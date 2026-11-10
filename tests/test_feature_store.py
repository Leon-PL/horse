"""Feature-store parity — build once, append, get the same data.

The store's promise: building from part of the history and then appending
the rest must produce featured rows identical to a single full batch
rebuild. This guards the "build once then append" workflow against silent
drift. Appends go through ``src.store_refresh.append_current_store``, the
path training uses.
"""

import numpy as np
import pandas as pd
import pytest
import config

from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.feature_store import FeatureStore
from src.featured_dataset import _stamp, data_signature, fe_signature
from src.model import get_feature_columns
from src.store_refresh import append_current_store
from tests.test_live_parity import _synthetic_raw


@pytest.fixture(autouse=True)
def all_feature_parity(monkeypatch):
    """Validate every feature, independently of a local/private allowlist."""
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)


@pytest.fixture(scope="module")
def proc_split():
    raw = _synthetic_raw(n_days=90, races_per_day=2, field=6, seed=23)
    proc = process_data(raw.copy())
    proc["race_date"] = pd.to_datetime(proc["race_date"], errors="coerce")
    dates = np.sort(proc["race_date"].unique())
    boundary = dates[len(dates) * 2 // 3]
    hist = proc[proc["race_date"] < boundary].copy()
    new = proc[proc["race_date"] >= boundary].copy()
    assert not hist.empty and not new.empty
    return proc, hist, new, boundary


def _build_then_append(store_root, hist, proc):
    """Build *hist*, stamp it current, then append the rest of *proc* as training does."""
    store = FeatureStore(root=str(store_root))
    built = store.build(hist.copy())
    _stamp(store, fe_signature(), data_signature(hist, cutoff=hist["race_date"].max()))
    assert append_current_store(proc.copy(), store=store), "a current store must append, not rebuild"
    return store, built


def _rows_from(featured, boundary):
    return featured[pd.to_datetime(featured["race_date"]) >= boundary]


def _assert_append_matches_batch(store_root, proc, hist, new, boundary):
    batch = engineer_features(proc.copy(), save=False)
    batch_new = batch[batch["race_date"] >= boundary]

    store, _ = _build_then_append(store_root, hist, proc)
    appended = _rows_from(store.load_featured(), boundary)
    assert len(appended) == len(new)

    model_cols = get_feature_columns(batch)
    assert model_cols
    assert get_feature_columns(appended) == model_cols, "appended feature schema differs from full rebuild"
    assert appended[model_cols].dtypes.equals(batch[model_cols].dtypes), "appended feature dtypes differ"

    key = ["race_id", "horse_id"]
    assert not batch_new.duplicated(key).any() and not appended.duplicated(key).any()
    b = batch_new.set_index(key)[model_cols].sort_index()
    a = appended.set_index(key)[model_cols].sort_index()
    assert b.index.equals(a.index), "appended row set differs from full rebuild"

    bad = []
    for col in model_cols:
        x = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(x) & np.isnan(y)
        close = np.isclose(x, y, rtol=1e-4, atol=1e-5, equal_nan=False) | both_nan
        if not close.all():
            n = int((~close).sum())
            j = int(np.argmax(~close))
            bad.append(f"{col}: {n}/{len(x)} differ (e.g. batch={x[j]!r} store={y[j]!r} at {b.index[j]})")
    assert not bad, (
        "Feature store append diverges from full rebuild:\n" + "\n".join(bad[:40])
    )


def test_append_matches_full_rebuild(proc_split, tmp_path):
    proc, hist, new, boundary = proc_split
    _assert_append_matches_batch(tmp_path / "store", proc, hist, new, boundary)


def test_persisted_featured_grows_and_dedupes(proc_split, tmp_path):
    proc, hist, new, _ = proc_split
    store, built = _build_then_append(tmp_path / "store2", hist, proc)

    full = store.load_featured()
    assert len(full) == len(built) + len(new)
    # Re-appending the same dates is a no-op (store holds settled history).
    assert append_current_store(proc.copy(), store=store)
    assert len(store.load_featured()) == len(full)


def test_appended_rating_state_matches_full_sweep(proc_split, tmp_path):
    """Building then appending must leave the same rating end-state as one
    full-history sweep — the persisted state stays bit-correct."""
    proc, hist, new, _ = proc_split
    store, _ = _build_then_append(tmp_path / "store3", hist, proc)
    incremental = store.load_state()["elo"]

    full = FeatureStore(root=str(tmp_path / "store_full"))
    full.build(proc.copy())
    reference = full.load_state()["elo"]

    # Compare the headline horse ratings dict.
    inc_r = incremental["horse_ratings"]
    ref_r = reference["horse_ratings"]
    assert set(inc_r) == set(ref_r)
    for k in ref_r:
        assert inc_r[k] == pytest.approx(ref_r[k], rel=1e-9, abs=1e-9)
