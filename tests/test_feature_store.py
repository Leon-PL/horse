"""Feature-store parity — build once, append, get the same data.

The store's promise: building from part of the history and then appending
the rest must produce featured rows identical to a single full batch
rebuild. This guards the "build once then append" workflow against silent
drift.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from tests.test_live_parity import _synthetic_raw


@pytest.fixture(scope="module")
def proc_split():
    raw = _synthetic_raw(n_days=90, races_per_day=2, field=6, seed=23)
    proc = process_data(raw.copy(), save=False)
    proc["race_date"] = pd.to_datetime(proc["race_date"], errors="coerce")
    dates = np.sort(proc["race_date"].unique())
    boundary = dates[len(dates) * 2 // 3]
    hist = proc[proc["race_date"] < boundary].copy()
    new = proc[proc["race_date"] >= boundary].copy()
    assert not hist.empty and not new.empty
    return proc, hist, new, boundary


def test_build_then_append_matches_full_rebuild(proc_split, tmp_path):
    proc, hist, new, boundary = proc_split

    # Full batch rebuild — the reference.
    batch = engineer_features(proc.copy(), save=False)
    batch_new = batch[batch["race_date"] >= boundary]

    # Store: build on history, append the new dates.
    store = FeatureStore(root=str(tmp_path / "store"))
    store.build(hist.copy())
    appended = store.append(new.copy())

    model_cols = [c for c in get_feature_columns(batch) if c in appended.columns]
    assert model_cols

    key = ["race_id", "horse_name"]
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
    assert not bad, "Feature store append diverges from full rebuild:\n" + "\n".join(bad[:40])


def test_persisted_featured_grows_and_dedupes(proc_split, tmp_path):
    proc, hist, new, _ = proc_split
    store = FeatureStore(root=str(tmp_path / "store2"))
    built = store.build(hist.copy())
    appended = store.append(new.copy())

    full = store.load_featured()
    assert len(full) == len(built) + len(appended)
    # Re-appending the same dates is a no-op (store holds settled history).
    again = store.append(new.copy())
    assert again.empty
    assert len(store.load_featured()) == len(full)


def test_appended_rating_state_matches_full_sweep(proc_split, tmp_path):
    """Building then appending must leave the same rating end-state as one
    full-history sweep — the persisted state stays bit-correct."""
    proc, hist, new, _ = proc_split
    store = FeatureStore(root=str(tmp_path / "store3"))
    store.build(hist.copy())
    store.append(new.copy())
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
