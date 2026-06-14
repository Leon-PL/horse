"""Incremental rating parity — the contract for the feature store.

The store will compute ratings once over history, persist the end-state,
and on append re-seed that state and sweep only the new rows. For that to
be safe, the seeded incremental sweep must produce *byte-identical*
pre-race rating columns to a full batch sweep over the same data.

Strategy: take a synthetic multi-month processed dataset, split it on a
date boundary, then compare:

  batch:        compute_elo_features(all)
  incremental:  state = compute_elo_features(part1, return_state=True)[1]
                compute_elo_features(part2, elo_state=state)

Every Elo column on the part2 rows must match the batch run's part2 rows.
Because the no-leakage rule makes a row's rating depend only on strictly
prior races, splitting on a date boundary means part2's pre-race state is
exactly what the batch had accumulated at the boundary.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src.data_processor import process_data
from src.ratings import compute_elo_features, compute_glicko_features
from tests.test_live_parity import _synthetic_raw


@pytest.fixture(scope="module")
def split_processed():
    raw = _synthetic_raw(n_days=80, races_per_day=2, field=6, seed=11)
    proc = process_data(raw.copy(), save=False)
    proc["race_date"] = pd.to_datetime(proc["race_date"], errors="coerce")
    dates = np.sort(proc["race_date"].unique())
    boundary = dates[len(dates) * 2 // 3]  # ~2/3 history, ~1/3 appended
    hist = proc[proc["race_date"] < boundary].copy()
    new = proc[proc["race_date"] >= boundary].copy()
    assert not hist.empty and not new.empty
    return proc, hist, new, boundary


def _elo_only_cols(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
    return [c for c in after.columns if c not in before.columns]


def test_incremental_elo_matches_batch(split_processed):
    proc, hist, new, boundary = split_processed

    # Batch over everything
    batch = compute_elo_features(proc.copy())
    elo_cols = _elo_only_cols(proc, batch)
    assert elo_cols, "no Elo columns produced — test wiring is wrong"

    # Incremental: build state from history, then sweep only the new rows
    _, state = compute_elo_features(hist.copy(), return_state=True)
    inc_new = compute_elo_features(new.copy(), elo_state=state)

    # Align on (race_id, horse_name) and compare the appended rows
    key = ["race_id", "horse_name"]
    batch_new = batch[batch["race_date"] >= boundary]
    b = batch_new.set_index(key)[elo_cols].sort_index()
    i = inc_new.set_index(key)[elo_cols].sort_index()
    assert b.index.equals(i.index), "row sets differ between batch and incremental"

    bad = []
    for col in elo_cols:
        a = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(i[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(c)
        close = np.isclose(a, c, rtol=1e-9, atol=1e-9, equal_nan=False) | both_nan
        if not close.all():
            n = int((~close).sum())
            j = int(np.argmax(~close))
            bad.append(f"{col}: {n}/{len(a)} differ (e.g. batch={a[j]!r} inc={c[j]!r} at {b.index[j]})")
    assert not bad, "Incremental Elo diverges from batch:\n" + "\n".join(bad[:40])


def test_returned_state_roundtrips_and_is_serializable(split_processed):
    """The end-state must be plain dicts (picklable) covering every dict."""
    import pickle

    _, hist, _, _ = split_processed
    _, state = compute_elo_features(hist.copy(), return_state=True)

    expected = {
        "horse_ratings", "jockey_ratings", "trainer_ratings",
        "horse_race_counts", "jockey_race_counts", "trainer_race_counts",
        "horse_momentum", "horse_surf_ratings", "horse_rt_ratings",
        "horse_dc_ratings", "jockey_rt_ratings", "horse_surf_counts",
        "horse_rt_counts", "horse_dc_counts", "jockey_rt_counts",
        "horse_margin_ratings", "horse_margin_race_counts",
        "horse_margin_momentum",
    }
    assert expected <= set(state)
    assert all(isinstance(v, dict) for v in state.values())
    # Round-trips through pickle unchanged (the store persists it this way).
    assert pickle.loads(pickle.dumps(state)).keys() == state.keys()
    assert state["horse_ratings"], "expected some horses to be rated"


def test_seeding_changes_nothing_when_state_empty(split_processed):
    """elo_state=None and elo_state={} must behave identically (full batch)."""
    proc, _, _, _ = split_processed
    a = compute_elo_features(proc.copy())
    b = compute_elo_features(proc.copy(), elo_state={})
    elo_cols = _elo_only_cols(proc, a)
    key = ["race_id", "horse_name"]
    aa = a.set_index(key)[elo_cols].sort_index()
    bb = b.set_index(key)[elo_cols].sort_index()
    pd.testing.assert_frame_equal(aa, bb, rtol=1e-12, atol=1e-12)


# ── Glicko ──────────────────────────────────────────────────────────────

@pytest.fixture
def all_glicko_flags_on(monkeypatch):
    """Exercise every Glicko sub-pass, not just the config defaults."""
    for flag in (
        "GLICKO_ENABLED", "GLICKO_JOCKEY_TRAINER", "GLICKO_MARGIN",
        "GLICKO2_ENABLED", "TRUESKILL_ENABLED", "GLICKO_DIMENSIONAL",
    ):
        monkeypatch.setattr(config, flag, True, raising=False)


def _assert_glicko_parity(proc, hist, new, boundary, rtol, atol):
    batch = compute_glicko_features(proc.copy())
    gcols = _elo_only_cols(proc, batch)
    assert gcols, "no Glicko columns produced — test wiring is wrong"

    _, state = compute_glicko_features(hist.copy(), return_state=True)
    inc_new = compute_glicko_features(new.copy(), glicko_state=state)

    key = ["race_id", "horse_name"]
    batch_new = batch[batch["race_date"] >= boundary]
    b = batch_new.set_index(key)[gcols].sort_index()
    i = inc_new.set_index(key)[gcols].sort_index()
    assert b.index.equals(i.index), "row sets differ between batch and incremental"

    bad = []
    for col in gcols:
        a = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(i[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(c)
        close = np.isclose(a, c, rtol=rtol, atol=atol, equal_nan=False) | both_nan
        if not close.all():
            n = int((~close).sum())
            j = int(np.argmax(~close))
            bad.append(f"{col}: {n}/{len(a)} differ (e.g. batch={a[j]!r} inc={c[j]!r} at {b.index[j]})")
    assert not bad, "Incremental Glicko diverges from batch:\n" + "\n".join(bad[:40])


def test_incremental_glicko_matches_batch_all_passes(split_processed, all_glicko_flags_on):
    proc, hist, new, boundary = split_processed
    _assert_glicko_parity(proc, hist, new, boundary, rtol=1e-7, atol=1e-7)


def test_incremental_glicko_matches_batch_default_config(split_processed):
    proc, hist, new, boundary = split_processed
    _assert_glicko_parity(proc, hist, new, boundary, rtol=1e-7, atol=1e-7)


def test_glicko_state_is_picklable(split_processed, all_glicko_flags_on):
    import pickle

    _, hist, _, _ = split_processed
    _, state = compute_glicko_features(hist.copy(), return_state=True)
    # horse pass always present; every sub-pass we enabled should appear.
    assert "horse" in state and state["horse"]["ratings"]
    assert {"trueskill", "glicko2", "margin", "jockey", "trainer"} <= set(state)
    assert pickle.loads(pickle.dumps(state)).keys() == state.keys()
