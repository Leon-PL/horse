"""Parity tests for the incremental cumulative primitive (src/fe_incremental.py).

Proves CountsAccumulator (1) matches the production race-safe formula exactly,
and (2) a seeded append reproduces a full recompute across the boundary — the
correctness guarantee the class-C incremental conversion rests on.
"""

import json

import numpy as np
import pandas as pd

import src.feature_engineer as fe
from src.feature_engineer import _race_safe_cumsum
from src.fe_incremental import (
    CountsAccumulator,
    PlainCountsAccumulator,
    cumulative_counts,
    cumulative_counts_plain,
)


class TestStateThreading:
    """The mechanism that lets class-C stages seed from / write back state."""

    def test_scope_sets_and_restores(self):
        assert fe._COUNTS_STATE is None
        with fe.counts_state_scope({"m": {}}):
            assert fe._COUNTS_STATE == {"m": {}}
        assert fe._COUNTS_STATE is None

    def test_cumulative_full_build_when_no_scope(self):
        keys = np.array(["A", "A", "B"]); races = np.array(["r1", "r2", "r3"])
        out = fe._cumulative(keys, races, {"m": np.array([1.0, 1.0, 1.0])})
        assert list(out["m"]) == [0.0, 1.0, 0.0]      # prior-to-race totals

    def test_cumulative_seeds_and_writes_back(self):
        keys = np.array(["A", "A", "B"]); races = np.array(["r1", "r2", "r3"])
        state: dict = {}
        with fe.counts_state_scope(state):
            fe._cumulative(keys, races, {"m": np.array([1.0, 1.0, 1.0])})
        assert state["m"]["A"] == 2.0 and state["m"]["B"] == 1.0   # written back

        with fe.counts_state_scope({"m": {"A": 5.0}}):              # seeded
            out = fe._cumulative(np.array(["A"]), np.array(["r9"]), {"m": np.array([1.0])})
        assert out["m"][0] == 5.0


def _frame(seed: int = 0) -> pd.DataFrame:
    """Chronological frame with trainers that have multiple runners per race
    (the multi-runner-per-race case race-safety exists for) and horses with one
    run per race."""
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    horse = 0
    for day in range(12):
        for _ in range(rng.integers(1, 3)):           # races per day
            rid += 1
            field = rng.integers(3, 9)
            for _ in range(field):
                rows.append({
                    "race_id": f"R{rid:04d}",
                    "trainer": f"T{rng.integers(0, 5)}",   # repeats within a race
                    "horse": f"H{horse}",                   # unique per row
                    "won": int(rng.random() < 0.12),
                    "ran": 1,
                })
                horse += 1
    return pd.DataFrame(rows)


def test_matches_race_safe_cumsum_multi_runner():
    df = _frame()
    expected = _race_safe_cumsum(df, "trainer", "won").to_numpy()
    got = CountsAccumulator().transform(df["trainer"], df["race_id"], df["won"])
    assert np.allclose(got, expected)


def test_matches_race_safe_cumsum_counts():
    # cumcount-style prior-runner counts == race-safe cumsum of a 1-flag.
    df = _frame(1)
    expected = _race_safe_cumsum(df, "trainer", "ran").to_numpy()
    got = CountsAccumulator().transform(df["trainer"], df["race_id"], df["ran"])
    assert np.allclose(got, expected)


def test_one_row_per_race_entity():
    # Horses run once per race: race-safe == plain cumsum()-current.
    df = _frame(2)
    plain = (df.groupby("horse")["won"].cumsum() - df["won"]).to_numpy()
    got = CountsAccumulator().transform(df["horse"], df["race_id"], df["won"])
    assert np.allclose(got, plain)


def test_seeded_append_matches_full_recompute():
    df = _frame(3)
    full = CountsAccumulator().transform(df["trainer"], df["race_id"], df["won"])

    # Split so the new races are strictly after the history races.
    cut = sorted(df["race_id"].unique())[len(df["race_id"].unique()) // 2]
    hist = df[df["race_id"] <= cut]
    new = df[df["race_id"] > cut]

    acc = CountsAccumulator()
    acc.transform(hist["trainer"], hist["race_id"], hist["won"])      # advance state
    got_new = acc.transform(new["trainer"], new["race_id"], new["won"])
    assert np.allclose(got_new, full[df["race_id"] > cut])


def test_cumulative_counts_multi_metric_and_seed():
    # The class-C stage API: several metrics for one entity, full and seeded.
    df = _frame(5)
    metrics = {"runs": df["ran"].to_numpy(), "wins": df["won"].to_numpy()}
    full, _ = cumulative_counts(df["trainer"], df["race_id"], metrics)
    assert np.allclose(full["runs"], _race_safe_cumsum(df, "trainer", "ran").to_numpy())
    assert np.allclose(full["wins"], _race_safe_cumsum(df, "trainer", "won").to_numpy())

    cut = sorted(df["race_id"].unique())[len(df["race_id"].unique()) // 2]
    h, n = df["race_id"] <= cut, df["race_id"] > cut
    _, state = cumulative_counts(
        df.loc[h, "trainer"], df.loc[h, "race_id"],
        {k: v[h.to_numpy()] for k, v in metrics.items()},
    )
    new, _ = cumulative_counts(
        df.loc[n, "trainer"], df.loc[n, "race_id"],
        {k: v[n.to_numpy()] for k, v in metrics.items()}, state=state,
    )
    assert np.allclose(new["runs"], full["runs"][n.to_numpy()])
    assert np.allclose(new["wins"], full["wins"][n.to_numpy()])


def test_plain_matches_cumsum_minus_current():
    # PlainCountsAccumulator == groupby(key).cumsum() - current (row-order).
    df = _frame(7)
    expected = (df.groupby("trainer")["won"].cumsum() - df["won"]).to_numpy()
    got = PlainCountsAccumulator().transform(df["trainer"], df["won"])
    assert np.allclose(got, expected)


def test_plain_seeded_append_matches_full_recompute():
    df = _frame(8)
    full = PlainCountsAccumulator().transform(df["trainer"], df["won"])
    cut = sorted(df["race_id"].unique())[len(df["race_id"].unique()) // 2]
    hist = df[df["race_id"] <= cut]
    new = df[df["race_id"] > cut]

    acc = PlainCountsAccumulator()
    acc.transform(hist["trainer"], hist["won"])
    got_new = acc.transform(new["trainer"], new["won"])
    assert np.allclose(got_new, full[(df["race_id"] > cut).to_numpy()])

    # API form
    _, state = cumulative_counts_plain(hist["trainer"], {"w": hist["won"].to_numpy()})
    new_out, _ = cumulative_counts_plain(new["trainer"], {"w": new["won"].to_numpy()}, state=state)
    assert np.allclose(new_out["w"], full[(df["race_id"] > cut).to_numpy()])


def test_state_is_json_serialisable_and_reloadable():
    df = _frame(4)
    cut = sorted(df["race_id"].unique())[len(df["race_id"].unique()) // 2]
    hist = df[df["race_id"] <= cut]
    new = df[df["race_id"] > cut]
    full = CountsAccumulator().transform(df["trainer"], df["race_id"], df["won"])

    acc = CountsAccumulator()
    acc.transform(hist["trainer"], hist["race_id"], hist["won"])
    blob = json.dumps(acc.to_state())                 # persist
    revived = CountsAccumulator(json.loads(blob))     # reload
    got_new = revived.transform(new["trainer"], new["race_id"], new["won"])
    assert np.allclose(got_new, full[df["race_id"] > cut])
