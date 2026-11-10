"""Parity tests for the incremental cumulative primitive (src/fe_incremental.py).

Proves CountsAccumulator (1) matches the production race-safe formula exactly,
and (2) a seeded append reproduces a full recompute across the boundary — the
correctness guarantee the class-C incremental conversion rests on.
"""

import json

import numpy as np
import pandas as pd
import pytest

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
        assert fe.current_counts_state() is None
        with fe.counts_state_scope({"m": {}}):
            assert fe.current_counts_state() == {"m": {}}
        assert fe.current_counts_state() is None

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


def _fractional_history(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, 800).astype(str), np.repeat(np.arange(200), 4), rng.uniform(0, 1, 800)


@pytest.mark.parametrize("split", [4, 40, 400, 796])
def test_resumed_race_safe_sum_has_the_same_bits_as_one_pass(split):
    keys, races, values = _fractional_history(split)
    whole, _ = cumulative_counts(keys, races, {"m": values})
    first, state = cumulative_counts(keys[:split], races[:split], {"m": values[:split]})
    second, _ = cumulative_counts(keys[split:], races[split:], {"m": values[split:]}, state=state)
    assert np.concatenate([first["m"], second["m"]]).tobytes() == whole["m"].tobytes()


@pytest.mark.parametrize("split", [1, 40, 399, 799])
def test_resumed_plain_sum_has_the_same_bits_as_one_pass(split):
    keys, _, values = _fractional_history(split)
    whole, _ = cumulative_counts_plain(keys, {"m": values})
    first, state = cumulative_counts_plain(keys[:split], {"m": values[:split]})
    second, _ = cumulative_counts_plain(keys[split:], {"m": values[split:]}, state=state)
    assert np.concatenate([first["m"], second["m"]]).tobytes() == whole["m"].tobytes()


def test_whole_number_metrics_carry_no_compensation():
    keys, races = np.array(["a", "b"]), np.array(["r1", "r2"])
    _, state = cumulative_counts(keys, races, {"m": np.array([1.0, 2.0])})
    assert state["m"] == {"a": 1.0, "b": 2.0}
    assert state["m::compensation"] == {}


def test_rolling_seeded_append_matches_full():
    """Seed-aware _fast_grouped_rolling: a seeded append == a full recompute for
    the new rows, across bounded ops and EWMA (within tolerance)."""
    rng = np.random.default_rng(11)
    rows = []
    rid = 0
    for day in range(40):
        for _ in range(int(rng.integers(1, 3))):
            rid += 1
            for _ in range(int(rng.integers(2, 6))):
                rows.append({
                    "race_id": f"R{rid:04d}",
                    "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "horse": f"H{int(rng.integers(0, 15))}",
                    "v": float(rng.normal()),
                })
    df = pd.DataFrame(rows)
    ops = [("m", "mean", 3), ("s", "sum", 5), ("mn", "min", 5),
           ("mx", "max", 5), ("sd", "std", 5), ("e", "ewma", 0.3)]

    full = fe._fast_grouped_rolling_core(df, "horse", "v", ops)
    rids = sorted(df["race_id"].unique())
    cut = rids[len(rids) // 2]
    hist = df[df["race_id"] <= cut]
    new = df[df["race_id"] > cut]
    newmask = (df["race_id"] > cut).to_numpy()

    _, state = fe._rolling_with_seed(hist, "horse", "v", ops, {})
    new_res, _ = fe._rolling_with_seed(new, "horse", "v", ops, state)
    for k in ["m", "s", "mn", "mx", "sd", "e"]:
        a = new_res[k]
        b = full[k][newmask]
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.all(np.isclose(a, b, rtol=1e-4, atol=1e-6) | both_nan), k


def test_time_window_seeded_append_matches_full():
    """Seed-aware _time_window_stats: seeded append == full recompute for new rows
    (sum and count), with the window straddling the history/new boundary."""
    rng = np.random.default_rng(13)
    rows = []
    rid = 0
    for day in range(60):
        for _ in range(int(rng.integers(1, 3))):
            rid += 1
            for _ in range(int(rng.integers(2, 6))):
                rows.append({
                    "race_id": f"R{rid:04d}",
                    "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "trainer": f"T{int(rng.integers(0, 8))}",
                    "v": float(rng.integers(0, 2)),
                })
    df = pd.DataFrame(rows)
    vs_full, vc_full = fe._time_window_stats_core(df, "trainer", "race_date", "v", 30)

    boundary = pd.Timestamp("2024-01-31")
    hist = df[df["race_date"] <= boundary]
    new = df[df["race_date"] > boundary]
    newmask = (df["race_date"] > boundary).to_numpy()

    _, state = fe._time_window_with_seed(hist, "trainer", "race_date", "v", 30, {})
    (vs, vc), _ = fe._time_window_with_seed(new, "trainer", "race_date", "v", 30, state)
    assert np.allclose(vs, vs_full[newmask])
    assert np.allclose(vc, vc_full[newmask])


def test_lag_seeded_append_matches_full():
    """Seed-aware _lag: seeded append == full groupby.shift(1) for new rows."""
    rng = np.random.default_rng(21)
    rows = []
    rid = 0
    for day in range(30):
        rid += 1
        for _ in range(int(rng.integers(2, 6))):
            rows.append({
                "race_id": f"R{rid:04d}",
                "horse": f"H{int(rng.integers(0, 10))}",
                "v": float(rng.normal()),
            })
    df = pd.DataFrame(rows)
    full = df.groupby("horse", sort=False)["v"].shift(1).to_numpy()
    cut = sorted(df["race_id"].unique())[len(df["race_id"].unique()) // 2]
    hist = df[df["race_id"] <= cut]
    new = df[df["race_id"] > cut]
    newmask = (df["race_id"] > cut).to_numpy()

    state: dict = {}
    with fe.counts_state_scope(state):
        fe._lag(hist, "horse", "v")
    with fe.counts_state_scope(state):
        got = fe._lag(new, "horse", "v").to_numpy()
    a, b = got, full[newmask]
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.all(np.isclose(a, b, equal_nan=False) | both_nan)


def _horse_value_frame(seed, n_horses=8, hi=20):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for _day in range(30):
        rid += 1
        for _ in range(int(rng.integers(2, 6))):
            rows.append({
                "race_id": f"R{rid:04d}",
                "horse": f"H{int(rng.integers(0, n_horses))}",
                "v": float(rng.integers(0, hi)),
            })
    return pd.DataFrame(rows)


def test_running_extreme_seeded_matches_full():
    df = _horse_value_frame(31)
    rids = sorted(df["race_id"].unique())
    cut = rids[len(rids) // 2]
    nm = (df["race_id"] > cut).to_numpy()
    for kind, prev in [("max", True), ("max", False), ("min", False)]:
        full = fe._running_extreme(df, "horse", "v", kind=kind, prev=prev).to_numpy()
        st: dict = {}
        with fe.counts_state_scope(st):
            fe._running_extreme(df[df["race_id"] <= cut], "horse", "v", kind=kind, prev=prev)
        with fe.counts_state_scope(st):
            got = fe._running_extreme(df[df["race_id"] > cut], "horse", "v", kind=kind, prev=prev).to_numpy()
        a, b = got, full[nm]
        bn = np.isnan(a) & np.isnan(b)
        assert np.all(np.isclose(a, b) | bn), (kind, prev)


def test_ffill_prev_seeded_matches_full():
    df = _horse_value_frame(32)
    df.loc[df.index % 3 != 0, "v"] = np.nan  # sparse non-nulls
    rids = sorted(df["race_id"].unique())
    cut = rids[len(rids) // 2]
    nm = (df["race_id"] > cut).to_numpy()
    full = fe._ffill_prev(df, "horse", "v").to_numpy()
    st: dict = {}
    with fe.counts_state_scope(st):
        fe._ffill_prev(df[df["race_id"] <= cut], "horse", "v")
    with fe.counts_state_scope(st):
        got = fe._ffill_prev(df[df["race_id"] > cut], "horse", "v").to_numpy()
    a, b = got, full[nm]
    bn = np.isnan(a) & np.isnan(b)
    assert np.all(np.isclose(a, b) | bn)


def test_expanding_prev_seeded_matches_full():
    rng = np.random.default_rng(33)
    df = pd.DataFrame({
        "k": [f"C{int(rng.integers(0, 4))}" for _ in range(400)],
        "v": rng.normal(size=400),
    })
    full = fe._expanding_prev(df, "k", "v", ["median", "std"])
    half = len(df) // 2
    hist, new = df.iloc[:half], df.iloc[half:]
    st: dict = {}
    with fe.counts_state_scope(st):
        fe._expanding_prev(hist, "k", "v", ["median", "std"])
    with fe.counts_state_scope(st):
        got = fe._expanding_prev(new, "k", "v", ["median", "std"])
    for s in ("median", "std"):
        a = got[s].to_numpy()
        b = full[s].to_numpy()[half:]
        bn = np.isnan(a) & np.isnan(b)
        assert np.all(np.isclose(a, b, rtol=1e-6) | bn), s


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
