"""Parity tests for the incremental cumulative primitive (src/fe_incremental.py).

Proves CountsAccumulator (1) matches the production race-safe formula exactly,
and (2) a seeded append reproduces a full recompute across the boundary — the
correctness guarantee the class-C incremental conversion rests on.
"""

import json

import numpy as np
import pandas as pd

from src.feature_engineer import _race_safe_cumsum
from src.fe_incremental import CountsAccumulator


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
