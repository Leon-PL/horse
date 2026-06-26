"""Regression test: grouped rolling features must be race-safe.

A trainer with several runners in one race must NOT let one stablemate's result
leak into another's rolling feature (the +56% ROI@BSP stablemate leak, 2026-06-24).
`_fast_grouped_rolling_core` row-shifts after sorting by (entity, date, race_id),
so same-race rows are consecutive — the fix broadcasts each (entity, race) block's
first (race-excluded) row across the whole block.
"""
import numpy as np
import pandas as pd

from src.feature_engineer import _fast_grouped_rolling_core


def _df():
    # Trainer T: R1 solo (loss); R2 with TWO runners (A wins, B loses); R3 solo.
    return pd.DataFrame({
        "trainer": ["T", "T", "T", "T"],
        "race_id": ["R1", "R2", "R2", "R3"],
        "race_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"]),
        "won": [0, 1, 0, 0],
    })


def test_stablemates_see_identical_prerace_history():
    df = _df()
    out = _fast_grouped_rolling_core(df, "trainer", "won", [("tw", "sum", 5)])
    tw = out["tw"]
    # R2's two stablemates must see the SAME pre-race trainer wins (neither sees
    # the other's same-race result) — both 0 (only prior race R1, a loss).
    assert tw[1] == tw[2] == 0.0
    # The NEXT race (R3) must still accumulate R2's win.
    assert tw[3] == 1.0


def test_no_leak_at_scale():
    # Many trainers, some with multiple same-race runners; the winner's own
    # rolling feature must never be inflated by being the winner this race.
    rng = np.random.default_rng(0)
    rows = []
    for r in range(300):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=r)
        for tr in rng.choice(["A", "B", "C", "D"], size=rng.integers(2, 5)):
            rows.append({"trainer": tr, "race_id": f"R{r}", "race_date": date,
                         "won": 0})
        rows[-1]["won"] = 1  # exactly one winner per race
    df = pd.DataFrame(rows)
    out = _fast_grouped_rolling_core(df, "trainer", "won", [("tw", "sum", 10)])
    df["tw"] = out["tw"]
    # Within any race, all of a trainer's runners must share one rolling value
    # (race-safe → identical pre-race history), regardless of who won.
    g = df.groupby(["race_id", "trainer"])["tw"].nunique()
    assert (g <= 1).all()
