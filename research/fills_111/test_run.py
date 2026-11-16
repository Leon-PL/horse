import numpy as np
import pandas as pd

from research.fills_111 import run


def _quote(clock, back1, size1, back2=np.nan, size2=np.nan, in_running=False):
    return {"taken_at": pd.Timestamp(f"2026-11-16 {clock}", tz="UTC"), "back1": back1, "back1_size": size1,
            "back2": back2, "back2_size": size2, "back3": np.nan, "back3_size": np.nan, "in_running": in_running,
            "market_volume": 100.}


def test_follow_times_how_long_the_price_stood_and_how_much_was_there():
    start, off = pd.Timestamp("2026-11-16 11:00", tz="UTC"), pd.Timestamp("2026-11-16 13:00", tz="UTC")
    path = pd.DataFrame([_quote("11:00", 3.0, 5.), _quote("11:30", 2.9, 50., 3.0, 20.),   # 3.0 still there at level 2
                         _quote("12:00", 2.8, 40.),                                         # gone
                         _quote("12:50", 3.1, 8.), _quote("13:01", 3.5, 9., in_running=True)])
    info = run.follow(path, 3.0, start, off)
    assert info["snapshots"] == 4 and info["stake_1100"] == 5. and info["max_stake"] == 20.
    assert info["held_minutes"] == 60.                       # first missing at 12:00
    assert abs(info["held_share"] - (60 + 10) / 120) < 1e-9   # 11:00-12:00 and 12:50-13:00 of 120 minutes
    assert info["held_to_close"] and info["close_back"] == 3.1


def test_stake_at_counts_only_levels_at_or_above_the_price():
    rows = pd.DataFrame([{"back1": 4.0, "back1_size": 2., "back2": 3.8, "back2_size": 5., "back3": 3.5, "back3_size": 9.}])
    assert run.stake_at(rows, 3.8)[0] == 7.
