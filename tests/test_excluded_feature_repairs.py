"""Repairs to excluded store features found in the 1 November 2026 review.

``runs_since_last_win`` (repaired here on 1 November) was removed on 7 November
(research 065): no feature set used it.
"""

import numpy as np
import pandas as pd

from src.data_processor import process_data
from src.features.common import _add_result_history_flags, _event_sort_key
from src.features.history import _time_window_stats_core
from src.features.horse import add_track_features
from tests.test_live_parity import _synthetic_raw


def _processed(n_days=30):
    df = process_data(_synthetic_raw(n_days=n_days, races_per_day=2, field=6))
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["_event_dt"] = _event_sort_key(df)
    df = df.sort_values(["_event_dt", "race_id"]).reset_index(drop=True)
    return _add_result_history_flags(df)


def test_horse_distance_record_groups_yards_into_half_furlongs():
    df = _processed(n_days=10)
    horse = df["horse_id"].iloc[0]
    mine = df.index[df.horse_id.eq(horse)]
    # 7f, 7f 13y and 7f 35y are one trip, not three.
    df.loc[mine, "distance_furlongs"] = np.resize([7.0, 7.06, 7.16], len(mine))
    out = add_track_features(df)
    runs = out.loc[mine, "horse_dist_runs"].to_numpy()
    assert list(runs) == list(range(len(mine)))


def test_time_window_sum_skips_missing_values():
    frame = pd.DataFrame({
        "jockey": ["A"] * 4,
        "race_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        "v": [1.0, np.nan, 1.0, 1.0],
    })
    total, count = _time_window_stats_core(frame, "jockey", "race_date", "v", 14)
    # A non-start (missing) earlier in the window used to blank the sum.
    assert list(total) == [0.0, 1.0, 1.0, 2.0]
    assert list(count) == [0.0, 1.0, 2.0, 3.0]
