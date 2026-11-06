"""Repairs to excluded store features found in the 1 November 2026 review."""

import numpy as np
import pandas as pd

from src.data_processor import process_data
from src.feature_engineer import (
    _add_result_history_flags, _event_sort_key, _time_window_stats_core,
    add_horse_features, add_track_features, counts_state_scope,
)
from tests.test_live_parity import _synthetic_raw


def _processed(n_days=30):
    df = process_data(_synthetic_raw(n_days=n_days, races_per_day=2, field=6))
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["_event_dt"] = _event_sort_key(df)
    df = df.sort_values(["_event_dt", "race_id"]).reset_index(drop=True)
    return _add_result_history_flags(df)


def _expected_runs_since_last_win(df):
    out = pd.Series(np.nan, index=df.index)
    for _, rows in df.groupby("horse_id", sort=False):
        since = None
        for i in rows.index:
            out[i] = np.nan if since is None else since
            if df.at[i, "won"] == 1:
                since = 0
            elif since is not None:
                since += 1
    return out


def test_runs_since_last_win_counts_each_horses_own_runs():
    df = _processed()
    out = add_horse_features(df.copy())
    expected = _expected_runs_since_last_win(df)
    # It was a constant 0: the kernel ran on rows in race order, not by horse.
    assert out["runs_since_last_win"].nunique() > 3
    pd.testing.assert_series_equal(
        out["runs_since_last_win"].astype(float), expected, check_names=False,
    )
    assert out.loc[out["horse_prev_races"].eq(0), "runs_since_last_win"].isna().all()


def test_runs_since_last_win_resumes_from_incremental_state():
    df = _processed()
    boundary = df.race_date.sort_values().unique()[20]
    expected = add_horse_features(df.copy())
    state = {}
    with counts_state_scope(state):
        add_horse_features(df[df.race_date < boundary].copy())
    with counts_state_scope(state):
        actual = add_horse_features(df[df.race_date >= boundary].copy())
    pd.testing.assert_series_equal(
        expected.loc[actual.index, "runs_since_last_win"], actual["runs_since_last_win"],
    )


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
