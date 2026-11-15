"""Outcome, date, and future-invariance guards for predictive features."""

import numpy as np
import pandas as pd
import pytest

from src.data_processor import clean_data, process_data
from src.feature_engineer import (
    _add_result_history_flags, add_pedigree_features,
    add_race_context_features, counts_state_scope, engineer_features,
)
from tests.test_live_parity import _synthetic_raw


def _history():
    raw = _synthetic_raw(n_days=4, races_per_day=2, field=6)
    raw["track"] = "York"
    raw["sire"] = "Shared sire"
    raw["dam"] = "Shared dam"
    raw["damsire"] = "Shared damsire"
    # Real results are usually in finishing order, the strongest leakage case.
    return process_data(raw).sort_values(
        ["race_date", "off_time", "finish_position"], kind="stable"
    ).reset_index(drop=True)


def test_draw_history_excludes_current_race_and_day():
    df = _history()
    # Enough same-bin history to exceed the feature's six-run threshold.
    df["draw"] = 1
    day = df["race_date"].max()
    expected = add_race_context_features(df.copy())
    pending = df.copy()
    pending.loc[pending.race_date == day, "won"] = 0
    actual = add_race_context_features(pending)
    pd.testing.assert_series_equal(
        expected.loc[df.race_date == day, "track_draw_win_rate"],
        actual.loc[df.race_date == day, "track_draw_win_rate"],
    )
    assert expected.loc[df.race_date == day, "track_draw_win_rate"].nunique() == 1


def test_pedigree_history_excludes_all_same_day_outcomes():
    df = _history()
    day = df["race_date"].max()
    expected = add_pedigree_features(_add_result_history_flags(df.copy()))
    pending = df.copy()
    pending.loc[pending.race_date == day, ["won", "finish_position"]] = 0
    actual = add_pedigree_features(_add_result_history_flags(pending))
    cols = [c for c in expected if c.startswith(("sire_", "dam_", "damsire_"))]
    pd.testing.assert_frame_equal(
        expected.loc[df.race_date == day, cols], actual.loc[df.race_date == day, cols]
    )


def test_corrected_history_features_resume_from_incremental_state():
    df = _history()
    df["draw"] = 1
    boundary = df.race_date.max()
    def features(frame):
        return add_race_context_features(
            add_pedigree_features(_add_result_history_flags(frame.copy()))
        )
    expected = features(df)
    state = {}
    with counts_state_scope(state):
        features(df[df.race_date < boundary])
    with counts_state_scope(state):
        actual = features(df[df.race_date >= boundary])
    cols = [c for c in expected if c.startswith(("sire_", "dam_", "damsire_"))]
    cols.append("track_draw_win_rate")
    pd.testing.assert_frame_equal(expected.loc[actual.index, cols], actual[cols])


def test_cleaning_is_invariant_to_unrelated_future_races():
    raw = _synthetic_raw(n_days=2, races_per_day=1, field=6)
    first = raw.race_date.min()
    raw.loc[raw.race_date == first, ["odds", "official_rating", "weight_lbs", "draw"]] = np.nan
    before = clean_data(raw[raw.race_date == first])
    after = clean_data(raw)
    pd.testing.assert_frame_equal(before, after.loc[before.index])
    assert before[["odds", "official_rating", "weight_lbs", "draw"]].isna().all().all()


@pytest.mark.parametrize("odds", [np.nan, -2.0, 0.0, 1.0, np.inf])
def test_missing_or_invalid_market_price_is_not_invented(odds):
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=6)
    raw.loc[0, "odds"] = odds
    cleaned = clean_data(raw)
    assert pd.isna(cleaned.loc[0, "odds"])


def test_cold_start_speed_features_stay_unknown():
    from src.store_columns import unpruned
    raw = _synthetic_raw(n_days=3, races_per_day=1, field=6)
    with unpruned():   # the speed-figure averages are computed but not stored
        featured = engineer_features(process_data(raw), save=False)
    first_runs = featured["horse_prev_races"] == 0
    cols = ["speed_fig_avg_5", "speed_fig_avg_10", "speed_fig_best_5", "speed_fig_adj_best_5"]
    assert cols
    assert featured.loc[first_runs, cols].isna().all().all()


def test_non_finisher_does_not_count_as_winning_adjusted_speed():
    from src.feature_engineer import add_speed_figure_features

    df = pd.DataFrame({
        "race_id": ["a", "b", "c"],
        "race_date": pd.date_range("2025-01-01", periods=3),
        "horse_name": ["Same horse"] * 3,
        "finish_position": [2, 0, 2],
        "lengths_behind": [12.0, np.nan, 1.0],
        "distance_furlongs": [6.0] * 3,
        "race_class": ["Class 5"] * 3,
    })
    featured = add_speed_figure_features(df)
    for column in ["speed_fig_adj_avg_5", "speed_fig_adj_best_5"]:
        assert featured.loc[2, column] == pytest.approx(featured.loc[1, column])


def test_full_features_for_settled_rows_do_not_change_with_future_data():
    raw = _synthetic_raw(n_days=5, races_per_day=2, field=6)
    raw["sire"] = "Shared sire"
    raw["damsire"] = "Shared damsire"
    cutoff = sorted(raw.race_date.unique())[2]
    prefix = engineer_features(process_data(raw[raw.race_date <= cutoff]), save=False)
    full = engineer_features(process_data(raw), save=False)
    columns = sorted(set(prefix.select_dtypes(include=np.number)) & set(full.select_dtypes(include=np.number)))
    key = ["race_id", "horse_name"]
    expected = prefix.set_index(key)[columns].sort_index()
    actual = full[full.race_date <= cutoff].set_index(key)[columns].sort_index()
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, rtol=1e-5, atol=1e-6)
