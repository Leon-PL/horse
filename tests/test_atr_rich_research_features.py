"""Measurement contracts and point-in-time isolation for new ATR candidates."""
import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.research.atr_rich_features import (
    derive_run_metrics, build_prior_features, RUN_FEATURES, PRIOR_FEATURES, COUNT, AGE,
)


def observation(day="2026-01-01", horse="101", race="1", **changes):
    return {"horse_id": horse, "race_id": race, "race_date": day, "off_time": "14:00",
        "atr_acceleration_0_30_seconds": 4.2, "atr_finishing_speed_pct": 105.0,
        "atr_race_finishing_speed_pct": 100.0, "atr_top_speed_mph": 44.0, "atr_final_speed_mph": 38.0,
        "atr_sectional_speeds": [30.0, 35.0, 40.0],
        "atr_sectional_speed_labels": ["Start to 3f", "3f to 2f", "2f to Finish"],
        "atr_stride_markers": ["3f", "2f", "1f", "Fin."],
        "atr_stride_frequency_markers": ["3f", "2f", "1f", "Fin."],
        "atr_stride_length": [30.0, 28.0, 26.0, 24.0],
        "atr_stride_frequency": [2.0, 2.0, 2.2, 2.2], **changes}


def test_seven_metrics_have_explicit_units_and_formulas():
    row = derive_run_metrics(pd.DataFrame([observation()])).iloc[0]
    assert row[RUN_FEATURES[0]] == 4.2
    assert row[RUN_FEATURES[1]] == 5.0
    assert row[RUN_FEATURES[2]] == pytest.approx(-5.0)
    assert row[RUN_FEATURES[3]] == pytest.approx(10.0)
    assert row[RUN_FEATURES[4]] == pytest.approx(100*(25/29-1))
    assert row[RUN_FEATURES[5]] == pytest.approx(10.0)
    assert row[RUN_FEATURES[6]] == pytest.approx(100*(25/29*1.1-1))


def test_curves_align_by_marker_and_support_json_independently():
    original = observation()
    altered = observation(atr_stride_frequency_markers=["Fin.", "1f", "2f", "3f"],
                          atr_stride_frequency=[2.2, 2.2, 2.0, 2.0])
    altered = {k: json.dumps(v) if isinstance(v, list) else v for k, v in altered.items()}
    assert_frame_equal(derive_run_metrics(pd.DataFrame([original])), derive_run_metrics(pd.DataFrame([altered])))


def test_missing_closing_section_is_not_replaced_by_an_earlier_speed():
    row = derive_run_metrics(pd.DataFrame([observation(atr_sectional_speeds=[30.0, 35.0, None])])).iloc[0]
    assert np.isnan(row[RUN_FEATURES[2]]) and np.isnan(row[RUN_FEATURES[3]])


def test_instantaneous_mph_and_acceleration_are_excluded_from_segment_speeds():
    row = observation(atr_sectional_speeds=[30, 35, 40, 1000, 900, 1.2],
        atr_sectional_speed_labels=["Start to 3f", "3f to 2f", "2f to Finish", "Top speed mph", "Final mph", "0 to 30 mph"])
    result = derive_run_metrics(pd.DataFrame([row])).iloc[0]
    assert result[RUN_FEATURES[2]] == pytest.approx(-5.0)
    assert result[RUN_FEATURES[3]] == pytest.approx(10.0)


@pytest.mark.parametrize("labels", [["3f", "2f", "1f"], ["3f", "2f", "Fin.", "Fin."], ["x", "y", "z", "Fin."]])
def test_ambiguous_stride_markers_remain_unknown(labels):
    row = derive_run_metrics(pd.DataFrame([observation(atr_stride_markers=labels)])).iloc[0]
    assert np.isnan(row[RUN_FEATURES[4]]) and np.isnan(row[RUN_FEATURES[6]])


def test_unknowns_and_invalid_native_measurements_stay_missing_without_mutation():
    frame = pd.DataFrame([observation(atr_acceleration_0_30_seconds=0, atr_final_speed_mph=np.inf)])
    before = frame.copy(deep=True)
    result = derive_run_metrics(frame)
    assert np.isnan(result.iloc[0][RUN_FEATURES[0]]) and np.isnan(result.iloc[0][RUN_FEATURES[2]])
    assert_frame_equal(frame, before)
    assert derive_run_metrics(pd.DataFrame(index=["empty"])).isna().all().all()


def test_publication_delay_excludes_exact_time_and_current_race_data():
    history = pd.DataFrame([observation()])
    targets = pd.DataFrame([
        observation("2026-01-02", race="2"),
        observation("2026-01-03", race="3"),
        observation("2026-01-03", race="4", off_time="14:01"),
    ], index=[8, 2, 8])
    result = build_prior_features(history, targets)
    assert result.index.tolist() == [8, 2, 8]
    assert result[COUNT].tolist() == [0, 0, 1]
    assert result[PRIOR_FEATURES[0]].iloc[2] == 4.2
    assert result[AGE].iloc[2] > 2
    assert result[list(PRIOR_FEATURES)].iloc[:2].isna().all().all()


def test_last_three_actual_races_include_runs_without_atr_data():
    rows = [observation("2026-01-01", race="1", atr_acceleration_0_30_seconds=1.0),
            observation("2026-01-05", race="2", atr_acceleration_0_30_seconds=3.0),
            {"horse_id": "101", "race_id": "3", "race_date": "2026-01-09", "off_time": "14:00"},
            observation("2026-01-13", race="4", atr_acceleration_0_30_seconds=9.0)]
    target = pd.DataFrame([observation("2026-01-20", race="5")])
    result = build_prior_features(pd.DataFrame(rows), target)
    assert result.iloc[0][PRIOR_FEATURES[0]] == 6.0  # median of 3 and 9; the old 1 is outside the window
    assert result.iloc[0][COUNT] == 2


def test_current_labels_and_future_appends_cannot_change_earlier_predictions():
    frame = pd.DataFrame([observation("2026-01-01", race="1"), observation("2026-01-10", race="2")])
    before = build_prior_features(frame).iloc[1:]
    changed = frame.copy(deep=True)
    changed.loc[1, ["atr_acceleration_0_30_seconds", "atr_finishing_speed_pct", "atr_final_speed_mph"]] = [99, 200, 99]
    changed["won"] = [0, 1]
    changed["finish_position"] = [9, 1]
    changed["bsp"] = [1000, 1.01]
    future = pd.DataFrame([observation("2026-02-10", race="3", atr_acceleration_0_30_seconds=0.1)])
    after = build_prior_features(pd.concat([changed, future], ignore_index=True)).iloc[1:2]
    assert_frame_equal(before, after)


def test_names_do_not_link_different_canonical_horses():
    history = pd.DataFrame([observation(horse="101", horse_name="Same Name")])
    targets = pd.DataFrame([observation("2026-01-10", horse="202", horse_name="Same Name")])
    result = build_prior_features(history, targets)
    assert result[COUNT].iloc[0] == 0 and result[list(PRIOR_FEATURES)].isna().all().all()


def test_unknown_identity_or_invalid_clock_does_not_receive_history():
    history = pd.DataFrame([observation()])
    targets = pd.DataFrame([observation("2026-01-10", horse=None), observation("2026-01-10", off_time="25:99"),
                            observation("2026-01-10", race=None)])
    result = build_prior_features(history, targets)
    assert result[COUNT].eq(0).all() and result[list(PRIOR_FEATURES)].isna().all().all()


def test_duplicate_appearances_are_rejected():
    frame = pd.DataFrame([observation(), observation()])
    with pytest.raises(ValueError, match="Ambiguous"):
        build_prior_features(frame)


@pytest.mark.parametrize("lag", ["0h", "24h", "NaT"])
def test_too_short_or_unknown_availability_delay_is_rejected(lag):
    with pytest.raises(ValueError, match="48 hours"):
        build_prior_features(pd.DataFrame([observation()]), availability_lag=lag)
