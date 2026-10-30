import numpy as np
import pandas as pd
import pytest

from src.feature_semantics import class_number, history_coverage_features, jumps_mask, normalise_going, observed_rating
from src.data_processor import _GOING_MAP


@pytest.mark.parametrize("raw,expected", [
    ("Yielding (Good To Yielding In Places)", "Yielding"),
    ("Soft (Soft To Heavy In Places)", "Soft"),
    ("Soft (Heavy In Places )", "Heavy"),
    ("Good To Yielding (Good In Places)", "Yielding"),
    ("Heavy (Standing Water In Places)", "Heavy"),
    ("Good To Soft (Good  In Places)", "Soft"),
    ("Standard / Fast", "Fast"), ("Standard / Slow", "Good"),
    ("Fast", "Fast"), ("", "Unknown"), (None, "Unknown"), ("Mystery", "Unknown"),
])
def test_going_preserves_semantics_and_is_idempotent(raw, expected):
    actual = normalise_going(raw, _GOING_MAP)
    assert actual == expected
    assert normalise_going(actual, _GOING_MAP) == actual


def test_all_previously_explicit_going_mappings_survive():
    for raw, expected in _GOING_MAP.items():
        assert normalise_going(raw, _GOING_MAP) == expected


def test_class_and_observed_mark_do_not_invent_numbers():
    values = class_number(pd.Series(["Class 1", "3.0", "Grade 1", "Group 2", "", None, "9"]))
    np.testing.assert_allclose(values, [1, 3, np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
    frame = pd.DataFrame({"official_rating": [100, 100, 0, np.nan], "has_official_rating": [1, 0, 1, 0]})
    np.testing.assert_allclose(observed_rating(frame), [100, np.nan, np.nan, np.nan], equal_nan=True)


def test_race_type_does_not_depend_on_unrelated_dummy_categories():
    frame = pd.DataFrame({"race_type": ["Chase", "Flat", "Hurdle"]})
    expected = jumps_mask(frame)
    pd.testing.assert_series_equal(expected, jumps_mask(frame.assign(race_type_Hurdle=0)))
    assert expected.tolist() == [True, False, True]
    assert jumps_mask(pd.DataFrame({"race_type_Hurdle": [1, 0]})).tolist() == [True, False]


def test_unseen_experience_and_contradictory_recency_are_distinguished():
    frame = pd.DataFrame({"horse_prev_races": [0, 0, 4, 4, 4],
        "days_since_last_race": [60, 60, 50, 50, 50],
        "days_since_last_run": [24, 0, 20, 70, np.nan], "prior_form_runs": [5, 0, 5, 5, 5]})
    out = history_coverage_features(frame)
    np.testing.assert_allclose(out.observed_days_since_run, [24, np.nan, 20, 50, 50], equal_nan=True)
    assert out.source_shows_unseen_experience.tolist() == [1, 0, 0, 0, 0]
    assert out.source_shows_intervening_run.tolist() == [0, 0, 1, 0, 0]
    assert out.source_recency_conflict.tolist() == [0, 0, 0, 1, 0]
    pd.testing.assert_frame_equal(out, history_coverage_features(frame.assign(won=1, finish_position=1)))
