import numpy as np
import pandas as pd
import pytest

from src.feature_semantics import (class_number, history_coverage_features, jumps_mask, normalise_going,
                                   observed_rating, rider_restriction_features)
from src.data_processor import _GOING_MAP


@pytest.mark.parametrize("raw,expected", [
    ("Yielding (Good To Yielding In Places)", "Yielding"),
    ("Soft (Soft To Heavy In Places)", "Soft"),
    ("Soft (Heavy In Places )", "Heavy"),
    ("Good To Yielding (Good In Places)", "Yielding"),
    ("Heavy (Standing Water In Places)", "Heavy"),
    ("Good To Soft (Good  In Places)", "Yielding"), ("Good To Soft", "Yielding"),
    ("Soft (Yielding To Soft In Places)", "Soft"),
    ("Standard", "Standard"), ("Standard / Fast", "Standard"), ("Standard / Slow", "Slow"),
    ("Standard (Good To Firm In Places)", "Standard"), ("Standard / Slow (Firm In Places)", "Slow"),
    ("Good To Firm", "Fast"),
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



def test_rider_restrictions_use_the_race_title_for_every_runner():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 3, 4, 5, 6, 7, 8],
        "race_name": ["Connacht Hotel (Q.R.) Handicap", None, "NH Flat Race (Amateurs/Conditionals)",
                      "Mares' Handicap Hurdle (CJs' & Amateurs)", "Claiming Stakes",
                      "John Mabbs Amateur Night In Cairo EBF Maiden Stakes", "",
                      "Amateur Riders' Handicap", "Conditional Jockeys’ Handicap"],
    })
    flags = rider_restriction_features(frame)
    np.testing.assert_allclose(flags.race_amateur_riders, [1, 1, 1, 1, 0, 0, np.nan, 1, 0])
    np.testing.assert_allclose(flags.race_conditional_jockeys, [0, 0, 1, 1, 0, 0, np.nan, 0, 1])
    np.testing.assert_allclose(flags.race_apprentice_jockeys, [0, 0, 0, 0, 0, 0, np.nan, 0, 0])



def test_amateur_restrictions_account_for_race_conditions_and_date():
    frame = pd.DataFrame({
        "race_id": range(7),
        "race_date": ["2025-03-14", "2021-03-19", "2021-03-19", "2021-01-18",
                      "2024-03-12", "2025-03-11", "2026-03-12"],
        "track": ["Cheltenham", "Cheltenham", "Downpatrick", "Kempton",
                  "Cheltenham", "Cheltenham", "Cheltenham"],
        "race_name": ["Festival Open Hunters' Chase", "Festival Open Hunters' Chase",
                      "Open Hunters' Chase", "Amateur Jockeys' Handicap",
                      "National Hunt Challenge Cup Novices' Chase",
                      "National Hunt Challenge Cup Novices' Handicap Chase",
                      "Fulke Walwyn Kim Muir Challenge Cup Handicap Chase"],
    })
    assert rider_restriction_features(frame).race_amateur_riders.tolist() == [1, 0, 1, 0, 1, 0, 1]


def test_claim_adjusted_weight_requires_observed_allocation_and_claim():
    from src.feature_semantics import claim_adjusted_weight

    f = pd.DataFrame({"weight_lbs": [140, 140, 140, 140, 1, 140, 140],
                      "reported_jockey_claim_lbs": [7, 0, None, 3, 3, -5, 3.5],
                      "has_observed_weight": [1, 1, 1, 0, 1, 1, 1]})
    np.testing.assert_allclose(claim_adjusted_weight(f), [133, 140, np.nan, np.nan, np.nan, np.nan, np.nan])
    assert claim_adjusted_weight(f.drop(columns="reported_jockey_claim_lbs")).isna().all()
