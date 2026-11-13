"""Reproductions of objective errors, independent of any fitted model."""
import numpy as np
import pandas as pd
import pytest

from src.data_processor import clean_data, process_data
from src.feature_engineer import (
    _add_result_history_flags, add_class_change_features, add_pedigree_features,
    add_race_context_features, add_race_relative_signal_features,
    add_speed_figure_features, apply_conditional_feature_masks, counts_state_scope,
    engineer_features,
)
from src.live_prediction import feature_engineer_with_history_core
from tests.test_live_parity import _synthetic_raw


def transitions():
    return pd.DataFrame({"horse_id": [1]*5, "race_id": list("abcde"),
        "race_date": pd.date_range("2025-01-01", periods=5),
        "race_type": ["Flat", "Flat", "Flat", "Flat", "Hurdle"],
        "surface": ["Turf"]*5,
        "race_class": ["Class 5", "", "Class 3", "Class 2", "Class 1"],
        "official_rating": [70., 80., 90., 95., 130.],
        "has_official_rating": [1, 0, 1, 1, 1]})


def test_missing_class_and_imputed_mark_are_not_real_changes():
    out = add_class_change_features(transitions())
    np.testing.assert_allclose(out.class_change, [np.nan, np.nan, np.nan, -1, -1], equal_nan=True)
    np.testing.assert_allclose(out.or_change, [np.nan, np.nan, np.nan, 5, np.nan], equal_nan=True)


def test_flat_surface_switch_is_not_a_proven_handicap_mark_change():
    frame = transitions().iloc[:4].copy()
    frame["has_official_rating"] = 1
    frame["surface"] = ["Turf", "Polytrack", "Allweather", "Turf"]
    out = add_class_change_features(frame)
    np.testing.assert_allclose(out.or_change, [np.nan, np.nan, 10., np.nan], equal_nan=True)


def test_unknown_flat_surface_cannot_prove_comparable_mark_changes():
    frame = transitions().iloc[:4].copy()
    frame["surface"] = None
    frame["has_official_rating"] = 1
    assert add_class_change_features(frame).or_change.isna().all()


def test_class_and_mark_changes_resume_without_reviving_an_older_known_value():
    frame = transitions()
    expected = add_class_change_features(frame.copy())
    state = {}
    actual = []
    for part in (frame.iloc[:2], frame.iloc[2:4], frame.iloc[4:]):
        with counts_state_scope(state):
            actual.append(add_class_change_features(part.copy()))
    pd.testing.assert_frame_equal(expected, pd.concat(actual))


def test_all_unknown_changes_still_produce_selected_columns():
    frame = transitions().iloc[:2].copy()
    frame["race_class"] = ""
    out = add_race_relative_signal_features(add_class_change_features(frame))
    assert out[["class_change_vs_field", "or_change_vs_field"]].isna().all().all()


def test_unknown_class_is_not_assumed_class_five_speed():
    frame = transitions().assign(horse_name="Same", race_class="", lengths_behind=1.,
        beaten_lengths=1., finish_position=2, distance_furlongs=8.)
    out = add_speed_figure_features(frame)
    assert out[["speed_fig_adj_best_5", "speed_z_class_best_5"]].isna().all().all()
    assert out.speed_fig_best_5.iloc[1:].notna().all()


def test_real_synthetic_surface_labels_count_in_pedigree_history():
    frame = _synthetic_raw(n_days=4, races_per_day=1, field=4)
    frame["sire"] = "One sire"
    frame["surface"] = np.tile(["Polytrack", "Allweather", "Fibresand", "Turf"], 4)
    source = _add_result_history_flags(process_data(frame))
    out = add_pedigree_features(source)
    assert out.loc[out.race_date.eq(out.race_date.max()), "sire_aw_win_rate_shrunk"].nunique() == 1
    assert not out.sire_aw_win_rate_shrunk.eq(.1).all()


def test_four_runner_place_prior_matches_top_three_and_unknown_stalls_stay_missing():
    raw = _synthetic_raw(n_days=2, races_per_day=1, field=4)
    raw["race_type"] = "Flat"  # Includes no-stalls bumpers under the legacy taxonomy.
    raw["draw"] = 0
    out = add_race_context_features(_add_result_history_flags(process_data(raw)))
    assert out.base_place_rate.eq(.75).all()
    out = apply_conditional_feature_masks(out)
    assert out[["draw_x_field_size", "track_draw_win_rate"]].isna().all().all()


def test_impossible_historical_weight_does_not_survive_cleaning():
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=4)
    raw.loc[0, "weight_lbs"] = 1
    expected = raw.loc[1:, "weight_lbs"].median()
    assert clean_data(raw).loc[0, "weight_lbs"] == expected
    assert raw.loc[0, "weight_lbs"] == 1


def test_field_weight_comparisons_need_only_one_observed_rival():
    raw = _synthetic_raw(n_days=1, races_per_day=2, field=4)
    raw["weight_lbs"] = 130.
    raw["reported_jockey_claim_lbs"] = [0., 5., np.nan, np.nan, 0., np.nan, np.nan, np.nan]
    out = add_race_context_features(process_data(raw)).reindex(raw.index)
    # An unverified claim falls back to the allocated weight, so every observed runner is compared.
    np.testing.assert_allclose(out.weight_vs_field, [1.25, -3.75, 1.25, 1.25, 0., 0., 0., 0.])
    np.testing.assert_allclose(out.weight_rank, [1., 4., 1., 1., 1., 1., 1., 1.])


def test_single_code_single_sex_card_has_fixed_selected_categorical_inputs():
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=4)
    raw["race_type"], raw["sex"] = "Flat", "G"
    out = process_data(raw)
    assert out.race_type_Chase.eq(0).all()
    assert out.sex_Female.eq(0).all()


def test_recleaning_cannot_turn_imputed_rating_or_weight_into_an_observation():
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=4)
    raw.loc[0, ["official_rating", "weight_lbs"]] = [0, 1]
    once = clean_data(raw)
    twice = clean_data(once)
    for col in ["official_rating", "has_official_rating", "weight_lbs", "has_observed_weight"]:
        pd.testing.assert_series_equal(once[col], twice[col])
    assert twice.loc[0, "has_official_rating"] == 0
    assert twice.loc[0, "has_observed_weight"] == 0


def test_missing_inputs_have_matching_historical_and_pending_card_features():
    raw = _synthetic_raw(n_days=6, races_per_day=2, field=6)
    raw["race_class"] = ""
    raw["official_rating"] = 0
    raw.loc[raw.index % 4 == 0, "weight_lbs"] = np.nan
    raw.loc[raw.index % 5 == 0, "draw"] = np.nan
    raw.loc[raw.index % 7 == 0, "lengths_behind"] = np.nan
    raw["going"] = "Yielding (Good To Yielding In Places)"
    day = raw.race_date.max()
    expected = engineer_features(process_data(raw), save=False)
    cards = raw.loc[raw.race_date.eq(day)].copy()
    cards[["finish_position", "won"]] = 0
    cards["lengths_behind"] = np.nan
    actual = feature_engineer_with_history_core(process_data(raw.loc[raw.race_date.lt(day)]),
        process_data(cards)).set_index(["race_id", "horse_id"])
    expected = expected.loc[expected.race_date.eq(day)].set_index(["race_id", "horse_id"])
    cols = ["class_change_vs_field", "dist_x_class_change", "or_change_vs_field", "speed_fig_best_5_vs_field",
            "weight_change_vs_field", "draw_x_field_size", "sire_aw_edge", "jockey_going_wr"]
    cols = [c for c in cols if c in expected and c in actual]
    assert len(cols) >= 6
    pd.testing.assert_frame_equal(expected[cols].sort_index(), actual[cols].sort_index(), check_dtype=False)
