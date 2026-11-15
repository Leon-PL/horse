from copy import deepcopy
import pickle

import numpy as np
import pandas as pd
import pytest

from src.established_form_features import FEATURES, STATE_KEY, established_form_features, mark_scale, race_discipline
from src.feature_engineer import add_established_form_features, counts_state_scope


def frame(rows):
    defaults = dict(horse_id=123, horse_name="Known Horse", race_type="Flat", race_name="Handicap",
        surface="Turf", handicap=1, official_rating=80., has_official_rating=1,
        weight_lbs=130., has_observed_weight=1, reported_jockey_claim_lbs=0., finish_position=2, num_runners=8,
        finish_pos_label="", off_time="14:00")
    return pd.DataFrame([{**defaults, "race_id": i, "race_date": f"2025-01-{i+1:02d}", **row}
                         for i, row in enumerate(rows)])


def test_cross_discipline_marks_are_not_comparable():
    f = frame([dict(official_rating=80, finish_position=1),
        dict(race_type="Hurdle", official_rating=120, finish_position=1),
        dict(race_type="Chase", official_rating=110),
        dict(race_type="Chase", official_rating=108, finish_position=1),
        dict(race_type="Chase", official_rating=105)])
    out = established_form_features(f)
    assert out.loc[2, ["or_vs_last_win_mark", "or_vs_last_place_mark", "or_vs_career_high_prev"]].isna().all()
    assert out.loc[4, "or_vs_last_win_mark"] == -3
    assert out.loc[4, "or_vs_last_place_mark"] == -3
    assert out.loc[4, "or_vs_career_high_prev"] == -5
    assert out.loc[2, "horse_discipline_exposure_share"] == 0


def test_turf_and_synthetic_marks_have_separate_histories():
    f = frame([dict(official_rating=80, finish_position=1),
               dict(surface="Tapeta", official_rating=70, finish_position=1), dict(official_rating=78)])
    out = established_form_features(f)
    assert np.isnan(out.loc[1, "or_vs_last_win_mark"])
    assert out.loc[2, "or_vs_last_win_mark"] == -2
    assert out.loc[2, "handicap_mark_vs_high_5"] == -2


def test_missing_marks_and_weights_are_not_zero_changes():
    f = frame([dict(finish_position=1, has_official_rating=0, has_observed_weight=0),
               dict(official_rating=75), dict(official_rating=90, has_official_rating=0)])
    out = established_form_features(f)
    assert out.loc[1, ["or_vs_last_win_mark", "handicap_weight_vs_last_win", "below_last_win_mark"]].isna().all()
    assert out.loc[1, "handicap_mark_support_5"] == 0
    assert out.loc[2, ["or_vs_last_place_mark", "or_vs_career_high_prev", "above_career_high_or"]].isna().all()


def test_last_winning_weight_uses_both_claims_and_retains_missing_values():
    f = frame([
        dict(finish_position=1, weight_lbs=138, reported_jockey_claim_lbs=7),
        dict(weight_lbs=137, reported_jockey_claim_lbs=3),
        dict(finish_position=1, reported_jockey_claim_lbs=np.nan),
        dict(weight_lbs=136, reported_jockey_claim_lbs=0),
    ])
    out = established_form_features(f)
    assert out.loc[1, "handicap_weight_vs_last_win"] == 3
    assert out.loc[2:, "handicap_weight_vs_last_win"].isna().all()
    _, state = established_form_features(f.iloc[:1], return_state=True)
    pd.testing.assert_frame_equal(out.iloc[1:], established_form_features(f.iloc[1:], state=state))


def test_conditions_win_is_not_a_winning_handicap_mark():
    f = frame([dict(handicap=0, finish_position=1), dict(official_rating=77)])
    out = established_form_features(f)
    assert out.loc[0, ["or_vs_last_win_mark", "handicap_discipline_runs_log"]].isna().all()
    assert np.isnan(out.loc[1, "or_vs_last_win_mark"])
    assert out.loc[1, "handicap_discipline_runs_log"] == 0
    assert out.loc[1, "horse_discipline_runs_log"] == np.log(2)


def test_career_high_survives_an_unrated_intervening_start():
    f = frame([dict(official_rating=90), dict(official_rating=200, has_official_rating=0), dict(official_rating=85)])
    out = established_form_features(f)
    assert out.loc[2, "or_vs_career_high_prev"] == -5
    assert out.loc[2, "handicap_mark_support_5"] == 1


def test_latest_winning_missing_mark_does_not_revive_an_older_win():
    f = frame([dict(finish_position=1), dict(finish_position=1, has_official_rating=0), dict(official_rating=75)])
    assert np.isnan(established_form_features(f).loc[2, "or_vs_last_win_mark"])


def test_current_and_future_outcomes_cannot_change_present_features():
    f = frame([{}, dict(race_type="Hurdle"), dict(race_type="Hurdle"), {}])
    expected = established_form_features(f)
    changed = f.copy()
    changed.loc[2:, ["finish_position", "finish_pos_label"]] = [0, "PU"]
    actual = established_form_features(changed)
    pd.testing.assert_frame_equal(expected.iloc[:3], actual.iloc[:3])
    pd.testing.assert_frame_equal(expected.iloc[:3], established_form_features(f.iloc[:3]))


def test_same_day_runners_only_see_prior_days_and_permutation_is_stable():
    f = frame([dict(finish_position=1), dict(race_date="2025-01-01", finish_position=3), {}])
    out = established_form_features(f)
    assert out.loc[:1, "horse_discipline_runs_log"].eq(0).all()
    assert out.loc[2, "horse_discipline_runs_log"] == np.log(3)
    shuffled = f.sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(out, established_form_features(shuffled).sort_index())


@pytest.mark.parametrize("status", ["", "NON-RUNNER", "WITHDRAWN"])
def test_pending_and_nonstarters_do_not_advance_history(status):
    f = frame([dict(finish_position=1), dict(finish_position=0, finish_pos_label=status), {}])
    out = established_form_features(f)
    assert out.loc[2, "horse_discipline_runs_log"] == np.log(2)
    assert out.loc[2, "horse_discipline_form_3"] == 1
    _, seed = established_form_features(f.iloc[:1], return_state=True)
    before = pickle.dumps(seed)
    _, after = established_form_features(f.iloc[1:2], state=seed, return_state=True)
    assert pickle.dumps(seed) == before
    assert after == seed


def test_nonfinish_is_experience_but_not_a_good_finishing_position():
    f = frame([dict(finish_position=0, finish_pos_label="F"), {}])
    out = established_form_features(f)
    assert out.loc[1, "horse_discipline_runs_log"] == np.log(2)
    assert out.loc[1, "horse_discipline_dnf_rate_shrunk"] == 1.5/11
    assert np.isnan(out.loc[1, "horse_discipline_form_3"])


def test_incremental_state_matches_batch_and_remains_immutable():
    f = frame([dict(finish_position=1), dict(race_type="Hurdle"), {}, dict(surface="Polytrack"), {}])
    expected = established_form_features(f)
    _, state = established_form_features(f.iloc[:2], return_state=True)
    before = deepcopy(state)
    actual = established_form_features(f.iloc[2:], state=pickle.loads(pickle.dumps(state)))
    pd.testing.assert_frame_equal(expected.iloc[2:], actual)
    assert state == before
    with pytest.raises(ValueError, match="complete seeded date"):
        established_form_features(f.iloc[1:], state=state)


def test_native_stage_uses_identical_state_and_values():
    f = frame([dict(finish_position=1), {}, dict(race_type="Hurdle"), {}])
    state = {}
    with counts_state_scope(state):
        add_established_form_features(f.iloc[:2].copy())
    assert STATE_KEY in state
    with counts_state_scope(state):
        actual = add_established_form_features(f.iloc[2:].copy())
    pd.testing.assert_frame_equal(actual[list(FEATURES)], established_form_features(f).iloc[2:][list(FEATURES)])


def test_float_and_text_ids_use_the_same_horse():
    f = frame([dict(horse_id="123", finish_position=1), dict(horse_id=123.), dict(horse_id=123)])
    assert established_form_features(f).horse_discipline_runs_log.tolist() == [0, np.log(2), np.log(3)]


def test_explicit_bumper_titles_split_flat_but_do_not_override_hurdles():
    f = frame([dict(race_name="Mares NH Flat Race"), dict(race_name="INH Flat Race"),
        dict(race_name="National Hunt Flat Race"), dict(race_name="Bumper", race_type="Hurdle"),
        dict(race_name="Summer Flat Race"), dict(race_name="Mares Novices' Limited Handicap Chase", race_type="Chase"),
        dict(race_name="Mares Maiden Hurdle", race_type="Hurdle")])
    assert race_discipline(f).tolist() == ["Bumper", "Bumper", "Bumper", "Hurdle", "Bumper", "Chase", "Hurdle"]
    out = established_form_features(f)
    assert out.race_is_bumper.tolist() == [1, 1, 1, 0, 1, 0, 0]
    assert out.loc[5, "race_is_novice"] == 1
    assert out.loc[6, "race_is_maiden"] == 1


def test_unknown_surface_cannot_supply_a_comparable_flat_mark():
    f = frame([dict(surface="???", finish_position=1), {}])
    assert pd.isna(mark_scale(f).iloc[0])
    assert np.isnan(established_form_features(f).loc[1, "or_vs_last_win_mark"])


def test_bumper_advertising_does_not_override_an_actual_handicap_title():
    f = frame([dict(race_name="Bumper Bonus Handicap"), dict(race_name="Champion Bumper (Grade 1)")])
    assert race_discipline(f).tolist() == ["Flat", "Bumper"]


def test_beginners_chase_has_maiden_eligibility_context():
    f = frame([dict(race_type="Chase", race_name="Irish EBF Beginners' Chase")])
    assert established_form_features(f).race_is_maiden.iloc[0] == 1


def test_recent_mark_window_is_bounded_to_five_actual_handicap_starts():
    f = frame([dict(official_rating=v) for v in [100, 90, 89, 88, 87, 86, 85]])
    out = established_form_features(f)
    assert out.loc[6, "handicap_mark_vs_high_5"] == -5
    assert out.loc[6, "handicap_mark_support_5"] == 5
    assert out.loc[6, "or_vs_career_high_prev"] == -15
