import pickle
import numpy as np
import pandas as pd

from src.trip_suitability_research import FEATURES, trip_suitability_features
from tests.test_handicap_performance_research import sample


def test_current_future_outcomes_and_permutations_are_invariant():
    f = sample()
    expected = trip_suitability_features(f)
    changed = f.copy()
    changed.loc[f.race_id.ge(3), ["won", "finish_position"]] = [0, 0]
    changed.loc[f.race_id.ge(3), "finish_pos_label"] = "PU"
    pd.testing.assert_frame_equal(expected.iloc[:12], trip_suitability_features(changed).iloc[:12])
    pd.testing.assert_frame_equal(expected.iloc[:12], trip_suitability_features(f.iloc[:12]))
    pd.testing.assert_frame_equal(expected, trip_suitability_features(f.sample(frac=1, random_state=4)).sort_index())


def test_identical_trips_have_no_specialist_edge():
    out = trip_suitability_features(sample())
    np.testing.assert_allclose(out.loc[out.horse_trip_specialist_edge_vs_field.notna(), FEATURES[1]], 0, atol=1e-14)
    assert out.iloc[:4][list(FEATURES[:2])].isna().all().all()


def test_two_mile_and_three_mile_are_different_and_codes_are_separate():
    f = sample()
    f[["race_type", "distance_furlongs"]] = ["Hurdle", 16.]
    f.loc[f.race_id.eq(3), "distance_furlongs"] = 24.
    out = trip_suitability_features(f)
    assert out.loc[f.race_id.eq(3), FEATURES[2]].max() < .05
    assert out.loc[f.race_id.eq(2), FEATURES[2]].min() == np.log(2)
    f.loc[f.race_id.eq(3), "race_type"] = "Chase"
    assert trip_suitability_features(f).loc[f.race_id.eq(3), FEATURES[2]].eq(0).all()


def test_seeded_live_matches_full_and_keeps_state_unchanged():
    f = sample()
    _, state = trip_suitability_features(f.iloc[:16], return_state=True)
    saved = pickle.dumps(state)
    card = f.iloc[16:].copy().assign(finish_position=0, won=0, finish_pos_label="")
    pd.testing.assert_frame_equal(trip_suitability_features(card, state=state), trip_suitability_features(f).iloc[16:])
    assert pickle.dumps(state) == saved


def test_nonstarters_do_not_create_support_and_stale_runs_expire():
    f = sample()
    f.loc[f.race_id.eq(1), ["finish_position", "won", "finish_pos_label"]] = [0, 0, "NR"]
    assert trip_suitability_features(f).loc[f.race_id.eq(2), FEATURES[2]].eq(0).all()
    f.loc[f.race_id.ge(3), "race_date"] += pd.Timedelta(days=800)
    assert trip_suitability_features(f).loc[f.race_id.eq(3), FEATURES[2]].eq(0).all()


def test_missing_distance_is_unknown_and_same_day_is_excluded():
    f = sample()
    f.loc[f.race_id.eq(2), "race_date"] = f.race_date.min()
    assert trip_suitability_features(f).iloc[:8][list(FEATURES[:2])].isna().all().all()
    f.loc[8, "distance_furlongs"] = np.nan
    assert trip_suitability_features(f).loc[8].isna().all()
