import pickle

import numpy as np
import pandas as pd
import pytest

from src.market_finish_surprise_research import FEATURES, finish_surprise_observations, market_finish_surprise_features
from tests.test_prior_market_research import fixture


def test_uniform_prices_give_the_expected_rank_residual_and_ties_are_half():
    frame = fixture().iloc[:4].assign(odds=4.)
    observation = finish_surprise_observations(frame)
    np.testing.assert_allclose(observation.expected_fraction_beaten, .5)
    np.testing.assert_allclose(observation.finish_surprise, [.5, 1/6, -1/6, -.5])
    frame.finish_position = [1, 2, 2, 4]
    np.testing.assert_allclose(finish_surprise_observations(frame).finish_surprise, [.5, 0, 0, -.5])


def test_an_unfancied_winner_exceeds_expectation_more_than_a_favourite():
    frame = fixture().iloc[:4].copy()
    favourite = finish_surprise_observations(frame).finish_surprise.iloc[0]
    frame.finish_position = [4, 2, 3, 1]
    frame.won = [0, 0, 0, 1]
    outsider = finish_surprise_observations(frame).finish_surprise.iloc[3]
    assert outsider > favourite > 0
    np.testing.assert_allclose(finish_surprise_observations(frame).finish_surprise.sum(), 0., atol=1e-12)


def test_nonfinishers_are_not_ranked_last_or_used_to_reward_survivors():
    frame = fixture().iloc[:4].assign(odds=4.)
    frame.loc[3, ["finish_position", "finish_pos_label"]] = [0, "PU"]
    surprise = finish_surprise_observations(frame).finish_surprise
    np.testing.assert_allclose(surprise.iloc[:3], [.5, 0, -.5])
    assert np.isnan(surprise.iloc[3])
    frame.loc[1:, ["finish_position", "finish_pos_label"]] = [0, "PU"]
    assert finish_surprise_observations(frame).finish_surprise.isna().all()


def test_partial_price_books_are_unknown_and_rating_units_do_not_enter():
    frame = fixture()
    expected = finish_surprise_observations(frame)
    pd.testing.assert_frame_equal(expected, finish_surprise_observations(frame.assign(odds=frame.odds*2)))
    pd.testing.assert_frame_equal(expected, finish_surprise_observations(frame.assign(official_rating=99999., has_official_rating=0)))
    frame.loc[0, "odds"] = np.nan
    assert finish_surprise_observations(frame).iloc[:4].isna().all().all()


def test_current_future_results_prices_and_row_order_do_not_change_present_features():
    frame = fixture()
    expected = market_finish_surprise_features(frame)
    changed = frame.copy()
    changed.loc[frame.race_id.ge(3), ["odds", "finish_position", "won", "finish_pos_label"]] = [100., 0, 0, "PU"]
    pd.testing.assert_frame_equal(expected.iloc[:12], market_finish_surprise_features(changed).iloc[:12])
    pd.testing.assert_frame_equal(expected.iloc[:12], market_finish_surprise_features(frame.iloc[:12]))
    pd.testing.assert_frame_equal(expected, market_finish_surprise_features(frame.sample(frac=1, random_state=4)).sort_index())


def test_pending_seeded_card_matches_full_features_without_mutating_history():
    frame = fixture()
    _, state = market_finish_surprise_features(frame.iloc[:16], return_state=True)
    saved = pickle.dumps(state)
    card = frame.iloc[16:].assign(odds=np.nan, won=0, finish_position=0, finish_pos_label="")
    pd.testing.assert_frame_equal(market_finish_surprise_features(card, state=state),
                                  market_finish_surprise_features(frame).iloc[16:])
    assert pickle.dumps(state) == saved
    with pytest.raises(ValueError, match="earlier complete date"):
        market_finish_surprise_features(frame, state=state)


def test_scale_and_expiry_are_respected_and_missing_performances_advance_the_window():
    frame = fixture()
    frame.loc[frame.race_id.eq(4), "surface"] = "Tapeta"
    assert market_finish_surprise_features(frame).loc[frame.race_id.eq(4), FEATURES[2]].eq(0).all()
    frame = fixture()
    frame.loc[frame.race_id.ge(2), "race_date"] += pd.Timedelta(days=400)
    assert market_finish_surprise_features(frame).loc[frame.race_id.eq(2), FEATURES[2]].eq(0).all()
    frame = fixture()
    frame.loc[frame.race_id.between(2, 4) & frame.horse_id.eq(1), "odds"] = np.nan
    assert market_finish_surprise_features(frame).loc[frame.race_id.eq(5), FEATURES[2]].eq(0).all()


def test_same_day_data_are_excluded_and_ambiguous_multiple_starts_rejected():
    frame = fixture()
    frame.loc[frame.race_id.eq(2), "race_date"] = frame.race_date.min()
    with pytest.raises(ValueError, match="ambiguous"):
        market_finish_surprise_features(frame)
    frame.loc[frame.race_id.eq(2), "horse_id"] += 10
    frame.loc[frame.race_id.eq(2), "horse_name"] += "other"
    assert market_finish_surprise_features(frame).iloc[:8][list(FEATURES[:2])].isna().all().all()
