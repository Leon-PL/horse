import pickle

import numpy as np
import pandas as pd
import pytest

from src.market_transfer_research import FEATURES, market_transfer_features
from src.prior_market_research import prior_market_features
from tests.test_prior_market_research import fixture


def test_turf_history_transfers_to_first_aw_with_an_explicit_switch():
    frame = fixture()
    frame.loc[frame.race_id.eq(4), "surface"] = "Tapeta"
    result = market_transfer_features(frame)
    aw = frame.race_id.eq(4)
    assert prior_market_features(frame).loc[aw, "horse_prior_market_support_3"].eq(0).all()
    assert result.loc[aw, FEATURES[2]].eq(3).all()
    assert result.loc[aw, FEATURES[3]].eq(1).all()
    assert result.loc[aw, FEATURES[4]].eq(0).all()
    assert result.loc[aw & frame.horse_id.eq(1), FEATURES[0]].gt(0).all()


def test_jump_code_and_flat_family_changes_are_distinguished():
    frame = fixture().assign(race_type="Hurdle", race_name="Handicap Hurdle")
    frame.loc[frame.race_id.eq(3), ["race_type", "race_name"]] = ["Chase", "Handicap Chase"]
    frame.loc[frame.race_id.eq(4), ["race_type", "race_name"]] = ["Flat", "Handicap"]
    result = market_transfer_features(frame)
    assert result.loc[frame.race_id.eq(3), FEATURES[3]].eq(1).all()
    assert result.loc[frame.race_id.eq(3), FEATURES[4]].eq(0).all()
    assert result.loc[frame.race_id.eq(4), FEATURES[4]].eq(1).all()


def test_current_future_and_row_order_cannot_change_earlier_features():
    frame = fixture()
    expected = market_transfer_features(frame)
    changed = frame.copy()
    mask = frame.race_id.ge(3)
    changed.loc[mask, ["odds", "official_rating", "won", "finish_position"]] = [200., 9999., 0, 0]
    changed.loc[mask, "finish_pos_label"] = "PU"
    pd.testing.assert_frame_equal(expected.iloc[:12], market_transfer_features(changed).iloc[:12])
    pd.testing.assert_frame_equal(expected.iloc[:12], market_transfer_features(frame.iloc[:12]))
    pd.testing.assert_frame_equal(expected, market_transfer_features(frame.sample(frac=1, random_state=6)).sort_index())


def test_pending_live_append_matches_full_and_preserves_input_state():
    frame = fixture()
    _, state = market_transfer_features(frame.iloc[:16], return_state=True)
    saved = pickle.dumps(state)
    card = frame.iloc[16:].assign(odds=np.nan, finish_position=0, won=0, finish_pos_label="")
    pd.testing.assert_frame_equal(market_transfer_features(card, state=state),
                                  market_transfer_features(frame).iloc[16:])
    assert pickle.dumps(state) == saved
    with pytest.raises(ValueError, match="earlier complete date"):
        market_transfer_features(frame, state=state)


def test_trend_uses_latest_actual_price_and_does_not_skip_a_missing_latest_book():
    frame = fixture()
    frame.loc[frame.race_id.eq(3) & frame.horse_id.eq(1), "odds"] = 1.5
    result = market_transfer_features(frame)
    assert result.loc[frame.race_id.eq(4) & frame.horse_id.eq(1), FEATURES[1]].gt(0).all()
    frame.loc[frame.race_id.eq(3) & frame.horse_id.eq(1), "odds"] = np.nan
    result = market_transfer_features(frame)
    assert result.loc[frame.race_id.eq(4), FEATURES[1]].isna().all()
    assert result.loc[frame.race_id.eq(4), FEATURES[2]].eq(2).all()


def test_missing_books_advance_three_start_window_and_expired_starts_are_unknown():
    frame = fixture()
    frame.loc[frame.race_id.between(2, 4) & frame.horse_id.eq(1), "odds"] = np.nan
    assert market_transfer_features(frame).loc[frame.race_id.eq(5), FEATURES[2]].eq(0).all()
    frame = fixture()
    frame.loc[frame.race_id.ge(2), "race_date"] += pd.Timedelta(days=400)
    result = market_transfer_features(frame).loc[frame.race_id.eq(2)]
    assert result[FEATURES[2]].eq(0).all()
    assert result[list(FEATURES[:2]) + list(FEATURES[3:])].isna().all().all()


def test_nonstarters_do_not_erase_history_and_ratings_never_enter_price_features():
    frame = fixture()
    frame.loc[frame.race_id.between(2, 4), ["finish_position", "won", "finish_pos_label"]] = [0, 0, "NR"]
    expected = market_transfer_features(frame)
    assert expected.loc[frame.race_id.eq(5), FEATURES[2]].eq(1).all()
    changed = frame.assign(official_rating=99999., has_official_rating=0)
    pd.testing.assert_frame_equal(expected, market_transfer_features(changed))
    pd.testing.assert_frame_equal(expected, market_transfer_features(frame.drop(columns="official_rating")))


def test_same_day_prices_from_other_races_cannot_enter_and_ambiguous_latest_is_rejected():
    frame = fixture()
    frame.loc[frame.race_id.eq(2), "race_date"] = frame.race_date.min()
    with pytest.raises(ValueError, match="ambiguous"):
        market_transfer_features(frame)
    frame.loc[frame.race_id.eq(2), "horse_id"] += 10
    frame.loc[frame.race_id.eq(2), "horse_name"] += "other"
    result = market_transfer_features(frame)
    assert result.iloc[:8][list(FEATURES[:2])].isna().all().all()
