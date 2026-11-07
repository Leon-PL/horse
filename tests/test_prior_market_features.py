import pickle
import numpy as np
import pandas as pd
import pytest

from src.prior_market_features import FEATURES, past_market_observations, prior_market_features


def sample():
    """Five four-runner handicaps (copied from the retired handicap-research tests)."""
    rows = []
    for day in range(1, 6):
        for horse in range(1, 5):
            rows.append(dict(race_id=day, horse_id=horse, horse_name=str(horse),
                race_date=pd.Timestamp(2025, 1, day), race_type="Flat", race_name="Handicap",
                surface="Turf", age=6, handicap=1, num_runners=4, weight_lbs=130.+horse,
                has_observed_weight=1, official_rating=70.+horse, has_official_rating=1,
                finish_position=horse, won=int(horse == 1), finish_pos_label=str(horse),
                distance_furlongs=8., beaten_lengths=float(horse-1)))
    return pd.DataFrame(rows)


def fixture():
    f = sample().assign(odds=4., horse_elo=1500.)
    f.loc[f.horse_id.eq(1), "odds"] = 2.
    return f


def test_uniform_book_has_zero_relative_opinion_and_rival_quality_excludes_self():
    f = fixture().assign(odds=4.)
    np.testing.assert_allclose(past_market_observations(f).logrel, 0., atol=1e-12)
    assert past_market_observations(fixture()).loc[0,"logrel"] > 0
    assert past_market_observations(fixture()).loc[0,"opposition"] == 73.


def test_current_and_future_prices_or_outcomes_cannot_change_present_features():
    f = fixture()
    expected = prior_market_features(f)
    changed = f.copy()
    changed.loc[f.race_id.ge(3), ["odds", "official_rating", "finish_position", "won"]] = [100., 5000., 0, 0]
    changed.loc[f.race_id.ge(3), "finish_pos_label"] = "PU"
    pd.testing.assert_frame_equal(expected.iloc[:12], prior_market_features(changed).iloc[:12])
    pd.testing.assert_frame_equal(expected.iloc[:12], prior_market_features(f.iloc[:12]))
    pd.testing.assert_frame_equal(expected, prior_market_features(f.sample(frac=1, random_state=9)).sort_index())


def test_seeded_pending_card_matches_full_history_without_prices():
    f = fixture()
    _, state = prior_market_features(f.iloc[:16], return_state=True)
    saved = pickle.dumps(state)
    card = f.iloc[16:].copy().assign(odds=np.nan, finish_position=0, won=0, finish_pos_label="")
    pd.testing.assert_frame_equal(prior_market_features(card, state=state), prior_market_features(f).iloc[16:])
    assert pickle.dumps(state) == saved


def test_partial_book_is_unknown_for_every_runner_and_advances_window():
    f = fixture()
    f.loc[f.race_id.between(2,4) & f.horse_id.eq(1), "odds"] = np.nan
    assert past_market_observations(f).loc[f.race_id.eq(2),"logrel"].isna().all()
    assert prior_market_features(f).loc[f.race_id.eq(5), FEATURES[2]].eq(0).all()


def test_same_day_and_other_scale_prices_are_excluded():
    f = fixture()
    f.loc[f.race_id.eq(2), "race_date"] = f.race_date.min()
    assert prior_market_features(f).iloc[:8][list(FEATURES[:2])].isna().all().all()
    f.loc[f.race_id.eq(3), "surface"] = "Tapeta"
    assert prior_market_features(f).loc[f.race_id.eq(3), FEATURES[2]].eq(0).all()


def test_expired_prices_do_not_create_false_support():
    f = fixture()
    f.loc[f.race_id.ge(2), "race_date"] += pd.Timedelta(days=400)
    assert prior_market_features(f).loc[f.race_id.eq(2), FEATURES[2]].eq(0).all()


def test_missing_seed_boundary_is_rejected():
    f = fixture()
    _, state = prior_market_features(f, return_state=True)
    with pytest.raises(ValueError, match="earlier complete date"):
        prior_market_features(f, state=state)


def test_imputed_marks_never_become_rival_quality():
    f = fixture().iloc[:4].copy()
    f.loc[0, ["official_rating", "has_official_rating"]] = [9999., 0]
    observations = past_market_observations(f)
    assert observations.loc[0,"opposition"] == 73.
    assert observations.loc[1:,"opposition"].isna().all()
