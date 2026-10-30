import numpy as np
import pandas as pd
import pytest

from src.handicap_performance_research import (
    FEATURES, handicap_performance_features, performance_observations, poundage,
)


def sample():
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


def test_weight_adjusted_pairwise_figures_and_fixed_scales():
    f = sample().iloc[:4]
    p = performance_observations(f)
    assert p.performance.notna().all()
    # Carrying 1lb more and finishing 1 length behind at8f.
    assert p.performance.iloc[1]-p.performance.iloc[0] == pytest.approx(1-17.05/8)
    np.testing.assert_allclose(poundage([18, 24, 34, 16, 4], ["Hurdle"]*5), [1.01, .85, .7, 1.07, np.nan])


@pytest.mark.parametrize("column,value", [("has_official_rating", 0), ("has_observed_weight", 0),
    ("age", 3), ("beaten_lengths", np.nan), ("num_runners", 5)])
def test_invalid_anchor_is_unknown_not_a_zero_figure(column, value):
    f = sample().iloc[:4].copy()
    f.loc[1, column] = value
    assert performance_observations(f).performance.isna().all()


def test_future_and_current_outcomes_do_not_change_present_features():
    f = sample()
    expected = handicap_performance_features(f)
    changed = f.copy()
    changed.loc[changed.race_id.ge(3), "beaten_lengths"] *= 50
    pd.testing.assert_frame_equal(expected.iloc[:12], handicap_performance_features(changed).iloc[:12])
    pd.testing.assert_frame_equal(expected.iloc[:12], handicap_performance_features(f.iloc[:12]))
    pd.testing.assert_frame_equal(expected, handicap_performance_features(f.sample(frac=1, random_state=2)).sort_index())
    assert expected.iloc[:4][list(FEATURES[:2])].isna().all().all()


def test_seeded_live_card_matches_full_history_and_does_not_mutate_state():
    f = sample()
    first, state = handicap_performance_features(f.iloc[:16], return_state=True)
    import pickle
    before = pickle.dumps(state)
    card = f.iloc[16:].copy()
    card[["finish_position", "won"]] = 0
    card["finish_pos_label"] = ""
    card["beaten_lengths"] = np.nan
    live = handicap_performance_features(card, state=state)
    pd.testing.assert_frame_equal(live, handicap_performance_features(f).iloc[16:])
    assert pickle.dumps(state) == before


def test_dnf_advances_window_but_nonstarter_does_not():
    f = sample()
    f.loc[(f.horse_id.eq(4)) & f.race_id.between(2,4), ["finish_position", "beaten_lengths", "finish_pos_label"]] = [0, np.nan, "PU"]
    out = handicap_performance_features(f)
    assert out.loc[19, FEATURES[2]] == 0
    f.loc[(f.horse_id.eq(4)) & f.race_id.between(2,4), "finish_pos_label"] = "NR"
    assert handicap_performance_features(f).loc[19, FEATURES[2]] == 1


def test_surface_and_code_histories_are_separate_and_stale_values_expire():
    f = sample()
    f.loc[f.race_id.eq(3), "surface"] = "Tapeta"
    out = handicap_performance_features(f)
    assert out.loc[f.race_id.eq(3), FEATURES[2]].eq(0).all()
    f.loc[f.race_id.ge(2), "race_date"] += pd.Timedelta(days=400)
    assert handicap_performance_features(f).loc[f.race_id.eq(2), FEATURES[2]].eq(0).all()


def test_same_day_outcomes_are_excluded():
    f = sample()
    f.loc[f.race_id.eq(2), "race_date"] = f.race_date.min()
    out = handicap_performance_features(f)
    assert out.iloc[:8][list(FEATURES[:2])].isna().all().all()


def test_empty_history_is_valid():
    out, state = handicap_performance_features(sample().iloc[:0], return_state=True)
    assert out.empty and state["last_date"] is None
