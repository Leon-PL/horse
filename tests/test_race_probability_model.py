import numpy as np
import pandas as pd
import pytest

import config
from src.race_probability_model import RaceProbabilityPredictor, race_loss_gradient, race_objective, race_softmax


def test_group_loss_gradient_and_hessian_match_finite_differences():
    scores = np.array([.5, -.1, .2, .7, -.4])
    groups, labels, weights = np.array([2, 3]), np.array([1, 0, 0, 1, 0]), np.array([2., .5])
    loss, gradient = race_loss_gradient(scores, labels, groups, weights)
    epsilon = 1e-5
    expected = np.array([(race_loss_gradient(scores + np.eye(5)[i] * epsilon, labels, groups, weights)[0]
                          - race_loss_gradient(scores - np.eye(5)[i] * epsilon, labels, groups, weights)[0])
                         / (2 * epsilon) for i in range(5)])
    np.testing.assert_allclose(gradient, expected, atol=1e-9)
    class Dataset:
        def get_label(self):
            return labels
    objective = race_objective(groups, weights)
    actual, hessian = objective(scores, Dataset())
    np.testing.assert_allclose(actual / weights.sum(), gradient, atol=1e-12)
    numerical = np.array([(objective(scores + np.eye(5)[i] * epsilon, Dataset())[0][i]
                           - objective(scores - np.eye(5)[i] * epsilon, Dataset())[0][i]) / (2 * epsilon)
                          for i in range(5)])
    np.testing.assert_allclose(hessian, numerical, atol=1e-9)


def test_equal_race_loss_and_shift_invariance_with_unequal_fields():
    groups = [2, 4]
    scores = np.array([1., -1, 1, -1, -1, -1])
    labels = [1, 0, 1, 0, 0, 0]
    p = race_softmax(scores, groups)
    expected = (-np.log(p[0]) - np.log(p[2])) / 2
    assert race_loss_gradient(scores, labels, groups, [1, 1])[0] == pytest.approx(expected)
    np.testing.assert_allclose(p, race_softmax(scores + np.repeat([1000, -1000], groups), groups))


def history():
    dates = pd.date_range("2024-01-01", periods=80, freq="7D")
    rows = []
    for race, date in enumerate(dates):
        for horse in range(3):
            rows.append({"race_id": race, "race_date": date, "off_time": "14:00", "horse_name": f"H{horse}",
                         "finish_position": [1, 2, 0][(horse + race) % 3],
                         "won": int((horse + race) % 3 == 0), "num_runners": 3, "handicap": 0,
                         "odds": 3., "lengths_behind": float(horse),
                         "signal": np.nan if race % 11 == 0 else float((horse + race) % 3)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("kind", ["conditional_logit", "race_softmax_gbm"])
def test_native_cutoff_purge_full_refit_permutation_and_saved_replay(kind, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 5)
    frame = history()
    cutoff = frame.race_date.drop_duplicates().iloc[69]
    predictor = RaceProbabilityPredictor(kind, ["signal"], threads=2,
        params={"n_estimators": 25, "min_child_samples": 3, "num_leaves": 4, "learning_rate": .1})
    info = predictor.fit_as_of(frame, cutoff)
    assert info["training_runners"] == 70 * 3
    assert info["full_fit"]["training_runners"] == 70 * 3
    assert pd.Timestamp(info["calibration_base"]["last_training_date"]) <= pd.Timestamp(info["calibration_first_date"]) - pd.Timedelta(days=7)
    assert pd.Timestamp(info["training_last_date"]) == cutoff
    test = frame.loc[frame.race_date > cutoff]
    expected = predictor.predict_races(test).sort_values(["race_id", "horse_name"])
    shuffled = predictor.predict_races(test.sample(frac=1, random_state=3)).sort_values(["race_id", "horse_name"])
    np.testing.assert_allclose(expected.win_probability, shuffled.win_probability, atol=1e-12)
    np.testing.assert_allclose(expected.groupby("race_id").win_probability.sum(), 1, atol=1e-12)
    predictor.save(tmp_path / "research.joblib")
    restored = RaceProbabilityPredictor.load(tmp_path / "research.joblib")
    np.testing.assert_allclose(expected.win_probability, restored.predict_races(test).win_probability, atol=1e-12)
    altered = frame.copy()
    altered.loc[altered.race_date > cutoff, "signal"] = 1e8
    altered.loc[altered.race_date > cutoff, "won"] = 0
    replica = RaceProbabilityPredictor(kind, ["signal"], params=predictor.params, threads=2)
    replica.fit_as_of(altered, cutoff)
    np.testing.assert_allclose(expected.win_probability, replica.predict_races(test).win_probability, atol=1e-12)
