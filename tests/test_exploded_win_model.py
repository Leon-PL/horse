"""The 057 exploded win objective inside the production win learner."""
import pickle

import numpy as np
import pytest

import config
from src.model import RacePredictor, _RaceSoftmaxClassifier
from src.probability_calibration import race_temperature
from src.race_objectives import race_softmax, training_objective, winner_metric


def _races(n_races=240, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 11, size=n_races)
    ability = rng.normal(size=sizes.sum())
    x = np.column_stack([ability + rng.normal(scale=0.5, size=len(ability)), rng.normal(size=len(ability))])
    performance = ability + rng.gumbel(size=len(ability))
    bounds = np.r_[0, np.cumsum(sizes)]
    positions = np.zeros(len(ability))
    for a, b in zip(bounds[:-1], bounds[1:]):
        positions[a:b] = np.argsort(np.argsort(-performance[a:b])) + 1
    won = (positions == 1).astype(float)
    market = race_softmax(ability, bounds)
    return x, sizes, positions, won, 0.5 * market + 0.5 * won


def _learner():
    learner = RacePredictor(random_seed=7, win_calibration_method="race_temperature")
    learner._autotune_njobs = 1
    learner.eval_split_info = {}
    return learner


def test_races_without_a_single_winner_get_no_weight():
    groups = np.array([3, 3, 3])
    positions = np.array([2, 1, 3, 1, 1, 2, 2, 3, 0])   # clean, dead heat, no winner
    target = np.array([.3, .6, .1, .5, .5, 0, 0, 0, 0])
    loss = training_objective(groups, positions, target * 2, np.ones(9), (1.0, 0.25, 0.0625))
    np.testing.assert_allclose(loss.weights, [1, 1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(loss.first_target[:3], [.3, .6, .1])     # normalised within the race
    _, grad, _ = loss.loss_and_gradients(np.random.default_rng(0).normal(size=9))
    assert np.all(grad[3:] == 0) and np.any(grad[:3] != 0)


def test_winner_metric_skips_races_without_a_single_winner():
    scores = np.array([0.0, 1.0, 0.5, 2.0, 2.0])
    groups = np.array([3, 2])
    name, value, higher_is_better = winner_metric(groups, np.array([0, 1, 0, 1, 1]))(None, scores)
    expected = -np.log(race_softmax(scores[:3], np.array([0, 3]))[1])
    assert name == "race_log_loss" and not higher_is_better
    assert value == pytest.approx(expected)


def test_exploded_win_model_trains_early_stops_and_refits():
    x, sizes, positions, won, soft = _races()
    learner = _learner()
    split = int(np.cumsum(sizes)[179])
    races = learner._win_races(sizes, positions, 0, split, eval_rows=(split, len(x)))
    selected = learner._train_win_classifier(x[:split], won[:split], soft_y=soft[:split], params={"n_estimators": 60},
                                             eval_set=[(x[split:], won[split:])], objective="exploded", **races)
    assert isinstance(selected, _RaceSoftmaxClassifier)
    assert 1 <= selected.reg.best_iteration_ <= 60
    assert selected.reg.random_state == 7
    fitted = learner._refit_selected_model(selected, "classifier", learner._train_win_classifier, x, won, None,
                                           soft_y=soft, objective="exploded",
                                           **learner._win_races(sizes, positions, 0, len(x)))
    assert fitted.reg.n_estimators == selected.reg.best_iteration_
    bounds = np.r_[0, np.cumsum(sizes)]
    raw = fitted.reg.predict(x, raw_score=True)
    served = race_temperature(fitted.predict_proba(x)[:, 1], sizes, 1.0)
    np.testing.assert_allclose(served, race_softmax(raw, bounds), rtol=1e-10)
    restored = pickle.loads(pickle.dumps(fitted))     # saved runs are joblib pickles
    np.testing.assert_allclose(restored.predict_proba(x), fitted.predict_proba(x))
    # It learned the signal: the strongest-rated runner wins far more often than chance.
    top = np.maximum.reduceat(served, bounds[:-1])
    assert np.mean(served[won == 1] == np.repeat(top, sizes)[won == 1]) > 2 / sizes.mean()


def test_exploded_objective_needs_races_and_race_temperature():
    x, sizes, positions, won, soft = _races(n_races=40)
    learner = _learner()
    with pytest.raises(ValueError, match="groups"):
        learner._train_win_classifier(x, won, soft_y=soft, objective="exploded")
    platt = RacePredictor(win_calibration_method="platt_isotonic")
    platt._autotune_njobs = 1
    with pytest.raises(ValueError, match="race_temperature"):
        platt._train_win_classifier(x, won, soft_y=soft, objective="exploded",
                                    **platt._win_races(sizes, positions, 0, len(x)))
    with pytest.raises(ValueError, match="race boundaries"):
        learner._win_races(sizes, positions, 0, 3)


class _Scores:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict(self, _x, raw_score=False):
        assert raw_score
        return self.scores


def test_serving_is_a_temperature_softmax_even_for_a_wide_score_range():
    scores = np.array([3.0, 1.0, 0.0, -60.0, -61.0, -64.0])   # a weak race 60 logits below a strong one
    groups = np.array([3, 3])
    predictor = _learner()
    predictor.clf_model = _RaceSoftmaxClassifier(_Scores(scores))
    predictor.calibration_temp = 1.3
    served = predictor._calibrate_win_probs(predictor.clf_model.predict_proba(np.zeros((6, 1)))[:, 1], groups)
    np.testing.assert_allclose(served, race_softmax(scores / 1.3, np.array([0, 3, 6])), rtol=1e-12)


def test_cross_entropy_learner_remains_available():
    x, sizes, positions, won, soft = _races(n_races=60)
    model = _learner()._train_win_classifier(x, won, soft_y=soft, params={"n_estimators": 20},
                                             objective="cross_entropy")
    assert not isinstance(model, _RaceSoftmaxClassifier)
    assert np.all((model.predict_proba(x)[:, 1] > 0) & (model.predict_proba(x)[:, 1] < 1))


def test_production_config_pairs_the_exploded_objective_with_race_temperature():
    assert config.WIN_OBJECTIVE == "exploded"
    assert tuple(config.WIN_EXPLODED_STAGE_WEIGHTS) == (1.0, 0.25, 0.0625)
    assert config.WIN_CALIBRATION_METHOD == "race_temperature"
