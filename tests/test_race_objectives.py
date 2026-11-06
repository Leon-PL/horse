import numpy as np
import pandas as pd
import pytest

from src.race_objectives import (CentredRegression, ExplodedSoftmax, centre_within, exploded_stages,
                                 fit_boosted, race_bounds, race_softmax, raw_scores)


def _numeric_gradient(fn, x, eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        up, down = x.copy(), x.copy()
        up[i] += eps
        down[i] -= eps
        grad[i] = (fn(up) - fn(down)) / (2 * eps)
    return grad


def test_stages_stop_at_dead_heats_missing_places_and_small_fields():
    # race A: clean 1-2-3 plus a non-finisher (0); race B: dead heat for 2nd;
    # race C: two runners, so only the winner stage exists.
    positions = [2, 1, 0, 3, 1, 2, 2, 3, 2, 1]
    chosen = exploded_stages(positions, [4, 4, 2], 3)
    assert chosen.tolist() == [[1, 0, 3], [4, -1, -1], [9, -1, -1]]


def test_stage_needs_two_runners_left():
    chosen = exploded_stages([1, 2, 3], [3], 3)
    assert chosen.tolist() == [[0, 1, -1]]


def test_exploded_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    race_ids = np.repeat(["a", "b"], [5, 4])
    positions = np.array([3, 1, 0, 2, 4, 2, 1, 3, 0])
    first = np.where(positions == 1, 1.0, 0.0) * 0.5
    first[:5] += 0.5 * np.array([.1, .4, .1, .3, .1])
    first[5:] += 0.5 * np.array([.25, .25, .25, .25])
    weights = np.repeat([0.7, 1.3], [5, 4])
    obj = ExplodedSoftmax(race_ids, positions, first, weights, (1.0, 0.5, 0.25))
    scores = rng.normal(size=9)
    _, grad, _ = obj.loss_and_gradients(scores)
    numeric = _numeric_gradient(lambda s: obj.loss_and_gradients(s)[0], scores)
    np.testing.assert_allclose(grad, numeric, atol=1e-6)


def test_flattened_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    race_ids = np.repeat(["a", "b"], [6, 4])
    positions = np.array([3, 1, 0, 2, 5, 4, 2, 1, 3, 4])
    first = (positions == 1).astype(float)
    weights = np.repeat([0.7, 1.3], [6, 4])
    obj = ExplodedSoftmax(race_ids, positions, first, weights, (1.0, 1.0, 0.5, 1.0, 1.0), (1.0, 0.8, 0.6, 0.4, 0.3))
    scores = rng.normal(size=10)
    _, grad, _ = obj.loss_and_gradients(scores)
    numeric = _numeric_gradient(lambda s: obj.loss_and_gradients(s)[0], scores)
    np.testing.assert_allclose(grad, numeric, atol=1e-6)


def test_flattened_loss_matches_a_direct_calculation():
    # one race: finishers 1-2-3-4 at rows 2, 0, 3, 1 and a non-finisher at row 4
    positions = np.array([2, 4, 1, 3, 0])
    scores = np.array([0.3, -0.2, 0.9, 0.1, 0.5])
    temps = np.array([1.0, 0.7, 0.5, 0.3])
    obj = ExplodedSoftmax(["a"] * 5, positions, (positions == 1).astype(float), np.ones(5), np.ones(4), temps)
    expected, field = 0.0, [0, 1, 2, 3, 4]
    for stage, row in enumerate([2, 0, 3, 1]):
        z = temps[stage] * scores[field]
        expected -= temps[stage] * scores[row] - np.log(np.exp(z).sum())
        field.remove(row)
    assert obj.loss_and_gradients(scores)[0] == pytest.approx(expected)


def test_temperatures_need_one_positive_value_per_stage():
    with pytest.raises(ValueError):
        ExplodedSoftmax(["a", "a"], [1, 2], [1.0, 0.0], [1, 1], (1.0,), (1.0, 0.5))
    with pytest.raises(ValueError):
        ExplodedSoftmax(["a", "a"], [1, 2], [1.0, 0.0], [1, 1], (1.0,), (0.0,))


def test_one_stage_is_the_race_softmax_gradient():
    race_ids = np.repeat(["a", "b"], [3, 3])
    positions = np.array([2, 1, 3, 1, 3, 2])
    won = (positions == 1).astype(float)
    obj = ExplodedSoftmax(race_ids, positions, won, np.ones(6), (1.0,))
    scores = np.array([0.2, -0.1, 0.4, 1.0, 0.0, -1.0])
    _, grad, hess = obj.loss_and_gradients(scores)
    p = race_softmax(scores, race_bounds(race_ids))
    np.testing.assert_allclose(grad, p - won)
    np.testing.assert_allclose(hess, p * (1 - p))


def test_non_finishers_are_never_chosen_but_stay_in_the_field():
    race_ids = np.array(["a"] * 4)
    positions = np.array([1, 0, 2, 3])
    obj = ExplodedSoftmax(race_ids, positions, (positions == 1).astype(float), np.ones(4), (0.0, 1.0))
    _, grad, _ = obj.loss_and_gradients(np.zeros(4))
    # stage 2 picks among rows 1-3: row 2 chosen, the non-finisher pushed down
    np.testing.assert_allclose(grad, [0.0, 1 / 3, 1 / 3 - 1, 1 / 3])


def test_soft_targets_must_sum_to_one_and_weights_be_per_race():
    with pytest.raises(ValueError):
        ExplodedSoftmax(["a", "a"], [1, 2], [0.7, 0.7], [1, 1], (1.0,))
    with pytest.raises(ValueError):
        ExplodedSoftmax(["a", "a"], [1, 2], [1.0, 0.0], [1.0, 2.0], (1.0,))


def test_slices_must_respect_race_boundaries():
    obj = ExplodedSoftmax(np.repeat(["a", "b"], [2, 2]), [1, 2, 2, 1], [1, 0, 0, 1], np.ones(4), (1.0,))
    obj.loss_and_gradients(np.zeros(2), 2, 4)
    with pytest.raises(ValueError):
        obj.loss_and_gradients(np.zeros(3), 1, 4)


def test_centred_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    race_ids = np.repeat(["a", "b", "c"], [3, 4, 2])
    target = rng.normal(size=9) * 5
    obj = CentredRegression(race_ids, target, np.repeat([0.5, 2.0, 1.1], [3, 4, 2]))
    pred = rng.normal(size=9)
    _, grad, _ = obj.loss_and_gradients(pred)
    numeric = _numeric_gradient(lambda p: obj.loss_and_gradients(p)[0], pred)
    np.testing.assert_allclose(grad, numeric, atol=1e-5)


def test_centred_loss_ignores_race_level():
    race_ids = np.repeat(["a", "b"], [2, 3])
    target = np.array([1.0, 3.0, 0.0, 5.0, 10.0])
    obj = CentredRegression(race_ids, target, np.ones(5))
    shifted = target + np.repeat([100.0, -7.0], [2, 3])
    assert obj.loss_and_gradients(shifted)[0] == pytest.approx(0.0)
    np.testing.assert_allclose(centre_within(shifted, race_bounds(race_ids)), centre_within(target, race_bounds(race_ids)))


def _synthetic(n_races=300, size=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_races * size, 3))
    ability = 1.5 * x[:, 0] - x[:, 1]
    noise = rng.gumbel(size=len(x))
    race_ids = np.repeat(np.arange(n_races), size).astype(str)
    order = pd.Series(ability + noise).groupby(race_ids).rank(ascending=False).to_numpy()
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.repeat(np.arange(n_races), size) // 3, unit="D")
    return x.astype(np.float32), order, race_ids, pd.DatetimeIndex(dates), ability


def test_exploded_fit_learns_ability_and_extra_stages_help():
    x, order, race_ids, dates, ability = _synthetic()
    won = (order == 1).astype(float)
    weights = np.ones(len(x))
    corr = {}
    for name, stages in {"winner": (1.0,), "exploded": (1.0, 1.0, 1.0)}.items():
        obj = ExplodedSoftmax(race_ids, order, won, weights, stages)
        model = fit_boosted(x, won, race_ids, dates, objective=obj.objective, metric=obj.metric,
                            capacity="screen", n_jobs=1)
        corr[name] = np.corrcoef(raw_scores(model, x), ability)[0, 1]
    assert corr["winner"] > 0.5
    assert corr["exploded"] > corr["winner"]


def test_native_fit_early_stops_and_refits():
    x, order, race_ids, dates, ability = _synthetic(n_races=400)
    target = -order.astype(float)
    obj = CentredRegression(race_ids, target, np.ones(len(x)))
    model, selected = fit_boosted(x, target, race_ids, dates, objective=obj.objective, metric=obj.metric,
                                  capacity="native", n_jobs=1, return_selected=True)
    assert selected is not None and model.selected_rounds_ >= 1
    assert np.corrcoef(raw_scores(model, x), ability)[0, 1] > 0.5
