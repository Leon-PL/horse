import numpy as np

from research.meta_learner.forecast_components import PairwiseBT


def toy(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [4, 3, 5]
    race = np.repeat(np.arange(len(sizes)), sizes)
    position = np.concatenate([rng.permutation(n) + 1 for n in sizes]).astype(float)
    position[2] = 0  # a faller
    teacher = np.concatenate([rng.dirichlet(np.ones(n)) for n in sizes])
    teacher[race == 1] = np.nan  # a race without a market falls back to finish order
    weights = np.repeat(rng.uniform(.5, 2, len(sizes)), sizes)
    return PairwiseBT(race, position, teacher, .75, weights), rng.normal(size=len(race))


def test_pairwise_gradient_matches_finite_differences():
    loss, scores = toy()
    _, grad, hess = loss.loss_and_gradients(scores)
    eps = 1e-6
    for i in range(len(scores)):
        up, down = scores.copy(), scores.copy()
        up[i] += eps
        down[i] -= eps
        numeric = (loss.loss_and_gradients(up)[0] - loss.loss_and_gradients(down)[0]) / (2 * eps)
        assert np.isclose(grad[i], numeric, atol=1e-6)
        numeric_h = (loss.loss_and_gradients(up)[1][i] - loss.loss_and_gradients(down)[1][i]) / (2 * eps)
        assert np.isclose(hess[i], numeric_h, atol=1e-5)


def test_pairwise_loss_prefers_the_finishing_order_without_a_market():
    loss, _ = toy()
    rows = slice(4, 7)  # race 1 has no market
    order = -loss.rank[rows]
    good, bad = np.zeros(12), np.zeros(12)
    good[rows], bad[rows] = order, -order
    assert loss.loss_and_gradients(good)[0] < loss.loss_and_gradients(bad)[0]
    assert not loss.has_teacher[1] and loss.has_teacher[0]
