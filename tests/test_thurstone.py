import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.thurstone import (CAP_LBS, figure_rows, fit_figure_model, fit_spread_scale, pounds_behind, race_halves,
                           win_probabilities, winner_log_loss)


def test_pounds_behind_scales_with_distance_and_caps():
    out = pounds_behind([1.0, 1.0, 2.0, 100.0, np.nan], [5.0, 16.0, 4.0, 8.0, 8.0])
    np.testing.assert_allclose(out[:4], [3.0, 15 / 16, 6.0, CAP_LBS])
    assert np.isnan(out[4])


def test_two_runner_win_chance_matches_closed_form():
    mu, sigma = np.array([2.0, 0.0]), np.array([3.0, 5.0])
    p = win_probabilities(mu, sigma, [2])
    expected = norm.cdf(2.0 / np.hypot(3.0, 5.0))
    assert p[0] == pytest.approx(expected, abs=1e-4)
    assert p.sum() == pytest.approx(1.0)


def test_win_chances_match_simulation_and_reward_uncertainty():
    rng = np.random.default_rng(0)
    mu = np.array([5.0, 3.0, 0.0, -2.0, -2.0])
    sigma = np.array([4.0, 4.0, 4.0, 4.0, 12.0])
    p = win_probabilities(mu, sigma, [5])
    draws = rng.normal(mu, sigma, size=(400_000, 5))
    simulated = np.bincount(draws.argmax(axis=1), minlength=5) / len(draws)
    np.testing.assert_allclose(p, simulated, atol=4e-3)
    assert p[4] > 2 * p[3]  # same mean, wider spread: more chance of the top draw


def test_mixed_field_sizes_are_independent():
    mu = np.array([1.0, 0.0, 3.0, 2.0, 1.0])
    sigma = np.full(5, 2.0)
    together = win_probabilities(mu, sigma, [2, 3])
    np.testing.assert_allclose(together[:2], win_probabilities(mu[:2], sigma[:2], [2]))
    np.testing.assert_allclose(together[2:], win_probabilities(mu[2:], sigma[2:], [3]))


def _simulate(n_races=1500, size=8, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_races * size, 3)).astype(np.float32)
    mu = 4.0 * x[:, 0]
    sigma = np.exp(1.2 + 0.6 * x[:, 1])
    perf = rng.normal(mu, sigma)
    race_ids = np.repeat(np.arange(n_races), size).astype(str)
    frame = pd.DataFrame({"race": race_ids, "perf": perf})
    best = frame.groupby("race").perf.transform("max").to_numpy()
    won = (perf == best).astype(int)
    figure = np.minimum(perf - best, 0.0) + 0.0
    figure[rng.random(len(figure)) < 0.03] = np.nan  # non-finishers
    figure[won == 1] = 0.0
    dates = pd.DatetimeIndex(pd.to_datetime("2023-01-01") + pd.to_timedelta(np.arange(len(x)) // (size * 4), "D"))
    return x, figure, won, race_ids, dates, mu, sigma


def test_spread_scale_recovers_a_known_multiplier():
    x, figure, won, race_ids, dates, mu, sigma = _simulate(n_races=2000)
    groups = np.full(2000, 8)
    scale = fit_spread_scale(mu, sigma / 2.0, won, groups)
    assert scale == pytest.approx(2.0, rel=0.15)


def test_figure_rows_need_two_measured_runners():
    rows = figure_rows([0.0, np.nan, np.nan, 0.0, -3.0], ["a", "a", "b", "b", "b"])
    assert rows.tolist() == [False, False, False, True, True]


def test_halves_split_races_not_runners():
    halves = race_halves(np.repeat(["r1", "r2", "r3", "r4"], 3))
    assert all(len(set(halves[i:i + 3])) == 1 for i in range(0, 12, 3))


def test_fitted_figure_model_learns_mean_and_spread():
    x, figure, won, race_ids, dates, mu, sigma = _simulate()
    model = fit_figure_model(x, figure, won, race_ids, dates, np.ones(len(x)), capacity="screen", n_jobs=1)
    pred = model.predict(x, race_ids)
    assert np.corrcoef(pred.mu, mu)[0, 1] > 0.8
    assert np.corrcoef(np.log(pred.sigma), np.log(sigma))[0, 1] > 0.3
    np.testing.assert_allclose(pred.p.groupby(race_ids).sum(), 1.0)
    uniform = winner_log_loss(np.full(len(x), 1 / 8), won, np.full(len(x) // 8, 8))
    assert winner_log_loss(pred.p.to_numpy(), won, np.full(len(x) // 8, 8)) < uniform - 0.2
