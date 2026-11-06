"""Performance figures with a spread, turned into win chances (experiment 058).

The production win model gives each runner one score. 055 found it over-rates
drifters and under-rates steamers and debutants: horses whose ability is
uncertain. Here each runner gets a mean performance figure and a spread, and
its win chance is the probability its draw is the best in the race
(a Thurstone model with normal performances), so an uncertain outsider keeps
more chance than a reliable one with the same mean.

* The label is pounds behind the winner: ``result_margins`` cumulative beaten
  lengths times a distance-dependent scale (:func:`pounds_behind`), capped so
  eased, tailed-off runners do not dominate. Every measured finisher has one.
* The mean figure is fitted with :class:`src.race_objectives.CentredRegression`
  (pairwise differences within a race, so no race par is needed).
* The spread is a second regression on the log squared residual. Its
  residuals are out-of-fold: two models, each fitted on half the training
  races at the mean model's tree count, predict the other half.
* One scale on every spread is fitted by winner likelihood on the purged
  early-stopping slab, using the early-stopped models, so it never sees the
  scored quarter.

Non-finishers have no figure and are left out of both regressions, but they
keep their predicted figure and spread at scoring time and stay in every
evaluated field (ledger 18).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from src.feature_selection import race_groups
from src.race_objectives import CentredRegression, centre_within, fit_boosted, race_bounds, raw_scores

#: Pounds per length at one furlong; per-length value falls with distance
#: (3 lb at 5f, about 1.9 lb at a mile, 1.25 lb at 1m4f, 0.94 lb at 2m).
LBS_PER_LENGTH_FURLONGS = 15.0
#: Margins beyond this many pounds are eased or tailed-off runners.
CAP_LBS = 30.0
#: Added to squared residuals before the log, in lb^2, so a near-exact
#: prediction does not produce an extreme target.
SPREAD_EPS = 0.25
SIGMA_FLOOR = 1.0


def pounds_behind(beaten_lengths, furlongs, cap: float = CAP_LBS) -> np.ndarray:
    """Pounds behind the winner; NaN where the margin is unknown."""
    lengths = np.asarray(beaten_lengths, dtype=np.float64)
    distance = np.clip(np.asarray(furlongs, dtype=np.float64), 5.0, None)
    return np.minimum(lengths * LBS_PER_LENGTH_FURLONGS / distance, cap)


def win_probabilities(mu, sigma, groups, *, points: int = 400, width: float = 6.0,
                      chunk_cells: int = 4_000_000) -> np.ndarray:
    """P(runner's normal performance draw is the race's best), by quadrature.

    For runner *i*: integral of ``phi_i(x) * prod_{j != i} Phi_j(x)`` over a
    per-race grid covering every runner's mean +- *width* spreads.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("Means must be finite and spreads positive")
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    out = np.empty(len(mu))
    unit = np.linspace(0.0, 1.0, points)
    for size in np.unique(groups):
        races = np.flatnonzero(groups == size)
        rows_all = starts[races][:, None] + np.arange(size)[None, :]
        step = max(1, chunk_cells // (size * points))
        for c in range(0, len(races), step):
            rows = rows_all[c:c + step]
            m, s = mu[rows], sigma[rows]
            lo = (m - width * s).min(axis=1)
            hi = (m + width * s).max(axis=1)
            grid = lo[:, None] + (hi - lo)[:, None] * unit[None, :]
            z = (grid[:, None, :] - m[:, :, None]) / s[:, :, None]
            log_cdf = log_ndtr(z)
            log_pdf = -0.5 * z * z - np.log(s)[:, :, None]
            integrand = np.exp(log_pdf + log_cdf.sum(axis=1, keepdims=True) - log_cdf)
            p = np.trapezoid(integrand, dx=1.0, axis=2)
            out[rows] = p / p.sum(axis=1, keepdims=True)
    return out


def winner_log_loss(p, won, groups) -> float:
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    return float(-np.add.reduceat(np.asarray(won) * np.log(np.clip(p, 1e-12, None)), starts).mean())


def scaled_spread(raw_spread, scale: float) -> np.ndarray:
    return np.maximum(scale * np.asarray(raw_spread, dtype=np.float64), SIGMA_FLOOR)


def fit_spread_scale(mu, raw_spread, won, groups) -> float:
    """The single multiplier on every spread that maximises the winners' likelihood."""
    result = minimize_scalar(
        lambda log_c: winner_log_loss(win_probabilities(mu, scaled_spread(raw_spread, np.exp(log_c)), groups),
                                      won, groups),
        bounds=(-2.0, 3.0), method="bounded", options={"xatol": 1e-3})
    return float(np.exp(result.x))


def race_halves(race_ids) -> np.ndarray:
    """A stable 0/1 split of races (by hashed race id), for cross-fitting."""
    ids = pd.Series(np.asarray(race_ids)).astype(str)
    uniques = ids.unique()
    halves = {r: hashlib.blake2b(r.encode(), digest_size=2).digest()[0] & 1 for r in uniques}
    return ids.map(halves).to_numpy(np.int8)


def figure_rows(figure, race_ids) -> np.ndarray:
    """Rows with a figure in races where at least two runners have one."""
    has = np.isfinite(np.asarray(figure, dtype=np.float64))
    count = pd.Series(has).groupby(np.asarray(race_ids)).transform("sum").to_numpy()
    return has & (count >= 2)


@dataclass
class FigureModel:
    mean: object
    spread: object
    scale: float
    info: dict

    def predict(self, x: np.ndarray, race_ids) -> pd.DataFrame:
        groups = race_groups(race_ids)
        mu = centre_within(raw_scores(self.mean, x), np.r_[0, np.cumsum(groups)])
        sigma = scaled_spread(np.exp(0.5 * raw_scores(self.spread, x)), self.scale)
        return pd.DataFrame({"mu": mu, "sigma": sigma, "p": win_probabilities(mu, sigma, groups)})


def fit_figure_model(x: np.ndarray, figure, won, race_ids, race_dates, weights, *,
                     capacity: str = "native", seed: int = 42, n_jobs: int = -1, log=None) -> FigureModel:
    """Fit the mean and spread regressions and the spread scale on one training window.

    *x* holds every training runner (finishers and not, races contiguous and
    in date order); *figure* is NaN where there is no measured figure.
    """
    log = log or (lambda *a: None)
    figure = np.asarray(figure, dtype=np.float64)
    race_ids = np.asarray(race_ids)
    dates = pd.DatetimeIndex(race_dates)
    weights = np.asarray(weights, dtype=np.float64)
    usable = figure_rows(figure, race_ids)
    rows = np.flatnonzero(usable)
    fx, fy, fr, fd, fw = x[rows], figure[rows], race_ids[rows], dates[rows], weights[rows]
    mean_obj = CentredRegression(fr, fy, fw)

    # Mean figure: early stopping on the purged slab, refit on every row.
    mean, mean_selected = fit_boosted(fx, fy, fr, fd, objective=mean_obj.objective, metric=mean_obj.metric,
                                      capacity=capacity, seed=seed, n_jobs=n_jobs, return_selected=True)
    rounds = getattr(mean, "selected_rounds_", None)
    log("mean figure: %s rows, %s trees", len(rows), rounds)

    # Out-of-fold residuals from two half-sample models at the same capacity.
    halves = race_halves(fr)
    oof = np.empty(len(rows))
    for half in (0, 1):
        train, other = halves == half, halves != half
        obj = CentredRegression(fr[train], fy[train], fw[train])
        if rounds is None:
            model = fit_boosted(fx[train], fy[train], fr[train], fd[train], objective=obj.objective,
                                metric=obj.metric, capacity=capacity, seed=seed, n_jobs=n_jobs)
        else:
            model = _fit_fixed(fx[train], fy[train], obj, rounds, seed, n_jobs)
        oof[other] = raw_scores(model, fx[other])
    bounds = race_bounds(fr)
    residual = centre_within(fy, bounds) - centre_within(oof, bounds)
    target = np.log(residual ** 2 + SPREAD_EPS)

    # Spread: an ordinary L2 regression on the log squared residual.
    spread, spread_selected = fit_boosted(fx, target, fr, fd, objective=lambda lo, hi: "regression",
                                          sample_weight=fw, capacity=capacity, seed=seed, n_jobs=n_jobs,
                                          return_selected=True)
    log("spread: %s trees", getattr(spread, "selected_rounds_", None))

    # Spread scale on the held-back slab, from models that never saw it.
    # Both were early-stopped on the same figure-row split, so the slab starts
    # at the same race for each; every runner of those races is scored.
    scale_x, scale_ids, scale_won, scale_mean, scale_spread = x, race_ids, np.asarray(won), mean, spread
    if mean_selected is not None and spread_selected is not None:
        first_race = fr[mean.split_[1]]
        validation = int(np.flatnonzero(race_ids == first_race)[0])
        scale_x, scale_ids, scale_won = x[validation:], race_ids[validation:], np.asarray(won)[validation:]
        scale_mean, scale_spread = mean_selected, spread_selected
    groups = race_groups(scale_ids)
    mu = centre_within(raw_scores(scale_mean, scale_x), np.r_[0, np.cumsum(groups)])
    scale = fit_spread_scale(mu, np.exp(0.5 * raw_scores(scale_spread, scale_x)), scale_won, groups)
    log("spread scale %.3f on %d races", scale, len(groups))
    info = {"figure_rows": int(len(rows)), "mean_trees": rounds,
            "spread_trees": getattr(spread, "selected_rounds_", None), "scale": scale,
            "scale_races": int(len(groups)), "oof_residual_sd": float(residual.std())}
    return FigureModel(mean, spread, scale, info)


def _fit_fixed(x, y, obj: CentredRegression, rounds: int, seed: int, n_jobs: int):
    """A mean-figure model at a fixed tree count (no early stopping)."""
    from lightgbm import LGBMRegressor

    from src.race_objectives import learner_params
    model = LGBMRegressor(objective=obj.objective(0, len(y)), **learner_params({"n_estimators": int(rounds)}),
                          subsample_freq=1, random_state=seed, n_jobs=n_jobs, verbose=-1)
    model.fit(x, y)
    return model
