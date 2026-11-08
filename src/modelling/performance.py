"""Complete-field performance/spread expert with chronological residual targets.

Reproduces 058's centred margin regression and normal-performance integration.
Spread targets now come from earlier-only fits, and an explicit later slab
calibrates scale using models fitted before that slab at either capacity.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

import config
from src.feature_selection import race_groups
from src.model import compute_recency_sample_weights
from src.modelling.data import numeric_matrix, ordered_frame
from src.modelling.spec import ModelSpec, _defaults, _screen_defaults
from src.race_objectives import CentredRegression, centre_within, fit_boosted, race_bounds, raw_scores
from src.wf_eval import oof_predict_expanding

logger = logging.getLogger(__name__)
TRAINING_COLUMNS = ("beaten_lengths", "distance_furlongs")


@dataclass(frozen=True)
class PerformanceSpec:
    id: str
    features: tuple[str, ...]
    params: tuple = field(default_factory=_defaults)
    screen_params: tuple = field(default_factory=_screen_defaults)
    seed: int = field(default_factory=lambda: config.RANDOM_SEED)
    purge_days: int = field(default_factory=lambda: config.PURGE_DAYS)
    early_stopping_rounds: int = field(default_factory=lambda: config.EARLY_STOPPING_ROUNDS)
    half_life_days: float = field(default_factory=lambda: config.RECENCY_HALF_LIFE_DAYS)
    seasonal_boost: float = field(default_factory=lambda: config.RECENCY_SEASONAL_BOOST)
    decay_shape: str = field(default_factory=lambda: config.RECENCY_DECAY_SHAPE)
    burn_in_months: int = field(default_factory=lambda: config.BURN_IN_MONTHS)
    calibration_fraction: float = .2
    residual_splits: int = 4
    margin_cap_lb: float = 30.
    lb_per_length_furlongs: float = 15.
    spread_epsilon: float = .25
    sigma_floor: float = 1.
    quadrature_points: int = 400

    def __post_init__(self):
        # Reuse the established validation of shared numerical/feature controls;
        # win targets and finishing-order weights are not part of this recipe.
        shared = {k: v for k, v in asdict(self).items() if k in ModelSpec.__dataclass_fields__}
        checked = ModelSpec(**shared)
        object.__setattr__(self, "params", checked.params)
        object.__setattr__(self, "screen_params", checked.screen_params)
        if (not isinstance(self.residual_splits, int) or self.residual_splits < 3
                or not isinstance(self.quadrature_points, int) or self.quadrature_points < 64):
            raise ValueError("Need at least three chronological blocks and 64 quadrature points")
        for name in ("margin_cap_lb", "lb_per_length_furlongs", "spread_epsilon", "sigma_floor"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self):
        value = asdict(self)
        value.update(kind="performance", features=list(self.features), params=dict(self.params),
                     screen_params=dict(self.screen_params))
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("kind", "performance") != "performance":
            raise ValueError("Expected a performance component recipe")
        value["features"] = tuple(value["features"])
        for name in ("params", "screen_params"):
            if name in value:
                value[name] = tuple(sorted(value[name].items()))
        return cls(**value)

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()


def performance_targets(frame, spec):
    """Negative capped pounds behind the winner; unknown margins/DNFs stay NaN.

    The 058 scale uses 15/furlongs lb per length, with distance floored at 5f.
    A winner's margin is exactly zero. A positive recorded finishing position
    and valid distance are required even if a non-finisher has a stored margin.
    """
    position = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(dtype=float)
    lengths = pd.to_numeric(frame.beaten_lengths, errors="coerce").to_numpy(dtype=float)
    distance = pd.to_numeric(frame.distance_furlongs, errors="coerce").to_numpy(dtype=float)
    lengths = np.where(position == 1, 0., lengths)
    valid = (np.isfinite(position) & (position >= 1) & np.isfinite(lengths) & (lengths >= 0)
             & np.isfinite(distance) & (distance > 0))
    figure = -np.minimum(lengths * spec.lb_per_length_furlongs / np.maximum(distance, 5.), spec.margin_cap_lb)
    return np.where(valid, figure, np.nan)


def figure_rows(figure, race_ids):
    finite = np.isfinite(figure)
    counts = pd.Series(finite).groupby(np.asarray(race_ids)).transform("sum").to_numpy()
    return finite & (counts >= 2)


def win_probabilities(mu, sigma, groups, *, points=400, width=6., chunk_cells=1_000_000):
    """Integrate P(normal performance is the largest) over every complete field."""
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    original = np.asarray(groups)
    if (mu.ndim != 1 or sigma.shape != mu.shape or not len(mu)
            or not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any()
            or original.ndim != 1 or not len(original) or not np.isfinite(original).all()
            or (original < 2).any() or (original != original.astype(np.int64)).any()
            or original.sum() != len(mu)):
        raise ValueError("Need finite means, positive spreads and complete race groups")
    if not isinstance(points, int) or points < 64 or not np.isfinite(width) or width <= 0 or chunk_cells < 1:
        raise ValueError("Invalid quadrature settings")
    groups = original.astype(np.int64)
    starts, out = np.r_[0, np.cumsum(groups)[:-1]], np.empty(len(mu))
    unit = np.linspace(0., 1., points)
    for size in np.unique(groups):
        races = np.flatnonzero(groups == size)
        step = max(1, chunk_cells // (size * points))
        for first in range(0, len(races), step):
            rows = starts[races[first:first + step], None] + np.arange(size)[None, :]
            m, s = mu[rows], sigma[rows]
            lo, hi = (m - width * s).min(axis=1), (m + width * s).max(axis=1)
            grid = lo[:, None] + (hi - lo)[:, None] * unit
            z = (grid[:, None, :] - m[:, :, None]) / s[:, :, None]
            log_cdf = log_ndtr(z)
            log_pdf = -.5 * z * z - np.log(s)[:, :, None]
            integrand = np.exp(log_pdf + log_cdf.sum(axis=1, keepdims=True) - log_cdf)
            p = np.trapezoid(integrand, axis=2)
            total = p.sum(axis=1, keepdims=True)
            if not np.isfinite(p).all() or (total <= 0).any():
                raise ValueError("Normal integration produced an invalid field distribution")
            out[rows] = p / total
    return out


def _regression(_lo, _hi):
    return "regression"


@dataclass
class PerformanceModel:
    spec: PerformanceSpec
    mean: object = None
    spread: object = None
    scale: float = 1.
    lineage: dict = None

    def _fit_mean(self, x, figure, ids, dates, weights, *, capacity, threads):
        rows = figure_rows(figure, ids)
        if rows.sum() < 100:
            raise ValueError("Insufficient measured finishers for a performance fit")
        loss = CentredRegression(ids[rows], figure[rows], weights[rows])
        model = self._fit(x[rows], figure[rows], ids[rows], dates[rows], capacity=capacity, threads=threads,
                          objective=loss.objective, metric=loss.metric)
        # Custom objective closures retain labels and gradient arrays unless
        # removed. Saved artifacts need only trees and explicit recipe lineage.
        model.objective = model._objective = "centred_margin"
        logger.info("Mean fitted on %d measured finishers, %d trees", int(rows.sum()), int(model.n_estimators_))
        return model

    def _fit(self, x, y, ids, dates, *, capacity, threads, objective=_regression, metric=None, weights=None):
        model = fit_boosted(x, y, ids, dates, objective=objective, metric=metric, sample_weight=weights,
            capacity=capacity, seed=self.spec.seed, n_jobs=threads, params=dict(self.spec.params),
            screen_params=dict(self.spec.screen_params), early_stopping_rounds=self.spec.early_stopping_rounds,
            purge_days=self.spec.purge_days)
        if weights is not None:
            logger.info("Spread fitted on %d earlier-only residuals, %d trees", len(y), int(model.n_estimators_))
        return model

    def _residual_targets(self, context, x, figure, weights, *, capacity, threads):
        narrow = context[["race_id", "race_date"]].assign(_row=np.arange(len(context)))
        receipts = []
        def predict(train, test):
            a, b = train._row.to_numpy(), test._row.to_numpy()
            logger.info("Residual mean: %d earlier / %d later runners", len(a), len(b))
            model = self._fit_mean(x[a], figure[a], context.race_id.to_numpy()[a], context.race_date.to_numpy()[a],
                                    weights[a], capacity=capacity, threads=threads)
            receipts.append({"training_last_date": str(train.race_date.max().date()),
                "test_first_date": str(test.race_date.min().date()), "test_last_date": str(test.race_date.max().date()),
                "selected_rounds": int(model.n_estimators_)})
            return raw_scores(model, x[b])
        scores = oof_predict_expanding(narrow, fit_predict=predict, n_splits=self.spec.residual_splits,
            embargo_days=self.spec.purge_days, min_train_rows=100)
        usable = figure_rows(figure, context.race_id) & np.isfinite(scores)
        if usable.sum() < 100:
            raise ValueError("Insufficient earlier-only residuals for the spread model")
        bounds = race_bounds(context.race_id.to_numpy()[usable])
        residual = centre_within(figure[usable], bounds) - centre_within(scores[usable], bounds)
        target = np.full(len(context), np.nan)
        target[usable] = np.log(residual ** 2 + self.spec.spread_epsilon)
        return target, receipts

    def _sigma(self, estimator, x):
        return np.exp(np.clip(.5 * raw_scores(estimator, x), -20., 20.))

    def _probabilities(self, mean, spread, x, ids, scale):
        mu = centre_within(raw_scores(mean, x), race_bounds(ids))
        sigma = np.maximum(scale * self._sigma(spread, x), self.spec.sigma_floor)
        p = win_probabilities(mu, sigma, race_groups(ids), points=self.spec.quadrature_points)
        return mu, sigma, p

    def fit(self, frame, *, dataset_start=None, capacity="native", threads=1):
        self.mean, self.spread, self.lineage, self.scale = None, None, None, 1.
        if threads < 1 or capacity not in {"screen", "native"}:
            raise ValueError("Positive threads and a supported capacity are required")
        if not set(TRAINING_COLUMNS).issubset(frame):
            raise ValueError("Performance fitting requires measured margins and distance as training context")
        ordered, _ = ordered_frame(frame)
        beginning = pd.Timestamp(dataset_start if dataset_start is not None else ordered.race_date.min())
        ordered = ordered.loc[ordered.race_date.ge(beginning + pd.DateOffset(months=self.spec.burn_in_months))].reset_index(drop=True)
        if ordered.empty:
            raise ValueError("Insufficient history after burn-in")
        x, figure = numeric_matrix(ordered, self.spec.features), performance_targets(ordered, self.spec)
        context = ordered[["race_id", "race_date", "won"]].copy()
        del ordered
        ids, dates = context.race_id.to_numpy(), context.race_date.to_numpy()
        weights = compute_recency_sample_weights(context.race_date, half_life_days=self.spec.half_life_days,
            seasonal_boost=self.spec.seasonal_boost, decay_shape=self.spec.decay_shape)
        groups = race_groups(ids)
        bnd = int(len(groups) * (1 - self.spec.calibration_fraction))
        if not 0 < bnd < len(groups):
            raise ValueError("Insufficient history for chronological scale calibration")
        cal_first = context.race_date.iloc[np.r_[0, np.cumsum(groups)[:-1]][bnd]]
        earlier = (context.race_date.lt(cal_first) & context.race_date.le(cal_first - pd.Timedelta(days=self.spec.purge_days))).to_numpy()
        calibration = context.race_date.ge(cal_first).to_numpy()
        if earlier.sum() < 100 or calibration.sum() < 50:
            raise ValueError("Insufficient history for chronological scale calibration")
        target, residual_receipts = self._residual_targets(context, x, figure, weights, capacity=capacity, threads=threads)
        spread_early = earlier & np.isfinite(target)
        if spread_early.sum() < 100:
            raise ValueError("Insufficient earlier residuals before the calibration boundary")
        logger.info("Fitting earlier calibration mean and spread")
        cal_mean = self._fit_mean(x[earlier], figure[earlier], ids[earlier], dates[earlier], weights[earlier],
                                   capacity=capacity, threads=threads)
        cal_spread = self._fit(x[spread_early], target[spread_early], ids[spread_early], dates[spread_early],
            weights=weights[spread_early], capacity=capacity, threads=threads)
        cal_x, cal_ids = x[calibration], ids[calibration]
        mu = centre_within(raw_scores(cal_mean, cal_x), race_bounds(cal_ids))
        raw_sigma = self._sigma(cal_spread, cal_x)
        won, cal_groups = context.won.to_numpy(dtype=float)[calibration], race_groups(cal_ids)
        def loss(log_scale):
            p = win_probabilities(mu, np.maximum(np.exp(log_scale) * raw_sigma, self.spec.sigma_floor),
                                   cal_groups, points=self.spec.quadrature_points)
            return float(-np.sum(won * np.log(np.clip(p, 1e-12, 1.))) / len(cal_groups))
        optimum = minimize_scalar(loss, bounds=(-2., 3.), method="bounded", options={"xatol": 1e-3})
        if not optimum.success or not np.isfinite(optimum.fun):
            raise ValueError("Spread scale calibration failed")
        scale = float(np.exp(optimum.x))
        del cal_mean, cal_spread, cal_x
        logger.info("Spread scale %.6f; fitting final mean and spread", scale)
        mean = self._fit_mean(x, figure, ids, dates, weights, capacity=capacity, threads=threads)
        usable = np.isfinite(target)
        spread = self._fit(x[usable], target[usable], ids[usable], dates[usable], weights=weights[usable],
                           capacity=capacity, threads=threads)
        self.mean, self.spread, self.scale = mean, spread, scale
        self.lineage = {"recipe": self.spec.to_dict(), "recipe_signature": self.spec.signature,
            "training_first_date": str(context.race_date.min().date()), "training_last_date": str(context.race_date.max().date()),
            "training_runners": len(context), "training_races": len(groups), "capacity": capacity,
            "figure_rows": int(figure_rows(figure, ids).sum()), "missing_figures": int((~np.isfinite(figure)).sum()),
            "spread_rows": int(usable.sum()), "residual_folds": residual_receipts,
            "calibration_first_date": str(cal_first.date()), "calibration_last_date": str(context.race_date.max().date()),
            "calibration_fit_boundary": str(context.race_date[earlier].max().date()),
            "scale": scale, "mean_trees": int(mean.n_estimators_), "spread_trees": int(spread.n_estimators_)}
        return self

    def predict_details(self, frame):
        if self.mean is None or self.spread is None:
            raise ValueError("Performance model is not fitted")
        ordered, positions = ordered_frame(frame, labelled=False)
        x = numeric_matrix(ordered, self.spec.features)
        values = np.column_stack(self._probabilities(self.mean, self.spread, x, ordered.race_id.to_numpy(), self.scale))
        out = np.empty_like(values)
        out[positions] = values
        return pd.DataFrame(out, columns=["mu", "sigma", "probability"], index=frame.index)

    def predict(self, frame):
        return self.predict_details(frame).probability.to_numpy()
