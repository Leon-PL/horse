"""Race-structured LightGBM objectives (experiments 057, 058; 057 is in production).

Until 057 the production win learner was a per-runner cross-entropy on soft
labels: each race taught it who won, one runner at a time. The research
helpers here keep that learner's hyper-parameters, purged early-stopping split
and full-window refit (``feature_selection.fit_win_model`` with
``capacity="native"``) and swap only the loss:

* :class:`ExplodedSoftmax` reads a result as a sequence of choices (Benter's
  rank-ordered logit): the winner from the whole field, then second from the
  runners left, then third. Stage 1 can take the production soft target
  (market blend), so a one-stage run differs from production only in being a
  race softmax instead of a per-runner sigmoid. Non-finishers stay in every
  stage's field and are never a stage's choice. A stage stops a race's
  sequence when its position is missing or dead-heated.
* :class:`CentredRegression` fits a performance figure up to a per-race
  constant: the squared error of within-race centred predictions against
  within-race centred targets. It equals a pairwise least-squares loss on
  every pair's difference, so it needs no race par.

Custom objectives own the sample weights (LightGBM does not apply them to a
callable's gradients), so the weights are passed to the objective, not to
``fit``.

The production win learner (``RacePredictor._train_win_classifier`` with
``config.WIN_OBJECTIVE = "exploded"``) builds its loss with
:func:`training_objective` and early-stops on :func:`winner_metric`. Nothing
here is imported by the feature build.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numba import njit

import config
from src.feature_selection import SCREEN_PARAMS, race_groups


def learner_params(params: dict | None = None) -> dict:
    """The production win learner's hyper-parameters, built as ``_train_win_classifier`` builds them."""
    base = config.CLASSIFIER_PARAMS
    hp = {
        "n_estimators": base.get("n_estimators", 500),
        "max_depth": base.get("max_depth", 7),
        "learning_rate": base.get("learning_rate", 0.05),
        "subsample": base.get("subsample", 0.8),
        "colsample_bytree": base.get("colsample_bytree", 0.8),
        "reg_alpha": base.get("reg_alpha", 0.05),
        "reg_lambda": base.get("reg_lambda", 1.0),
        "linear_tree": bool(base.get("linear_tree", False)),
    }
    if "num_leaves" in base:
        hp["num_leaves"] = base["num_leaves"]
    hp.update(params or {})
    hp.setdefault("min_child_samples", base.get("min_child_samples", 10))
    return hp


def race_bounds(race_ids) -> np.ndarray:
    """Row boundaries of contiguous races: race r is rows ``bounds[r]:bounds[r+1]``."""
    return np.r_[0, np.cumsum(race_groups(race_ids))].astype(np.int64)


def fit_boosted(
    x: np.ndarray,
    y: np.ndarray,
    race_ids,
    race_dates,
    *,
    objective: Callable[[int, int], object],
    metric: Callable[[int, int], object] | None = None,
    sample_weight: np.ndarray | None = None,
    capacity: str = "native",
    seed: int = 42,
    n_jobs: int = -1,
    return_selected: bool = False,
    params: dict | None = None,
    early_stopping_rounds: int | None = None,
    purge_days: int | None = None,
    refit: bool = True,
    screen_params: dict | None = None,
):
    """Fit one LightGBM regressor the way the production win learner is fitted.

    *objective(lo, hi)* returns the LightGBM objective (a callable or a
    built-in name) for rows ``lo:hi``, which always start and end on race
    boundaries; *metric(lo, hi)* returns the early-stopping metric for those
    rows. *sample_weight* is passed to LightGBM only for a built-in objective.

    ``capacity="screen"`` is 400 trees at 0.04 with no early stopping.
    ``"native"`` is ``config.CLASSIFIER_PARAMS`` with the production purged
    early-stopping split, then a refit on every row at the selected tree count.
    With *return_selected* the early-stopped model is returned as well (or
    None when there was no split), for honest predictions on the held-back
    slab.
    """
    from lightgbm import LGBMRegressor, early_stopping

    from src.model import RacePredictor

    n = len(x)
    # Explicit recipes are already resolved; never merge them with mutable
    # defaults. Legacy callers retain the established defaults.
    hp = learner_params() if params is None else dict(params)
    es_rounds = int(config.EARLY_STOPPING_ROUNDS if early_stopping_rounds is None else early_stopping_rounds)
    gap = int(config.PURGE_DAYS if purge_days is None else purge_days)

    def fit(lo, hi, params=None, eval_rows=None):
        obj = objective(lo, hi)
        extra = {} if metric is None else {"metric": "None"}
        model = LGBMRegressor(objective=obj, **{**hp, **(params or {})}, **extra, subsample_freq=1,
                              random_state=seed, n_jobs=n_jobs, verbose=-1)
        kwargs = {}
        if sample_weight is not None and isinstance(obj, str):
            kwargs["sample_weight"] = sample_weight[lo:hi]
        if eval_rows is not None:
            a, b = eval_rows
            kwargs["eval_set"] = [(x[a:b], y[a:b])]
            if metric is not None:
                kwargs["eval_metric"] = metric(a, b)
            kwargs["callbacks"] = [early_stopping(es_rounds,
                                                  verbose=False)]
        model.fit(x[lo:hi], y[lo:hi], **kwargs)
        return model

    if capacity == "screen":
        model = fit(0, n, params=SCREEN_PARAMS if screen_params is None else screen_params)
        return (model, None) if return_selected else model
    if capacity != "native":
        raise ValueError(f"Unknown capacity {capacity!r}")
    groups = race_groups(race_ids)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    split = (RacePredictor._early_stop_split(groups, pd.DatetimeIndex(race_dates)[starts],
                                             purge_days=gap)
             if es_rounds > 0 else None)
    if split is None:
        model = fit(0, n)
        return (model, None) if return_selected else model
    end, validation = split
    selected = fit(0, end, eval_rows=(validation, n))
    selected.split_ = (int(end), int(validation))
    if not refit:
        return (selected, selected) if return_selected else selected
    rounds = selected.best_iteration_ or selected.n_estimators_
    if not rounds or int(rounds) < 1:
        raise ValueError("Cannot determine the selected tree count; refusing an ambiguous refit")
    model = fit(0, n, params={"n_estimators": int(rounds)})
    model.selected_rounds_ = int(rounds)
    model.split_ = (int(end), int(validation))
    return (model, selected) if return_selected else model


def raw_scores(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.booster_.predict(x, raw_score=True), dtype=np.float64)


def race_softmax(scores, bounds) -> np.ndarray:
    """Softmax of *scores* within each race."""
    scores = np.asarray(scores, dtype=np.float64)
    sizes = np.diff(bounds)
    top = np.repeat(np.maximum.reduceat(scores, bounds[:-1]), sizes)
    e = np.exp(scores - top)
    return e / np.repeat(np.add.reduceat(e, bounds[:-1]), sizes)


# --------------------------------------------------------------- 057 ------

def exploded_stages(finish_positions, groups, n_stages: int) -> np.ndarray:
    """``(n_races, n_stages)`` row of the runner chosen at each stage, ``-1`` where absent.

    Stage *k* (0-based) chooses the runner placed *k + 1*. A race's sequence
    stops at the first stage whose position is missing or shared (a dead
    heat), or when fewer than two runners would be left to choose from.
    """
    pos = np.asarray(finish_positions, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    n_races = len(groups)
    if groups.sum() != len(pos):
        raise ValueError("Group sizes do not cover the rows")
    race = np.repeat(np.arange(n_races), groups)
    chosen = np.full((n_races, n_stages), -1, dtype=np.int64)
    alive = np.ones(n_races, dtype=bool)
    for k in range(n_stages):
        rows = np.flatnonzero(pos == k + 1)
        counts = np.bincount(race[rows], minlength=n_races)
        pick = np.full(n_races, -1, dtype=np.int64)
        pick[race[rows]] = rows
        alive &= (counts == 1) & (groups - k >= 2)
        chosen[alive, k] = pick[alive]
    return chosen


@njit(cache=True)
def _exploded_pass(scores, bounds, chosen, first_target, weights, stage_weights, stage_temps, grad, hess):
    """Accumulate the exploded-softmax gradient and Hessian diagonal; return the weighted loss."""
    loss = 0.0
    n_races, n_stages = chosen.shape
    if n_races == 0:
        return loss
    taken = np.zeros(np.max(bounds[1:] - bounds[:-1]), dtype=np.bool_)
    for r in range(n_races):
        a, b = bounds[r], bounds[r + 1]
        taken[:b - a] = False
        for k in range(n_stages):
            c = chosen[r, k]
            if c < 0:
                break
            lam, tau = stage_weights[k], stage_temps[k]
            if lam != 0.0:
                top = -np.inf
                for i in range(a, b):
                    if not taken[i - a] and tau * scores[i] > top:
                        top = tau * scores[i]
                total = 0.0
                for i in range(a, b):
                    if not taken[i - a]:
                        total += np.exp(tau * scores[i] - top)
                log_total = np.log(total)
                for i in range(a, b):
                    if taken[i - a]:
                        continue
                    log_p = tau * scores[i] - top - log_total
                    p = np.exp(log_p)
                    if k == 0:
                        t = first_target[i]
                    else:
                        t = 1.0 if i == c else 0.0
                    g = lam * weights[i]
                    grad[i] += g * tau * (p - t)
                    hess[i] += g * tau * tau * p * (1.0 - p)
                    if t > 0.0:
                        loss -= g * t * log_p
            taken[c - a] = True
    return loss


@njit(cache=True)
def _winner_log_loss(scores, bounds, winners):
    """Mean winner log loss over races with a single winner (``winners[r] >= 0``)."""
    total = 0.0
    counted = 0
    n_races = len(bounds) - 1
    for r in range(n_races):
        if winners[r] < 0:
            continue
        counted += 1
        a, b = bounds[r], bounds[r + 1]
        top = -np.inf
        for i in range(a, b):
            if scores[i] > top:
                top = scores[i]
        s = 0.0
        for i in range(a, b):
            s += np.exp(scores[i] - top)
        total += np.log(s) + top - scores[winners[r]]
    return total / max(counted, 1)


class ExplodedSoftmax:
    """Rank-ordered (exploded) race softmax over contiguous races.

    *first_target* is stage 1's target distribution per runner (the
    production soft label, or the 0/1 winner); later stages choose the
    runner actually placed. *stage_weights* weight the stages, e.g.
    ``(1, 0.5, 0.25)``; ``(1,)`` is a winner-only race softmax.

    *stage_temps* (default all 1) multiply the scores inside each stage's
    softmax. A value below 1 flattens that choice: a lower place is treated
    as noisier rather than less important, as in the discounted Harville
    models of place betting.

    A race with zero weight is ignored, so it may lack a single winner (a
    dead heat or no recorded winner); every weighted race needs one.
    """

    def __init__(self, race_ids, finish_positions, first_target, weights, stage_weights, stage_temps=None):
        self.bounds = race_bounds(race_ids)
        self.stage_weights = np.asarray(stage_weights, dtype=np.float64)
        self.stage_temps = (np.ones_like(self.stage_weights) if stage_temps is None
                            else np.asarray(stage_temps, dtype=np.float64))
        if self.stage_temps.shape != self.stage_weights.shape or (self.stage_temps <= 0).any():
            raise ValueError("Need one positive temperature per stage")
        self.chosen = exploded_stages(finish_positions, np.diff(self.bounds), len(self.stage_weights))
        self.first_target = np.asarray(first_target, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        starts = self.bounds[:-1]
        if not np.allclose(np.maximum.reduceat(self.weights, starts), np.minimum.reduceat(self.weights, starts)):
            raise ValueError("Weights must be constant within each race (a race is one observation)")
        weighted = self.weights[starts] > 0
        if (self.chosen[weighted, 0] < 0).any():
            raise ValueError("Every weighted race needs a single winner")
        sums = np.add.reduceat(self.first_target, starts)
        if not np.allclose(sums[weighted], 1.0, atol=1e-6):
            raise ValueError("Stage-1 targets must sum to one within each race")

    def _slice(self, lo, hi):
        r0, r1 = np.searchsorted(self.bounds, [lo, hi])
        if self.bounds[r0] != lo or self.bounds[r1] != hi:
            raise ValueError("Rows must start and end on race boundaries")
        chosen = self.chosen[r0:r1]
        return (self.bounds[r0:r1 + 1] - lo, np.where(chosen >= 0, chosen - lo, -1),
                self.first_target[lo:hi], self.weights[lo:hi])

    def loss_and_gradients(self, scores, lo=0, hi=None):
        hi = len(self.weights) if hi is None else hi
        bounds, chosen, target, weights = self._slice(lo, hi)
        scores = np.asarray(scores, dtype=np.float64)
        grad, hess = np.zeros(len(scores)), np.zeros(len(scores))
        loss = _exploded_pass(scores, bounds, chosen, target, weights, self.stage_weights, self.stage_temps,
                              grad, hess)
        return loss, grad, np.maximum(hess, 1e-12)

    def objective(self, lo, hi):
        def fobj(_y_true, y_pred):
            _, grad, hess = self.loss_and_gradients(y_pred, lo, hi)
            return grad, hess
        return fobj

    def metric(self, lo, hi):
        """Early-stopping metric: race log loss of the actual winner (stage 1, hard label)."""
        bounds, chosen, _, _ = self._slice(lo, hi)
        winners = chosen[:, 0].copy()

        def feval(_y_true, y_pred):
            return "race_log_loss", float(_winner_log_loss(np.asarray(y_pred, np.float64), bounds, winners)), False
        return feval


def training_objective(groups, finish_positions, target, weights, stage_weights) -> ExplodedSoftmax:
    """The production win loss for one contiguous block of races.

    *target* is stage 1's per-runner target (the soft label, or 0/1 won) and
    is normalised within each race. Races without a single winner (dead heats,
    no recorded winner) or without any target get zero weight: 057 was
    confirmed on single-winner fields only, and the race softmax has no
    single-choice reading of a shared win.
    """
    groups = np.asarray(groups, dtype=np.int64)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    target = np.asarray(target, dtype=np.float64)
    sums = np.add.reduceat(target, starts)
    usable = (exploded_stages(finish_positions, groups, 1)[:, 0] >= 0) & (sums > 0)
    size = np.repeat(groups, groups)
    first = np.where(np.repeat(usable, groups), target / np.repeat(np.where(usable, sums, 1.0), groups), 1.0 / size)
    weights = np.asarray(weights, dtype=np.float64) * np.repeat(usable, groups)
    return ExplodedSoftmax(np.repeat(np.arange(len(groups)), groups), finish_positions, first, weights,
                           stage_weights)


def winner_metric(groups, won):
    """Early-stopping metric for a block of races: mean race log loss of the single winner."""
    groups = np.asarray(groups, dtype=np.int64)
    bounds = np.r_[0, np.cumsum(groups)].astype(np.int64)
    won = np.asarray(won, dtype=np.float64)
    starts = bounds[:-1]
    single = np.add.reduceat(won, starts) == 1
    race = np.repeat(np.arange(len(groups)), groups)
    winners = np.full(len(groups), -1, dtype=np.int64)
    rows = np.flatnonzero(won == 1)
    winners[race[rows]] = rows
    winners[~single] = -1

    def feval(_y_true, y_pred):
        return "race_log_loss", float(_winner_log_loss(np.asarray(y_pred, np.float64), bounds, winners)), False
    return feval


# --------------------------------------------------------------- 058 ------

@njit(cache=True)
def _centred_pass(pred, target, bounds, weights, grad, hess):
    loss = 0.0
    for r in range(len(bounds) - 1):
        a, b = bounds[r], bounds[r + 1]
        n = b - a
        mp, mt = 0.0, 0.0
        for i in range(a, b):
            mp += pred[i]
            mt += target[i]
        mp /= n
        mt /= n
        for i in range(a, b):
            e = (pred[i] - mp) - (target[i] - mt)
            grad[i] = weights[i] * e
            hess[i] = weights[i] * (1.0 - 1.0 / n)
            loss += 0.5 * weights[i] * e * e
    return loss


class CentredRegression:
    """Squared error after centring predictions and targets within each race.

    Every race needs at least two rows; the race's level is unidentified, so
    predictions are meaningful only as differences within a race.
    """

    def __init__(self, race_ids, target, weights):
        self.bounds = race_bounds(race_ids)
        if (np.diff(self.bounds) < 2).any():
            raise ValueError("Every race needs at least two rows")
        self.target = np.asarray(target, dtype=np.float64)
        if not np.isfinite(self.target).all():
            raise ValueError("Targets must be finite")
        self.weights = np.asarray(weights, dtype=np.float64)
        starts = self.bounds[:-1]
        if not np.allclose(np.maximum.reduceat(self.weights, starts), np.minimum.reduceat(self.weights, starts)):
            raise ValueError("Weights must be constant within each race")

    def _slice(self, lo, hi):
        r0, r1 = np.searchsorted(self.bounds, [lo, hi])
        if self.bounds[r0] != lo or self.bounds[r1] != hi:
            raise ValueError("Rows must start and end on race boundaries")
        return self.bounds[r0:r1 + 1] - lo, self.target[lo:hi], self.weights[lo:hi]

    def loss_and_gradients(self, pred, lo=0, hi=None):
        hi = len(self.target) if hi is None else hi
        bounds, target, weights = self._slice(lo, hi)
        pred = np.asarray(pred, dtype=np.float64)
        grad, hess = np.zeros(len(pred)), np.zeros(len(pred))
        loss = _centred_pass(pred, target, bounds, weights, grad, hess)
        return loss, grad, hess

    def objective(self, lo, hi):
        def fobj(_y_true, y_pred):
            _, grad, hess = self.loss_and_gradients(y_pred, lo, hi)
            return grad, hess
        return fobj

    def metric(self, lo, hi):
        """Early-stopping metric: unweighted mean squared centred error."""
        bounds, target, _ = self._slice(lo, hi)
        ones = np.ones(len(target))

        def feval(_y_true, y_pred):
            grad, hess = np.zeros(len(target)), np.zeros(len(target))
            loss = _centred_pass(np.asarray(y_pred, np.float64), target, bounds, ones, grad, hess)
            return "centred_mse", 2.0 * loss / len(target), False
        return feval


def centre_within(values, bounds) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sizes = np.diff(bounds)
    return values - np.repeat(np.add.reduceat(values, bounds[:-1]) / sizes, sizes)
