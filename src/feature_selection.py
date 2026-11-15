"""Walk-forward feature selection for the win model (experiment 048).

The allowlist this replaces was the top 150 by LightGBM *gain* from one pilot
fit, on one split, on a store that still held three leaks, and it was then
patched by hand for four months (``docs/archive/research/FEATURE_SELECTION_048.md``). Gain
rewards leaked and high-cardinality columns and splits credit between
correlated copies; a single in-sample fit says nothing about generalisation.

This module selects on the quantity the win model exists to get right: its
own **out-of-sample race log loss**, the negative log-probability given to
each race's winner after per-race normalisation. It is the likelihood of a
one-winner race, strictly proper, and unlike Brier it is not indifferent to a
2% horse priced at 0.5%. Brier is reported alongside it as a guard. The
market is not an objective here; its log loss is reported only as a leak
tripwire, because a public-form model that beats the closing price is far
more likely to be reading the result than predicting it.

Protocol, each step a function below:

1. :func:`hygiene_filter` drops, without looking at labels, columns that are
   near-empty, constant or exact duplicates of another column.
   :func:`near_copy_filter` then keeps one representative of each cluster of
   near-identical columns, and :func:`permutation_importance` ranks inputs by
   the loss they prevent on unseen races (both added 12 November 2026 and used
   by ``src.modelling.selection``).
2. :func:`run_elimination` performs recursive elimination over expanding,
   purged quarterly folds (:func:`src.wf_eval.expanding_period_folds`). At each
   step every fold's model is fitted on its training window and scored on its
   quarter; each feature's importance is its mean |SHAP| *on that unseen
   quarter*, normalised per fold and averaged across folds, so the ranking is
   out-of-sample and averaged over several fits. The weakest fraction is
   dropped and the smaller set refitted, which lets a correlated feature
   inherit the credit its dropped twin was sharing.
3. :func:`one_se_choice` stops at the smallest step whose pooled loss is within
   one paired, date-clustered standard error of the best step (Breiman's
   one-SE rule): the best step's loss is biased low by having been chosen.
4. The chosen set is then confirmed on later quarters that took no part in
   selection (``scripts/select_features.py confirm``).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import config
from src.forecast_evaluation import probability_metrics

logger = logging.getLogger(__name__)

#: The project's cheap screen capacity (scripts/quick_win_feature_screen.py):
#: every other learner setting stays at ``config.CLASSIFIER_PARAMS``.
SCREEN_PARAMS = {"n_estimators": 400, "learning_rate": 0.04}

#: Rows per ``pred_contrib`` call, so a quarter's SHAP matrix stays small.
SHAP_CHUNK_ROWS = 20_000


def candidate_feature_columns(frame):
    """All safe numeric candidates, before the existing production allowlist.

    Identifiers, outcomes and other excluded fields keep the model's normal
    exclusions. New BHA columns can enter selection without changing live input
    selection or mutating shared configuration.
    """
    from src.model import get_feature_columns
    return get_feature_columns(frame, apply_allowlist=False)


# ---------------------------------------------------------------- hygiene --

def hygiene_filter(
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    min_present: float = 0.005,
    prefer: Sequence[str] = (),
    rows=None,
    dtype=np.float32,
) -> tuple[list[str], dict[str, str]]:
    """Columns worth offering the model, and the reason each other was dropped.

    Label-free, so it can see every development row without biasing anything.
    A column is dropped when fewer than *min_present* of rows are finite, when
    it holds a single value and no missing rows, or when it is bit-for-bit the
    same as another column (047 found five such duplicates in the old list). A
    column that is constant where present but sometimes missing is kept: it is
    a missingness indicator, which trees can split on. Among duplicates the
    first name in *prefer*, then the earlier column, survives.
    """
    names = list(names)
    if matrix.shape[1] != len(names):
        raise ValueError(f"{matrix.shape[1]} columns but {len(names)} names")
    preferred = {name: i for i, name in enumerate(prefer)}
    order = sorted(range(len(names)), key=lambda i: (preferred.get(names[i], len(preferred)), i))
    seen: dict[bytes, str] = {}
    reasons: dict[str, str] = {}
    for i in order:
        name = names[i]
        column = np.asarray(matrix[:, i] if rows is None else matrix[rows, i], dtype=dtype)
        finite = np.isfinite(column)
        if finite.mean() < min_present:
            reasons[name] = f"present in {finite.mean():.2%} of rows"
            continue
        values = column[finite]
        if finite.all() and values.min() == values.max():
            reasons[name] = "constant"
            continue
        # +0.0 folds -0.0 into 0.0, and one NaN constant fixes the payload, so
        # equal columns always hash equal.
        canonical = np.where(finite, column + dtype(0.0), dtype(np.nan))
        digest = hashlib.blake2b(canonical.tobytes(), digest_size=16).digest()
        if digest in seen:
            reasons[name] = f"duplicate of {seen[digest]}"
            continue
        seen[digest] = name
    kept = [name for name in names if name not in reasons]
    return kept, reasons


def near_copy_filter(
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    threshold: float = 0.95,
    prefer: Sequence[str] = (),
    rows=None,
    sample: int = 100_000,
    seed: int = 0,
) -> tuple[list[str], dict[str, str]]:
    """One representative of each cluster of near-identical columns.

    The store holds many versions of one measurement (a rating, the same rating
    against the field, its z-score, its rank). Recursive elimination splits the
    credit between them, so a useful family can be dropped piece by piece and
    the first fits carry hundreds of redundant inputs. This is label-free:
    columns are ranked, missing values take the middle rank, and a column whose
    absolute rank correlation with an already-kept column reaches *threshold*
    is dropped in favour of it. Columns are considered in order of *prefer*,
    then fewest missing values, then position, so a current model input or the
    better-covered column represents its cluster.
    """
    names = list(names)
    if matrix.shape[1] != len(names):
        raise ValueError(f"{matrix.shape[1]} columns but {len(names)} names")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    index = np.arange(matrix.shape[0]) if rows is None else np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows)
    if len(index) > sample:
        index = np.sort(np.random.default_rng(seed).choice(index, size=sample, replace=False))
    z = np.empty((len(index), len(names)), dtype=np.float32)
    missing = np.empty(len(names))
    for j in range(len(names)):
        column = np.asarray(matrix[:, j], dtype=np.float64)[index]
        finite = np.isfinite(column)
        missing[j] = 1 - finite.mean()
        ranks = np.full(len(column), (finite.sum() + 1) / 2.0)
        if finite.any():
            ranks[finite] = pd.Series(column[finite]).rank().to_numpy()
        ranks -= ranks.mean()
        scale = np.sqrt(np.square(ranks).sum())
        z[:, j] = ranks / scale if scale > 0 else 0.0
    preferred = {name: i for i, name in enumerate(prefer)}
    order = sorted(range(len(names)), key=lambda i: (preferred.get(names[i], len(preferred)), missing[i], i))
    kept: list[int] = []
    reasons: dict[str, str] = {}
    for i in order:
        if kept:
            r = np.abs(z[:, kept].T @ z[:, i])
            best = int(np.argmax(r))
            if r[best] >= threshold:
                reasons[names[i]] = f"near-copy of {names[kept[best]]} (|rank r| {float(r[best]):.3f})"
                continue
        kept.append(i)
    chosen = set(kept)
    return [name for i, name in enumerate(names) if i in chosen], reasons


def permutation_importance(predict: Callable[[np.ndarray], np.ndarray], x: np.ndarray, won, groups,
                           *, seed: int = 0) -> np.ndarray:
    """Rise in race log loss when each column of *x* is shuffled across runners.

    Mean |SHAP| measures how much a model leans on an input; this measures what
    the forecast loses without it, on races the model has not seen. A value at
    or below zero means the input did not help there. *predict(x)* must return
    race-normalised win probabilities for the rows of *x*.
    """
    won = np.asarray(won, dtype=np.float64)
    starts = np.r_[0, np.cumsum(groups)[:-1]]

    def loss(matrix):
        p = np.clip(np.asarray(predict(matrix), dtype=np.float64), 1e-12, None)
        return float((-np.add.reduceat(won * np.log(p), starts)).mean())

    base = loss(x)
    rng = np.random.default_rng(seed)
    out = np.zeros(x.shape[1])
    work = np.array(x, copy=True)
    for j in range(x.shape[1]):
        original = work[:, j].copy()
        work[:, j] = original[rng.permutation(len(original))]
        out[j] = loss(work) - base
        work[:, j] = original
    return out


# ---------------------------------------------------------------- scoring --

def race_groups(race_ids) -> np.ndarray:
    """Sizes of contiguous races, in order; raises if a race is split."""
    ids = pd.Series(np.asarray(race_ids)).reset_index(drop=True)
    starts = ids.ne(ids.shift()).to_numpy()
    starts[0] = True
    labels = ids[starts]
    if labels.duplicated().any():
        raise ValueError("Races must be contiguous")
    return np.diff(np.r_[np.flatnonzero(starts), len(ids)])


def race_normalise(raw, groups) -> np.ndarray:
    """Scale raw win probabilities to sum to one within each race."""
    raw = np.asarray(raw, dtype=np.float64)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    totals = np.add.reduceat(raw, starts)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("Every race needs positive finite probability mass")
    return raw / np.repeat(totals, groups)


def score_races(probabilities, won, groups) -> tuple[dict, np.ndarray, np.ndarray]:
    """Shared project metrics plus per-race log loss and Brier."""
    return probability_metrics(probabilities, won, groups)


def clustered_se(delta, clusters) -> float:
    """Standard error of a mean over races, clustered (normally by date).

    Races on one day share going, weather and a market regime, so they are not
    independent; the clustered linearisation is what a date-block bootstrap of
    the same ratio-of-sums estimates, without the draws.
    """
    delta = np.asarray(delta, dtype=np.float64)
    codes, uniques = pd.factorize(np.asarray(clusters), sort=False)
    n_clusters = len(uniques)
    if n_clusters < 2:
        return float("nan")
    totals = np.bincount(codes, weights=delta, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    mean = totals.sum() / counts.sum()
    residual = totals - mean * counts
    variance = n_clusters / (n_clusters - 1) * np.square(residual).sum() / counts.sum() ** 2
    return float(np.sqrt(variance))


def one_se_choice(sizes: Sequence[int], race_losses: np.ndarray, clusters) -> dict:
    """Smallest step within one clustered standard error of the best step.

    *race_losses* is ``(n_steps, n_races)``: every step scored on the same
    races, so each comparison is paired and most race-to-race noise cancels.
    """
    losses = np.asarray(race_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] != len(sizes):
        raise ValueError("Expected one row of race losses per step")
    means = losses.mean(axis=1)
    best = int(np.argmin(means))
    rows = []
    for i, size in enumerate(sizes):
        delta = losses[i] - losses[best]
        se = clustered_se(delta, clusters) if i != best else 0.0
        rows.append({"step": i, "n_features": int(size), "race_log_loss": float(means[i]),
                     "delta_vs_best": float(delta.mean()), "se": float(se),
                     "within_one_se": bool(i == best or delta.mean() <= se)})
    eligible = [r for r in rows if r["within_one_se"]]
    chosen = min(eligible, key=lambda r: (r["n_features"], r["race_log_loss"]))
    return {"best_step": best, "chosen_step": int(chosen["step"]), "steps": rows}


# ---------------------------------------------------------- elimination ----

def elimination_schedule(
    n_start: int,
    *,
    n_min: int = 30,
    coarse: float = 0.2,
    fine: float = 0.1,
    fine_below: int = 250,
) -> list[int]:
    """Feature counts to visit: big cuts through the long tail, small near the knee."""
    if n_start < 1 or n_min < 1:
        raise ValueError("Feature counts must be positive")
    sizes = [int(n_start)]
    n = int(n_start)
    while n > n_min:
        frac = coarse if n > fine_below else fine
        n = max(n_min, min(n - 1, int(round(n * (1 - frac)))))
        sizes.append(n)
    return sizes


def aggregate_importance(per_fold: Sequence[np.ndarray]) -> np.ndarray:
    """Average of each fold's importance share, so no one fold dominates."""
    shares = []
    for importance in per_fold:
        importance = np.asarray(importance, dtype=np.float64)
        total = importance.sum()
        shares.append(importance / total if total > 0 else np.zeros_like(importance))
    return np.mean(shares, axis=0)


def keep_strongest(names: Sequence[str], importance: np.ndarray, n_keep: int) -> list[str]:
    """The *n_keep* most important names; ties broken by name, deterministically."""
    order = sorted(range(len(names)), key=lambda i: (-float(importance[i]), names[i]))
    return [names[i] for i in order[:n_keep]]


def mean_abs_shap(model, x: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature over *x*, in the booster's raw score."""
    booster = model.booster_
    total = np.zeros(x.shape[1], dtype=np.float64)
    for start in range(0, len(x), SHAP_CHUNK_ROWS):
        contrib = booster.predict(x[start:start + SHAP_CHUNK_ROWS], pred_contrib=True)
        total += np.abs(contrib[:, :-1]).sum(axis=0)
    return total / max(len(x), 1)


# ------------------------------------------------------------- fitting -----

@dataclass
class FoldData:
    """One walk-forward fold's rows, as positions into the shared matrix."""

    label: str
    train_rows: np.ndarray
    test_rows: np.ndarray


@dataclass
class Targets:
    """Row-aligned labels and context for the shared feature matrix."""

    won: np.ndarray                   # 0/1 outcome
    soft_y: np.ndarray | None         # production distillation target, or None
    race_ids: np.ndarray
    race_dates: pd.DatetimeIndex
    market: np.ndarray | None = None  # per-race market probability, tripwire only
    extra: dict = field(default_factory=dict)


def recency_weights(dates) -> np.ndarray:
    """Production recency weights, measured from this window's last race."""
    from src.model import compute_recency_sample_weights
    return compute_recency_sample_weights(pd.Series(dates))


def fit_win_model(x, y, soft_y, weights, race_ids, race_dates, *, capacity: str = "screen",
                  seed: int = 42, n_jobs: int = -1):
    """Fit the production win learner on one training window.

    ``capacity="screen"`` is the project's cheap screen (400 trees at 0.04, no
    early stopping). ``"native"`` is ``config.CLASSIFIER_PARAMS`` with the
    production purged early-stopping split and a refit on every row at the
    selected tree count, as ``RacePredictor.fit_as_of`` does.

    The loss is the per-runner cross-entropy of the 048–056 production learner,
    which every comparison up to 060 used. The 057 exploded objective, in
    production since 3 November 2026, is fitted by
    ``race_objectives.fit_boosted`` with ``ExplodedSoftmax``.
    """
    from src.model import RacePredictor
    learner = RacePredictor(random_seed=seed, frameworks={"classifier": "lgbm", "place": "lgbm"})
    learner._autotune_njobs = n_jobs
    learner.eval_split_info = {}
    kwargs = {"sample_weight": weights, "soft_y": soft_y, "objective": "cross_entropy"}
    if capacity == "screen":
        return learner._train_win_classifier(x, y, params=SCREEN_PARAMS, **kwargs)
    if capacity != "native":
        raise ValueError(f"Unknown capacity {capacity!r}")
    groups = race_groups(race_ids)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    split = (RacePredictor._early_stop_split(groups, pd.DatetimeIndex(race_dates)[starts],
                                             purge_days=int(config.PURGE_DAYS))
             if int(config.EARLY_STOPPING_ROUNDS) > 0 else None)
    if split is None:
        return learner._train_win_classifier(x, y, **kwargs)
    end, validation = split
    selected = learner._train_win_classifier(
        x[:end], y[:end], sample_weight=weights[:end], eval_set=[(x[validation:], y[validation:])],
        soft_y=soft_y[:end] if soft_y is not None else None, objective="cross_entropy",
    )
    return learner._refit_selected_model(selected, "classifier", learner._train_win_classifier,
                                         x, y, None, **kwargs)


def evaluate_subset(
    take: Callable[[np.ndarray, list[str]], np.ndarray],
    features: list[str],
    folds: Sequence[FoldData],
    targets: Targets,
    *,
    capacity: str = "screen",
    seed: int = 42,
    n_jobs: int = -1,
    importance: bool = True,
) -> dict:
    """Fit and score *features* on every fold; optionally measure OOS importance.

    *take(rows, names)* returns the float32 matrix for those rows and columns,
    so callers decide whether it comes from memory or a memory map.
    """
    per_fold, fold_importance, race_rows = [], [], []
    for fold in folds:
        started = time.monotonic()
        train, test = fold.train_rows, fold.test_rows
        y = targets.won[train]
        soft = targets.soft_y[train] if targets.soft_y is not None else None
        x_train = take(train, features)
        model = fit_win_model(x_train, y, soft, recency_weights(targets.race_dates[train]),
                              targets.race_ids[train], targets.race_dates[train],
                              capacity=capacity, seed=seed, n_jobs=n_jobs)
        del x_train
        x_test = take(test, features)
        groups = race_groups(targets.race_ids[test])
        probabilities = race_normalise(model.predict_proba(x_test)[:, 1], groups)
        metrics, race_loss, race_brier = score_races(probabilities, targets.won[test], groups)
        starts = np.r_[0, np.cumsum(groups)[:-1]]
        frame = pd.DataFrame({
            "fold": fold.label, "race_id": targets.race_ids[test][starts],
            "race_date": targets.race_dates[test][starts], "race_log_loss": race_loss,
            "race_brier": race_brier,
        })
        if targets.market is not None:
            frame["market_log_loss"] = _market_race_loss(targets.market[test], targets.won[test], groups)
        race_rows.append(frame)
        if importance:
            fold_importance.append(mean_abs_shap(model, x_test))
        del x_test, model
        per_fold.append({"fold": fold.label, "races": int(len(groups)), **metrics,
                         "seconds": round(time.monotonic() - started, 1)})
        logger.info("  %s: %d features, race LL %.5f (%.0fs)", fold.label, len(features),
                    metrics["race_log_loss"], per_fold[-1]["seconds"])
    races = pd.concat(race_rows, ignore_index=True)
    result = {"features": list(features), "folds": per_fold, "races": races,
              "pooled": pooled_metrics(races)}
    if importance:
        result["importance"] = aggregate_importance(fold_importance)
    return result


def _market_race_loss(market, won, groups) -> np.ndarray:
    """Race log loss of the market where its field is complete, else NaN."""
    market = np.asarray(market, dtype=np.float64)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    complete = np.minimum.reduceat(np.where(np.isfinite(market) & (market > 0), 1, 0), starts) == 1
    safe = np.where(np.isfinite(market) & (market > 0), market, 1.0)
    totals = np.add.reduceat(safe, starts)
    p = safe / np.repeat(totals, groups)
    loss = -np.add.reduceat(np.asarray(won, dtype=np.float64) * np.log(np.clip(p, 1e-12, None)), starts)
    return np.where(complete, loss, np.nan)


def pooled_metrics(races: pd.DataFrame) -> dict:
    """Pooled race log loss and Brier, plus the market tripwire where present."""
    out = {"races": int(len(races)), "race_log_loss": float(races.race_log_loss.mean()),
           "race_mean_brier": float(races.race_brier.mean())}
    if "market_log_loss" in races:
        priced = races.market_log_loss.notna()
        out["market_races"] = int(priced.sum())
        if priced.any():
            out["model_log_loss_priced"] = float(races.race_log_loss[priced].mean())
            out["market_log_loss_priced"] = float(races.market_log_loss[priced].mean())
            out["gap_to_market"] = out["model_log_loss_priced"] - out["market_log_loss_priced"]
    return out


def run_elimination(
    take: Callable[[np.ndarray, list[str]], np.ndarray],
    features: list[str],
    folds: Sequence[FoldData],
    targets: Targets,
    *,
    n_min: int = 30,
    seed: int = 42,
    n_jobs: int = -1,
    on_step: Callable[[int, dict], None] | None = None,
    resume: Callable[[int], dict | None] | None = None,
) -> list[dict]:
    """Recursive elimination along :func:`elimination_schedule`.

    *on_step(i, result)* persists a step as soon as it finishes and
    *resume(i)* returns a stored step (or None), so a long run can stop and
    restart without refitting completed steps.
    """
    sizes = elimination_schedule(len(features), n_min=n_min)
    steps: list[dict] = []
    current = list(features)
    for i, size in enumerate(sizes):
        if i:
            previous = steps[-1]
            current = keep_strongest(previous["features"], previous["importance"], size)
        stored = resume(i) if resume else None
        if stored is not None and stored["features"] == current:
            logger.info("Step %d (%d features): reused", i, size)
            steps.append(stored)
            continue
        logger.info("Step %d: %d features", i, size)
        result = evaluate_subset(take, current, folds, targets, seed=seed, n_jobs=n_jobs)
        result["step"] = i
        if on_step:
            on_step(i, result)
        steps.append(result)
    return steps
