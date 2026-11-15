"""Shared walk-forward evaluation primitives — the measuring instrument.

Every edge claim in this project is decided by a walk-forward harness, and
those harnesses were written one at a time, each re-implementing folds,
confidence intervals and second-stage fitting. That duplication produced real,
decision-changing errors:

* **Second-stage models were fitted on in-sample first-stage predictions.**
  ``scripts/wf_benter_combination.py`` trained a win model on the fold's
  training rows, then fitted the market blend on that same model's *training*
  predictions. In-sample probabilities are far sharper than out-of-sample
  ones, so the blend learned to over-trust the model and under-weight the
  market; applied to the test quarter it then over-trusted a much weaker
  signal. The measured ROI was biased **downward**, and on that basis the
  market-combination lead was recorded as dead. :func:`oof_predict_expanding`
  exists so a second stage is never again fitted on in-sample predictions.

* **Confidence intervals resampled individual bets.** Bets are not
  independent: several runners in one race share the same conditions and are
  mutually exclusive, and bets in one quarter share a model and a market
  regime. An i.i.d. bootstrap over bets therefore reports an interval that is
  narrower than the data supports, which is how a result gets called
  "significant". :func:`bootstrap_roi_ci` resamples whole races, and
  :func:`fold_summary` reports the across-time spread that actually matters.

* **Research folds had no purge gap** while the production training path
  purges ``config.PURGE_DAYS``. Two protocols means two incomparable numbers.
  :func:`expanding_period_folds` applies the production purge by default.

Use these helpers rather than hand-rolling a fold loop. See
``docs/ALREADY_TRIED.md`` for the bar an edge claim has to clear.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

#: Bootstrap draws used unless a caller asks for more.
DEFAULT_N_BOOT = 2000

#: Index-matrix cells held in memory per bootstrap chunk. A long backtest has
#: tens of thousands of races, and a full (n_boot x n_clusters) matrix would be
#: gigabytes; the statistic is a ratio of sums, so chunking is exact.
BOOT_CHUNK_CELLS = 4_000_000


def purge_days() -> int:
    """The project-wide train/test embargo, so research matches production."""
    return max(0, int(config.PURGE_DAYS))


def golden_holdout_start() -> pd.Timestamp:
    """First race date of the pre-registered golden hold-out (config)."""
    return pd.Timestamp(config.GOLDEN_HOLDOUT_START)


def development_mask(race_dates) -> np.ndarray:
    """Rows research may use: strictly before the golden hold-out, minus the
    purge gap, so rolling features cannot straddle into it.

    ACTION_PLAN 7 only counts an edge that also holds on data never used for
    tuning. Every harness filters with this before building folds, and scores
    the hold-out separately and once (:func:`holdout_mask`).
    """
    dates = pd.to_datetime(pd.Series(race_dates).reset_index(drop=True), errors="coerce")
    cutoff = golden_holdout_start() - pd.Timedelta(days=purge_days())
    return (dates < cutoff).to_numpy()


def holdout_mask(race_dates) -> np.ndarray:
    """Rows in the golden hold-out."""
    dates = pd.to_datetime(pd.Series(race_dates).reset_index(drop=True), errors="coerce")
    return (dates >= golden_holdout_start()).to_numpy()


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: train strictly before test, with a purge gap."""

    label: str
    train: np.ndarray          # boolean mask over the input frame
    test: np.ndarray           # boolean mask over the input frame
    n_purged: int              # training rows dropped by the embargo

    @property
    def n_train(self) -> int:
        return int(self.train.sum())

    @property
    def n_test(self) -> int:
        return int(self.test.sum())


def expanding_period_folds(
    race_dates,
    *,
    freq: str = "Q",
    min_train_periods: int = 8,
    min_train_rows: int = 1000,
    embargo_days: int | None = None,
) -> list[Fold]:
    """Expanding-window folds over calendar periods, with a purge gap.

    Train on everything before period *p* and test on *p* alone. Training rows
    within *embargo_days* of the test period's first race are dropped: rolling
    and target-encoded features straddle the boundary, so without the gap the
    training set partly describes the same recent form the test set is being
    judged on. Defaults to :func:`purge_days` to match the production path.
    """
    dates = pd.to_datetime(pd.Series(race_dates).reset_index(drop=True), errors="coerce")
    gap = purge_days() if embargo_days is None else max(0, int(embargo_days))
    periods = dates.dt.to_period(freq)
    ordered = sorted(p for p in periods.dropna().unique())

    folds: list[Fold] = []
    for period in ordered[min_train_periods:]:
        test = (periods == period).to_numpy()
        if not test.any():
            continue
        before = (periods < period).to_numpy()
        test_start = dates[test].min()
        cutoff = test_start - pd.Timedelta(days=gap)
        train = before & (dates <= cutoff).to_numpy()
        n_purged = int(before.sum() - train.sum())
        if train.sum() < min_train_rows:
            continue
        folds.append(Fold(str(period), train, test, n_purged))
    if not folds:
        logger.warning(
            "No usable folds: need %d training rows after a %d-day embargo.",
            min_train_rows, gap,
        )
    return folds


def oof_predict_expanding(
    frame: pd.DataFrame,
    *,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    date_col: str = "race_date",
    n_splits: int = 4,
    embargo_days: int | None = None,
    min_train_rows: int = 1000,
    folds: Sequence[Fold] | None = None,
    race_col: str | None = None,
) -> np.ndarray:
    """Out-of-fold predictions for *frame*, for fitting a second stage.

    Accepts explicit calendar ``folds`` from :func:`expanding_period_folds`.
    Without them, splits complete dates into *n_splits* time blocks and predicts each
    block from a model fitted only on earlier rows (minus the embargo). The
    earliest block has no history to learn from and is returned as ``NaN``, so
    callers must mask non-finite values before fitting on the result.

    **This is the function that makes a stacked/blended model honest.** Fitting
    a blend on predictions a model made over its own training rows teaches the
    blend to trust a sharpness the model will not have at serving time.

    ``fit_predict(train_df, predict_df)`` must return one probability per row
    of *predict_df*.
    """
    dates = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("Walk-forward dates must be present and valid")
    out = np.full(len(frame), np.nan, dtype=np.float64)
    gap = purge_days() if embargo_days is None else max(0, int(embargo_days))
    if folds is None:
        blocks = np.array_split(np.sort(dates.unique()), max(2, int(n_splits)))
        folds = []
        for i, block in enumerate(blocks[1:], 1):
            if not len(block):
                continue
            before = dates.lt(block[0]).to_numpy()
            train = before & dates.le(pd.Timestamp(block[0]) - pd.Timedelta(days=gap)).to_numpy()
            if train.sum() >= min_train_rows:
                folds.append(Fold(str(i), train, dates.isin(block).to_numpy(), int(before.sum() - train.sum())))
    validate_forward_folds(frame, folds, date_col=date_col, race_col=race_col, embargo_days=gap)
    order = np.argsort(dates.to_numpy(), kind="stable")
    for fold in folds:
        train_rows, block = order[fold.train[order]], order[fold.test[order]]
        if len(train_rows) < min_train_rows:
            raise ValueError(f"Fold {fold.label} has insufficient training rows")
        preds = np.asarray(
            fit_predict(frame.iloc[train_rows], frame.iloc[block]), dtype=np.float64,
        )
        if preds.ndim != 1 or len(preds) != len(block):
            raise ValueError(
                f"fit_predict returned {len(preds)} predictions for {len(block)} rows",
            )
        out[block] = preds
    return out


def validate_forward_folds(frame, folds, *, date_col="race_date", race_col=None, embargo_days=None):
    """Reject overlapping, partial-day/race or nonchronological OOF folds.

    Masks are positional, independent of a caller's DataFrame index. All folds
    are checked before any learner runs, so a later malformed fold cannot leave
    a partially produced prediction cache.
    """
    dates = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize().reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("Walk-forward dates must be present and valid")
    race_col = race_col or ("race_id" if "race_id" in frame else None)
    if race_col:
        ids = frame[race_col].reset_index(drop=True)
        if ids.isna().any() or dates.groupby(ids, dropna=False).nunique().gt(1).any():
            raise ValueError("Each race needs one valid date and identity")
    gap = purge_days() if embargo_days is None else max(0, int(embargo_days))
    seen, labels = np.zeros(len(frame), dtype=bool), set()
    for fold in folds:
        if fold.label in labels:
            raise ValueError("Fold labels must be unique")
        labels.add(fold.label)
        for mask in (fold.train, fold.test):
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != (len(frame),):
                raise ValueError("Fold masks must be positional boolean arrays matching the frame")
            selected = pd.Series(mask)
            if selected.groupby(dates).nunique().gt(1).any():
                raise ValueError("A fold cannot split a date or race")
        if not fold.train.any() or not fold.test.any() or (fold.train & fold.test).any():
            raise ValueError("Folds need disjoint nonempty training and prediction sets")
        start = dates[fold.test].min()
        end = dates[fold.train].max()
        if end >= start or end > start - pd.Timedelta(days=gap):
            raise ValueError("Fold training violates the chronological purge boundary")
        if (seen & fold.test).any():
            raise ValueError("A row may be predicted in only one OOF fold")
        seen |= fold.test


def _cluster_means(pnl: np.ndarray, clusters) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster PnL sums and counts."""
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(inverse.max()) + 1 if len(inverse) else 0
    sums = np.bincount(inverse, weights=pnl, minlength=n_clusters)
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    return sums, counts


def bootstrap_roi_ci(
    pnl,
    *,
    clusters=None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    scale: float = 100.0,
) -> dict:
    """Bootstrap CI for mean PnL, resampling *clusters* (normally races).

    Pass ``clusters=race_ids``. Resampling whole races keeps correlated bets
    together, which an i.i.d. bootstrap over bets does not — and that
    difference decides whether an interval excludes zero. With ``clusters``
    omitted this degrades to the i.i.d. bootstrap, which is only correct when
    every bet is independent.

    Returns ``{"mean", "ci_low", "ci_high", "n", "n_clusters", "clustered"}``,
    all PnL figures multiplied by *scale* (percent by default).
    """
    pnl = np.asarray(pnl, dtype=np.float64).ravel()
    n = len(pnl)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None,
                "n": 0, "n_clusters": 0, "clustered": clusters is not None}

    rng = np.random.default_rng(seed)
    if clusters is None:
        sums, counts = pnl, np.ones(n)
    else:
        sums, counts = _cluster_means(pnl, clusters)
    n_units = len(sums)

    means = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, min(n_boot, BOOT_CHUNK_CELLS // max(n_units, 1) or 1))
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n_units, size=(size, n_units))
        drawn_counts = counts[idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[done:done + size] = sums[idx].sum(axis=1) / drawn_counts
        done += size

    means = means[np.isfinite(means)] * scale
    lo, hi = (np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if len(means) else (np.nan, np.nan))
    return {
        "mean": round(float(pnl.mean() * scale), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n": n,
        "n_clusters": n_units if clusters is not None else n,
        "clustered": clusters is not None,
    }


def bootstrap_median_ci(
    values,
    *,
    clusters,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
    log_mean: bool = False,
) -> dict:
    """Race-clustered bootstrap CI for the median (or geometric mean) of *values*.

    ACTION_PLAN 7's bar is "median CLV > 1.0 with a race-clustered CI
    excluding 1.0"; :func:`bootstrap_roi_ci` only handles means. Each
    replicate draws races with replacement, which is equivalent to giving every
    value its race's draw count as a weight, so the median is a weighted median
    over values sorted once. ``log_mean=True`` reports the geometric mean.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    c = np.asarray(clusters).ravel()
    keep = np.isfinite(v) & ((v > 0) if log_mean else True)
    v, c = v[keep], c[keep]
    if len(v) == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0, "n_clusters": 0}
    if log_mean:
        v = np.log(v)
    order = np.argsort(v, kind="stable")
    v, c = v[order], c[order]
    _, inverse = np.unique(c, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    uniform = np.full(n_clusters, 1.0 / n_clusters)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        weights = rng.multinomial(n_clusters, uniform)[inverse].astype(np.float64)
        if log_mean:
            stats[b] = np.dot(weights, v) / weights.sum()
        else:
            cum = np.cumsum(weights)
            stats[b] = v[np.searchsorted(cum, cum[-1] / 2.0)]
    point = float(v.mean()) if log_mean else float(np.median(v))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if log_mean:
        point, lo, hi = float(np.exp(point)), float(np.exp(lo)), float(np.exp(hi))
    return {"estimate": round(point, 6), "ci_low": round(float(lo), 6),
            "ci_high": round(float(hi), 6), "n": int(len(v)), "n_clusters": n_clusters}


def clv_summary(entry_odds, close_odds) -> dict:
    """CLV of a set of bets, reported on every measure that matters.

    Delegates to :func:`src.clv.summarise_clv`, which documents why the mean
    of a price ratio must never be read on its own.
    """
    from src.clv import summarise_clv

    entry = np.asarray(entry_odds, dtype=np.float64)
    close = np.asarray(close_odds, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return summarise_clv(entry / close)


def fold_summary(fold_rois: Sequence[float]) -> dict:
    """Across-time dispersion of per-fold ROI — the read that survives.

    A pooled ROI with a tight interval can still come from one lucky quarter.
    How many folds were positive, and how wide the fold-to-fold spread is,
    answers "would this have worked whenever I switched it on?".
    """
    values = np.asarray([r for r in fold_rois if np.isfinite(r)], dtype=np.float64)
    if len(values) == 0:
        return {"n_folds": 0, "folds_positive": 0, "mean": None,
                "median": None, "std": None, "worst": None, "best": None}
    return {
        "n_folds": int(len(values)),
        "folds_positive": int((values > 0).sum()),
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4),
        "worst": round(float(values.min()), 4),
        "best": round(float(values.max()), 4),
    }


def check_combination_nesting(
    combined_logloss: float,
    market_logloss: float,
    *,
    tolerance: float = 0.01,
) -> dict:
    """Diagnostic tripwire for observed loss exceeding a chosen tolerance.

    The family contains the market at a=0, b=1. This constrains a successfully
    optimised, unpenalised *fitting* objective, not out-of-sample performance.
    Worse test scores can reflect overfitting, sampling noise or regime change
    as well as a pipeline error. This check alone cannot distinguish them.
    ``ok`` means the observed excess is within the caller's tolerance; it is
    neither an equivalence test nor a guarantee that the fit is correct.
    """
    if not (np.isfinite(combined_logloss) and np.isfinite(market_logloss)):
        return {"ok": True, "reason": "log loss unavailable; nesting not checked",
                "combined_logloss": combined_logloss, "market_logloss": market_logloss}
    excess = float(combined_logloss - market_logloss)
    ok = excess <= float(tolerance)
    return {
        "ok": ok,
        "excess_logloss": round(excess, 6),
        "combined_logloss": round(float(combined_logloss), 6),
        "market_logloss": round(float(market_logloss), 6),
        "reason": (
            "observed combination loss is within the allowed tolerance of market-only"
            if ok else
            f"combination is {excess:.4f} nats worse than market-only; inspect "
            "generalisation and sampling uncertainty, and suspect the fit if "
            "training-data checks also fail. Out-of-sample worsening alone "
            "does not prove a pipeline error"
        ),
    }


def race_logloss(probs, won, race_ids) -> float:
    """Mean negative log-probability assigned to the actual winner per race.

    The natural score for a per-race, exactly-one-winner problem, and the
    quantity :func:`check_combination_nesting` compares. Races without exactly
    one recorded winner are skipped.
    """
    frame = pd.DataFrame({
        "p": pd.to_numeric(pd.Series(probs).reset_index(drop=True), errors="coerce"),
        "won": pd.to_numeric(pd.Series(won).reset_index(drop=True), errors="coerce").fillna(0),
        "race": pd.Series(race_ids).reset_index(drop=True),
    })
    winners = frame[frame["won"] == 1]
    valid = winners["race"].value_counts()
    valid = set(valid[valid == 1].index)
    winners = winners[winners["race"].isin(valid)]
    if winners.empty:
        return float("nan")
    return float(-np.log(np.clip(winners["p"].to_numpy(), 1e-12, None)).mean())
