"""Research-only relative floors for severely suppressed tail probabilities.

The forecast is raised only below 1%, and only up to a small fraction of its
pre-isotonic Platt probability. Whole-field normalization redistributes the
added mass. No model/serving default is changed by importing this module.
"""
from __future__ import annotations

import numpy as np

from src.forecast_evaluation import probability_metrics

TAIL_THRESHOLD = .01
STRENGTHS = (0., .001, .003, .01, .03, .1, .25)


def _arrays(existing, platt, groups):
    p, q, sizes = (np.asarray(v, dtype=float) for v in (existing, platt, groups))
    if (p.ndim != 1 or q.shape != p.shape or sizes.ndim != 1 or not len(sizes)
            or not np.isfinite(sizes).all() or (sizes < 2).any()
            or (sizes != np.floor(sizes)).any() or sizes.sum() != len(p)):
        raise ValueError("Expected aligned complete race distributions and integer group sizes")
    sizes = sizes.astype(np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    if (not np.isfinite(p).all() or not np.isfinite(q).all()
            or ((p < 0) | (p > 1)).any() or ((q <= 0) | (q > 1)).any()):
        raise ValueError("Existing probabilities must be nonnegative; Platt probabilities strictly positive")
    for value in (p, q):
        if not np.allclose(np.add.reduceat(value, starts), 1, atol=1e-8, rtol=0):
            raise ValueError("Every input field must sum to one; partial fields cannot be repaired")
    return p, q, sizes, starts


def smooth_tail(existing, platt, groups, strength):
    """Apply max(p, strength*q) below 1%, then normalize each whole race.

    The fixed cap .25 means only forecasts suppressed by at least 75% relative
    to their Platt stage can receive extra mass. Other runners keep their
    relative odds. The denominator is at most 1+strength, giving an affected
    runner a lower bound strength*q/(1+strength). Identity/no activation are
    exact. This inference function never needs outcomes.
    """
    p, q, sizes, starts = _arrays(existing, platt, groups)
    if not np.isfinite(strength) or not 0 <= strength <= max(STRENGTHS):
        raise ValueError("Tail strength must be in [0, .25]")
    if strength == 0:
        return p.copy()
    unnormalized = np.maximum(p, np.where(p < TAIL_THRESHOLD, strength * q, 0.))
    if np.array_equal(unnormalized, p):
        return p.copy()
    affected = np.add.reduceat((unnormalized > p).astype(int), starts) > 0
    denominators = np.where(affected, np.add.reduceat(unnormalized, starts), 1.)
    return unnormalized / np.repeat(denominators, sizes)


def fit_strength(existing, platt, winners, groups):
    """Choose one fixed-grid coefficient on earlier out-of-sample forecasts.

    The seven strengths, 1% tail boundary and 25% cap are fixed before scoring.
    Winner log loss selects among fits with no worse runner Brier/binary loss.
    Validation may reject this coefficient but must not choose a different one.
    """
    p, q, sizes, _ = _arrays(existing, platt, groups)
    original = probability_metrics(p, winners, sizes)[0]
    best_strength, best_loss = 0., original["race_log_loss"]
    candidates = []
    for strength in STRENGTHS:
        metrics = probability_metrics(smooth_tail(p, q, sizes, strength), winners, sizes)[0]
        eligible = bool(metrics["brier"] <= original["brier"] + 1e-12
                        and metrics["binary_log_loss"] <= original["binary_log_loss"] + 1e-12)
        candidates.append({"strength": strength, "metrics": metrics, "eligible": eligible})
        if eligible and metrics["race_log_loss"] < best_loss - 1e-12:
            best_strength, best_loss = strength, metrics["race_log_loss"]
    return {"learned_strength": best_strength, "original": original, "candidates": candidates}


def validate_strength(strength, existing, platt, winners, groups):
    """Accept the earlier coefficient or identity; do not refit on validation."""
    p, q, sizes, _ = _arrays(existing, platt, groups)
    original = probability_metrics(p, winners, sizes)[0]
    adjusted = probability_metrics(smooth_tail(p, q, sizes, strength), winners, sizes)[0]
    accepted = bool(strength > 0
                    and adjusted["race_log_loss"] < original["race_log_loss"] - 1e-12
                    and adjusted["brier"] <= original["brier"] + 1e-12
                    and adjusted["binary_log_loss"] <= original["binary_log_loss"] + 1e-12)
    return {"learned_strength": float(strength), "accepted": accepted,
            "selected_strength": float(strength) if accepted else 0.,
            "original": original, "adjusted": adjusted}


def activation_summary(existing, platt, groups, strength):
    """Outcome-free coverage and mass redistribution, including normalization."""
    p, q, sizes, starts = _arrays(existing, platt, groups)
    adjusted = smooth_tail(p, q, sizes, strength)
    active = (p < TAIL_THRESHOLD) & (strength * q > p)
    lifted_mass = np.add.reduceat(np.maximum(adjusted - p, 0), starts)
    return {"directly_lifted_runners": int(active.sum()),
            "affected_races": int(np.add.reduceat(active.astype(int), starts).astype(bool).sum()),
            "mean_redistributed_probability_mass_per_race": float(lifted_mass.mean()),
            "max_redistributed_probability_mass_per_race": float(lifted_mass.max()),
            "max_probability_change": float(np.max(np.abs(adjusted - p)))}
