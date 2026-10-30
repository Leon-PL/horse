"""Research-only shrinkage of isotonic forecasts toward their Platt stage.

This module does not change serving or fit trees. A single mixture coefficient
is learned on earlier out-of-sample forecasts, subject to no increase in their
runner Brier score. A later validation slab can reject it in favour of identity.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from src.forecast_evaluation import probability_metrics, validate_probabilities


def _pair(existing, platt, winners, groups):
    p, y, g, starts = validate_probabilities(existing, winners, groups)
    q, _, _, _ = validate_probabilities(platt, winners, groups)
    if (p <= 0).any() or (q <= 0).any():
        raise ValueError("Both source forecasts must be strictly positive")
    return p, q, y, g, starts


def blend(existing, platt, weight):
    """Convex mixture; identity is exact and no extra clipping is introduced."""
    p, q = (np.asarray(v, dtype=float) for v in (existing, platt))
    if (p.shape != q.shape or p.ndim != 1 or not np.isfinite(p).all()
            or not np.isfinite(q).all() or ((p < 0) | (p > 1)).any()
            or ((q < 0) | (q > 1)).any()):
        raise ValueError("Expected aligned finite probability vectors")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Mixture weight must be in [0, 1]")
    return p.copy() if weight == 0 else (1 - weight) * p + weight * q


def fit_weight(existing, platt, winners, groups):
    """Minimise winner log loss on the Brier-feasible interval analytically.

    DeltaBrier(w) = A*w**2 + B*w. The log-loss objective is convex, so
    its derivative identifies either an endpoint or the unique minimum.
    Test predictions and labels must never be supplied to this function.
    """
    p, q, y, g, _ = _pair(existing, platt, winners, groups)
    d = q - p
    a = float(np.mean(d * d))
    b = float(2 * np.mean((p - y) * d))
    upper = min(1., -b / a) if a > 0 and b < 0 else 0.
    winner = y == 1
    pw, dw = p[winner], d[winner]

    def derivative(w):
        return float(-np.mean(dw / (pw + w * dw)))

    if upper == 0 or derivative(0) >= 0:
        weight = 0.
    elif derivative(upper) <= 0:
        weight = upper
    else:
        weight = float(brentq(derivative, 0., upper, xtol=1e-13))
    original = probability_metrics(p, y, g)[0]
    adjusted = probability_metrics(blend(p, q, weight), y, g)[0]
    if adjusted["race_log_loss"] >= original["race_log_loss"] - 1e-12:
        weight = 0.
        adjusted = original.copy()
    return {"weight": weight, "brier_feasible_upper": upper,
            "brier_quadratic_a": a, "brier_linear_b": b,
            "original": original, "adjusted": adjusted}


def validate_weight(weight, existing, platt, winners, groups):
    """Freeze the learned coefficient, or reject it. Never refit on validation."""
    p, q, y, g, _ = _pair(existing, platt, winners, groups)
    original = probability_metrics(p, y, g)[0]
    adjusted = probability_metrics(blend(p, q, weight), y, g)[0]
    accepted = bool(weight > 0
                    and adjusted["race_log_loss"] < original["race_log_loss"] - 1e-12
                    and adjusted["brier"] <= original["brier"] + 1e-12
                    and adjusted["binary_log_loss"] <= original["binary_log_loss"] + 1e-12)
    return {"learned_weight": float(weight), "accepted": accepted,
            "selected_weight": float(weight) if accepted else 0.,
            "original": original, "adjusted": adjusted,
            "rule": "Lower race log loss; runner Brier and binary log loss no worse"}
