"""Race-level temperature calibration shared by training and benchmarks.

Inputs contain contiguous runners for each race; ``groups`` contains their
field sizes. Temperature scales log probabilities, retaining within-race
rankings while producing a single probability distribution over each field.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _race_inputs(probabilities, groups):
    p = np.asarray(probabilities, dtype=np.float64)
    sizes = np.asarray(groups)
    if p.ndim != 1 or sizes.ndim != 1:
        raise ValueError("Probabilities and race sizes must be one-dimensional.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and between zero and one.")
    if (not np.all(np.isfinite(sizes)) or np.any(sizes <= 0)
            or np.any(sizes != np.floor(sizes)) or sizes.sum() != len(p)):
        raise ValueError("Positive integer race sizes must sum to the number of runners.")
    sizes = sizes.astype(np.intp)
    starts = np.r_[0, np.cumsum(sizes)[:-1]] if len(sizes) else np.array([], dtype=np.intp)
    return p, sizes, starts


def _temperature_probs(log_p, groups, starts, temperature):
    # Centre before dividing so even an extremely small positive temperature
    # cannot produce inf - inf. Negative overflow just gives zero probability.
    centred = log_p - np.repeat(np.maximum.reduceat(log_p, starts), groups)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(centred / temperature)
    return weights / np.repeat(np.add.reduceat(weights, starts), groups)


def race_temperature(probabilities, groups, temperature):
    """Apply one positive temperature; each race sums to one, even for zeros."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive.")
    p, groups, starts = _race_inputs(probabilities, groups)
    if not len(p):
        return p.copy()
    return _temperature_probs(np.log(np.clip(p, 1e-12, 1.0)), groups, starts, temperature)


def fit_temperature(probabilities, groups, winners):
    """Minimise mean winner log loss on held-out, single-winner races.

    No-winner races, dead heats and walkovers do not define the target used by
    this calibrator and are omitted as whole races. No usable races returns
    the identity temperature. Callers must keep these data out of model fit,
    early stopping and final evaluation.
    """
    p, groups, starts = _race_inputs(probabilities, groups)
    y = np.asarray(winners, dtype=np.float64)
    if y.shape != p.shape or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("Winners must contain one binary label per runner.")
    if not len(groups):
        return 1.0
    valid = (np.add.reduceat(y, starts) == 1) & (groups > 1)
    if not np.any(valid):
        return 1.0
    rows = np.repeat(valid, groups)
    p, y, groups = p[rows], y[rows], groups[valid]
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    log_p = np.log(np.clip(p, 1e-12, 1.0))

    def loss(log_temperature):
        calibrated = _temperature_probs(log_p, groups, starts, np.exp(log_temperature))
        return -float(np.sum(y * np.log(np.clip(calibrated, 1e-12, 1.0))) / len(groups))

    result = minimize_scalar(loss, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
    if not result.success or not np.isfinite(result.fun) or result.fun >= loss(0.0):
        return 1.0
    return float(np.exp(result.x))
