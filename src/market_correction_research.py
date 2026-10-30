"""Small, bounded market corrections with a pre-test identity fallback.

Research-only: source/horizon contracts are supplied by the registered caller.
The independent predictor and its feature pipeline are unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _inputs(model, market, groups):
    model = np.asarray(model, dtype=float)
    market = np.asarray(market, dtype=float)
    raw_groups = np.asarray(groups)
    if (raw_groups.ndim != 1 or raw_groups.size == 0 or not np.isfinite(raw_groups).all()
            or (raw_groups < 2).any() or not np.equal(raw_groups, np.floor(raw_groups)).all()):
        raise ValueError("Use complete groups with at least two runners")
    groups = raw_groups.astype(np.int64)
    if (model.ndim != 1 or market.shape != model.shape or groups.sum() != len(model)
            or not np.isfinite(model).all() or not np.isfinite(market).all()
            or (model < 0).any() or (model > 1).any() or (market <= 0).any() or (market > 1).any()):
        raise ValueError("Invalid model/market probability arrays")
    offsets = np.r_[0, np.cumsum(groups)[:-1]]
    for values in (model, market):
        if not np.allclose(np.add.reduceat(values, offsets), 1, atol=1e-9, rtol=0):
            raise ValueError("Every whole-field probability book must sum to one")
    return model, market, groups, offsets


def apply_correction(model, market, groups, alpha=0., beta=1.):
    """Normalize model**alpha * market**beta within each whole field."""
    model, market, groups, offsets = _inputs(model, market, groups)
    if not (np.isfinite(alpha) and np.isfinite(beta) and 0 <= alpha <= 1 and .5 <= beta <= 1.5):
        raise ValueError("Correction coefficients outside the registered bounds")
    if alpha == 0 and beta == 1:
        return market.copy()
    logs = alpha*np.log(np.clip(model, 1e-6, 1)) + beta*np.log(market)
    logits = logs - np.repeat(np.maximum.reduceat(logs, offsets), groups)
    values = np.exp(logits)
    return values / np.repeat(np.add.reduceat(values, offsets), groups)


def correction_objective(parameters, model, market, groups, won, ridge=.001):
    """Convex winner log loss and analytic gradient, with a market-centred ridge."""
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("The market-centred ridge must be nonnegative")
    model, market, groups, offsets = _inputs(model, market, groups)
    won = np.asarray(won, dtype=float)
    if (won.shape != model.shape or not np.isin(won, [0, 1]).all()
            or not np.equal(np.add.reduceat(won, offsets), 1).all()):
        raise ValueError("One observed winner is required in each field")
    alpha, beta = np.asarray(parameters, dtype=float)
    probabilities = apply_correction(model, market, groups, alpha, beta)
    log_model = np.log(np.clip(model, 1e-6, 1))
    log_market = np.log(market)
    loss = -np.log(probabilities[won.astype(bool)]).mean()
    penalty = ridge * (alpha*alpha + (beta-1)*(beta-1))
    error = probabilities - won
    gradient = np.array([(error*log_model).sum(), (error*log_market).sum()]) / len(groups)
    gradient += 2*ridge*np.array([alpha, beta-1])
    return float(loss + penalty), gradient


def fit_correction(model, market, groups, won, *, model_allowed, ridge=.001):
    """Solve inside the coefficient box rather than clipping an outside optimum."""
    bounds = [(0., 1.) if model_allowed else (0., 0.), (.5, 1.5)]
    result = minimize(correction_objective, [0., 1.],
        args=(model, market, groups, won, ridge), method="L-BFGS-B", jac=True,
        bounds=bounds, options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
    baseline, _ = correction_objective([0., 1.], model, market, groups, won, ridge)
    accepted = bool(result.success and np.isfinite(result.fun) and result.fun <= baseline + 1e-10)
    return {"alpha": float(result.x[0]) if accepted else 0.,
            "beta": float(result.x[1]) if accepted else 1.,
            "converged": bool(result.success), "accepted_fit": accepted,
            "iterations": int(result.nit), "ridge": float(ridge),
            "objective": float(result.fun) if np.isfinite(result.fun) else None,
            "market_objective": baseline}
