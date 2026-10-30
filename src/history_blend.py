"""Research-only race-context mixtures of two independently fitted models.

Each race gets one coefficient, including every runner in that field. The
context is the availability of external prior form, known before the race.
Coefficients fit earlier OOS forecasts and can be rejected on later validation.
"""
from __future__ import annotations

import numpy as np

from src.forecast_evaluation import validate_probabilities
from src.tail_calibration import blend, fit_weight, validate_weight

CONTEXTS = ("no_external_form", "contains_external_form")
MIN_RACES = 50


def _groups(groups, rows):
    g = np.asarray(groups, dtype=float)
    if (g.ndim != 1 or not len(g) or not np.isfinite(g).all()
            or (g < 2).any() or (g != np.floor(g)).any() or g.sum() != rows):
        raise ValueError("Expected complete races with integer group sizes")
    return g.astype(np.int64)


def contexts_from_flags(flags, groups):
    """One context per complete race; no outcome input exists in this API."""
    flags = np.asarray(flags, dtype=float)
    if flags.ndim != 1 or not np.isin(flags, [0, 1]).all():
        raise ValueError("External-form flags must be observed binary values")
    g = _groups(groups, len(flags))
    return np.maximum.reduceat(flags, np.r_[0, np.cumsum(g)[:-1]]).astype(np.int8)


def apply_context_blend(baseline, history, groups, contexts, weights):
    p, q = (np.asarray(v, dtype=float) for v in (baseline, history))
    result = blend(p, q, 0)  # Shared vector validation and exact identity.
    g = _groups(groups, len(p))
    context = np.asarray(contexts)
    if context.shape != g.shape or not np.isin(context, [0, 1]).all():
        raise ValueError("Expected one binary context per race")
    starts = np.r_[0, np.cumsum(g)[:-1]]
    if any(not np.allclose(np.add.reduceat(v, starts), 1, atol=1e-8, rtol=0) for v in (p, q)):
        raise ValueError("Both forecasts must normalize over every complete field")
    if set(weights) != set(CONTEXTS):
        raise ValueError("Specify both context coefficients")
    for i, name in enumerate(CONTEXTS):
        # Validate even a coefficient whose context is absent from this batch.
        weight = weights[name]
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Context coefficients must be in [0, 1]")
        mask = np.repeat(context == i, g)
        if mask.any():
            result[mask] = blend(p[mask], q[mask], weight)
    return result


def fit_context_blend(fit, validation):
    """Fit and validate each fixed context; identity if data or scores fail.

    Inputs are tuples (baseline, history, winners, groups, contexts). The
    caller enforces chronology. The existing convex log-loss optimizer and
    Brier constraint are reused; no parameter grid or test-based fallback.
    """
    prepared = []
    for p, q, y, g, context in (fit, validation):
        p, y, g, _ = validate_probabilities(p, y, g)
        q, _, _, _ = validate_probabilities(q, y, g)
        c = np.asarray(context)
        if c.shape != g.shape or not np.isin(c, [0, 1]).all():
            raise ValueError("Expected one binary context per complete race")
        prepared.append((p, q, y, g, c))
    selection = {}
    for i, name in enumerate(CONTEXTS):
        arrays = []
        counts = []
        for p, q, y, g, context in prepared:
            chosen = context == i
            mask = np.repeat(chosen, g)
            counts.append(int(chosen.sum()))
            arrays.append((p[mask], q[mask], y[mask], g[chosen]))
        if min(counts) < MIN_RACES:
            selection[name] = {"fit_races": counts[0], "validation_races": counts[1],
                               "selected_weight": 0., "reason": "Insufficient earlier fields"}
            continue
        learned = fit_weight(*arrays[0])
        accepted = validate_weight(learned["weight"], *arrays[1])
        selection[name] = {"fit_races": counts[0], "validation_races": counts[1],
                           "fit": learned, "validation": accepted,
                           "selected_weight": accepted["selected_weight"]}
    return selection
