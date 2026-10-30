"""Use the raw place head as an auxiliary ranking channel; research only.

The place classifier's log odds are scores, not asserted win probabilities.
They are never the served place probabilities, which already contain win floors.
The bounded solver is shared with036; no current market input is required.
"""
from __future__ import annotations

import numpy as np

from src.market_correction_research import _inputs, apply_correction, fit_correction


def place_utility_book(win, raw_place, groups):
    """Normalise raw place log-odds scores within complete fields.

    The win book validates the field layout but cannot change these scores.
    Numerical clipping precedes log odds; the shared solver also floors its
    auxiliary-book logarithm at1e-6. Neither operation changes the identity arm.
    """
    win, _, groups, offsets = _inputs(win, win, groups)
    raw = np.asarray(raw_place, dtype=float)
    if raw.shape != win.shape or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("One finite raw place probability per runner is required")
    clipped = np.clip(raw, 1e-6, 1 - 1e-6)
    scores = np.log(clipped) - np.log1p(-clipped)
    values = np.exp(scores - np.repeat(np.maximum.reduceat(scores, offsets), groups))
    return values / np.repeat(np.add.reduceat(values, offsets), groups)


def auxiliary_win_probabilities(win, raw_place, groups, *, alpha=0., beta=1.):
    book = place_utility_book(win, raw_place, groups)
    return apply_correction(book, win, groups, alpha=alpha, beta=beta)


def fit_auxiliary_win(win, raw_place, groups, won, *, allow_place=True, ridge=.001):
    book = place_utility_book(win, raw_place, groups)
    fitted = fit_correction(book, win, groups, won, model_allowed=allow_place, ridge=ridge)
    fitted["baseline_objective"] = fitted.pop("market_objective")
    return fitted
