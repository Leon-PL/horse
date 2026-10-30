"""Training-only, date-local probability refinement candidates.

This module is deliberately downstream of feature engineering. Nothing here
changes an installed forecast until a separately evaluated candidate is saved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit


def calendar_split(dates, *, days=90, purge_days=7):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if dates.hasnans or days < 14 or purge_days < 1 or not len(dates):
        raise ValueError("Require dated history, at least14 calibration days and a positive purge")
    first = dates.max()-pd.Timedelta(days=days-1)
    train = dates <= first-pd.Timedelta(days=purge_days)
    calibrate = dates >= first
    if not train.any() or not calibrate.any():
        raise ValueError("Insufficient earlier history for calendar calibration")
    return train, calibrate


def _layout(groups, n):
    groups = np.asarray(groups, dtype=int)
    if groups.ndim != 1 or not len(groups) or (groups < 2).any() or groups.sum() != n:
        raise ValueError("Require complete contiguous fields with at least two runners")
    return groups, np.r_[0, np.cumsum(groups)[:-1]]


def normalise(values, groups):
    values = np.asarray(values, dtype=float)
    groups, starts = _layout(groups, len(values))
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Require finite positive probabilities")
    return values/np.repeat(np.add.reduceat(values, starts), groups)


@dataclass
class SmoothWinMap:
    slope: float = 1.
    intercept: float = 0.

    def predict(self, raw, groups):
        raw = np.asarray(raw, dtype=float)
        if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError("Invalid raw win probabilities")
        return normalise(expit(self.slope*logit(np.clip(raw, 1e-12, 1-1e-12))+self.intercept), groups)

    @classmethod
    def fit(cls, raw, won, groups):
        raw, won = np.asarray(raw, float), np.asarray(won, float)
        groups, starts = _layout(groups, len(raw))
        if won.shape != raw.shape or not np.isin(won, [0, 1]).all() or not np.all(np.add.reduceat(won, starts) == 1):
            raise ValueError("Win calibration requires one observed winner per field")
        def loss(ab):
            p = cls(*ab).predict(raw, groups)
            return float(-(won*np.log(p)).sum()/len(groups)+1e-4*((ab[0]-1)**2+ab[1]**2))
        result = minimize(loss, [1., 0.], method="L-BFGS-B", bounds=[(.1, 3.), (-3., 3.)])
        if not result.success or not np.isfinite(result.fun) or result.fun > loss([1., 0.])+1e-10:
            raise ValueError("Smooth win calibration did not converge to a non-worsening fit")
        return cls(*map(float, result.x))


def project_places(base, win, groups, places, *, strength=2., expected_slots=None):
    """Project onto feasible marginal place probabilities, in bounded batches.

    Expected filled places may be lower than the offered places. No runner
    can have a place probability below its win probability.
    """
    base, win = np.asarray(base, float), np.asarray(win, float)
    groups, starts = _layout(groups, len(base))
    places = np.asarray(places, float)
    target = places.copy() if expected_slots is None else np.asarray(expected_slots, float).copy()
    if places.shape != groups.shape or target.shape != groups.shape or win.shape != base.shape:
        raise ValueError("Place arrays do not align")
    if (not np.isfinite(base).all() or not np.isfinite(win).all() or not np.isfinite(target).all()
            or ((base < 0) | (base > 1) | (win < 0) | (win > 1)).any()
            or ((places < 1) | (places > groups) | (target > places) | (target < 1)).any()
            or not 0 <= strength <= 4):
        raise ValueError("Invalid place probabilities, slot expectation or shrinkage")
    if (np.add.reduceat(win, starts) > target+1e-10).any():
        raise ValueError("Win probability floors exceed expected filled places")
    logits = logit(np.clip(base, 1e-6, 1-1e-6))
    means = np.add.reduceat(logits, starts)/groups
    shrink = np.clip(1-strength*np.maximum(0, places/groups-.25), .25, 1.)
    logits = np.repeat(means, groups)+np.repeat(shrink, groups)*(logits-np.repeat(means, groups))
    lo, hi = np.full(len(groups), -100.), np.full(len(groups), 100.)
    for _ in range(60):
        mid = (lo+hi)/2
        projected = np.maximum(expit(logits+np.repeat(mid, groups)), win)
        above = np.add.reduceat(projected, starts) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.maximum(expit(logits+np.repeat((lo+hi)/2, groups)), win)


@dataclass
class RecentPlaceMap:
    slope: float = 1.
    intercept: float = 0.
    strength: float = 2.
    # (discipline, offered places) -> shrinkage-estimated missing slots.
    deficits: dict | None = None

    def expected_slots(self, codes, places):
        return np.asarray([max(1., k-(self.deficits or {}).get((str(c), int(k)), 0.))
                           for c, k in zip(codes, places)])

    def predict(self, raw, win, groups, places, codes):
        base = expit(self.slope*logit(np.clip(np.asarray(raw, float), 1e-9, 1-1e-9))+self.intercept)
        return project_places(base, win, groups, places, strength=self.strength,
                              expected_slots=self.expected_slots(codes, places))

    @classmethod
    def fit(cls, raw, win, placed, finish_positions, groups, places, codes):
        raw, placed, fp = (np.asarray(v, float) for v in (raw, placed, finish_positions))
        groups, starts = _layout(groups, len(raw))
        places = np.asarray(places, int)
        codes = np.asarray(codes, str)
        if (placed.shape != raw.shape or fp.shape != raw.shape or not np.isin(placed, [0, 1]).all()
                or codes.shape != groups.shape or places.shape != groups.shape):
            raise ValueError("Invalid complete place-calibration fields")
        completed = np.add.reduceat((fp > 0).astype(int), starts)
        deficits = np.maximum(places-completed, 0)
        # A zero-deficit prior of100 fields prevents a rare casualty race from
        # asserting that many future place slots will be unfilled.
        estimates = {}
        for key in sorted(set(zip(codes, places))):
            mask = (codes == key[0]) & (places == key[1])
            estimates[(str(key[0]), int(key[1]))] = float(deficits[mask].sum()/(mask.sum()+100.))
        def brier(ab):
            p = expit(ab[0]*logit(np.clip(raw, 1e-9, 1-1e-9))+ab[1])
            return float(np.mean((p-placed)**2)+1e-4*((ab[0]-1)**2+ab[1]**2))
        fitted = minimize(brier, [1., 0.], method="L-BFGS-B", bounds=[(.1, 3.), (-3., 3.)])
        if not fitted.success or not np.isfinite(fitted.fun):
            raise ValueError("Place calibration did not converge")
        mapping = cls(*map(float, fitted.x), deficits=estimates)
        def objective(s):
            mapping.strength = float(s)
            return float(np.mean((mapping.predict(raw, win, groups, places, codes)-placed)**2))
        result = minimize_scalar(objective, bounds=(0., 4.), method="bounded", options={"xatol": 1e-3})
        if not result.success or not np.isfinite(result.fun):
            raise ValueError("Place shrinkage fit did not converge")
        mapping.strength = float(result.x)
        return mapping
