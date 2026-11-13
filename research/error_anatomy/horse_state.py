"""Run-by-run state of each horse's errors, and short-memory stable form.

The third pass found that the corrected forecast's disagreement with the
market still repeats for the same horse (correlation 0.18 with its previous
run, 0.16 two runs back on the uncorrected forecast). The own-horse inputs of
``entity_residuals`` are one decayed mean per signal with a 365-day half-life,
which cannot tell a horse whose last run was the surprise from one whose
surprise is a year old.

Two groups, declared before either was scored:

- ``horse-state``: the horse's last five race-centred market gaps and last
  three rivals-beaten and win residuals, days back to each of the last three
  runs, run-weighted moving averages of each signal (weights 0.5 and 0.25 per
  run), and the horse's mean gap and rivals-beaten residual under today's
  code, surface and trip band.
- ``entity-recent``: the residual and market-gap means of ``entity_residuals``
  for trainer, jockey, trainer and jockey together, and sire, with half-lives
  of 30 and 90 days instead of 365.

Everything uses runs on strictly earlier dates, against the seed-42
out-of-fold forecast, with the earlier race's closing price.
"""
import numba
import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.meta_learner import screening as s

KEYS = s.KEYS
from src.correction.inputs import (ALPHAS, CONDITIONS, HALF_LIVES, LAGS, RECENT,  # noqa: F401  (one owner: src)
                                   entity_recent, horse_state)
from src.correction.inputs import moving_mean as _moving  # noqa: F401
from src.correction.inputs import run_signals as signals  # noqa: F401


# Ablations of ``horse-state``, to see which part carries it.
SUBSETS = {
    "horse-state-gap": ("hs_gap_",),                 # past closing prices against the forecast
    "horse-state-result": ("hs_win_", "hs_order_"),  # past results against the forecast
    "horse-state-lags": ("hs_gap_l", "hs_win_l", "hs_order_l", "hs_days_l"),  # no averages, no conditions
}


def load(name, frame):
    if name in SUBSETS:
        values = horse_state(frame)
        columns = [c for c in values if c.startswith(SUBSETS[name])]
        return values[[*KEYS, *columns]], columns
    values = {"horse-state": horse_state, "entity-recent": entity_recent}[name](frame)
    columns = [c for c in values if c not in KEYS]
    return values, columns
