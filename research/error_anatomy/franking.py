"""Has the form worked out? Later runs of a horse's past rivals, against the model.

A rating system rates a race by its runners as they were on the day. A
handicapper also looks at what those runners did next: if the horses that
finished around ours have since run above expectations, the race was stronger
than it looked. Nothing in the store carries this.

For a runner today and each of its last three races R, take every other runner
in R and its runs strictly after R and strictly before today. Each such run has
residuals against the model's out-of-fold forecast (``entity_residuals``):
rivals beaten, win and the market gap. The inputs are shrunk means of those
residuals over the last race and over the last three, the number of rival runs
behind them, and the means relative to today's field.

Rivals are split by where they finished relative to our horse in R, because
the two halves say different things: horses that beat us franking the form
says the race was strong; horses we beat franking it says our beating them
was worth more.
"""
import numpy as np
import pandas as pd
from numba import njit

from research.error_anatomy.entity_residuals import KEYS, run_residuals

from src.correction.inputs import FRANK_LOOKBACK as LOOKBACK  # noqa: F401  (one owner: src)
from src.correction.inputs import FRANK_SHRINK as SHRINK  # noqa: F401
from src.correction.inputs import _frank, franking_inputs  # noqa: F401


def load(frame):
    """Candidate loader: (values, columns) for the frame's races, from the long out-of-fold history."""
    from research.error_anatomy import entity_residuals as er
    from research.meta_learner import screen_candidates as sc
    history = sc.read_store(["finish_position", "won"])
    forecasts = er.long_frame()
    values = franking_inputs(history, forecasts, market="p_market")
    values = values.loc[values.race_id.isin(frame.race_id.astype(str).unique())]
    return values, [c for c in values if c not in KEYS]
