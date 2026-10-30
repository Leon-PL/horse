"""A horse's observed local racing cadence; unselected research descriptors.

Elapsed gaps are not a diagnosis of fitness or a claim of complete overseas
history. Compare today's gap with its own last five observed intervals, rather
than assuming the same break means the same thing for every horse.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.established_form_features import _horse_keys
from src.result_status import NONSTARTER_LABELS, result_labels

HISTORY_DAYS = 730
MAX_INTERVALS = 5
MIN_INTERVALS = 3
FEATURES = ("horse_gap_vs_own_median_log", "horse_own_median_gap_log", "horse_gap_interval_support_5")


def recovery_pattern_features(frame, *, state=None, return_state=False):
    """Start-of-day, last-five local intervals ending within the preceding730d.

    Missing routine remains unknown until three positive prior intervals exist.
    Genuine nonfinishers are starts; pending cards/nonstarters never advance it.
    A seed must precede every input date and remains unmodified.
    """
    state = state or {}
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Racing cadence requires unambiguous local dates")
    if len(frame) and state.get("last_date") and dates.min() <= pd.Timestamp(state["last_date"]):
        raise ValueError("Cadence seed must precede every input day")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    keys = _horse_keys(frame)
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    nonstarter = labels.isin(NONSTARTER_LABELS).to_numpy()
    started = ((finish > 0) | labels.ne("").to_numpy()) & ~nonstarter
    history = {key: {"last_start": value["last_start"], "intervals": list(value["intervals"])}
               for key, value in state.get("history", {}).items()}
    output = {column: np.full(len(frame), np.nan) for column in FEATURES}
    order = np.lexsort((frame.horse_name.fillna("").astype(str).to_numpy(),
                        frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order])) + 1, len(frame)]
    last_day = state.get("last_date")
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        if not len(rows):
            continue
        day = int(days[rows[0]])
        for i in rows:
            key = keys[i]
            prior = history.get(key)
            output[FEATURES[2]][i] = 0.
            if prior is None:
                continue
            intervals = [gap for end, gap in prior["intervals"] if day - end <= HISTORY_DAYS]
            output[FEATURES[2]][i] = len(intervals)
            if len(intervals) >= MIN_INTERVALS:
                median = float(np.median([math.log1p(gap) for gap in intervals]))
                gap = day - prior["last_start"]
                if gap <= 0:
                    raise ValueError("Cadence history must contain strictly earlier starts")
                output[FEATURES[1]][i] = median
                output[FEATURES[0]][i] = math.log1p(gap) - median
        settled_keys = [keys[i] for i in rows if started[i] and keys[i] is not None]
        if len(settled_keys) != len(set(settled_keys)):
            raise ValueError("Multiple actual starts for one horse/day need explicit ordering")
        for i in rows:
            if not started[i] or keys[i] is None:
                continue
            key = keys[i]
            prior = history.get(key)
            intervals = [] if prior is None else [event for event in prior["intervals"] if day - event[0] <= HISTORY_DAYS]
            if prior is not None:
                gap = day - prior["last_start"]
                if gap <= 0:
                    raise ValueError("A stored interval must have positive elapsed days")
                intervals.append((day, gap))
            history[key] = {"last_start": day, "intervals": intervals[-MAX_INTERVALS:]}
            last_day = str(dates.iloc[i].date())
    result = pd.DataFrame(output, index=frame.index)
    updated = {"last_date": last_day, "history": history}
    return (result, updated) if return_state else result
