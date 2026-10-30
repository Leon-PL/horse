"""Prior-day market-adjusted connection form; unselected research inputs.

A descriptive observed-minus-expected win rate, not an ability or profit claim.
Only complete, settled, priced fields update histories. Unknown names or books
never become pooled connections or invented neutral observations.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.forecast_evaluation import valid_race_mask
from src.result_status import NONSTARTER_LABELS, result_labels

WINDOW_DAYS = 60
PRIOR_STARTS = 20
FEATURES = (
    "trainer_market_excess_60d_vs_field", "jockey_market_excess_60d_vs_field",
    "trainer_market_support_60d_log", "jockey_market_support_60d_log",
)


def complete_market_residuals(frame):
    """Normalise a whole earlier SP book; one missing price invalidates its book."""
    price = pd.to_numeric(frame.odds, errors="coerce")
    known = valid_race_mask(frame) & price.ge(1.01) & np.isfinite(price)
    complete = known.groupby(frame.race_id, sort=False).transform("all")
    inverse = (1 / price).where(complete)
    probability = inverse / inverse.groupby(frame.race_id, sort=False).transform("sum")
    won = pd.to_numeric(frame.won, errors="coerce")
    return pd.DataFrame({"probability": probability, "residual": won - probability}, index=frame.index)


def _connection_keys(values):
    keys = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()
    keys = keys.mask(keys.isin({"", "unknown", "none", "nan", "<na>", "n/a"}))
    return keys.to_numpy(dtype=object, na_value=None)


def connection_market_features(frame, *, state=None, return_state=False):
    """Emit start-of-day 60-day excess wins/(priced starts+20), plus support.

    All runners on a date share an unchanged prior-day view. Pending cards do
    not advance history. A seeded append must begin after its last settled day.
    Input state is never mutated; dated aggregates keep the state bounded.
    """
    state = state or {}
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Connection form requires unambiguous local race dates")
    if len(frame) and state.get("last_date") and dates.min() <= pd.Timestamp(state["last_date"]):
        raise ValueError("Connection seed must end before every input day")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    names = {kind: _connection_keys(frame[kind]) for kind in ("trainer", "jockey")}
    observations = complete_market_residuals(frame)
    probabilities = observations.probability.to_numpy(float)
    wins = pd.to_numeric(frame.won, errors="coerce").to_numpy(float)
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    started = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    history = {key: list(events) for key, events in state.get("history", {}).items()}
    output = {column: np.full(len(frame), np.nan) for column in FEATURES}
    runner = frame.horse_name.fillna("").astype(str).to_numpy()
    order = np.lexsort((runner, frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order])) + 1, len(frame)]
    last_day = state.get("last_date")
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if lo == hi:
            continue
        rows = order[lo:hi]
        day = int(days[rows[0]])
        cutoff = day - WINDOW_DAYS
        for kind, excess_column, support_column in (
            ("trainer", FEATURES[0], FEATURES[2]), ("jockey", FEATURES[1], FEATURES[3]),
        ):
            values = {}
            for i in rows:
                key = names[kind][i]
                output[support_column][i] = 0.
                if key is None:
                    continue
                if key not in values:
                    prior = [event for event in history.get((kind, key), []) if event[0] >= cutoff]
                    count = sum(event[3] for event in prior)
                    excess = ((math.fsum(event[1] for event in prior) - math.fsum(event[2] for event in prior))
                              / (count + PRIOR_STARTS)) if count else np.nan
                    values[key] = (excess, math.log1p(count))
                output[excess_column][i], output[support_column][i] = values[key]
        # Only after every forecast on this date has been emitted may results
        # update the next day's state. Multiple same-connection rides count once
        # each, not once per copy of the connection's name in the current field.
        if started[rows].any():
            history = {key: [event for event in events if event[0] >= cutoff]
                       for key, events in history.items()}
            history = {key: events for key, events in history.items() if events}
            updates = {}
            for i in rows:
                if not np.isfinite(probabilities[i]):
                    continue
                for kind in names:
                    key = names[kind][i]
                    if key is not None:
                        updates.setdefault((kind, key), []).append((wins[i], probabilities[i]))
            for key, records in updates.items():
                history.setdefault(key, []).append((day, math.fsum(v[0] for v in records),
                                                    math.fsum(v[1] for v in records), len(records)))
            last_day = str(dates.iloc[rows[0]].date())
    result = pd.DataFrame(output, index=frame.index)
    for column in FEATURES[:2]:
        result[column] -= result[column].groupby(frame.race_id, sort=False).transform("mean")
    updated = {"last_date": last_day, "history": history}
    return (result, updated) if return_state else result
