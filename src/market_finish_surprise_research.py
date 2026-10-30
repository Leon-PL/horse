"""Earlier finishing performance relative to price-implied pair comparisons.

Research descriptor, not an official performance figure or a calibrated placing
forecast. Win-price strengths use the Bradley--Terry comparison formula. This
is an assumption to test, not a claim that win odds determine finishing order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.established_form_features import _horse_keys, mark_scale
from src.prior_market_research import past_market_observations
from src.result_status import NONSTARTER_LABELS, result_labels

FEATURES = (
    "horse_market_finish_surprise_3_vs_field",
    "horse_market_last_finish_surprise_vs_field",
    "horse_market_finish_surprise_support_3",
)


def finish_surprise_observations(frame):
    """Compare only actual finishers; a nonfinish is unknown ability evidence.

    A full valid original price book is required, including nonfinishers'
    prices. For each finisher, actual fraction of other finishers beaten
    (half for a tie) minus mean p_i/(p_i+p_j) gives a bounded descriptor.
    Comparisons involving nonfinishers are omitted, so merely surviving a
    rival's fall does not earn an unexpected-performance bonus. This is a
    post-result, conditional descriptor; it is not an unbiased market error.
    """
    work = frame if "official_rating" in frame else frame.assign(official_rating=np.nan)
    logrel = past_market_observations(work).logrel.to_numpy()
    strength = np.exp(logrel)
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    expected = np.full(len(frame), np.nan)
    actual = np.full(len(frame), np.nan)
    for rows in frame.groupby("race_id", sort=False).indices.values():
        if not np.isfinite(strength[rows]).all():
            continue
        completed = rows[np.isfinite(finish[rows]) & (finish[rows] > 0)]
        if len(completed) < 2:
            continue
        values = strength[completed]
        positions = finish[completed]
        probabilities = values[:, None] / (values[:, None] + values[None, :])
        observed = ((positions[:, None] < positions[None, :]).astype(float)
                    + .5 * (positions[:, None] == positions[None, :]))
        expected[completed] = (probabilities.sum(axis=1) - .5) / (len(completed) - 1)
        actual[completed] = (observed.sum(axis=1) - .5) / (len(completed) - 1)
    return pd.DataFrame({"expected_fraction_beaten": expected, "observed_fraction_beaten": actual,
                         "finish_surprise": actual - expected}, index=frame.index)


def market_finish_surprise_features(frame, *, state=None, return_state=False):
    """Last three actual same-scale starts within365 days, prior-day only."""
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    state = state or {}
    if dates.isna().any() or (len(frame) and state.get("last_date")
                            and dates.min() <= pd.Timestamp(state["last_date"])):
        raise ValueError("Finish-surprise seed must end on an earlier complete date")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    identities = _horse_keys(frame)
    scales = mark_scale(frame).to_numpy(dtype=object, na_value=None)
    observations = finish_surprise_observations(frame).finish_surprise.to_numpy()
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    known = np.array([key is not None and scale is not None
                      for key, scale in zip(identities, scales)], dtype=bool)
    actual = pd.DataFrame({"identity": identities, "day": days})
    if actual.loc[starts & known].duplicated(["identity", "day"]).any():
        raise ValueError("Multiple actual starts for one identity/day make latest performance ambiguous")
    output = {column: np.full(len(frame), np.nan) for column in FEATURES}
    history = {key: list(values) for key, values in state.get("history", {}).items()}
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order])) + 1, len(frame)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        for i in rows:
            if not known[i]:
                continue
            recent = [entry for entry in history.get((identities[i], scales[i]), [])
                      if days[i] - entry[0] <= 365]
            measured = [value for _, value in recent if np.isfinite(value)]
            output[FEATURES[2]][i] = len(measured)
            if measured:
                output[FEATURES[0]][i] = sum(measured) / len(measured)
            if recent and np.isfinite(recent[-1][1]):
                output[FEATURES[1]][i] = recent[-1][1]
        for i in rows:
            if starts[i] and known[i]:
                key = (identities[i], scales[i])
                history[key] = (history.get(key, []) + [(int(days[i]), float(observations[i]))])[-3:]
    result = pd.DataFrame(output, index=frame.index)
    for column in FEATURES[:2]:
        result[column] -= result[column].groupby(frame.race_id, sort=False).transform("mean")
    if return_state:
        return result, {"last_date": str(dates.max().date()) if len(frame) else state.get("last_date"),
                        "history": history}
    return result
