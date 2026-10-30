"""Research-only, prior-day continuous trip suitability by racing discipline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.established_form_features import _horse_keys, mark_scale
from src.forecast_evaluation import valid_race_mask
from src.result_status import NONSTARTER_LABELS, result_labels

FEATURES = ("horse_trip_quality_vs_field", "horse_trip_specialist_edge_vs_field", "horse_trip_support_log")


def trip_suitability_features(frame, *, state=None, return_state=False):
    """Use20 earlier starts/730 days, separate Turf/AW/Hurdle/Chase/Bumper.

    A fixed15% bandwidth smoothly weights comparable trips. Quality is finishing
    percentile (DNF0), with5 pseudo-starts at0.5. Specialist edge centres each
    horse's results on its own recent average before weighting for today's trip.
    No fitted parameters, winner-selected categories or cross-code victories.
    """
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    state = state or {}
    if dates.isna().any() or (len(frame) and state.get("last_date")
            and dates.min() <= pd.Timestamp(state["last_date"])):
        raise ValueError("Trip seed must end on an earlier complete date")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    keys = _horse_keys(frame)
    scales = mark_scale(frame).to_numpy(dtype=object, na_value=None)
    distance = pd.to_numeric(frame.distance_furlongs, errors="coerce").to_numpy(float)
    distance = np.where((distance >= 4) & (distance <= 50), distance, np.nan)
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    size = pd.to_numeric(frame.num_runners, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    quality = np.where(finish > 0, (size-finish)/np.maximum(size-1, 1), 0.)
    quality = np.where(valid_race_mask(frame).to_numpy() & starts, quality, np.nan)
    quality = np.where((quality >= 0) & (quality <= 1), quality, np.nan)
    output = {c: np.full(len(frame), np.nan) for c in FEATURES}
    history = {k: list(v) for k, v in state.get("history", {}).items()}
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order]))+1, len(frame)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        for i in rows:
            if keys[i] is None or scales[i] is None or not np.isfinite(distance[i]):
                continue
            recent = [(d, q) for day, d, q in history.get((keys[i], scales[i]), [])
                      if days[i]-day <= 730 and np.isfinite(d) and np.isfinite(q)]
            output[FEATURES[2]][i] = 0.
            if recent:
                ds, qs = np.asarray(recent).T
                w = np.exp(-.5*(np.log(ds/distance[i])/np.log(1.15))**2)
                support = w.sum()
                output[FEATURES[0]][i] = (np.dot(w, qs)+2.5)/(support+5.)
                output[FEATURES[1]][i] = np.dot(w, qs-qs.mean())/(support+5.)
                output[FEATURES[2]][i] = np.log1p(support)
        for i in rows:
            if starts[i] and keys[i] is not None and scales[i] is not None:
                key = (keys[i], scales[i])
                history[key] = (history.get(key, [])+[(int(days[i]), float(distance[i]), float(quality[i]))])[-20:]
    out = pd.DataFrame(output, index=frame.index)
    for c in FEATURES[:2]:
        out[c] -= out[c].groupby(frame.race_id, sort=False).transform("mean")
    if return_state:
        return out, {"last_date":str(dates.max().date()) if len(frame) else state.get("last_date"), "history":history}
    return out
