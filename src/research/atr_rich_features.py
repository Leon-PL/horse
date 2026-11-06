"""Candidate features from richer ATR observations, outside the normal recipe.

Run metrics describe a completed race. Only ``build_prior_features`` produces
prediction inputs: medians over the horse's last three eligible race appearances,
with at least 48 hours allowed for publication. That delay is an availability
assumption, not proof of when a historical page first appeared.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.runner_identity import horse_ids, identity_values
from src.atr_features import efficiency_error_scores
from src.utils import normalise_off_time_key

RUN_FEATURES = (
    "atr_rich_acceleration_seconds",
    "atr_rich_finish_vs_race_pp",
    "atr_rich_line_speed_change_pct",
    "atr_rich_burst_reserve_pct",
    "atr_rich_late_stride_length_change_pct",
    "atr_rich_late_cadence_change_pct",
    "atr_rich_late_stride_speed_change_pct",
    "atr_rich_efficiency_log_error",
    "atr_rich_efficiency_grade_rank",
    "atr_rich_profile_worst_gap_pp",
    "atr_rich_profile_irregularity_pp",
    "atr_rich_cadence_jitter_pct",
)
MEDIAN_FEATURES = tuple(name + "_median_3" for name in RUN_FEATURES)
CONSISTENCY = {
    "atr_rich_efficiency_mad_5": "atr_rich_efficiency_log_error",
    "atr_rich_finish_vs_race_mad_5": "atr_rich_finish_vs_race_pp",
}
PRIOR_FEATURES = MEDIAN_FEATURES + tuple(CONSISTENCY)
GRADE_ORDER = {grade: rank for rank, grade in enumerate(("A++", "A+", "A", "B", "C", "D", "E", "F"))}
COUNT = "atr_rich_observed_runs_last_3"
AGE = "atr_rich_days_since_observation"
RAW_COLUMNS = (
    "atr_acceleration_0_30_seconds", "atr_finishing_speed_pct", "atr_race_finishing_speed_pct",
    "atr_final_speed_mph", "atr_top_speed_mph", "atr_sectional_speeds", "atr_sectional_speed_labels",
    "atr_stride_length", "atr_stride_frequency", "atr_stride_markers", "atr_stride_frequency_markers",
    "atr_efficiency_score", "atr_efficiency_grade", "atr_energy_dist", "atr_optimum_energy",
)


def _array(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def _positive(value):
    try:
        number = float(value)
        return number if np.isfinite(number) and number > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def _numeric(frame, column):
    value = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
    result = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


def _section_anchors(values, labels):
    values, labels = _array(values), _array(labels)
    if labels:
        if len(values) != len(labels):
            return np.nan, np.nan
        positions = [i for i, name in enumerate(labels)
                     if re.search(r"\bto\b", str(name), re.I) and "mph" not in str(name).casefold()]
    elif len(values) == 3:
        positions = list(range(3))
    else:
        return np.nan, np.nan
    if not positions:
        return np.nan, np.nan
    sections = [_positive(values[i]) for i in positions]
    # A missing final section must never turn the previous section into the
    # closing speed. The peak also requires all named sections to be observed.
    peak = max(sections) if all(np.isfinite(sections)) else np.nan
    return sections[-1], peak


def _stride_points(values, labels):
    values, labels = _array(values), _array(labels)
    if len(values) != len(labels):
        return None
    found = {}
    for label, value in zip(labels, values):
        label = str(label).strip().casefold()
        match = re.fullmatch(r"([123])\s*f", label)
        marker = int(match[1]) if match else 0 if label in {"fin", "fin.", "finish"} else None
        if marker is None:
            continue
        if marker in found:
            return None
        found[marker] = _positive(value)
    if set(found) != {0, 1, 2, 3} or not all(np.isfinite(list(found.values()))):
        return None
    return np.array([found[3], found[2], found[1], found[0]])


def _stride_anchors(values, labels):
    points = _stride_points(values, labels)
    if points is None:
        return None
    middle, late = np.mean(points[:2]), np.mean(points[2:])
    return late / middle


def derive_run_metrics(frame):
    """Post-race metrics, keeping unknown measurements as missing."""
    n = len(frame)
    out = {name: np.full(n, np.nan) for name in RUN_FEATURES}
    acceleration = _numeric(frame, "atr_acceleration_0_30_seconds")
    out[RUN_FEATURES[0]] = np.where(acceleration > 0, acceleration, np.nan)
    runner_fsp, race_fsp = (_numeric(frame, name) for name in
                          ("atr_finishing_speed_pct", "atr_race_finishing_speed_pct"))
    out[RUN_FEATURES[1]] = np.where((runner_fsp > 0) & (race_fsp > 0), runner_fsp-race_fsp, np.nan)
    efficiency = efficiency_error_scores(frame)
    out["atr_rich_efficiency_log_error"] = np.log1p(efficiency)
    if "atr_efficiency_grade" in frame:
        grades = frame.atr_efficiency_grade.astype("string").str.strip().str.upper()
        out["atr_rich_efficiency_grade_rank"] = grades.map(GRADE_ORDER).to_numpy(dtype=float, na_value=np.nan)
    if "atr_energy_dist" in frame and "atr_optimum_energy" in frame:
        for row, (actual, optimum) in enumerate(zip(frame.atr_energy_dist, frame.atr_optimum_energy)):
            actual, optimum = _array(actual), _array(optimum)
            if len(actual) != len(optimum):
                continue
            paired = []
            for a, o in zip(actual, optimum):
                try:
                    a, o = float(a), float(o)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(a) and np.isfinite(o):
                    paired.append(a-o)
            if len(paired) >= 3:
                residual = np.array(paired)
                out["atr_rich_profile_worst_gap_pp"][row] = np.max(np.abs(residual))
                out["atr_rich_profile_irregularity_pp"][row] = np.median(np.abs(residual-np.median(residual)))
    final, top = (_numeric(frame, name) for name in ("atr_final_speed_mph", "atr_top_speed_mph"))
    if "atr_sectional_speeds" in frame:
        labels = frame.get("atr_sectional_speed_labels", pd.Series(None, index=frame.index, dtype=object)).to_numpy()
        values = frame.atr_sectional_speeds.to_numpy()
        for row in np.flatnonzero(frame.atr_sectional_speeds.notna().to_numpy()):
            closing, peak = _section_anchors(values[row], labels[row])
            if np.isfinite(closing) and final[row] > 0:
                out[RUN_FEATURES[2]][row] = 100*(final[row]/closing-1)
            if np.isfinite(peak) and top[row] > 0:
                out[RUN_FEATURES[3]][row] = 100*(top[row]/peak-1)
    stride_columns = ["atr_stride_length", "atr_stride_markers", "atr_stride_frequency", "atr_stride_frequency_markers"]
    if any(column in frame for column in stride_columns):
        arrays = [frame[column].to_numpy() if column in frame else np.full(n, None, dtype=object) for column in stride_columns]
        present = frame.reindex(columns=stride_columns).notna().any(axis=1).to_numpy()
        for row in np.flatnonzero(present):
            length = _stride_anchors(arrays[0][row], arrays[1][row])
            cadence = _stride_anchors(arrays[2][row], arrays[3][row])
            points = _stride_points(arrays[2][row], arrays[3][row])
            if points is not None:
                # Remove a smooth linear change before measuring rhythm jitter;
                # this remains a coarse cadence proxy, not an official fluency grade.
                centered = np.array([-1.5, -.5, .5, 1.5])
                mean = np.mean(points)
                slope = np.dot(centered, points-mean)/5.0
                residual = points-(mean+slope*centered)
                out["atr_rich_cadence_jitter_pct"][row] = 100*np.sqrt(np.mean(residual**2))/mean
            if length is not None:
                out[RUN_FEATURES[4]][row] = 100*(length-1)
            if cadence is not None:
                out[RUN_FEATURES[5]][row] = 100*(cadence-1)
            if length is not None and cadence is not None:
                out[RUN_FEATURES[6]][row] = 100*(length*cadence-1)
    return pd.DataFrame(out, index=frame.index)


def _event_times(frame):
    days = pd.to_datetime(frame.race_date, errors="coerce", utc=True).dt.normalize()
    clock = frame.off_time.map(normalise_off_time_key)
    clock = clock.where(clock.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", na=False))
    return days + pd.to_timedelta(clock+":00", errors="coerce")


def _mad(values):
    return np.nanmedian(np.abs(values-np.nanmedian(values)))


def build_prior_features(history, targets=None, *, availability_lag="48h"):
    """Last-three-race medians using only observations available before the off.

    Supply the full canonical race history, including runs without ATR data:
    those appearances still occupy a window slot. Identity is canonical horse
    ID, never display name. Unknown IDs/times remain missing. Current-race raw
    values, outcomes and future observations cannot enter the returned inputs.
    """
    targets = history if targets is None else targets
    required = {"horse_id", "race_id", "race_date", "off_time"}
    for name, frame in (("history", history), ("targets", targets)):
        if required-set(frame.columns):
            raise ValueError(f"{name} requires canonical horse/race IDs, race_date and UTC off_time")
    lag = pd.Timedelta(availability_lag)
    if pd.isna(lag) or lag < pd.Timedelta(hours=48):
        raise ValueError("Allow at least 48 hours for historical ATR publication")
    out = pd.DataFrame(np.nan, index=range(len(targets)), columns=[*PRIOR_FEATURES, COUNT, AGE])
    out[COUNT] = np.zeros(len(targets), dtype=np.int8)
    if history.empty or targets.empty:
        out.index = targets.index
        return out
    work = derive_run_metrics(history).reset_index(drop=True)
    work["_horse"] = horse_ids(history).to_numpy()
    work["_race"] = identity_values(history.race_id).to_numpy()
    work["_event"] = _event_times(history).to_numpy()
    work = work.loc[work[["_horse", "_race", "_event"]].notna().all(axis=1)]
    if work.duplicated(["_horse", "_race"]).any() or work.duplicated(["_horse", "_event"]).any():
        raise ValueError("Ambiguous duplicate race appearances for a canonical horse ID")
    work = work.sort_values(["_horse", "_event", "_race"]).reset_index(drop=True)
    if work.empty:
        out.index = targets.index
        return out
    medians = work.groupby("_horse", sort=False)[list(RUN_FEATURES)].rolling(3, min_periods=1).median()
    medians = medians.reset_index(level=0, drop=True).sort_index()
    work[list(MEDIAN_FEATURES)] = medians.to_numpy()
    for output, metric in CONSISTENCY.items():
        spread = work.groupby("_horse", sort=False)[metric].rolling(5, min_periods=3).apply(_mad, raw=True)
        work[output] = spread.reset_index(level=0, drop=True).sort_index().to_numpy()
    work["_observed"] = work[list(RUN_FEATURES)].notna().any(axis=1).astype(float)
    counts = work.groupby("_horse", sort=False)._observed.rolling(3, min_periods=1).sum()
    work[COUNT] = counts.reset_index(level=0, drop=True).sort_index().to_numpy()
    work["_last_observation"] = work._event.where(work._observed.eq(1)).groupby(work._horse).ffill()
    work["_available"] = work._event + lag
    left = pd.DataFrame({"_horse": horse_ids(targets).to_numpy(), "_race": identity_values(targets.race_id).to_numpy(),
                         "_off": _event_times(targets).to_numpy(),
                         "_row": np.arange(len(targets))})
    left = left.loc[left[["_horse", "_race", "_off"]].notna().all(axis=1)].sort_values("_off")
    if left.empty:
        out.index = targets.index
        return out
    # As-of on publication time performs the lag: no current-row shift or
    # cross-horse shifting is involved. Exact-time publication is excluded too.
    right = work[["_horse", "_available", "_last_observation", *PRIOR_FEATURES, COUNT]].sort_values("_available")
    joined = pd.merge_asof(left, right, left_on="_off", right_on="_available", by="_horse",
                           direction="backward", allow_exact_matches=False).set_index("_row")
    for column in [*PRIOR_FEATURES, COUNT]:
        out[column] = joined[column].reindex(out.index).to_numpy()
    out[COUNT] = out[COUNT].fillna(0).astype(np.int8)
    age = (joined._off-joined._last_observation).dt.total_seconds()/86400
    out[AGE] = age.reindex(out.index).to_numpy()
    out.index = targets.index
    return out
