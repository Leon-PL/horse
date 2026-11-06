"""Prior-only candidates from ATR jumping and distance observations.

Raw measurements describe their completed race. Prediction inputs come only
from ``build_prior_features``. Native grades/course pars additionally respect
their dated revision; no model fitting or source-cache writes happen here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.runner_identity import horse_ids, identity_values
from src.utils import normalise_off_time_key

JUMP_METRICS = (
    "atr_jump_loss_pct", "atr_jump_early_loss_pct", "atr_jump_late_loss_pct",
    "atr_jump_late_change_pp", "atr_jump_phase_range_pp", "atr_jump_consistency_pct",
    "atr_jump_grade_rank", "atr_jump_vs_course_par_pp",
)
PATH_METRICS = ("atr_path_extra_lengths", "atr_path_extra_pct", "atr_path_above_median")
RUN_METRICS = JUMP_METRICS + PATH_METRICS
MEDIAN_METRICS = RUN_METRICS[:-1]
MEDIAN_FEATURES = tuple(name + "_median_3" for name in MEDIAN_METRICS)
EXTRA_FEATURES = (
    "atr_path_above_median_rate_3", "atr_jump_loss_best_3",
    "atr_jump_grade_ab_rate_3", "atr_jump_loss_mad_5", "atr_path_extra_mad_5",
    "atr_jump_loss_recent_change", "atr_path_extra_recent_change",
)
QUALITY_FEATURES = (
    "atr_jump_observed_runs_3", "atr_path_observed_runs_3",
    "atr_jump_observation_age_days", "atr_path_observation_age_days",
)
PRIOR_FEATURES = MEDIAN_FEATURES + EXTRA_FEATURES
ALL_FEATURES = PRIOR_FEATURES + QUALITY_FEATURES
REVISION_METRICS = ("atr_jump_grade_rank", "atr_jump_vs_course_par_pp")
GRADE_ORDER = {grade: rank for rank, grade in enumerate("ABCDE")}
RAW_COLUMNS = (
    "atr_jump_speed_lost_early_pct", "atr_jump_speed_lost_middle_pct",
    "atr_jump_speed_lost_late_pct", "atr_jump_speed_lost_avg_pct",
    "atr_jump_consistency_pct", "atr_jump_fluency_grade",
    "atr_jump_course_winner_speed_lost_pct", "atr_jumps_par_updated_date",
    "atr_distance_ran_yards", "atr_distance_diff_vs_median_lengths",
    "atr_race_median_distance_ran_yards", "atr_official_position",
)


def _number(frame, name):
    values = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
    result = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
    result[~np.isfinite(result)] = np.nan
    return result


def _percentage(frame, name):
    values = _number(frame, name)
    values[(values < 0) | (values > 100)] = np.nan
    return values


def _event_times(frame):
    days = pd.to_datetime(frame.race_date, errors="coerce", utc=True).dt.normalize()
    clock = frame.off_time.map(normalise_off_time_key)
    clock = clock.where(clock.str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", na=False))
    return days + pd.to_timedelta(clock + ":00", errors="coerce")


def _revision_end(frame):
    """A date-only revision is conservatively available after its UK day ends."""
    values = frame.get("atr_jumps_par_updated_date", pd.Series(pd.NA, index=frame.index, dtype="string"))
    values = values.astype("string")
    values = values.where(values.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False))
    dates = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce") + pd.Timedelta(days=1)
    return dates.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def derive_run_metrics(frame):
    """Eleven post-race descriptors; unknowns and incomplete races stay missing.

    Published speed loss/consistency are percentages, not fractions. Distance
    percentages use rounded source yards; published extra lengths are retained
    independently because they can have finer precision.
    """
    finish = _number(frame, "finish_position" if "finish_position" in frame else "atr_official_position")
    completed = finish > 0
    early, middle, late, average = (
        _percentage(frame, "atr_jump_speed_lost_" + part + "_pct")
        for part in ("early", "middle", "late", "avg")
    )
    consistency = _percentage(frame, "atr_jump_consistency_pct")
    grade = frame.get("atr_jump_fluency_grade", pd.Series(pd.NA, index=frame.index, dtype="string"))
    rank = grade.astype("string").str.strip().str.upper().map(GRADE_ORDER).to_numpy(dtype=float, na_value=np.nan)
    # An ungraded, otherwise unsupported zero is not evidence of perfect jumps.
    phases = np.column_stack([early, middle, late])
    placeholder = ((average == 0) & ~np.isfinite(rank)
                   & (~np.isfinite(phases) | (phases == 0)).all(axis=1)
                   & (~np.isfinite(consistency) | (consistency == 0)))
    for values in (early, middle, late, average, consistency, rank):
        values[~completed | placeholder] = np.nan
    phases = np.column_stack([early, middle, late])
    spread = np.full(len(frame), np.nan)
    full = np.isfinite(phases).all(axis=1)
    spread[full] = np.ptp(phases[full], axis=1)
    par = _percentage(frame, "atr_jump_course_winner_speed_lost_pct")
    par[par <= 0] = np.nan
    distance, median, extra = (_number(frame, name) for name in
        ("atr_distance_ran_yards", "atr_race_median_distance_ran_yards", "atr_distance_diff_vs_median_lengths"))
    path_ok = completed & (distance > 0) & (median > 0)
    percent = np.full(len(frame), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        percent[path_ok] = 100 * (distance[path_ok] / median[path_ok] - 1)
    # A signed source difference with no completed distance is not a path-cost
    # observation. In particular, pulled-up horses did not take a shortcut.
    extra[~path_ok] = np.nan
    above = np.where(np.isfinite(extra), (extra > 0).astype(float), np.nan)
    return pd.DataFrame(dict(zip(RUN_METRICS, (
        average, early, late, late - early, spread, consistency, rank, average - par,
        extra, percent, above,
    ))), index=frame.index).replace([np.inf,-np.inf],np.nan)


def _stat(values, operation="median"):
    """Row statistic without warnings or invented zeros for empty histories."""
    present = np.isfinite(values).any(axis=1)
    safe = values.copy()
    safe[~present, 0] = 0
    result = {"median": np.nanmedian, "min": np.nanmin, "mean": np.nanmean}[operation](safe, axis=1)
    result[~present] = np.nan
    return result


def build_prior_features(history, targets=None, *, availability_lag="48h", batch_size=50000):
    """Query the last 3/5 eligible canonical appearances, not just tracked runs.

    The source race + at least 48 hours must precede the target off. For each
    grade/par cell its revision day must also end before that off. Per-cell
    masking handles later revisions without an old revised run replacing more
    recent appearances. The lag for physical measurements is an assumption;
    it does not prove historical first-publication/revision timestamps.
    """
    targets = history if targets is None else targets
    required = {"horse_id", "race_id", "race_date", "off_time"}
    if required - set(history) or required - set(targets):
        raise ValueError("Canonical horse/race IDs, race_date and UTC off_time are required")
    lag = pd.Timedelta(availability_lag)
    if pd.isna(lag) or lag < pd.Timedelta(hours=48):
        raise ValueError("Allow at least 48 hours for ATR publication")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    out = pd.DataFrame(np.nan, index=pd.RangeIndex(len(targets)), columns=ALL_FEATURES)
    out[list(QUALITY_FEATURES[:2])] = 0.
    if history.empty or targets.empty:
        out.index = targets.index
        return out
    work = derive_run_metrics(history).reset_index(drop=True)
    work["_horse"] = horse_ids(history).to_numpy()
    work["_race"] = identity_values(history.race_id).to_numpy()
    work["_event"] = _event_times(history).to_numpy()
    work["_revision"] = _revision_end(history).to_numpy()
    work = work.loc[work[["_horse", "_race", "_event"]].notna().all(axis=1)]
    if work.duplicated(["_horse", "_race"]).any() or work.duplicated(["_horse", "_event"]).any():
        raise ValueError("Ambiguous canonical race appearances")
    work = work.sort_values(["_horse", "_event", "_race"]).reset_index(drop=True)
    if work.empty:
        out.index = targets.index
        return out
    work["_available"] = work._event + lag
    work["_position"] = np.arange(len(work))
    left = pd.DataFrame({"_horse":horse_ids(targets).to_numpy(),
                         "_race":identity_values(targets.race_id).to_numpy(),
                         "_off":_event_times(targets).to_numpy(), "_row":np.arange(len(targets))})
    left = left.loc[left[["_horse", "_race", "_off"]].notna().all(axis=1)]
    if left.empty:
        out.index = targets.index
        return out
    joined = pd.merge_asof(left.sort_values("_off"),
        work[["_horse", "_available", "_position"]].sort_values("_available"),
        left_on="_off", right_on="_available", by="_horse", direction="backward", allow_exact_matches=False)
    values = work[list(RUN_METRICS)].to_numpy(dtype=float)
    horses, races = work._horse.to_numpy(), work._race.to_numpy()
    events = work._event.astype("int64").to_numpy()
    revisions = work._revision.astype("int64").to_numpy()
    revisions = np.where(revisions == np.iinfo(np.int64).min, np.iinfo(np.int64).max, revisions)
    columns = {name: i for i, name in enumerate(RUN_METRICS)}
    for start in range(0, len(joined), batch_size):
        chunk = joined.iloc[start:start + batch_size]
        positions = chunk._position.fillna(-1).to_numpy(dtype=np.int64)[:, None] - np.arange(5)
        safe = np.maximum(positions, 0)
        valid = ((positions >= 0) & (horses[safe] == chunk._horse.to_numpy()[:, None])
                 & (races[safe] != chunk._race.to_numpy()[:, None]))
        past = values[safe].copy()
        past[~valid] = np.nan
        off = chunk._off.astype("int64").to_numpy()
        revised_later = revisions[safe] >= off[:, None]
        for name in REVISION_METRICS:
            column = columns[name]
            cell = past[:, :, column]
            cell[revised_later] = np.nan
        result = {}
        for name in MEDIAN_METRICS:
            result[name + "_median_3"] = _stat(past[:, :3, columns[name]])
        result["atr_path_above_median_rate_3"] = _stat(past[:, :3, columns[PATH_METRICS[-1]]], "mean")
        loss = past[:, :, columns[JUMP_METRICS[0]]]
        path = past[:, :, columns[PATH_METRICS[0]]]
        result["atr_jump_loss_best_3"] = _stat(loss[:, :3], "min")
        grades = past[:, :3, columns["atr_jump_grade_rank"]]
        result["atr_jump_grade_ab_rate_3"] = _stat(np.where(np.isfinite(grades), grades <= 1, np.nan), "mean")
        for data, mad_name, change_name in (
            (loss, "atr_jump_loss_mad_5", "atr_jump_loss_recent_change"),
            (path, "atr_path_extra_mad_5", "atr_path_extra_recent_change"),
        ):
            mad = _stat(np.abs(data - _stat(data)[:, None]))
            mad[np.isfinite(data).sum(axis=1) < 3] = np.nan
            result[mad_name] = mad
            change = data[:, 0] - _stat(data[:, 1:])
            change[np.isfinite(data[:, 1:]).sum(axis=1) < 2] = np.nan
            result[change_name] = change
        for lo, hi, count_name, age_name in (
            (0, len(JUMP_METRICS), QUALITY_FEATURES[0], QUALITY_FEATURES[2]),
            (len(JUMP_METRICS), len(RUN_METRICS), QUALITY_FEATURES[1], QUALITY_FEATURES[3]),
        ):
            observed = np.isfinite(past[:, :3, lo:hi]).any(axis=2)
            result[count_name] = observed.sum(axis=1)
            latest = np.where(observed, events[safe[:, :3]], np.iinfo(np.int64).min).max(axis=1)
            age = np.full(len(chunk), np.nan)
            has = observed.any(axis=1)
            age[has] = (off[has] - latest[has]) / (86400 * 1e9)
            result[age_name] = age
        for name, data in result.items():
            out.iloc[chunk._row.to_numpy(), out.columns.get_loc(name)] = data
    out.index = targets.index
    return out.replace([np.inf,-np.inf],np.nan)
