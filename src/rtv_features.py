"""Quality-filtered prior RTV performance and timestamped pre-race context.

Raw current-run telemetry is never returned. Prior-run observations mature
after two calendar days; historical backfill remains retrospective development
data because original publication/revision times were not captured. Pre-race
context requires an actual snapshot before the forecast decision cutoff.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import numba as nb

import config
from src.utils import canonical_track_key, normalise_horse_key, normalise_off_time_key

RUN_METRICS = ("fsp", "top_speed", "acceleration", "avg_length", "avg_frequency", "stride_range",
    "jump_index", "lengths_gained_jumping", "speed_lost", "recovery_time", "positive_jump_percentage",
    "fsp_relative", "top_speed_relative", "late_speed_ratio", "early_position", "late_gain", "pace_variability")
LAST_METRICS = ("fsp_relative", "top_speed_relative", "jump_index", "late_speed_ratio", "late_gain")
HISTORY_FEATURES = tuple("rtv_"+m+"_mean3" for m in RUN_METRICS) + tuple("rtv_"+m+"_last" for m in LAST_METRICS) + (
    "rtv_history_count3", "rtv_jumps_count3", "rtv_sectional_count3", "rtv_days_since_observation")
PRE_FEATURES = ("rtv_pre_timeform_rating", "rtv_pre_timeform_stars", "rtv_pre_pace_position", "rtv_pre_draw_advantage")
FEATURES = HISTORY_FEATURES + PRE_FEATURES
BOUNDS = {"fsp": (60., 140.), "top_speed": (15., 50.), "acceleration": (1., 8.),
    "avg_length": (4., 10.), "avg_frequency": (1.5, 3.5), "min_length": (3., 10.), "max_length": (4., 11.),
    "jump_index": (0., 10.), "lengths_gained_jumping": (-40., 40.), "speed_lost": (-25., 5.),
    "recovery_time": (0., 10.), "positive_jump_percentage": (0., 1.)}
SECTIONAL_BAD = {"sector_count_vs_distance", "sector_glitch", "cumulative_mismatch",
    "sector_position_mismatch", "order_vs_time", "leader_gap_mismatch"}


def runs_path():
    return Path(getattr(config, "RTV_FEATURE_RUNS", Path(config.DATA_DIR)/"rtv_cache/feature_runs.parquet"))


def pre_path():
    return Path(getattr(config, "RTV_FEATURE_PRE", Path(config.DATA_DIR)/"rtv_cache/feature_pre.parquet"))


def _number(frame, name):
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce").astype("float64")


def build_run_observations(races, runners, metrics, sectionals, flags, links):
    """Build one numeric observation per exact SL/RTV starter identity.

    Uses measured values, never provider ranks/percentages/averages or prices.
    Bad sectional races/runners are masked before summaries are calculated.
    FSP and completed-race sectional summaries require a physical finisher.
    """
    columns = ["race_id", "horse_id", "race_date", "available_date", *RUN_METRICS]
    if links.empty:
        return pd.DataFrame(columns=columns)
    base = links.merge(runners[["race_id", "runner_id", "is_finisher"]],
        left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"], validate="one_to_one")
    if base.duplicated(["sl_race_id", "sl_horse_id"]).any():
        raise ValueError("Duplicate RTV observation identity")
    if not metrics.empty:
        values = metrics.loc[metrics.metric.isin(BOUNDS), ["race_id", "runner_id", "metric", "value"]].copy()
        if values.duplicated(["race_id", "runner_id", "metric"]).any():
            raise ValueError("Duplicate RTV metric")
        values["value"] = pd.to_numeric(values.value, errors="coerce")
        for name, (lo, hi) in BOUNDS.items():
            bad = values.metric.eq(name) & ~values.value.between(lo, hi)
            values.loc[bad, "value"] = np.nan
        wide = values.pivot(index=["race_id", "runner_id"], columns="metric", values="value").reset_index()
        base = base.merge(wide, on=["race_id", "runner_id"], how="left", validate="one_to_one")
    base["fsp"] = _number(base, "fsp").where(base.is_finisher.fillna(False))
    base["stride_range"] = (_number(base, "max_length")-_number(base, "min_length")).where(
        _number(base, "max_length").ge(_number(base, "min_length")))
    for metric, floor in (("fsp", 1.), ("top_speed", .5)):
        value = _number(base, metric)
        group = value.groupby(base.rtv_race_id)
        base[metric+"_relative"] = ((value-group.transform("mean"))/group.transform("std").clip(lower=floor)).clip(-5, 5)
    if not sectionals.empty:
        sector = sectionals.copy()
        bad_flags = flags.loc[flags["check"].isin(SECTIONAL_BAD)] if not flags.empty else flags
        if not bad_flags.empty:
            bad_races = bad_flags.loc[bad_flags.runner_id.isna(), "race_id"]
            bad_keys = pd.MultiIndex.from_frame(bad_flags.dropna(subset=["runner_id"])[["race_id", "runner_id"]])
            sector = sector.loc[~sector.race_id.isin(bad_races)
                & ~pd.MultiIndex.from_frame(sector[["race_id", "runner_id"]]).isin(bad_keys)]
        sector["_time"] = _number(sector, "sector_time")
        sector["_yards"] = _number(sector, "sector_length_yards")
        sector["_speed"] = sector._yards/sector._time
        sector = sector.loc[sector._time.gt(0) & sector._yards.gt(0) & (sector._speed*3600/1760).between(10, 50)]
        sector = sector.sort_values(["race_id", "runner_id", "sector_number"], kind="stable")
        if sector.duplicated(["race_id", "runner_id", "sector_number"]).any():
            raise ValueError("Duplicate RTV sector")
        keys = ["race_id", "runner_id"]
        group = sector.groupby(keys, sort=False)
        sector["_n"] = group.sector_number.transform("size")
        sector["_last"] = group.cumcount(ascending=False).lt(2)
        sector["_tail_time"] = sector._time.where(sector._last, 0)
        sector["_tail_yards"] = sector._yards.where(sector._last, 0)
        summary = sector.groupby(keys).agg(n=("sector_number", "size"), yards=("_yards", "sum"),
            seconds=("_time", "sum"), tail_yards=("_tail_yards", "sum"), tail_seconds=("_tail_time", "sum"),
            speed_std=("_speed", "std"), speed_mean=("_speed", "mean"))
        summary["late_speed_ratio"] = (summary.tail_yards/summary.tail_seconds)/(summary.yards/summary.seconds)
        summary["pace_variability"] = summary.speed_std/summary.speed_mean
        first = sector.groupby(keys).head(1).set_index(keys)
        before_tail = sector.loc[sector.groupby(keys).cumcount(ascending=False).eq(2)].set_index(keys)
        last = sector.groupby(keys).tail(1).set_index(keys)
        field_size = runners.loc[~runners.is_nonrunner].groupby("race_id").size()
        denominator = summary.index.get_level_values("race_id").map(field_size).to_numpy()-1
        summary["early_position"] = (_number(first, "race_position")-1).reindex(summary.index).to_numpy()/np.maximum(denominator, 1)
        summary["late_gain"] = (_number(before_tail, "race_position")-_number(last, "race_position")).reindex(summary.index).to_numpy()/np.maximum(denominator, 1)
        for name in ("late_speed_ratio", "early_position", "late_gain", "pace_variability"):
            complete = summary.n.ge(4)
            if "distance_furlongs" in races:
                distance = summary.index.get_level_values("race_id").map(races.set_index("race_id").distance_furlongs).to_numpy()
                complete &= summary.n.ge(np.ceil(distance)) & (summary.yards-distance*220).abs().le(20.)
            summary[name] = summary[name].where(complete)
        base = base.merge(summary[["late_speed_ratio", "early_position", "late_gain", "pace_variability"]],
            on=keys, how="left", validate="one_to_one")
        for name in ("late_speed_ratio", "early_position", "late_gain", "pace_variability"):
            base[name] = base[name].where(base.is_finisher.fillna(False))
    output = pd.DataFrame({"race_id": base.sl_race_id.astype(str), "horse_id": pd.to_numeric(base.sl_horse_id),
        "race_date": pd.to_datetime(base.race_date).dt.normalize()})
    output["available_date"] = output.race_date+pd.Timedelta(days=2)
    for name in RUN_METRICS:
        output[name] = _number(base, name).to_numpy()
    return output[columns]


@nb.njit(cache=False)
def _rolling_three(values, horse_ids):
    means = np.full(values.shape, np.nan)
    counts = np.zeros(values.shape)
    start = 0
    for i in range(len(values)):
        if i == 0 or horse_ids[i] != horse_ids[i-1]:
            start = i
        lo = max(start, i-2)
        for column in range(values.shape[1]):
            total, count = 0., 0
            for row in range(lo, i+1):
                value = values[row, column]
                if not np.isnan(value):
                    total += value
                    count += 1
            counts[i, column] = count
            if count:
                means[i, column] = total/count
    return means, counts


def _history_observations(runs):
    if runs.empty:
        return pd.DataFrame()
    runs = runs.copy()
    for name in RUN_METRICS:
        runs[name] = _number(runs, name)
    runs["_horse"] = pd.to_numeric(runs.horse_id, errors="coerce")
    runs = runs.loc[runs._horse.notna() & runs._horse.gt(0)]
    runs = runs.loc[~runs.duplicated(["_horse", "race_date"], keep=False)]
    runs = runs.sort_values(["_horse", "race_date", "race_id"], kind="stable")
    # No telemetry means no observation, rather than a fictitious zero run.
    runs = runs.loc[runs[list(RUN_METRICS)].notna().any(axis=1)].reset_index(drop=True)
    group = runs.groupby("_horse", sort=False)
    result = runs[["_horse", "race_date", "available_date"]].copy()
    rolled, support = _rolling_three(runs[list(RUN_METRICS)].to_numpy(dtype="float64"), runs._horse.to_numpy(dtype="int64"))
    for column, name in enumerate(RUN_METRICS):
        result["rtv_"+name+"_mean3"] = rolled[:, column]
    for name in LAST_METRICS:
        result["rtv_"+name+"_last"] = runs[name]
    result["rtv_history_count3"] = (group.cumcount()+1).clip(upper=3)
    for short, name in (("jumps", "jump_index"), ("sectional", "late_speed_ratio")):
        result["rtv_"+short+"_count3"] = support[:, RUN_METRICS.index(name)]
    result["_horse"] = result._horse.astype("int64")
    result["available_date"] = pd.to_datetime(result.available_date)
    result["race_date"] = pd.to_datetime(result.race_date)
    return result


@lru_cache(maxsize=2)
def _cached_history(path, modified, size):
    return _history_observations(pd.read_parquet(path))


def add_history_features(frame, *, runs=None):
    output = pd.DataFrame(np.nan, index=pd.RangeIndex(len(frame)), columns=HISTORY_FEATURES)
    counts = ["rtv_history_count3", "rtv_jumps_count3", "rtv_sectional_count3"]
    output[counts] = 0.
    path = runs_path()
    if runs is not None:
        history = _history_observations(runs)
    elif path.exists():
        stat = path.stat()
        history = _cached_history(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    else:
        history = pd.DataFrame()
    if not history.empty and "horse_id" in frame:
        target = pd.DataFrame({"_row": np.arange(len(frame)), "_horse": pd.to_numeric(frame.horse_id, errors="coerce").to_numpy(),
            "_date": pd.to_datetime(frame.race_date, errors="coerce").to_numpy()})
        target = target.loc[target._horse.notna() & target._date.notna() & target._horse.gt(0)].copy()
        target["_horse"] = target._horse.astype("int64")
        joined = pd.merge_asof(target.sort_values("_date"), history.sort_values("available_date"),
            left_on="_date", right_on="available_date", by="_horse", direction="backward", tolerance=pd.Timedelta(days=730))
        joined["rtv_days_since_observation"] = (joined._date-joined.race_date).dt.days
        output.loc[joined._row, list(HISTORY_FEATURES)] = joined[list(HISTORY_FEATURES)].to_numpy(dtype="float64", na_value=np.nan)
        output[counts] = output[counts].fillna(0.)
    output.index = frame.index
    return pd.concat([frame.drop(columns=list(FEATURES), errors="ignore"), output], axis=1)


def add_pre_race_features(frame, *, observations=None, as_of=None):
    output = pd.DataFrame(np.nan, index=pd.RangeIndex(len(frame)), columns=PRE_FEATURES)
    path = pre_path()
    observations = (pd.read_parquet(path) if path.exists() else pd.DataFrame()) if observations is None else observations.copy()
    required = {"track", "off_time", "horse_name", "race_date"}
    if not observations.empty and required.issubset(frame):
        target = pd.DataFrame({"_row": np.arange(len(frame))})
        target["race_date"] = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d").to_numpy()
        target["track_key"] = frame.track.map(canonical_track_key).to_numpy()
        target["off_key"] = frame.off_time.map(normalise_off_time_key).to_numpy()
        target["horse_key"] = frame.horse_name.map(normalise_horse_key).to_numpy()
        cutoff = pd.to_datetime(target.race_date+" "+target.off_key, utc=True, errors="coerce")-pd.Timedelta(minutes=5)
        if "prediction_as_of" in frame:
            moment = pd.to_datetime(frame.prediction_as_of, utc=True, errors="raise").reset_index(drop=True)
            cutoff = cutoff.where(cutoff.le(moment), moment)
        ceiling = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
        if ceiling.tzinfo is None:
            raise ValueError("RTV snapshot cutoff must include a timezone")
        target["_cutoff"] = cutoff.where(cutoff.le(ceiling), ceiling)
        observations["observed_at"] = pd.to_datetime(observations.observed_at, utc=True, errors="raise")
        keys = ["race_date", "track_key", "off_key", "horse_key"]
        observations = observations.loc[~observations.duplicated(keys+["observed_at"], keep=False)]
        target = target.loc[target._cutoff.notna()]
        joined = pd.merge_asof(target.sort_values("_cutoff"), observations.sort_values("observed_at"),
            left_on="_cutoff", right_on="observed_at", by=keys, direction="backward")
        output.loc[joined._row, list(PRE_FEATURES)] = joined[list(PRE_FEATURES)].to_numpy()
    output.index = frame.index
    return pd.concat([frame.drop(columns=list(PRE_FEATURES), errors="ignore"), output], axis=1)


def add_rtv_features(frame):
    return add_pre_race_features(add_history_features(frame))
