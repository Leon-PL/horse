"""Quality-filtered prior RTV performance and timestamped pre-race context.

Raw current-run telemetry is never returned. Prior-run observations mature
after two calendar days; historical backfill remains retrospective development
data because original publication/revision times were not captured. Pre-race
context requires an actual snapshot before the forecast decision cutoff.

The Timeform card (``CARD_FEATURES``) is the one exception: it describes the
current race. Live rows read it from a pre-decision snapshot; stored history
falls back to the backfilled card, whose published values matched the
evening-before snapshot (1 change in 490 runners, 2026-10-30) and carried no
information beyond the T-5 market (ledger 046).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import numba as nb

from src import rtv_text
from src.utils import normalise_off_time_key

RUN_METRICS = ("fsp", "top_speed", "acceleration", "avg_length", "avg_frequency", "stride_range",
    "jump_index", "lengths_gained_jumping", "speed_lost", "recovery_time", "positive_jump_percentage",
    "fsp_relative", "top_speed_relative", "late_speed_ratio", "early_position", "late_gain", "pace_variability")
LAST_METRICS = ("fsp_relative", "top_speed_relative", "jump_index", "late_speed_ratio", "late_gain")
# Measures normalised within each race, so trip and going cancel out of the
# stride and jumping values; exit_entry is jump exit speed over entry speed.
REL_METRICS = ("acceleration", "avg_length", "avg_frequency", "max_frequency", "stride_length",
    "exit_speed", "time_in_envelope", "speed_lost", "recovery_time", "exit_entry")
DERIVED_METRICS = ("time_fig",) + tuple(m+"_rel" for m in REL_METRICS)
DERIVED_FEATURES = ("rtv_time_fig_mean3", "rtv_time_fig_best3", "rtv_time_fig_last") + tuple(
    "rtv_"+m+"_rel_mean3" for m in REL_METRICS)
HISTORY_FEATURES = tuple("rtv_"+m+"_mean3" for m in RUN_METRICS) + tuple("rtv_"+m+"_last" for m in LAST_METRICS) + (
    "rtv_history_count3", "rtv_jumps_count3", "rtv_sectional_count3", "rtv_days_since_observation") + DERIVED_FEATURES
PRE_FEATURES = ("rtv_pre_timeform_rating", "rtv_pre_timeform_stars", "rtv_pre_pace_position", "rtv_pre_draw_advantage")
CARD_FEATURES = ("rtv_card_tf_rating", "rtv_card_tf_stars", "rtv_card_tf_tipped", "rtv_card_tf_improver",
    "rtv_card_tf_doubtful", "rtv_card_tf_focus")
# The Timeform comment score (src/rtv_text.py) is snapshot-first like the card.
SNAPSHOT_FEATURES = PRE_FEATURES + CARD_FEATURES + rtv_text.SNAPSHOT_FEATURES
FEATURES = HISTORY_FEATURES + SNAPSHOT_FEATURES + ("rtv_text_prob",)
# Time-figure pars are fitted once, on RaceIQ-era races (the first RaceIQ day
# onwards) up to PAR_END, so backfilling older results cannot move the par.
PAR_START = "2023-06-10"
PAR_END = "2024-06-30"
AW_SURFACES = {"AllWeather", "Polytrack", "Sand", "Tapeta"}
_GOING = {"hard": 0., "firm": 1., "good to firm": 2., "good": 3., "good to yielding": 3.5,
    "good to soft": 4., "yielding": 4., "yielding to soft": 4.5, "soft": 5., "soft to heavy": 5.5,
    "heavy": 6., "fast": 2., "standard to fast": 2.5, "standard/fast": 2.5, "standard": 3.,
    "standard to slow": 3.5, "standard/slow": 3.5, "slow": 4.}
BOUNDS = {"fsp": (60., 140.), "top_speed": (15., 50.), "acceleration": (1., 8.),
    "avg_length": (4., 10.), "avg_frequency": (1.5, 3.5), "min_length": (3., 10.), "max_length": (4., 11.),
    "jump_index": (0., 10.), "lengths_gained_jumping": (-40., 40.), "speed_lost": (-25., 5.),
    "recovery_time": (0., 10.), "positive_jump_percentage": (0., 1.)}
SECTIONAL_BAD = {"sector_count_vs_distance", "sector_glitch", "cumulative_mismatch",
    "sector_position_mismatch", "order_vs_time", "leader_gap_mismatch"}


def runs_path():
    """The selected dataset's prior-run observations."""
    from src.source_dataset import source_path
    return source_path("rtv", "runs")


def pre_path():
    """The selected dataset's pre-race snapshots."""
    from src.source_dataset import source_path
    return source_path("rtv", "pre")


def cards_path():
    """The selected dataset's backfilled cards."""
    from src.source_dataset import source_path
    return source_path("rtv", "cards")


def _number(frame, name):
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce").astype("float64")


def going_number(text):
    """Ordinal going; '(X in places)' moves a quarter step toward X."""
    if text is None or pd.isna(text):
        return np.nan
    main, _, rest = str(text).strip().casefold().replace(" / ", "/").partition("(")
    value = _GOING.get(main.strip())
    if value is None:
        return np.nan
    other = _GOING.get(rest.replace("in places", "").replace(")", "").strip())
    return value if other is None else value+.25*np.sign(other-value)


def _text(frame, name):
    return frame[name] if name in frame else pd.Series(pd.NA, index=frame.index, dtype=object)


def _par_inputs(races):
    return pd.DataFrame({"distance_yards": _number(races, "distance_yards"),
        "going_num": _text(races, "going").map(going_number).astype("float64"),
        "aw": _text(races, "surface").isin(AW_SURFACES).astype("float64"),
        "race_type": _text(races, "race_type").astype("category"),
        "track_name": _text(races, "track_name").astype("category")}, index=races.index)


def _log_win_time(races):
    seconds = _number(races, "win_time_s")
    return np.log(seconds.where((_number(races, "distance_yards")/seconds).between(10, 20)))


def fit_time_par(races):
    """Log winning time expected from trip, going, surface, race type and course.

    Fitted on races from ``PAR_START`` to ``PAR_END`` only, so every later
    refresh reproduces the same par. Returns None without enough timed races.
    """
    import lightgbm as lgb
    if races.empty or "win_time_s" not in races:
        return None
    races = races.loc[pd.to_datetime(races.race_date).between(PAR_START, PAR_END)
        & ~_text(races, "is_abandoned").fillna(False).astype(bool)
        & ~_text(races, "is_arab_race").fillna(False).astype(bool) & _text(races, "race_type").notna()]
    races = races.sort_values("race_id", kind="stable")
    target = _log_win_time(races)
    fit = target.notna()
    if fit.sum() < 500:
        return None
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=40,
        deterministic=True, force_row_wise=True, n_jobs=1, random_state=0, verbose=-1).fit(
        _par_inputs(races)[fit], target[fit])


def _expected_log_time(races, par):
    """Par plus the meeting's leave-one-race-out time variant, by RTV race."""
    if par is None or races.empty or "win_time_s" not in races:
        return pd.Series(dtype="float64")
    expected = pd.Series(par.predict(_par_inputs(races)), index=races.index)
    deviation = _log_win_time(races)-expected
    meeting = [races.race_date.astype(str), _text(races, "track_name").astype(str), _text(races, "surface").isin(AW_SURFACES)]
    total = deviation.groupby(meeting).transform("sum")
    others = deviation.groupby(meeting).transform("count")-deviation.notna().astype(int)
    variant = ((total-deviation.fillna(0))/others.where(others.gt(0))).where(others.ge(2), 0.)
    return pd.Series((expected+variant).to_numpy(), index=races.race_id.to_numpy())


def _relative_metrics(runners, metrics):
    """Within-race z of stride and jumping measures over every RTV starter."""
    columns = ["race_id", "runner_id", *(m+"_rel" for m in REL_METRICS)]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    names = {m for m in REL_METRICS if m != "exit_entry"} | {"entry_speed"}
    values = metrics.loc[metrics.metric.isin(names), ["race_id", "runner_id", "metric", "value"]].copy()
    if values.duplicated(["race_id", "runner_id", "metric"]).any():
        raise ValueError("Duplicate RTV metric")
    values["value"] = pd.to_numeric(values.value, errors="coerce")
    for name, (lo, hi) in BOUNDS.items():
        values.loc[values.metric.eq(name) & ~values.value.between(lo, hi), "value"] = np.nan
    wide = values.pivot(index=["race_id", "runner_id"], columns="metric", values="value").reset_index()
    starters = runners.loc[~runners.is_nonrunner.fillna(False).astype(bool), ["race_id", "runner_id"]]
    wide = starters.merge(wide, on=["race_id", "runner_id"], how="left", validate="one_to_one")
    entry = _number(wide, "entry_speed")
    wide["exit_entry"] = _number(wide, "exit_speed")/entry.where(entry.gt(0))
    for name in REL_METRICS:
        value = _number(wide, name)
        group = value.groupby(wide.race_id)
        z = (value-group.transform("mean"))/group.transform("std").clip(lower=1e-9)
        wide[name+"_rel"] = z.where(group.transform("count").ge(4)).clip(-4, 4)
    return wide[columns]


def _time_figures(races, sector, par):
    """100 x (expected log time - own log time) at the final sector: + is faster."""
    expected = _expected_log_time(races, par)
    if expected.empty or "cumulative_sector_time" not in sector:
        return pd.DataFrame(columns=["race_id", "runner_id", "time_fig"])
    final = sector.loc[sector.sector_number.eq(sector.groupby("race_id").sector_number.transform("max"))]
    final = final.loc[~final.duplicated(["race_id", "runner_id"], keep=False)].copy()
    elapsed = _number(final, "cumulative_sector_time")
    yards = pd.to_numeric(final.race_id.map(races.drop_duplicates("race_id").set_index("race_id").distance_yards), errors="coerce")
    elapsed = elapsed.where((yards/elapsed).between(10, 20))
    final["time_fig"] = 100*(final.race_id.map(expected)-np.log(elapsed))
    return final[["race_id", "runner_id", "time_fig"]]


def build_run_observations(races, runners, metrics, sectionals, flags, links, *, par=None):
    """Build one numeric observation per exact SL/RTV starter identity.

    Uses measured values, never provider ranks/percentages/averages or prices.
    Bad sectional races/runners are masked before summaries are calculated.
    FSP and completed-race sectional summaries require a physical finisher.
    ``par`` (from :func:`fit_time_par`) enables sectional time figures.
    """
    columns = ["race_id", "horse_id", "race_date", "available_date", *RUN_METRICS, *DERIVED_METRICS]
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
    base = base.merge(_relative_metrics(runners, metrics), on=["race_id", "runner_id"], how="left", validate="one_to_one")
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
        base = base.merge(_time_figures(races, sector, par), on=["race_id", "runner_id"], how="left", validate="one_to_one")
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
    base["time_fig"] = _number(base, "time_fig").where(base.is_finisher.fillna(False))
    output = pd.DataFrame({"race_id": base.sl_race_id.astype(str), "horse_id": pd.to_numeric(base.sl_horse_id),
        "race_date": pd.to_datetime(base.race_date).dt.normalize()})
    output["available_date"] = output.race_date+pd.Timedelta(days=2)
    for name in RUN_METRICS+DERIVED_METRICS:
        output[name] = _number(base, name).to_numpy()
    return output[columns]


def card_values(frame):
    """Numeric Timeform card fields: one parser for backfilled and snapshot cards.

    The rating text carries Timeform's symbols ('111p', '94?'); '-' and '0'
    mean unrated. Flags are defined only where a card was published.
    """
    text = _text(frame, "card_timeform_rating").astype("string").str.strip()
    stars = pd.to_numeric(_text(frame, "card_timeform_star_rating"), errors="coerce")
    symbol = text.str.replace(r"^\d+", "", regex=True).fillna("")
    rating = pd.to_numeric(text.str.extract(r"^(\d+)", expand=False), errors="coerce")
    output = pd.DataFrame({
        "rtv_card_tf_rating": rating.where(rating.gt(0)),
        "rtv_card_tf_stars": stars,
        "rtv_card_tf_tipped": pd.to_numeric(_text(frame, "card_timeform_tipped_place"), errors="coerce").map({1: 3., 2: 2., 3: 1.}).fillna(0.),
        "rtv_card_tf_improver": symbol.str.contains("p", case=False).astype("float64"),
        "rtv_card_tf_doubtful": symbol.str.contains("?", regex=False).astype("float64"),
        "rtv_card_tf_focus": _text(frame, "card_focus_horse").astype("string").str.lower().eq("true").fillna(False).astype("float64"),
    }, index=frame.index).astype("float64")
    output.loc[text.isna().to_numpy() & stars.isna().to_numpy()] = np.nan
    return output


def build_card_observations(runners, links):
    """The published Timeform card for each exact SL/RTV starter identity."""
    columns = ["race_id", "horse_id", "race_date", *CARD_FEATURES]
    if links.empty:
        return pd.DataFrame(columns=columns)
    base = links.merge(runners.drop(columns=["race_date"], errors="ignore"),
        left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"], validate="one_to_one")
    output = pd.concat([pd.DataFrame({"race_id": base.sl_race_id.astype(str), "horse_id": pd.to_numeric(base.sl_horse_id),
        "race_date": pd.to_datetime(base.race_date).dt.normalize()}), card_values(base)], axis=1)
    return output.loc[output[list(CARD_FEATURES)].notna().any(axis=1), columns].reset_index(drop=True)


@nb.njit(cache=True)
def _rolling_three(values, horse_ids):
    means = np.full(values.shape, np.nan)
    maxima = np.full(values.shape, np.nan)
    counts = np.zeros(values.shape)
    start = 0
    for i in range(len(values)):
        if i == 0 or horse_ids[i] != horse_ids[i-1]:
            start = i
        lo = max(start, i-2)
        for column in range(values.shape[1]):
            total, count, largest = 0., 0, -np.inf
            for row in range(lo, i+1):
                value = values[row, column]
                if not np.isnan(value):
                    total += value
                    count += 1
                    largest = max(largest, value)
            counts[i, column] = count
            if count:
                means[i, column] = total/count
                maxima[i, column] = largest
    return means, counts, maxima


def _history_observations(runs):
    if runs.empty:
        return pd.DataFrame()
    runs = runs.copy()
    measured = RUN_METRICS+DERIVED_METRICS
    for name in measured:
        runs[name] = _number(runs, name)
    runs["_horse"] = pd.to_numeric(runs.horse_id, errors="coerce")
    runs = runs.loc[runs._horse.notna() & runs._horse.gt(0)]
    runs = runs.loc[~runs.duplicated(["_horse", "race_date"], keep=False)]
    runs = runs.sort_values(["_horse", "race_date", "race_id"], kind="stable")
    # No telemetry means no observation, rather than a fictitious zero run.
    runs = runs.loc[runs[list(measured)].notna().any(axis=1)].reset_index(drop=True)
    group = runs.groupby("_horse", sort=False)
    result = runs[["_horse", "race_date", "available_date"]].copy()
    rolled, support, best = _rolling_three(runs[list(measured)].to_numpy(dtype="float64"), runs._horse.to_numpy(dtype="int64"))
    for column, name in enumerate(measured):
        result["rtv_"+name+"_mean3"] = rolled[:, column]
    result["rtv_time_fig_best3"] = best[:, measured.index("time_fig")]
    # The latest figure the horse has, however many untimed runs followed it.
    result["rtv_time_fig_last"] = group.time_fig.ffill()
    # Match the three-observation window used by the rolling features. A
    # newer observation carrying only other metrics does not erase this one.
    latest = group[list(LAST_METRICS)].ffill(limit=2)
    for name in LAST_METRICS:
        result["rtv_"+name+"_last"] = latest[name]
    result["rtv_history_count3"] = (group.cumcount()+1).clip(upper=3)
    for short, name in (("jumps", "jump_index"), ("sectional", "late_speed_ratio")):
        result["rtv_"+short+"_count3"] = support[:, measured.index(name)]
    result["_horse"] = result._horse.astype("int64")
    result["available_date"] = pd.to_datetime(result.available_date)
    result["race_date"] = pd.to_datetime(result.race_date)
    return result


@lru_cache(maxsize=2)
def _cached_history(path, modified, size):
    # The path names one immutable version of the selected dataset's table; saved per version and code.
    from src import fe_cache
    from src.source_dataset import model_table
    return fe_cache.frame("rtv_history", [path, modified, size, fe_cache.source_hash(__file__)],
                          lambda: _history_observations(model_table("rtv", "runs")))


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
    return rtv_text.replaced_columns(frame, FEATURES, output)


RIDER_CLAIM_SNAPSHOT_COLUMNS = (
    "rider_card_observed", "rider_name", "rider_weight_lbs", "rider_claim_lbs",
)


def _latest_pre_race_rows(frame, observations, *, as_of=None):
    """Join only the latest eligible, unambiguous snapshot to each runner."""
    keys = ["sl_race_id", "sl_horse_id"]
    required = {"race_id", "horse_id", "off_time", "race_date"}
    if observations.empty or not required.issubset(frame):
        return pd.DataFrame()
    if not {*keys, "match_status"}.issubset(observations):
        raise ValueError("RTV snapshots must carry their canonical links (sl_race_id, sl_horse_id, match_status)")
    observations = observations.loc[observations.match_status.eq("matched")].copy()
    for key in keys:
        observations[key] = observations[key].astype("string").str.replace(r"\.0$", "", regex=True)
    target = pd.DataFrame({
        "_row": np.arange(len(frame)),
        "sl_race_id": frame.race_id.astype("string").str.replace(r"\.0$", "", regex=True).to_numpy(),
        "sl_horse_id": frame.horse_id.astype("string").str.replace(r"\.0$", "", regex=True).to_numpy(),
    }).astype(dict.fromkeys(keys, "string"))
    day = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d")
    off = frame.off_time.map(normalise_off_time_key)
    cutoff = pd.to_datetime(day + " " + off, utc=True, errors="coerce") - pd.Timedelta(minutes=5)
    if "prediction_as_of" in frame:
        moment = pd.to_datetime(frame.prediction_as_of, utc=True, errors="raise")
        cutoff = cutoff.mask(cutoff.gt(moment), moment)
    ceiling = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if ceiling.tzinfo is None:
        raise ValueError("RTV snapshot cutoff must include a timezone")
    target["_cutoff"] = cutoff.clip(upper=ceiling).array
    observations["observed_at"] = pd.to_datetime(observations.observed_at, utc=True, errors="raise")
    observations = observations.loc[~observations.duplicated(keys + ["observed_at"], keep=False)]
    target = target.loc[target._cutoff.notna() & ~target.duplicated(keys, keep=False)]
    return pd.merge_asof(target.sort_values("_cutoff"), observations.sort_values("observed_at"),
                         left_on="_cutoff", right_on="observed_at", by=keys, direction="backward")


def add_pre_race_features(frame, *, observations=None, as_of=None):
    """Use the latest snapshot captured before each runner's decision cutoff."""
    if observations is None:
        from src.source_dataset import read_table
        observations = read_table("rtv", "pre", linked=True)
    joined = _latest_pre_race_rows(frame, observations, as_of=as_of)
    output = pd.DataFrame(np.nan, index=pd.RangeIndex(len(frame)), columns=SNAPSHOT_FEATURES)
    if not joined.empty:
        output.loc[joined._row, list(SNAPSHOT_FEATURES)] = joined.reindex(
            columns=list(SNAPSHOT_FEATURES)).to_numpy(dtype="float64", na_value=np.nan)
    output.index = frame.index
    return rtv_text.replaced_columns(frame, SNAPSHOT_FEATURES, output)


def add_rider_claims(frame):
    """Resolve per-runner claims from accepted, rider-matched evidence."""
    required = {"race_id", "horse_id", "race_date", "jockey", "weight_lbs", "finish_position"}
    if not required.issubset(frame) or frame.empty:
        return frame
    from src.feature_semantics import claim_pounds, rider_name_key
    from src.source_dataset import read_table

    explicit = claim_pounds(frame.get("reported_jockey_claim_lbs", pd.Series(np.nan, index=frame.index)))
    fallback = pd.Series(np.nan, index=pd.RangeIndex(len(frame)))
    jockey = rider_name_key(frame.jockey).reset_index(drop=True)
    weight = pd.to_numeric(frame.weight_lbs, errors="coerce").reset_index(drop=True)
    settled = pd.to_numeric(frame.finish_position, errors="coerce").gt(0).groupby(
        frame.race_id, sort=False).transform("any").fillna(False).to_numpy(dtype=bool)
    if settled.any():
        history = read_table("rtv", "rider_claims")
        if not history.empty:
            keys = ["race_id", "horse_id", "race_date"]
            target = frame.loc[settled, keys].copy()
            target["_row"] = np.flatnonzero(settled)
            for key in keys[:2]:
                target[key] = target[key].astype("string").str.replace(r"\.0$", "", regex=True)
                history[key] = history[key].astype("string")
            target["race_date"] = pd.to_datetime(target.race_date).dt.strftime("%Y-%m-%d")
            matched = target.merge(history, on=keys, validate="many_to_one")
            rows = matched._row.to_numpy()
            valid = (matched.rider_name.eq(jockey.iloc[rows].array) &
                     matched.rider_weight_lbs.eq(weight.iloc[rows].array))
            fallback.iloc[rows] = claim_pounds(matched.rider_claim_lbs).where(valid).to_numpy()
    if (~settled).any():
        pre = read_table("rtv", "pre", linked=True)
        if not pre.empty and set(RIDER_CLAIM_SNAPSHOT_COLUMNS).issubset(pre):
            positions = np.flatnonzero(~settled)
            joined = _latest_pre_race_rows(frame.iloc[positions], pre.loc[pre.rider_card_observed.eq(1)])
            if not joined.empty:
                rows = positions[joined._row.to_numpy()]
                valid = (joined.rider_name.eq(jockey.iloc[rows].array) &
                         joined.rider_weight_lbs.eq(weight.iloc[rows].array))
                fallback.iloc[rows] = claim_pounds(joined.rider_claim_lbs).where(valid).to_numpy()
    fallback.index = frame.index
    conflict = explicit.notna() & fallback.notna() & explicit.ne(fallback)
    return frame.assign(reported_jockey_claim_lbs=explicit.fillna(fallback).mask(conflict))


def _card_table(cards):
    cards = cards.assign(race_id=cards.race_id.astype(str), horse_id=pd.to_numeric(cards.horse_id, errors="coerce"))
    return cards.loc[~cards.duplicated(["race_id", "horse_id"], keep=False)]


@lru_cache(maxsize=2)
def _cached_cards(path, modified, size):
    # The path names one immutable version of the selected dataset's table.
    from src.source_dataset import model_table
    return _card_table(model_table("rtv", "cards"))


def add_card_features(frame, *, cards=None):
    """Fill rows without a pre-decision card snapshot from the backfilled card.

    Snapshot values win: they are what a live forecast sees. Upcoming races
    have no backfilled card, so live rows can only use the snapshot.
    """
    path = cards_path()
    if cards is not None:
        cards = _card_table(cards)
    elif path.exists():
        stat = path.stat()
        cards = _cached_cards(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if cards is None or cards.empty or not {"race_id", "horse_id"}.issubset(frame):
        return frame
    target = pd.DataFrame({"race_id": frame.race_id.astype(str).to_numpy(),
        "horse_id": pd.to_numeric(frame.horse_id, errors="coerce").to_numpy()})
    joined = target.merge(cards[["race_id", "horse_id", *CARD_FEATURES]], on=["race_id", "horse_id"], how="left")
    current = frame.reindex(columns=list(CARD_FEATURES)).to_numpy(dtype="float64", na_value=np.nan)
    missing = np.isnan(current).all(axis=1)
    if not missing.any():
        return frame
    current[missing] = joined.loc[missing, list(CARD_FEATURES)].to_numpy(dtype="float64", na_value=np.nan)
    output = pd.DataFrame(current, index=frame.index, columns=list(CARD_FEATURES))
    return rtv_text.replaced_columns(frame, CARD_FEATURES, output)


def add_rtv_features(frame):
    from src.ingestion.rtv_storage import pin_generation
    with pin_generation():
        return rtv_text.add_text_features(add_card_features(add_pre_race_features(add_history_features(frame))))
