"""Materialise numeric RTV observations; keep network/collection out of FE."""
from datetime import datetime, timezone
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src import rtv_features as features
from src import rtv_scraper as rtv
from src.source_pipeline import pipeline_dir, _read, match_runners, canonical_rows
from src.utils import canonical_track_key, normalise_horse_key


def _atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_columns(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    schema = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in columns if c in schema])


def observation_tables(start, end, *, source_root=None):
    root = Path(source_root or Path(config.DATA_DIR)/"rtv_cache")
    overrides = _read(root/"pipeline/history.json").get("days", {})
    overrides = {d: v for d, v in overrides.items() if start <= d <= end}
    columns = {
        "races": ["race_id", "race_date", "track_name", "off_time_utc", "is_abandoned", "is_arab_race", "distance_furlongs"],
        "runners": ["race_id", "runner_id", "horse_name", "is_finisher", "is_nonrunner", "in_results"],
        "metrics": ["race_id", "runner_id", "metric", "value"],
        "sectionals": ["race_id", "runner_id", "sector_number", "sector_time", "sector_length_yards", "sector_position"],
        "quality_flags": ["race_id", "runner_id", "check", "severity"]}
    parts = {name: [] for name in columns}
    for race_path in sorted((root/"tables/races").glob("*.parquet")):
        year, quarter = int(race_path.stem[:4]), int(race_path.stem[-1])
        lo = f"{year}-{3*quarter-2:02d}-01"
        hi = str((pd.Timestamp(lo)+pd.offsets.QuarterEnd()).date())
        if hi < start or lo > end:
            continue
        races = _read_columns(race_path, columns["races"])
        races = races.loc[races.race_date.between(start, end) & ~races.race_date.isin(overrides)]
        parts["races"].append(races)
        for name in columns.keys()-{"races"}:
            frame = _read_columns(root/"tables"/name/race_path.name, columns[name])
            if not frame.empty:
                parts[name].append(frame.loc[frame.race_id.isin(races.race_id)])
    for day, entry in overrides.items():
        for name, cols in columns.items():
            frame = _read_columns(root/"pipeline"/entry["directory"]/(name+".parquet"), cols)
            if not frame.empty:
                parts[name].append(frame)
    tables = {name: pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns[name])
              for name, chunks in parts.items()}
    tables["sectionals"] = tables["sectionals"].rename(columns={"sector_position": "race_position"})
    return tables


def refresh_run_observations(start, end, *, database=None, source_root=None, destination=None):
    tables = observation_tables(start, end, source_root=source_root)
    canonical = canonical_rows(start, end, db_path=database)
    links = match_runners(tables["races"], tables["runners"], canonical)
    output = features.build_run_observations(tables["races"], tables["runners"], tables["metrics"],
        tables["sectionals"], tables["quality_flags"], links)
    path = Path(destination or features.runs_path())
    if path.exists():
        previous = pd.read_parquet(path)
        previous = previous.loc[~pd.to_datetime(previous.race_date).between(start, end)]
        output = pd.concat([previous, output], ignore_index=True)
    output = output.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    if output.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate materialised RTV run")
    _atomic_parquet(output, path)
    return {"runs": len(output), "horses": int(output.horse_id.nunique()), "path": str(path)}


def refresh_pre_observations(day=None, *, raw_path=None, destination=None):
    store = rtv.RawStore(str(raw_path or Path(config.DATA_DIR)/"rtv_cache/rtv_raw.sqlite"))
    try:
        tables = rtv.build_snapshot_tables(start=day, end=day, store=store)
    finally:
        store.close()
    observations = []
    specs = (("pre_runners", "card_horse_name", {"card_timeform_rating": "rtv_pre_timeform_rating",
        "card_timeform_star_rating": "rtv_pre_timeform_stars"}),
        ("pace_maps", "pm_runner_name", {"pm_pace_map__position": "rtv_pre_pace_position",
        "pm_draw__lengths_advantage": "rtv_pre_draw_advantage"}))
    for table, horse_name, mappings in specs:
        frame = tables[table]
        if frame.empty:
            continue
        pieces = frame.race_key.str.split("/", expand=True)
        out = pd.DataFrame({"race_date": pieces[0], "track_key": pieces[1].str.replace("-", " ").map(canonical_track_key),
            "horse_key": frame[horse_name].map(normalise_horse_key),
            "observed_at": pd.to_datetime(frame.fetched_at, utc=True)})
        local = pieces[0]+" "+pieces[2].str[:2]+":"+pieces[2].str[2:]
        off = pd.to_datetime(local).dt.tz_localize("Europe/London").dt.tz_convert("UTC")
        out["off_key"] = off.dt.strftime("%H:%M")
        for field in features.PRE_FEATURES:
            out[field] = np.nan
        for source, target in mappings.items():
            out[target] = pd.to_numeric(frame.get(source), errors="coerce")
        # Captures made after the intended five-minute forecast decision cannot
        # become pre-race observations for that race.
        out = out.loc[out.observed_at.le(off-pd.Timedelta(minutes=5))]
        observations.append(out)
    keys = ["race_date", "track_key", "off_key", "horse_key"]
    output = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame(columns=keys+["observed_at", *features.PRE_FEATURES])
    if not output.empty:
        # Coalesce concurrent payloads only at their actual timestamps; ffill
        # cannot bring a later pace observation into an earlier card capture.
        output = output.groupby(keys+["observed_at"], as_index=False)[list(features.PRE_FEATURES)].first()
        output = output.sort_values("observed_at", kind="stable")
        output[list(features.PRE_FEATURES)] = output.groupby(keys)[list(features.PRE_FEATURES)].ffill()
    path = Path(destination or features.pre_path())
    if day and path.exists():
        previous = pd.read_parquet(path)
        output = pd.concat([previous.loc[previous.race_date.ne(day)], output], ignore_index=True)
    _atomic_parquet(output, path)
    return {"observations": len(output), "races": int(output[keys[:3]].drop_duplicates().shape[0]),
            "published_at": datetime.now(timezone.utc).isoformat()}
