"""SL canonical results plus independently stored Racing TV observations.

The orchestrators call this module, outside the feature builder. RTV payloads,
identity links and comparisons are sidecars: no result, price or model feature
is overwritten. Post-race telemetry needs a strictly prior-race transform before
it can become a predictor; historical cards are not pre-race observations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

import config
from src import rtv_scraper as rtv
from src.utils import normalise_horse_key, normalise_off_time_key

logger = logging.getLogger(__name__)


def pipeline_dir():
    return Path(config.DATA_DIR) / "rtv_cache" / "pipeline"


def _now():
    return datetime.now(timezone.utc)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _writer(root):
    """One publishing collector; competing app/recorder refreshes skip safely."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if not os.fstat(stream.fileno()).st_size:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            lock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            unlock()


def match_runners(races, runners, canonical):
    """Exact date/track/UTC-time/name links; ambiguous keys never join.

    SL and RTV numeric identities belong to different namespaces. A unique
    crosswalk is required before any RTV observation can be used with SL.
    Outcome-based renamed-horse pairing remains confined to the audit.
    """
    columns = ["rtv_race_id", "rtv_runner_id", "sl_race_id", "sl_horse_id", "horse_name", "race_date"]
    if races.empty or runners.empty or canonical.empty:
        return pd.DataFrame(columns=columns)
    races = races.loc[~races.is_abandoned.fillna(False).astype(bool)
                      & ~races.is_arab_race.fillna(False).astype(bool)].copy()
    race_cols = ["race_id", "race_date", "track_name", "off_time_utc"]
    left = runners[["race_id", "runner_id", "horse_name"]].merge(races[race_cols], on="race_id", validate="many_to_one")
    right = canonical.copy()
    for frame, track, off in ((left, "track_name", "off_time_utc"), (right, "track", "off_time")):
        frame["_date"] = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d")
        frame["_track"] = frame[track].map(rtv._track_join_key)
        frame["_off"] = frame[off].map(normalise_off_time_key)
        frame["_horse"] = frame.horse_name.map(normalise_horse_key)
    keys = ["_date", "_track", "_off", "_horse"]
    for frame in (left, right):
        frame.drop(frame.index[frame.duplicated(keys, keep=False) | frame[keys].isin(["", "None", "Nan"]).any(axis=1)], inplace=True)
    joined = left.merge(right[keys + ["race_id", "horse_id"]], on=keys, suffixes=("_rtv", "_sl"), validate="one_to_one")
    return joined.rename(columns={"race_id_rtv": "rtv_race_id", "runner_id": "rtv_runner_id",
        "race_id_sl": "sl_race_id", "horse_id": "sl_horse_id"})[columns]


def canonical_rows(start, end, db_path=None):
    db_path = Path(db_path or Path(config.DATA_DIR) / "races.db").resolve()
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as con:
        return pd.read_sql_query("SELECT race_id, horse_id, horse_name, race_date, track, off_time "
                                 "FROM results WHERE race_date BETWEEN ? AND ?", con, params=(start, end))


def _save_generation(root, kind, day, tables):
    relative = Path(kind) / day / uuid4().hex
    rtv.write_tables(tables, str(root / relative))
    return relative.as_posix()


def _refresh_rtv_features(kind, start, end=None):
    from src.rtv_feature_collection import refresh_run_observations, refresh_pre_observations
    if kind == "history":
        return refresh_run_observations(start, end)
    return refresh_pre_observations(start)


def refresh_rtv_history(days_back=10, *, today=None, min_interval_seconds=21600):
    """Collect/reparse a rolling ten-day window; leave long backfills explicit.

    Publishing a manifest follows writing *all* tables and the comparison.
    Failed refreshes retain the last complete generation and a visible error.
    """
    if days_back < 0:
        raise ValueError("days_back must be non-negative")
    today = today or date.today()
    root = pipeline_dir()
    with _writer(root) as acquired:
        if not acquired:
            return {"state": "busy"}
        manifest = _read(root / "history.json")
        days = manifest.setdefault("days", {})
        now = _now()
        wanted = [(today - timedelta(days=i+1)).isoformat() for i in range(min(days_back, rtv.RECENT_DAYS))
                  if today - timedelta(days=i+1) >= rtv.RACEIQ_START]
        wanted = [d for d in wanted if d not in days or not days[d].get("complete") or not days[d].get("features_complete")
                  or (now - datetime.fromisoformat(days[d]["checked_at"])).total_seconds() >= min_interval_seconds]
        if not wanted:
            return {"state": "cached", "days": 0}
        try:
            store = rtv.RawStore(str(root.parent / "rtv_raw.sqlite"))
            try:
                summaries = rtv.scrape_days(wanted, rtv.RTVClient(), store, today=today)
                for day in wanted:
                    tables = rtv.build_tables(day, day, store=store)
                    races, runners = tables["races"], tables["runners"]
                    # The comparison is diagnostic. Canonical labels are never repaired here.
                    comparable = (not races.empty and (~races.is_abandoned & ~races.is_arab_race).any())
                    comparison = (rtv.validate_against_db(tables, db_path=str(Path(config.DATA_DIR)/"races.db"))
                                  if comparable else {"races": 0})
                    comparison.pop("_joined", None)
                    tables["runner_links"] = match_runners(races, runners, canonical_rows(day, day))
                    generation = _save_generation(root, "history", day, tables)
                    _publish(root / generation / "comparison.json", comparison)
                    summary = summaries[day]
                    days[day] = {"directory": generation, "checked_at": now.isoformat(),
                        "complete": store.day_status(day) == "complete", "races": len(races),
                        "linked_runners": len(tables["runner_links"]), "collection": summary}
                    _publish(root / "history.json", manifest)
            finally:
                store.close()
            feature_receipt = _refresh_rtv_features("history", min(wanted), max(wanted))
            for day in wanted:
                days[day]["features_complete"] = True
            _publish(root / "history.json", manifest)
            result = {"state": "ok" if all(days[d]["complete"] for d in wanted) else "partial",
                      "checked_at": _now().isoformat(), "days": len(wanted), "latest_date": max(days), "features": feature_receipt}
            _publish(root / "history_status.json", result)
            return result
        except Exception as error:
            _publish(root / "history_status.json", {"state": "error", "checked_at": _now().isoformat(), "error": str(error)})
            raise


def capture_rtv_snapshot(date_str=None, *, min_interval_seconds=900):
    """Capture expiring race-day information without adding it to model inputs."""
    day = date_str or date.today().isoformat()
    if day != date.today().isoformat():
        return {"state": "skipped", "reason": "Race-day-only source; historical cards cannot be captured retrospectively"}
    root = pipeline_dir()
    with _writer(root) as acquired:
        if not acquired:
            return {"state": "busy"}
        manifest = _read(root / "snapshots.json")
        previous = manifest.get(day)
        if previous and (_now() - datetime.fromisoformat(previous["checked_at"])).total_seconds() < min_interval_seconds:
            return {"state": "cached"}
        try:
            store = rtv.RawStore(str(root.parent / "rtv_raw.sqlite"))
            try:
                started = _now().replace(microsecond=0)
                summary = rtv.snapshot_day(day, store=store)
                tables = rtv.build_snapshot_tables(day, day, store=store)
                # RawStore retains every capture. Each parsed generation stores
                # only this capture, avoiding quadratic daily snapshot copies.
                for name, frame in tables.items():
                    if "fetched_at" in frame:
                        tables[name] = frame.loc[pd.to_datetime(frame.fetched_at, utc=True) >= started].copy()
            finally:
                store.close()
            generation = _save_generation(root, "snapshots", day, tables)
            features = _refresh_rtv_features("snapshot", day)
            value = {"state": "ok" if summary["ok"] else "empty", "checked_at": _now().isoformat(),
                     "directory": generation, "features": features, **summary}
            manifest[day] = value
            _publish(root / "snapshots.json", manifest)
            _publish(root / "snapshot_status.json", value)
            return value
        except Exception as error:
            _publish(root / "snapshot_status.json", {"state": "error", "checked_at": _now().isoformat(), "error": str(error)})
            raise


def _optional(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception as error:
        logger.warning("Racing TV sidecar refresh failed; see rtv_cache/pipeline status: %s", error)
        return {"state": "error", "error": str(error)}


def sync_database(days_back=90, *, enrich=True):
    """Main source refresh: SL canonical history and RTV data/validation sidecars."""
    from src import database
    frame = database.sync_database(days_back=days_back, enrich=enrich)
    # RaceiQ can mature several days after SL's result. Keep its ten-day
    # revision window even when the ordinary SL catch-up requests only three.
    _optional(refresh_rtv_history, days_back=max(days_back, rtv.RECENT_DAYS) if days_back else 0)
    return frame


def get_scraped_racecards(date_str=None, uk_only=True, progress_callback=None, force_refresh=False):
    from src import data_scraper
    frame = data_scraper.get_scraped_racecards(date_str=date_str, uk_only=uk_only,
        progress_callback=progress_callback, force_refresh=force_refresh)
    _optional(capture_rtv_snapshot, date_str)
    return frame


def load_history_table(name, start=None, end=None):
    """Read backfill plus published daily replacements, without duplicate races."""
    root = pipeline_dir()
    manifests = _read(root / "history.json").get("days", {})
    manifests = {d: v for d, v in manifests.items() if (start is None or d >= start) and (end is None or d <= end)}
    # Restrict the backfill before reading its wide runner/sectional tables.
    base_dir = root.parent / "tables" / name
    paths = sorted(base_dir.glob("*.parquet")) if base_dir.is_dir() else [root.parent / "tables" / (name+".parquet")]
    chunks = []
    for path in paths:
        if not path.exists():
            continue
        if path.stem[:4].isdigit() and ((start and path.stem[:4] < start[:4]) or (end and path.stem[:4] > end[:4])):
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "race_date" not in df:
            race_path = root.parent / "tables/races" / path.name
            races = pd.read_parquet(race_path, columns=["race_id", "race_date"])
            df = df.merge(races, on="race_id", validate="many_to_one")
        df = df.loc[(df.race_date >= (start or "0000")) & (df.race_date <= (end or "9999"))
                    & ~df.race_date.isin(manifests)]
        chunks.append(df)
    for day, value in sorted(manifests.items()):
        path = root / value["directory"] / (name+".parquet")
        if path.exists():
            df = pd.read_parquet(path)
            if "race_date" not in df:
                df["race_date"] = day
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_matched_history(start, end):
    """RTV run observations keyed to SL IDs for research; contains post-race data.

    Metrics/sectionals remain separate long tables, connected by RTV runner IDs.
    Never merge these current-run observations straight into prediction inputs.
    """
    races = load_history_table("races", start, end)
    runners = load_history_table("runners", start, end)
    links = match_runners(races, runners, canonical_rows(start, end))
    return links.merge(runners, left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"],
                       validate="one_to_one", suffixes=("_sl", "")) if not links.empty else links


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sync", type=int, metavar="DAYS")
    parser.add_argument("--rtv-only", action="store_true")
    parser.add_argument("--results-only", action="store_true", help="Skip SL weather/pedigree backfills; RTV is still collected")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.sync is not None:
        print(refresh_rtv_history(args.sync) if args.rtv_only else {"canonical_rows": len(sync_database(args.sync, enrich=not args.results_only))})
    if args.snapshot:
        print(capture_rtv_snapshot())
