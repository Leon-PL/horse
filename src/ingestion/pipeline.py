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
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import pandas as pd

import config
from src.database import result_rows
from src.ingestion import racing_tv as rtv
from src.utils import normalise_horse_key, normalise_off_time_key
from src.ingestion.rtv_storage import history_manifest, generation, accepted_days, partition_paths
from src.ingestion.storage import writer_lock, atomic_json

logger = logging.getLogger(__name__)
DEFAULT_SOURCES = ("sl", "rtv", "atr", "betfair", "fx", "bha", "foreign")


def pipeline_dir():
    return Path(config.DATA_DIR) / "rtv_cache" / "pipeline"


def _now():
    return datetime.now(timezone.utc)


def _read(path):
    if path.name == "history.json":
        return history_manifest(path.parent.parent)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@contextmanager
def _writer(root):
    """One publishing collector; competing app/recorder refreshes skip safely."""
    with writer_lock(root/".writer.lock") as acquired:
        yield acquired


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
    return result_rows(start, end, ("race_id", "horse_id", "horse_name", "race_date", "track", "off_time"),
                       db_path=db_path or Path(config.DATA_DIR) / "races.db")


def _save_generation(root, kind, day, tables):
    relative = Path(kind) / day / uuid4().hex
    rtv.write_tables(tables, str(root / relative))
    return relative.as_posix()


def _refresh_rtv_features(kind, start, end=None, *, manifest=None):
    from src.ingestion.rtv_observations import refresh_run_observations, refresh_pre_observations
    if kind == "history":
        return refresh_run_observations(start, end, manifest=manifest, publish=False)
    return refresh_pre_observations(start)


def refresh_rtv_history(days_back=10, *, today=None, min_interval_seconds=21600, start=None, end=None, reparse=False):
    """Recover the requested range and recheck the recent revision window.

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
        last = date.fromisoformat(end) if end else today-timedelta(days=1)
        first = date.fromisoformat(start) if start else last-timedelta(days=max(days_back-1, 0))
        if last >= today or first > last:
            raise ValueError("Invalid RTV historical range")
        requested = [(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)] if days_back or start or end else []
        base_days = set()
        for path in partition_paths(root.parent/"tables/races", first.isoformat(), last.isoformat()):
            base_days.update(pd.read_parquet(path, columns=["race_date"]).race_date.astype(str).str[:10])
        wanted = []
        for day in requested:
            entry = days.get(day)
            recent = (today-date.fromisoformat(day)).days <= rtv.RECENT_DAYS
            if entry is None and day in base_days and not recent and not reparse:
                continue
            if (reparse or entry is None or not entry.get("complete") or not entry.get("features_complete")
                    or (recent and (now-datetime.fromisoformat(entry["checked_at"])).total_seconds() >= min_interval_seconds)):
                wanted.append(day)
        if not wanted:
            return {"state": "cached", "days": 0, "start": first.isoformat(), "end": last.isoformat()}
        try:
            store = rtv.RawStore(str(root.parent / "rtv_raw.sqlite"))
            try:
                summaries = ({day: {"reparsed": True} for day in wanted} if reparse else
                    rtv.scrape_days(wanted, rtv.RTVClient(), store, today=today, profile="core"))
                incomplete = []
                for day in wanted:
                    tables = rtv.build_tables(day, day, store=store)
                    races, runners = tables["races"], tables["runners"]
                    # The comparison is diagnostic. Canonical labels are never repaired here.
                    comparable = (not races.empty and (~races.is_abandoned & ~races.is_arab_race).any())
                    comparison = (rtv.validate_against_db(tables, db_path=str(Path(config.DATA_DIR)/"races.db"))
                                  if comparable else {"races": 0})
                    comparison.pop("_joined", None)
                    canonical = canonical_rows(day, day)
                    tables["runner_links"] = match_runners(races, runners, canonical)
                    directory = _save_generation(root, "history", day, tables)
                    atomic_json(root / directory / "comparison.json", comparison)
                    summary = summaries.get(day, {"races": len(races), "done": 0})
                    status = store.day_status(day)
                    complete = status == "complete" or (status == "list_404" and canonical.empty)
                    if status == "list_404" and canonical.empty:
                        summary = {**summary, "state": "confirmed_empty"}
                    if not canonical.empty and races.empty:
                        complete = False
                    prior = days.get(day, {})
                    expected_links = min(prior.get("linked_runners", 0), len(canonical))
                    if len(tables["runner_links"]) < expected_links:
                        complete = False
                    if not complete:
                        incomplete.append(day)
                        continue
                    days[day] = {"directory": directory, "checked_at": now.isoformat(),
                        "complete": True, "races": len(races),
                        "linked_runners": len(tables["runner_links"]), "collection": summary}
            finally:
                store.close()
            accepted = [day for day in wanted if day not in incomplete]
            if not accepted:
                result = {"state": "partial", "days": len(wanted), "pending_dates": incomplete,
                          "start": first.isoformat(), "end": last.isoformat(), "checked_at": _now().isoformat()}
                atomic_json(root/"history_status.json", result)
                return result
            feature_receipt = _refresh_rtv_features("history", min(accepted), max(accepted), manifest=manifest)
            for day in accepted:
                days[day]["features_complete"] = True
            previous = generation(root.parent)
            previous.update(history=manifest, files=feature_receipt.get("files", previous.get("files", {})))
            atomic_json(root/"generation.json", previous)
            result = {"state": "partial" if incomplete else "ok", "pending_dates": incomplete,
                      "checked_at": _now().isoformat(), "days": len(accepted), "latest_date": max(days), "features": feature_receipt}
            atomic_json(root / "history_status.json", result)
            return result
        except Exception as error:
            atomic_json(root / "history_status.json", {"state": "error", "checked_at": _now().isoformat(), "error": str(error)})
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
            return {**previous, "state": "cached" if previous["state"] == "ok" else previous["state"]}
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
            directory = _save_generation(root, "snapshots", day, tables)
            features = _refresh_rtv_features("snapshot", day)
            value = {"state": "ok" if summary["ok"] == summary["requests"]-1 else "partial" if summary["ok"] else "empty", "checked_at": _now().isoformat(),
                     "directory": directory, "features": features, **summary}
            manifest[day] = value
            atomic_json(root / "snapshots.json", manifest)
            atomic_json(root / "snapshot_status.json", value)
            return value
        except Exception as error:
            atomic_json(root / "snapshot_status.json", {"state": "error", "checked_at": _now().isoformat(), "error": str(error)})
            raise


def _optional(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except Exception as error:
        logger.warning("Optional source %s refresh failed: %s", getattr(call, "__name__", "unknown"), error)
        return {"state": "error", "error": str(error)}


def _publish_source_health(path, report, previous):
    """Keep per-source coverage when a run only checks selected providers."""
    merged = {}
    for name, body in previous.get("sources", {}).items():
        if name not in DEFAULT_SOURCES:
            continue
        merged[name] = {**body,
            "checked_at": body.get("checked_at") or previous.get("checked_at"),
            "requested_start": body.get("requested_start", previous.get("start")),
            "requested_end": body.get("requested_end", previous.get("end"))}
    for name, body in report["sources"].items():
        attempt = {**body, "checked_at": report["checked_at"],
                   "requested_start": report["start"], "requested_end": report["end"]}
        from src.ingestion.publication import latest_source_health
        attempt["health"] = latest_source_health(name, attempt)
        prior = merged.get(name, {})
        # An older explicit backfill does not recheck or erase newer coverage.
        if (prior.get("requested_end") or "") > report["end"]:
            merged[name] = {**prior, "last_attempt": attempt}
        else:
            merged[name] = attempt
    checked = [body["checked_at"] for body in merged.values() if body.get("checked_at")]
    starts = [body["requested_start"] for body in merged.values() if body.get("requested_start")]
    ends = [body["requested_end"] for body in merged.values() if body.get("requested_end")]
    lo, hi = max(starts, default=report["start"]), min(ends, default=report["end"])
    complete = (set(DEFAULT_SOURCES).issubset(merged) and lo <= hi
                and all(merged[name].get("state") in {"ok", "cached"} for name in DEFAULT_SOURCES))
    health = {**report, "required_sources": list(DEFAULT_SOURCES), "sources": merged,
              "start": min(lo, hi), "end": hi, "checked_at": min(checked, default=report["checked_at"]),
              "updated_at": report["checked_at"],
              "state": "running" if report["state"] == "running" else "ok" if complete else "partial"}
    health.pop("seconds", None)  # Different provider checks have no shared elapsed time.
    health["latest_run"] = {key: report[key] for key in ("state", "checked_at", "required_sources", "start", "end", "seconds")
                            if key in report}
    atomic_json(path.with_name("last_run.json"), report)
    atomic_json(path, health)


def refresh_sources(days_back=14, *, enrich=False, start=None, end=None, recheck=False,
                    sources=DEFAULT_SOURCES, atr_max_races=50, pedigree_limit=100,
                    weather=False, offline=False, max_source_workers=3, progress_callback=None,
                    sl_backlog_max_races=0):
    """One bounded source plan; return coverage and failures without loading history."""
    from src.ingestion import results
    from src.ingestion.auxiliary import refresh_atr, refresh_betfair
    from src.ingestion.fx import refresh_prize_references
    from src.ingestion.bha_stewards import refresh_reports
    from src.ingestion.foreign_form import refresh_foreign_form
    allowed = set(DEFAULT_SOURCES)
    if days_back < 0 or set(sources)-allowed or not sources:
        raise ValueError("Choose nonnegative days and sources from sl, rtv, atr, betfair, fx, bha, foreign")
    if atr_max_races < 1 or pedigree_limit < 1:
        raise ValueError("ATR and pedigree budgets must be positive")
    if max_source_workers < 1:
        raise ValueError("Source worker count must be positive")
    if sl_backlog_max_races < 0 or (sl_backlog_max_races and "sl" not in sources):
        raise ValueError("SL historical recovery needs a non-negative budget and the sl source")
    if offline and not set(sources).issubset({"sl", "rtv", "foreign"}):
        raise ValueError("Offline replay supports --sources sl,rtv,foreign")
    last = date.fromisoformat(end) if end else date.today()-timedelta(days=1)
    first = date.fromisoformat(start) if start else last-timedelta(days=max(days_back-1, 0))
    if first > last or last >= date.today():
        raise ValueError("Source range must end before today and start no later than its end")
    root = Path(config.DATA_DIR)/"source_refresh"
    status_path = root/("backlog_status.json" if not days_back and not start and not end and sl_backlog_max_races else "status.json")
    with writer_lock(root/".writer.lock") as acquired:
        if not acquired:
            return {"state": "busy", "sources": {}, "error": "Another source refresh is running"}
        started = time.monotonic()
        previous = _read(status_path)
        report = {"state": "running", "checked_at": _now().isoformat(), "start": first.isoformat(),
                  "end": last.isoformat(), "required_sources": list(sources),
                  "sources": {name: {"state": "pending"} for name in sources}}

        def refresh_rtv():
            history = refresh_rtv_history(days_back=max(days_back, rtv.RECENT_DAYS) if days_back else 0,
                **({"start": first.isoformat(), "end": last.isoformat()} if start or end else {}),
                **({"reparse": True} if offline else {}))
            if enrich and not offline and history.get("state") in {"ok", "cached"}:
                from src.ingestion.pedigree import backfill_rtv_foaling_profiles
                history["foaling_profiles"] = _optional(backfill_rtv_foaling_profiles, limit=pedigree_limit,
                    start=first.isoformat(), end=last.isoformat())
            return history

        operations = {
            "sl": lambda: results.sync_database(days_back=days_back, enrich=enrich,
                start=start, end=end, recheck=recheck or offline, pedigree_limit=pedigree_limit, weather=weather, offline=offline,
                backlog_max_races=sl_backlog_max_races),
            "rtv": refresh_rtv,
            "atr": lambda: refresh_atr(first.isoformat(), last.isoformat(), max_races=atr_max_races),
            "betfair": lambda: refresh_betfair(first.isoformat(), last.isoformat()),
            "fx": refresh_prize_references,
            "bha": lambda: refresh_reports(first.isoformat(), last.isoformat(), recheck=recheck),
            # Complete French/UAE fields: raiders' overseas form and rating-only races.
            "foreign": lambda: refresh_foreign_form(first.isoformat(), last.isoformat(), replay=offline),
        }
        def execute(source):
            source_started = time.monotonic()
            try:
                requested = days_back or start or end or (source == "sl" and sl_backlog_max_races)
                value = operations[source]() if requested else {"state": "skipped", "reason": "Empty source window"}
            except Exception as error:
                logger.warning("%s refresh failed: %s", source, error)
                value = {"state": "error", "error": str(error)}
            return {**value, "seconds": round(time.monotonic()-source_started, 3)}

        def record(source, value):
            report["sources"][source] = value
            report["checked_at"] = _now().isoformat()
            report["seconds"] = round(time.monotonic()-started, 3)
            publish_report()
            if progress_callback is not None:
                progress_callback(source, dict(value))

        def publish_report():
            if status_path.name == "status.json":
                _publish_source_health(status_path, report, previous)
            else:
                atomic_json(status_path, report)

        publish_report()
        # Canonical sporting records precede every identity join. The remaining
        # collectors have separate stores and per-source request-rate limits.
        if "sl" in sources:
            record("sl", {"state": "running"})
            record("sl", execute("sl"))
        independent = [name for name in ("fx", "betfair", "rtv", "atr", "bha", "foreign") if name in sources]
        if independent:
            with ThreadPoolExecutor(max_workers=min(max_source_workers, len(independent))) as pool:
                jobs = {}
                for source in independent:
                    record(source, {"state": "running"})
                    jobs[pool.submit(execute, source)] = source
                for job in as_completed(jobs):
                    source = jobs[job]
                    record(source, job.result())
                    logger.info("Source %s: %s (%.1fs)", source,
                                report["sources"][source]["state"], report["sources"][source]["seconds"])
        report["state"] = ("ok" if all(value.get("state") in {"ok", "cached", "skipped"}
                                       for value in report["sources"].values()) else "partial")
        report["checked_at"] = _now().isoformat()
        report["seconds"] = round(time.monotonic()-started, 3)
        # Publish through the common boundary after native stores are accepted.
        # Isolated/test coordinators without a canonical database have nothing
        # to adopt. A failed adoption retains all previous accepted versions.
        if (Path(config.DATA_DIR) / "races.db").exists():
            from src.source_dataset import capture_dataset
            try:
                manifest = capture_dataset(include_prices=True)
                report["dataset_manifest_id"] = manifest["id"]
            except Exception as error:
                report["publication_error"] = str(error)
                report["state"] = "partial"
        from src.source_health import consumer_readiness
        known = {**previous.get("sources", {}), **report["sources"]}
        report["readiness"] = {purpose: consumer_readiness(known, purpose=purpose, requested_end=report["end"])
                               for purpose in ("features", "training", "serving", "evaluation")}
        if report.get("publication_error"):
            for value in report["readiness"].values():
                value["ready"] = False
                value["blockers"].append("source publication failed")
        publish_report()
        return report


def refresh_summary(report):
    labels = {"sl": "Sporting Life", "rtv": "Racing TV", "atr": "ATR", "betfair": "Betfair archives", "fx": "Exchange rates", "bha": "BHA stewards"}
    values = [f"{labels.get(name, name)}: {value.get('state', 'unknown')}" for name, value in report.get("sources", {}).items()]
    return f"Source refresh {report['state']} — " + "; ".join(values)


def get_scraped_racecards(date_str=None, uk_only=True, progress_callback=None, force_refresh=False):
    from src.ingestion import sporting_life
    frame = sporting_life.get_scraped_racecards(date_str=date_str, uk_only=uk_only,
        progress_callback=progress_callback, force_refresh=force_refresh)
    result = _optional(capture_rtv_snapshot, date_str)
    if frame is not None:
        frame.attrs["source_refresh"] = {"rtv_snapshot": result}
        if date_str is None or date_str == date.today().isoformat():
            from src.ingestion.fx import refresh_prize_references
            frame.attrs["source_refresh"]["fx"] = _optional(refresh_prize_references)
    return frame


def load_history_table(name, start=None, end=None):
    """Read backfill plus published daily replacements, without duplicate races."""
    root = pipeline_dir()
    manifests = accepted_days(root.parent)
    manifests = {d: v for d, v in manifests.items() if (start is None or d >= start) and (end is None or d <= end)}
    # Restrict the backfill before reading its wide runner/sectional tables.
    base_dir = root.parent / "tables" / name
    paths = partition_paths(base_dir, start, end) if base_dir.is_dir() else [root.parent / "tables" / (name+".parquet")]
    chunks = []
    seen = set()
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
        if "race_id" in df:
            df = df.loc[~df.race_id.isin(seen)]
            seen.update(df.race_id)
        chunks.append(df)
    for day, value in sorted(manifests.items()):
        path = root / value["directory"] / (name+".parquet")
        if path.exists():
            df = pd.read_parquet(path)
            if "race_date" not in df:
                df["race_date"] = day
            chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


#: Racing TV text by when it is written: the Timeform card before the off, the
#: close-up comment after the race (usable only as history of earlier runs).
RTV_PRE_RACE_TEXT = {"card_analyst_comment_text": "tf_comment", "card_analyst_verdict_text": "tf_verdict",
                     "card_ip_hints_specific_pace_text": "pace_hint", "card_draw_comment": "draw_comment"}
RTV_POST_RACE_TEXT = {"comment": "close_up"}


def load_matched_text(start, end):
    """Racing TV text per Sporting Life runner, for research text features.

    One row per linked starter keyed by SL ``race_id``/``horse_id``. Race-level
    text (verdict, pace and draw notes) repeats on each runner. Pre-race card
    text has been checked against a live snapshot only for 2026-10-30.
    """
    races = load_history_table("races", start, end)
    runners = load_history_table("runners", start, end)
    links = match_runners(races, runners, canonical_rows(start, end))
    columns = ["race_id", "horse_id", "race_date", *RTV_PRE_RACE_TEXT.values(), *RTV_POST_RACE_TEXT.values()]
    if links.empty:
        return pd.DataFrame(columns=columns)
    text = {**RTV_PRE_RACE_TEXT, **RTV_POST_RACE_TEXT}
    race_text = races.reindex(columns=["race_id", *(c for c in text if c in races)])
    runner_text = runners.reindex(columns=["race_id", "runner_id", *(c for c in text if c in runners and c not in races)])
    out = (links.merge(runner_text, left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"], validate="one_to_one")
           .drop(columns=["race_id", "runner_id"])
           .merge(race_text, left_on="rtv_race_id", right_on="race_id", validate="many_to_one")
           .drop(columns=["race_id"]).rename(columns={"sl_race_id": "race_id", "sl_horse_id": "horse_id", **text}))
    return out.reindex(columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sync", type=int, metavar="DAYS")
    parser.add_argument("--publish-sources", action="store_true", help="Publish local accepted sources and promote validated ATR work; no network")
    parser.add_argument("--source-status", action="store_true", help="Show accepted source versions and quality; no network")
    parser.add_argument("--rtv-only", action="store_true")
    parser.add_argument("--foreign-form", action="store_true",
                        help="Backfill complete France and UAE results through Sporting Life (history only, never training rows)")
    parser.add_argument("--publish-foreign-form", action="store_true",
                        help="Publish retained overseas form alongside the committed domestic results")
    parser.add_argument("--results-only", action="store_true", help="Skip optional pedigree/weather; required sources are still collected")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--start", help="Explicit first historical date")
    parser.add_argument("--end", help="Explicit last historical date")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--atr-max-races", type=int, default=50)
    parser.add_argument("--sl-backlog-max-races", type=int, default=0,
                        help="Recover this many due historical SL jobs using their persisted URLs")
    parser.add_argument("--enrich", action="store_true", help="Collect bounded pedigree and foaling-date batches")
    parser.add_argument("--rtv-profile-backfill", nargs="?", const=True, type=int, metavar="LIMIT",
                        help="Collect and publish missing Racing TV foaling dates across the archive")
    parser.add_argument("--weather", action="store_true", help="Explicit research weather collection")
    parser.add_argument("--recheck", action="store_true", help="Revalidate completed SL races in the requested range")
    parser.add_argument("--replay", action="store_true", help="Offline SL parser replay from retained responses")
    args = parser.parse_args()
    if args.foreign_form:
        if not args.start or not args.end or args.sync is not None or args.rtv_only:
            parser.error("--foreign-form requires --start and --end and runs independently of domestic sync")
        from src.ingestion.foreign_form import refresh_foreign_form
        from src.ingestion.publication import publish_foreign_form_dataset
        logging.basicConfig(level=logging.INFO)
        result = refresh_foreign_form(args.start, args.end, replay=args.replay)
        manifest = publish_foreign_form_dataset()
        result["dataset_manifest_id"] = manifest["id"]
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["state"] == "ok" else 1)
    if args.publish_foreign_form:
        from src.ingestion.publication import publish_foreign_form_dataset
        manifest = publish_foreign_form_dataset()
        print(json.dumps({"dataset_manifest_id": manifest["id"]}, indent=2))
    if args.rtv_profile_backfill is not None:
        from src.ingestion.pedigree import backfill_rtv_foaling_and_publish
        limit = None if args.rtv_profile_backfill is True else args.rtv_profile_backfill
        if limit is not None and limit < 1:
            parser.error("--rtv-profile-backfill LIMIT must be positive")
        logging.basicConfig(level=logging.INFO)
        result = backfill_rtv_foaling_and_publish(limit=limit,
                                      progress=lambda value: logger.info("RTV profiles %s", value))
        print(json.dumps(result, indent=2))
        if result["state"] not in {"ok", "cached"}:
            raise SystemExit(1)
    if args.publish_sources:
        from src.source_dataset import capture_dataset
        manifest = capture_dataset(include_prices=True)
        print(json.dumps({"dataset_manifest_id": manifest["id"], "sources": {
            name: {"version": value["version"], "coverage": value["reports"]} for name, value in manifest["sources"].items()}}, indent=2))
    if args.source_status:
        from src.source_dataset import registry_root
        from src.ingestion.publication import source_health
        print(json.dumps({path.parent.name: source_health({}, _read(path))
                          for path in registry_root().glob("*/latest.json")}, indent=2))
    if args.sync is not None or args.start or args.end:
        args.sync = 14 if args.sync is None else args.sync
        result = (refresh_rtv_history(args.sync, start=args.start, end=args.end, reparse=args.replay) if args.rtv_only else
            refresh_sources(args.sync, enrich=args.enrich and not args.results_only, start=args.start, end=args.end,
                sources=("sl",) if args.replay and args.sources == ",".join(DEFAULT_SOURCES) else tuple(args.sources.split(",")), atr_max_races=args.atr_max_races,
                weather=args.weather and not args.results_only, recheck=args.recheck, offline=args.replay,
                sl_backlog_max_races=args.sl_backlog_max_races))
        print(json.dumps(result, indent=2, default=str))
        if result["state"] not in {"ok", "cached"}:
            raise SystemExit(1)
    if args.snapshot:
        print(capture_rtv_snapshot())


if __name__ == "__main__":
    main()
