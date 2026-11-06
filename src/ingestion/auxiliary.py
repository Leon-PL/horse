"""Bounded ATR and full Betfair historical refreshes for the source coordinator."""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import config
from src import database
from src.ingestion.storage import atomic_json, read_json, writer_lock


def _utc_now():
    return datetime.now(timezone.utc)


def canonical_fields(start, end):
    return database.result_rows(start, end, ("race_id", "race_date", "track", "off_time", "horse_name", "num_runners"))


def _atr_race_keys(frame, atr):
    """Batch the date parser; scalar parsing dominated cached ATR refreshes."""
    days = pd.to_datetime(frame.race_date, format="mixed", errors="coerce")
    dated = days.dt.strftime("%Y-%m-%d").fillna(frame.race_date.astype(str))
    return dated + "|" + frame.track.map(atr._normalise_track_key) + "|" + frame.off_time.map(atr._normalise_off_time_key)


def refresh_atr(start, end, *, max_races=50):
    from src.ingestion import attheraces as atr
    if max_races < 1:
        raise ValueError("ATR request budget must be positive")
    rebuild_root = Path(config.DATA_DIR)/'atr_cache/rebuild_v2'
    rebuild = read_json(rebuild_root/'status.json')
    if rebuild.get('state') in {'running','waiting'}:
        with writer_lock(rebuild_root/'.worker.lock') as available:
            if not available:
                counts = rebuild.get('races', {})
                return {'state':'partial','reason':'Fresh ATR backfill is running',
                        'start':start,'end':end,'fetched_races':0,
                        'pending_races':sum(counts.get(key,0) for key in ('pending','retry','review')),
                        'backfill':{key:rebuild.get(key) for key in
                                    ('state','checked_at','total_races','saved_rows','races','publication_deferred_races',
                                     'collection_version','tabs','tab_inventory')}}
    cursor_path = Path(config.DATA_DIR)/"source_refresh/atr_recovery.json"
    cursor = read_json(cursor_path)
    cache = atr.load_atr_cache()
    if cursor.get("pending_races"):
        start = min(start, cursor["start"])
    elif cursor.get("covered_through"):
        start = min(start, (date.fromisoformat(cursor["covered_through"])+timedelta(days=1)).isoformat())
    elif not cache.empty:
        start = min(start, (pd.to_datetime(cache.race_date).max()+pd.Timedelta(days=1)).date().isoformat())
    fields = canonical_fields(start, end)
    races = fields[["race_date", "track", "off_time"]].drop_duplicates()
    eligible = races.loc[~races.track.map(atr._normalise_track_key).isin(atr._NO_SECTIONALS_TRACKS)]
    current = atr.current_collection_rows(cache)
    existing = set(_atr_race_keys(current[["race_date", "track", "off_time"]].drop_duplicates(), atr)) if not current.empty else set()
    absent = atr._load_no_data_race_keys()
    resolved = existing | absent
    pending = eligible.loc[~_atr_race_keys(eligible, atr).isin(resolved)]
    if pending.empty:
        atomic_json(cursor_path, {"start": start, "covered_through": end, "pending_races": 0})
        return {"state": "cached", "start": start, "end": end, "pending_races": 0,
                "eligible_races": len(eligible), "excluded_races": len(races)-len(eligible)}
    now = _utc_now()
    ready = atr._publication_ready(pending, now=now)
    deferred = int((~ready).sum())
    if not ready.any():
        atomic_json(cursor_path, {**cursor, "start": start, "end": end,
                                  "pending_races": len(pending), "deferred_recent_races": deferred})
        return {"state": "partial", "reason": "Waiting for ATR's 48-hour publication window",
                "start": start, "end": end, "pending_races": len(pending), "fetched_races": 0,
                "deferred_recent_races": deferred, "eligible_races": len(eligible),
                "excluded_races": len(races)-len(eligible)}
    retry_after = cursor.get("retry_after")
    if retry_after and now < datetime.fromisoformat(retry_after):
        atomic_json(cursor_path, {**cursor, "start": start, "end": end, "pending_races": len(pending)})
        return {"state": "partial", "reason": "Provider retry cooldown", "retry_after": retry_after,
                "start": start, "end": end, "pending_races": len(pending), "fetched_races": 0,
                "deferred_recent_races": deferred,
                "eligible_races": len(eligible), "excluded_races": len(races)-len(eligible)}
    # Automatic refreshes yield after one polite retry. The explicit historical
    # backfill retains its longer retry policy; repeated app refreshes must not
    # restart that wait or turn a blocked response into permanent no-data.
    result = atr.backfill_atr_sectionals_for_races(pending.loc[ready], max_races=max_races, max_block_retries=1)
    remaining = max(0, len(pending)-result.get("resolved_races", 0))
    recovery = {"start": start, "end": end, "pending_races": remaining,
                "deferred_recent_races": deferred,
                "covered_through": end if not remaining else cursor.get("covered_through")}
    if result.get("aborted_blocked"):
        recovery["last_blocked_at"] = _utc_now().isoformat()
        recovery["retry_after"] = (_utc_now()+timedelta(hours=1)).isoformat()
    atomic_json(cursor_path, recovery)
    return {**result, "state": "partial" if remaining or result.get("aborted_blocked") else "ok",
            "start": start, "end": end, "pending_races": remaining,
            "deferred_recent_races": deferred,
            **({"retry_after": recovery["retry_after"]} if "retry_after" in recovery else {}),
            "eligible_races": len(eligible), "excluded_races": len(races)-len(eligible)}


def refresh_betfair(start, end):
    """Refresh full market streams; publication lag remains explicit."""
    from src.ingestion.betfair_history import configured, sync_history, HistoryError
    if not configured():
        return {"state": "needs_login", "reason": "Connect Betfair under Data → Historical exchange data"}
    first = date.fromisoformat(start)
    last = min(date.fromisoformat(end), _utc_now().date()-timedelta(days=6))
    if first > last:
        return {"state": "awaiting_publication", "available_through": str(last),
                "start": start, "end": end, "publication_lag_days": 6}
    try:
        report = sync_history(start, str(last), budget=500, _source_lock_held=True)
    except HistoryError as error:
        report = {"state": "needs_login", "error": str(error)}
    deferred = max(0, (date.fromisoformat(end)-last).days)
    if deferred and report["state"] in {"ok", "cached"}:
        report["state"] = "partial"
    if report["state"] == "busy":
        report["backfill"] = read_json(Path(config.DATA_DIR)/"betfair/history/status.json")
    return {**report, "available_through": str(last), "deferred_recent_days": deferred,
            "publication_lag_days": 6}
