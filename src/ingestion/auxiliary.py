"""Bounded ATR and public BSP refreshes using the existing source parsers."""
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

import config
from src import database
from src.betfair import attach_bsp, bf_track_key, bf_horse_key
from src.utils import normalise_off_time_key
from src.ingestion.storage import atomic_json, atomic_parquet, read_json, writer_lock


def _utc_now():
    return datetime.now(timezone.utc)


def canonical_fields(start, end):
    with sqlite3.connect(Path(database.DB_PATH).resolve().as_uri()+"?mode=ro", uri=True) as con:
        return pd.read_sql_query("SELECT race_id,race_date,track,off_time,horse_name,num_runners "
                                 "FROM results WHERE race_date BETWEEN ? AND ?", con, params=(start, end))


def _atr_race_keys(frame, atr):
    """Batch the date parser; scalar parsing dominated cached ATR refreshes."""
    days = pd.to_datetime(frame.race_date, format="mixed", errors="coerce")
    dated = days.dt.strftime("%Y-%m-%d").fillna(frame.race_date.astype(str))
    return dated + "|" + frame.track.map(atr._normalise_track_key) + "|" + frame.off_time.map(atr._normalise_off_time_key)


def refresh_atr(start, end, *, max_races=50):
    from src import attheraces_scraper as atr
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
                                    ('state','checked_at','total_races','saved_rows','races','publication_deferred_races')}}
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
    existing = set(_atr_race_keys(cache[["race_date", "track", "off_time"]].drop_duplicates(), atr)) if not cache.empty else set()
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


def _price_keys(frame):
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d")
    frame["track_key"] = frame.track.map(bf_track_key)
    frame["off_key"] = frame.off_time.map(normalise_off_time_key)
    frame["horse_key"] = frame.horse_name.map(bf_horse_key)
    return frame


def merge_public_bsp(existing, incoming):
    """Fill absent BSP only; preserve archive timestamps and every known price.

    No public VWAP is substituted for a timed archive trade. Ambiguous source
    identities are rejected, including conflicting duplicate public prices.
    """
    keys = ["race_date", "track_key", "off_key", "horse_key"]
    left, right = _price_keys(existing), _price_keys(incoming)
    for frame in (left, right):
        frame.drop_duplicates(inplace=True)
        if frame.duplicated(keys).any():
            raise ValueError("Ambiguous BSP identities; canonical merge was not published")
    right = right.loc[np.isfinite(right.bsp) & right.bsp.gt(1)].copy()
    right["bsp_source"] = "public_daily"
    if "selection_id" in right:
        right["bf_runner_id"] = right.selection_id
    # Archive IDs are integers, while the public CSV parser retains strings.
    # Use nullable integers on both sides before concatenating/serialising.
    for frame in (left, right):
        if "bf_runner_id" in frame:
            frame["bf_runner_id"] = pd.to_numeric(frame.bf_runner_id, errors="raise").astype("Int64")
    right["n_runners"] = right.groupby(keys[:3]).bsp.transform("size")
    indexed = left.set_index(keys)
    additions = right.set_index(keys)
    overlap = indexed.index.intersection(additions.index)
    known = pd.to_numeric(indexed.loc[overlap, "bsp"], errors="coerce")
    proposed = pd.to_numeric(additions.loc[overlap, "bsp"], errors="coerce")
    conflicts = int((known.gt(1) & ~np.isclose(known, proposed)).sum())
    missing = known.index[~np.isfinite(known) | known.le(1)]
    indexed.loc[missing, "bsp"] = additions.loc[missing, "bsp"]
    if "bsp_source" in indexed:
        indexed.loc[missing, "bsp_source"] = "public_daily"
    new = additions.loc[~additions.index.isin(indexed.index)]
    if not indexed.empty:
        new = new.reindex(columns=indexed.columns)
    out = pd.concat([indexed, new], sort=False).reset_index()
    return out, {"new_rows": len(new), "filled_prices": len(missing), "preserved_conflicts": conflicts}


def _bsp_signature(paths, cache_path, fields, start, end):
    """An unchanged refresh needs neither a whole-cache merge nor another join.

    Hash contents rather than timestamps: external archive repairs and public
    files replaced with preserved metadata must invalidate the receipt too.
    """
    digest = hashlib.sha256(f"{start}/{end}".encode())
    digest.update(pd.util.hash_pandas_object(fields, index=False).values.tobytes())
    source = Path(__file__).resolve()
    project = Path(config.BASE_DIR)
    dependencies = [source, project/"src/betfair.py", project/"src/utils.py",
                    project/"scripts/fetch_betfair_sp_public.py"]
    for path in [*dependencies, *paths, cache_path]:
        digest.update(str(path).encode())
        if path.exists():
            with path.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _refresh_public_bsp(start, end):
    from scripts.fetch_betfair_sp_public import download, raw_path, parse_file
    raw_dir = Path(config.DATA_DIR)/"betfair_sp_public"
    stats = download(date.fromisoformat(start), date.fromisoformat(end), raw_dir=str(raw_dir))
    paths = []
    for offset in range((date.fromisoformat(end)-date.fromisoformat(start)).days+1):
        day = date.fromisoformat(start)+timedelta(days=offset)
        for region in ("uk", "ire"):
            path = Path(raw_path("win", region, day, raw_dir=str(raw_dir)))
            if path.exists():
                paths.append((path, region))
    path = Path(config.PROCESSED_DATA_DIR)/"betfair_bsp.parquet"
    fields = canonical_fields(start, end)
    receipt_path = Path(config.DATA_DIR)/"source_refresh/bsp_receipt.json"
    signature = _bsp_signature([p for p, _ in paths], path, fields, start, end)
    receipt = read_json(receipt_path)
    if receipt.get("signature") == signature:
        return {**receipt["coverage"], **stats, "state": "partial" if stats["failed"] else "cached",
                "new_rows": 0, "filled_prices": 0, "start": start, "end": end}
    frames = [parse_file(str(p), region) for p, region in paths]
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["race_date", "track", "off_time", "horse_name", "bsp"])
    merge = {"new_rows": 0, "filled_prices": 0, "preserved_conflicts": 0}
    if frames:
        combined, merge = merge_public_bsp(existing, pd.concat(frames, ignore_index=True))
        if merge["new_rows"] or merge["filled_prices"]:
            atomic_parquet(combined, path)
            existing = combined
    matched = attach_bsp(fields, existing, cols=("bsp",))
    complete = matched.assign(matched=np.isfinite(matched.bsp) & matched.bsp.gt(1)).groupby("race_id").matched.all()
    pending = int((~complete).sum())
    coverage = {"coverage_state": "partial" if pending else "complete", "fallback_fields": pending,
                "pending_fields": pending, "complete_fields": int(complete.sum()),
                "preserved_conflicts": merge["preserved_conflicts"]}
    atomic_json(receipt_path, {"signature": _bsp_signature([p for p, _ in paths], path, fields, start, end),
                               "coverage": coverage})
    return {**stats, **merge, **coverage, "state": "partial" if stats["failed"] else "ok",
            "start": start, "end": end}


def refresh_bsp(start, end):
    """Daily BSP plus optional full historical-market downloads after setup."""
    from src.ingestion.betfair_history import configured, sync_history, HistoryError
    report = _refresh_public_bsp(start, end)
    if configured():
        last = min(date.fromisoformat(end), date.today()-timedelta(days=6))
        if date.fromisoformat(start) > last:
            report["history"] = {"state": "awaiting_publication", "available_through": str(last)}
        else:
            try:
                report["history"] = sync_history(start, str(last), budget=500, _source_lock_held=True)
            except HistoryError as error:
                report["history"] = {"state": "needs_login", "error": str(error)}
    return report
