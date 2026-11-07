"""Sporting Life results: the resumable listing and race-page collector.

``src.database`` owns the tables, reads, writes and the published snapshot. This module
decides what to fetch and records how each attempt ended, so an interrupted
collection resumes without hiding a gap.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src import database
from src.ingestion.storage import atomic_json, writer_lock

logger = logging.getLogger(__name__)

# Legacy rows have no scrape manifest. Recheck a small recent window without
# unexpectedly re-scraping years of existing history on the first upgraded run.
RECENT_RELIST_DAYS = 7


def sync_database(days_back: int = 90, *, enrich: bool = False,
                  start=None, end=None, recheck=False, revision_interval_seconds=21600,
                  pedigree_limit=100, weather=False, offline=False, backlog_max_races=0):
    """Resume result ingestion and atomically export the complete history.

    Keep one idle connection for the operation's lifetime. Transactions use
    their own short-lived handles; the idle handle prevents SQLite from
    checkpointing the WAL after every individual race's connection closes.
    It never opens a read or write transaction across network requests.
    Set ``enrich=False`` to collect results without optional source backfills.
    """
    if days_back < 0:
        raise ValueError("days_back must be non-negative")
    if backlog_max_races < 0:
        raise ValueError("Historical recovery budget must be non-negative")
    with writer_lock(Path(database.DB_PATH).with_suffix(".source.lock")) as acquired:
        if not acquired:
            raise RuntimeError("Results refresh is already running")
        database.init_db()
        with database._get_conn() as idle:
            idle.execute("SELECT race_id FROM results LIMIT 1").fetchall()
            report = _sync_database(days_back, enrich=enrich, start=start, end=end, recheck=recheck,
                revision_interval_seconds=revision_interval_seconds, pedigree_limit=pedigree_limit, weather=weather, offline=offline,
                backlog_max_races=backlog_max_races)
        atomic_json(Path(database.DB_PATH).with_suffix(".refresh.json"), report)
        return report


def _sync_database(days_back: int, *, enrich=False, start=None, end=None, recheck=False,
                   revision_interval_seconds=21600, pedigree_limit=100, weather=False, offline=False,
                   backlog_max_races=0):
    """Resume incomplete race scrapes and publish the full history.

    Each listed unfinished race is attempted once per invocation, within the
    requested window. Recent dates are relisted to discover delayed results.
    Older legacy dates without a manifest are preserved, not assumed audited.
    Empty listings remain retryable: the scraper also returns [] on parse
    failures, so they cannot certify that a day had no races.
    """
    from src.ingestion.sporting_life import SportingLifeScraper

    if days_back < 0:
        raise ValueError("days_back must be non-negative")
    database.init_db()
    existing_dates = database.get_dates_in_db()
    today = datetime.now()
    all_dates = [
        (today - timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(days_back)
    ]
    if start is not None or end is not None:
        last = datetime.fromisoformat(end).date() if end else (today-timedelta(days=1)).date()
        first = datetime.fromisoformat(start).date() if start else last-timedelta(days=max(0, days_back-1))
        if first > last or last >= today.date():
            raise ValueError("Results range must end before today and start no later than its end")
        all_dates = [(last-timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
    date_window = set(all_dates)
    recent_dates = {d for d in all_dates if d >= (today-timedelta(days=RECENT_RELIST_DAYS)).date().isoformat()}
    with database._get_conn() as conn:
        before_revision = conn.execute("SELECT revision FROM result_source_state WHERE id=1").fetchone()[0]
        managed_dates = {r[0] for r in conn.execute("SELECT race_date FROM result_sync_dates")}
        incomplete_dates = {r[0] for r in conn.execute("""
            SELECT d.race_date FROM result_sync_dates d
            WHERE d.listing_succeeded = 0 OR EXISTS (
                SELECT 1 FROM result_sync_races r
                WHERE r.race_date = d.race_date AND r.completed = 0
            )
        """)}
    legacy_dates = (existing_dates & date_window) - managed_dates - recent_dates
    if legacy_dates:
        logger.info("Preserving %d older legacy dates without a completeness audit", len(legacy_dates))
    listing_dates = sorted(
        (date_window - existing_dates - managed_dates)
        | (incomplete_dates & date_window) | recent_dates
    )
    if recheck:
        listing_dates = sorted(date_window)
    scraper = SportingLifeScraper()
    scraper.raw_store_path = Path(database.DB_PATH).parent / "sl_cache/raw.sqlite"
    scraper.offline = offline

    # Persist the listing attempt and every expected race before fetching any
    # result. An interruption or partial failure therefore cannot hide a gap.
    for date_str in listing_dates:
        with database._get_conn() as conn:
            conn.execute("""
                INSERT INTO result_sync_dates
                    (race_date, listing_attempts, last_attempt_at) VALUES (?, 1, datetime('now'))
                ON CONFLICT(race_date) DO UPDATE SET listing_succeeded = 0,
                    listing_attempts = listing_attempts + 1, last_attempt_at = datetime('now')
            """, (date_str,))
        try:
            links = scraper.get_results_urls(date_str, uk_only=True)
            if not links and date_str not in getattr(scraper, "empty_result_dates", set()):
                raise ValueError("Empty or unreadable results listing")
            race_links = {str(info.get("race_id") or ""): info["url"] for info in links}
            if any(not race_id or not url for race_id, url in race_links.items()):
                raise ValueError("Listing has a missing race ID or URL")
            with database._get_conn() as conn:
                for race_id, url in race_links.items():
                    # Adopt complete legacy races without refetching their
                    # pages; a stored fragment must still be retried.
                    stored = conn.execute("""
                        SELECT COUNT(*), MAX(num_runners), MAX(finish_position = 1),
                               MIN(num_runners)
                        FROM results WHERE race_date = ? AND race_id = ?
                    """, (date_str, race_id)).fetchone()
                    complete = int(bool(stored[0] and stored[2]
                                        and stored[0] == stored[1] == stored[3]))
                    conn.execute("""
                        INSERT INTO result_sync_races (race_date, race_id, url, completed)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(race_date, race_id) DO UPDATE SET
                            url = excluded.url,
                            completed = CASE
                                WHEN result_sync_races.last_error LIKE 'not run:%'
                                THEN result_sync_races.completed
                                ELSE excluded.completed END
                    """, (date_str, race_id, url, complete))
                conn.execute("""
                    UPDATE result_sync_dates SET listing_succeeded = 1, last_error = NULL
                    WHERE race_date = ?
                """, (date_str,))
        except Exception as exc:
            logger.warning("Failed to verify listing for %s: %s", date_str, exc)
            with database._get_conn() as conn:
                conn.execute("UPDATE result_sync_dates SET last_error = ? WHERE race_date = ?",
                             (str(exc), date_str))

    # Persisted URLs remain usable even when a subsequent listing request
    # fails or omits the race. Never truncate the number of races in a day.
    with database._get_conn() as conn:
        candidates = conn.execute("SELECT race_date,race_id,url,completed,last_attempt_at,last_error "
                                  "FROM result_sync_races ORDER BY race_date,race_id").fetchall()
        jobs = []
        for ds, race_id, url, closed, attempted, error in candidates:
            if ds not in date_window:
                continue
            due = (attempted is None or (today-datetime.fromisoformat(attempted)).total_seconds() >= revision_interval_seconds)
            if not closed or ((recheck or (ds in recent_dates and due)) and not str(error or "").startswith("not run:")):
                jobs.append((ds, race_id, url))
        backlog_jobs = []
        if backlog_max_races:
            cutoff = (today-timedelta(days=1)).isoformat()
            candidates = conn.execute("SELECT race_date,race_id,url FROM result_sync_races "
                "WHERE completed=0 AND race_date<? AND (last_attempt_at IS NULL OR last_attempt_at<?) "
                "AND COALESCE(last_error,'') NOT LIKE 'Review required:%' "
                "AND COALESCE(last_error,'') NOT LIKE 'Missing or duplicate%' "
                "ORDER BY attempts,COALESCE(last_attempt_at,''),race_date,race_id",
                (today.date().isoformat(), cutoff)).fetchall()
            backlog_jobs = [job for job in candidates if job[0] not in date_window][:backlog_max_races]
            jobs.extend(backlog_jobs)
        conn.executemany("""
            UPDATE result_sync_races SET attempts = attempts + 1, last_attempt_at = ?
            WHERE race_date = ? AND race_id = ?
        """, [(today.isoformat(), ds, race_id) for ds, race_id, _ in jobs])

    # Existing URLs identify the recovery targets. A listing can prove a
    # terminal state for many of them in one request; unseen or conflicting
    # entries still need their individual page. This does not declare the
    # rest of an old date complete or add unexamined races to its ledger.
    backlog_listing_failures = 0
    backlog_blocked = False
    backlog_listing_loads = 0
    for ds in sorted({job[0] for job in backlog_jobs}):
        if ds in getattr(scraper, "result_listing_evidence", {}):
            continue
        try:
            backlog_listing_loads += 1
            scraper.get_results_urls(ds, uk_only=True)
        except Exception as exc:
            backlog_listing_failures += 1
            logger.warning("Historical listing unavailable on %s: %s", ds, exc)
            if getattr(getattr(exc, "response", None), "status_code", None) in {403, 429, 503}:
                backlog_blocked = True
                break
    listing_closed = _close_listing_terminals(scraper, jobs, today)
    listing_closed_ids = set(listing_closed)
    jobs = [job for job in jobs if (job[0], job[1]) not in listing_closed_ids]
    backlog_ids = {(ds, identity) for ds, identity, _ in backlog_jobs}
    if backlog_blocked:
        jobs = [job for job in jobs if (job[0], job[1]) not in backlog_ids]
    backlog_completed = len(backlog_ids & listing_closed_ids)
    inserted = 0
    completed = 0
    not_run = 0
    pages_finished = 0
    inserted_dates: set[str] = set()
    # A DORMANT page inside the relist window may simply not be updated yet;
    # only older ones are known never to have run.
    settled_before = (today - timedelta(days=RECENT_RELIST_DAYS)).strftime("%Y-%m-%d")
    if jobs:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.ingestion.sporting_life import MAX_WORKERS, RaceNotRun

        logger.info("Fetching %d unfinished race pages (%d workers)", len(jobs), MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(scraper.scrape_race_result, url, date_str=ds,
                            raise_if_not_run=True): (ds, race_id)
                for ds, race_id, url in jobs
            }
            for future in as_completed(futures):
                ds, race_id = futures[future]
                closed_this_job = False
                try:
                    frame = pd.DataFrame(future.result())
                    database._validate_race_results(frame, ds, race_id)
                    with database._get_conn() as conn:
                        added = database._replace_race_results(conn, frame)
                        conn.execute("""
                            UPDATE result_sync_races SET completed = 1, last_error = NULL
                            WHERE race_date = ? AND race_id = ?
                        """, (ds, race_id))
                    inserted += added
                    completed += 1
                    closed_this_job = True
                    if added:
                        inserted_dates.add(ds)
                except RaceNotRun as exc:
                    # Abandoned, void, never run or Arabian: close it with the
                    # reason, so "pending" keeps meaning "failed unexplained".
                    terminal = exc.reason != "DORMANT" or ds < settled_before
                    with database._get_conn() as conn:
                        conn.execute("""
                            UPDATE result_sync_races SET completed = ?, last_error = ?
                            WHERE race_date = ? AND race_id = ?
                        """, (int(terminal), f"not run: {exc.reason}", ds, race_id))
                    if terminal:
                        completed += 1
                        not_run += 1
                        closed_this_job = True
                except Exception as exc:
                    logger.warning("Race %s on %s remains pending: %s", race_id, ds, exc)
                    with database._get_conn() as conn:
                        conn.execute("""
                            UPDATE result_sync_races SET completed = 0, last_error = ?
                            WHERE race_date = ? AND race_id = ?
                        """, (str(exc), ds, race_id))
                if (ds, race_id) in backlog_ids:
                    backlog_completed += int(closed_this_job)
                pages_finished += 1
                if pages_finished % 25 == 0 or pages_finished == len(jobs):
                    logger.info("Result recovery: %d races closed; %d of %d pages finished",
                                completed+len(listing_closed), pages_finished, len(jobs))
    logger.info("Inserted %d new runner rows; %d races closed as not run; "
                "%d race requests remain pending", inserted, not_run, len(jobs) - completed)

    enrichment = {}
    if weather and date_window and not offline:
        try:
            from src.ingestion.weather import backfill_weather_in_db
            enrichment["weather"] = {"state": "ok", "updated": backfill_weather_in_db(dates=sorted(date_window))}
        except Exception as exc:
            enrichment["weather"] = {"state": "error", "error": str(exc)}
    if enrich and not offline:
        try:
            from src.ingestion.pedigree import backfill_pedigree
            enrichment["pedigree"] = {"state": "ok", "stored": backfill_pedigree(limit=pedigree_limit)}
        except Exception as exc:
            enrichment["pedigree"] = {"state": "error", "error": str(exc)}

    # Publish the snapshot for the pipeline. This MUST be the full history, not
    # the scrape window: feature engineering computes cumulative/rolling features
    # across each horse's whole career, and the batched feature-store build reads
    # this snapshot. Writing only `days_back` here previously clobbered
    # the feature store down to the scrape window.
    snapshot = database.publish_results_snapshot()
    lo, hi = min(all_dates, default="9999"), max(all_dates, default="0000")
    with database._get_conn() as conn:
        pending = conn.execute("SELECT COUNT(*) FROM result_sync_races WHERE completed=0 AND race_date BETWEEN ? AND ?", (lo, hi)).fetchone()[0]
        listings = conn.execute("SELECT COUNT(*) FROM result_sync_dates WHERE listing_succeeded=0 AND race_date BETWEEN ? AND ?", (lo, hi)).fetchone()[0]
        total_pending = conn.execute("SELECT COUNT(*) FROM result_sync_races WHERE completed=0").fetchone()[0]
        change = conn.execute("SELECT MIN(race_date),COUNT(DISTINCT race_id) FROM result_source_changes WHERE revision>?", (before_revision,)).fetchone()
        latest = conn.execute("SELECT MAX(race_date) FROM results").fetchone()[0]
        current_revision = conn.execute("SELECT revision FROM result_source_state WHERE id=1").fetchone()[0]
        backlog_counts = {}
        if backlog_max_races:
            review = "(COALESCE(last_error,'') LIKE 'Review required:%' OR COALESCE(last_error,'') LIKE 'Missing or duplicate%')"
            count, needs_review, due = conn.execute(
                f"SELECT COUNT(*),COALESCE(SUM({review}),0),COALESCE(SUM(NOT {review} "
                "AND (last_attempt_at IS NULL OR last_attempt_at<?)),0) FROM result_sync_races "
                "WHERE completed=0 AND race_date<? AND (race_date<? OR race_date>?)",
                ((today-timedelta(days=1)).isoformat(), today.date().isoformat(), lo, hi)).fetchone()
            backlog_counts = {"review_required_races": needs_review, "due_races": due,
                              "retry_waiting_races": count-needs_review-due}
    incomplete = (pending or listings or current_revision != snapshot["revision"]
                  or any(v["state"] == "error" for v in enrichment.values())
                  or (backlog_max_races and total_pending-pending))
    return {"source": "sporting_life", "state": "partial" if incomplete else "ok",
            "start": min(all_dates, default=None), "end": max(all_dates, default=None), "latest_date": latest,
            "rows": snapshot["rows"], "new_rows": inserted, "changed_races": change[1], "changed_from": change[0],
            "requests": len(jobs), "pending_races": pending, "unverified_dates": listings,
            "listing_resolved_races": len(listing_closed),
            **({"backlog": {"state": "partial" if total_pending-pending else "ok",
                             "budget": backlog_max_races, "attempted": len(backlog_jobs),
                             "completed": backlog_completed, "unresolved_attempts": len(backlog_jobs)-backlog_completed,
                             "listing_loads": backlog_listing_loads,
                             "listing_failures": backlog_listing_failures, "aborted_blocked": backlog_blocked,
                             "remaining_races": total_pending-pending, **backlog_counts}}
               if backlog_max_races else {}),
            "older_pending_races": total_pending-pending, "snapshot": snapshot, "enrichment": enrichment}


def _close_listing_terminals(scraper, jobs, today):
    """Close only dated, unambiguous source exclusions with no accepted field."""
    if not jobs:
        return []
    from src.ingestion.sporting_life import NOT_RUN_STAGES
    evidence = getattr(scraper, "result_listing_evidence", {})
    settled_before = (today-timedelta(days=RECENT_RELIST_DAYS)).date().isoformat()
    closed = []
    observed_at = datetime.now().isoformat()
    with database._get_conn() as conn:
        _ensure_terminal_evidence(conn)
        for ds, identity, _ in jobs:
            item = evidence.get(ds, {}).get(identity, {})
            reason = item.get("reason")
            if (item.get("source_date") != ds or not item.get("source_url")
                    or len(str(item.get("payload_sha256") or "")) != 64
                    or reason not in NOT_RUN_STAGES | {"ARABIAN"}
                    or item.get("has_results") and reason != "ARABIAN"
                    or reason == "DORMANT" and ds >= settled_before):
                continue
            if conn.execute("SELECT 1 FROM results WHERE race_id=? LIMIT 1", (identity,)).fetchone():
                continue
            conn.execute("UPDATE result_sync_races SET completed=1,last_error=? WHERE race_date=? AND race_id=?",
                         (f"not run: {reason}", ds, identity))
            conn.execute("INSERT OR REPLACE INTO result_terminal_evidence "
                         "(race_date,race_id,reason,source_url,observed_at,payload_sha256) VALUES (?,?,?,?,?,?)",
                         (ds, identity, reason, item["source_url"], observed_at, item["payload_sha256"]))
            closed.append((ds, identity))
    if closed:
        logger.info("Resolved %d terminal races from dated source listings", len(closed))
    return closed


def _ensure_terminal_evidence(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS result_terminal_evidence ("
                 "race_date TEXT, race_id TEXT, reason TEXT, source_url TEXT, observed_at TEXT, "
                 "payload_sha256 TEXT, PRIMARY KEY(race_date,race_id))")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(result_terminal_evidence)")}
    if "payload_sha256" not in columns:
        conn.execute("ALTER TABLE result_terminal_evidence ADD COLUMN payload_sha256 TEXT")
