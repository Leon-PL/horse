"""
Targeted backfill of the starting-stall ``draw`` column in ``data/races.db``.

Why: the scraper historically stored ``cloth_number`` (saddle-cloth) in the
``draw`` column instead of ``draw_number`` (the starting-stall draw). That
corrupted every draw-bias feature in ``src/feature_engineer.py``. The scraper is
now fixed for NEW scrapes; this repairs the existing rows WITHOUT re-running the
full results scrape (no pedigree/weather/odds re-fetch, only the ``draw`` column
is touched).

Two phases:

  1. JUMPS (Hurdle/Chase) -> no starting stalls, so ``draw_number`` is always
     null. A single bulk ``UPDATE ... SET draw=0``. No HTTP at all.

  2. FLAT (incl. all-weather) -> the true draw lives only on each race-result
     page, so each flat race is fetched ONCE. We reuse ``SportingLifeScraper``
     (its retry/backoff ``_get``, slug-correct URLs from the day listing, and
     the corrected ``draw_number`` parse) and then UPDATE only the ``draw``
     column. Resumable via a ``draw_backfill_done`` checkpoint table: re-running
     skips races already completed, and progress is committed per-date.

Usage:
  python scripts/backfill_draw.py                       # jumps + all flat
  python scripts/backfill_draw.py --start 2022-01-01 --end 2022-12-31
  python scripts/backfill_draw.py --jumps-only          # phase 1 only
  python scripts/backfill_draw.py --flat-only           # phase 2 only
  python scripts/backfill_draw.py --dry-run             # report scope, no writes
  python scripts/backfill_draw.py --limit 50            # first N flat races (smoke test)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_scraper import MAX_WORKERS, SportingLifeScraper  # noqa: E402

JUMPS_TYPES = ("Hurdle", "Chase")


def _ensure_checkpoint(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS draw_backfill_done ("
        "  race_id TEXT PRIMARY KEY,"
        "  n_updated INTEGER,"
        "  ts TEXT"
        ")"
    )
    con.commit()


def backfill_jumps(con: sqlite3.Connection, dry_run: bool) -> int:
    """Bulk-set draw=0 for hurdle/chase rows (no stalls -> no draw)."""
    placeholders = ",".join("?" for _ in JUMPS_TYPES)
    where = f"race_type IN ({placeholders}) AND draw <> 0"
    n = con.execute(
        f"SELECT COUNT(*) FROM results WHERE {where}", JUMPS_TYPES
    ).fetchone()[0]
    print(f"[jumps] {n:,} rows have a non-zero (cloth) draw to clear")
    if n and not dry_run:
        con.execute(f"UPDATE results SET draw = 0 WHERE {where}", JUMPS_TYPES)
        con.commit()
        print(f"[jumps] set draw=0 on {n:,} rows")
    return n


def _pending_flat(con: sqlite3.Connection, start: str, end: str) -> dict[str, list[str]]:
    """race_date -> [race_id, ...] for flat races not yet checkpointed."""
    rows = con.execute(
        "SELECT DISTINCT r.race_id, r.race_date FROM results r "
        "WHERE r.race_type = 'Flat' AND r.race_date BETWEEN ? AND ? "
        "AND r.race_id NOT IN (SELECT race_id FROM draw_backfill_done) "
        "ORDER BY r.race_date",
        (start, end),
    ).fetchall()
    by_date: dict[str, list[str]] = defaultdict(list)
    for race_id, race_date in rows:
        by_date[race_date].append(str(race_id))
    return by_date


def _apply_race(con: sqlite3.Connection, race_id: str, runners: list[dict]) -> int:
    """UPDATE draw for one race's runners; return rows changed."""
    changed = 0
    for rn in runners:
        draw = rn.get("draw", 0)
        hid = str(rn.get("horse_id", "") or "")
        hname = rn.get("horse_name", "")
        if hid:
            cur = con.execute(
                "UPDATE results SET draw = ? WHERE race_id = ? AND horse_id = ?",
                (draw, race_id, hid),
            )
        else:
            from src.identity_schema import result_rowid
            cur = con.execute(
                "UPDATE results SET draw = ? WHERE rowid = ?",
                (draw, result_rowid(con,{"race_id":race_id,"horse_name":hname})),
            )
        changed += cur.rowcount
    return changed


def backfill_flat(
    con: sqlite3.Connection, start: str, end: str, dry_run: bool, limit: int | None
) -> None:
    scraper = SportingLifeScraper()
    by_date = _pending_flat(con, start, end)
    total_races = sum(len(v) for v in by_date.values())
    print(
        f"[flat] {total_races:,} flat races pending across {len(by_date):,} dates "
        f"({start}..{end})"
    )
    if dry_run:
        print("[flat] dry-run: no fetching/writes")
        return
    if not total_races:
        return

    done_races = 0
    rows_changed = 0
    unresolved = 0
    budget = limit if limit is not None else total_races

    for race_date in sorted(by_date):
        if done_races >= budget:
            break
        wanted = set(by_date[race_date])
        try:
            links = scraper.get_results_urls(race_date, uk_only=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {race_date}: listing failed ({e}) — skipping date")
            continue
        url_by_id = {str(li["race_id"]): li["url"] for li in links}

        targets = []
        for rid in wanted:
            url = url_by_id.get(rid)
            if url is None:
                unresolved += 1
                continue
            targets.append((rid, url))
        if limit is not None:
            targets = targets[: max(0, budget - done_races)]
        if not targets:
            continue

        results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(scraper.scrape_race_result, url, race_date): rid
                for rid, url in targets
            }
            for fut in as_completed(futs):
                rid = futs[fut]
                try:
                    results[rid] = fut.result() or []
                except Exception as e:  # noqa: BLE001
                    print(f"  ! race {rid}: fetch failed ({e})")

        ts = dt.datetime.now().isoformat(timespec="seconds")
        for rid, runners in results.items():
            n = _apply_race(con, rid, runners)
            con.execute(
                "INSERT OR REPLACE INTO draw_backfill_done(race_id, n_updated, ts) "
                "VALUES (?, ?, ?)",
                (rid, n, ts),
            )
            rows_changed += n
            done_races += 1
        con.commit()
        print(
            f"  {race_date}: {len(results)} races, {rows_changed:,} rows updated so far "
            f"({done_races:,}/{total_races:,})"
        )

    print(
        f"[flat] done: {done_races:,} races, {rows_changed:,} rows updated, "
        f"{unresolved:,} race_ids unresolved in listing"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.path.join("data", "races.db"))
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--jumps-only", action="store_true")
    ap.add_argument("--flat-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on flat races to process (smoke test / resume in chunks)")
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=30.0)
    _ensure_checkpoint(con)

    if not args.flat_only:
        backfill_jumps(con, args.dry_run)
    if not args.jumps_only:
        backfill_flat(con, args.start, args.end, args.dry_run, args.limit)

    con.close()
    if not args.dry_run:
        from src.database import publish_after_repair
        publish_after_repair(args.db)


if __name__ == "__main__":
    main()
