"""Targeted backfill of the ``wind_surgery`` column in ``data/races.db``.

Sporting Life result pages carry ``medication: [{"short_name": "WS"}]`` on the
first run after a declared wind operation (BHA declarations began Jan 2018, so
the whole DB — which starts 2020-09 — is inside the declaration era). The
scraper now stores this for NEW scrapes; this script repairs existing rows by
re-fetching each race-result page once and UPDATE-ing only ``wind_surgery``.

Mirrors ``scripts/backfill_draw.py``: reuses ``SportingLifeScraper`` (retry /
backoff ``_get``, slug-correct URLs from the day listing, the new
``wind_surgery`` parse) and is resumable via a ``ws_backfill_done`` checkpoint
table — re-running skips races already completed, progress commits per-date.

Usage:
  python scripts/backfill_wind_surgery.py                      # everything
  python scripts/backfill_wind_surgery.py --start 2024-01-01 --end 2024-12-31
  python scripts/backfill_wind_surgery.py --dry-run            # report scope
  python scripts/backfill_wind_surgery.py --limit 50           # smoke test
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


def _ensure_checkpoint(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS ws_backfill_done ("
        "  race_id TEXT PRIMARY KEY,"
        "  n_updated INTEGER,"
        "  n_ws INTEGER,"
        "  ts TEXT"
        ")"
    )
    # The column itself (init_db adds it too, but keep the script standalone).
    try:
        con.execute("ALTER TABLE results ADD COLUMN wind_surgery INTEGER")
    except sqlite3.OperationalError:
        pass
    con.commit()


def _pending(con: sqlite3.Connection, start: str, end: str) -> dict[str, list[str]]:
    """race_date -> [race_id, ...] for races not yet checkpointed."""
    rows = con.execute(
        "SELECT DISTINCT r.race_id, r.race_date FROM results r "
        "WHERE r.race_date BETWEEN ? AND ? "
        "AND r.race_id NOT IN (SELECT race_id FROM ws_backfill_done) "
        "ORDER BY r.race_date",
        (start, end),
    ).fetchall()
    by_date: dict[str, list[str]] = defaultdict(list)
    for race_id, race_date in rows:
        by_date[race_date].append(str(race_id))
    return by_date


def _apply_race(
    con: sqlite3.Connection, race_id: str, runners: list[dict]
) -> tuple[int, int]:
    """UPDATE wind_surgery for one race's runners; return (rows changed, WS)."""
    changed = ws_count = 0
    for rn in runners:
        ws = int(rn.get("wind_surgery", 0) or 0)
        ws_count += ws
        hid = str(rn.get("horse_id", "") or "")
        hname = rn.get("horse_name", "")
        if hid:
            cur = con.execute(
                "UPDATE results SET wind_surgery = ? "
                "WHERE race_id = ? AND horse_id = ?",
                (ws, race_id, hid),
            )
        else:
            cur = con.execute(
                "UPDATE results SET wind_surgery = ? "
                "WHERE race_id = ? AND horse_name = ?",
                (ws, race_id, hname),
            )
        changed += cur.rowcount
    return changed, ws_count


def backfill(
    con: sqlite3.Connection, start: str, end: str, dry_run: bool, limit: int | None
) -> None:
    scraper = SportingLifeScraper()
    by_date = _pending(con, start, end)
    total_races = sum(len(v) for v in by_date.values())
    print(
        f"{total_races:,} races pending across {len(by_date):,} dates "
        f"({start}..{end})"
    )
    if dry_run:
        print("dry-run: no fetching/writes")
        return
    if not total_races:
        return

    done_races = 0
    rows_changed = 0
    ws_total = 0
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
            n, n_ws = _apply_race(con, rid, runners)
            con.execute(
                "INSERT OR REPLACE INTO ws_backfill_done"
                "(race_id, n_updated, n_ws, ts) VALUES (?, ?, ?, ?)",
                (rid, n, n_ws, ts),
            )
            rows_changed += n
            ws_total += n_ws
            done_races += 1
        con.commit()
        print(
            f"  {race_date}: {len(results)} races, {rows_changed:,} rows / "
            f"{ws_total:,} WS flags so far ({done_races:,}/{total_races:,})"
        )

    print(
        f"done: {done_races:,} races, {rows_changed:,} rows updated, "
        f"{ws_total:,} WS-flagged runs, {unresolved:,} race_ids unresolved"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.path.join("data", "races.db"))
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on races to process (smoke test / resume in chunks)")
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=30.0)
    _ensure_checkpoint(con)
    backfill(con, args.start, args.end, args.dry_run, args.limit)
    con.close()


if __name__ == "__main__":
    main()
