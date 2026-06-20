"""
Remove leaked non-runner rows from ``data/races.db``.

Withdrawn horses (Sporting Life ``ride_status == "NONRUNNER"``) were stored as
bogus ``finish_position == 0`` runners because the old filter matched the wrong
spelling (``"NON-RUNNER"``). The scraper is now fixed; this repairs history.

The authoritative field size is ``num_runners`` (Betfair's runner count matches
it 100%). Each race must therefore end with exactly ``num_runners`` rows. The
contaminating rows are all ``finish_position == 0``, but NOT every ``fp == 0``
row is a non-runner — jumps fallers / pulled-up horses genuinely ran. So:

  Phase 1 (deterministic, all years, no network):
      Races where (stored - num_runners) == count(fp == 0). Here every fp==0
      row must be a non-runner (no genuine fallers), so delete them all.

  Phase 2 (Betfair-validated, the ambiguous remainder, 2022+):
      Races where some fp==0 rows are real fallers. Delete only fp==0 rows whose
      horse is absent from Betfair's runner set for that race. Applied ONLY when
      the count of Betfair-absent fp==0 rows equals the gap exactly — otherwise
      the race is left untouched (a name mismatch must never delete a real
      runner).

Deleted rows are backed up to CSV first (``data/`` is gitignored). Idempotent:
re-running finds less to do. Each touched race is verified to end at
``num_runners`` rows.

Usage:
  python scripts/cleanup_non_runners.py --dry-run
  python scripts/cleanup_non_runners.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.betfair import bf_horse_key, bf_track_key, load_bsp  # noqa: E402
from src.utils import normalise_off_time_key  # noqa: E402


def _betfair_runner_set(bsp_path: str) -> set | None:
    """Set of (race_date, track_key, off_key, horse_key) Betfair runners."""
    bf = load_bsp(bsp_path)
    if bf is None or bf.empty:
        return None
    rd = bf["race_date"].astype(str).str.slice(0, 10)
    tk = bf["track"].map(bf_track_key)
    ok = bf["off_key"] if "off_key" in bf.columns else bf["off_time"].map(normalise_off_time_key)
    hk = bf["horse_key"] if "horse_key" in bf.columns else bf["horse_name"].map(bf_horse_key)
    return set(zip(rd, tk, ok, hk))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.path.join("data", "races.db"))
    ap.add_argument("--bsp", default=os.path.join("data", "processed", "betfair_bsp.parquet"))
    ap.add_argument("--backup", default=os.path.join("data", "processed", "deleted_non_runners.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=30.0)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT rowid, race_id, race_date, track, off_time, num_runners, "
        "finish_position, horse_name FROM results"
    ).fetchall()

    by_race: dict[str, list] = defaultdict(list)
    for r in rows:
        by_race[r["race_id"]].append(r)

    bf_runners = _betfair_runner_set(args.bsp)
    print(f"loaded {len(rows):,} rows / {len(by_race):,} races; "
          f"Betfair runner set: {0 if bf_runners is None else len(bf_runners):,}")

    to_delete: list = []            # sqlite Row objects
    stats = defaultdict(int)

    for race_id, rr in by_race.items():
        num = rr[0]["num_runners"]
        gap = len(rr) - num
        if gap <= 0:
            stats["clean"] += 1
            continue
        fp0 = [r for r in rr if r["finish_position"] == 0]
        if gap == len(fp0):
            # Deterministic: every fp==0 row is a non-runner.
            to_delete.extend(fp0)
            stats["deterministic_races"] += 1
            stats["deterministic_rows"] += len(fp0)
        elif gap < len(fp0) and bf_runners is not None:
            # Ambiguous: some fp==0 rows are genuine fallers. Use Betfair.
            rdate = str(rr[0]["race_date"])[:10]
            tk = bf_track_key(rr[0]["track"])
            ok = normalise_off_time_key(rr[0]["off_time"])
            absent = [
                r for r in fp0
                if (rdate, tk, ok, bf_horse_key(r["horse_name"])) not in bf_runners
            ]
            if len(absent) == gap:
                to_delete.extend(absent)
                stats["betfair_races"] += 1
                stats["betfair_rows"] += len(absent)
            else:
                stats["ambiguous_unresolved"] += 1
        else:
            stats["ambiguous_no_betfair"] += 1

    print("\nplan:")
    print(f"  deterministic : {stats['deterministic_races']:,} races, {stats['deterministic_rows']:,} rows")
    print(f"  betfair-valid : {stats['betfair_races']:,} races, {stats['betfair_rows']:,} rows")
    print(f"  total to delete: {len(to_delete):,} rows")
    print(f"  left unresolved: {stats['ambiguous_unresolved']:,} (betfair count != gap) + "
          f"{stats['ambiguous_no_betfair']:,} (no betfair cover)")

    if not to_delete:
        print("nothing to delete.")
        con.close()
        return
    if args.dry_run:
        print("dry-run: no deletions.")
        con.close()
        return

    # Backup, then delete by rowid.
    os.makedirs(os.path.dirname(args.backup), exist_ok=True)
    with open(args.backup, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(to_delete[0].keys())
        for r in to_delete:
            w.writerow(list(r))
    print(f"\nbacked up {len(to_delete):,} rows -> {args.backup}")

    touched_races = {r["race_id"] for r in to_delete}
    con.executemany("DELETE FROM results WHERE rowid = ?", [(r["rowid"],) for r in to_delete])
    con.commit()

    # Verify each touched race now has exactly num_runners rows.
    bad = 0
    for race_id in touched_races:
        row = con.execute(
            "SELECT num_runners, COUNT(*) FROM results WHERE race_id = ? GROUP BY num_runners",
            (race_id,),
        ).fetchone()
        if row and row[1] != row[0]:
            bad += 1
    after = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    print(f"deleted; results now {after:,} rows")
    print(f"verification: {bad:,} touched races still != num_runners "
          f"({'OK' if bad == 0 else 'INVESTIGATE'})")
    con.close()


if __name__ == "__main__":
    main()
