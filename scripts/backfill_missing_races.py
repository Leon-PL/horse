"""
Backfill races the scraper dropped, identified by cross-referencing Betfair.
==========================================================================

Reads the missing-races worklist from ``scripts/betfair_db_audit.py``
(``data/processed/missing_races.csv``), re-scrapes exactly those races from
Sporting Life and inserts them. ``insert_results`` is ``INSERT OR IGNORE`` on
``(race_id, horse_name)``, so already-present rows are untouched.

The original gaps came from transient fetch failures with no retry; ``_get``
now retries with backoff, so this run should be complete. Any race that still
fails to resolve is written to ``data/processed/backfill_failures.csv``.

Usage:
    python scripts/backfill_missing_races.py            # dry-run summary
    python scripts/backfill_missing_races.py --commit   # scrape + insert
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import bf_track_key  # noqa: E402
from src.data_scraper import SportingLifeScraper  # noqa: E402
from src.database import insert_results  # noqa: E402
from src.utils import normalise_off_time_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", default="data/processed/missing_races.csv")
    ap.add_argument("--commit", action="store_true", help="actually scrape + insert")
    args = ap.parse_args()

    miss = pd.read_csv(args.missing)
    miss["tk"] = miss["track"].map(bf_track_key)
    miss["ok"] = miss["off_time"].map(normalise_off_time_key)
    want = set(zip(miss["race_date"], miss["tk"], miss["ok"]))
    dates = sorted(miss["race_date"].unique())
    print(f"{len(miss)} missing races across {len(dates)} dates")
    if not args.commit:
        print("dry-run — pass --commit to scrape + insert")
        for d in dates:
            sub = miss[miss["race_date"] == d]
            print(f"  {d}: {len(sub)} races — {sorted(sub['track'].unique())}")
        return

    scraper = SportingLifeScraper()
    all_rows: list[dict] = []
    failures: list[dict] = []
    total_inserted = 0

    for d in dates:
        try:
            urls = scraper.get_results_urls(d)
        except Exception as exc:
            print(f"  {d}: listing failed: {exc!r}")
            continue
        targets = [
            u for u in urls
            if (d, bf_track_key(u["track"]), normalise_off_time_key(u["time"])) in want
        ]
        print(f"  {d}: {len(targets)} target races")
        day_rows: list[dict] = []
        for u in targets:
            try:
                rows = scraper.scrape_race_result(u["url"], d)
            except Exception as exc:
                rows = []
                print(f"     ERROR {u['track']} {u['time']}: {exc!r}")
            if rows:
                day_rows.extend(rows)
            else:
                failures.append({"race_date": d, "track": u["track"],
                                 "off_time": u["time"], "url": u["url"]})
        if day_rows:
            df = pd.DataFrame(day_rows)
            ins = insert_results(df)
            total_inserted += ins
            all_rows.extend(day_rows)
            print(f"     scraped {len(day_rows)} runners, inserted {ins} new")

    print(f"\nDONE: inserted {total_inserted} new runner rows from "
          f"{len(all_rows)} scraped")
    if failures:
        fp = "data/processed/backfill_failures.csv"
        pd.DataFrame(failures).to_csv(fp, index=False)
        print(f"{len(failures)} races still failed -> {fp}")


if __name__ == "__main__":
    main()
