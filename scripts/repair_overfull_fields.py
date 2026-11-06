"""
Remove non-runners stored as runners (fields with more rows than runners).
=========================================================================

A field is *overfull* when ``results`` holds more rows for a race than its
``num_runners`` (Sporting Life's ``ride_count``). Every case found on
2026-11-01 was a non-runner or a withdrawn horse stored as a starter, e.g.
Yarmouth 2026-05-13 with 13 rows for 7 runners. They inflate field sizes and
add phantom runs to horse histories.

Each overfull race is re-scraped. A field that passes the sync validator
replaces the stored one through ``_replace_race_results`` (the sync path,
which also deletes rows the source no longer lists). A field that cannot be
validated (two different horses with the same name, which the
``(race_id, horse_name)`` key cannot hold) only loses the rows the page marks
``NONRUNNER`` or ``WITHDRAWN``. Every changed row is first written to
``data/backups/overfull_fields_<timestamp>.csv``. The whole-history CSV
snapshot that feature builds read is then re-exported from the database.

Usage:
    python scripts/repair_overfull_fields.py            # dry run
    python scripts/repair_overfull_fields.py --apply    # back up and repair
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.ingestion.sporting_life import SportingLifeScraper, _slugify  # noqa: E402
from src.database import (  # noqa: E402
    DB_PATH, _get_conn, _replace_race_results, _validate_race_results, publish_after_repair,
)

NOT_STARTED = {"NONRUNNER", "WITHDRAWN"}


def overfull_races() -> pd.DataFrame:
    with _get_conn() as conn:
        return pd.read_sql("""
            SELECT race_id, race_date, track, race_name, MAX(num_runners) AS num_runners, COUNT(*) AS rows
            FROM results GROUP BY race_id HAVING COUNT(*) > MAX(num_runners) ORDER BY race_date
        """, conn)


def _url(race) -> str:
    return (f"https://www.sportinglife.com/racing/results/{race.race_date}/{_slugify(race.track)}/"
            f"{race.race_id}/{_slugify(race.race_name)}")


def _not_started_ids(scraper: SportingLifeScraper, url: str) -> set[str]:
    data = scraper._fetch_next_data(url) or {}
    rides = (((data.get("props") or {}).get("pageProps") or {}).get("race") or {}).get("rides") or []
    return {str(r["horse"]["horse_reference"]["id"]) for r in rides
            if re.sub(r"[^A-Z]", "", str(r.get("ride_status", "")).upper()) in NOT_STARTED}


def plan(scraper: SportingLifeScraper) -> list[dict]:
    jobs = []
    for race in overfull_races().itertuples():
        url = _url(race)
        fresh = pd.DataFrame(scraper.scrape_race_result(url, race.race_date))
        try:
            _validate_race_results(fresh, race.race_date, str(race.race_id))
            jobs.append({"race": race, "mode": "replace", "fresh": fresh})
        except ValueError as exc:
            ids = _not_started_ids(scraper, url)
            jobs.append({"race": race, "mode": "delete_not_started", "ids": ids, "reason": str(exc)})
    return jobs


def apply(jobs: list[dict]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(config.DATA_DIR, "backups", f"overfull_fields_{stamp}.csv")
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    with _get_conn() as conn:
        ids = [str(j["race"].race_id) for j in jobs]
        before = pd.read_sql(f"SELECT * FROM results WHERE race_id IN ({','.join('?' * len(ids))})", conn, params=ids)
        before.to_csv(backup, index=False)
        for job in jobs:
            race_id = str(job["race"].race_id)
            if job["mode"] == "replace":
                _replace_race_results(conn, job["fresh"])
            else:
                conn.executemany("DELETE FROM results WHERE race_id = ? AND horse_id = ?",
                                 [(race_id, horse_id) for horse_id in sorted(job["ids"])])
    publish_after_repair(DB_PATH)
    return backup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="back up and repair (default: dry run)")
    args = ap.parse_args()
    jobs = plan(SportingLifeScraper())
    if not jobs:
        print("No overfull fields.")
        return
    for job in jobs:
        race = job["race"]
        detail = (f"replace with {len(job['fresh'])} validated runners" if job["mode"] == "replace"
                  else f"delete {len(job['ids'])} non-started rows ({job['reason']})")
        print(f"{race.race_date} {race.track} {race.race_id}: {race.rows} rows for {race.num_runners} runners -> {detail}")
    if args.apply:
        print(f"Backed up {len(jobs)} fields to {apply(jobs)}")
        left = overfull_races()
        print(f"Overfull fields remaining: {len(left)}")
    else:
        print("Dry run; pass --apply to repair.")


if __name__ == "__main__":
    main()
