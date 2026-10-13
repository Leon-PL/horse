"""Collect BHA going reports: stalls position, rail movements, GoingStick.

ACTION_PLAN 5a-bis; the collection logic is in ``src/bha_reports.py`` and the
join onto results in ``src/course_reports.py`` (ledger 17). This is the
backfill CLI; the daily sync keeps recent days current by itself.

Usage::

    python scripts/fetch_bha_going.py --download --start 2020-09-01 --end 2026-10-12
    python scripts/fetch_bha_going.py --download --going      # also per-fixture reports
    python scripts/fetch_bha_going.py --download --races --delay 3   # rail-adjusted distances
    python scripts/fetch_bha_going.py --refresh 2026-10-13 2026-10-14
    python scripts/fetch_bha_going.py --build

The API enforces a long-window quota (HTTP 429 for every request for over an
hour once tripped); a 3 s delay completes a full backfill, ~9,300 fixtures in
about six hours.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import bha_reports  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--going", action="store_true", help="also fetch per-fixture going reports")
    ap.add_argument("--races", action="store_true",
                    help="also fetch per-fixture race lists (rail-adjusted distances)")
    ap.add_argument("--refresh", nargs="+", metavar="DATE",
                    help="re-list and re-fetch these dates, then rebuild")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="2020-09-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    if args.download:
        bha_reports.download(date.fromisoformat(args.start), date.fromisoformat(args.end),
                             with_going=args.going, with_races=args.races, delay=args.delay)
    if args.refresh:
        bha_reports.refresh(args.refresh, delay=max(args.delay, 1.0))
    if args.build:
        bha_reports.build()
        bha_reports.build_races()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
