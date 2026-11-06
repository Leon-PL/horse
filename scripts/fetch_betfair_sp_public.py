"""Download and build Betfair's free, public daily BSP files.

See ``src/ingestion/betfair_public.py`` for the source and its traps.

Usage::

    python scripts/fetch_betfair_sp_public.py --download --start 2020-09-14 --end 2026-10-12
    python scripts/fetch_betfair_sp_public.py --build
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.betfair_public import build, download  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="2020-09-14", help="first RACE date")
    ap.add_argument("--end", default=None, help="last RACE date (default: yesterday)")
    ap.add_argument("--market", default="win", choices=("win", "place"))
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    if args.download:
        end = (date.fromisoformat(args.end) if args.end
               else (datetime.now() - timedelta(days=1)).date())
        download(date.fromisoformat(args.start), end, market=args.market, delay=args.delay)
    if args.build:
        build(args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
