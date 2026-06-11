"""CLI: backfill horse pedigree from Sporting Life profiles.

Usage:
    python scripts/backfill_pedigree.py [limit]

Fetches pedigree for horses (most recent runners first) that have no
row in horse_pedigree yet. Rate-limited by config.REQUEST_DELAY inside
the scraper. Safe to stop and restart — progress is committed every 50.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.pedigree_backfill import backfill_pedigree

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
backfill_pedigree(limit=limit)
