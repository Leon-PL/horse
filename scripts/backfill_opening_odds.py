"""Backfill opening_odds on existing results by re-fetching result pages.

insert_results uses INSERT OR IGNORE, so re-scraping never updates
existing rows — this script UPDATEs opening_odds in place instead.
Opening vs SP odds is the raw material for closing-line-value analysis.

Usage:
    python scripts/backfill_opening_odds.py [days_back]   # default 120

Walks dates newest-first, skipping dates whose rows already have
opening_odds, so it is safe to stop and restart.
"""
import logging
import sqlite3
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backfill_opening_odds")

import numpy as np

from src.database import DB_PATH, init_db
from src.data_scraper import SportingLifeScraper

days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 120

init_db()
conn = sqlite3.connect(DB_PATH)
dates = [
    row[0] for row in conn.execute(
        """
        SELECT race_date FROM results
        WHERE race_date >= date('now', ?)
        GROUP BY race_date
        HAVING SUM(CASE WHEN opening_odds IS NOT NULL THEN 1 ELSE 0 END) = 0
        ORDER BY race_date DESC
        """,
        (f"-{days_back} days",),
    )
]
logger.info("%d dates need opening odds (last %d days)", len(dates), days_back)

scraper = SportingLifeScraper()
total_updated = 0
for n, date_str in enumerate(dates, 1):
    try:
        urls = scraper.get_results_urls(date_str)
    except Exception:
        logger.exception("listing failed for %s", date_str)
        continue
    day_updated = 0
    for info in urls:
        try:
            runners = scraper.scrape_race_result(info["url"], date_str)
        except Exception:
            logger.exception("result fetch failed: %s", info.get("url"))
            continue
        for row in runners:
            oo = row.get("opening_odds")
            if oo is None or (isinstance(oo, float) and np.isnan(oo)):
                continue
            cur = conn.execute(
                "UPDATE results SET opening_odds = ? "
                "WHERE race_id = ? AND horse_name = ? AND opening_odds IS NULL",
                (float(oo), row["race_id"], row["horse_name"]),
            )
            day_updated += cur.rowcount
    conn.commit()
    total_updated += day_updated
    logger.info("[%d/%d] %s: %d rows updated (running total %d)",
                n, len(dates), date_str, day_updated, total_updated)

conn.close()
logger.info("Done: %d rows updated", total_updated)
