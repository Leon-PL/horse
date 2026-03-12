"""
Historical Database Module
==========================
SQLite-backed persistent storage for scraped race results.

Instead of re-scraping months of data every time you retrain, this module
stores results in a local ``data/races.db`` file and only fetches *new*
days that are not yet in the database.

Usage::

    from src.database import sync_database, load_from_database, db_stats

    # Scrape only the days we're missing (incremental)
    sync_database(days_back=90)

    # Load everything (or a window) as a DataFrame
    df = load_from_database()                # all records
    df = load_from_database(days_back=30)    # last 30 days only

    # Quick summary
    print(db_stats())
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(config.DATA_DIR, "races.db")

# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    race_id           TEXT    NOT NULL,
    race_date         TEXT    NOT NULL,
    off_time          TEXT,
    track             TEXT,
    region            TEXT,
    race_name         TEXT,
    race_class        TEXT,
    race_type         TEXT,
    distance_furlongs REAL,
    going             TEXT,
    prize_money       REAL,
    num_runners       INTEGER,
    horse_name        TEXT    NOT NULL,
    horse_id          TEXT,
    jockey            TEXT,
    trainer           TEXT,
    age               INTEGER,
    sex               TEXT,
    headgear          TEXT,
    weight_lbs        REAL,
    draw              INTEGER,
    form              TEXT,
    days_since_last_run INTEGER,
    odds              REAL,
    official_rating   INTEGER,
    finish_position   INTEGER,
    won               INTEGER,
    lengths_behind    REAL,
    horse_runs        INTEGER,
    horse_wins        INTEGER,
    horse_places      INTEGER,
    surface           TEXT,
    handicap          INTEGER,
    sire              TEXT,
    dam               TEXT,
    damsire           TEXT,
    weather_temp_max  REAL,
    weather_temp_min  REAL,
    weather_precip_mm REAL,
    weather_wind_kmh  REAL,
    weather_precip_prev3 REAL,
    scraped_at        TEXT    DEFAULT (datetime('now')),

    UNIQUE(race_id, horse_name)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_results_date ON results(race_date);
"""


# -----------------------------------------------------------------------
# Connection helpers
# -----------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a connection to the database (creates it if needed)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Create the results table and indexes if they don't already exist."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        # Migrate: add columns that may not exist in older databases
        for col, col_type in [
            ("sire", "TEXT"),
            ("dam", "TEXT"),
            ("damsire", "TEXT"),
            ("weather_temp_max", "REAL"),
            ("weather_temp_min", "REAL"),
            ("weather_precip_mm", "REAL"),
            ("weather_wind_kmh", "REAL"),
            ("weather_precip_prev3", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # column already exists
    logger.info(f"Database ready at {DB_PATH}")


# -----------------------------------------------------------------------
# Insert / upsert
# -----------------------------------------------------------------------

_COLUMNS = [
    "race_id", "race_date", "off_time", "track", "region",
    "race_name", "race_class", "race_type", "distance_furlongs",
    "going", "prize_money", "num_runners", "horse_name", "horse_id",
    "jockey", "trainer", "age", "sex", "headgear", "weight_lbs",
    "draw", "form", "days_since_last_run", "odds", "official_rating",
    "finish_position", "won", "lengths_behind", "horse_runs",
    "horse_wins", "horse_places", "surface", "handicap",
    "sire", "dam", "damsire",
    "weather_temp_max", "weather_temp_min", "weather_precip_mm",
    "weather_wind_kmh", "weather_precip_prev3",
]


def insert_results(df: pd.DataFrame) -> int:
    """
    Insert a DataFrame of race results into the database.

    Rows whose ``(race_id, horse_name)`` already exist are silently
    skipped (``INSERT OR IGNORE``).

    Returns:
        Number of *new* rows inserted.
    """
    if df is None or df.empty:
        return 0

    init_db()

    # Keep only columns that exist in both the DF and the schema
    cols = [c for c in _COLUMNS if c in df.columns]
    subset = df[cols].copy()

    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO results ({col_names}) VALUES ({placeholders})"

    rows_before = _count_rows()

    with _get_conn() as conn:
        conn.executemany(sql, subset.values.tolist())

    rows_after = _count_rows()
    inserted = rows_after - rows_before
    logger.info(f"  Inserted {inserted} new rows ({rows_after} total in DB)")
    return inserted


# -----------------------------------------------------------------------
# Query helpers
# -----------------------------------------------------------------------

def _count_rows() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]


def get_dates_in_db() -> set[str]:
    """Return the set of distinct race_date strings already stored."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT race_date FROM results ORDER BY race_date"
        ).fetchall()
    return {r[0] for r in rows}


def get_latest_date() -> str | None:
    """Return the most recent race_date in the DB, or None if empty."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(race_date) FROM results"
        ).fetchone()
    return row[0] if row and row[0] else None


def load_from_database(days_back: int | None = None) -> pd.DataFrame:
    """
    Load results from the database as a DataFrame.

    Args:
        days_back: If given, only return the last *N* days.
                   If ``None``, return everything.

    Returns:
        DataFrame matching the same schema as ``scrape_results()``.
    """
    init_db()

    if days_back is not None:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = "SELECT * FROM results WHERE race_date >= ? ORDER BY race_date"
        params: tuple = (cutoff,)
    else:
        query = "SELECT * FROM results ORDER BY race_date"
        params = ()

    with _get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    # Drop the internal scraped_at column if present
    if "scraped_at" in df.columns:
        df = df.drop(columns=["scraped_at"])

    logger.info(f"Loaded {len(df)} records from database" +
                (f" (last {days_back} days)" if days_back else ""))
    return df


# -----------------------------------------------------------------------
# Sync — incremental scrape + store
# -----------------------------------------------------------------------

def sync_database(days_back: int = 90) -> pd.DataFrame:
    """
    Incrementally update the database.

    1. Determine which dates in the requested window are already stored.
    2. Scrape only the *missing* dates.
    3. Insert new results into the DB.
    4. Return the full dataset for the requested window.

    Args:
        days_back: How many days of history to ensure are in the DB.

    Returns:
        DataFrame with all results for the requested window.
    """
    from src.data_scraper import SportingLifeScraper

    init_db()

    existing_dates = get_dates_in_db()

    # Build list of dates we need
    today = datetime.now()
    all_dates = [
        (today - timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(days_back)
    ]
    missing_dates = [d for d in all_dates if d not in existing_dates]

    if not missing_dates:
        logger.info(
            f"✅ Database is up-to-date — all {days_back} days already stored "
            f"({len(existing_dates)} dates in DB)"
        )
        return load_from_database(days_back=days_back)

    logger.info(
        f"📦 Database has {len(existing_dates)} dates. "
        f"Need to scrape {len(missing_dates)} missing day(s)..."
    )

    # Scrape each missing date — collect all race URLs first, then fetch
    # individual race pages in parallel across all dates.
    scraper = SportingLifeScraper()
    new_runners: list[dict] = []

    # Phase 1: gather race URLs (one listing request per date — sequential)
    all_race_jobs: list[tuple[str, dict]] = []   # (date_str, link_info)
    for i, date_str in enumerate(sorted(missing_dates)):
        logger.info(
            f"  📅 [{i+1}/{len(missing_dates)}] Listing {date_str} …"
        )
        try:
            race_links = scraper.get_results_urls(date_str, uk_only=True)
        except Exception as e:
            logger.warning(f"    ⚠️ Failed to get listing for {date_str}: {e}")
            continue

        if not race_links:
            logger.info(f"    No races found for {date_str}")
            continue

        race_links = race_links[:50]  # cap per day
        for link_info in race_links:
            all_race_jobs.append((date_str, link_info))

    # Phase 2: fetch individual race pages in parallel
    if all_race_jobs:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data_scraper import MAX_WORKERS

        logger.info(
            f"  ⚡ Fetching {len(all_race_jobs)} race pages "
            f"({MAX_WORKERS} workers) …"
        )

        def _fetch_race(job: tuple[str, dict]) -> list[dict]:
            ds, info = job
            return scraper.scrape_race_result(info["url"], date_str=ds)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch_race, j): j for j in all_race_jobs}
            done = 0
            for future in as_completed(futures):
                done += 1
                ds, info = futures[future]
                try:
                    runners = future.result()
                    new_runners.extend(runners)
                    if done % 50 == 0 or done == len(all_race_jobs):
                        logger.info(
                            f"    [{done}/{len(all_race_jobs)}] races fetched"
                        )
                except Exception as e:
                    logger.warning(
                        f"    ⚠️ Failed {info.get('race_name', '?')}: {e}"
                    )

    if new_runners:
        new_df = pd.DataFrame(new_runners)
        inserted = insert_results(new_df)
        logger.info(f"🆕 Scraped {len(new_runners)} runners, inserted {inserted} new rows")

        # Fetch weather for the newly-scraped dates and write it into the DB
        if inserted > 0:
            try:
                from src.weather import backfill_weather_in_db
                scraped_dates = sorted(set(d for d, _ in all_race_jobs))
                backfill_weather_in_db(dates=scraped_dates)
            except Exception as e:
                logger.warning(f"⚠️ Weather backfill for new rows failed: {e}")
    else:
        logger.info("  No new runners found for missing dates")

    # Save a CSV copy for the pipeline
    full_df = load_from_database(days_back=days_back)
    csv_path = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
    full_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved CSV snapshot → {csv_path}")

    return full_df


# -----------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------

def db_stats() -> dict:
    """Return a summary of the database contents."""
    init_db()
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        dates = conn.execute(
            "SELECT COUNT(DISTINCT race_date) FROM results"
        ).fetchone()[0]
        tracks = conn.execute(
            "SELECT COUNT(DISTINCT track) FROM results"
        ).fetchone()[0]
        races = conn.execute(
            "SELECT COUNT(DISTINCT race_id) FROM results"
        ).fetchone()[0]
        min_date = conn.execute(
            "SELECT MIN(race_date) FROM results"
        ).fetchone()[0]
        max_date = conn.execute(
            "SELECT MAX(race_date) FROM results"
        ).fetchone()[0]

    return {
        "total_runners": total,
        "total_dates": dates,
        "total_tracks": tracks,
        "total_races": races,
        "earliest_date": min_date or "—",
        "latest_date": max_date or "—",
        "db_path": DB_PATH,
        "db_size_mb": round(
            os.path.getsize(DB_PATH) / (1024 * 1024), 2
        ) if os.path.exists(DB_PATH) else 0,
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage the historical race database"
    )
    parser.add_argument(
        "--sync",
        type=int,
        metavar="DAYS",
        help="Sync the database (scrape missing days within DAYS window)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print database statistics",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export the full database to a CSV file",
    )
    args = parser.parse_args()

    if args.stats or (not args.sync and not args.export):
        s = db_stats()
        print("\n📦  Race Database Statistics")
        print("=" * 40)
        for k, v in s.items():
            print(f"  {k:20s}: {v}")
        print()

    if args.sync:
        print(f"\n🔄 Syncing database (last {args.sync} days)...")
        df = sync_database(days_back=args.sync)
        print(f"  Done — {len(df)} records available\n")

    if args.export:
        df = load_from_database()
        df.to_csv(args.export, index=False)
        print(f"  Exported {len(df)} records to {args.export}")
