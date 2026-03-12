"""
Data Collector Module (Unified)
===============================
Unified interface for collecting horse racing data.

Sources:
1. Historical database (SQLite) — incremental scrape + persistent store  [RECOMMENDED]
2. Web scraper (Sporting Life) — UK & Ireland results + racecards
3. Sample data generator (for testing/offline fallback)
"""

# --- Historical Database ---
from src.database import (
    sync_database,
    load_from_database,
    db_stats,
    insert_results,
)

# --- Web Scraper (primary real-data source) ---
from src.data_scraper import (
    scrape_results,
    scrape_todays_racecards,
    scrape_todays_results,
    collect_scraped_data,
    get_scraped_racecards,
    load_cached_racecards,
    save_racecards_cache,
)

# --- Sample / synthetic data ---
from src.data_collector_sample import (
    generate_sample_data,
    HORSE_NAMES,
    JOCKEY_NAMES,
    TRAINER_NAMES,
    TRACKS,
    GOING_CONDITIONS,
    RACE_CLASSES,
    RACE_TYPES,
    DISTANCE_FURLONGS,
)


def collect_data(
    source: str = "database",
    num_races: int = 1500,
    days_back: int = 90,
):
    """
    Main entry point for data collection.

    Args:
        source: "database" (incremental, recommended), "scrape" (full re-scrape), or "sample" (synthetic)
        num_races: Number of races for sample data
        days_back: Days of history for scraping / database

    Returns:
        DataFrame with race data
    """
    if source == "database":
        return sync_database(days_back=days_back)
    elif source == "scrape":
        df = collect_scraped_data(days_back=days_back)
        # Also store in the database for next time
        if df is not None and not df.empty:
            insert_results(df)
        return df
    else:
        return generate_sample_data(num_races=num_races, save=True)
