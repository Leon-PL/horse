"""
Weather Data Module
===================
Fetches historical weather for racecourse locations via the free
Open-Meteo Archive API and caches results in a local SQLite database.

API docs: https://open-meteo.com/en/docs/historical-weather-api

Features returned per (track, date):
    - temp_max_c         : max temperature (°C)
    - temp_min_c         : min temperature (°C)
    - precipitation_mm   : total precipitation (mm)
    - wind_max_kmh       : max wind speed at 10 m (km/h)
    - precip_prev3_mm    : total precipitation in the 3 days before (excl. race day)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.track_config import get_track_config

logger = logging.getLogger(__name__)

# ── paths ───────────────────────────────────────────────────────
_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_WEATHER_DB = _DB_DIR / "weather_cache.db"
_RACES_DB = _DB_DIR / "races.db"

# ── Open-Meteo endpoints ────────────────────────────────────────
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_DAILY_VARS = (
    "temperature_2m_max,"
    "temperature_2m_min,"
    "precipitation_sum,"
    "wind_speed_10m_max"
)


# ═══════════════════════════════════════════════════════════════
# SQLite cache
# ═══════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_WEATHER_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            lat          REAL    NOT NULL,
            lon          REAL    NOT NULL,
            date         TEXT    NOT NULL,
            temp_max_c   REAL,
            temp_min_c   REAL,
            precip_mm    REAL,
            wind_max_kmh REAL,
            PRIMARY KEY (lat, lon, date)
        )
    """)
    conn.commit()
    return conn


def _load_cached(
    conn: sqlite3.Connection,
    lat: float,
    lon: float,
    dates: list[str],
) -> dict[str, dict]:
    """Return {date_str: {col: val}} for rows already in the cache."""
    if not dates:
        return {}
    placeholders = ",".join("?" for _ in dates)
    rows = conn.execute(
        f"SELECT date, temp_max_c, temp_min_c, precip_mm, wind_max_kmh "
        f"FROM weather WHERE lat=? AND lon=? AND date IN ({placeholders})",
        [lat, lon, *dates],
    ).fetchall()
    return {
        r[0]: {
            "temp_max_c": r[1],
            "temp_min_c": r[2],
            "precip_mm": r[3],
            "wind_max_kmh": r[4],
        }
        for r in rows
    }


def _save_to_cache(
    conn: sqlite3.Connection,
    lat: float,
    lon: float,
    records: list[dict],
) -> None:
    """Insert (or replace) weather records into the cache."""
    if not records:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO weather "
        "(lat, lon, date, temp_max_c, temp_min_c, precip_mm, wind_max_kmh) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (lat, lon, r["date"], r["temp_max_c"], r["temp_min_c"],
             r["precip_mm"], r["wind_max_kmh"])
            for r in records
        ],
    )
    conn.commit()


# ═══════════════════════════════════════════════════════════════
# Open-Meteo fetcher
# ═══════════════════════════════════════════════════════════════

def _fetch_open_meteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    is_forecast: bool = False,
) -> dict[str, dict]:
    """
    Call Open-Meteo and return {date_str: {col: val}}.
    Batches up to ~1 year per call (API limit).
    """
    url = _FORECAST_URL if is_forecast else _ARCHIVE_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": _DAILY_VARS,
        "timezone": "Europe/London",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo request failed (%s): %s", url, exc)
        return {}

    daily = data.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return {}

    t_max = daily.get("temperature_2m_max", [None] * len(times))
    t_min = daily.get("temperature_2m_min", [None] * len(times))
    precip = daily.get("precipitation_sum", [None] * len(times))
    wind = daily.get("wind_speed_10m_max", [None] * len(times))

    return {
        d: {
            "date": d,
            "temp_max_c": t_max[i],
            "temp_min_c": t_min[i],
            "precip_mm": precip[i],
            "wind_max_kmh": wind[i],
        }
        for i, d in enumerate(times)
    }


def _fetch_location_weather(
    lat: float,
    lon: float,
    dates: list[str],
    conn: sqlite3.Connection,
) -> dict[str, dict]:
    """
    Return weather for a single location across many dates,
    using the cache first and filling gaps from the API.
    Requests are grouped into contiguous date-range chunks
    to minimise API calls (Open-Meteo allows multi-year ranges).
    """
    cached = _load_cached(conn, lat, lon, dates)
    missing = sorted(set(dates) - set(cached))
    if not missing:
        return cached

    # Build contiguous date-range chunks (with 3-day buffer for precip_prev3)
    result = dict(cached)

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Build yearly chunks from the missing dates (one API call per year
    # is far more efficient than dozens of small contiguous ranges).
    # We extend 3 days before the first date for precip_prev3 look-back.
    ranges = _yearly_ranges(missing, buffer_days=3)

    for start, end in ranges:
        # Decide archive vs forecast
        is_forecast = end >= today_str
        if is_forecast:
            archive_end = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
            if start < archive_end:
                fetched = _fetch_open_meteo(lat, lon, start, archive_end, is_forecast=False)
                _save_to_cache(conn, lat, lon, list(fetched.values()))
                result.update(fetched)
                time.sleep(0.2)
            fc_start = max(start, archive_end)
            fetched = _fetch_open_meteo(lat, lon, fc_start, end, is_forecast=True)
            _save_to_cache(conn, lat, lon, list(fetched.values()))
            result.update(fetched)
        else:
            fetched = _fetch_open_meteo(lat, lon, start, end, is_forecast=False)
            _save_to_cache(conn, lat, lon, list(fetched.values()))
            result.update(fetched)
        time.sleep(0.2)  # polite rate-limiting

    return result


def _yearly_ranges(
    dates_sorted: list[str],
    buffer_days: int = 3,
) -> list[tuple[str, str]]:
    """
    Group sorted ISO dates into per-year ranges.

    Each range covers [min_date - buffer_days, max_date] within a
    calendar year.  This produces at most ~5 API calls for 5 years
    of data instead of hundreds of tiny ranges.
    """
    if not dates_sorted:
        return []

    # Group by year
    by_year: dict[int, list[str]] = {}
    for d in dates_sorted:
        y = int(d[:4])
        by_year.setdefault(y, []).append(d)

    ranges: list[tuple[str, str]] = []
    for year in sorted(by_year):
        ds = by_year[year]
        first = datetime.strptime(min(ds), "%Y-%m-%d") - timedelta(days=buffer_days)
        last = max(ds)
        ranges.append((first.strftime("%Y-%m-%d"), last))

    return ranges


# ═══════════════════════════════════════════════════════════════
# Public API – DataFrame-level enrichment
# ═══════════════════════════════════════════════════════════════

def get_weather_for_races(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach weather columns to a race DataFrame.

    **Fast path**: If the DataFrame already contains weather columns
    (populated from the database), only fetch for rows where they're NULL.

    Expects columns: ``track`` (or ``track_name``) and ``race_date``.
    Returns the original DataFrame with new columns:
        weather_temp_max, weather_temp_min,
        weather_precip_mm, weather_wind_kmh,
        weather_precip_prev3
    """
    track_col = "track" if "track" in df.columns else "track_name"
    if track_col not in df.columns or "race_date" not in df.columns:
        logger.warning("Weather: missing track/race_date columns — skipping")
        return df

    # Check if weather columns already exist and are fully populated
    _W_COLS = ["weather_temp_max", "weather_temp_min", "weather_precip_mm",
               "weather_wind_kmh", "weather_precip_prev3"]
    if all(c in df.columns for c in _W_COLS):
        n_null = df["weather_temp_max"].isna().sum()
        if n_null == 0:
            logger.info("Weather: all rows already have weather from DB — skipping API")
            return df
        # Partial coverage: only fetch for the NULL rows
        logger.info(f"Weather: {n_null}/{len(df)} rows need weather — fetching gaps")
        mask_need = df["weather_temp_max"].isna()
        if mask_need.sum() > 0:
            df_need = df.loc[mask_need].copy()
            # Strip existing weather columns so the fetch fills them
            df_need = df_need.drop(columns=_W_COLS, errors="ignore")
            df_filled = _fetch_weather_for_df(df_need, track_col)
            for c in _W_COLS:
                if c in df_filled.columns:
                    df.loc[mask_need, c] = df_filled[c].values
        # Fill any remaining NaNs with defaults
        defaults = {
            "weather_temp_max": 14.0, "weather_temp_min": 7.0,
            "weather_precip_mm": 1.5, "weather_wind_kmh": 18.0,
            "weather_precip_prev3": 4.5,
        }
        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    # Full fetch path (no weather columns exist yet)
    return _fetch_weather_for_df(df, track_col)


def _fetch_weather_for_df(df: pd.DataFrame, track_col: str) -> pd.DataFrame:
    """Internal: fetch weather from API/cache and merge onto DataFrame."""
    # Normalise race_date to str (YYYY-MM-DD)
    dates_raw = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.copy()
    df["_w_date"] = dates_raw.dt.strftime("%Y-%m-%d")

    # Group unique (lat, lon, date) triples
    track_dates: dict[tuple[float, float], set[str]] = {}
    track_coords: dict[str, tuple[float, float]] = {}

    for track_name in df[track_col].unique():
        cfg = get_track_config(str(track_name))
        lat, lon = cfg.get("lat", 52.0), cfg.get("lon", -1.5)
        track_coords[track_name] = (lat, lon)
        key = (round(lat, 4), round(lon, 4))
        race_dates = df.loc[df[track_col] == track_name, "_w_date"].dropna().unique().tolist()
        track_dates.setdefault(key, set()).update(race_dates)

    # Fetch all weather (cached + API)
    conn = _get_conn()
    all_weather: dict[tuple[float, float], dict[str, dict]] = {}

    total_locs = len(track_dates)
    for idx, ((lat, lon), date_set) in enumerate(track_dates.items(), 1):
        date_list = sorted(date_set)
        logger.info(
            "Weather: fetching loc %d/%d (%.2f, %.2f) — %d dates",
            idx, total_locs, lat, lon, len(date_list),
        )
        loc_weather = _fetch_location_weather(lat, lon, date_list, conn)
        all_weather[(round(lat, 4), round(lon, 4))] = loc_weather

    conn.close()

    # Vectorised approach: build weather as a mapping keyed on (track, date)
    w_rows: list[dict] = []
    for track_name, (lat, lon) in track_coords.items():
        key = (round(lat, 4), round(lon, 4))
        loc_data = all_weather.get(key, {})
        track_dates_list = df.loc[df[track_col] == track_name, "_w_date"].dropna().unique()
        for date_str in track_dates_list:
            w = loc_data.get(date_str, {})
            # 3-day prior precip
            precip_prev3 = 0.0
            try:
                race_dt = datetime.strptime(date_str, "%Y-%m-%d")
                for d_offset in range(1, 4):
                    prev_d = (race_dt - timedelta(days=d_offset)).strftime("%Y-%m-%d")
                    pw = loc_data.get(prev_d, {})
                    precip_prev3 += pw.get("precip_mm", 0.0) or 0.0
            except (ValueError, TypeError):
                pass

            w_rows.append({
                "_w_track": track_name,
                "_w_date": date_str,
                "weather_temp_max": w.get("temp_max_c"),
                "weather_temp_min": w.get("temp_min_c"),
                "weather_precip_mm": w.get("precip_mm"),
                "weather_wind_kmh": w.get("wind_max_kmh"),
                "weather_precip_prev3": precip_prev3,
            })

    if w_rows:
        w_df = pd.DataFrame(w_rows)
        df = df.merge(
            w_df,
            left_on=[track_col, "_w_date"],
            right_on=["_w_track", "_w_date"],
            how="left",
        )
        df.drop(columns=["_w_track"], inplace=True, errors="ignore")
    else:
        for c in ("weather_temp_max", "weather_temp_min",
                   "weather_precip_mm", "weather_wind_kmh",
                   "weather_precip_prev3"):
            df[c] = np.nan

    df.drop(columns=["_w_date"], inplace=True, errors="ignore")

    # Fill missing weather with sensible UK defaults
    defaults = {
        "weather_temp_max": 14.0,
        "weather_temp_min": 7.0,
        "weather_precip_mm": 1.5,
        "weather_wind_kmh": 18.0,
        "weather_precip_prev3": 4.5,
    }
    for col, val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


# ═══════════════════════════════════════════════════════════════
# Backfill weather into the main races database
# ═══════════════════════════════════════════════════════════════

def backfill_weather_in_db(
    dates: list[str] | None = None,
    batch_size: int = 200,
) -> int:
    """
    Find rows in ``results`` where weather columns are NULL and fill them.

    Works in batches of unique (track, date) pairs to keep API calls
    efficient.  Weather is first pulled from / saved to the cache DB,
    then written into the main ``races.db``.

    Args:
        dates: Optional list of ISO date strings to limit the backfill to.
               If None, processes ALL rows with NULL weather.
        batch_size: How many (track, date) pairs to process per round.

    Returns:
        Number of rows updated.
    """
    import sqlite3 as _sqlite3

    races_conn = _sqlite3.connect(str(_RACES_DB))
    cache_conn = _get_conn()

    # Find distinct (track, date) pairs that need weather
    if dates:
        placeholders = ",".join("?" for _ in dates)
        query = (
            f"SELECT DISTINCT track, race_date FROM results "
            f"WHERE weather_temp_max IS NULL "
            f"AND race_date IN ({placeholders}) "
            f"ORDER BY race_date"
        )
        pairs = races_conn.execute(query, dates).fetchall()
    else:
        pairs = races_conn.execute(
            "SELECT DISTINCT track, race_date FROM results "
            "WHERE weather_temp_max IS NULL "
            "ORDER BY race_date"
        ).fetchall()

    if not pairs:
        logger.info("Weather backfill: all rows already have weather data ✅")
        races_conn.close()
        cache_conn.close()
        return 0

    logger.info(f"🌦️  Weather backfill: {len(pairs)} (track, date) pairs to fill")

    # Group by location to batch API calls
    loc_dates: dict[tuple[float, float, str], list[tuple[str, str]]] = {}
    for track, date_str in pairs:
        cfg = get_track_config(str(track))
        lat, lon = cfg.get("lat", 52.0), cfg.get("lon", -1.5)
        key = (round(lat, 4), round(lon, 4))
        loc_dates.setdefault(key, []).append((track, date_str))

    total_updated = 0

    for loc_idx, ((lat, lon), track_date_list) in enumerate(loc_dates.items(), 1):
        # Unique dates for this location
        unique_dates = sorted(set(d for _, d in track_date_list))

        logger.info(
            "  📍 Location %d/%d (%.2f, %.2f) — %d dates",
            loc_idx, len(loc_dates), lat, lon, len(unique_dates),
        )

        # Fetch weather (cache-first)
        loc_weather = _fetch_location_weather(lat, lon, unique_dates, cache_conn)

        # Compute precip_prev3 for each date
        date_weather: dict[str, dict] = {}
        for date_str in unique_dates:
            w = loc_weather.get(date_str, {})
            precip_prev3 = 0.0
            try:
                race_dt = datetime.strptime(date_str, "%Y-%m-%d")
                for d_offset in range(1, 4):
                    prev_d = (race_dt - timedelta(days=d_offset)).strftime("%Y-%m-%d")
                    pw = loc_weather.get(prev_d, {})
                    precip_prev3 += pw.get("precip_mm", 0.0) or 0.0
            except (ValueError, TypeError):
                pass
            date_weather[date_str] = {
                "temp_max": w.get("temp_max_c"),
                "temp_min": w.get("temp_min_c"),
                "precip_mm": w.get("precip_mm"),
                "wind_kmh": w.get("wind_max_kmh"),
                "precip_prev3": precip_prev3,
            }

        # Write into races.db
        for track, date_str in track_date_list:
            wdata = date_weather.get(date_str, {})
            if wdata.get("temp_max") is None:
                continue  # no weather data available
            races_conn.execute(
                "UPDATE results SET "
                "weather_temp_max = ?, weather_temp_min = ?, "
                "weather_precip_mm = ?, weather_wind_kmh = ?, "
                "weather_precip_prev3 = ? "
                "WHERE track = ? AND race_date = ? "
                "AND weather_temp_max IS NULL",
                (
                    wdata["temp_max"], wdata["temp_min"],
                    wdata["precip_mm"], wdata["wind_kmh"],
                    wdata["precip_prev3"],
                    track, date_str,
                ),
            )
            total_updated += races_conn.execute(
                "SELECT changes()"
            ).fetchone()[0]

        races_conn.commit()

    races_conn.close()
    cache_conn.close()
    logger.info(f"🌦️  Weather backfill complete: {total_updated} rows updated")
    return total_updated


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Weather data management")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill weather for all DB rows that have NULL weather columns",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show how many rows have / lack weather data",
    )
    args = parser.parse_args()

    if args.stats or (not args.backfill):
        import sqlite3 as _sqlite3

        rconn = _sqlite3.connect(str(_RACES_DB))
        total = rconn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        filled = rconn.execute(
            "SELECT COUNT(*) FROM results WHERE weather_temp_max IS NOT NULL"
        ).fetchone()[0]
        rconn.close()
        print(f"\n🌦️  Weather coverage: {filled:,} / {total:,} rows "
              f"({filled/total*100:.1f}%)\n")

    if args.backfill:
        updated = backfill_weather_in_db()
        print(f"Done — {updated:,} rows updated")
