"""
RacingTV RACEiQ Scraper
=======================
Fetches per-runner race performance metrics from the RacingTV API's
``iq-results`` comparison endpoint.

**Metrics available:**

* **Jump races** — ``jump_index``, ``lengths_gained_jumping``, ``fsp``,
  ``top_speed``, ``speed_lost``, ``entry_speed``
* **Flat races** — ``acceleration``, ``stride_length``, ``fsp``, ``top_speed``

Each metric includes ``value`` (float) and ``rank`` (int within race).

Usage::

    from src.rtv_scraper import fetch_rtv_metrics, backfill_rtv_metrics

    # Fetch metrics for a single race
    rows = fetch_rtv_metrics("2026-03-14", "kempton-park", "1408")

    # Backfill from existing race_results.csv
    backfill_rtv_metrics()
"""

import logging
import os
import re
import time
import json
from datetime import datetime, timedelta

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RTV_API_BASE = "https://api.racingtv.com/"
_RTV_API_KEY = "936fdbdf-1e46-4f82-9d18-d81b4803537d"
_RTV_HEADERS = {
    "API-KEY": _RTV_API_KEY,
    "User-Agent": config.USER_AGENT,
    "Accept": "application/json",
}

# Minimum gap between requests (seconds)
_REQUEST_DELAY = 0.05

RTV_CACHE_DIR = os.path.join(config.DATA_DIR, "rtv_cache")
RTV_CACHE_FILE = os.path.join(RTV_CACHE_DIR, "rtv_metrics.parquet")
RTV_NO_DATA_FILE = os.path.join(RTV_CACHE_DIR, "rtv_no_data_keys.json")
os.makedirs(RTV_CACHE_DIR, exist_ok=True)

# All metric columns we store (values are float, NaN when absent)
RTV_METRIC_COLS = [
    "rtv_jump_index",
    "rtv_lengths_gained_jumping",
    "rtv_fsp",
    "rtv_top_speed",
    "rtv_speed_lost",
    "rtv_entry_speed",
    "rtv_acceleration",
    "rtv_stride_length",
]

# Rank columns (integer, NaN when absent)
RTV_RANK_COLS = [c.replace("rtv_", "rtv_rank_") for c in RTV_METRIC_COLS]

# ---------------------------------------------------------------------------
# Track name → slug mapping
# ---------------------------------------------------------------------------

# Manual overrides for names that cannot be derived by simple slugification
_TRACK_SLUG_OVERRIDES: dict[str, str] = {
    "Ascot": "ascot",
    "Aintree": "aintree",
    "Bangor-on-Dee": "bangor-on-dee",
    "Bath": "bath",
    "Beverley": "beverley",
    "Brighton": "brighton",
    "Carlisle": "carlisle",
    "Cartmel": "cartmel",
    "Catterick": "catterick-bridge",
    "Chelmsford City": "chelmsford-city",
    "Cheltenham": "cheltenham",
    "Chepstow": "chepstow",
    "Chester": "chester",
    "Doncaster": "doncaster",
    "Epsom Downs": "epsom-downs",
    "Exeter": "exeter",
    "Fakenham": "fakenham",
    "Ffos Las": "ffos-las",
    "Fontwell": "fontwell-park",
    "Goodwood": "goodwood",
    "Hamilton": "hamilton-park",
    "Haydock": "haydock-park",
    "Hereford": "hereford",
    "Hexham": "hexham",
    "Huntingdon": "huntingdon",
    "Kelso": "kelso",
    "Kempton": "kempton-park",
    "Leicester": "leicester",
    "Lingfield": "lingfield-park",
    "Ludlow": "ludlow",
    "Market Rasen": "market-rasen",
    "Musselburgh": "musselburgh",
    "Newbury": "newbury",
    "Newcastle": "newcastle",
    "Newmarket": "newmarket",
    "Newton Abbot": "newton-abbot",
    "Nottingham": "nottingham",
    "Perth": "perth",
    "Plumpton": "plumpton",
    "Pontefract": "pontefract",
    "Redcar": "redcar",
    "Ripon": "ripon",
    "Royal Ascot": "ascot",
    "Salisbury": "salisbury",
    "Sandown": "sandown-park",
    "Sedgefield": "sedgefield",
    "Southwell": "southwell",
    "Stratford": "stratford-on-avon",
    "Taunton": "taunton",
    "Thirsk": "thirsk",
    "Uttoxeter": "uttoxeter",
    "Warwick": "warwick",
    "Wetherby": "wetherby",
    "Wincanton": "wincanton",
    "Windsor": "windsor",
    "Wolverhampton": "wolverhampton",
    "Worcester": "worcester",
    "Yarmouth": "yarmouth",
    "York": "york",
    # Irish tracks — RacingTV coverage is inconsistent; include anyway
    "Ballinrobe": "ballinrobe",
    "Bellewstown": "bellewstown",
    "Clonmel": "clonmel",
    "Cork": "cork",
    "Curragh": "curragh",
    "Down Royal": "down-royal",
    "Downpatrick": "downpatrick",
    "Dundalk": "dundalk",
    "Fairyhouse": "fairyhouse",
    "Galway": "galway",
    "Gowran Park": "gowran-park",
    "Kilbeggan": "kilbeggan",
    "Killarney": "killarney",
    "Laytown": "laytown",
    "Leopardstown": "leopardstown",
    "Limerick": "limerick",
    "Listowel": "listowel",
    "Naas": "naas",
    "Navan": "navan",
    "Punchestown": "punchestown",
    "Roscommon": "roscommon",
    "Sligo": "sligo",
    "Thurles": "thurles",
    "Tipperary": "tipperary",
    "Tramore": "tramore",
    "Wexford": "wexford",
}


def _track_to_slug(track_name: str) -> str:
    """Convert a track name to the RacingTV API slug."""
    if track_name in _TRACK_SLUG_OVERRIDES:
        return _TRACK_SLUG_OVERRIDES[track_name]
    # Fallback: lowercase, spaces/underscores → hyphens
    slug = re.sub(r"[^\w\s-]", "", track_name.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug


def _off_time_to_hhmm(off_time: str) -> str:
    """Convert off_time like '14:08' or '14:08:00' to '1408'."""
    parts = str(off_time).strip().split(":")
    if len(parts) >= 2:
        return f"{int(parts[0]):02d}{int(parts[1]):02d}"
    return str(off_time).replace(":", "")


def _normalise_off_time_key(off_time) -> str:
    """Normalise off-time values to a stable ``HH:MM`` merge key."""
    s = str(off_time).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""
    # 1408 -> 14:08
    if re.fullmatch(r"\d{4}", s):
        return f"{s[:2]}:{s[2:]}"
    # 14:08[:SS] -> 14:08
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    # Last-resort digit extraction (e.g. "1408 BST")
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 4:
        return f"{digits[:2]}:{digits[2:4]}"
    return s


def _normalise_track_key(track) -> str:
    """Normalise track names for robust joins."""
    return re.sub(r"\s+", " ", str(track).strip().lower())


def _normalise_horse_key(name) -> str:
    """Normalise horse names for robust joins."""
    return re.sub(r"\s+", " ", str(name).strip().title())


def _is_bst(date_str: str) -> bool:
    """Return True if *date_str* (YYYY-MM-DD) falls within UK BST.

    BST: last Sunday of March 01:00 UTC  →  last Sunday of October 01:00 UTC.
    We ignore the hour boundary and treat the whole day as BST/GMT.
    """
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except ValueError:
        return False
    year = dt.year
    # Last Sunday of March
    mar31 = datetime(year, 3, 31)
    bst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
    # Last Sunday of October
    oct31 = datetime(year, 10, 31)
    bst_end = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
    return bst_start <= dt < bst_end


def _add_one_hour(hhmm: str) -> str:
    """Add 1 hour to an HHMM string, e.g. '1640' → '1740'."""
    h = int(hhmm[:2])
    m = int(hhmm[2:])
    h += 1
    if h >= 24:
        h -= 24
    return f"{h:02d}{m:02d}"


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

_last_request_time = 0.0


def _rate_limited_get(url: str) -> requests.Response | None:
    """GET with rate limiting and error handling."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _REQUEST_DELAY:
        time.sleep(_REQUEST_DELAY - elapsed)
    try:
        r = requests.get(url, headers=_RTV_HEADERS, timeout=15)
        _last_request_time = time.time()
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None  # no data for this race
        logger.warning("RTV API %s returned %d", url, r.status_code)
        return None
    except requests.RequestException as exc:
        logger.warning("RTV API request failed: %s", exc)
        return None


def fetch_rtv_comparison(
    date: str, track_slug: str, time_hhmm: str,
) -> list[dict] | None:
    """Fetch RACEiQ comparison data for a single race.

    Returns a list of dicts (one per runner) with metric values, or None.
    """
    url = (
        f"{_RTV_API_BASE}racing/iq-results/{date}/{track_slug}"
        f"/{time_hhmm}/comparison"
    )
    resp = _rate_limited_get(url)
    if resp is None:
        return None

    data = resp.json()
    horses = data.get("horses", [])
    if not horses:
        return None

    rows = []
    for h in horses:
        row = {
            "horse_name": h.get("horse_name", ""),
            "finish_position": h.get("finish_position"),
        }
        metrics = h.get("metrics", {})
        for api_key, col_name in [
            ("jump_index", "rtv_jump_index"),
            ("lengths_gained_jumping", "rtv_lengths_gained_jumping"),
            ("fsp", "rtv_fsp"),
            ("top_speed", "rtv_top_speed"),
            ("speed_lost", "rtv_speed_lost"),
            ("entry_speed", "rtv_entry_speed"),
            ("acceleration", "rtv_acceleration"),
            ("stride_length", "rtv_stride_length"),
        ]:
            m = metrics.get(api_key)
            if m is not None:
                try:
                    row[col_name] = float(m["value"])
                except (ValueError, KeyError, TypeError):
                    pass
                rank_col = col_name.replace("rtv_", "rtv_rank_")
                if m.get("rank") is not None:
                    row[rank_col] = int(m["rank"])
        rows.append(row)
    return rows


def fetch_rtv_metrics(
    date: str, track_name: str, off_time: str,
) -> list[dict] | None:
    """High-level wrapper: convert friendly names and fetch.

    The raw data stores off_time in UTC, but the RTV API uses UK local
    time (GMT in winter, BST = UTC+1 in summer).  During BST we try the
    +1 h adjusted time first, then fall back to the original.
    """
    slug = _track_to_slug(track_name)
    hhmm = _off_time_to_hhmm(off_time)
    if _is_bst(date):
        bst_hhmm = _add_one_hour(hhmm)
        result = fetch_rtv_comparison(date, slug, bst_hhmm)
        if result is not None:
            return result
    return fetch_rtv_comparison(date, slug, hhmm)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def load_rtv_cache() -> pd.DataFrame:
    """Load cached RTV metrics (or return empty frame)."""
    if os.path.exists(RTV_CACHE_FILE):
        return pd.read_parquet(RTV_CACHE_FILE)
    return pd.DataFrame()


def save_rtv_cache(df: pd.DataFrame) -> None:
    """Save RTV metrics cache."""
    df.to_parquet(RTV_CACHE_FILE, index=False, engine="pyarrow")
    logger.info("Saved RTV cache: %d rows → %s", len(df), RTV_CACHE_FILE)


def _cache_key_cols() -> list[str]:
    return ["race_date", "track", "off_time", "horse_name"]


def _race_key(date_str: str, track: str, off_time: str) -> str:
    """Stable race-level key for cache skip-lists."""
    _d = pd.to_datetime(date_str, errors="coerce")
    _d_s = _d.strftime("%Y-%m-%d") if pd.notna(_d) else str(date_str)
    _t = _normalise_track_key(track)
    _o = _normalise_off_time_key(off_time)
    return f"{_d_s}|{_t}|{_o}"


def _load_no_data_race_keys() -> set[str]:
    """Load races previously confirmed as no-data from RTV endpoint."""
    if not os.path.exists(RTV_NO_DATA_FILE):
        return set()
    try:
        with open(RTV_NO_DATA_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        keys = payload.get("race_keys", []) if isinstance(payload, dict) else []
        return {str(k) for k in keys}
    except Exception:
        return set()


def _save_no_data_race_keys(keys: set[str]) -> None:
    """Persist known no-data race keys to disk."""
    payload = {
        "updated_at": datetime.now().isoformat(),
        "race_keys": sorted(keys),
    }
    with open(RTV_NO_DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill_rtv_metrics(
    max_races: int | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Scrape RTV comparison data for all races in race_results.csv.

    Appends to the cache incrementally so progress is not lost on
    interruption.  Call with ``skip_existing=True`` (default) to resume.

    Args:
        max_races: Cap the number of new races to fetch (for testing).
        skip_existing: If True, skip races already in the cache.

    Returns:
        The updated cache DataFrame.
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
    if not os.path.exists(raw_path):
        logger.error("race_results.csv not found at %s", raw_path)
        return pd.DataFrame()

    raw = pd.read_csv(raw_path, usecols=["race_date", "track", "off_time"])
    # Unique races
    races = (
        raw.drop_duplicates(subset=["race_date", "track", "off_time"])
        .sort_values("race_date", ascending=False)  # newest first
        .reset_index(drop=True)
    )
    logger.info("Total unique races in raw data: %d", len(races))

    cache = load_rtv_cache()

    if skip_existing and not cache.empty:
        existing = set(
            cache[["race_date", "track", "off_time"]]
            .drop_duplicates()
            .apply(tuple, axis=1)
        )
        races = races[
            ~races.apply(
                lambda r: (r["race_date"], r["track"], r["off_time"]) in existing,
                axis=1,
            )
        ].reset_index(drop=True)
        logger.info("Races remaining after skipping cached: %d", len(races))

    if max_races is not None:
        races = races.head(max_races)

    if races.empty:
        logger.info("No new races to fetch.")
        return cache

    new_rows: list[dict] = []
    fetched = 0
    batch_size = 200  # save every N races

    for idx, race in races.iterrows():
        date_str = str(race["race_date"])
        track = str(race["track"])
        off_time = str(race["off_time"])

        data = fetch_rtv_metrics(date_str, track, off_time)
        if data:
            for row in data:
                row["race_date"] = date_str
                row["track"] = track
                row["off_time"] = off_time
                new_rows.append(row)
        else:
            logger.info("NO DATA  %s  %-20s  %s", date_str, track, off_time)
        fetched += 1

        if fetched % 50 == 0:
            logger.info(
                "Backfill progress: %d / %d races (%.1f%%)",
                fetched, len(races), 100 * fetched / len(races),
            )

        # Periodic save
        if fetched % batch_size == 0 and new_rows:
            batch_df = pd.DataFrame(new_rows)
            cache = pd.concat([cache, batch_df], ignore_index=True)
            save_rtv_cache(cache)
            new_rows = []

    # Final save
    if new_rows:
        batch_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, batch_df], ignore_index=True)
        save_rtv_cache(cache)

    logger.info(
        "Backfill complete: fetched %d races, cache now %d rows",
        fetched, len(cache),
    )
    return cache


def backfill_rtv_metrics_for_races(
    races_df: pd.DataFrame,
    max_races: int | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Incrementally fetch RTV metrics for a specific race subset.

    Args:
        races_df: DataFrame containing at least ``race_date``, ``track``,
            and ``off_time`` columns.
        max_races: Optional cap on number of missing races to fetch.
        skip_existing: If True, only fetch races not already in cache.

    Returns:
        Summary stats dict.
    """
    required = {"race_date", "track", "off_time"}
    if races_df is None or races_df.empty or not required.issubset(races_df.columns):
        return {
            "target_races": 0,
            "missing_races": 0,
            "fetched_races": 0,
            "new_rows": 0,
            "cache_rows": int(len(load_rtv_cache())),
        }

    races = (
        races_df[["race_date", "track", "off_time"]]
        .dropna(subset=["race_date", "track", "off_time"])
        .copy()
    )
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    races["track"] = races["track"].astype(str).str.strip()
    races["off_time"] = races["off_time"].astype(str).str.strip()
    races = races.dropna(subset=["race_date"]).drop_duplicates()
    races = races.sort_values("race_date", ascending=False).reset_index(drop=True)

    cache = load_rtv_cache()
    known_no_data = _load_no_data_race_keys()

    # ── Only check races newer than the latest date already covered ──
    # The union of cache dates and no-data dates tells us the latest
    # date we've already attempted.  Only fetch races strictly after
    # that cutoff (i.e. newly-added races since the last backfill).
    _covered_dates: list[str] = []
    if not cache.empty:
        _cd = pd.to_datetime(cache["race_date"], errors="coerce").dropna()
        if not _cd.empty:
            _covered_dates.append(_cd.max().strftime("%Y-%m-%d"))
    if known_no_data:
        _nd_dates = [k.split("|")[0] for k in known_no_data if "|" in k]
        if _nd_dates:
            _covered_dates.append(max(_nd_dates))
    if _covered_dates:
        _cutoff = max(_covered_dates)
        _before = len(races)
        races = races.loc[races["race_date"] > _cutoff].reset_index(drop=True)
        _skipped_old = _before - len(races)
        if _skipped_old:
            logger.info(
                "RTV backfill: skipped %d races on or before %s (already covered)",
                _skipped_old, _cutoff,
            )

    # Skip races we've previously confirmed as no-data (e.g. abandoned meetings).
    if known_no_data:
        _rk = races.apply(
            lambda r: _race_key(r["race_date"], r["track"], r["off_time"]),
            axis=1,
        )
        races = races.loc[~_rk.isin(known_no_data)].reset_index(drop=True)

    skipped_known_missing = 0

    if skip_existing and not cache.empty:
        _cache_keys = cache[["race_date", "track", "off_time"]].copy()
        _cache_keys["race_date"] = pd.to_datetime(_cache_keys["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        _cache_keys["track"] = _cache_keys["track"].map(_normalise_track_key)
        _cache_keys["off_time"] = _cache_keys["off_time"].map(_normalise_off_time_key)
        existing = set(_cache_keys.dropna().drop_duplicates().apply(tuple, axis=1))

        _r = races.copy()
        _r["track_norm"] = _r["track"].map(_normalise_track_key)
        _r["off_norm"] = _r["off_time"].map(_normalise_off_time_key)
        races = _r[
            ~_r.apply(lambda x: (x["race_date"], x["track_norm"], x["off_norm"]) in existing, axis=1)
        ][["race_date", "track", "off_time"]].reset_index(drop=True)

    # Re-apply known-no-data skip after existing-cache reduction.
    if known_no_data and not races.empty:
        _rk2 = races.apply(
            lambda r: _race_key(r["race_date"], r["track"], r["off_time"]),
            axis=1,
        )
        _mask_known = _rk2.isin(known_no_data)
        skipped_known_missing = int(_mask_known.sum())
        if skipped_known_missing:
            races = races.loc[~_mask_known].reset_index(drop=True)

    if max_races is not None:
        races = races.head(max_races)

    if races.empty:
        return {
            "target_races": int(len(races_df[["race_date", "track", "off_time"]].drop_duplicates())),
            "missing_races": 0,
            "fetched_races": 0,
            "new_rows": 0,
            "cache_rows": int(len(cache)),
        }

    logger.info("RTV incremental backfill: fetching %d missing races", len(races))
    new_rows: list[dict] = []
    total_new_rows = 0
    fetched = 0
    batch_size = 100
    newly_confirmed_no_data = 0
    today = datetime.now().date()

    for _, race in races.iterrows():
        date_str = str(race["race_date"])
        track = str(race["track"])
        off_time = str(race["off_time"])

        data = fetch_rtv_metrics(date_str, track, off_time)
        if data:
            for row in data:
                row["race_date"] = date_str
                row["track"] = track
                row["off_time"] = off_time
                new_rows.append(row)
            # If this race now has data, ensure it is not cached as no-data.
            known_no_data.discard(_race_key(date_str, track, off_time))
        else:
            # Cache no-data only for races that are definitely in the past.
            _rd = pd.to_datetime(date_str, errors="coerce")
            if pd.notna(_rd) and _rd.date() < today:
                _k = _race_key(date_str, track, off_time)
                if _k not in known_no_data:
                    known_no_data.add(_k)
                    newly_confirmed_no_data += 1
        fetched += 1

        if fetched % batch_size == 0 and new_rows:
            batch_df = pd.DataFrame(new_rows)
            cache = pd.concat([cache, batch_df], ignore_index=True)
            save_rtv_cache(cache)
            total_new_rows += len(batch_df)
            new_rows = []

    if new_rows:
        batch_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, batch_df], ignore_index=True)
        save_rtv_cache(cache)
        total_new_rows += len(batch_df)

    # Persist updated no-data skip-list.
    if newly_confirmed_no_data > 0:
        _save_no_data_race_keys(known_no_data)

    return {
        "target_races": int(len(races_df[["race_date", "track", "off_time"]].drop_duplicates())),
        "missing_races": int(len(races)),
        "fetched_races": int(fetched),
        "new_rows": int(total_new_rows),
        "cache_rows": int(len(cache)),
        "skipped_known_missing": int(skipped_known_missing),
        "new_known_missing": int(newly_confirmed_no_data),
    }


# ---------------------------------------------------------------------------
# Merge helper (used by feature_engineer)
# ---------------------------------------------------------------------------

def merge_rtv_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join cached RTV metrics onto a race DataFrame.

    Matches on ``(race_date, track, off_time, horse_name)``.
    All RTV metric columns are added (NaN where not available).
    """
    cache = load_rtv_cache()
    if cache.empty:
        for c in RTV_METRIC_COLS + RTV_RANK_COLS:
            df[c] = float("nan")
        return df

    # Normalise join keys
    cache["_rd_str"] = pd.to_datetime(cache["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cache["_trk_str"] = cache["track"].map(_normalise_track_key)
    cache["_ot_str"] = cache["off_time"].map(_normalise_off_time_key)
    cache["_hn_str"] = cache["horse_name"].map(_normalise_horse_key)

    df["_rd_str"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["_trk_str"] = df["track"].map(_normalise_track_key)
    df["_ot_str"] = df["off_time"].map(_normalise_off_time_key)
    df["_hn_str"] = df["horse_name"].map(_normalise_horse_key)

    # Ensure all metric columns exist in cache
    for c in RTV_METRIC_COLS + RTV_RANK_COLS:
        if c not in cache.columns:
            cache[c] = float("nan")

    # Keep only what we need from cache
    merge_cols = ["_rd_str", "_trk_str", "_ot_str", "_hn_str"] + RTV_METRIC_COLS + RTV_RANK_COLS
    cache_slim = cache[merge_cols].drop_duplicates(
        subset=["_rd_str", "_trk_str", "_ot_str", "_hn_str"], keep="last"
    )

    merged = df.merge(
        cache_slim,
        on=["_rd_str", "_trk_str", "_ot_str", "_hn_str"],
        how="left",
        indicator="_rtv_merge",
    )

    # Drop merge helper columns
    drop_cols = ["_rd_str", "_trk_str", "_ot_str", "_hn_str"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Ensure metric columns exist even if merge added nothing
    for c in RTV_METRIC_COLS + RTV_RANK_COLS:
        if c not in merged.columns:
            merged[c] = float("nan")

    has_any_metric = merged[RTV_METRIC_COLS].notna().any(axis=1)
    matched_rows = (merged["_rtv_merge"] == "both") if "_rtv_merge" in merged.columns else pd.Series(False, index=merged.index)

    if "_is_future" in merged.columns:
        is_future = merged["_is_future"].fillna(0).astype(int) == 1
        is_hist = ~is_future

        def _pct(mask: pd.Series, vec: pd.Series) -> float:
            den = int(mask.sum())
            if den <= 0:
                return 0.0
            return 100.0 * float(vec[mask].sum()) / den

        n_hist = int(is_hist.sum())
        logger.info(
            "RTV merge (historical): key matches %d / %d (%.1f%%), any-metric coverage %d / %d (%.1f%%)",
            int(matched_rows[is_hist].sum()), n_hist,
            _pct(is_hist, matched_rows),
            int(has_any_metric[is_hist].sum()), n_hist,
            _pct(is_hist, has_any_metric),
        )
        logger.info(
            "RTV merge note: future racecard rows are excluded from raw RTV coverage logging (pre-race raw metrics are expected missing)."
        )
    else:
        logger.info(
            "RTV merge: key matches %d / %d (%.1f%%), any-metric coverage %d / %d (%.1f%%)",
            matched_rows.sum(), len(merged),
            100 * matched_rows.sum() / max(1, len(merged)),
            has_any_metric.sum(), len(merged),
            100 * has_any_metric.sum() / max(1, len(merged)),
        )
    merged = merged.drop(columns=["_rtv_merge"], errors="ignore")
    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    max_r = int(sys.argv[1]) if len(sys.argv) > 1 else None
    backfill_rtv_metrics(max_races=max_r)
