"""
Real Data Collector Module
==========================
Collects real UK horse racing data from free API sources.

Supported sources:
1. The Racing API (free tier) — UK & Ireland racecards, results, horse history
2. RapidAPI Horse Racing (free tier, 50 req/day) — UK & Ireland racecards & results

Sign-up links:
- The Racing API:  https://www.theracingapi.com  (free plan, get username/password)
- RapidAPI:        https://rapidapi.com/ortegalex/api/horse-racing  (free plan, get API key)
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================================
# The Racing API  (free tier)
# =========================================================================

class TheRacingAPIClient:
    """
    Client for The Racing API (https://www.theracingapi.com).

    Free tier includes:
    - Daily racecards (basic data)
    - Results for all races on daily racecards (basic data)

    Sign up at https://www.theracingapi.com to get credentials.
    """

    BASE_URL = "https://api.theracingapi.com/v1"

    def __init__(self, username: str, password: str):
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.auth = self.auth

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make a GET request with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            time.sleep(0.5)  # Rate limit
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 401:
                logger.error("Authentication failed. Check your Racing API credentials.")
                return None
            if resp.status_code == 403:
                logger.warning(f"Access denied for {endpoint} (may require paid plan)")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Racing API request failed: {e}")
            return None

    def get_racecards(self, day: str = "today") -> Optional[list]:
        """
        Get racecards for today or tomorrow.

        Tries the paid endpoint first, then falls back to the free endpoint.

        Args:
            day: "today" or "tomorrow"

        Returns:
            List of race dicts with runners
        """
        # Try paid endpoint first
        data = self._get("racecards", params={"day": day})
        if data and "racecards" in data:
            logger.info(f"Retrieved {len(data['racecards'])} racecards for {day}")
            return data["racecards"]

        # Fall back to free endpoint (no day parameter — always returns today)
        logger.info("Falling back to free racecards endpoint...")
        data = self._get("racecards/free")
        if data and "racecards" in data:
            logger.info(f"Retrieved {len(data['racecards'])} racecards (free tier)")
            return data["racecards"]
        return None

    def get_results(
        self,
        start_date: str = None,
        end_date: str = None,
        limit: int = 50,
        skip: int = 0,
    ) -> Optional[dict]:
        """
        Get race results for a date range.

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            limit: Max results per page
            skip: Pagination offset
        """
        params = {"limit": limit, "skip": skip}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        return self._get("results", params=params)

    def get_horse_results(self, horse_id: str, limit: int = 20) -> Optional[dict]:
        """Get historical results for a specific horse."""
        return self._get(f"horses/{horse_id}/results", params={"limit": limit})

    def racecards_to_dataframe(self, racecards: list) -> pd.DataFrame:
        """Convert racecard JSON data to a flat DataFrame."""
        rows = []
        for race in racecards:
            race_info = {
                "race_id": race.get("race_id", ""),
                "race_date": race.get("date", ""),
                "off_time": race.get("off_time", ""),
                "track": race.get("course", ""),
                "region": race.get("region", ""),
                "race_name": race.get("race_name", ""),
                "race_class": race.get("race_class", ""),
                "race_type": _map_pattern(race.get("pattern", ""), race.get("type", "")),
                "distance_furlongs": _safe_float(race.get("distance_f")) or _parse_distance(race.get("distance", "")),
                "distance_raw": race.get("distance", race.get("distance_f", "")),
                "going": race.get("going", ""),
                "prize_money": _parse_prize(race.get("prize", "")),
                "field_size": _safe_int(race.get("field_size", 0)),
                "age_band": race.get("age_band", ""),
            }

            for runner in race.get("runners", []):
                # Skip non-runners
                number = str(runner.get("number", "")).strip().upper()
                if number == "NR":
                    continue

                row = {**race_info}
                row["horse_name"] = runner.get("horse", "")
                row["horse_id"] = runner.get("horse_id", "")
                row["jockey"] = runner.get("jockey", "")
                row["jockey_id"] = runner.get("jockey_id", "")
                row["trainer"] = runner.get("trainer", "")
                row["trainer_id"] = runner.get("trainer_id", "")
                row["age"] = _safe_int(runner.get("age"))
                row["weight_lbs"] = _parse_weight(runner.get("lbs", runner.get("weight_lbs", "")))
                row["draw"] = _safe_int(runner.get("draw"))
                row["form"] = runner.get("form", "")
                row["days_since_last_run"] = _safe_int(runner.get("last_run"))
                row["odds"] = _parse_odds(runner.get("odds", []))
                row["sex"] = runner.get("sex", runner.get("sex_code", ""))
                row["colour"] = runner.get("colour", "")
                row["sire"] = runner.get("sire", "")
                row["dam"] = runner.get("dam", "")
                row["owner"] = runner.get("owner", "")
                row["num_runners"] = race_info["field_size"] or len(race.get("runners", []))
                # Racecard entries don't have results yet
                row["finish_position"] = 0
                row["won"] = 0
                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def results_to_dataframe(self, results_data: dict) -> pd.DataFrame:
        """Convert results JSON data to a flat DataFrame."""
        rows = []
        results_list = results_data.get("results", [])

        for race in results_list:
            race_info = {
                "race_id": race.get("race_id", ""),
                "race_date": race.get("date", ""),
                "off_time": race.get("off_time", ""),
                "track": race.get("course", ""),
                "region": race.get("region", ""),
                "race_name": race.get("race_name", ""),
                "race_class": race.get("race_class", ""),
                "race_type": _map_pattern(race.get("pattern", ""), race.get("type", "")),
                "distance_furlongs": _parse_distance(race.get("distance", "")),
                "distance_raw": race.get("distance", ""),
                "going": race.get("going", ""),
                "prize_money": _parse_prize(race.get("prize", "")),
                "field_size": race.get("field_size", 0),
                "age_band": race.get("age_band", ""),
            }

            for runner in race.get("runners", []):
                row = {**race_info}
                row["horse_name"] = runner.get("horse", "")
                row["horse_id"] = runner.get("horse_id", "")
                row["jockey"] = runner.get("jockey", "")
                row["jockey_id"] = runner.get("jockey_id", "")
                row["trainer"] = runner.get("trainer", "")
                row["trainer_id"] = runner.get("trainer_id", "")
                row["age"] = _safe_int(runner.get("age"))
                row["weight_lbs"] = _parse_weight(runner.get("lbs", runner.get("weight_lbs", "")))
                row["draw"] = _safe_int(runner.get("draw"))
                row["form"] = runner.get("form", "")
                row["days_since_last_run"] = _safe_int(runner.get("last_run"))
                row["sex"] = runner.get("sex", runner.get("sex_code", ""))
                row["colour"] = runner.get("colour", "")
                row["sire"] = runner.get("sire", "")
                row["dam"] = runner.get("dam", "")
                row["owner"] = runner.get("owner", "")

                # Parse odds — could be SP (Starting Price) or array
                row["odds"] = _parse_odds(runner.get("odds", runner.get("sp_dec", "")))

                # Result fields
                pos_raw = runner.get("position", "")
                row["finish_position"] = _parse_position(pos_raw)
                row["won"] = 1 if row["finish_position"] == 1 else 0
                row["num_runners"] = race_info["field_size"] or len(race.get("runners", []))
                row["lengths_behind"] = _safe_float(runner.get("distance_btn", ""))
                row["finish_time_secs"] = 0.0  # Not always available

                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def collect_results_range(
        self,
        start_date: str,
        end_date: str,
        region: str = "gb",
    ) -> pd.DataFrame:
        """
        Collect all results for a date range, handling pagination.

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            region: "gb" for Great Britain, "ire" for Ireland
        """
        all_frames = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            limit = 50
            skip = 0

            while True:
                data = self.get_results(
                    start_date=date_str,
                    end_date=date_str,
                    limit=limit,
                    skip=skip,
                )
                if not data:
                    break

                df = self.results_to_dataframe(data)
                if not df.empty:
                    # Filter to requested region
                    if region:
                        df = df[df["region"].str.lower() == region.lower()]
                    all_frames.append(df)

                total = data.get("total", 0)
                skip += limit
                if skip >= total:
                    break

            logger.info(f"  {date_str}: collected results")
            current += timedelta(days=1)

        if all_frames:
            result = pd.concat(all_frames, ignore_index=True)
            logger.info(f"Total: {len(result)} runner entries collected")
            return result
        return pd.DataFrame()


# =========================================================================
# RapidAPI Horse Racing  (free tier — 50 requests/day)
# =========================================================================

class RapidAPIRacingClient:
    """
    Client for the Horse Racing API on RapidAPI.
    Free tier: 50 requests/day.

    Sign up at https://rapidapi.com/ortegalex/api/horse-racing
    to get your API key (x-rapidapi-key).
    """

    BASE_URL = "https://horse-racing.p.rapidapi.com"

    def __init__(self, api_key: str):
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "horse-racing.p.rapidapi.com",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _get(self, endpoint: str, params: dict = None) -> Optional[dict | list]:
        """Make a GET request with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            time.sleep(1.0)  # Be respectful of rate limits
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 403:
                logger.error("RapidAPI authentication failed. Check your API key.")
                return None
            if resp.status_code == 429:
                logger.warning("RapidAPI rate limit exceeded (50 requests/day on free tier)")
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"RapidAPI request failed: {e}")
            return None

    def get_racecards(self, date: str = None) -> Optional[list]:
        """
        Get today's racecards.

        Args:
            date: Optional date in YYYY-MM-DD format
        """
        params = {}
        if date:
            params["date"] = date
        data = self._get("racecards", params=params)
        if isinstance(data, list):
            logger.info(f"RapidAPI: Retrieved {len(data)} races")
            return data
        return None

    def get_results(self, date: str = None) -> Optional[list]:
        """
        Get race results.

        Args:
            date: Date in YYYY-MM-DD format
        """
        params = {}
        if date:
            params["date"] = date
        data = self._get("results", params=params)
        if isinstance(data, list):
            logger.info(f"RapidAPI: Retrieved {len(data)} race results")
            return data
        return None

    def get_race_detail(self, race_id: str) -> Optional[dict]:
        """Get detailed info for a specific race."""
        return self._get(f"race/{race_id}")

    def racecards_to_dataframe(self, racecards: list) -> pd.DataFrame:
        """Convert RapidAPI racecard data to a flat DataFrame."""
        rows = []
        for race in racecards:
            race_info = {
                "race_id": race.get("id_race", race.get("race_id", "")),
                "race_date": race.get("date", ""),
                "off_time": race.get("off", race.get("off_time", "")),
                "track": race.get("course", ""),
                "race_name": race.get("race", race.get("race_name", "")),
                "race_class": race.get("class", ""),
                "distance_raw": race.get("distance", ""),
                "distance_furlongs": _parse_distance(race.get("distance", "")),
                "going": race.get("going", ""),
                "age_band": race.get("age", ""),
                "region": "gb",
                "race_type": race.get("type", "Flat"),
                "prize_money": _parse_prize(race.get("prize", "")),
            }

            runners = race.get("runners", [])
            race_info["field_size"] = len(runners)

            for runner in runners:
                row = {**race_info}
                row["horse_name"] = runner.get("horse", "")
                row["horse_id"] = runner.get("id_horse", "")
                row["jockey"] = runner.get("jockey", "")
                row["trainer"] = runner.get("trainer", "")
                row["age"] = _safe_int(runner.get("age"))
                row["weight_lbs"] = _parse_weight(runner.get("lbs", ""))
                row["draw"] = _safe_int(runner.get("draw"))
                row["form"] = runner.get("form", "")
                row["days_since_last_run"] = _safe_int(runner.get("last_run"))
                row["num_runners"] = race_info["field_size"]
                row["sex"] = runner.get("sex", "")
                row["sire"] = runner.get("sire", "")
                row["dam"] = runner.get("dam", "")
                row["owner"] = runner.get("owner", "")

                # Odds
                odds_list = runner.get("odds", [])
                row["odds"] = _parse_odds(odds_list)

                # Position (for results)
                pos_raw = runner.get("position", "")
                row["finish_position"] = _parse_position(pos_raw)
                row["won"] = 1 if row["finish_position"] == 1 else 0
                row["lengths_behind"] = _safe_float(runner.get("distance", ""))
                row["finish_time_secs"] = 0.0

                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def collect_results_range(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Collect results for a date range (limited by free tier's 50 req/day).

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
        """
        all_frames = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        request_count = 0

        while current <= end:
            if request_count >= 45:  # Leave some buffer
                logger.warning("Approaching RapidAPI daily limit. Stopping collection.")
                break

            date_str = current.strftime("%Y-%m-%d")
            results = self.get_results(date=date_str)
            request_count += 1

            if results:
                df = self.racecards_to_dataframe(results)
                if not df.empty:
                    all_frames.append(df)
                    logger.info(f"  {date_str}: {len(df)} runners")

            current += timedelta(days=1)

        if all_frames:
            result = pd.concat(all_frames, ignore_index=True)
            logger.info(f"Total: {len(result)} runner entries collected")
            return result
        return pd.DataFrame()


# =========================================================================
# Helper functions for parsing API data
# =========================================================================

def _safe_int(val) -> int:
    """Safely convert a value to int."""
    if val is None or val == "":
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _safe_float(val) -> float:
    """Safely convert a value to float."""
    if val is None or val == "":
        return 0.0
    try:
        # Handle strings like "2½" or "nk" or "hd"
        s = str(val).strip().lower()
        if s in ("", "dh", "nk", "hd", "nse", "shd", "sht-hd", "dist"):
            return {"dh": 0.0, "nk": 0.2, "hd": 0.1, "nse": 0.05,
                    "shd": 0.1, "sht-hd": 0.1, "dist": 30.0}.get(s, 0.0)
        # Handle fractions like "1½", "2¼"
        s = s.replace("½", ".5").replace("¼", ".25").replace("¾", ".75")
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _parse_prize(prize_val) -> float:
    """
    Parse a prize money value (e.g. "£10,000", "10000", 10000) to float.
    """
    if prize_val is None:
        return 0.0
    try:
        s = str(prize_val).strip()
        # Remove currency symbols and commas
        s = s.replace("£", "").replace("€", "").replace("$", "").replace(",", "").strip()
        return float(s) if s else 0.0
    except (ValueError, TypeError):
        return 0.0


def _parse_distance(dist_str: str) -> float:
    """
    Parse a distance string to furlongs.
    E.g. "1m2f" -> 10, "7f" -> 7, "2m" -> 16, "1m" -> 8
    """
    if not dist_str:
        return 0.0
    dist_str = str(dist_str).lower().strip()

    miles = 0.0
    furlongs = 0.0
    yards = 0.0

    import re

    # Match miles
    m_match = re.search(r"(\d+)m", dist_str)
    if m_match:
        miles = float(m_match.group(1))

    # Match furlongs
    f_match = re.search(r"(\d+)f", dist_str)
    if f_match:
        furlongs = float(f_match.group(1))

    # Match yards
    y_match = re.search(r"(\d+)y", dist_str)
    if y_match:
        yards = float(y_match.group(1))

    # Also handle half-furlongs: ½
    if "½" in dist_str:
        furlongs += 0.5

    total_furlongs = (miles * 8) + furlongs + (yards / 220)

    # If parsing failed, try treating it as a plain number
    if total_furlongs == 0:
        try:
            total_furlongs = float(dist_str)
        except ValueError:
            pass

    return round(total_furlongs, 1)


def _parse_weight(weight_val) -> int:
    """
    Parse weight to pounds (lbs).
    Handles formats: 130, "9-7" (stones-lbs), "130lbs"
    """
    if weight_val is None or weight_val == "":
        return 0
    s = str(weight_val).strip().lower().replace("lbs", "").replace("lb", "")

    # Try stones-pounds format: "9-7" = 9*14 + 7 = 133
    if "-" in s:
        parts = s.split("-")
        try:
            stones = int(parts[0])
            lbs = int(parts[1])
            return stones * 14 + lbs
        except (ValueError, IndexError):
            pass

    try:
        return int(float(s))
    except ValueError:
        return 0


def _parse_odds(odds_val) -> float:
    """
    Parse odds to decimal format.
    Handles: float, list of bookmaker odds, fractional string "5/1"
    """
    if odds_val is None or odds_val == "":
        return 0.0

    # If it's already a number
    if isinstance(odds_val, (int, float)):
        return float(odds_val) if odds_val > 0 else 0.0

    # If it's a list of bookmaker odds, take the first or average
    if isinstance(odds_val, list):
        decimal_odds = []
        for item in odds_val:
            if isinstance(item, dict):
                dec = item.get("decimal", item.get("odds_decimal", 0))
                if dec and float(dec) > 1:
                    decimal_odds.append(float(dec))
            elif isinstance(item, (int, float)):
                if item > 1:
                    decimal_odds.append(float(item))
        if decimal_odds:
            return round(sum(decimal_odds) / len(decimal_odds), 2)
        return 0.0

    # If it's a string — try fractional "5/1" or decimal "6.0"
    s = str(odds_val).strip()
    if "/" in s:
        parts = s.split("/")
        try:
            return round(float(parts[0]) / float(parts[1]) + 1, 2)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        v = float(s)
        return v if v > 0 else 0.0
    except ValueError:
        return 0.0


def _parse_position(pos_val) -> int:
    """Parse finish position from various formats."""
    if pos_val is None or pos_val == "":
        return 0
    s = str(pos_val).strip().lower()

    # Handle common non-numeric results
    non_finish = {"f": 0, "ur": 0, "pu": 0, "bd": 0, "su": 0,
                  "rr": 0, "ro": 0, "co": 0, "dsq": 0, "void": 0,
                  "nr": 0, "lft": 0, "ref": 0}
    if s in non_finish:
        return 0

    # Remove ordinal suffixes
    s = s.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
    try:
        return int(s)
    except ValueError:
        return 0


def _map_pattern(pattern: str, race_type: str) -> str:
    """Map race pattern/type to a standard race type."""
    p = str(pattern).lower() if pattern else ""
    t = str(race_type).lower() if race_type else ""

    if "chase" in p or "chase" in t:
        return "Chase"
    elif "hurdle" in p or "hurdle" in t:
        return "Hurdle"
    elif "nh flat" in p or "bumper" in p or "nh flat" in t:
        return "National Hunt Flat"
    else:
        return "Flat"


# =========================================================================
# Unified data collection interface
# =========================================================================

def collect_real_data(
    source: str = "racing_api",
    days_back: int = 7,
    save: bool = True,
    **credentials,
) -> pd.DataFrame:
    """
    Collect real UK horse racing data.

    Args:
        source: "racing_api" or "rapidapi"
        days_back: Number of days of historical results to collect
        save: Whether to save to CSV
        **credentials: API credentials
            For racing_api: username, password
            For rapidapi: api_key

    Returns:
        DataFrame with race results
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    if source == "racing_api":
        username = credentials.get("username", "") or getattr(config, "RACING_API_USERNAME", "")
        password = credentials.get("password", "") or getattr(config, "RACING_API_PASSWORD", "")
        if not username or not password:
            raise ValueError(
                "The Racing API requires username and password. "
                "Sign up free at https://www.theracingapi.com"
            )

        client = TheRacingAPIClient(username, password)
        df = client.collect_results_range(start_date, end_date, region="gb")

    elif source == "rapidapi":
        api_key = credentials.get("api_key", "") or getattr(config, "RAPIDAPI_KEY", "")
        if not api_key:
            raise ValueError(
                "RapidAPI requires an API key. "
                "Sign up free at https://rapidapi.com/ortegalex/api/horse-racing"
            )

        client = RapidAPIRacingClient(api_key)
        df = client.collect_results_range(start_date, end_date)

    else:
        raise ValueError(f"Unknown source: {source}. Use 'racing_api' or 'rapidapi'")

    if df.empty:
        logger.warning("No data collected. Check API credentials and date range.")
        return df

    if save:
        output_path = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
        # Append to existing data if it exists
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            df = pd.concat([existing, df], ignore_index=True)
            # Remove duplicates based on race_id + horse_name
            df = df.drop_duplicates(subset=["race_id", "horse_name"], keep="last")

        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} entries to {output_path}")

    return df


def get_todays_racecards(
    source: str = "racing_api",
    day: str = "today",
    **credentials,
) -> pd.DataFrame:
    """
    Get today's (or tomorrow's) racecards for prediction.

    Args:
        source: "racing_api" or "rapidapi"
        day: "today" or "tomorrow"
        **credentials: API credentials

    Returns:
        DataFrame with racecard entries
    """
    if source == "racing_api":
        username = credentials.get("username", "") or getattr(config, "RACING_API_USERNAME", "")
        password = credentials.get("password", "") or getattr(config, "RACING_API_PASSWORD", "")
        client = TheRacingAPIClient(username, password)
        racecards = client.get_racecards(day=day)
        if racecards:
            df = client.racecards_to_dataframe(racecards)
            if not df.empty:
                logger.info(f"Loaded {len(df)} runners from {df['race_id'].nunique()} races")
                return df

    elif source == "rapidapi":
        api_key = credentials.get("api_key", "") or getattr(config, "RAPIDAPI_KEY", "")
        client = RapidAPIRacingClient(api_key)
        date_str = None
        if day == "tomorrow":
            date_str = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        racecards = client.get_racecards(date=date_str)
        if racecards:
            df = client.racecards_to_dataframe(racecards)
            if not df.empty:
                logger.info(f"Loaded {len(df)} runners from {df['race_id'].nunique()} races")
                return df

    logger.warning("No racecards retrieved. Check credentials and try again.")
    return pd.DataFrame()


# =========================================================================
# Keep the sample data generator for testing/fallback
# =========================================================================

def generate_sample_data(num_races: int = 1500, save: bool = True) -> pd.DataFrame:
    """Generate synthetic data for testing (imported from original module)."""
    from src.data_collector_sample import generate_sample_data as _gen
    return _gen(num_races=num_races, save=save)


def collect_data(
    source: str = "sample",
    num_races: int = 1500,
    days_back: int = 7,
    **credentials,
) -> pd.DataFrame:
    """
    Main entry point for data collection.

    Args:
        source: "sample", "racing_api", or "rapidapi"
        num_races: Number of races for sample data
        days_back: Days of history for API sources
        **credentials: API credentials

    Returns:
        DataFrame with race data
    """
    if source == "sample":
        return generate_sample_data(num_races=num_races, save=True)
    else:
        return collect_real_data(
            source=source, days_back=days_back, save=True, **credentials
        )
