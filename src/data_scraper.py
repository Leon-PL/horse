"""
Web Scraper Module
==================
Scrapes real UK horse racing data from Sporting Life (sportinglife.com).

Data is extracted from Next.js ``__NEXT_DATA__`` JSON embedded in each page,
so no JavaScript execution is needed — a simple HTTP GET is sufficient.

**Data sources:**

* **Results listing** — ``/racing/results/{YYYY-MM-DD}``
  Returns all meetings / races for a day with top-3 finishers.

* **Individual result** — ``/racing/results/{date}/{course}/{race_id}/{slug}``
  Returns full ``rides[]`` array with position, odds, jockey, trainer, form …

* **Racecard listing** — ``/racing/racecards/{YYYY-MM-DD}``
  Returns meetings / races for a day (no runners).

* **Individual racecard** — ``/racing/racecards/{date}/{course}/racecard/{race_id}/{slug}``
  Returns full ``rides[]`` with pre-race data (odds, form, commentary …).

Usage::

    from src.data_scraper import scrape_results, scrape_todays_racecards

    # Historical results for the last 14 days
    df = scrape_results(days_back=14)

    # Today's racecards with odds
    df = scrape_todays_racecards()
"""

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock, Semaphore
from typing import Optional

import numpy as np
import pandas as pd
import requests

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

SPORTING_LIFE_BASE = "https://www.sportinglife.com"

REQUEST_DELAY = 0.5   # seconds between requests (per-thread minimum)
MAX_WORKERS = 5       # concurrent HTTP requests for race pages

# Pre-compiled regex to extract __NEXT_DATA__ JSON — much faster than
# parsing the full DOM tree with BeautifulSoup.
_NEXT_DATA_RE = re.compile(
    r'<script\s+id="__NEXT_DATA__"[^>]*>(.+?)</script>',
    re.DOTALL,
)

# Countries we consider UK & Ireland
UK_IRE_COUNTRIES = {
    "england", "scotland", "wales", "northern ireland", "ireland",
    "great britain", "united kingdom",
}

# Fallback track-slug list for UK & Ireland courses
UK_IRE_TRACK_SLUGS = {
    # GB
    "aintree", "ascot", "ayr", "bangor-on-dee", "bath", "beverley",
    "brighton", "carlisle", "cartmel", "catterick", "chelmsford-city",
    "cheltenham", "chepstow", "chester", "doncaster", "epsom",
    "exeter", "fakenham", "ffos-las", "fontwell", "goodwood",
    "hamilton", "haydock", "hereford", "hexham", "huntingdon",
    "kelso", "kempton", "kempton-park", "leicester", "lingfield",
    "ludlow", "market-rasen", "musselburgh", "newbury", "newcastle",
    "newmarket", "newton-abbot", "nottingham", "perth", "plumpton",
    "pontefract", "redcar", "ripon", "salisbury", "sandown",
    "sandown-park", "sedgefield", "southwell", "stratford",
    "taunton", "thirsk", "towcester", "uttoxeter", "warwick",
    "wetherby", "wincanton", "windsor", "wolverhampton", "worcester",
    "yarmouth", "york",
    # IRE
    "cork", "curragh", "down-royal", "downpatrick", "dundalk",
    "fairyhouse", "galway", "gowran-park", "kilbeggan", "killarney",
    "laytown", "leopardstown", "limerick", "listowel", "naas",
    "navan", "punchestown", "roscommon", "sligo", "thurles",
    "tipperary", "tramore", "wexford",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug (lowercase, hyphens)."""
    text = text.lower()
    text = re.sub(r"[''']", "", text)      # remove apostrophes
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _parse_distance_to_furlongs(dist_str: str) -> float:
    """
    Convert distance strings like ``'2m 7f 96y'`` to furlongs.

    1 mile = 8 furlongs, 1 furlong = 220 yards.
    """
    if not dist_str:
        return 10.0

    dist_str = dist_str.strip().lower()

    miles = furlongs = yards = 0
    m_match = re.search(r'(\d+)\s*m(?:ile)?', dist_str)
    f_match = re.search(r'(\d+)\s*f', dist_str)
    y_match = re.search(r'(\d+)\s*y', dist_str)

    if m_match:
        miles = int(m_match.group(1))
    if f_match:
        furlongs = int(f_match.group(1))
    if y_match:
        yards = int(y_match.group(1))

    total = miles * 8 + furlongs + yards / 220.0
    return round(total, 2) if total > 0 else 10.0


def _parse_weight_to_lbs(weight_str: str) -> float:
    """
    Convert weight like ``'11-9'`` (11 st 9 lb) to total pounds.
    1 stone = 14 lb.
    """
    if not weight_str:
        return 140.0

    match = re.search(r'(\d+)\s*[-–]\s*(\d+)', str(weight_str))
    if match:
        return int(match.group(1)) * 14 + int(match.group(2))

    match = re.search(r'(\d+)', str(weight_str))
    if match:
        return float(match.group(1))

    return 140.0


def _parse_fractional_odds(odds_str: str) -> float:
    """
    Convert fractional odds like ``'5/2'``, ``'11/4f'``, ``'evs'``
    to decimal odds (e.g. 5/2 → 3.5).
    """
    if not odds_str:
        return 5.0

    odds_str = str(odds_str).strip().lower()
    odds_str = odds_str.replace("f", "").replace("j", "").strip()

    if odds_str in ("evs", "evens", "ev"):
        return 2.0

    match = re.match(r'(\d+)\s*/\s*(\d+)', odds_str)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den > 0:
            return round(num / den + 1, 2)

    try:
        return float(odds_str)
    except (ValueError, TypeError):
        return 5.0


def _parse_lengths_behind(dist_str: str) -> float:
    """
    Parse finish distance strings like ``'8 ½'``, ``'nk'``, ``'1¼'``
    into a numeric lengths-behind value.
    """
    if not dist_str:
        return 0.0

    s = str(dist_str).strip().lower()

    # Named distances
    named = {
        "dht": 0.0, "nse": 0.05, "shd": 0.1, "sh": 0.1,
        "hd": 0.2, "nk": 0.3, "sn": 0.1, "snk": 0.2,
    }
    for k, v in named.items():
        if k in s:
            return v

    # Unicode fractions
    frac_map = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 0.33, "⅔": 0.67}
    total = 0.0
    for fchar, fval in frac_map.items():
        if fchar in s:
            total += fval
            s = s.replace(fchar, "")

    # Whole number part
    whole_match = re.search(r'(\d+)', s)
    if whole_match:
        total += int(whole_match.group(1))

    return total


def _infer_race_type(race_name: str, run_type: str = "") -> str:
    """Infer race type from race name or ``run_type`` field."""
    if run_type:
        rt = run_type.upper()
        if rt == "CHASE":
            return "Chase"
        if rt in ("HURDLE", "HURDL"):
            return "Hurdle"
        if rt in ("FLAT", "AW FLAT"):
            return "Flat"
        if rt in ("BUMPER", "NHF"):
            return "Flat"

    name = (race_name or "").lower()
    if "chase" in name:
        return "Chase"
    if "hurdle" in name:
        return "Hurdle"
    if "bumper" in name or "nh flat" in name:
        return "Flat"
    return "Flat"


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _is_uk_ire_meeting(meeting: dict) -> bool:
    """Check if a meeting is at a UK or Ireland venue."""
    ms = meeting.get("meeting_summary", {})
    course = ms.get("course", {})

    # Method 1: country long_name
    country = course.get("country", {})
    if isinstance(country, dict):
        long_name = country.get("long_name", "")
        if long_name and long_name.lower() in UK_IRE_COUNTRIES:
            return True

    # Method 2: track slug in known list
    course_name = course.get("name", "")
    if _slugify(course_name) in UK_IRE_TRACK_SLUGS:
        return True

    # Method 3: feed source (RUK / ATR → UK/IRE broadcasters)
    feed = course.get("feed_source", "")
    if feed in ("RUK", "ATR", "SIS"):
        return True

    return False


def _extract_prize_money(prizes_data) -> float:
    """Extract first-place prize money from the prizes structure."""
    if not prizes_data:
        return 0.0

    prize_list = prizes_data
    if isinstance(prizes_data, dict):
        prize_list = prizes_data.get("prize", [])

    if isinstance(prize_list, list):
        for p in prize_list:
            if isinstance(p, dict) and p.get("position") == 1:
                prize_str = str(p.get("prize", "0"))
                # "34170.0 GBP" → 34170.0
                val = re.sub(
                    r'[^0-9.]', '',
                    prize_str.split()[0] if ' ' in prize_str else prize_str,
                )
                return _safe_float(val, 0)
    return 0.0


# ---------------------------------------------------------------------------
# Sporting Life Scraper
# ---------------------------------------------------------------------------

class SportingLifeScraper:
    """
    Scrapes horse racing data from sportinglife.com.

    Extracts data from the ``__NEXT_DATA__`` JSON that Sporting Life
    embeds in every page as a Next.js application.
    """

    _HEADERS = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(self):
        # Thread-local storage so each worker thread gets its own
        # ``requests.Session`` (Session is NOT thread-safe).
        self._local = threading.local()
        # Global rate-limiter — at most MAX_WORKERS concurrent requests,
        # each separated by REQUEST_DELAY seconds.
        self._req_sem = Semaphore(MAX_WORKERS)
        self._req_lock = Lock()
        self._last_req_time = 0.0

    @property
    def session(self) -> requests.Session:
        """Return a per-thread ``requests.Session``."""
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(self._HEADERS)
            self._local.session = s
        return s

    # -- network helpers ---------------------------------------------------

    def _get(self, url: str) -> requests.Response:
        """Rate-limited GET request (thread-safe)."""
        with self._req_sem:
            # Enforce minimum gap between requests across all threads
            with self._req_lock:
                elapsed = time.time() - self._last_req_time
                if elapsed < REQUEST_DELAY:
                    time.sleep(REQUEST_DELAY - elapsed)
                self._last_req_time = time.time()
            logger.debug(f"  GET {url}")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp

    def _fetch_next_data(self, url: str) -> Optional[dict]:
        """Fetch a page and return the embedded ``__NEXT_DATA__`` dict."""
        resp = self._get(url)
        m = _NEXT_DATA_RE.search(resp.text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse __NEXT_DATA__ from {url}")
        return None

    # ==================================================================
    # RESULTS — listing page → race URLs
    # ==================================================================

    def get_results_urls(
        self, date_str: str, uk_only: bool = True
    ) -> list[dict]:
        """
        Return race-result URL info for every race on *date_str*.

        Each item is a dict with keys:
        ``url``, ``track``, ``race_name``, ``time``, ``race_id``.
        """
        url = f"{SPORTING_LIFE_BASE}/racing/results/{date_str}"
        data = self._fetch_next_data(url)
        if not data:
            logger.warning(f"No __NEXT_DATA__ found for {url}")
            return []

        races: list[dict] = []
        try:
            meetings = data["props"]["pageProps"].get("meetings", [])
            for meeting in meetings:
                if uk_only and not _is_uk_ire_meeting(meeting):
                    continue

                ms = meeting.get("meeting_summary", {})
                course_name = ms.get("course", {}).get("name", "")
                course_slug = _slugify(course_name)

                for race in meeting.get("races", []):
                    ref = race.get("race_summary_reference", {})
                    race_id = ref.get("id", "")
                    race_name = race.get("name", "")
                    race_slug = _slugify(race_name) if race_name else ""
                    off_time = race.get("time", "")

                    if not race_id or not course_slug:
                        continue

                    race_url = (
                        f"{SPORTING_LIFE_BASE}/racing/results/"
                        f"{date_str}/{course_slug}/{race_id}/{race_slug}"
                    )
                    races.append({
                        "url": race_url,
                        "track": course_name,
                        "race_name": race_name,
                        "time": off_time,
                        "race_id": str(race_id),
                    })
        except (KeyError, TypeError) as e:
            logger.warning(f"Error parsing results listing: {e}")

        logger.info(f"  Found {len(races)} race result links for {date_str}")
        return races

    # ==================================================================
    # RESULTS — individual race page → runner records
    # ==================================================================

    def scrape_race_result(
        self, race_url: str, date_str: str = ""
    ) -> list[dict]:
        """
        Scrape a single race-result page and return pipeline-ready dicts.
        """
        data = self._fetch_next_data(race_url)
        if not data:
            logger.warning(f"No __NEXT_DATA__ for result: {race_url}")
            return []

        runners: list[dict] = []
        try:
            page_props = data["props"]["pageProps"]
            race_data = page_props.get("race", {})
            if not race_data:
                return []

            summary = race_data.get("race_summary", {})

            # ---- race-level metadata ----
            race_id = str(
                summary.get("race_summary_reference", {}).get("id", "")
            )
            race_name = summary.get("name", "")
            course_name = summary.get("course_name", "")
            going = summary.get("going", "")
            distance = summary.get("distance", "")
            race_class = str(summary.get("race_class", ""))
            off_time = summary.get("time", "")
            num_runners = _safe_int(summary.get("ride_count", 0))
            has_handicap = summary.get("has_handicap", False)
            surface = summary.get("course_surface", {})
            surface_str = (
                surface.get("surface", "Turf")
                if isinstance(surface, dict) else "Turf"
            )

            if not date_str:
                date_str = summary.get("date", "")

            distance_f = _parse_distance_to_furlongs(distance)
            race_type = _infer_race_type(race_name)
            prize_money = _extract_prize_money(race_data.get("prizes"))

            is_uk = page_props.get("isUkorIreMeeting", True)
            region = "UK" if is_uk else "INT"

            # ---- parse each ride (runner) ----
            for ride in race_data.get("rides", []):
                row = self._ride_to_dict(
                    ride,
                    race_id=race_id,
                    date_str=date_str,
                    off_time=off_time,
                    course_name=course_name,
                    region=region,
                    race_name=race_name,
                    race_class=race_class,
                    race_type=race_type,
                    distance_f=distance_f,
                    going=going,
                    prize_money=prize_money,
                    num_runners=num_runners,
                    surface_str=surface_str,
                    has_handicap=has_handicap,
                    is_result=True,
                )
                if row:
                    runners.append(row)

        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Error parsing result JSON: {e}")

        return runners

    # ==================================================================
    # RACECARDS — listing page → race URLs
    # ==================================================================

    def get_racecard_urls(
        self, date_str: str = None, uk_only: bool = True
    ) -> list[dict]:
        """
        Return racecard URL info for every race on *date_str*
        (defaults to today).

        Uses the internal Sporting Life JSON API
        ``/api/horse-racing/racing/racecards/{date}`` which returns the
        correct meetings for any date including future dates.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        api_url = f"{SPORTING_LIFE_BASE}/api/horse-racing/racing/racecards/{date_str}"
        resp = self._get(api_url)
        if resp.status_code != 200:
            logger.warning(f"Racecard API returned HTTP {resp.status_code} for {api_url}")
            return []
        try:
            meetings = resp.json()
        except Exception as e:
            logger.warning(f"Failed to parse racecard API response: {e}")
            return []

        races: list[dict] = []
        try:
            for meeting in meetings:
                if uk_only and not _is_uk_ire_meeting(meeting):
                    continue

                ms = meeting.get("meeting_summary", {})
                course_name = ms.get("course", {}).get("name", "")
                course_slug = _slugify(course_name)
                meeting_date = ms.get("date") or date_str

                for race in meeting.get("races", []):
                    ref = race.get("race_summary_reference", {})
                    race_id = ref.get("id", "")
                    race_name = race.get("name", "")
                    race_slug = _slugify(race_name) if race_name else ""
                    off_time = race.get("time", "")

                    if not race_id or not course_slug:
                        continue

                    race_date = race.get("date") or meeting_date

                    race_url = (
                        f"{SPORTING_LIFE_BASE}/racing/racecards/"
                        f"{race_date}/{course_slug}/racecard"
                        f"/{race_id}/{race_slug}"
                    )
                    races.append({
                        "url": race_url,
                        "track": course_name,
                        "race_name": race_name,
                        "time": off_time,
                        "race_id": str(race_id),
                    })
        except (KeyError, TypeError) as e:
            logger.warning(f"Error parsing racecard listing: {e}")

        logger.info(f"  Found {len(races)} racecard links for {date_str}")
        return races

    # ==================================================================
    # RACECARDS — individual page → runner records
    # ==================================================================

    def scrape_racecard(
        self, race_url: str, date_str: str = ""
    ) -> list[dict]:
        """
        Scrape a single racecard page and return pipeline-ready dicts.
        ``finish_position`` is 0 (race not yet run).
        """
        data = self._fetch_next_data(race_url)
        if not data:
            logger.warning(f"No __NEXT_DATA__ for racecard: {race_url}")
            return []

        runners: list[dict] = []
        try:
            page_props = data["props"]["pageProps"]
            race_data = page_props.get("race", {})
            if not race_data:
                return []

            summary = race_data.get("race_summary", {})

            race_id = str(
                summary.get("race_summary_reference", {}).get("id", "")
            )
            race_name = summary.get("name", "")
            course_name = summary.get("course_name", "")
            going = summary.get("going", "")
            distance = summary.get("distance", "")
            race_class = str(summary.get("race_class", ""))
            off_time = summary.get("time", "")
            num_runners = _safe_int(summary.get("ride_count", 0))
            has_handicap = summary.get("has_handicap", False)
            surface = summary.get("course_surface", {})
            surface_str = (
                surface.get("surface", "Turf")
                if isinstance(surface, dict) else "Turf"
            )

            if not date_str:
                date_str = summary.get("date", "")

            distance_f = _parse_distance_to_furlongs(distance)
            race_type = _infer_race_type(race_name)
            prize_money = _extract_prize_money(race_data.get("prizes"))

            is_uk = page_props.get("isUkorIreMeeting", True)
            region = "UK" if is_uk else "INT"

            for ride in race_data.get("rides", []):
                row = self._ride_to_dict(
                    ride,
                    race_id=race_id,
                    date_str=date_str,
                    off_time=off_time,
                    course_name=course_name,
                    region=region,
                    race_name=race_name,
                    race_class=race_class,
                    race_type=race_type,
                    distance_f=distance_f,
                    going=going,
                    prize_money=prize_money,
                    num_runners=num_runners,
                    surface_str=surface_str,
                    has_handicap=has_handicap,
                    is_result=False,
                )
                if row:
                    runners.append(row)

        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Error parsing racecard JSON: {e}")

        return runners

    # ==================================================================
    # Shared ride → dict conversion
    # ==================================================================

    def _ride_to_dict(
        self,
        ride: dict,
        *,
        race_id: str,
        date_str: str,
        off_time: str,
        course_name: str,
        region: str,
        race_name: str,
        race_class: str,
        race_type: str,
        distance_f: float,
        going: str,
        prize_money: float,
        num_runners: int,
        surface_str: str,
        has_handicap: bool,
        is_result: bool,
    ) -> Optional[dict]:
        """
        Convert a single Sporting Life *ride* object into a flat dict
        matching the pipeline's expected columns.
        """
        status = ride.get("ride_status", "")
        if status == "NON-RUNNER":
            return None

        # --- horse ---
        horse = ride.get("horse", {})
        horse_name = horse.get("name", "")
        horse_id = str(horse.get("horse_reference", {}).get("id", ""))
        age = _safe_int(horse.get("age", 0))

        sex_data = horse.get("sex", {})
        sex = sex_data.get("type", "") if isinstance(sex_data, dict) else ""

        form_data = horse.get("formsummary", {})
        form = (
            form_data.get("display_text", "")
            if isinstance(form_data, dict) else ""
        )
        days_since = _safe_int(horse.get("last_ran_days", 0))

        # --- jockey / trainer ---
        jockey_data = ride.get("jockey", {})
        jockey = (
            jockey_data.get("name", "")
            if isinstance(jockey_data, dict) else str(jockey_data or "")
        )
        trainer_data = ride.get("trainer", {})
        trainer = (
            trainer_data.get("name", "")
            if isinstance(trainer_data, dict) else str(trainer_data or "")
        )

        # --- weight / draw / rating ---
        weight_lbs = _parse_weight_to_lbs(ride.get("handicap", ""))
        draw = _safe_int(ride.get("cloth_number", 0))
        official_rating = _safe_int(ride.get("official_rating", 0))

        # --- headgear ---
        headgear_data = ride.get("headgear", "")
        if isinstance(headgear_data, dict):
            headgear = (
                headgear_data.get("description", "")
                or headgear_data.get("abbreviation", "")
                or headgear_data.get("type", "")
            )
        else:
            headgear = str(headgear_data) if headgear_data else ""

        # --- odds ---
        betting = ride.get("betting", {})
        odds_str = ""
        if isinstance(betting, dict):
            odds_str = betting.get("current_odds", "")
        # Racecard pages sometimes have bookmakerOdds
        if not odds_str:
            bk = ride.get("bookmakerOdds", {})
            if isinstance(bk, dict):
                odds_str = bk.get("fractional", "") or bk.get("decimal", "")
        odds = _parse_fractional_odds(odds_str)

        # --- result-specific fields ---
        finish_pos = 0
        finish_pos_label = ""
        won = 0
        lengths_behind = 0.0

        if is_result:
            _raw_fp = ride.get("finish_position", 0)
            finish_pos = _safe_int(_raw_fp)
            # Keep the raw label for non-finishers (PU, F, UR, BD, RR, WO, etc.)
            if finish_pos == 0 and _raw_fp not in (None, "", 0):
                finish_pos_label = str(_raw_fp).strip().upper()
            won = 1 if finish_pos == 1 else 0
            lengths_behind = _parse_lengths_behind(
                ride.get("finish_distance", "")
            )

        # --- lifetime stats ---
        lifetime = ride.get("horse_lifetime_stats", [])
        runs = wins = places = 0
        if lifetime and isinstance(lifetime, list) and len(lifetime) > 0:
            stat = lifetime[0]
            runs = _safe_int(stat.get("run_count", 0))
            wins = _safe_int(stat.get("win_count", 0))
            places = _safe_int(stat.get("place_count", 0))

        return {
            "race_id": race_id,
            "race_date": date_str,
            "off_time": off_time,
            "track": course_name,
            "region": region,
            "race_name": race_name,
            "race_class": race_class,
            "race_type": race_type,
            "distance_furlongs": distance_f,
            "going": going,
            "prize_money": prize_money,
            "num_runners": num_runners,
            "horse_name": horse_name,
            "horse_id": horse_id,
            "jockey": jockey,
            "trainer": trainer,
            "age": age,
            "sex": sex,
            "headgear": headgear,
            "weight_lbs": weight_lbs,
            "draw": draw,
            "form": form,
            "days_since_last_run": days_since,
            "odds": odds,
            "official_rating": official_rating,
            "finish_position": finish_pos,
            "finish_pos_label": finish_pos_label,
            "won": won,
            "lengths_behind": lengths_behind,
            "horse_runs": runs,
            "horse_wins": wins,
            "horse_places": places,
            "surface": surface_str,
            "handicap": 1 if has_handicap else 0,
        }


# =====================================================================
# High-level public functions
# =====================================================================

def scrape_results(
    days_back: int = 14,
    uk_only: bool = True,
    save: bool = True,
    max_races_per_day: int = 50,
) -> pd.DataFrame:
    """
    Scrape historical UK race results from Sporting Life.

    Args:
        days_back: Number of days of history to scrape.
        uk_only:   Only scrape UK & Ireland tracks.
        save:      Save to CSV in ``data/raw/``.
        max_races_per_day: Max races per day (rate limiting).

    Returns:
        DataFrame with all runner records.
    """
    scraper = SportingLifeScraper()
    all_runners: list[dict] = []

    logger.info(f"🌐 Scraping race results for the last {days_back} days...")

    for day_offset in range(days_back):
        date = datetime.now() - timedelta(days=day_offset + 1)
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"\n📅 {date_str} ...")

        try:
            race_links = scraper.get_results_urls(date_str, uk_only=uk_only)
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to get listing for {date_str}: {e}")
            continue

        if not race_links:
            logger.info(f"  No races found for {date_str}")
            continue

        race_links = race_links[:max_races_per_day]
        logger.info(f"  Scraping {len(race_links)} races...")

        # Fetch individual race pages in parallel
        def _fetch_one(info: dict) -> list[dict]:
            return scraper.scrape_race_result(info["url"], date_str)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_one, ri): ri for ri in race_links
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                ri = futures[future]
                try:
                    runners = future.result()
                    if runners:
                        all_runners.extend(runners)
                        logger.info(
                            f"    [{done}/{len(race_links)}] "
                            f"{ri.get('track', '?')} "
                            f"- {len(runners)} runners"
                        )
                    else:
                        logger.info(
                            f"    [{done}/{len(race_links)}] No runners parsed"
                        )
                except Exception as e:
                    logger.warning(f"    [{done}/{len(race_links)}] Error: {e}")
                    continue

    if not all_runners:
        logger.warning("❌ No results scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runners)
    logger.info(
        f"\n✅ Scraped {len(df)} runner records from "
        f"{df['race_id'].nunique()} races"
    )

    if save:
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        filepath = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"  Saved to {filepath}")

    return df


def scrape_todays_racecards(
    uk_only: bool = True,
    date_str: str = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Scrape today's racecards from Sporting Life.

    Args:
        uk_only:  Only scrape UK & Ireland tracks.
        date_str: Specific date (default: today).
        progress_callback: Optional ``callable(current, total, track)``
                           invoked after each race is scraped, useful
                           for UI progress bars.

    Returns:
        DataFrame with runner records (``finish_position`` = 0).
    """
    scraper = SportingLifeScraper()
    all_runners: list[dict] = []

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"🌐 Scraping racecards for {date_str}...")

    try:
        race_links = scraper.get_racecard_urls(date_str, uk_only=uk_only)
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to get racecard listing: {e}")
        return pd.DataFrame()

    if not race_links:
        logger.info("  No racecards found")
        return pd.DataFrame()

    logger.info(f"  Found {len(race_links)} races, scraping details...")

    # Fetch individual racecard pages in parallel
    def _fetch_card(info: dict) -> list[dict]:
        return scraper.scrape_racecard(info["url"], date_str)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_card, ri): ri for ri in race_links}
        done = 0
        for future in as_completed(futures):
            done += 1
            ri = futures[future]
            track_name = ri.get("track", "?")
            if progress_callback is not None:
                try:
                    progress_callback(done, len(race_links), track_name)
                except Exception:
                    pass
            try:
                runners = future.result()
                if runners:
                    all_runners.extend(runners)
                    logger.info(
                        f"    [{done}/{len(race_links)}] "
                        f"{track_name} "
                        f"- {len(runners)} runners"
                    )
                else:
                    logger.info(
                        f"    [{done}/{len(race_links)}] No runners parsed"
                    )
            except Exception as e:
                logger.warning(f"    [{done}/{len(race_links)}] Error: {e}")
                continue

    if not all_runners:
        logger.warning("❌ No racecards scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(all_runners)
    logger.info(
        f"\n✅ Scraped {len(df)} runners across "
        f"{df['race_id'].nunique()} races at "
        f"{df['track'].nunique()} venues"
    )

    return df



# =====================================================================
# Today's results (for live settlement)
# =====================================================================

def scrape_todays_results(
    uk_only: bool = True,
    date_str: str | None = None,
) -> pd.DataFrame:
    """Scrape results for today (or *date_str*) from Sporting Life.

    Unlike ``scrape_results`` which iterates over past days, this
    fetches only the single day requested — ideal for settling
    today's picks as races finish.

    Returns:
        DataFrame of runner records with ``finish_position``, ``won``,
        ``lengths_behind`` populated for completed races.  Races that
        haven't run yet will simply be absent from the results listing.
    """
    scraper = SportingLifeScraper()
    all_runners: list[dict] = []

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"🌐 Scraping today's results for {date_str} …")

    try:
        race_links = scraper.get_results_urls(date_str, uk_only=uk_only)
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to get results listing: {e}")
        return pd.DataFrame()

    if not race_links:
        logger.info("  No results available yet")
        return pd.DataFrame()

    logger.info(f"  Found {len(race_links)} completed races, scraping …")

    def _fetch_one(info: dict) -> list[dict]:
        return scraper.scrape_race_result(info["url"], date_str)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, ri): ri for ri in race_links}
        for future in as_completed(futures):
            try:
                runners = future.result()
                if runners:
                    all_runners.extend(runners)
            except Exception:
                continue

    if not all_runners:
        logger.info("  No result runners parsed")
        return pd.DataFrame()

    df = pd.DataFrame(all_runners)
    logger.info(
        f"  ✅ {len(df)} runners across "
        f"{df['race_id'].nunique()} completed races"
    )
    return df


# =====================================================================
# Integration helpers (match the collect_data interface)
# =====================================================================

def collect_scraped_data(
    days_back: int = 14,
    uk_only: bool = True,
) -> pd.DataFrame:
    """Drop-in replacement for ``collect_real_data()``."""
    return scrape_results(days_back=days_back, uk_only=uk_only, save=True)


def _racecard_cache_path(date_str: str | None = None) -> str:
    """Return path to the cached racecards file for *date_str*."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "racecards_cache",
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"racecards_{date_str}.csv")


def load_cached_racecards(date_str: str | None = None) -> pd.DataFrame | None:
    """Load cached racecards for *date_str*.  Returns ``None`` if no cache."""
    path = _racecard_cache_path(date_str)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                logger.info(f"📂 Loaded {len(df)} cached runners from {path}")
                return df
        except Exception as e:
            logger.warning(f"Cache read failed ({path}): {e}")
    return None


def save_racecards_cache(df: pd.DataFrame, date_str: str | None = None) -> None:
    """Persist racecards DataFrame to disk."""
    if df is None or df.empty:
        return
    path = _racecard_cache_path(date_str)
    df.to_csv(path, index=False)
    logger.info(f"💾 Saved {len(df)} runners to cache: {path}")


def get_scraped_racecards(
    date_str: str = None,
    uk_only: bool = True,
    progress_callback=None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return today's racecards, using a local cache when available.

    On the first call of the day the racecards are scraped from Sporting
    Life and saved to ``data/racecards_cache/racecards_YYYY-MM-DD.csv``.
    Subsequent calls return the cached file instantly unless
    *force_refresh* is ``True``.
    """
    if not force_refresh:
        cached = load_cached_racecards(date_str)
        if cached is not None:
            return cached

    df = scrape_todays_racecards(
        uk_only=uk_only, date_str=date_str,
        progress_callback=progress_callback,
    )
    if df is not None and not df.empty:
        save_racecards_cache(df, date_str)
    return df


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape UK horse racing data from Sporting Life"
    )
    parser.add_argument(
        "--mode",
        choices=["results", "racecards", "both"],
        default="both",
        help="What to scrape",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Days of history (results mode)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date for racecards (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Include non-UK/Ireland tracks",
    )

    args = parser.parse_args()
    uk_only = not args.all_tracks

    if args.mode in ("results", "both"):
        print(f"\n{'='*60}")
        print(f"  Scraping results (last {args.days_back} days)")
        print(f"{'='*60}\n")
        df = scrape_results(days_back=args.days_back, uk_only=uk_only)
        if not df.empty:
            print(f"\nResults summary:")
            print(f"  Records : {len(df)}")
            print(f"  Races   : {df['race_id'].nunique()}")
            print(f"  Tracks  : {df['track'].nunique()}")
            print(
                f"  Dates   : {df['race_date'].min()} → "
                f"{df['race_date'].max()}"
            )

    if args.mode in ("racecards", "both"):
        print(f"\n{'='*60}")
        print(f"  Scraping today's racecards")
        print(f"{'='*60}\n")
        df = scrape_todays_racecards(uk_only=uk_only, date_str=args.date)
        if not df.empty:
            print(f"\nRacecard summary:")
            print(f"  Runners : {len(df)}")
            print(f"  Races   : {df['race_id'].nunique()}")
            print(f"  Tracks  : {', '.join(df['track'].unique())}")
