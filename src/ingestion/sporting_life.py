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

Historical results are collected through ``src.ingestion.results``, which keeps
the resumable ledger. Today's cards come from ``scrape_todays_racecards()``.
"""

import html
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock, Semaphore
from typing import Optional

import numpy as np
import pandas as pd
import requests

import config
from pathlib import Path

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

# Structured casualty enums verified in cached full Sporting Life results.
# Unknown reasons remain unknown; never infer a result from profile prose.
_NONFINISH_REASON_LABELS = {
    "PulledUp": "PU",
    "Fell": "F",
    "UnseatedRider": "UR",
    "Refused": "R",
}


def _official_finish_value(ride: dict, num_runners: int = 0):
    """Use an explicit final placing, preserving legacy raw nonfinish labels.

    A malformed nonempty amendment cannot silently restore the original winner.
    Only positive integral amended placings have a verified source contract.
    """
    original = ride.get("finish_position", 0)
    amended = ride.get("amended_position")
    if amended is None or amended == "":
        return original
    try:
        value = float(amended)
    except (TypeError, ValueError):
        raise ValueError("Invalid amended finishing position") from None
    if (isinstance(amended, bool) or not math.isfinite(value) or value <= 0
            or not value.is_integer() or (num_runners > 0 and value > num_runners)):
        raise ValueError("Invalid amended finishing position")
    return int(value)

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
    "great britain", "united kingdom", "eire",
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

def _verdict_text(summary: dict) -> Optional[str]:
    """The race verdict as plain text (the source marks selections with <b>)."""
    raw = summary.get("verdict")
    if not isinstance(raw, str) or not raw.strip():
        return None
    inline = re.sub(r"</?(?:b|i|em|strong)>", "", raw, flags=re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", inline))).strip() or None


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
    text = str(dist_str or "").strip().lower()
    for symbol, fraction in (("¼", .25), ("½", .5), ("¾", .75)):
        text = re.sub(r"(\d*)" + symbol,
                      lambda m: str(float(m.group(1) or 0) + fraction), text)
    number = r"(\d+(?:\.\d+)?)"
    match = re.fullmatch(rf"(?:{number}\s*m(?:iles?)?\s*)?(?:{number}\s*f(?:urlongs?)?\s*)?(?:{number}\s*y(?:ards?)?\s*)?", text)
    if not match or not any(match.groups()):
        return float("nan")
    miles, furlongs, yards = (float(v or 0) for v in match.groups())
    total = miles * 8 + furlongs + yards / 220.
    # The provider renders UK/Irish distances in miles/furlongs/yards. An
    # unrecognised token is unknown, never an invented ten-furlong race.
    return round(total, 2) if math.isfinite(total) and 1 <= total <= 64 else float("nan")


def _parse_reported_lbs(value) -> float | None:
    """A reported claim/penalty; absence is unknown, never an observed zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= 30 and number.is_integer() else None


def _parse_weight_to_lbs(weight_str: str) -> float:
    """
    Convert weight like ``'11-9'`` (11 st 9 lb) to total pounds.
    1 stone = 14 lb.
    """
    text = str(weight_str or "").strip().lower()
    text = re.sub(r"(\d*)½", lambda m: str(float(m.group(1) or 0) + .5), text)
    match = re.fullmatch(r"(\d+)\s*(?:[-–—−]|st(?:ones?)?\s*)\s*(\d+(?:\.\d+)?)\s*(?:lbs?)?", text)
    if match:
        stones, pounds = (float(v) for v in match.groups())
        total = stones * 14 + pounds if pounds < 14 else float("nan")
    elif re.fullmatch(r"\d+(?:\.\d+)?\s*(?:lbs?)?", text):
        total = float(re.sub(r"\s*lbs?$", "", text))
    else:
        return float("nan")
    # Broad physical sanity bounds, not a typical-weight imputation. The
    # source itself contains impossible "0-1" and "0-5" historical weights.
    return total if math.isfinite(total) and 50 <= total <= 220 else float("nan")


def _parse_fractional_odds(odds_str: str) -> float:
    """
    Convert fractional odds like ``'5/2'``, ``'11/4f'``, ``'evs'``
    to decimal odds (e.g. 5/2 → 3.5).
    """
    if not odds_str:
        return float("nan")

    odds_str = str(odds_str).strip().lower()
    odds_str = odds_str.replace("f", "").replace("j", "").strip()

    if odds_str in ("evs", "evens", "ev"):
        return 2.0

    match = re.fullmatch(r'(\d+)\s*/\s*(\d+)', odds_str)
    if match:
        num, den = int(match.group(1)), int(match.group(2))
        if den > 0:
            return round(num / den + 1, 2)

    try:
        value = float(odds_str)
        return value if np.isfinite(value) and value > 1 else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def _parse_lengths_behind(dist_str: str) -> float:
    """
    Parse finish distance strings like ``'8 ½'``, ``'nk'``, ``'1¼'``
    into a numeric gap to the PREVIOUS finisher. This is not a cumulative
    distance behind the winner; data_processor derives that separately.
    """
    if not dist_str:
        return float('nan')

    s = str(dist_str).strip().lower()

    # Named margins (no numeric part). Match the whole token, longest code
    # first, so "snk" isn't shadowed by "nk" and a stray substring never
    # matches (the old `if k in s` returned nk=0.3 for "snk").
    named = {
        "dht": 0.0, "nse": 0.05, "shd": 0.1, "sh": 0.1,
        "hd": 0.2, "nk": 0.3, "sn": 0.1, "snk": 0.2,
    }
    s_compact = s.replace(" ", "")
    for k in sorted(named, key=len, reverse=True):
        if s_compact == k:
            return named[k]

    # Unicode fractions
    frac_map = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 0.33, "⅔": 0.67}
    total = 0.0
    for fchar, fval in frac_map.items():
        if fchar in s:
            total += fval
            s = s.replace(fchar, "")

    # Accept decimal and ASCII fraction variants too. An unrecognised token
    # is unknown, never a fabricated zero-length gap/dead heat.
    s = s.strip()
    if s:
        try:
            from fractions import Fraction
            parts = s.split()
            if len(parts) > 2:
                return float("nan")
            total += sum(float(Fraction(part)) for part in parts)
        except (ValueError, ZeroDivisionError):
            return float("nan")
    elif total == 0.0:
        return float("nan")
    return total if np.isfinite(total) and total >= 0 else float("nan")


_WINNING_TIME_RE = re.compile(r"(?:(\d+)\s*m)?\s*([\d.]+)\s*s", re.I)


def _parse_winning_time_secs(raw) -> float:
    """Parse a Sporting Life winning time into seconds.

    ``'1m 13.24s'`` -> 73.24, ``'59.34s'`` -> 59.34. Empty/garbage -> NaN.
    """
    if not raw:
        return float('nan')
    m = _WINNING_TIME_RE.search(str(raw).strip())
    if not m:
        return float('nan')
    mins = float(m.group(1)) if m.group(1) else 0.0
    secs = float(m.group(2))
    v = mins * 60.0 + secs
    return v if 5.0 < v < 1200.0 else float('nan')


_SOURCE_RACE_CODES = {"FLAT": "Flat", "AW FLAT": "Flat", "CHASE": "Chase",
    "STEEPLECHASE": "Chase", "HURDLE": "Hurdle", "HURDL": "Hurdle",
    "NH FLAT": "Flat", "NHF": "Flat", "INH FLAT": "Flat", "BUMPER": "Flat"}
_NAMED_CHASES = re.compile(r"\b(?:punchestown|irish)\s+gold\s+cup\b", re.I)
_RACE_DISCIPLINE = re.compile(r"\b(?:national\s+hunt\s+flat|[i.]?n[.]?h[.]?\s+flat|flat\s+race|bumper|hurdle|steeplechase|chase|stakes)\b", re.I)


def _infer_race_type(race_name: str, run_type: str = "") -> str:
    """Prefer an explicit source code, then match actual discipline words.

    Sponsor words such as Purchase/Chipchase cannot imply fences. Where a
    title advertises another race before its own kind, the final kind wins.
    The two named Gold Cups omit 'Chase' in this provider's titles; their
    discipline is confirmed by the cached, dated public performances in025.
    This does not extend to the Ascot Gold Cup or any generic 'Gold Cup'.
    Keep bumpers in the existing coarse Flat label until a separate feature
    representation demonstrates a gain.
    """
    source = re.sub(r"\s+", " ", str(run_type or "").upper().replace(".", "").strip())
    if source in _SOURCE_RACE_CODES:
        return _SOURCE_RACE_CODES[source]
    name = str(race_name or "")
    matches = []
    for match in _RACE_DISCIPLINE.finditer(name):
        kind, tail = match.group().lower(), name[match.end():]
        # Event-specific source codes in032 confirm these sponsor phrases.
        # Ignore the sponsor occurrence, leaving any later actual Chase intact.
        if kind == "chase" and (
                re.match(r"\s+(?:day|distillery)\b", tail, re.I)
                or re.search(r"\b(?:cruises|hullabaloos)\s*$", name[:match.start()], re.I)):
            continue
        # A rider restriction describes the jockey, not an NH Flat race.
        if re.fullmatch(r"flat\s+race", kind, re.I) and re.match(r"\s+jockeys?\b", tail, re.I):
            continue
        matches.append(match)
    if matches:
        kind = matches[-1].group().lower()
        if kind in ("chase", "steeplechase"):
            return "Chase"
        if kind == "hurdle":
            return "Hurdle"
        return "Flat"
    if _NAMED_CHASES.search(name):
        return "Chase"
    return "Flat"


#: Sporting Life ``race_stage`` values for a listed race that has no result.
NOT_RUN_STAGES = frozenset({"ABANDONED", "RACEVOID", "DORMANT"})


class RaceNotRun(Exception):
    """A listed race with no result to store: abandoned, void, never run, or
    an Arabian race this dataset excludes. Raised only on request, so callers
    can tell "nothing to store" apart from "the fetch or the parse failed"."""

    def __init__(self, reason: str, url: str = ""):
        super().__init__(f"{reason}: {url}")
        self.reason = reason


def _is_arabian_race(race_name: str, course_name: str, runners: list[dict]) -> bool:
    """True if a race is a Purebred Arabian race (not Thoroughbred).

    Arabian races are a different breed with different form dynamics and must
    not pollute the Thoroughbred dataset. Sporting Life suffixes Arabian horses
    with ``(A)`` and broadcasts some on feed-only channels mislabelled as the
    course (e.g. ``"RMG Extras"`` / ``"TRP Extras"``).
    """
    rn = str(race_name or "").lower()
    if str(course_name or "").strip().endswith("Extras"):
        return True
    if "arab race" in rn or "(arab" in rn:
        return True
    return any(
        str(r.get("horse_name", "")).strip().endswith("(A)") for r in runners
    )


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
        if long_name:
            return long_name.strip().lower() in UK_IRE_COUNTRIES

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
        self.raw_store_path = None  # enabled by the source orchestrator
        self.offline = False
        self.empty_result_dates = set()
        self.result_listing_evidence = {}

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

    def _get(self, url: str, *, max_retries: int = 4) -> requests.Response:
        """Rate-limited GET with retry/backoff (thread-safe).

        Retries transient failures — timeouts, connection errors and HTTP
        429/5xx — with exponential backoff. Without this, a single transient
        failure silently dropped a race during backfill (the cause of the
        Irish/late-meeting gaps the Betfair cross-reference exposed).
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            with self._req_sem:
                with self._req_lock:
                    elapsed = time.time() - self._last_req_time
                    if elapsed < REQUEST_DELAY:
                        time.sleep(REQUEST_DELAY - elapsed)
                    self._last_req_time = time.time()
                logger.debug(f"  GET {url}")
                try:
                    resp = self.session.get(url, timeout=30)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        raise requests.HTTPError(
                            f"HTTP {resp.status_code}", response=resp
                        )
                    resp.raise_for_status()
                    return resp
                except requests.RequestException as exc:
                    if (getattr(exc, "response", None) is not None
                            and 400 <= exc.response.status_code < 500
                            and exc.response.status_code not in {408, 425, 429}):
                        raise
                    last_exc = exc
            # backoff outside the semaphore so we don't hold the rate-limit slot
            if attempt < max_retries - 1:
                backoff = 2.0 ** attempt
                logger.warning(
                    "GET failed (%s), retry %d/%d in %.1fs: %s",
                    type(last_exc).__name__, attempt + 1, max_retries - 1, backoff, url,
                )
                time.sleep(backoff)
        raise last_exc  # type: ignore[misc]

    def _fetch_next_data(self, url: str) -> Optional[dict]:
        """Fetch a page and return the embedded ``__NEXT_DATA__`` dict."""
        self._local.payload_digest = None
        if self.raw_store_path:
            from src.ingestion.storage import RawPayloadStore
            store = getattr(self._local, "raw_store", None)
            if store is None or store.path != Path(self.raw_store_path):
                store = RawPayloadStore(self.raw_store_path)
                self._local.raw_store = store
        else:
            store = None
        if self.offline:
            body = store.latest(url) if store else None
            if body is None:
                raise FileNotFoundError(f"No archived Sporting Life response for {url}")
            page_text = body.decode("utf-8")
            self._local.payload_digest = hashlib.sha256(body).hexdigest()
        else:
            try:
                resp = self._get(url)
            except requests.RequestException as error:
                if store and getattr(error, "response", None) is not None:
                    store.put(url, error.response.status_code, error.response.content)
                raise
            page_text = resp.text
            if store:
                self._local.payload_digest = store.put(url, resp.status_code, resp.content)
        m = _NEXT_DATA_RE.search(page_text)
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse __NEXT_DATA__ from {url}")
                return None
            # A redirected/malformed horse page must not be cached under the
            # requested horse's identity (including by the pedigree backfill).
            from urllib.parse import urlsplit
            requested = re.search(r"/racing/profiles/horse/(\d+)(?:/|$)", urlsplit(url).path)
            if requested:
                profile = data.get("props", {}).get("pageProps", {}).get("profile") or {}
                returned = (profile.get("horse_reference") or {}).get("id")
                if str(returned) != requested.group(1):
                    logger.warning("Profile identity mismatch for horse %s (returned %s)", requested.group(1), returned)
                    return None
            return data
        return None

    # ==================================================================
    # RESULTS — listing page → race URLs
    # ==================================================================

    def get_results_urls(
        self, date_str: str, uk_only: bool = True, *, countries=None,
    ) -> list[dict]:
        """
        Return race-result URL info for every race on *date_str*.

        Each item is a dict with keys:
        ``url``, ``track``, ``race_name``, ``time``, ``race_id``.
        """
        url = f"{SPORTING_LIFE_BASE}/racing/results/{date_str}"
        self.empty_result_dates.discard(date_str)
        self.result_listing_evidence.pop(date_str, None)
        data = self._fetch_next_data(url)
        if not data:
            logger.warning(f"No __NEXT_DATA__ found for {url}")
            return []

        races: list[dict] = []
        evidence = {}
        try:
            if data["props"]["pageProps"].get("hasError"):
                raise ValueError("Source reported an error for the results listing")
            meetings = data["props"]["pageProps"].get("meetings", [])
            if data["props"]["pageProps"].get("meetings") == []:
                self.empty_result_dates.add(date_str)
            for meeting in meetings:
                ms = meeting.get("meeting_summary", {})
                country = ((ms.get("course") or {}).get("country") or {}).get("long_name", "")
                if countries is not None:
                    if country.strip().lower() not in countries:
                        continue
                elif uk_only and not _is_uk_ire_meeting(meeting):
                    continue
                meeting_date = str(ms.get("date") or "")[:10]
                if meeting_date and meeting_date != date_str:
                    raise ValueError("Results listing belongs to another meeting date")
                course_name = ms.get("course", {}).get("name", "")
                course_slug = _slugify(course_name)

                for race in meeting.get("races", []):
                    ref = race.get("race_summary_reference", {})
                    race_id = ref.get("id", "")
                    race_name = race.get("name", "")
                    race_slug = _slugify(race_name) if race_name else ""
                    off_time = race.get("time", "")
                    source_date = str(race.get("date") or meeting_date)[:10]
                    if source_date and source_date != date_str:
                        raise ValueError("Results listing belongs to another race date")

                    if not race_id or not course_slug:
                        continue
                    if str(race_id) in evidence:
                        raise ValueError("Results listing has duplicate race identities")
                    is_arabian = _is_arabian_race(race_name, course_name, [])
                    evidence[str(race_id)] = {
                        "source_date": source_date, "source_url": url,
                        "payload_sha256": getattr(self._local, "payload_digest", None),
                        "reason": "ARABIAN" if is_arabian else str(race.get("race_stage") or "").upper(),
                        "has_results": bool(race.get("top_horses") or race.get("winning_time")),
                    }
                    # The page scraper rejects these anyway; dropping them
                    # here keeps them out of the sync ledger, where they
                    # would otherwise stay "pending" and be refetched forever.
                    if is_arabian:
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
                        "country": country,
                    })
        except (KeyError, TypeError) as e:
            logger.warning(f"Error parsing results listing: {e}")
            return []

        self.result_listing_evidence[date_str] = evidence
        logger.info(f"  Found {len(races)} race result links for {date_str}")
        return races

    # ==================================================================
    # RESULTS — individual race page → runner records
    # ==================================================================

    def scrape_race_result(
        self, race_url: str, date_str: str = "", *, raise_if_not_run: bool = False,
    ) -> list[dict]:
        """
        Scrape a single race-result page and return pipeline-ready dicts.

        Returns ``[]`` when there is nothing to store. With
        ``raise_if_not_run=True`` a race that was abandoned, voided, never run
        or is Arabian raises :class:`RaceNotRun` instead, so the sync ledger
        can close it rather than retry it forever.
        """
        data = self._fetch_next_data(race_url)
        if not data:
            logger.warning(f"No __NEXT_DATA__ for result: {race_url}")
            return []

        runners: list[dict] = []
        race_stage = ""
        try:
            page_props = data["props"]["pageProps"]
            race_data = page_props.get("race", {})
            if not race_data:
                return []

            summary = race_data.get("race_summary", {})
            expected = re.search(r"/racing/results/(\d{4}-\d{2}-\d{2})/[^/]+/(\d+)(?:[/?#]|$)", race_url)
            source_identity = str((summary.get("race_summary_reference") or {}).get("id", ""))
            expected_date = date_str or (expected.group(1) if expected else "")
            if expected and date_str and date_str != expected.group(1):
                raise ValueError("Requested result date disagrees with its URL")
            if expected and source_identity != expected.group(2):
                raise ValueError("Source result belongs to another race identity")
            source_date = str(summary.get("date") or "")[:10]
            if expected_date and source_date and source_date != expected_date:
                raise ValueError("Source result belongs to another race date")
            race_stage = str(summary.get("race_stage") or "").upper()

            # A first-past-post winner can already exist while an inquiry is
            # pending. Do not let that provisional order close the sync job.
            # Legacy payloads without a stage retain their existing contract.
            if race_stage in NOT_RUN_STAGES:
                if raise_if_not_run:
                    raise RaceNotRun(race_stage, race_url)
                return []
            if race_stage and race_stage != "WEIGHEDIN":
                logger.info("Result is not yet weighed in (%s): %s", race_stage, race_url)
                return []

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
            winning_time_secs = _parse_winning_time_secs(
                summary.get("winning_time", "")
            )
            off_time_actual = summary.get("off_time", "")
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
            race_type = _infer_race_type(race_name, summary.get("run_type", ""))
            prize_money = _extract_prize_money(race_data.get("prizes"))

            is_uk = page_props.get("isUkorIreMeeting", True)
            region = "UK" if is_uk else "INT"

            # Amended placings and physical arrival gaps describe different
            # orderings. Never accumulate the original gaps in the final order.
            rides = race_data.get("rides", [])
            # A disqualified horse can be placed after all declarations,
            # including withdrawn horses (e.g. eighth among seven starters).
            # Keep the source's explicit ordinal, bounded by the declaration
            # roster; field size continues to mean actual starters.
            position_limit = max(num_runners, len(rides))
            starters = [r for r in rides if re.sub(r"[^A-Z]", "", str(r.get("ride_status", "")).upper()) not in {"NONRUNNER", "WITHDRAWN"}]
            finals = [_safe_int(_official_finish_value(r, position_limit)) for r in starters]
            beyond = [i for i, value in enumerate(finals) if value > num_runners > 0]
            if beyond:
                positive = sorted(value for value in finals if value > 0)
                terminal = (len(beyond) == 1 and finals[beyond[0]] == len(rides)
                    and positive == [*range(1, len(positive)), len(rides)]
                    and _safe_int(starters[beyond[0]].get("finish_position")) < finals[beyond[0]])
                if not terminal:
                    raise ValueError("Invalid amended finishing position beyond the started field")
            positions_changed = any(
                _safe_int(_official_finish_value(ride, position_limit)) != _safe_int(ride.get("finish_position", 0))
                for ride in rides
                if re.sub(r"[^A-Z]", "", str(ride.get("ride_status", "")).upper()) not in {"NONRUNNER", "WITHDRAWN"}
            )
            # ---- parse each ride (runner) ----
            for ride in rides:
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
                    winning_time_secs=winning_time_secs,
                    off_time_actual=off_time_actual,
                    result_positions_changed=positions_changed,
                    result_position_limit=position_limit,
                    verdict=_verdict_text(summary),
                )
                if row:
                    prizes = race_data.get("prizes") or []
                    if isinstance(prizes, dict):
                        prizes = prizes.get("prize", [])
                    text = " ".join(str(p.get("prize", "")) for p in prizes if isinstance(p, dict))
                    unit = re.search(r"\b(GBP|EUR)\b", text)
                    row["prize_currency"] = unit.group(1) if unit else None
                    runners.append(row)

        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Error parsing result JSON: {e}")

        # Purebred Arabian races must not pollute the Thoroughbred dataset.
        if runners and _is_arabian_race(race_name, course_name, runners):
            logger.info("Skipping Purebred Arabian race: %s", race_url)
            if raise_if_not_run:
                raise RaceNotRun("ARABIAN", race_url)
            return []

        # Some Sporting Life "result" URLs occasionally resolve to an
        # unresolved or non-settled page shape (for example entries-like
        # pages with all runners at finish_position=0 and blank jockeys).
        # Those must not be stored as historical results.
        settled_count = sum(
            1 for row in runners
            if _safe_int(row.get("finish_position", 0)) > 0
        )
        if settled_count == 0:
            if raise_if_not_run and race_stage in NOT_RUN_STAGES:
                raise RaceNotRun(race_stage, race_url)
            if runners:
                logger.warning(
                    "Skipping unresolved result page with no settled finishers: %s",
                    race_url,
                )
            return []

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
            race_type = _infer_race_type(race_name, summary.get("run_type", ""))
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
                    verdict=_verdict_text(summary),
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
        winning_time_secs: float = float('nan'),
        off_time_actual: str = "",
        result_positions_changed: bool = False,
        result_position_limit: int = 0,
        verdict: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Convert a single Sporting Life *ride* object into a flat dict
        matching the pipeline's expected columns.
        """
        # Drop non-runners (withdrawn horses that never went to post). Sporting
        # Life uses "NONRUNNER" on result pages but "NON-RUNNER" elsewhere, so
        # normalise away separators rather than matching one exact spelling —
        # the old exact "NON-RUNNER" check silently let every result-page
        # non-runner through as a finish_position=0 row.
        status_norm = re.sub(r"[^A-Z]", "", str(ride.get("ride_status", "")).upper())
        # WITHDRAWN is a separate source status on some result/card pages.
        # Keep started non-finishers (RUNNER + Fell/PulledUp/etc.) in history.
        if status_norm in {"NONRUNNER", "WITHDRAWN"}:
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
        # ``draw`` is the starting-stall number (``draw_number``), NOT the
        # saddle-cloth number. Jumps races have no stalls so ``draw_number`` is
        # null -> 0. (Historically this read ``cloth_number`` by mistake, which
        # corrupted every draw-bias feature; backfilled via
        # scripts/backfill_draw.py.)
        draw = _safe_int(ride.get("draw_number", 0))
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

        # --- wind surgery ---
        # ``medication`` is ``[{"short_name": "WS"}]`` when the horse has had
        # a declared wind operation since its last run (BHA declarations began
        # 2018; the flag appears on both racecard and historical result pages).
        # Lower-case "ws" occurs on some result pages, so compare folded.
        wind_surgery = 0
        med_list = ride.get("medication") or []
        if isinstance(med_list, list):
            for med in med_list:
                if (
                    isinstance(med, dict)
                    and str(med.get("short_name", "")).strip().lower() == "ws"
                ):
                    wind_surgery = 1
                    break

        # --- odds ---
        # bookmakerOdds is a list of per-bookmaker prices; prefer the
        # entry flagged ``bestOdds`` which matches what Sporting Life
        # displays on the racecard page.  Fall back to betting.current_odds.
        odds_str = ""
        bk_list = ride.get("bookmakerOdds", [])
        if isinstance(bk_list, list) and bk_list:
            best = next((b for b in bk_list if b.get("bestOdds")), None)
            if best is None:
                best = bk_list[0]
            odds_str = best.get("fractionalOdds", "") or str(best.get("decimalOdds", ""))
        if not odds_str:
            betting = ride.get("betting", {})
            if isinstance(betting, dict):
                odds_str = betting.get("current_odds", "")
        odds = _parse_fractional_odds(odds_str)

        # --- result-specific fields ---
        finish_pos = 0
        finish_pos_label = ""
        won = 0
        lengths_behind = float('nan')
        amendment_json = None

        if is_result:
            original_fp = ride.get("finish_position", 0)
            _raw_fp = _official_finish_value(ride, result_position_limit or num_runners)
            finish_pos = _safe_int(_raw_fp)
            changed_order = result_positions_changed or finish_pos != _safe_int(original_fp)
            if changed_order:
                # Preserve physical source evidence separately from model inputs.
                amendment_json = json.dumps({
                    "original_finish_position": original_fp,
                    "amended_position": ride.get("amended_position"),
                    "original_finish_distance": ride.get("finish_distance"),
                    "original_winning_time_secs": (float(winning_time_secs)
                        if np.isfinite(winning_time_secs) else None),
                    "physical_margins_and_time_withheld": True,
                }, sort_keys=True, allow_nan=False)
                winning_time_secs = float('nan')
            if finish_pos == 0:
                # Some feeds put the label in finish_position; full result
                # pages instead use numeric 0 plus casualty.reason.
                raw_label = "" if _raw_fp is None else str(_raw_fp).strip().upper()
                if raw_label not in {"", "0", "0.0"}:
                    finish_pos_label = raw_label
                else:
                    casualty = ride.get("casualty")
                    reason = casualty.get("reason") if isinstance(casualty, dict) else None
                    if isinstance(reason, str):
                        finish_pos_label = _NONFINISH_REASON_LABELS.get(reason.strip(), "")
            won = 1 if finish_pos == 1 else 0
            if finish_pos >= 1:
                # Only parse lengths behind for horses that finished
                if not changed_order:
                    lengths_behind = _parse_lengths_behind(ride.get("finish_distance", ""))
                # Winner has 0.0 lengths behind (parse returns NaN for empty string)
                if finish_pos == 1 and np.isnan(lengths_behind):
                    lengths_behind = 0.0

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
            "wind_surgery": wind_surgery,
            "weight_lbs": weight_lbs,
            # Keep allocated weight, rider claim and penalty in separate units.
            # A penalty may already be included in the allocated weight.
            "reported_jockey_claim_lbs": _parse_reported_lbs(ride.get("jockey_claim")),
            "weight_penalty_lbs": _parse_reported_lbs(ride.get("weight_penalty")),
            "draw": draw,
            "form": form,
            "days_since_last_run": days_since,
            "odds": odds,
            "official_rating": official_rating,
            "finish_position": finish_pos,
            "finish_pos_label": finish_pos_label,
            "won": won,
            "lengths_behind": lengths_behind,  # legacy raw name: gap to preceding finisher
            "horse_runs": runs,
            "horse_wins": wins,
            "horse_places": places,
            "surface": surface_str,
            "handicap": 1 if has_handicap else 0,
            # Race-level result fields (same for every runner). off_time stays the
            # SCHEDULED time (used by merge keys); these are the actual values.
            "winning_time_secs": winning_time_secs,
            "off_time_actual": off_time_actual,
            # Pre-race text, stored apart from results (runner_text_observations)
            # with when it was seen, so card and result copies can be compared.
            "sl_commentary": ride.get("commentary") if isinstance(ride.get("commentary"), str) else None,
            "sl_verdict": verdict,
            # Event-specific result evidence, never current profile commentary.
            "ride_description": (ride.get("ride_description")
                                 if is_result and isinstance(ride.get("ride_description"), str) else None),
            "nonfinish_reason": ((ride.get("casualty") or {}).get("reason")
                                 if is_result and isinstance(ride.get("casualty"), dict)
                                 and isinstance(ride["casualty"].get("reason"), str) else None),
            "result_amendment_json": amendment_json,
        }


# =====================================================================
# High-level public functions
# =====================================================================

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

    Unlike the historical results sync, this
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
# Racecard cache
# =====================================================================

def _racecard_cache_path(date_str: str | None = None) -> str:
    """Return path to the cached racecards file for *date_str*."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cache_dir = os.path.join(
        config.DATA_DIR, "racecards_cache",
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
    from src.database import write_csv_atomically
    write_csv_atomically(df, path)
    logger.info(f"💾 Saved {len(df)} runners to cache: {path}")


def enrich_racecard_pedigree(df, *, scraper=None, cache_dir=None, known=None):
    """Supply pre-race immutable breeding for horses missing from local results.

    The historical pedigree backfill only queues horses already in results.
    A first local visitor must not wait until after its debut for its pedigree.
    This card-only cache never updates historical results/pedigree tables and
    never imports current form, lifetime totals, ratings, age or trainer data.
    """
    from pathlib import Path
    from datetime import timezone
    import tempfile
    from src.database import load_pedigree
    from src.ingestion.pedigree import fetch_horse_pedigree
    from src.utils import normalise_horse_key

    required = {"horse_id", "horse_name", "race_date"}
    if df is None or df.empty or not required.issubset(df):
        return df
    out = df.copy()
    parents = ("sire", "dam", "damsire")
    for parent in parents:
        if parent not in out:
            out[parent] = pd.Series(None, index=out.index, dtype=object)
    def present(values):
        return values.notna() & ~values.astype(str).str.strip().str.lower().isin(["", "nan", "none", "<na>"])
    ids = pd.to_numeric(out.horse_id, errors="coerce")
    ids = ids.where(ids.gt(0) & ids.mod(1).eq(0)).astype("Int64").astype("string")
    if known is None:
        try:
            known = load_pedigree()
        except Exception as error:
            logger.warning("Local pedigree lookup unavailable: %s", error)
            known = pd.DataFrame()
    if not known.empty:
        lookup = known.copy()
        lookup["horse_id"] = pd.to_numeric(lookup.horse_id, errors="coerce").astype("Int64").astype("string")
        lookup = lookup.set_index("horse_id")
        for parent in parents:
            out[parent] = out[parent].where(present(out[parent]), ids.map(lookup[parent]))
    missing = ~pd.concat([present(out[c]) for c in parents], axis=1).all(axis=1)
    directory = Path(cache_dir or Path(config.DATA_DIR) / "racecard_pedigree")
    if missing.any():
        directory.mkdir(parents=True, exist_ok=True)
    for identity in ids.loc[missing].dropna().unique():
        rows = ids.eq(identity).fillna(False)
        expected_names = out.loc[rows, "horse_name"].map(normalise_horse_key).unique()
        if len(expected_names) != 1 or not expected_names[0]:
            continue
        destination = directory / f"{identity}.json"
        payload = None
        if destination.exists():
            try:
                payload = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = None
        def matches(value):
            return (isinstance(value, dict) and str(value.get("horse_id")) == identity
                    and normalise_horse_key(value.get("horse_name", "")) == expected_names[0])
        if not matches(payload):
            try:
                if scraper is None:
                    scraper = SportingLifeScraper()
                fetched = fetch_horse_pedigree(scraper, identity)
            except Exception as error:
                logger.warning("Card pedigree unavailable for horse %s: %s", identity, error)
                continue
            if not matches(fetched):
                continue
            payload = {k: fetched.get(k) for k in ("horse_id", "horse_name", *parents)}
            payload.update(source_url=f"{SPORTING_LIFE_BASE}/racing/profiles/horse/{identity}",
                           fetched_at=datetime.now(timezone.utc).isoformat())
            # Look-ahead cards can contain the same horse on several dates.
            # Independent readers must not race over one shared .tmp filename.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=f".{identity}-", suffix=".tmp", delete=False) as stream:
                json.dump(payload, stream)
                temporary = Path(stream.name)
            try:
                # Windows can briefly refuse a replacement while another
                # look-ahead card publishes/reads the same horse's cache.
                # The fetched identity-checked parents remain usable even
                # if this optional cache stays locked after bounded retries.
                for attempt in range(4):
                    try:
                        temporary.replace(destination)
                        break
                    except PermissionError as error:
                        if attempt == 3:
                            logger.warning("Could not publish card pedigree cache for %s: %s", identity, error)
                        else:
                            time.sleep(.02 * 2**attempt)
            finally:
                temporary.unlink(missing_ok=True)
        for parent in parents:
            if payload.get(parent):
                out.loc[rows & ~present(out[parent]), parent] = payload[parent]
    return out


def get_scraped_racecards(
    date_str: str = None,
    uk_only: bool = True,
    progress_callback=None,
    force_refresh: bool = False,
    cache_ttl_seconds: int = 900,
) -> pd.DataFrame:
    """
    Return today's racecards, using a local cache when available.

    On the first call of the day the racecards are scraped from Sporting
    Life and saved to ``data/racecards_cache/racecards_YYYY-MM-DD.csv``.
    Subsequent calls reuse the cached card unless *force_refresh* is ``True``.
    Missing immutable pedigree may need one profile request per horse, cached
    separately so a first local runner receives the same information as training.
    """
    path = _racecard_cache_path(date_str)
    fresh = os.path.exists(path) and time.time()-os.path.getmtime(path) < cache_ttl_seconds
    if not force_refresh and fresh:
        cached = load_cached_racecards(date_str)
        if cached is not None:
            return enrich_racecard_pedigree(cached)

    df = scrape_todays_racecards(
        uk_only=uk_only, date_str=date_str,
        progress_callback=progress_callback,
    )
    if df is not None and not df.empty:
        # Keep the morning text as a pre-race observation; it is not a model
        # input, so it stays out of the cached card.
        from src.database import _TEXT_FIELDS, record_runner_text
        try:
            record_runner_text(df, "racecard")
        except Exception as exc:  # never lose a card to the research store
            logger.warning("Could not record racecard commentary: %s", exc)
        df = df.drop(columns=list(_TEXT_FIELDS), errors="ignore")
        save_racecards_cache(df, date_str)
        # Return what the cache returns. A fresh scrape holds horse_id as text;
        # the CSV round trip (and the training store) hold integers, and the
        # store's carried history is keyed by those integers. Serving a fresh
        # scrape silently dropped every horse's history from 25 of the 149
        # model features (2026-10-13).
        reloaded = load_cached_racecards(date_str)
        if reloaded is not None:
            return enrich_racecard_pedigree(reloaded)
    return enrich_racecard_pedigree(df)
