"""
At The Races Sectionals Scraper
===============================
Fetches per-runner sectional, stride, jumping and distance-travelled data
from At The Races (``attheraces.com``). It retains per-furlong split times,
running totals, pace classifications, sectional speeds (mph), finishing
speed %, an efficiency grade/score, energy distribution and stride
metrics, plus the in-running comment and a sectional-adjusted placing.

How it works
------------
ATR can return a JavaScript "Client Challenge" even with browser headers.
Challenged responses are retryable blocks and never confirmed absence.

1. GET ``/results/{DD-Month-YYYY}`` once per day/session. Each result article
   pairs its exact racecard URL with ATR's native race ID. Fall back to the
   racecard only when the index is genuinely unavailable or lacks that race.
2. GET the public fragments advertised by the current sectional tabs::

       /ajax/sectionals-razor/{raceid}/times
       /ajax/sectionals-razor/{raceid}/tools
       /ajax/sectionals-razor/{raceid}/stride
       /ajax/sectionals-razor/{raceid}/jumpsdata
       /ajax/sectionals-razor/{raceid}/distanceran

3. Parse one row per runner and cache to parquet, keyed (like RTV) on
   ``(race_date, track, off_time, horse_name)`` so it merges onto the
   main results frame with the shared normalisers in :mod:`src.utils`.

Usage::

    from src.ingestion.attheraces import fetch_atr_sectionals, backfill_atr_sectionals

    rows = fetch_atr_sectionals("2026-06-10", "Kempton", "18:10")
    backfill_atr_sectionals(max_races=50)

Feature building joins the accepted cache with
:func:`src.atr_features.merge_atr_sectionals`. The columns, the cache location and
the cache reader are in :mod:`src.ingestion.atr_tabs`, which both sides import.

ATR/TPD sectional coverage is mostly UK tracks (all-weather plus many
turf courses from ~2021 on); Irish meetings are largely uncovered.  A
missing race returns HTTP 404 and is recorded in the no-data skip-list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.ingestion.atr_tabs import (ALL_ATR_COLS, ATR_ARRAY_COLS, ATR_CACHE_DIR, ATR_CACHE_FILE,
                                    COLLECTION_VERSION, EXTRA_ARRAYS, EXTRA_METRICS, EXTRA_TEXT,
                                    PERFORMANCE_TABS, current_collection_rows, load_atr_cache,
                                    signal_runner_count)
from src.utils import (
    normalise_horse_key as _normalise_horse_key,  # noqa: F401  the worker calls it through this module
    normalise_off_time_key as _normalise_off_time_key,
    normalise_track_key as _normalise_track_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATR_BASE = "https://www.attheraces.com"
ATR_PARSER_VERSION = 2

# Browser headers improve compatibility but do not guarantee access.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "sec-ch-ua": '"Chromium";v="126", "Google Chrome";v="126", "Not?A_Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

# Minimum gap between requests (seconds).  ATR's Fastly edge rate-limits
# aggressively: sustained fast crawling earns 429s and JS challenges, so be
# deliberately gentle.  Tunable via the CLI / the ``request_delay`` arg.
# Usually one results-index request per day plus five fragments per race.
_REQUEST_DELAY = 1.5
_PUBLICATION_DELAY_HOURS = 48

# Status codes that mean "backed off / blocked" rather than "no such race".
# These must NOT be cached as no-data — they are transient.
_BLOCKED_STATUSES = {403, 429, 503}

# Absence under the old three-tab contract does not establish five-tab absence.
ATR_NO_DATA_FILE = os.path.join(ATR_CACHE_DIR, "atr_no_data_keys_v3.json")
ATR_INDEX_DIR = os.path.join(ATR_CACHE_DIR, "results_index")
os.makedirs(ATR_CACHE_DIR, exist_ok=True)


# Every known performance endpoint is checked, even when not advertised for a
# particular race. The page-inventory guard separately flags unknown new tabs.
_AJAX_KINDS = PERFORMANCE_TABS

# ---------------------------------------------------------------------------
# Track name -> ATR course slug
# ---------------------------------------------------------------------------

# ATR course path segments are the course display name with spaces turned
# into hyphens, preserving the existing capitalisation
# (e.g. "Chelmsford City" -> "Chelmsford-City", "Lingfield" -> "Lingfield",
# "Bangor-on-Dee" -> "Bangor-on-Dee").  The project's track names already
# match ATR's display names, so the default slugifier covers almost
# everything; only genuinely different names need an override.  Slugs with
# the wrong case 301-redirect to the canonical course (which the fetcher
# follows), so minor mismatches usually self-correct.
_TRACK_SLUG_OVERRIDES: dict[str, str] = {
    "Royal Ascot": "Ascot",
}


def _track_to_slug(track_name: str) -> str:
    """Convert a track name to the ATR racecard URL course slug."""
    if track_name in _TRACK_SLUG_OVERRIDES:
        return _TRACK_SLUG_OVERRIDES[track_name]
    cleaned = re.sub(r"[^\w\s-]", "", str(track_name).strip())
    return re.sub(r"\s+", "-", cleaned)


# Tracks confirmed to have NO ATR/TPD sectionals — all Irish meetings
# (verified no-data across multiple dates, incl. Dundalk, which is AW).
# Skipped by default in backfills to avoid wasting requests; override with
# ``exclude_tracks=set()`` / the ``--include-irish`` CLI flag.  Matched on
# the normalised track key.
_NO_SECTIONALS_TRACKS = {
    _normalise_track_key(t) for t in (
        "Ballinrobe", "Bellewstown", "Clonmel", "Cork", "Curragh",
        "Down Royal", "Downpatrick", "Dundalk", "Fairyhouse", "Galway",
        "Gowran Park", "Kilbeggan", "Killarney", "Laytown", "Leopardstown",
        "Limerick", "Listowel", "Naas", "Navan", "Punchestown", "Roscommon",
        "Sligo", "Thurles", "Tipperary", "Tramore", "Wexford",
    )
}


_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _date_to_atr(date_str: str) -> str | None:
    """Convert ``YYYY-MM-DD`` to ATR URL date ``DD-Month-YYYY``."""
    dt = pd.to_datetime(str(date_str), errors="coerce")
    if pd.isna(dt):
        return None
    return f"{dt.day:02d}-{_MONTHS[dt.month - 1]}-{dt.year}"


def _off_time_to_hhmm(off_time) -> str:
    """Convert off_time like '14:08' or '14:08:00' to '1408'."""
    parts = str(off_time).strip().split(":")
    if len(parts) >= 2:
        return f"{int(parts[0]):02d}{int(parts[1]):02d}"
    return str(off_time).replace(":", "")


def _is_bst(date_str: str) -> bool:
    """Return True if *date_str* (YYYY-MM-DD) falls within UK BST.

    BST runs from the last Sunday of March to the last Sunday of October.
    Stored off-times are UTC; the ATR URL uses UK local time.
    """
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except ValueError:
        return False
    year = dt.year
    mar31 = datetime(year, 3, 31)
    bst_start = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)
    oct31 = datetime(year, 10, 31)
    bst_end = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)
    return bst_start <= dt < bst_end


def _add_one_hour(hhmm: str) -> str:
    """Add 1 hour to an HHMM string, e.g. '1640' -> '1740'."""
    h = int(hhmm[:2])
    m = int(hhmm[2:])
    h = (h + 1) % 24
    return f"{h:02d}{m:02d}"


def _publication_ready(races: pd.DataFrame, now=None) -> pd.Series:
    """ATR advises allowing about 48 hours; stored race times are UTC."""
    now = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    days = pd.to_datetime(races.race_date, errors="coerce").dt.strftime("%Y-%m-%d")
    times = races.off_time.map(_normalise_off_time_key)
    scheduled = pd.to_datetime(days + " " + times, errors="coerce", utc=True)
    return scheduled.le(now - pd.Timedelta(hours=_PUBLICATION_DELAY_HOURS))


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------
# ATR's Fastly edge fingerprints the TLS ClientHello (JA3): plain
# ``requests``/urllib3 is blocked with a 403 even when the browser headers
# are perfect, whereas a real browser (or curl) sails through.  We therefore
# fetch with, in order of preference:
#   1. curl_cffi  — impersonates Chrome's TLS fingerprint (pip install curl_cffi)
#   2. the curl binary — ships with Windows 10+/Git Bash and passes the check
#   3. plain requests — works wherever the challenge is absent
# All three keep a cookie jar so any Fastly clearance cookie carries from the
# racecard GET to the AJAX fragment GETs.

try:  # optional, strongly preferred
    from curl_cffi import requests as _curl_cffi_requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _curl_cffi_requests = None

_CURL_BIN = shutil.which("curl")


def _ajax_headers(referer: str) -> dict:
    return {
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "text/html, */*; q=0.01",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }


def _retry_after_seconds(value, now=None):
    if value is None:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            deadline = parsedate_to_datetime(str(value))
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = deadline.timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0,seconds) if math.isfinite(seconds) else None


class _Fetcher:
    """Throttled public HTTP client; challenges remain retryable blocks."""

    def __init__(self, request_delay: float | None = None) -> None:
        self._delay = _REQUEST_DELAY if request_delay is None else float(request_delay)
        self._last_request = 0.0
        self._cookie_jar: str | None = None
        self._cffi = None
        self._req = None
        self._results_indexes: dict[str, dict[str, str] | None] = {}
        self.index_dir = Path(ATR_INDEX_DIR)
        self.raw_store_path = Path(ATR_CACHE_DIR)/"raw.sqlite"
        self._raw_store = None
        self.last_response = {}
        self.request_count = 0
        self.request_seconds = 0.0
        self.throttle_seconds = 0.0

        if _curl_cffi_requests is not None:
            self.mode = "curl_cffi"
            self._cffi = _curl_cffi_requests.Session(impersonate="chrome")
            self._cffi.headers.update(_BROWSER_HEADERS)
        elif _CURL_BIN:
            self.mode = "curl_bin"
            fd, self._cookie_jar = tempfile.mkstemp(prefix="atr_cookies_", suffix=".txt")
            os.close(fd)
        else:
            self.mode = "requests"
            self._req = requests.Session()
            self._req.headers.update(_BROWSER_HEADERS)
        logger.debug("ATR fetcher transport: %s", self.mode)

    def _throttle(self) -> None:
        # Gap + small jitter so we don't hammer a fixed cadence.
        gap = self._delay + random.uniform(0, self._delay * 0.4)
        elapsed = time.time() - self._last_request
        if elapsed < gap:
            self.throttle_seconds += gap - elapsed
            time.sleep(gap - elapsed)
        self._last_request = time.time()

    def get(self, url: str, *, referer: str | None = None) -> tuple[int, str] | None:
        """GET *url*; return ``(status_code, text)`` or None on transport error.

        The JS "Client Challenge" stub is reported as status 429 so callers
        treat it as a transient block, never as genuine absence of data.
        """
        self._throttle()
        self.request_count += 1
        started = time.monotonic()
        self.last_response = {"url": url, "reason": "transport_error"}
        response_headers = {}
        headers = _ajax_headers(referer) if referer else {}
        try:
            if self.mode == "curl_bin":
                status, text = self._get_curl_bin(url, headers)
            elif self.mode == "curl_cffi":
                r = self._cffi.get(url, headers=headers, timeout=25)
                status, text = r.status_code, r.text
                response_headers = getattr(r, 'headers', {})
            else:
                r = self._req.get(url, headers=headers, timeout=25, allow_redirects=True)
                status, text = r.status_code, r.text
                response_headers = getattr(r, 'headers', {})
        except Exception as exc:  # pragma: no cover - network drift
            self.request_seconds += time.monotonic() - started
            logger.warning("ATR request failed (%s): %s", self.mode, exc)
            return None

        self.request_seconds += time.monotonic() - started

        # Retain the exact decoded input to the parsers, including real status
        # codes. This permits diagnosis/reparse without repeating HTTP requests.
        if self._raw_store is None:
            from src.ingestion.storage import RawPayloadStore
            self._raw_store = RawPayloadStore(self.raw_store_path)
        digest = self._raw_store.put(url, status, text.encode("utf-8"))
        self.last_response = {"url": url, "status": status, "sha256": digest,
                              "provider":self.mode}
        retry_after = _retry_after_seconds(response_headers.get('Retry-After'))
        if retry_after is not None:
            self.last_response['retry_after_seconds'] = retry_after
        if "Client Challenge" in text[:4000]:
            self.last_response["reason"] = "client_challenge"
            logger.warning("ATR returned HTTP %s Client Challenge for %s (retryable block)", status, url)
            return 429, ""
        return status, text

    def _get_curl_bin(self, url: str, extra_headers: dict) -> tuple[int, str]:
        merged = {**_BROWSER_HEADERS, **extra_headers}
        fd, body_path = tempfile.mkstemp(prefix="atr_body_", suffix=".html")
        os.close(fd)
        try:
            cmd = [_CURL_BIN, "-s", "-S", "--compressed", "-L",
                   "--connect-timeout", "15", "--max-time", "30",
                   "-c", self._cookie_jar, "-b", self._cookie_jar,
                   "-o", body_path, "-w", "%{http_code}"]
            for k, v in merged.items():
                cmd += ["-H", f"{k}: {v}"]
            cmd.append(url)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
            status = int(proc.stdout.strip() or 0)
            with open(body_path, "rb") as fh:
                text = fh.read().decode("utf-8", errors="replace")
            return status, text
        finally:
            try:
                os.remove(body_path)
            except OSError:
                pass

    def close(self) -> None:
        if self._cffi is not None:
            self._cffi.close()
        if self._req is not None:
            self._req.close()
        if self._cookie_jar and os.path.exists(self._cookie_jar):
            try:
                os.remove(self._cookie_jar)
            except OSError:
                pass


def _extract_raceid(html_text: str) -> str | None:
    """Read the active sectional-tab ID, with legacy racecard fallbacks."""
    ids = set(re.findall(r"/ajax/sectionals-razor/(\d+)/(?:times|tools|stride|jumpsdata|distanceran)\b", html_text))
    if ids:
        return ids.pop() if len(ids) == 1 else None
    m = re.search(r"RaceID:\s*(\d+)", html_text)
    if m:
        return m.group(1)
    # Fallback: the raceid query param attached to sectional/form links.
    ids = set(re.findall(r"[?&]raceid=(\d+)(?:[&\"'\s<>]|$)", html_text))
    return ids.pop() if len(ids) == 1 else None


def _parse_results_index(html_text: str) -> dict[str, str]:
    """Pair each result article's exact card path with its own native ID.

    Scope IDs to their article: the page also contains unrelated navigation,
    horse forms and other meetings. Never infer IDs from numeric sequences.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    result: dict[str, str] = {}
    for article in soup.select('article.panel-content--medium'):
        card = article.select_one('h3 a[href^="/racecard/"]')
        if card is None:
            continue
        path = card["href"].split("?", 1)[0].rstrip("/")
        ids = {match.group(1) for link in article.select('a[href]')
               if (match := re.search(r"[?&]raceid=(\d+)(?:&|$)", link["href"]))}
        if len(ids) > 1:
            raise ValueError(f"Conflicting ATR race IDs in result article: {path}")
        if not ids:
            continue
        raceid = ids.pop()
        if path in result and result[path] != raceid:
            raise ValueError(f"Conflicting ATR result articles: {path}")
        result[path] = raceid
    return result


def _results_index(fetcher: "_Fetcher", atr_date: str) -> tuple[str, dict[str, str]]:
    """Load each day's index once; do not memoise transient challenges."""
    if not hasattr(fetcher, "_results_indexes"):
        fetcher._results_indexes = {}
    indexes = fetcher._results_indexes
    def resolved(value, receipt=None):
        callback = getattr(fetcher, 'on_index', None)
        if callback is not None:
            callback(atr_date, value, receipt or {})
        return DATA, value
    if atr_date in indexes:
        value = indexes[atr_date]
        return resolved(value) if value is not None else (NOT_FOUND, {})
    index_dir = getattr(fetcher, "index_dir", None)
    index_path = Path(index_dir)/f"{atr_date}.json" if index_dir is not None else None
    if index_path is not None and index_path.exists():
        from src.ingestion.storage import read_json
        try:
            receipt = read_json(index_path)
            value = receipt["race_ids"]
            if not value or not isinstance(value, dict) or any(
                not re.fullmatch(rf"/racecard/[^/]+/{re.escape(atr_date)}/\d{{4}}", path)
                or not re.fullmatch(r"\d+", str(raceid)) for path, raceid in value.items()
            ):
                raise ValueError("Invalid saved ATR results index")
            indexes[atr_date] = value
            return resolved(value, receipt)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid ATR index %s: %s", index_path, exc)
    response = fetcher.get(f"{ATR_BASE}/results/{atr_date}")
    if response is None:
        return BLOCKED, {}
    status, text = response
    if status == 404:
        indexes[atr_date] = None
        return NOT_FOUND, {}
    if status != 200 or not text or "Client Challenge" in text[:4000]:
        return BLOCKED, {}
    try:
        value = _parse_results_index(text)
    except ValueError as exc:
        fetcher.last_response = {**getattr(fetcher,'last_response',{}),'reason':'unrecognised_index'}
        logger.warning("ATR index rejected: %s", exc)
        return BLOCKED, {}
    value = {path: raceid for path, raceid in value.items()
             if re.fullmatch(rf"/racecard/[^/]+/{re.escape(atr_date)}/\d{{4}}", path)}
    if not value:
        fetcher.last_response = {**getattr(fetcher,'last_response',{}),'reason':'unrecognised_index'}
        # An unrecognised page is not evidence that its races lack sectionals.
        return BLOCKED, {}
    indexes[atr_date] = value
    receipt = {"source_url": f"{ATR_BASE}/results/{atr_date}",
               "retrieved_at": datetime.now(timezone.utc).isoformat(),
               "source_sha256":getattr(fetcher,'last_response',{}).get('sha256'), "race_ids": value}
    if index_path is not None:
        from src.ingestion.storage import atomic_json
        atomic_json(index_path, receipt)
    return resolved(value, receipt)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m.group(0)) if m else None


def _sectional_entries(soup: BeautifulSoup) -> list:
    """Return the per-runner card entries from the sectional column.

    Sectional AJAX fragments contain two ``card-body`` columns: the first
    repeats the runner info, the second carries the sectional cells.  We
    want the second; fall back to the first if only one is present.
    """
    bodies = soup.find_all("div", class_="card-body")
    if not bodies:
        return []
    target = bodies[1] if len(bodies) > 1 else bodies[0]
    return target.find_all("div", class_="card-entry")


def _runner_meta(entry) -> dict:
    """Extract the shared per-runner identity/meta fields from a card entry."""
    meta: dict = {}

    link = entry.find("a", class_="horse__link")
    name = link.get_text(" ", strip=True) if link else ""
    meta["horse_name"] = name
    href = (link.get("href") or link.get("data-url") or "") if link else ""
    m = re.search(r"/horse/[^/]+/(?:[A-Za-z]{2,3}/)?(\d+)(?:[/?#]|$)", href)
    meta["atr_horse_id"] = m.group(1) if m else None

    cloth = entry.find("span", class_="p--small visible--inline-block")
    if cloth:
        meta["atr_saddle_cloth"] = cloth.get_text(strip=True).rstrip(".") or None

    draw = entry.find("span", class_="p--x-small visible--inline-block")
    if draw:
        d = draw.get_text(strip=True).strip("()")
        meta["atr_draw"] = d or None

    cmt = entry.find("span", class_="p--x-small tooltip")
    if cmt and cmt.get("title"):
        meta["atr_in_running_comment"] = cmt.get("title").strip()

    if entry.get("data-officialplacing"):
        meta["atr_official_position"] = entry.get("data-officialplacing")
    if entry.get("data-adjplacing"):
        meta["atr_adj_placing"] = _to_float(entry.get("data-adjplacing"))

    return meta


def _class_flag(element, prefix: str) -> str | None:
    if element is None:
        return None
    return next((c[len(prefix):] for c in element.get("class", [])
                 if c.startswith(prefix) and c != prefix + "has-icon"), None)


def _parse_race_analysis(soup: BeautifulSoup) -> dict:
    """Race context is separate from the individual horse's finishing speed."""
    root = soup.select_one('.sectional-analysis')
    if root is None:
        return {}
    out = {}
    pace = root.select_one('.sectional-analysis__pace .sectional-text-highlight')
    if pace is not None:
        # Exclude the nested help icon's question mark.
        out['atr_race_early_pace'] = ' '.join(pace.find_all(string=True, recursive=False)).strip()
        out['atr_race_early_pace_flag'] = _class_flag(pace, 'sectional-text-highlight--')
    finish = root.select_one('.sectional-analysis__finish .value')
    if finish is not None:
        out['atr_race_finishing_speed_pct'] = _to_float(finish.get_text(' ', strip=True))
        out['atr_race_finishing_pace_flag'] = _class_flag(
            finish.select_one('.sectional-text-highlight'), 'sectional-text-highlight--')
    first = root.select_one('.sectional-pacebar__first-text')
    if first is not None:
        out['atr_first_section_furlongs'] = _to_float(first.get_text(' ', strip=True))
    stops = root.select('.sectional-pacebar stop')
    if stops:
        out['atr_race_pace_profile'] = [{'offset': x.get('offset'), 'colour': x.get('stop-color'),
                                       'opacity': x.get('stop-opacity')} for x in stops]
    return out


def _finish_seconds(text: str) -> float | None:
    match = re.fullmatch(r'\s*(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s\s*', text)
    return (60 * int(match[1] or 0) + float(match[2])) if match else None


def _source_card_context(soup: BeautifulSoup) -> dict:
    """Use dated race-silk filenames as an additional identity cross-check."""
    found = set()
    for image in soup.select('.card-entry img[src]'):
        match = re.search(r'/images/silks/(\d{8})/\d{8}[A-Za-z]{2,8}(\d{4})\d{2,3}\.(?:png|gif)', image['src'])
        if match:
            day, off = match.groups()
            found.add((f'{day[:4]}-{day[4:6]}-{day[6:]}',f'{off[:2]}:{off[2:]}'))
    if len(found) > 1:
        raise ValueError('Mixed source race dates/off-times')
    if not found:
        return {}
    day, off = found.pop()
    return {'atr_source_race_date':day,'atr_source_off_time_local':off}


def _parse_times(soup: BeautifulSoup) -> dict[str, dict]:
    """Parse the ``times`` fragment: split times, running totals, pace flags."""
    out: dict[str, dict] = {}
    for entry in _sectional_entries(soup):
        meta = _runner_meta(entry)
        name = meta.get("horse_name")
        if not name:
            continue

        splits: list[float] = []
        running: list[float | None] = []
        flags: list[str] = []
        front, notes = [], []
        for cell in entry.find_all("div", class_="card-cell--sectional-time"):
            sec = cell.find("div", class_=lambda c: c and "card-sectional" in c)
            vis = cell.find("span", class_="visible")
            if vis is None:
                continue  # finish marker cell carries no value
            val = _to_float(vis.get_text(strip=True))
            splits.append(val)
            inner = cell.find("div", class_="card-sectional__inner")
            rt = _to_float(inner.get("title")) if inner and inner.get("title") else None
            running.append(rt)  # Preserve alignment when a tooltip is absent.
            labels = [p.get_text(' ', strip=True) for p in cell.select('.text-pill')]
            notes.append(labels)
            front.append(any(label.casefold() == 'front' for label in labels))
            # pace flag = class suffix after "card-sectional--"
            flag = ""
            for cls in (sec.get("class") or []) if sec else []:
                if cls.startswith("card-sectional--"):
                    flag = cls.replace("card-sectional--", "")
                    break
            flags.append(flag)

        rec = dict(meta)
        rec["atr_split_times"] = splits
        rec["atr_running_totals"] = running
        rec["atr_pace_flags"] = flags
        rec['atr_front_flags'] = front
        rec['atr_section_notes'] = notes
        rec["atr_n_sections"] = float(len(splits)) if splits else None
        finish = entry.select_one('.card-sectional--finish .card-sectional__inner')
        final = _finish_seconds(finish.get_text(' ', strip=True)) if finish else None
        rec["atr_final_time"] = final if final is not None else (running[-1] if running else None)
        out[name] = rec
    return out


def _parse_tools(soup: BeautifulSoup) -> dict[str, dict]:
    """Parse the ``tools`` fragment: speeds, finishing speed %, efficiency, energy."""
    out: dict[str, dict] = {}
    for entry in _sectional_entries(soup):
        name = (entry.find("a", class_="horse__link") or {})
        name = name.get_text(" ", strip=True) if hasattr(name, "get_text") else ""
        if not name:
            continue
        rec: dict = _runner_meta(entry)

        speeds: list[float] = []
        labels: list[str] = []
        source_metrics = []
        for pill in entry.find_all("span", class_=lambda c: c and "text-pill" in c):
            classes = pill.get("class") or []
            if "text-pill--grey" in classes:
                # efficiency grade pill, e.g. "C" with title "Efficency Score: 17.18"
                rec["atr_efficiency_grade"] = pill.get_text(strip=True) or None
                rec["atr_efficiency_score"] = _to_float(pill.get("title"))
                continue
            v = _to_float(pill.get_text(strip=True))
            if v is not None:
                label = (pill.get('title') or '').strip()
                compact = re.sub(r'\s+', '', label.casefold()).replace('–', '-').replace('−', '-')
                unit = 'mph'
                if compact == 'topspeed':
                    rec['atr_top_speed_mph'] = v
                elif compact == 'finalspeed':
                    rec['atr_final_speed_mph'] = v
                elif compact in {'0-30mph', '0to30mph'}:
                    rec['atr_acceleration_0_30_seconds'] = v
                    unit = 'seconds'
                elif re.search(r'\bto\b', label, re.IGNORECASE):
                    speeds.append(v)
                    labels.append(label)
                else:
                    # Retain unknown source columns without treating them as speeds.
                    unit = None
                source_metrics.append({'label': label, 'value': v, 'unit': unit,
                                       'display': pill.get_text(' ', strip=True)})
        if speeds:
            rec["atr_sectional_speeds"] = speeds
            rec["atr_sectional_speed_labels"] = labels
        rec['atr_source_metrics'] = source_metrics

        # Per-runner finishing speed %: the width--24 p--medium block.
        fs = entry.find("div", class_="width--24 p--medium")
        if fs is not None:
            rec["atr_finishing_speed_pct"] = _to_float(fs.get_text(strip=True))

        # Energy distribution chart: [[marker, energy%, optimum%], ...]
        chart = entry.find("div", class_="chart-energy-distribution")
        markers, energy, optimum = _parse_chart(chart)
        if markers:
            rec["atr_section_markers"] = markers
            rec["atr_energy_dist"] = energy
            rec["atr_optimum_energy"] = optimum

        out[name] = rec
    return out


def _parse_stride(soup: BeautifulSoup) -> dict[str, dict]:
    """Parse the ``stride`` fragment: per-furlong stride length + peaks."""
    out: dict[str, dict] = {}
    for entry in _sectional_entries(soup):
        link = entry.find("a", class_="horse__link")
        name = link.get_text(" ", strip=True) if link else ""
        if not name:
            continue
        rec: dict = _runner_meta(entry)

        charts = entry.select('.chart-energy-distribution')
        source_charts = []
        for index, chart in enumerate(charts):
            markers, values, reference = _parse_chart(chart)
            title = chart.get('data-line1-title') or ('Stride Length (Feet)' if index == 0 else '')
            source_charts.append({'title': title, 'attributes': {
                key: value for key, value in chart.attrs.items() if key.startswith('data-')},
                'markers': markers, 'values': values, 'reference': reference})
            if 'length' in title.casefold():
                rec['atr_stride_length'], rec['atr_stride_markers'] = values, markers
                rec['atr_stride_length_reference'] = reference
            elif 'frequency' in title.casefold():
                rec['atr_stride_frequency'], rec['atr_stride_frequency_markers'] = values, markers
                rec['atr_stride_frequency_reference'] = reference
        rec['atr_source_charts'] = source_charts

        # Peak stride length / frequency: "26.94 / 2.28" in a width--24 block.
        for w in entry.find_all("div", class_="width--24"):
            txt = w.get_text(" ", strip=True)
            if "/" in txt:
                left, _, right = txt.partition("/")
                pl, pf = _to_float(left), _to_float(right)
                if pl is not None:
                    rec["atr_peak_stride_length"] = pl
                if pf is not None:
                    rec["atr_peak_stride_freq"] = pf
                break
        out[name] = rec
    return out


def _parse_chart(chart) -> tuple[list[str], list[float], list[float]]:
    """Parse a ``chart-energy-distribution`` data-data JSON triple-array.

    Returns ``(markers, series1, series2)`` where each row is
    ``[marker, value1, value2]``.
    """
    if chart is None:
        return [], [], []
    raw = chart.get("data-data")
    if not raw:
        return [], [], []
    try:
        rows = json.loads(raw)
    except (ValueError, TypeError):
        return [], [], []
    markers, s1, s2 = [], [], []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        markers.append(str(row[0]))
        s1.append(_to_float(row[1]))
        s2.append(_to_float(row[2]))
    return markers, s1, s2


# ---------------------------------------------------------------------------
# High-level fetch
# ---------------------------------------------------------------------------

# Outcome of a fetch attempt.  The distinction is critical for the backfill:
# only NOT_FOUND may be cached as no-data; BLOCKED means "back off and retry".
DATA = "data"
NOT_FOUND = "not_found"   # racecard 404, or 200 with genuinely no sectionals
BLOCKED = "blocked"       # 403/429/503/challenge/transport error — transient
INVALID = "invalid"       # unparseable inputs (bad date)
UNRESOLVED = "unresolved" # unexpected layout / unconfirmed absence, retryable


def _fetch_one(
    fetcher: "_Fetcher", slug: str, atr_date: str, hhmm: str,
) -> tuple[str, list[dict] | None]:
    """Fetch + parse a single exact (slug, date, hhmm).

    Returns ``(outcome, rows)`` where outcome is one of :data:`DATA`,
    :data:`NOT_FOUND` or :data:`BLOCKED`.
    """
    page_path = f"/racecard/{slug}/{atr_date}/{hhmm}"
    index_outcome, index = _results_index(fetcher, atr_date)
    if index_outcome == BLOCKED:
        return BLOCKED, None
    raceid = index.get(page_path)
    if not raceid:
        card = fetcher.get(ATR_BASE + page_path)
        if card is None:
            return BLOCKED, None
        status, text = card
        if status == 404:
            return NOT_FOUND, None
        if status != 200 or not text or "Client Challenge" in text[:4000]:
            return BLOCKED, None
        from src.ingestion.atr_tabs import inventory
        tab_inventory = inventory(text)
        if tab_inventory['unknown_tabs']:
            fetcher.last_response = {**getattr(fetcher,'last_response',{}),
                                     'reason':'unknown_tabs', 'inventory':tab_inventory}
            return UNRESOLVED, None
        raceid = _extract_raceid(text)
        if not raceid:
            # HTML drift must not permanently suppress a real race.
            return BLOCKED, None
        callback = getattr(fetcher,'on_index',None)
        if callback is not None:
            callback(atr_date,{page_path:raceid},{'source_url':ATR_BASE+page_path,
                     'source_sha256':getattr(fetcher,'last_response',{}).get('sha256')})

    return _fetch_fragments(fetcher, raceid, page_path)


def _fetch_fragments(fetcher: "_Fetcher", raceid: str, page_path: str) -> tuple[str, list[dict] | None]:
    """Fetch every public tab for a known native ID, retaining provenance."""
    if not re.fullmatch(r'\d+', str(raceid)):
        return INVALID, None
    referer = ATR_BASE + page_path
    parsed: dict[str, dict] = {}
    context, payloads = {}, []
    fetcher.tab_receipts = payloads
    statuses = []
    parsers = {
        "times": _parse_times,
        "tools": _parse_tools,
        "stride": _parse_stride,
    }
    for kind in _AJAX_KINDS:
        url = f"{ATR_BASE}/ajax/sectionals-razor/{raceid}/{kind}"
        resp = fetcher.get(url, referer=referer)
        if resp is None:
            payloads.append({'kind':kind, 'url':url, 'state':'blocked', 'status':None,
                             'reason':'transport_error'})
            return BLOCKED, None
        s, t = resp
        statuses.append(s)
        receipt = {'kind': kind, 'url': url, 'status': s, 'state':'received',
                   'sha256': hashlib.sha256(t.encode('utf-8')).hexdigest()}
        payloads.append(receipt)
        if s == 404:
            receipt['state'] = 'not_found'
            continue  # this fragment is simply absent for the race
        if s in _BLOCKED_STATUSES or s != 200 or not t:
            receipt['state'] = 'blocked'
            # Do not permanently skip a race whose later tabs were blocked.
            return BLOCKED, None
        soup = BeautifulSoup(t, "html.parser")
        providers = [x.get('alt') for x in soup.select('img.site-logo') if x.get('alt')]
        receipt['providers'] = providers
        if 'Client Challenge' in t[:4000]:
            receipt['state'] = 'blocked'
            return BLOCKED, None
        try:
            source_context = _source_card_context(soup)
        except ValueError:
            receipt['state'] = 'unresolved'
            fetcher.last_response = {**getattr(fetcher,'last_response',{}),'reason':'source_identity_mismatch'}
            return UNRESOLVED, None
        if source_context:
            _, _, _, day_text, local_time = page_path.split('/')
            day, month, year = day_text.split('-')
            expected_day = f'{year}-{_MONTHS.index(month)+1:02d}-{day}'
            expected_time = f'{local_time[:2]}:{local_time[2:]}'
            if source_context['atr_source_race_date'] != expected_day or source_context['atr_source_off_time_local'] != expected_time:
                receipt['state'] = 'unresolved'
                fetcher.last_response = {**getattr(fetcher,'last_response',{}),'reason':'source_identity_mismatch',
                                         'source_context':source_context}
                return UNRESOLVED, None
            context.update(source_context)
        for key, value in _parse_race_analysis(soup).items():
            if key in context and context[key] != value:
                return UNRESOLVED, None
            context[key] = value
        try:
            if kind in parsers:
                records = parsers[kind](soup)
                if not records and not soup.select_one('.card-body, .js-sectional, .sectional-analysis'):
                    raise ValueError('Unrecognised performance tab layout')
            else:
                from src.ingestion.atr_tabs import parse_extra
                records, extra_context = parse_extra(kind, t, raceid)
                context.update({k:v for k,v in extra_context.items() if k in ALL_ATR_COLS})
            for name, rec in records.items():
                old = parsed.get(name, {})
                if old.get('atr_horse_id') and rec.get('atr_horse_id') and old['atr_horse_id'] != rec['atr_horse_id']:
                    raise ValueError('Native horse identity differs between tabs')
                if kind in ('jumpsdata','distanceran') and old:
                    # Additional tabs must not overwrite core placing/comment
                    # metadata with the optional endpoint's empty placeholders.
                    rec = {k:v for k,v in rec.items() if k not in old or k in EXTRA_METRICS+EXTRA_TEXT+EXTRA_ARRAYS}
                parsed.setdefault(name, {}).update(rec)
            signal_count = signal_runner_count(kind, records)
            receipt.update(state='data' if signal_count else 'empty', runner_count=len(records),
                           signal_runner_count=signal_count,
                           horse_names=sorted(records),
                           horse_ids=sorted(str(r['atr_horse_id']) for r in records.values() if r.get('atr_horse_id')))
        except ValueError as exc:
            receipt.update(state='unresolved', reason=str(exc))
            fetcher.last_response = {**getattr(fetcher,'last_response',{}),'url':url,
                                     'reason':'unrecognised_tab', 'detail':str(exc)}
            return UNRESOLVED, None

    native_ids = [r['atr_horse_id'] for r in parsed.values() if r.get('atr_horse_id')]
    if len(native_ids) != len(set(native_ids)):
        fetcher.last_response = {**getattr(fetcher,'last_response',{}),'reason':'duplicate_native_horse_identity'}
        return UNRESOLVED, None

    if not any(signal_runner_count(kind, parsed) for kind in _AJAX_KINDS):
        # A changed HTML template is not proof of no telemetry. Only explicit
        # missing fragments or actual runner rows with no signals are terminal.
        return (NOT_FOUND, None) if all(s == 404 for s in statuses) or parsed else (UNRESOLVED, None)
    rows: list[dict] = []
    for name, rec in parsed.items():
        # Keep untracked runners' identities when other runners have telemetry.
        rec.setdefault("horse_name", name)
        rec["atr_raceid"] = raceid
        rec['atr_parser_version'] = str(ATR_PARSER_VERSION)
        rec['atr_collection_version'] = str(COLLECTION_VERSION)
        rec['atr_source_payloads'] = payloads
        rec['atr_tab_manifest'] = payloads
        rec.update(context)
        rows.append(rec)

    if rows:
        return DATA, rows
    return NOT_FOUND, None


def fetch_atr_sectionals_outcome(
    date: str, track_name: str, off_time: str,
    fetcher: "_Fetcher | None" = None,
    *, raceid: str | None = None,
) -> tuple[str, list[dict] | None]:
    """Fetch ATR sectionals, returning ``(outcome, rows)``.

    Off-times are stored in UTC; the ATR URL uses UK local time. Use exactly
    that conversion: an unadjusted summer time can identify a different race.
    """
    slug = _track_to_slug(track_name)
    atr_date = _date_to_atr(date)
    if not atr_date:
        return INVALID, None
    hhmm = _off_time_to_hhmm(off_time)
    if _is_bst(date):
        hhmm = _add_one_hour(hhmm)

    owns_fetcher = fetcher is None
    f = fetcher or _Fetcher()
    try:
        if raceid is not None:
            return _fetch_fragments(f, str(raceid), f'/racecard/{slug}/{atr_date}/{hhmm}')
        return _fetch_one(f, slug, atr_date, hhmm)
    finally:
        if owns_fetcher:
            f.close()


def fetch_atr_sectionals(
    date: str, track_name: str, off_time: str,
    fetcher: "_Fetcher | None" = None,
) -> list[dict] | None:
    """Convenience wrapper returning just the rows (or None)."""
    outcome, rows = fetch_atr_sectionals_outcome(date, track_name, off_time, fetcher)
    return rows if outcome == DATA else None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _encode_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-encode list-valued columns so the parquet stays flat/portable."""
    out = df.copy()
    for col in ATR_ARRAY_COLS:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: json.dumps(v) if isinstance(v, (list, tuple)) else v
            )
    return out


def save_atr_cache(df: pd.DataFrame) -> None:
    """Save the ATR sectionals cache."""
    from src.ingestion.storage import atomic_parquet
    atomic_parquet(_encode_arrays(df), ATR_CACHE_FILE)
    logger.info("Saved ATR cache: %d rows -> %s", len(df), ATR_CACHE_FILE)


def _race_key(date_str: str, track: str, off_time: str) -> str:
    """Stable race-level key for cache skip-lists."""
    _d = pd.to_datetime(date_str, errors="coerce")
    _d_s = _d.strftime("%Y-%m-%d") if pd.notna(_d) else str(date_str)
    return f"{_d_s}|{_normalise_track_key(track)}|{_normalise_off_time_key(off_time)}"


def _load_no_data_race_keys() -> set[str]:
    if not os.path.exists(ATR_NO_DATA_FILE):
        return set()
    try:
        with open(ATR_NO_DATA_FILE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        keys = payload.get("race_keys", []) if isinstance(payload, dict) else []
        return {str(k) for k in keys}
    except Exception:
        return set()


def _save_no_data_race_keys(keys: set[str]) -> None:
    from src.ingestion.storage import atomic_json
    payload = {"updated_at": datetime.now().isoformat(), "race_keys": sorted(keys)}
    atomic_json(ATR_NO_DATA_FILE, payload)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill_atr_sectionals(
    max_races: int | None = None,
    skip_existing: bool = True,
    request_delay: float | None = None,
    exclude_tracks: set[str] | None = None,
) -> pd.DataFrame:
    """Scrape ATR sectionals for every race in the published results snapshot.

    Appends to the cache incrementally so progress survives interruption.
    """
    from src.database import read_results_snapshot, results_snapshot_path
    raw_path = results_snapshot_path()
    if not raw_path.exists():
        logger.error("Results snapshot not found at %s", raw_path)
        return pd.DataFrame()

    raw = read_results_snapshot(raw_path, columns=["race_date", "track", "off_time"])
    races = (
        raw.drop_duplicates(subset=["race_date", "track", "off_time"])
        .sort_values("race_date", ascending=False)
        .reset_index(drop=True)
    )
    summary = backfill_atr_sectionals_for_races(
        races, max_races=max_races, skip_existing=skip_existing,
        request_delay=request_delay, exclude_tracks=exclude_tracks,
    )
    logger.info("ATR backfill summary: %s", summary)
    return load_atr_cache()


def _exclusive_collection(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from src.ingestion.storage import writer_lock
        with writer_lock(Path(ATR_CACHE_FILE).parent/'.collector.lock') as acquired:
            if not acquired:
                raise RuntimeError('An ATR collector is already running')
            return function(*args, **kwargs)
    return wrapped


@_exclusive_collection
def backfill_atr_sectionals_for_races(
    races_df: pd.DataFrame,
    max_races: int | None = None,
    skip_existing: bool = True,
    request_delay: float | None = None,
    max_block_retries: int = 5,
    block_cooldown: int = 30,
    exclude_tracks: set[str] | None = None,
) -> dict[str, int]:
    """Incrementally fetch ATR sectionals for a specific race subset.

    ``races_df`` must contain ``race_date``, ``track`` and ``off_time``.

    ``exclude_tracks`` (normalised track keys) are skipped entirely; it
    defaults to :data:`_NO_SECTIONALS_TRACKS` (Irish meetings, which have no
    ATR sectionals).  Pass an empty set to scrape every track.
    """
    required = {"race_date", "track", "off_time"}
    if races_df is None or races_df.empty or not required.issubset(races_df.columns):
        return {
            "target_races": 0, "missing_races": 0, "fetched_races": 0,
            "new_rows": 0, "cache_rows": int(len(load_atr_cache())),
        }

    if exclude_tracks is None:
        exclude_tracks = _NO_SECTIONALS_TRACKS

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
    target_total = int(len(races))

    excluded_races = 0
    if exclude_tracks:
        keep = ~races["track"].map(_normalise_track_key).isin(exclude_tracks)
        excluded_races = int((~keep).sum())
        races = races[keep].reset_index(drop=True)
        if excluded_races:
            logger.info(
                "ATR backfill: skipping %d races at %d known-uncovered tracks",
                excluded_races, len(exclude_tracks),
            )

    cache = load_atr_cache()
    known_no_data = _load_no_data_race_keys()

    # Skip races already cached (matched on normalised keys).
    if skip_existing and not cache.empty:
        ck = current_collection_rows(cache)[["race_date", "track", "off_time"]].copy()
        ck["race_date"] = pd.to_datetime(ck["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        ck["track"] = ck["track"].map(_normalise_track_key)
        ck["off_time"] = ck["off_time"].map(_normalise_off_time_key)
        existing = set(ck.dropna().drop_duplicates().apply(tuple, axis=1))
        r = races.copy()
        r["_t"] = r["track"].map(_normalise_track_key)
        r["_o"] = r["off_time"].map(_normalise_off_time_key)
        races = r[
            ~r.apply(lambda x: (x["race_date"], x["_t"], x["_o"]) in existing, axis=1)
        ][["race_date", "track", "off_time"]].reset_index(drop=True)

    # Skip races previously confirmed as no-data.
    skipped_known_missing = 0
    if known_no_data and not races.empty:
        rk = races.apply(
            lambda r: _race_key(r["race_date"], r["track"], r["off_time"]), axis=1,
        )
        mask = rk.isin(known_no_data)
        skipped_known_missing = int(mask.sum())
        races = races.loc[~mask].reset_index(drop=True)

    # Fresh absence is not terminal: allow time for TPD's delayed publication.
    ready = _publication_ready(races)
    deferred_recent = int((~ready).sum())
    races = races.loc[ready].reset_index(drop=True)

    if max_races is not None:
        races = races.head(max_races)

    if races.empty:
        return {
            "target_races": target_total, "excluded_tracks_races": excluded_races,
            "missing_races": 0, "fetched_races": 0,
            "new_rows": 0, "cache_rows": int(len(cache)),
            "skipped_known_missing": skipped_known_missing,
            "deferred_recent_races": deferred_recent,
        }

    logger.info("ATR backfill: fetching %d races (request_delay=%.2fs)",
                len(races), request_delay or _REQUEST_DELAY)
    fetcher = _Fetcher(request_delay=request_delay)
    logger.info("ATR transport: %s", fetcher.mode)
    new_rows: list[dict] = []
    total_new_rows = 0
    fetched = 0
    n_data = 0
    n_no_data = 0
    batch_size = 50
    newly_no_data = 0
    aborted = False

    def _flush() -> None:
        nonlocal cache, new_rows, total_new_rows
        if new_rows:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            cache = cache.drop_duplicates(subset=['race_date','track','off_time','horse_name'],keep='last')
            save_atr_cache(cache)
            total_new_rows += len(new_rows)
            new_rows = []
        if newly_no_data:
            _save_no_data_race_keys(known_no_data)

    try:
        for _, race in races.iterrows():
            date_str = str(race["race_date"])
            track = str(race["track"])
            off_time = str(race["off_time"])

            # Retry transient blocks with escalating cooldown; never let a
            # block be mistaken for genuine no-data.
            outcome, data = (BLOCKED, None)
            for attempt in range(max_block_retries + 1):
                outcome, data = fetch_atr_sectionals_outcome(
                    date_str, track, off_time, fetcher=fetcher,
                )
                if outcome != BLOCKED:
                    break
                if attempt < max_block_retries:
                    cooldown = block_cooldown * (attempt + 1)
                    logger.warning(
                        "BLOCKED by ATR (rate limit / challenge) on %s %s %s — "
                        "cooling down %ds (attempt %d/%d)",
                        date_str, track, off_time, cooldown,
                        attempt + 1, max_block_retries,
                    )
                    time.sleep(cooldown)

            if outcome == BLOCKED:
                logger.error(
                    "ATR still blocking after %d retries — aborting backfill to "
                    "avoid polluting the no-data list. Progress is saved; rerun "
                    "later to resume.", max_block_retries,
                )
                aborted = True
                break

            if outcome == DATA and data:
                for row in data:
                    row["race_date"] = date_str
                    row["track"] = track
                    row["off_time"] = off_time
                    new_rows.append(row)
                known_no_data.discard(_race_key(date_str, track, off_time))
                n_data += 1
            elif outcome == NOT_FOUND:
                logger.info("NO DATA  %s  %-18s  %s", date_str, track, off_time)
                n_no_data += 1
                k = _race_key(date_str, track, off_time)
                if k not in known_no_data:
                    known_no_data.add(k)
                    newly_no_data += 1
            fetched += 1

            if fetched % 25 == 0:
                logger.info(
                    "ATR backfill progress: %d / %d races (data=%d, no-data=%d)",
                    fetched, len(races), n_data, n_no_data,
                )
            if fetched % batch_size == 0:
                _flush()
    finally:
        _flush()
        fetcher.close()

    return {
        "target_races": target_total,
        "excluded_tracks_races": excluded_races,
        "missing_races": int(len(races)),
        "fetched_races": int(fetched),
        "races_with_data": int(n_data),
        "races_no_data": int(n_no_data),
        "new_rows": int(total_new_rows),
        "cache_rows": int(len(cache)),
        "skipped_known_missing": skipped_known_missing,
        "new_known_missing": int(newly_no_data),
        "resolved_races": int(n_data + newly_no_data),
        "deferred_recent_races": deferred_recent,
        "aborted_blocked": aborted,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch At The Races sectionals")
    p.add_argument("--date", help="Single race date YYYY-MM-DD (with --track/--off-time)")
    p.add_argument("--track", help="Track name, e.g. 'Kempton'")
    p.add_argument("--off-time", dest="off_time", help="Off time, e.g. 18:10 (UTC)")
    p.add_argument("--max-races", type=int, default=None, help="Cap races in backfill")
    p.add_argument(
        "--request-delay", dest="request_delay", type=float, default=None,
        help=f"Min seconds between requests (default {_REQUEST_DELAY}). "
             "Raise if ATR rate-limits (429).",
    )
    p.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Re-fetch races already in the cache",
    )
    p.add_argument(
        "--include-irish", dest="include_irish", action="store_true",
        help="Also scrape Irish tracks (skipped by default — they have no "
             "ATR sectionals)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.date and args.track and args.off_time:
        rows = fetch_atr_sectionals(args.date, args.track, args.off_time)
        print(json.dumps({"runners": len(rows) if rows else 0,
                          "rows": rows}, indent=2, default=str))
        return

    backfill_atr_sectionals(
        max_races=args.max_races, skip_existing=args.skip_existing,
        request_delay=args.request_delay,
        exclude_tracks=set() if args.include_irish else None,
    )


if __name__ == "__main__":
    main()
