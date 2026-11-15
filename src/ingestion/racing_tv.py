"""
Racing TV (RaceiQ) scraper
==========================
Collects everything Racing TV publishes about GB and IE races, from the JSON
API its website calls (``https://api.racingtv.com/racing``). See
``docs/operations/RTV_SCRAPER.md`` for coverage, validation and data-quality findings.

Per race (:func:`scrape_days`): ``results`` (2009-), ``racecard`` (2009-),
``tips``, bookmaker ``odds`` (by runner id), RaceiQ ``iq_metrics`` and
``iq_sectional`` (2023-06-10-); per day the ``day_list`` and ``tipsters``;
per course the ``track`` page. Race-day-only payloads (pace maps, RaceiQ
prompts, the pre-race card and odds) are captured by :func:`snapshot_day`
and kept per capture; horse profiles by :func:`scrape_horse_profiles`.
``results`` and ``racecard`` duplicate much of the Sporting Life feed on
purpose: they are a validation source and a fallback.

Design
------
* **Raw first.** Every response body is stored, zlib-compressed, in
  ``data/rtv_cache/rtv_raw.sqlite`` keyed by API path. Parsing never touches
  the network, so tables can be rebuilt after a parser fix.
* **No time-zone guessing.** Race URLs use UK/IE local time. The day list
  gives each race's ``start_time_scheduled`` with its UTC offset, so the URL
  time and the UTC off-time used by ``data/races.db`` both come from it.
  Races the day list omits are found from ``data/races.db``
  (:func:`discover_unlisted_races`).
* **Blocked is not missing.** Only 200 and 404 are final. 429/5xx/network
  errors pause every worker, are retried with backoff and never cached; a
  run that cannot get through stops, and the next run resumes.
* **Concurrent but capped.** A thread pool shares one request-rate cap
  (default 10/s); several days share each phase so workers stay busy.
* **The API key is discovered**, as the BHA scraper does: it is the public
  key in the site's JavaScript config, read at run time (``RTV_API_KEY``
  overrides it). The open ``results`` endpoint works without it.

The parsed tables (:func:`build_tables`) keep every scalar field of every
payload under a source prefix (``res_``, ``card_``, ``sect_``) plus cleaned,
typed derived columns. :func:`quality_checks` flags spurious values and
:func:`validate_against_db` compares the result with ``data/races.db``.

Racing TV's website terms permit downloading text content to a single
personal computer for personal, non-commercial use.

Usage::

    python -m src.ingestion.racing_tv scrape --start 2023-06-10 --end 2026-10-28
    python -m src.ingestion.racing_tv snapshot            # today's race-day data
    python -m src.ingestion.racing_tv profiles --start 2026-01-01
    python -m src.ingestion.racing_tv learn-glyphs
    python -m src.ingestion.racing_tv build
    python -m src.ingestion.racing_tv validate --start 2025-08-01 --end 2025-08-31
    python -m src.ingestion.racing_tv status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
import zlib
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config
from src.utils import (
    canonical_track_key,
    normalise_horse_key,
    normalise_off_time_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://api.racingtv.com/racing"
SITE_BASE = "https://www.racingtv.com"

RTV_DIR = os.path.join(config.DATA_DIR, "rtv_cache")
RAW_DB_PATH = os.path.join(RTV_DIR, "rtv_raw.sqlite")
TABLES_DIR = os.path.join(RTV_DIR, "tables")
VALIDATION_DIR = os.path.join(RTV_DIR, "validation")
RESULTS_DB_PATH = os.path.join(config.DATA_DIR, "races.db")

# First day with RaceiQ data (verified: 2023-06-09 has none, 06-10 has).
RACEIQ_START = date(2023, 6, 10)
DEFAULT_COUNTRIES = ("gb", "ie")

# Overall request rate across all worker threads, and the number of workers
# (latency is ~0.2-0.4 s, so a few workers are needed to reach the rate).
# A 429 pauses every worker; the rate can be raised once a run shows none.
DEFAULT_MAX_RPS = 10.0
DEFAULT_WORKERS = 12
# Days for which day lists are re-read and incomplete RaceiQ data re-fetched.
RECENT_DAYS = 10

FINAL_STATUSES = (200, 404)
# Bump when scrape_days starts collecting a new endpoint: days marked
# complete by an older version are revisited (only the missing payloads
# are fetched, the rest are cached).
SCRAPE_VERSION = 2
_TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}

# The bundle embeds the app config as an escaped JSON string: \"API_KEY\":\"...\"
_API_KEY_RE = re.compile(r'\\?"API_KEY\\?"\s*:\s*\\?"([0-9A-Za-z-]{20,64})\\?"')
_BUNDLE_RE = re.compile(r'(/_expo/static/js/web/[A-Za-z0-9_-]+\.js)')

RACE_KINDS = ("results", "racecard", "tips", "iq_metrics", "iq_sectional", "iq_comparison", "odds")


class RTVError(RuntimeError):
    """Base error for the Racing TV scraper."""


class RTVTransientError(RTVError):
    """Retries exhausted on rate limiting, server or network errors."""


class RTVForbiddenError(RTVError):
    """The API refused the request even with a freshly discovered key."""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class RTVClient:
    """Thread-safe, rate-limited JSON client for the Racing TV API.

    ``max_rps`` caps the request rate across every thread using the client
    (a shared schedule of request slots). A rate-limit or server error makes
    *all* threads wait out the backoff, not just the one that saw it.
    """

    def __init__(
        self,
        max_rps: float = DEFAULT_MAX_RPS,
        max_retries: int = 5,
        backoff_base: float = 5.0,
        api_key: str | None = None,
        session: requests.Session | None = None,
        request_delay: float | None = None,
    ) -> None:
        if request_delay is not None:  # older interface: seconds between requests
            max_rps = 1.0 / request_delay if request_delay > 0 else float("inf")
        self.max_rps = float(max_rps)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self._shared_session = session
        self._local = threading.local()
        self._lock = threading.Lock()
        self._key_lock = threading.Lock()
        self._next_slot = 0.0
        self._paused_until = 0.0
        self._api_key = api_key or os.environ.get("RTV_API_KEY") or None
        self._key_fresh = False
        self.n_requests = 0
        self.n_backoffs = 0

    @property
    def session(self) -> requests.Session:
        if self._shared_session is not None:
            return self._shared_session
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": config.USER_AGENT,
                "Accept": "application/json",
                "Origin": SITE_BASE,
                "Referer": SITE_BASE + "/",
            })
            self._local.session = s
        return s

    # -- key discovery -------------------------------------------------
    def discover_api_key(self, stale: str | None = None) -> str:
        """Read the public API key from the site's JavaScript config.

        With ``stale``, another thread may already have replaced that key;
        then its value is reused instead of fetching the bundle again.
        """
        with self._key_lock:
            # Another thread got here first: reuse its key unless it is the
            # one that was just refused.
            if self._api_key and self._api_key != stale:
                return self._api_key
            page = self._raw_get(SITE_BASE + "/results", accept="text/html")
            bundles = list(dict.fromkeys(_BUNDLE_RE.findall(page.text)))
            # The config lives in the app bundle ("index-*.js"); try it first.
            bundles.sort(key=lambda b: (not b.rsplit("/", 1)[-1].startswith("index-"), b))
            for bundle in bundles:
                js = self._raw_get(SITE_BASE + bundle, accept="*/*")
                m = _API_KEY_RE.search(js.text)
                if m:
                    self._api_key = m.group(1)
                    self._key_fresh = True
                    logger.info("RTV API key discovered from %s", bundle.rsplit("/", 1)[-1])
                    return self._api_key
            raise RTVError("Could not find API_KEY in the Racing TV site bundles")

    @property
    def api_key(self) -> str:
        if not self._api_key:
            self.discover_api_key()
        return self._api_key  # type: ignore[return-value]

    # -- requests ------------------------------------------------------
    def _throttle(self) -> None:
        """Wait for this thread's request slot under the shared rate cap."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot, self._paused_until)
            gap = 0.0 if self.max_rps == float("inf") else 1.0 / self.max_rps
            self._next_slot = slot + gap
            self.n_requests += 1
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def _pause_all(self, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)
            self.n_backoffs += 1

    def _raw_get(self, url: str, accept: str) -> requests.Response:
        self._throttle()
        r = self.session.get(url, headers={"Accept": accept}, timeout=30)
        r.raise_for_status()
        return r

    def get(self, path: str) -> tuple[int, bytes]:
        """GET an API path; return ``(status, body)`` for any final answer.

        Retries 429/5xx/network errors with exponential backoff (honouring
        ``Retry-After``, pausing every thread) and raises
        :class:`RTVTransientError` when they do not clear. A 403 triggers one
        key re-discovery before it is final.
        """
        url = API_BASE + path
        rediscovered = False
        attempt = 0
        while True:
            self._throttle()
            key = self.api_key
            try:
                r = self.session.get(url, headers={"API-KEY": key}, timeout=30)
                status, body = r.status_code, r.content
                retry_after = r.headers.get("Retry-After")
            except requests.RequestException as exc:
                status, body, retry_after = None, b"", None
                logger.warning("RTV request error on %s: %s", path, exc)

            if status == 403 and not rediscovered and not self._key_fresh:
                logger.warning("RTV 403 on %s; re-reading the API key", path)
                self.discover_api_key(stale=key)
                rediscovered = True
                continue
            if status is not None and status not in _TRANSIENT_STATUSES:
                if status == 403:
                    raise RTVForbiddenError(f"403 Forbidden for {path}")
                return status, body
            if attempt >= self.max_retries:
                raise RTVTransientError(f"{path}: still failing after {attempt} retries (last status {status})")
            delay = self.backoff_base * (2 ** attempt) + random.uniform(0, self.backoff_base)
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
            logger.warning("RTV %s on %s; pausing all requests %.0fs (retry %d/%d)",
                           status or "network error", path, delay, attempt + 1, self.max_retries)
            self._pause_all(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Raw response store
# ---------------------------------------------------------------------------

class RawStore:
    """SQLite store of raw API responses, keyed by API path."""

    def __init__(self, path: str = RAW_DB_PATH, *, readonly: bool = False) -> None:
        self.path = path
        if readonly:
            self.con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(
            """
            CREATE TABLE IF NOT EXISTS responses (
                path TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                race_date TEXT NOT NULL,
                http_status INTEGER NOT NULL,
                fetched_at TEXT NOT NULL,
                body BLOB
            );
            CREATE INDEX IF NOT EXISTS responses_kind_date ON responses(kind, race_date);
            CREATE TABLE IF NOT EXISTS days (
                race_date TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                n_meetings INTEGER,
                n_races INTEGER,
                n_races_done INTEGER,
                updated_at TEXT NOT NULL
            );
            -- Race-day-only payloads (pace maps, prompts, pre-race cards and
            -- odds): every capture is kept, since they change through the day
            -- and disappear afterwards.
            CREATE TABLE IF NOT EXISTS snapshots (
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                race_date TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                http_status INTEGER NOT NULL,
                body BLOB,
                PRIMARY KEY (path, fetched_at)
            );
            CREATE INDEX IF NOT EXISTS snapshots_kind_date ON snapshots(kind, race_date);
            """
        )
        cols = {r[1] for r in self.con.execute("PRAGMA table_info(days)")}
        if "version" not in cols:
            self.con.execute("ALTER TABLE days ADD COLUMN version INTEGER")
        if "profile" not in cols:
            self.con.execute("ALTER TABLE days ADD COLUMN profile TEXT NOT NULL DEFAULT 'research'")
        self.con.commit()

    def put_snapshot(self, path: str, kind: str, race_date: str, status: int, body: bytes,
                     fetched_at: str | None = None) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?)",
            (path, kind, race_date, fetched_at or _utc_now(), int(status), zlib.compress(body or b"", 9)),
        )
        self.con.commit()

    def snapshots(self, kind: str, start: str | None = None, end: str | None = None
                  ) -> Iterator[tuple[str, str, str, Any]]:
        """Yield ``(path, race_date, fetched_at, json)`` for successful captures."""
        sql = "SELECT path, race_date, fetched_at, body FROM snapshots WHERE kind=? AND http_status=200"
        args: list[Any] = [kind]
        if start:
            sql += " AND race_date>=?"
            args.append(start)
        if end:
            sql += " AND race_date<=?"
            args.append(end)
        for path, race_date, fetched_at, body in self.con.execute(sql + " ORDER BY race_date, path, fetched_at", args):
            yield path, race_date, fetched_at, json.loads(zlib.decompress(body).decode("utf-8"))

    def close(self) -> None:
        self.con.close()

    def status(self, path: str) -> int | None:
        row = self.con.execute("SELECT http_status FROM responses WHERE path=?", (path,)).fetchone()
        return None if row is None else int(row[0])

    def fetched_at(self, path: str) -> str | None:
        row = self.con.execute("SELECT fetched_at FROM responses WHERE path=?", (path,)).fetchone()
        return None if row is None else row[0]

    def put(self, path: str, kind: str, race_date: str, status: int, body: bytes) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?)",
            (path, kind, race_date, int(status), _utc_now(), zlib.compress(body or b"", 9)),
        )
        self.con.commit()

    def get_json(self, path: str) -> Any | None:
        row = self.con.execute(
            "SELECT http_status, body FROM responses WHERE path=?", (path,)
        ).fetchone()
        if row is None or int(row[0]) != 200 or row[1] is None:
            return None
        return json.loads(zlib.decompress(row[1]).decode("utf-8"))

    def paths(self, kind: str, start: str | None = None, end: str | None = None) -> list[tuple[str, str]]:
        sql = "SELECT path, race_date FROM responses WHERE kind=? AND http_status=200"
        args: list[Any] = [kind]
        if start:
            sql += " AND race_date>=?"
            args.append(start)
        if end:
            sql += " AND race_date<=?"
            args.append(end)
        return [(p, d) for p, d in self.con.execute(sql + " ORDER BY race_date, path", args)]

    def horse_profiles(self) -> Iterator[tuple[str, str, dict]]:
        """Yield the capture day, capture timestamp and payload from one read."""
        sql = ("SELECT race_date, fetched_at, body FROM responses "
               "WHERE kind='horse' AND http_status=200 AND body IS NOT NULL ORDER BY race_date, path")
        for day, fetched_at, body in self.con.execute(sql):
            horse = json.loads(zlib.decompress(body)).get("horse")
            if isinstance(horse, dict):
                yield day, fetched_at, horse

    def mark_day(self, race_date: str, status: str, n_meetings: int, n_races: int, n_done: int,
                 *, profile="research") -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO days (race_date, status, n_meetings, n_races, n_races_done, updated_at, version, profile)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (race_date, status, n_meetings, n_races, n_done, _utc_now(), SCRAPE_VERSION, profile),
        )
        self.con.commit()

    def day_status(self, race_date: str, *, profile=None) -> str | None:
        """The day's status, or ``'outdated'`` if an older scraper version set it."""
        row = self.con.execute("SELECT status, version, profile FROM days WHERE race_date=?", (race_date,)).fetchone()
        if row is None:
            return None
        if profile == "research" and row[2] != "research":
            return "core_only"
        return row[0] if row[1] == SCRAPE_VERSION else "outdated"

    def summary(self) -> pd.DataFrame:
        return pd.read_sql(
            "SELECT kind, http_status, COUNT(*) AS n, MIN(race_date) AS first, MAX(race_date) AS last "
            "FROM responses GROUP BY kind, http_status ORDER BY kind, http_status",
            self.con,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Paths and time helpers
# ---------------------------------------------------------------------------

def day_list_path(race_date: str) -> str:
    return f"/results/list/{race_date}"


def tipsters_path(race_date: str) -> str:
    return f"/tips/{race_date}/tipsters"


def track_path(track_slug: str) -> str:
    return f"/tracks/{track_slug.strip('/')}"


def race_paths(race_date: str, track_slug: str, hhmm: str) -> dict[str, str]:
    """API paths of every per-race endpoint (``track_slug`` without slashes).

    ``odds`` is a storage key, not a URL: the odds endpoint is queried by
    runner ids (see :func:`odds_url`), which are only known from the result.
    """
    base = f"{race_date}/{track_slug.strip('/')}/{hhmm}"
    return {
        "results": f"/results/{base}",
        "racecard": f"/racecards/{base}",
        "tips": f"/racecards/{base}/tips",
        "iq_metrics": f"/iq-results/{base}/metrics",
        "iq_sectional": f"/iq-results/{base}/sectional",
        "iq_comparison": f"/iq-results/{base}/comparison",
        "odds": f"/runner/odds#{base}",
    }


def odds_url(runner_ids: Iterable[Any]) -> str:
    """Bookmaker odds for the given runners (the site batches up to 200)."""
    return "/runner/odds?" + "&".join(f"runner_ids[]={int(i)}" for i in runner_ids)


def pre_race_paths(race_date: str, track_slug: str, hhmm: str) -> dict[str, str]:
    """Race-day-only endpoints: gone (404) once the day's racing is over."""
    base = f"{race_date}/{track_slug.strip('/')}/{hhmm}"
    return {
        "pre_racecard": f"/racecards/{base}",
        "pre_pace_maps": f"/iq-races/{base}/pace-maps",
        "pre_prompts": f"/iq-races/{base}/prompts",
        "pre_tips": f"/racecards/{base}/tips",
        "pre_odds": f"/runner/odds#{base}",
    }


_RESULTS_PATH_RE = re.compile(r"^/results/(\d{4}-\d{2}-\d{2})/([^/]+)/(\d{4})$")


def parse_results_path(path: str) -> tuple[str, str, str] | None:
    m = _RESULTS_PATH_RE.match(path)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp with offset; None when absent/invalid."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else None


def local_hhmm(start_time_scheduled: str) -> str | None:
    """URL time (UK/IE local wall clock) from a scheduled start with offset."""
    dt = parse_iso(start_time_scheduled)
    return dt.strftime("%H%M") if dt else None


def utc_off_time(start_time_scheduled: str) -> str | None:
    """``HH:MM`` in UTC, the convention of ``data/races.db``."""
    dt = parse_iso(start_time_scheduled)
    return dt.astimezone(timezone.utc).strftime("%H:%M") if dt else None


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

# Named margins, valued as in sporting_life._parse_lengths_behind so RTV and
# Sporting Life margins are on one scale.
_NAMED_MARGINS = {
    "dead heat": 0.0, "dht": 0.0, "dh": 0.0,
    "nose": 0.05, "nse": 0.05,
    "short head": 0.1, "shd": 0.1, "sh": 0.1,
    "head": 0.2, "hd": 0.2,
    "short neck": 0.2, "snk": 0.2,
    "neck": 0.3, "nk": 0.3,
}
_UNICODE_FRACTIONS = {"½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅓": " 1/3", "⅔": " 2/3"}


def parse_margin(text: Any) -> float:
    """Lengths from a Racing TV margin (``'2 1/2 l'``, ``'Neck'``, ``'Short Head'``).

    Unknown or unmeasured margins (``'Dist'``, blanks) are NaN, never zero.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return float("nan")
    s = str(text).strip().lower()
    for ch, rep in _UNICODE_FRACTIONS.items():
        s = s.replace(ch, rep)
    s = re.sub(r"\s*(lengths?|l)\.?$", "", s).strip()
    if not s:
        return float("nan")
    if s in _NAMED_MARGINS:
        return _NAMED_MARGINS[s]
    try:
        parts = s.split()
        if not 1 <= len(parts) <= 2:
            return float("nan")
        value = float(sum(Fraction(p) for p in parts))
    except (ValueError, ZeroDivisionError):
        return float("nan")
    return value if np.isfinite(value) and value >= 0 else float("nan")


_WIN_TIME_RE = re.compile(r"^\s*(?:(\d+)\s*m[,\s]*)?(?:(\d+(?:\.\d+)?)\s*s)?\s*$", re.I)


def parse_race_time(text: Any) -> float:
    """Seconds from ``'1m, 28.11s'`` / ``'59.3s'`` / ``'4m'``; NaN otherwise."""
    if not isinstance(text, str) or not text.strip():
        return float("nan")
    m = _WIN_TIME_RE.match(text)
    if not m or not (m.group(1) or m.group(2)):
        return float("nan")
    return (float(m.group(1)) * 60 if m.group(1) else 0.0) + (float(m.group(2)) if m.group(2) else 0.0)


def parse_weight_lbs(text: Any) -> float:
    """``'9-2'`` (stones-pounds) -> 128; the ``'0-0'`` placeholder -> NaN."""
    if not isinstance(text, str):
        return float("nan")
    m = re.fullmatch(r"\s*(\d{1,2})-(\d{1,2})\s*", text)
    if not m:
        return float("nan")
    lbs = float(int(m.group(1)) * 14 + int(m.group(2)))
    return lbs if lbs > 0 else float("nan")


def parse_equipment(text: Any) -> tuple[str | None, int | None]:
    """``'tp<sup>1</sup>'`` -> (``'tp'``, 1): codes, and whether first time."""
    if not isinstance(text, str) or not text.strip():
        return None, None
    first = 1 if re.search(r"<sup>\s*1\s*</sup>", text, re.I) else 0
    codes = re.sub(r"<sup>.*?</sup>", "", text, flags=re.I)
    codes = re.sub(r"<[^>]+>", "", codes).strip()
    return (codes or None), first


# Racing TV prints equipment as glyphs from a Timeform-style font, one glyph
# per item group, each optionally followed by <sup>n</sup> (first or second
# time in that equipment), groups separated by spaces: "w<sup>1</sup> z".
# Values are Sporting Life's symbols (b blinkers, v visor, h hood,
# p cheekpieces, t tongue strap, e/s eyeshield, e/c eye cover).
#
# The first block is the site's own lookup table (formatEquipmentText in the
# app bundle, 2026-10-29): 0:p +:tp B:b b:tb D:eb H:h h:ht V:v v:tv z:t w:w.
# The second block is glyphs the site leaves blank, decoded from the same
# runners' Sporting Life headgear (count of agreeing runners in the 13-day
# sample of 2026-10-29; every observation agreed).
EQUIPMENT_GLYPHS: dict[str, tuple[str, ...]] = {
    "0": ("p",), "+": ("p", "t"), "B": ("b",), "b": ("b", "t"),
    "D": ("b", "e/s"), "H": ("h",), "h": ("h", "t"), "V": ("v",),
    "v": ("v", "t"), "z": ("t",),
    # Inferred from Sporting Life headgear:
    "-": ("h", "p"),            # 10 runners
    "E": ("e/s",),              # 4
    "e": ("e/c", "t"),          # 2
    "A": ("h", "v"),            # 2
    "*": ("h", "p", "t"),       # 1
    "g": ("b", "e/s", "h", "t"),  # 1
    "d": ("b", "e/s"),          # 1 (no tongue strap, unlike b/h/v lower case)
    "C": ("e/s", "h"),          # 2
    "O": ("b", "h"),            # 2
    # From the full 2023-06..2026-10 backfill (467k runners, 2026-10-30):
    "}": ("p", "v"),            # 7 of 7
    "j": ("e/c", "t"),          # 4 of 4
    "J": ("e/c",),              # 2 of 2
    "a": ("h", "t", "v"),       # 2 of 2
    "=": ("e/s", "p"),          # 2 of 2
    # Split evidence, resolved by the lower-case = upper-case + tongue strap
    # convention (O = bh, g = bh e/s t) and the majority of observations:
    "o": ("b", "h", "t"),       # 9 of 11 (2 without t)
    "G": ("b", "e/s", "h"),     # 5 of 8 (3 without h)
    "F": ("e/s", "v"),          # 1
    "c": ("e/c", "h", "t"),     # 1
    "L": ("b", "e/c"),          # 1
    # "^" seen once, on a non-runner (Tribal Moon, Dundalk 2024-11-27):
    # no evidence, left undecoded.
}
WIND_OP_GLYPH = "w"
# Glyphs learned later by learn_equipment_glyphs() (see that function).
LEARNED_GLYPHS_FILE = os.path.join(RTV_DIR, "equipment_glyphs_learned.json")
_learned_glyphs: dict[str, tuple[str, ...]] | None = None


def _glyph_table() -> dict[str, tuple[str, ...]]:
    """The static glyph table plus any glyphs learned from the results DB."""
    global _learned_glyphs
    if _learned_glyphs is None:
        _learned_glyphs = {}
        if os.path.exists(LEARNED_GLYPHS_FILE):
            try:
                with open(LEARNED_GLYPHS_FILE, encoding="utf-8") as fh:
                    learned = json.load(fh).get("glyphs", {})
                _learned_glyphs = {g: tuple(v["symbols"]) for g, v in learned.items()}
            except (ValueError, OSError, KeyError, TypeError):
                logger.warning("Could not read %s; using the static glyph table", LEARNED_GLYPHS_FILE)
    return {**_learned_glyphs, **EQUIPMENT_GLYPHS}
_EQUIP_TOKEN_RE = re.compile(r"^(?P<glyph>.)(?:<sup>\s*(?P<times>\d+)\s*</sup>)?$", re.I)


def parse_equipment_tokens(text: Any) -> list[tuple[str, int | None]]:
    """``'w<sup>1</sup> z'`` -> ``[('w', 1), ('z', None)]``; unparseable -> ``[(text, None)]``."""
    if not isinstance(text, str) or not text.strip():
        return []
    out: list[tuple[str, int | None]] = []
    for token in text.strip().split():
        m = _EQUIP_TOKEN_RE.match(token)
        if not m:
            out.append((token, None))
            continue
        out.append((m.group("glyph"), int(m.group("times")) if m.group("times") else None))
    return out


def decode_equipment(text: Any) -> dict[str, Any]:
    """Equipment text -> Sporting Life symbols, first-time items and wind-op flags.

    ``decoded`` is False only for a glyph in neither table above; such runners
    are flagged by :func:`quality_checks` (``equipment_glyph_unknown``).
    """
    tokens = parse_equipment_tokens(text)
    table = _glyph_table()
    symbols: set[str] = set()
    first: set[str] = set()
    ambiguous: list[str] = []
    unknown: list[str] = []
    wind = wind_first = 0
    for glyph, times in tokens:
        if glyph == WIND_OP_GLYPH:
            wind = 1
            wind_first = int(times == 1)
            continue
        items = table.get(glyph)
        if items is None:
            unknown.append(glyph)
            continue
        symbols.update(items)
        if times == 1:
            # On a combined glyph the mark does not say which item is new
            # ("+1" is p1,t or p,t1 in Sporting Life); keep the glyph instead.
            if len(items) == 1:
                first.update(items)
            else:
                ambiguous.append(glyph)
    return {
        "symbols": ",".join(sorted(symbols)) or None,
        "first_time": ",".join(sorted(first)) or None,
        "first_time_ambiguous": "".join(ambiguous) or None,
        "wind_op": wind if tokens else None,
        "wind_op_first_time": wind_first if tokens else None,
        "decoded": not unknown,
        "unknown": "".join(unknown) or None,
    }


def learn_equipment_glyphs(store: RawStore | None = None, db_path: str = RESULTS_DB_PATH,
                           min_count: int = 2, write: bool = True) -> dict[str, Any]:
    """Decode unknown glyphs from Sporting Life headgear for the same runners.

    For every stored runner whose equipment holds exactly one glyph not in
    the table (besides ``w``), the runner's ``results.headgear`` symbols in
    ``data/races.db`` are collected. A glyph is learned only when it has at
    least ``min_count`` observations and every one of them agrees; anything
    else is reported, not guessed. Learned glyphs are written to
    :data:`LEARNED_GLYPHS_FILE` and used by :func:`decode_equipment`.
    """
    global _learned_glyphs
    own_store = store is None
    store = store or RawStore()
    table = _glyph_table()
    seen: list[tuple[str, str, str]] = []  # (glyph, race_date, horse key)
    try:
        for path, race_date in store.paths("results"):
            for r in ((store.get_json(path) or {}).get("race") or {}).get("runners") or []:
                tokens = [g for g, _ in parse_equipment_tokens(r.get("equipment_text")) if g != WIND_OP_GLYPH]
                unknown = [g for g in tokens if g not in table]
                if len(tokens) == 1 and len(unknown) == 1:
                    seen.append((unknown[0], race_date, normalise_horse_key(r.get("horse_name"))))
    finally:
        if own_store:
            store.close()
    evidence: dict[str, list[str]] = {}
    if seen and os.path.exists(db_path):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for glyph, race_date, horse in seen:
                for (hg,) in con.execute("SELECT headgear FROM results WHERE race_date=? AND horse_name=?",
                                         (race_date, horse)):
                    sym = ",".join(sorted({s.rstrip("1") for s in _headgear_symbols(hg).split(",") if s}))
                    evidence.setdefault(glyph, []).append(sym)
        finally:
            con.close()
    learned, unresolved = {}, {}
    for glyph, syms in evidence.items():
        values = set(syms)
        if len(syms) >= min_count and len(values) == 1 and next(iter(values)):
            learned[glyph] = {"symbols": next(iter(values)).split(","), "count": len(syms)}
        else:
            unresolved[glyph] = {"observations": len(syms), "seen_as": sorted(values)}
    for glyph, _, _ in seen:
        if glyph not in evidence:
            unresolved.setdefault(glyph, {"observations": 0, "seen_as": []})
    if write and learned:
        existing = {}
        if os.path.exists(LEARNED_GLYPHS_FILE):
            with open(LEARNED_GLYPHS_FILE, encoding="utf-8") as fh:
                existing = json.load(fh).get("glyphs", {})
        existing.update(learned)
        os.makedirs(os.path.dirname(LEARNED_GLYPHS_FILE), exist_ok=True)
        with open(LEARNED_GLYPHS_FILE, "w", encoding="utf-8") as fh:
            json.dump({"updated_at": _utc_now(), "glyphs": existing}, fh, indent=2)
        _learned_glyphs = None  # reload on next decode
    return {"learned": learned, "unresolved": unresolved}


def strip_markup(text: Any) -> Any:
    """Remove Racing TV's HTML/React-Native markup from a text field."""
    if not isinstance(text, str):
        return text
    s = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", s).strip()


def to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        f = float(str(value).replace(",", "").strip().rstrip("%"))
    except ValueError:
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _distance_yards_from_text(text: Any) -> float:
    """``'2m 6f 7y'`` -> 4847 (cross-check for the numeric yards field)."""
    if not isinstance(text, str):
        return float("nan")
    m = re.fullmatch(r"\s*(?:(\d+)m)?\s*(?:(\d+)f)?\s*(?:(\d+)y)?\s*", text)
    if not m or not any(m.groups()):
        return float("nan")
    miles, furlongs, yards = (int(g) if g else 0 for g in m.groups())
    return float(miles * 1760 + furlongs * 220 + yards)


def _sector_furlongs(name: Any) -> float:
    """Sector label (``'7f'``, ``'1m'``, ``'2m 5f'``) as furlongs from the start."""
    y = _distance_yards_from_text(name)
    return y / 220.0 if np.isfinite(y) else float("nan")


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

class _Task(NamedTuple):
    key: str        # storage key (the API path, or a synthetic key for odds)
    url: str        # API path actually requested
    kind: str
    race_date: str


def _needs_fetch(store: RawStore, key: str, force: bool = False) -> bool:
    return force or store.status(key) not in FINAL_STATUSES


def _run_tasks(client: RTVClient, store: RawStore, tasks: list[_Task], workers: int,
               snapshot: bool = False) -> dict[str, int]:
    """Fetch tasks on a thread pool; store each answer from this thread.

    The store is only touched here (SQLite connections are per-thread). A
    transient failure cancels the remaining tasks and is re-raised; every
    answer received before it is already stored, so a rerun resumes. With
    ``snapshot`` the answers are appended to the snapshots table instead.
    """
    statuses: dict[str, int] = {}
    if not tasks:
        return statuses
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = {pool.submit(client.get, t.url): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            status, body = fut.result()
            if snapshot:
                store.put_snapshot(t.key, t.kind, t.race_date, status, body)
            else:
                store.put(t.key, t.kind, t.race_date, status, body)
            statuses[t.key] = status
            if status not in FINAL_STATUSES:
                logger.warning("RTV %s returned HTTP %s (stored, will retry next run)", t.url, status)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return statuses


def _iq_needs_refresh(results: dict | None, race_date: str, today: date) -> bool:
    """Recent races whose RaceiQ data Racing TV still marks as incomplete."""
    if not results:
        return False
    if (today - date.fromisoformat(race_date)).days > RECENT_DAYS:
        return False
    state = (results.get("race") or {}).get("race_iq_state") or {}
    return bool(
        (state.get("show_metrics") and not state.get("metrics_complete"))
        or (state.get("show_sectionals") and not state.get("sectionals_complete"))
    )


def list_races(day_list: dict, countries: Iterable[str] | None = DEFAULT_COUNTRIES,
               key: str = "completed_races") -> list[dict]:
    """Races in a day list, each with its URL slug, local and UTC time and meeting.

    ``key`` is ``completed_races`` for result lists and ``races`` for racecard
    lists (``/racecards/list/{date}``).
    """
    allowed = None if countries is None else {c.lower() for c in countries}
    out = []
    for meeting in day_list.get("meetings") or []:
        track = meeting.get("track") or {}
        iso = ((track.get("country_code") or {}).get("iso") or "").lower()
        if allowed is not None and iso not in allowed:
            continue
        slug = (track.get("slug") or "").strip("/")
        for race in meeting.get(key) or []:
            hhmm = local_hhmm(race.get("start_time_scheduled"))
            if not slug or not hhmm:
                logger.warning("RTV list entry without slug/time: %s", race.get("id"))
                continue
            out.append({
                "race_id": race.get("id"), "track_slug": slug, "hhmm": hhmm,
                "track_name": track.get("name"), "off_time_utc": utc_off_time(race.get("start_time_scheduled")),
                "country_iso": iso, "meeting_id": meeting.get("id"),
                "status": (race.get("status") or {}).get("state"), "source": "day_list",
            })
    return out


# Racing TV URL slugs for results-DB track names that do not slugify directly
# (from the deleted 2026 scraper's mapping; Irish names slugify as-is).
_DB_TRACK_SLUGS: dict[str, str] = {
    "Royal Ascot": "ascot", "Catterick": "catterick-bridge", "Fontwell": "fontwell-park",
    "Hamilton": "hamilton-park", "Haydock": "haydock-park", "Kempton": "kempton-park",
    "Lingfield": "lingfield-park", "Sandown": "sandown-park", "Stratford": "stratford-on-avon",
}
_UK_TZ = ZoneInfo("Europe/London")  # Irish racing keeps the same clock


def _db_track_slug(track: str) -> str:
    if track in _DB_TRACK_SLUGS:
        return _DB_TRACK_SLUGS[track]
    return re.sub(r"[\s_]+", "-", re.sub(r"[^\w\s-]", "", str(track).lower())).strip("-")


def discover_unlisted_races(race_date: str, listed: list[dict], db_path: str = RESULTS_DB_PATH,
                            tolerance_min: int = 5) -> list[dict]:
    """Races in ``data/races.db`` that Racing TV's day list leaves out.

    Racing TV's older day lists omit some races that its results endpoint
    still serves (2021-07-10 lacks the first race at Newmarket, York and
    Chester). Each results-DB race with no listed race at the same track
    within ``tolerance_min`` minutes is returned for a direct fetch; a wrong
    guess costs one 404.
    """
    if not os.path.exists(db_path):
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT DISTINCT track, off_time FROM results WHERE race_date=?",
                           (race_date,)).fetchall()
    finally:
        con.close()
    listed_min: dict[str, list[int]] = {}
    for r in listed:
        if r.get("off_time_utc"):
            h, m = map(int, r["off_time_utc"].split(":"))
            listed_min.setdefault(_track_join_key(r.get("track_name") or r["track_slug"]), []).append(h * 60 + m)
    out = []
    for track, off in rows:
        off = normalise_off_time_key(off)
        if not off or ":" not in off:
            continue
        h, m = map(int, off.split(":"))
        if any(abs(h * 60 + m - t) <= tolerance_min for t in listed_min.get(_track_join_key(track), [])):
            continue
        utc = datetime.fromisoformat(f"{race_date}T{off}:00+00:00")
        out.append({
            "race_id": None, "track_slug": _db_track_slug(track),
            "hhmm": utc.astimezone(_UK_TZ).strftime("%H%M"), "track_name": track, "off_time_utc": off,
            "country_iso": None, "meeting_id": None, "status": None, "source": "results_db",
        })
    return out


def scrape_days(
    race_dates: list[str],
    client: RTVClient,
    store: RawStore,
    countries: Iterable[str] | None = DEFAULT_COUNTRIES,
    with_comparison: bool = False,
    force: bool = False,
    today: date | None = None,
    workers: int = DEFAULT_WORKERS,
    discover: bool = True,
    profile: str = "research",
) -> dict[str, dict[str, int]]:
    """Fetch every endpoint for every race on *race_dates*; resumable.

    The days share each phase's thread pool, so short days do not leave
    workers idle. Phase 1 fetches day lists and tipsters; phase 2 each race's
    results, racecard and tips (plus unseen track pages); phase 3 the
    payloads that need the result: RaceiQ metrics and sectionals (when
    Racing TV flags them) and bookmaker odds (queried by runner id).
    Returns a summary per day.
    """
    today = today or date.today()
    if profile not in {"core", "research"}:
        raise ValueError("RTV profile must be core or research")
    iq_kinds = ["iq_metrics", "iq_sectional"] + (["iq_comparison"] if with_comparison else [])

    # Phase 1: day lists and tipsters.
    phase1 = []
    for d in race_dates:
        recent = (today - date.fromisoformat(d)).days <= RECENT_DAYS
        listings = [(day_list_path(d), "day_list")]
        if profile == "research":
            listings.append((tipsters_path(d), "tipsters"))
        for path, kind in listings:
            if _needs_fetch(store, path, force or recent):
                phase1.append(_Task(path, path, kind, d))
    _run_tasks(client, store, phase1, workers)

    races_by_day: dict[str, list[dict]] = {}
    for d in race_dates:
        if store.status(day_list_path(d)) != 200:
            store.mark_day(d, f"list_{store.status(day_list_path(d))}", 0, 0, 0)
            continue
        races = list_races(store.get_json(day_list_path(d)) or {}, countries)
        if discover:
            races += discover_unlisted_races(d, races)
        seen: set[str] = set()
        races_by_day[d] = [r for r in races if not (f"{r['track_slug']}/{r['hhmm']}" in seen
                                                    or seen.add(f"{r['track_slug']}/{r['hhmm']}"))]

    # Phase 2: results, racecard, tips; track pages once per course.
    refresh: dict[str, bool] = {}
    phase2: list[_Task] = []
    slugs: dict[str, str] = {}
    for d, races in races_by_day.items():
        for race in races:
            paths = race_paths(d, race["track_slug"], race["hhmm"])
            # A stored result whose RaceiQ data Racing TV still marks
            # incomplete (recent races only) is fetched again, with its
            # RaceiQ payloads.
            refresh[paths["results"]] = _iq_needs_refresh(store.get_json(paths["results"]), d, today)
            kinds = ("results", "racecard", "tips") if profile == "research" else ("results", "racecard")
            recent = 0 <= (today-date.fromisoformat(d)).days <= RECENT_DAYS
            for kind in kinds:
                fetched = store.fetched_at(paths[kind])
                due = not fetched or (datetime.now(timezone.utc)-datetime.fromisoformat(fetched)).total_seconds() >= 21600
                revising = recent and due and kind in {"results", "racecard"}
                if _needs_fetch(store, paths[kind], force or revising or (kind == "results" and refresh[paths["results"]])):
                    phase2.append(_Task(paths[kind], paths[kind], kind, d))
            if race["source"] == "day_list":
                slugs.setdefault(race["track_slug"], d)
    for slug, d in sorted(slugs.items()):
        if _needs_fetch(store, track_path(slug)):
            phase2.append(_Task(track_path(slug), track_path(slug), "track", d))
    _run_tasks(client, store, phase2, workers)

    # Phase 3: RaceiQ and odds, which need the stored result.
    phase3: list[_Task] = []
    for d, races in races_by_day.items():
        for race in races:
            paths = race_paths(d, race["track_slug"], race["hhmm"])
            results = store.get_json(paths["results"])
            if not results:
                continue
            race_obj = results.get("race") or {}
            state = race_obj.get("race_iq_state") or {}
            if state.get("show_metrics") or state.get("show_sectionals"):
                for kind in iq_kinds:
                    if _needs_fetch(store, paths[kind], force or refresh[paths["results"]]):
                        phase3.append(_Task(paths[kind], paths[kind], kind, d))
            runner_ids = [r.get("id") for r in race_obj.get("runners") or [] if r.get("id")]
            if profile == "research" and runner_ids and _needs_fetch(store, paths["odds"], force):
                phase3.append(_Task(paths["odds"], odds_url(runner_ids), "odds", d))
    _run_tasks(client, store, phase3, workers)

    # A race is done when every payload it should have holds a final answer.
    summaries: dict[str, dict[str, int]] = {}
    for d, races in races_by_day.items():
        done = 0
        for race in races:
            paths = race_paths(d, race["track_slug"], race["hhmm"])
            ok = store.status(paths["results"]) in FINAL_STATUSES
            results = store.get_json(paths["results"])
            if results:
                # Odds are queried by runner id: an abandoned race with no
                # runners has none to ask for.
                has_runners = bool((results.get("race") or {}).get("runners"))
                needed = ["racecard"]
                if profile == "research":
                    needed += ["tips"] + (["odds"] if has_runners else [])
                state = (results.get("race") or {}).get("race_iq_state") or {}
                if state.get("show_metrics") or state.get("show_sectionals"):
                    needed += iq_kinds
                ok &= all(store.status(paths[k]) in FINAL_STATUSES for k in needed)
            done += int(ok)
        listed = [r for r in races if r["source"] == "day_list"]
        store.mark_day(d, "complete" if done == len(races) else "partial",
                       len({r["meeting_id"] for r in listed}), len(races), done, profile=profile)
        probed = [r for r in races if r["source"] == "results_db"]
        summaries[d] = {
            "races": len(races), "done": done, "unlisted_probed": len(probed),
            "unlisted_found": sum(1 for r in probed
                                  if store.status(race_paths(d, r["track_slug"], r["hhmm"])["results"]) == 200),
        }
    return summaries


def scrape_day(race_date: str, client: RTVClient, store: RawStore, *args, **kwargs) -> dict[str, int]:
    """:func:`scrape_days` for a single day."""
    return scrape_days([race_date], client, store, *args, **kwargs).get(race_date, {"races": 0, "done": 0})


def scrape_range(
    start: str,
    end: str,
    countries: Iterable[str] | None = DEFAULT_COUNTRIES,
    max_rps: float = DEFAULT_MAX_RPS,
    with_comparison: bool = False,
    force: bool = False,
    newest_first: bool = True,
    store: RawStore | None = None,
    client: RTVClient | None = None,
    workers: int = DEFAULT_WORKERS,
    discover: bool = True,
    batch_days: int = 7,
) -> dict[str, int]:
    """Scrape every day in ``[start, end]`` in batches; skips days already complete."""
    own_store = store is None
    store = store or RawStore()
    client = client or RTVClient(max_rps=max_rps)
    today = date.today()
    days = [date.fromisoformat(start) + timedelta(days=i)
            for i in range((date.fromisoformat(end) - date.fromisoformat(start)).days + 1)]
    if newest_first:
        days.reverse()
    todo = [d.isoformat() for d in days
            if force or (today - d).days <= RECENT_DAYS or store.day_status(d.isoformat(), profile="research") != "complete"]
    totals = {"days": 0, "races": 0, "done": 0, "unlisted_found": 0}
    t0 = time.monotonic()
    try:
        for i in range(0, len(todo), max(1, batch_days)):
            batch = todo[i:i + batch_days]
            before, t_batch = client.n_requests, time.monotonic()
            summaries = scrape_days(batch, client, store, countries, with_comparison, force, today,
                                    workers, discover)
            n_req = client.n_requests - before
            for d in batch:
                s = summaries.get(d, {"races": 0, "done": 0})
                totals["days"] += 1
                for k in ("races", "done", "unlisted_found"):
                    totals[k] += s.get(k, 0)
                logger.info("RTV %s: %d/%d races complete, %d unlisted found",
                            d, s["done"], s["races"], s.get("unlisted_found", 0))
            logger.info("RTV batch %s..%s: %d requests, %.1f req/s, %d/%d days done",
                        batch[-1], batch[0], n_req, n_req / max(time.monotonic() - t_batch, 1e-6),
                        min(i + batch_days, len(todo)), len(todo))
    finally:
        if own_store:
            store.close()
    totals["seconds"] = round(time.monotonic() - t0, 1)
    totals["requests"] = client.n_requests
    totals["backoffs"] = client.n_backoffs
    return totals


def racecard_list_path(race_date: str) -> str:
    return f"/racecards/list/{race_date}"


def snapshot_day(
    race_date: str | None = None,
    client: RTVClient | None = None,
    store: RawStore | None = None,
    countries: Iterable[str] | None = DEFAULT_COUNTRIES,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, int]:
    """Capture the race-day-only payloads for every race on *race_date*.

    Pace maps (each runner's average early position and category, course
    run-style statistics, draw advantage) and RaceiQ prompts exist only on
    race day: the next day both return 404. The pre-race racecard and
    bookmaker odds are captured alongside, so each snapshot is a consistent
    picture of the card at ``fetched_at``. Run it on race day (e.g. morning
    and again near the first race); every capture is kept.
    """
    race_date = race_date or date.today().isoformat()
    own_store = store is None
    store = store or RawStore()
    client = client or RTVClient()
    try:
        list_path = racecard_list_path(race_date)
        _run_tasks(client, store, [_Task(list_path, list_path, "pre_list", race_date)], 1, snapshot=True)
        latest = None
        for _, _, _, payload in store.snapshots("pre_list", race_date, race_date):
            latest = payload
        races = list_races(latest or {}, countries, key="races")
        tasks = []
        for race in races:
            paths = pre_race_paths(race_date, race["track_slug"], race["hhmm"])
            tasks += [_Task(paths[k], paths[k], k, race_date) for k in ("pre_racecard", "pre_pace_maps",
                                                                        "pre_prompts", "pre_tips")]
        statuses = _run_tasks(client, store, tasks, workers, snapshot=True)
        # Odds by runner id, from the cards just captured.
        odds_tasks = []
        cards = {p: j for p, _, _, j in store.snapshots("pre_racecard", race_date, race_date)}
        for race in races:
            paths = pre_race_paths(race_date, race["track_slug"], race["hhmm"])
            ids = [r.get("id") for r in (cards.get(paths["pre_racecard"]) or {}).get("runners") or [] if r.get("id")]
            if ids:
                odds_tasks.append(_Task(paths["pre_odds"], odds_url(ids), "pre_odds", race_date))
        statuses.update(_run_tasks(client, store, odds_tasks, workers, snapshot=True))
    finally:
        if own_store:
            store.close()
    ok = sum(1 for s in statuses.values() if s == 200)
    return {"races": len(races), "requests": len(statuses) + 1, "ok": ok,
            "pace_maps": sum(1 for k, s in statuses.items() if "/pace-maps" in k and s == 200)}


def horse_profile_path(slug: str) -> str:
    return f"/horses/{slug.strip('/')}"


def scrape_horse_profiles(
    slugs: Iterable[str],
    client: RTVClient | None = None,
    store: RawStore | None = None,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
) -> dict[str, int]:
    """Fetch horse profile pages (foaling date, pedigree with countries, form).

    Profiles describe the horse *as of the scrape*: age, last run, future
    entries and form all move on, so never treat them as race-day values.
    The form list (``performances``) includes runs outside GB/IE.
    """
    own_store = store is None
    store = store or RawStore()
    client = client or RTVClient()
    today = date.today().isoformat()
    try:
        tasks = [_Task(horse_profile_path(s), horse_profile_path(s), "horse", today)
                 for s in sorted({s.strip("/") for s in slugs if s})
                 if _needs_fetch(store, horse_profile_path(s), force)]
        statuses = _run_tasks(client, store, tasks, workers)
    finally:
        if own_store:
            store.close()
    return {"requested": len(tasks), "ok": sum(s == 200 for s in statuses.values()),
            "unavailable": sum(s == 404 for s in statuses.values()),
            "retryable": sum(s not in FINAL_STATUSES for s in statuses.values())}


def build_snapshot_tables(start: str | None = None, end: str | None = None,
                          store: RawStore | None = None) -> dict[str, pd.DataFrame]:
    """Pace maps, course run-style statistics, prompts and pre-race odds.

    Keyed by ``race_key`` (``date/track/hhmm``) and ``fetched_at``; ``race_id``
    comes from the racecard captured in the same snapshot run.
    """
    own_store = store is None
    store = store or RawStore()
    out: dict[str, list[dict]] = {"pace_maps": [], "pace_course_stats": [], "prompts": [], "pre_odds": [],
                                  "pre_runners": []}
    try:
        race_ids: dict[str, Any] = {}
        for path, race_date, fetched_at, j in store.snapshots("pre_racecard", start, end):
            key = path.split("/racecards/", 1)[1]
            race = j.get("race") or {}
            race_ids[key] = race.get("id")
            for r in j.get("runners") or []:
                out["pre_runners"].append({"race_key": key, "race_id": race.get("id"), "fetched_at": fetched_at,
                                           "card_analyst_verdict": race.get("analyst_verdict"),
                                           "runner_id": r.get("id"), **_flatten(r, "card", _RUNNER_SKIP)})
        for path, race_date, fetched_at, j in store.snapshots("pre_pace_maps", start, end):
            key = path.split("/iq-races/", 1)[1].rsplit("/", 1)[0]
            base = {"race_key": key, "race_id": race_ids.get(key), "fetched_at": fetched_at,
                    **_flatten(j.get("race") or {}, "pm_race")}
            for s in j.get("course_statistics") or []:
                out["pace_course_stats"].append({**base, **_flatten(s, "stat")})
            for r in j.get("runners") or []:
                out["pace_maps"].append({**base, **_flatten(r, "pm")})
        for path, race_date, fetched_at, j in store.snapshots("pre_prompts", start, end):
            key = path.split("/iq-races/", 1)[1].rsplit("/", 1)[0]
            for r in j.get("runners") or []:
                for name, p in (r.get("prompts") or {}).items():
                    if isinstance(p, dict) and p.get("prompt"):
                        out["prompts"].append({"race_key": key, "race_id": race_ids.get(key), "fetched_at": fetched_at,
                                               "runner_id": r.get("id"), "prompt_type": name,
                                               "title": p.get("title"), "prompt": p.get("prompt")})
        for path, race_date, fetched_at, j in store.snapshots("pre_odds", start, end):
            key = path.split("#", 1)[1]
            for row in parse_odds(j, race_ids.get(key)):
                out["pre_odds"].append({"race_key": key, "fetched_at": fetched_at, **row})
    finally:
        if own_store:
            store.close()
    tables = {k: pd.DataFrame(v) for k, v in out.items()}
    pm = tables["pace_maps"]
    for col in ("pm_pace_map__position", "pm_draw__lengths_advantage", "pm_draw__lengths_advantage_significance"):
        if col in pm:
            pm[col] = pd.to_numeric(pm[col], errors="coerce")
    return tables


def build_profile_tables(store: RawStore | None = None) -> dict[str, pd.DataFrame]:
    """Horse profiles (scrape-time) and their full form lists."""
    own_store = store is None
    store = store or RawStore()
    profiles, perfs, stats = [], [], []
    try:
        for scraped, _, horse in store.horse_profiles():
            hid = horse.get("id")
            profiles.append({"horse_id": hid, "scraped_on": scraped,
                             **_flatten(horse, "scrape_horse", {"performances", "statistics", "future_runners"})})
            for p in horse.get("performances") or []:
                race = {k: v for k, v in (p.get("race") or {}).items() if k != "replay_videos"}
                perfs.append({"horse_id": hid, "scraped_on": scraped, **_flatten({**p, "race": race}, "perf")})
            for s in horse.get("statistics") or []:
                stats.append({"horse_id": hid, "scraped_on": scraped, **_flatten(s, "scrape_stat")})
    finally:
        if own_store:
            store.close()
    return {"horse_profiles": pd.DataFrame(profiles), "horse_performances": pd.DataFrame(perfs),
            "horse_statistics": pd.DataFrame(stats)}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _flatten(obj: dict, prefix: str, skip: Iterable[str] = (), out: dict | None = None,
             sep: str = "_") -> dict:
    """Flatten nested dicts into scalars; lists become JSON text.

    Top-level keys join the prefix with ``_`` and nested keys with ``__``
    (``res_trainer_name`` vs ``res_trainer__name``): the payloads carry both
    a ``trainer_name`` field and a ``trainer.name`` object, with different
    values, so a single separator would silently overwrite one of them.
    """
    out = {} if out is None else out
    skip = set(skip)
    for key, value in obj.items():
        if key in skip:
            continue
        name = f"{prefix}{sep}{key}"
        if isinstance(value, dict):
            _flatten(value, name, (), out, sep="__")
        elif isinstance(value, list):
            out[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value else None
        else:
            out[name] = value
    return out


# Media and navigation blobs: links to images/videos, not race data.
_RACE_SKIP = {"replay_videos", "custom_tabs", "meeting", "runners", "totes", "sectionals"}
_RUNNER_SKIP = {"silk", "sectionals"}
# Fields that describe the horse as of the scrape, not race day. Prefixed
# ``scrape_`` so nobody mistakes them for race-day values (cf. the ``sex``
# look-ahead leak).
_SCRAPE_TIME_FIELDS = {
    "card_horse__age": "scrape_card_horse__age",
    "card_horse__age_days": "scrape_card_horse__age_days",
    "card_horse__gender": "scrape_card_horse__gender",
}


def _silk_filename(silk: Any) -> str | None:
    if not isinstance(silk, dict):
        return None
    paf = silk.get("press_association_file") or {}
    return paf.get("filename") or silk.get("filename")


def parse_race(
    results: dict,
    racecard: dict | None = None,
    metrics: dict | None = None,
    sectional: dict | None = None,
    path: str | None = None,
) -> dict[str, list[dict]]:
    """Turn one race's payloads into table rows (no derived columns yet)."""
    race = results.get("race") or {}
    race_id = race.get("id")
    meeting = race.get("meeting") or {}
    track = meeting.get("track") or {}
    url_date, url_slug, url_hhmm = parse_results_path(path) if path else (None, None, None)

    row: dict[str, Any] = {
        "race_id": race_id,
        "meeting_id": meeting.get("id"),
        "race_date": meeting.get("date") or url_date,
        "track_id": track.get("id"),
        "track_name": track.get("name"),
        "track_slug": (track.get("slug") or "").strip("/") or url_slug,
        "url_hhmm": url_hhmm,
        "track_tv_channel": track.get("tv_channel"),
        "track_country_name": track.get("country_name"),
        "meeting_track_condition": meeting.get("track_condition"),
        "meeting_weather_icon": meeting.get("weather_icon"),
        "meeting_status": (meeting.get("status") or {}).get("state"),
        "meeting_n_races": len(meeting.get("ordered_races") or []) or None,
        "tpd_sectional_pdfs": "|".join(
            (s.get("pdf") or {}).get("filename") or "" for s in race.get("sectionals") or []
        ) or None,
        "n_replay_videos": len(race.get("replay_videos") or []),
    }
    _flatten(race, "res", _RACE_SKIP, row)

    card_race: dict = {}
    smart_stats: list[dict] = []
    last_winners: list[dict] = []
    if racecard:
        card_race = racecard.get("race") or {}
        card_meeting = card_race.get("meeting") or {}
        card_track = card_meeting.get("track") or {}
        row["country_iso"] = ((card_track.get("country_code") or {}).get("iso") or "").lower() or None
        row["country_racing"] = (card_track.get("country_code") or {}).get("racing")
        # The meeting's copy of this race sometimes carries fields the race
        # object leaves null (draw_comment, rating_limit_range).
        twin = next((r for r in card_meeting.get("races") or [] if r.get("id") == race_id), {})
        _flatten(card_race, "card", {"meeting", "custom_tabs"}, row)
        # Keep only the copy's fields that add to or contradict the race's own.
        for key, value in _flatten(twin, "card", {"meeting"}).items():
            if value is not None and row.get(key) != value:
                row[key.replace("card_", "card_meeting_copy_", 1)] = value
        row["card_sort"] = racecard.get("sort")
        for s in racecard.get("smart_stats") or []:
            smart_stats.append({"race_id": race_id, **_flatten(s, "ss")})
        for w in racecard.get("last_winners") or []:
            last_winners.append({"race_id": race_id, **_flatten(w, "lw")})

    if sectional:
        sect_race = sectional.get("race") or {}
        _flatten(sect_race, "sect", {"runners"}, row)

    totes = [
        {"race_id": race_id, **_flatten(t, "tote")} for t in race.get("totes") or []
    ]

    # Runners: results is the spine; racecard/sectional/metrics join on runner id.
    card_runners = {r.get("id"): r for r in (racecard or {}).get("runners") or []}
    sect_runners = {r.get("id"): r for r in ((sectional or {}).get("race") or {}).get("runners") or []}
    runner_rows: list[dict] = []
    ids_seen = set()
    for r in race.get("runners") or []:
        rid = r.get("id")
        ids_seen.add(rid)
        rr: dict[str, Any] = {"race_id": race_id, "runner_id": rid,
                              "res_silk_filename": _silk_filename(r.get("silk"))}
        _flatten(r, "res", _RUNNER_SKIP, rr)
        if rid in card_runners:
            c = card_runners[rid]
            rr["card_silk_filename"] = _silk_filename(c.get("silk"))
            _flatten(c, "card", _RUNNER_SKIP, rr)
        if rid in sect_runners:
            _flatten(sect_runners[rid], "sect", _RUNNER_SKIP, rr)
        for old, new in _SCRAPE_TIME_FIELDS.items():
            if old in rr:
                rr[new] = rr.pop(old)
        rr["in_results"] = True
        rr["in_racecard"] = rid in card_runners
        rr["in_sectional"] = rid in sect_runners
        runner_rows.append(rr)
    # Declared runners missing from the result (reserves, late withdrawals).
    for rid, c in card_runners.items():
        if rid in ids_seen:
            continue
        rr = {"race_id": race_id, "runner_id": rid, "card_silk_filename": _silk_filename(c.get("silk"))}
        _flatten(c, "card", _RUNNER_SKIP, rr)
        for old, new in _SCRAPE_TIME_FIELDS.items():
            if old in rr:
                rr[new] = rr.pop(old)
        rr.update(in_results=False, in_racecard=True, in_sectional=rid in sect_runners)
        runner_rows.append(rr)

    metric_rows: list[dict] = []
    for r in (metrics or {}).get("runners") or []:
        for group, items in (r.get("race_iq") or {}).items():
            if not isinstance(items, dict):
                continue
            for metric, m in items.items():
                if not isinstance(m, dict):
                    continue
                metric_rows.append({
                    "race_id": race_id, "runner_id": r.get("id"), "group": group, "metric": metric,
                    "metric_title": m.get("metric_title"), "value_raw": m.get("value"),
                    "value": to_float(m.get("value")), "unit": m.get("unit"),
                    "rank": m.get("rank"), "average": to_float(m.get("average")),
                    "percentage": to_float(m.get("percentage")),
                    "extra": json.dumps({k: v for k, v in m.items() if k not in
                                         {"metric_title", "value", "unit", "rank", "average", "percentage"}},
                                        separators=(",", ":")) if set(m) - {"metric_title", "value", "unit", "rank", "average", "percentage"} else None,
                })

    sector_rows: list[dict] = []
    for rid, s in sect_runners.items():
        for sec in s.get("sectionals") or []:
            sector_rows.append({
                "race_id": race_id, "runner_id": rid,
                **{k: sec.get(k) for k in sec},
                "sector_furlongs": _sector_furlongs(sec.get("sector_name")),
            })

    return {
        "races": [row], "runners": runner_rows, "metrics": metric_rows,
        "sectionals": sector_rows, "totes": totes,
        "smart_stats": smart_stats, "last_winners": last_winners,
    }


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------

def _derive_races(races: pd.DataFrame) -> pd.DataFrame:
    df = races.copy()
    sched = df.get("res_start_time_scheduled", pd.Series(index=df.index, dtype=object))
    df["off_time_local"] = sched.map(lambda s: (lambda d: d.strftime("%H:%M") if d else None)(parse_iso(s)))
    df["utc_offset_hours"] = sched.map(
        lambda s: (lambda d: d.utcoffset().total_seconds() / 3600 if d else np.nan)(parse_iso(s)))
    df["off_time_utc"] = sched.map(utc_off_time)
    started = df.get("res_started_at", pd.Series(index=df.index, dtype=object)).map(parse_iso)
    sched_dt = sched.map(parse_iso)
    df["off_delay_s"] = [
        (a - b).total_seconds() if a is not None and b is not None else np.nan
        for a, b in zip(started, sched_dt)
    ]
    if "sect_scheduled_start_time" in df:
        sect_dt = df["sect_scheduled_start_time"].map(parse_iso)
        df["sect_schedule_shift_min"] = [
            (a - b).total_seconds() / 60 if a is not None and b is not None else np.nan
            for a, b in zip(sect_dt, sched_dt)
        ]
    # The day list's "completed" races include abandoned ones (runners left
    # in state "entered"); they have no result.
    df["race_status"] = df.get("res_status__state")
    df["is_abandoned"] = df["race_status"].isin(["abandoned", "void", "cancelled"])
    df["distance_yards"] = pd.to_numeric(df.get("card_distance"), errors="coerce")
    df["distance_yards_text"] = df.get("card_distance_formatted", pd.Series(index=df.index, dtype=object)).map(
        _distance_yards_from_text)
    df["distance_furlongs"] = df["distance_yards"] / 220.0
    df["win_time_s"] = df.get("res_win_time_formatted", pd.Series(index=df.index, dtype=object)).map(parse_race_time)
    df["going"] = df.get("res_track_condition")
    df["race_class"] = df.get("card_race_class")
    df["race_type"] = df.get("res_race_type")
    df["surface"] = df.get("card_track_type")
    df["first_prize"] = pd.to_numeric(df.get("card_first_prize__amount"), errors="coerce")
    df["first_prize_currency"] = df.get("card_first_prize__currency")
    title = df.get("res_title_full", pd.Series(index=df.index, dtype=object)).fillna("")
    df["is_handicap"] = title.str.contains(r"\bhandicap\b|\bh'cap\b", case=False, regex=True)
    # Purebred Arabian races: Racing TV lists them, Sporting Life does not.
    # Not plain "Arabian": sponsors use it ("Arabian Horse Bahrain ... Chase").
    df["is_arab_race"] = title.str.contains(
        r"\barab (?:race|group|handicap|stakes|maiden|conditions)\b|\bconditions arab\b|\bpurebred arabian\b",
        case=False, regex=True)
    # Official winning time implies an average speed; outside 18-45 mph one
    # of the time or the distance is wrong.
    df["win_speed_mph"] = df["distance_yards"] / df["win_time_s"] * 3600 / 1760
    for col in ("card_analyst_verdict", "card_ip_hints_specific_pace", "card_ip_hints_individual_pace"):
        if col in df:
            df[col + "_text"] = df[col].map(strip_markup)
    return df


def _derive_runners(runners: pd.DataFrame) -> pd.DataFrame:
    df = runners.copy()
    get = lambda c: df[c] if c in df else pd.Series(index=df.index, dtype=object)  # noqa: E731
    df["horse_name"] = get("res_horse_name").fillna(get("card_horse_name"))
    df["horse_key"] = df["horse_name"].map(normalise_horse_key)
    df["horse_id"] = pd.to_numeric(get("res_horse__id").fillna(get("card_horse_id")), errors="coerce")
    df["horse_country"] = get("res_horse__country_code__racing").fillna(get("card_horse__country_code__racing"))
    state = get("res_status__state").fillna(get("card_status__state"))
    df["state"] = state
    withdrawn = get("card_withdrawn").astype("boolean").fillna(False).astype(bool)
    # States seen: finished, casualty (fell/PU/UR), disqualified, scratched
    # (non-runner) and reserve (declared reserve that did not get a run).
    df["is_nonrunner"] = state.isin(["scratched", "withdrawn", "non_runner", "nonrunner", "reserve"]) | withdrawn
    # finish_position is the physical finishing order; amended_position the
    # official result after stewards' changes (e.g. disqualifications).
    df["finish_position"] = pd.to_numeric(get("res_finish_position"), errors="coerce")
    df["amended_position"] = pd.to_numeric(get("res_amended_position"), errors="coerce")
    df["final_position"] = df["amended_position"].fillna(df["finish_position"])
    # Some casualty payloads carry an ordinal tail number. It is not a
    # completed physical finish; retain that raw number for source inspection.
    df["is_finisher"] = df["finish_position"].ge(1) & ~df["is_nonrunner"] & ~state.eq("casualty")
    df["is_disqualified"] = state.eq("disqualified")
    # between_distance is the gap to the preceding finisher (blank for the
    # winner), the same convention as Sporting Life's lengths_behind.
    df["margin_text"] = get("res_between_distance")
    df["margin_to_prev"] = df["margin_text"].map(parse_margin)
    df["cloth_number"] = pd.to_numeric(get("res_cloth_number").fillna(get("card_cloth_number")), errors="coerce")
    # Barrier 0 means no stalls (jumps, some flat starts): no draw.
    df["draw"] = pd.to_numeric(get("res_barrier_number").fillna(get("card_barrier_number")),
                               errors="coerce").where(lambda s: s > 0)
    df["age"] = pd.to_numeric(get("res_age").fillna(get("card_age")), errors="coerce")
    df["weight_lbs"] = get("res_format_weight").fillna(get("card_format_weight")).map(parse_weight_lbs)
    df["claim_lbs"] = pd.to_numeric(get("res_apprentice_claim").fillna(get("card_apprentice_claim")), errors="coerce")
    eq_text = get("res_equipment_text").fillna(get("card_equipment_text"))
    eq = eq_text.map(parse_equipment)
    df["equipment"] = eq.map(lambda t: t[0])
    df["equipment_first_time"] = eq.map(lambda t: t[1])
    dec = eq_text.map(decode_equipment)
    for key in ("symbols", "first_time", "first_time_ambiguous", "wind_op", "wind_op_first_time", "decoded", "unknown"):
        df[f"equipment_{key}" if key != "first_time" else "equipment_first_time_symbols"] = dec.map(lambda d: d[key])
    df["sp_decimal"] = pd.to_numeric(get("res_starting_price__decimal"), errors="coerce").where(lambda s: s > 1)
    df["sp_fractional"] = get("res_starting_price__fractional")
    # 1 = favourite, 2 = joint favourite (seen), 3 = co-favourite (assumed).
    df["sp_favourite_code"] = pd.to_numeric(get("res_starting_price_favorite"), errors="coerce")
    # Each payload names people twice (``x_name`` and ``x.name``). Usually one
    # is the initials form Sporting Life uses ("A M Balding") and the other the
    # full name, but which field holds which varies by payload and even by
    # person, so pick by shape rather than by field.
    df["jockey"], df["jockey_full"] = _name_forms(
        df, ["card_jockey_name", "card_jockey__name", "res_jockey_name", "res_jockey__name"])
    df["trainer"], df["trainer_full"] = _name_forms(
        df, ["res_trainer_name", "res_trainer__name", "card_trainer_name", "card_trainer__name"])
    df["owner"] = get("res_owner_name").fillna(get("card_owner_name"))
    df["comment"] = get("res_close_up_comment")
    tf = get("card_timeform_rating").astype("string")
    df["timeform_rating"] = pd.to_numeric(tf.str.extract(r"(\d+)", expand=False), errors="coerce")
    df["timeform_rating_symbol"] = tf.str.replace(r"^\s*\d+", "", regex=True).str.strip().replace("", pd.NA)
    df["timeform_rating_provisional"] = get("card_format_timeform_rating").astype("string").str.contains(r"\(", regex=True)
    if "card_analyst_comment" in df:
        df["card_analyst_comment_text"] = df["card_analyst_comment"].map(strip_markup)
    df["sect_finishing_time_s"] = get("sect_finishing_time").map(parse_race_time)
    df["sect_fsp"] = pd.to_numeric(get("sect_finishing_speed_percentage"), errors="coerce")
    df["beaten_lengths"] = _beaten_lengths(df)
    return df


_INITIALS_RE = re.compile(r"^(?:[A-Z]\.?\s+)+\S")
_COUNTRY_SUFFIX_RE = re.compile(r",\s*(Ireland|France|Germany|Great Britain|GB|IRE|FR|USA|UAE)\s*$", re.I)


def _clean_person(name: str) -> str:
    """'J. A. Powell' -> 'J A Powell'; 'P. Mulligan, Ireland' -> 'P Mulligan'."""
    s = _COUNTRY_SUFFIX_RE.sub("", name.strip())
    s = re.sub(r"\b([A-Z])\.(?=\s|$)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _name_forms(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """(initials form, full form) from the candidate name columns, in order.

    The initials form is the first candidate that starts with initials
    ("J & T Gosden" and single-word names count as their own short form);
    the full form is the longest candidate.
    """
    present = [c for c in cols if c in df]
    if not present:
        empty = pd.Series(None, index=df.index, dtype=object)
        return empty, empty

    def pick(row) -> tuple[Any, Any]:
        names = [_clean_person(v) for v in row if isinstance(v, str) and v.strip()]
        if not names:
            return None, None
        short = next((n for n in names if _INITIALS_RE.match(n)), names[0])
        full = max(names, key=len)
        return short, full

    picked = [pick(row) for row in df[present].itertuples(index=False, name=None)]
    return (pd.Series([p[0] for p in picked], index=df.index, dtype=object),
            pd.Series([p[1] for p in picked], index=df.index, dtype=object))


def _derive_sectionals(sectors: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Add each sector's length and the runner's speed over it.

    Sector names are cumulative distance labels; every sector is a furlong
    except the first, which carries the odd yards (2m 5f 122y over 22
    sectors starts with a 122-yard sector; 2m 6f 7y with 227 yards).
    """
    if sectors.empty:
        return sectors
    df = sectors.copy()
    n = df.groupby("race_id")["sector_number"].transform("max") + 1
    dist = df["race_id"].map(races.set_index("race_id")["distance_yards"])
    first = dist - 220.0 * (n - 1)
    # Only trust the partial first sector when it is a plausible length.
    first = first.where((first > 0) & (first < 440))
    df["sector_length_yards"] = np.where(df["sector_number"] == 0, first, 220.0)
    time_ = pd.to_numeric(df["sector_time"], errors="coerce")
    df["sector_speed_mph"] = (df["sector_length_yards"] / time_) * 3600 / 1760
    return df


def _beaten_lengths(df: pd.DataFrame) -> pd.Series:
    """Cumulative lengths behind the winner from RTV's per-gap margins.

    Delegates the chain rules (ties, broken chains) to
    :func:`src.result_margins.cumulative_beaten_lengths`. Margins describe the
    physical finish, so the chain follows ``finish_position`` (a disqualified
    horse keeps its place on the track).
    """
    from src.result_margins import cumulative_beaten_lengths

    out = pd.Series(np.nan, index=df.index)
    fin = df.loc[df["is_finisher"] & df["finish_position"].notna(),
                 ["race_id", "finish_position", "margin_to_prev"]].rename(
                     columns={"margin_to_prev": "lengths_behind"})
    if fin.empty:
        return out
    out.loc[fin.index] = cumulative_beaten_lengths(fin).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def iter_race_payloads(
    store: RawStore, start: str | None = None, end: str | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(results_path, payloads)`` for every stored race."""
    for path, _ in store.paths("results", start, end):
        parts = parse_results_path(path)
        if not parts:
            continue
        paths = race_paths(*parts)
        yield path, {kind: store.get_json(paths[kind]) for kind in RACE_KINDS}


def build_tables(
    start: str | None = None,
    end: str | None = None,
    store: RawStore | None = None,
) -> dict[str, pd.DataFrame]:
    """Parse stored payloads into tables (+ derived columns and quality flags)."""
    own_store = store is None
    store = store or RawStore()
    rows: dict[str, list[dict]] = {k: [] for k in
                                    ("races", "runners", "metrics", "sectionals", "totes", "smart_stats",
                                     "last_winners", "odds", "race_tips")}
    try:
        for path, p in iter_race_payloads(store, start, end):
            if not p["results"] or not (p["results"].get("race") or {}).get("id"):
                logger.warning("RTV %s: results payload without a race", path)
                continue
            parsed = parse_race(p["results"], p["racecard"], p["iq_metrics"], p["iq_sectional"], path)
            for key, value in parsed.items():
                rows[key].extend(value)
            race_id = parsed["races"][0]["race_id"]
            rows["odds"].extend(parse_odds(p.get("odds"), race_id))
            rows["race_tips"].extend(parse_race_tips(p.get("tips"), race_id))
            comparison = p.get("iq_comparison")
            if comparison and p["iq_metrics"]:
                rows.setdefault("comparison_check", []).extend(
                    _compare_comparison(p["iq_metrics"], comparison, race_id))
        meetings = _meetings_table(store, start, end)
        tipster_tips = _tipster_tips_table(store, start, end)
        tracks = _tracks_table(store)
    finally:
        if own_store:
            store.close()

    tables = {k: pd.DataFrame(v) for k, v in rows.items()}
    tables["meetings"] = meetings
    tables["tipster_tips"] = tipster_tips
    tables["tracks"] = tracks
    if not tables["races"].empty:
        tables["races"] = _derive_races(tables["races"])
        tables["runners"] = _derive_runners(tables["runners"])
        tables["sectionals"] = _derive_sectionals(tables["sectionals"], tables["races"])
        _attach_race_context(tables)
    tables["quality_flags"] = quality_checks(tables)
    return tables


def _attach_race_context(tables: dict[str, pd.DataFrame]) -> None:
    races = tables["races"].set_index("race_id")
    for name in ("runners",):
        df = tables[name]
        for col in ("race_date", "track_name", "off_time_utc", "race_type"):
            df[col] = df["race_id"].map(races[col])


def parse_odds(payload: dict | None, race_id: Any) -> list[dict]:
    """One row per runner x bookmaker: the bookmaker's last price before the off.

    Prices are the site's final displayed odds (with each-way places and
    fraction, and whether the price was shortening or drifting), not a time
    series. Available from at least December 2023; some races have none.
    """
    if not payload:
        return []
    books = {str(k): v for k, v in (payload.get("bookmakers") or {}).items()}
    out = []
    for r in payload.get("runners") or []:
        for o in r.get("odds") or []:
            book = books.get(str(o.get("bookmaker_id"))) or {}
            price = o.get("price") or {}
            out.append({
                "race_id": race_id, "runner_id": r.get("id"), "odds_id": o.get("id"),
                "bookmaker_id": o.get("bookmaker_id"), "bookmaker": book.get("name"),
                "price_decimal": to_float(price.get("decimal")), "price_fractional": price.get("fractional"),
                "each_way_places": o.get("each_way_places"), "each_way_denominator": o.get("each_way_denominator"),
                "fluctuation": o.get("fluctuation_type"),
                **_flatten({k: v for k, v in o.items() if k not in
                            {"id", "price", "bookmaker_id", "each_way_places", "each_way_denominator",
                             "fluctuation_type"}}, "odds"),
            })
    return out


def parse_race_tips(payload: dict | None, race_id: Any) -> list[dict]:
    """Racing TV's selection(s) for a race (``/racecards/.../tips``)."""
    if not payload:
        return []
    out = []
    for t in (payload.get("race") or {}).get("tips") or []:
        runner = t.get("runner") or {}
        out.append({"race_id": race_id, "tip_id": t.get("id"), "runner_id": runner.get("id"),
                    "horse_name": runner.get("horse_name"), "description": t.get("description"),
                    **_flatten({k: v for k, v in t.items() if k not in {"id", "runner", "description"}}, "tip")})
    return out


def _tipster_tips_table(store: RawStore, start: str | None, end: str | None) -> pd.DataFrame:
    """Named tipsters' daily selections, with advised price and outcome.

    ``tip_runner__last_replay__*`` describes the horse's latest run at the
    time of the scrape, not the tipped race.
    """
    rows = []
    for path, race_date in store.paths("tipsters", start, end):
        for tipster in (store.get_json(path) or {}).get("tipsters") or []:
            name = (tipster.get("name") or {}).get("formatted")
            for t in tipster.get("tips") or []:
                rows.append({"race_date": race_date, "tipster": name, "tipster_slug": tipster.get("slug"),
                             "runner_id": (t.get("runner") or {}).get("id"),
                             **_flatten(t, "tip", {"preview_video"}),
                             "video_title": (t.get("preview_video") or {}).get("title")})
    return pd.DataFrame(rows)


def _tracks_table(store: RawStore) -> pd.DataFrame:
    """Course pages: name, surface, track type and the profile text.

    Jockey/trainer statistics and upcoming races on these pages are as of
    the scrape and are not tabulated.
    """
    rows = []
    for path, _ in store.paths("track"):
        track = (store.get_json(path) or {}).get("track") or {}
        row = _flatten({k: v for k, v in track.items() if not isinstance(v, (dict, list)) or k == "country_code"},
                       "track")
        for col in ("track_profile_content", "track_overview_header_content"):
            if col in row:
                row[col + "_text"] = strip_markup(row[col])
        row["track_slug"] = path.rsplit("/", 1)[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def _meetings_table(store: RawStore, start: str | None, end: str | None) -> pd.DataFrame:
    rows = []
    for path, race_date in store.paths("day_list", start, end):
        payload = store.get_json(path) or {}
        for m in payload.get("meetings") or []:
            track = m.get("track") or {}
            row = {"race_date": race_date, "n_completed_races": len(m.get("completed_races") or [])}
            _flatten(m, "meeting", {"completed_races"}, row)
            row["country_iso"] = ((track.get("country_code") or {}).get("iso") or "").lower() or None
            rows.append(row)
    return pd.DataFrame(rows)


def _compare_comparison(metrics: dict, comparison: dict, race_id: Any) -> list[dict]:
    """Rows where the comparison endpoint disagrees with metrics (expect none)."""
    by_name = {}
    for r in metrics.get("runners") or []:
        flat = {}
        for items in (r.get("race_iq") or {}).values():
            if isinstance(items, dict):
                for k, m in items.items():
                    if isinstance(m, dict):
                        flat[k] = m.get("value")
        by_name[r.get("id")] = flat
    values = [set(v.items()) for v in by_name.values()]
    out = []
    for h in comparison.get("horses") or []:
        cm = {k: m.get("value") for k, m in (h.get("metrics") or {}).items() if isinstance(m, dict)}
        if not any(set(cm.items()) <= v for v in values):
            out.append({"race_id": race_id, "horse_name": h.get("horse_name"), "comparison": json.dumps(cm)})
    return out


def write_tables(tables: dict[str, pd.DataFrame], directory: str = TABLES_DIR, part: str | None = None) -> None:
    """Write each table to ``{directory}/{name}.parquet`` or ``{name}/{part}.parquet``."""
    for name, df in tables.items():
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
        if part is None:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{name}.parquet")
        else:
            os.makedirs(os.path.join(directory, name), exist_ok=True)
            path = os.path.join(directory, name, f"{part}.parquet")
        temporary = path+"."+uuid4().hex+".tmp"
        try:
            out.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        logger.info("RTV table %s%s: %d rows", name, f" [{part}]" if part else "", len(out))


def load_table(name: str, directory: str = TABLES_DIR) -> pd.DataFrame:
    """Concatenate every part of a written table (columns unioned)."""
    folder = os.path.join(directory, name)
    if not os.path.isdir(folder):
        path = os.path.join(directory, f"{name}.parquet")
        return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    parts = [pd.read_parquet(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith(".parquet")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_by_quarter(store: RawStore | None = None, directory: str = TABLES_DIR,
                     *, start=None, end=None) -> dict[str, int]:
    """Rebuild every table one calendar quarter at a time (bounded memory).

    Writes ``{directory}/{table}/{year}-Q{n}.parquet``; read a table back with
    :func:`load_table` (years can differ in columns as the API adds fields).
    """
    own_store = store is None
    store = store or RawStore()
    counts: dict[str, int] = {}
    requested_start, requested_end = start, end
    try:
        # One calendar quarter per part: building a whole year at once
        # peaked at ~9 GB of memory.
        quarters = sorted({(d[:4], (int(d[5:7]) - 1) // 3) for _, d in store.paths("results")})
        for year, q in quarters:
            start = f"{year}-{3 * q + 1:02d}-01"
            end = (date(int(year), 3 * q + 3, 1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)
            if (requested_start and end.isoformat() < requested_start) or (requested_end and start > requested_end):
                continue
            tables = build_tables(start, end.isoformat(), store=store)
            # Course pages are not dated: written once below, not per part.
            tables.pop("tracks", None)
            write_tables(tables, directory, part=f"{year}-Q{q + 1}")
            for name, df in tables.items():
                counts[name] = counts.get(name, 0) + len(df)
        tracks = _tracks_table(store)
        write_tables({"tracks": tracks}, directory)
        counts["tracks"] = len(tracks)
    finally:
        if own_store:
            store.close()
    return counts


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

# Plausible bands per metric (outside -> flagged, value kept).
METRIC_BANDS: dict[str, tuple[float, float]] = {
    "fsp": (60.0, 140.0),
    "top_speed": (15.0, 50.0),
    "acceleration": (1.0, 8.0),
    "stride_length": (4.0, 10.0),
    "min_length": (3.0, 10.0), "avg_length": (4.0, 10.0), "max_length": (4.0, 11.0),
    "min_frequency": (1.2, 3.5), "avg_frequency": (1.5, 3.5), "max_frequency": (1.5, 4.0),
    "jump_index": (0.0, 10.0),
    "lengths_gained_jumping": (-40.0, 40.0),
    "entry_speed": (10.0, 45.0), "exit_speed": (10.0, 45.0),
    "speed_lost": (-25.0, 5.0),
    "time_in_envelope": (1.0, 15.0), "recovery_time": (0.0, 10.0),
    "positive_jump_percentage": (0.0, 1.0),
}


def _flag(rows: list, check: str, severity: str, race_id: Any, runner_id: Any = None,
          value: Any = None, detail: str = "") -> None:
    rows.append({"race_id": race_id, "runner_id": runner_id, "check": check,
                 "severity": severity, "value": value, "detail": detail})


def quality_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flag spurious or internally inconsistent values. Nothing is altered."""
    flags: list[dict] = []
    races = tables.get("races", pd.DataFrame())
    runners = tables.get("runners", pd.DataFrame())
    metrics = tables.get("metrics", pd.DataFrame())
    sectors = tables.get("sectionals", pd.DataFrame())
    if races.empty:
        return pd.DataFrame(columns=["race_id", "runner_id", "check", "severity", "value", "detail"])

    # -- race level ----------------------------------------------------
    rr = races.copy()
    if "sect_schedule_shift_min" in rr:
        _emit(flags, rr[rr["sect_schedule_shift_min"].fillna(0) != 0], "sectional_schedule_shift", "info",
              "sect_schedule_shift_min",
              lambda r: "sectional endpoint's scheduled start differs from the results endpoint")
    rr["_dd"] = rr["distance_yards"] - rr["distance_yards_text"]
    _emit(flags, rr[rr["_dd"].abs() > 1], "distance_text_mismatch", "warn", "_dd",
          lambda r: f"{r.distance_yards} yards vs '{getattr(r, 'card_distance_formatted', None)}'")
    if not runners.empty and "res_runner_count" in rr:
        took_part = runners[runners.get("in_results", pd.Series(True, index=runners.index)).fillna(False).astype(bool)
                            & ~runners["is_nonrunner"]]
        rr["_n_run"] = rr["race_id"].map(took_part.groupby("race_id").size()).fillna(0)
        rr["_rc"] = pd.to_numeric(rr["res_runner_count"], errors="coerce")
        bad = rr[rr["_rc"].notna() & (rr["_rc"] != rr["_n_run"]) & ~rr["is_abandoned"]]
        _emit(flags, bad, "runner_count_mismatch", "warn", "_rc",
              lambda r: f"runner_count {r._rc:.0f} vs {r._n_run:.0f} runners in the result")
    if "res_win_time_formatted" in rr:
        bad = rr[rr["win_time_s"].isna() & rr["res_win_time_formatted"].notna()
                 & (rr["res_win_time_formatted"].astype(str).str.strip() != "") & ~rr["is_abandoned"]]
        _emit(flags, bad, "win_time_unparsed", "warn", None, lambda r: str(r.res_win_time_formatted))
    _emit(flags, rr[rr["is_abandoned"]], "race_abandoned", "info", "race_status",
          lambda r: "race listed but abandoned")
    if "win_speed_mph" in rr:
        bad = rr[rr["win_speed_mph"].notna() & ~rr["win_speed_mph"].between(18, 45)]
        _emit(flags, bad, "win_time_implausible", "error", "win_speed_mph",
              lambda r: f"official time {r.res_win_time_formatted} over {r.distance_furlongs:.1f}f "
                        f"is {r.win_speed_mph:.1f} mph")

    # -- runner level --------------------------------------------------
    if not runners.empty:
        for col_a, col_b, check, numeric in (
            ("res_cloth_number", "card_cloth_number", "cloth_mismatch", True),
            ("res_barrier_number", "card_barrier_number", "draw_mismatch", True),
            ("res_finish_position", "sect_finish_position", "position_mismatch_sectional", True),
            ("res_amended_position", "sect_amended_position", "amended_mismatch_sectional", True),
            ("res_cloth_number", "sect_cloth_number", "cloth_mismatch_sectional", True),
            ("res_barrier_number", "sect_barrier_number", "draw_mismatch_sectional", True),
            ("res_between_distance", "sect_between_distance", "margin_mismatch_sectional", False),
            ("res_starting_price__decimal", "card_starting_price__decimal", "sp_mismatch_card", True),
            ("res_age", "card_age", "age_mismatch_card", True),
            ("res_format_weight", "card_format_weight", "weight_mismatch_card", False),
            ("res_horse__id", "card_horse_id", "horse_id_mismatch_card", True),
        ):
            if col_a in runners and col_b in runners:
                a, b = runners[col_a], runners[col_b]
                if numeric:
                    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
                    bad = a.notna() & b.notna() & ((a - b).abs() > 1e-9)
                else:
                    bad = a.notna() & b.notna() & (a.astype(str).str.strip() != b.astype(str).str.strip())
                sub = runners.loc[bad, ["race_id", "runner_id"]].assign(_a=a[bad], _b=b[bad])
                _emit(flags, sub, check, "warn", None, lambda r: f"{r._a!r} vs {r._b!r}")
        if "res_horse_name" in runners and "card_horse_name" in runners:
            a = runners["res_horse_name"].map(normalise_horse_key)
            b = runners["card_horse_name"].map(normalise_horse_key)
            bad = runners["res_horse_name"].notna() & runners["card_horse_name"].notna() & (a != b)
            for row in runners.loc[bad, ["race_id", "runner_id", "res_horse_name", "card_horse_name"]].itertuples(index=False):
                _flag(flags, "name_mismatch_card", "warn", row[0], row[1], detail=f"{row[2]!r} vs {row[3]!r}")
        unparsed = runners["margin_text"].notna() & runners["margin_to_prev"].isna() & runners["is_finisher"]
        for row in runners.loc[unparsed, ["race_id", "runner_id", "margin_text"]].itertuples(index=False):
            _flag(flags, "margin_unparsed", "info", row[0], row[1], detail=repr(row[2]))
        if "equipment_decoded" in runners:
            unk = runners[~runners["equipment_decoded"].fillna(True).astype(bool)]
            _emit(flags, unk, "equipment_glyph_unknown", "warn", "equipment_unknown",
                  lambda r: f"glyph {r.equipment_unknown!r} not in the glyph table")
        # Below the code's minimum (Flat 8-0, jumps 10-0) is a long-handicap
        # weight: what the horse was rated to carry, not what it carried.
        rtype = runners.get("race_type", pd.Series(index=runners.index, dtype=object))
        minimum = np.where(rtype.isin(["Hurdle", "Chase", "Bumper"]), 140.0, 112.0)
        low = runners["weight_lbs"].notna() & (runners["weight_lbs"] < minimum)
        _emit(flags, runners[low], "weight_long_handicap", "info", "weight_lbs",
              lambda r: "below the minimum weight: long-handicap figure")
        _emit(flags, runners[runners["weight_lbs"] > 182], "weight_out_of_range", "warn", "weight_lbs",
              lambda r: "above 13 st")
        zero = runners.get("res_format_weight", pd.Series(index=runners.index, dtype=object)).astype(str).str.strip() == "0-0"
        _emit(flags, runners[zero], "weight_placeholder", "info", None, lambda r: "weight published as 0-0")
        # Favourite code: every shortest-priced runner, and only those.
        ran = runners[runners["sp_decimal"].notna()]
        if not ran.empty:
            min_sp = ran.groupby("race_id")["sp_decimal"].transform("min")
            is_min = (ran["sp_decimal"] - min_sp).abs() < 1e-9
            flagged = ran["sp_favourite_code"].fillna(0) >= 1
            bad = ran[is_min != flagged]
            _emit(flags, bad, "favourite_code_inconsistent", "warn", "sp_favourite_code",
                  lambda r: f"SP {r.sp_decimal} code {r.sp_favourite_code}")
        if "sect_fsp" in runners and not metrics.empty:
            mf = metrics.loc[metrics.metric == "fsp"].set_index("runner_id")["value"]
            joined = runners.set_index("runner_id")["sect_fsp"].to_frame().join(mf.rename("iq_fsp"), how="inner")
            bad = joined.dropna()
            bad = bad[(bad.sect_fsp - bad.iq_fsp).abs() > 0.011]
            for rid_, row in bad.iterrows():
                race_id = runners.loc[runners.runner_id == rid_, "race_id"].iloc[0]
                _flag(flags, "fsp_mismatch_sectional", "warn", race_id, rid_, detail=f"{row.sect_fsp} vs {row.iq_fsp}")

    # -- metrics -------------------------------------------------------
    if not metrics.empty:
        fin = runners.set_index("runner_id")["is_finisher"] if not runners.empty else pd.Series(dtype=bool)
        m = metrics.copy()
        m["is_finisher"] = m["runner_id"].map(fin).fillna(False).astype(bool)
        lo = m["metric"].map(lambda k: METRIC_BANDS.get(k, (np.nan, np.nan))[0])
        hi = m["metric"].map(lambda k: METRIC_BANDS.get(k, (np.nan, np.nan))[1])
        out = m[m["value"].notna() & lo.notna() & ((m["value"] < lo) | (m["value"] > hi))]
        _emit(flags, out, "metric_out_of_range", "warn", "value",
              lambda r: f"{r.metric} outside {METRIC_BANDS[r.metric]}" + ("" if r.is_finisher else " (non-finisher)"))
        unparsed = m[m["value"].isna() & m["value_raw"].notna() & (m["value_raw"].astype(str) != "")]
        _emit(flags, unparsed, "metric_unparsed", "warn", None, lambda r: f"{r.metric}={r.value_raw!r}")
        # Ranks must order the values. Ranks are computed on unrounded values,
        # so equal displayed values may take any order; only a strict
        # reversal beyond the 0.01 display precision is flagged. Direction
        # (higher better or lower better) is read from each group.
        k = m.dropna(subset=["value", "rank"]).copy()
        k["rank"] = pd.to_numeric(k["rank"], errors="coerce")
        keys = ["race_id", "metric"]
        k = k.sort_values(keys + ["rank"])
        k["direction"] = k.groupby(keys)["value"].transform(
            lambda v: -1.0 if len(v) < 2 or v.iloc[0] >= v.iloc[-1] else 1.0)
        k["step"] = (k.groupby(keys)["value"].diff() * k["direction"])
        rev = k[k["step"] < -0.011]
        _emit(flags, rev, "metric_rank_inconsistent", "warn", "value",
              lambda r: f"{r.metric}: rank {r.rank:.0f} has value {r.value} out of order")
        # 'average' is not the field mean and 'percentage' is the value
        # min-max scaled within the race; both are kept as published.

    # -- sectionals ----------------------------------------------------
    if not sectors.empty:
        _sectional_checks(sectors, races, runners, flags)

    return pd.DataFrame(flags, columns=["race_id", "runner_id", "check", "severity", "value", "detail"])


def _emit(flags: list, df: pd.DataFrame, check: str, severity: str, value_col: str | None, detail) -> None:
    """Append one flag per row of *df* (``detail`` is a row -> str callable)."""
    for row in df.itertuples(index=False):
        _flag(flags, check, severity, row.race_id, getattr(row, "runner_id", None),
              None if value_col is None else getattr(row, value_col), detail(row))


def _sectional_checks(sectors: pd.DataFrame, races: pd.DataFrame, runners: pd.DataFrame, flags: list) -> None:
    s = sectors.copy()
    for c in ("sector_time", "cumulative_sector_time", "time_behind_leader", "sector_position", "sector_number"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.sort_values(["race_id", "runner_id", "sector_number"])
    race_info = races.set_index("race_id")
    # Physical finishing order: sectional times describe the race as run.
    pos = runners.set_index("runner_id")["finish_position"] if not runners.empty else pd.Series(dtype=float)
    fin = runners.set_index("runner_id")["is_finisher"] if not runners.empty else pd.Series(dtype=bool)
    timed = s.dropna(subset=["sector_time"]).copy()
    if timed.empty:
        return

    # Sector count vs distance.
    n_sec = s.groupby("race_id")["sector_number"].nunique().rename("n_sec").reset_index()
    n_sec["dist_f"] = n_sec["race_id"].map(race_info["distance_furlongs"])
    n_sec["diff"] = n_sec["n_sec"] - n_sec["dist_f"].round()
    # A whole meeting can carry splits that do not fit the race (Chelmsford
    # 2024-04-13: 10 sectors for 8f, 5 for 14f): the sectionals are unusable.
    _emit(flags, n_sec[n_sec["diff"].abs() > 1], "sector_count_vs_distance", "error", "n_sec",
          lambda r: f"{r.n_sec} sectors for {r.dist_f:.2f}f")

    # Cumulative = running sum of sector times (per runner, timed sectors only).
    timed["run_sum"] = timed.groupby(["race_id", "runner_id"])["sector_time"].cumsum()
    timed["cum_err"] = (timed["run_sum"] - timed["cumulative_sector_time"]).abs()
    cum = timed.groupby(["race_id", "runner_id"])["cum_err"].max().reset_index()
    _emit(flags, cum[cum["cum_err"] > 0.06], "cumulative_mismatch", "warn", "cum_err",
          lambda r: f"sum of sector times differs from cumulative by {r.cum_err:.2f}s")

    # Winner's final cumulative vs official winning time.
    last = timed.groupby(["race_id", "runner_id"]).tail(1).copy()
    last["pos"] = last["runner_id"].map(pos)
    last["fin"] = last["runner_id"].map(fin).fillna(False).astype(bool)
    w = last[last["pos"] == 1].copy()
    w["win_time"] = w["race_id"].map(race_info["win_time_s"])
    w["diff"] = (w["cumulative_sector_time"] - w["win_time"]).abs()
    w = w.groupby("race_id", as_index=False)["diff"].min()
    # Over jumps ~28% of races differ by 0.1-3 s (median 0.7 s): a timing
    # convention, not an error (flat: 0.5% of races). Beyond 5 s one of the
    # two times is wrong (some official times are impossible).
    for lo, hi, severity in ((0.1, 1.0, "info"), (1.0, 5.0, "warn"), (5.0, np.inf, "error")):
        _emit(flags, w[(w["diff"] > lo) & (w["diff"] <= hi)], "winner_time_mismatch", severity, "diff",
              lambda r: f"winner's sectional total is {r.diff:.2f}s off the official time")

    # Leader consistency: time_behind_leader = cumulative - fastest cumulative.
    timed["lead_cum"] = timed.groupby(["race_id", "sector_number"])["cumulative_sector_time"].transform("min")
    timed["gap_err"] = (timed["cumulative_sector_time"] - timed["lead_cum"] - timed["time_behind_leader"]).abs()
    # Early (2023) payloads can omit some runners' sectionals while positions
    # and gaps still count the whole field: a missing leader shifts gaps, a
    # missing runner ahead shifts positions by up to the number missing.
    if not runners.empty:
        ran = runners[runners.get("in_results", pd.Series(True, index=runners.index)).fillna(False).astype(bool)
                      & ~runners["is_nonrunner"].astype(bool)]
        n_ran = ran.groupby("race_id").size()
        n_rows = s.groupby("race_id")["runner_id"].nunique()
        timed["n_missing"] = timed["race_id"].map((n_ran - n_rows).clip(lower=0)).fillna(0)
    else:
        timed["n_missing"] = 0
    gap = timed[(timed["gap_err"] > 0.06) & (timed["n_missing"] == 0)]
    _emit(flags, gap, "leader_gap_mismatch", "info", "gap_err",
          lambda r: f"sector {r.sector_number:.0f}: time_behind_leader {r.time_behind_leader} vs cumulative gap")
    # Sector positions must rank cumulative times; tied times may take any
    # of the tied positions.
    grp = timed.groupby(["race_id", "sector_number"])["cumulative_sector_time"]
    timed["pos_lo"] = grp.rank(method="min")
    # Runners with a row but no time at this marker still hold a position.
    cnt = (s.assign(_timed=s["sector_time"].notna())
           .groupby(["race_id", "sector_number"]).agg(_rows=("runner_id", "size"), _t=("_timed", "sum"))
           .reset_index())
    cnt["n_untimed"] = cnt["_rows"] - cnt["_t"]
    pos_hi = grp.rank(method="max")
    timed["n_untimed"] = timed[["race_id", "sector_number"]].merge(
        cnt[["race_id", "sector_number", "n_untimed"]], on=["race_id", "sector_number"], how="left"
    )["n_untimed"].fillna(0).to_numpy()
    timed["pos_hi"] = pos_hi + timed["n_missing"] + timed["n_untimed"]
    pbad = timed[timed["sector_position"].notna()
                 & ((timed["sector_position"] < timed["pos_lo"]) | (timed["sector_position"] > timed["pos_hi"]))]
    _emit(flags, pbad, "sector_position_mismatch", "warn", "sector_position",
          lambda r: f"sector {r.sector_number:.0f}: position {r.sector_position:.0f} vs {r.pos_lo:.0f}-{r.pos_hi:.0f} by time")

    # Tracking glitches: a sector far off the field median, compensated by
    # the next sector so the running total is unaffected.
    timed["med"] = timed.groupby(["race_id", "sector_number"])["sector_time"].transform("median")
    timed["dev"] = timed["sector_time"] - timed["med"]
    timed["next_dev"] = timed.groupby(["race_id", "runner_id"])["dev"].shift(-1)
    glitch = timed[(timed["dev"].abs() > 0.2 * timed["med"])
                   & (np.sign(timed["next_dev"]) == -np.sign(timed["dev"]))
                   & ((timed["dev"] + timed["next_dev"]).abs() < 0.5 * timed["dev"].abs())]
    _emit(flags, glitch, "sector_glitch", "warn", "dev",
          lambda r: f"sector {r.sector_number:.0f}: {r.dev:+.2f}s vs field median, next {r.next_dev:+.2f}s")

    # Finishing order vs sectional total (finishers only, 0.25s ~ 1 length).
    lf = last[last["fin"] & last["pos"].notna()].sort_values(["race_id", "pos"]).copy()
    lf["slowest_ahead"] = lf.groupby("race_id")["cumulative_sector_time"].transform(
        lambda x: x.cummax().shift(1))
    ob = lf[lf["cumulative_sector_time"] < lf["slowest_ahead"] - 0.25]
    _emit(flags, ob, "order_vs_time", "warn", "pos",
          lambda r: f"finished {r.pos:.0f} in {r.cumulative_sector_time}s, faster than a horse ahead ({r.slowest_ahead}s)")


# ---------------------------------------------------------------------------
# Validation against data/races.db (Sporting Life)
# ---------------------------------------------------------------------------

# Results-DB names Racing TV spells differently, after canonical_track_key.
_DB_TRACK_ALIAS = {"royal ascot": "ascot"}


def _norm_str(s: pd.Series) -> pd.Series:
    """Strings for comparison: None/NaN/'' all become ''."""
    return s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip())


def _pedigree_key(name: Any) -> str:
    """Sire/dam names without country suffix, case or punctuation."""
    s = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", str(name), flags=re.I)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _headgear_symbols(value: Any) -> str:
    """Sporting Life headgear list -> symbols, '1' marking first time ('p,t1')."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        import ast
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return "?"
    return ",".join(f"{i.get('symbol')}{'1' if i.get('count') == 1 else ''}"
                    for i in items if isinstance(i, dict))


def _surname(name: Any) -> str:
    parts = re.sub(r"[^a-z' -]", "", str(name).lower()).split()
    return parts[-1] if parts else ""


_TITLES = {"mr", "mrs", "miss", "ms", "sir", "dr", "lady", "lord", "capt", "major", "mme"}
_SUFFIXES = {"bt", "jnr", "jr", "snr", "sr"}


def _person_key(name: Any) -> str:
    """First initial + surname, titles and suffixes dropped.

    'Mrs A J Perrett' / 'Amanda Perrett' -> 'a perrett';
    'Sir Mark Prescott Bt' -> 'm prescott'.
    """
    parts = [p for p in re.sub(r"[^a-z' -]", "", str(name).lower()).split() if p not in _TITLES]
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else (parts[0] if parts else "")


def _track_join_key(name: Any) -> str:
    key = canonical_track_key(name)
    return _DB_TRACK_ALIAS.get(key, key)


def validate_against_db(
    tables: dict[str, pd.DataFrame], db_path: str = RESULTS_DB_PATH,
) -> dict[str, Any]:
    """Compare RTV races/runners with ``data/races.db``; return a report."""
    races, runners = tables["races"], tables["runners"]
    if races.empty:
        return {"races": 0}
    n_abandoned = int(races["is_abandoned"].sum())
    # Sporting Life does not carry purebred Arabian races; they are not gaps.
    arab = races["is_arab_race"].fillna(False).astype(bool) if "is_arab_race" in races else False
    n_arab = int((~races["is_abandoned"] & arab).sum()) if "is_arab_race" in races else 0
    races = races[~races["is_abandoned"] & ~arab]
    runners = runners[runners["race_id"].isin(races["race_id"])]
    start, end = races["race_date"].min(), races["race_date"].max()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # Pedigree lives in horse_pedigree (results.sire etc. are mostly blank).
    db = pd.read_sql(
        "SELECT r.race_id AS db_race_id, r.race_date, r.off_time, r.track, r.race_class, r.race_type, "
        "r.distance_furlongs, r.going, r.prize_money, r.num_runners, r.horse_name, r.jockey, r.trainer, "
        "r.age, r.weight_lbs, r.draw, r.odds, r.finish_position, r.finish_pos_label, r.lengths_behind, "
        "r.surface, r.headgear, r.wind_surgery, r.days_since_last_run, "
        "COALESCE(NULLIF(r.sire, ''), p.sire) AS sire, COALESCE(NULLIF(r.dam, ''), p.dam) AS dam, "
        "COALESCE(NULLIF(r.damsire, ''), p.damsire) AS damsire "
        "FROM results r LEFT JOIN horse_pedigree p ON p.horse_id = r.horse_id "
        "WHERE r.race_date BETWEEN ? AND ?", con, params=(start, end))
    con.close()
    db["_t"] = db["track"].map(_track_join_key)
    db["_o"] = db["off_time"].map(normalise_off_time_key)
    db["_h"] = db["horse_name"].map(normalise_horse_key)
    r = races.copy()
    r["_t"] = r["track_name"].map(_track_join_key)
    r["_o"] = r["off_time_utc"].map(normalise_off_time_key)

    db_races = db.drop_duplicates(["race_date", "_t", "_o"])
    rk = r.merge(db_races[["race_date", "_t", "_o", "db_race_id", "race_class", "distance_furlongs",
                           "going", "prize_money", "num_runners", "race_type", "surface"]]
                 .rename(columns={"race_class": "db_class", "distance_furlongs": "db_dist",
                                  "going": "db_going", "prize_money": "db_prize", "num_runners": "db_runners",
                                  "race_type": "db_race_type", "surface": "db_surface"}),
                 on=["race_date", "_t", "_o"], how="left", indicator=True)
    db_tracks = set(db["_t"])
    report: dict[str, Any] = {
        "date_range": [start, end],
        "rtv_abandoned_races_excluded": n_abandoned,
        "rtv_arab_races_excluded": n_arab,
        "rtv_races": int(len(r)),
        "rtv_races_matched": int((rk["_merge"] == "both").sum()),
        "rtv_unmatched_at_db_tracks": rk.loc[(rk["_merge"] == "left_only") & rk["_t"].isin(db_tracks),
                                             ["race_date", "track_name", "off_time_utc", "res_title"]].to_dict("records")[:50],
        "rtv_unmatched_other_tracks": sorted(set(rk.loc[(rk["_merge"] == "left_only") & ~rk["_t"].isin(db_tracks), "track_name"])),
    }
    rtv_keys = set(zip(r["race_date"], r["_t"], r["_o"]))
    db_missing = db_races[[k not in rtv_keys for k in zip(db_races["race_date"], db_races["_t"], db_races["_o"])]]
    report["db_races"] = int(len(db_races))
    report["db_races_missing_in_rtv"] = db_missing[["race_date", "track", "off_time"]].to_dict("records")[:50]
    report["n_db_races_missing_in_rtv"] = int(len(db_missing))

    m = rk[rk["_merge"] == "both"].copy()
    def rate(mask):
        return round(float(mask.mean()), 4) if len(mask) else None
    m["race_class"] = _norm_str(m["race_class"])
    m["db_class"] = _norm_str(m["db_class"])
    race_checks = {
        "class": rate(m["race_class"] == m["db_class"]),
        "distance_within_0.05f": rate((m["distance_furlongs"] - m["db_dist"]).abs() <= 0.05),
        "going_exact_ci": rate(m["going"].astype(str).str.lower().str.strip() == m["db_going"].astype(str).str.lower().str.strip()),
        "first_prize_within_1": rate((m["first_prize"] - m["db_prize"]).abs() <= 1),
        "race_type": rate(m["race_type"].astype(str).str.lower() == m["db_race_type"].astype(str).str.lower()),
        # Sporting Life files NH flat races ("Bumper" here) under Flat.
        "race_type_bumper_as_flat": rate(m["race_type"].replace({"Bumper": "Flat"}).astype(str).str.lower()
                                         == m["db_race_type"].astype(str).str.lower()),
        "n_bumpers_filed_as_flat": int(((m["race_type"] == "Bumper") & (m["db_race_type"] == "Flat")).sum()),
    }
    report["race_field_agreement"] = race_checks

    # Runners
    ru = runners.merge(rk.loc[rk["_merge"] == "both", ["race_id", "db_race_id"]], on="race_id")
    ru = ru[ru["in_results"].fillna(False).astype(bool) & ~ru["is_nonrunner"].astype(bool)].copy()
    dbr = db.copy()
    # Racing TV shows a horse's *current* name on old results (a horse
    # renamed after the race no longer matches Sporting Life's race-day
    # name). Pair leftovers within a race by finishing position and SP.
    ru["_join_h"] = ru["horse_key"]
    db_keys = set(zip(dbr["db_race_id"], dbr["_h"]))
    rtv_keys = set(zip(ru["db_race_id"], ru["horse_key"]))
    lo = ru[[k not in db_keys for k in zip(ru["db_race_id"], ru["horse_key"])]]
    ro = dbr[dbr["db_race_id"].isin(lo["db_race_id"])
             & [k not in rtv_keys for k in zip(dbr["db_race_id"], dbr["_h"])]]
    renamed = []
    if not lo.empty and not ro.empty:
        cand = lo[["runner_id", "db_race_id", "horse_name", "finish_position", "sp_decimal"]].merge(
            ro[["db_race_id", "_h", "horse_name", "finish_position", "odds"]], on="db_race_id", suffixes=("", "_db"))
        same_pos = (cand["finish_position"].fillna(0) == pd.to_numeric(cand["finish_position_db"], errors="coerce").fillna(0))
        same_sp = ((cand["sp_decimal"] - cand["odds"]).abs() <= 0.01 * cand["odds"])
        cand = cand[same_pos & same_sp]
        cand = cand[~cand["runner_id"].duplicated(keep=False) & ~cand.duplicated(["db_race_id", "_h"], keep=False)]
        ru.loc[ru["runner_id"].isin(cand["runner_id"]), "_join_h"] = ru["runner_id"].map(
            cand.set_index("runner_id")["_h"])
        renamed = cand[["horse_name", "horse_name_db"]].to_dict("records")
    report["renamed_horses"] = renamed
    joined = ru.merge(dbr, left_on=["db_race_id", "_join_h"], right_on=["db_race_id", "_h"],
                      how="outer", indicator="_m", suffixes=("", "_db"))
    report["runners_rtv"] = int(len(ru))
    report["runners_matched"] = int((joined["_m"] == "both").sum())
    report["runners_only_rtv"] = joined.loc[joined["_m"] == "left_only", ["race_date", "track_name", "horse_name"]].to_dict("records")[:30]
    only_db = joined.loc[(joined["_m"] == "right_only") & joined["db_race_id"].isin(ru["db_race_id"])]
    report["n_runners_only_db"] = int(len(only_db))
    report["runners_only_db"] = only_db[["race_date_db", "track", "horse_name_db", "finish_pos_label"]].to_dict("records")[:30] \
        if "race_date_db" in only_db else only_db[["track", "horse_name_db"]].to_dict("records")[:30]

    b = joined[joined["_m"] == "both"].copy()
    # The outer join turns boolean columns into object; restore them.
    for col in ("is_finisher", "is_nonrunner", "is_disqualified"):
        b[col] = b[col].fillna(False).astype(bool)
    b["db_pos"] = pd.to_numeric(b["finish_position_db"], errors="coerce")
    fin_both = b["is_finisher"] & b["db_pos"].ge(1)
    sp_ok = b["sp_decimal"].notna() & b["odds"].notna()
    amended = fin_both & (b["final_position"] != b["finish_position"])
    runner_checks = {
        "finish_position": rate((b.loc[fin_both, "final_position"] == b.loc[fin_both, "db_pos"])),
        # Where stewards changed the result, which order does races.db hold?
        "n_amended_runners": int(amended.sum()),
        "amended_db_has_official": rate(b.loc[amended, "final_position"] == b.loc[amended, "db_pos"]),
        "amended_db_has_physical": rate(b.loc[amended, "finish_position"] == b.loc[amended, "db_pos"]),
        "unamended_finish_position": rate(b.loc[fin_both & ~amended, "finish_position"]
                                          == b.loc[fin_both & ~amended, "db_pos"]),
        "nonfinisher_agreement": rate((~b["is_finisher"]) == (b["db_pos"].fillna(0) == 0)),
        "draw": rate(b.loc[b["draw"].notna() & b["draw_db"].notna(), "draw"] == b.loc[b["draw"].notna() & b["draw_db"].notna(), "draw_db"]),
        "age": rate(b["age"] == b["age_db"]),
        "weight_lbs": rate((b["weight_lbs"] - b["weight_lbs_db"]).abs() < 0.5),
        "sp_within_1pct": rate(((b.loc[sp_ok, "sp_decimal"] - b.loc[sp_ok, "odds"]).abs() / b.loc[sp_ok, "odds"]) <= 0.01),
        "jockey_exact": rate(b["jockey"].astype(str).str.lower() == b["jockey_db"].astype(str).str.lower()),
        "jockey_initial_surname": rate(b["jockey"].map(_person_key) == b["jockey_db"].map(_person_key)),
        "jockey_surname": rate(b["jockey"].map(_surname) == b["jockey_db"].map(_surname)),
        "trainer_exact": rate(b["trainer"].astype(str).str.lower() == b["trainer_db"].astype(str).str.lower()),
        "trainer_initial_surname": rate(b["trainer"].map(_person_key) == b["trainer_db"].map(_person_key)),
    }
    # Same convention on both sides: gap to the preceding finisher.
    db_gap = pd.to_numeric(b["lengths_behind"], errors="coerce")
    gap_ok = b["is_finisher"] & b["db_pos"].ge(2) & b["margin_to_prev"].notna() & db_gap.notna()
    runner_checks["margin_within_0.01"] = rate((b.loc[gap_ok, "margin_to_prev"] - db_gap[gap_ok]).abs() <= 0.01)
    runner_checks["comment_present"] = rate(b["comment"].notna())
    for ours, theirs in (("card_horse__sire_name", "sire"), ("card_horse__dam_name", "dam"),
                         ("card_horse__dam_sire_name", "damsire")):
        if ours in b:
            both = b[ours].notna() & b[theirs].notna() & (b[theirs].astype(str) != "")
            runner_checks[theirs] = rate(b.loc[both, ours].map(_pedigree_key) == b.loc[both, theirs].map(_pedigree_key))
    if "card_days_since_last_run" in b:
        dl = pd.to_numeric(b["card_days_since_last_run"], errors="coerce")
        dd = pd.to_numeric(b["days_since_last_run"], errors="coerce")
        runner_checks["days_since_last_run"] = rate(dl[dl.notna() & dd.notna()] == dd[dl.notna() & dd.notna()])
    # Equipment: Racing TV's codes vs Sporting Life's symbols, as a cross-tab.
    db_sym = b["headgear"].map(_headgear_symbols)
    sl_set = db_sym.map(lambda s: ",".join(sorted({x.rstrip("1") for x in s.split(",") if x})) or None)
    dec_ok = b["equipment_decoded"].fillna(True).astype(bool)
    runner_checks["equipment_symbols"] = rate(b.loc[dec_ok, "equipment_symbols"].fillna("")
                                              == sl_set[dec_ok].fillna(""))
    runner_checks["equipment_undecoded_share"] = rate(~dec_ok)
    sl_first = db_sym.map(lambda s: ",".join(sorted({x[:-1] for x in s.split(",") if x.endswith("1")})) or None)
    # Only where Racing TV's mark is unambiguous (single-item glyphs).
    unamb = dec_ok & b["equipment_first_time_ambiguous"].isna()
    runner_checks["equipment_first_time"] = rate(b.loc[unamb, "equipment_first_time_symbols"].fillna("")
                                                 == sl_first[unamb].fillna(""))
    runner_checks["equipment_first_time_ambiguous_share"] = rate(~unamb & dec_ok)
    ws = pd.to_numeric(b["wind_surgery"], errors="coerce")
    wop = b["equipment_wind_op_first_time"].fillna(0)
    runner_checks["wind_op_first_time_vs_sl_wind_surgery"] = rate((wop == 1) == (ws.fillna(0) == 1))
    equipment_examples = b.loc[dec_ok & (b["equipment_symbols"].fillna("") != sl_set.fillna("")),
                               ["race_date", "track_name", "horse_name", "equipment", "equipment_symbols",
                                "headgear"]].head(20).to_dict("records")
    report["equipment_crosstab"] = (
        pd.DataFrame({"rtv": b["equipment"].fillna(""), "first": b["equipment_first_time"].fillna(-1),
                      "sl": db_sym, "sl_wind_op": pd.to_numeric(b["wind_surgery"], errors="coerce").fillna(0)})
        .value_counts().head(60).reset_index().to_dict("records"))
    report["runner_field_agreement"] = runner_checks
    report["examples"] = {
        "finish_position": b.loc[fin_both & (b["final_position"] != b["db_pos"]),
                                 ["race_date", "track_name", "off_time_utc", "horse_name", "finish_position",
                                  "final_position", "db_pos", "state"]].head(40).to_dict("records"),
        "sp": b.loc[sp_ok & (((b["sp_decimal"] - b["odds"]).abs() / b["odds"]) > 0.01),
                    ["race_date", "track_name", "horse_name", "sp_decimal", "sp_fractional", "odds"]].head(15).to_dict("records"),
        "weight": b.loc[(b["weight_lbs"] - b["weight_lbs_db"]).abs() >= 0.5,
                        ["race_date", "track_name", "horse_name", "weight_lbs", "weight_lbs_db"]].head(15).to_dict("records"),
        "draw": b.loc[b["draw"].notna() & b["draw_db"].notna() & (b["draw"] != b["draw_db"]),
                      ["race_date", "track_name", "horse_name", "draw", "draw_db"]].head(15).to_dict("records"),
        "class": m.loc[m["race_class"] != m["db_class"],
                       ["race_date", "track_name", "off_time_utc", "race_class", "db_class"]].head(15).to_dict("records"),
        "jockey": b.loc[b["jockey"].astype(str).str.lower() != b["jockey_db"].astype(str).str.lower(),
                        ["race_date", "track_name", "horse_name", "jockey", "jockey_full", "jockey_db"]].head(15).to_dict("records"),
        "trainer": b.loc[b["trainer"].astype(str).str.lower() != b["trainer_db"].astype(str).str.lower(),
                         ["race_date", "track_name", "horse_name", "trainer", "trainer_full", "trainer_db"]].head(15).to_dict("records"),
        "distance": m.loc[(m["distance_furlongs"] - m["db_dist"]).abs() > 0.05,
                          ["race_date", "track_name", "off_time_utc", "card_distance_formatted", "distance_furlongs", "db_dist"]].head(15).to_dict("records"),
        "going": m.loc[m["going"].astype(str).str.lower().str.strip() != m["db_going"].astype(str).str.lower().str.strip(),
                       ["race_date", "track_name", "going", "db_going"]].head(15).to_dict("records"),
        "prize": m.loc[(m["first_prize"] - m["db_prize"]).abs() > 1,
                       ["race_date", "track_name", "off_time_utc", "first_prize", "first_prize_currency", "db_prize"]].head(15).to_dict("records"),
    }
    report["examples"]["equipment"] = equipment_examples
    report["examples"]["margin"] = b.loc[gap_ok & ((b["margin_to_prev"] - db_gap).abs() > 0.01),
                                         ["race_date", "track_name", "horse_name", "final_position",
                                          "margin_text", "lengths_behind"]].head(15).to_dict("records")
    report["_joined"] = b  # for interactive inspection; dropped when saved
    return report


def save_report(report: dict[str, Any], name: str) -> str:
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    path = os.path.join(VALIDATION_DIR, f"{name}.json")
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Racing TV (RaceiQ) scraper")
    sub = p.add_subparsers(dest="cmd", required=True)
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    s = sub.add_parser("scrape", help="Fetch raw payloads for a date range")
    s.add_argument("--start", default=RACEIQ_START.isoformat())
    s.add_argument("--end", default=yesterday)
    s.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES),
                   help="Comma-separated ISO codes, or 'all'")
    s.add_argument("--rps", type=float, default=DEFAULT_MAX_RPS, help="Overall requests per second")
    s.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    s.add_argument("--no-discover", dest="discover", action="store_false",
                   help="Do not probe races.db races missing from Racing TV's day lists")
    s.add_argument("--with-comparison", action="store_true")
    s.add_argument("--force", action="store_true", help="Re-fetch even final responses")
    s.add_argument("--oldest-first", action="store_true")

    n = sub.add_parser("snapshot", help="Capture race-day-only data (pace maps, prompts, odds) for a day")
    n.add_argument("--date", default=date.today().isoformat())
    n.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES))
    n.add_argument("--rps", type=float, default=DEFAULT_MAX_RPS)
    n.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    h = sub.add_parser("profiles", help="Fetch horse profiles for runners in stored results")
    h.add_argument("--start")
    h.add_argument("--end")
    h.add_argument("--rps", type=float, default=DEFAULT_MAX_RPS)
    h.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    sub.add_parser("learn-glyphs", help="Decode unknown equipment glyphs from data/races.db headgear")

    b = sub.add_parser("build", help="Parse stored payloads into parquet tables")
    b.add_argument("--start")
    b.add_argument("--end")

    v = sub.add_parser("validate", help="Quality flags + comparison with data/races.db")
    v.add_argument("--start", required=True)
    v.add_argument("--end", required=True)

    sub.add_parser("status", help="Summarise the raw store")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    countries_arg = getattr(args, "countries", ",".join(DEFAULT_COUNTRIES))
    countries = None if countries_arg.lower() == "all" else tuple(countries_arg.split(","))
    if args.cmd == "scrape":
        totals = scrape_range(args.start, args.end, countries, args.rps, args.with_comparison, args.force,
                              newest_first=not args.oldest_first, workers=args.workers, discover=args.discover)
        print(json.dumps(totals))
    elif args.cmd == "snapshot":
        print(json.dumps(snapshot_day(args.date, RTVClient(max_rps=args.rps), countries=countries,
                                      workers=args.workers)))
    elif args.cmd == "profiles":
        store = RawStore()
        slugs = set()
        for path, _ in store.paths("results", args.start, args.end):
            for r in (store.get_json(path) or {}).get("race", {}).get("runners") or []:
                slugs.add(((r.get("horse") or {}).get("slug") or "").strip("/"))
        print(json.dumps(scrape_horse_profiles(slugs, RTVClient(max_rps=args.rps), store, args.workers)))
        store.close()
    elif args.cmd == "learn-glyphs":
        print(json.dumps(learn_equipment_glyphs(), indent=2))
    elif args.cmd == "build":
        print(build_by_quarter(start=args.start, end=args.end))
        extra = {**build_snapshot_tables(), **build_profile_tables()}
        write_tables({k: v for k, v in extra.items() if not v.empty})
        print({k: len(v) for k, v in extra.items()})
    elif args.cmd == "validate":
        tables = build_tables(args.start, args.end)
        flags = tables["quality_flags"]
        report = validate_against_db(tables)
        report["quality_flag_counts"] = flags.groupby(["check", "severity"]).size().to_dict() if not flags.empty else {}
        path = save_report(report, f"validation_{args.start}_{args.end}")
        print(json.dumps({k: v for k, v in report.items() if k in
                          ("rtv_races", "rtv_races_matched", "db_races", "n_db_races_missing_in_rtv",
                           "runners_rtv", "runners_matched", "race_field_agreement",
                           "runner_field_agreement", "quality_flag_counts")}, indent=2, default=str))
        print("report:", path)
    elif args.cmd == "status":
        store = RawStore()
        print(store.summary().to_string(index=False))
        store.close()


if __name__ == "__main__":
    main()
