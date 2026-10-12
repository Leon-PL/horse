"""
At The Races Sectionals Scraper
===============================
Fetches rich per-runner sectional / stride / pace data from At The Races
(``attheraces.com``).  ATR sources its sectionals from Total Performance
Data (TPD), which is considerably richer than the RacingTV RACEiQ feed
(see :mod:`src.rtv_scraper`) — it provides per-furlong split times,
running totals, pace classifications, sectional speeds (mph), finishing
speed %, an efficiency grade/score, energy distribution and stride
metrics, plus the in-running comment and a sectional-adjusted placing.

How it works
------------
ATR sits behind a Fastly JavaScript "Client Challenge".  A plain
``requests`` GET returns a ~3 KB challenge stub, **but** sending a full
set of browser headers (``Sec-Fetch-*``, ``sec-ch-ua`` …) passes the
edge check and returns the real page.  No headless browser is required.

The flow mirrors the (now archived) ``smkirwan77/web_scraper`` project:

1. GET the racecard page
   ``/racecard/{Course-Slug}/{DD-Month-YYYY}/{HHMM}`` and read the
   ``RaceID`` from an HTML comment.
2. GET the four sectional AJAX fragments, which require that ``RaceID``::

       /ajax/sectionals/{raceid}/times?page=...&raceid=...
       /ajax/sectionals/{raceid}/tools?page=...&raceid=...
       /ajax/sectionals/{raceid}/stride?page=...&raceid=...
       /ajax/sectionals/{raceid}/distanceran?page=...&raceid=...

3. Parse one row per runner and cache to parquet, keyed (like RTV) on
   ``(race_date, track, off_time, horse_name)`` so it merges onto the
   main results frame with the shared normalisers in :mod:`src.utils`.

Usage::

    from src.attheraces_scraper import (
        fetch_atr_sectionals, backfill_atr_sectionals, merge_atr_sectionals,
    )

    rows = fetch_atr_sectionals("2026-06-10", "Kempton", "18:10")
    backfill_atr_sectionals(max_races=50)
    df = merge_atr_sectionals(results_df)

ATR/TPD sectional coverage is mostly UK tracks (all-weather plus many
turf courses from ~2021 on); Irish meetings are largely uncovered.  A
missing race returns HTTP 404 and is recorded in the no-data skip-list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from src.utils import (
    normalise_horse_key as _normalise_horse_key,
    normalise_off_time_key as _normalise_off_time_key,
    normalise_track_key as _normalise_track_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATR_BASE = "https://www.attheraces.com"

# A full browser header set is *required* to pass the Fastly JS challenge;
# a bare User-Agent gets the ~3 KB "Client Challenge" stub instead.
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
# Each race costs up to 4 requests (1 racecard + 3 AJAX fragments).
_REQUEST_DELAY = 1.5

# Status codes that mean "backed off / blocked" rather than "no such race".
# These must NOT be cached as no-data — they are transient.
_BLOCKED_STATUSES = {403, 429, 503}

ATR_CACHE_DIR = os.path.join(config.DATA_DIR, "atr_cache")
ATR_CACHE_FILE = os.path.join(ATR_CACHE_DIR, "atr_sectionals.parquet")
ATR_NO_DATA_FILE = os.path.join(ATR_CACHE_DIR, "atr_no_data_keys.json")
os.makedirs(ATR_CACHE_DIR, exist_ok=True)

# Scalar numeric feature columns (NaN when absent) exposed by the merge.
ATR_METRIC_COLS = [
    "atr_final_time",            # total race time for the runner (seconds)
    "atr_finishing_speed_pct",   # finishing speed as % of average (TPD)
    "atr_efficiency_score",      # TPD efficiency score
    "atr_peak_stride_length",    # peak stride length (TPD units)
    "atr_peak_stride_freq",      # peak stride frequency (strides/sec)
    "atr_n_sections",            # number of furlong splits captured
    "atr_adj_placing",           # sectional-adjusted finishing placing
]

# Text / identifier columns carried through the merge.
ATR_TEXT_COLS = [
    "atr_raceid",
    "atr_horse_id",
    "atr_saddle_cloth",
    "atr_draw",
    "atr_official_position",
    "atr_efficiency_grade",
    "atr_in_running_comment",
]

# Variable-length per-furlong series, stored as JSON strings so the cache
# stays a flat parquet table.  Nothing is discarded — downstream code can
# ``json.loads`` these to recover the full sectional profile.
ATR_ARRAY_COLS = [
    "atr_section_markers",       # ["5f","4f",...,"Fin."]
    "atr_split_times",           # per-furlong split time (s)
    "atr_running_totals",        # cumulative time at each marker (s)
    "atr_pace_flags",            # per-split pace class (even / fast / slow ...)
    "atr_sectional_speeds",      # segment speeds (mph)
    "atr_sectional_speed_labels",
    "atr_energy_dist",           # energy distribution % per marker
    "atr_optimum_energy",        # optimum energy distribution % per marker
    "atr_stride_length",         # stride length per marker
]

ALL_ATR_COLS = ATR_METRIC_COLS + ATR_TEXT_COLS + ATR_ARRAY_COLS

# The four AJAX sectional fragments.
_AJAX_KINDS = ("times", "tools", "stride", "distanceran")

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


class _Fetcher:
    """A small HTTP client that can clear the ATR JS/TLS challenge."""

    def __init__(self, request_delay: float | None = None) -> None:
        self._delay = _REQUEST_DELAY if request_delay is None else float(request_delay)
        self._last_request = 0.0
        self._cookie_jar: str | None = None
        self._cffi = None
        self._req = None

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
            time.sleep(gap - elapsed)
        self._last_request = time.time()

    def get(self, url: str, *, referer: str | None = None) -> tuple[int, str] | None:
        """GET *url*; return ``(status_code, text)`` or None on transport error.

        The JS "Client Challenge" stub is reported as status 429 so callers
        treat it as a transient block, never as genuine absence of data.
        """
        self._throttle()
        headers = _ajax_headers(referer) if referer else {}
        try:
            if self.mode == "curl_bin":
                status, text = self._get_curl_bin(url, headers)
            elif self.mode == "curl_cffi":
                r = self._cffi.get(url, headers=headers, timeout=25)
                status, text = r.status_code, r.text
            else:
                r = self._req.get(url, headers=headers, timeout=25, allow_redirects=True)
                status, text = r.status_code, r.text
        except Exception as exc:  # pragma: no cover - network drift
            logger.warning("ATR request failed (%s): %s", self.mode, exc)
            return None

        if status == 200 and "Client Challenge" in text[:4000]:
            logger.debug("ATR served the JS challenge for %s (treating as blocked)", url)
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
    """Read the ATR RaceID embedded in an HTML comment on the racecard."""
    m = re.search(r"RaceID:\s*(\d+)", html_text)
    if m:
        return m.group(1)
    # Fallback: the raceid query param attached to sectional/form links.
    m = re.search(r"raceid=(\d+)", html_text)
    return m.group(1) if m else None


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
    m = re.search(r"/horse/[^/]+/(\d+)", href)
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


def _parse_times(soup: BeautifulSoup) -> dict[str, dict]:
    """Parse the ``times`` fragment: split times, running totals, pace flags."""
    out: dict[str, dict] = {}
    for entry in _sectional_entries(soup):
        meta = _runner_meta(entry)
        name = meta.get("horse_name")
        if not name:
            continue

        splits: list[float] = []
        running: list[float] = []
        flags: list[str] = []
        for cell in entry.find_all("div", class_="card-cell--sectional-time"):
            sec = cell.find("div", class_=lambda c: c and "card-sectional" in c)
            vis = cell.find("span", class_="visible")
            if vis is None:
                continue  # finish marker cell carries no value
            val = _to_float(vis.get_text(strip=True))
            if val is None:
                continue
            splits.append(val)
            inner = cell.find("div", class_="card-sectional__inner")
            rt = _to_float(inner.get("title")) if inner and inner.get("title") else None
            if rt is not None:
                running.append(rt)
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
        rec["atr_n_sections"] = float(len(splits)) if splits else None
        rec["atr_final_time"] = running[-1] if running else None
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
        rec: dict = {}

        speeds: list[float] = []
        labels: list[str] = []
        for pill in entry.find_all("span", class_=lambda c: c and "text-pill" in c):
            classes = pill.get("class") or []
            if "text-pill--grey" in classes:
                # efficiency grade pill, e.g. "C" with title "Efficency Score: 17.18"
                rec["atr_efficiency_grade"] = pill.get_text(strip=True) or None
                rec["atr_efficiency_score"] = _to_float(pill.get("title"))
                continue
            v = _to_float(pill.get_text(strip=True))
            if v is not None:
                speeds.append(v)
                labels.append((pill.get("title") or "").strip())
        if speeds:
            rec["atr_sectional_speeds"] = speeds
            rec["atr_sectional_speed_labels"] = labels

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
        rec: dict = {}

        chart = entry.find("div", class_="chart-energy-distribution")
        markers, length, _peak_line = _parse_chart(chart)
        if length:
            rec["atr_stride_length"] = length
            if markers and "atr_section_markers" not in rec:
                rec["atr_section_markers"] = markers

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


def _fetch_one(
    fetcher: "_Fetcher", slug: str, atr_date: str, hhmm: str,
) -> tuple[str, list[dict] | None]:
    """Fetch + parse a single exact (slug, date, hhmm).

    Returns ``(outcome, rows)`` where outcome is one of :data:`DATA`,
    :data:`NOT_FOUND` or :data:`BLOCKED`.
    """
    page_path = f"/racecard/{slug}/{atr_date}/{hhmm}"
    card = fetcher.get(ATR_BASE + page_path)
    if card is None:
        return BLOCKED, None
    status, text = card
    if status == 404:
        return NOT_FOUND, None
    if status in _BLOCKED_STATUSES or status != 200 or not text:
        return BLOCKED, None

    raceid = _extract_raceid(text)
    if not raceid:
        # A real racecard with no RaceID comment carries no sectionals.
        return NOT_FOUND, None

    referer = ATR_BASE + page_path
    parsed: dict[str, dict] = {}
    any_blocked = False
    parsers = {
        "times": _parse_times,
        "tools": _parse_tools,
        "stride": _parse_stride,
    }
    for kind, parser in parsers.items():
        url = (
            f"{ATR_BASE}/ajax/sectionals/{raceid}/{kind}"
            f"?page={page_path}&raceid={raceid}"
        )
        resp = fetcher.get(url, referer=referer)
        if resp is None:
            any_blocked = True
            continue
        s, t = resp
        if s == 404:
            continue  # this fragment is simply absent for the race
        if s in _BLOCKED_STATUSES or s != 200 or not t:
            any_blocked = True
            continue
        soup = BeautifulSoup(t, "html.parser")
        for name, rec in parser(soup).items():
            parsed.setdefault(name, {}).update(rec)

    rows: list[dict] = []
    for name, rec in parsed.items():
        # Require at least one real sectional signal to count as data.
        if not (rec.get("atr_split_times") or rec.get("atr_sectional_speeds")
                or rec.get("atr_stride_length")):
            continue
        rec.setdefault("horse_name", name)
        rec["atr_raceid"] = raceid
        rows.append(rec)

    if rows:
        return DATA, rows
    # No usable rows: if a fragment was blocked we can't be sure there is no
    # data, so report BLOCKED (retryable) rather than NOT_FOUND.
    return (BLOCKED, None) if any_blocked else (NOT_FOUND, None)


def fetch_atr_sectionals_outcome(
    date: str, track_name: str, off_time: str,
    fetcher: "_Fetcher | None" = None,
) -> tuple[str, list[dict] | None]:
    """Fetch ATR sectionals, returning ``(outcome, rows)``.

    Off-times are stored in UTC; the ATR URL uses UK local time, so during
    BST we try the +1 h time first and fall back to the unadjusted time.
    """
    slug = _track_to_slug(track_name)
    atr_date = _date_to_atr(date)
    if not atr_date:
        return INVALID, None
    hhmm = _off_time_to_hhmm(off_time)

    owns_fetcher = fetcher is None
    f = fetcher or _Fetcher()
    try:
        candidates = [hhmm]
        if _is_bst(date):
            candidates.insert(0, _add_one_hour(hhmm))
        saw_blocked = False
        for cand in candidates:
            outcome, rows = _fetch_one(f, slug, atr_date, cand)
            if outcome == DATA:
                return DATA, rows
            if outcome == BLOCKED:
                saw_blocked = True
        # If any candidate was blocked we cannot conclude "no data".
        return (BLOCKED, None) if saw_blocked else (NOT_FOUND, None)
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

def load_atr_cache() -> pd.DataFrame:
    """Load cached ATR sectionals (or return empty frame)."""
    if os.path.exists(ATR_CACHE_FILE):
        return pd.read_parquet(ATR_CACHE_FILE)
    return pd.DataFrame()


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
    _encode_arrays(df).to_parquet(ATR_CACHE_FILE, index=False, engine="pyarrow")
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
    payload = {"updated_at": datetime.now().isoformat(), "race_keys": sorted(keys)}
    with open(ATR_NO_DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill_atr_sectionals(
    max_races: int | None = None,
    skip_existing: bool = True,
    request_delay: float | None = None,
    exclude_tracks: set[str] | None = None,
) -> pd.DataFrame:
    """Scrape ATR sectionals for every race in ``race_results.csv``.

    Appends to the cache incrementally so progress survives interruption.
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
    if not os.path.exists(raw_path):
        logger.error("race_results.csv not found at %s", raw_path)
        return pd.DataFrame()

    raw = pd.read_csv(raw_path, usecols=["race_date", "track", "off_time"])
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
    Mirrors :func:`src.rtv_scraper.backfill_rtv_metrics_for_races`.

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
        ck = cache[["race_date", "track", "off_time"]].copy()
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

    if max_races is not None:
        races = races.head(max_races)

    if races.empty:
        return {
            "target_races": target_total, "excluded_tracks_races": excluded_races,
            "missing_races": 0, "fetched_races": 0,
            "new_rows": 0, "cache_rows": int(len(cache)),
            "skipped_known_missing": skipped_known_missing,
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
    today = datetime.now().date()

    def _flush() -> None:
        nonlocal cache, new_rows, total_new_rows
        if new_rows:
            cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
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
                _rd = pd.to_datetime(date_str, errors="coerce")
                if pd.notna(_rd) and _rd.date() < today:
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
        "aborted_blocked": aborted,
    }


# ---------------------------------------------------------------------------
# Merge helper (used by feature_engineer)
# ---------------------------------------------------------------------------

def merge_atr_sectionals(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join cached ATR sectionals onto a race DataFrame.

    Matches on ``(race_date, track, off_time, horse_name)`` using the
    shared normalisers.  All ATR columns are added (NaN where absent);
    per-furlong series remain JSON strings (see :data:`ATR_ARRAY_COLS`).
    """
    cache = load_atr_cache()
    if cache.empty:
        for c in ALL_ATR_COLS:
            df[c] = float("nan")
        return df

    cache = cache.copy()
    cache["_rd"] = pd.to_datetime(cache["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cache["_trk"] = cache["track"].map(_normalise_track_key)
    cache["_ot"] = cache["off_time"].map(_normalise_off_time_key)
    cache["_hn"] = cache["horse_name"].map(_normalise_horse_key)

    for c in ALL_ATR_COLS:
        if c not in cache.columns:
            cache[c] = float("nan")

    merge_cols = ["_rd", "_trk", "_ot", "_hn"] + ALL_ATR_COLS
    cache_slim = cache[merge_cols].drop_duplicates(
        subset=["_rd", "_trk", "_ot", "_hn"], keep="last",
    )

    df["_rd"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["_trk"] = df["track"].map(_normalise_track_key)
    df["_ot"] = df["off_time"].map(_normalise_off_time_key)
    df["_hn"] = df["horse_name"].map(_normalise_horse_key)

    merged = df.merge(
        cache_slim, on=["_rd", "_trk", "_ot", "_hn"], how="left", indicator="_atr_merge",
    )
    merged = merged.drop(columns=["_rd", "_trk", "_ot", "_hn"], errors="ignore")

    for c in ALL_ATR_COLS:
        if c not in merged.columns:
            merged[c] = float("nan")

    matched = (merged["_atr_merge"] == "both") if "_atr_merge" in merged.columns else pd.Series(False, index=merged.index)
    logger.info(
        "ATR merge: key matches %d / %d (%.1f%%)",
        int(matched.sum()), len(merged), 100 * matched.sum() / max(1, len(merged)),
    )
    return merged.drop(columns=["_atr_merge"], errors="ignore")


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
