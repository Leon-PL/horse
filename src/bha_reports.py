"""Collect BHA going reports and race lists (stalls, rail movements, GoingStick).

The BHA API behind britishhorseracing.com publishes, per GB fixture back to at
least 2020:

* ``fixtures`` - per fixture: ``stalls`` (free text by distance), ``rail``
  (free text), the official going;
* ``fixtures/{year}/{id}/races`` - per race: advertised distance and
  ``distanceChange``, the yards rail movements add to it;
* ``fixtures/{year}/{id}/going`` - GoingStick, sometimes split inside /
  centre / outside, plus watering.

Raw responses are cached under ``data/bha_going/``; :func:`build` and
:func:`build_races` flatten them into ``data/processed/bha_going.parquet`` and
``bha_races.parquet``, which ``src.course_reports`` joins onto results inside
``process_data``. This module only collects; it never changes a feature value
itself, and what it writes reaches features through ``process_data``, where
``data_signature`` covers it.

:func:`refresh` keeps recent days current. The daily sync calls it for the
dates it adds, *before* those results are appended to the feature store: a
BHA file that changed under already-stored rows would change
``data_signature`` and force a full rebuild.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date

import pandas as pd
import requests

import config
from src import bha_stewards_scraper as bha
from src import course_reports
from src.course_reports import bha_track_key

logger = logging.getLogger(__name__)

BASE = "https://api09.horseracing.software/bha/v1/"
ROOT = os.path.join(config.DATA_DIR, "bha_going")
#: After this many consecutive failures the long-window quota has tripped;
#: every request keeps failing until it resets, so pause instead of burning
#: through the fixture list.
_QUOTA_STRIKES = 3
_QUOTA_PAUSE_S = 60 * 60


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(bha._DEFAULT_HEADERS)
    s.headers["Authorization"] = f"Bearer {bha._extract_bearer_token(s)}"
    return s


def _get(s: requests.Session, path: str, params=None, delay: float = 0.5) -> dict:
    time.sleep(delay)
    resp = bha._get_with_backoff(s, BASE + path, params or {}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _write_json(path: str, payload) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(path + ".tmp", path)


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _list_fixtures(s: requests.Session, start: date, end: date, delay: float) -> list:
    items, page = [], 1
    while True:
        body = _get(s, "fixtures", {"fromdate": start.isoformat(), "todate": end.isoformat(),
                                    "page": page}, delay)
        items.extend(body.get("data", []))
        if page >= int(body.get("last_page") or 1):
            return items
        page += 1


def _fetch_per_fixture(s: requests.Session, fixtures: list, kind: str, delay: float,
                       strikes: list, *, refetch: bool = False) -> None:
    """Fetch ``fixtures/{year}/{id}/{kind}`` for each fixture (cached on disk)."""
    for f in fixtures:
        if kind == "races" and f.get("abandonedReasonCode"):
            continue
        path = os.path.join(ROOT, kind, f"{f['fixtureYear']}_{f['fixtureId']}.json")
        if os.path.exists(path) and not refetch:
            continue
        try:
            body = _get(s, f"fixtures/{f['fixtureYear']}/{f['fixtureId']}/{kind}", delay=delay)
        except Exception as exc:  # retried next run
            logger.warning("BHA %s %s/%s: %s", kind, f["fixtureYear"], f["fixtureId"], exc)
            strikes[0] += 1
            if strikes[0] >= _QUOTA_STRIKES:
                logger.warning("%d BHA failures in a row: pausing %d min",
                               strikes[0], _QUOTA_PAUSE_S // 60)
                time.sleep(_QUOTA_PAUSE_S)
                strikes[0] = 0
            continue
        strikes[0] = 0
        _write_json(path, body.get("data", {}))


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, 1)


def download(start: date, end: date, *, with_going: bool = False, with_races: bool = False,
             delay: float = 0.5) -> None:
    """Backfill: month listings (cached once), then per-fixture endpoints."""
    s = _session()
    strikes = [0]
    for kind in ("fixtures", "going", "races"):
        os.makedirs(os.path.join(ROOT, kind), exist_ok=True)
    month = _month_start(start)
    while month <= end:
        nxt = _next_month(month)
        target = os.path.join(ROOT, "fixtures", f"{month:%Y-%m}.json")
        if not os.path.exists(target):
            items = _list_fixtures(s, month, nxt - pd.Timedelta(days=1), delay)
            _write_json(target, items)
            logger.info("%s: %d fixtures", f"{month:%Y-%m}", len(items))
        fixtures = [f for f in json.load(open(target, encoding="utf-8"))
                    if start.isoformat() <= str(f.get("fixtureDate")) <= end.isoformat()]
        if with_going:
            _fetch_per_fixture(s, fixtures, "going", delay, strikes)
        if with_races:
            _fetch_per_fixture(s, fixtures, "races", delay, strikes)
            logger.info("%s: races fetched", f"{month:%Y-%m}")
        month = nxt


def refresh(dates, *, delay: float = 1.0) -> int:
    """Re-list and re-fetch the fixtures on *dates*, then rebuild both files.

    Month listings are cached, and a month in progress gains fixtures (and
    going-report updates) after its file was written, so the listing for the
    window is fetched again and merged in; race lists are re-fetched too.
    Returns the number of fixtures in the window.
    """
    days = sorted({pd.Timestamp(d).date() for d in dates})
    if not days:
        return 0
    start, end = days[0], days[-1]
    s = _session()
    for kind in ("fixtures", "races"):
        os.makedirs(os.path.join(ROOT, kind), exist_ok=True)
    listed = [f for f in _list_fixtures(s, start, end, delay)
              if str(f.get("fixtureDate")) in {d.isoformat() for d in days}]
    month = _month_start(start)
    while month <= end:
        target = os.path.join(ROOT, "fixtures", f"{month:%Y-%m}.json")
        cached = json.load(open(target, encoding="utf-8")) if os.path.exists(target) else []
        fresh = [f for f in listed if str(f.get("fixtureDate", "")).startswith(f"{month:%Y-%m}")]
        fresh_dates = {str(f.get("fixtureDate")) for f in fresh} | {
            d.isoformat() for d in days if d.strftime("%Y-%m") == f"{month:%Y-%m}"}
        merged = [f for f in cached if str(f.get("fixtureDate")) not in fresh_dates] + fresh
        merged.sort(key=lambda f: (str(f.get("fixtureDate")), f.get("fixtureId", 0)))
        _write_json(target, merged)
        month = _next_month(month)
    _fetch_per_fixture(s, listed, "races", delay, [0], refetch=True)
    build()
    build_races()
    return len(listed)


def build() -> pd.DataFrame:
    """Flatten the cached fixture listings into ``bha_going.parquet``."""
    rows = []
    folder = os.path.join(ROOT, "fixtures")
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not name.endswith(".json"):
            continue
        for f in json.load(open(os.path.join(folder, name), encoding="utf-8")):
            going_path = os.path.join(ROOT, "going", f"{f['fixtureYear']}_{f['fixtureId']}.json")
            cond = {}
            if os.path.exists(going_path):
                cond = (json.load(open(going_path, encoding="utf-8")) or {}).get("conditions") or {}
            rows.append({
                "race_date": f.get("fixtureDate"),
                "course": f.get("courseName"),
                "track_key": bha_track_key(f.get("courseName")),
                "fixture_type": f.get("fixtureType"),
                "track_type": f.get("racingTrackType"),
                "session": f.get("fixtureSession"),
                "going": f.get("going"),
                "stalls": f.get("stalls") or cond.get("stalls"),
                "rail": f.get("rail") or cond.get("rails"),
                "goingstick": pd.to_numeric(cond.get("goingStick"), errors="coerce"),
                "goingstick_comment": cond.get("goingStickComment"),
                "watering": cond.get("watering"),
                "abandoned": f.get("abandonedReasonCode"),
                "fixture_year": f.get("fixtureYear"), "fixture_id": f.get("fixtureId"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["fixture_year", "fixture_id"]).sort_values(
        ["race_date", "course", "fixture_id"]).reset_index(drop=True)
    _write_parquet(df, course_reports.GOING_PATH)
    logger.info("built %s: %d fixtures, %s..%s", course_reports.GOING_PATH, len(df),
                df["race_date"].min(), df["race_date"].max())
    return df


def build_races() -> pd.DataFrame:
    """One row per BHA race: advertised distance and the rail-movement change."""
    fixtures = {}
    folder = os.path.join(ROOT, "fixtures")
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.endswith(".json"):
            for f in json.load(open(os.path.join(folder, name), encoding="utf-8")):
                fixtures[f"{f['fixtureYear']}_{f['fixtureId']}"] = f
    folder = os.path.join(ROOT, "races")
    rows = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if not name.endswith(".json"):
            continue
        f = fixtures.get(name[:-5], {})
        body = json.load(open(os.path.join(folder, name), encoding="utf-8")) or []
        for r in body if isinstance(body, list) else []:
            rows.append({
                "race_date": r.get("raceDate"),
                "race_time": r.get("raceTime"),
                "course": f.get("courseName"),
                "track_key": bha_track_key(f.get("courseName")),
                "race_name": r.get("raceName"),
                "race_type": r.get("raceCriteriaRaceType"),
                "distance_text": r.get("distanceText"),
                "distance_yards": pd.to_numeric(r.get("distanceValue"), errors="coerce"),
                "distance_change_yards": pd.to_numeric(r.get("distanceChange"), errors="coerce"),
                "abandoned": r.get("abandonedReasonCode"),
                "fixture_year": f.get("fixtureYear"), "fixture_id": f.get("fixtureId"),
                "bha_race_id": r.get("raceId"), "division": r.get("divisionSequence"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["fixture_year", "bha_race_id", "division"]).sort_values(
        ["race_date", "course", "race_time", "bha_race_id"]).reset_index(drop=True)
    _write_parquet(df, course_reports.RACES_PATH)
    logger.info("built %s: %d races, %s..%s; distance change non-zero on %.0f%%", course_reports.RACES_PATH,
                len(df), df["race_date"].min(), df["race_date"].max(),
                100 * (df["distance_change_yards"].fillna(0) != 0).mean())
    return df
