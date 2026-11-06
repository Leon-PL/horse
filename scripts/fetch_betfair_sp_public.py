"""Download and extract Betfair's free, public daily BSP files.

Betfair publishes one CSV per market type, region and day at
``https://promo.betfair.com/betfairsp/prices/dwbfprices{region}{market}{DDMMYYYY}.csv``.
Each row is one runner: BSP, the pre-play and morning volume-weighted average
prices (``ppwap``, ``morningwap``), the pre-play and in-play price range, and
the matched volume traded in the morning, pre-play and in-play.

Why this exists. The Betfair Historical BASIC tars behind
``data/processed/betfair_bsp.parquet`` are downloaded by hand, stop at
2026-06-10, and miss 2020Q4 and 2021Q3-Q4 - so no CLV can be measured on the
most recent months, which are the only ones no research has touched. These
files fill both gaps, and they carry **traded volume**, which the BASIC tier
does not have at all (ACTION_PLAN 5a).

Three traps, each of which would fail silently:

* **File date is the day after the races.** ``...ukwin01072026.csv`` holds
  races run on 30 June 2026.
* **Off times are UK local time**; ``data/races.db`` and the BASIC parquet hold
  UTC. Without conversion every British Summer Time race - half the year -
  fails to join. Ireland keeps the same clock, so Europe/London covers both.
* **Two header styles.** Older files use upper-case headers and menu hints
  like ``"GB / Thirsk  14th Sep"``; newer ones ``"Thirsk 14th Sep"``.

Usage::

    python scripts/fetch_betfair_sp_public.py --download --start 2020-09-14 --end 2026-10-12
    python scripts/fetch_betfair_sp_public.py --build
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import bf_horse_key, bf_track_key  # noqa: E402
from src.utils import normalise_off_time_key  # noqa: E402
from scripts.extract_betfair_bsp import _clean_horse, _track_from_event  # noqa: E402

BASE_URL = "https://promo.betfair.com/betfairsp/prices/dwbfprices{region}{market}{ddmmyyyy}.csv"
RAW_DIR = os.path.join("data", "betfair_sp_public")
OUT_PATH = os.path.join("data", "processed", "betfair_sp_public.parquet")
REGIONS = {"uk": "GB", "ire": "IE"}
LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")
MISSING_LOG = "missing.json"

NUMERIC = ["bsp", "ppwap", "morningwap", "ppmax", "ppmin", "ipmax", "ipmin",
           "morningtradedvol", "pptradedvol", "iptradedvol"]
# "GB / Thirsk  14th Sep" -> "Thirsk  14th Sep"
COUNTRY_PREFIX_RE = re.compile(r"^\s*[A-Z]{2,3}\s*/\s*")


def raw_path(market: str, region: str, race_date: date, *, raw_dir=None) -> str:
    return os.path.join(raw_dir or RAW_DIR, market, region, f"{race_date.isoformat()}.csv")


def download(start: date, end: date, *, regions=("uk", "ire"), market: str = "win",
             delay: float = 0.3, timeout: int = 30, raw_dir=None) -> dict:
    """Fetch every file for races in [start, end]; resumable, polite."""
    session = requests.Session()
    session.headers["User-Agent"] = "horse-research/1.0 (historical BSP archive)"
    stats = {"fetched": 0, "cached": 0, "missing": 0, "failed": 0}
    for region in regions:
        folder = os.path.join(raw_dir or RAW_DIR, market, region)
        os.makedirs(folder, exist_ok=True)
        missing_path = os.path.join(folder, MISSING_LOG)
        missing = set(json.load(open(missing_path))) if os.path.exists(missing_path) else set()
        day = start
        while day <= end:
            target = raw_path(market, region, day, raw_dir=raw_dir)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                stats["cached"] += 1
            elif day.isoformat() in missing and (date.today()-day).days > 10:
                stats["missing"] += 1
            else:
                file_day = day + timedelta(days=1)  # the file is dated the day after
                url = BASE_URL.format(region=region, market=market,
                                      ddmmyyyy=file_day.strftime("%d%m%Y"))
                try:
                    resp = session.get(url, timeout=timeout)
                    # The server rate-limits bursts (HTTP 429). Back off and
                    # retry rather than recording a failure for every file.
                    for attempt in range(5):
                        if resp.status_code != 429:
                            break
                        wait = float(resp.headers.get("Retry-After") or 0) or 5.0 * 2 ** attempt
                        time.sleep(min(wait, 120.0))
                        resp = session.get(url, timeout=timeout)
                    if resp.status_code == 404:
                        missing.add(day.isoformat())
                        stats["missing"] += 1
                    else:
                        resp.raise_for_status()
                        body = resp.content
                        if not body.lstrip().lower().startswith(b"event_id"):
                            raise ValueError(f"unexpected body for {url}: {body[:60]!r}")
                        tmp = target + ".tmp"
                        with open(tmp, "wb") as fh:
                            fh.write(body)
                        os.replace(tmp, target)
                        missing.discard(day.isoformat())
                        stats["fetched"] += 1
                except Exception as exc:  # retried on the next run
                    stats["failed"] += 1
                    print(f"  {region} {day}: {exc}", flush=True)
                time.sleep(delay)
            day += timedelta(days=1)
        with open(missing_path, "w") as fh:
            json.dump(sorted(missing), fh)
        print(f"{region}: {stats}", flush=True)
    return stats


def parse_file(path: str, region: str) -> pd.DataFrame:
    """One raw CSV -> runner rows with UTC off times and Betfair join keys."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if df.empty:
        return df
    local = pd.to_datetime(df["event_dt"].str.strip(), format="%d-%m-%Y %H:%M", errors="coerce")
    utc = (local.dt.tz_localize(LONDON, ambiguous="NaT", nonexistent="NaT")
                .dt.tz_convert(UTC))
    out = pd.DataFrame({
        "race_date": utc.dt.strftime("%Y-%m-%d"),
        "off_time": utc.dt.strftime("%H:%M"),
        "track": df["menu_hint"].map(lambda s: _track_from_event(COUNTRY_PREFIX_RE.sub("", str(s)))),
        "horse_name": df["selection_name"].map(lambda s: _clean_horse(str(s))),
        "region": REGIONS[region],
        "event_id": df["event_id"].str.strip(),
        "event_name": df["event_name"].str.strip(),
        "selection_id": df["selection_id"].str.strip(),
        "won": pd.to_numeric(df["win_lose"], errors="coerce"),
    })
    for col in NUMERIC:
        out[col] = pd.to_numeric(df.get(col), errors="coerce")
    out["track_key"] = out["track"].map(bf_track_key)
    out["off_key"] = out["off_time"].map(normalise_off_time_key)
    out["horse_key"] = out["horse_name"].map(bf_horse_key)
    return out[utc.notna().to_numpy()]


def build(market: str = "win") -> pd.DataFrame:
    frames = []
    for region in REGIONS:
        folder = os.path.join(RAW_DIR, market, region)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".csv"):
                frames.append(parse_file(os.path.join(folder, name), region))
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["race_date", "track_key", "off_key", "horse_key"], keep="first")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    races = df.groupby(["race_date", "track_key", "off_key"]).ngroups
    print(f"built {OUT_PATH}: {len(df):,} runners ({before - len(df):,} duplicates dropped), "
          f"{races:,} races, {df['race_date'].min()}..{df['race_date'].max()}", flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="2020-09-14", help="first RACE date")
    ap.add_argument("--end", default=None, help="last RACE date (default: yesterday)")
    ap.add_argument("--market", default="win", choices=("win", "place"))
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()
    if args.download:
        end = (date.fromisoformat(args.end) if args.end
               else (datetime.now() - timedelta(days=1)).date())
        download(date.fromisoformat(args.start), end, market=args.market, delay=args.delay)
    if args.build:
        build(args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
