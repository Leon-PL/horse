"""Fetch archived day-ahead weather forecasts for every course (EXPLORATION_PLAN C).

Idea C needs forecasts *as they were issued* before a decision time - never
observations (the ``weather_*`` columns come from Open-Meteo's archive, which
records what happened and is look-ahead for any forecast test). Open-Meteo's
Previous Runs API serves, for each valid hour, the value forecast 24 h
(``_previous_day1``) and 48 h (``_previous_day2``) earlier. Coverage begins
in early 2024 for the default model (2024-01-15 empty, 2024-03-01 present).

One request per course for the whole period, cached raw under
``data/weather_forecast/``; resumable (a cached course is skipped) and slow.

Usage::

    python scripts/fetch_forecast_previous_runs.py --start 2024-01-01 --end 2026-10-12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from src.track_config import TRACK_CONFIG

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OUT_DIR = os.path.join("data", "weather_forecast")
VARS = [f"{v}_previous_day{d}" for v in ("precipitation", "temperature_2m", "wind_speed_10m",
                                          "relative_humidity_2m") for d in (1, 2)] + ["precipitation"]


def fetch(track: str, lat: float, lon: float, start: str, end: str, delay: float) -> str:
    path = os.path.join(OUT_DIR, f"{track.lower().replace(' ', '_')}.json")
    if os.path.exists(path):
        return "cached"
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(VARS),
              "start_date": start, "end_date": end, "timezone": "UTC"}
    for attempt in range(5):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(60 * (attempt + 1))
            continue
        r.raise_for_status()
        payload = r.json()
        payload["_track"] = track
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        time.sleep(delay)
        return "ok"
    return "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-10-12")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    counts: dict[str, int] = {}
    for track, cfg in sorted(TRACK_CONFIG.items()):
        lat, lon = cfg.get("lat"), cfg.get("lon")
        if lat is None or lon is None:
            counts["no_coords"] = counts.get("no_coords", 0) + 1
            continue
        status = fetch(track, float(lat), float(lon), args.start, args.end, args.delay)
        counts[status] = counts.get(status, 0) + 1
        print(f"  {track}: {status}", flush=True)
    print(counts, flush=True)
    return 0 if not counts.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
