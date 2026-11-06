"""Collect BHA going reports: stalls position, rail movements, GoingStick.

ACTION_PLAN 5a-bis. The store has 11 draw columns and no stalls side, so a
low draw means one thing with stalls on the inside and the opposite with
stalls on the stands side. The BHA API behind britishhorseracing.com
publishes, per GB fixture back to at least 2020:

* ``stalls`` - free text by distance ("5f, 6f & 1m4f: Centre / 7f: Inside"),
* ``rail``   - rail movements ("Bend dolled out 2yds"),
* ``going``  - the official description,

and, from ``fixtures/{year}/{id}/going``, the GoingStick reading, sometimes
split inside / centre / outside, plus watering. A single GoingStick value is
race-level and cancels within a race (ledger 2g); split readings combined
with stalls side and draw are runner-specific, which is the only way such
data could pay.

Usage::

    python scripts/fetch_bha_going.py --download --start 2020-09-01 --end 2026-10-12
    python scripts/fetch_bha_going.py --download --going      # also per-fixture reports
    python scripts/fetch_bha_going.py --build
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion import bha_stewards as bha  # noqa: E402
from src.utils import canonical_track_key  # noqa: E402

BASE = "https://api09.horseracing.software/bha/v1/"
#: BHA course names canonical_track_key does not map to our results' names.
#: Kept here, not in src/utils._TRACK_ALIAS, because utils.py is hashed into
#: fe_signature: an alias no feature uses would force a full store rebuild.
#: Move them into utils at the next rebuild that is needed anyway.
_BHA_ALIAS = {"great yarmouth": "yarmouth", "royal ascot": "ascot"}


def bha_track_key(name) -> str:
    """Join key for BHA fixtures: apply to BOTH sides (BHA courses and DB tracks)."""
    key = canonical_track_key(name)
    return _BHA_ALIAS.get(key, key)
ROOT = os.path.join("data", "bha_going")
OUT = os.path.join("data", "processed", "bha_going.parquet")


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


def download(start: date, end: date, *, with_going: bool, delay: float) -> None:
    s = _session()
    os.makedirs(os.path.join(ROOT, "fixtures"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "going"), exist_ok=True)
    month = date(start.year, start.month, 1)
    while month <= end:
        nxt = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        target = os.path.join(ROOT, "fixtures", f"{month:%Y-%m}.json")
        if not os.path.exists(target):
            items, page = [], 1
            while True:
                body = _get(s, "fixtures", {"fromdate": month.isoformat(),
                                            "todate": (nxt - pd.Timedelta(days=1)).isoformat(),
                                            "page": page}, delay)
                items.extend(body.get("data", []))
                if page >= int(body.get("last_page") or 1):
                    break
                page += 1
            with open(target + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(items, fh)
            os.replace(target + ".tmp", target)
            print(f"{month:%Y-%m}: {len(items)} fixtures", flush=True)
        if with_going:
            for f in json.load(open(target, encoding="utf-8")):
                path = os.path.join(ROOT, "going", f"{f['fixtureYear']}_{f['fixtureId']}.json")
                if os.path.exists(path):
                    continue
                try:
                    body = _get(s, f"fixtures/{f['fixtureYear']}/{f['fixtureId']}/going", delay=delay)
                except Exception as exc:  # retried next run
                    print(f"  going {f['fixtureYear']}/{f['fixtureId']}: {exc}", flush=True)
                    continue
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(body.get("data", {}), fh)
        month = nxt


def build() -> pd.DataFrame:
    rows = []
    folder = os.path.join(ROOT, "fixtures")
    for name in sorted(os.listdir(folder)):
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
    df = pd.DataFrame(rows).drop_duplicates(["fixture_year", "fixture_id"])
    df.to_parquet(OUT, index=False)
    print(f"built {OUT}: {len(df):,} fixtures, {df['race_date'].min()}..{df['race_date'].max()}; "
          f"stalls text on {df['stalls'].astype(bool).mean():.0%}, rail text on "
          f"{df['rail'].astype(bool).mean():.0%}, goingstick on {df['goingstick'].notna().mean():.0%}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--going", action="store_true", help="also fetch per-fixture going reports")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--start", default="2020-09-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    if args.download:
        download(date.fromisoformat(args.start), date.fromisoformat(args.end),
                 with_going=args.going, delay=args.delay)
    if args.build:
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
