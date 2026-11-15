"""Matchbook's public price feed for British and Irish horse racing: read-only, no account.

``GET https://api.matchbook.com/edge/rest/events`` with ``tag-url-names=horse-racing,uk`` (or ``ireland``) lists
open races with every runner's best three back and lay prices and the stake on offer. No session token is sent, so
nothing here can place, cancel or see bets. Prices are decimal odds; Matchbook's ``back`` side is the price a
backer can take now.

Event names are ``"HH:MM Track"`` in UK time; ``start`` is UTC. Runner names carry the saddle-cloth number
(``"4 Mysterland"``), which is stripped into ``cloth``.
"""
import re
import time

import numpy as np
import pandas as pd
import requests

URL = "https://api.matchbook.com/edge/rest/events"
REGIONS = {"uk": "GB", "ireland": "IE"}
DEPTH = 3
PAGE = 100
USER_AGENT = "racing-research/1.0 (read-only price recorder)"
LONDON = "Europe/London"
_CLOTH = re.compile(r"^\s*(\d+)[A-Za-z]?\s+(.*\S)\s*$")


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def _get(http, params, retries=4):
    for attempt in range(retries):
        try:
            response = http.get(URL, params=params, timeout=30)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def fetch_events(region, http=None):
    """Every listed race of a region with prices, following pages."""
    http = http or session()
    events, offset = [], 0
    while True:
        body = _get(http, {"tag-url-names": f"horse-racing,{region}", "per-page": PAGE, "offset": offset,
                           "include-prices": "true", "price-depth": DEPTH, "odds-type": "DECIMAL",
                           "exchange-type": "back-lay", "currency": "GBP"})
        events += body.get("events", [])
        offset += PAGE
        if offset >= body.get("total", 0) or not body.get("events"):
            return events


def split_name(name):
    match = _CLOTH.match(str(name))
    return (int(match.group(1)), match.group(2)) if match else (np.nan, str(name).strip())


def flatten(events, region, taken_at):
    """One row per runner of each race's WIN market."""
    rows = []
    for event in events:
        if event.get("sport-id") != 24735152712200:
            continue
        start = pd.Timestamp(event["start"])
        local = start.tz_convert(LONDON)
        name = str(event.get("name", ""))
        track = name.split(" ", 1)[1].strip() if re.match(r"^\d{1,2}:\d{2}\s", name) else name
        for market in event.get("markets", []):
            if str(market.get("name", "")).upper() != "WIN":
                continue
            for runner in market.get("runners", []):
                cloth, horse = split_name(runner.get("name"))
                row = {"taken_at": taken_at, "region": REGIONS[region], "event_id": str(event["id"]),
                       "event_name": name, "start_utc": start, "race_date": str(local.date()),
                       "off_time": local.strftime("%H:%M"), "track": track, "event_status": event.get("status"),
                       "market_id": str(market["id"]), "market_status": market.get("status"),
                       "in_running": bool(market.get("in-running-flag", False)),
                       "market_volume": float(market.get("volume") or 0.),
                       "runner_id": str(runner["id"]), "runner_name": runner.get("name"), "cloth": cloth,
                       "horse_name": horse, "runner_status": runner.get("status"),
                       "withdrawn": bool(runner.get("withdrawn", False)),
                       "runner_volume": float(runner.get("volume") or 0.)}
                for side in ("back", "lay"):
                    quotes = [p for p in runner.get("prices", []) if p.get("side") == side]
                    quotes.sort(key=lambda p: -p["decimal-odds"] if side == "back" else p["decimal-odds"])
                    for level in range(DEPTH):
                        quote = quotes[level] if level < len(quotes) else {}
                        row[f"{side}{level + 1}"] = float(quote.get("decimal-odds", np.nan))
                        row[f"{side}{level + 1}_size"] = float(quote.get("available-amount", np.nan))
                rows.append(row)
    return pd.DataFrame(rows)


def snapshot_frame(http=None):
    """Both regions now, one row per runner."""
    http = http or session()
    taken_at = pd.Timestamp.now(tz="UTC")
    frames = [flatten(fetch_events(region, http), region, taken_at) for region in REGIONS]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reference_price(back, lay, max_spread=1.25):
    """A traded-price stand-in: the geometric middle of best back and lay, else the best back.

    Betfair's last traded price, which the model was fitted on, sits between the two. When the lay is
    more than ``max_spread`` times the back (thin evening books: back 6.0, lay 220) the middle says
    nothing and would shrink the runner's share into a false edge, so the best back is used. A runner
    with no back price has none (NaN).
    """
    back = pd.to_numeric(pd.Series(back), errors="coerce").to_numpy(float)
    lay = pd.to_numeric(pd.Series(lay), errors="coerce").to_numpy(float)
    both = (back > 1) & (lay >= back) & (lay <= back * max_spread)
    middle = np.sqrt(np.where(both, back * lay, 1.))
    return np.where(both, middle, np.where(back > 1, back, np.nan))
