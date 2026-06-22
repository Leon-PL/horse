"""
Extract Betfair PLACE-market BSP from the historical BASIC streams.
==================================================================
Companion to ``extract_betfair_bsp.py`` (which keeps only WIN markets). The same
``data/betfair/*.tar`` archives also carry the **PLACE** ("to be placed") market
per race, with its own BSP and ``numberOfWinners`` (= places paid). The place
market is a *softer* market than win-BSP, so it's where the place model's
selection value might actually convert to edge (see memory: place-market-edge-lead).

Emits one row per runner:
    * ``place_bsp``    — Betfair Starting Price of the PLACE market (low-margin,
                         the realistic price you'd get for a place bet)
    * ``bf_places_paid`` — numberOfWinners (places the market settled)
    * ``placed_bf``    — 1 if the runner was a WINNER (i.e. placed) in the market
    * merge keys (``track_key``/``off_key``/``horse_key`` + ``race_date``)

Usage (mirror the win extractor):
    python scripts/extract_betfair_place_bsp.py --tar "data/betfair/2025.tar" \
        --tar "data/betfair/2026.tar" --out data/processed/betfair_place_bsp.parquet
"""
from __future__ import annotations
import argparse, bz2, json, os, sys, tarfile, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import bf_horse_key  # noqa: E402
from src.utils import normalise_off_time_key, normalise_track_key  # noqa: E402
# Reuse the win extractor's parsing helpers so behaviour stays identical.
from scripts.extract_betfair_bsp import (  # noqa: E402
    KEEP_COUNTRIES, MARKET_FILE_RE, _track_from_event, _clean_horse,
)


def parse_place_market(stream: bytes) -> list[dict] | None:
    """Parse one market's mcm stream -> place rows, or None if not a PLACE market."""
    final_def = None
    for raw in stream.split(b"\n"):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for mc in d.get("mc", []):
            if mc.get("marketDefinition"):
                final_def = mc["marketDefinition"]
    if not final_def:
        return None
    if final_def.get("marketType") != "PLACE":
        return None
    if final_def.get("countryCode") not in KEEP_COUNTRIES:
        return None
    if final_def.get("eventTypeId") != "7":
        return None

    places_paid = final_def.get("numberOfWinners")
    event_name = final_def.get("eventName", "")
    track = _track_from_event(event_name)
    market_time = final_def.get("marketTime", "")
    race_date = market_time[:10] if market_time else ""
    off_hhmm = market_time[11:16] if len(market_time) >= 16 else ""

    rows = []
    for r in final_def.get("runners", []):
        status = r.get("status")
        if status not in ("WINNER", "LOSER"):
            continue
        horse = _clean_horse(r.get("name", ""))
        rows.append({
            "race_date": race_date,
            "off_time": off_hhmm,
            "track": track,
            "horse_name": horse,
            "place_bsp": r.get("bsp"),
            "bf_places_paid": places_paid,
            "placed_bf": 1 if status == "WINNER" else 0,
            "track_key": normalise_track_key(track),
            "off_key": normalise_off_time_key(off_hhmm),
            "horse_key": bf_horse_key(horse),
        })
    return rows or None


def process_tar(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    n_markets = 0
    truncated = False
    with tarfile.open(path, "r") as tf:
        try:
            for member in tf:
                if not (member.isfile() and MARKET_FILE_RE.search(member.name)):
                    continue
                if limit and n_markets >= limit:
                    break
                try:
                    stream = bz2.decompress(tf.extractfile(member).read())
                except Exception:
                    continue
                parsed = parse_place_market(stream)
                if parsed:
                    rows.extend(parsed)
                    n_markets += 1
        except tarfile.ReadError as e:
            truncated = True  # truncated/corrupt archive — keep what we read
    tag = " (TRUNCATED, partial)" if truncated else ""
    print(f"  {os.path.basename(path)}: {n_markets} PLACE markets, {len(rows)} runners{tag}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="append", required=True)
    ap.add_argument("--out", default="data/processed/betfair_place_bsp.parquet")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    all_rows: list[dict] = []
    for tar in args.tar:
        if not os.path.exists(tar):
            print(f"  (missing: {tar})", flush=True); continue
        try:
            all_rows.extend(process_tar(tar, args.limit))
        except Exception as e:
            print(f"  (skipped {os.path.basename(tar)}: {type(e).__name__}: {e})", flush=True)
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No place rows extracted."); return

    if args.append and os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        df = pd.concat([prev, df], ignore_index=True)
    df["place_bsp"] = pd.to_numeric(df["place_bsp"], errors="coerce")
    # Same race's place BSP is identical across downloads — keep the row with a
    # real BSP, drop dupes on the merge keys.
    df = df.sort_values("place_bsp", na_position="last").drop_duplicates(
        ["race_date", "track_key", "off_key", "horse_key"], keep="first")
    df.to_parquet(args.out, index=False)
    cov = df["place_bsp"].notna().mean()
    print(f"\nWrote {len(df):,} place rows -> {args.out}  (place_bsp present {cov:.1%}, "
          f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
