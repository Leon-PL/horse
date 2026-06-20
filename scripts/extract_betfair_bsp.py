"""
Extract Betfair Exchange BSP + pre-off prices from historical BASIC streams.
==========================================================================

The ``data/betfair/*.tar`` archives are Betfair Historical "BASIC" data:

    BASIC/<year>/<Mon>/<day>/<eventId>/<marketId>.bz2

Each ``.bz2`` is a JSON-lines ``mcm`` (market-change) stream — the same feed
the live Exchange API emits. We keep only **GB/IE WIN** markets and emit one
row per runner with:

    * ``bsp``            — Betfair Starting Price (the sharp "fair" closing price)
    * ``ltp_preoff``     — last traded price at/just before the off
    * ``ltp_60s``/``300s`` — last trade 1 min / 5 min before the off
    * ``won``            — settled WINNER flag
    * merge keys (``track_key``/``off_key``/``horse_key`` + ``race_date``) so the
      result joins straight onto ``data/races.db`` via ``src/utils.py`` keys.

The folder layout is *not* one meeting per folder (a single eventId can mix
tracks/days), so every market is parsed purely from its own
``marketDefinition`` (``eventName``/``marketTime``), never the path.

Usage:
    # Full rebuild from every tar:
    python scripts/extract_betfair_bsp.py --tar "data/betfair/2025.tar" \
        --tar "data/betfair/2026.tar" --out data/processed/betfair_bsp.parquet --validate

    # Incremental: process ONLY the new tar and merge it into the existing
    # parquet (seconds, not a full rebuild):
    python scripts/extract_betfair_bsp.py --tar "data/betfair/2022-06to07.tar" \
        --out data/processed/betfair_bsp.parquet --append --validate
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import sys
import tarfile
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import bf_horse_key  # noqa: E402
from src.utils import (  # noqa: E402
    normalise_off_time_key,
    normalise_track_key,
)

KEEP_COUNTRIES = {"GB", "IE"}
MARKET_FILE_RE = re.compile(r"(?:^|/)1\.\d+\.bz2$")
# "Nottingham 1st Jun" -> "Nottingham"; "Chelmsford City 1st Jun" -> "Chelmsford City".
# Month is optional so a malformed "Newton Abbot 16th" (no month) still strips.
TRACK_DATE_RE = re.compile(r"\s+\d{1,2}(?:st|nd|rd|th)(?:\s+\w{3,})?\s*$", re.IGNORECASE)
COUNTRY_SUFFIX_RE = re.compile(r"\s*\([A-Z]{2,3}\)\s*$")
CLOTH_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")


def _track_from_event(event_name: str) -> str:
    return TRACK_DATE_RE.sub("", event_name or "").strip()


def _clean_horse(name: str) -> str:
    name = CLOTH_PREFIX_RE.sub("", name or "")
    name = COUNTRY_SUFFIX_RE.sub("", name)
    return name.strip()


def _iso_to_epoch_ms(iso: str) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return int(dt.timestamp() * 1000)


def parse_market(stream: bytes) -> list[dict] | None:
    """Parse one market's mcm stream -> list of per-runner rows (or None to skip)."""
    final_def: dict | None = None
    names: dict[int, str] = {}
    # runner_id -> list[(pt_ms, ltp)] kept only as a running "last trade" tracker
    last_ltp: dict[int, float] = {}
    ltp_60: dict[int, float] = {}
    ltp_300: dict[int, float] = {}
    ltp_pre: dict[int, float] = {}
    off_ms: int | None = None

    # First pass to find off-time (marketTime) — appears in every marketDefinition.
    # We parse line-by-line, updating definition and prices together; because the
    # marketTime is present from the first definition, the windowed cutoffs work.
    lines = stream.split(b"\n")
    parsed = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    # Establish off-time and the final settled definition.
    for d in parsed:
        for mc in d.get("mc", []):
            md = mc.get("marketDefinition")
            if md:
                final_def = md
                if off_ms is None:
                    off_ms = _iso_to_epoch_ms(md.get("marketTime", ""))

    if not final_def:
        return None
    if final_def.get("marketType") != "WIN":
        return None
    if final_def.get("countryCode") not in KEEP_COUNTRIES:
        return None
    if final_def.get("eventTypeId") != "7":  # horse racing
        return None

    for r in final_def.get("runners", []):
        names[r["id"]] = r.get("name", "")

    cut_60 = (off_ms - 60_000) if off_ms else None
    cut_300 = (off_ms - 300_000) if off_ms else None

    for d in parsed:
        pt = d.get("pt")
        for mc in d.get("mc", []):
            for rc in mc.get("rc", []):
                if "ltp" not in rc:
                    continue
                rid = rc["id"]
                ltp = rc["ltp"]
                last_ltp[rid] = ltp
                if off_ms is not None and pt is not None and pt <= off_ms:
                    ltp_pre[rid] = ltp
                if cut_60 is not None and pt is not None and pt <= cut_60:
                    ltp_60[rid] = ltp
                if cut_300 is not None and pt is not None and pt <= cut_300:
                    ltp_300[rid] = ltp

    event_name = final_def.get("eventName", "")
    track = _track_from_event(event_name)
    market_time = final_def.get("marketTime", "")
    race_date = market_time[:10] if market_time else ""
    off_hhmm = market_time[11:16] if len(market_time) >= 16 else ""
    n_active = sum(
        1 for r in final_def.get("runners", []) if r.get("status") in ("WINNER", "LOSER")
    )

    rows = []
    for r in final_def.get("runners", []):
        status = r.get("status")
        if status == "REMOVED":
            continue
        rid = r["id"]
        raw_name = names.get(rid, r.get("name", ""))
        horse = _clean_horse(raw_name)
        rows.append(
            {
                "race_date": race_date,
                "off_time": off_hhmm,
                "track": track,
                "horse_name": horse,
                "bf_runner_id": rid,
                "bsp": r.get("bsp"),
                "ltp_preoff": ltp_pre.get(rid),
                "ltp_60s": ltp_60.get(rid),
                "ltp_300s": ltp_300.get(rid),
                "ltp_last": last_ltp.get(rid),
                "won": 1 if status == "WINNER" else 0,
                "n_runners": n_active,
                "market_id": final_def.get("eventId", ""),
                "track_key": normalise_track_key(track),
                "off_key": normalise_off_time_key(off_hhmm),
                "horse_key": bf_horse_key(horse),
            }
        )
    return rows


def process_tar(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    n_files = n_markets = n_bad = 0
    t0 = time.time()
    # These libarchive-written tars end with a malformed/short trailing block
    # that makes Python's tarfile raise mid-iteration. Verified harmless: the
    # ReadError fires AFTER the last market file (the per-tar market-file counts
    # reconcile exactly with system `tar`), so no market data is lost. The
    # ignore_zeros + graceful iterator-stop below are belt-and-braces.
    tf = tarfile.open(path, "r:", ignore_zeros=True)
    try:
        member_iter = iter(tf)
        while True:
            try:
                member = next(member_iter)
            except StopIteration:
                break
            except Exception as exc:  # corrupt header mid-stream -> stop this tar
                print(f"  {os.path.basename(path)}: iterator stopped at "
                      f"{n_files} files: {exc!r}", flush=True)
                break
            if not member.isfile():
                continue
            if not MARKET_FILE_RE.search(member.name):
                continue
            n_files += 1
            try:
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                data = bz2.decompress(fh.read())
            except Exception:  # bad/truncated member -> skip, keep going
                n_bad += 1
                continue
            try:
                out = parse_market(data)
            except Exception:
                n_bad += 1
                continue
            if out:
                n_markets += 1
                rows.extend(out)
            if n_files % 5000 == 0:
                print(
                    f"  {os.path.basename(path)}: {n_files} market files, "
                    f"{n_markets} GB/IE WIN markets, {len(rows)} runners "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            if limit and n_files >= limit:
                break
    finally:
        tf.close()
    print(
        f"  {os.path.basename(path)}: DONE {n_files} files -> "
        f"{n_markets} markets, {len(rows)} runners, {n_bad} skipped "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return rows


def validate(df: pd.DataFrame) -> None:
    import sqlite3

    db = os.path.join("data", "races.db")
    if not os.path.exists(db):
        print("validate: data/races.db not found", flush=True)
        return
    con = sqlite3.connect(db)
    res = pd.read_sql_query(
        "SELECT race_date, off_time, track, horse_name, odds, won FROM results "
        "WHERE race_date >= ?",
        con,
        params=[df["race_date"].min()],
    )
    con.close()
    res["track_key"] = res["track"].map(normalise_track_key)
    res["off_key"] = res["off_time"].map(normalise_off_time_key)
    res["horse_key"] = res["horse_name"].map(bf_horse_key)

    # Betfair-side coverage: of BF runners whose race exists in the DB, how
    # many join to a DB horse row. (Irish-only races absent from the GB DB are
    # excluded from the denominator — they can never match.)
    db_races = set(zip(res["race_date"], res["track_key"], res["off_key"]))
    bf = df.copy()
    bf["_race_in_db"] = [
        (d, t, o) in db_races
        for d, t, o in zip(bf["race_date"], bf["track_key"], bf["off_key"])
    ]
    bf_indb = bf[bf["_race_in_db"]]
    bf_m = bf_indb.merge(
        res[["race_date", "track_key", "off_key", "horse_key", "won"]].assign(_db=1),
        on=["race_date", "track_key", "off_key", "horse_key"],
        how="left",
    )
    print(f"\n=== VALIDATION (period {df['race_date'].min()}..{df['race_date'].max()}) ===", flush=True)
    print(f"Betfair runners                  : {len(bf):,}", flush=True)
    print(f"  whose race is in the GB DB     : {len(bf_indb):,}", flush=True)
    print(f"  of those, matched to DB horse  : {int(bf_m['_db'].notna().sum()):,}  "
          f"({bf_m['_db'].notna().mean():.1%})", flush=True)

    merged = res.merge(
        df[["race_date", "track_key", "off_key", "horse_key", "bsp", "ltp_preoff", "won"]],
        on=["race_date", "track_key", "off_key", "horse_key"],
        how="left",
        suffixes=("_db", "_bf"),
    )
    both = merged.dropna(subset=["bsp", "odds"])
    if len(both):
        agree = (both["won_db"] == both["won_bf"]).mean()
        print(f"winner agreement (DB vs BF): {agree:.3%}  on {len(both):,} rows", flush=True)
        # SP vs BSP value: do they line up?
        bsp = pd.to_numeric(both["bsp"], errors="coerce")
        sp = pd.to_numeric(both["odds"], errors="coerce")
        ok = bsp.notna() & sp.notna() & (bsp > 1) & (sp > 1)
        if ok.sum():
            corr = bsp[ok].rank().corr(sp[ok].rank())
            print(f"BSP vs SP rank corr      : {corr:.3f}  on {ok.sum():,} rows", flush=True)
            print(f"median BSP={bsp[ok].median():.2f}  median SP={sp[ok].median():.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="append", required=True, help="tar path (repeatable)")
    ap.add_argument("--out", default="data/processed/betfair_bsp.parquet")
    ap.add_argument("--limit", type=int, default=None, help="cap market files per tar (debug)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="merge the given tar(s) into the existing --out parquet "
                         "instead of rebuilding from scratch (process only new files)")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for tar in args.tar:
        if not os.path.exists(tar):
            print(f"missing: {tar}", flush=True)
            continue
        print(f"processing {tar} ...", flush=True)
        all_rows.extend(process_tar(tar, limit=args.limit))

    if not all_rows:
        print("no rows extracted", flush=True)
        return

    df = pd.DataFrame(all_rows)
    # Older (2022) streams occasionally encode prices as strings; coerce so the
    # sort/dedup and downstream maths stay numeric.
    for _c in ("bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_last"):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
    # Incremental update: merge into the existing parquet instead of rebuilding
    # from every tar. The same race's BSP is identical across downloads, so the
    # dedup below collapses overlaps — only the new tar(s) need processing.
    if args.append and os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        print(f"append: merging {len(df):,} new rows into existing {len(prev):,}",
              flush=True)
        df = pd.concat([prev, df], ignore_index=True)
    # Drop exact dup runners (a market can appear in >1 archive); keep best-priced.
    df = df.sort_values("bsp", na_position="last").drop_duplicates(
        subset=["race_date", "track_key", "off_key", "horse_key"], keep="first"
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {len(df):,} runner rows -> {args.out}", flush=True)
    print(
        f"races: {df.groupby(['race_date', 'track_key', 'off_key']).ngroups:,} | "
        f"date range {df['race_date'].min()}..{df['race_date'].max()} | "
        f"bsp present {df['bsp'].notna().mean():.1%} | "
        f"ltp_preoff present {df['ltp_preoff'].notna().mean():.1%}",
        flush=True,
    )
    if args.validate:
        validate(df)


if __name__ == "__main__":
    main()
