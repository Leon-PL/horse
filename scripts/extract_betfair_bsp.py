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
import os
import sys
import tarfile
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import bf_horse_key  # noqa: E402
from src.ingestion.betfair_streams import MARKET_FILE_RE, parse_market  # noqa: E402
from src.utils import normalise_off_time_key, normalise_track_key  # noqa: E402


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


def _process_tar_frame(path: str, limit: int | None = None) -> pd.DataFrame:
    """One tar -> DataFrame (a frame pickles far smaller than a list of dicts)."""
    print(f"processing {path} ...", flush=True)
    return pd.DataFrame(process_tar(path, limit=limit))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", action="append", required=True, help="tar path (repeatable)")
    ap.add_argument("--out", default="data/processed/betfair_bsp.parquet")
    ap.add_argument("--limit", type=int, default=None, help="cap market files per tar (debug)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="merge the given tar(s) into the existing --out parquet "
                         "instead of rebuilding from scratch (process only new files)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="process this many tars in parallel (one process each)")
    args = ap.parse_args()

    tars = [t for t in args.tar if os.path.exists(t)]
    for t in set(args.tar) - set(tars):
        print(f"missing: {t}", flush=True)
    frames: list[pd.DataFrame] = []
    if args.jobs > 1 and len(tars) > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for rows in pool.map(partial(_process_tar_frame, limit=args.limit), tars):
                frames.append(rows)
    else:
        for tar in tars:
            frames.append(_process_tar_frame(tar, limit=args.limit))
    frames = [f for f in frames if not f.empty]

    if not frames:
        print("no rows extracted", flush=True)
        return

    df = pd.concat(frames, ignore_index=True)
    # Older (2022) streams occasionally encode prices as strings; coerce so the
    # sort/dedup and downstream maths stay numeric.
    for _c in ("bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_3600s", "ltp_1100", "ltp_last",
               "ltp_eve1800", "ltp_0800"):
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
