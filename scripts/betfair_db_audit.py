"""
Two-directional integrity check between Betfair and ``data/races.db``.

A) Betfair-present / DB-absent  -> the SCRAPER missed a race (backfill worklist,
   ``missing_races.csv``). Only races on DB-known tracks are flagged.

B) DB-present / Betfair-absent, scoped to meetings Betfair already covers
   (same date+track) -> a BETFAIR gap or a join/alias problem
   (``db_races_not_in_betfair.csv``). Scoping to covered meetings avoids
   false-flagging the truncated back-half of each year and tracks Betfair
   never carries (Galway/Laytown). Should be ~0 when joins are clean.

Usage:
    python scripts/betfair_db_audit.py \
        --bsp data/processed/betfair_bsp.parquet \
        --out data/processed/missing_races.csv
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import bf_track_key  # noqa: E402
from src.utils import normalise_off_time_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsp", default="data/processed/betfair_bsp.parquet")
    ap.add_argument("--db", default="data/races.db")
    ap.add_argument("--out", default="data/processed/missing_races.csv")
    args = ap.parse_args()

    bf = pd.read_parquet(args.bsp)
    con = sqlite3.connect(args.db)
    res = pd.read_sql_query(
        "SELECT race_date, track, off_time FROM results WHERE race_date >= ?",
        con,
        params=[bf["race_date"].min()],
    )
    con.close()

    res["track_key"] = res["track"].map(bf_track_key)
    res["off_key"] = res["off_time"].map(normalise_off_time_key)
    db_race_keys = set(zip(res["race_date"], res["track_key"], res["off_key"]))
    db_tracks = set(res["track_key"].unique())

    # one row per Betfair race
    bf_races = (
        bf.assign(track_key=bf["track"].map(bf_track_key)).groupby(["race_date", "track", "track_key", "off_key", "off_time"])
        .agg(n_runners=("horse_name", "size"),
             winner=("won", lambda s: bf.loc[s.index][s.eq(1)]["horse_name"].head(1).squeeze()
                     if s.eq(1).any() else ""))
        .reset_index()
    )
    bf_races["key"] = list(zip(bf_races["race_date"], bf_races["track_key"], bf_races["off_key"]))
    bf_races["in_db"] = bf_races["key"].isin(db_race_keys)
    bf_races["track_known"] = bf_races["track_key"].isin(db_tracks)

    missing = bf_races[(~bf_races["in_db"]) & (bf_races["track_known"])].copy()
    untracked = bf_races[~bf_races["track_known"]]

    print(f"Betfair races (GB/IE WIN)     : {len(bf_races):,}")
    print(f"  present in DB               : {int(bf_races['in_db'].sum()):,} "
          f"({bf_races['in_db'].mean():.1%})")
    print(f"  MISSING (DB-known track)    : {len(missing):,}")
    print(f"  on tracks the DB never has  : {len(untracked):,} "
          f"(tracks: {sorted(untracked['track'].unique())})")

    print("\nMissing races by track:")
    by_track = missing.groupby("track").size().sort_values(ascending=False)
    for t, n in by_track.items():
        print(f"  {t:<16} {n:>4}")

    print("\nMissing races by month:")
    missing["month"] = missing["race_date"].str[:7]
    for mth, n in missing.groupby("month").size().items():
        print(f"  {mth}  {n:>4}")

    out_cols = ["race_date", "track", "off_time", "n_runners", "winner"]
    missing.sort_values(["race_date", "track", "off_time"])[out_cols].to_csv(
        args.out, index=False
    )
    print(f"\nwrote {len(missing):,} missing races -> {args.out}")

    # Are whole meetings missing, or partial cards? (informs scraper diagnosis)
    db_meetings = set(zip(res["race_date"], res["track_key"]))
    missing["meeting"] = list(zip(missing["race_date"], missing["track_key"]))
    whole = missing[~missing["meeting"].isin(db_meetings)]
    partial = missing[missing["meeting"].isin(db_meetings)]
    print(f"\nof missing races: {len(whole):,} are in meetings ENTIRELY absent from DB, "
          f"{len(partial):,} are missing races from PARTIALLY-scraped meetings")

    # ── Direction B: DB races Betfair lacks, within meetings Betfair covers ──
    bf_race_keys = set(bf_races["key"])
    bf_meetings = set(zip(bf_races["race_date"], bf_races["track_key"]))
    res = res.copy()
    res["key"] = list(zip(res["race_date"], res["track_key"], res["off_key"]))
    res["meeting"] = list(zip(res["race_date"], res["track_key"]))
    # DB races whose meeting Betfair covers, but the specific race is absent.
    db_gap = res[res["meeting"].isin(bf_meetings) & ~res["key"].isin(bf_race_keys)]
    db_gap = db_gap.drop_duplicates(subset=["race_date", "track_key", "off_key"])

    print("\n" + "=" * 60)
    print("Direction B — DB races MISSING from Betfair (within covered meetings):")
    print(f"  flagged: {len(db_gap):,}  (Betfair gap or unfixed join/alias)")
    if len(db_gap):
        print("  by track:")
        for t, n in db_gap.groupby("track").size().sort_values(ascending=False).head(20).items():
            print(f"    {t:<16} {n:>4}")
        db_gap.sort_values(["race_date", "track", "off_time"])[
            ["race_date", "track", "off_time"]
        ].to_csv("data/processed/db_races_not_in_betfair.csv", index=False)
        print("  wrote data/processed/db_races_not_in_betfair.csv")

    # Meetings on Betfair-covered DATES that Betfair lacks entirely (e.g. tracks
    # Betfair never carries — informational, usually not actionable).
    bf_dates = set(bf_races["race_date"])
    db_on_cov = res[res["race_date"].isin(bf_dates)]
    absent_meetings = sorted(set(db_on_cov["meeting"]) - bf_meetings)
    if absent_meetings:
        from collections import Counter
        trk = Counter(t for _d, t in absent_meetings)
        print(f"\n  (info) {len(absent_meetings)} DB meetings on Betfair-covered dates "
              f"are absent from Betfair entirely — top tracks: {trk.most_common(8)}")


if __name__ == "__main__":
    main()
