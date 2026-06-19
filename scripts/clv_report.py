"""
CLV / exchange-settlement harness
=================================

Answers the question SP-based backtests cannot: does the form model have a real
edge at a *bettable* price? It joins Betfair BSP + pre-off traded prices onto a
run's ``bets.csv`` and reports, per strategy:

  (A) Realised ROI settled at **BSP** (after commission) vs at bookmaker **SP**.
      BSP is a price you can actually get matched at, so this is the honest
      bottom line. SP and BSP are different markets (bookies hold more margin),
      so do NOT read SP->BSP as "CLV" — (A) is an absolute settlement, not a
      cross-market comparison.

  (B) Closing-line drift of the selections: ltp_preoff / ltp_300s. <1 means the
      market shortened the horse in the last 5 min (smart money agreed).

  (C) Probability edge on the selections: log-loss / Brier of model_prob vs
      BSP-implied prob, and how often the model's prob beat BSP's on the winner.

Bet selections, strategies and stakes come straight from ``bets.csv`` (written
by the run using ``src/bet_settlement.py``), so the settlement maths is shared.

Usage:
    python scripts/clv_report.py                      # latest run
    python scripts/clv_report.py --run data/runs/20260614_213017 --commission 0.02
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.betfair import (  # noqa: E402
    attach_bsp,
    bf_horse_key,
    bf_track_key,
    bsp_implied_prob,
    load_bsp,
)
from src.utils import normalise_off_time_key as _ok  # noqa: E402


def _latest_run() -> str:
    runs = sorted(glob.glob("data/runs/*/"), reverse=True)
    for r in runs:
        if os.path.exists(os.path.join(r, "bets.csv")):
            return r.rstrip("/\\")
    raise SystemExit("no run with bets.csv found")


def _off_time_by_race_id(race_ids, db="data/races.db") -> dict:
    con = sqlite3.connect(db)
    q = "SELECT DISTINCT race_id, off_time FROM results"
    m = pd.read_sql_query(q, con)
    con.close()
    m["race_id"] = m["race_id"].astype(str)
    return dict(zip(m["race_id"], m["off_time"]))


def _boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype="float64")
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: latest with bets.csv)")
    ap.add_argument("--bsp", default="data/processed/betfair_bsp.parquet")
    ap.add_argument("--commission", type=float, default=0.02,
                    help="exchange commission on net winnings (default 0.02)")
    args = ap.parse_args()

    run = args.run or _latest_run()
    bets = pd.read_csv(os.path.join(run, "bets.csv"))
    bets["race_id"] = bets["race_id"].astype(str)
    print(f"run: {run}  |  bets: {len(bets):,}  |  strategies: "
          f"{sorted(bets['strategy'].unique())}")

    bsp = load_bsp(args.bsp)
    if bsp is None:
        raise SystemExit(f"BSP parquet not found: {args.bsp}")

    # bets.csv lacks off_time; bridge it from the DB by race_id for a clean
    # date/track/off/horse join.
    off_map = _off_time_by_race_id(bets["race_id"].unique().tolist())
    bets["off_time"] = bets["race_id"].map(off_map).fillna("")

    bets = attach_bsp(bets, bsp)
    cov = bets["bsp"].notna().mean()
    print(f"BSP coverage of bets: {bets['bsp'].notna().sum():,}/{len(bets):,} ({cov:.1%})  "
          f"(date range {bets['race_date'].min()}..{bets['race_date'].max()})")

    m = bets[bets["bsp"].notna()].copy()
    m["won"] = m["won"].astype(int)
    comm = args.commission

    # ---- no-skill baselines settled at BSP, over the SAME races as the bets ----
    # Random-runner and favourite (shortest BSP) returns at BSP define the
    # bar the model must clear; an efficient market pays ~ -overround-comm.
    bsp = bsp.copy()
    bsp["rkey"] = list(zip(bsp["race_date"], bsp["track_key"], bsp["off_key"]))
    # m has no track_key column; rebuild from raw for the race set
    m_rkey = list(zip(m["race_date"], m["track"].map(bf_track_key), m["off_time"].map(_ok)))
    bet_race_set = set(m_rkey)
    fld = bsp[bsp["rkey"].isin(bet_race_set)].copy()
    fld["won"] = fld["won"].astype(int)

    def _roi_at_bsp(d):
        return np.where(d["won"] == 1, (d["bsp"] - 1.0) * (1 - comm), -1.0).mean()

    rand_roi = _roi_at_bsp(fld)
    fav = fld.loc[fld.groupby("rkey")["bsp"].idxmin()]
    fav_roi = _roi_at_bsp(fav)
    print(f"\n=== no-skill baselines at BSP (same {fld['rkey'].nunique():,} races) ===")
    print(f"  back EVERY runner   : ROI {rand_roi:+.1%}  (efficient-market floor)")
    print(f"  back the FAVOURITE   : ROI {fav_roi:+.1%}  (n={len(fav):,} races)")

    # ---- (A) realised ROI at SP vs BSP ----
    def settle(odds, won, stake):
        gross = np.where(won == 1, (odds - 1.0) * (1.0 - comm) * stake, -stake)
        return gross

    print("\n=== (A) realised ROI — SP vs BSP (after %.0f%% commission on BSP wins) ==="
          % (comm * 100))
    print(f"{'strategy':<12}{'n':>6}{'ROI@SP':>10}{'ROI@BSP':>10}{'BSP 95% CI':>22}")
    for strat in ["top_pick", "value", "each_way", "ALL"]:
        d = m if strat == "ALL" else m[m["strategy"] == strat]
        if len(d) == 0:
            continue
        stake = d["stake"].to_numpy()
        won_a = d["won"].to_numpy()
        # bookmaker SP: no exchange commission
        sp_pnl = np.where(won_a == 1, (d["odds"].to_numpy() - 1.0) * stake, -stake)
        bsp_pnl = settle(d["bsp"].to_numpy(), d["won"].to_numpy(), stake)
        roi_sp = sp_pnl.sum() / stake.sum()
        roi_bsp = bsp_pnl.sum() / stake.sum()
        lo, hi = _boot_ci(bsp_pnl / stake.mean())
        print(f"{strat:<12}{len(d):>6}{roi_sp:>9.1%}{roi_bsp:>10.1%}"
              f"{f'[{lo:+.1%}, {hi:+.1%}]':>22}")

    # ---- (B) closing-line drift of selections ----
    drift = m["ltp_preoff"] / m["ltp_300s"]
    drift = drift.replace([np.inf, -np.inf], np.nan).dropna()
    if len(drift):
        print("\n=== (B) last-5-min drift of selections (ltp_preoff / ltp_300s) ===")
        print(f"  median {drift.median():.3f}  mean {drift.mean():.3f}  "
              f"(<1 = market shortened them; n={len(drift):,})")
        print(f"  shortened: {(drift < 1).mean():.1%}   drifted: {(drift > 1).mean():.1%}")

    # ---- (C) probability edge vs BSP on the selections ----
    # BSP-implied prob must be normalised over the FULL race field, then looked
    # up for the selected horses (normalising selections alone -> prob 1.0).
    bsp["bsp_ip"] = bsp_implied_prob(bsp["bsp"].to_numpy(), bsp["rkey"].to_numpy())
    lut = bsp.set_index(["race_date", "track_key", "off_key", "horse_key"])["bsp_ip"]
    m_idx = pd.MultiIndex.from_arrays(
        [m["race_date"], m["track"].map(bf_track_key), m["off_time"].map(_ok),
         m["horse_name"].map(bf_horse_key)]
    )
    bp = lut.reindex(m_idx).to_numpy()
    ok = np.isfinite(bp) & m["model_prob"].notna().to_numpy()
    if ok.sum():
        won = m["won"].to_numpy()[ok]
        mp = np.clip(m["model_prob"].to_numpy()[ok], 1e-6, 1 - 1e-6)
        bpc = np.clip(bp[ok], 1e-6, 1 - 1e-6)
        ll_model = -(won * np.log(mp) + (1 - won) * np.log(1 - mp)).mean()
        ll_bsp = -(won * np.log(bpc) + (1 - won) * np.log(1 - bpc)).mean()
        br_model = ((mp - won) ** 2).mean()
        br_bsp = ((bpc - won) ** 2).mean()
        print("\n=== (C) model vs BSP on the SELECTED bets (lower = better) ===")
        print(f"  log-loss : model {ll_model:.4f}   BSP {ll_bsp:.4f}   "
              f"{'model better' if ll_model < ll_bsp else 'BSP better'}")
        print(f"  brier    : model {br_model:.4f}   BSP {br_bsp:.4f}")
        print(f"  model_prob > BSP_prob on these picks: {(mp > bpc).mean():.1%}")

    print("\nNote: SP and BSP are different markets; (A) is absolute settlement, not "
          "cross-market CLV. BSP coverage starts 2025-01 — earlier bets are excluded.")


if __name__ == "__main__":
    main()
