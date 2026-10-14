"""Exploratory diagnostics for ledger 20's descriptive lead. NOT a test.

Ledger 20's pre-registered test failed, but the policy's early strikes, read
as bets of their own, beat the close (median CLV ~1.016 with a one-tick
haircut and non-runner reductions). Before anyone treats that as a lead,
check the ways it could be an artefact of the LTP being a poor proxy for an
available back price:

* **staleness** - an LTP is the last *change*; a stale one may sit above
  where the book has since moved, so it was never available;
* **spread** - in thin markets the back/lay spread is wider than one tick;
* **where the strikes come from** - 11:00 (thin) versus T-5 (liquid).

Same walk-forward predictions as ``execution_timing.py`` (deterministic);
results are labelled exploratory and only a pre-registered hold-out test
(BASIC from 2026-06-11) could promote anything found here.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.execution_timing import CUTS, TAU, frame, policy, strike_price, universes, walk_forward
from src.bet_settlement import settle_win_exchange_bets
from src.wf_eval import bootstrap_median_ci, bootstrap_roi_ci, clv_summary

PREDS = os.path.join("data", "audit", "ledger20_preds.parquet")


def get_preds(d: pd.DataFrame, unis: dict) -> pd.DataFrame:
    if os.path.exists(PREDS):
        p = pd.read_parquet(PREDS)
        if len(p) == len(d):
            return p
    preds, _ = walk_forward(d, train_mask=unis["U1 all in band"])
    preds.to_parquet(PREDS)
    return preds


def cell(s: pd.DataFrame, taken: np.ndarray, n_boot: int) -> str:
    if len(s) < 30:
        return f"n={len(s)}"
    cl = s["race_id"].to_numpy()
    won = s["won"].to_numpy().astype(bool)
    m = bootstrap_median_ci(taken / s["bsp"].to_numpy(float), clusters=cl, n_boot=n_boot)
    c = clv_summary(taken, s["bsp"])
    r = bootstrap_roi_ci(settle_win_exchange_bets(taken, won), clusters=cl, n_boot=n_boot)
    return (f"n={len(s):6,} medCLV {m['estimate']:.4f} [{m['ci_low']:.4f},{m['ci_high']:.4f}] "
            f"geo {c['geometric_mean_clv']:.4f} beat {c['pct_beat_close']:.3f} "
            f"ROI {r['mean']:+.2f}% [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]")


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-morning", action="store_true",
                    help="refit the 11:00 model without the morning fields instead")
    ap.add_argument("--bounce", action="store_true",
                    help="split early strikes by the direction of the last LTP change")
    args = ap.parse_args()
    if args.bounce:
        bounce(args.n_boot)
        return 0
    if args.no_morning:
        no_morning(args.n_boot)
        return 0
    d = frame()
    unis = universes(d)
    preds = get_preds(d, unis)
    strike = policy(preds, TAU)
    u1 = unis["U1 all in band"] & preds["fold"].notna().to_numpy()
    print("EXPLORATORY - not a test. U1 early strikes, by decision point and haircut:")
    for i, cut in enumerate(CUTS):
        m = u1 & (strike == i)
        s = d[m]
        for label, kw in (("1 tick", dict(ticks=1)), ("2 ticks", dict(ticks=2)), ("no haircut", dict(ticks=0))):
            print(f"  {cut:6s} {label:10s} {cell(s, strike_price(s, cut, **kw), args.n_boot)}")
        age = s[f"age_{cut}"].to_numpy(float)
        all_age = d.loc[u1, f"age_{cut}"].to_numpy(float)
        print(f"         LTP age median {np.nanmedian(age):.0f}s (all U1 at this cut: {np.nanmedian(all_age):.0f}s); "
              f"changes so far median {np.nanmedian(s[f'nchg_{cut}']):.0f} "
              f"(all: {np.nanmedian(d.loc[u1, f'nchg_{cut}']):.0f})")
        for lo, hi in ((0, 60), (60, 600), (600, 1e9)):
            mm = (age >= lo) & (age < hi)
            print(f"         age {lo:>4.0f}-{hi:<9.0f}s 1 tick: {cell(s[mm], strike_price(s[mm], cut), args.n_boot)}")
    return 0



MORNING_FEATURES = ("excess_vol", "log_vol", "log_race_vol", "move_mwap")


def no_morning(n_boot: int) -> None:
    """Refit the 11:00 model without the public files' morning fields.

    Betfair does not document when its "morning" ends, and the race-level
    morning aggregates are normalised over the final field, which can betray
    a later withdrawal. A model that never sees them is immune to both.
    """
    from scripts.execution_timing import (_lgbm_fit_predict, _ols_fit_predict, features,
                                          MIN_TRAIN_QUARTERS)
    from src.wf_eval import expanding_period_folds
    d = frame()
    unis = universes(d)
    band = unis["U1 all in band"]
    X = features(d, "1100").drop(columns=list(MORNING_FEATURES)).to_numpy(float)
    y = np.log(strike_price(d, "1100") / d["bsp"].to_numpy(float))
    folds = expanding_period_folds(d["race_date"], freq="Q", min_train_periods=MIN_TRAIN_QUARTERS)
    yhat = {m: np.full(len(d), np.nan) for m in ("ols", "lgbm")}
    for f in folds:
        tr = f.train & band
        yhat["ols"][f.test] = _ols_fit_predict(X[tr], y[tr], X[f.test])
        yhat["lgbm"][f.test] = _lgbm_fit_predict(X[tr], y[tr], X[f.test])
    tested = np.isfinite(yhat["ols"])
    later_rf = (d["rf_after_1100"].fillna(1.0) < 1.0).to_numpy()
    print("\nEXPLORATORY - 11:00 model WITHOUT morning-volume/VWAP features, 11:00 strikes alone (tau 0.01):")
    for m in ("ols", "lgbm"):
        ok = tested & band
        r2 = 1 - np.mean((y[ok] - yhat[m][ok]) ** 2) / np.var(y[ok])
        print(f"  {m}: OOF R2 {r2:.4f}")
        for uname, umask in unis.items():
            hit = umask & tested & (yhat[m] > TAU)
            s = d[hit]
            print(f"    {uname:24s} 1 tick  {cell(s, strike_price(s, '1100'), n_boot)}")
            print(f"    {'':24s} 2 ticks {cell(s, strike_price(s, '1100', ticks=2), n_boot)}")
        hit = band & tested & (yhat[m] > TAU)
        print(f"    strike rate where a runner is later withdrawn {hit[later_rf & band & tested].mean():.4f} "
              f"vs not {hit[~later_rf & band & tested].mean():.4f}")
        for lab, mm in (("no later withdrawal", hit & ~later_rf), ("later withdrawal", hit & later_rf)):
            s = d[mm]
            print(f"    U1 {lab:20s} 1 tick  {cell(s, strike_price(s, '1100'), n_boot)}")


def bounce(n_boot: int) -> None:
    """Split the policy's early strikes by the direction of the last LTP change.

    If the last change before the cut was a *fall*, a backer had just taken
    that price, so it was there to back; if a *rise*, a layer may have matched
    at the top of the spread and the best back price sat below the LTP. A
    selection edge that is really bid-ask bounce lives in the rises.
    """
    from src.betfair import attach_bsp
    d = frame()
    extra = tuple(f"prev_ltp_{c}" for c in CUTS)
    d = attach_bsp(d, pd.read_parquet(os.path.join("data", "processed", "betfair_timing.parquet")), cols=extra)
    unis = universes(d)
    preds = get_preds(d, unis)
    strike = policy(preds, TAU)
    u1 = unis["U1 all in band"] & preds["fold"].notna().to_numpy()
    print("\nEXPLORATORY - early strikes by the direction of the last LTP change before the cut:")
    for i, cut in enumerate(CUTS):
        prev = pd.to_numeric(d[f"prev_ltp_{cut}"], errors="coerce").to_numpy(float)
        cur = d[f"ltp_{cut}"].to_numpy(float)
        direction = np.where(~np.isfinite(prev), "none", np.where(cur < prev, "fell", "rose"))
        m = u1 & (strike == i)
        base = u1
        print(f"  {cut}: strikes fell {np.mean(direction[m] == 'fell'):.3f} rose {np.mean(direction[m] == 'rose'):.3f} "
              f"(all U1: fell {np.mean(direction[base] == 'fell'):.3f} rose {np.mean(direction[base] == 'rose'):.3f})")
        for lab in ("fell", "rose"):
            mm = m & (direction == lab)
            s = d[mm]
            for hl, kw in (("no haircut", dict(ticks=0)), ("1 tick", dict(ticks=1))):
                print(f"    last change {lab:4s} {hl:10s} {cell(s, strike_price(s, cut, **kw), n_boot)}")


if __name__ == "__main__":
    raise SystemExit(main())
