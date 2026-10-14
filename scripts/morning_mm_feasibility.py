"""EXPLORATION_PLAN E, phase 1: could a passive morning market-maker earn the spread?

Feasibility only, free data. A market-maker posting offers on both sides is
filled by whoever crosses the spread. Matched volume is zero-sum at the
traded price: per GBP of backer stake matched at price P on a runner whose
fair (overround-free BSP) probability is p, the backer's expected value is
``P*p - 1`` and the layer's is ``1 - P*p`` (before commission). So the
*markout* of a fill is known from the price, the close and which side the
market-maker was on. The public files give morning VWAP and volume but not
who was the aggressor, so two bounds are reported:

* **informed side** - runners where the morning money is concentrated
  (ledger 14's excess volume): assume the aggressor backed, so the
  market-maker laid;
* **the rest** - assume balanced flow: the market-maker lays half and backs
  half, and its markout is the spread plus the drift it cannot control.

Outputs, by price band and volume-weighted: markout per GBP matched for a lay
and a back filled at ``morningwap`` and at ``ltp_1100`` (with later non-runner
reductions applied to the morning price), the break-even half-spread in
Betfair ticks, and capacity (morning matched volume per runner and per day).
What BASIC cannot show: the ladder, queue position, how often a passive offer
is filled at all, and who crossed. Development period only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.microstructure_screen import load
from src.bet_settlement import BETFAIR_TICK_BANDS, EXCHANGE_COMMISSION, apply_reduction_factor
from src.betfair import attach_bsp
from src.wf_eval import bootstrap_roi_ci

TIMING = os.path.join("data", "processed", "betfair_timing.parquet")
BANDS = [(1.01, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 13.0), (13.0, 21.0), (21.0, 50.0), (50.0, 1000.0)]


def tick_size(price: np.ndarray) -> np.ndarray:
    uppers = np.array([b for b, _ in BETFAIR_TICK_BANDS])
    incs = np.array([i for _, i in BETFAIR_TICK_BANDS])
    return incs[np.clip(np.searchsorted(uppers, price - 1e-9), 0, len(incs) - 1)]


def frame() -> pd.DataFrame:
    df = load().drop(columns=["ltp_300s", "p_t5", "move_m2t5", "move_t52c"], errors="ignore")
    df = attach_bsp(df, pd.read_parquet(TIMING), cols=("ltp_1100", "rf_after_1100", "off_ms"))
    for c in ("ltp_1100", "rf_after_1100"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    off = pd.to_datetime(df["race_date"].astype(str) + " " + df["off_time"].astype(str), errors="coerce")
    local = off.dt.tz_localize("UTC").dt.tz_convert("Europe/London")
    # Races after the morning closes, so "morning" volume is all pre-11:00.
    df = df[(local.dt.hour * 60 + local.dt.minute).to_numpy() >= 11 * 60 + 30].copy()
    rf = df["rf_after_1100"].fillna(1.0).to_numpy(float)
    df["mwap_eff"] = apply_reduction_factor(df["morningwap"].to_numpy(float), rf)
    df["l11_eff"] = np.where(df["ltp_1100"] > 1, apply_reduction_factor(df["ltp_1100"].to_numpy(float), rf), np.nan)
    top = df.groupby("race_id")["excess_vol"].transform("max")
    df["informed_top"] = df["excess_vol"] == top
    df["informed_q5"] = df["excess_vol"] >= df["excess_vol"].quantile(0.8)
    return df


def markouts(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Expected value at the close, per GBP of backer stake, for each side."""
    p = df["p_bsp"].to_numpy(float)
    P = df[price_col].to_numpy(float)
    c = EXCHANGE_COMMISSION
    back = p * (P - 1.0) * (1.0 - c) - (1.0 - p)
    lay = (1.0 - p) * (1.0 - c) - p * (P - 1.0)
    return pd.DataFrame({"back": back, "lay": lay, "price": P, "w": df["morningtradedvol"].to_numpy(float),
                         "race_id": df["race_id"].to_numpy()}, index=df.index)


def summarise(m: pd.DataFrame, mask: np.ndarray, label: str) -> dict:
    s = m[mask & np.isfinite(m["price"].to_numpy())]
    if s.empty:
        return {"label": label, "n": 0}
    w = s["w"].to_numpy()
    tick_ev = tick_size(s["price"].to_numpy()) / s["price"].to_numpy()  # EV per GBP of one tick of spread
    out = {"label": label, "n": int(len(s)), "volume": round(float(w.sum()), 0)}
    for side in ("lay", "back"):
        v = s[side].to_numpy()
        out[f"{side}_vw"] = round(float(np.average(v, weights=w)) * 100, 3)
        out[f"{side}_mean"] = bootstrap_roi_ci(v, clusters=s["race_id"].to_numpy(), n_boot=500)
    balanced = 0.5 * (s["lay"].to_numpy() + s["back"].to_numpy())
    out["balanced_vw"] = round(float(np.average(balanced, weights=w)) * 100, 3)
    out["one_tick_ev_vw"] = round(float(np.average(tick_ev, weights=w)) * 100, 3)
    # Ticks of half-spread a market-maker must earn on each fill to break even.
    need_lay = np.maximum(-s["lay"].to_numpy(), 0.0)
    out["breakeven_ticks_lay_vw"] = round(float(np.average(-s["lay"].to_numpy(), weights=w) /
                                                np.average(tick_ev, weights=w)), 2)
    out["breakeven_ticks_balanced_vw"] = round(float(np.average(-balanced, weights=w) /
                                                     np.average(tick_ev, weights=w)), 2)
    out["share_lay_negative"] = round(float((need_lay > 0).mean()), 3)
    return out


def run() -> dict:
    df = frame()
    print(f"frame: {df['race_id'].nunique():,} races, {len(df):,} runners", flush=True)
    report: dict = {}
    for price_col in ("mwap_eff", "l11_eff"):
        m = markouts(df, price_col)
        rows = [summarise(m, np.ones(len(m), bool), "all runners"),
                summarise(m, df["informed_top"].to_numpy(), "informed: top excess volume in race"),
                summarise(m, ~df["informed_top"].to_numpy(), "rest"),
                summarise(m, df["informed_q5"].to_numpy(), "informed: excess volume top quintile")]
        for lo, hi in BANDS:
            band = (m["price"] >= lo).to_numpy() & (m["price"] < hi).to_numpy()
            rows.append(summarise(m, band & df["informed_top"].to_numpy(), f"informed, {lo}-{hi}"))
            rows.append(summarise(m, band & ~df["informed_top"].to_numpy(), f"rest, {lo}-{hi}"))
        report[price_col] = rows
    per_runner = df["morningtradedvol"]
    per_day = df.groupby("race_date")["morningtradedvol"].sum()
    report["capacity"] = {
        "morning_vol_per_runner_median": round(float(per_runner.median()), 0),
        "morning_vol_per_runner_mean": round(float(per_runner.mean()), 0),
        "morning_vol_per_day_median": round(float(per_day.median()), 0),
        "morning_vol_per_day_mean": round(float(per_day.mean()), 0),
        "informed_vol_per_day_median": round(float(df[df["informed_top"]].groupby("race_date")["morningtradedvol"].sum().median()), 0),
        "days": int(per_day.size),
    }
    return report


def render(report: dict) -> str:
    L = []
    for price_col in ("mwap_eff", "l11_eff"):
        L.append(f"\n=== markout per GBP matched at {price_col} (EV at the close, 2% commission, %) ===")
        for r in report[price_col]:
            if not r.get("n"):
                continue
            L.append(f"  {r['label']:40s} n={r['n']:7,} vol {r['volume']:>13,.0f}  lay vw {r['lay_vw']:+7.2f}  "
                     f"lay mean {r['lay_mean']['mean']:+7.2f} [{r['lay_mean']['ci_low']:+.2f},{r['lay_mean']['ci_high']:+.2f}]  "
                     f"back vw {r['back_vw']:+7.2f}  balanced vw {r['balanced_vw']:+6.2f}  1 tick = {r['one_tick_ev_vw']:.2f}  "
                     f"break-even ticks: lay {r['breakeven_ticks_lay_vw']:+.2f} balanced {r['breakeven_ticks_balanced_vw']:+.2f}")
    L.append(f"\ncapacity: {report['capacity']}")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report = run()
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
