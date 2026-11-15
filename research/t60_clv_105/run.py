"""Does betting at the T-60 price on the corrected forecast's disagreements beat Betfair SP? (research 105)

    python -m research.t60_clv_105.run

Forecast: the joint correction ``with-3600`` (research 104) on the native 87-input forecasts, scored quarters
2025Q3-2026Q2 (retrospective, reused; hold-out untouched), normalised within race. The same for ``with-1100`` at the
11:00 price, for comparison.

Bet: back at the last traded price at the cut (T-60 or 11:00), cut by the reduction factors of runners withdrawn
after it (``rf_after_*``; ledger reduction-factor trap). Selection is ``src.bet_settlement.value_bet_selection`` against
the race-normalised share of the field at the cut, at several base thresholds. Settled at the taken price with
exchange commission; ROI interval resamples races (``src.wf_eval.bootstrap_roi_ci``). CLV is taken / BSP, read as
median, geometric mean and share beating the close (``src.clv.summarise_clv``). Every rule is compared with all
runners priced at the same cut and in the same odds band (the universe the rule bets in).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.error_anatomy import joint_variants as jv
from research.error_anatomy import soft_corrections as soft
from research.error_anatomy.morning_price import _table
from research.meta_learner import screening as s
from src.bet_settlement import (apply_reduction_factor, betfair_tick_down, settle_win_exchange_bets,
                                value_bet_selection)
from src.clv import summarise_clv
from src.wf_eval import bootstrap_roi_ci

OUT = Path("data/research/t60_clv_105")
KEYS = s.KEYS
THRESHOLDS = (0.0, 0.02, 0.05, 0.10)
MAX_ODDS = 51.0
CUTS = {"3600s": "with-3600", "1100": "with-1100"}


def prices():
    t = _table("timing")[["bsp", "ltp_3600s", "rf_after_3600s", "ltp_1100", "rf_after_1100"]].reset_index(
        names="source_row")
    links = _table("links_timing")
    links = links.loc[links.match_status.eq("matched"), ["source_row", "sl_race_id", "sl_horse_id"]]
    out = t.merge(links, on="source_row", validate="one_to_one")
    out["race_id"] = out.sl_race_id.astype(str)
    out["horse_id"] = pd.to_numeric(out.sl_horse_id).astype("int64")
    out = out.drop(columns=["source_row", "sl_race_id", "sl_horse_id"])
    return out.loc[~out.duplicated(KEYS, keep=False)]


def race_norm(values, race):
    v = pd.Series(np.asarray(values, dtype=float))
    return (v / v.groupby(np.asarray(race)).transform("sum")).to_numpy()


def frame():
    saved = pd.read_parquet(jv.SAVED)
    base = soft.frames()["long"]
    rows = base.quarter.isin(s.SCORED).to_numpy()
    scored = base.loc[rows, [*KEYS, "race_date", "quarter", "won", "p_single"]].reset_index(drop=True)
    scored = scored.merge(saved[[*KEYS, *CUTS.values()]], on=KEYS, how="left", validate="one_to_one")
    f = jv.native_frame(scored)
    p = prices()
    p["horse_id"] = p.horse_id.astype(f.horse_id.dtype)
    f = f.merge(p, on=KEYS, how="left", validate="one_to_one")
    race = f.race_id.to_numpy()
    for cut, variant in CUTS.items():
        f[f"p_{cut}"] = race_norm(np.exp(np.log(f.p_native) + f[variant]), race)
        price = f[f"ltp_{cut}"].where(f[f"ltp_{cut}"] > 1)
        whole = price.notna().groupby(race).transform("all")
        f[f"mkt_{cut}"] = np.where(whole, race_norm(1 / price, race), np.nan)
    return f


def evaluate(f, cut, threshold, ticks=0):
    ok = f[f"mkt_{cut}"].notna() & f.bsp.gt(1)
    g = f[ok].copy()
    taken = apply_reduction_factor(g[f"ltp_{cut}"], g[f"rf_after_{cut}"])
    if ticks:
        taken = betfair_tick_down(taken, ticks)
    band = (taken > 1) & (taken <= MAX_ODDS)
    g, taken = g[band], taken[band]
    pick = value_bet_selection(g[f"p_{cut}"], taken, threshold, implied_prob=g[f"mkt_{cut}"])["mask"]

    def summary(mask):
        pnl = settle_win_exchange_bets(taken[mask], g.won.to_numpy()[mask])
        roi = bootstrap_roi_ci(pnl, clusters=g.race_id.to_numpy()[mask])
        clv = summarise_clv(taken[mask] / g.bsp.to_numpy()[mask])
        return {"bets": int(mask.sum()), "races": int(g.race_id[mask].nunique()),
                "median_odds": float(np.median(taken[mask])), "roi": roi,
                "median_clv": clv["median_clv"], "geometric_mean_clv": clv["geometric_mean_clv"],
                "pct_beat_close": clv["pct_beat_close"]}
    return {"rule": summary(pick), "universe": summary(np.ones(len(g), bool))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = frame()
    report = {}
    for cut in CUTS:
        for threshold in THRESHOLDS:
            key = f"{cut} threshold {threshold}"
            report[key] = evaluate(f, cut, threshold)
            r, u = report[key]["rule"], report[key]["universe"]
            print(f"{key:<24} bets {r['bets']:>6}  median odds {r['median_odds']:5.1f}  "
                  f"medCLV {r['median_clv']:.4f}  geoCLV {r['geometric_mean_clv']:.4f}  beat {r['pct_beat_close']:.3f}  "
                  f"ROI {r['roi']['mean']:+.2f}% [{r['roi']['ci_low']:+.2f}, {r['roi']['ci_high']:+.2f}]   "
                  f"| all {u['bets']} geoCLV {u['geometric_mean_clv']:.4f} ROI {u['roi']['mean']:+.2f}%", flush=True)
    key = "3600s threshold 0.05, one tick worse"
    report[key] = evaluate(f, "3600s", 0.05, ticks=1)
    r = report[key]["rule"]
    print(f"{key}: bets {r['bets']} geoCLV {r['geometric_mean_clv']:.4f} ROI {r['roi']['mean']:+.2f}% "
          f"[{r['roi']['ci_low']:+.2f}, {r['roi']['ci_high']:+.2f}]")
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
