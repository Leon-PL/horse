"""The correction's teacher at market weights 0.5, 0.75 and 0.9: log loss and 11:00 CLV (research 106).

    python -m research.blend_weight_106.clv

Variants ``with-1100-blend50`` and ``with-1100-blend90`` (declared in ``research/error_anatomy/joint_variants.py``)
against ``with-1100`` (weight 0.75). Native log loss paired by race; then the research 105 betting test at 11:00
(value rule, thresholds 0 and 0.02, at the last traded price and one tick worse).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.error_anatomy import joint_variants as jv
from research.error_anatomy import soft_corrections as soft
from research.meta_learner import screening as s
from research.meta_learner.specialists import race_losses
from research.t60_clv_105 import run as clv
from src.bet_settlement import (apply_reduction_factor, betfair_tick_down, settle_win_exchange_bets,
                                value_bet_selection)
from src.clv import summarise_clv
from src.wf_eval import bootstrap_roi_ci

OUT = Path("data/research/blend_weight_106")
VARIANTS = {"0.50": "with-1100-blend50", "0.75": "with-1100", "0.90": "with-1100-blend90"}


def log_loss():
    saved = pd.read_parquet(jv.SAVED)
    base = soft.frames()["long"]
    rows = base.quarter.isin(s.SCORED).to_numpy()
    scored = base.loc[rows, [*s.KEYS, "race_date", "quarter", "won", "p_single"]].reset_index(drop=True)
    scored = scored.merge(saved[[*s.KEYS, *VARIANTS.values()]], on=s.KEYS, how="left", validate="one_to_one")
    frame = jv.native_frame(scored)
    bounds = s.bounds_of(frame.race_id.to_numpy())
    races = frame.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    native = np.log(frame.p_native.to_numpy())
    loss = {w: race_losses(frame, native + frame[v].to_numpy()) for w, v in VARIANTS.items()}
    out = {}
    for w in ("0.50", "0.90"):
        r = s.paired(loss[w], loss["0.75"], races)
        out[w] = {"native_log_loss": float(loss[w].mean()), "vs_075": r}
        print(f"weight {w}: {loss[w].mean():.5f}  vs 0.75 ({loss['0.75'].mean():.5f}): {r['mean']:+.5f} "
              f"[{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4", flush=True)
    out["0.75"] = {"native_log_loss": float(loss["0.75"].mean())}
    return out


def betting():
    out = {}
    for w, variant in VARIANTS.items():
        clv.CUTS = {"1100": variant}
        f = clv.frame()
        g = f[f["mkt_1100"].notna() & f.bsp.gt(1)]
        base = apply_reduction_factor(g.ltp_1100, g.rf_after_1100)
        for ticks in (0, 1):
            taken = betfair_tick_down(base, ticks) if ticks else base
            band = (taken > 1) & (taken <= clv.MAX_ODDS)
            for threshold in (0.0, 0.02):
                mask = value_bet_selection(g.p_1100.to_numpy(), taken, threshold,
                                           implied_prob=g.mkt_1100.to_numpy())["mask"] & band
                pnl = settle_win_exchange_bets(taken[mask], g.won.to_numpy()[mask])
                roi = bootstrap_roi_ci(pnl, clusters=g.race_id.to_numpy()[mask])
                c = summarise_clv(taken[mask] / g.bsp.to_numpy()[mask])
                key = f"{w} ticks {ticks} threshold {threshold}"
                out[key] = {"bets": int(mask.sum()), "geometric_mean_clv": c["geometric_mean_clv"],
                            "median_clv": c["median_clv"], "roi": roi}
                print(f"{key:<28} bets {mask.sum():>6}  geoCLV {c['geometric_mean_clv']:.4f}  "
                      f"ROI {roi['mean']:+.2f}% [{roi['ci_low']:+.2f}, {roi['ci_high']:+.2f}]", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"log_loss": log_loss(), "betting": betting()}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
