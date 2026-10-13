"""Economic follow-up to the microstructure screen: is the timing signal bankable?

microstructure_screen found that excess morning volume predicts the price
move *after* the observation point (T5: +0.014 partial with the T-5 -> BSP
move; T1/T3 at morning prices), but predicts nothing beyond BSP (T2 +0.0004).
So the only possible edge is timing: take an early price on horses the money
is backing, before the close absorbs it. This measures that directly, with
one rule per protocol fixed in advance:

    in each race, back the runner with the highest excess morning volume,
    if its decision price is in the 1.5-50 band.

Matched baseline: every runner in the band (the universe the rule bets in).
Reported: CLV (median and geometric mean with race-clustered CIs) and ROI
both at the taken price and at BSP, 2% commission, per-quarter consistency.
Development period only; the golden hold-out is not read.
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
from src.bet_settlement import settle_win_exchange_bets
from src.wf_eval import bootstrap_median_ci, bootstrap_roi_ci, clv_summary, fold_summary

BAND = (1.5, 50.0)


def evaluate(df: pd.DataFrame, entry_col: str, n_boot: int) -> dict:
    # load() keeps complete markets only; the T-5 frame passed in is complete too.
    d = df.copy()
    assert (d[entry_col] > 1).all()
    d["band"] = d[entry_col].between(*BAND)
    top = d[d["band"]].groupby("race_id")["excess_vol"].transform("max")
    d["pick"] = False
    d.loc[d["band"], "pick"] = d.loc[d["band"], "excess_vol"] == top
    d["q"] = pd.to_datetime(d["race_date"]).dt.to_period("Q").astype(str)
    won = d["won"].to_numpy().astype(bool)
    d["pnl_taken"] = settle_win_exchange_bets(d[entry_col].to_numpy(), won)
    d["pnl_bsp"] = settle_win_exchange_bets(d["bsp"].to_numpy(), won)
    out = {}
    for name, mask in (("bet_all_in_band", d["band"]), ("top_excess_volume", d["pick"])):
        s = d[mask]
        ratio = (s[entry_col] / s["bsp"]).to_numpy()
        out[name] = {
            "n": int(len(s)),
            "clv": {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in clv_summary(s[entry_col], s["bsp"]).items()},
            "median_clv_ci": bootstrap_median_ci(ratio, clusters=s["race_id"].to_numpy(), n_boot=n_boot),
            "roi_taken": bootstrap_roi_ci(s["pnl_taken"], clusters=s["race_id"].to_numpy(), n_boot=n_boot),
            "roi_bsp": bootstrap_roi_ci(s["pnl_bsp"], clusters=s["race_id"].to_numpy(), n_boot=n_boot),
            "quarters_taken": fold_summary(s.groupby("q")["pnl_taken"].mean().mul(100).tolist()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    df = load()
    report = {"t5": evaluate(df[df["p_t5"].notna()], "ltp_300s", args.n_boot),
              "morning": evaluate(df, "morningwap", args.n_boot)}
    for proto, res in report.items():
        print(f"\n=== protocol {proto} (decision price -> BSP) ===")
        for name, r in res.items():
            c, mc = r["clv"], r["median_clv_ci"]
            print(f"  {name:18s} n={r['n']:7,}  medCLV {mc['estimate']:.4f} [{mc['ci_low']:.4f},{mc['ci_high']:.4f}]"
                  f"  geo {c.get('geometric_mean_clv')}  beat {c.get('pct_beat_close')}"
                  f"  ROI@taken {r['roi_taken']['mean']:+.2f}% [{r['roi_taken']['ci_low']:+.2f},{r['roi_taken']['ci_high']:+.2f}]"
                  f"  ROI@BSP {r['roi_bsp']['mean']:+.2f}%"
                  f"  q+ {r['quarters_taken']['folds_positive']}/{r['quarters_taken']['n_folds']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
