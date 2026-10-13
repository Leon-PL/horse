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

The **morning protocol is look-ahead biased by construction**: excess morning
volume is only known when the morning ends, but ``morningwap`` averages the
whole morning's prices, including the longer ones a steamer traded at before
the money came. Its CLV is reported for completeness and must not be read as
a result. The T-5 protocol is clean wherever the morning ended before T-5;
``--min-off-utc 14:00`` checks that the effect survives in races run hours
after any plausible morning cutoff (Betfair does not publish it).
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


def _with_early_price(df: pd.DataFrame, path: str, col: str, min_off_minutes: int) -> pd.DataFrame:
    """Honest early protocols: strike at a price available after the morning.

    Full-morning volume is known when the morning closes (~11:00 UK), so the
    bet takes a price available then or later - never the morning VWAP.
    ``ltp_1100`` (m11): races off >= 11:30 UK. ``ltp_3600s`` (t60, one hour
    before the off): races off >= 12:05 UK, so T-60 falls after the morning.
    Races with any runner untraded by the cut have no coherent market.
    """
    from src.betfair import attach_bsp
    out = attach_bsp(df.drop(columns=[col], errors="ignore"), pd.read_parquet(path), cols=(col,))
    out[col] = pd.to_numeric(out[col], errors="coerce")
    off = pd.to_datetime(out["race_date"].astype(str) + " " + out["off_time"].astype(str), errors="coerce")
    local = off.dt.tz_localize("UTC").dt.tz_convert("Europe/London")
    late_enough = (local.dt.hour * 60 + local.dt.minute) >= min_off_minutes
    priced = (out[col] > 1).groupby(out["race_id"]).transform("all")
    return out[late_enough & priced].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-off-utc", default=None,
                    help="only races off at/after HH:MM UTC: hours after any morning "
                         "cutoff, so the morning volume cannot include post-decision trades")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    df = load()
    if args.min_off_utc:
        from src.utils import normalise_off_time_key
        df = df[df["off_time"].map(normalise_off_time_key) >= args.min_off_utc]
        print(f"restricted to races off at/after {args.min_off_utc} UTC: {df['race_id'].nunique():,} races")
    report = {"t5": evaluate(df[df["p_t5"].notna()], "ltp_300s", args.n_boot),
              "morning": evaluate(df, "morningwap", args.n_boot)}
    early = os.path.join("data", "processed", "betfair_bsp_early.parquet")
    if os.path.exists(early):
        report["m11"] = evaluate(_with_early_price(df, early, "ltp_1100", 11 * 60 + 30),
                                 "ltp_1100", args.n_boot)
        report["t60"] = evaluate(_with_early_price(df, early, "ltp_3600s", 12 * 60 + 5),
                                 "ltp_3600s", args.n_boot)
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
