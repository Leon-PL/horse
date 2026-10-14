"""Ledger 19: is the 11:00 morning-money value concentrated in debutants and favourites?

Pre-registered in docs/NEGATIVE_RESULTS.md section 19 (commit a5e566f) before
this was run. Same rule, data and protocol as the m11 row of ledger 14: in
each race, back the runner with the highest excess morning volume if its
11:00 price is in the 1.5-50 band; strike at ``ltp_1100``, close at BSP, races
off at or after 11:30 UK. Development period only.

Primary tests: the rule's selections that are debutants (H1) and that are the
11:00 favourite (H2), each against its matched baseline (every debutant /
favourite in the band). Pass: median CLV > 1 with a race-clustered 97.5% CI
excluding 1 (Bonferroni over two), geometric CLV > 1, ROI at the taken price
after commission > 0. Complements and within-segment rules are descriptive.

Debutant = empty pre-race form string: the results begin 2020-09-15, so a
count of earlier database runs would mislabel older horses as debutants,
whereas the racecard form string grows run by run and includes earlier runs.

Usage::

    python scripts/morning_money_segments.py
    python scripts/morning_money_segments.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.microstructure_clv import BAND, _with_early_price
from scripts.microstructure_screen import load
from src.bet_settlement import settle_win_exchange_bets
from src.wf_eval import bootstrap_median_ci, bootstrap_roi_ci, clv_summary, fold_summary

EARLY = os.path.join("data", "processed", "betfair_bsp_early.parquet")
PRIMARY_ALPHA = 0.025


def debutant_flags(df: pd.DataFrame) -> pd.Series:
    """True where the runner's pre-race form string is empty."""
    con = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    form = pd.read_sql("select race_id, horse_name, form from results", con)
    form["race_id"] = form["race_id"].astype(str)
    form = form.drop_duplicates(["race_id", "horse_name"])
    keyed = df[["race_id", "horse_name"]].astype({"race_id": str}).merge(
        form, on=["race_id", "horse_name"], how="left")
    return pd.Series(keyed["form"].fillna("").astype(str).str.strip().eq("").to_numpy(), index=df.index)


def frame() -> pd.DataFrame:
    df = load()
    df["debutant"] = debutant_flags(df)
    d = _with_early_price(df, EARLY, "ltp_1100", 11 * 60 + 30)
    d["band"] = d["ltp_1100"].between(*BAND)
    top = d[d["band"]].groupby("race_id")["excess_vol"].transform("max")
    d["pick"] = False
    d.loc[d["band"], "pick"] = d.loc[d["band"], "excess_vol"] == top
    d["fav"] = d["ltp_1100"] == d.groupby("race_id")["ltp_1100"].transform("min")
    deb = d[d["band"] & d["debutant"]]
    deb_top = deb.groupby("race_id")["excess_vol"].transform("max")
    d["pick_within_deb"] = False
    d.loc[deb.index, "pick_within_deb"] = deb["excess_vol"] == deb_top
    won = d["won"].to_numpy().astype(bool)
    d["pnl_taken"] = settle_win_exchange_bets(d["ltp_1100"].to_numpy(), won)
    d["pnl_bsp"] = settle_win_exchange_bets(d["bsp"].to_numpy(), won)
    d["q"] = pd.to_datetime(d["race_date"]).dt.to_period("Q").astype(str)
    return d


def cell(d: pd.DataFrame, mask: pd.Series, n_boot: int, alpha: float) -> dict:
    s = d[mask]
    if s.empty:
        return {"n": 0}
    ratio = (s["ltp_1100"] / s["bsp"]).to_numpy()
    clusters = s["race_id"].to_numpy()
    clv = clv_summary(s["ltp_1100"], s["bsp"])
    return {
        "n": int(len(s)),
        "win_rate": round(float(s["won"].mean()), 4),
        "median_clv_ci": bootstrap_median_ci(ratio, clusters=clusters, n_boot=n_boot, alpha=alpha),
        "geo_clv": clv.get("geometric_mean_clv"),
        "beat_close": clv.get("pct_beat_close"),
        "roi_taken": bootstrap_roi_ci(s["pnl_taken"], clusters=clusters, n_boot=n_boot, alpha=alpha),
        "roi_bsp": bootstrap_roi_ci(s["pnl_bsp"], clusters=clusters, n_boot=n_boot, alpha=alpha),
        "quarters": fold_summary(s.groupby("q")["pnl_taken"].mean().mul(100).tolist()),
        "alpha": alpha,
    }


def passes(c: dict) -> bool:
    if not c.get("n"):
        return False
    return bool(c["median_clv_ci"]["ci_low"] > 1.0 and (c["geo_clv"] or 0) > 1.0
                and c["roi_taken"]["mean"] > 0)


def run(n_boot: int) -> dict:
    d = frame()
    band, pick = d["band"], d["pick"]
    print(f"m11 frame: {d['race_id'].nunique():,} races, {len(d):,} runners; "
          f"debutants in band {int((band & d['debutant']).sum()):,}, "
          f"favourites in band {int((band & d['fav']).sum()):,}", flush=True)
    spec = [
        ("H1 pick & debutant", pick & d["debutant"], PRIMARY_ALPHA, True),
        ("   baseline: debutants in band", band & d["debutant"], PRIMARY_ALPHA, False),
        ("H2 pick & favourite", pick & d["fav"], PRIMARY_ALPHA, True),
        ("   baseline: favourites in band", band & d["fav"], PRIMARY_ALPHA, False),
        ("all picks (ledger 14 m11)", pick, 0.05, False),
        ("pick & not debutant", pick & ~d["debutant"], 0.05, False),
        ("pick & not favourite", pick & ~d["fav"], 0.05, False),
        ("within-debutant top excess", d["pick_within_deb"], 0.05, False),
    ]
    out = {}
    for name, mask, alpha, primary in spec:
        c = cell(d, mask, n_boot, alpha)
        c["primary"] = primary
        if primary:
            c["pass"] = passes(c)
        out[name] = c
    return out


def render(report: dict) -> str:
    lines = ["", "=== ledger 19: morning money by segment (11:00 price -> BSP, dev period) ==="]
    for name, c in report.items():
        if not c.get("n"):
            lines.append(f"  {name:32s} n=0")
            continue
        m, r = c["median_clv_ci"], c["roi_taken"]
        tag = f"  PASS={c['pass']}" if c.get("primary") else ""
        lines.append(
            f"  {name:32s} n={c['n']:6,}  win {c['win_rate']:.3f}  medCLV {m['estimate']:.4f} "
            f"[{m['ci_low']:.4f},{m['ci_high']:.4f}]@{1 - c['alpha']:.3f}  geo {c['geo_clv']:.4f}  "
            f"beat {c['beat_close']:.3f}  ROI@11:00 {r['mean']:+.2f}% [{r['ci_low']:+.2f},{r['ci_high']:+.2f}]  "
            f"ROI@BSP {c['roi_bsp']['mean']:+.2f}%  q+ {c['quarters']['folds_positive']}/{c['quarters']['n_folds']}"
            f"{tag}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report = run(args.n_boot)
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
