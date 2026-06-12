"""Walk-forward validation focused on the each-way strategy.

The test-set simulations keep showing EW ROI > +100% on ~130 bets.
This runs the proper expanding-window backtest (monthly folds, purged,
fold-trained models, shared bet_settlement maths) on the full-rebuild
featured snapshot to see whether the edge survives out-of-sample
repetition or was holdout luck.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("wf_each_way")

import pandas as pd

from src.backtester import walk_forward_validation

RUN = "20260612_172827"
featured = pd.read_parquet(f"data/runs/{RUN}/featured_races.parquet")
logger.info("Featured: %d rows", len(featured))

report = walk_forward_validation(
    featured,
    min_train_months=6,
    test_window_months=1,
    value_threshold=0.05,
    ew_min_place_edge=0.15,
    fast_fold=True,
)

summary = report["summary"]
bets = report["bets"]
print("\n================ PER-FOLD SUMMARY ================")
print(summary.to_string(index=False))

print("\n================ EACH-WAY ACROSS FOLDS ================")
ew = bets[bets["strategy"] == "each_way"] if "strategy" in bets.columns else pd.DataFrame()
if not ew.empty:
    per_fold = ew.groupby("test_period").agg(
        bets=("pnl", "size"), pnl=("pnl", "sum"), winners=("won", "sum"),
    )
    per_fold["roi_%"] = (per_fold["pnl"] / (2.0 * per_fold["bets"]) * 100).round(1)
    print(per_fold.to_string())
    total_stake = 2.0 * len(ew)
    print(f"\nTOTAL: {len(ew)} bets, pnl={ew['pnl'].sum():.1f}, "
          f"roi={100 * ew['pnl'].sum() / total_stake:.1f}%, "
          f"profitable folds: {(per_fold['pnl'] > 0).sum()}/{len(per_fold)}")
else:
    print("No each-way bets recorded.")

print("\n================ VALUE ACROSS FOLDS ================")
vb = bets[bets["strategy"] == "value"] if "strategy" in bets.columns else pd.DataFrame()
if not vb.empty:
    per_fold = vb.groupby("test_period").agg(bets=("pnl", "size"), pnl=("pnl", "sum"))
    per_fold["roi_%"] = (per_fold["pnl"] / per_fold["bets"] * 100).round(1)
    print(per_fold.to_string())
    print(f"\nTOTAL: {len(vb)} bets, pnl={vb['pnl'].sum():.1f}, "
          f"roi={100 * vb['pnl'].sum() / len(vb):.1f}%, "
          f"profitable folds: {(per_fold['pnl'] > 0).sum()}/{len(per_fold)}")

bets.to_csv("data/wf_each_way_bets.csv", index=False)
print("\nbets saved to data/wf_each_way_bets.csv")
sys.stdout.flush()
