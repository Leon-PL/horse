"""Retrain win + place models on the leak-fixed (clean) feature store.

Production models were trained before the trainer_freq stablemate-leak fix
(commit 150506e) + store rebuild. This retrains on the clean store via the real
pipeline (src.training.train_run) so the production artifacts are honest. No data
refresh — train on the rebuilt store as-is. Does NOT auto-promote; prints run_id +
held-out evals (calibration / ROI / CLV) for inspection first.
"""
import os, sys, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
from src.training import TrainingConfig, train_run
from src.run_store import load_run_meta

cfg = TrainingConfig(
    name="clean-store retrain (post trainer_freq leak fix)",
    refresh_data=False,            # use the rebuilt clean store as-is
    train_place_market=True,       # retrain the place model alongside win/EW
    place_market_features=True,
    run_walk_forward=False,        # combination-edge WF validated separately
)
run_id = train_run(cfg)
print("\n=== RUN SAVED:", run_id, "===")
meta = load_run_meta(run_id)
evals = (meta or {}).get("evals", {})
# print the headline eval blocks compactly
for k in ("calibration", "backtest", "roi", "clv", "place_market"):
    if k in evals:
        print(f"\n--- {k} ---")
        print(json.dumps(evals[k], indent=1, default=str)[:1500])
