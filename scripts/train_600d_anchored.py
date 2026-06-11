"""600-day training with market anchor + fold early stopping.

Reuses the featured snapshot from run 20260611_000627 so metrics and
wall time are directly comparable to the '600d baseline check' run."""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_600d_anchored")

import pandas as pd

import config
from src.model import RacePredictor, get_feature_columns
from src.run_store import get_run_featured_path, get_run_processed_path, save_run

BASE_RUN = "20260611_000627"
VALUE_CONFIG = {
    "staking_mode": "flat",
    "value_threshold": 0.05,
    "value_min_odds": 1.0,
    "value_max_odds": 101.0,
    "kelly_fraction": 0.25,
    "bankroll": 100.0,
    "ew_enabled": True,
    "ew_fraction": 0.20,
    "ew_min_place_edge": 0.15,
    "ew_min_odds": 4.0,
    "ew_max_odds": 51.0,
}

t0 = time.time()
featured = pd.read_parquet(get_run_featured_path(BASE_RUN))
logger.info("Featured: %d rows x %d cols", len(featured), featured.shape[1])

logger.info("Training RacePredictor (anchor=%s, ranker=%s) ...",
            config.MARKET_ANCHOR, config.TRAIN_RANKER)
predictor = RacePredictor(frameworks=dict(config.SUB_MODEL_FRAMEWORKS))
metrics = predictor.train(featured, save=True, value_config=VALUE_CONFIG)

train_elapsed = time.time() - t0
processed_path = get_run_processed_path(BASE_RUN)
processed = pd.read_parquet(processed_path) if processed_path else None
run_id = save_run(
    name="600d anchored + fold ES",
    model_type="race_predictor",
    data_source="database (600d, market anchor)",
    data_rows=len(featured),
    n_features=len(get_feature_columns(featured)),
    elapsed_seconds=train_elapsed,
    metrics=metrics,
    train_metrics=getattr(predictor, "train_metrics", None),
    test_analysis=predictor.test_analysis,
    training_config={"value_config": VALUE_CONFIG, "days_back": 600,
                     "frameworks": dict(config.SUB_MODEL_FRAMEWORKS),
                     "train_ranker": bool(config.TRAIN_RANKER),
                     "market_anchor": predictor.market_anchor,
                     "base_run": BASE_RUN},
    featured_df=featured,
    processed_df=processed,
)
logger.info("Run saved: %s (model phase %.1f min)", run_id, train_elapsed / 60)

print("\n================ SUMMARY (anchored) ================")
print("market_anchor:", predictor.market_anchor)
for key in ["win_classifier", "baseline_win", "baseline_market", "place_classifier", "baseline_place"]:
    m = metrics.get(key)
    if not isinstance(m, dict):
        continue
    if "ndcg_at_1" in m:
        print(f"{key:18s} ndcg@1={m['ndcg_at_1']:.4f} top1={m['top1_accuracy']:.4f} "
              f"brier={m['brier_score']:.6f} logloss={m['log_loss']:.4f} ece={m.get('ece', float('nan')):.4f}")
    else:
        print(f"{key:18s} brier_cal={m.get('brier_calibrated')} place_precision={m.get('place_precision')}")

ta = predictor.test_analysis or {}
stats = ta.get("stats", {})
for strat in ["top_pick", "value", "each_way"]:
    s = stats.get(strat, {})
    if s:
        print(f"{strat:10s} bets={s.get('bets')} strike={s.get('strike_rate')}% roi={s.get('roi')}% pnl={s.get('pnl')}")
print(f"model-phase wall time: {train_elapsed/60:.1f} min (previous run: ~21 min model phase)")
print("run_id:", run_id)
sys.stdout.flush()
