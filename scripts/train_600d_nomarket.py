"""600-day training with market/odds-derived features dropped.

Reuses the featured dataset snapshot from run 20260611_000627 (same
600-day window) so results are directly comparable to the
'600d baseline check' run."""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_600d_nomarket")

import pandas as pd

import config
from src.app_helpers import _drop_market_feature_columns
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
featured_path = get_run_featured_path(BASE_RUN)
logger.info("Loading featured snapshot from %s ...", featured_path)
featured = pd.read_parquet(featured_path)
logger.info("Featured: %d rows x %d cols", len(featured), featured.shape[1])

featured, dropped = _drop_market_feature_columns(featured)
logger.info("Dropped %d market features: %s", len(dropped), ", ".join(dropped))

logger.info("Training RacePredictor without market features (ranker=%s) ...", config.TRAIN_RANKER)
predictor = RacePredictor(frameworks=dict(config.SUB_MODEL_FRAMEWORKS))
metrics = predictor.train(featured, save=True, value_config=VALUE_CONFIG)

elapsed = time.time() - t0
processed_path = get_run_processed_path(BASE_RUN)
processed = pd.read_parquet(processed_path) if processed_path else None
run_id = save_run(
    name="600d no-market features",
    model_type="race_predictor",
    data_source="database (600d, market features dropped)",
    data_rows=len(featured),
    n_features=len(get_feature_columns(featured)),
    elapsed_seconds=elapsed,
    metrics=metrics,
    train_metrics=getattr(predictor, "train_metrics", None),
    test_analysis=predictor.test_analysis,
    training_config={"value_config": VALUE_CONFIG, "days_back": 600,
                     "frameworks": dict(config.SUB_MODEL_FRAMEWORKS),
                     "train_ranker": bool(config.TRAIN_RANKER),
                     "dropped_market_features": dropped,
                     "base_run": BASE_RUN},
    featured_df=featured,
    processed_df=processed,
)
logger.info("Run saved: %s (%.1f min total)", run_id, elapsed / 60)

print("\n================ SUMMARY (no market features) ================")
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
        print(f"{strat:10s} bets={s.get('bets')} strike={s.get('strike_rate')}% roi={s.get('roi')}% "
              f"pnl={s.get('pnl')} avg_odds={s.get('avg_odds_all')}")
print("run_id:", run_id)
sys.stdout.flush()
