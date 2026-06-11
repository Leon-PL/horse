"""One-off training run: 600 days of database history, ranker off,
baselines on. Saves a normal run snapshot so it appears in the
Experiments page."""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_600d")

import config
from src.database import load_from_database
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import RacePredictor, get_feature_columns
from src.run_store import save_run
from src.utils import compact_numeric_dtypes

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
logger.info("Loading 600 days from database ...")
raw = load_from_database(days_back=600)
logger.info("Loaded %d rows, %s -> %s", len(raw), raw["race_date"].min(), raw["race_date"].max())

logger.info("Processing ...")
processed = process_data(raw, save=False)
logger.info("Processed: %d rows", len(processed))

logger.info("Engineering features ...")
featured = engineer_features(processed, save=False)
featured = compact_numeric_dtypes(featured, label="600d featured")
logger.info("Featured: %d rows x %d cols", len(featured), featured.shape[1])

logger.info("Training RacePredictor (ranker=%s) ...", config.TRAIN_RANKER)
predictor = RacePredictor(frameworks=dict(config.SUB_MODEL_FRAMEWORKS))
metrics = predictor.train(featured, save=True, value_config=VALUE_CONFIG)

elapsed = time.time() - t0
run_id = save_run(
    name="600d baseline check",
    model_type="race_predictor",
    data_source="database (600d)",
    data_rows=len(featured),
    n_features=len(get_feature_columns(featured)),
    elapsed_seconds=elapsed,
    metrics=metrics,
    train_metrics=getattr(predictor, "train_metrics", None),
    test_analysis=predictor.test_analysis,
    training_config={"value_config": VALUE_CONFIG, "days_back": 600,
                     "frameworks": dict(config.SUB_MODEL_FRAMEWORKS),
                     "train_ranker": bool(config.TRAIN_RANKER)},
    featured_df=featured,
    processed_df=processed,
)
logger.info("Run saved: %s (%.1f min total)", run_id, elapsed / 60)

print("\n================ SUMMARY ================")
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
print("run_id:", run_id)
sys.stdout.flush()
