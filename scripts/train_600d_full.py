"""Full 600-day end-to-end run: DB -> process (pedigree join) -> FE -> train.

Unlike train_600d_anchored.py (which reused a frozen featured snapshot),
this regenerates everything so it picks up: pedigree backfill, TrueSkill,
Glicko + swept FE constants, RTV NaN handling, dtype-filter fix, and the
market anchor. The resulting run is the new reference baseline.
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_600d_full")

import config
from src.database import load_from_database
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import RacePredictor, get_feature_columns
from src.run_store import save_run

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
raw = load_from_database(days_back=600)
logger.info("Raw: %d rows", len(raw))

processed = process_data(raw, save=False)
ped_cov = processed["sire"].notna().mean() if "sire" in processed.columns else 0.0
logger.info("Processed: %d rows (pedigree coverage %.1f%%)", len(processed), 100 * ped_cov)

featured = engineer_features(processed.copy(), save=False)
fe_elapsed = time.time() - t0
logger.info("Featured: %d rows x %d cols (FE %.1f min)",
            len(featured), featured.shape[1], fe_elapsed / 60)

t1 = time.time()
logger.info("Training RacePredictor (anchor=%s, ranker=%s, trueskill=%s) ...",
            config.MARKET_ANCHOR, config.TRAIN_RANKER,
            getattr(config, "TRUESKILL_ENABLED", False))
predictor = RacePredictor(frameworks=dict(config.SUB_MODEL_FRAMEWORKS))
metrics = predictor.train(featured, save=True, value_config=VALUE_CONFIG)
train_elapsed = time.time() - t1

run_id = save_run(
    name="600d full rebuild (pedigree+trueskill)",
    model_type="race_predictor",
    data_source="database (600d, full regen)",
    data_rows=len(featured),
    n_features=len(get_feature_columns(featured)),
    elapsed_seconds=time.time() - t0,
    metrics=metrics,
    train_metrics=getattr(predictor, "train_metrics", None),
    test_analysis=predictor.test_analysis,
    training_config={"value_config": VALUE_CONFIG, "days_back": 600,
                     "frameworks": dict(config.SUB_MODEL_FRAMEWORKS),
                     "train_ranker": bool(config.TRAIN_RANKER),
                     "market_anchor": predictor.market_anchor,
                     "trueskill": bool(getattr(config, "TRUESKILL_ENABLED", False)),
                     "pedigree_coverage": round(float(ped_cov), 4)},
    featured_df=featured,
    processed_df=processed,
)
logger.info("Run saved: %s (FE %.1f min, model %.1f min)",
            run_id, fe_elapsed / 60, train_elapsed / 60)

print("\n================ SUMMARY (full rebuild) ================")
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
print(f"wall time: FE {fe_elapsed/60:.1f} min + model {train_elapsed/60:.1f} min")
print("run_id:", run_id)
sys.stdout.flush()
