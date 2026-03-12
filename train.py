"""
Training Script
===============
End-to-end pipeline: collect data -> process -> engineer features -> train model.

Usage:
    python train.py                                      # Full pipeline (sample data)
    python train.py --source database --days-back 90     # Incremental DB (recommended)
    python train.py --source scrape --days-back 14       # Real data via web scraping
    python train.py --model rank_ensemble                # Rank ensemble
    python train.py --races 2000                         # More synthetic races
    python train.py --skip-collection                    # Reuse existing data
    python train.py --backtest                           # Walk-forward backtest after training
"""

import argparse
import logging
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_collector import collect_data
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import (
    RankingPredictor, RankEnsemblePredictor, TripleEnsemblePredictor,
    RANKER_MODELS, DEFAULT_FRAMEWORKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    model_type: str = "xgb_ranker",
    num_races: int = 1500,
    skip_collection: bool = False,
    data_source: str = "sample",
    days_back: int = 90,
    backtest: bool = False,
    frameworks: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
):
    """
    Run the full training pipeline.

    Args:
        model_type: Model to train ("xgb_ranker", "lgbm_ranker", "rank_ensemble")
        num_races: Number of races for synthetic data
        skip_collection: Skip data collection (reuse existing data)
        data_source: "database" (incremental, recommended), "scrape", or "sample"
        days_back: Number of days of history for real data
        frameworks: Per-sub-model framework overrides for triple_ensemble
        weights: Manual ensemble weights (skips Optuna if provided)
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  Horse Racing Prediction - Training Pipeline")
    logger.info("=" * 60)

    # --- Step 1: Data Collection ---
    if not skip_collection:
        logger.info(f"\n📊 Step 1/4: Collecting data (source={data_source})...")
        if data_source in ("scrape", "database"):
            try:
                raw_data = collect_data(source=data_source, days_back=days_back)
            except Exception as e:
                logger.error(f"  ❌ Scraping failed: {e}")
                logger.info("  Falling back to sample data...")
                raw_data = collect_data(source="sample", num_races=num_races)
        else:
            raw_data = collect_data(source="sample", num_races=num_races)

        if raw_data is None or raw_data.empty:
            logger.warning("  No data collected from API. Falling back to sample data...")
            raw_data = collect_data(source="sample", num_races=num_races)

        logger.info(f"  Collected {len(raw_data)} race entries")
    else:
        logger.info("\n📊 Step 1/4: Skipping data collection (using existing data)")
        raw_data = None

    # --- Step 2: Data Processing ---
    logger.info("\n🔧 Step 2/4: Processing data...")
    processed_data = process_data(df=raw_data)
    logger.info(f"  Processed data shape: {processed_data.shape}")

    # --- Step 3: Feature Engineering ---
    logger.info("\n⚙️ Step 3/4: Engineering features...")
    featured_data = engineer_features(processed_data)
    logger.info(f"  Featured data shape: {featured_data.shape}")

    # --- Step 4: Model Training ---
    logger.info(f"\n🤖 Step 4/4: Training {model_type} model...")

    if model_type == "triple_ensemble":
        predictor = TripleEnsemblePredictor(frameworks=frameworks)
        metrics = predictor.train(featured_data, weights=weights)
        logger.info(f"\nTriple Ensemble Results (frameworks: {predictor.frameworks}):")
        for name, m in metrics.items():
            logger.info(
                f"  {name}: Brier = {m.get('brier_score', 0):.6f}, "
                f"NDCG@1 = {m.get('ndcg_at_1', 0):.4f}, "
                f"Top-1 = {m.get('top1_accuracy', 0):.4f}"
            )
    elif model_type == "rank_ensemble":
        predictor = RankEnsemblePredictor()
        metrics = predictor.train(featured_data)
        logger.info("\nRank Ensemble Results:")
        for name, m in metrics.items():
            logger.info(f"  {name}: NDCG@1 = {m['ndcg_at_1']:.4f}, Top-1 = {m['top1_accuracy']:.4f}")
    else:
        predictor = RankingPredictor()
        metrics = predictor.train(featured_data, model_type=model_type)

    # --- Summary ---
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("  Training Complete!")
    logger.info(f"  Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"  Model saved to: {config.MODELS_DIR}")
    logger.info("=" * 60)

    # --- Demo prediction ---
    logger.info("\n🏇 Running demo prediction on a sample race...")
    demo_race = featured_data[
        featured_data["race_id"] == featured_data["race_id"].unique()[-1]
    ].copy()

    predictions = predictor.predict_race(demo_race)

    from src.utils import print_race_prediction
    print_race_prediction(predictions)

    # --- Optional: Walk-forward backtest ---
    if backtest:
        logger.info("\n🔄 Running walk-forward backtest...")
        from src.backtester import walk_forward_validation
        report = walk_forward_validation(
            featured_data, model_type=model_type,
        )
        # Save backtest results
        import os as _os
        bt_dir = _os.path.join(config.DATA_DIR, "backtest")
        _os.makedirs(bt_dir, exist_ok=True)
        report["summary"].to_csv(
            _os.path.join(bt_dir, "fold_summary.csv"), index=False
        )
        if not report["bets"].empty:
            report["bets"].to_csv(
                _os.path.join(bt_dir, "all_bets.csv"), index=False
            )
        if not report["curves"].empty:
            report["curves"].to_csv(
                _os.path.join(bt_dir, "cumulative_pnl.csv"), index=False
            )
        logger.info(f"  Backtest results saved to {bt_dir}/")

    return predictor, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train horse racing prediction model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgb_ranker",
        choices=[
            "xgb_ranker", "lgbm_ranker", "cat_ranker", "rank_ensemble",
            "triple_ensemble",
        ],
        help="Model type to train (default: xgb_ranker)",
    )
    parser.add_argument(
        "--races",
        type=int,
        default=1500,
        help="Number of races for synthetic data (default: 1500)",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection and reuse existing data",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sample",
        choices=["sample", "scrape", "database"],
        help="Data source: 'database' for incremental DB (recommended), 'scrape' for full re-scrape, 'sample' for synthetic (default: sample)",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Days of history for real data sources (default: 90)",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward backtest after training",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default=None,
        help=(
            "Comma-separated key=value pairs to override sub-model frameworks "
            "for triple_ensemble, e.g. 'classifier=lgbm,regressor=lgbm'. "
            f"Valid keys: {list(DEFAULT_FRAMEWORKS.keys())}. "
            "Values: 'xgb', 'lgbm', or 'cat'."
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Comma-separated key=value pairs to set manual ensemble weights "
            "for triple_ensemble (skips Optuna optimisation), e.g. "
            "'ltr=0.30,regressor=0.16,classifier=0.16,"
            "place=0.12,norm_pos=0.14,residual=0.12'. "
            "Values are auto-normalised to sum to 1."
        ),
    )

    args = parser.parse_args()

    # Parse --frameworks into a dict
    fw_overrides: dict[str, str] | None = None
    if args.frameworks:
        fw_overrides = {}
        for pair in args.frameworks.split(","):
            k, v = pair.strip().split("=")
            if k not in DEFAULT_FRAMEWORKS:
                parser.error(f"Unknown framework key '{k}'. Valid: {list(DEFAULT_FRAMEWORKS.keys())}")
            if v not in ("xgb", "lgbm", "cat"):
                parser.error(f"Invalid framework value '{v}'. Must be 'xgb', 'lgbm', or 'cat'.")
            fw_overrides[k] = v

    # Parse --weights into a dict
    _valid_weight_keys = {"ltr", "regressor", "classifier", "place", "norm_pos", "residual"}
    weight_overrides: dict[str, float] | None = None
    if args.weights:
        weight_overrides = {}
        for pair in args.weights.split(","):
            k, v = pair.strip().split("=")
            if k not in _valid_weight_keys:
                parser.error(f"Unknown weight key '{k}'. Valid: {sorted(_valid_weight_keys)}")
            try:
                weight_overrides[k] = float(v)
            except ValueError:
                parser.error(f"Invalid weight value '{v}' for key '{k}'. Must be a number.")

    run_pipeline(
        model_type=args.model,
        num_races=args.races,
        skip_collection=args.skip_collection,
        data_source=args.source,
        days_back=args.days_back,
        backtest=args.backtest,
        frameworks=fw_overrides,
        weights=weight_overrides,
    )


if __name__ == "__main__":
    main()
