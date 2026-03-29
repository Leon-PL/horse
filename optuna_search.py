"""
Optuna Hyperparameter Search
=====================================================
Tunes the win classifier and place classifier independently using a fast
2-fold walk-forward validation.  The objective is average
NDCG@1 across folds (higher = better ranking quality).

Usage::

    # Full search (default 60 trials per model)
    python optuna_search.py

    # Quick test run
    python optuna_search.py --trials 10

    # Tune a single sub-model
    python optuna_search.py --model classifier --trials 40

    # Resume from a previous study
    python optuna_search.py --resume

Results are saved to ``data/optuna/`` and a ready-to-paste
config snippet is printed at the end.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score as _ndcg

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Sub-model definitions ────────────────────────────────────────
SUB_MODELS = {
    "classifier": {
        "config_key": "CLASSIFIER_PARAMS",
        "kind": "classifier",
        "description": "Win Classifier",
    },
    "place": {
        "config_key": "PLACE_CLASSIFIER_PARAMS",
        "kind": "place",
        "description": "Place Classifier",
    },
}


# ── Search spaces ────────────────────────────────────────────────
def _common_search_space(trial):
    """Hyperparameters shared by all LGBM sub-models."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0, step=0.05),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 60),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ── Data preparation ─────────────────────────────────────────────
def _load_featured_data() -> pd.DataFrame:
    """Load feature-engineered data from disk or database."""
    pq_path = os.path.join(
        config.PROCESSED_DATA_DIR, "featured_races.parquet",
    )
    csv_path = os.path.join(
        config.PROCESSED_DATA_DIR, "featured_races.csv",
    )
    if os.path.exists(pq_path):
        logger.info(f"Loading featured data from {pq_path} …")
        df = pd.read_parquet(pq_path, engine="pyarrow")
    elif os.path.exists(csv_path):
        logger.info(f"Loading featured data from {csv_path} …")
        df = pd.read_csv(csv_path)
    else:
        logger.info("No featured data found — building from database …")
        from src.data_processor import process_data
        from src.feature_engineer import engineer_features

        processed = process_data(save=False)
        df = engineer_features(processed, save=True)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df[df["finish_position"].notna() & (df["finish_position"] > 0)].copy()
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows, {df['race_id'].nunique():,} races")
    return df


def _build_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 2,
    min_train_months: int = 6,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build temporal expanding-window folds for validation.

    Returns a list of (train_df, val_df) pairs.
    """
    from src.model import get_feature_columns

    df["_ym"] = df["race_date"].dt.to_period("M")
    months = sorted(df["_ym"].unique())

    total_months = len(months)
    # Reserve roughly equal test windows for each fold
    test_months_per_fold = max(1, (total_months - min_train_months) // n_folds)

    folds = []
    for i in range(n_folds):
        test_start_idx = min_train_months + i * test_months_per_fold
        test_end_idx = min(
            test_start_idx + test_months_per_fold, total_months,
        )
        if test_start_idx >= total_months:
            break

        train_months = months[:test_start_idx]
        test_months = months[test_start_idx:test_end_idx]

        train_mask = df["_ym"].isin(train_months)
        test_mask = df["_ym"].isin(test_months)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        # Purge gap
        _purge_days = getattr(config, "PURGE_DAYS", 7)
        if _purge_days > 0:
            _test_start = test_df["race_date"].min()
            _purge_cutoff = _test_start - pd.Timedelta(days=_purge_days)
            train_df = train_df[train_df["race_date"] <= _purge_cutoff].copy()

        folds.append((train_df, test_df))
        logger.info(
            f"  Fold {len(folds)}: train {train_months[0]}–{train_months[-1]} "
            f"({len(train_df):,} rows) → test {test_months[0]}–{test_months[-1]} "
            f"({len(test_df):,} rows)"
        )

    return folds


# ── Scoring helpers ──────────────────────────────────────────────
def _ndcg_at_1(raw_scores, test_df):
    """Compute mean NDCG@1 across races."""
    y_test_rel = np.where(
        test_df["finish_position"] == 1, 10,
        np.where(test_df["finish_position"] == 2, 4,
        np.where(test_df["finish_position"] == 3, 2, 1)),
    ).astype(int)

    ndcg_list = []
    for race_id in test_df["race_id"].unique():
        rm = test_df["race_id"].values == race_id
        rs = raw_scores[rm]
        rl = y_test_rel[rm]
        if len(rs) < 2 or rl.max() == rl.min():
            continue
        try:
            ndcg_list.append(_ndcg([rl], [rs], k=1))
        except ValueError:
            pass
    return float(np.mean(ndcg_list)) if ndcg_list else 0.0


def _recency_weights(train_df):
    """Build recency sample weights matching production logic."""
    dates = pd.to_datetime(train_df["race_date"])
    days_ago = (dates.max() - dates).dt.days.values.astype(np.float64)
    _hl = getattr(config, "RECENCY_HALF_LIFE_DAYS", 180)
    sw = np.exp(-np.log(2) * days_ago / _hl)
    _seasonal = getattr(config, "RECENCY_SEASONAL_BOOST", 0.0)
    if _seasonal > 0:
        _cur_month = dates.max().month
        _same_month = (dates.dt.month.values == _cur_month).astype(np.float64)
        sw *= (1.0 + _seasonal * _same_month)
    sw /= sw.mean()
    return sw


# ── Per-model training + scoring ─────────────────────────────────
def _train_and_score(
    kind: str,
    params: dict,
    folds: list[tuple],
    feature_cols: list[str],
) -> float:
    """Train a single sub-model type on each fold and return avg NDCG@1."""
    from lightgbm import LGBMClassifier
    from src.model import (
        _focal_binary_objective,
        _FocalLGBMClassifier,
        TripleEnsemblePredictor,
    )

    ndcg_scores = []

    for train_df, test_df in folds:
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        X_train_s = X_train
        X_test_s = X_test

        fp_train = train_df["finish_position"].values.astype(np.float32)
        y_won = train_df["won"].fillna(0).values.astype(int)
        y_placed = (fp_train <= 3).astype(int)

        sw = _recency_weights(train_df)

        # Common LGBM kwargs
        base_kw = dict(random_state=config.RANDOM_SEED, n_jobs=-1, verbose=-1)
        hp = {k: v for k, v in params.items()}

        if kind == "classifier":
            model = _FocalLGBMClassifier(
                objective=_focal_binary_objective, **hp, **base_kw,
            )
            n_pos = max(int(y_won.sum()), 1)
            model.fit(X_train_s, y_won, sample_weight=sw)
            raw_scores = model.predict_proba(X_test_s)[:, 1]

        elif kind == "place":
            model = _FocalLGBMClassifier(
                objective=_focal_binary_objective, **hp, **base_kw,
            )
            model.fit(X_train_s, y_placed, sample_weight=sw)
            raw_scores = model.predict_proba(X_test_s)[:, 1]

        else:
            raise ValueError(f"Unknown model kind: {kind}")

        score = _ndcg_at_1(raw_scores, test_df)
        ndcg_scores.append(score)

    return float(np.mean(ndcg_scores))


# ── Optuna objective factory ─────────────────────────────────────
def _make_objective(kind, folds, feature_cols):
    """Return an Optuna objective function for a given sub-model type."""

    def objective(trial):
        params = _common_search_space(trial)
        score = _train_and_score(kind, params, folds, feature_cols)
        return score

    return objective


# ── Main search loop ─────────────────────────────────────────────
def run_search(
    models: list[str] | None = None,
    n_trials: int = 60,
    n_folds: int = 2,
    min_train_months: int = 6,
    resume: bool = False,
):
    """Run the full Optuna search and save results."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    output_dir = os.path.join(config.DATA_DIR, "optuna")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = _load_featured_data()
    from src.model import get_feature_columns

    feature_cols = get_feature_columns(df)

    # Optional feature pruning (to match production)
    _prune_frac = getattr(config, "FEATURE_PRUNE_FRACTION", 0.0)
    if _prune_frac > 0:
        from src.model import _prune_features_quick
        feature_cols = _prune_features_quick(df, feature_cols, _prune_frac)

    logger.info(f"Using {len(feature_cols)} features after pruning")

    # Build folds
    logger.info(f"Building {n_folds} walk-forward folds …")
    folds = _build_walk_forward_folds(df, n_folds, min_train_months)
    if not folds:
        logger.error("No valid folds — not enough data")
        return

    # Which models to tune
    to_tune = models or list(SUB_MODELS.keys())
    results = {}

    for model_name in to_tune:
        if model_name not in SUB_MODELS:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        info = SUB_MODELS[model_name]
        kind = info["kind"]
        desc = info["description"]
        config_key = info["config_key"]

        logger.info(f"\n{'═'*60}")
        logger.info(f"  Tuning {desc} ({model_name})  —  {n_trials} trials")
        logger.info(f"{'═'*60}")

        # Storage for persistence / resume
        db_path = os.path.join(output_dir, f"study_{model_name}.db")
        storage = f"sqlite:///{db_path}"
        study_name = f"horse_{model_name}"

        if resume and os.path.exists(db_path):
            logger.info(f"  Resuming from {db_path}")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True,
            )
            remaining = max(0, n_trials - len(study.trials))
            if remaining == 0:
                logger.info(f"  Already have {len(study.trials)} trials — skipping")
                results[model_name] = {
                    "best_params": study.best_params,
                    "best_score": study.best_value,
                    "n_trials": len(study.trials),
                }
                continue
        else:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",  # higher NDCG = better
                load_if_exists=False,
            )
            remaining = n_trials

        objective = _make_objective(kind, folds, feature_cols)

        t0 = time.time()
        study.optimize(
            objective,
            n_trials=remaining,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        elapsed = time.time() - t0

        best = study.best_params
        best_score = study.best_value

        logger.info(
            f"\n  ✅ {desc}: best NDCG@1 = {best_score:.4f}  "
            f"({len(study.trials)} trials in {elapsed:.0f}s)"
        )
        logger.info(f"     Best params: {best}")

        # Compare with current config
        current_params = getattr(config, config_key, {})
        logger.info(f"     Current config ({config_key}): {current_params}")

        results[model_name] = {
            "best_params": best,
            "best_score": round(best_score, 4),
            "n_trials": len(study.trials),
            "elapsed_s": round(elapsed, 1),
        }

    # ── Save results ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n💾 Results saved to {results_path}")

    # ── Print config snippet ─────────────────────────────────────
    print("\n" + "═" * 65)
    print("  READY-TO-PASTE CONFIG SNIPPET")
    print("  Copy the blocks below into config.py to apply the best params")
    print("═" * 65)
    for model_name, res in results.items():
        config_key = SUB_MODELS[model_name]["config_key"]
        print(f"\n# {SUB_MODELS[model_name]['description']}  "
              f"(NDCG@1 = {res['best_score']:.4f})")
        print(f"{config_key} = {{")
        for k, v in res["best_params"].items():
            if isinstance(v, float):
                print(f'    "{k}": {v:.6g},')
            else:
                print(f'    "{k}": {v},')
        print("}")

    return results


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for win and place classifiers",
    )
    parser.add_argument(
        "--trials", type=int, default=60,
        help="Number of Optuna trials per sub-model (default: 60)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(SUB_MODELS.keys()),
        help="Tune a single sub-model (default: all)",
    )
    parser.add_argument(
        "--folds", type=int, default=2,
        help="Number of walk-forward folds (default: 2)",
    )
    parser.add_argument(
        "--min-train-months", type=int, default=6,
        help="Minimum training months for first fold (default: 6)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous studies (if they exist)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else None
    run_search(
        models=models,
        n_trials=args.trials,
        n_folds=args.folds,
        min_train_months=args.min_train_months,
        resume=args.resume,
    )
