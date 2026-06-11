"""Walk-forward sweep over feature-engineering constants.

For each candidate config (one-at-a-time deviations from defaults),
re-runs feature engineering on the fixed 600-day processed snapshot,
drops market features, and scores a fast LGBM win classifier on two
purged walk-forward folds. The metric that matters here is the
NO-MARKET model's ranking skill (NDCG@1 / top-1) — pure fundamental
signal, no odds.

Results land in data/fe_sweep_results.csv.
"""

import gc
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for noisy in ["src.feature_engineer", "src.ratings", "src.weather", "src.track_config", "src.database"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("fe_sweep")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, ndcg_score

import config
import src.ratings as ratings
from src.app_helpers import _drop_market_feature_columns
from src.feature_engineer import engineer_features
from src.model import get_feature_columns

BASE_RUN = "20260611_000627"
PROCESSED_PATH = f"data/runs/{BASE_RUN}/processed_races.parquet"

# config name -> module-level binding in src.ratings (bound at import)
RATINGS_BINDINGS = {
    "ELO_K_BASE": "K_BASE",
    "ELO_K_MIN": "K_MIN",
    "ELO_K_DECAY": "K_DECAY",
    "MARGIN_ELO_SCALE": "MARGIN_SCALE",
    "MARGIN_ELO_REF_DIST": "MARGIN_REF_DIST",
    "MARGIN_ELO_DNF_PENALTY": "DNF_PENALTY_LB",
    "GLICKO_C": "GLICKO_C",
    "GLICKO_RD_INIT": "GLICKO_RD_INIT",
    "GLICKO_RD_MIN": "GLICKO_RD_MIN",
}

CANDIDATES: list[tuple[str, dict]] = [
    ("baseline (defaults + glicko)", {}),
    ("no_glicko", {"GLICKO_ENABLED": False}),
    ("ELO_K_BASE=16", {"ELO_K_BASE": 16.0}),
    ("ELO_K_BASE=64", {"ELO_K_BASE": 64.0}),
    ("MARGIN_ELO_SCALE=3", {"MARGIN_ELO_SCALE": 3.0}),
    ("MARGIN_ELO_SCALE=8", {"MARGIN_ELO_SCALE": 8.0}),
    ("TE_EWMA_HALF_LIFE=5", {"TE_EWMA_HALF_LIFE_RACES": 5}),
    ("TE_EWMA_HALF_LIFE=20", {"TE_EWMA_HALF_LIFE_RACES": 20}),
    ("GLICKO_C=40", {"GLICKO_C": 40.0}),
    ("GLICKO_C=110", {"GLICKO_C": 110.0}),
]

_DEFAULTS = {}


def _apply_overrides(overrides: dict) -> None:
    for key, value in overrides.items():
        if key not in _DEFAULTS:
            _DEFAULTS[key] = getattr(config, key, None)
        setattr(config, key, value)
        if key in RATINGS_BINDINGS:
            setattr(ratings, RATINGS_BINDINGS[key], value)


def _restore_defaults() -> None:
    for key, value in _DEFAULTS.items():
        setattr(config, key, value)
        if key in RATINGS_BINDINGS:
            setattr(ratings, RATINGS_BINDINGS[key], value)


def quick_walk_forward(featured: pd.DataFrame) -> dict:
    """Two purged temporal folds; fast LGBM win classifier; no market features."""
    df = featured[featured["finish_position"] > 0].copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df = df.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    feature_cols = get_feature_columns(df)

    race_ids = df["race_id"].drop_duplicates().values
    race_dates = df.drop_duplicates("race_id")["race_date"].values
    n_races = len(race_ids)

    fold_metrics = []
    for lo, hi in [(0.70, 0.85), (0.85, 1.0)]:
        ev_beg, ev_end = int(n_races * lo), int(n_races * hi)
        ev_start_date = race_dates[ev_beg]
        purge_cut = ev_start_date - np.timedelta64(7, "D")

        tr_races = set(race_ids[:ev_beg][race_dates[:ev_beg] <= purge_cut])
        ev_races = set(race_ids[ev_beg:ev_end])
        tr_mask = df["race_id"].isin(tr_races).values
        ev_mask = df["race_id"].isin(ev_races).values

        X_tr = df.loc[tr_mask, feature_cols].values
        y_tr = (df.loc[tr_mask, "finish_position"] == 1).astype(int).values
        X_ev = df.loc[ev_mask, feature_cols].values

        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.7,
            n_jobs=-1, verbose=-1, random_state=42,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_ev)[:, 1]

        ev_df = df.loc[ev_mask, ["race_id", "finish_position"]].copy()
        ev_df["prob"] = probs
        ndcg1, top1, total = [], 0, 0
        for _, grp in ev_df.groupby("race_id", sort=False):
            fp = grp["finish_position"].values
            p = grp["prob"].values
            if len(grp) < 2 or p.max() == p.min():
                continue
            rel = np.clip(8.0 - fp, 0, 7)
            total += 1
            ndcg1.append(ndcg_score([rel], [p], k=1))
            if fp[np.argmax(p)] == 1:
                top1 += 1
        won = (ev_df["finish_position"] == 1).astype(int).values
        fold_metrics.append({
            "ndcg1": float(np.mean(ndcg1)) if ndcg1 else 0.0,
            "top1": top1 / total if total else 0.0,
            "logloss": float(log_loss(won, np.clip(probs, 1e-12, 1 - 1e-12))),
            "races": total,
        })
    return {
        "ndcg1": float(np.mean([m["ndcg1"] for m in fold_metrics])),
        "top1": float(np.mean([m["top1"] for m in fold_metrics])),
        "logloss": float(np.mean([m["logloss"] for m in fold_metrics])),
        "fold_ndcg1": [round(m["ndcg1"], 4) for m in fold_metrics],
    }


processed = pd.read_parquet(PROCESSED_PATH)
logger.info("Processed snapshot: %d rows", len(processed))

results = []
for name, overrides in CANDIDATES:
    t0 = time.time()
    _apply_overrides(overrides)
    try:
        featured = engineer_features(processed.copy(), save=False)
        featured, _ = _drop_market_feature_columns(featured)
        metrics = quick_walk_forward(featured)
        metrics.update({"candidate": name, "minutes": round((time.time() - t0) / 60, 1)})
        results.append(metrics)
        logger.info("%-28s ndcg1=%.4f top1=%.4f logloss=%.4f folds=%s (%.1f min)",
                    name, metrics["ndcg1"], metrics["top1"], metrics["logloss"],
                    metrics["fold_ndcg1"], metrics["minutes"])
    except Exception:
        logger.exception("Candidate %s failed", name)
    finally:
        _restore_defaults()
        del featured
        gc.collect()

out = pd.DataFrame(results)[["candidate", "ndcg1", "top1", "logloss", "fold_ndcg1", "minutes"]]
out.to_csv("data/fe_sweep_results.csv", index=False)
print("\n================ FE SWEEP RESULTS (no-market model) ================")
print(out.sort_values("ndcg1", ascending=False).to_string(index=False))
sys.stdout.flush()
