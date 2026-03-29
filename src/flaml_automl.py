from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.model import prepare_multi_target_data

logger = logging.getLogger(__name__)


def flaml_is_available() -> tuple[bool, str]:
    """Check whether FLAML is importable without failing app startup."""
    try:
        import flaml  # noqa: F401
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _grouped_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    probs = np.zeros(len(scores), dtype=np.float64)
    if len(scores) == 0:
        return probs
    tmp = pd.DataFrame({"race_id": race_ids, "score": scores.astype(float)})
    for rid, grp in tmp.groupby("race_id", sort=False):
        idx = grp.index.values
        s = grp["score"].values.astype(np.float64)
        s = s - s.max()
        e = np.exp(s)
        probs[idx] = e / max(e.sum(), 1e-12)
    return probs


def _race_level_metrics(
    race_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | None]:
    if len(race_ids) == 0:
        return {"top1_accuracy": None, "ndcg_at_1": None}

    race_df = pd.DataFrame(
        {
            "race_id": race_ids,
            "y_true": y_true.astype(int),
            "y_prob": y_prob.astype(float),
        }
    )

    top_hits: list[float] = []
    ndcg_vals: list[float] = []
    for _, grp in race_df.groupby("race_id", sort=False):
        if grp.empty:
            continue
        top_row = grp.iloc[int(np.argmax(grp["y_prob"].values))]
        top_hits.append(float(top_row["y_true"]))
        if grp["y_true"].sum() > 0 and len(grp) > 1:
            try:
                ndcg_vals.append(
                    float(
                        ndcg_score(
                            [grp["y_true"].values.astype(float)],
                            [grp["y_prob"].values.astype(float)],
                            k=1,
                        )
                    )
                )
            except Exception:
                continue

    return {
        "top1_accuracy": _safe_float(np.mean(top_hits)) if top_hits else None,
        "ndcg_at_1": _safe_float(np.mean(ndcg_vals)) if ndcg_vals else None,
    }


def _ranking_metrics_from_groups(
    scores: np.ndarray,
    y_rel: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | None]:
    """Compute ranking metrics directly from grouped race scores."""
    if len(scores) == 0 or len(groups) == 0:
        return {
            "rank_top1_accuracy": None,
            "rank_ndcg_at_1": None,
            "rank_ndcg_at_3": None,
        }

    off = 0
    top1_hits: list[float] = []
    ndcg1_vals: list[float] = []
    ndcg3_vals: list[float] = []
    for g in groups.astype(int):
        if g <= 1:
            off += g
            continue
        sl = slice(off, off + g)
        s = np.asarray(scores[sl], dtype=np.float64)
        y = np.asarray(y_rel[sl], dtype=np.float64)
        off += g
        if np.all(y == y[0]):
            continue

        top1_hits.append(float(np.argmax(s) == np.argmax(y)))
        try:
            ndcg1_vals.append(float(ndcg_score([y], [s], k=1)))
            ndcg3_vals.append(float(ndcg_score([y], [s], k=3)))
        except Exception:
            continue

    return {
        "rank_top1_accuracy": _safe_float(np.mean(top1_hits)) if top1_hits else None,
        "rank_ndcg_at_1": _safe_float(np.mean(ndcg1_vals)) if ndcg1_vals else None,
        "rank_ndcg_at_3": _safe_float(np.mean(ndcg3_vals)) if ndcg3_vals else None,
    }


def _best_estimator_table(automl) -> pd.DataFrame | None:
    losses = getattr(automl, "best_loss_per_estimator", None)
    if not isinstance(losses, dict) or not losses:
        return None
    rows = []
    for est, loss in losses.items():
        rows.append({"estimator": est, "best_loss": loss})
    return pd.DataFrame(rows).sort_values("best_loss", ascending=True)


def run_flaml_automl(
    featured_df: pd.DataFrame,
    *,
    mode: str = "classification",
    target: str = "won",
    time_budget: int = 300,
    metric: str = "auto",
    estimator_list: list[str] | None = None,
    verbose: int = 2,
    log_file_name: str | None = None,
    log_training_metric: bool = True,
) -> dict[str, Any]:
    """Run FLAML AutoML using the project's leak-safe train/test split."""
    from flaml import AutoML

    if mode not in {"classification", "ranking"}:
        raise ValueError("mode must be one of {'classification', 'ranking'}")
    if target not in {"won", "placed"}:
        raise ValueError("target must be one of {'won', 'placed'}")

    payload = prepare_multi_target_data(featured_df)
    feature_cols = list(payload["feature_cols"])

    X_train = pd.DataFrame(payload["X_train"], columns=feature_cols)
    X_test = pd.DataFrame(payload["X_test"], columns=feature_cols)

    y_test_bin = payload["y_test_won"] if target == "won" else payload["y_test_placed"]
    y_test_rel = payload["y_test_rel"]
    groups_test = payload["groups_test"]
    race_ids = payload["test_df"]["race_id"].values

    automl = AutoML()
    fit_settings: dict[str, Any] = {
        "time_budget": int(max(10, time_budget)),
        "seed": int(getattr(config, "RANDOM_SEED", 42)),
        "verbose": int(max(0, verbose)),
        "log_training_metric": bool(log_training_metric),
    }
    if log_file_name:
        os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
        fit_settings["log_file_name"] = str(log_file_name)

    logger.info(
        "Starting FLAML run: mode=%s target=%s time_budget=%ss metric=%s estimators=%s verbose=%s",
        mode,
        target,
        fit_settings.get("time_budget"),
        fit_settings.get("metric"),
        fit_settings.get("estimator_list"),
        fit_settings.get("verbose"),
    )

    if mode == "classification":
        y_train = payload["y_train_won"] if target == "won" else payload["y_train_placed"]
        fit_settings["task"] = "classification"
        fit_settings["metric"] = "log_loss" if metric == "auto" else metric
        fit_settings["estimator_list"] = estimator_list or [
            "lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree",
        ]
        automl.fit(X_train=X_train, y_train=y_train.astype(int), **fit_settings)

        try:
            pred_proba = automl.predict_proba(X_test)
            y_prob = np.asarray(pred_proba)[:, -1]
        except Exception:
            # Fallback for estimators without predict_proba
            y_pred_label = np.asarray(automl.predict(X_test)).astype(int)
            y_prob = np.clip(y_pred_label.astype(float), 1e-9, 1 - 1e-9)

    else:
        y_train_rank = payload["y_train_rel"].astype(float)
        fit_settings["task"] = "rank"
        fit_settings["metric"] = "ndcg" if metric == "auto" else metric
        fit_settings["estimator_list"] = estimator_list or ["lgbm", "xgboost", "xgb_limitdepth"]
        fit_settings["groups"] = payload["groups_train"].astype(int).tolist()
        automl.fit(X_train=X_train, y_train=y_train_rank, **fit_settings)

        rank_scores = np.asarray(automl.predict(X_test)).reshape(-1)
        y_prob = _grouped_softmax(rank_scores, race_ids)

    y_prob = np.clip(np.asarray(y_prob).reshape(-1), 1e-9, 1 - 1e-9)
    y_true = y_test_bin.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics: dict[str, float | None] = {
        "brier": _safe_float(brier_score_loss(y_true, y_prob)),
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["log_loss"] = _safe_float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        metrics["log_loss"] = None

    try:
        metrics["roc_auc"] = _safe_float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size >= 2 else None
    except Exception:
        metrics["roc_auc"] = None

    metrics.update(_race_level_metrics(race_ids=race_ids, y_true=y_true, y_prob=y_prob))

    if mode == "ranking":
        rank_metrics = _ranking_metrics_from_groups(
            scores=rank_scores,
            y_rel=y_test_rel,
            groups=groups_test,
        )
        metrics.update(rank_metrics)
        # Promote rank-target metrics as primary ranking display metrics.
        metrics["top1_accuracy"] = rank_metrics.get("rank_top1_accuracy")
        metrics["ndcg_at_1"] = rank_metrics.get("rank_ndcg_at_1")

    est_table = _best_estimator_table(automl)
    best_loss = getattr(automl, "best_loss", None)
    best_estimator = getattr(automl, "best_estimator", None)

    logger.info(
        "FLAML run complete: best_estimator=%s best_loss=%s",
        best_estimator,
        best_loss,
    )

    return {
        "automl": automl,
        "mode": mode,
        "target": target,
        "training_target": "relevance_labels" if mode == "ranking" else str(target),
        "best_estimator": str(best_estimator) if best_estimator is not None else None,
        "best_loss": _safe_float(best_loss),
        "best_config": getattr(automl, "best_config", None),
        "best_estimator_table": est_table,
        "metrics": metrics,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(len(feature_cols)),
        "log_file_name": fit_settings.get("log_file_name"),
        "settings": {
            "time_budget": int(max(10, time_budget)),
            "metric": fit_settings.get("metric"),
            "task": fit_settings.get("task"),
            "estimators": list(fit_settings.get("estimator_list") or []),
            "verbose": fit_settings.get("verbose"),
            "log_training_metric": fit_settings.get("log_training_metric"),
        },
    }
