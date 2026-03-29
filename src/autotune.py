from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.data_collector import collect_data
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import TripleEnsemblePredictor, prepare_multi_target_data

logger = logging.getLogger(__name__)

AUTOTUNE_DIR = os.path.join(config.DATA_DIR, "autotune")
AUTOTUNE_MODEL_INFO = {
    "classifier": {"label": "Win Classifier", "training_metric": "LogLoss"},
    "place": {"label": "Place Classifier", "training_metric": "LogLoss"},
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_json(obj):
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "study"


def autotune_session_dir(session_id: str) -> str:
    return os.path.join(AUTOTUNE_DIR, session_id)


def autotune_manifest_path(session_id: str) -> str:
    return os.path.join(autotune_session_dir(session_id), "manifest.json")


def study_storage_path(session_id: str, model_key: str) -> str:
    return os.path.join(autotune_session_dir(session_id), f"study_{model_key}.db")


def study_storage_url(session_id: str, model_key: str) -> str:
    return f"sqlite:///{study_storage_path(session_id, model_key)}"


def _write_manifest(manifest: dict) -> None:
    _ensure_dir(autotune_session_dir(manifest["session_id"]))
    manifest["updated_at"] = datetime.now().isoformat()
    with open(autotune_manifest_path(manifest["session_id"]), "w", encoding="utf-8") as f:
        json.dump(_safe_json(manifest), f, indent=2, default=str)


def load_autotune_session(session_id: str) -> dict | None:
    path = autotune_manifest_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_autotune_sessions() -> list[dict]:
    _ensure_dir(AUTOTUNE_DIR)
    sessions: list[dict] = []
    for entry in os.listdir(AUTOTUNE_DIR):
        manifest = load_autotune_session(entry)
        if manifest is not None:
            sessions.append(manifest)
    sessions.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return sessions


def delete_autotune_session(session_id: str) -> bool:
    """Delete an autotune session directory and all its contents."""
    import shutil
    session_dir = autotune_session_dir(session_id)
    if not os.path.isdir(session_dir):
        return False
    shutil.rmtree(session_dir)
    logger.info(f"Deleted autotune session {session_id}")
    return True


def build_autotune_dataset(
    *,
    data_source: str,
    days_back: int | None = None,
    num_races: int = 1500,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    if data_source in {"database", "scrape"}:
        raw = collect_data(source=data_source, days_back=int(days_back or 90))
    else:
        raw = collect_data(source="sample", num_races=int(num_races))
    processed = process_data(df=raw)
    featured = engineer_features(processed, save=False)
    return featured, processed, dataset_meta_from_frame(
        featured,
        data_source=data_source,
        requested_days=days_back if data_source != "sample" else None,
        origin="autotune_fresh_build",
    )


def dataset_meta_from_frame(
    df: pd.DataFrame,
    *,
    data_source: str | None,
    requested_days: int | None,
    origin: str,
) -> dict:
    dates = pd.to_datetime(df["race_date"], errors="coerce") if "race_date" in df.columns else pd.Series(dtype="datetime64[ns]")
    date_min = dates.min() if not dates.empty else pd.NaT
    date_max = dates.max() if not dates.empty else pd.NaT
    actual_days = None
    months = None
    if pd.notna(date_min) and pd.notna(date_max):
        actual_days = int((date_max - date_min).days) + 1
        months = int(dates.dt.to_period("M").nunique())
    return {
        "data_source": data_source,
        "requested_days": int(requested_days) if requested_days is not None else None,
        "actual_days": actual_days,
        "date_start": date_min.date().isoformat() if pd.notna(date_min) else None,
        "date_end": date_max.date().isoformat() if pd.notna(date_max) else None,
        "months": months,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "origin": origin,
    }


def create_autotune_session(
    *,
    name: str,
    dataset_meta: dict,
    frameworks: dict[str, str],
    models: list[str],
    n_trials: int,
    n_folds: int = 3,
) -> dict:
    _ensure_dir(AUTOTUNE_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_{_slugify(name)}"
    manifest = {
        "session_id": session_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": "created",
        "dataset_meta": dataset_meta,
        "frameworks": dict(frameworks),
        "models": list(models),
        "target_trials": int(n_trials),
        "target_folds": int(n_folds),
        "summaries": {},
        "best_params": {},
    }
    _write_manifest(manifest)
    return manifest


def _build_phase1b_payload(
    featured_df: pd.DataFrame,
    frameworks: dict[str, str],
    n_folds: int = 3,
) -> tuple[TripleEnsemblePredictor, dict]:
    predictor = TripleEnsemblePredictor(frameworks=dict(frameworks))
    data = prepare_multi_target_data(featured_df)

    return predictor, _build_autotune_payload(data, n_folds=n_folds)


def _build_autotune_payload(data: dict, n_folds: int = 3) -> dict:
    groups_train = data["groups_train"]
    train_dates = pd.to_datetime(data["train_race_dates"])
    cum_g = np.cumsum(groups_train)
    race_starts_idx = np.concatenate([[0], cum_g[:-1]])
    race_dates_idx = pd.DatetimeIndex(train_dates.values[race_starts_idx])
    n_races = len(race_dates_idx)
    if n_races < 20:
        raise ValueError("Not enough races to build a leak-safe autotune split. Prepare a larger dataset first.")

    purge_gap = int(getattr(config, "PURGE_DAYS", 7))
    date_min = race_dates_idx.min()
    date_max = race_dates_idx.max()
    total_span = max((date_max - date_min).days, 1)
    chunk_days = total_span / (max(int(n_folds), 1) + 1)
    boundaries = [
        date_min + pd.Timedelta(days=int(chunk_days * i))
        for i in range(max(int(n_folds), 1) + 2)
    ]
    boundaries[-1] = date_max + pd.Timedelta(days=1)

    cv_folds: list[dict] = []
    for fold_idx in range(max(int(n_folds), 1)):
        val_start_dt = boundaries[fold_idx + 1]
        val_end_dt = boundaries[fold_idx + 2]
        purge_cutoff_dt = val_start_dt - pd.Timedelta(days=purge_gap)

        train_mask = race_dates_idx <= purge_cutoff_dt
        val_mask = (race_dates_idx >= val_start_dt) & (race_dates_idx < val_end_dt)
        train_races = np.where(train_mask)[0]
        val_races = np.where(val_mask)[0]
        if len(train_races) < 20 or len(val_races) < 5:
            continue

        train_end = int(cum_g[train_races[-1]])
        val_begin = int(race_starts_idx[val_races[0]])
        val_end = int(cum_g[val_races[-1]])

        cv_folds.append({
            "fold_index": fold_idx + 1,
            "X_train": data["X_train"][:train_end],
            "X_val": data["X_train"][val_begin:val_end],
            "groups_train": groups_train[:train_races[-1] + 1],
            "groups_val": groups_train[val_races[0]:val_races[-1] + 1],
            "sw_train": data["sample_weight_train"][:train_end],
            "train_dates": pd.to_datetime(data["train_df"]["race_date"].iloc[:train_end]),
            "targets_train": {
                "rel": data["y_train_rel"][:train_end],
                "lb": data["y_train_lb"][:train_end],
                "won": data["y_train_won"][:train_end],
                "resid": data["y_train_resid"][:train_end],
                "placed": data["y_train_placed"][:train_end],
                "norm_pos": data["y_train_norm_pos"][:train_end],
                "fp": data["fp_train"][:train_end],
                "ip": data["ip_train"][:train_end],
            },
            "targets_val": {
                "rel": data["y_train_rel"][val_begin:val_end],
                "lb": data["y_train_lb"][val_begin:val_end],
                "won": data["y_train_won"][val_begin:val_end],
                "resid": data["y_train_resid"][val_begin:val_end],
                "placed": data["y_train_placed"][val_begin:val_end],
                "norm_pos": data["y_train_norm_pos"][val_begin:val_end],
                "fp": data["fp_train"][val_begin:val_end],
                "ip": data["ip_train"][val_begin:val_end],
            },
            "summary": {
                "fold": fold_idx + 1,
                "train_races": int(len(groups_train[:train_races[-1] + 1])),
                "validation_races": int(len(groups_train[val_races[0]:val_races[-1] + 1])),
                "train_rows": int(train_end),
                "validation_rows": int(val_end - val_begin),
                "purged_rows": int(val_begin - train_end),
                "val_start": str(val_start_dt.date()),
                "val_end": str((val_end_dt - pd.Timedelta(days=1)).date()),
            },
        })

    return {
        "data": data,
        "cv_folds": cv_folds,
        "split_summary": {
            "outer_train_races": int(len(data["groups_train"])),
            "outer_test_races": int(len(data["groups_test"])),
            "outer_train_rows": int(len(data["X_train"])),
            "outer_test_rows": int(len(data["X_test"])),
            "purge_days": purge_gap,
            "cv_folds": len(cv_folds),
        },
        "cv_fold_summaries": [fold["summary"] for fold in cv_folds],
    }


def run_autotune_session(
    *,
    session_id: str,
    featured_df: pd.DataFrame,
    frameworks: dict[str, str],
    models: list[str],
    n_trials: int,
    n_folds: int = 3,
    progress_callback=None,
) -> dict:
    manifest = load_autotune_session(session_id)
    if manifest is None:
        raise FileNotFoundError(f"Unknown autotune session: {session_id}")

    if progress_callback is not None:
        progress_callback(
            "setup",
            {
                "message": "Building purged walk-forward folds",
                "target_folds": int(n_folds),
            },
        )

    predictor, payload = _build_phase1b_payload(featured_df, frameworks, n_folds=n_folds)
    if not payload.get("cv_folds"):
        raise ValueError("No valid purged walk-forward folds could be built for autotuning. Increase the dataset window or reduce fold count.")

    anchor_fold = payload["cv_folds"][0]

    manifest["status"] = "running"
    manifest["frameworks"] = dict(frameworks)
    manifest["models"] = list(models)
    manifest["target_trials"] = int(n_trials)
    manifest["target_folds"] = int(n_folds)
    manifest["split_summary"] = payload["split_summary"]
    manifest["cv_fold_summaries"] = payload.get("cv_fold_summaries", [])
    _write_manifest(manifest)

    summaries = dict(manifest.get("summaries") or {})
    best_params = dict(manifest.get("best_params") or {})

    for model_index, model_key in enumerate(models, start=1):
        if model_key not in AUTOTUNE_MODEL_INFO:
            continue

        if progress_callback is not None:
            progress_callback(
                "model_start",
                {
                    "model_key": model_key,
                    "model_index": model_index,
                    "model_total": len(models),
                    "cv_folds": len(payload.get("cv_folds", [])),
                },
            )

        def _trial_cb(trial_num: int, total: int, score: float, *, _model_key=model_key, _model_index=model_index):
            if progress_callback is not None:
                progress_callback(
                    "trial",
                    {
                        "model_key": _model_key,
                        "model_index": _model_index,
                        "model_total": len(models),
                        "trial_num": int(trial_num),
                        "trial_total": int(total),
                        "score": float(score),
                        "cv_folds": len(payload.get("cv_folds", [])),
                    },
                )

        result = predictor._auto_tune_model(
            model_key,
            anchor_fold["X_train"],
            anchor_fold["X_val"],
            anchor_fold["targets_train"],
            anchor_fold["targets_val"],
            anchor_fold["groups_train"],
            anchor_fold["groups_val"],
            sw_train=anchor_fold["sw_train"],
            train_dates=anchor_fold["train_dates"],
            n_trials=int(n_trials),
            callback=_trial_cb,
            storage=study_storage_url(session_id, model_key),
            study_name=f"{session_id}_{model_key}",
            load_if_exists=True,
            folds=payload.get("cv_folds"),
        )
        summaries[model_key] = result
        best_params[model_key] = result.get("best_params", {})
        manifest["summaries"] = summaries
        manifest["best_params"] = best_params
        _write_manifest(manifest)

    manifest["status"] = "complete"
    _write_manifest(manifest)
    if progress_callback is not None:
        progress_callback(
            "complete",
            {
                "session_id": session_id,
                "models": list(models),
                "cv_folds": len(payload.get("cv_folds", [])),
            },
        )
    return manifest


def load_optuna_study(session_id: str, model_key: str):
    import optuna

    return optuna.load_study(
        study_name=f"{session_id}_{model_key}",
        storage=study_storage_url(session_id, model_key),
    )


def build_config_snippet(manifest: dict) -> str:
    model_to_config_key = {
        "classifier": "CLASSIFIER_PARAMS",
        "place": "PLACE_CLASSIFIER_PARAMS",
    }
    lines = []
    summaries = manifest.get("summaries") or {}
    for model_key in manifest.get("models", []):
        result = summaries.get(model_key) or {}
        params = result.get("best_params") or {}
        config_key = model_to_config_key.get(model_key)
        if not config_key or not params:
            continue
        score = result.get("best_score")
        metric_name = result.get("metric_name", "score")
        lines.append(f"# {AUTOTUNE_MODEL_INFO[model_key]['label']} ({metric_name} = {score})")
        lines.append(f"{config_key} = {{")
        for key, value in params.items():
            if isinstance(value, float):
                lines.append(f'    "{key}": {value:.6g},')
            else:
                lines.append(f'    "{key}": {value},')
        lines.append("}")
        lines.append("")
    return "\n".join(lines).strip()