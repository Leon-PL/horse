"""
Run Store — persist and manage training-run snapshots.
=====================================================
Each training run is saved as a folder under ``data/runs/<run_id>/``
containing:

- **meta.json** — all JSON-serialisable metadata (name, timestamp,
  metrics, test_analysis stats, calibration, hyperparameters, etc.)
- **bets.csv** — the full bets DataFrame from ``analyse_test_set``.
- **curves.csv** — cumulative P&L curves DataFrame.
- **model.joblib** — a snapshot of the model artifact so it can be
  restored later.

The module exposes a small set of functions that the Streamlit app
(and nothing else) needs to call.
"""

from __future__ import annotations

import json
import os
import shutil
import logging
from datetime import datetime

import pandas as pd

import config

logger = logging.getLogger(__name__)

RUNS_DIR = os.path.join(config.DATA_DIR, "runs")


# ── helpers ──────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(obj):
    """Make an object JSON-safe (numpy scalars, etc.)."""
    import numpy as np

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
        return str(obj)
    return obj


# ── public API ───────────────────────────────────────────────────────

def save_run(
    *,
    name: str,
    model_type: str,
    data_source: str,
    data_rows: int,
    n_features: int,
    elapsed_seconds: float,
    hyperparameters: dict | None = None,
    ensemble_weights: dict | None = None,
    metrics: dict | None = None,
    train_metrics: dict | None = None,
    test_analysis: dict | None = None,
    auto_tune: dict | None = None,
    training_config: dict | None = None,
    wf_report: dict | None = None,
) -> str:
    """
    Persist a complete training run to disk.

    Returns the ``run_id`` string (e.g. ``"20250715_143022"``).
    """
    run_id = _make_run_id()
    run_dir = os.path.join(RUNS_DIR, run_id)
    _ensure_dir(run_dir)

    # ── Separate DataFrames from test_analysis ───────────────────
    bets_df: pd.DataFrame | None = None
    curves_df: pd.DataFrame | None = None
    predictions_df: pd.DataFrame | None = None
    ta_serialisable: dict | None = None

    if test_analysis is not None:
        ta_serialisable = {}
        for k, v in test_analysis.items():
            if k == "bets" and isinstance(v, pd.DataFrame):
                bets_df = v
            elif k == "curves" and isinstance(v, pd.DataFrame):
                curves_df = v
            elif k == "predictions" and isinstance(v, pd.DataFrame):
                predictions_df = v
            else:
                ta_serialisable[k] = v

    # ── Build meta dict ──────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "data_source": data_source,
        "data_rows": data_rows,
        "n_features": n_features,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "hyperparameters": hyperparameters or {},
        "ensemble_weights": ensemble_weights or {},
        "metrics": metrics or {},
        "train_metrics": train_metrics or {},
        "test_analysis": ta_serialisable or {},
        "auto_tune": auto_tune,
        "training_config": training_config or {},
    }

    # ── Write files ──────────────────────────────────────────────
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(_safe_json(meta), f, indent=2, default=str)

    if bets_df is not None and not bets_df.empty:
        bets_df.to_csv(os.path.join(run_dir, "bets.csv"), index=False)

    if curves_df is not None and not curves_df.empty:
        curves_df.to_csv(os.path.join(run_dir, "curves.csv"), index=False)

    if predictions_df is not None and not predictions_df.empty:
        predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    # ── Walk-forward data ────────────────────────────────────────
    if wf_report is not None:
        wf_summary = wf_report.get("summary")
        if isinstance(wf_summary, pd.DataFrame) and not wf_summary.empty:
            wf_summary.to_csv(os.path.join(run_dir, "wf_summary.csv"), index=False)
        wf_bets = wf_report.get("bets")
        if isinstance(wf_bets, pd.DataFrame) and not wf_bets.empty:
            wf_bets.to_csv(os.path.join(run_dir, "wf_bets.csv"), index=False)
        wf_curves = wf_report.get("curves")
        if isinstance(wf_curves, pd.DataFrame) and not wf_curves.empty:
            wf_curves.to_csv(os.path.join(run_dir, "wf_curves.csv"), index=False)

    # ── Snapshot model artifact ──────────────────────────────────
    _snapshot_model(run_dir)

    logger.info(f"Run {run_id} ({name}) saved to {run_dir}")
    return run_id


def list_runs() -> list[dict]:
    """
    Return a list of run metadata dicts, newest first.

    Each dict is the full ``meta.json`` content so the UI can render
    summary tables without loading DataFrames.
    """
    if not os.path.isdir(RUNS_DIR):
        return []

    runs: list[dict] = []
    for entry in sorted(os.listdir(RUNS_DIR), reverse=True):
        meta_path = os.path.join(RUNS_DIR, entry, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    runs.append(json.load(f))
            except Exception:
                logger.warning(f"Skipping corrupt run dir: {entry}")
    return runs


def load_run(run_id: str) -> dict:
    """
    Load a full run snapshot, including bets/curves DataFrames.

    Returns a dict with keys:
    - All keys from ``meta.json``
    - ``bets_df``  — ``pd.DataFrame`` or ``None``
    - ``curves_df`` — ``pd.DataFrame`` or ``None``
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No run found with id {run_id}")

    with open(meta_path, "r") as f:
        data = json.load(f)

    bets_path = os.path.join(run_dir, "bets.csv")
    data["bets_df"] = (
        pd.read_csv(bets_path) if os.path.isfile(bets_path) else None
    )

    curves_path = os.path.join(run_dir, "curves.csv")
    data["curves_df"] = (
        pd.read_csv(curves_path) if os.path.isfile(curves_path) else None
    )

    predictions_path = os.path.join(run_dir, "predictions.csv")
    data["predictions_df"] = (
        pd.read_csv(predictions_path) if os.path.isfile(predictions_path) else None
    )

    # Walk-forward DataFrames
    wf_summary_path = os.path.join(run_dir, "wf_summary.csv")
    wf_bets_path = os.path.join(run_dir, "wf_bets.csv")
    wf_curves_path = os.path.join(run_dir, "wf_curves.csv")
    data["wf_summary_df"] = (
        pd.read_csv(wf_summary_path) if os.path.isfile(wf_summary_path) else None
    )
    data["wf_bets_df"] = (
        pd.read_csv(wf_bets_path) if os.path.isfile(wf_bets_path) else None
    )
    data["wf_curves_df"] = (
        pd.read_csv(wf_curves_path) if os.path.isfile(wf_curves_path) else None
    )

    return data


def load_run_meta(run_id: str) -> dict:
    """
    Load run metadata only (meta.json), without bets/curves CSVs.

    Use this for lightweight reads where only metrics/config are needed.
    """
    run_dir = os.path.join(RUNS_DIR, run_id)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No run found with id {run_id}")

    with open(meta_path, "r") as f:
        return json.load(f)


def delete_run(run_id: str) -> bool:
    """Delete a run folder. Returns ``True`` if successfully removed."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
        logger.info(f"Run {run_id} deleted")
        return True
    return False


def rename_run(run_id: str, new_name: str) -> bool:
    """Update the display name stored in a run's meta.json.  Returns ``True`` on success."""
    meta_path = os.path.join(RUNS_DIR, run_id, "meta.json")
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["name"] = new_name.strip()
    with open(meta_path, "w") as f:
        json.dump(_safe_json(meta), f, indent=2, default=str)
    logger.info(f"Run {run_id} renamed to '{meta['name']}'")
    return True


def get_latest_run_id() -> str | None:
    """Return the ``run_id`` of the most recent run, or ``None``."""
    runs = list_runs()
    return runs[0]["run_id"] if runs else None


# ── Model snapshot helpers ───────────────────────────────────────────

# The canonical model files written by TripleEnsemblePredictor.save()
_MODEL_FILENAMES = [
    "triple_ensemble_models.joblib",
    "rank_ensemble_models.joblib",
    "ranker_model.joblib",
]


def _snapshot_model(run_dir: str) -> None:
    """
    Copy the active model file and processed dataset into *run_dir*
    so predictions remain reproducible across subsequent training runs.
    """
    import shutil as _shutil

    # ── Model snapshot ────────────────────────────────────────────
    for fname in _MODEL_FILENAMES:
        src = os.path.join(config.MODELS_DIR, fname)
        if os.path.isfile(src):
            _shutil.copy2(src, os.path.join(run_dir, "model.joblib"))
            logger.info(f"  Model snapshot saved ({fname})")
            break
    else:
        logger.warning("No model file found to snapshot.")

    # ── Processed-data snapshot ───────────────────────────────────
    # This ensures feature engineering uses the same history the model
    # was trained on, even after future training runs overwrite the
    # global processed_races file.
    proc_pq = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
    proc_csv = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.csv")
    if os.path.isfile(proc_pq):
        _shutil.copy2(proc_pq, os.path.join(run_dir, "processed_races.parquet"))
        logger.info("  Processed-data snapshot saved (parquet)")
    elif os.path.isfile(proc_csv):
        _shutil.copy2(proc_csv, os.path.join(run_dir, "processed_races.csv"))
        logger.info("  Processed-data snapshot saved (csv)")


def restore_run_model(run_id: str) -> bool:
    """
    Copy a run's ``model.joblib`` back to the global models directory.

    Returns ``True`` if the model was restored successfully.
    """
    import shutil as _shutil

    run_dir = os.path.join(RUNS_DIR, run_id)
    snapshot = os.path.join(run_dir, "model.joblib")
    if not os.path.isfile(snapshot):
        logger.warning(f"Run {run_id} has no model snapshot.")
        return False

    # Figure out which canonical filename to restore to.
    # Peek inside the joblib to detect the model type.
    import joblib

    data = joblib.load(snapshot)
    if isinstance(data, dict) and "ltr_model" in data:
        dest_name = "triple_ensemble_models.joblib"
    elif isinstance(data, dict) and "ranker" in data:
        dest_name = "rank_ensemble_models.joblib"
    else:
        dest_name = "ranker_model.joblib"

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    dest = os.path.join(config.MODELS_DIR, dest_name)
    _shutil.copy2(snapshot, dest)
    logger.info(f"Model from run {run_id} restored to {dest}")
    return True


def run_has_model(run_id: str) -> bool:
    """Check whether a run has a saved model snapshot."""
    return os.path.isfile(os.path.join(RUNS_DIR, run_id, "model.joblib"))


def get_run_processed_path(run_id: str) -> str | None:
    """Return the path to the processed-data snapshot for *run_id*, or ``None``."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    pq = os.path.join(run_dir, "processed_races.parquet")
    csv = os.path.join(run_dir, "processed_races.csv")
    if os.path.isfile(pq):
        return pq
    if os.path.isfile(csv):
        return csv
    return None
