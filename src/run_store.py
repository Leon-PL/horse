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
- **featured_races.parquet** — the exact featured dataset used to train
    the saved model, when available.
- **processed_races.parquet** — the exact processed history used to train
    the saved model, when available.

Training, promotion and serving code call these functions; there is no UI.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

import config

if TYPE_CHECKING:
    from src.model import RacePredictor

logger = logging.getLogger(__name__)

RUNS_DIR = os.path.join(config.DATA_DIR, "runs")


# ── helpers ──────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _allocate_run_dir() -> tuple[str, str]:
    """Reserve one directory exclusively, even when two clocks return one ID."""
    _ensure_dir(RUNS_DIR)
    base_id = _make_run_id()
    suffix = 0
    while True:
        run_id = base_id if suffix == 0 else f"{base_id}_{suffix:04d}"
        run_dir = os.path.join(RUNS_DIR, run_id)
        try:
            os.mkdir(run_dir)
        except FileExistsError:
            suffix += 1
        else:
            return run_id, run_dir


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
    featured_df: pd.DataFrame | None = None,
    processed_df: pd.DataFrame | None = None,
    predictor: RacePredictor | None = None,
) -> str:
    """
    Persist a complete training run to disk.

    Pass the trained ``predictor`` to serialize that exact object. Omitting it
    retains the legacy global-model snapshot behavior. Metadata is published
    last, so registry readers never see a partially saved run.

    Returns the unique ``run_id`` string.
    """

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
    run_id, run_dir = _allocate_run_dir()
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
    try:
        if bets_df is not None and not bets_df.empty:
            bets_df.to_csv(os.path.join(run_dir, "bets.csv"), index=False)

        if curves_df is not None and not curves_df.empty:
            curves_df.to_csv(os.path.join(run_dir, "curves.csv"), index=False)

        if predictions_df is not None and not predictions_df.empty:
            predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

        # ── Walk-forward data ────────────────────────────────────
        if wf_report is not None:
            for key in ("summary", "bets", "curves"):
                frame = wf_report.get(key)
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    frame.to_csv(os.path.join(run_dir, f"wf_{key}.csv"), index=False)

        # Serialize the caller's predictor, never a shared mutable checkpoint.
        if predictor is not None:
            predictor.save(os.path.join(run_dir, "model.joblib"))
        else:
            _snapshot_model(run_dir)
        _snapshot_datasets(run_dir, featured_df=featured_df, processed_df=processed_df)

        temporary_meta = os.path.join(run_dir, "meta.json.tmp")
        with open(temporary_meta, "w", encoding="utf-8") as f:
            json.dump(_safe_json(meta), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_meta, os.path.join(run_dir, "meta.json"))
    except BaseException:
        # This is our exclusively allocated directory; other saves and the
        # currently promoted run are never cleanup targets.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

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
    """Delete an unpromoted run; the production artifact is always retained."""
    if run_id == get_production_run_id():
        logger.warning("Refusing to delete production run %s; promote another run first.", run_id)
        return False
    root = os.path.realpath(RUNS_DIR)
    run_dir = os.path.realpath(os.path.join(root, run_id))
    if os.path.dirname(run_dir) != root:
        raise ValueError("run_id must identify one run directory")
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


def run_disk_usage() -> list[tuple[str, int]]:
    """Return ``(run_id, bytes)`` per run folder, newest first."""
    if not os.path.isdir(RUNS_DIR):
        return []
    usage: list[tuple[str, int]] = []
    for entry in sorted(os.listdir(RUNS_DIR), reverse=True):
        run_dir = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(run_dir):
            continue
        total = sum(
            os.path.getsize(os.path.join(run_dir, f))
            for f in os.listdir(run_dir)
            if os.path.isfile(os.path.join(run_dir, f))
        )
        usage.append((entry, total))
    return usage


def prune_runs(keep_latest: int = 5) -> list[str]:
    """Delete all but the *keep_latest* most recent runs.

    Run folders sort chronologically by id (``YYYYMMDD_HHMMSS``).
    The production run is retained even when it is older than this limit.
    Returns the list of deleted run ids.
    """
    if keep_latest < 1:
        raise ValueError("keep_latest must be >= 1")
    if not os.path.isdir(RUNS_DIR):
        return []
    # A folder without published metadata may still be saving its artifacts.
    run_ids = [run["run_id"] for run in list_runs()]
    deleted: list[str] = []
    for run_id in run_ids[keep_latest:]:
        if delete_run(run_id):
            deleted.append(run_id)
    return deleted


# ── Model snapshot helpers ───────────────────────────────────────────

# The canonical model file written by RacePredictor.save()
_MODEL_FILENAMES = [
    "triple_ensemble_models.joblib",
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



def _snapshot_datasets(
    run_dir: str,
    *,
    featured_df: pd.DataFrame | None = None,
    processed_df: pd.DataFrame | None = None,
) -> None:
    """Persist the exact datasets used for the training run when available.

    Snapshots are downcast to compact numeric dtypes and written with
    zstd compression — featured datasets run to hundreds of MB per run
    otherwise.
    """
    from src.utils import compact_numeric_dtypes

    if isinstance(featured_df, pd.DataFrame) and not featured_df.empty:
        featured_path = os.path.join(run_dir, "featured_races.parquet")
        snapshot = compact_numeric_dtypes(featured_df.copy(), label="featured snapshot")
        snapshot.to_parquet(featured_path, index=False, compression="zstd")
        logger.info("  Featured-data snapshot saved (parquet)")

    if isinstance(processed_df, pd.DataFrame) and not processed_df.empty:
        processed_path = os.path.join(run_dir, "processed_races.parquet")
        snapshot = compact_numeric_dtypes(processed_df.copy(), label="processed snapshot")
        snapshot.to_parquet(processed_path, index=False, compression="zstd")
        logger.info("  Processed-data snapshot saved (parquet)")


def run_has_model(run_id: str) -> bool:
    """Check whether a run has a saved model snapshot."""
    return os.path.isfile(os.path.join(RUNS_DIR, run_id, "model.joblib"))


# ── Production pointer (the model the serving app loads) ──────────────
# Instead of copying a run's model file into MODELS_DIR to "activate" it, we
# write a tiny JSON pointer naming the promoted run. The serving app reads the
# pointer and loads that run's model.joblib directly — no file shuffling, and
# the experimental runs stay untouched.

PRODUCTION_POINTER = os.path.join(config.MODELS_DIR, "production.json")


# ── Model ↔ feature-generation contract ───────────────────────────────
# A model is only meaningful against features built by the same FE code it was
# trained on. ``train_run`` records the ``fe_signature`` of the build it
# consumed; the live path (``live_prediction.features_for_live_cards``) already
# refuses to featurise when the store disagrees with the current FE code. The
# missing link was the model itself: without this check a model trained on one
# feature generation can be served against another, which produces confidently
# wrong probabilities and no error. The project has already lost months to this
# class of bug (the 2026-06 stale-store "place edge", the trainer_freq leak),
# so the mismatch is surfaced rather than discovered in the P&L.

GENERATION_MATCH = "match"
GENERATION_MISMATCH = "mismatch"
GENERATION_UNKNOWN = "unknown"      # run predates signature recording
GENERATION_NO_MODEL = "no_model"


def run_signatures(run_id: str) -> dict:
    """The data/FE signatures recorded when *run_id* was trained (may be empty)."""
    try:
        meta = load_run_meta(run_id)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    config_block = meta.get("training_config")
    if not isinstance(config_block, dict):
        return {}
    signatures = config_block.get("signatures")
    return signatures if isinstance(signatures, dict) else {}


def store_fe_signature() -> str | None:
    """The ``fe_signature`` stamped on the feature store, or ``None``.

    Reads the store's ``meta.json`` directly — a few hundred bytes — so the
    serving app can check the contract without touching the 2 GB parquet.
    """
    from src.feature_store import FeatureStore

    return FeatureStore().meta().get("fe_signature")


def feature_generation_status(run_id: str | None = None) -> dict:
    """Does *run_id* (default: production) match the store's feature generation?

    Returns ``{"state", "run_id", "run_fe_signature", "store_fe_signature",
    "message"}``. ``state`` is one of the ``GENERATION_*`` constants;
    ``unknown`` means the run was trained before signatures were recorded and
    the match cannot be verified either way.
    """
    rid = run_id if run_id is not None else get_production_run_id()
    if not rid:
        return {
            "state": GENERATION_NO_MODEL, "run_id": None,
            "run_fe_signature": None, "store_fe_signature": None,
            "message": "No model is promoted to production.",
        }

    run_sig = run_signatures(rid).get("fe_signature")
    try:
        store_sig = store_fe_signature()
    except Exception as error:
        busy = isinstance(error, RuntimeError) and "FeatureStore is in use by another operation" in str(error)
        if not busy:
            logger.warning("Could not read the feature-store signature", exc_info=True)
        return {"state": GENERATION_UNKNOWN, "run_id": rid,
                "run_fe_signature": run_sig, "store_fe_signature": None,
                "store_state": "busy" if busy else "unreadable", "store_error": str(error),
                "message": ("Feature history is in use by another operation. Retry when it finishes."
                            if busy else f"Could not read feature-history metadata: {error}. Check access and retry.")}

    result = {
        "state": GENERATION_UNKNOWN, "run_id": rid,
        "run_fe_signature": run_sig, "store_fe_signature": store_sig,
        "store_state": "ready" if store_sig else "unsigned",
        "message": "",
    }
    if not run_sig:
        result["message"] = (
            f"Run {rid} records no feature signature (trained before signatures "
            "were tracked), so it cannot be verified against the current "
            "feature store. Retrain to make it checkable."
        )
    elif not store_sig:
        result["message"] = (
            "The feature store carries no signature; rebuild it on the training "
            "app's Data page before trusting live picks."
        )
    elif run_sig == store_sig:
        result["state"] = GENERATION_MATCH
        result["message"] = f"Run {rid} matches the feature store ({run_sig})."
        # A calibration/helper edit can alter probabilities without changing
        # any engineered feature. New runs may pin those runtime sources;
        # check their actual files before permitting serving or promotion.
        metadata = load_run_meta(rid)
        recorded = (metadata.get("training_config") or {}).get("runtime_model_sources")
        if recorded:
            changed = []
            if not isinstance(recorded, dict):
                changed.append("unreadable probability-code evidence")
            else:
                root = Path(config.BASE_DIR).resolve()
                for relative, digest in recorded.items():
                    try:
                        source = (root / relative).resolve()
                        if Path(relative).is_absolute() or not source.is_relative_to(root):
                            changed.append(str(relative))
                            continue
                        with source.open("rb") as stream:
                            actual = hashlib.file_digest(stream, "sha256").hexdigest()
                        if actual != digest:
                            changed.append(str(relative))
                    except (OSError, TypeError, ValueError):
                        changed.append(str(relative))
            if changed:
                result["state"] = GENERATION_MISMATCH
                result["runtime_source_mismatches"] = changed
                result["message"] = (f"Probability code changed since run {rid} was recorded: " + ", ".join(changed)
                    + ". Restore that code or validate and publish a compatible model.")
    else:
        result["state"] = GENERATION_MISMATCH
        result["message"] = (
            f"Run {rid} was trained on feature generation {run_sig} but the "
            f"store now holds {store_sig}. Live features would not match the "
            "model's training features — retrain and promote before betting."
        )
    return result


def _run_model_type(run_id: str):
    """Older artifact-only checkpoints have no metadata to identify a type."""
    try:
        return load_run_meta(run_id).get("model_type")
    except FileNotFoundError:
        return None


def promote(run_id: str, *, allow_feature_mismatch: bool = False) -> bool:
    """Mark *run_id* as the production model. Returns ``True`` on success.

    Refuses a run whose feature generation disagrees with the store, or cannot
    be verified against it, because serving it would score live features the
    model may never have seen. Pass ``allow_feature_mismatch=True`` to override
    deliberately (for example to promote a model while a rebuild is in flight).

    Unverified runs used to promote with only a log warning. On 2026-10-13
    every saved run predated the ``trainer_freq`` leak fix and six of eight
    carried no signature, so one click would have served a leaky model.
    """
    if not run_has_model(run_id):
        logger.warning("Cannot promote %s: no model snapshot.", run_id)
        return False
    if _run_model_type(run_id) == "win_stack_candidate":
        logger.warning("Refusing to promote %s: win-only modelling candidate has not qualified the serving/place contract", run_id)
        return False
    status = feature_generation_status(run_id)
    if (status["state"] in (GENERATION_MISMATCH, GENERATION_UNKNOWN)
            and not allow_feature_mismatch):
        logger.warning("Refusing to promote %s: %s", run_id, status["message"])
        return False
    if status["state"] in (GENERATION_UNKNOWN, GENERATION_MISMATCH):
        logger.warning("Promoting %s unverified: %s", run_id, status["message"])
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    # Readers must never observe a partially written pointer.
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(PRODUCTION_POINTER), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(
                {"run_id": run_id, "promoted_at": datetime.now().isoformat(timespec="seconds")},
                fh, indent=2,
            )
        os.replace(temporary, PRODUCTION_POINTER)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    logger.info("Promoted run %s to production", run_id)
    return True


def get_production_run_id() -> str | None:
    """Return the promoted run id, or ``None`` if nothing is promoted."""
    if not os.path.isfile(PRODUCTION_POINTER):
        return None
    try:
        with open(PRODUCTION_POINTER, encoding="utf-8") as fh:
            rid = json.load(fh).get("run_id")
    except (OSError, json.JSONDecodeError):
        return None
    # Guard against a pointer to a deleted/incomplete run.
    return rid if rid and run_has_model(rid) else None


def production_model_revision() -> tuple[str, int, int] | None:
    """Identity for serving caches; changes on promotion or artifact replacement."""
    rid = get_production_run_id()
    if rid is None:
        return None
    try:
        stat = os.stat(os.path.join(RUNS_DIR, rid, "model.joblib"))
    except OSError:
        return None
    return rid, stat.st_mtime_ns, stat.st_size


def production_model_path() -> str | None:
    """Path to the promoted run's model.joblib, or ``None``."""
    rid = get_production_run_id()
    return os.path.join(RUNS_DIR, rid, "model.joblib") if rid else None


def run_model_path(run_id: str) -> str | None:
    """Path to a specific run's model.joblib, or ``None`` if it has none."""
    path = os.path.join(RUNS_DIR, run_id, "model.joblib")
    return path if os.path.isfile(path) else None


def load_run_predictor(run_id: str):
    """Load a saved predictor, or None. Win candidates expose ``predict`` only;
    they are not the production win/place serving adapter.
    """
    path = run_model_path(run_id)
    if not path:
        return None
    if _run_model_type(run_id) == "win_stack_candidate":
        from src.modelling.candidate import WinCandidate
        return WinCandidate.load(path)
    from src.model import RacePredictor

    p = RacePredictor()
    p.load(path)
    return p


def load_production_predictor():
    """Load the promoted model as a ``RacePredictor`` (or ``None``).

    This is the serving app's single entry point — it loads ONLY the lean model
    artifact, never a featured dataset.
    """
    path = production_model_path()
    if not path:
        return None
    from src.model import RacePredictor

    p = RacePredictor()
    p.load(path)
    return p


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


def get_run_featured_path(run_id: str) -> str | None:
    """Return the path to the featured-data snapshot for *run_id*, or ``None``."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    pq = os.path.join(run_dir, "featured_races.parquet")
    csv = os.path.join(run_dir, "featured_races.csv")
    if os.path.isfile(pq):
        return pq
    if os.path.isfile(csv):
        return csv
    return None
