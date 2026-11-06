"""Bind deployment refits to a separately evaluated, unchanged recipe."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import config

ROOT = Path(config.BASE_DIR)
RECIPE_SOURCES = ("src/model.py", "src/race_objectives.py", "src/probability_calibration.py", "src/utils.py",
                  "src/each_way.py",
                  "src/forecast_evaluation.py", "src/wf_eval.py", "src/training.py",
                  "src/store_refresh.py", "src/parquet_io.py", "src/deployment_validation.py")
RECIPE_OPTIONS = ("frameworks", "seed", "hyperparams", "win_calibration_method",
                  "train_window_days", "blend_market_odds")


def file_sha256(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def training_recipe(cfg) -> dict:
    """Record effective settings as well as sources; runtime overrides matter."""
    from src.model import _resolve_feature_allowlist

    names = set()
    hashes = {}
    for relative in RECIPE_SOURCES:
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        hashes[relative] = file_sha256(path)
        names.update(re.findall(r"config\.([A-Z][A-Z0-9_]*)", source))
        names.update(re.findall(r"getattr\(config,\s*['\"]([A-Z][A-Z0-9_]*)['\"]", source))
    # Locations and worker limits do not define the statistical recipe.
    names = {n for n in names if not n.endswith("_DIR") and n != "FEATURE_ALLOWLIST"}
    allowlist = _resolve_feature_allowlist()
    result = {
        "options": {name: getattr(cfg, name) for name in RECIPE_OPTIONS},
        "feature_allowlist": sorted(allowlist) if allowlist is not None else None,
        "model_config": {name: getattr(config, name, None) for name in sorted(names)},
        "source_sha256": hashes,
    }
    return json.loads(json.dumps(result, sort_keys=True, default=str))


def _run_paths(run_id):
    from src import run_store

    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_id)):
        raise ValueError("Validation run ID must identify one saved run")
    base = Path(run_store.RUNS_DIR) / run_id
    return base / "meta.json", base / "model.joblib"


def _evaluated_run(run_id):
    from src.run_store import load_run_meta

    _run_paths(run_id)
    meta = load_run_meta(run_id)
    training = meta.get("training_config") or {}
    split = training.get("split") or {}
    if split.get("mode") == "fit_as_of" or split.get("holdout_races", 0) <= 0:
        raise ValueError("Validation reference must have a separate chronological holdout")
    for role in ("win_classifier", "place_classifier"):
        scores = (meta.get("metrics") or {}).get(role) or {}
        keys = ("brier_score", "log_loss") if role == "win_classifier" else ("brier_calibrated",)
        if not all(isinstance(scores.get(key), (int, float)) and math.isfinite(scores[key])
                   for key in keys):
            raise ValueError(f"Validation run has no held-out {role} probability scores")
    if not (training.get("signatures") or {}).get("fe_signature"):
        raise ValueError("Validation run has no feature provenance")
    return meta


def write_validation_report(run_id, path, *, evidence_paths=()) -> dict:
    """Create a portable certificate from a newly recorded evaluation recipe."""
    meta = _evaluated_run(run_id)
    recipe = meta["training_config"].get("recipe")
    if not recipe:
        raise ValueError("Legacy run needs an explicit audit-backed recipe certificate")
    from src.model import RacePredictor

    predictor = RacePredictor()
    metadata_path, artifact_path = _run_paths(run_id)
    predictor.load(str(artifact_path))
    paths = [metadata_path, artifact_path, *map(Path, evidence_paths)]
    report = {
        "schema_version": 1, "validation_run_id": run_id, "recipe": recipe,
        "features": predictor.feature_cols,
        "evidence_sha256": {str(p.resolve()): file_sha256(p) for p in paths},
        "scope": "Historical recipe validation; not performance of a subsequent deployment refit",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def load_validation_report(path, recipe) -> dict:
    """Fail before collection/fitting when evidence or the locked recipe differs."""
    from src.model import RacePredictor

    source = Path(path).resolve()
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("schema_version") != 1 or report.get("recipe") != recipe:
        raise ValueError("Deployment recipe differs from its validation report")
    meta = _evaluated_run(report.get("validation_run_id"))
    recorded_recipe = meta["training_config"].get("recipe")
    if recorded_recipe and recorded_recipe != recipe:
        raise ValueError("Validation run records a different training recipe")
    reference_config = meta["training_config"].get("config") or {}
    if any(reference_config.get(name) != value for name, value in recipe["options"].items()):
        raise ValueError("Deployment settings differ from the evaluated run")
    evidence = report.get("evidence_sha256") or {}
    if not evidence:
        raise ValueError("Validation report has no frozen evidence")
    expected = {str(Path(name).resolve()): value for name, value in evidence.items()}
    metadata_path, artifact_path = _run_paths(report["validation_run_id"])
    if any(str(p.resolve()) not in expected for p in (metadata_path, artifact_path)):
        raise ValueError("Validation report must fingerprint the reference metadata and model")
    if any(file_sha256(name) != value for name, value in expected.items()):
        raise ValueError("Validation evidence changed after qualification")
    predictor = RacePredictor()
    predictor.load(str(artifact_path))
    if predictor.clf_model is None or predictor.place_model is None:
        raise ValueError("Validation reference must contain fitted win and place models")
    if predictor.feature_cols != report.get("features"):
        raise ValueError("Validation feature list differs from its saved artifact")
    if (predictor.frameworks != recipe["options"]["frameworks"]
            or predictor.random_seed != recipe["options"]["seed"]
            or predictor.win_calibration_method != recipe["options"]["win_calibration_method"]):
        raise ValueError("Validation artifact differs from the declared recipe")
    return {"report_path": str(source), "report_sha256": hashlib.sha256(raw).hexdigest(),
            "validation_run_id": report["validation_run_id"], "features": report["features"],
            "fe_signature": meta["training_config"]["signatures"]["fe_signature"],
            "evidence_sha256": evidence, "scope": report.get("scope")}
