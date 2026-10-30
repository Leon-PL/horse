"""One diagnostic historical comparison, then normal deployment of mandatory fixes.

Feature definitions are the ordinary main-pipeline definitions. Verified errors
are not retained on the strength of a retrospective score. This script only
records the comparison, fits matching models and preserves restart checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.assess_forecast_parity import upper_bound
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.canonical_training_cache import CheckpointPredictor
from src.deployment_validation import (RECIPE_SOURCES, file_sha256 as sha,
    training_recipe, write_validation_report, load_validation_report)
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, placed_mask
from src.run_store import save_run, promote, get_production_run_id, RUNS_DIR
from src.training import TrainingConfig

OUTPUT = ROOT / "data/audit/canonical_repairs_026"
PARENT = "20261028_084608_624611"


def write(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def native_component_semantics(cfg):
    """Settings of the normal native estimator calls, excluding data/report paths."""
    return {"schema":"native_estimator_cache_v1", "seed":cfg.seed, "threads":2,
        "frameworks":cfg.frameworks, "win_calibration_method":cfg.win_calibration_method,
        "params":cfg.hyperparams, "config_sha256":sha(ROOT / "config.py"),
        "runtime":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scikit-learn","lightgbm","joblib")},
        "sources":{p:sha(ROOT/p) for p in RECIPE_SOURCES}}


def verified_historical_cache_scope(status, cfg):
    """Transfer exact estimator fits only from matching recorded learner settings."""
    scope = status.get("historical_component_cache_scope")
    if scope is None:
        return None
    semantics = status.get("historical_component_cache_semantics")
    if semantics != native_component_semantics(cfg) or canonical_hash(semantics) != scope:
        raise ValueError("Historical component reuse has different learner settings/source/runtime")
    return scope


def prepare():
    status = json.loads((OUTPUT / "rebuild_status.json").read_text())
    if status["stage"] != "rebuilt" or status["fe_signature"] != fe_signature():
        raise ValueError("Finish the canonical corrected rebuild first")
    updated = {name: {"rebuild_sha256": digest, "current_sha256": sha(ROOT / name)}
               for name, digest in status["source_sha256"].items() if sha(ROOT / name) != digest}
    if set(updated)-{"src/training.py"}:
        raise ValueError("Source changed after rebuilding outside the reviewed training-field safeguard")
    sources = list(dict.fromkeys([*_FE_SOURCE_FILES, *RECIPE_SOURCES, "src/live_prediction.py", "src/data_scraper.py",
        "config.py", "feature_sets/win.txt", "scripts/integrate_canonical_repairs.py", "scripts/canonical_training_cache.py",
        *status.get("extra_source_paths", [])]))
    label = status.get("label", "026 canonical correctness")
    cfg = TrainingConfig(name=label+" reference", refresh_data=False, threads=2)
    cache_scope = verified_historical_cache_scope(status, cfg)
    if (OUTPUT / "manifest.json").exists():
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        verify(manifest, cfg)
        return manifest, cfg
    teacher = status.get("teacher_inputs_sha256")
    if teacher is None:
        teacher = json.loads((ROOT / "data/audit/model_improvement/022_full_capacity/manifest.json").read_text())["teacher_inputs_sha256"]
    inputs = list(dict.fromkeys(["corrected_inputs.parquet", "baseline_2026Q1_predictions.parquet",
        *status.get("extra_input_paths", []), *([status["reference_win_predictions"]] if status.get("reference_win_predictions") else [])]))
    manifest = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "recipe": training_recipe(cfg), "label":label,
        "fe_signature": fe_signature(), "features": status["features"], "data_signature": status["store_meta"]["data_signature"],
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "input_sha256": {name: sha(OUTPUT / name) for name in inputs},
        "reference_sha256": {name: sha(ROOT / name) for name in ("reference/ecb_gbp_per_eur.csv",)},
        "teacher_inputs_sha256": teacher,
        "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "numba")},
        "post_rebuild_source_updates": updated,
        "historical_cutoff": "2025-12-22", "test_period": "2026Q1", "production_cutoff": status["store_meta"]["last_date"],
        "adoption_policy": status.get("adoption_policy", "User-mandated verified correctness repairs become canonical regardless of diagnostic retrospective score; no new feature design or calibration selection"), "parent_run": PARENT}
    if cache_scope is not None:
        manifest["historical_component_cache_scope"] = cache_scope
        manifest["historical_component_cache_semantics"] = status["historical_component_cache_semantics"]
    if status.get("reference_win_predictions"):
        manifest["reference_win_predictions"] = status["reference_win_predictions"]
    write(OUTPUT / "manifest.json", manifest)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
    verify(manifest, cfg)
    return manifest, cfg


def verify(manifest, cfg):
    if fe_signature() != manifest["fe_signature"] or training_recipe(cfg) != manifest["recipe"]:
        raise ValueError("Corrected feature/training definitions changed")
    verified_historical_cache_scope(manifest, cfg)
    for name, digest in manifest["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen canonical source changed: {name}")
    for field in ("reference_sha256", "teacher_inputs_sha256"):
        for name, digest in manifest[field].items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Frozen input changed: {name}")
    for name, digest in manifest["input_sha256"].items():
        if sha(OUTPUT / name) != digest:
            raise ValueError("Frozen corrected feature inputs changed")
    for name, version in manifest["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Native training runtime changed")


def outputs(predictor, frame):
    result = predictor.predict_races(frame)
    return frame[["race_id", "horse_name"]].merge(result[["race_id", "horse_name", "win_probability", "place_probability"]],
        on=["race_id", "horse_name"], how="left", sort=False, validate="one_to_one")


def qualify(manifest, cfg):
    certificate = OUTPUT / "validation.json"
    if certificate.exists():
        return load_validation_report(certificate, manifest["recipe"])
    frame = pd.read_parquet(OUTPUT / "corrected_inputs.parquet")
    original = pd.read_parquet(OUTPUT / "baseline_2026Q1_predictions.parquet")
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    test = frame.loc[frame.race_id.isin(original.race_id)].copy().reset_index(drop=True)
    keys = ["race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap"]
    pd.testing.assert_frame_equal(test[keys], original[keys], check_dtype=False)
    # Release later history before fitting. Its outcomes are not part of this
    # historical reference, and retaining it needlessly raises the memory peak.
    frame = frame.loc[frame.race_date.le(manifest["historical_cutoff"])].copy()
    gc.collect()
    path = OUTPUT / "historical_model.joblib"
    predictor = CheckpointPredictor(checkpoint_directory=OUTPUT / "components/historical",
        checkpoint_manifest=manifest.get("historical_component_cache_scope") or sha(OUTPUT / "manifest.json"),
        random_seed=cfg.seed, frameworks=cfg.frameworks, win_calibration_method=cfg.win_calibration_method)
    predictor._autotune_njobs = 2
    started = time.perf_counter()
    if path.exists():
        predictor.load(str(path))
        if predictor.eval_split_info.get("canonical_manifest_sha256") != sha(OUTPUT / "manifest.json"):
            raise ValueError("Historical checkpoint belongs to another pipeline")
    else:
        write(OUTPUT / "status.json", {"stage": "historical_reference_fit", "cutoff": manifest["historical_cutoff"]})
        predictor.fit_as_of(frame, manifest["historical_cutoff"], params=cfg.hyperparams,
                            dataset_start=frame.race_date.min(), win_only=False)
        predictor.eval_split_info["canonical_manifest_sha256"] = sha(OUTPUT / "manifest.json")
        verify(manifest, cfg)
        predictor.save(str(path))
    probabilities = outputs(predictor, test)
    reference_error = None
    if manifest.get("reference_win_predictions"):
        prior = pd.read_parquet(OUTPUT / manifest["reference_win_predictions"])
        aligned = test[["race_id", "horse_name", "race_date"]].merge(
            prior[["race_id", "horse_name", "race_date", "candidate_probability"]],
            on=["race_id", "horse_name", "race_date"], how="left", validate="one_to_one")
        if aligned.candidate_probability.isna().any():
            raise ValueError("Native win-only reference misses heldout runners")
        reference_error = float(np.max(np.abs(probabilities.win_probability.to_numpy()-aligned.candidate_probability.to_numpy())))
        if reference_error > 1e-12:
            raise ValueError("Canonical win probabilities differ from the checked native win-only model")
    scored = test[keys].assign(baseline_probability=original.win_probability.to_numpy(),
        candidate_probability=probabilities.win_probability.to_numpy(), baseline_place_probability=original.place_probability.to_numpy(),
        candidate_place_probability=probabilities.place_probability.to_numpy())
    scored.to_parquet(OUTPUT / "historical_predictions.parquet", index=False)
    comparison = score_pair(scored)
    nr, hcap = test.num_runners.to_numpy(), test.handicap.to_numpy().astype(bool)
    slots = np.where(nr<=4, 3, np.where(nr<=7, 2, np.where(nr<=15, 3, np.where(hcap, 4, 3))))
    y = placed_mask(test.finish_position.to_numpy(), slots)
    place_change = np.square(scored.candidate_place_probability-y)-np.square(scored.baseline_place_probability-y)
    _, matrix = predictor._prepare_prediction_frame(test)
    raw_place = predictor.place_model.predict_proba(matrix)[:, 1]
    place_metrics = predictor._evaluate_place_model(scored.candidate_place_probability.to_numpy(), raw_place,
        test, test.groupby("race_id", sort=False).size().to_numpy())
    report = {"win": comparison, "place": place_metrics,
        "reference_win_reproduction_max_error":reference_error,
        "place_brier_delta_race_upper95": upper_bound(place_change.to_numpy(), test.race_id.to_numpy(), seed=42),
        "place_brier_delta_date_upper95": upper_bound(place_change.to_numpy(), test.race_date.astype(str).to_numpy(), seed=42),
        "policy": manifest["adoption_policy"], "scope": "One already inspected historical development period; no reserved/future performance evaluation"}
    write(OUTPUT / "comparison.json", report)
    metrics = {"win_classifier": {"brier_score": comparison["candidate"]["brier"], "log_loss": comparison["candidate"]["binary_log_loss"],
        "race_log_loss": comparison["candidate"]["race_log_loss"]}, "place_classifier": place_metrics}
    validation_run = save_run(predictor=predictor, name=manifest.get("label", "026 canonical correctness")+": separate Q1 validation", model_type="race_predictor",
        data_source="Main corrected pipeline; whole original Q1 test fields", data_rows=predictor.eval_split_info["training_runners"],
        n_features=len(predictor.feature_cols), elapsed_seconds=time.perf_counter()-started, metrics=metrics,
        training_config={"config": cfg.to_dict(), "recipe": manifest["recipe"], "signatures": {"fe_signature": manifest["fe_signature"]},
            "split": {"mode": "external_walk_forward", "holdout_races": int(test.race_id.nunique()),
                "holdout_first_date": str(test.race_date.min().date()), "holdout_last_date": str(test.race_date.max().date()),
                "native_fit": dict(predictor.eval_split_info)}})
    evidence_names = list(dict.fromkeys(["manifest.json", "source_snapshot.zip", "corrected_inputs.parquet",
        "comparison.json", "historical_predictions.parquet", *manifest.get("input_sha256", {})]))
    write_validation_report(validation_run, certificate, evidence_paths=[OUTPUT / name for name in evidence_names])
    return load_validation_report(certificate, manifest["recipe"])


def deploy(manifest, cfg, validation):
    from src import training
    status_path = OUTPUT / "deployment_status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {}
    run_id = state.get("run_id")
    if not run_id:
        if get_production_run_id() != PARENT:
            raise ValueError("Production changed during canonical repair work")
        deployment_cfg = TrainingConfig.deployment(OUTPUT / "validation.json", manifest["production_cutoff"],
            name=manifest.get("label", "026 canonical corrected features")+": full win/place refit", refresh_data=False, threads=2)
        # Execution-only checkpoints wrap the ordinary training core; all feature
        # preparation, labels, fits, calibration, provenance and saves stay native.
        ordinary = training.RacePredictor
        training.RacePredictor = partial(CheckpointPredictor, checkpoint_directory=OUTPUT / "components/current",
            checkpoint_manifest=sha(OUTPUT / "manifest.json"))
        write(status_path, {"stage": "normal_full_refit", "as_of": manifest["production_cutoff"]})
        try:
            run_id = training.train_run(deployment_cfg)
        finally:
            training.RacePredictor = ordinary
        write(status_path, {"stage": "saved", "run_id": run_id})
    verify(manifest, cfg)
    predictor = RacePredictor(); predictor.load(str(Path(RUNS_DIR) / run_id / "model.joblib"))
    if predictor.feature_cols != validation["features"]:
        raise ValueError("Latest model uses different corrected inputs")
    if not promote(run_id):
        raise ValueError("Corrected model and canonical history disagree")
    write(status_path, {"stage": "installed", "run_id": run_id, "model_sha256": sha(Path(RUNS_DIR) / run_id / "model.joblib"),
        "fe_signature": manifest["fe_signature"], "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Mandatory verified correctness corrections; separate historical recipe scores"})
    print(status_path.read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--case-directory", type=Path, help="Prepared canonical repair batch; previous batches stay immutable")
    parser.add_argument("--parent-run", help="Installed parent required for a new repair batch")
    args = parser.parse_args()
    if args.case_directory is not None:
        if not args.parent_run:
            parser.error("--case-directory requires an explicit --parent-run")
        OUTPUT = args.case_directory.resolve()
        PARENT = args.parent_run
    manifest, cfg = prepare()
    reference = qualify(manifest, cfg)
    if args.deploy:
        deploy(manifest, cfg, reference)
