"""Qualify the unchanged 154-input recipe and refit corrected histories once."""
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"data/audit/rtv_repair_features_044"
PROJECT = CASE/"project"
OUTPUT = CASE/"refit"
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
import config
from src.deployment_validation import file_sha256 as sha, training_recipe, write_validation_report, load_validation_report
from src.featured_dataset import fe_signature, _FE_SOURCE_FILES
from src.model import RacePredictor
from src.run_store import save_run, load_run_meta, RUNS_DIR
from src.training import TrainingConfig
from src.source_pipeline import _publish

PARENT = "20261030_015917_823047"


def qualify():
    OUTPUT.mkdir(exist_ok=True)
    rebuilt = json.loads((CASE/"rebuilt.json").read_text())
    if rebuilt["fe_signature"] != fe_signature():
        raise ValueError("Finish rebuilding this exact feature generation")
    parent = json.loads((ROOT/"data/runs"/PARENT/"meta.json").read_text())
    features = parent["training_config"]["validation_reference"]["features"]
    repaired = pd.read_parquet(CASE/"repair_before.parquet")
    first = pd.to_datetime(repaired.race_date).min()
    columns = ["race_id", "horse_name", "race_date", *features]
    columns = list(dict.fromkeys(columns))
    filters = [("race_date", "<", first)]
    old = pd.read_parquet(ROOT/"data/processed/feature_store/featured.parquet", columns=columns, filters=filters)
    new = pd.read_parquet(PROJECT/"data/processed/feature_store/featured.parquet", columns=columns, filters=filters)
    keys = ["race_id", "horse_name"]
    for frame in (old, new):
        frame["race_id"] = frame.race_id.astype(str)
    old, new = (x.sort_values(keys).reset_index(drop=True) for x in (old, new))
    pd.testing.assert_frame_equal(old, new[old.columns], check_dtype=False, check_exact=True)
    _publish(OUTPUT/"qualification.json", {"qualified_at": datetime.now(timezone.utc).isoformat(),
        "fe_signature": fe_signature(), "selected_features": len(features), "prefix_before": first.isoformat(),
        "prefix_rows": len(old), "all_existing_selected_prefix_values_exact": True,
        "scope": "Mandatory source facts change later histories. Added RTV features do not change the retained 154 definitions; historical scores retain their old label version."})
    print({"passed": True, "prefix_rows": len(old), "selected_features": len(features)}, flush=True)


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT/"plan.json").exists():
        raise ValueError("Prepared refit already exists; resume fit rather than making another")
    qualification = json.loads((OUTPUT/"qualification.json").read_text())
    if qualification["fe_signature"] != fe_signature() or not qualification["all_existing_selected_prefix_values_exact"]:
        raise ValueError("Qualify the unchanged selected definitions first")
    parent = json.loads((ROOT/"data/runs"/PARENT/"meta.json").read_text())
    cfg = TrainingConfig(**parent["training_config"]["config"])
    recipe = training_recipe(cfg)
    prior_recipe = parent["training_config"]["recipe"]
    for name in ("options", "feature_allowlist", "model_config"):
        if recipe[name] != prior_recipe[name]:
            raise ValueError("The production statistical recipe changed: "+name)
    original_id = parent["training_config"]["validation_reference"]["validation_run_id"]
    original_path = ROOT/"data/runs"/original_id
    original = json.loads((original_path/"meta.json").read_text())
    model = RacePredictor()
    model.load(str(original_path/"model.joblib"))
    if len(model.feature_cols) != 154 or sorted(model.feature_cols) != recipe["feature_allowlist"]:
        raise ValueError("Historical reference does not match the retained recipe")
    tc = deepcopy(original["training_config"])
    tc["recipe"] = recipe
    tc["signatures"]["fe_signature"] = fe_signature()
    tc["source_repair_044"] = {"historical_reference": original_id,
        "historical_scores_label_version": "pre044; no scores recomputed or relabelled",
        "qualification_path": str(OUTPUT/"qualification.json"), "qualification_sha256": sha(OUTPUT/"qualification.json"),
        "source_data_repair": str(CASE/"prepared_source.json"), "new_reference_fits": 0,
        "scope": "Only the feature generation and source-contract fingerprints migrate; selected statistical settings and historical artifact bytes stay fixed"}
    reference = save_run(predictor=model, name="044 historical reference: unchanged154; original pre044 scores",
        model_type=original["model_type"], data_source=original["data_source"], data_rows=original["data_rows"],
        n_features=original["n_features"], elapsed_seconds=0, metrics=deepcopy(original["metrics"]),
        train_metrics=deepcopy(original.get("train_metrics")), training_config=tc)
    local = Path(RUNS_DIR)/reference
    shutil.copy2(original_path/"model.joblib", local/"model.joblib")
    # The certificate includes both locations, so it remains verifiable after
    # this draft reference and the refit are installed in the main registry.
    installed = ROOT/"data/runs"/reference
    shutil.copytree(local, installed)
    old_report = Path(parent["training_config"]["validation_reference"]["report_path"])
    old_evidence = json.loads(old_report.read_text())["evidence_sha256"]
    evidence = [Path(p) for p in old_evidence]
    evidence += [old_report, OUTPUT/"qualification.json", CASE/"prepared_source.json", CASE/"feature_plan.json",
        CASE/"rebuilt.json", installed/"meta.json", installed/"model.joblib"]
    certificate = OUTPUT/"validation.json"
    write_validation_report(reference, certificate, evidence_paths=evidence)
    cfg = TrainingConfig.deployment(certificate, "2026-10-28", name="044 verified RTV/SL result repairs: current unchanged154 refit", threads=2)
    if training_recipe(cfg) != recipe:
        raise ValueError("Deployment would change the validated statistical recipe")
    load_validation_report(certificate, recipe)
    paths = list(dict.fromkeys([*_FE_SOURCE_FILES, *recipe["source_sha256"], "src/feature_inputs.py",
        "src/source_pipeline.py", "src/rtv_feature_collection.py", "src/live_prediction.py", "src/verified_result_repair.py"]))
    plan = {"prepared_at": datetime.now(timezone.utc).isoformat(), "parent_run": PARENT,
        "parent_model_sha256": sha(ROOT/"data/runs"/PARENT/"model.joblib"), "recipe": recipe,
        "config": cfg.to_dict(), "reference_run": reference, "fe_signature": fe_signature(),
        "reference_model_bytes_identical": sha(local/"model.joblib") == sha(original_path/"model.joblib"),
        "source_sha256": {p: sha(PROJECT/p) for p in paths}, "teacher_sha256": sha(PROJECT/"data/processed/betfair_bsp.parquet"),
        "input_sha256": {str(p): sha(p) for p in (certificate, CASE/"rebuilt.json", CASE/"prepared_source.json", CASE/"feature_plan.json")},
        "scope": "Mandatory corrected-data refit; no new predictive-score claim and no automatic RTV predictor selection"}
    _publish(OUTPUT/"plan.json", plan)
    print({"prepared": True, "reference": reference, "new_reference_fits": 0, "features": 154}, flush=True)


def fit():
    from src import training
    from scripts.canonical_training_cache import CheckpointPredictor
    plan = json.loads((OUTPUT/"plan.json").read_text())
    if (OUTPUT/"fitted.json").exists():
        raise ValueError("The current refit already completed")
    for name, digest in plan["source_sha256"].items():
        if sha(PROJECT/name) != digest:
            raise ValueError("Pinned fitting source changed: "+name)
    if fe_signature() != plan["fe_signature"]:
        raise ValueError("Feature generation changed")
    for name, digest in plan["input_sha256"].items():
        if sha(name) != digest:
            raise ValueError("Pinned fitting input changed")
    if sha(PROJECT/"data/processed/betfair_bsp.parquet") != plan["teacher_sha256"]:
        raise ValueError("Teaching source changed")
    cfg = TrainingConfig(**plan["config"])
    if training_recipe(cfg) != plan["recipe"]:
        raise ValueError("Fitting recipe changed")
    predictor = partial(CheckpointPredictor, checkpoint_directory=OUTPUT/"components", checkpoint_manifest=sha(OUTPUT/"plan.json"))
    with patch.object(training, "RacePredictor", predictor):
        run = training.train_run(cfg)
    metadata = load_run_meta(run)
    for name, digest in plan["source_sha256"].items():
        if sha(PROJECT/name) != digest:
            raise ValueError("Source changed during fitting")
    value = {"completed_at": datetime.now(timezone.utc).isoformat(), "run_id": run,
        "model_sha256": sha(Path(RUNS_DIR)/run/"model.joblib"), "fe_signature": fe_signature(),
        "split": metadata["training_config"]["split"], "signatures": metadata["training_config"]["signatures"],
        "selected_features": metadata["n_features"], "promoted": False, "canonical_writes": False}
    _publish(OUTPUT/"fitted.json", value)
    print(value, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["qualify", "prepare", "fit"])
    {"qualify": qualify, "prepare": prepare, "fit": fit}[parser.parse_args().phase]()
