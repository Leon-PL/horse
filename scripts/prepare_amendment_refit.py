"""Transfer exact-generation validation evidence, then fit corrected data once.

The historical reference retains its original model bytes, scores and source
labels. Only its exactly qualified feature-generation identity is migrated.
The new current-data fit uses ordinary train_run in process-local staged data
directories. No canonical store, production pointer or recorder is changed.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from src.deployment_validation import file_sha256 as sha, load_validation_report, training_recipe, write_validation_report
from src.featured_dataset import _model_load_columns, data_signature, fe_signature
from src.model import RacePredictor
from src.run_store import RUNS_DIR, get_production_run_id, load_run_meta, save_run
from src.training import TrainingConfig

CASE = ROOT / "data/audit/result_amendments_042/consolidated"
OUTPUT = CASE / "refit"
SPEED = ROOT / "data/audit/live_feature_speed_20261029/installation"


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare():
    OUTPUT.mkdir(exist_ok=False)
    qualification = json.loads((SPEED / "qualification.json").read_text())
    activation = json.loads((SPEED / "activation.json").read_text())
    migration_plan = json.loads((SPEED / "plan.json").read_text())
    parent_run = get_production_run_id()
    if (not qualification["passed"] or not qualification["all_values_and_dtypes_exact"]
            or not qualification["processed_exact"] or parent_run != activation["run_id"]
            or fe_signature() != activation["fe_signature"]):
        raise ValueError("Finish the exact feature-generation qualification before transferring validation")
    parent = load_run_meta(parent_run)
    current_cfg = TrainingConfig(**parent["training_config"]["config"])
    recipe = training_recipe(current_cfg)
    if parent["training_config"]["recipe"] != recipe:
        raise ValueError("Production statistical recipe changed")
    parent_artifact = Path(RUNS_DIR) / parent_run / "model.joblib"
    if sha(parent_artifact) != activation["model_sha256"]:
        raise ValueError("The qualified migration's trained artifact changed")
    reference_path = Path(parent["training_config"]["validation_reference"]["report_path"])
    reference = load_validation_report(reference_path, recipe)
    if reference["fe_signature"] != migration_plan["before_fe_signature"]:
        raise ValueError("The old validation belongs to another feature generation")
    original_id = reference["validation_run_id"]
    original = load_run_meta(original_id)
    original_path = Path(RUNS_DIR) / original_id / "model.joblib"
    model = RacePredictor()
    model.load(str(original_path))
    if len(model.feature_cols) != 154 or sorted(model.feature_cols) != recipe["feature_allowlist"]:
        raise ValueError("The qualified model's selected inputs changed")
    tc = deepcopy(original["training_config"])
    tc["signatures"]["fe_signature"] = fe_signature()
    tc["feature_generation_migration"] = {"source_run": original_id, "source_fe_signature": reference["fe_signature"],
        "target_fe_signature": fe_signature(), "qualification_path": str(SPEED / "qualification.json"),
        "qualification_sha256": sha(SPEED / "qualification.json"), "new_model_fits": 0,
        "scope": "Same historical model bytes, data identity, original predictions/scores and pre042 labels; full stored-feature equivalence qualifies only the generation identity"}
    migrated_id = save_run(predictor=model, name="042 validation reference: exact speed-generation transfer; original pre-repair scores",
        model_type=original["model_type"], data_source=original["data_source"], data_rows=original["data_rows"],
        n_features=original["n_features"], elapsed_seconds=0, metrics=deepcopy(original["metrics"]),
        train_metrics=deepcopy(original.get("train_metrics")), training_config=tc)
    migrated_path = Path(RUNS_DIR) / migrated_id / "model.joblib"
    shutil.copy2(original_path, migrated_path)
    if sha(migrated_path) != sha(original_path):
        raise ValueError("Migration altered the historical reference model")
    certificate = OUTPUT / "validation.json"
    evidence = [Path(name) for name in reference["evidence_sha256"]]
    evidence += [reference_path, SPEED / "qualification.json", SPEED / "plan.json", SPEED / "activation.json",
                 CASE / "audit.json", CASE / "audit_plan.json"]
    write_validation_report(migrated_id, certificate, evidence_paths=evidence)
    cfg = TrainingConfig.deployment(certificate, "2026-10-28", name="042 official-result corrections: one current win/place refit", threads=2)
    if training_recipe(cfg) != recipe:
        raise ValueError("Current-data fitting would change the validated statistical recipe")
    verified = load_validation_report(certificate, recipe)
    if verified["fe_signature"] != fe_signature():
        raise ValueError("Transferred certificate does not qualify the current generation")
    write(OUTPUT / "plan.json", {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run": parent_run, "parent_model_sha256": sha(parent_artifact),
        "historical_reference": original_id, "migrated_reference": migrated_id,
        "reference_model_bytes_identical": True, "recipe": recipe, "config": cfg.to_dict(),
        "fe_signature": fe_signature(), "certificate_sha256": sha(certificate),
        "script_sha256": sha(Path(__file__)), "new_reference_fits": 0,
        "scope": "One unchanged-recipe current refit on corrected histories; no new historical score or predictive-gain claim"})
    print("Exact-generation validation transfer qualified; one staged current-data refit prepared.", flush=True)


def fit():
    from src import training
    from src.feature_store import FeatureStore
    plan = json.loads((OUTPUT / "final_plan.json").read_text())
    if (OUTPUT / "fitted.json").exists():
        raise ValueError("Preserve the completed current-data fit")
    if sha(Path(__file__)) != plan["script_sha256"] or fe_signature() != plan["fe_signature"]:
        raise ValueError("Pinned implementation changed")
    cfg = TrainingConfig(**plan["config"])
    if training_recipe(cfg) != plan["recipe"]:
        raise ValueError("Statistical recipe changed after preparation")
    built = json.loads((CASE / "history_rebuild/rebuilt.json").read_text())
    if built["fe_signature"] != plan["fe_signature"]:
        raise ValueError("Corrected histories belong to another generation")
    staged_processed = CASE / "history_rebuild"
    staged_raw = CASE / "staged_history_v2"
    original_store = FeatureStore().meta()
    # Route the ordinary processing/store seams, preserving config: directory
    # settings are included in the feature signature even for private inputs.
    import pandas as pd
    private = FeatureStore(str(staged_processed / "feature_store"))
    ordinary_processing = training.process_data

    def process_staged_history(*, save=False):
        if save:
            raise ValueError("Private fitting must not write normal processed files")
        return ordinary_processing(df=pd.read_csv(staged_raw / "race_results.csv"), save=False)

    def load_staged_features(*, processed):
        with private._access():
            store_meta = private.meta()
            if (store_meta["fe_signature"] != fe_signature()
                    or len(processed) != store_meta["processed_rows"]
                    or data_signature(processed) != store_meta["data_signature"]):
                raise ValueError("Prepared features differ from normal processing of the corrected source")
            return private.load_featured(columns=_model_load_columns(private))

    with patch.object(training, "process_data", process_staged_history), patch.object(training, "get_featured_dataset", load_staged_features):
        run_id = training.train_run(cfg)
    meta = load_run_meta(run_id)
    if meta["training_config"]["signatures"]["fe_signature"] != plan["fe_signature"]:
        raise ValueError("Saved current model has another feature generation")
    if meta["training_config"]["recipe"] != plan["recipe"] or training_recipe(cfg) != plan["recipe"]:
        raise ValueError("Saved current model has another statistical recipe")
    if FeatureStore().meta() != original_store:
        raise ValueError("The canonical history changed while fitting the staged model; review before publication")
    if meta["metrics"]:
        raise ValueError("A current-data refit must not claim held-out scores")
    write(OUTPUT / "fitted.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "model_sha256": sha(Path(RUNS_DIR) / run_id / "model.joblib"),
        "parent_run_at_preparation": plan["parent_run"], "fe_signature": plan["fe_signature"],
        "split": meta["training_config"]["split"], "signatures": meta["training_config"]["signatures"],
        "canonical_writes": False, "promoted": False, "scope": plan["scope"]})
    print("Corrected current-data win/place model saved; production remains unchanged pending coherent installation.", flush=True)


def seal():
    """Use immutable local receipts after the other rollout's final update."""
    initial = json.loads((OUTPUT / "plan.json").read_text())
    completed = json.loads((SPEED / "final_status.json").read_text())
    if (not completed["installed_in_actual_project"] or not completed["deep_readiness_passed"]
            or get_production_run_id() != initial["parent_run"] or fe_signature() != initial["fe_signature"]):
        raise ValueError("Finish the other rollout before sealing the refit evidence")
    snapshots = OUTPUT / "pinned_speed_evidence"
    snapshots.mkdir(exist_ok=False)
    for name in ("qualification.json", "plan.json", "activation.json", "final_status.json"):
        shutil.copy2(SPEED / name, snapshots / name)
    parent = load_run_meta(initial["parent_run"])
    original_certificate = Path(parent["training_config"]["validation_reference"]["report_path"])
    original = load_validation_report(original_certificate, initial["recipe"])
    evidence = [Path(name) for name in original["evidence_sha256"]]
    evidence += [original_certificate, *snapshots.glob("*.json"), CASE / "audit.json", CASE / "audit_plan.json"]
    certificate = OUTPUT / "validation_final.json"
    write_validation_report(initial["migrated_reference"], certificate, evidence_paths=evidence)
    cfg = TrainingConfig.deployment(certificate, initial["config"]["fit_as_of"],
        name=initial["config"]["name"], threads=2)
    if training_recipe(cfg) != initial["recipe"]:
        raise ValueError("Statistical recipe changed while sealing the evidence")
    verified = load_validation_report(certificate, initial["recipe"])
    if verified["fe_signature"] != fe_signature():
        raise ValueError("Final certificate has another feature generation")
    final = deepcopy(initial)
    final.update({"sealed_at_utc": datetime.now(timezone.utc).isoformat(), "config": cfg.to_dict(),
        "certificate_sha256": sha(certificate), "script_sha256": sha(Path(__file__)),
        "supersedes": "Initial certificate pinned the other rollout's still-changing activation receipt. It is preserved but is not used for fitting. Final evidence is copied locally after completion; no model or historical score changed.",
        "pinned_speed_evidence_sha256": {p.name: sha(p) for p in snapshots.glob("*.json")}})
    write(OUTPUT / "final_plan.json", final)
    print("Completed rollout evidence frozen locally; final refit certificate verifies.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=["prepare", "seal", "fit"])
    {"prepare": prepare, "seal": seal, "fit": fit}[cli.parse_args().phase]()
