"""Qualify an already-saved refit if the recorder only appended a newer tail.

No fitting. The native fit driver deliberately refuses a concurrent canonical
metadata change after saving. This recovery proves that the earlier canonical
prefix is unchanged and the model's private corrected prefix/recipe still match.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.deployment_validation import file_sha256 as sha, training_recipe
from src.featured_dataset import data_signature, fe_signature
from src.feature_store import FeatureStore
from src.model import RacePredictor
from src.run_store import RUNS_DIR, load_run_meta, get_production_run_id
from src.training import TrainingConfig

CASE = ROOT / "data/audit/result_amendments_042/consolidated"
OUTPUT = CASE / "refit"


def main():
    if (OUTPUT / "fitted.json").exists():
        raise ValueError("The completed refit is already qualified")
    saved = json.loads((OUTPUT / "run_saved.json").read_text())
    execution = json.loads((OUTPUT / "execution_plan.json").read_text())
    if any(sha(ROOT / name) != digest for name, digest in execution["runtime_model_sources"].items()):
        raise ValueError("Runtime probability math changed; do not recover this automatically")
    if sha(ROOT / "data/processed/betfair_bsp.parquet") != execution["teacher_source_sha256"]:
        raise ValueError("Market-teacher source changed; inspect the completed fit")
    plan = json.loads((OUTPUT / "final_plan.json").read_text())
    attempts = sorted((OUTPUT / "attempts").glob("*.json"))
    if not attempts:
        raise ValueError("No captured fit-start source state")
    attempt = json.loads(attempts[-1].read_text())
    cfg = TrainingConfig(**plan["config"])
    meta = load_run_meta(saved["run_id"])
    artifact = Path(RUNS_DIR) / saved["run_id"] / "model.joblib"
    if (sha(artifact) != saved["model_sha256"] or training_recipe(cfg) != plan["recipe"]
            or meta["training_config"]["recipe"] != plan["recipe"] or meta["metrics"]
            or fe_signature() != plan["fe_signature"] or get_production_run_id() != plan["parent_run"]):
        raise ValueError("Saved model, statistical recipe or installed parent changed")
    prior = attempt["canonical_store_at_start"]
    current = FeatureStore()
    actual = pd.read_parquet(current.processed_path,
        filters=[("race_date", "<=", pd.Timestamp(prior["last_date"]))])
    if len(actual) != prior["processed_rows"] or data_signature(actual) != prior["data_signature"]:
        raise ValueError("The canonical historical prefix changed; this is not a harmless append")
    del actual
    private = FeatureStore(str(CASE / "history_rebuild/feature_store"))
    fitted_prefix = pd.read_parquet(private.processed_path,
        filters=[("race_date", "<=", pd.Timestamp(cfg.fit_as_of))])
    if data_signature(fitted_prefix) != meta["training_config"]["signatures"]["data_signature"]:
        raise ValueError("Private fitted source prefix changed")
    model = RacePredictor()
    model.load(str(artifact))
    certificate = json.loads(Path(cfg.validation_report).read_text())
    if model.feature_cols != certificate["features"] or model.clf_model is None or model.place_model is None:
        raise ValueError("Saved refit is incomplete or has different selected inputs")
    if model.eval_split_info.get("as_of") != cfg.fit_as_of:
        raise ValueError("Saved model has another settled cutoff")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": saved["run_id"],
        "model_sha256": saved["model_sha256"], "parent_run_at_preparation": plan["parent_run"],
        "fe_signature": plan["fe_signature"], "split": meta["training_config"]["split"],
        "signatures": meta["training_config"]["signatures"], "canonical_writes": False, "promoted": False,
        "new_fits_during_recovery": 0, "recovery": "Canonical source prefix unchanged; only newer tail/metadata advanced during native fitting",
        "fit_start_receipt_sha256": sha(attempts[-1]), "scope": plan["scope"]}
    with (OUTPUT / "fitted.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    with (OUTPUT / "execution_completed.json").open("x", encoding="utf-8") as stream:
        json.dump({"completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_plan_sha256": sha(OUTPUT / "execution_plan.json"),
            "fitted_receipt_sha256": sha(OUTPUT / "fitted.json"),
            "runtime_model_sources": execution["runtime_model_sources"], "runtime_sources_unchanged": True,
            "recovered_without_fitting": True}, stream, indent=2)
    print("Already-saved current refit qualified after a harmless source append; zero repeated fits.", flush=True)


if __name__ == "__main__":
    main()
