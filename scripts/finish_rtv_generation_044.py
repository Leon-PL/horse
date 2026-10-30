"""Verify the installed 044 generation and its fresh running recorder."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = ROOT / "data/audit/rtv_repair_features_044"
OUT = CASE / "installation"

import pandas as pd
import psutil
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature, _rtv_signature
from src.forward_store import ForwardStudy
from src.run_store import get_production_run_id
from src.source_pipeline import _publish


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if (OUT / "final_status.json").exists():
        raise ValueError("Preserve the completed installation receipt")
    published, fit = read(OUT / "published.json"), read(CASE / "refit/fitted.json")
    contract = read(CASE / "source_contract/published.json")
    qa = read(OUT / "qa_contract_after/report.json")
    scheduler = read(CASE / "source_contract/scheduler.json")
    ready = read(OUT / "deep_status_final.json")
    if not ready["servable"] or not ready["sections"]["feature_store"].get("data_checked"):
        raise ValueError("Finish full database-content readiness qualification")
    if not qa["passed"] or not qa["staged_installed_inputs_and_probabilities_exact"]:
        raise ValueError("Installed serving parity has not passed")
    settings = read(ROOT / "data/forecasts/recorder.json")
    study = ForwardStudy(settings["study"])
    study.verify()
    heartbeat = read(study.path / "recorder_heartbeat.json")
    now = datetime.now(timezone.utc)
    age = (now - datetime.fromisoformat(heartbeat["observed_at"])).total_seconds()
    discovery_age = (now - datetime.fromisoformat(heartbeat["last_discovery"])).total_seconds()
    if (heartbeat["mode"] != "recording" or not 0 <= age <= 90 or not 0 <= discovery_age <= 1200
            or heartbeat["model_id"] != fit["run_id"] or heartbeat["study"] != settings["study"]
            or not settings["scheduled_task_verified"] or not scheduler["verified"]
            or not scheduler["triggers_preserved"] or scheduler["study"] != settings["study"]):
        raise ValueError("Fresh recorder identity, heartbeat, discovery or retained schedule failed")
    process = psutil.Process(heartbeat["pid"])
    args = process.cmdline()
    if not process.is_running() or "--study" not in args or args[args.index("--study") + 1] != settings["study"]:
        raise ValueError("The verified recorder child is not running for this study")
    artifact = ROOT / "data/runs" / fit["run_id"] / "model.joblib"
    if (get_production_run_id() != fit["run_id"] or published["run_id"] != fit["run_id"]
            or sha(artifact) != fit["model_sha256"] or sha(study.path / "model.joblib") != fit["model_sha256"]):
        raise ValueError("Installed and pinned native model artifacts disagree")
    store = FeatureStore()
    metadata = store.meta()
    model_metadata = read(ROOT / "data/runs" / fit["run_id"] / "meta.json")
    if model_metadata["training_config"]["signatures"]["data_signature"] != metadata["data_signature"]:
        raise ValueError("Normalized model/source history fingerprints disagree")
    if metadata["fe_signature"] != fe_signature() or fe_signature() != fit["fe_signature"]:
        raise ValueError("Installed source/store/model generations differ")
    if metadata["rtv_signature"] != _rtv_signature(metadata["last_date"]):
        raise ValueError("The recorder changed available historical RTV inputs beneath the store")
    raw_hash = sha(ROOT / "data/raw/race_results.csv")
    if raw_hash != contract["raw_sha256"]:
        raise ValueError("Canonical source changed after the qualified publication")
    ids = list(set(pd.read_parquet(CASE / "repair_after.parquet").race_id.astype(str))
               | set(pd.read_parquet(CASE / "new_fields.parquet").race_id.astype(str)))
    with closing(sqlite3.connect((ROOT / "data/races.db").as_uri() + "?mode=ro", uri=True)) as con:
        totals = con.execute("SELECT COUNT(*),COUNT(DISTINCT race_id),MAX(race_date) FROM results").fetchone()
        marker = con.execute("SELECT content_sha256 FROM verified_result_batches WHERE repair_id='rtv_verified_results_044'").fetchone()
        ledger = con.execute("SELECT COUNT(*),SUM(completed),SUM(last_error IS NOT NULL) FROM result_sync_races WHERE race_id IN ("
                             + ",".join("?" for _ in ids) + ")", ids).fetchone()
        excluded = con.execute("SELECT completed,last_error FROM result_sync_races WHERE race_id='933576'").fetchone()
    if (totals != (755357, 80302, "2026-10-29") or marker[0] != published["application"]["content_sha256"]
            or ledger != (89, 89, 0) or excluded != (1, "not run: ARABIAN")):
        raise ValueError("Durable canonical source or reviewed sync ledger differs")
    health = {name: read(ROOT / "data/rtv_cache/pipeline" / (name + "_status.json")) for name in ("history", "snapshot")}
    if any(value["state"] != "ok" for value in health.values()):
        raise ValueError("RTV history/snapshot collection is not healthy")
    for name, value in health.items():
        elapsed = (now - datetime.fromisoformat(value["checked_at"])).total_seconds()
        if not 0 <= elapsed <= (86400 if name == "history" else 3600) or "features" not in value:
            raise ValueError("RTV collection is stale or has not materialised numeric features")
    result = {"completed_at": now.isoformat(), "status": "COMPLETE_REPAIRS_FEATURES_INSTALLATION",
        "run_id": fit["run_id"], "model_sha256": fit["model_sha256"], "fe_signature": fe_signature(),
        "database_rows": totals[0], "database_races": totals[1], "last_date": totals[2], "raw_sha256": raw_hash,
        "repairs": published["application"], "reviewed_sync_fields_completed": 89, "excluded_arab_listing_closed": True,
        "implemented_history_features": 26, "implemented_pre_race_features": 4, "selected_rtv_inputs": 0,
        "mandatory_native_components": 8, "development_screen_fits": 2, "new_accuracy_claim": False,
        "deep_readiness_passed": True, "native_serving_parity_exact": True,
        "training_prefix_signature": qa["training_prefix_signature"], "loaded_source_signature": metadata["data_signature"],
        "normal_operational_timestamp_contract": True, "native_model_bytes_unchanged_after_contract_fix": True,
        "study": settings["study"], "protocol_sha256": study.protocol_sha256,
        "first_eligible_date": study.protocol["first_eligible_date"], "recorder_running": True,
        "recorder_pid": heartbeat["pid"], "heartbeat_age_seconds": age, "discovery_age_seconds": discovery_age,
        "daily_logon_triggers_preserved": True, "rtv_history_state": health["history"]["state"],
        "rtv_snapshot_state": health["snapshot"]["state"], "rtv_snapshot_features": health["snapshot"]["features"]}
    _publish(OUT / "final_status.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
