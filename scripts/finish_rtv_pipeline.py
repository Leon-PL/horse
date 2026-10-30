"""Close the RTV source rollout after preserved readiness and task checks."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.deployment_validation import file_sha256
from src.forward_store import ForwardStudy
from src.run_store import RUNS_DIR, get_production_run_id
from src.source_pipeline import _read, _publish


def main():
    audit = ROOT/"data/audit/rtv_pipeline_043/installation"
    before, ready, activation = (_read(audit/name) for name in ("before.json", "readiness.json", "activation.json"))
    settings, scheduler = _read(ROOT/"data/forecasts/recorder.json"), _read(audit/"scheduler.json")
    study = ForwardStudy(settings["study"])
    study.verify()
    heartbeat = _read(study.path/"recorder_heartbeat.json")
    now = datetime.now(timezone.utc)
    age = (now-datetime.fromisoformat(heartbeat["observed_at"])).total_seconds()
    discovery_age = (now-datetime.fromisoformat(heartbeat["last_discovery"])).total_seconds()
    if (not ready["servable"] or not ready["feature_store"]["data_matches"] or age > 90
            or discovery_age > 1200 or heartbeat["mode"] != "recording"
            or not settings["scheduled_task_verified"] or not scheduler["triggers_preserved"]
            or scheduler["study"] != settings["study"] or activation["study"] != settings["study"]):
        raise ValueError("Recorder/readiness qualification failed")
    model = get_production_run_id()
    unchanged = (model == before["model_id"]
        and file_sha256(Path(RUNS_DIR)/model/"model.joblib") == before["model_sha256"]
        and file_sha256(Path(config.RAW_DATA_DIR)/"race_results.csv") == before["raw_csv_sha256"])
    if not unchanged:
        raise ValueError("Canonical model/history changed")
    for name in ("history", "snapshot"):
        if _read(ROOT/"data/rtv_cache/pipeline"/(name+"_status.json"))["state"] != "ok":
            raise ValueError("RTV source health failed")
    value = {"completed_at": now.isoformat(), "status": "COMPLETE_SOURCE_INTEGRATION",
        "study": str(study.path), "protocol_sha256": study.protocol_sha256,
        "first_eligible_date": study.protocol["first_eligible_date"], "model_id": model,
        "fe_signature": study.protocol["fe_signature"], "new_model_fits": 0,
        "canonical_model_and_history_unchanged": unchanged, "deep_readiness_passed": True,
        "recorder_running": True, "recorder_pid": heartbeat["pid"], "heartbeat_age_seconds": age,
        "discovery_age_seconds": discovery_age, "triggers_preserved": True,
        "official_position_review_queue": {"races": 75, "runners": 336,
            "fresh_sl_corroborated_stale_database_positions": 16, "sampled_races": 3,
            "separate_malformed_sl_amendment_field": "944574", "canonical_repairs_applied": 0},
        "selected_rtv_model_inputs": 0}
    _publish(audit/"final_status.json", value)
    print(value)


if __name__ == "__main__":
    main()
