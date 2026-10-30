"""Pin the source-only RTV rollout without changing model weights or histories."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.deployment_validation import file_sha256
from src.featured_dataset import fe_signature
from src.forward_store import create_study
from src.forward_readout import register_plan
from src.run_store import get_production_run_id, RUNS_DIR
from src.source_pipeline import _publish, load_matched_history, load_history_table


def main():
    audit = ROOT / "data/audit/rtv_pipeline_043/installation"
    audit.mkdir(parents=True, exist_ok=True)
    if (audit / "activation.json").exists():
        raise ValueError("This rollout is already activated; preserve its receipt")
    model = get_production_run_id()
    if model != "20261030_015917_823047" or fe_signature() != "3697230d16bbd0d8":
        raise ValueError("The installed generation changed during qualification")
    pointer = ROOT / "data/forecasts/recorder.json"
    before = {"recorder": json.loads(pointer.read_text()), "model_id": model, "fe_signature": fe_signature(),
        "model_sha256": file_sha256(Path(RUNS_DIR)/model/"model.joblib"),
        "raw_csv_sha256": file_sha256(Path(config.RAW_DATA_DIR)/"race_results.csv")}
    _publish(audit / "before.json", before)
    linked = load_matched_history("2026-10-29", "2026-10-29")
    metrics = load_history_table("metrics", "2026-10-29", "2026-10-29")
    sectionals = load_history_table("sectionals", "2026-10-29", "2026-10-29")
    if len(linked) != 387 or linked.sl_race_id.nunique() != 38 or linked.duplicated(["sl_race_id", "sl_horse_id"]).any():
        raise ValueError("Yesterday's complete identity projection did not replay")
    now = datetime.now(timezone.utc)
    path = ROOT / "data/forecasts/forward" / (now.strftime("%Y%m%dT%H%M%SZ")+"_rtv_sources_043_"+model)
    study = create_study(path)
    register_plan(study, path/"readout_plan.json")
    study.verify()
    if (before["model_sha256"] != file_sha256(Path(RUNS_DIR)/model/"model.joblib")
            or before["raw_csv_sha256"] != file_sha256(Path(config.RAW_DATA_DIR)/"race_results.csv")):
        raise ValueError("Canonical model/history changed during the source-only rollout")
    value = {"study": str(path), "model_id": model, "protocol_sha256": study.protocol_sha256,
        "first_eligible_date": study.protocol["first_eligible_date"], "configured_at": now.isoformat(),
        "scheduled_task": "HorseForecastRecorder_038", "scheduled_task_verified": False}
    _publish(pointer, value)
    result = {"configured_at": now.isoformat(), "study": str(path), "model_id": model,
        "model_sha256": before["model_sha256"], "fe_signature": fe_signature(),
        "raw_csv_sha256": before["raw_csv_sha256"], "new_model_fits": 0,
        "linked_runners": len(linked), "linked_races": linked.sl_race_id.nunique(),
        "metric_rows": len(metrics), "sectional_rows": len(sectionals),
        "canonical_model_and_history_unchanged": True, "recorder_task_retarget_pending": True}
    _publish(audit/"activation.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
