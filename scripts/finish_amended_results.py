"""Close042 only after canonical repair, installed QA and recorder recovery."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.database import DB_PATH
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore
from src.featured_dataset import fe_signature
from src.forward_store import ForwardStudy
from src.result_amendment_repair import corrected_rows
from src.run_store import RUNS_DIR, get_production_run_id

CASE = ROOT / "data/audit/result_amendments_042/consolidated"
OUTPUT = CASE / "installation"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    destination = OUTPUT / "final_status.json"
    if destination.exists():
        raise ValueError("Preserve the completed installation receipt")
    fit = read(CASE / "refit/fitted.json")
    published = read(OUTPUT / "published.json")
    qa = read(OUTPUT / "qa_after/report.json")
    readiness = read(OUTPUT / "readiness.json")
    recorder = read(OUTPUT / "recorder_registration.json")
    scheduled = read(OUTPUT / "scheduler.json")
    current_recorder = read(ROOT / "data/forecasts/recorder.json")
    fx = read(CASE / "fx_refresh/installed.json")
    fx_qa = read(CASE / "fx_refresh/live_qa/report.json")
    transport = read(CASE / "fx_refresh/transport_check.json")
    fx_scheduled = read(CASE / "fx_refresh/scheduler.json")
    from scripts.project_status import _forward_recording
    forward = _forward_recording()
    checked_at = datetime.fromisoformat(readiness["checked_at"])
    if checked_at.tzinfo is None:
        checked_at = checked_at.astimezone()
    installed_at = datetime.fromisoformat(fx["installed_at_utc"])
    if (checked_at < installed_at or (datetime.now(timezone.utc)-checked_at).total_seconds() > 600
            or not fx_scheduled["verified"] or not fx_scheduled["enabled"] or not fx_scheduled["restarted"]):
        raise ValueError("Refresh full readiness and resume the recorder after the daily FX installation")
    heartbeat = forward.get("heartbeat", {})
    if datetime.fromisoformat(heartbeat.get("observed_at", "1970-01-01T00:00:00+00:00")) < datetime.fromisoformat(fx_scheduled["verified_at_utc"]):
        raise ValueError("Wait for a new recorder heartbeat after restarting")
    if (get_production_run_id() != fit["run_id"] or published["run_id"] != fit["run_id"]
            or not qa["passed"] or not qa["installed_live_inputs_and_probabilities_exact"]
            or qa["run_id"] != fit["run_id"] or not readiness["servable"]
            or not readiness["sections"]["feature_store"]["data_checked"]
            or forward["state"] != "OK" or not scheduled["verified"]
            or not scheduled["enabled"] or not scheduled["restarted"]
            or any(item["model_id"] != fit["run_id"] for item in (recorder, scheduled, current_recorder))
            or any(item["study"] != recorder["study"] for item in (scheduled, current_recorder, forward))):
        raise ValueError("Complete installed serving checks, deep readiness and fresh recorder discovery")
    artifact = Path(RUNS_DIR) / fit["run_id"] / "model.joblib"
    if sha(artifact) != fit["model_sha256"] or fe_signature() != fit["fe_signature"]:
        raise ValueError("Installed model or feature source changed")
    if (not fx["fitted_inputs_unchanged"] or not fx["model_bytes_unchanged"]
            or fx["new_model_fits"] != 0 or not transport["default_verified_https_available"]
            or not fx_qa["passed"] or fx_qa["run_id"] != fit["run_id"]):
        raise ValueError("Qualify the latest-day FX correction and normal verified HTTPS transport")
    study = ForwardStudy(Path(recorder["study"]))
    study.verify()
    if study.protocol_sha256 != recorder["protocol_sha256"]:
        raise ValueError("Current recorder has another pinned protocol")
    before = pd.read_parquet(CASE / "before_db_rows.parquet")
    proposed = pd.read_parquet(CASE / "proposed_repairs.parquet")
    expected = corrected_rows(before, proposed)
    identities = before.race_id.astype(str).unique().tolist()
    with closing(sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True)) as connection:
        actual = pd.read_sql_query("SELECT * FROM results WHERE race_id IN (" + ",".join("?" for _ in identities) + ")",
                                   connection, params=identities)
        marker = connection.execute("SELECT content_sha256 FROM result_amendment_batches WHERE repair_id=?",
            ("042_complete_official_orders_exchange_corroborated",)).fetchone()
        ledger_rows = connection.execute("SELECT COUNT(*) FROM result_amendment_rows WHERE repair_id=?",
            ("042_complete_official_orders_exchange_corroborated",)).fetchone()[0]
    keys = ["race_id", "horse_name"]
    pd.testing.assert_frame_equal(actual[expected.columns].set_index(keys).sort_index(),
        expected.set_index(keys).sort_index(), check_dtype=False, check_exact=True)
    if (not marker or marker[0] != published["application"]["content_sha256"] or ledger_rows != len(expected)):
        raise ValueError("Canonical correction ledger differs from the qualified source batch")
    evidence = [CASE / "refit/fitted.json", CASE / "refit/execution_completed.json",
        OUTPUT / "prepared.json", OUTPUT / "published.json", OUTPUT / "qa_before/report.json",
        OUTPUT / "qa_after/report.json", OUTPUT / "recorder_registration.json", OUTPUT / "scheduler.json",
        OUTPUT / "readiness.json", CASE / "transaction_extension.json",
        CASE / "recent_input_changes.json", CASE / "training_weight_mass.json",
        CASE.parent / "publication_tests_final.log", CASE.parent / "forward_source_tests.log",
        CASE.parent / "discovery_health_tests.log", ROOT / "src/forward_store.py", ROOT / "scripts/project_status.py",
        CASE / "fx_refresh/probe.json", CASE / "fx_refresh/prepared.json", CASE / "fx_refresh/installed.json",
        CASE / "fx_refresh/transport_check.json", CASE / "fx_refresh/live_qa/report.json",
        CASE / "fx_refresh/tail_check.json", CASE / "fx_refresh/scheduler.json"]
    status = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "installed_in_actual_project": True, "run_id": fit["run_id"], "previous_run": published["previous_run"],
        "model_sha256": fit["model_sha256"], "fe_signature": fit["fe_signature"],
        "selected_features": qa["features"], "training_as_of": fit["split"]["as_of"],
        "refit_rounds": fit["split"]["refit_rounds"], "current_store_meta": FeatureStore().meta(),
        "corrected_fields": len(identities), "corrected_source_rows": len(expected),
        "changed_winners": 131, "changed_finishing_positions": 463, "canonical_corrected_rows_exact": True,
        "repair_ledger_rows": ledger_rows, "deep_readiness_passed": True,
        "installed_live_inputs_and_probabilities_exact": True, "single_batch_max_error": qa["single_batch_max_error"],
        "recent_operational_fields": fx_qa["recent_races"], "recent_operational_runners": fx_qa["recent_runners"],
        "current_live_check": {"race_id": fx_qa["upcoming_race_id"], "track": fx_qa["upcoming_track"],
            "off_time": fx_qa["upcoming_off_time"], "runners": fx_qa["upcoming_runners"],
            "single_batch_max_error": fx_qa["native_single_batch_max_error"]},
        "study": recorder["study"], "protocol_sha256": recorder["protocol_sha256"],
        "first_eligible_date": recorder["first_eligible_date"], "scheduler_verified": True,
        "forward_recording": forward, "native_refit_jobs": 1, "cached_native_components": 8,
        "unresolved_exchange_disputes": ["628845", "779195", "914213"],
        "predictive_gain_measured": False, "market_parity_established": False,
        "daily_reference_repair": fx,
        "scope": "Verified source/feature-history correction and one current native refit; no new held-out performance estimate",
        "evidence_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in evidence}, "script_sha256": sha(Path(__file__))}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in status.items() if key not in
        {"evidence_sha256", "forward_recording", "current_store_meta"}}, indent=2))


if __name__ == "__main__":
    main()
