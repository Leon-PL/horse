"""Qualify caller-owned transactions without changing the sealed repair math."""
from contextlib import closing
from datetime import datetime, timezone
import importlib.util
import inspect
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from scripts import repair_result_amendments as source_repair
from src import result_amendment_repair as current
from src.deployment_validation import file_sha256 as sha

CASE = ROOT / "data/audit/result_amendments_042/consolidated"


def main():
    destination = CASE / "transaction_extension.json"
    if destination.exists():
        raise ValueError("Preserve the completed transaction qualification")
    original_path = CASE / "result_amendment_repair_at_staging.py"
    stage = CASE / "staged_history_v2"
    stage_plan = json.loads((stage / "plan.json").read_text())
    if sha(original_path) != stage_plan["source_sha256"]["src/result_amendment_repair.py"]:
        raise ValueError("The original staged repair implementation was not preserved exactly")
    spec = importlib.util.spec_from_file_location("repair_at_staging_042", original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    if inspect.getsource(original.corrected_rows) != inspect.getsource(current.corrected_rows):
        raise ValueError("The pure correction definition changed")
    source_repair.CASE = CASE
    before, proposed, evidence = source_repair.inputs()
    expected = original.corrected_rows(before, proposed)
    pd.testing.assert_frame_equal(expected, current.corrected_rows(before, proposed), check_exact=True)
    expected_digest = json.loads((stage / "report.json").read_text())["application"]["content_sha256"]
    with closing(sqlite3.connect(":memory:")) as connection:
        before.to_sql("results", connection, index=False)
        connection.execute("BEGIN IMMEDIATE")
        first = current.apply_amendments(connection, before, proposed,
            repair_id=source_repair.REPAIR_ID, evidence=evidence, manage_transaction=False)
        if not connection.in_transaction or first["content_sha256"] != expected_digest:
            raise ValueError("Caller-owned transaction changed the reviewed batch")
        connection.rollback()
        pd.testing.assert_frame_equal(pd.read_sql_query("SELECT * FROM results", connection), before,
                                     check_dtype=False, check_exact=True)
        if connection.execute("SELECT name FROM sqlite_master WHERE name LIKE 'result_amendment_%'").fetchall():
            raise ValueError("Rollback retained partial repair evidence")
        second = current.apply_amendments(connection, before, proposed,
            repair_id=source_repair.REPAIR_ID, evidence=evidence)
        if second["content_sha256"] != expected_digest:
            raise ValueError("Default transaction changed the reviewed batch")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "rows": len(before), "fields": int(before.race_id.nunique()), "pure_correction_source_exact": True,
        "full_data_result_exact": True, "default_manifest_digest_exact": True,
        "caller_owned_manifest_digest_exact": True, "caller_rollback_restores_all_rows_and_ledger": True,
        "original_source_sha256": sha(original_path),
        "current_source_sha256": sha(ROOT / "src/result_amendment_repair.py"),
        "verification_script_sha256": sha(Path(__file__)), "canonical_writes": False}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
