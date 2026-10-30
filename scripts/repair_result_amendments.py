"""Stage verified amendments, then install the reviewed source-data repair.

Staging has no effect on normal results/store/model/recorder paths. Installation
refuses an unfinished store migration. It updates reviewed rows transactionally
and exports the complete history; ordinary training then detects the changed
data fingerprint and rebuilds histories. It never relabels old predictions or
claims their scores measure a subsequently fitted model.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.database import DB_PATH
from src.deployment_validation import file_sha256 as sha
from src.result_amendment_repair import apply_amendments

CASE = ROOT / "data/audit/result_amendments_042"
STAGE = CASE / "staged_history_v2"
REPAIR_ID = "042_complete_official_orders_exchange_corroborated"


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def inputs():
    report = json.loads((CASE / "audit.json").read_text())
    before = pd.read_parquet(CASE / "before_db_rows.parquet")
    proposed = pd.read_parquet(CASE / "proposed_repairs.parquet")
    if report["eligible_fields"] != proposed.race_id.nunique() or report["eligible_field_runners"] != len(proposed):
        raise ValueError("Preserve the completed historical audit population")
    evidence = {"scope": "Final official placings; original arrival margins withheld on reordered fields",
        "inputs_sha256": {name: sha(CASE / name) for name in
                          ("audit.json", "audit_plan.json", "before_db_rows.parquet", "proposed_repairs.parquet")},
        "reference_input_sha256": json.loads((CASE / "audit_plan.json").read_text())["input_sha256"],
        "limitations": report["limitations"]}
    return before, proposed, evidence


def export_csv(connection, destination):
    temporary = destination.with_suffix(destination.suffix + ".pending")
    if temporary.exists():
        raise ValueError("Preserve the interrupted export for inspection")
    rows = 0
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        for chunk in pd.read_sql_query("SELECT * FROM results ORDER BY race_date", connection, chunksize=25000):
            chunk.drop(columns="scraped_at", errors="ignore").to_csv(stream, index=False, header=rows == 0)
            rows += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return rows


def stage():
    before, proposed, evidence = inputs()
    STAGE.mkdir(exist_ok=False)
    write(STAGE / "plan.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": evidence, "source_sha256": {name: sha(ROOT / name) for name in
            ("scripts/repair_result_amendments.py", "src/result_amendment_repair.py")},
        "canonical_writes": False, "new_model_fits": 0})
    snapshot = STAGE / "original.db"
    corrected = STAGE / "corrected.db"
    with closing(sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
    shutil.copy2(snapshot, corrected)
    with closing(sqlite3.connect(corrected)) as connection:
        if connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' AND tbl_name='results'").fetchone()[0]:
            raise ValueError("Unreviewed result triggers prevent automatic repair")
        receipt = apply_amendments(connection, before, proposed, repair_id=REPAIR_ID, evidence=evidence)
        repeated = apply_amendments(connection, before, proposed, repair_id=REPAIR_ID, evidence=evidence)
        assert repeated["already_applied"] and repeated["updated_rows"] == 0
        connection.execute("ATTACH DATABASE ? AS original", (str(snapshot),))
        count = connection.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        old_count = connection.execute("SELECT COUNT(*) FROM original.results").fetchone()[0]
        matched = connection.execute("SELECT COUNT(*) FROM results a JOIN original.results b USING(race_id,horse_name)").fetchone()[0]
        if count != old_count or matched != count:
            raise ValueError("Repair changed the canonical starter population")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(results)")]
        expressions = []
        for col in columns:
            quoted = '"' + col.replace('"', '""') + '"'
            expressions.append(f"SUM(a.{quoted} IS NOT b.{quoted})")
        changed = connection.execute("SELECT " + ",".join(expressions) +
            " FROM results a JOIN original.results b USING(race_id,horse_name)").fetchone()
        changes = dict(zip(columns, changed))
        if any(value for col, value in changes.items() if col not in {"finish_position", "finish_pos_label", "won", "lengths_behind"}):
            raise ValueError("An unrelated canonical value changed")
        audit = json.loads((CASE / "audit.json").read_text())
        if changes["finish_position"] != audit["eligible_position_changes"] or changes["won"] != 2 * audit["eligible_changed_winners"]:
            raise ValueError("Changed labels differ from the source audit")
        connection.execute("DETACH DATABASE original")
        exported = export_csv(connection, STAGE / "race_results.csv")
        if exported != count or connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Staged repaired snapshot is incomplete")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "application": receipt, "source_rows": count, "exported_rows": exported,
        "changed_cells_by_column": changes, "all_other_columns_exact": True,
        "whole_field_identity_exact": True, "repeat_application_noop": True,
        "outputs_sha256": {name: sha(STAGE / name) for name in ("original.db", "corrected.db", "race_results.csv")},
        "canonical_writes": False, "new_model_fits": 0}
    write(STAGE / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def require_installation_window():
    # Another chat owns this generation migration. Never publish new raw
    # history underneath its qualification/activation transaction.
    migration = ROOT / "data/audit/live_feature_speed_20261029/installation"
    if (migration / "plan.json").exists() and not (migration / "final_status.json").exists():
        raise RuntimeError("The other chat's feature-store migration is unfinished; staged repairs are ready, canonical results were not changed")


def install():
    require_installation_window()
    before, proposed, evidence = inputs()
    report = json.loads((STAGE / "report.json").read_text())
    if not report["passed"] or not report["all_other_columns_exact"]:
        raise ValueError("Finish and verify the complete staged repair first")
    plan = json.loads((STAGE / "plan.json").read_text())
    if evidence != plan["evidence"]:
        raise ValueError("Reviewed evidence changed after staging")
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Repair implementation changed after staging")
    for name, digest in report["outputs_sha256"].items():
        if sha(STAGE / name) != digest:
            raise ValueError("Reviewed staged evidence changed")
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature
    store = FeatureStore()
    with store._access():
        if store.meta().get("fe_signature") != fe_signature():
            raise ValueError("Finish the feature-generation rollout before installing source repairs")
        history = ROOT / "data/raw/race_results.csv"
        backup = CASE / "canonical_raw_before.csv"
        if not backup.exists():
            shutil.copy2(history, backup)
        with closing(sqlite3.connect(DB_PATH)) as connection:
            receipt = apply_amendments(connection, before, proposed, repair_id=REPAIR_ID, evidence=evidence)
            exported = export_csv(connection, history)
        write(CASE / "installation.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "application": receipt, "exported_rows": exported, "raw_sha256": sha(history),
            "next_training_requires_full_history_rebuild": True,
            "model_refitted": False, "old_store_retained_until_normal_rebuild": True,
            "scope": "Mandatory result-source correction; no new accuracy claim or model promotion"})
    print("Official source corrections installed; ordinary training will rebuild changed histories.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["stage", "install"])
    parser.add_argument("--case", type=Path, default=CASE)
    args = parser.parse_args()
    CASE = args.case.resolve()
    STAGE = CASE / "staged_history_v2"
    {"stage": stage, "install": install}[args.phase]()
