"""Prepare and publish one coherent corrected-source/model generation.

All expensive processing and validation happens before publication. During
publication an explicit SQLite transaction blocks source writers; source,
CSV, store and model have preserved rollback copies. The recorder is paused
separately by the accompanying Windows helper and restarted only after QA.
"""
from contextlib import closing
from datetime import datetime, timezone
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

import config
from scripts import repair_result_amendments as repair
from src.database import DB_PATH
from src.data_processor import process_data
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import data_signature, fe_signature
from src.feature_store import FeatureStore, _stringify_objects
from src.result_amendment_repair import apply_amendments
from src.run_store import RUNS_DIR, PRODUCTION_POINTER, get_production_run_id, load_run_meta, promote
from src.store_refresh import append_current_store

CASE = ROOT / "data/audit/result_amendments_042/consolidated"
OUTPUT = CASE / "installation"
RAW = ROOT / "data/raw/race_results.csv"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verified_inputs():
    fit = read(CASE / "refit/fitted.json")
    execution = read(CASE / "refit/execution_completed.json")
    if (not execution["runtime_sources_unchanged"]
            or sha(CASE / "refit/fitted.json") != execution["fitted_receipt_sha256"]):
        raise ValueError("Finish runtime-source qualification of the saved refit")
    for name, digest in execution["runtime_model_sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Runtime probability math changed after fitting")
    plan = read(CASE / "refit/final_plan.json")
    extension = read(CASE / "transaction_extension.json")
    primary = read(CASE.parent / "atr_primary_verification/report.json")
    if not primary["passed"] or primary["fields"] != 70:
        raise ValueError("Finish primary verification of the cached official orders")
    if not extension["passed"] or sha(ROOT / "src/result_amendment_repair.py") != extension["current_source_sha256"]:
        raise ValueError("Repair transaction implementation is not qualified")
    if fe_signature() != fit["fe_signature"] or fit["fe_signature"] != plan["fe_signature"]:
        raise ValueError("Fitted model and current feature generation differ")
    model = Path(RUNS_DIR) / fit["run_id"] / "model.joblib"
    if sha(model) != fit["model_sha256"]:
        raise ValueError("Fitted artifact changed")
    if get_production_run_id() != plan["parent_run"]:
        raise ValueError("Another production model was selected; review before publication")
    if sha(Path(RUNS_DIR) / plan["parent_run"] / "model.joblib") != plan["parent_model_sha256"]:
        raise ValueError("The installed parent artifact changed")
    repair.CASE = CASE
    before, proposed, evidence = repair.inputs()
    return fit, plan, before, proposed, evidence


def prefix_compatible(processed, store, fit, plan):
    meta = store.meta()
    prefix = processed.loc[pd.to_datetime(processed.race_date).le(meta["last_date"])]
    if data_signature(prefix) != meta["data_signature"] or len(prefix) != meta["processed_rows"]:
        raise ValueError("Current source history changed inside the prepared corrected generation")
    if data_signature(processed, cutoff=plan["config"]["fit_as_of"]) != fit["signatures"]["data_signature"]:
        raise ValueError("Current source training prefix differs from the fitted model")


def prepare():
    fit, plan, before, proposed, evidence = verified_inputs()
    OUTPUT.mkdir(exist_ok=False)
    write(OUTPUT / "plan.json", {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run": plan["parent_run"], "run_id": fit["run_id"], "fe_signature": fit["fe_signature"],
        "model_sha256": fit["model_sha256"], "source_sha256": {name: sha(ROOT / name) for name in
            ("scripts/install_amended_results.py", "src/result_amendment_repair.py", "src/database.py")},
        "evidence_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in
            (CASE / "refit/fitted.json", CASE / "refit/final_plan.json", CASE / "transaction_extension.json",
             CASE / "audit.json", CASE / "audit_plan.json", CASE.parent / "atr_primary_verification/report.json")}})
    snapshot, preview = OUTPUT / "source_before.db", OUTPUT / "source_preview.db"
    with closing(sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(snapshot)) as destination:
            source.backup(destination)
    shutil.copy2(snapshot, preview)
    with closing(sqlite3.connect(preview)) as connection:
        applied = apply_amendments(connection, before, proposed, repair_id=repair.REPAIR_ID, evidence=evidence)
        rows = repair.export_csv(connection, OUTPUT / "source_preview.csv")
    # CSV is the ordinary pipeline contract. Loading directly from SQLite
    # would change some ID dtypes and has caused false fingerprints before.
    raw = pd.read_csv(OUTPUT / "source_preview.csv")
    processed = _stringify_objects(process_data(df=raw, save=False))
    del raw
    gc.collect()
    private = FeatureStore(str(CASE / "history_rebuild/feature_store"))
    prefix_compatible(processed, private, fit, plan)
    if not append_current_store(processed, store=private):
        raise ValueError("Additional current source rows cannot be appended safely")
    prefix_compatible(processed, private, fit, plan)
    processed.to_parquet(OUTPUT / "current_processed.parquet", index=False)
    ready = FeatureStore(str(OUTPUT / "ready_store"))
    with private._access():
        for name in private._FILES:
            os.link(Path(private.root) / name, Path(ready.root) / name)
    write(OUTPUT / "prepared.json", {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": fit["run_id"], "model_sha256": fit["model_sha256"], "preview_application": applied,
        "source_rows": rows, "processed_rows": len(processed), "last_date": ready.meta()["last_date"],
        "raw_preview_sha256": sha(OUTPUT / "source_preview.csv"), "processed_sha256": sha(OUTPUT / "current_processed.parquet"),
        "store_files_sha256": {name: sha(Path(ready.root) / name) for name in ready._FILES},
        "training_prefix_signature": fit["signatures"]["data_signature"],
        "canonical_writes": False, "model_fits": 0})
    print("Publication source/history snapshot prepared; any new settled tail appended natively, without refitting.", flush=True)


def replace_copy(source, destination):
    fd, temporary = tempfile.mkstemp(dir=Path(destination).parent, suffix=".maintenance.tmp")
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def publish():
    fit, plan, before, proposed, evidence = verified_inputs()
    prepared = read(OUTPUT / "prepared.json")
    install_plan = read(OUTPUT / "plan.json")
    paused = read(OUTPUT / "recorder_paused.json")
    qa = read(OUTPUT / "qa_before/report.json")
    if (not paused["paused"] or paused["model_id"] != plan["parent_run"]
            or not qa["passed"] or qa["run_id"] != fit["run_id"]):
        raise ValueError("Finish staged serving QA and pause the old recorder before publication")
    for name, digest in install_plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Publication implementation changed after preparation")
    for name, digest in install_plan["evidence_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Publication evidence changed after preparation")
    ready = FeatureStore(str(OUTPUT / "ready_store"))
    for name, digest in prepared["store_files_sha256"].items():
        if sha(Path(ready.root) / name) != digest:
            raise ValueError("Qualified corrected history changed")
    canonical = FeatureStore()
    backup = OUTPUT / "previous_files"
    backup.mkdir(exist_ok=False)
    raw_changed = store_attempted = pointer_changed = False
    source_committed = False
    application = None
    try:
        with canonical._access():
            previous_meta = canonical.meta()
            for name in canonical._FILES:
                os.link(Path(canonical.root) / name, backup / name)
            shutil.copy2(RAW, backup / "race_results.csv")
            shutil.copy2(PRODUCTION_POINTER, backup / "production.json")
            with closing(sqlite3.connect(DB_PATH)) as connection:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    application = apply_amendments(connection, before, proposed,
                        repair_id=repair.REPAIR_ID, evidence=evidence, manage_transaction=False)
                    repair.export_csv(connection, OUTPUT / "source_committing.csv")
                    if sha(OUTPUT / "source_committing.csv") != prepared["raw_preview_sha256"]:
                        raise ValueError("Current source differs from the qualified preview; no generation published")
                    store_attempted = True
                    with canonical._staged_update() as pending:
                        for name in canonical._FILES:
                            os.link(Path(ready.root) / name, Path(pending.root) / name)
                        replace_copy(OUTPUT / "source_committing.csv", RAW)
                        raw_changed = True
                    # Reuse the actual, already locked store's metadata for
                    # the normal promotion guard. Creating another store
                    # instance here would contend with our own OS file lock.
                    from unittest.mock import patch
                    from src import run_store
                    with patch.object(run_store, "store_fe_signature", lambda: canonical.meta().get("fe_signature")):
                        if not promote(fit["run_id"]):
                            raise ValueError("The ordinary model-generation promotion guard rejected the candidate")
                    pointer_changed = True
                    if fe_signature() != fit["fe_signature"]:
                        raise ValueError("Feature sources changed during publication")
                    intent = {"published_at_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": fit["run_id"], "previous_run": plan["parent_run"], "model_sha256": fit["model_sha256"],
                        "application": application, "source_rows": prepared["source_rows"],
                        "fe_signature": fit["fe_signature"], "store_meta": canonical.meta(),
                        "previous_store_meta": previous_meta, "new_raw_sha256": sha(RAW),
                        "scope": "Mandatory official-result correction and one current-data refit; no held-out accuracy estimate"}
                    write(OUTPUT / "commit_intent.json", intent)
                # The outer SQLite commit completes only after the filesystem
                # and pointer changes succeeded. A failure rolls back the
                # complete source batch and its ledger.
            source_committed = True
            write(OUTPUT / "published.json", intent)
    except Exception as error:
        if not source_committed and application and not application["already_applied"]:
            # Resolve the narrow interruption window immediately after COMMIT
            # by reading the durable ledger, rather than trusting a Python flag.
            with closing(sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True)) as probe:
                exists = probe.execute("SELECT name FROM sqlite_master WHERE name='result_amendment_batches'").fetchone()
                marker = probe.execute("SELECT content_sha256 FROM result_amendment_batches WHERE repair_id=?",
                                       (repair.REPAIR_ID,)).fetchone() if exists else None
                source_committed = bool(marker and marker[0] == application["content_sha256"])
        if source_committed:
            # The generation is already committed and coherent. Preserve it
            # if only writing the final receipt failed; commit_intent.json
            # provides an exact recovery record, without another fit.
            raise RuntimeError("Generation committed; recover published.json from the verified commit intent") from error
        # A failed DB commit has already rolled back before this restoration.
        # Keep all prior files and the failure receipt for inspection.
        if store_attempted:
            with canonical._staged_update() as pending:
                for name in canonical._FILES:
                    os.link(backup / name, Path(pending.root) / name)
        if raw_changed:
            replace_copy(backup / "race_results.csv", RAW)
        if pointer_changed:
            replace_copy(backup / "production.json", PRODUCTION_POINTER)
        write(OUTPUT / "publication_failure.json", {"failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(error), "store_restored": store_attempted, "raw_restored": raw_changed,
            "pointer_restored": pointer_changed, "source_transaction_rolled_back": True})
        raise
    print("Corrected source, histories and current model published coherently; recorder remains paused for final QA.", flush=True)


def register_recorder():
    from src.forward_store import create_study
    from src.forward_readout import register_plan
    from src.forward_recorder import replace_json
    published = read(OUTPUT / "published.json")
    qa = read(OUTPUT / "qa_after/report.json")
    if (not qa["passed"] or qa["run_id"] != published["run_id"]
            or get_production_run_id() != published["run_id"]):
        raise ValueError("Finish installed serving QA before registering the new study")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = create_study(ROOT / f"data/forecasts/forward/{stamp}_amended_results_042_{published['run_id']}")
    register_plan(study, study.path / "readout_plan.json")
    settings = {"study": str(study.path), "model_id": published["run_id"],
        "protocol_sha256": study.protocol_sha256, "first_eligible_date": study.protocol["first_eligible_date"],
        "configured_at": datetime.now(timezone.utc).isoformat(), "scheduled_task": None, "scheduled_task_verified": False}
    replace_json(ROOT / "data/forecasts/recorder.json", settings)
    write(OUTPUT / "recorder_registration.json", settings)
    print("Fresh source/model-pinned study registered; existing studies remain preserved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "publish", "register-recorder"])
    {"prepare": prepare, "publish": publish, "register-recorder": register_recorder}[parser.parse_args().phase]()
