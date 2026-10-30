"""Publish the reviewed 044 source, feature and model generation with rollback.

Run prepare after the native refit, pause the existing recorder, then publish.
The source transaction commits only after code, CSV, caches, store and pointer
are coherent. A fresh process must pass serving QA before recorder registration.
"""
from contextlib import closing
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/audit/rtv_repair_features_044"
PROJECT = CASE / "project"
OUTPUT = CASE / "installation"
sys.path.insert(0, str(ROOT))

SOURCES = [
    "src/data_scraper.py", "src/feature_engineer.py", "src/featured_dataset.py",
    "src/fe_taxonomy.py", "src/feature_inputs.py", "src/store_refresh.py",
    "src/live_prediction.py", "src/source_pipeline.py", "src/forward_store.py",
    "src/rtv_features.py", "src/rtv_feature_collection.py", "src/verified_result_repair.py",
    "src/rtv_scraper.py", "tests/test_rtv_scraper.py",
    "scripts/build_feature_store_batched.py", "scripts/project_status.py",
    "scripts/quick_win_feature_screen.py", "train.py", "tests/test_rtv_features.py",
    "tests/test_rtv_feature_collection.py", "tests/test_verified_result_repair.py",
    "tests/test_source_pipeline.py", "feature_sets/rtv_history.txt",
    "feature_sets/rtv_pre_race.txt", "feature_sets/win_with_rtv.txt",
]
CACHES = ["data/rtv_cache/feature_runs.parquet", "data/rtv_cache/feature_pre.parquet"]
RAW = ROOT / "data/raw/race_results.csv"
POINTER = ROOT / "models/production.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def candidates():
    quality = read(CASE / "casualty_quality.json")
    if (not quality["persisted_selected_inputs_unchanged_exact"]
            or quality["rows_checked"] != 755357
            or sha(PROJECT / "src/rtv_scraper.py") != quality["parser_sha256"]
            or sha(PROJECT / CACHES[0]) != quality["after_runs_sha256"]):
        raise ValueError("Finish casualty classification and retained-input qualification")
    return SOURCES + CACHES + list(quality["parsed_files"])


def replace_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".044.tmp")
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def verified_inputs():
    import pandas as pd
    fit, plan = read(CASE / "refit/fitted.json"), read(CASE / "refit/plan.json")
    if read(POINTER)["run_id"] != plan["parent_run"]:
        raise ValueError("Production changed before the reviewed installation")
    if sha(ROOT / "data/runs" / plan["parent_run"] / "model.joblib") != plan["parent_model_sha256"]:
        raise ValueError("Parent artifact changed")
    if sha(PROJECT / "data/runs" / fit["run_id"] / "model.joblib") != fit["model_sha256"]:
        raise ValueError("Corrected model changed")
    for name, digest in plan["source_sha256"].items():
        if sha(PROJECT / name) != digest:
            raise ValueError("Qualified source changed: " + name)
    for name, digest in plan["input_sha256"].items():
        if sha(name) != digest:
            raise ValueError("Qualified evidence changed: " + name)
    prepared = read(CASE / "prepared_source.json")
    before, after, additional = [pd.read_parquet(CASE / (name + ".parquet"))
                                for name in ("repair_before", "repair_after", "new_fields")]
    # Preserve the exact observation timestamps in the qualified CSV preview.
    # Keep every original extra source column, including nonfinish context.
    with closing(sqlite3.connect(PROJECT / "data/races.db")) as con:
        stamps = pd.read_sql_query("SELECT race_id,horse_name,scraped_at FROM results", con)
    additional = additional.merge(stamps, on=["race_id", "horse_name"], how="left", validate="one_to_one")
    if additional.scraped_at.isna().any():
        raise ValueError("A reviewed new starter lost its observation timestamp")
    evidence = dict(prepared["evidence"])
    evidence["reviewed_insert_timestamps_sha256"] = hashlib.sha256(
        additional[["race_id", "horse_name", "scraped_at"]].to_json(orient="records").encode()).hexdigest()
    for name, digest in prepared["evidence"]["input_sha256"].items():
        if sha(CASE / (name + ".parquet")) != digest:
            raise ValueError("Reviewed source field changed")
    for name in ("review_roster", "verification"):
        if sha(CASE / (name + ".json")) != evidence[name + "_sha256"]:
            raise ValueError("Reviewed decisions changed")
    for name, digest in evidence["source_sha256"].items():
        if sha(CASE / "sources" / name) != digest:
            raise ValueError("Reviewed original source changed: " + name)
    spec = importlib.util.spec_from_file_location("repair_044", PROJECT / "src/verified_result_repair.py")
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    return fit, plan, before, after, additional, evidence, repair


def export(connection, path):
    import pandas as pd
    rows = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", connection).drop(columns=["scraped_at"], errors="ignore")
    rows.to_csv(path, index=False)
    return len(rows)


def close_exclusions(connection):
    """Close a reviewed Arabian listing without inserting an excluded field."""
    spec = importlib.util.spec_from_file_location("scraper_044", PROJECT / "src/data_scraper.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    closed = []
    for item in read(CASE / "verification.json")["decisions"]:
        if item.get("state") != "review_required" or not item.get("error", "").startswith("ARABIAN:"):
            continue
        source = CASE / "sources" / (item["race_id"] + ".json")
        page = read(source)["page"]
        scraper = parser.SportingLifeScraper()
        scraper._fetch_next_data = lambda url, page=page: page
        try:
            scraper.scrape_race_result(item["url"], item["race_date"], raise_if_not_run=True)
        except parser.RaceNotRun as error:
            if error.reason != "ARABIAN":
                raise ValueError("The reviewed exclusion reason changed") from error
        else:
            raise ValueError("An excluded field now parses as an eligible result")
        if connection.execute("SELECT COUNT(*) FROM results WHERE race_id=?", (item["race_id"],)).fetchone()[0]:
            raise ValueError("An excluded Arabian field is present in canonical results")
        connection.execute("UPDATE result_sync_races SET completed=1,last_error='not run: ARABIAN' WHERE race_id=? AND race_date=?",
                           (item["race_id"], item["race_date"]))
        closed.append({"race_id": item["race_id"], "race_date": item["race_date"], "reason": "ARABIAN",
                       "source_sha256": sha(source)})
    return closed


def prepare():
    fit, plan, before, after, additional, evidence, repair = verified_inputs()
    OUTPUT.mkdir(exist_ok=True)
    if (OUTPUT / "prepared.json").exists():
        raise ValueError("Publication is already prepared")
    names_to_install = candidates()
    for name in names_to_install:
        if not (PROJECT / name).is_file():
            raise ValueError("Missing candidate: " + name)
    for name, digest in plan["source_sha256"].items():
        if name not in SOURCES and sha(ROOT / name) != digest:
            raise ValueError("Unmodified runtime source differs from the refit: " + name)
    snapshot, preview = OUTPUT / "source_before.db", OUTPUT / "source_preview.db"
    if snapshot.exists() or preview.exists():
        raise ValueError("Preserve the previous preparation files")
    with closing(sqlite3.connect((ROOT / "data/races.db").as_uri() + "?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(snapshot)) as destination:
            source.backup(destination)
    shutil.copyfile(snapshot, preview)
    with closing(sqlite3.connect(preview)) as con:
        applied = repair.apply_verified_results(con, before, after, additional,
            repair_id="rtv_verified_results_044", evidence=evidence)
        with con:
            exclusions = close_exclusions(con)
        rows = export(con, OUTPUT / "source_preview.csv")
    expected = read(CASE / "prepared_source.json")["raw_sha256"]
    if sha(OUTPUT / "source_preview.csv") != expected:
        raise ValueError("Actual source plus verified repairs differs from the rebuilt preview")
    files = {name: {"before": sha(ROOT / name) if (ROOT / name).exists() else None,
                    "after": sha(PROJECT / name)} for name in names_to_install}
    store = PROJECT / "data/processed/feature_store"
    names = ["processed.parquet", "featured.parquet", "ratings_state.pkl", "counts_state.pkl", "meta.json"]
    write(OUTPUT / "prepared.json", {"prepared_at": datetime.now(timezone.utc).isoformat(),
        "run_id": fit["run_id"], "fe_signature": fit["fe_signature"], "model_sha256": fit["model_sha256"],
        "rows": rows, "files": files, "raw_sha256": expected, "application": applied,
        "closed_exclusions": exclusions,
        "casualty_quality_sha256": sha(CASE / "casualty_quality.json"),
        "store_files_sha256": {name: sha(store / name) for name in names},
        "helper_sha256": sha(__file__), "canonical_writes": False})
    print("Exact source preview and coherent installation prepared", flush=True)


def publish():
    fit, plan, before, after, additional, evidence, repair = verified_inputs()
    ready = read(OUTPUT / "prepared.json")
    candidates()
    if sha(CASE / "casualty_quality.json") != ready["casualty_quality_sha256"]:
        raise ValueError("Casualty quality evidence changed")
    paused, qa = read(OUTPUT / "recorder_paused.json"), read(OUTPUT / "qa_before/report.json")
    if not paused["paused"] or paused["model_id"] != plan["parent_run"] or not qa["passed"] or qa["run_id"] != fit["run_id"]:
        raise ValueError("Pause the old recorder and finish native staged serving QA first")
    if sha(__file__) != ready["helper_sha256"]:
        raise ValueError("Publication implementation changed")
    for name, digests in ready["files"].items():
        current = sha(ROOT / name) if (ROOT / name).exists() else None
        if current != digests["before"] or sha(PROJECT / name) != digests["after"]:
            raise ValueError("Installation input changed: " + name)
    store_path = PROJECT / "data/processed/feature_store"
    for name, digest in ready["store_files_sha256"].items():
        if sha(store_path / name) != digest:
            raise ValueError("Qualified store changed")
    # Import the unchanged storage primitive before replacing source files;
    # probability and FE modules are reloaded after the complete source copy.
    from src.feature_store import FeatureStore
    canonical = FeatureStore()
    backup = OUTPUT / "previous_files"
    backup.mkdir(exist_ok=False)
    changed, store_attempted, source_committed, application = [], False, False, None
    try:
        with canonical._access():
            old_meta = canonical.meta()
            for name in canonical._FILES:
                os.link(Path(canonical.root) / name, backup / name)
            for name in [*ready["files"], "data/raw/race_results.csv", "models/production.json", "data/forecasts/recorder.json"]:
                if (ROOT / name).exists():
                    target = backup / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
            with closing(sqlite3.connect(ROOT / "data/races.db")) as con:
                with con:
                    con.execute("BEGIN IMMEDIATE")
                    application = repair.apply_verified_results(con, before, after, additional,
                        repair_id="rtv_verified_results_044", evidence=evidence, manage_transaction=False)
                    exclusions = close_exclusions(con)
                    if exclusions != ready["closed_exclusions"]:
                        raise ValueError("Reviewed terminal exclusions changed")
                    export(con, OUTPUT / "source_committing.csv")
                    if sha(OUTPUT / "source_committing.csv") != ready["raw_sha256"]:
                        raise ValueError("Source changed after preview; publication cancelled")
                    for name in ready["files"]:
                        changed.append(name)
                        replace_copy(PROJECT / name, ROOT / name)
                    changed.append("data/raw/race_results.csv")
                    replace_copy(OUTPUT / "source_committing.csv", RAW)
                    store_attempted = True
                    with canonical._staged_update() as pending:
                        for name in canonical._FILES:
                            shutil.copyfile(store_path / name, Path(pending.root) / name)
                    destination = ROOT / "data/runs" / fit["run_id"]
                    if destination.exists():
                        raise ValueError("Fitted run already exists in the actual registry")
                    shutil.copytree(PROJECT / "data/runs" / fit["run_id"], destination)
                    from src import featured_dataset, run_store
                    importlib.reload(featured_dataset)
                    if featured_dataset.fe_signature() != fit["fe_signature"]:
                        raise ValueError("Installed FE sources do not match the fitted generation")
                    from unittest.mock import patch
                    changed.append("models/production.json")
                    with patch.object(run_store, "store_fe_signature", lambda: canonical.meta()["fe_signature"]):
                        if not run_store.promote(fit["run_id"]):
                            raise ValueError("The standard promotion guard rejected the corrected model")
                    intent = {"published_at": datetime.now(timezone.utc).isoformat(), "run_id": fit["run_id"],
                        "previous_run": plan["parent_run"], "model_sha256": fit["model_sha256"],
                        "fe_signature": fit["fe_signature"], "rows": ready["rows"], "application": application,
                        "raw_sha256": sha(RAW), "store_meta": canonical.meta(), "previous_store_meta": old_meta,
                        "closed_exclusions": exclusions,
                        "new_feature_selection": False, "scope": "Verified source repairs and unchanged-recipe refit"}
                    write(OUTPUT / "commit_intent.json", intent)
            source_committed = True
            write(OUTPUT / "published.json", intent)
    except Exception as error:
        if not source_committed and application and not application["already_applied"]:
            with closing(sqlite3.connect((ROOT / "data/races.db").as_uri() + "?mode=ro", uri=True)) as probe:
                table = probe.execute("SELECT name FROM sqlite_master WHERE name='verified_result_batches'").fetchone()
                marker = probe.execute("SELECT content_sha256 FROM verified_result_batches WHERE repair_id='rtv_verified_results_044'").fetchone() if table else None
                source_committed = bool(marker and marker[0] == application["content_sha256"])
        if source_committed:
            raise RuntimeError("Generation committed; recover the published receipt from commit_intent.json") from error
        if store_attempted:
            with canonical._staged_update() as pending:
                for name in canonical._FILES:
                    shutil.copyfile(backup / name, Path(pending.root) / name)
        for name in reversed(changed):
            if (backup / name).exists():
                replace_copy(backup / name, ROOT / name)
            elif name in ready["files"] and ready["files"][name]["before"] is None:
                (ROOT / name).unlink(missing_ok=True)
        write(OUTPUT / "publication_failure.json", {"failed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(error), "source_rolled_back": True, "files_restored": changed, "store_restored": store_attempted})
        raise
    print("Corrected source, code, history and model published; recorder paused for installed QA", flush=True)


def register():
    from src.forward_store import create_study
    from src.forward_readout import register_plan
    from src.forward_recorder import replace_json
    from src.run_store import get_production_run_id
    published, qa = read(OUTPUT / "published.json"), read(OUTPUT / "qa_after/report.json")
    if not qa["passed"] or qa["run_id"] != published["run_id"] or get_production_run_id() != published["run_id"]:
        raise ValueError("Complete installed native serving QA first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    study = create_study(ROOT / f"data/forecasts/forward/{stamp}_rtv_features_044_{published['run_id']}")
    register_plan(study, study.path / "readout_plan.json")
    study.verify()
    settings = {"study": str(study.path), "model_id": published["run_id"],
        "protocol_sha256": study.protocol_sha256, "first_eligible_date": study.protocol["first_eligible_date"],
        "configured_at": datetime.now(timezone.utc).isoformat(),
        "scheduled_task": "HorseForecastRecorder_038", "scheduled_task_verified": False}
    replace_json(ROOT / "data/forecasts/recorder.json", settings)
    write(OUTPUT / "recorder_registration.json", settings)
    print(json.dumps(settings, indent=2), flush=True)


def stop_recorder():
    """Stop only child processes whose exact argument names the current study."""
    import psutil
    settings = read(ROOT / "data/forecasts/recorder.json")
    expected = str(Path(settings["study"]).resolve()).casefold()
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        args = process.info["cmdline"] or []
        if not any(Path(arg).name == "run_forecast_recorder.py" for arg in args):
            continue
        if "--study" not in args or args.index("--study") + 1 >= len(args):
            continue
        if str(Path(args[args.index("--study") + 1]).resolve()).casefold() == expected:
            matches.append(process)
    for process in matches:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(matches, timeout=10)
    if alive:
        raise ValueError("A matched recorder child did not stop")
    write(OUTPUT / "stopped_children.json", {"study": settings["study"], "pids": [p.pid for p in matches]})
    print({"stopped_recorder_children": [p.pid for p in matches]}, flush=True)


def recover():
    """Resolve a process interruption using the durable source-batch marker."""
    ready = read(OUTPUT / "prepared.json")
    if (OUTPUT / "published.json").exists():
        raise ValueError("The completed generation does not need interruption recovery")
    if not read(OUTPUT / "recorder_paused.json")["paused"]:
        raise ValueError("Keep the recorder paused during recovery")
    backup = OUTPUT / "previous_files"
    if not backup.is_dir():
        raise ValueError("No preserved publication to recover")
    with closing(sqlite3.connect((ROOT / "data/races.db").as_uri() + "?mode=ro", uri=True)) as con:
        table = con.execute("SELECT name FROM sqlite_master WHERE name='verified_result_batches'").fetchone()
        marker = con.execute("SELECT content_sha256 FROM verified_result_batches WHERE repair_id='rtv_verified_results_044'").fetchone() if table else None
    if marker:
        intent = read(OUTPUT / "commit_intent.json")
        if marker[0] != intent["application"]["content_sha256"]:
            raise ValueError("Committed source batch differs from the preserved intent")
        if read(POINTER)["run_id"] != ready["run_id"] or sha(RAW) != ready["raw_sha256"]:
            raise ValueError("Committed source/files differ; inspect the preserved generation")
        for name, digest in ready["files"].items():
            if sha(ROOT / name) != digest["after"]:
                raise ValueError("Committed installed source/cache differs: " + name)
        for name, digest in ready["store_files_sha256"].items():
            if sha(ROOT / "data/processed/feature_store" / name) != digest:
                raise ValueError("Committed installed history differs: " + name)
        write(OUTPUT / "published.json", intent)
        print("Committed generation verified; recovered the publication receipt", flush=True)
        return
    # An uncommitted SQLite transaction was rolled back by process death.
    # Restore all filesystem members, including files not yet reached by the
    # interrupted copy loop; completed before-files are immutable snapshots.
    from src.feature_store import FeatureStore
    store = FeatureStore()
    with store._staged_update() as pending:
        for name in store._FILES:
            shutil.copyfile(backup / name, Path(pending.root) / name)
    for name in [*ready["files"], "data/raw/race_results.csv", "models/production.json"]:
        if (backup / name).exists():
            replace_copy(backup / name, ROOT / name)
        elif name in ready["files"] and ready["files"][name]["before"] is None:
            (ROOT / name).unlink(missing_ok=True)
    write(OUTPUT / "interruption_recovered.json", {"restored_at": datetime.now(timezone.utc).isoformat(),
        "source_batch_committed": False, "previous_generation_restored": True, "recorder_still_paused": True})
    print("Previous filesystem generation restored; source transaction had rolled back", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "publish", "register-recorder", "stop-recorder", "recover"])
    {"prepare": prepare, "publish": publish, "register-recorder": register,
     "stop-recorder": stop_recorder, "recover": recover}[parser.parse_args().phase]()
