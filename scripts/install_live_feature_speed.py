"""Rebuild and qualify an exact feature-generation migration, without tree fits.

Run from the actual project. Preserve the old store, code, weights and study;
activate only after an exhaustive stored-row/state comparison succeeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "data/audit/live_feature_speed_20261029/installation"


def write(name, value):
    path = OUTPUT / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare(candidate):
    from src.deployment_validation import file_sha256
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature
    from src.rating_reuse import freeze_rating_source
    from src.run_store import get_production_run_id, load_run_meta, RUNS_DIR

    manifest = json.loads((candidate / "data/audit/live_feature_speed_20261029/implementation.json").read_text())
    if fe_signature() != manifest["installed_fe_signature"]:
        raise ValueError("Installed generation changed before preparation")
    for name in ("src/feature_store.py", "src/feature_engineer.py"):
        if file_sha256(ROOT / name) != manifest["files"][name]["before_sha256"]:
            raise ValueError(f"Installed source changed: {name}")
        if file_sha256(candidate / name) != manifest["files"][name]["after_sha256"]:
            raise ValueError(f"Candidate source changed: {name}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    old = FeatureStore()
    backup = FeatureStore(str(OUTPUT / "previous_store"))
    with old._access():
        if old.meta().get("fe_signature") != fe_signature():
            raise ValueError("Store and installed code must agree before migration")
        for name in old._FILES:
            # Commits replace complete files; these immutable links retain the
            # previous generation without copying its multi-GB data twice.
            os.link(Path(old.root) / name, Path(backup.root) / name)
        freeze_rating_source(backup, OUTPUT / "rating_source.json")
    for name in ("src/feature_store.py", "src/feature_engineer.py", "models/production.json",
                 "data/forecasts/recorder.json"):
        target = OUTPUT / "previous_files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    parent = get_production_run_id()
    parent_meta = load_run_meta(parent)
    shutil.copy2(Path(RUNS_DIR) / parent / "meta.json", OUTPUT / "parent_meta.json")
    write("plan.json", {"prepared_at": datetime.now(timezone.utc).isoformat(), "parent_run": parent,
          "parent_model_sha256": file_sha256(Path(RUNS_DIR) / parent / "model.joblib"),
          "before_fe_signature": fe_signature(), "candidate_fe_signature": manifest["candidate_fe_signature"],
          "candidate_source_sha256": {name: manifest["files"][name]["after_sha256"]
                                      for name in ("src/feature_store.py", "src/feature_engineer.py")},
          "previous_store_meta": backup.meta(), "training_cutoff": parent_meta["training_config"]["split"]["training_last_date"],
          "policy": "Full rebuild; exact full stored-row and state parity required; original model bytes retained; zero fits"})
    print("Prepared immutable store/code/model/recorder references", flush=True)


def verify_sources(plan):
    from src.deployment_validation import file_sha256
    from src.featured_dataset import fe_signature
    if fe_signature() != plan["candidate_fe_signature"]:
        raise ValueError("Active implementation is not the qualified candidate")
    for name, expected in plan["candidate_source_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Candidate source changed during migration: {name}")


def fixed_batches(path, columns, size=8192):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pending, rows = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=columns):
        offset = 0
        while offset < batch.num_rows:
            count = min(size - rows, batch.num_rows - offset)
            pending.append(batch.slice(offset, count))
            rows += count
            offset += count
            if rows == size:
                yield pa.Table.from_batches(pending).to_pandas()
                pending, rows = [], 0
    if pending:
        yield pa.Table.from_batches(pending).to_pandas()


def rebuild():
    import joblib
    import pandas as pd
    import pyarrow.parquet as pq
    from itertools import zip_longest
    from src.feature_store import FeatureStore
    from src.deployment_validation import file_sha256
    from src.featured_dataset import data_signature
    from src.rating_reuse import ValidatedRatingCache
    from scripts.build_feature_store_batched import _build

    logging.getLogger("src").setLevel(logging.WARNING)
    started = time.monotonic()
    plan = json.loads((OUTPUT / "plan.json").read_text())
    verify_sources(plan)
    old = FeatureStore(str(OUTPUT / "previous_store"))
    rebuilt = FeatureStore(str(OUTPUT / "rebuilt_store"))
    if not rebuilt.exists():
        processed = old.load_processed()
        ratings = ValidatedRatingCache(old.root, OUTPUT / "rating_source.json")
        with rebuilt._staged_update() as pending:
            _build(pending, 60_000, processed=processed, ratings_cache=ratings)
        del processed, ratings
        gc.collect()
    verify_sources(plan)
    old_meta, new_meta = old.meta(), rebuilt.meta()
    for key in ("data_signature", "last_date", "processed_rows", "featured_rows", "atr_signature", "prize_reference_signature"):
        if old_meta.get(key) != new_meta.get(key):
            raise ValueError(f"Rebuilt input provenance differs: {key}")
    columns = sorted(pq.read_schema(old.featured_path).names)
    if columns != sorted(pq.read_schema(rebuilt.featured_path).names):
        raise ValueError("Rebuilt feature schema differs")
    compared = 0
    for before, after in zip_longest(fixed_batches(old.featured_path, columns), fixed_batches(rebuilt.featured_path, columns)):
        if before is None or after is None:
            raise ValueError("Rebuilt row count differs")
        pd.testing.assert_frame_equal(before, after, check_exact=True)
        compared += len(before)
        if compared % 65536 == 0:
            print(f"Exact stored-row comparison: {compared:,} / {old_meta['featured_rows']:,}", flush=True)
    pd.testing.assert_frame_equal(old.load_processed(), rebuilt.load_processed(), check_exact=True)
    state_receipts = {}
    for name, loader in (("ratings", "load_state"), ("counts", "load_counts_state")):
        filename = "ratings_state.pkl" if name == "ratings" else "counts_state.pkl"
        original_hash = file_sha256(Path(old.root) / filename)
        if original_hash == file_sha256(Path(rebuilt.root) / filename):
            state_receipts[name] = {"all_families_equal": True, "byte_identical": True, "sha256": original_hash}
            continue
        before = getattr(old, loader)()
        after = getattr(rebuilt, loader)()
        if before.keys() != after.keys():
            raise ValueError(f"Rebuilt {name} state families differ")
        differences = [str(key) for key in before if joblib.hash(before[key]) != joblib.hash(after[key])]
        if differences:
            raise ValueError(f"Rebuilt {name} state differs: {differences[:12]}")
        state_receipts[name] = {"families": len(before), "all_families_equal": True}
        del before, after
        gc.collect()
    processed = rebuilt.load_processed()
    prefix = data_signature(processed, cutoff=plan["training_cutoff"])
    parent = json.loads((OUTPUT / "parent_meta.json").read_text())
    if prefix != parent["training_config"]["signatures"]["data_signature"]:
        raise ValueError("Rebuilt training-prefix inputs differ from the fitted parent")
    write("qualification.json", {"passed": True, "qualified_at": datetime.now(timezone.utc).isoformat(),
          "elapsed_seconds": time.monotonic() - started, "rows_compared": compared, "columns_compared": len(columns),
          "all_values_and_dtypes_exact": True, "processed_exact": True, "state": state_receipts,
          "historical_boundary_preservation": new_meta.get("historical_boundary_preservation"),
          "training_prefix_signature": prefix, "store_meta": new_meta, "new_model_fits": 0})
    print("Rebuild and exhaustive exact-parity qualification passed", flush=True)


def activate():
    from src.deployment_validation import file_sha256, training_recipe
    from src.feature_store import FeatureStore
    from src.model import RacePredictor
    from src.run_store import save_run, get_production_run_id, promote, RUNS_DIR
    from src.training import TrainingConfig
    from src.forward_store import create_study
    from src.forward_readout import register_plan
    from src.forward_recorder import replace_json

    plan = json.loads((OUTPUT / "plan.json").read_text())
    verify_sources(plan)
    report = json.loads((OUTPUT / "qualification.json").read_text())
    if not report.get("passed") or not report.get("all_values_and_dtypes_exact"):
        raise ValueError("No exhaustive exact-parity qualification")
    if get_production_run_id() != plan["parent_run"]:
        raise ValueError("Production changed during qualification")
    parent_path = Path(RUNS_DIR) / plan["parent_run"] / "model.joblib"
    if file_sha256(parent_path) != plan["parent_model_sha256"]:
        raise ValueError("Parent model changed during qualification")
    parent = json.loads((OUTPUT / "parent_meta.json").read_text())
    tc = deepcopy(parent["training_config"])
    cfg = TrainingConfig(**tc["config"])
    if training_recipe(cfg) != tc["recipe"]:
        raise ValueError("Statistical recipe changed; this is not an exact migration")
    tc["signatures"]["fe_signature"] = plan["candidate_fe_signature"]
    tc["feature_generation_migration"] = {"source_run": plan["parent_run"],
        "source_fe_signature": plan["before_fe_signature"], "target_fe_signature": plan["candidate_fe_signature"],
        "qualification_sha256": file_sha256(OUTPUT / "qualification.json"),
        "qualification_path": str(OUTPUT / "qualification.json"), "new_fits": 0,
        "scope": "Original fitted model bytes and prior evaluation retained; exact rebuilt input/state equivalence; no new accuracy estimate"}
    predictor = RacePredictor()
    predictor.load(str(parent_path))
    run_id = save_run(predictor=predictor, name="Exact speed-only feature migration; original 038 model retained",
        model_type=parent["model_type"], data_source=parent["data_source"], data_rows=parent["data_rows"],
        n_features=parent["n_features"], elapsed_seconds=0, metrics=deepcopy(parent["metrics"]),
        train_metrics=deepcopy(parent.get("train_metrics")), training_config=tc)
    # Preserve the exact previously fitted artifact, rather than a reserialised equivalent.
    exact_model = Path(RUNS_DIR) / run_id / "model.exact.tmp"
    shutil.copy2(parent_path, exact_model)
    exact_model.replace(Path(RUNS_DIR) / run_id / "model.joblib")
    if file_sha256(Path(RUNS_DIR) / run_id / "model.joblib") != plan["parent_model_sha256"]:
        raise ValueError("Migration altered the fitted artifact")
    canonical, rebuilt = FeatureStore(), FeatureStore(str(OUTPUT / "rebuilt_store"))
    with canonical._staged_update() as pending:
        for name in canonical._FILES:
            if not os.path.samefile(Path(canonical.root) / name, OUTPUT / "previous_store" / name):
                raise ValueError(f"Installed store changed during qualification: {name}")
        for name in canonical._FILES:
            os.link(Path(rebuilt.root) / name, Path(pending.root) / name)
    if not promote(run_id):
        raise ValueError("Qualified migration could not be promoted")
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    study = create_study(ROOT / f"data/forecasts/forward/{day}_feature_speed_{run_id}")
    register_plan(study, study.path / "readout_plan.json")
    settings = {"study": str(study.path), "model_id": run_id, "protocol_sha256": study.protocol_sha256,
        "first_eligible_date": study.protocol["first_eligible_date"], "configured_at": datetime.now(timezone.utc).isoformat(),
        "scheduled_task": None, "scheduled_task_verified": False}
    replace_json(ROOT / "data/forecasts/recorder.json", settings)
    write("activation.json", {"run_id": run_id, "model_sha256": plan["parent_model_sha256"],
          "fe_signature": plan["candidate_fe_signature"], "study": str(study.path),
          "protocol_sha256": study.protocol_sha256, "new_fits": 0, "recorder_task_update_pending": True})
    print(json.dumps(settings, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "rebuild", "activate"))
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.candidate is None:
            parser.error("prepare requires --candidate")
        prepare(args.candidate)
    elif args.phase == "rebuild":
        rebuild()
    else:
        activate()
