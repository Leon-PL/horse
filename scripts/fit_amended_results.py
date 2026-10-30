"""Run the prepared native042 refit with exact resumable component caching."""
from datetime import datetime, timezone
from copy import deepcopy
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.canonical_training_cache import CheckpointPredictor
from scripts import prepare_amendment_refit as driver
from src import training
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore
from src.run_store import RUNS_DIR, load_run_meta

OUTPUT = driver.OUTPUT
MATH_SOURCES = ("config.py", "feature_sets/win.txt", "src/model.py", "src/probability_calibration.py",
    "src/each_way.py", "src/bet_settlement.py", "src/betfair.py", "src/utils.py", "src/run_store.py")


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    if (OUTPUT / "fitted.json").exists() or (OUTPUT / "run_saved.json").exists():
        raise ValueError("A current refit is already saved; qualify/recover it without fitting again")
    built = driver.CASE / "history_rebuild/rebuilt.json"
    if not built.exists():
        raise ValueError("Finish the native corrected history build first")
    identity = {"purpose": "Same native train_run with process-local staged directories and resumable numeric component fits",
        "final_plan_sha256": sha(OUTPUT / "final_plan.json"), "rebuilt_history_sha256": sha(built),
        "implementation_sha256": {name: sha(ROOT / name) for name in
            ("scripts/fit_amended_results.py", "scripts/prepare_amendment_refit.py", "scripts/canonical_training_cache.py")},
        "runtime_model_sources": {name: sha(ROOT / name) for name in MATH_SOURCES},
        "teacher_source_sha256": sha(ROOT / "data/processed/betfair_bsp.parquet"),
        "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "lightgbm", "scikit-learn", "joblib")}}}
    manifest = OUTPUT / "execution_plan.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if any(recorded[name] != value for name, value in identity.items()):
            raise ValueError("Checkpoint implementation/runtime/inputs changed")
    else:
        write(manifest, {"created_at_utc": datetime.now(timezone.utc).isoformat(), **identity})
    capsule = OUTPUT / "runtime_source_snapshot.zip"
    capture = {**identity["runtime_model_sources"], **identity["implementation_sha256"],
               **json.loads((OUTPUT / "final_plan.json").read_text())["recipe"]["source_sha256"]}
    if not capsule.exists():
        with zipfile.ZipFile(capsule, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, digest in capture.items():
                if sha(ROOT / name) != digest:
                    raise ValueError("Source changed before the runtime snapshot")
                archive.write(ROOT / name, arcname=name)
    import hashlib
    with zipfile.ZipFile(capsule) as archive:
        for name, digest in capture.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Runtime source snapshot differs from the pinned fit")
    attempts = OUTPUT / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = attempts / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".json")
    write(attempt, {"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_plan_sha256": sha(manifest), "canonical_store_at_start": FeatureStore().meta(),
        "component_cache_files_at_start": len(list((OUTPUT / "component_cache").glob("*.joblib")))})
    checkpointed = partial(CheckpointPredictor, checkpoint_directory=OUTPUT / "component_cache",
                           checkpoint_manifest=sha(manifest))
    ordinary_save = training.save_run

    def observe_save(*args, **kwargs):
        if any(sha(ROOT / name) != digest for name, digest in identity["runtime_model_sources"].items()):
            raise ValueError("Runtime probability math changed while fitting")
        if sha(ROOT / "data/processed/betfair_bsp.parquet") != identity["teacher_source_sha256"]:
            raise ValueError("Market-teacher source changed while fitting")
        kwargs = dict(kwargs)
        kwargs["training_config"] = deepcopy(kwargs["training_config"])
        kwargs["training_config"]["runtime_model_sources"] = identity["runtime_model_sources"]
        kwargs["training_config"]["runtime_model_source_manifest"] = str(manifest)
        run_id = ordinary_save(*args, **kwargs)
        meta = load_run_meta(run_id)
        write(OUTPUT / "run_saved.json", {"saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id, "model_sha256": sha(Path(RUNS_DIR) / run_id / "model.joblib"),
            "signatures": meta["training_config"]["signatures"],
            "execution_plan_sha256": sha(manifest), "promoted": False})
        return run_id

    # The existing tested cache hashes every numeric input, target, split and
    # fit argument. It calls the unchanged native fitting method on cache misses.
    with patch.object(training, "RacePredictor", checkpointed), patch.object(training, "save_run", observe_save):
        driver.fit()
    if any(sha(ROOT / name) != digest for name, digest in identity["runtime_model_sources"].items()):
        raise ValueError("Runtime probability math changed before final qualification")
    write(OUTPUT / "execution_completed.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_plan_sha256": sha(manifest), "fitted_receipt_sha256": sha(OUTPUT / "fitted.json"),
        "runtime_model_sources": identity["runtime_model_sources"], "runtime_sources_unchanged": True,
        "source_snapshot_sha256": sha(capsule)})
    print("Prepared native refit completed; component checkpoints and saved-run receipt retained.", flush=True)


if __name__ == "__main__":
    main()
