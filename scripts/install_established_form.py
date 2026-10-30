"""Validate the explicitly chosen029 inputs, then refit normal win/place heads.

Call only after reviewing the completed matched comparison. Objective repairs
apply in either choice. Exact native win-component checkpoints are reused when
their actual numeric training arguments match; place heads are freshly fitted.
"""
import argparse
import importlib.metadata
from functools import partial
import gc
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_established_form import write
from scripts.canonical_training_cache import CheckpointPredictor
from src.deployment_validation import file_sha256 as sha, training_recipe, write_validation_report, load_validation_report, RECIPE_SOURCES
from src.featured_dataset import fe_signature, _FE_SOURCE_FILES
from src.forecast_evaluation import valid_race_mask
from src.model import placed_mask
from src.run_store import save_run, promote, get_production_run_id, RUNS_DIR, load_run_meta
from src.training import TrainingConfig

COMPARISON = ROOT / "data/audit/established_form_029/native_canonical"
OUTPUT = ROOT / "data/audit/established_form_029/installation"


def verify(manifest, cfg):
    if training_recipe(cfg) != manifest["recipe"] or fe_signature() != manifest["fe_signature"]:
        raise ValueError("The chosen normal recipe changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Chosen source changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Chosen evidence changed: {name}")
    for name, expected in manifest["runtime"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Chosen runtime changed: {name}")


def run(choice, reason):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initial = json.loads((COMPARISON / "manifest.json").read_text())
    report = json.loads((COMPARISON / "report.json").read_text())
    selected = initial[choice+"_features"]
    if report["model_promoted"]:
        raise ValueError("Do not reinterpret a comparison's historical state")
    # Changing selection happens only after the single comparison is complete.
    feature_file = ROOT / "feature_sets/win.txt"
    backup = OUTPUT / "previous_win.txt"
    if not backup.exists():
        backup.write_bytes(feature_file.read_bytes())
    expected_text = "# 029 reviewed established-form selection; normal correctness repairs always apply\n"+"\n".join(selected)+"\n"
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.exists():
        feature_file.write_text(expected_text, encoding="utf-8")
    cfg = TrainingConfig(name="029 separate Q1 validation", refresh_data=False, threads=2)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["choice"] != choice or manifest["decision_reason"] != reason:
            raise ValueError("Resume the recorded decision without silently changing it")
    else:
        # Model maths/settings must be the same ones used by the selected arm.
        for name in (*RECIPE_SOURCES, "config.py"):
            if sha(ROOT/name) != initial["source_sha256"][name]:
                raise ValueError("Do not reuse historical native fits after changing the learner")
        sources = list(dict.fromkeys([*_FE_SOURCE_FILES, *RECIPE_SOURCES, "config.py", "feature_sets/win.txt",
            "scripts/install_established_form.py", "scripts/canonical_training_cache.py"]))
        store_meta = json.loads((ROOT / "data/processed/feature_store/meta.json").read_text())
        manifest = {"choice": choice, "decision_reason": reason, "features": selected,
            "runtime": initial["runtime"],
            "parent_run": get_production_run_id(), "recipe": training_recipe(cfg), "fe_signature": fe_signature(),
            "source_sha256": {name: sha(ROOT/name) for name in sources},
            "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [COMPARISON/"manifest.json", COMPARISON/"inputs.parquet",
                COMPARISON/"report.json", COMPARISON/"predictions.parquet", ROOT/"data/processed/betfair_bsp.parquet"]},
            "comparison_checkpoint_scope": sha(COMPARISON/"manifest.json"), "historical_cutoff": initial["history_cutoff"],
            "production_cutoff": store_meta["last_date"], "store_meta": store_meta}
        write(manifest_path, manifest)
        with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT/name, name)
    verify(manifest, cfg)
    certificate = OUTPUT / "validation.json"
    if not certificate.exists():
        frame = pd.read_parquet(COMPARISON / "inputs.parquet")
        dataset_start = frame.race_date.min()
        frame = frame.loc[valid_race_mask(frame)].copy()
        test = frame.loc[frame.race_date.between(initial["evaluation_start"], initial["evaluation_end"])].copy()
        train = frame.loc[frame.race_date.le(manifest["historical_cutoff"])].copy()
        del frame
        gc.collect()
        predictor = CheckpointPredictor(random_seed=cfg.seed, frameworks=cfg.frameworks, win_calibration_method=cfg.win_calibration_method,
            checkpoint_directory=COMPARISON / "components" / choice, checkpoint_manifest=manifest["comparison_checkpoint_scope"])
        predictor._autotune_njobs = 2
        path = OUTPUT / "historical_full_model.joblib"
        started = time.monotonic()
        write(OUTPUT / "status.json", {"stage": "historical_full_heads", "choice": choice})
        if path.exists():
            predictor.load(str(path))
        else:
            predictor.fit_as_of(train, manifest["historical_cutoff"], dataset_start=dataset_start, win_only=False)
            verify(manifest, cfg)
            predictor.save(str(path))
        if (predictor.eval_split_info.get("training_checkpoint_manifest") != manifest["comparison_checkpoint_scope"]
                or str(predictor.eval_split_info.get("as_of"))[:10] != manifest["historical_cutoff"]):
            raise ValueError("Historical checkpoint belongs to another scope")
        if predictor.feature_cols != selected:
            raise ValueError("Normal training input order differs from the chosen comparison")
        forecast = predictor.predict_races(test)
        paired = test[["race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap"]].merge(
            forecast[["race_id", "horse_name", "win_probability", "place_probability"]], on=["race_id", "horse_name"], validate="one_to_one")
        original = pd.read_parquet(COMPARISON / "predictions.parquet")
        check = paired.merge(original[["race_id", "horse_name", choice+"_probability"]], on=["race_id", "horse_name"], validate="one_to_one")
        np.testing.assert_allclose(check.win_probability, check[choice+"_probability"], rtol=0, atol=1e-12)
        _, matrix = predictor._prepare_prediction_frame(test)
        raw_place = predictor.place_model.predict_proba(matrix)[:, 1]
        place = predictor._evaluate_place_model(paired.place_probability.to_numpy(), raw_place, test,
            test.groupby("race_id", sort=False).size().to_numpy())
        nr, hc = paired.num_runners.to_numpy(), paired.handicap.to_numpy().astype(bool)
        slots = np.where(nr<=4, 3, np.where(nr<=7, 2, np.where(nr<=15, 3, np.where(hc, 4, 3))))
        place["brier_calibrated"] = float(np.mean((paired.place_probability.to_numpy()-placed_mask(paired.finish_position.to_numpy(), slots))**2))
        win = report["comparison"][choice]
        metrics = {"win_classifier": {"brier_score": win["brier"], "log_loss": win["binary_log_loss"], "race_log_loss": win["race_log_loss"]},
                   "place_classifier": place}
        paired.to_parquet(OUTPUT / "historical_predictions.parquet", index=False)
        write(OUTPUT / "historical_scores.json", {"win": win, "place": place, "maximum_win_reproduction_error": float((check.win_probability-check[choice+"_probability"]).abs().max())})
        run_id = save_run(predictor=predictor, name="029 chosen established-form recipe: separate Q1 validation", model_type="race_predictor",
            data_source="Canonical rebuilt normal features; full unchanged Q1 fields", data_rows=predictor.eval_split_info["training_runners"],
            n_features=len(selected), elapsed_seconds=time.monotonic()-started, metrics=metrics,
            training_config={"config": cfg.to_dict(), "recipe": manifest["recipe"], "signatures": {"fe_signature": manifest["fe_signature"]},
                "split": {"mode": "external_walk_forward", "holdout_races": int(test.race_id.nunique()),
                    "holdout_first_date": str(test.race_date.min().date()), "holdout_last_date": str(test.race_date.max().date()),
                    "native_fit": predictor.eval_split_info}})
        write_validation_report(run_id, certificate, evidence_paths=[manifest_path, OUTPUT/"source_snapshot.zip", OUTPUT/"historical_scores.json",
            OUTPUT/"historical_predictions.parquet", COMPARISON/"manifest.json", COMPARISON/"report.json", COMPARISON/"predictions.parquet"])
        del predictor, train, test, paired, check, matrix, raw_place
        gc.collect()
    validation = load_validation_report(certificate, manifest["recipe"])
    status = json.loads((OUTPUT/"status.json").read_text()) if (OUTPUT/"status.json").exists() else {}
    run_id = status.get("run_id")
    if not run_id:
        from src import training
        if get_production_run_id() != manifest["parent_run"]:
            raise ValueError("Another model was installed during this work")
        deploy = TrainingConfig.deployment(certificate, manifest["production_cutoff"], name="029 established-horse corrections and reviewed form inputs", threads=2)
        ordinary = training.RacePredictor
        training.RacePredictor = partial(CheckpointPredictor, checkpoint_directory=OUTPUT/"components/current", checkpoint_manifest=sha(manifest_path))
        write(OUTPUT/"status.json", {"stage": "normal_full_refit", "choice": choice, "cutoff": manifest["production_cutoff"]})
        try:
            run_id = training.train_run(deploy)
        finally:
            training.RacePredictor = ordinary
        write(OUTPUT/"status.json", {"stage": "saved", "run_id": run_id, "choice": choice})
    verify(manifest, cfg)
    if load_run_meta(run_id)["training_config"]["signatures"]["fe_signature"] != manifest["fe_signature"]:
        raise ValueError("Saved deployment uses different features")
    if not promote(run_id):
        raise ValueError("The new model and normal feature store do not match")
    write(OUTPUT/"status.json", {"stage": "installed", "run_id": run_id, "choice": choice,
        "model_sha256": sha(Path(RUNS_DIR)/run_id/"model.joblib"), "fe_signature": manifest["fe_signature"]})
    print((OUTPUT/"status.json").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choice", choices=["baseline", "candidate"], required=True)
    parser.add_argument("--reason", required=True, help="Explicit reviewed interpretation of the completed comparison")
    args = parser.parse_args()
    run(args.choice, args.reason)
