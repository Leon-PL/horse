"""021: one fixed doubling of the matched cheap model's tree allowance.

Original inputs, labels, training dates, seed, rates and calibration remain
fixed. Native fitting selects capacity and refits as usual. Learning-curve
capture is observational and does not change or replace the native learner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.benchmark_forecasts import align_predictions
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.screen_feature_batch import OUTPUT as PREVIOUS, sha, runtime_versions, verify as verify_previous
from src.model import RacePredictor

OUTPUT = ROOT / "data/audit/model_improvement/021_model_capacity"


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))


class CurvePredictor(RacePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_curves = []

    def _train_win_classifier(self, *args, **kwargs):
        fitted = super()._train_win_classifier(*args, **kwargs)
        estimator = getattr(fitted, "reg", fitted)
        history = getattr(estimator, "evals_result_", None)
        if history:
            self.learning_curves.append({"training_rows": len(args[0]),
                "best_iteration": int(estimator.best_iteration_), "metrics": history})
        return fitted


def prepare():
    old = json.loads((PREVIOUS / "manifest.json").read_text()); verify_previous(old)
    if old["params"]["classifier"]["n_estimators"] != 500 or old["params"]["classifier"]["learning_rate"] != .025:
        raise ValueError("Expected the exact matched500-tree/.025 baseline")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise ValueError("Already registered; resume without --prepare")
    parameters = json.loads(json.dumps(old["params"]))
    parameters["classifier"]["n_estimators"] = 1000
    names = ["manifest.json", "baseline_inputs.parquet", "baseline_predictions.parquet", "baseline.joblib", "baseline_fit.json"]
    value = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_artifacts_sha256": {name: sha(PREVIOUS / name) for name in names},
        "source_sha256": sha(Path(__file__)), "runtime_versions": runtime_versions(), "params": parameters,
        "features": old["features"]["baseline"], "dataset_start": old["dataset_start"],
        "training_cutoff": old["training_cutoff"], "test_start": old["test_start"], "test_end": old["test_end"],
        "selection": "One predeclared1000-tree cap at unchanged.025 learning rate; shortlist only if race log loss and runner Brier improve versus exact500-tree baseline, report paired race/date uncertainty; no further capacity or rate search",
        "scope": "Already inspected retrospective development window, original151 inputs and same two-year fitting rows; win-only native fitting/calibration,seed42,two threads; no deployment or additional quarters"}
    write(OUTPUT / "manifest.json", value)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(Path(__file__), str(Path(__file__).relative_to(ROOT)))
    print(json.dumps({"stage": "registered", "manifest_sha256": canonical_hash(value)}), flush=True)


def verify(manifest):
    old = json.loads((PREVIOUS / "manifest.json").read_text()); verify_previous(old)
    if runtime_versions() != manifest["runtime_versions"] or sha(Path(__file__)) != manifest["source_sha256"]:
        raise ValueError("Registered capacity code or runtime changed")
    for name, expected in manifest["previous_artifacts_sha256"].items():
        if sha(PREVIOUS / name) != expected:
            raise ValueError("Matched baseline changed")


def run():
    manifest = json.loads((OUTPUT / "manifest.json").read_text()); verify(manifest)
    started = time.perf_counter()
    frame = pd.read_parquet(PREVIOUS / "baseline_inputs.parquet")
    config.FEATURE_ALLOWLIST = manifest["features"]
    config.WIN_SOFT_LABEL_DISTILL, config.WIN_SOFT_LABEL_BLEND = True, .5
    predictor = CurvePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 2
    path = OUTPUT / "candidate.joblib"
    if path.exists():
        predictor.load(str(path))
        if predictor.eval_split_info.get("capacity_screen_manifest_sha256") != canonical_hash(manifest):
            raise ValueError("Saved model belongs to another capacity test")
        curves = json.loads((OUTPUT / "learning_curves.json").read_text())
    else:
        train = frame.loc[frame.race_date.le(manifest["training_cutoff"])].copy()
        print(json.dumps({"stage": "fitting", "rows": len(train), "trees": 1000}), flush=True)
        predictor.fit_as_of(train, pd.Timestamp(manifest["training_cutoff"]), params=manifest["params"],
                            dataset_start=pd.Timestamp(manifest["dataset_start"]), win_only=True)
        curves = predictor.learning_curves
        write(OUTPUT / "learning_curves.json", curves)
        predictor.eval_split_info["capacity_screen_manifest_sha256"] = canonical_hash(manifest)
        verify(manifest)
        temporary = path.with_suffix(".incomplete.joblib")
        predictor.save(str(temporary)); temporary.replace(path)
    test = frame.loc[frame.race_date.ge(manifest["test_start"])].copy()
    baseline = pd.read_parquet(PREVIOUS / "baseline_predictions.parquet")
    keys = ["race_id", "horse_name", "race_date", "won"]
    if not baseline[keys].equals(test[keys].reset_index(drop=True)):
        raise ValueError("Capacity comparison contains different race fields")
    result = baseline.drop(columns="probability").assign(baseline_probability=baseline.probability,
                                                       candidate_probability=align_predictions(predictor, test))
    replay = RacePredictor(); replay.load(str(path))
    np.testing.assert_array_equal(result.candidate_probability, align_predictions(replay, test))
    result.to_parquet(OUTPUT / "paired_predictions.parquet", index=False)
    comparison = score_pair(result)
    summary = []
    for curve in curves:
        for dataset, metrics in curve["metrics"].items():
            for metric, values in metrics.items():
                summary.append({"training_rows": curve["training_rows"], "best_iteration": curve["best_iteration"],
                    "validation_dataset": dataset, "metric": metric, "iterations": len(values),
                    "loss_at_500": values[min(len(values), 500)-1], "final_recorded_loss": values[-1],
                    "best_recorded_loss": min(values)})
    verify(manifest)
    report = {"manifest_sha256": canonical_hash(manifest), "comparison": comparison,
        "point_improves_both": bool(comparison["candidate"]["race_log_loss"] < comparison["baseline"]["race_log_loss"]
                                    and comparison["candidate"]["brier"] < comparison["baseline"]["brier"]),
        "learning_curve_summary": summary, "fit": predictor.eval_split_info,
        "saved_model_replay_max_error": 0., "model_sha256": sha(path), "elapsed_seconds": time.perf_counter()-started,
        "scope": manifest["scope"], "production_changed": False, "additional_quarters_queued": False}
    write(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    arguments = parser.parse_args()
    prepare() if arguments.prepare else run()
