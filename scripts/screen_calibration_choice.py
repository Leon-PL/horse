"""019: choose an existing calibration family using training history only.

Replay just the native disposable calibration learner from the frozen 018
repair inputs, then split its later, out-of-sample predictions chronologically.
Reuse the saved final trees. No full model refit or additional quarter is run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.benchmark_forecasts import align_predictions
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.screen_feature_batch import OUTPUT as PREVIOUS, sha, runtime_versions, verify as verify_previous
from src.model import RacePredictor, prepare_multi_target_data

OUTPUT = ROOT / "data/audit/model_improvement/019_calibration_choice"
FAMILIES = ("native", "platt_only", "race_temperature")


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))


def temporal_split(frame, fraction=.7, purge_days=7):
    """Keep whole dates/fields; fit calibration before a purged later choice slab."""
    dates = pd.to_datetime(frame.race_date)
    if dates.isna().any() or not dates.is_monotonic_increasing:
        raise ValueError("Expected valid chronological calibration rows")
    race_dates = frame.groupby("race_id", sort=False).race_date.first()
    if len(race_dates) < 40:
        raise ValueError("Too few calibration races for a separate family choice")
    first_validation = pd.Timestamp(race_dates.iloc[int(len(race_dates)*fraction)])
    fit = dates.le(first_validation-pd.Timedelta(days=purge_days)) & dates.lt(first_validation)
    validation = dates.ge(first_validation)
    if frame.loc[fit, "race_id"].nunique() < 20 or frame.loc[validation, "race_id"].nunique() < 20:
        raise ValueError("Chronological calibration slabs are too small")
    return fit, validation


def groups(frame):
    sizes = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id, np.repeat(frame.race_id.drop_duplicates(), sizes)):
        raise ValueError("Expected contiguous complete fields")
    return sizes


def fit_map(family, frame):
    method = "race_temperature" if family == "race_temperature" else "platt_isotonic"
    fitted = RacePredictor(win_calibration_method=method)
    fitted._fit_win_calibrator(frame.raw_probability.to_numpy(), groups(frame), frame.finish_position.to_numpy())
    if family == "platt_only":
        fitted.win_iso_cal = None
    return fitted


def apply_map(fitted, frame):
    return fitted._calibrate_win_probs(frame.raw_probability.to_numpy(), groups(frame))


def eligible(result):
    delta = result["paired_delta_candidate_minus_baseline"]
    return bool(delta["race_log_loss"]["ci95"][1] < 0
        and delta["date_cluster_race_log_loss"]["ci95"][1] < 0
        and result["candidate"]["brier"] <= result["baseline"]["brier"]
        and result["candidate"]["binary_log_loss"] <= result["baseline"]["binary_log_loss"])


def prepare():
    old = json.loads((PREVIOUS / "manifest.json").read_text())
    verify_previous(old)
    if not (PREVIOUS / "report.json").exists():
        raise ValueError("Finish 018 before registering a modelling follow-up")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "manifest.json"
    if target.exists():
        raise ValueError("Calibration screen already registered; resume without --prepare")
    pins = ["manifest.json", "report.json", "diagnostics.json", "repairs_inputs.parquet",
            "repairs.joblib", "repairs_fit.json", "repairs_predictions.parquet", "baseline_predictions.parquet"]
    value = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_artifacts_sha256": {name: sha(PREVIOUS / name) for name in pins},
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                          "tests/test_calibration_choice.py": sha(ROOT / "tests/test_calibration_choice.py")},
        "runtime_versions": runtime_versions(), "families": FAMILIES,
        "inner_calibration_fit_fraction": .7, "purge_days": 7,
        "selection": "Prefer native unless an alternative improves race and date-cluster paired log-loss upper95 below zero, with no worse Brier or binary loss on the later training-only choice slab; select lowest winner log loss if both pass",
        "base": "Reuse exact 018 repaired151 final trees; replay only its disposable calibration learner and verify all native fitted calibration parameters",
        "test": {k: old[k] for k in ("test_start", "test_end", "training_cutoff")},
        "scope": "Retrospective development motivated by 018 diagnostics; already inspected test period, no fresh confirmation or automatic promotion",
        "extra_quarters_queued": False}
    write(target, value)
    print(json.dumps({"stage": "registered", "manifest_sha256": canonical_hash(value)}), flush=True)


def verify(manifest):
    old = json.loads((PREVIOUS / "manifest.json").read_text())
    verify_previous(old)
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("Runtime changed")
    for name, expected in manifest["previous_artifacts_sha256"].items():
        if sha(PREVIOUS / name) != expected:
            raise ValueError("Frozen 018 evidence changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Registered calibration procedure changed")
    return old


class CaptureCalibration(RacePredictor):
    def _fit_win_calibrator(self, probabilities, sizes, finish_positions):
        self.captured = np.asarray(probabilities).copy()
        return super()._fit_win_calibrator(probabilities, sizes, finish_positions)


def calibration_predictions(old, manifest):
    path, receipt = OUTPUT / "training_only_calibration.parquet", OUTPUT / "calibration_replay.json"
    if path.exists():
        saved = json.loads(receipt.read_text())
        if saved["manifest_sha256"] != canonical_hash(manifest) or saved["projection_sha256"] != sha(path):
            raise ValueError("Calibration replay belongs to another procedure")
        return pd.read_parquet(path)
    source = pd.read_parquet(PREVIOUS / "repairs_inputs.parquet",
                             filters=[("race_date", "<=", pd.Timestamp(old["training_cutoff"]))])
    config.FEATURE_ALLOWLIST = old["features"]["repairs"]
    config.WIN_SOFT_LABEL_DISTILL, config.WIN_SOFT_LABEL_BLEND = True, .5
    data = prepare_multi_target_data(source, dataset_start=pd.Timestamp(old["dataset_start"]), training_only=True)
    sizes = data["groups_train"]
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    dates = pd.DatetimeIndex(pd.to_datetime(data["train_df"].race_date).to_numpy()[starts])
    teacher, won = data["win_teacher_train"], data["y_train_won"]
    soft_y = np.where(teacher > 0, .5*teacher+.5*won, won)
    engine = CaptureCalibration(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    engine._autotune_njobs = 2
    passed = engine._fit_training_slab_calibrators(data["X_train"], sizes, won,
        data["y_train_placed"], data["fp_train"], old["params"], data["sample_weight_train"],
        race_dates=dates, soft_y=soft_y, fit_place=False)
    if not passed:
        raise ValueError("Native disposable calibration fit failed")
    saved = RacePredictor(); saved.load(str(PREVIOUS / "repairs.joblib"))
    np.testing.assert_allclose([engine.platt_a, engine.platt_b], [saved.platt_a, saved.platt_b], rtol=0, atol=1e-12)
    for attribute in ("X_thresholds_", "y_thresholds_"):
        np.testing.assert_array_equal(getattr(engine.win_iso_cal, attribute), getattr(saved.win_iso_cal, attribute))
    boundary = int(np.cumsum(sizes)[max(1, int(len(sizes)*.8))-1])
    columns = ["race_id", "race_date", "horse_name", "won", "finish_position", "num_runners"]
    result = data["train_df"].iloc[boundary:][columns].copy().reset_index(drop=True)
    result["raw_probability"] = engine.captured
    verify(manifest)
    result.to_parquet(path, index=False)
    write(receipt, {"manifest_sha256": canonical_hash(manifest), "projection_sha256": sha(path),
        "native_calibration_parameter_replay": "exact isotonic arrays; Platt coefficients within1e-12",
        "races": len(groups(result)), "rows": len(result), "dates": [str(result.race_date.min()), str(result.race_date.max())]})
    return result


def run():
    manifest = json.loads((OUTPUT / "manifest.json").read_text()); old = verify(manifest)
    slab = calibration_predictions(old, manifest)
    fit, validation = temporal_split(slab, manifest["inner_calibration_fit_fraction"], manifest["purge_days"])
    native = fit_map("native", slab.loc[fit])
    choice = slab.loc[validation].copy()
    choice["baseline_probability"] = apply_map(native, choice)
    comparisons = {}
    for family in FAMILIES[1:]:
        fitted = fit_map(family, slab.loc[fit])
        choice["candidate_probability"] = apply_map(fitted, choice)
        comparisons[family] = score_pair(choice)
    allowed = [name for name, result in comparisons.items() if eligible(result)]
    selected = min(allowed, key=lambda name: comparisons[name]["candidate"]["race_log_loss"]) if allowed else "native"
    print(json.dumps({"stage": "training_only_choice", "selected": selected,
                     "eligible": allowed, "comparisons": comparisons}), flush=True)
    full_map = fit_map(selected, slab)
    final = RacePredictor(); final.load(str(PREVIOUS / "repairs.joblib"))
    final.win_calibration_method = full_map.win_calibration_method
    final.platt_a, final.platt_b = full_map.platt_a, full_map.platt_b
    final.win_iso_cal, final.calibration_temp = full_map.win_iso_cal, full_map.calibration_temp
    test = pd.read_parquet(PREVIOUS / "repairs_inputs.parquet",
                           filters=[("race_date", ">=", pd.Timestamp(old["test_start"]))])
    frozen = pd.read_parquet(PREVIOUS / "repairs_predictions.parquet")
    result = frozen.drop(columns="probability").assign(baseline_probability=frozen.probability,
                                                       candidate_probability=align_predictions(final, test))
    if selected == "native":
        np.testing.assert_allclose(result.baseline_probability, result.candidate_probability, rtol=0, atol=1e-12)
    verify(manifest)
    path = OUTPUT / "selected.joblib"
    final.save(str(path))
    replay = RacePredictor(); replay.load(str(path))
    np.testing.assert_array_equal(result.candidate_probability, align_predictions(replay, test))
    result.to_parquet(OUTPUT / "selected_predictions.parquet", index=False)
    baseline = pd.read_parquet(PREVIOUS / "baseline_predictions.parquet")
    keys = ["race_id", "horse_name", "race_date", "won"]
    if not baseline[keys].equals(result[keys]):
        raise ValueError("Original baseline population differs")
    versus_original = result.assign(baseline_probability=baseline.probability)
    report = {"manifest_sha256": canonical_hash(manifest), "selected_family": selected,
        "training_only_choice": comparisons, "calibration_fit_races": int(slab.loc[fit, "race_id"].nunique()),
        "calibration_choice_races": int(choice.race_id.nunique()),
        "calibration_fit_last_date": str(slab.loc[fit, "race_date"].max())[:10],
        "calibration_choice_first_date": str(choice.race_date.min())[:10],
        "test_vs_repaired_native": score_pair(result), "test_vs_original_baseline": score_pair(versus_original),
        "saved_model_replay_max_error": 0., "selected_model_sha256": sha(path),
        "scope": manifest["scope"], "production_changed": False, "extra_quarters_queued": False}
    write(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    arguments = parser.parse_args()
    prepare() if arguments.prepare else run()
