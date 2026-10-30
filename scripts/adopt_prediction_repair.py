"""Select completed registered repairs, validate, refit and adopt a coherent generation.

No fitting or deployment happens without --deploy. Unsuccessful comparisons
leave the current production pointer intact. Completed evidence stays immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
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
from scripts.replay_market_snapshots import digest
from src.data_processor import process_data
from src.deployment_validation import training_recipe, write_validation_report, load_validation_report
from src.featured_dataset import data_signature, fe_signature
from src.feature_store import FeatureStore
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.run_store import save_run, promote, RUNS_DIR, PRODUCTION_POINTER, get_production_run_id, load_run_meta
from src.training import TrainingConfig

BASE = ROOT / "data/audit/model_improvement"
EXPERIMENTS = ("015_nonfinisher_history", "016_market_teacher")
OUTPUT = BASE / "015_016_adoption"
RESULTS_DOC = ROOT / "docs/MODEL_HISTORY_TEACHER_015_016.md"


def write(path, contents):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(contents, indent=2, default=str))
    temporary.replace(path)


def completed(directory):
    receipt = json.loads((directory / "completion.json").read_text())
    for name, expected in receipt["evidence_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Completed evidence changed: {directory.name}/{name}")
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if (canonical_hash(manifest) != receipt["manifest_sha256"]
            or report["manifest_sha256"] != receipt["manifest_sha256"]
            or set(receipt["completed_periods"]) != {"2025Q3", "2025Q4", "2026Q1"}
            or {fold["period"] for fold in report["folds"]} != set(receipt["completed_periods"])):
        raise ValueError("Completed evidence does not cover the registered three quarters")
    return report


def better(scores, reference):
    return (scores["race_log_loss"] < reference["race_log_loss"]
            and scores["brier"] <= reference["brier"]
            and scores["binary_log_loss"] <= reference["binary_log_loss"])


def write_results(selection, old, reports):
    original = reports[EXPERIMENTS[0]]["pooled"]["baseline"]
    lines = ["# Corrected histories and market teaching: completed results", "",
        "Retrospective chronological research on all three registered quarters (2025Q3, 2025Q4, 2026Q1), 9,356 races.", "",
        "| Recipe | Race log loss | Runner Brier | Top-pick accuracy |",
        "|---|---:|---:|---:|"]
    for label, metrics in [("Original 149 inputs", original), ("014: corrected margins", old),
                           *[(name, reports[name]["pooled"]["candidate"]) for name in EXPERIMENTS]]:
        lines.append(f"| {label} | {metrics['race_log_loss']:.6f} | {metrics['brier']:.8f} | {metrics['top1_accuracy']:.2%} |")
    lines += ["", "Lower log loss and Brier are better. Accuracy is a secondary diagnostic.", ""]
    for name in EXPERIMENTS:
        report = reports[name]
        interval = report["pooled"]["paired_delta_candidate_minus_baseline"]["race_log_loss"]
        market = report["market"]["metrics"]
        lines += [f"{name}: log-loss change versus original {interval['mean']:+.6f}; race-bootstrap 95% interval {interval['ci95']}.",
            f"On the same 8,942 market-covered fields, candidate log loss {market['candidate']['race_log_loss']:.6f}; market {market['market']['race_log_loss']:.6f}.",
            f"Original market gap closed (point estimate): {report['market_gap_closed_point_estimate']:.2%}.", ""]
    lines += [f"Selected for final qualification: **{selection['selected'] or 'none'}**.",
        "Selection requires the registered complete-period gate and improvement over the best completed 014 model; teaching must also improve over 015. No race-type subgroup selection was used.", "",
        "The 50/50 teacher uses historical complete-field market probabilities only during training; its live forecasts do not require odds.",
        "A separate final-quarter win/place artifact validates the recipe. Any later full-history deployment refit has no new held-out performance estimate.", "",
        "The public-profile prototype and raw-source-recency finding are not inputs to these comparisons.",
        "Original evidence is preserved under data/audit/model_improvement/015_nonfinisher_history and 016_market_teacher.",
        "Qualification and deployment state: data/audit/model_improvement/015_016_adoption/status.json.", ""]
    RESULTS_DOC.write_text("\n".join(lines))


def select():
    old = completed(BASE / "014_margin_repair")["pooled"]["candidate"]
    reports = {name: completed(BASE / name) for name in EXPERIMENTS}
    hard = reports[EXPERIMENTS[0]]["pooled"]["candidate"]
    eligible = []
    for name, report in reports.items():
        scores = report["pooled"]["candidate"]
        if (report["candidate_passes_numeric_advancement"] and better(scores, old)
                and (name == EXPERIMENTS[0] or better(scores, hard))):
            eligible.append(name)
    selected = min(eligible, key=lambda name: reports[name]["pooled"]["candidate"]["race_log_loss"]) if eligible else None
    hard_predictions = pd.read_parquet(BASE / EXPERIMENTS[0] / "paired_predictions.parquet")
    soft_predictions = pd.read_parquet(BASE / EXPERIMENTS[1] / "paired_predictions.parquet")
    paired = soft_predictions.drop(columns="baseline_probability").merge(
        hard_predictions[["race_id", "horse_name", "race_date", "won", "candidate_probability"]].rename(
            columns={"candidate_probability": "baseline_probability"}),
        on=["race_id", "horse_name", "race_date", "won"], how="left", validate="one_to_one")
    if len(paired) != len(hard_predictions) or paired.baseline_probability.isna().any():
        raise ValueError("Head-to-head comparison lost a field or outcome")
    selection = {"selected": selected, "teacher_vs_hard": score_pair(paired),
            "rule": "Primary registered gate, lower pooled LL and no worse Brier/binaryLL than014; teacher must also improve on015; all3quarters, no subgroup selection",
            "scope": "Retrospective selection, no independent confirmation or betting-edge claim",
            "completion_sha256": {name: digest(BASE / name / "completion.json") for name in EXPERIMENTS}}
    write_results(selection, old, reports)
    return selection


def native_settings(manifest):
    cfg = TrainingConfig(**manifest["native_training_config"])
    spec = manifest["registration"]
    config.WIN_SOFT_LABEL_DISTILL = bool(spec.get("market_teacher", False))
    config.WIN_SOFT_LABEL_BLEND = float(spec.get("market_teacher_blend", .5))
    return cfg


def verify_precision_upgrade(directory, manifest):
    """Prove the only source upgrade preserves the evaluated float32 inputs."""
    with zipfile.ZipFile(directory / "source_snapshot.zip") as archive:
        old = archive.read("src/model.py")
    if hashlib.sha256(old).hexdigest() != manifest["source_sha256"]["src/model.py"]:
        raise ValueError("Original model source archive changed")
    start = old.index(b"class _IdentityScaler:")
    end = old.index(b"    def __reduce__", start)
    part = old[start:end]
    if part.count(b"return np.asarray(X)") != 2:
        raise ValueError("Unexpected original precision contract")
    part = part.replace(b"return np.asarray(X)", b"return np.asarray(X, dtype=np.float32)")
    part = part.replace(b"No-op scaler for tree-based models (keeps API compatibility).",
                        b"Use the same float32 input precision in training and live predictions.")
    if (ROOT / "src/model.py").read_bytes() != old[:start]+part+old[end:]:
        raise ValueError("Model source changed beyond the exact input-precision repair")
    frame = pd.read_parquet(directory / "inputs.parquet")
    if any(frame[name].dtype != np.dtype("float32") for name in manifest["features"]):
        raise ValueError("Registered model inputs were not uniformly float32")
    proofs = []
    spec = manifest["registration"]
    source_store = ROOT / spec["feature_store_directory"] if spec.get("feature_store_directory") else directory / "candidate_store"
    if digest(source_store / "meta.json") != manifest["feature_store_meta_sha256"]:
        raise ValueError("Original corrected-store generation changed")
    for period in manifest["registration"]["candidate_periods"]:
        frozen = pd.read_parquet(directory / f"{period}_predictions.parquet")
        test = frame.loc[frame.race_id.isin(frozen.race_id)].sort_values(
            ["race_date", "race_id", "horse_name"], kind="stable")
        predictor = RacePredictor(); predictor.load(str(directory / f"{period}_candidate.joblib"))
        actual = align_predictions(predictor, test)
        np.testing.assert_allclose(actual, frozen.candidate_probability, atol=1e-12, rtol=0)
        proof = {"period": period, "races": frozen.race_id.nunique(),
                 "max_win_error": float(np.max(np.abs(actual-frozen.candidate_probability.to_numpy())))}
        raw = pd.read_parquet(source_store / "featured.parquet",
            columns=list(dict.fromkeys([*manifest["features"], "race_id", "horse_name", "race_date"])),
            filters=[("race_date", ">=", test.race_date.min()), ("race_date", "<=", test.race_date.max())])
        raw = raw.loc[raw.race_id.isin(frozen.race_id)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
        if not raw[["race_id", "horse_name"]].reset_index(drop=True).equals(test[["race_id", "horse_name"]].reset_index(drop=True)):
            raise ValueError("Full-precision serving replay lost a runner")
        live_precision = align_predictions(predictor, raw)
        np.testing.assert_allclose(live_precision, frozen.candidate_probability, atol=1e-12, rtol=0)
        proof["max_original_store_precision_win_error"] = float(np.max(np.abs(live_precision-frozen.candidate_probability.to_numpy())))
        if "place_probability" in frozen:
            received = predictor.predict_races(raw)
            places = test[["race_id", "horse_name"]].merge(
                received[["race_id", "horse_name", "place_probability"]],
                on=["race_id", "horse_name"], how="left", validate="one_to_one").place_probability.to_numpy()
            np.testing.assert_allclose(places, frozen.place_probability, atol=1e-12, rtol=0)
            proof["max_original_store_precision_place_error"] = float(np.max(np.abs(places-frozen.place_probability.to_numpy())))
        proofs.append(proof)
    result = {"passed": True, "old_model_source_sha256": manifest["source_sha256"]["src/model.py"],
        "new_model_source_sha256": digest(ROOT / "src/model.py"), "replays": proofs,
        "scope": "Exact scaler-only cast; all evaluated inputs already float32, their values/forecasts unchanged; future training and serving now both cast to the evaluated precision"}
    write(OUTPUT / "precision_upgrade.json", result)
    return result


def capture_qualification_sources(destination, names, *, root=ROOT):
    """Keep the ZIP handle scoped separately from the recovery directory."""
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as source_zip:
        for name in dict.fromkeys(names):
            source_zip.write(Path(root) / name, name)


def install_saved_store(candidate, canonical, recovery_root, expected_hashes):
    """Publish five verified files, preserving the original generation on retry."""
    recovery_root = Path(recovery_root)
    if set(expected_hashes) != set(candidate._FILES):
        raise ValueError("Qualification must fingerprint all five store files")
    recovery_root.mkdir(parents=True, exist_ok=True)
    for name, expected in expected_hashes.items():
        if digest(Path(candidate.root) / name) != expected:
            raise ValueError(f"Candidate generation changed: {name}")
    receipt = recovery_root / "store_backup_sha256.json"
    with canonical._access():
        needed = sum((Path(candidate.root) / n).stat().st_size for n in candidate._FILES)
        needed += sum((Path(canonical.root) / n).stat().st_size for n in canonical._FILES
                      if not (recovery_root / n).exists())
        if shutil.disk_usage(canonical.root).free < needed:
            raise ValueError("Not enough free disk space for the old backup and staged new generation")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
        else:
            previous = {name: digest(Path(canonical.root) / name) for name in canonical._FILES}
            write(receipt, previous)
        for name, expected in previous.items():
            target = recovery_root / name
            if not target.exists():
                if digest(Path(canonical.root) / name) != expected:
                    raise ValueError(f"Cannot reconstruct the original backup: {name}")
                shutil.copy2(Path(canonical.root) / name, target)
            if digest(target) != expected:
                raise ValueError(f"Original recovery file changed: {name}")
        with canonical._staged_update() as staged:
            for name in canonical._FILES:
                shutil.copy2(Path(candidate.root) / name, Path(staged.root) / name)
        if any(digest(Path(canonical.root) / name) != expected for name, expected in expected_hashes.items()):
            raise ValueError("Published generation does not match the qualified files")


def settings_text(source, desired):
    for name, value in desired.items():
        source, count = re.subn(rf"(?m)^{name}\s*=\s*[^\r\n]+", f"{name} = {value!r}", source)
        if count != 1:
            raise ValueError(f"Cannot persist qualified setting {name}")
    return source


def adopt(selection):
    directory = BASE / selection["selected"]
    manifest = json.loads((directory / "manifest.json").read_text())
    checks = json.loads((OUTPUT / "checks.json").read_text())
    if not checks["passed"] or any(digest(ROOT / name) != expected for name, expected in checks["sha256"].items()):
        raise ValueError("Deployment checks changed or did not pass")
    for relative, expected in manifest["source_sha256"].items():
        if relative != "src/model.py" and digest(ROOT / relative) != expected:
            raise ValueError(f"Current sources differ from the qualified recipe: {relative}")
    precision = verify_precision_upgrade(directory, manifest)
    for relative, expected in manifest["market_teacher_inputs_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Market-teacher inputs changed")
    live = json.loads((ROOT / "data/audit/live_features_repair_015.json").read_text())
    if (not live["passed"] or not live.get("all_known_horses_matched_history")
            or live["fe_signature"] != manifest["candidate_fe_signature"]):
        raise ValueError("Corrected live feature generation has not passed")
    store = FeatureStore(str(BASE / EXPERIMENTS[0] / "candidate_store"))
    meta = store.meta()
    if meta["fe_signature"] != fe_signature() or fe_signature() != manifest["candidate_fe_signature"]:
        raise ValueError("Candidate store and source generation differ")
    processed = process_data(save=False)
    if data_signature(processed, cutoff=meta["last_date"]) != meta["data_signature"]:
        raise ValueError("Raw history changed since the corrected store was built")
    del processed
    gc.collect()
    store_hashes = {name: digest(Path(store.root) / name) for name in store._FILES}
    archive = OUTPUT / "previous_generation"
    if archive.exists():
        raise ValueError("Adoption already started; inspect its checkpoint rather than overwriting recovery files")
    archive.mkdir(parents=True)
    write(OUTPUT / "source_store_sha256.json", store_hashes)
    for relative in ("config.py", "feature_sets/win.txt"):
        shutil.copy2(ROOT / relative, archive / Path(relative).name)
    if Path(PRODUCTION_POINTER).exists():
        shutil.copy2(PRODUCTION_POINTER, archive / "production.json")
    # Preserve the default path; contents select columns, not numeric FE values.
    feature_path = ROOT / "feature_sets/win.txt"
    feature_path.write_text(f"# Qualified {directory.name}; see its registered evidence and 015_016_adoption/precision_upgrade.json\n"
                            + "\n".join(manifest["features"]) + "\n")
    cfg = native_settings(manifest)
    recipe = training_recipe(cfg)
    expected_recipe = json.loads(json.dumps(manifest["native_training_recipe"]))
    expected_recipe["source_sha256"]["src/model.py"] = precision["new_model_source_sha256"]
    if recipe != expected_recipe or fe_signature() != meta["fe_signature"]:
        raise ValueError("Effective deployment recipe differs from validation")
    source_archive = OUTPUT / "qualification_source_snapshot.zip"
    capture_qualification_sources(source_archive, [*manifest["source_sha256"], "feature_sets/win.txt",
        "scripts/adopt_prediction_repair.py", "scripts/smoke_prediction_repair.py",
        "tests/test_model_input_precision.py", "tests/test_prediction_repair_adoption.py"])
    predictor = RacePredictor()
    predictor.load(str(directory / "2026Q1_candidate.joblib"))
    if predictor.place_model is None or predictor.feature_cols != manifest["features"]:
        raise ValueError("Qualification requires the final-quarter full model")
    predictions = pd.read_parquet(directory / "2026Q1_predictions.parquet")
    inputs = pd.read_parquet(directory / "inputs.parquet")
    test = inputs.loc[inputs.race_id.isin(predictions.race_id)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable")
    np.testing.assert_allclose(align_predictions(predictor, test), predictions.candidate_probability, atol=1e-12, rtol=0)
    report = json.loads((directory / "2026Q1_report.json").read_text())
    if not np.isfinite(report["place_metrics"]["brier_calibrated"]):
        raise ValueError("No held-out place validation")
    win = report["candidate"]
    metrics = {"win_classifier": {"brier_score": win["brier"], "log_loss": win["binary_log_loss"],
                                   "race_log_loss": win["race_log_loss"]},
               "place_classifier": report["place_metrics"]}
    fit = dict(predictor.eval_split_info)
    evaluation = {"mode": "external_walk_forward", "holdout_races": report["races"],
                  "holdout_first_date": str(test.race_date.min().date()),
                  "holdout_last_date": str(test.race_date.max().date()), "native_fit": fit}
    validation_run = save_run(predictor=predictor, name=f"{directory.name}: external 2026Q1 validation",
        model_type="race_predictor", data_source="registered corrected feature store",
        data_rows=fit["training_runners"], n_features=len(predictor.feature_cols), elapsed_seconds=0,
        metrics=metrics, training_config={"config": cfg.to_dict(), "recipe": recipe, "split": evaluation,
            "signatures": {"fe_signature": meta["fe_signature"]},
            "input_precision_qualification": precision,
            "evidence_directory": str(directory), "scope": "Q1 metrics only; pooled three-fold scores belong to separate models"})
    certificate = OUTPUT / "validation.json"
    write_validation_report(validation_run, certificate, evidence_paths=[directory / name for name in
        ("manifest.json", "completion.json", "2026Q1_report.json", "2026Q1_predictions.parquet")]
        + [OUTPUT / "precision_upgrade.json", OUTPUT / "checks.json", source_archive])
    validation = load_validation_report(certificate, recipe)
    del predictor, inputs, test, predictions
    gc.collect()
    columns = list(dict.fromkeys(manifest["features"] + ["race_id", "horse_name", "race_date", "won",
        "finish_position", "num_runners", "handicap", "odds", "lengths_behind", "beaten_lengths", "track", "off_time"]))
    frame = store.load_featured(columns=columns)
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    if get_feature_columns(frame) != manifest["features"]:
        raise ValueError("Full refit would use another feature order")
    predictor = RacePredictor(random_seed=42, frameworks=dict(cfg.frameworks), win_calibration_method=cfg.win_calibration_method)
    predictor._autotune_njobs = 2
    started = time.monotonic()
    write(OUTPUT / "status.json", {"stage": "full_refit", "selected": directory.name,
        "validation_run": validation_run, "as_of": meta["last_date"], "production_changed": False})
    predictor.fit_as_of(frame, meta["last_date"], params=cfg.hyperparams, dataset_start=frame.race_date.min())
    if training_recipe(cfg) != recipe or fe_signature() != meta["fe_signature"]:
        raise ValueError("Sources changed during the full refit")
    if any(digest(Path(store.root) / name) != expected for name, expected in store_hashes.items()):
        raise ValueError("Candidate history changed during fitting")
    for relative, expected in manifest["market_teacher_inputs_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Market teacher changed during the full refit")
    if digest(ROOT / "config.py") != manifest["source_sha256"]["config.py"]:
        raise ValueError("Default configuration changed during fitting")
    deployment_cfg = TrainingConfig.deployment(certificate, meta["last_date"], threads=2)
    deployment = save_run(predictor=predictor, name=f"{directory.name}: full-history deployment",
        model_type="race_predictor", data_source="qualified corrected feature store", data_rows=len(frame),
        n_features=len(predictor.feature_cols), elapsed_seconds=time.monotonic()-started,
        metrics={}, training_config={"config": deployment_cfg.to_dict(), "recipe": recipe,
            "seed": cfg.seed,
            "split": dict(predictor.eval_split_info), "validation_reference": validation,
            "signatures": {"fe_signature": meta["fe_signature"], "data_signature": meta["data_signature"]},
            "evals": {"scope": "Latest refit has no held-out performance estimate"}})
    restored = RacePredictor(); restored.load(str(Path(RUNS_DIR) / deployment / "model.joblib"))
    replay = frame.loc[frame.race_date.ge(frame.race_date.max()-pd.Timedelta(days=2))]
    before, after = predictor.predict_races(replay), restored.predict_races(replay)
    for column in ("win_probability", "place_probability"):
        np.testing.assert_allclose(before[column], after[column], atol=1e-12, rtol=0)
        if not after[column].between(0, 1).all():
            raise ValueError("Invalid saved-model probabilities")
    np.testing.assert_allclose(after.groupby("race_id").win_probability.sum(), 1, atol=1e-12, rtol=0)
    # Keep the complete previous store as a recoverable copy. The transaction
    # publishes only the already-built candidate's five coherent files.
    canonical = FeatureStore()
    install_saved_store(store, canonical, archive, store_hashes)
    text = settings_text((ROOT / "config.py").read_bytes().decode(),
                         {name: getattr(config, name) for name in ("WIN_SOFT_LABEL_DISTILL", "WIN_SOFT_LABEL_BLEND")})
    (ROOT / "config.py").write_bytes(text.encode())
    if fe_signature() != meta["fe_signature"] or not promote(deployment):
        raise ValueError("Refused incoherent production promotion")
    result = {"stage": "adopted", "selected": directory.name, "production_run": deployment,
        "validation_run": validation_run, "fe_signature": meta["fe_signature"],
        "as_of": meta["last_date"], "saved_model_replay_max_error": 0.0,
        "production_changed": True, "previous_generation": str(archive),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write(OUTPUT / "status.json", result)
    with RESULTS_DOC.open("a") as stream:
        stream.write(f"\nProduction upgraded to `{deployment}` using settled history through {meta['last_date']}. "
                     "The table above describes the separate historical evaluation fits. "
                     "The previous model, store, settings and pointer remain recoverable.\n")
    # The old source-pinned study cannot accept forecasts from a new model.
    # Start a new prospective record without changing any old protocol/attempt.
    from src.forward_store import create_study
    from src.forward_readout import register_plan
    study_root = ROOT / "data/forecasts/forward"
    study = create_study(study_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_prediction_repair"))
    register_plan(study, study.path / "readout_plan.json")
    study.verify()
    result["forward_study"] = str(study.path)
    result["first_eligible_forward_date"] = study.protocol["first_eligible_date"]
    write(OUTPUT / "status.json", result)
    print(json.dumps(result, indent=2), flush=True)


def resume_saved_deployment(run_id):
    """Finish installation from a qualified completed fit; never retrain it."""
    saved = load_run_meta(run_id)
    tc = saved["training_config"]
    recipe = tc["recipe"]
    desired = {name: recipe["model_config"][name]
               for name in ("WIN_SOFT_LABEL_DISTILL", "WIN_SOFT_LABEL_BLEND")}
    for name, value in desired.items():
        setattr(config, name, value)
    cfg = TrainingConfig(**tc["config"])
    if training_recipe(cfg) != recipe:
        raise ValueError("Saved deployment recipe differs from current model code or settings")
    validation = load_validation_report(cfg.validation_report, recipe)
    if validation != tc["validation_reference"]:
        raise ValueError("Saved deployment references different qualification evidence")
    store = FeatureStore(str(BASE / "015_nonfinisher_history/candidate_store"))
    meta = store.meta()
    if (meta["fe_signature"] != fe_signature() or meta["fe_signature"] != validation["fe_signature"]
            or tc["signatures"] != {"fe_signature": meta["fe_signature"], "data_signature": meta["data_signature"]}
            or tc["split"].get("as_of") != meta["last_date"]):
        raise ValueError("Saved deployment does not match the corrected history generation")
    expected = json.loads((OUTPUT / "source_store_sha256.json").read_text())
    for name, checksum in expected.items():
        if digest(Path(store.root) / name) != checksum:
            raise ValueError(f"Qualified history changed: {name}")
    processed = process_data(save=False)
    if data_signature(processed, cutoff=meta["last_date"]) != meta["data_signature"]:
        raise ValueError("Current raw history differs from the trained source")
    del processed
    gc.collect()
    recovery_root = OUTPUT / "previous_generation"
    previous = json.loads((recovery_root / "production.json").read_text())
    if get_production_run_id() not in (previous["run_id"], run_id):
        raise ValueError("Production has independently changed; refusing to overwrite it")
    predictor = RacePredictor()
    artifact = Path(RUNS_DIR) / run_id / "model.joblib"
    artifact_sha = digest(artifact)
    predictor.load(str(artifact))
    if (predictor.clf_model is None or predictor.place_model is None
            or predictor.feature_cols != validation["features"]
            or predictor.eval_split_info != tc["split"]):
        raise ValueError("Saved fitted artifact differs from its qualified metadata")
    columns = list(dict.fromkeys([*predictor.feature_cols, "race_id", "horse_name", "race_date",
                                  "won", "finish_position", "odds", "handicap", "num_runners"]))
    replay = pd.read_parquet(store.featured_path, columns=columns,
        filters=[("race_date", ">=", pd.Timestamp(meta["last_date"])-pd.Timedelta(days=2))])
    replay = replay.loc[valid_race_mask(replay)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    before = predictor.predict_races(replay)
    copy_path = OUTPUT / "resume_roundtrip.joblib"
    predictor.save(str(copy_path))
    restored = RacePredictor(); restored.load(str(copy_path))
    after = restored.predict_races(replay)
    replay_error = 0.
    for col in ("win_probability", "place_probability"):
        np.testing.assert_allclose(before[col], after[col], atol=1e-12, rtol=0)
        replay_error = max(replay_error, float(np.max(np.abs(before[col].to_numpy()-after[col].to_numpy()))))
        if not after[col].between(0, 1).all():
            raise ValueError("Invalid saved probability")
    np.testing.assert_allclose(after.groupby("race_id").win_probability.sum(), 1, atol=1e-12, rtol=0)
    pending = pd.read_parquet(ROOT / "data/audit/live_features_repair_015.parquet")
    live = restored.predict_races(pending)
    np.testing.assert_allclose(live.groupby("race_id").win_probability.sum(), 1, atol=1e-12, rtol=0)
    if not all(live[col].between(0, 1).all() for col in ("win_probability", "place_probability")):
        raise ValueError("Pending-card prediction failed")
    updated_settings = settings_text((ROOT / "config.py").read_bytes().decode(), desired)
    # Original validation/checks/archive files remain unchanged. This records
    # the operational recovery separately from the statistical qualification.
    write(OUTPUT / "resume_install_receipt.json", {"production_run": run_id,
        "model_sha256": artifact_sha, "metadata_sha256": digest(Path(RUNS_DIR) / run_id / "meta.json"),
        "validation_report_sha256": digest(Path(cfg.validation_report)),
        "installer_source_sha256": digest(Path(__file__)),
        "replay_races": int(replay.race_id.nunique()), "pending_runners": len(pending),
        "maximum_save_reload_error": replay_error, "retraining_performed": False})
    write(OUTPUT / "status.json", {"stage": "installing_saved_run", "selected": "016_market_teacher",
        "production_run": run_id, "production_changed": get_production_run_id() == run_id})
    install_saved_store(store, FeatureStore(), recovery_root, expected)
    temporary_settings = ROOT / "config.py.installing"
    temporary_settings.write_bytes(updated_settings.encode())
    temporary_settings.replace(ROOT / "config.py")
    if training_recipe(cfg) != recipe or fe_signature() != meta["fe_signature"] or not promote(run_id):
        raise ValueError("Refused incoherent production promotion")
    if digest(artifact) != artifact_sha:
        raise ValueError("Installation changed the trained model")
    result = {"stage": "adopted", "selected": "016_market_teacher", "production_run": run_id,
        "validation_run": validation["validation_run_id"], "fe_signature": meta["fe_signature"],
        "as_of": meta["last_date"], "production_changed": True, "retraining_performed": False,
        "saved_model_replay_max_error": replay_error, "previous_generation": str(recovery_root),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    write(OUTPUT / "status.json", result)
    from src.forward_store import create_study, ForwardStudy
    from src.forward_readout import register_plan
    study_root = ROOT / "data/forecasts/forward"
    existing = [p.parent for p in study_root.glob("*/protocol.json")
                if json.loads(p.read_text()).get("model_id") == run_id]
    study = ForwardStudy(sorted(existing)[-1]) if existing else create_study(
        study_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_prediction_repair"))
    if not (study.path / "readout_plan.json").exists():
        register_plan(study, study.path / "readout_plan.json")
    study.verify()
    result.update(forward_study=str(study.path), first_eligible_forward_date=study.protocol["first_eligible_date"])
    write(OUTPUT / "status.json", result)
    if (OUTPUT / "failure.json").exists():
        (OUTPUT / "failure.json").replace(OUTPUT / "resolved_install_failure.json")
    with RESULTS_DOC.open("a", encoding="utf-8") as stream:
        stream.write(f"\nProduction upgraded to `{run_id}` using settled history through {meta['last_date']}. "
            "Installation resumed from the saved completed fit; no retraining. "
            "The historical table describes separate evaluation fits. Previous model, store and settings are recoverable.\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Wait for the existing native comparisons, at most four hours")
    parser.add_argument("--resume-run", help="Install an already completed qualified refit, without training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    OUTPUT.mkdir(exist_ok=True)
    if args.resume_run:
        if args.deploy or args.wait:
            parser.error("--resume-run installs its saved artifact directly; omit --wait and --deploy")
        return resume_saved_deployment(args.resume_run)
    status_path = OUTPUT / "status.json"
    if args.deploy and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("stage") == "adopted":
            if get_production_run_id() != status["production_run"]:
                raise ValueError("Qualified deployment was already adopted; production has subsequently changed")
            print(json.dumps(status, indent=2), flush=True)
            return
    if args.wait:
        write(OUTPUT / "status.json", {"stage": "waiting_for_registered_comparisons", "production_changed": False})
        deadline = time.monotonic()+4*3600
        while not all((BASE / name / "completion.json").exists() for name in EXPERIMENTS):
            if time.monotonic() > deadline:
                raise TimeoutError("Native comparison did not complete; production unchanged")
            log = ROOT / "data/audit/model-repair-native-015-016.log"
            if "Traceback (most recent call last)" in log.read_text(errors="replace")[-12000:]:
                raise RuntimeError("Native comparison failed; inspect its log before deployment")
            time.sleep(15)
    selection = select()
    write(OUTPUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)
    if args.deploy:
        if selection["selected"] is None:
            write(OUTPUT / "status.json", {"stage": "no_qualified_candidate", "production_changed": False})
        else:
            adopt(selection)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(exist_ok=True)
        write(OUTPUT / "failure.json", {"error": str(exc), "production_run_at_failure": get_production_run_id(),
              "failed_at_utc": datetime.now(timezone.utc).isoformat(), "action": "Inspect status and preserved checkpoints; do not waive guards"})
        raise
