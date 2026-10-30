"""Qualify and optionally install the selected win-capacity upgrade.

Requires both022 periods to pass. Validate the final-quarter win head together
with its existing, same-cutoff place head; refit only the changed win component
through the installed model's settled cutoff. Keep recovery files and reuse a
completed refit on restart. No feature-store rewrite or extra place fit.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.assess_forecast_parity import upper_bound
from scripts.benchmark_forecasts import align_predictions
from src.data_processor import process_data
from src.deployment_validation import file_sha256 as sha, training_recipe, write_validation_report, load_validation_report, RECIPE_SOURCES
from src.featured_dataset import data_signature, fe_signature, _FE_SOURCE_FILES
from src.feature_store import FeatureStore
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns, placed_mask
from src.run_store import save_run, promote, RUNS_DIR, PRODUCTION_POINTER, get_production_run_id, load_run_meta
from src.training import TrainingConfig

COMPARISON = ROOT / "data/audit/model_improvement/022_full_capacity"
REFERENCE = ROOT / "data/audit/model_improvement/016_market_teacher"
OUTPUT = ROOT / "data/audit/model_improvement/023_capacity_adoption"
PARENT = "20261027_165656_616818"
PLACE_BRIER_NONINFERIORITY = .0001


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def atomic_bytes(path, contents):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".capacity-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capacity_defaults(source, cap=4000):
    """Replace only CLASSIFIER_PARAMS' integer tree cap, preserving all bytes."""
    tree = ast.parse(source)
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CLASSIFIER_PARAMS" for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError("Expected one explicit classifier parameter dictionary")
    dictionary = assignments[0].value
    values = [v for k, v in zip(dictionary.keys, dictionary.values)
              if isinstance(k, ast.Constant) and k.value == "n_estimators"]
    if len(values) != 1 or not isinstance(values[0], ast.Constant) or values[0].value not in (2000, cap):
        raise ValueError("Unexpected classifier tree cap")
    node = values[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1]))+node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1]))+node.end_col_offset
    result = source[:start]+str(cap).encode()+source[end:]
    node.value = cap
    if ast.dump(tree) != ast.dump(ast.parse(result)):
        raise ValueError("Configuration rewrite changed more than the declared tree cap")
    return result


def borrow_place(target, parent, origin):
    if target.feature_cols != parent.feature_cols or target.frameworks["place"] != parent.frameworks["place"]:
        raise ValueError("Place reuse requires identical input order and framework")
    if parent.place_model is None:
        raise ValueError("Parent has no fitted place component")
    for name in ("place_model", "place_platt_a", "place_platt_b", "place_iso_cal"):
        setattr(target, name, getattr(parent, name))
    target.eval_split_info["reused_place_component"] = origin


def aligned_outputs(predictor, frame):
    result = predictor.predict_races(frame)
    return frame[["race_id", "horse_name"]].merge(
        result[["race_id", "horse_name", "win_probability", "place_probability"]],
        on=["race_id", "horse_name"], how="left", sort=False, validate="one_to_one")


def certificate_settings(manifest):
    original = json.loads((REFERENCE / "manifest.json").read_text())
    cfg = TrainingConfig(**original["native_training_config"])
    cfg.name = "022 selected win-capacity recipe"
    cfg.hyperparams = manifest["params"]
    config.FEATURE_ALLOWLIST = manifest["features"]
    config.WIN_SOFT_LABEL_DISTILL, config.WIN_SOFT_LABEL_BLEND = True, .5
    config.CLASSIFIER_PARAMS = {**config.CLASSIFIER_PARAMS, "n_estimators": 4000}
    return cfg


def sealed_comparison():
    report = json.loads((COMPARISON / "report.json").read_text())
    if not report["passes_registered_advancement"]:
        raise ValueError("The selected capacity change did not pass both registered periods")
    manifest = json.loads((COMPARISON / "manifest.json").read_text())
    if manifest["periods"] != ["2025Q4", "2026Q1"]:
        raise ValueError("Unexpected validation periods")
    completion = COMPARISON / "completion.json"
    if not completion.exists():
        from scripts.validate_model_capacity import verify
        verify(manifest)
        evidence = {p.name: sha(p) for p in sorted(COMPARISON.iterdir()) if p.is_file()
                    and p.suffix in (".json", ".parquet", ".zip", ".joblib")}
        write(completion, {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "evidence_sha256": evidence})
    recorded = json.loads(completion.read_text())
    if any(sha(COMPARISON / name) != expected for name, expected in recorded["evidence_sha256"].items()):
        raise ValueError("Completed capacity evidence changed")
    return manifest, report


def ensure_certificate(cfg, checks):
    certificate = OUTPUT / "validation.json"
    if not certificate.exists():
        write_validation_report(checks["validation_run"], certificate, evidence_paths=[COMPARISON / name for name in
            ("manifest.json", "completion.json", "report.json", "2026Q1_predictions.parquet")]
            + [OUTPUT / "checks.json", OUTPUT / "qualification_source_snapshot.zip", OUTPUT / "planned_config.py"])
    return load_validation_report(certificate, training_recipe(cfg))


def qualify():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, report = sealed_comparison()
    if get_production_run_id() != PARENT:
        raise ValueError("The installed parent changed; review before replacing it")
    store = FeatureStore(); meta = store.meta()
    parent_meta = load_run_meta(PARENT)
    if (meta["fe_signature"] != fe_signature()
            or meta["fe_signature"] != parent_meta["training_config"]["signatures"]["fe_signature"]
            or meta["data_signature"] != parent_meta["training_config"]["signatures"]["data_signature"]):
        raise ValueError("Canonical history differs from the unchanged-feature parent")
    cfg = certificate_settings(manifest)
    if meta["fe_signature"] != fe_signature():
        raise ValueError("Changing the win cap unexpectedly changed feature semantics")
    recipe = training_recipe(cfg)
    if (OUTPUT / "checks.json").exists():
        checks = json.loads((OUTPUT / "checks.json").read_text())
        if checks["recipe"] != recipe:
            raise ValueError("The saved qualification uses another effective recipe")
        verify_checks(checks)
        ensure_certificate(cfg, checks)
        return cfg, checks
    processed = process_data(save=False)
    if data_signature(processed, cutoff=meta["last_date"]) != meta["data_signature"]:
        raise ValueError("Raw historical data differ from the feature store")
    del processed; gc.collect()
    p = RacePredictor(); p.load(str(COMPARISON / "2026Q1_candidate.joblib"))
    parent = RacePredictor(); parent.load(str(REFERENCE / "2026Q1_candidate.joblib"))
    frozen = pd.read_parquet(COMPARISON / "2026Q1_predictions.parquet")
    inputs = pd.read_parquet(REFERENCE / "inputs.parquet")
    test = inputs.loc[inputs.race_id.isin(frozen.race_id)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    np.testing.assert_allclose(align_predictions(p, test), frozen.candidate_probability, rtol=0, atol=1e-12)
    _, target_matrix = p._prepare_prediction_frame(test)
    _, parent_matrix = parent._prepare_prediction_frame(test)
    np.testing.assert_array_equal(target_matrix, parent_matrix)
    borrow_place(p, parent, {"source": str(REFERENCE / "2026Q1_candidate.joblib"),
                            "source_sha256": sha(REFERENCE / "2026Q1_candidate.joblib"), "as_of": "2025-12-22"})
    new, old = aligned_outputs(p, test), aligned_outputs(parent, test)
    np.testing.assert_allclose(new.win_probability, frozen.candidate_probability, rtol=0, atol=1e-12)
    _, matrix = p._prepare_prediction_frame(test)
    raw_place = p.place_model.predict_proba(matrix)[:, 1]
    sizes = test.groupby("race_id", sort=False).size().to_numpy()
    place_metrics = p._evaluate_place_model(new.place_probability.to_numpy(), raw_place, test, sizes)
    old_place = parent._evaluate_place_model(old.place_probability.to_numpy(), raw_place, test, sizes)
    nr, handicap = test.num_runners.to_numpy(), test.handicap.to_numpy().astype(bool)
    slots = np.where(nr <= 4, 3, np.where(nr <= 7, 2, np.where(nr <= 15, 3, np.where(handicap, 4, 3))))
    y = placed_mask(test.finish_position.to_numpy(), slots)
    change = np.square(new.place_probability-y)-np.square(old.place_probability-y)
    race_bound = upper_bound(change.to_numpy(), test.race_id.to_numpy(), seed=42)
    date_bound = upper_bound(change.to_numpy(), test.race_date.astype(str).to_numpy(), seed=42)
    if max(race_bound["upper95"], date_bound["upper95"]) > PLACE_BRIER_NONINFERIORITY:
        raise ValueError("The combined win/place forecasts fail place-score non-inferiority")
    for values in (new.win_probability, new.place_probability):
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError("Invalid combined probability")
    recovery = OUTPUT / "previous_generation"; recovery.mkdir(exist_ok=True)
    if not (recovery / "config.py").exists():
        shutil.copy2(ROOT / "config.py", recovery / "config.py")
        shutil.copy2(PRODUCTION_POINTER, recovery / "production.json")
    planned_config = capacity_defaults((ROOT / "config.py").read_bytes())
    (OUTPUT / "planned_config.py").write_bytes(planned_config)
    source_files = tuple(dict.fromkeys([*_FE_SOURCE_FILES, *RECIPE_SOURCES, "src/run_store.py",
        "src/data_scraper.py", "src/live_prediction.py", "config.py", "feature_sets/win.txt",
        "scripts/adopt_capacity_upgrade.py", "tests/test_capacity_adoption.py", "tests/test_live_serving_inputs.py"]))
    source_sha = {name: sha(ROOT / name) for name in source_files}
    archive = OUTPUT / "qualification_source_snapshot.zip"
    if archive.exists():
        import hashlib
        with zipfile.ZipFile(archive) as bundle:
            if any(hashlib.sha256(bundle.read(name)).hexdigest() != digest for name, digest in source_sha.items()):
                raise ValueError("Existing source archive describes another qualification")
    else:
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in source_files:
                bundle.write(ROOT / name, name)
            bundle.write(OUTPUT / "planned_config.py", "planned_config.py")
    final_quarter = next(r for r in report["periods"] if r["period"] == "2026Q1")
    win = final_quarter["candidate"]
    metrics = {"win_classifier": {"brier_score": win["brier"], "log_loss": win["binary_log_loss"], "race_log_loss": win["race_log_loss"]},
               "place_classifier": place_metrics}
    evaluation = {"mode": "external_walk_forward", "holdout_races": final_quarter["races"],
        "holdout_first_date": str(test.race_date.min().date()), "holdout_last_date": str(test.race_date.max().date()),
        "native_fit": dict(p.eval_split_info)}
    validation_run = save_run(predictor=p, name="022 capacity: separate Q1 win/place validation",
        model_type="race_predictor", data_source="frozen016 inputs; new win head and same-cutoff saved place head",
        data_rows=p.eval_split_info["training_runners"], n_features=len(p.feature_cols), elapsed_seconds=0,
        metrics=metrics, training_config={"config": cfg.to_dict(), "recipe": recipe, "split": evaluation,
            "signatures": {"fe_signature": meta["fe_signature"]}, "evidence_directory": str(COMPARISON),
            "scope": "Q1 metrics only; two-period pooled metrics describe separate historical models"})
    checks = {"passed": True, "recipe": recipe, "source_sha256": source_sha,
        "parent_run": PARENT, "parent_model_sha256": sha(Path(RUNS_DIR) / PARENT / "model.joblib"),
        "validation_run": validation_run, "source_store_sha256": {name: sha(Path(store.root) / name) for name in store._FILES},
        "meta": meta, "planned_config_sha256": sha(OUTPUT / "planned_config.py"),
        "place_metrics": place_metrics, "old_place_metrics": old_place,
        "place_race_brier_delta_upper95": race_bound, "place_date_brier_delta_upper95": date_bound,
        "place_brier_noninferiority_margin": PLACE_BRIER_NONINFERIORITY,
        "place_max_prediction_change": float(np.max(np.abs(new.place_probability-old.place_probability))),
        "default_change": "Only CLASSIFIER_PARAMS.n_estimators2000->4000; historical models already used explicit4000, other settings and code unchanged"}
    write(OUTPUT / "checks.json", checks)
    ensure_certificate(cfg, checks)
    write(OUTPUT / "status.json", {"stage": "qualified", "validation_run": validation_run, "production_changed": False})
    return cfg, checks


def verify_checks(checks):
    if not checks["passed"] or sha(Path(RUNS_DIR) / PARENT / "model.joblib") != checks["parent_model_sha256"]:
        raise ValueError("Qualified parent evidence changed")
    for name, expected in checks["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Qualified source changed: {name}")
    store = FeatureStore()
    for name, expected in checks["source_store_sha256"].items():
        if sha(Path(store.root) / name) != expected:
            raise ValueError("Qualified feature history changed")


def finish_forward(value):
    checks = json.loads((OUTPUT / "checks.json").read_text())
    if (get_production_run_id() != value["production_run"]
            or sha(Path(RUNS_DIR) / value["production_run"] / "model.joblib") != value["production_model_sha256"]
            or sha(ROOT / "config.py") != checks["planned_config_sha256"]):
        raise ValueError("Promoted artifact or planned configuration differs from its checkpoint")
    for name, expected in checks["source_sha256"].items():
        if name != "config.py" and sha(ROOT / name) != expected:
            raise ValueError("Serving source changed after the promoted checkpoint")
    from src.forward_store import create_study, ForwardStudy
    from src.forward_readout import register_plan
    study_root = ROOT / "data/forecasts/forward"
    suffix = "_capacity_upgrade_"+value["production_run"]
    existing = sorted(study_root.glob("*"+suffix))
    study = ForwardStudy(existing[-1]) if existing else create_study(study_root /
        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+suffix))
    if not (study.path / "readout_plan.json").exists():
        register_plan(study, study.path / "readout_plan.json")
    study.verify()
    value.update(stage="adopted", production_changed=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        forward_study=str(study.path), first_eligible_forward_date=study.protocol["first_eligible_date"])
    write(OUTPUT / "status.json", value)
    print(json.dumps(value, indent=2), flush=True)


def adopt():
    existing = OUTPUT / "status.json"
    state = json.loads(existing.read_text()) if existing.exists() else {}
    if state.get("stage") == "adopted":
        print(existing.read_text(), flush=True); return
    if state.get("stage") in ("deployment_saved", "promoted_pending_forward"):
        if get_production_run_id() == state["production_run"]:
            finish_forward(state); return
        checks = json.loads((OUTPUT / "checks.json").read_text())
        if get_production_run_id() != PARENT:
            raise ValueError("Another model was promoted during capacity adoption")
        if sha(ROOT / "config.py") == checks["planned_config_sha256"]:
            atomic_bytes(ROOT / "config.py", (OUTPUT / "previous_generation/config.py").read_bytes())
    cfg, checks = qualify()
    verify_checks(checks)
    recipe = checks["recipe"]
    certificate = OUTPUT / "validation.json"
    validation = load_validation_report(certificate, recipe)
    meta = checks["meta"]; store = FeatureStore()
    columns = list(dict.fromkeys([*json.loads((COMPARISON / "manifest.json").read_text())["features"],
        "race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap", "odds",
        "lengths_behind", "track", "off_time"]))
    frame = store.load_featured(columns=columns)
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    parent = RacePredictor(); parent.load(str(Path(RUNS_DIR) / PARENT / "model.joblib"))
    if get_feature_columns(frame) != parent.feature_cols:
        raise ValueError("The full refit would use different model inputs")
    path = OUTPUT / "full_refit.joblib"
    started = time.perf_counter()
    p = RacePredictor(random_seed=42, frameworks=cfg.frameworks, win_calibration_method=cfg.win_calibration_method)
    p._autotune_njobs = 2
    if path.exists():
        p.load(str(path))
        if p.eval_split_info.get("capacity_adoption_checks_sha256") != sha(OUTPUT / "checks.json"):
            raise ValueError("Saved full refit belongs to another qualification")
    else:
        write(OUTPUT / "status.json", {"stage": "full_win_refit", "validation_run": checks["validation_run"],
            "as_of": meta["last_date"], "production_changed": False})
        p.fit_as_of(frame, meta["last_date"], params=cfg.hyperparams, dataset_start=frame.race_date.min(), win_only=True)
        for key in ("training_runners", "training_races", "training_first_date", "training_last_date"):
            if p.eval_split_info[key] != parent.eval_split_info[key]:
                raise ValueError("The unchanged-data component refit has another fitting population")
        recent = frame.loc[frame.race_date.ge(frame.race_date.max()-pd.Timedelta(days=2))]
        np.testing.assert_array_equal(p._prepare_prediction_frame(recent)[1], parent._prepare_prediction_frame(recent)[1])
        borrow_place(p, parent, {"run_id": PARENT, "model_sha256": checks["parent_model_sha256"],
                               "as_of": parent.eval_split_info["as_of"]})
        p.eval_split_info["capacity_adoption_checks_sha256"] = sha(OUTPUT / "checks.json")
        verify_checks(checks)
        temporary = path.with_suffix(".incomplete.joblib"); p.save(str(temporary)); temporary.replace(path)
        write(OUTPUT / "status.json", {"stage": "full_refit_saved", "as_of": meta["last_date"], "production_changed": False})
    if training_recipe(cfg) != recipe or fe_signature() != meta["fe_signature"]:
        raise ValueError("Effective recipe changed during fitting")
    samples = frame.loc[frame.race_date.ge(frame.race_date.max()-pd.Timedelta(days=2))]
    restored = RacePredictor(); restored.load(str(path))
    before, after = aligned_outputs(p, samples), aligned_outputs(restored, samples)
    for name in ("win_probability", "place_probability"):
        np.testing.assert_allclose(before[name], after[name], rtol=0, atol=1e-12)
        if not np.isfinite(after[name]).all() or not after[name].between(0, 1).all():
            raise ValueError("Invalid saved probabilities")
    np.testing.assert_allclose(after.groupby("race_id").win_probability.sum(), 1, rtol=0, atol=1e-12)
    verify_checks(checks)
    deployment_cfg = TrainingConfig.deployment(certificate, meta["last_date"], threads=2)
    if state.get("stage") == "deployment_saved":
        deployment = state["production_run"]
        if sha(Path(RUNS_DIR) / deployment / "model.joblib") != state["production_model_sha256"]:
            raise ValueError("Completed deployment artifact changed before promotion")
    else:
        deployment = save_run(predictor=p, name="022 capacity: full win refit with unchanged saved place head",
            model_type="race_predictor", data_source="unchanged canonical151-input history", data_rows=len(frame),
            n_features=len(p.feature_cols), elapsed_seconds=time.perf_counter()-started, metrics={},
            training_config={"config": deployment_cfg.to_dict(), "recipe": recipe, "split": dict(p.eval_split_info),
                "validation_reference": validation, "signatures": {"fe_signature": meta["fe_signature"], "data_signature": meta["data_signature"]},
                "evals": {"scope": "Latest refit has no held-out performance estimate"}})
    value = {"stage": "deployment_saved", "production_run": deployment,
        "production_model_sha256": sha(Path(RUNS_DIR) / deployment / "model.joblib"),
        "validation_run": checks["validation_run"], "as_of": meta["last_date"],
        "fe_signature": meta["fe_signature"], "production_changed": False, "saved_model_replay_max_error": 0.,
        "recovery_directory": str(OUTPUT / "previous_generation")}
    write(OUTPUT / "status.json", value)
    # The only persisted default change follows the already completed refit.
    planned = (OUTPUT / "planned_config.py").read_bytes()
    if sha(OUTPUT / "planned_config.py") != checks["planned_config_sha256"]:
        raise ValueError("Planned capacity default changed")
    config_before = (ROOT / "config.py").read_bytes()
    try:
        atomic_bytes(ROOT / "config.py", planned)
        if fe_signature() != meta["fe_signature"] or not promote(deployment):
            raise ValueError("Refused incoherent capacity promotion")
    except Exception:
        atomic_bytes(ROOT / "config.py", config_before)
        raise
    value.update(stage="promoted_pending_forward", production_changed=True)
    write(OUTPUT / "status.json", value)
    finish_forward(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    arguments = parser.parse_args()
    adopt() if arguments.deploy else qualify()
