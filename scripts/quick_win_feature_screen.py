"""Cached baseline and one capped candidate fit; no production qualification.

Uses the already-frozen029 inputs. Capacity, dates and recipe are fixed, with
no inner validation, calibration learner, place learner or current-data refit.
Completed baselines are cached by their actual numeric training arguments and
learner settings, independently of which candidate is being investigated.
"""
from datetime import datetime, timezone
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config

from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.canonical_training_cache import cached_component, fit_key
from src.deployment_validation import file_sha256, RECIPE_SOURCES
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, prepare_multi_target_data

REFERENCE = ROOT / "data/audit/established_form_029/native_canonical"
OUTPUT = ROOT / "data/audit/handicap_weight_review_030/quick_screen"
START, CUTOFF = "2023-09-24", "2025-09-23"
TEST_START, TEST_END = "2025-10-01", "2025-12-31"
PARAMS = {"n_estimators": 400, "learning_rate": .04}


def run(candidate_path, projection_path=None, *, allow_new_baseline=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads((REFERENCE / "manifest.json").read_text())
    if file_sha256(REFERENCE / "inputs.parquet") != reference["inputs_sha256"]:
        raise ValueError("The frozen feature inputs changed")
    baseline = reference["baseline_features"]
    candidate = [line.strip() for line in Path(candidate_path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")]
    if not candidate or len(candidate) != len(set(candidate)):
        raise ValueError("Choose nonempty distinct candidate inputs")
    added = [c for c in candidate if c not in baseline]
    if added and projection_path is None:
        raise ValueError("New inputs require a separately audited historical projection")
    teacher = ROOT / "data/processed/betfair_bsp.parquet"
    if file_sha256(teacher) != reference["teacher_sha256"]:
        raise ValueError("Keep the same teaching observations")
    started = time.monotonic()
    frame = pd.read_parquet(REFERENCE / "inputs.parquet", filters=[
        ("race_date", ">=", pd.Timestamp(START)), ("race_date", "<=", pd.Timestamp(TEST_END))])
    if projection_path is not None:
        projection = pd.read_parquet(projection_path, columns=["race_id", "horse_name", "race_date"]+added)
        # Prevent silently changing an existing feature, identity, order or date.
        if set(added) & set(frame):
            raise ValueError("Projection cannot replace baseline inputs")
        frame = frame.merge(projection, on=["race_id", "horse_name", "race_date"],
            how="left", validate="one_to_one", indicator=True)
        if not frame.pop("_merge").eq("both").all():
            raise ValueError("Projection lacks frozen input rows")
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    train = frame.loc[frame.race_date.le(CUTOFF)].copy()
    test = frame.loc[frame.race_date.between(TEST_START, TEST_END)].copy()
    del frame
    with patch.object(config, "FEATURE_ALLOWLIST", baseline):
        data = prepare_multi_target_data(train, training_only=True, dataset_start="2020-09-15")
    del train
    gc.collect()
    if data["feature_cols"] != baseline:
        raise ValueError("Preserve the native baseline feature order")
    all_columns = baseline+added
    x_train = data["X_train"]
    x_test = data["scaler"].transform(test[baseline].to_numpy())
    if added:
        # The native scaler is deliberately identity/float32. Reuse its
        # transform so all baseline numeric training arguments stay identical.
        x_train = np.column_stack((x_train, data["scaler"].transform(data["train_df"][added].to_numpy())))
        x_test = np.column_stack((x_test, data["scaler"].transform(test[added].to_numpy())))
    y = data["y_train_won"]
    weights = data["sample_weight_train"]
    soft_y = None
    if config.WIN_SOFT_LABEL_DISTILL:
        blend = float(config.WIN_SOFT_LABEL_BLEND)
        observations = data["win_teacher_train"]
        soft_y = np.where(observations > 0, blend*observations+(1-blend)*y, y)
    runtime = {name:importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scikit-learn", "lightgbm")}
    sources = {name:file_sha256(ROOT/name) for name in RECIPE_SOURCES}
    semantics = {"cache_schema": "quick_win_v1", "seed": 42, "threads": 2,
        "frameworks": {"classifier": "lgbm", "place": "lgbm"}, "runtime": runtime,
        "sources": sources, "config_sha256": file_sha256(ROOT/"config.py"),
        "native_classifier_params": config.CLASSIFIER_PARAMS, "overrides": PARAMS,
        "soft_labels": bool(config.WIN_SOFT_LABEL_DISTILL)}
    # Only learner semantics and actual training args define a component key.
    # Candidate names, reports, and unrelated feature code cannot force an
    # unchanged baseline to fit again.
    scope = canonical_hash(semantics)
    registration = {"scope": "Cheap development triage; no independent confirmation or market-parity claim",
        "training_dates": [START, CUTOFF], "test_dates": [TEST_START, TEST_END],
        "training_runners": len(y), "test_runners": len(test), "test_races": int(test.race_id.nunique()),
        "baseline_features": baseline, "candidate_features": candidate, "semantics": semantics,
        "feature_inputs_sha256": reference["inputs_sha256"], "teacher_sha256": reference["teacher_sha256"],
        "candidate_file_sha256": file_sha256(candidate_path), "script_sha256": file_sha256(Path(__file__)),
        "projection_sha256": file_sha256(projection_path) if projection_path is not None else None,
        "calibration": "Identical simple race normalisation in both arms; no calibrator fitting",
        "selection_rule": "One preset screen only. No automatic deployment, capacity search or favourable retuning."}
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != registration:
            raise ValueError("A different screen is already registered here")
    else:
        manifest_path.write_text(json.dumps(registration, indent=2), encoding="utf-8")
    result = test[["race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap", "horse_prev_races"]].copy()
    cache = ROOT / "data/research/quick_win_components"
    fits = []
    for label, columns in (("baseline", baseline), ("candidate", candidate)):
        indices = [all_columns.index(name) for name in columns]
        x = x_train[:, indices]
        learner = RacePredictor(random_seed=42, frameworks=semantics["frameworks"])
        learner.feature_cols = columns
        learner._autotune_njobs = 2
        args = (x, y)
        kwargs = {"params": PARAMS, "sample_weight": weights, "soft_y": soft_y,
                  "scale_pos_weight": (len(y)-y.sum())/max(y.sum(),1)}
        key = fit_key(scope, "win", columns, args, kwargs)
        reused = (cache/(key+".joblib")).exists()
        if label == "baseline" and projection_path is not None and not reused and not allow_new_baseline:
            raise ValueError("Expected an exactly reusable baseline; inspect numeric preparation before fitting")
        print(json.dumps({"stage":label, "trees":400, "features":len(columns), "reused":reused}),flush=True)
        model = cached_component(cache, key, learner._train_win_classifier, args, kwargs)
        raw = model.predict_proba(x_test[:,indices])[:,1]
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Invalid raw probabilities")
        total = pd.Series(raw, index=result.index).groupby(result.race_id,sort=False).transform("sum").to_numpy()
        if np.any(total <= 0):
            raise ValueError("Empty probability mass")
        result[label+"_probability"] = raw/total
        fits.append({"arm":label, "cache_key":key, "reused":reused, "features":len(columns)})
        del x, model, learner
        gc.collect()
    established = result.groupby("race_id", sort=False).horse_prev_races.transform("min").ge(5)
    handicap = result.groupby("race_id", sort=False).handicap.transform("min").eq(1)
    report = {"completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":registration["scope"], "comparison":score_pair(result), "fits":fits,
        "established_handicaps": score_pair(result.loc[established & handicap]),
        "actual_new_fits":sum(not item["reused"] for item in fits),
        "elapsed_seconds":time.monotonic()-started, "production_changed":False,
        "limitations":["Small capacity and two-year fitting window; this screen does not replace native full-model validation.",
            "No market comparator or production gain claim. No follow-up tuning on this window."]}
    result.to_parquet(OUTPUT/"predictions.parquet",index=False)
    (OUTPUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate",type=Path,default=ROOT/"feature_sets/handicap_compact_030.txt")
    parser.add_argument("--output",type=Path,default=OUTPUT,help="Separate receipt directory for each fixed candidate screen")
    parser.add_argument("--projection",type=Path,help="Audited extra features keyed by race, horse and date")
    arguments=parser.parse_args()
    OUTPUT=arguments.output.resolve()
    run(arguments.candidate, arguments.projection)
