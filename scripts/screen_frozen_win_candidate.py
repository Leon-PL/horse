"""One small candidate against an exactly cached frozen win learner.

The original model source is loaded from its sealed archive. No production
module is replaced. Matching every numeric baseline training argument, runtime
and classifier setting is mandatory before reusing that baseline. Saved test
forecasts must also replay exactly. Place-layer or live-speed edits therefore
need not force an unchanged win baseline to fit again.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import config

from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.canonical_training_cache import cached_component, fit_key
from src.deployment_validation import file_sha256 as sha
from src.forecast_evaluation import valid_race_mask

REFERENCE = ROOT / "data/audit/market_transfer_034"
KEY = ["race_id", "horse_name", "race_date"]


def run(case):
    case = case.resolve()
    output = case / "quick_screen"
    output.mkdir(exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Preserve completed screens")
    preparation = json.loads((case / "preparation.json").read_text())
    for name, digest in preparation["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Audited source changed: {name}")
    for name, digest in preparation["output_sha256"].items():
        if sha(case / name) != digest:
            raise ValueError(f"Audited inputs changed: {name}")
    original = json.loads((REFERENCE / "quick_screen/manifest.json").read_text())
    semantics = original["semantics"]
    baseline_manifest = json.loads((REFERENCE / "baseline/manifest.json").read_text())
    columns = baseline_manifest["baseline_features"]
    candidate = [line.strip() for line in (case / "features.txt").read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    extra = candidate[len(columns):]
    if candidate != columns + extra or len(candidate) != len(set(candidate)) or not extra:
        raise ValueError("Only append the fixed proposal to the frozen baseline")
    runtime = {name: importlib.metadata.version(name) for name in semantics["runtime"]}
    if runtime != semantics["runtime"] or config.CLASSIFIER_PARAMS != semantics["native_classifier_params"]:
        raise ValueError("Frozen learner runtime/classifier settings differ")
    for path, digest in [(REFERENCE / "baseline/inputs.parquet", baseline_manifest["inputs_sha256"]),
                         (ROOT / "data/processed/betfair_bsp.parquet", baseline_manifest["teacher_sha256"])]:
        if sha(path) != digest:
            raise ValueError("Frozen feature/teacher inputs changed")
    archive_path = REFERENCE / "native_check/source_snapshot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        raw_source = archive.read("src/model.py")
    import hashlib
    if hashlib.sha256(raw_source).hexdigest() != semantics["sources"]["src/model.py"]:
        raise ValueError("Original win learner was not preserved")
    name = "src._frozen_screen_win_034"
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / "src/model.py")
    sys.modules[name] = module
    exec(compile(raw_source, module.__file__, "exec"), module.__dict__)
    started = time.monotonic()
    frame = pd.read_parquet(REFERENCE / "baseline/inputs.parquet", filters=[
        ("race_date", ">=", pd.Timestamp("2023-09-24")), ("race_date", "<=", pd.Timestamp("2025-12-31"))])
    projection = pd.read_parquet(case / "projection.parquet", columns=KEY + extra)
    frame = frame.merge(projection, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not frame.pop("_merge").eq("both").all() or set(extra) & set(columns):
        raise ValueError("Projection misses or replaces baseline inputs")
    del projection
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    train = frame.loc[frame.race_date.le("2025-09-23")].copy()
    test = frame.loc[frame.race_date.between("2025-10-01", "2025-12-31")].copy()
    del frame
    with patch.object(config, "FEATURE_ALLOWLIST", columns):
        data = module.prepare_multi_target_data(train, training_only=True, dataset_start="2020-09-15")
    del train
    gc.collect()
    if data["feature_cols"] != columns:
        raise ValueError("Baseline feature order differs")
    base_x = data["X_train"]
    y, weights = data["y_train_won"], data["sample_weight_train"]
    teacher = data["win_teacher_train"]
    soft_y = (np.where(teacher > 0, float(config.WIN_SOFT_LABEL_BLEND) * teacher +
                      (1 - float(config.WIN_SOFT_LABEL_BLEND)) * y, y)
              if config.WIN_SOFT_LABEL_DISTILL else None)
    kwargs = {"params": semantics["overrides"], "sample_weight": weights, "soft_y": soft_y,
              "scale_pos_weight": (len(y) - y.sum()) / max(y.sum(), 1)}
    scope = canonical_hash(semantics)
    baseline_key = fit_key(scope, "win", columns, (base_x, y), kwargs)
    saved_report = json.loads((REFERENCE / "quick_screen/report.json").read_text())
    expected_key = saved_report["fits"][0]["cache_key"]
    cache = ROOT / "data/research/quick_win_components"
    if baseline_key != expected_key or not (cache / (baseline_key + ".joblib")).exists():
        raise ValueError("Every numeric baseline training argument must match its exact cached fit")
    base_model = joblib.load(cache / (baseline_key + ".joblib"))
    if base_model["fit_key"] != baseline_key:
        raise ValueError("Cached baseline identity differs")
    test_base = data["scaler"].transform(test[columns].to_numpy())
    raw = base_model["model"].predict_proba(test_base)[:, 1]
    result = test[KEY + ["won", "finish_position", "num_runners", "handicap", "horse_prev_races"]].copy()
    result["baseline_probability"] = raw / pd.Series(raw, index=test.index).groupby(test.race_id, sort=False).transform("sum").to_numpy()
    saved = pd.read_parquet(REFERENCE / "quick_screen/predictions.parquet")
    replay = result[KEY + ["baseline_probability"]].merge(saved[KEY + ["baseline_probability"]], on=KEY, suffixes=("_now", "_saved"), validate="one_to_one")
    if len(replay) != len(result):
        raise ValueError("Saved baseline forecast population differs")
    np.testing.assert_array_equal(replay.baseline_probability_now, replay.baseline_probability_saved)
    del base_model
    x = np.column_stack((base_x, data["scaler"].transform(data["train_df"][extra].to_numpy())))
    test_x = np.column_stack((test_base, data["scaler"].transform(test[extra].to_numpy())))
    candidate_key = fit_key(scope, "win", candidate, (x, y), kwargs)
    registration = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_cache_key": baseline_key, "all_baseline_numeric_arguments_exact": True,
        "baseline_forecasts_replay_exact": True, "frozen_model_source_sha256": semantics["sources"]["src/model.py"],
        "frozen_source_archive_sha256": sha(archive_path), "frozen_semantics": semantics,
        "script_sha256": sha(Path(__file__)), "preparation_sha256": sha(case / "preparation.json"),
        "candidate_key": candidate_key, "candidate_features": candidate,
        "scope": "One small fixed win candidate, no baseline/place/calibration fit; inspected Q4 development only"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        registration["registered_at_utc"] = previous["registered_at_utc"]
        if registration != previous:
            raise ValueError("Registered screen changed")
    else:
        manifest_path.write_text(json.dumps(registration, indent=2), encoding="utf-8")
    learner = module.RacePredictor(random_seed=42, frameworks=semantics["frameworks"])
    learner.feature_cols = candidate
    learner._autotune_njobs = 2
    reused = (cache / (candidate_key + ".joblib")).exists()
    print(json.dumps({"stage": "one_small_candidate", "baseline_cache_reused": True,
                      "baseline_replay_max_error": 0., "features": len(candidate)}), flush=True)
    model = cached_component(cache, candidate_key, learner._train_win_classifier, (x, y), kwargs)
    raw = model.predict_proba(test_x)[:, 1]
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid candidate probabilities")
    total = pd.Series(raw, index=test.index).groupby(test.race_id, sort=False).transform("sum").to_numpy()
    if (total <= 0).any():
        raise ValueError("Missing candidate field mass")
    result["candidate_probability"] = raw / total
    established = result.groupby("race_id").horse_prev_races.transform("min").ge(5)
    handicap = result.groupby("race_id").handicap.transform("min").eq(1)
    overall = score_pair(result)
    target = score_pair(result.loc[established & handicap])
    point = all(s["candidate"]["race_log_loss"] < s["baseline"]["race_log_loss"] and
                s["candidate"]["brier"] <= s["baseline"]["brier"] for s in (overall, target))
    precise = overall["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] < 0
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "comparison": overall,
        "established_handicaps": target, "baseline_reused": True, "baseline_forecast_replay_max_error": 0.,
        "actual_new_fits": int(not reused), "candidate_cache_key": candidate_key,
        "point_rule_passed": bool(point), "overall_date_interval_excludes_zero_in_favour": bool(precise),
        "advance_to_native": bool(point and precise), "elapsed_seconds": time.monotonic() - started,
        "canonical_writes": False, "scope": registration["scope"],
        "limitations": ["400 trees/two-year fit is triage, not native qualification or a production score.",
                        "Original frozen win training source reused; other production/place/speed changes are outside this experiment."]}
    result.to_parquet(output / "predictions.parquet", index=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    run(parser.parse_args().case)
