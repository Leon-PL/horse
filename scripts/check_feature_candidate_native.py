"""One registered win-only native check against existing frozen forecasts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.canonical_training_cache import CheckpointPredictor
from scripts.integrate_canonical_repairs import native_component_semantics
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.model import get_feature_columns
from src.run_store import get_production_run_id
from src.training import TrainingConfig

KEY = ["race_id", "horse_name", "race_date"]


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def verify(plan, manifest):
    if fe_signature() != manifest["base_fe_signature"] or get_production_run_id() != manifest["parent_run"]:
        raise ValueError("Installed baseline changed during the native check")
    for name, digest in {**manifest["input_sha256"], **manifest["source_sha256"]}.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen native-check source/input changed: {name}")
    cfg = TrainingConfig(refresh_data=False, threads=2)
    if native_component_semantics(cfg) != manifest["learner_semantics"]:
        raise ValueError("Native learner settings/runtime changed")


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT / plan["output"]
    output.mkdir(parents=True, exist_ok=True)
    cfg = TrainingConfig(refresh_data=False, threads=2)
    semantics = native_component_semantics(cfg)
    candidate = [s.strip() for s in (ROOT / plan["feature_file"]).read_text().splitlines()
                 if s.strip() and not s.lstrip().startswith("#")]
    base = json.loads((ROOT / plan["baseline_manifest"]).read_text())["baseline_features"]
    extra = [c for c in candidate if c not in base]
    if candidate != base + extra or len(candidate) != len(set(candidate)):
        raise ValueError("Candidate may only append the fixed audited inputs")
    sources = list(dict.fromkeys([*semantics["sources"], *plan["source_paths"],
                                 "scripts/check_feature_candidate_native.py", "config.py"]))
    inputs = list(dict.fromkeys([str(plan_path.relative_to(ROOT)), plan["features"], plan["baseline_manifest"],
        plan["projection"], plan["feature_file"], plan["baseline_predictions"], plan["market_predictions"],
        "data/processed/betfair_bsp.parquet", *plan["evidence_inputs"]]))
    manifest = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One already-inspected Q1 development check; no new independent confirmation",
        "plan": plan, "features": candidate, "baseline_features": base,
        "base_fe_signature": fe_signature(), "parent_run": get_production_run_id(),
        "learner_semantics": semantics, "component_cache_scope": canonical_hash(semantics),
        "source_sha256": {p: sha(ROOT / p) for p in sources},
        "input_sha256": {p: sha(ROOT / p) for p in inputs}}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        manifest["registered_at_utc"] = existing["registered_at_utc"]
        if manifest != existing:
            raise ValueError("The registered native check has different definitions")
    else:
        write(manifest_path, manifest)
        with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for p in sources:
                archive.write(ROOT / p, p)
    verify(plan, manifest)
    reference = pd.read_parquet(ROOT / plan["baseline_predictions"])
    frame = pd.read_parquet(ROOT / plan["features"])
    projection = pd.read_parquet(ROOT / plan["projection"], columns=KEY + extra)
    frame = frame.merge(projection, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not frame.pop("_merge").eq("both").all():
        raise ValueError("Projection misses baseline runners")
    del projection
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    test = frame.loc[frame.race_date.between(*plan["test_dates"])].copy().reset_index(drop=True)
    labels = KEY + ["won", "finish_position", "num_runners", "handicap"]
    reference = reference.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(test[labels], reference[labels], check_dtype=False)
    start = frame.race_date.min()
    train = frame.loc[frame.race_date.le(plan["historical_cutoff"])].copy()
    del frame
    gc.collect()
    predictor = CheckpointPredictor(random_seed=cfg.seed, frameworks=cfg.frameworks,
        win_calibration_method=cfg.win_calibration_method, checkpoint_directory=output / "components",
        checkpoint_manifest=manifest["component_cache_scope"])
    predictor._autotune_njobs = 2
    started = time.monotonic()
    model_path = output / "win_only_model.joblib"
    if model_path.exists():
        predictor.load(str(model_path))
        if predictor.eval_split_info.get("feature_check_manifest") != sha(manifest_path):
            raise ValueError("Whole-model checkpoint belongs to another proposal")
    else:
        print(json.dumps({"stage": "one_native_win_candidate", "features": len(candidate),
                          "baseline_refit": False, "place_fit": False}), flush=True)
        with patch.object(config, "FEATURE_ALLOWLIST", candidate):
            if get_feature_columns(train) != candidate:
                raise ValueError("Native feature order differs from the preset candidate")
            predictor.fit_as_of(train, plan["historical_cutoff"], params=cfg.hyperparams,
                                dataset_start=start, win_only=True)
        predictor.eval_split_info["feature_check_manifest"] = sha(manifest_path)
        verify(plan, manifest)
        predictor.save(str(model_path))
    paired = test[labels + ["horse_prev_races"]].assign(
        baseline_probability=reference[plan["baseline_probability_column"]].to_numpy(),
        candidate_probability=align_predictions(predictor, test))
    established = paired.groupby("race_id", sort=False).horse_prev_races.transform("min").ge(5)
    handicap = paired.groupby("race_id", sort=False).handicap.transform("min").eq(1)
    overall = score_pair(paired)
    target = score_pair(paired.loc[established & handicap])
    primary = all(s["candidate"]["race_log_loss"] < s["baseline"]["race_log_loss"]
                  and s["candidate"]["brier"] <= s["baseline"]["brier"] for s in (overall, target))
    precise = overall["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] < 0
    # Stronger evidence is required before another expensive installed refit.
    advance = bool(primary and precise)
    check = test.race_id.isin(test.race_id.drop_duplicates().head(100))
    wiped = align_predictions(predictor, test.loc[check].assign(odds=np.nan, bsp=np.nan))
    price_error = float(np.max(np.abs(wiped - paired.loc[check.to_numpy(), "candidate_probability"].to_numpy())))
    if price_error > 1e-12:
        raise ValueError("Current prices affected the prior-history candidate")
    market = pd.read_parquet(ROOT / plan["market_predictions"], columns=KEY + ["market_probability"])
    matched = paired.merge(market, on=KEY, validate="one_to_one")
    market_report = score_forecasts(matched, {name: matched[name + "_probability"].to_numpy()
                                             for name in ("baseline", "candidate", "market")})
    verify(plan, manifest)
    paired.to_parquet(output / "predictions.parquet", index=False)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "overall": overall,
        "established_handicaps": target, "market": market_report,
        "primary_point_rule_passed": bool(primary), "overall_date_interval_excludes_zero_in_favour": bool(precise),
        "advance_to_canonical": advance,
        "decision": "qualify_for_installation" if advance else ("uncertain_not_selected" if primary else "not_selected"),
        "native_fit": predictor.eval_split_info, "component_cache_scope": manifest["component_cache_scope"],
        "current_price_independence_100_races_max_error": price_error,
        "elapsed_seconds": time.monotonic() - started, "production_changed": False,
        "limitations": ["Previously inspected Q1 development; uncertainty intervals do not correct for the wider research selection.",
                        "No current-data fit, place fit or baseline refit in this check; no future or profitability claim."]}
    write(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    run(args.plan.resolve())
