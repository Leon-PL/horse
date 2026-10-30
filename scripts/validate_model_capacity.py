"""022: wider development validation of the one shortlisted capacity change.

Reuse saved016 production-recipe baselines, verify their native forecasts,
and double only the win tree cap2000->4000 at the original.008 learning rate.
Complete the two declared periods before judging the candidate. No feature,
calibration, rate search, reserved holdout or automatic promotion is included.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
import config
from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.screen_feature_batch import OUTPUT as SCREEN, REFERENCE, sha, runtime_versions, verify as verify_screen
from scripts.screen_model_capacity import CurvePredictor, OUTPUT as CAPACITY, verify as verify_capacity
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.wf_eval import development_mask

OUTPUT = ROOT / "data/audit/model_improvement/022_full_capacity"
PERIODS = ("2025Q4", "2026Q1")


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))
    temporary.replace(path)


def preflight_reference(original):
    frame = pd.read_parquet(REFERENCE / "inputs.parquet")
    frame = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    checked = {}
    keys = ["race_id", "horse_name", "race_date", "won"]
    burn = pd.Timestamp(original["dataset_start"])+pd.DateOffset(months=config.BURN_IN_MONTHS)
    for fold in original["folds"]:
        if fold["period"] not in PERIODS:
            continue
        period = fold["period"]
        test = frame.loc[frame.race_date.between(fold["test_start"], fold["test_end"])]
        frozen = pd.read_parquet(REFERENCE / f"{period}_predictions.parquet")
        if not test[keys].reset_index(drop=True).equals(frozen[keys]):
            raise ValueError("Preflight: original test field identities or labels differ")
        baseline = RacePredictor(); baseline.load(str(REFERENCE / f"{period}_candidate.joblib"))
        np.testing.assert_allclose(align_predictions(baseline, test), frozen.candidate_probability, rtol=0, atol=1e-12)
        train = frame.loc[frame.race_date.ge(burn) & frame.race_date.le(fold["cutoff"])]
        fit = json.loads((REFERENCE / f"{period}_fit.json").read_text())
        if len(train) != fit["training_runners"] or train.race_id.nunique() != fit["training_races"]:
            raise ValueError("Preflight: original post-burn-in fitting population differs")
        checked[period] = {"test_races": int(test.race_id.nunique()), "test_runners": len(test),
            "training_runners": len(train), "training_races": int(train.race_id.nunique()),
            "saved_baseline_native_replay": "within1e-12"}
    return checked


def prepare():
    screened = json.loads((CAPACITY / "report.json").read_text())
    capacity_manifest = json.loads((CAPACITY / "manifest.json").read_text())
    verify_capacity(capacity_manifest)
    scored = screened["comparison"]
    delta = scored["paired_delta_candidate_minus_baseline"]
    if (not screened["point_improves_both"] or delta["race_log_loss"]["ci95"][1] >= 0
            or delta["date_cluster_race_log_loss"]["ci95"][1] >= 0):
        raise ValueError("Do not widen validation without a clear matched screening gain")
    original = json.loads((REFERENCE / "manifest.json").read_text())
    complete = json.loads((REFERENCE / "completion.json").read_text())
    if original["params"]["classifier"]["n_estimators"] != 2000 or original["params"]["classifier"]["learning_rate"] != .008:
        raise ValueError("Expected the original production recipe")
    files = ["manifest.json", "completion.json", "inputs.parquet", "paired_predictions.parquet", "market_context_predictions.parquet"]
    files += [f"{period}_{kind}" for period in PERIODS for kind in ("candidate.joblib", "fit.json", "predictions.parquet")]
    for name in files:
        if name in complete["evidence_sha256"] and sha(REFERENCE / name) != complete["evidence_sha256"][name]:
            raise ValueError("Original completed evidence changed")
    if sha(REFERENCE / "inputs.parquet") != original["inputs_sha256"]:
        raise ValueError("Full original input projection changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise ValueError("Already registered; resume without --prepare")
    preflight = preflight_reference(original)
    write(OUTPUT / "reference_preflight.json", preflight)
    params = json.loads(json.dumps(original["params"]))
    params["classifier"]["n_estimators"] = 4000
    value = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "features": original["features"],
        "params": params, "dataset_start": original["dataset_start"], "seed": 42, "threads": 2,
        "periods": list(PERIODS), "folds": [f for f in original["folds"] if f["period"] in PERIODS],
        "source_sha256": {name: sha(ROOT / name) for name in ("scripts/validate_model_capacity.py", "scripts/screen_model_capacity.py")},
        "runtime_versions": runtime_versions(), "reference_sha256": {name: sha(REFERENCE / name) for name in files},
        "reference_preflight_sha256": sha(OUTPUT / "reference_preflight.json"),
        "screen_evidence_sha256": {name: sha(CAPACITY / name) for name in ("manifest.json", "report.json", "learning_curves.json")},
        "teacher_inputs_sha256": original["market_teacher_inputs_sha256"],
        "advancement": "Complete both periods; each race-log-loss point estimate improves, pooled paired race/date log-loss upper95<0, pooled Brier and binary loss no worse; report market gap and rare winner errors; no automatic promotion",
        "scope": "Wider retrospective development validation, two already inspected periods; preserve full historical fitting rows/native calibration, reuse exact saved016 baselines; no reserved holdout or other candidate sweep"}
    write(OUTPUT / "manifest.json", value)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in value["source_sha256"]:
            archive.write(ROOT / name, name)
    print(json.dumps({"stage": "registered", "periods": PERIODS, "candidate_fits": 2,
                      "baseline_fits": 0, "manifest_sha256": canonical_hash(value)}), flush=True)


def verify(manifest):
    verify_screen(json.loads((SCREEN / "manifest.json").read_text()))
    verify_capacity(json.loads((CAPACITY / "manifest.json").read_text()))
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("Runtime changed")
    if sha(OUTPUT / "reference_preflight.json") != manifest["reference_preflight_sha256"]:
        raise ValueError("Original baseline preflight changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Registered validation procedure changed")
    for name, expected in manifest["reference_sha256"].items():
        if sha(REFERENCE / name) != expected:
            raise ValueError("Full-budget baseline evidence changed")
    for name, expected in manifest["screen_evidence_sha256"].items():
        if sha(CAPACITY / name) != expected:
            raise ValueError("Shortlisting evidence changed")
    for name, expected in manifest["teacher_inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Market-teacher data changed")


def fit_period(frame, manifest, fold):
    period = fold["period"]
    test = frame.loc[frame.race_date.between(fold["test_start"], fold["test_end"])].copy()
    frozen = pd.read_parquet(REFERENCE / f"{period}_predictions.parquet")
    keys = ["race_id", "horse_name", "race_date", "won"]
    if not test[keys].reset_index(drop=True).equals(frozen[keys]):
        raise ValueError("Test population differs from the original comparison")
    baseline = RacePredictor(); baseline.load(str(REFERENCE / f"{period}_candidate.joblib"))
    if baseline.feature_cols != manifest["features"] or baseline.eval_split_info["as_of"] != fold["cutoff"]:
        raise ValueError("Baseline model recipe differs")
    original = align_predictions(baseline, test)
    np.testing.assert_allclose(original, frozen.candidate_probability, rtol=0, atol=1e-12)
    replay_error = float(np.max(np.abs(original-frozen.candidate_probability)))
    del baseline
    predictor = CurvePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 2
    path = OUTPUT / f"{period}_candidate.joblib"
    started = time.perf_counter()
    if path.exists():
        predictor.load(str(path))
        if predictor.eval_split_info.get("full_capacity_manifest_sha256") != canonical_hash(manifest):
            raise ValueError("Saved candidate belongs to another validation")
    else:
        train = frame.loc[frame.race_date.le(fold["cutoff"])].copy()
        if get_feature_columns(train) != manifest["features"]:
            raise ValueError("Feature order differs from the saved baseline")
        print(json.dumps({"stage": "fitting", "period": period, "input_rows_before_native_burn_in": len(train),
                          "tree_cap": 4000, "learning_rate": .008, "baseline_replay_max_error": replay_error}), flush=True)
        predictor.fit_as_of(train, pd.Timestamp(fold["cutoff"]), params=manifest["params"],
                            dataset_start=pd.Timestamp(manifest["dataset_start"]), win_only=True)
        baseline_fit = json.loads((REFERENCE / f"{period}_fit.json").read_text())
        for key in ("training_runners", "training_races", "training_first_date", "training_last_date"):
            if predictor.eval_split_info[key] != baseline_fit[key]:
                raise ValueError(f"Full-budget fitting population differs: {key}")
        predictor.eval_split_info.update(full_capacity_manifest_sha256=canonical_hash(manifest), period=period)
        write(OUTPUT / f"{period}_learning_curves.json", predictor.learning_curves)
        verify(manifest)
        temporary = path.with_suffix(".incomplete.joblib")
        predictor.save(str(temporary)); temporary.replace(path)
        del train
    result = frozen.drop(columns=[c for c in ("baseline_probability", "candidate_probability", "place_probability") if c in frozen])
    result["baseline_probability"] = original
    result["candidate_probability"] = align_predictions(predictor, test)
    native = RacePredictor(); native.load(str(path))
    np.testing.assert_array_equal(result.candidate_probability, align_predictions(native, test))
    result.to_parquet(OUTPUT / f"{period}_predictions.parquet", index=False)
    report = {"period": period, "elapsed_seconds": time.perf_counter()-started,
        "baseline_native_replay_max_error": replay_error, "candidate_saved_replay_max_error": 0.,
        "model_sha256": sha(path), "fit": predictor.eval_split_info, **score_pair(result)}
    write(OUTPUT / f"{period}_report.json", report)
    print(json.dumps({"stage": "period_complete", **report}), flush=True)
    del test, frozen, predictor, native, result
    gc.collect()


def summarise(manifest):
    reports = [json.loads((OUTPUT / f"{period}_report.json").read_text()) for period in manifest["periods"]]
    pooled = pd.concat([pd.read_parquet(OUTPUT / f"{period}_predictions.parquet") for period in manifest["periods"]], ignore_index=True)
    comparison = score_pair(pooled)
    delta = comparison["paired_delta_candidate_minus_baseline"]
    advances = bool(all(r["candidate"]["race_log_loss"] < r["baseline"]["race_log_loss"] for r in reports)
        and delta["race_log_loss"]["ci95"][1] < 0 and delta["date_cluster_race_log_loss"]["ci95"][1] < 0
        and comparison["candidate"]["brier"] <= comparison["baseline"]["brier"]
        and comparison["candidate"]["binary_log_loss"] <= comparison["baseline"]["binary_log_loss"])
    keys = ["race_id", "horse_name", "race_date", "won"]
    market = pd.read_parquet(REFERENCE / "market_context_predictions.parquet")
    eligible_ids = pooled.race_id.unique()
    market = market.loc[market.race_id.isin(eligible_ids)].copy()
    paired = market.drop(columns=["baseline_probability", "candidate_probability"]).merge(
        pooled[keys + ["baseline_probability", "candidate_probability"]], on=keys, how="left", sort=False, validate="one_to_one")
    if paired[["baseline_probability", "candidate_probability"]].isna().any().any() or not valid_race_mask(paired).all():
        raise ValueError("Market comparison requires unchanged complete original fields")
    market_scores = score_forecasts(paired, {"market": paired.market_probability.to_numpy(),
        "baseline": paired.baseline_probability.to_numpy(), "candidate": paired.candidate_probability.to_numpy()})
    tails = {name: {str(threshold): int((pooled.won.eq(1) & pooled[f"{name}_probability"].le(threshold)).sum())
                    for threshold in (1e-8, .001, .01)} for name in ("baseline", "candidate")}
    current, proposed, benchmark = (market_scores["metrics"][n]["race_log_loss"] for n in ("baseline", "candidate", "market"))
    report = {"manifest_sha256": canonical_hash(manifest), "periods": reports, "pooled": comparison,
        "passes_registered_advancement": advances, "market": market_scores, "winner_tail_errors": tails,
        "fraction_of_remaining_market_gap_closed_point_estimate": (current-proposed)/(current-benchmark) if current > benchmark else None,
        "scope": manifest["scope"], "production_changed": False, "additional_quarters_queued": False}
    verify(manifest)
    write(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run():
    manifest = json.loads((OUTPUT / "manifest.json").read_text()); verify(manifest)
    config.FEATURE_ALLOWLIST = manifest["features"]
    config.WIN_SOFT_LABEL_DISTILL, config.WIN_SOFT_LABEL_BLEND = True, .5
    frame = pd.read_parquet(REFERENCE / "inputs.parquet")
    frame = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    for fold in manifest["folds"]:
        fit_period(frame, manifest, fold)
    summarise(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--start", action="store_true", help="Register if needed, then complete both periods in this process")
    arguments = parser.parse_args()
    if arguments.start:
        if not (OUTPUT / "manifest.json").exists():
            prepare()
        run()
    elif arguments.prepare:
        prepare()
    else:
        run()
