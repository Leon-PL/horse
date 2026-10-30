"""One registered comparison of two race-level recipes and frozen binary forecasts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.assess_forecast_parity import parity_evidence
from scripts.assess_independent_market import fresh_field_mask
from scripts.benchmark_forecasts import align_predictions, gap_segments, score_forecasts
from scripts.benchmark_independent_features import META, advancement, score_pair
from scripts.replay_market_snapshots import digest
from src.featured_dataset import fe_signature
from src.forecast_benchmark import match_market_fields
from src.forecast_evaluation import valid_race_mask
from src.race_probability_model import RaceProbabilityPredictor
from src.wf_eval import development_mask, expanding_period_folds

KINDS = ("conditional_logit", "race_softmax_gbm")
SOURCES = ["config.py", "src/model.py", "src/race_probability_model.py", "src/wf_eval.py",
           "src/probability_calibration.py", "src/forecast_evaluation.py", "src/forecast_benchmark.py",
           "src/utils.py", "scripts/benchmark_race_objective.py", "scripts/benchmark_independent_features.py",
           "scripts/benchmark_forecasts.py", "scripts/assess_forecast_parity.py",
           "scripts/assess_independent_market.py", "scripts/replay_market_snapshots.py"]


def candidate_advances(pooled, folds):
    # Two directional comparisons: central 95% intervals give 97.5% one-sided
    # upper bounds, conservatively controlling the two-candidate family at 5%.
    return bool(advancement(pooled, folds)
                and pooled["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] < 0)


def market_context(paired, snapshots, specification, seed):
    matched, coverage = match_market_fields(paired, snapshots)
    def describe(frame):
        if frame.empty:
            return {"races": 0, "runners": 0}
        forecasts = {name: frame[f"{name}_probability"].to_numpy(float) for name in ("baseline", *KINDS, "market")}
        return {**score_forecasts(frame, forecasts, seed=seed),
                "parity_evidence": parity_evidence(frame, forecasts, seed=seed),
                "segments": gap_segments(frame, forecasts)}
    mask = fresh_field_mask(matched, specification)
    return matched, {"coverage": coverage, "pooled": describe(matched), "freshness": describe(matched.loc[mask])}


def run(args):
    registration_path = Path(args.registration)
    registration = json.loads(registration_path.read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # The frozen 002 projection and forecasts already encode the exact fields,
    # warm-up, fitted binary recipe and earlier guard-free source provenance.
    previous = Path(registration["baseline_experiment"])
    prior_manifest = json.loads((previous / "manifest.json").read_text())
    prior_report = json.loads((previous / "report.json").read_text())
    if prior_report["candidate_advances"] or len(prior_report["folds"]) != 3:
        raise ValueError("Expected completed rejected ATR family and retained baseline")
    if registration["features"] != prior_manifest["variants"]["baseline"]:
        raise ValueError("Baseline feature recipe differs")
    store_meta = json.loads((ROOT / "data/processed/feature_store/meta.json").read_text())
    if (store_meta["fe_signature"] != fe_signature()
            or store_meta["fe_signature"] != registration["fe_signature"]
            or store_meta["data_signature"] != registration["data_signature"]):
        raise ValueError("Feature/store provenance changed")
    inputs = {path.as_posix(): digest(path) for path in [previous / "inputs.parquet", previous / "paired_predictions.parquet",
              previous / "manifest.json", previous / "report.json", Path(registration["snapshots"]),
              Path(registration["snapshots"]).with_suffix(".json")]
              + [previous / f"{period}_baseline_fit.json" for period in registration["periods"]]
              + [previous / f"{period}_baseline.joblib" for period in registration["periods"]]}
    if inputs != registration["input_sha256"]:
        raise ValueError("A registered input artifact changed")
    sources = {name: digest(ROOT / name) for name in SOURCES}
    manifest = {"experiment": "003 bounded independent race-objective comparison", "registration": registration,
                "registration_sha256": digest(registration_path), "code_sha256": sources,
                "source_store_meta": store_meta, "input_sha256": inputs,
                "baseline_scope": "Exact cached independent probabilities from completed 002; selected features unchanged by later guard",
                "selection": "Both fixed candidates, all three development quarters; no partial-score selection; no promotion"}
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise ValueError("Manifest changed; use a separately registered experiment")
    manifest_path.write_bytes(encoded)
    if not (output / "source_snapshot.zip").exists():
        with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCES:
                archive.write(ROOT / name, name)
    names = registration["features"]
    columns = list(dict.fromkeys(names + META))
    frame = pd.read_parquet(previous / "inputs.parquet", columns=columns)
    frame.race_date = pd.to_datetime(frame.race_date).dt.normalize()
    start = frame.race_date.min()
    frame = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    folds = {fold.label: fold for fold in expanding_period_folds(frame.race_date)}
    cached = pd.read_parquet(previous / "paired_predictions.parquet")
    original_allowlist = config.FEATURE_ALLOWLIST
    started = time.monotonic()
    try:
        config.FEATURE_ALLOWLIST = names
        for kind in KINDS:  # Complete the simple linear reference before trees.
            for period in registration["periods"]:
                fold = folds[period]
                cutoff = frame.loc[fold.train, "race_date"].max()
                path = output / f"{period}_{kind}.joblib"
                if path.exists():
                    predictor = RaceProbabilityPredictor.load(path)
                else:
                    print(json.dumps({"stage": "fitting", "period": period, "kind": kind,
                                      "cutoff": str(cutoff.date()), "features": len(names)}), flush=True)
                    predictor = RaceProbabilityPredictor(kind, names, params=registration["tree_params"],
                        threads=registration["threads"], seed=registration["seed"], linear_l2=registration["linear_l2"])
                    predictor.fit_as_of(frame.loc[fold.train].copy(), cutoff, dataset_start=start)
                    if {name: digest(ROOT / name) for name in SOURCES} != sources:
                        raise RuntimeError("Source changed during native fitting")
                    predictor.eval_split_info.update(experiment_manifest_sha256=manifest_hash, period=period)
                    old_info = json.loads((previous / f"{period}_baseline_fit.json").read_text())
                    for key in ("training_runners", "training_races", "training_first_date", "training_last_date"):
                        if predictor.eval_split_info[key] != old_info[key]:
                            raise ValueError(f"Historical training population differs: {key}")
                    predictor.save(path)
                if (predictor.eval_split_info.get("experiment_manifest_sha256") != manifest_hash
                        or predictor.feature_cols != names):
                    raise ValueError("Cached predictor belongs to another experiment")
                test = frame.loc[fold.test].copy()
                predictions = test[[name for name in META if name in test]].copy()
                predictions["probability"] = align_predictions(predictor, test)
                predictions.to_parquet(output / f"{period}_{kind}_predictions.parquet", index=False)
                (output / f"{period}_{kind}_fit.json").write_text(json.dumps(predictor.eval_split_info, indent=2))
                print(json.dumps({"stage": "fit_complete", "period": period, "kind": kind}), flush=True)
                del predictor, test, predictions
                gc.collect()
    finally:
        config.FEATURE_ALLOWLIST = original_allowlist
    # No full-data performance is inspected until both recipes finish all folds.
    paired = cached[[name for name in META if name in cached] + ["baseline_probability"]].copy()
    for kind in KINDS:
        current = pd.concat([pd.read_parquet(output / f"{period}_{kind}_predictions.parquet")
                             for period in registration["periods"]], ignore_index=True)
        paired = paired.merge(current[["race_id", "horse_name", "probability"]].rename(
            columns={"probability": f"{kind}_probability"}), on=["race_id", "horse_name"],
            how="outer", validate="one_to_one")
    if paired.isna()[[f"{kind}_probability" for kind in KINDS] + ["baseline_probability"]].any().any():
        raise ValueError("Candidate and baseline prediction fields differ")
    paired = paired.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    result = {"manifest_sha256": manifest_hash, "candidates": {}, "production_changed": False,
              "scope": "Previously inspected retrospective development dates, not fresh forward confirmation"}
    for kind in KINDS:
        current = paired.rename(columns={f"{kind}_probability": "candidate_probability"})
        by_period = pd.to_datetime(current.race_date).dt.to_period("Q").astype(str)
        fold_scores = [{"period": period, **score_pair(current.loc[by_period == period], seed=registration["seed"])}
                       for period in registration["periods"]]
        pooled = score_pair(current, seed=registration["seed"])
        result["candidates"][kind] = {"pooled": pooled, "folds": fold_scores,
                                       "advances": candidate_advances(pooled, fold_scores)}
    paired.to_parquet(output / "paired_predictions.parquet", index=False)
    snapshots = pd.read_parquet(registration["snapshots"])
    matched, context = market_context(paired, snapshots, registration["freshness"], registration["seed"])
    matched.to_parquet(output / "market_predictions.parquet", index=False)
    (output / "market_context.json").write_text(json.dumps(context, indent=2))
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="data/audit/market_performance/003_registration.json")
    parser.add_argument("--output-dir", default="data/audit/market_performance/003_race_objective")
    logging.basicConfig(level=logging.INFO)
    run(parser.parse_args())
