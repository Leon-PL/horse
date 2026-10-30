"""Describe frozen independent feature forecasts against their same-time market.

Requires a finished experiment and an earlier market-context registration.
Does not fit corrections, select features or change the advancement decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.assess_forecast_parity import parity_evidence
from scripts.benchmark_forecasts import gap_segments, score_forecasts
from scripts.benchmark_independent_features import score_pair
from scripts.replay_market_snapshots import digest
from src.forecast_benchmark import match_market_fields
from src.forecast_evaluation import probability_metrics


def verify_complete_experiment(frame, primary, specification):
    periods = specification["periods"]
    if [fold["period"] for fold in primary["folds"]] != periods or len(periods) < 3:
        raise ValueError("Complete all registered independent periods before market scoring")
    actual = pd.to_datetime(frame.race_date).dt.to_period("Q").astype(str)
    if set(actual) != set(periods):
        raise ValueError("Prediction dates differ from the registered periods")
    sizes = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id, np.repeat(frame.race_id.drop_duplicates(), sizes)):
        raise ValueError("Expected contiguous complete prediction fields")
    if len(frame) != primary["pooled"]["runners"] or len(sizes) != primary["pooled"]["races"]:
        raise ValueError("Prediction coverage differs from the completed experiment")
    for name in ("baseline", "candidate"):
        actual_metrics = probability_metrics(frame[f"{name}_probability"], frame.won, sizes)[0]
        expected = primary["pooled"][name]
        if any(not np.isclose(actual_metrics[key], expected[key], rtol=0, atol=1e-12)
               for key in actual_metrics):
            raise ValueError("Frozen predictions do not reproduce the primary report")


def fresh_field_mask(frame, specification):
    runner_age = (frame.decision_ms - frame.price_changed_ms) / 1000
    market_age = (frame.decision_ms - frame.observed_ms) / 1000
    eligible = (runner_age.between(0, specification["every_runner_update_max_age_seconds"])
                & market_age.between(0, specification["market_observation_max_age_seconds"]))
    return eligible.groupby(frame.race_id).transform("all").to_numpy()


def verify_snapshot_manifest(manifest, specification, snapshot_sha256):
    if manifest["output_sha256"] != snapshot_sha256:
        raise ValueError("Market snapshot fingerprint differs from its replay manifest")
    if manifest["seconds_before"] != specification["horizon_seconds"]:
        raise ValueError("Market replay uses a different decision horizon")


def describe(frame, seed):
    if frame.empty:
        return {"races": 0, "runners": 0, "status": "No complete eligible fields"}
    forecasts = {name: frame[f"{name}_probability"].to_numpy(float)
                 for name in ("baseline", "candidate", "market")}
    return {**score_forecasts(frame, forecasts, seed=seed),
            "parity_evidence": parity_evidence(frame, forecasts, seed=seed),
            "candidate_vs_baseline": score_pair(frame, seed=seed),
            "segments": gap_segments(frame, forecasts)}


def run(args):
    directory = Path(args.experiment)
    registration_path = directory / "market_context_registration.json"
    prediction_path = directory / "paired_predictions.parquet"
    primary_path = directory / "report.json"
    manifest_path = directory / "manifest.json"
    snapshots_path = Path(args.snapshots)
    # Fail before scoring an incomplete run; this report cannot pick an early winner.
    if not prediction_path.exists() or not primary_path.exists():
        raise ValueError("The independent comparison must finish before this assessment")
    if registration_path.stat().st_mtime_ns >= prediction_path.stat().st_mtime_ns:
        raise ValueError("Register market context before independent predictions are scored")
    paths = [registration_path, prediction_path, primary_path, manifest_path, snapshots_path,
             snapshots_path.with_suffix(".json")]
    before = {str(path): digest(path) for path in paths}
    specification = json.loads(registration_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    snapshot_manifest = json.loads(snapshots_path.with_suffix(".json").read_text(encoding="utf-8"))
    verify_snapshot_manifest(snapshot_manifest, specification, before[str(snapshots_path)])
    # The fitter hashes its encoded manifest text before writing. Windows
    # translates newlines on disk; retain the byte fingerprint separately.
    manifest_text_sha256 = hashlib.sha256(manifest_path.read_text(encoding="utf-8").encode()).hexdigest()
    if primary["manifest_sha256"] != manifest_text_sha256:
        raise ValueError("The primary report belongs to a different manifest")
    frame = pd.read_parquet(prediction_path).sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    verify_complete_experiment(frame, primary, specification)
    snapshots = pd.read_parquet(snapshots_path)
    dates = pd.to_datetime(snapshots.race_date).dt.normalize()
    snapshots = snapshots.loc[dates.between(specification["snapshot_start"], specification["snapshot_end"])].copy()
    paired, coverage = match_market_fields(frame, snapshots)
    report = {
        "scope": specification["scope"], "specification": specification,
        "seed": args.seed,
        "bootstrap_draws": {"two_sided_paired_intervals": 1000, "one_sided_parity_bounds": 2000},
        "selection": "Descriptive secondary context only; primary advancement decision unchanged",
        "primary_candidate_advances": primary["candidate_advances"], "coverage": coverage,
        "input_sha256": before, "market_replay": snapshot_manifest,
        "primary_manifest_text_sha256": manifest_text_sha256,
        "pooled": describe(paired, args.seed),
        "folds": {}, "freshness": {},
    }
    periods = pd.to_datetime(paired.race_date).dt.to_period("Q").astype(str)
    for period in specification["periods"]:
        report["folds"][period] = describe(paired.loc[periods == period], args.seed)
    mask = fresh_field_mask(paired, specification["freshness_diagnostic"])
    report["freshness"] = describe(paired.loc[mask], args.seed)
    report["freshness"]["excluded_races"] = int(paired.loc[~mask, "race_id"].nunique())
    sources = ["scripts/assess_independent_market.py", "scripts/assess_forecast_parity.py",
               "scripts/benchmark_independent_features.py", "scripts/benchmark_forecasts.py",
               "src/forecast_benchmark.py", "src/forecast_evaluation.py", "src/wf_eval.py"]
    report["code_sha256"] = {path: digest(ROOT / path) for path in sources}
    if {str(path): digest(path) for path in paths} != before:
        raise RuntimeError("Frozen assessment input changed while scoring")
    paired.to_parquet(directory / "market_context_predictions.parquet", index=False)
    excluded = frame.loc[~frame.race_id.isin(paired.race_id)]
    excluded.to_parquet(directory / "market_context_excluded_fields.parquet", index=False)
    (directory / "market_context_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {"coverage": coverage, "metrics": report["pooled"].get("metrics"),
               "parity_evidence": report["pooled"].get("parity_evidence"),
               "fresh_races": report["freshness"]["races"]}
    print(json.dumps(summary, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
