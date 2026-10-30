"""Registered monthly versus quarterly refitting, with the forecast recipe fixed."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.assess_independent_market import run as assess_market
from scripts.benchmark_forecasts import align_predictions
from scripts.benchmark_independent_features import META, advancement, score_pair
from scripts.replay_market_snapshots import digest
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask, validate_probabilities
from src.model import RacePredictor
from src.wf_eval import development_mask, purge_days

SOURCES = ["config.py", "src/model.py", "src/wf_eval.py", "src/utils.py", "src/probability_calibration.py",
           "src/forecast_evaluation.py", "src/forecast_benchmark.py", "scripts/benchmark_refit_cadence.py",
           "scripts/benchmark_forecasts.py", "scripts/benchmark_independent_features.py",
           "scripts/assess_independent_market.py", "scripts/assess_forecast_parity.py",
           "scripts/replay_market_snapshots.py"]


def month_schedule(periods, embargo_days):
    """Anchor refits to calendar starts, even if the first day has no races."""
    quarters = [pd.Period(period, freq="Q") for period in periods]
    if len(quarters) != len(set(quarters)) or quarters != sorted(quarters) or embargo_days < 1:
        raise ValueError("Ordered unique quarters and a positive embargo required")
    return [{"period": str(quarter), "month": str(month),
             "decision_start": str(month.start_time.date()),
             "as_of": str((month.start_time.normalize() - pd.Timedelta(days=embargo_days)).date()),
             "reuse_quarterly_model": index == 0}
            for quarter in quarters
            for index, month in enumerate(pd.period_range(quarter.start_time, quarter.end_time, freq="M"))]


def assemble_month(expected, actual, *, baseline_last, candidate_last, specification):
    result = expected.merge(actual, on=["race_id", "horse_name"], how="outer", validate="one_to_one")
    if result[["baseline_probability", "candidate_probability", "won", "race_date"]].isna().any().any():
        raise ValueError("Monthly and quarterly complete prediction fields differ")
    result = result.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    groups = result.groupby("race_id", sort=False).size().to_numpy()
    validate_probabilities(result.candidate_probability, result.won, groups)
    dates = pd.to_datetime(result.race_date)
    if not dates.dt.to_period("M").astype(str).eq(specification["month"]).all():
        raise ValueError("Forecasts do not belong to the registered month")
    if pd.Timestamp(candidate_last) > pd.Timestamp(specification["as_of"]):
        raise ValueError("Monthly fit used labels after the registered cutoff")
    result["prediction_month"] = specification["month"]
    result["refitted_for_month"] = not specification["reuse_quarterly_model"]
    result["baseline_training_last_date"] = baseline_last
    result["candidate_training_last_date"] = candidate_last
    result["baseline_parameter_age_days"] = (dates - pd.Timestamp(baseline_last)).dt.days
    result["candidate_parameter_age_days"] = (dates - pd.Timestamp(candidate_last)).dt.days
    return result


def cadence_advances(pooled, folds):
    return bool(advancement(pooled, folds)
                and pooled["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] < 0)


def run(args):
    registration_path = Path(args.registration)
    registration = json.loads(registration_path.read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = Path(registration["baseline_experiment"])
    previous = json.loads((baseline / "manifest.json").read_text())
    names = registration["features"]
    if (names != previous["variants"]["baseline"] or registration["params"] != previous["params"]
            or registration["calibration"] != previous["calibration"]
            or bool(config.WIN_SOFT_LABEL_DISTILL) != previous["distillation"]
            or digest(ROOT / "config.py") != previous["code_sha256"]["config.py"]
            or registration["purge_days"] != purge_days()):
        raise ValueError("Refitting cadence must not change the forecast recipe")
    meta = json.loads((ROOT / "data/processed/feature_store/meta.json").read_text())
    if (meta["fe_signature"] != fe_signature() or meta["fe_signature"] != registration["fe_signature"]
            or meta["data_signature"] != registration["data_signature"]):
        raise ValueError("Registered feature/store snapshot changed")
    inputs = {name: digest(ROOT / name) for name in registration["input_sha256"]}
    if inputs != registration["input_sha256"]:
        raise ValueError("Frozen refit-cadence input changed")
    schedule = month_schedule(registration["periods"], registration["purge_days"])
    if schedule != registration["schedule"]:
        raise ValueError("Registered calendar schedule differs")
    sources = {name: digest(ROOT / name) for name in SOURCES}
    manifest = {"experiment": "004 fixed-recipe monthly versus quarterly refitting", "registration": registration,
                "registration_sha256": digest(registration_path), "code_sha256": sources,
                "source_store_meta": meta, "input_sha256": inputs,
                "scope": "Retrospective development replay, not a new untouched holdout or production promotion"}
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise ValueError("Manifest changed; use a separately registered experiment")
    manifest_path.write_bytes(encoded)
    manifest_hash = hashlib.sha256(encoded).hexdigest()
    if not (output / "source_snapshot.zip").exists():
        with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCES:
                archive.write(ROOT / name, name)
    frame = pd.read_parquet(baseline / "inputs.parquet", columns=list(dict.fromkeys(names + META)))
    frame.race_date = pd.to_datetime(frame.race_date).dt.normalize()
    dataset_start = frame.race_date.min()
    frame = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    cached = pd.read_parquet(baseline / "paired_predictions.parquet")
    cached = cached[[name for name in META if name in cached] + ["baseline_probability"]].copy()
    cached.race_date = pd.to_datetime(cached.race_date).dt.normalize()
    original_allowlist = config.FEATURE_ALLOWLIST
    months = []
    started = time.monotonic()
    try:
        config.FEATURE_ALLOWLIST = names
        for entry in schedule:
            month, period = entry["month"], entry["period"]
            expected = cached.loc[cached.race_date.dt.to_period("M").astype(str).eq(month)].copy()
            if expected.empty:
                raise ValueError("A registered month has no complete forecast fields")
            baseline_info = json.loads((baseline / f"{period}_baseline_fit.json").read_text())
            baseline_last = baseline_info["training_last_date"]
            if entry["reuse_quarterly_model"]:
                available_last = frame.loc[frame.race_date.le(entry["as_of"]), "race_date"].max()
                if available_last != pd.Timestamp(baseline_last):
                    raise ValueError("Shared first-month training history differs")
                actual = expected[["race_id", "horse_name", "baseline_probability"]].rename(
                    columns={"baseline_probability": "candidate_probability"})
                candidate_last = baseline_last
            else:
                path = output / f"{month}_monthly.joblib"
                predictor = RacePredictor(random_seed=registration["seed"],
                    frameworks={"classifier": "lgbm", "place": "lgbm"},
                    win_calibration_method=registration["calibration"])
                predictor._autotune_njobs = registration["threads"]
                if path.exists():
                    predictor.load(str(path))
                else:
                    print(json.dumps({"stage": "fitting", "month": month, "as_of": entry["as_of"]}), flush=True)
                    training = frame.loc[frame.race_date.le(entry["as_of"])].copy()
                    predictor.fit_as_of(training, entry["as_of"], params=registration["params"],
                                        dataset_start=dataset_start, win_only=True)
                    del training
                    if {name: digest(ROOT / name) for name in SOURCES} != sources:
                        raise RuntimeError("Source changed during cadence fitting")
                    predictor.eval_split_info.update(experiment_manifest_sha256=manifest_hash, month=month)
                    predictor.save(str(path))
                if (predictor.eval_split_info.get("experiment_manifest_sha256") != manifest_hash
                        or predictor.feature_cols != names or predictor.eval_split_info["as_of"] != entry["as_of"]):
                    raise ValueError("Cached monthly predictor belongs to another recipe or cutoff")
                candidate_last = predictor.eval_split_info["training_last_date"]
                test = frame.loc[frame.race_id.isin(expected.race_id)].copy()
                actual = test[["race_id", "horse_name"]].copy()
                actual["candidate_probability"] = align_predictions(predictor, test)
                (output / f"{month}_fit.json").write_text(json.dumps(predictor.eval_split_info, indent=2))
                print(json.dumps({"stage": "fit_complete", "month": month}), flush=True)
                del predictor, test
                gc.collect()
            scored = assemble_month(expected, actual, baseline_last=baseline_last,
                                    candidate_last=candidate_last, specification=entry)
            scored.to_parquet(output / f"{month}_predictions.parquet", index=False)
            months.append(scored)
    finally:
        config.FEATURE_ALLOWLIST = original_allowlist
    paired = pd.concat(months, ignore_index=True).sort_values(["race_date", "race_id", "horse_name"])
    by_period = paired.race_date.dt.to_period("Q").astype(str)
    folds = [{"period": period, **score_pair(paired.loc[by_period.eq(period)], seed=registration["seed"])}
             for period in registration["periods"]]
    pooled = score_pair(paired, seed=registration["seed"])
    report = {"manifest_sha256": manifest_hash, "pooled": pooled, "folds": folds,
              "candidate_advances": cadence_advances(pooled, folds), "production_changed": False,
              "changed_months_diagnostic": score_pair(paired.loc[paired.refitted_for_month], seed=registration["seed"]),
              "parameter_age_days": {column: paired[column].describe().to_dict() for column in (
                  "baseline_parameter_age_days", "candidate_parameter_age_days")},
              "elapsed_seconds": round(time.monotonic() - started, 2),
              "scope": manifest["scope"], "rule": registration["advancement_rule"]}
    paired.to_parquet(output / "paired_predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    assess_market(SimpleNamespace(experiment=str(output), snapshots=registration["snapshots"], seed=registration["seed"]))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="data/audit/market_performance/004_registration.json")
    parser.add_argument("--output-dir", default="data/audit/market_performance/004_refit_cadence")
    logging.basicConfig(level=logging.INFO)
    run(parser.parse_args())
