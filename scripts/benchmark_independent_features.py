"""Registered, purged comparison of two independent probability feature sets.

Shares native fits, calibration, inference, folds and scoring with production
and the forecast benchmark. Does not collect data, fit place models, simulate
bets, search test scores or promote a model. All registered quarters must
finish before deciding whether the candidate advances.
"""
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
import pyarrow.parquet as pq
import config
from scripts.benchmark_forecasts import align_predictions
from scripts.replay_market_snapshots import digest
from scripts.assess_forecast_parity import upper_bound
from src.featured_dataset import fe_signature
from src.forecast_evaluation import paired_interval, probability_metrics, valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.wf_eval import development_mask, expanding_period_folds, purge_days

META = ["race_id", "race_date", "horse_name", "won", "finish_position", "num_runners", "odds",
        "lengths_behind", "handicap", "track", "off_time", "race_type", "prior_form_runs", "has_atr_history"]
SOURCES = ["config.py", "src/model.py", "src/wf_eval.py", "src/forecast_evaluation.py",
           "src/probability_calibration.py", "scripts/benchmark_independent_features.py",
           "scripts/benchmark_forecasts.py", "scripts/assess_forecast_parity.py",
           "scripts/replay_market_snapshots.py", "scripts/audit_atr_invariance.py"]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def score_pair(frame, *, seed=42):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id.to_numpy(), np.repeat(frame.race_id.drop_duplicates().to_numpy(), groups)):
        raise ValueError("Score contiguous complete race fields")
    y = frame.won.to_numpy(float)
    base = frame.baseline_probability.to_numpy(float)
    candidate = frame.candidate_probability.to_numpy(float)
    bm, bl, bb = probability_metrics(base, y, groups)
    cm, cl, cb = probability_metrics(candidate, y, groups)
    dates = frame.groupby("race_id", sort=False).race_date.first().astype(str).to_numpy()
    runner_delta = np.square(candidate - y) - np.square(base - y)
    return {"races": len(groups), "runners": len(frame), "baseline": bm, "candidate": cm,
            "paired_delta_candidate_minus_baseline": {
                "race_log_loss": paired_interval(cl - bl, seed=seed),
                "race_mean_brier": paired_interval(cb - bb, seed=seed),
                "date_cluster_race_log_loss": paired_interval(cl - bl, seed=seed, blocks=dates),
                "runner_brier_upper95": upper_bound(runner_delta, frame.race_id.to_numpy(), seed=seed),
            }}


def advancement(pooled, folds):
    if len(folds) < 3:
        return False
    return bool(
        pooled["paired_delta_candidate_minus_baseline"]["race_log_loss"]["ci95"][1] < 0
        and pooled["candidate"]["brier"] <= pooled["baseline"]["brier"]
        and pooled["candidate"]["binary_log_loss"] <= pooled["baseline"]["binary_log_loss"]
        and sum(f["candidate"]["race_log_loss"] < f["baseline"]["race_log_loss"] for f in folds) >= 2
    )


def check_eligibility(registration, audit):
    added = set(registration["candidate_features"]) - set(registration["baseline_features"])
    if (not audit["passed"] or set(audit["tested_columns"]) != added
            or audit["registration_content_sha256"] != canonical_hash(registration)):
        raise ValueError("The exact added feature family must pass the invariance audit")
    for source, expected in audit["source_sha256"].items():
        if digest(ROOT / source) != expected:
            raise ValueError(f"Feature audit source changed: {source}")
    unsafe = set(registration["unsafe_composites_requiring_exclusion_or_rebuild"])
    if unsafe.intersection(registration["candidate_features"]):
        raise ValueError("Registered candidate includes a quarantined future-dependent feature")


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registration = json.loads(Path(args.registration).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.invariance_audit).read_text(encoding="utf-8"))
    check_eligibility(registration, audit)
    source_meta = json.loads(Path(args.features).with_name("meta.json").read_text(encoding="utf-8"))
    if (source_meta["fe_signature"] != fe_signature()
            or source_meta["fe_signature"] != registration["fe_signature"]
            or source_meta["data_signature"] != registration["data_signature"]):
        raise ValueError("The registered feature/data snapshot does not match the current store")
    source_hashes = {p: digest(ROOT / p) for p in SOURCES}
    source_file_hash = digest(args.features)
    periods = registration["candidate_periods"]
    if len(periods) != len(set(periods)) or len(periods) < 3:
        raise ValueError("All three distinct predeclared quarters are required")
    end = max(pd.Period(p, freq="Q").end_time.normalize() for p in periods)
    schema = pq.read_schema(args.features)
    names = list(dict.fromkeys(registration["baseline_features"] + registration["candidate_features"]))
    missing = set(names) - set(schema.names)
    if missing:
        raise ValueError(f"Registered feature columns missing: {sorted(missing)}")
    columns = [c for c in schema.names if c in set(names)]
    metadata = [c for c in META if c in schema.names and c not in columns]
    projection_path = output / "inputs.parquet"
    projection_meta = output / "inputs.json"
    if projection_path.exists():
        cached = json.loads(projection_meta.read_text(encoding="utf-8"))
        if cached["source_sha256"] != source_file_hash or cached["columns"] != columns + metadata:
            raise ValueError("Cached input projection belongs to another dataset or feature family")
        if digest(projection_path) != cached["projection_sha256"]:
            raise ValueError("Input projection fingerprint changed")
        frame = pd.read_parquet(projection_path)
    else:
        frame = pd.read_parquet(args.features, columns=columns + metadata,
                                filters=[("race_date", "<=", end)])
        frame[columns] = frame[columns].astype(np.float32)
        if digest(args.features) != source_file_hash:
            raise RuntimeError("Source feature file changed while reading the research snapshot")
        frame.to_parquet(projection_path, index=False, compression="zstd")
        projection_meta.write_text(json.dumps({"source_sha256": source_file_hash, "columns": columns + metadata,
                                               "projection_sha256": digest(projection_path), "dtype": "float32 features"}, indent=2), encoding="utf-8")
    frame["race_date"] = pd.to_datetime(frame.race_date).dt.normalize()
    dataset_start = frame.race_date.min()
    valid = valid_race_mask(frame)
    excluded = int((~valid).sum())
    frame = frame.loc[valid & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    folds = {fold.label: fold for fold in expanding_period_folds(frame.race_date)}
    if any(period not in folds for period in periods):
        raise ValueError("A registered quarter is unavailable inside the permitted research boundary")
    variants = {name: [c for c in columns if c in set(registration[f"{name}_features"])]
                for name in ("baseline", "candidate")}
    params = {"classifier": {**config.CLASSIFIER_PARAMS, "n_estimators": args.max_trees}}
    manifest = {
        "experiment": registration.get("experiment", "002 independent ATR family probability comparison"), "registration": registration,
        "registration_sha256": canonical_hash(registration), "feature_invariance_audit": audit,
        "source_meta": source_meta, "source_file_sha256": source_file_hash,
        "projection_sha256": digest(projection_path), "code_sha256": source_hashes,
        "variants": variants, "params": params, "seed": args.seed, "threads": args.threads,
        "feature_dtype": "float32 for both variants", "calibration": config.WIN_CALIBRATION_METHOD,
        "recency_half_life_days": config.RECENCY_HALF_LIFE_DAYS,
        "distillation": bool(config.WIN_SOFT_LABEL_DISTILL), "purge_days": purge_days(),
        "invalid_rows_excluded": excluded, "research_boundary": "development_dates_only",
        "folds": [{"period": p, "cutoff": str(frame.loc[folds[p].train, "race_date"].max())[:10],
                   "test_start": str(frame.loc[folds[p].test, "race_date"].min())[:10],
                   "test_end": str(frame.loc[folds[p].test, "race_date"].max())[:10]} for p in periods],
        "scope": "Retrospective independent forecast comparison on complete final-starter fields; dates may have been examined by earlier project research",
        "selection": "All registered quarters, fixed recipes; no test-driven tuning or early favourable stopping; no automatic promotion",
    }
    path = output / "manifest.json"
    encoded = json.dumps(manifest, sort_keys=True, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise ValueError("Experiment manifest changed; choose a new output directory")
    path.write_text(encoded, encoding="utf-8")
    manifest_hash = hashlib.sha256(encoded.encode()).hexdigest()
    if not (output / "source_snapshot.zip").exists():
        with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in dict.fromkeys(SOURCES + list(audit["source_sha256"])):
                archive.write(ROOT / source, source)
    print(json.dumps({"stage": "registered", "manifest_sha256": manifest_hash,
                      "variants": {k: len(v) for k, v in variants.items()}, "periods": periods}), flush=True)
    started = time.monotonic()
    results, predictions = [], []
    for period in periods:
        fold = folds[period]
        cutoff = frame.loc[fold.train, "race_date"].max()
        test = frame.loc[fold.test].copy()
        scored = test[[c for c in META if c in test]].copy()
        for name, feature_columns in variants.items():
            model_path = output / f"{period}_{name}.joblib"
            predictor = RacePredictor(random_seed=args.seed, frameworks={"classifier": "lgbm", "place": "lgbm"})
            predictor._autotune_njobs = args.threads
            if model_path.exists():
                predictor.load(str(model_path))
            else:
                print(json.dumps({"stage": "fitting", "period": period, "variant": name,
                                  "cutoff": str(cutoff.date()), "features": len(feature_columns)}), flush=True)
                previous = config.FEATURE_ALLOWLIST
                try:
                    config.FEATURE_ALLOWLIST = feature_columns
                    allowed_columns = list(dict.fromkeys(feature_columns + [c for c in META if c in frame]))
                    train = frame.loc[fold.train, allowed_columns].copy()
                    if get_feature_columns(train) != feature_columns:
                        raise ValueError("Requested feature order/eligibility differs from native fitting")
                    predictor.fit_as_of(train, cutoff, params=params, dataset_start=dataset_start, win_only=True)
                    del train
                finally:
                    config.FEATURE_ALLOWLIST = previous
                if {p: digest(ROOT / p) for p in SOURCES} != source_hashes:
                    raise RuntimeError("Research source changed during fitting")
                predictor.eval_split_info.update(experiment_manifest_sha256=manifest_hash, variant=name, period=period)
                predictor.save(str(model_path))
            if (predictor.eval_split_info.get("experiment_manifest_sha256") != manifest_hash
                    or predictor.feature_cols != feature_columns):
                raise ValueError("Cached predictor belongs to another manifest or feature set")
            scored[f"{name}_probability"] = align_predictions(predictor, test)
            (output / f"{period}_{name}_fit.json").write_text(json.dumps(predictor.eval_split_info, indent=2), encoding="utf-8")
            del predictor
            gc.collect()
        scored.to_parquet(output / f"{period}_predictions.parquet", index=False)
        result = {"period": period, **score_pair(scored, seed=args.seed)}
        results.append(result)
        predictions.append(scored)
        (output / "progress.json").write_text(json.dumps({"stage": "partial_diagnostic_only", "completed_folds": results,
                                                          "selection_permitted": False}, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "fold_complete", **result}), flush=True)
        del test
        gc.collect()
    paired = pd.concat(predictions, ignore_index=True).sort_values(["race_date", "race_id", "horse_name"])
    pooled = score_pair(paired, seed=args.seed)
    report = {"manifest_sha256": manifest_hash, "folds": results, "pooled": pooled,
              "candidate_advances": advancement(pooled, results), "production_changed": False,
              "elapsed_seconds": round(time.monotonic() - started, 2), "scope": manifest["scope"],
              "rule": registration["advancement_rule"]}
    paired.to_parquet(output / "paired_predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/feature_store/featured.parquet")
    parser.add_argument("--registration", default="data/audit/market_performance/002_atr_family_registration.json")
    parser.add_argument("--invariance-audit", default="data/audit/market_performance/002_feature_invariance.json")
    parser.add_argument("--output-dir", default="data/audit/market_performance/002_atr_family")
    parser.add_argument("--max-trees", type=int, default=config.CLASSIFIER_PARAMS["n_estimators"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(build_parser().parse_args())
