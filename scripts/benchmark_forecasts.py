"""Frozen chronological comparison of independent and same-time market forecasts.

Pre-register a manifest before fitting. Fit one historical model before the
market-calibration period, generate OOS calibration predictions, select market
corrections on a later calibration slab, then refit the independent recipe
through the purged test boundary. Test outcomes never select corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import config
from scripts.replay_market_snapshots import digest
from src.featured_dataset import fe_signature
from src.forecast_benchmark import corrected_probabilities, fit_corrections, match_market_fields
from src.forecast_evaluation import paired_interval, probability_metrics, valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.wf_eval import development_mask, expanding_period_folds, purge_days


def align_predictions(predictor, frame):
    result = predictor.predict_races(frame, win_only=True)
    aligned = frame[["race_id", "horse_name"]].merge(
        result[["race_id", "horse_name", "win_probability"]],
        on=["race_id", "horse_name"], how="left", sort=False, validate="one_to_one")
    return aligned.win_probability.to_numpy(float)


def score_forecasts(frame, forecasts, seed=42):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    y = frame.won.to_numpy(float)
    race_days = frame.groupby("race_id", sort=False).race_date.first().astype(str).to_numpy()
    metrics, losses, briers = {}, {}, {}
    for name, p in forecasts.items():
        metrics[name], losses[name], briers[name] = probability_metrics(p, y, groups)
    deltas = {}
    for name in forecasts:
        if name == "market":
            continue
        deltas[name] = {
            "race_log_loss": paired_interval(losses[name] - losses["market"], seed=seed),
            "race_mean_brier": paired_interval(briers[name] - briers["market"], seed=seed),
            "date_cluster_race_log_loss": paired_interval(losses[name] - losses["market"], seed=seed, blocks=race_days),
        }
    return {"races": len(groups), "runners": len(frame), "metrics": metrics,
            "paired_delta_vs_market": deltas}


def gap_segments(frame, forecasts):
    """Predeclared diagnostics, never a gate selecting a profitable subgroup."""
    field = frame.groupby("race_id", sort=False)
    descriptors = pd.DataFrame(index=field.size().index)
    if "race_type" in frame:
        descriptors["race_type"] = field.race_type.first().fillna("unknown").astype(str)
    if "handicap" in frame:
        descriptors["handicap"] = field.handicap.first().fillna(-1).astype(str)
    descriptors["field_size"] = pd.cut(field.size(), [1, 7, 11, np.inf], labels=["2-7", "8-11", "12+"])
    if "prior_form_runs" in frame:
        descriptors["exposure"] = np.where(field.prior_form_runs.min() < 3, "contains lightly raced", "all at least 3 starts")
    descriptors["favourite_probability"] = pd.cut(field.market_probability.max(), [0, .25, .5, 1.01],
                                                  labels=["up to .25", ".25-.50", "above .50"])
    output = {}
    for dimension in descriptors:
        for value in descriptors[dimension].dropna().unique():
            ids = descriptors.index[descriptors[dimension] == value]
            mask = frame.race_id.isin(ids).to_numpy()
            subset = frame.loc[mask]
            if subset.race_id.nunique() < 30:
                continue
            groups = subset.groupby("race_id", sort=False).size().to_numpy()
            output[f"{dimension}={value}"] = {"races": len(groups), "metrics": {
                name: probability_metrics(p[mask], subset.won.to_numpy(float), groups)[0]
                for name, p in forecasts.items()}}
    return output


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    store_meta = json.loads(Path(args.features).with_name("meta.json").read_text(encoding="utf-8"))
    if store_meta["fe_signature"] != fe_signature():
        raise ValueError("Feature-store code signature is stale; inspect project status before benchmarking")
    schema = pq.read_schema(args.features)
    if args.feature_set:
        names = [line.strip() for line in Path(args.feature_set).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        config.FEATURE_ALLOWLIST = names
    features = get_feature_columns(schema.empty_table().to_pandas())
    metadata = [c for c in ("race_id", "race_date", "horse_name", "won", "finish_position", "num_runners",
                            "odds", "lengths_behind", "handicap", "track", "off_time", "race_type", "prior_form_runs")
                if c in schema.names]
    frame = pd.read_parquet(args.features, columns=list(dict.fromkeys(features + metadata)))
    frame["race_date"] = pd.to_datetime(frame.race_date).dt.normalize()
    dataset_start = frame.race_date.min()
    frame = frame.loc[valid_race_mask(frame)].copy()
    excluded_boundary = int((~development_mask(frame.race_date)).sum())
    if not args.include_reserved_holdout:
        frame = frame.loc[development_mask(frame.race_date)].copy()
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    snapshots = pd.read_parquet(args.snapshots)
    paired, coverage = match_market_fields(frame, snapshots)
    folds = {f.label: f for f in expanding_period_folds(frame.race_date)}
    if args.period not in folds:
        raise ValueError(f"No permissible fold {args.period}; use an explicit retrospective override for reserved dates")
    fold = folds[args.period]
    test_ids = frame.loc[fold.test, "race_id"]
    test = paired.loc[paired.race_id.isin(test_ids)].copy()
    if test.race_id.nunique() < 100:
        raise ValueError("At least 100 identical test fields are required")
    # Boundaries use the entire fold, not the first successfully joined race.
    test_start = frame.loc[fold.test, "race_date"].min()
    final_cutoff = frame.loc[fold.train, "race_date"].max()
    calibration_start = test_start - pd.Timedelta(days=args.calibration_days)
    calibration = paired.loc[(paired.race_date >= calibration_start) & (paired.race_date <= final_cutoff)].copy()
    history_cutoff = calibration_start - pd.Timedelta(days=purge_days())
    if calibration.race_id.nunique() < 200:
        raise ValueError("Insufficient complete same-time fields for correction fitting and validation")
    params = {"classifier": {**config.CLASSIFIER_PARAMS, "n_estimators": args.max_trees}}
    sources = ["src/model.py", "src/forecast_benchmark.py", "src/forecast_evaluation.py", "src/wf_eval.py",
               "src/probability_calibration.py", "scripts/benchmark_forecasts.py", "config.py"]
    manifest = {
        "experiment": args.name, "period": args.period, "features": features,
        "feature_store_meta": store_meta, "feature_file_sha256": digest(args.features),
        "snapshots_sha256": digest(args.snapshots),
        "snapshot_manifest": json.loads(Path(args.snapshots).with_suffix(".json").read_text(encoding="utf-8")),
        "code_sha256": {p: digest(ROOT / p) for p in sources},
        "params": params, "seed": args.seed, "threads": args.threads,
        "calibration_method": config.WIN_CALIBRATION_METHOD,
        "recency_half_life_days": config.RECENCY_HALF_LIFE_DAYS,
        "distillation": bool(config.WIN_SOFT_LABEL_DISTILL),
        "history_cutoff": str(history_cutoff.date()), "final_cutoff": str(final_cutoff.date()),
        "test_start": str(test_start.date()), "calibration_start": str(calibration_start.date()),
        "research_scope": "retrospective_including_reserved_dates" if args.include_reserved_holdout else "development_dates_only",
        "excluded_boundary_rows": 0 if args.include_reserved_holdout else excluded_boundary,
        "decision_rule": "Select correction on pre-test validation only; report every test forecast; no automatic promotion",
        "stop_rule": "One frozen independent recipe, one market temperature and one bounded combination; no test-driven search",
        "confirmation_limitation": "Dates previously examined by the project; this is not a new untouched holdout",
    }
    manifest_path = output / "manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("Experiment manifest changed; choose a new output directory")
    manifest_path.write_text(encoded, encoding="utf-8")
    manifest_hash = hashlib.sha256(encoded.encode()).hexdigest()
    print(json.dumps({"stage": "registered", "manifest_sha256": manifest_hash, "coverage": coverage}), flush=True)

    def fit(cutoff, label):
        path = output / f"{label}.joblib"
        predictor = RacePredictor(random_seed=args.seed, frameworks={"classifier": "lgbm", "place": "lgbm"})
        predictor._autotune_njobs = args.threads
        if path.exists():
            predictor.load(str(path))
        else:
            print(json.dumps({"stage": "fitting", "label": label, "cutoff": str(cutoff.date())}), flush=True)
            predictor.fit_as_of(frame, cutoff, params=params, dataset_start=dataset_start, win_only=True)
            if {p: digest(ROOT / p) for p in sources} != manifest["code_sha256"]:
                raise RuntimeError("Source code changed during fit; do not publish ambiguous evidence")
            predictor.eval_split_info["experiment_manifest_sha256"] = manifest_hash
            predictor.save(str(path))
        if predictor.eval_split_info.get("experiment_manifest_sha256") != manifest_hash:
            raise ValueError("Cached model belongs to another experiment manifest")
        return predictor

    started = time.monotonic()
    historical = fit(history_cutoff, "calibration_model")
    calibration["model_probability"] = align_predictions(historical, calibration)
    calibration.to_parquet(output / "calibration_predictions.parquet", index=False)
    recipe = fit_corrections(calibration)
    (output / "correction_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    final = fit(final_cutoff, "test_model")
    test["model_probability"] = align_predictions(final, test)
    forecasts = corrected_probabilities(test, recipe)
    report = score_forecasts(test, forecasts, seed=args.seed)
    report.update(coverage=coverage, manifest_sha256=manifest_hash, recipe=recipe,
                  calibration_fit=historical.eval_split_info, final_fit=final.eval_split_info,
                  segments=gap_segments(test, forecasts), elapsed_seconds=round(time.monotonic() - started, 2),
                  scope=manifest["research_scope"], confirmation_limitation=manifest["confirmation_limitation"])
    for name, p in forecasts.items():
        test[f"probability_{name}"] = p
    test.to_parquet(output / "test_predictions.parquet", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "segments"}, indent=2), flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/processed/feature_store/featured.parquet")
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="001 independent versus T-5 market ladder")
    parser.add_argument("--period", required=True, help="Calendar quarter, e.g. 2026Q3")
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--max-trees", type=int, default=config.CLASSIFIER_PARAMS["n_estimators"])
    parser.add_argument("--feature-set")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-reserved-holdout", action="store_true")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(build_parser().parse_args())
