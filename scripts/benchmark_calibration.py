"""Compare calibration on strictly chronological, disjoint race windows.

This is an offline research benchmark: no scraping, model saving, promotion,
or mutation of the feature store. Use the audit's repaired feature snapshot,
or a fully rebuilt store. Results from a partial repair are diagnostic only.
The reserved project holdout and its purge gap are excluded by default.
Opting in is retrospective analysis, not evidence of an untouched holdout.

Example:
    python scripts/benchmark_calibration.py --features data/audit/20261011/repaired_features.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src.model import RacePredictor, _grouped_normalize, get_feature_columns
from src.probability_calibration import fit_temperature, race_temperature
from src.forecast_evaluation import paired_interval, probability_metrics, valid_race_mask
from src.wf_eval import development_mask, golden_holdout_start, holdout_mask, purge_days


def research_boundary(df, *, include_reserved_holdout=False):
    """Apply the shared research cutoff before constructing any windows."""
    allowed = development_mask(df["race_date"])
    selected = df if include_reserved_holdout else df.loc[allowed]
    metadata = {
        "include_reserved_holdout": bool(include_reserved_holdout),
        "reserved_holdout_start": str(golden_holdout_start().date()),
        "development_purge_days": purge_days(),
        "excluded_by_research_boundary_rows": 0 if include_reserved_holdout else int((~allowed).sum()),
        "reserved_holdout_rows_in_benchmark": int(holdout_mask(selected["race_date"]).sum()),
        "evaluation_scope": "retrospective_including_reserved_dates" if include_reserved_holdout else "development_dates_only",
        "confirmation_limitation": (
            "Confirmation is held out only from candidate selection within this run. "
            "It may have been examined in prior research and is not a claim of universally unseen data."
        ),
    }
    return selected.reset_index(drop=True), metadata


def window_masks(dates, test_start, *, train_days=730, calibration_days=90,
                 stopping_days=60, test_days=90, purge_days=7):
    test_end = test_start + pd.Timedelta(days=test_days)
    calibration_end = test_start - pd.Timedelta(days=purge_days)
    calibration_start = calibration_end - pd.Timedelta(days=calibration_days)
    stopping_end = calibration_start - pd.Timedelta(days=purge_days)
    stopping_start = stopping_end - pd.Timedelta(days=stopping_days)
    train_end = stopping_start - pd.Timedelta(days=purge_days)
    train_start = train_end - pd.Timedelta(days=train_days)
    return {
        "train": (dates >= train_start) & (dates < train_end),
        "stopping": (dates >= stopping_start) & (dates < stopping_end),
        "calibration": (dates >= calibration_start) & (dates < calibration_end),
        "test": (dates >= test_start) & (dates < test_end),
    }


def run(args):
    from lightgbm import LGBMClassifier, early_stopping
    schema = pq.read_schema(args.features)
    # Discover the model schema without loading the 1,000+ unused columns.
    empty = schema.empty_table().to_pandas()
    features = get_feature_columns(empty)
    metadata = [c for c in ("race_id", "race_date", "horse_name", "won",
                            "finish_position", "num_runners", "odds") if c in schema.names]
    df = pd.read_parquet(args.features, columns=sorted(set(features + metadata)))
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.normalize()
    df = df.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    valid = valid_race_mask(df)
    excluded = int((~valid).sum())
    df = df[valid].reset_index(drop=True)
    df["finish_position"] = pd.to_numeric(df["finish_position"])
    df, boundary = research_boundary(df, include_reserved_holdout=args.include_reserved_holdout)
    if df.empty:
        raise ValueError("No valid races remain within the permitted research dates")
    y = df["won"].to_numpy(dtype=float)
    X = df[features].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    end = df["race_date"].max() + pd.Timedelta(days=1)
    output = {
        "features_file": str(Path(args.features).resolve()),
        "source_note": args.source_note,
        "feature_count": len(features), "excluded_rows": excluded,
        "research_boundary": boundary,
        "folds": [], "protocol": "train -> purge -> early stopping -> purge -> calibration -> purge -> test",
        "selection_rule": "Development folds first; latest fold is within-run confirmation only. Lower Brier/log loss is better.",
    }
    params = {**config.CLASSIFIER_PARAMS, "objective": "binary", "n_jobs": args.threads,
              "verbosity": -1, "random_state": args.seed, "subsample_freq": 1}
    params.pop("min_child_weight", None)
    for fold in range(args.folds):
        start = end - pd.Timedelta(days=args.test_days * (args.folds - fold))
        masks = window_masks(df["race_date"], start, train_days=args.train_days, test_days=args.test_days)
        if any(m.sum() < 100 for m in masks.values()):
            raise ValueError(f"Insufficient data for fold beginning {start.date()}")
        tr, es, ca, te = [m.to_numpy() for m in masks.values()]
        model = LGBMClassifier(**params)
        t0 = time.monotonic()
        model.fit(X[tr], y[tr], eval_set=[(X[es], y[es])],
                  callbacks=[early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)])
        raw_cal, raw_test = model.predict_proba(X[ca])[:, 1], model.predict_proba(X[te])[:, 1]
        gc = df.loc[ca].groupby("race_id", sort=False).size().to_numpy()
        gt = df.loc[te].groupby("race_id", sort=False).size().to_numpy()
        predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"},
                                  win_calibration_method="platt_isotonic")
        fp = df.loc[ca, "finish_position"].to_numpy()
        predictor.platt_a, predictor.platt_b = predictor._optimise_platt_calibration(raw_cal, gc, fp)
        cal_platt = predictor._calibrate_win_probs(raw_cal, gc)
        platt = predictor._calibrate_win_probs(raw_test, gt)
        predictor.win_iso_cal = predictor._fit_isotonic(cal_platt, y[ca])
        iso = predictor._calibrate_win_probs(raw_test, gt)
        temperature = fit_temperature(raw_cal, gc, y[ca])
        candidates = {"raw_normalised": _grouped_normalize(raw_test, gt),
                      "platt_isotonic": iso, "platt_only": platt,
                      "race_temperature": race_temperature(raw_test, gt, temperature)}
        metrics, losses, briers = {}, {}, {}
        for name, p in candidates.items():
            metrics[name], losses[name], briers[name] = probability_metrics(p, y[te], gt)
        deltas = {name: {"race_log_loss": paired_interval(losses[name] - losses["platt_isotonic"]),
                         "race_mean_brier": paired_interval(briers[name] - briers["platt_isotonic"])}
                  for name in candidates if name != "platt_isotonic"}
        details = {
            "test_start": str(start.date()), "test_end": str((start + pd.Timedelta(days=args.test_days)).date()),
            "role": "confirmation" if fold == args.folds - 1 else "development",
            "races": len(gt), "rows": int(te.sum()), "best_iteration": int(model.best_iteration_),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "temperature": temperature, "platt": [predictor.platt_a, predictor.platt_b],
            "metrics": metrics, "delta_vs_platt_isotonic": deltas,
            "windows": {k: {"start": str(df.loc[m, "race_date"].min().date()),
                              "end": str(df.loc[m, "race_date"].max().date()), "rows": int(m.sum())}
                        for k, m in masks.items()},
        }
        output["folds"].append(details)
        if fold == args.folds - 2:
            # Lock the choice before the confirmation model is even trained.
            totals = {
                name: sum(f["rows"] * f["metrics"][name]["brier"] for f in output["folds"])
                for name in candidates
            }
            output["selected_candidate"] = min(totals, key=totals.get)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(details), flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="data/audit/calibration_benchmark.json")
    parser.add_argument("--source-note", default="Partial feature repair; rebuild and revalidate before production.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-reserved-holdout", action="store_true",
                        help="Include reserved holdout dates for retrospective analysis; results are not an untouched-holdout test.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
