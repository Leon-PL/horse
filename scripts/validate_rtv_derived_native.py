"""Native win-model check of the RTV-derived features and the Timeform card.

Fits the production recipe (``RacePredictor.fit_as_of``, win head) twice per
development quarter, once on ``feature_sets/win.txt`` and once with the
family added (``feature_sets/win_rtv_derived.txt``), each through eight days
before the quarter, and scores the quarter's complete fields against each
other and against the Betfair SP. The golden hold-out is never loaded.

Fits are cached per quarter and arm, so an interrupted run resumes. Delete
``data/audit/rtv_derived_native`` after a store rebuild.

Usage::

    python scripts/validate_rtv_derived_native.py [--threads 6]
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from scripts.benchmark_forecasts import align_predictions, score_forecasts  # noqa: E402
from scripts.benchmark_independent_features import META, score_pair  # noqa: E402
from src.betfair import attach_bsp  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.featured_dataset import _model_load_columns, fe_signature  # noqa: E402
from src.forecast_evaluation import valid_race_mask  # noqa: E402
from src.model import RacePredictor, _resolve_feature_allowlist  # noqa: E402
from src.wf_eval import development_mask  # noqa: E402

CASE = ROOT / "data/audit/rtv_derived_native"
FOLDS = {"2026Q1": ("2026-01-01", "2026-03-31"), "2026Q2": ("2026-04-01", "2026-06-03")}
ARMS = {"baseline": ROOT / "feature_sets/win.txt", "candidate": ROOT / "feature_sets/win_rtv_derived.txt"}
KEY = ["race_id", "horse_name"]


def load_frame():
    store = FeatureStore()
    if store.meta().get("fe_signature") != fe_signature():
        raise SystemExit("The feature store predates the current feature code; rebuild it first")
    config.FEATURE_ALLOWLIST = str(ARMS["candidate"])
    missing = set(_resolve_feature_allowlist()) - set(store.featured_schema_names())
    if missing:
        raise SystemExit(f"Store lacks candidate features: {sorted(missing)}")
    columns = _model_load_columns(store)
    frame = pd.read_parquet(store.featured_path, columns=columns,
                            filters=[("race_date", "<=", pd.Timestamp(max(e for _, e in FOLDS.values())))])
    frame = frame.loc[development_mask(frame.race_date)]
    dataset_start = pd.to_datetime(frame.race_date).min()
    frame = frame.loc[valid_race_mask(frame)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    return frame.reset_index(drop=True), dataset_start


def fit_arm(frame, dataset_start, fold, arm, threads):
    path = CASE / fold / f"{arm}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    test_start, test_end = FOLDS[fold]
    cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=8)
    dates = pd.to_datetime(frame.race_date)
    test = frame.loc[dates.between(test_start, test_end)]
    config.FEATURE_ALLOWLIST = str(ARMS[arm])
    predictor = RacePredictor(frameworks={"classifier": "lgbm", "place": "lgbm"}, random_seed=config.RANDOM_SEED,
                              win_calibration_method=config.WIN_CALIBRATION_METHOD)
    predictor._autotune_njobs = threads
    started = time.time()
    predictor.fit_as_of(frame.loc[dates.le(cutoff)], cutoff.strftime("%Y-%m-%d"),
                        dataset_start=dataset_start, win_only=True)
    expected = [c for c in _resolve_feature_allowlist() if c in frame]
    dropped = sorted(set(expected) - set(predictor.feature_cols))
    out = test[KEY].copy()
    out["probability"] = align_predictions(predictor, test)
    path.parent.mkdir(parents=True, exist_ok=True)
    (CASE / fold / f"{arm}.json").write_text(json.dumps({
        "cutoff": str(cutoff.date()), "features": len(predictor.feature_cols), "dropped_by_recipe": dropped,
        "seconds": round(time.time() - started), "split": predictor.eval_split_info}, indent=2, default=str))
    out.to_parquet(path, index=False)
    del predictor
    gc.collect()
    return out


def market_probability(frame, column):
    """A Betfair price normalised within race; NaN for the whole race if any price is missing."""
    implied = 1 / pd.to_numeric(frame[column], errors="coerce").where(lambda s: (s > 1) & np.isfinite(s))
    total = implied.groupby(frame.race_id).transform("sum")
    complete = implied.notna().groupby(frame.race_id).transform("all")
    return (implied / total).where(complete)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    frame, dataset_start = load_frame()
    rows, paired = [], []
    for fold, (test_start, test_end) in FOLDS.items():
        test = frame.loc[pd.to_datetime(frame.race_date).between(test_start, test_end)]
        scored = test[[c for c in META if c in test]].copy()
        for arm in ARMS:
            print(json.dumps({"stage": "fit", "fold": fold, "arm": arm}), flush=True)
            predictions = fit_arm(frame, dataset_start, fold, arm, args.threads)
            scored[f"{arm}_probability"] = scored[KEY].merge(predictions, on=KEY, how="left",
                                                             validate="one_to_one").probability.to_numpy()
        if scored[["baseline_probability", "candidate_probability"]].isna().any().any():
            raise ValueError(f"{fold}: a runner has no prediction")
        paired.append(scored)
        report = score_pair(scored)
        delta = report["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]
        rows.append({"fold": fold, "races": report["races"], "baseline_ll": report["baseline"]["race_log_loss"],
                     "candidate_ll": report["candidate"]["race_log_loss"], "delta_ll": delta["mean"],
                     "ci_lo": delta["ci95"][0], "ci_hi": delta["ci95"][1]})
    pooled = pd.concat(paired, ignore_index=True)
    report = score_pair(pooled)
    delta = report["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]
    rows.append({"fold": "pooled", "races": report["races"], "baseline_ll": report["baseline"]["race_log_loss"],
                 "candidate_ll": report["candidate"]["race_log_loss"], "delta_ll": delta["mean"],
                 "ci_lo": delta["ci95"][0], "ci_hi": delta["ci95"][1]})
    pooled = attach_bsp(pooled, cols=("bsp", "ltp_300s"))
    summary = {"folds": rows, "pooled": report, "vs_market": {}, "store_fe_signature": fe_signature()}
    print("\nRace log loss, candidate minus baseline (negative = better), date-cluster 95% interval:")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.5f}" if abs(v) < 1 else f"{v:.5f}"))
    for column, label in (("bsp", "Betfair SP"), ("ltp_300s", "T-5 Betfair price")):
        pooled["market_probability"] = market_probability(pooled, column)
        priced = pooled.loc[pooled.groupby("race_id").market_probability.transform(lambda s: s.notna().all())]
        market = score_forecasts(priced, {n: priced[f"{n}_probability"].to_numpy(float)
                                          for n in ("baseline", "candidate", "market")})
        metrics = market["metrics"]
        gap = metrics["baseline"]["race_log_loss"] - metrics["market"]["race_log_loss"]
        closed = (metrics["baseline"]["race_log_loss"] - metrics["candidate"]["race_log_loss"]) / gap if gap > 0 else None
        summary["vs_market"][column] = {**market, "gap_closed": closed}
        print(f"{label}: {priced.race_id.nunique():,} fully priced races; race log loss baseline "
              f"{metrics['baseline']['race_log_loss']:.5f}, candidate {metrics['candidate']['race_log_loss']:.5f}, "
              f"market {metrics['market']['race_log_loss']:.5f}; share of the gap closed {closed:.3f}")
    pooled.drop(columns="market_probability").to_parquet(CASE / "paired_predictions.parquet", index=False)
    (CASE / "report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
