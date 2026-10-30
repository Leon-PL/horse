"""Replay two saved models on identical complete historical race fields.

This is retrospective diagnostics, not an untouched-holdout claim. The start
date is also constrained by both runs' saved test exports, keeping every
evaluated row later than their respective training splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forecast_evaluation import paired_interval, probability_metrics, valid_race_mask
from src.run_store import RUNS_DIR, load_run_predictor, run_signatures


def compare(frame, baseline, candidate, prediction_path=None):
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    valid = valid_race_mask(frame)
    excluded = int((~valid).sum())
    frame = frame[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete single-winner races remain")
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    winners = pd.to_numeric(frame.won).to_numpy(dtype=float)
    scored = frame[["race_id", "race_date", "horse_name", "finish_position", "won", "num_runners"]].copy()
    output, losses, briers = {}, {}, {}
    for name, predictor in (("baseline", baseline), ("candidate", candidate)):
        prediction = predictor.predict_races(frame)
        aligned = frame[["race_id", "horse_name"]].merge(
            prediction[["race_id", "horse_name", "win_probability"]],
            on=["race_id", "horse_name"], how="left", sort=False, validate="one_to_one",
        )
        p = aligned.win_probability.to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f"{name} returned invalid probabilities")
        np.testing.assert_allclose(np.add.reduceat(p, np.r_[0, np.cumsum(groups)[:-1]]),
                                   1.0, atol=1e-10)
        output[name], losses[name], briers[name] = probability_metrics(p, winners, groups)
        scored[f"{name}_probability"] = p
        # This comparison's decision uses proper probability scores. The
        # production rank report separately handles partial and full ties.
        output[name].pop("top1_accuracy")
    scores = ("brier", "binary_log_loss", "race_log_loss")
    if prediction_path is not None:
        Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(prediction_path, index=False)
    return {
        "races": len(groups), "runners": len(frame), "excluded_rows": excluded,
        "start": str(frame.race_date.min())[:10], "end": str(frame.race_date.max())[:10],
        "metrics": output,
        "paired_delta_candidate_minus_baseline": {
            "race_log_loss": paired_interval(losses["candidate"] - losses["baseline"]),
            "race_mean_brier": paired_interval(briers["candidate"] - briers["baseline"]),
        },
        "non_worsening_probability_scores": all(output["candidate"][s] <= output["baseline"][s]
                                                for s in scores),
        "scope": "Retrospective shared-race evaluation; dates were previously inspected. No betting edge claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--start", default="2026-07-15")
    parser.add_argument("--features", default="data/processed/feature_store/featured.parquet")
    parser.add_argument("--output", default="data/audit/20261025/probability_run_comparison.json")
    args = parser.parse_args()
    baseline = load_run_predictor(args.baseline)
    candidate = load_run_predictor(args.candidate)
    if baseline is None or candidate is None:
        raise ValueError("Both runs must have a saved predictor")
    store_meta = json.loads(Path(args.features).with_name("meta.json").read_text())
    store_signature = store_meta["fe_signature"]
    start = pd.Timestamp(args.start)
    for run_id in (args.baseline, args.candidate):
        if run_signatures(run_id).get("fe_signature") != store_signature:
            raise ValueError(f"Run {run_id} does not match the feature-generation code")
        saved = pd.read_csv(Path(RUNS_DIR) / run_id / "predictions.csv", usecols=["race_date"])
        start = max(start, pd.to_datetime(saved.race_date).min())
    metadata = ["race_id", "race_date", "horse_name", "finish_position", "won", "odds", "num_runners", "handicap"]
    columns = list(dict.fromkeys(baseline.feature_cols + candidate.feature_cols + metadata))
    frame = pd.read_parquet(args.features, columns=columns)
    frame["race_date"] = pd.to_datetime(frame.race_date)
    frame = frame[frame.race_date >= start]
    prediction_path = Path(args.output).with_suffix(".parquet")
    report = compare(frame, baseline, candidate, prediction_path=prediction_path)
    report.update(baseline_run=args.baseline, candidate_run=args.candidate,
                  fe_signature=store_signature,
                  data_signature=store_meta.get("data_signature"),
                  paired_predictions=str(prediction_path.resolve()),
                  selection_gate="Do not replace baseline if Brier, binary log loss or race log loss worsens.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
