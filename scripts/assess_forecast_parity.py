"""Descriptive parity bounds for frozen forecasts; never select on this report.

The accepted plan fixes +.002 mean race log loss and +.0001 runner Brier
margins. These retrospective bounds do not replace fresh forward confirmation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.benchmark_forecasts import gap_segments, score_forecasts
from scripts.replay_market_snapshots import digest
from src.forecast_evaluation import validate_probabilities
from src.wf_eval import bootstrap_roi_ci


def upper_bound(values, clusters, *, seed=42, draws=2000):
    # This shared cluster bootstrap estimates a mean, despite its historic
    # ROI name. Scaling preserves precision through its four-decimal output.
    result = bootstrap_roi_ci(values, clusters=clusters, n_boot=draws,
                              seed=seed, alpha=.10, scale=1_000_000)
    return {"mean": result["mean"] / 1_000_000,
            "upper95": result["ci_high"] / 1_000_000,
            "clusters": result["n_clusters"]}


def parity_evidence(frame, forecasts, *, seed=42, draws=2000):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    race_ids = frame.race_id.to_numpy()
    if not np.array_equal(race_ids, np.repeat(frame.race_id.drop_duplicates().to_numpy(), groups)):
        raise ValueError("Sort whole contiguous races before assessing paired forecasts")
    dates = pd.to_datetime(frame.race_date).dt.strftime("%Y-%m-%d").to_numpy()
    race_dates = frame.groupby("race_id", sort=False).race_date.first().astype(str).to_numpy()
    y = frame.won.to_numpy(float)
    q = forecasts["market"]
    validate_probabilities(q, y, groups)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    output = {}
    for name, p in forecasts.items():
        if name == "market":
            continue
        validate_probabilities(p, y, groups)
        runner_brier_delta = np.square(p - y) - np.square(q - y)
        race_ll_delta = np.add.reduceat(y * (np.log(np.clip(q, 1e-12, 1))
                                               - np.log(np.clip(p, 1e-12, 1))), starts)
        by_race = {
            "runner_brier": upper_bound(runner_brier_delta, race_ids, seed=seed, draws=draws),
            "race_log_loss": upper_bound(race_ll_delta, np.arange(len(groups)), seed=seed, draws=draws),
        }
        by_date = {
            "runner_brier": upper_bound(runner_brier_delta, dates, seed=seed, draws=draws),
            "race_log_loss": upper_bound(race_ll_delta, race_dates, seed=seed, draws=draws),
        }
        output[name] = {
            "race_bootstrap": by_race, "date_cluster_sensitivity": by_date,
            "within_proposed_margins_in_both_analyses": all(
                result["runner_brier"]["upper95"] <= .0001
                and result["race_log_loss"]["upper95"] <= .002 for result in (by_race, by_date)),
        }
    return output


def run(args):
    directory = Path(args.experiment)
    path = directory / "test_predictions.parquet"
    frame = pd.read_parquet(path).sort_values(["race_date", "race_id", "horse_name"]).reset_index(drop=True)
    forecasts = {c.removeprefix("probability_"): frame[c].to_numpy(float)
                 for c in frame if c.startswith("probability_")}
    if set(forecasts) != {"market", "independent", "market_temperature", "market_combination"}:
        raise ValueError("Expected the four frozen experiment forecasts")
    report = {
        "scope": "Retrospective evidence on the declared LTP reference; not forward confirmation or profitability",
        "margins": {"race_log_loss": .002, "runner_brier": .0001},
        "criterion": "One-sided 95% upper paired bounds below both margins, with race and date cluster resampling",
        "predictions_sha256": digest(path),
        "evidence": parity_evidence(frame, forecasts, seed=args.seed),
        "corrected_segments": gap_segments(frame, forecasts),
        "diagnostic_correction": "Experience slices use prior_form_runs; supersedes raw report slices based on scrape-time horse_runs. Forecasts and overall scores are unchanged.",
    }
    specification = directory / "freshness_sensitivity_registration.json"
    if specification.exists():
        spec = json.loads(specification.read_text(encoding="utf-8"))
        if specification.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            raise ValueError("Freshness diagnostic must be registered before test scoring")
        eligible = ((frame.decision_ms - frame.price_changed_ms).le(spec["max_runner_ltp_age_seconds"] * 1000)
                    & (frame.decision_ms - frame.observed_ms).le(spec["max_market_observation_age_seconds"] * 1000))
        mask = eligible.groupby(frame.race_id).transform("all").to_numpy()
        subset = frame.loc[mask]
        if subset.race_id.nunique() >= spec["minimum_races_for_reporting"]:
            filtered = {name: p[mask] for name, p in forecasts.items()}
            report["freshness_diagnostic"] = {
                "specification": spec, **score_forecasts(subset, filtered, seed=args.seed),
                "parity_evidence": parity_evidence(subset, filtered, seed=args.seed),
                "segments": gap_segments(subset, filtered),
            }
    report["code_sha256"] = {p: digest(ROOT / p) for p in (
        "scripts/assess_forecast_parity.py", "scripts/benchmark_forecasts.py", "src/forecast_benchmark.py",
        "src/wf_eval.py", "src/forecast_evaluation.py")}
    (directory / "parity_assessment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "freshness_diagnostic"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
