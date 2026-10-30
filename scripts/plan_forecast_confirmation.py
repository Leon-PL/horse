"""Freeze assumptions and estimate confirmation feasibility from archived forecasts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.confirmation_planning import race_differences, variance_estimates, planning_scenario
from src.deployment_validation import file_sha256
from src.forward_store import ForwardStudy, publish_json, utcnow

SOURCES = ["src/confirmation_planning.py", "scripts/plan_forecast_confirmation.py",
           "src/forecast_evaluation.py", "src/utils.py", "src/deployment_validation.py", "src/forward_store.py"]
MARGINS = {"race_log_loss": .002, "runner_brier": .0001}


def run(output, study_path):
    study = ForwardStudy(study_path)
    study.verify()
    root = ROOT / "data/audit/market_performance"
    inputs = {
        "independent_recipe": {"path": root / "003_race_objective/market_predictions.parquet",
                               "column": "baseline_probability", "market": "market_probability",
                               "scope": "2025Q3/2025Q4/2026Q1 inspected development recipe; not current-refit performance"},
        "unpromoted_market_correction": {"path": root / "001_t5_2026Q3/test_predictions.parquet",
                               "column": "probability_market_combination", "market": "probability_market",
                               "scope": "Inspected2026Q3 closed correction; variance illustration only, no advancement"},
    }
    for item in inputs.values():
        item["sha256"] = file_sha256(item["path"])
        item["path"] = str(item["path"].relative_to(ROOT).as_posix())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": utcnow().isoformat(), "purpose": "Variance and feasibility planning; no formal confirmation activation",
        "study_protocol_sha256": study.protocol_sha256, "study": str(study.path.relative_to(ROOT).as_posix()),
        "inputs": inputs, "source_sha256": {name: file_sha256(ROOT/name) for name in SOURCES},
        "margins": MARGINS, "alpha_one_sided": .05, "joint_power_target": .8, "conjunctive_gates": 4,
        "normal_approximation": "N=(z(.95)+z(.95))^2*variance/(margin-assumed_difference)^2; 95% marginal power gives >=80% joint power by union bound under the approximation",
        "slices": "all fields and one frozen freshness rule, each pooled and by calendar quarter; maximum required N across all",
        "freshness": {"runner_price_age_seconds": 300, "market_observation_age_seconds": 90},
        "scenarios": {"exact_parity_mean": {"race_log_loss": 0., "runner_brier": 0.},
                      "half_margin_mean": {"race_log_loss": .001, "runner_brier": .00005},
                      "modest_advantage_mean": {"race_log_loss": -.005, "runner_brier": -.0002}},
        "calendar_sensitivity": {"eligible_races_per_day": [10, 25, 40], "minimum_calendar_days": 84,
                                 "note": "Illustrative throughput, not measured Matchbook capture coverage"},
        "transfer_limit": "Historical Betfair BASIC LTP and older fits differ from the live Matchbook best-back population; forward variance and coverage need a pilot",
        "runtime": {"python": sys.version, "packages": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow")}},
    }
    publish_json(output / "manifest.json", manifest)  # Freeze before reading scores.
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT/name, name)
    reports = {}
    for name, spec in inputs.items():
        path = ROOT/spec["path"]
        columns = ["race_id", "race_date", "horse_name", "won", "finish_position", "num_runners",
                   "decision_ms", "observed_ms", "price_changed_ms", spec["column"], spec["market"]]
        frame = pd.read_parquet(path, columns=columns)
        frame.race_date = pd.to_datetime(frame.race_date).dt.normalize()
        # Validate the full archived field before applying any quality subset.
        race_differences(frame, spec["column"], spec["market"])
        price_age = frame.decision_ms-frame.price_changed_ms
        obs_age = frame.decision_ms-frame.observed_ms
        fresh = (price_age.between(0, 300_000) & obs_age.between(0, 90_000)
                 & frame.price_changed_ms.le(frame.observed_ms)).groupby(frame.race_id).transform("all")
        summaries = {}
        for quality, selected in (("all", frame), ("fresh", frame.loc[fresh])):
            blocks = [("pooled", selected), *[(str(q), block) for q, block in selected.groupby(selected.race_date.dt.to_period("Q"), sort=True)]]
            for period, block in blocks:
                if block.race_id.nunique() < 50 or block.race_date.nunique() < 10:
                    raise ValueError("Too few archived fields/dates for the frozen variance slice")
                losses = race_differences(block, spec["column"], spec["market"])
                summaries[f"{quality}:{period}"] = variance_estimates(losses)
                losses.to_parquet(output / f"{name}_{quality}_{period}.parquet", index=False)
        scenarios = {label: planning_scenario(summaries, margins=MARGINS, assumed_differences=assumed)
                     for label, assumed in manifest["scenarios"].items()}
        observed = summaries["all:pooled"]["observed_mean_candidate_minus_market"]
        scenarios["observed_historical_mean"] = planning_scenario(summaries, margins=MARGINS, assumed_differences=observed)
        for scenario in scenarios.values():
            n = scenario["required_races"]
            scenario["illustrative_calendar_days"] = {str(rate): max(84, math.ceil(n/rate)) if n is not None else None
                for rate in manifest["calendar_sensitivity"]["eligible_races_per_day"]}
        reports[name] = {"scope": spec["scope"], "variance_slices": summaries, "scenarios": scenarios}
        if file_sha256(path) != spec["sha256"]:
            raise ValueError("Archived input changed during planning")
    study.verify()
    if manifest["source_sha256"] != {name: file_sha256(ROOT/name) for name in SOURCES}:
        raise ValueError("Planning implementation changed during calculation")
    report = {"manifest_sha256": file_sha256(output / "manifest.json"), "confirmation_registered": False,
              "candidate_promoted": False, "identity_market_baseline": "M0 equals its reference by construction; statistical testing cannot establish independent skill",
              "comparisons": reports}
    publish_json(output / "report.json", report)
    print(json.dumps({name: {s: {"races": v["required_races"], "days_at_25": v["illustrative_calendar_days"]["25"]}
                            for s, v in result["scenarios"].items()} for name, result in reports.items()}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.study)
