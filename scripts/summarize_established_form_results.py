"""Describe the fixed029 comparison on experienced fields without fitting.

Secondary subgroup/error diagnostics, never a feature search or promotion gate.
All groups and both model arms are retained, including unfavourable results.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_forecasts import score_forecasts
from scripts.benchmark_independent_features import score_pair
from src.deployment_validation import file_sha256
from src.forecast_evaluation import valid_race_mask

OUTPUT = ROOT / "data/audit/established_form_029/native_canonical"


def run():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    report = json.loads((OUTPUT / "report.json").read_text())
    if report["manifest_sha256"] != file_sha256(OUTPUT / "manifest.json"):
        raise ValueError("Comparison registration changed")
    frame = pd.read_parquet(OUTPUT / "market_paired_predictions.parquet")
    if not valid_race_mask(frame).all():
        raise ValueError("Describe complete single-winner fields only")
    field = frame.groupby("race_id", sort=False)
    established = field.horse_prev_races.min().ge(5)
    handicap = field.handicap.first().eq(1)
    context = pd.read_parquet(OUTPUT / "inputs.parquet", columns=["race_id", "race_is_bumper"])
    bumper = context.groupby("race_id", sort=False).race_is_bumper.first().reindex(established.index).eq(1)
    discipline = field.race_type.first().where(~bumper, "Bumper")
    selections = {
        "all_at_least_5_starts": established,
        "established_handicaps": established & handicap,
        "established_conditions": established & ~handicap,
        **{f"established_{code.lower()}": established & discipline.eq(code)
           for code in ("Flat", "Bumper", "Hurdle", "Chase")},
    }
    scores = {}
    winners = []
    for label, mask in selections.items():
        selected = frame.loc[frame.race_id.isin(mask.index[mask])].copy()
        if selected.empty:
            scores[label] = {"races": 0}
            continue
        scores[label] = {
            "vs_market": score_forecasts(selected, {name: selected[f"{name}_probability"].to_numpy()
                for name in ("baseline", "candidate", "market")}),
            "candidate_vs_baseline": score_pair(selected),
        }
        if label == "all_at_least_5_starts":
            rows = selected.loc[selected.won.eq(1)].copy()
            for name in ("baseline", "candidate"):
                rows[f"{name}_winner_loss_vs_market"] = np.log(
                    rows.market_probability.clip(1e-15) / rows[f"{name}_probability"].clip(1e-15))
            rows["candidate_minus_baseline_winner_loss"] = np.log(
                rows.baseline_probability.clip(1e-15) / rows.candidate_probability.clip(1e-15))
            winners.append(rows)
    if winners:
        pd.concat(winners).to_csv(OUTPUT / "established_winner_errors.csv", index=False)
    result = {
        "scope": "Secondary descriptive diagnostics on the same inspected Q1 development fields; no additional fitting or selection search",
        "definition": "Every runner in each field has at least five prior local starts; handicap/conditions and all four disciplines are all reported",
        "scores": scores,
        "input_sha256": {name: file_sha256(OUTPUT / name) for name in
            ("manifest.json", "report.json", "inputs.parquet", "market_paired_predictions.parquet")},
        "source_sha256": file_sha256(Path(__file__)),
    }
    (OUTPUT / "established_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {"races": value.get("vs_market", {}).get("races", 0),
        "metrics": value.get("vs_market", {}).get("metrics", {})} for name, value in scores.items()}, indent=2), flush=True)


if __name__ == "__main__":
    run()
