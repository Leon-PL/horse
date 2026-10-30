"""Describe remaining losses using frozen022 forecasts; never fit or select a model.

Cohorts contain complete races, including all runners, to preserve the meaning
of race log loss. Historical prices retain their original benchmark scope.
These already inspected development periods cannot become a fresh holdout.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_forecasts import score_forecasts
from src.deployment_validation import file_sha256
from src.forecast_evaluation import valid_race_mask

SOURCE = ROOT / "data/audit/model_improvement/022_full_capacity"
REFERENCE = ROOT / "data/audit/model_improvement/016_market_teacher"
OUTPUT = ROOT / "data/audit/model_improvement/024_gap_diagnostics"


def run():
    files = [SOURCE / f"{period}_predictions.parquet" for period in ("2025Q4", "2026Q1")]
    files += [SOURCE / "report.json", REFERENCE / "market_context_predictions.parquet"]
    original = pd.concat([pd.read_parquet(path) for path in files[:2]], ignore_index=True)
    market = pd.read_parquet(files[-1])
    keys = ["race_id", "horse_name", "race_date", "won"]
    price = market.loc[market.race_id.isin(original.race_id), keys+["market_probability"]]
    frame = original.merge(price, on=keys, how="inner", validate="one_to_one")
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    if not valid_race_mask(frame).all() or frame.market_probability.isna().any():
        raise ValueError("Use only unchanged complete paired fields")
    expected = json.loads((SOURCE / "report.json").read_text())["market"]
    if (frame.race_id.nunique(), len(frame)) != (expected["races"], expected["runners"]):
        raise ValueError("Paired population differs from completed022 comparison")
    summaries = {}
    field = frame.groupby("race_id", sort=False)
    constant_columns = ["race_date", "race_type", "handicap", "num_runners"]
    if field[constant_columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Race-level cohort labels disagree within a field")
    races = field[constant_columns].first()
    races["local_debutants"] = frame.horse_prev_races.eq(0).groupby(frame.race_id).sum()
    races["small_history_runners"] = frame.horse_prev_races.lt(3).groupby(frame.race_id).sum()
    races["atr_covered"] = frame.has_atr_history.gt(0).groupby(frame.race_id).sum()
    races["field_size"] = field.size()
    races["field_bin"] = pd.cut(races.field_size, [0, 7, 12, np.inf], labels=["2-7", "8-12", "13+"])
    races["any_local_debutant"] = races.local_debutants.gt(0)
    races["sparse_history_majority"] = races.small_history_runners.ge(races.field_size/2)
    races["any_atr_history"] = races.atr_covered.gt(0)
    races["period"] = pd.to_datetime(races.race_date).dt.to_period("Q").astype(str)
    scores = {name: frame[f"{name}_probability"].to_numpy() for name in ("baseline", "candidate", "market")}
    overall = score_forecasts(frame, scores)
    for name in scores:
        for metric in ("race_log_loss", "brier", "top1_accuracy"):
            np.testing.assert_allclose(overall["metrics"][name][metric], expected["metrics"][name][metric], rtol=0, atol=1e-12)
    for column in ("race_type", "handicap", "field_bin", "any_local_debutant",
                   "sparse_history_majority", "any_atr_history", "period"):
        groups = []
        for label, rows in races.groupby(column, sort=False, observed=True):
            selected = frame.loc[frame.race_id.isin(rows.index)].copy()
            values = {name: selected[f"{name}_probability"].to_numpy() for name in scores}
            result = score_forecasts(selected, values)
            current, old, reference = (result["metrics"][name] for name in ("candidate", "baseline", "market"))
            groups.append({"label": str(label), "races": result["races"], "runners": result["runners"],
                "metrics": result["metrics"],
                "candidate_minus_baseline_log_loss": current["race_log_loss"]-old["race_log_loss"],
                "candidate_minus_market_log_loss": current["race_log_loss"]-reference["race_log_loss"],
                "contribution_to_overall_gap": (current["race_log_loss"]-reference["race_log_loss"])*len(rows)/len(races)})
        summaries[column] = groups
    winners = frame.loc[frame.won.eq(1)].copy()
    winners["candidate_minus_market_winner_loss"] = np.log(winners.market_probability/winners.candidate_probability)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    winners.sort_values("candidate_minus_market_winner_loss", ascending=False).to_csv(OUTPUT / "winner_gap_examples.csv", index=False)
    report = {"scope": "Descriptive attribution on already inspected development data; no candidate selection, test, fit or prospective claim",
        "market_scope": "Unchanged016 historical benchmark, not newly observed pre-off exchange quotes",
        "input_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in files},
        "source_sha256": file_sha256(Path(__file__)), "overall": overall["metrics"], "cohorts": summaries,
        "foreign_history_caveat": "Zero local history is not proof of a debut or of French origin; known overseas starts remain a distinct information gap",
        "model_or_features_changed": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"races": len(races), "runners": len(frame), "cohorts": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    run()
