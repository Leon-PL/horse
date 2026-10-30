"""Explain saved028 cases with the saved026 historical learner, without fitting."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256
from src.model import RacePredictor


def explain_cases(directory, model_path, inputs_path, cases, context, probability_column):
    """Reuse a saved historical learner; explain requested runners only."""
    cases = cases.copy()
    context = context.copy()
    cases.race_id = cases.race_id.astype(str)
    context.race_id = context.race_id.astype(str)
    model = RacePredictor()
    model.load(str(model_path))
    cutoff = model.eval_split_info.get("as_of") or model.eval_split_info.get("training_last_date")
    if not cutoff or pd.Timestamp(cutoff) >= pd.to_datetime(cases.race_date).min():
        raise ValueError("Use a saved model fitted strictly before these cases")
    inputs = pd.read_parquet(inputs_path, filters=[("race_id", "in", cases.race_id.astype(int).unique().tolist())])
    inputs.race_id = inputs.race_id.astype(str)
    inputs = inputs.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    scored = model.predict_races(inputs, win_only=True)
    checked = context.loc[context.race_id.isin(cases.race_id)].merge(
        scored[["race_id", "horse_name", "win_probability"]], on=["race_id", "horse_name"], validate="one_to_one")
    np.testing.assert_allclose(checked[probability_column], checked.win_probability, rtol=0, atol=1e-12)
    requested = inputs.merge(cases[KEY].drop_duplicates(), on=KEY, validate="one_to_one")
    _, transformed = model._prepare_prediction_frame(requested)
    contributions = model.clf_model.booster_.predict(transformed, pred_contrib=True, num_threads=2)
    raw_scores = model.clf_model.booster_.predict(transformed, raw_score=True, num_threads=2)
    np.testing.assert_allclose(contributions.sum(axis=1), raw_scores, rtol=0, atol=1e-10)
    values = np.asarray(transformed)
    feature_names = model.feature_cols
    lookup = {(row.race_id, row.horse_name): i for i, row in enumerate(requested.itertuples())}
    explanations = []
    for case in cases.itertuples():
        i = lookup[(case.race_id, case.horse_name)]
        order = np.argsort(contributions[i, :-1])
        for direction, indices in [("decreases_raw_score", order[:8]), ("increases_raw_score", order[-8:][::-1])]:
            for rank, j in enumerate(indices, 1):
                explanations.append({"case_group": case.case_group, "race_id": case.race_id,
                    "horse_name": case.horse_name, "direction": direction, "rank": rank,
                    "feature": feature_names[j], "value": float(values[i, j]),
                    "raw_feature_value": float(requested.iloc[i][feature_names[j]]),
                    "raw_score_contribution": float(contributions[i, j]),
                    "base_score": float(contributions[i, -1]), "raw_score": float(raw_scores[i])})
    pd.DataFrame(explanations).to_csv(directory / "case_explanations.csv", index=False)
    importance = pd.DataFrame({"feature": feature_names,
        "gain": model.clf_model.booster_.feature_importance(importance_type="gain"),
        "splits": model.clf_model.booster_.feature_importance(importance_type="split")})
    importance.sort_values("gain", ascending=False).to_csv(directory / "historical_feature_importance.csv", index=False)
    report = {"model_sha256": file_sha256(model_path),
        "source_sha256": file_sha256(Path(__file__)), "cases": len(cases),
        "complete_fields_reproduced": checked.race_id.nunique(), "runners_reproduced": len(checked),
        "maximum_probability_reproduction_error": float((checked[probability_column]-checked.win_probability).abs().max()),
        "scope": "Additive contributions to the historical learner's raw score; descriptive, not causal effects or final calibrated probability contributions",
        "training_performed": False}
    (directory / "explanation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    selected = pd.DataFrame(explanations)
    for name in cases.loc[cases.case_group.isin(["missed_winners", "overrated_losers"])].groupby("case_group", sort=False).head(5).horse_name:
        print(name, flush=True)
        print(selected.loc[selected.horse_name.eq(name) & selected["rank"].le(3), ["direction", "feature", "value", "raw_score_contribution"]].to_string(index=False), flush=True)


KEY = ["race_id", "horse_name"]


def run(handicap_review=False):
    if handicap_review:
        directory = ROOT / "data/audit/handicap_weight_review_030"
        source = ROOT / "data/audit/established_form_029/native_canonical"
        context = pd.read_parquet(directory / "cohort_context.parquet")
        winners = context.loc[context.won.eq(1)].sort_values("winner_loss_vs_market", ascending=False).head(8).copy()
        losers = context.loc[context.won.eq(0)].sort_values("runner_brier_gap", ascending=False).head(8).copy()
        winners["case_group"], losers["case_group"] = "missed_winners", "overrated_losers"
        cases = pd.concat([winners, losers], ignore_index=True)
        cases.to_csv(directory / "selected_cases.csv", index=False)
        explain_cases(directory, source / "baseline.joblib", source / "inputs.parquet", cases, context, "baseline_probability")
    else:
        directory = ROOT / "data/audit/established_horse_review_028"
        source = ROOT / "data/audit/canonical_repairs_026"
        explain_cases(directory, source / "historical_model.joblib", source / "corrected_inputs.parquet",
            pd.read_csv(directory / "selected_cases.csv"), pd.read_parquet(directory / "paired_context.parquet"), "candidate_probability")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handicap-review", action="store_true", help="Explain the saved029 established-handicap errors without fitting")
    run(parser.parse_args().handicap_review)
