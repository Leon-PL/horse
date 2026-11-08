"""Verify the saved performance experiment without fitting or selecting anything."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_selection import race_groups
from src.forecast_evaluation import probability_metrics
from src.ingestion.storage import atomic_json
from src.modelling.evaluation import code_signature
from src.modelling.performance import win_probabilities


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    root = Path(args.experiment)
    registered = json.loads((root / "registration.json").read_text())
    report = json.loads((root / "report.json").read_text())
    receipt = json.loads((root / "input_receipt.json").read_text())
    reference = Path(registered["design"]["reference"])
    if (code_signature() != registered["code_signature"]
            or sha(root / "input.parquet") != registered["input_sha256"]
            or registered["input_sha256"] != receipt["input_sha256"]
            or sha(reference / "oof.parquet") != registered["design"]["reference_oof_sha256"]):
        raise ValueError("Experiment source/input/reference changed")
    model = joblib.load(root / "performance_model.joblib")
    if model.spec.to_dict() != registered["spec"] or model.lineage != report["lineage"]:
        raise ValueError("Saved component differs from its registered recipe/lineage")
    columns = list(dict.fromkeys(["race_id", "horse_id", "horse_name", "race_date", "num_runners", "won"] + list(model.spec.features)))
    frame = pd.read_parquet(root / "input.parquet", columns=columns)
    first, last = registered["design"]["outer_test_start"], registered["design"]["outer_test_end"]
    frame = frame.loc[frame.race_date.between(first, last)].reset_index(drop=True)
    expected = pd.read_parquet(root / "performance_oof.parquet")
    keys = ["race_id", "horse_id"]
    if not np.array_equal(frame[keys].astype(str).to_numpy(), expected[keys].astype(str).to_numpy()):
        raise ValueError("Saved forecast runner alignment changed")
    details = model.predict_details(frame.drop(columns="won"))
    np.testing.assert_allclose(details.probability, expected.probability, rtol=0, atol=1e-12)
    groups = race_groups(frame.race_id)
    np.testing.assert_allclose(details.probability.groupby(frame.race_id).sum(), 1., rtol=0, atol=1e-12)
    refined = win_probabilities(details.mu, details.sigma, groups, points=2 * model.spec.quadrature_points)
    quadrature_error = float(np.max(np.abs(details.probability - refined)))
    if quadrature_error > 1e-6:
        raise ValueError(f"Quadrature convergence needs review: maximum probability change {quadrature_error}")
    predicted = pd.read_parquet(root / "predictions.parquet")
    later = frame.race_date.ge(registered["design"]["pool_score_from"])
    if not np.array_equal(frame.loc[later, keys].astype(str).to_numpy(), predicted[keys].astype(str).to_numpy()):
        raise ValueError("Pool evaluation field alignment changed")
    np.testing.assert_allclose(details.probability[later], predicted._component_1, rtol=0, atol=1e-12)
    scored_groups = race_groups(predicted.race_id)
    pooled = probability_metrics(predicted.pool_probability, predicted.won, scored_groups)[0]
    for name, value in pooled.items():
        np.testing.assert_allclose(value, report["pool"][name], rtol=0, atol=1e-12)
    standalone = probability_metrics(predicted._component_1, predicted.won, scored_groups)[0]
    result = {"verified": True, "code_signature": registered["code_signature"],
        "reloaded_max_probability_error": float(np.max(np.abs(details.probability - expected.probability))),
        "quadrature_max_probability_change": quadrature_error, "refined_points": 2 * model.spec.quadrature_points,
        "full_quarter_races": int(frame.race_id.nunique()), "scored_races": int(predicted.race_id.nunique()),
        "scored_dates": int(predicted.race_date.nunique()), "performance_only_scored_period": standalone,
        "artifact_sha256": {name: sha(root / name) for name in (
            "input.parquet", "performance_model.joblib", "performance_oof.parquet", "predictions.parquet", "races.parquet", "report.json")}}
    atomic_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
