"""The predeclared text contribution test against the generalist/performance pool."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast_evaluation import paired_interval
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.evaluation import nested_pool_evaluation
from src.wf_eval import Fold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    root = Path(args.experiment)
    design = json.loads((root / "ensemble_pre_registration.json").read_text())
    selected = json.loads((root / "selection.json").read_text())["selected"]
    frame = pd.read_parquet(root / "corpus.parquet", columns=["race_id", "horse_id", "horse_name", "race_date", "num_runners", "won", "finish_position"])
    frame = frame.loc[frame.race_date.between("2026-04-01", "2026-06-03")].reset_index(drop=True)
    parts = []
    for name, reference in design["references"].items():
        if sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Ensemble reference forecasts changed")
        part = pd.read_parquet(reference["path"])
        part["race_id"] = part.race_id.astype(str)
        part["component"] = name
        part.attrs = {}
        parts.append(part)
    base = pd.concat(parts, ignore_index=True)
    fold = Fold("later-quarter-text", frame.race_date.le(design["pool_train_through"]).to_numpy(),
                frame.race_date.ge(design["pool_score_from"]).to_numpy(), 0)
    reports, losses, predictions = {}, {}, {}
    for name, recipe in [("generalist_performance", None), ("with_corrected_text", "corrected"), ("with_selected_text", selected)]:
        records = base
        components = ("generalist", "performance")
        if recipe is not None:
            text = pd.read_parquet(root / f"confirm_{recipe}.parquet")
            text = text.loc[text.quarter.eq("2026Q2"), ["race_id", "horse_id", "race_date", "probability"]].copy()
            text["race_id"] = text.race_id.astype(str)
            text["component"], text["available"] = "text", True
            records = pd.concat([base, text], ignore_index=True)
            components += ("text",)
        report, predicted, races = nested_pool_evaluation(frame, records, components, folds=[fold],
            purge_days=design["purge_days"], shrinkage=design["shrinkage"])
        reports[name], predictions[name] = report, predicted
        losses[name] = races.set_index("race_id")
        atomic_parquet(predicted, root / f"ensemble_{name}.parquet")
    baseline = losses["generalist_performance"]
    comparisons = {}
    for current, reference in (("with_corrected_text", "generalist_performance"),
                               ("with_selected_text", "generalist_performance"),
                               ("with_selected_text", "with_corrected_text")):
        a = losses[current].loc[baseline.index].race_log_loss.to_numpy()
        b = losses[reference].loc[baseline.index].race_log_loss.to_numpy()
        comparisons[current + "_minus_" + reference] = paired_interval(a - b, blocks=baseline.race_date)
    result = {"design": design, "selected": selected, "reports": reports, "comparisons": comparisons,
        "races": len(baseline), "evaluation_status": "Retrospective within-quarter ensemble probe; not independent confirmation"}
    for predicted in predictions.values():
        assert np.isfinite(predicted.pool_probability).all()
    atomic_json(root / "ensemble_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
