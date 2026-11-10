"""Check the full 063 recipe at native capacity against one completed control fold."""
import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from research.meta_learner import features_063
from research.meta_learner.first_ensemble import (
    KEYS, difference, expert_forecasts, folds_for, get_selected, immutable_json, metrics, read_input, read_json, sha,
)
from src.ingestion.storage import atomic_json, writer_lock
from src.modelling.evaluation import code_signature
from src.modelling.spec import ModelSpec
from src.modelling.win import WinModel


def compare(root, quarter):
    design = read_json(root / "registration.json")
    native = read_json(root / "native_registration.json")
    model_code = code_signature()
    if (model_code != design["model_code"] or model_code != native["model_code"]
            or sha(root / "input.parquet") != design["input_sha256"]
            or native["input_sha256"] != design["input_sha256"]
            or sha(Path(features_063.__file__)) != design["feature_reconstruction_sha256"]
            or sha(root / "screen.json") != native["screen_sha256"] or quarter not in native["folds"]):
        raise ValueError("The registered input, code or fold changed")
    baseline, challenger = ModelSpec.from_dict(native["specs"][0]), get_selected(root, design)
    if baseline.to_dict() != design["specs"][0] or challenger.to_dict() != native["specs"][1]:
        raise ValueError("Pilot recipes differ from the original registration")
    threads = native["threads"]
    baseline_path = root / f"native-{baseline.id}_{quarter}.parquet"
    receipt = read_json(baseline_path.with_suffix(".json"))
    expected = {"dataset": design["receipt"]["dataset_manifest"]["id"], "input": design["input_sha256"],
                "recipe": {"spec": baseline.to_dict(), "capacity": "native", "threads": threads,
                           "driver_sha256": native["driver_sha256"]},
                "code": model_code, "fold": quarter, "purge": baseline.purge_days}
    if receipt["identity"] != expected or sha(baseline_path) != receipt["sha256"]:
        raise ValueError("Saved baseline forecasts fail their receipt")
    reference = pd.read_parquet(baseline_path)
    frame = read_input(root, design, challenger.features)
    fold, = folds_for(frame, (quarter,), challenger.purge_days)
    target = frame.loc[fold.test].reset_index(drop=True)
    boundary = target.race_date.min() - pd.Timedelta(days=baseline.purge_days)
    if (not reference[KEYS].equals(target[KEYS]) or not reference.available.all()
            or pd.Timestamp(receipt["lineage"]["calibration_last_date"]) > boundary
            or pd.Timestamp(receipt["lineage"]["training_last_date"]) > boundary):
        raise ValueError("Saved baseline does not cover the pilot fold chronologically")

    output = root / f"native_quick_{quarter}"
    driver = sha(Path(__file__))
    recipe = {"spec": challenger.to_dict(), "capacity": "native", "threads": threads, "driver_sha256": driver}
    immutable_json(output / "registration.json", {"baseline": expected, "baseline_sha256": receipt["sha256"],
        "challenger": recipe, "selection_rule": "Advance the single 063 recipe if the date-paired interval is below zero."})

    def fit_predict(train, test):
        start = perf_counter()
        model = WinModel(challenger).fit(train, dataset_start=design["receipt"]["dataset_start"], capacity="native", threads=threads)
        fit_seconds = perf_counter() - start
        inputs = test.drop(columns=["won", "finish_position", "odds", "bsp"])
        start = perf_counter()
        prediction = model.predict(inputs)
        return prediction, {**model.lineage, "fit_seconds": fit_seconds, "predict_seconds": perf_counter() - start}

    name = f"native-{challenger.id}"
    prediction = expert_forecasts(output, design, frame, name, recipe, (quarter,), challenger.purge_days, fit_predict)
    fit = read_json(output / f"{name}_{quarter}.json")["lineage"]
    if (code_signature() != model_code or sha(Path(__file__)) != driver
            or pd.Timestamp(fit["calibration_last_date"]) > boundary):
        raise ValueError("Pilot implementation or calibration boundary changed")
    before, baseline_loss = metrics(frame, reference, (quarter,))
    after, challenger_loss = metrics(frame, prediction, (quarter,))
    change = difference(challenger_loss, baseline_loss)
    timings = ("fit_seconds", "predict_seconds", "selected_rounds", "temperature")
    report = {"quarter": quarter, "scope": "Single-quarter retrospective native-capacity check",
              "baseline": before, "challenger": after, "change": change,
              "baseline_fit": {k: receipt["lineage"][k] for k in timings},
              "challenger_fit": {k: fit[k] for k in timings},
              "baseline_sha256": receipt["sha256"], "challenger_sha256": sha(output / f"{name}_oof.parquet")}
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--quarter", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    with writer_lock(args.study / ".benchmark.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another phase owns this benchmark directory")
        compare(args.study, args.quarter)


if __name__ == "__main__":
    main()
