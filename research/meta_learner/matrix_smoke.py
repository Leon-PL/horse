"""Bounded real-data exercise of the explicit wide matrix path; not model selection."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import psutil

from src.feature_selection import race_groups
from src.forecast_evaluation import probability_metrics
from src.ingestion.storage import atomic_json
from src.modelling.evaluation import code_signature, frame_signature
from src.modelling.selection import matrix_signature
from src.modelling.spec import ModelSpec
from src.modelling.win import WinModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.prepared)
    description = json.loads((root / "prepared.json").read_text())
    context = pd.read_parquet(root / "context.parquet")
    matrix = np.load(root / "matrix.npy", mmap_mode="r")
    names = tuple(description["features"])
    if (frame_signature(context, context.columns) != description["context_signature"]
            or matrix_signature(matrix, names) != description["matrix_signature"]):
        raise ValueError("Prepared matrix/context content changed")
    dates = pd.to_datetime(context.race_date)
    earlier, later = dates.le("2026-04-30").to_numpy(), dates.ge("2026-05-08").to_numpy()
    spec = replace(ModelSpec("wide-matrix-smoke", names),
        screen_params=(("n_estimators", 20), ("learning_rate", .1)))
    source = code_signature()
    process, done = psutil.Process(), threading.Event()
    peak = [process.memory_info().rss]
    def sample():
        while not done.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    try:
        train = context.loc[earlier].reset_index(drop=True)
        x_train = np.ascontiguousarray(matrix[np.flatnonzero(earlier)])
        model = WinModel(spec).fit(train, matrix=x_train, matrix_columns=names,
            dataset_start=description["receipt"]["dataset_start"], capacity="screen", threads=2)
        del x_train
        test = context.loc[later].reset_index(drop=True)
        x_test = np.ascontiguousarray(matrix[np.flatnonzero(later)])
        p = model.predict(test, matrix=x_test, matrix_columns=names)
        metrics, _, _ = probability_metrics(p, test.won, race_groups(test.race_id))
    finally:
        done.set()
        observer.join()
    if code_signature() != source:
        raise RuntimeError("Modelling source changed during the smoke check")
    result = {"purpose": "Resource/routing check only; not a selected model or accuracy claim",
        "features": len(names), "training_runners": len(train), "scored_runners": len(test),
        "peak_rss_gb": peak[0] / 2**30, "spec": spec.to_dict(), "code_signature": source,
        "metrics": metrics, "dataset_id": description["receipt"]["dataset_manifest"]["id"]}
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "spec"}, indent=2))


if __name__ == "__main__":
    main()
