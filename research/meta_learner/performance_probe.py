"""Registered retrospective reproduction of the 058 performance/spread head.

Preparation uses the supported accepted-data loader. The completed, immutable
generalist benchmark is a reference measurement, not a current-code fitting
receipt. No historical fingerprint or cached prediction is restamped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.modelling.data import load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root, benchmark = Path(args.output), Path(args.benchmark)
    root.mkdir(parents=True, exist_ok=True)
    registered = json.loads((benchmark / "registration.json").read_text())
    report = json.loads((benchmark / "report.json").read_text())
    oof_path = benchmark / "oof.parquet"
    oof_sha = hashlib.sha256(oof_path.read_bytes()).hexdigest()
    if oof_sha != report["oof_sha256"] or len(registered["specs"]) != 1:
        raise ValueError("Expected one completed, intact generalist reference")
    base = pd.read_parquet(oof_path)
    if args.command == "prepare":
        design = {
            "reference": str(benchmark.resolve()), "reference_oof_sha256": oof_sha,
            "reference_code_signature": registered["code_signature"],
            "experiment": "058 initial native reproduction with strictly chronological residuals",
            "capacity": "native", "input_recipe": registered["specs"][0],
            "outer_test_start": str(pd.to_datetime(base.race_date).min().date()),
            "outer_test_end": str(pd.to_datetime(base.race_date).max().date()),
            "pool_train_through": "2026-04-30", "pool_score_from": "2026-05-08",
            "purge_days": 7, "pool_shrinkage": .01, "residual_splits": 4,
            "margin_cap_lb": 30., "lb_per_length_furlongs": 15., "spread_epsilon": .25,
            "sigma_floor": 1., "quadrature_points": 400, "calibration_fraction": .2,
            "choices": "Fixed legacy numerical constants; no tuning on the scored period",
            "interpretation": "Retrospective single-quarter pilot; no adoption or independent confirmation",
            "later_evidence": "063 improved generalist reduced the earlier spread benefit; not yet reproduced here",
        }
        path = root / "pre_registration.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if {k: v for k, v in previous.items() if k != "created_at"} != design:
                raise ValueError("Output belongs to a different experiment")
        else:
            atomic_json(path, {**design, "created_at": datetime.now(timezone.utc).isoformat()})
        with writer_lock(root / ".prepare.lock") as acquired:
            if not acquired:
                raise RuntimeError("Performance preparation already running")
            frame, receipt = load_dataset(args.store_root, features=registered["specs"][0]["features"],
                end=design["outer_test_end"], label_columns=("beaten_lengths", "distance_furlongs"))
            old = registered["receipt"]
            for name in ("fe_signature", "data_signature", "store_last_date"):
                if receipt[name] != old[name]:
                    raise ValueError("Prepared data differs from the reference store generation")
            if receipt["dataset_manifest"]["id"] != old["dataset_manifest"]["id"]:
                raise ValueError("Prepared data differs from the reference accepted dataset")
            atomic_parquet(frame, root / "input.parquet")
            atomic_json(root / "input_receipt.json", {"receipt": receipt,
                "input_sha256": hashlib.sha256((root / "input.parquet").read_bytes()).hexdigest(),
                "rows": len(frame), "reference_oof_sha256": oof_sha})
            print(json.dumps({"rows": len(frame), "prepared": str(root.resolve())}), flush=True)
        return
    run(args, root, registered, base, oof_sha)


def run(args, root, registered, base, oof_sha):
    # Imported only after the explicitly registered data preparation is complete.
    from src.modelling.performance import PerformanceModel, PerformanceSpec
    from src.modelling.evaluation import code_signature, nested_pool_evaluation
    from src.wf_eval import Fold, oof_predict_expanding
    from src.feature_selection import race_groups
    from src.forecast_evaluation import probability_metrics
    import numpy as np
    import time
    import joblib
    import psutil
    from threading import Event, Thread

    design = json.loads((root / "pre_registration.json").read_text())
    receipt = json.loads((root / "input_receipt.json").read_text())
    if (design["reference_oof_sha256"] != oof_sha or receipt["reference_oof_sha256"] != oof_sha
            or receipt["input_sha256"] != hashlib.sha256((root / "input.parquet").read_bytes()).hexdigest()):
        raise ValueError("Prepared/reference data changed")
    frame = pd.read_parquet(root / "input.parquet")
    settings = registered["specs"][0]
    controls = {k: v for k, v in settings.items() if k not in {"objective", "soft_label_blend", "stage_weights"}}
    constants = {name: design[name] for name in ("residual_splits", "margin_cap_lb", "lb_per_length_furlongs",
        "spread_epsilon", "sigma_floor", "quadrature_points", "calibration_fraction")}
    spec = PerformanceSpec.from_dict({**controls, **constants, "id": "performance-spread"})
    snapshot = Path(args.benchmark) / "source_snapshot"
    source_manifest = json.loads((snapshot / "manifest.json").read_text())
    if source_manifest["benchmark_code_signature"] != registered["code_signature"]:
        raise ValueError("Reference source snapshot differs from its measurement")
    for name, expected in source_manifest["files"].items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Reference source snapshot is not intact")
    first, last = pd.Timestamp(design["outer_test_start"]), pd.Timestamp(design["outer_test_end"])
    fold = Fold("2026Q2", frame.race_date.le(first - pd.Timedelta(days=spec.purge_days)).to_numpy(),
                frame.race_date.between(first, last).to_numpy(), 0)
    registration = {"design": design, "spec": spec.to_dict(), "code_signature": code_signature(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": receipt["input_sha256"], "threads": args.threads,
        "reference_source_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()}
    with writer_lock(root / ".run.lock") as acquired:
        if not acquired:
            raise RuntimeError("Performance probe is already running")
        path = root / "registration.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("Run output belongs to a different registered source/recipe")
        atomic_json(path, registration)
        atomic_json(root / "performance_spec.json", spec.to_dict())
        started = time.monotonic()
        rss = [psutil.Process().memory_info().rss]
        stopped = Event()
        def measure_memory():
            while not stopped.wait(1.):
                rss[0] = max(rss[0], psutil.Process().memory_info().rss)
        Thread(target=measure_memory, daemon=True).start()
        fitted = []
        def predict(train, test):
            model = PerformanceModel(spec).fit(train, dataset_start=receipt["receipt"]["dataset_start"],
                                               capacity=design["capacity"], threads=args.threads)
            fitted.append(model)
            return model.predict(test.drop(columns=["won", "finish_position", "beaten_lengths"]))
        p = oof_predict_expanding(frame, folds=[fold], fit_predict=predict, min_train_rows=100,
                                  embargo_days=spec.purge_days)
        test = frame.loc[fold.test].reset_index(drop=True)
        keys = ["race_id", "horse_id"]
        if not np.array_equal(test[keys].astype(str).to_numpy(), base[keys].astype(str).to_numpy()):
            raise ValueError("Reference and reproduction do not score identical ordered fields")
        component = test[keys + ["race_date"]].copy()
        component["component"], component["probability"], component["available"] = spec.id, p[fold.test], True
        component["training_last_date"] = fitted[0].lineage["training_last_date"]
        component["calibration_last_date"] = fitted[0].lineage["calibration_last_date"]
        component["recipe_signature"] = spec.signature
        records = pd.concat([base, component], ignore_index=True)
        pool_fold = Fold("later-quarter-performance-probe",
            test.race_date.le(pd.Timestamp(design["pool_train_through"])).to_numpy(),
            test.race_date.ge(pd.Timestamp(design["pool_score_from"])).to_numpy(), 0)
        report, predicted, races = nested_pool_evaluation(test, records, (settings["id"], spec.id),
            folds=[pool_fold], purge_days=spec.purge_days, shrinkage=design["pool_shrinkage"])
        report["performance_standalone"] = probability_metrics(p[fold.test], test.won, race_groups(test.race_id))[0]
        report["lineage"] = fitted[0].lineage
        report["registration"] = registration
        report["elapsed_seconds"] = time.monotonic() - started
        stopped.set()
        report["sampled_peak_fit_rss_gb"] = rss[0] / 1024 ** 3
        report["serving_qualified"] = False
        if (code_signature() != registration["code_signature"]
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != registration["probe_sha256"]):
            raise RuntimeError("Modelling code changed during the experiment")
        atomic_parquet(component, root / "performance_oof.parquet")
        atomic_parquet(predicted, root / "predictions.parquet")
        atomic_parquet(races, root / "races.parquet")
        joblib.dump(fitted[0], root / "performance_model.joblib")
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
