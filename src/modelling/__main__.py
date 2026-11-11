"""Supported modelling operations: benchmark, prepare/select features, and fit candidates."""
import argparse
from datetime import datetime
import json
import hashlib
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

import config
from src.feature_selection import race_groups
from src.forecast_evaluation import probability_metrics
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.modelling.data import load_dataset
from src.modelling.evaluation import component_oof, nested_pool_evaluation, code_signature
from src.modelling.spec import ModelSpec, read_features
from src.modelling.components import spec_from_dict, training_columns
from src.wf_eval import expanding_period_folds


def _specs(args):
    if args.spec:
        specs = [spec_from_dict(json.loads(Path(p).read_text(encoding="utf-8"))) for p in args.spec]
        if len({s.id for s in specs}) != len(specs):
            raise ValueError("Model IDs must be unique")
        return specs
    return [ModelSpec("generalist", read_features(args.features or config.FEATURE_ALLOWLIST))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="chronological OOF benchmark; never promotes")
    prepare = commands.add_parser("prepare", help="prepare a projected, bounded-memory selection matrix")
    selection = commands.add_parser("select", help="select on earlier folds; confirm on separate later folds")
    admit = commands.add_parser("admit", help="forward test: would adding named inputs to a recipe help?")
    fit = commands.add_parser("fit", help="fit a win-only candidate from a benchmark's locked recipes/OOF")
    for command in (benchmark, prepare, fit):
        command.add_argument("--store-root", default=None)
    for command in (benchmark, prepare):
        command.add_argument("--start", default=None)
        command.add_argument("--end", default=None)
    for command in (benchmark, selection, admit):
        command.add_argument("--spec", action="append", help="resolved ModelSpec JSON; repeat for several win experts")
        command.add_argument("--features", default=None, help="feature-list file when creating a default recipe")
        command.add_argument("--capacity", choices=("native", "screen"), default="native")
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--min-train-periods", type=int, default=8)
    for command in (benchmark, prepare, selection, admit):
        command.add_argument("--output", required=True, help="new output directory, or the same registration for a resumed run")
    benchmark.add_argument("--fold-start", default=None)
    benchmark.add_argument("--pool", action="store_true", help="evaluate a pool fitted on earlier OOF quarters")
    selection.add_argument("--prepared", required=True)
    selection.add_argument("--selection-end", required=True)
    selection.add_argument("--confirmation-start", required=True)
    selection.add_argument("--min-features", type=int, default=30)
    selection.add_argument("--near-copy", type=float, default=0.95,
        help="keep one of each cluster of columns at or above this absolute rank correlation; 0 keeps all")
    selection.add_argument("--importance", choices=("auto", "shap", "permutation"), default="auto",
        help="how inputs are ranked for elimination; auto uses SHAP share for the long tail, loss prevented near the choice")
    selection.add_argument("--permutation-below", type=int, default=150)
    selection.add_argument("--confirm-capacity", choices=("native", "screen"), default=None,
        help="also confirm the baseline and selected lists at this capacity")
    admit.add_argument("--prepared", required=True)
    admit.add_argument("--candidates", nargs="+", required=True,
        help="input names, or one text file with a name per line")
    admit.add_argument("--from", dest="start", required=True, help="first test date of the folds to score")
    fit.add_argument("--benchmark", required=True)
    fit.add_argument("--as-of", required=True)
    fit.add_argument("--threads", type=int, default=2)
    fit.add_argument("--name", default="win modelling candidate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if getattr(args, "threads", 1) < 1:
        parser.error("--threads must be positive")
    if args.command == "fit":
        from src.modelling.candidate import fit_candidate, save_candidate
        cutoff = pd.Timestamp(args.as_of)
        if pd.isna(cutoff) or cutoff != cutoff.normalize() or cutoff >= pd.Timestamp.now().normalize():
            parser.error("--as-of must be a settled date before today")
        folder = Path(args.benchmark)
        registered = json.loads((folder / "registration.json").read_text())
        if registered["code_signature"] != code_signature():
            raise ValueError("Benchmark used different modelling code; rerun it before fitting")
        specs = [spec_from_dict(s) for s in registered["specs"]]
        report = json.loads((folder / "report.json").read_text())
        if report.get("oof_sha256") != hashlib.sha256((folder / "oof.parquet").read_bytes()).hexdigest():
            raise ValueError("Benchmark OOF content differs from its completed report")
        started = time.monotonic()
        frame, receipt = load_dataset(args.store_root, features=list(dict.fromkeys(n for s in specs for n in s.features)),
            development=False, end=args.as_of, label_columns=training_columns(specs))
        if receipt["dataset_manifest"]["id"] != registered["receipt"]["dataset_manifest"]["id"]:
            raise ValueError("Candidate data differs from the benchmark data")
        candidate = fit_candidate(frame, specs, pd.read_parquet(folder / "oof.parquet"), as_of=args.as_of,
            dataset_start=receipt["dataset_start"], capacity=registered["capacity"], threads=args.threads)
        run_id = save_candidate(candidate, receipt, name=args.name, elapsed_seconds=time.monotonic() - started,
                                evaluation=str((folder / "report.json").resolve()))
        print(json.dumps({"run_id": run_id, "serving_qualified": False}))
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root / ".operation.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another modelling operation owns this output directory")
        if args.command == "prepare":
            from src.modelling.selection import prepare_matrix
            context, matrix, names, receipt = prepare_matrix(args.store_root, root, start=args.start, end=args.end)
            print(json.dumps({"rows": len(context), "candidate_features": len(names), "directory": str(root.resolve())}))
            return
        specs = _specs(args)
        if args.command == "select":
            from src.modelling.selection import select_features, matrix_signature
            if len(specs) != 1 or not isinstance(specs[0], ModelSpec):
                raise ValueError("Feature selection currently evaluates one numeric win recipe at a time")
            prepared = Path(args.prepared)
            description = json.loads((prepared / "prepared.json").read_text())
            frame = pd.read_parquet(prepared / "context.parquet")
            matrix = np.load(prepared / "matrix.npy", mmap_mode="r")
            from src.modelling.evaluation import frame_signature
            if frame_signature(frame, frame.columns) != description["context_signature"]:
                raise ValueError("Selection context changed after matrix preparation")
            if matrix_signature(matrix, description["features"]) != description["matrix_signature"]:
                raise ValueError("Selection matrix changed after preparation")
            result = select_features(frame, matrix, description["features"], specs[0],
                selection_end=args.selection_end, confirmation_start=args.confirmation_start,
                dataset_start=description["receipt"]["dataset_start"], n_min=args.min_features,
                min_train_periods=args.min_train_periods, capacity=args.capacity, threads=args.threads, output=root,
                near_copy=args.near_copy or None, importance=args.importance,
                permutation_below=args.permutation_below, confirm_capacity=args.confirm_capacity)
            print(json.dumps({k: v for k, v in result.items() if k.startswith("confirmation")}, indent=2))
            return
        if args.command == "admit":
            from src.modelling.selection import admit_features, matrix_signature
            if len(specs) != 1 or not isinstance(specs[0], ModelSpec):
                raise ValueError("Feature admission evaluates one numeric win recipe at a time")
            prepared = Path(args.prepared)
            description = json.loads((prepared / "prepared.json").read_text())
            frame = pd.read_parquet(prepared / "context.parquet")
            matrix = np.load(prepared / "matrix.npy", mmap_mode="r")
            if matrix_signature(matrix, description["features"]) != description["matrix_signature"]:
                raise ValueError("Selection matrix changed after preparation")
            candidates = args.candidates
            if len(candidates) == 1 and Path(candidates[0]).is_file():
                candidates = read_features(candidates[0])
            result = admit_features(frame, matrix, description["features"], specs[0], candidates, start=args.start,
                dataset_start=description["receipt"]["dataset_start"], min_train_periods=args.min_train_periods,
                capacity=args.capacity, threads=args.threads, output=root)
            print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
            return
        # The default baseline preserves the existing model's store column order.
        names = list(dict.fromkeys(n for s in specs for n in s.features))
        frame, receipt = load_dataset(args.store_root, features=names, start=args.start, end=args.end,
                                      label_columns=training_columns(specs))
        if not args.spec:
            from src.feature_store import FeatureStore
            import pyarrow.parquet as pq
            store = FeatureStore(args.store_root)
            with store._access():
                order = pq.read_schema(store.featured_path).names
            specs = [specs[0].subset([n for n in order if n in set(specs[0].features)])]
        folds = expanding_period_folds(frame.race_date, min_train_periods=args.min_train_periods,
            embargo_days=specs[0].purge_days)
        if args.fold_start:
            folds = [f for f in folds if frame.loc[f.test, "race_date"].min() >= pd.Timestamp(args.fold_start)]
        registration = {"specs": [s.to_dict() for s in specs], "receipt": receipt,
            "capacity": args.capacity, "code_signature": code_signature(),
            "folds": [{"label": f.label, "training_last_date": str(frame.loc[f.train, "race_date"].max().date()),
                "test_first_date": str(frame.loc[f.test, "race_date"].min().date()),
                "test_last_date": str(frame.loc[f.test, "race_date"].max().date())} for f in folds],
            "pool": args.pool}
        path = root / "registration.json"
        if path.exists() and json.loads(path.read_text()) != registration:
            raise ValueError("Output belongs to a different modelling registration; choose a new directory")
        atomic_json(path, registration)
        records, _ = component_oof(frame, specs, folds=folds, capacity=args.capacity, threads=args.threads,
            dataset_start=receipt["dataset_start"], cache_dir=root / "cache")
        atomic_parquet(records, root / "oof.parquet")
        reports = {}
        for spec in specs:
            part = records.loc[records.component.eq(spec.id)]
            aligned = part.merge(frame[["race_id", "horse_id", "won"]], on=["race_id", "horse_id"], validate="one_to_one")
            metrics, _, _ = probability_metrics(aligned.probability, aligned.won, race_groups(aligned.race_id))
            reports[spec.id] = metrics
        report = {"created_at": datetime.now().isoformat(), "components": reports,
            "evaluation_status": receipt["evaluation_status"], "serving_qualified": False,
            "oof_sha256": hashlib.sha256((root / "oof.parquet").read_bytes()).hexdigest()}
        if args.pool:
            report["nested_pool"], predictions, races = nested_pool_evaluation(frame, records,
                tuple(s.id for s in specs), purge_days=specs[0].purge_days)
            atomic_parquet(predictions, root / "pool_predictions.parquet")
            atomic_parquet(races, root / "pool_races.parquet")
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
