"""Compare saved before/after FE implementations on one unchanged saved card.

Reads the source store under its normal lock; never rebuilds, stamps, trains,
promotes or records a forecast. Each variant runs in a fresh subprocess.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def install_baseline(directory, fe, store_module):
    for module, filename, names in (
        (fe, "feature_engineer.py", ("_expanding_prev", "_distinct_count")),
        (store_module, "feature_store.py", ("featurize_live",)),
    ):
        path = directory / "src" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = list(tree.body)
        if module is store_module:
            functions = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "FeatureStore").body
        selected = [node for node in functions
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(selected) != len(names):
            raise ValueError(f"Baseline functions missing in {path}")
        module.__dict__["copy"] = copy
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
        if module is store_module:
            module.FeatureStore.featurize_live = module.featurize_live


def worker(args):
    # Use the installed data/reference configuration, with candidate code.
    spec = importlib.util.spec_from_file_location("config", args.source / "config.py")
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    import pandas as pd
    import psutil
    from src import feature_engineer as fe, feature_store as fs
    from src.data_processor import process_data
    from src.live_prediction import stable_card_indicators, _match_store_id_types
    from src.result_status import pending_card

    if args.worker == "baseline":
        install_baseline(args.baseline, fe, fs)
    store = fs.FeatureStore(str(args.source / "data/processed/feature_store"))
    before = {name: (Path(store.root) / name).stat() for name in store._FILES}
    stages = {}
    for name in ("load_counts_state", "_continue_freq_counts", "_seeded_rating_frame"):
        original = getattr(store, name)

        def timed(*values, _name=name, _function=original, **kwargs):
            started = time.perf_counter()
            result = _function(*values, **kwargs)
            elapsed = time.perf_counter() - started
            stages.setdefault(_name, []).append(elapsed)
            print(f"{args.worker} {_name}: {elapsed:.3f}s", flush=True)
            return result

        setattr(store, name, timed)
    card = pd.read_parquet(args.card)
    prep = time.perf_counter()
    processed = stable_card_indicators(process_data(df=pending_card(_match_store_id_types(card)), save=False))
    preparation_seconds = time.perf_counter() - prep
    gc.collect()
    started = time.perf_counter()
    features = store.featurize_live(processed)
    elapsed = time.perf_counter() - started
    memory = psutil.Process().memory_info()
    for name, stat in before.items():
        after = (Path(store.root) / name).stat()
        if (stat.st_mtime_ns, stat.st_size) != (after.st_mtime_ns, after.st_size):
            raise ValueError(f"Source store changed during benchmark: {name}")
    features.to_parquet(args.output / f"{args.worker}_features.parquet", index=False)
    result = {"variant": args.worker, "rows": len(features), "columns": len(features.columns),
              "preparation_seconds": preparation_seconds, "fe_seconds": elapsed,
              "peak_rss_bytes": memory.peak_wset, "stages": stages,
              "store_unchanged": True}
    if args.model:
        from src.model import RacePredictor
        predictor = RacePredictor()
        predictor.load(str(args.model))
        started = time.perf_counter()
        predictions = predictor.predict_races(features)
        result["prediction_seconds"] = time.perf_counter() - started
        predictions.to_parquet(args.output / f"{args.worker}_predictions.parquet", index=False)
    write(args.output / f"{args.worker}.json", result)
    print(json.dumps(result), flush=True)


def main(args):
    import numpy as np
    import pandas as pd

    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for variant in ("baseline", "candidate"):
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", variant,
                   "--source", str(args.source), "--card", str(args.card),
                   "--baseline", str(args.baseline), "--output", str(args.output)]
        if args.model:
            command += ["--model", str(args.model)]
        subprocess.run(command, env=env, check=True)
    baseline = pd.read_parquet(args.output / "baseline_features.parquet")
    candidate = pd.read_parquet(args.output / "candidate_features.parquet")
    pd.testing.assert_frame_equal(baseline, candidate, check_exact=True)
    before = json.loads((args.output / "baseline.json").read_text())
    after = json.loads((args.output / "candidate.json").read_text())
    report = {"features_exact": True, "baseline": before, "candidate": after,
              "fe_speedup": before["fe_seconds"] / after["fe_seconds"],
              "fe_seconds_saved": before["fe_seconds"] - after["fe_seconds"],
              "scope": "One saved card, fresh processes, baseline first; not a throughput estimate",
              "card_sha256": sha(args.card),
              "before_sources": {name: sha(args.baseline / "src" / name)
                                 for name in ("feature_store.py", "feature_engineer.py")},
              "candidate_sources": {name: sha(ROOT / "src" / name)
                                    for name in ("feature_store.py", "feature_engineer.py")}}
    if args.model:
        old = pd.read_parquet(args.output / "baseline_predictions.parquet")
        new = pd.read_parquet(args.output / "candidate_predictions.parquet")
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        report.update(predictions_exact=True, model_sha256=sha(args.model),
                      win_probability_max_error=float(np.max(np.abs(old.win_probability - new.win_probability))),
                      place_probability_max_error=float(np.max(np.abs(old.place_probability - new.place_probability))))
    write(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Installed project with the source store")
    parser.add_argument("--card", type=Path, required=True, help="Saved raw racecard parquet")
    parser.add_argument("--baseline", type=Path, required=True, help="Snapshot directory containing src/")
    parser.add_argument("--output", type=Path, required=True, help="New receipt directory")
    parser.add_argument("--model", type=Path, help="Optional installed model for exact forecast comparison")
    parser.add_argument("--worker", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        main(arguments)
