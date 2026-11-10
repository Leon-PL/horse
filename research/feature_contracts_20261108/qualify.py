"""Compare the historical FE kernel with current code on accepted source rows.

Reads the committed dataset and writes audit outputs only. It does not refresh
sources, rebuild the installed store or train a model. The baseline engine uses
the current accepted-source readers so this isolates the kernel refactor.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd

import config
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import get_feature_columns
from src.source_dataset import committed_dataset, load_manifest, read_table, use_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="55b9e6e")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dataset", help="Replay this accepted manifest ID instead of the latest commit")
    parser.add_argument("--out", type=Path, default=Path("data/audit/feature_contracts_20261108/qualification"))
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.run(["git", "show", f"{args.baseline}:src/feature_engineer.py"],
                              check=True, capture_output=True).stdout
    path = args.out.resolve()/"baseline_engine.py"
    path.write_bytes(baseline)
    spec = importlib.util.spec_from_file_location("src._fe_qualified_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    manifest = load_manifest(args.dataset) if args.dataset else committed_dataset()
    if manifest is None:
        raise RuntimeError("Publish a dataset before qualification; this command never publishes")
    with use_manifest(manifest):
        raw = read_table("sl", "results")
        days = pd.to_datetime(raw.race_date).dt.normalize()
        end = days.max()
        raw = raw.loc[days.ge(end-pd.Timedelta(days=args.days-1))].copy()
        processed = process_data(raw)
        previous_state, current_state = {}, {}
        previous = module.engineer_features(processed.copy(), save=False, counts_state=previous_state)
        current = engineer_features(processed.copy(), save=False, counts_state=current_state)
        pd.testing.assert_frame_equal(previous, current, check_exact=True, check_dtype=True)
        old_state = pickle.dumps(previous_state, protocol=pickle.HIGHEST_PROTOCOL)
        new_state = pickle.dumps(current_state, protocol=pickle.HIGHEST_PROTOCOL)
        if old_state != new_state:
            raise AssertionError("Carried state differs from the historical kernel")
        result = {
            "baseline": args.baseline, "baseline_engine_sha256": hashlib.sha256(baseline).hexdigest(),
            "dataset": manifest["id"], "start": str(pd.to_datetime(raw.race_date).min()), "end": str(end),
            "rows": len(current), "columns": len(current.columns),
            "selected_features": len(get_feature_columns(current)),
            "exact_values_dtypes_order": True, "exact_carried_state": True,
            "bha_policy": config.BHA_FEATURE_AVAILABILITY_POLICY,
            "runners_with_history_support": {family: int(current[column].fillna(0).gt(0).sum())
                for family, column in (("atr", "atr_observation_count_5"), ("rtv", "rtv_history_count3"),
                                       ("bha", "bha_report_count_5"))},
            "scope": "Historical FE kernel and current accepted-source dependencies; no installed store or model changed",
        }
    (args.out/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
