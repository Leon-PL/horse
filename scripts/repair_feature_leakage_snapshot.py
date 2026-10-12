"""Produce an isolated audit snapshot with known leaky history columns repaired.

This is a partial correction for controlled model comparisons, not a production
FeatureStore migration. Preprocessing, speed, and other changed feature families
still require a complete rebuild before training or promoting a production model.

Usage:
    python -m scripts.repair_feature_leakage_snapshot --output data/audit/repaired.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineer import (
    _add_result_history_flags, _event_sort_key,
    add_pedigree_features, add_race_context_features,
)
from src.feature_store import FeatureStore, _stringify_objects
from src.featured_dataset import _model_load_columns, fe_signature
from src.utils import normalise_horse_key


def _runner_index(frame: pd.DataFrame) -> pd.MultiIndex:
    keys = pd.MultiIndex.from_arrays(
        [frame["race_id"].astype(str), frame["horse_name"].map(normalise_horse_key)],
        names=["race_id", "horse_key"],
    )
    if not keys.is_unique:
        raise ValueError("Ambiguous duplicate race/horse keys; repair requires unique runners")
    return keys


def repair_known_leaky_columns(
    featured: pd.DataFrame, processed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Replace only draw-history and pedigree-history columns, aligned by key.

    All available processed history must be supplied even when featured contains
    a date subset; otherwise prior counts would be incomplete. Neither input nor
    the store's persisted rating/count state is mutated.
    """
    history = processed.copy()
    history["race_date"] = pd.to_datetime(history["race_date"])
    history["_event_dt"] = _event_sort_key(history)
    history = history.sort_values(["_event_dt", "race_id"], kind="stable")
    corrected = add_race_context_features(
        add_pedigree_features(_add_result_history_flags(history))
    )
    source_keys = _runner_index(corrected)
    target_keys = _runner_index(featured)
    positions = source_keys.get_indexer(target_keys)
    if (positions < 0).any():
        raise ValueError("Featured runners are absent from the processed history")
    cols = [c for c in corrected if c in featured and (
        c == "track_draw_win_rate" or (
            c.startswith(("sire_", "dam_", "damsire_"))
            and c.endswith(("_shrunk", "_edge"))
        )
    )]
    if not cols:
        raise ValueError("No known leaky columns are present in the snapshot")
    result = featured.copy()
    changes = {}
    for col in cols:
        values = corrected[col].to_numpy()[positions]
        old = pd.to_numeric(result[col], errors="coerce").to_numpy(dtype=float)
        changes[col] = int((~np.isclose(old, values, equal_nan=True)).sum())
        result[col] = values
    report = {
        "partial_repair": True,
        "production_ready": False,
        "rows": len(result),
        "history_rows": len(processed),
        "changed_rows_by_column": changes,
        "fe_signature": fe_signature(),
        "remaining_requirement": "Full feature-store rebuild before production training",
    }
    return result, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-columns", action="store_true",
                        help="Include all stored features, instead of model/evaluation columns")
    args = parser.parse_args()
    store = FeatureStore(args.store_root)
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(store.root).resolve()):
        raise ValueError("Audit output must be outside the live feature store")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")
    featured = store.load_featured(columns=None if args.all_columns else _model_load_columns(store))
    processed = store.load_processed()
    if featured is None or processed is None:
        raise FileNotFoundError("The source feature store is incomplete")
    repaired, report = repair_known_leaky_columns(featured, processed)
    report["source_store"] = str(Path(store.root).resolve())
    report["source_store_metadata"] = store.meta()
    output.parent.mkdir(parents=True, exist_ok=True)
    _stringify_objects(repaired).to_parquet(output, index=False)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
