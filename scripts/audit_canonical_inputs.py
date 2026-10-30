"""Confirm that the corrected store changes input values without losing runners.

This compares data, never predictive scores. Bounded calendar slices keep the
two full-history generations from occupying memory together.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256
from src.featured_dataset import fe_signature
from src.feature_store import FeatureStore

OUTPUT = ROOT / "data/audit/canonical_repairs_026"


def run():
    status = json.loads((OUTPUT / "rebuild_status.json").read_text())
    if status["stage"] != "rebuilt" or status["fe_signature"] != fe_signature():
        raise ValueError("Finish the corrected normal history first")
    current = FeatureStore()
    path = OUTPUT / "previous_generation/feature_store" / Path(current.featured_path).name
    features = status["features"]
    keys = ["race_date", "race_id", "horse_id", "horse_name"]
    columns = keys+features
    old_meta = json.loads((path.parent / Path(current.meta_path).name).read_text())
    expected = current.meta()
    if expected["fe_signature"] != status["fe_signature"]:
        raise ValueError("Store generation changed after the rebuild")
    changes = {name: {"changed_rows": 0, "finite_to_missing": 0,
                      "missing_to_finite": 0, "maximum_absolute_finite_change": 0.}
               for name in features}
    rows = rows_changed = races = 0
    last = pd.Timestamp(expected["last_date"]).year
    with current._access():
        first = pd.read_parquet(current.featured_path, columns=["race_date"]).race_date.min().year
        for year in range(first, last+1):
            filters = [("race_date", ">=", pd.Timestamp(year=year, month=1, day=1)),
                       ("race_date", "<", pd.Timestamp(year=year+1, month=1, day=1))]
            before = pd.read_parquet(path, columns=columns, filters=filters).sort_values(keys).reset_index(drop=True)
            after = pd.read_parquet(current.featured_path, columns=columns, filters=filters).sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(before[keys], after[keys], check_dtype=False)
            if before.duplicated(keys).any():
                raise ValueError("Ambiguous runner keys in the generation comparison")
            changed = np.zeros(len(after), dtype=bool)
            for name in features:
                a = pd.to_numeric(before[name], errors="raise").to_numpy(dtype=float)
                b = pd.to_numeric(after[name], errors="raise").to_numpy(dtype=float)
                finite_a, finite_b = np.isfinite(a), np.isfinite(b)
                different = ~np.isclose(a, b, atol=1e-10, rtol=1e-10, equal_nan=True)
                changed |= different
                result = changes[name]
                result["changed_rows"] += int(different.sum())
                result["finite_to_missing"] += int((finite_a & np.isnan(b)).sum())
                result["missing_to_finite"] += int((np.isnan(a) & finite_b).sum())
                both = finite_a & finite_b
                if both.any():
                    result["maximum_absolute_finite_change"] = max(result["maximum_absolute_finite_change"], float(np.abs(b[both]-a[both]).max()))
            rows += len(after)
            races += after.race_id.nunique()
            rows_changed += int(changed.sum())
            print(f"{year}: {len(after)} runners, {int(changed.sum())} changed selected inputs", flush=True)
    if rows != status["rows"] or rows != old_meta["featured_rows"]:
        raise ValueError("Generation comparison lost runners")
    report = {"passed": True, "rows": rows, "races": races, "rows_with_changed_inputs": rows_changed,
              "numeric_comparison_tolerance": {"absolute": 1e-10, "relative": 1e-10},
              "changed_selected_features": sum(item["changed_rows"]>0 for item in changes.values()),
              "features": changes, "old_fe_signature": old_meta["fe_signature"],
              "new_fe_signature": expected["fe_signature"],
              "source_sha256": file_sha256(__file__),
              "scope": "Input-value and whole-population comparison; no outcome scores or feature selection"}
    (OUTPUT / "feature_changes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
