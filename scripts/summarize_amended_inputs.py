"""Describe recent corrected input changes; no fitting or accuracy scoring."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256 as sha

CASE = ROOT / "data/audit/result_amendments_042/consolidated"


def main():
    destination = CASE / "recent_input_changes.json"
    if destination.exists():
        raise ValueError("Preserve the completed input summary")
    old_root = ROOT / "data/audit/live_feature_speed_20261029/installation/previous_store"
    new_root = CASE / "history_rebuild/feature_store"
    old_meta = json.loads((old_root / "meta.json").read_text())
    new_meta = json.loads((new_root / "meta.json").read_text())
    if old_meta["last_date"] != new_meta["last_date"]:
        raise ValueError("Compare the same last historical date")
    features = json.loads((CASE / "refit/validation_final.json").read_text())["features"]
    first = pd.Timestamp(new_meta["last_date"]) - pd.Timedelta(days=2)
    keys = ["race_id", "horse_name"]
    frames = [pd.read_parquet(folder / "featured.parquet", columns=[*keys, *features],
        filters=[("race_date", ">=", first)]).set_index(keys).sort_index()
        for folder in (old_root, new_root)]
    before, after = frames
    pd.testing.assert_index_equal(before.index, after.index)
    affected = np.zeros(len(before), dtype=bool)
    changes = []
    for name in features:
        a, b = before[name].to_numpy(dtype=float), after[name].to_numpy(dtype=float)
        missing = np.isnan(a) != np.isnan(b)
        changed = ~np.isclose(a, b, atol=1e-10, rtol=1e-12, equal_nan=True)
        affected |= changed
        if changed.any():
            finite = np.isfinite(a) & np.isfinite(b)
            changes.append({"feature": name, "runners_changed": int(changed.sum()),
                "availability_changes": int(missing.sum()),
                "maximum_finite_absolute_change": float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None})
    changes.sort(key=lambda item: (-item["runners_changed"], item["feature"]))
    report = {"summarized_at_utc": datetime.now(timezone.utc).isoformat(),
        "dates": [first.date().isoformat(), new_meta["last_date"]], "runners": len(before),
        "fields": int(before.index.get_level_values("race_id").nunique()), "selected_features": len(features),
        "runners_with_changed_inputs": int(affected.sum()), "features_changed": len(changes),
        "numeric_comparison": {"atol": 1e-10, "rtol": 1e-12, "equal_nan": True},
        "changes": changes, "old_store_meta_sha256": sha(old_root / "meta.json"),
        "new_store_meta_sha256": sha(new_root / "meta.json"), "script_sha256": sha(Path(__file__)),
        "scope": "Descriptive recent input differences, including rebuilt rounding. No accuracy score, causal-effect estimate or additional model fit."}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, indent=2))


if __name__ == "__main__":
    main()
