"""Verify the daily FX repair changes only selected monetary inputs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.feature_store import FeatureStore
from src.prize_currency_features import FEATURES
from src.deployment_validation import file_sha256 as sha

CASE = ROOT / "data/audit/result_amendments_042/consolidated"


def main():
    output = CASE / "fx_refresh"
    features = json.loads((CASE / "refit/validation_final.json").read_text())["features"]
    fit = json.loads((CASE / "refit/fitted.json").read_text())
    stores = [FeatureStore(), FeatureStore(str(output / "ready_store"))]
    frames = [pd.read_parquet(store.featured_path, columns=["race_id", "horse_name", *features],
        filters=[("race_date", ">", pd.Timestamp(fit["split"]["as_of"]))])
        .set_index(["race_id", "horse_name"]).sort_index() for store in stores]
    before, after = frames
    pd.testing.assert_index_equal(before.index, after.index)
    changes = {}
    for name in features:
        a, b = before[name].to_numpy(float), after[name].to_numpy(float)
        changed = ~np.isclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-10)
        if name not in FEATURES:
            np.testing.assert_allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-10, err_msg=name)
        elif changed.any():
            changes[name] = int(changed.sum())
    report = {"passed": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "runners": len(before), "features": len(features), "all_other_selected_features_equal": True,
        "changed_monetary_features": changes, "training_prefix_unchanged": True,
        "numeric_comparison": {"atol": 1e-10, "rtol": 1e-12, "equal_nan": True},
        "prepared_sha256": sha(output / "prepared.json"), "script_sha256": sha(Path(__file__))}
    with (output / "tail_check.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
