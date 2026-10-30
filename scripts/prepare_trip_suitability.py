"""Audit one continuous distance-history proposal, without model fitting."""
from pathlib import Path
from unittest.mock import patch
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.deployment_validation import file_sha256
from src.trip_suitability_research import FEATURES, trip_suitability_features
from src.verified_race_metadata import apply_verified_race_types

OUTPUT = ROOT / "data/audit/trip_suitability_032"


def main():
    start = time.monotonic()
    f = pd.read_parquet(ROOT / "data/processed/feature_store/processed.parquet",
        columns=["race_id", "horse_id", "horse_name", "race_date", "race_name", "race_type", "track", "surface",
            "distance_furlongs", "finish_position", "finish_pos_label", "num_runners", "won"],
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    catalog = json.loads((OUTPUT / "verified_race_types_merged.json").read_text())["races"]
    with patch("src.verified_race_metadata._read", return_value=catalog):
        f = apply_verified_race_types(f)
    added = trip_suitability_features(f)
    pd.concat([f[["race_id", "horse_name", "race_date"]], added], axis=1).to_parquet(OUTPUT / "projection.parquet", index=False)
    mask = f.race_date.le("2025-12-31")
    prefix, state = trip_suitability_features(f.loc[mask], return_state=True)
    pd.testing.assert_frame_equal(prefix, added.loc[mask])
    pd.testing.assert_frame_equal(trip_suitability_features(f.loc[~mask], state=state), added.loc[~mask])
    reference = json.loads((ROOT / "data/audit/established_form_029/native_canonical/manifest.json").read_text())
    (ROOT / "feature_sets/trip_suitability_032.txt").write_text("# Research only: continuous discipline-specific trip history032\n"+
        "\n".join(reference["baseline_features"]+list(FEATURES))+"\n", encoding="utf-8")
    report = {"feature_rows":len(added), "features":list(FEATURES), "real_prefix_and_seeded_append_exact":True,
        "source_sha256":file_sha256(ROOT / "src/trip_suitability_research.py"),
        "script_sha256":file_sha256(Path(__file__)), "projection_sha256":file_sha256(OUTPUT / "projection.parquet"),
        "metadata_catalog_sha256":file_sha256(OUTPUT / "verified_race_types_merged.json"),
        "coverage":{c:int(added[c].notna().sum()) for c in FEATURES},
        "quantiles":added.quantile([0, .01, .5, .99, 1]).to_dict(), "elapsed_seconds":time.monotonic()-start,
        "production_changed":False, "parameters":"20 starts,730 days,15% relative-distance bandwidth,5 pseudo-starts; fixed before any score"}
    (OUTPUT / "trip_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
