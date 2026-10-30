"""Prepare one preset past-market proposal while ordinary history rebuilds."""
from pathlib import Path
from unittest.mock import patch
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.deployment_validation import file_sha256
from src.prior_market_research import FEATURES, prior_market_features
from src.verified_race_metadata import apply_verified_race_types

OUTPUT = ROOT / "data/audit/prior_market_feasibility_033"


def main():
    start = time.monotonic()
    source = ROOT / "data/audit/trip_suitability_032/previous_generation/feature_store"
    f = pd.read_parquet(source / "processed.parquet", columns=["race_id", "horse_id", "horse_name", "race_date",
        "race_name", "race_type", "track", "surface", "finish_position", "finish_pos_label", "num_runners", "won", "odds",
        "official_rating", "has_official_rating"],
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    f = apply_verified_race_types(f)
    added = prior_market_features(f)
    pd.concat([f[["race_id", "horse_name", "race_date"]], added], axis=1).to_parquet(OUTPUT / "projection.parquet", index=False)
    mask = f.race_date.le("2025-12-31")
    prefix, state = prior_market_features(f.loc[mask], return_state=True)
    pd.testing.assert_frame_equal(prefix, added.loc[mask])
    pd.testing.assert_frame_equal(prior_market_features(f.loc[~mask], state=state), added.loc[~mask])
    reference = json.loads((ROOT / "data/audit/established_form_029/native_canonical/manifest.json").read_text())
    (ROOT / "feature_sets/prior_market_033.txt").write_text("# Research only: last-three prior market opinion and opposition033\n"+
        "\n".join(reference["baseline_features"]+list(FEATURES))+"\n", encoding="utf-8")
    report = {"feature_rows":len(added), "features":list(FEATURES), "real_prefix_and_seeded_append_exact":True,
        "source_sha256":file_sha256(ROOT / "src/prior_market_research.py"), "script_sha256":file_sha256(Path(__file__)),
        "projection_sha256":file_sha256(OUTPUT / "projection.parquet"),
        "coverage":{c:int(added[c].notna().sum()) for c in FEATURES}, "quantiles":added.quantile([0,.01,.5,.99,1]).to_dict(),
        "elapsed_seconds":time.monotonic()-start, "production_changed":False,
        "parameters":"3 actual starts,365 days,same mark scale,mean log(N*p) and observed rival marks; no coefficient fitting"}
    (OUTPUT / "feature_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
