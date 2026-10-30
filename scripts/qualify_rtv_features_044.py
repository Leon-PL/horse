"""Seal the preset RTV feature family and audit real-data causal projections."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"data/audit/rtv_repair_features_044"
PROJECT = CASE/"project"
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from src import rtv_features as features
from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.source_pipeline import _publish


def main():
    files = PROJECT/"feature_sets"
    (files/"rtv_history.txt").write_text("# Mature, quality-filtered prior-race RTV observations\n"+"\n".join(features.HISTORY_FEATURES)+"\n", encoding="utf-8")
    (files/"rtv_pre_race.txt").write_text("# Genuine timestamped pre-race snapshots; not eligible for the historical trial\n"+"\n".join(features.PRE_FEATURES)+"\n", encoding="utf-8")
    (files/"win_with_rtv.txt").write_text((files/"win.txt").read_text().rstrip()+"\n\n# Fixed RTV history candidate044\n"+"\n".join(features.HISTORY_FEATURES)+"\n", encoding="utf-8")
    run = pd.read_parquet(features.runs_path())
    source = pd.read_csv(PROJECT/"data/raw/race_results.csv", usecols=["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"])
    sample = source.loc[source.race_date.between("2025-10-01", "2025-10-14")].copy()
    if sample.empty:
        raise ValueError("Real-data cohort disappeared")
    full = features.add_history_features(sample, runs=run)
    if any(full[c].notna().sum() == 0 for c in features.HISTORY_FEATURES):
        raise ValueError("A requested RTV history input has no usable real observations")
    shuffled = features.add_history_features(sample.sample(frac=1, random_state=42), runs=run.sample(frac=1, random_state=43)).sort_index()
    pd.testing.assert_frame_equal(full[list(features.HISTORY_FEATURES)].sort_index(), shuffled[list(features.HISTORY_FEATURES)], check_exact=True)
    max_error = 0.
    for indexes in np.array_split(np.arange(len(sample)), 4):
        part = features.add_history_features(sample.iloc[indexes], runs=run)
        pd.testing.assert_frame_equal(full.iloc[indexes][list(features.HISTORY_FEATURES)], part[list(features.HISTORY_FEATURES)], check_exact=True)
    future = run.copy()
    later = pd.to_datetime(future.available_date).gt("2025-10-14")
    future.loc[later, list(features.RUN_METRICS)] = -999.
    unchanged = features.add_history_features(sample.assign(won=999, finish_position=999, odds=999), runs=future)
    pd.testing.assert_frame_equal(full[list(features.HISTORY_FEATURES)], unchanged[list(features.HISTORY_FEATURES)], check_exact=True)
    plan = {"registered_at": datetime.now(timezone.utc).isoformat(), "fe_signature": fe_signature(),
        "history_features": list(features.HISTORY_FEATURES), "pre_features": list(features.PRE_FEATURES),
        "family_count": len(features.HISTORY_FEATURES), "publication_lag_calendar_days": 2,
        "history_lookback_observations": 3, "max_history_age_days": 730,
        "small_screen": {"trees": 400, "train": ["2023-09-24", "2025-09-23"],
            "test": ["2025-10-01", "2025-12-31"], "arms": ["repaired154", "repaired154_plus_fixed26"],
            "rule": "One preset whole-family comparison; no subset/quarter/capacity search; no automatic predictive adoption"},
        "real_projection_audit": {"rows": len(sample), "permutation_exact": True, "subset_append_exact": True,
            "future_current_labels_prices_independent": True, "coverage": {c: int(full[c].notna().sum()) for c in features.HISTORY_FEATURES}},
        "observation_rows": len(run), "source_sha256": {name: sha(PROJECT/name) for name in
            ("src/rtv_features.py", "src/rtv_feature_collection.py", "src/featured_dataset.py", "src/feature_inputs.py")},
        "cache_sha256": {"runs": sha(features.runs_path()), "pre": sha(features.pre_path())},
        "limitations": ["Historical backfill does not reconstruct original publication/revision timestamps.",
            "Only prior matured runs are used, but this is retrospective development, not prospective confirmation.",
            "Pre-race Timeform/pace/draw inputs require saved captures; one day is insufficient for predictive adoption."]}
    _publish(CASE/"feature_plan.json", plan)
    print({"features": len(features.HISTORY_FEATURES), "pre_features": len(features.PRE_FEATURES),
        "projection_rows": len(sample), "observation_rows": len(run), "passed": True}, flush=True)


if __name__ == "__main__":
    main()
