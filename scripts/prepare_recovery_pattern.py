"""Prepare one own-cadence proposal on frozen history, before any fitting."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256 as sha
from src.forecast_evaluation import valid_race_mask
from src.recovery_pattern_research import FEATURES, HISTORY_DAYS, MAX_INTERVALS, MIN_INTERVALS, recovery_pattern_features

CASE = ROOT / "data/audit/recovery_pattern_040"
REFERENCE = ROOT / "data/audit/market_transfer_034/baseline"
KEY = ["race_id", "horse_name", "race_date"]
SOURCES = ["src/recovery_pattern_research.py", "src/established_form_features.py", "src/result_status.py",
           "src/forecast_evaluation.py", "src/utils.py", "scripts/prepare_recovery_pattern.py",
           "scripts/screen_frozen_win_candidate.py", "tests/test_recovery_pattern_research.py"]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    started = time.monotonic()
    CASE.mkdir(parents=True, exist_ok=True)
    if (CASE / "plan.json").exists():
        raise ValueError("Keep completed research definitions")
    reference = json.loads((REFERENCE / "manifest.json").read_text())
    raw_path = ROOT / "data/audit/connection_form_039/source_inputs.parquet"
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    plan = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Current elapsed days may mean different things for horses with different observed racing patterns; compare the current local gap with its own routine.",
        "intervals": MAX_INTERVALS, "minimum_prior_intervals": MIN_INTERVALS, "history_days": HISTORY_DAYS,
        "features": list(FEATURES), "training_dates": ["2023-09-24", "2025-09-23"],
        "screen_dates": ["2025-10-01", "2025-12-31"],
        "advance_rule": "Both overall and established-handicap LL lower, neither runner Brier worse, and overall date-cluster95%LL upper bound below zero before any expensive native follow-up. No tuning, subsets or combined rescue.",
        "source_sha256": source_hashes, "raw_sha256": sha(raw_path),
        "fit_budget": "Reuse exact154-input small baseline; at most one157-input400-tree win candidate. No place/calibration/current-data/quarter fitting.",
        "scope": "Inspected development and observed local calendar only; not physiology or complete overseas history. No canonical/recorder writes."}
    write(CASE / "plan.json", plan)
    frame = pd.read_parquet(raw_path)
    full = recovery_pattern_features(frame)
    cut = frame.race_date.le("2025-09-23")
    prefix, seed = recovery_pattern_features(frame.loc[cut], return_state=True)
    pd.testing.assert_frame_equal(prefix, full.loc[cut], check_exact=True)
    pd.testing.assert_frame_equal(recovery_pattern_features(frame.loc[~cut], state=seed), full.loc[~cut], check_exact=True)
    pilot = frame.loc[frame.race_date.between("2025-08-01", "2025-10-07")].copy()
    control = recovery_pattern_features(pilot)
    pd.testing.assert_frame_equal(recovery_pattern_features(pilot.sample(frac=1, random_state=42)).sort_index(), control.sort_index(), check_exact=True)
    changed = pilot.copy()
    future = changed.race_date.ge("2025-10-01")
    changed.loc[future, ["finish_position", "won", "finish_pos_label"]] = [0, 0, ""]
    pd.testing.assert_frame_equal(recovery_pattern_features(changed).loc[~future], control.loc[~future], check_exact=True)
    pd.concat([frame[KEY], full], axis=1).to_parquet(CASE / "projection.parquet", index=False)
    (CASE / "features.txt").write_text("# Research only: own observed racing cadence040\n" +
        "\n".join(reference["baseline_features"] + list(FEATURES)) + "\n", encoding="utf-8")
    q4 = valid_race_mask(frame) & frame.race_date.between("2025-10-01", "2025-12-31")
    target = (frame.groupby("race_id").horse_prev_races.transform("min").ge(5) &
              frame.groupby("race_id").handicap.transform("min").eq(1))
    coverage = {}
    for label, mask in [("all_scored_Q4", q4), ("established_handicaps_Q4", q4 & target)]:
        part = full.loc[mask]
        coverage[label] = {"runners": len(part), "fields": int(frame.loc[mask, "race_id"].nunique()),
            "routine_known": int(part[FEATURES[0]].notna().sum()),
            "gap_relative_to_own_routine_over_twofold": int(part[FEATURES[0]].gt(np.log(2)).sum()),
            "gap_relative_to_own_routine_under_half": int(part[FEATURES[0]].lt(-np.log(2)).sum())}
    if any(sha(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Source changed during feature audit")
    with zipfile.ZipFile(CASE / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    report = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "full_prefix_seed_exact": True, "real_permutation_exact": True, "future_independent": True,
        "source_sha256": source_hashes, "coverage": coverage,
        "output_sha256": {name: sha(CASE / name) for name in ["plan.json", "projection.parquet", "features.txt", "source_snapshot.zip"]},
        "elapsed_seconds": time.monotonic() - started,
        "limitations": ["Observed local gaps can span unrecorded overseas activity; no claim that all exercise or starts are known.",
                        "Historical intervals are descriptions, not physiological recovery estimates or a probability bonus."]}
    write(CASE / "preparation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
