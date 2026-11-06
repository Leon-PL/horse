"""Check the existing pace/setup family before its independent model trial."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.audit_atr_invariance import fixture, differences
from src import feature_engineer as fe


def derive(raw):
    raw = raw.sort_values(["race_date", "off_time", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    raw = fe._add_result_history_flags(raw)
    with patch("src.atr_features.merge_atr_sectionals", side_effect=lambda frame: frame), patch.object(fe, "_COUNTS_STATE", None):
        out = fe.add_atr_features(raw)
        out = fe.add_pace_style_features(out)
        out = fe.add_track_config_features(out)
        out = fe.add_course_dynamics_features(out)
        out = fe.add_race_relative_signal_features(out)
    return out.set_index(["race_id", "horse_id"])


def raw_fixture(days=24):
    raw = fixture(days=days)
    raw["distance_furlongs"] = raw.race_date.dt.day.map(lambda n: 5. if n % 2 else 10.)
    return raw


def audit(registration):
    added = sorted(set(registration["candidate_features"])-set(registration["baseline_features"]))
    raw = raw_fixture()
    original = derive(raw)
    if set(added)-set(original):
        raise ValueError("Pace fixture does not exercise every candidate feature")
    boundary = pd.Timestamp("2024-03-25")
    prefix = derive(raw.loc[raw.race_date < boundary].copy())
    changed = raw.copy()
    future = changed.race_date.ge(boundary)
    changed.loc[future, "atr_sectional_speeds"] = json.dumps([60., 20., 10.])
    changed.loc[future, "atr_finishing_speed_pct"] = 160.
    changed.loc[future, "won"] = 1-changed.loc[future, "won"]
    future_changed = derive(changed)
    extra = raw_fixture(days=30).iloc[len(raw):].copy()
    extended = derive(pd.concat([raw, extra], ignore_index=True))
    current_day = raw.race_date.unique()[15]
    current = raw.copy()
    selected = current.race_date.eq(current_day)
    current.loc[selected, "atr_sectional_speeds"] = json.dumps([70., 10., 90.])
    current.loc[selected, "atr_finishing_speed_pct"] = 170.
    current.loc[selected, "won"] = 1-current.loc[selected, "won"]
    current.loc[selected, "finish_position"] = 9
    current_changed = derive(current)
    keys = pd.MultiIndex.from_frame(current.loc[selected, ["race_id", "horse_id"]])
    checks = {
        "prefix_vs_full_history": differences(prefix, original, added),
        "changed_future_outcomes": differences(original.loc[prefix.index], future_changed, added),
        "appended_future_races": differences(original, extended, added),
        "current_day_results_excluded": differences(original.loc[keys], current_changed, added),
        "input_row_permutation": differences(original, derive(raw.sample(frac=1, random_state=13)), added),
    }
    unsafe = [c for c in registration["unsafe_composites_requiring_exclusion_or_rebuild"] if c in original]
    negative = differences(original, extended, unsafe)
    positive = differences(original, future_changed, added)
    return {"passed": all(not v for v in checks.values()) and bool(negative) and bool(positive),
        "tested_columns": added, "checks": checks, "quarantined_negative_control": negative,
        "prior_information_sensitivity_columns": len(positive),
        "registration_content_sha256": hashlib.sha256(json.dumps(registration, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            "scripts/audit_pace_invariance.py", "scripts/audit_atr_invariance.py", "src/feature_engineer.py",
            "src/atr_features.py", "src/fe_incremental.py", "src/track_config.py")},
        "scope": "Actual existing feature functions on synthetic date-strict histories, with positive/negative controls; no outcome-score selection"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.registration.read_text()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
