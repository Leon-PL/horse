"""Gate an ATR family trial on date/future invariance, without rebuilding data.

The frozen store contains two known full-sample composites and descendants.
They are excluded by the registered candidate. Test every remaining ATR
column through the real historical derivation and relative-feature functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import feature_engineer as fe


def fixture(days=24, *, seed=42):
    rng = np.random.default_rng(seed)
    comments = ["led, stayed on", "held up, short of room, ran on", "outpaced, weakened",
                "mistake, green, hung left", "eased when beaten", "won comfortably"]
    rows = []
    for day in range(days):
        for horse in range(8):
            position = (horse + day) % 4 + 1
            missing = (horse + day) % 7 == 0
            rows.append({
                "race_id": f"r{day}-{horse // 4}",
                "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day * 7),
                "off_time": f"{13 + horse // 4}:00", "track": ["Kempton", "Ascot", "Chester"][day % 3],
                "horse_id": f"h{horse}", "horse_name": f"Horse {horse}", "num_runners": 4,
                "finish_position": position, "won": int(position == 1), "lengths_behind": float(position - 1),
                "atr_sectional_speeds": None if missing else json.dumps([30 + horse * .4, 38 + day * .2, 34 + rng.uniform(0, 5)]),
                "atr_energy_dist": json.dumps([90 + horse, 100 + day % 4]),
                "atr_optimum_energy": json.dumps([92, 101]),
                "atr_finishing_speed_pct": np.nan if missing else 85 + (day * 3 + horse) % 45,
                "atr_efficiency_score": 10 + (horse + day) % 20,
                "atr_peak_stride_length": 20 + horse + rng.uniform(0, 1),
                "atr_peak_stride_freq": 1.7 + horse * .1,
                "atr_official_position": position, "atr_adj_placing": max(1, position - horse % 3),
                "atr_in_running_comment": comments[(horse + day) % len(comments)],
            })
    return pd.DataFrame(rows)


def derive(raw):
    raw = raw.sort_values(["race_date", "off_time", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    with patch("src.atr_features.merge_atr_sectionals", side_effect=lambda df: df), patch.object(fe, "_COUNTS_STATE", None):
        out = fe.add_race_relative_signal_features(fe.add_atr_features(raw))
    return out.set_index(["race_id", "horse_id"])


def differences(before, after, columns):
    aligned = after.reindex(before.index)
    output = {}
    for col in columns:
        a, b = before[col].to_numpy(float), aligned[col].to_numpy(float)
        changed = ~np.isclose(a, b, equal_nan=True, atol=1e-10, rtol=1e-10)
        if changed.any():
            finite = np.isfinite(a) & np.isfinite(b)
            output[col] = {"changed_rows": int(changed.sum()),
                           "max_finite_difference": float(np.abs(a[finite] - b[finite]).max()) if finite.any() else None}
    return output


def audit(registration):
    columns = [c for c in registration["candidate_features"] if c.startswith("atr_")]
    raw = fixture()
    original = derive(raw)
    missing = sorted(set(columns) - set(original))
    if missing:
        raise ValueError(f"Fixture does not exercise registered columns: {missing}")
    boundary = pd.Timestamp("2024-03-25")
    earlier = raw.loc[raw.race_date < boundary]
    prefix = derive(earlier)
    changed = raw.copy()
    mask = changed.race_date >= boundary
    changed.loc[mask, "atr_finishing_speed_pct"] = 155.
    changed.loc[mask, "atr_sectional_speeds"] = json.dumps([48., 55., 52.])
    changed.loc[mask, "atr_peak_stride_length"] = 33.
    changed.loc[mask, "atr_in_running_comment"] = "led, stayed on comfortably"
    future_changed = derive(changed)
    future_rows = fixture(days=30).iloc[len(raw):].copy()
    future_rows["track"] = "Newmarket"
    extended = derive(pd.concat([raw, future_rows], ignore_index=True))
    current_day = raw.race_date.unique()[15]
    current = raw.copy()
    day_mask = current.race_date == current_day
    current.loc[day_mask, "atr_finishing_speed_pct"] = 150.
    current.loc[day_mask, "atr_sectional_speeds"] = json.dumps([50., 59., 56.])
    current.loc[day_mask, "atr_in_running_comment"] = "hampered, outpaced, weakened"
    current.loc[day_mask, "won"] = 1 - current.loc[day_mask, "won"]
    current.loc[day_mask, "lengths_behind"] = 30.
    changed_current = derive(current)
    current_keys = pd.MultiIndex.from_frame(current.loc[day_mask, ["race_id", "horse_id"]])
    checks = {
        "prefix_vs_full_history": differences(prefix, original, columns),
        "changed_future_outcomes": differences(original.loc[prefix.index], future_changed, columns),
        "appended_future_races": differences(original, extended, columns),
        "current_day_outcomes_excluded": differences(original.loc[current_keys], changed_current, columns),
        "input_row_permutation": differences(original, derive(raw.sample(frac=1, random_state=13)), columns),
    }
    excluded = [c for c in registration["unsafe_composites_requiring_exclusion_or_rebuild"] if c in original]
    negative_control = differences(original, extended, excluded)
    positive_control = differences(original, future_changed, columns)
    passed = all(not failures for failures in checks.values()) and bool(negative_control) and bool(positive_control)
    return {"passed": passed, "eligible_atr_columns": len(columns), "tested_columns": columns,
            "registration_content_sha256": hashlib.sha256(json.dumps(registration, sort_keys=True).encode()).hexdigest(),
            "historical_rows": len(raw),
            "checks": checks, "quarantined_negative_control": negative_control,
            "prior_information_sensitivity_columns": len(positive_control),
            "scope": "Real feature functions on synthetic multi-horse histories, date prefixes, future appends/changes and current-day label changes; not a publication-time availability proof",
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                "scripts/audit_atr_invariance.py", "src/feature_engineer.py", "src/atr_features.py", "src/fe_incremental.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", default="data/audit/market_performance/002_atr_family_registration.json")
    parser.add_argument("--output", default="data/audit/market_performance/002_feature_invariance.json")
    args = parser.parse_args()
    result = audit(json.loads(Path(args.registration).read_text(encoding="utf-8")))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
