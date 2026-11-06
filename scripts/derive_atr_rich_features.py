"""Derive unselected ATR research candidates from the fresh backfill snapshot.

No feature-store build, model fit, cache promotion or collector edit is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.research.atr_rich_features import (  # noqa: E402
    derive_run_metrics, build_prior_features, PRIOR_FEATURES, COUNT, AGE, RAW_COLUMNS,
)
from src.runner_identity import horse_ids, identity_values  # noqa: E402
from src.research import atr_new_tabs_features  # noqa: E402
from src.ingestion.atr_rebuild import CODE_FILES  # noqa: E402


def read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)


def coverage(frame):
    result = {}
    for column in frame:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        result[column] = {"rows": len(values), "percent": round(100*len(values)/max(len(frame), 1), 2),
                          "p05": float(np.quantile(values, .05)) if len(values) else None,
                          "median": float(np.median(values)) if len(values) else None,
                          "p95": float(np.quantile(values, .95)) if len(values) else None}
    return result


def derive(output, *, lag_hours=48, family="rich"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(config.BASE_DIR)
    if family not in {"rich", "new-tabs"}:
        raise ValueError("Unknown ATR feature family")
    new_tabs = family == "new-tabs"
    raw_columns = atr_new_tabs_features.RAW_COLUMNS if new_tabs else RAW_COLUMNS
    prior_columns = atr_new_tabs_features.PRIOR_FEATURES if new_tabs else PRIOR_FEATURES
    quality_columns = atr_new_tabs_features.QUALITY_FEATURES if new_tabs else (COUNT, AGE)
    run_builder = atr_new_tabs_features.derive_run_metrics if new_tabs else derive_run_metrics
    history_builder = atr_new_tabs_features.build_prior_features if new_tabs else build_prior_features
    pinned = list(CODE_FILES)
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in pinned}
    feature_files = ["scripts/derive_atr_rich_features.py",
                     "src/research/atr_new_tabs_features.py" if new_tabs else "src/research/atr_rich_features.py"]
    feature_hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in feature_files}
    queue = Path(config.DATA_DIR)/"atr_cache/rebuild_v2/rebuild.sqlite"
    expected = {}
    rows, missing_identity = [], 0
    with read_only(queue) as con:
        for race, horse_key, payload, identity in con.execute(
                "SELECT r.race_id,r.horse_key,r.payload,j.expected FROM runners r JOIN jobs j USING(race_id)"):
            if race not in expected:
                expected[race] = {item["horse_key"]: item.get("horse_id") for item in json.loads(identity)}
            horse = expected[race].get(horse_key)
            if horse is None:
                missing_identity += 1
                continue
            record = json.loads(payload)
            rows.append({"race_id": race, "horse_id": horse, "atr_field_coverage": record.get("atr_field_coverage"),
                         **{column: record.get(column) for column in raw_columns}})
    observations = pd.DataFrame(rows)
    if observations.empty:
        raise RuntimeError("No fresh ATR observations with canonical runner identities are available")
    all_source_rows = len(observations)
    if new_tabs:
        # Metadata/context repeated across untracked runners is not telemetry.
        measured = [name for name in raw_columns if name not in {
            "atr_official_position", "atr_jump_course_winner_speed_lost_pct",
            "atr_jumps_par_updated_date", "atr_race_median_distance_ran_yards",
            "atr_distance_diff_vs_median_lengths"}]
        observations = observations.loc[observations[measured].notna().any(axis=1)].copy()
        if observations.empty:
            raise RuntimeError("The fresh backfill does not yet contain new-tab observations")
    observations["horse_id"] = horse_ids(observations)
    observations["race_id"] = identity_values(observations.race_id)
    with read_only(Path(config.DATA_DIR)/"races.db") as con:
        canonical = pd.read_sql_query("SELECT race_id,horse_id,race_date,off_time,finish_position,"
                                      "race_type,track,distance_furlongs FROM results", con)
    canonical["horse_id"] = horse_ids(canonical)
    canonical["race_id"] = identity_values(canonical.race_id)
    all_canonical_rows = len(canonical)
    # These predictors depend only on a horse's own history. Keep every race
    # appearance for cohort horses while avoiding work for unrelated horses.
    canonical = canonical.loc[canonical.horse_id.isin(observations.horse_id.dropna().unique())].copy()
    combined = canonical.merge(observations, on=["horse_id", "race_id"], how="left", validate="one_to_one", indicator=True)
    matched = combined.loc[combined._merge.eq("both")].copy()
    if new_tabs and len(matched) != len(observations):
        raise ValueError("New-tab source rows do not all match canonical runner identities")
    # New candidates are emitted for every appearance of cohort horses, even
    # where the current race has no telemetry. Future raw rows cannot enter a
    # historical prediction through the as-of boundary.
    targets = combined if new_tabs else matched
    run = run_builder(matched)
    prior = history_builder(combined, targets, availability_lag=pd.Timedelta(hours=lag_hours))
    metadata = targets[["race_id", "horse_id", "race_date", "off_time"]].reset_index(drop=True)
    run_metadata = matched[["race_id", "horse_id", "race_date", "off_time"]].reset_index(drop=True)
    pd.concat([run_metadata, run.reset_index(drop=True)], axis=1).to_parquet(output/"run_metrics.parquet", index=False)
    pd.concat([metadata, prior.reset_index(drop=True)], axis=1).to_parquet(output/"prior_candidates.parquet", index=False)
    if new_tabs:
        # Freeze the exact inputs: the live queue continues changing while
        # subsequent experiments need to reproduce this feature set.
        snapshot_columns = list(dict.fromkeys([
            "race_id","horse_id","race_date","off_time","finish_position",
            "race_type","track","distance_furlongs",*raw_columns]))
        combined[snapshot_columns].to_parquet(output/"history_inputs.parquet", index=False)
    (output/"feature_names.json").write_text(json.dumps({
        "identity_columns":["race_id","horse_id"], "prior_predictors":list(prior_columns),
        "quality_features":list(quality_columns), "post_race_diagnostics":list(run.columns),
        "selected_in_normal_model":False,
    },indent=2),encoding="utf-8")
    status = json.loads((Path(config.DATA_DIR)/"atr_cache/rebuild_v2/status.json").read_text())
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "source": str(queue), "family": family,
        "all_source_rows_with_identity":all_source_rows,
        "fresh_rows_with_identity": len(observations), "missing_canonical_identity": missing_identity,
        "matched_canonical_runners": len(matched), "unmatched_source_rows": len(observations)-len(matched),
        "output_target_runners":len(targets),
        "source_rows_with_usable_run_metrics":int(run.notna().any(axis=1).sum()),
        "canonical_history_rows": len(canonical), "all_canonical_history_rows": all_canonical_rows,
        "availability_lag_hours": lag_hours,
        "scope": "Coverage within the partially rebuilt source cohort; not whole-project coverage or evidence of predictive improvement",
        "source_field_coverage": matched.atr_field_coverage.value_counts(dropna=False).to_dict(),
        "run_metrics": coverage(run), "prior_candidates": coverage(prior[list(prior_columns)]),
        "runners_with_prior_candidates": int(prior[list(prior_columns)].notna().any(axis=1).sum()),
        "prior_observed_runs": coverage(prior[list(quality_columns)]),
        "revision_policy":"New-tab grade and course par require their revision day to end before the target off" if new_tabs else None,
        "completion_policy":"New-tab measurements require a verified finisher; nonfinishers still occupy history slots" if new_tabs else None,
        "collector_hashes_unchanged": all(hashlib.sha256((root/name).read_bytes()).hexdigest() == value for name, value in hashes.items()),
        "collector_code_hashes":hashes, "feature_code_hashes":feature_hashes,
        "collector_state_after": status["state"], "collector_rows_after": status["saved_rows"],
        "model_evaluation": "not run; candidates are unselected research outputs"}
    (output/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if new_tabs:
        print(json.dumps({key:report[key] for key in (
            "family", "fresh_rows_with_identity", "matched_canonical_runners", "output_target_runners",
            "runners_with_prior_candidates", "collector_hashes_unchanged", "collector_state_after")}, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family", choices=("rich","new-tabs"), default="rich")
    parser.add_argument("--lag-hours", type=int, default=48)
    args = parser.parse_args()
    default_name = "atr_new_tabs_20261106" if args.family=="new-tabs" else "atr_rich_20261106"
    derive(args.output or Path(config.DATA_DIR)/"research"/default_name,
           lag_hours=args.lag_hours, family=args.family)


if __name__ == "__main__":
    main()
