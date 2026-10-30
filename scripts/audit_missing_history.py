"""Validate the Cheltenham dated-history pilot without fitting or viewing returns."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.public_history_features import external_history_features, load_cached_results


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["cheltenham", "jumps_class1"], default="cheltenham")
    args = parser.parse_args()
    directory = ROOT / f"data/research/public_prior_results_{args.scope}"
    progress = json.loads((directory / "progress.json").read_text())
    registration = json.loads((directory / "cohort.json").read_text())
    if progress["stage"] != "complete" or progress["errors"]:
        raise ValueError("Finish the registered source collection first")
    files = sorted((directory / "horses").glob("*.json"))
    if len(files) != len(registration["horses"]):
        raise ValueError("A registered horse is missing from the snapshot")
    local = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet",
        columns=["race_id", "horse_id", "horse_name", "race_date", "race_type", "track",
                 "distance_furlongs", "horse_prev_races"])
    past = load_cached_results(directory / "horses")
    dates = pd.to_datetime(past.date)
    eligibility = pd.to_datetime(past.eligible_from_date)
    earlier = past.loc[dates.lt(eligibility)]
    values = external_history_features(local, past, local)
    projection = pd.concat([local[["race_id", "horse_id", "horse_name", "race_date"]], values], axis=1)
    target = directory / "features.parquet"
    if target.exists():
        raise ValueError("Preserve the first audited source projection")
    projection.to_parquet(target, index=False, compression="zstd")
    populated = values.external_prior_runs.gt(0)
    summaries = []
    for horse in registration["horses"]:
        before = earlier.loc[earlier.horse_id.astype(str).eq(horse["horse_id"])]
        row = local.loc[local.horse_id.eq(int(horse["horse_id"]))
                        & local.race_date.eq(horse["first_local_date"])]
        summaries.append({**horse, "known_dated_profile_records_before_local": len(before),
            "courses_before_local": "; ".join(sorted(before.course_name.dropna().unique())),
            "usable_external_results_at_first_local": float(values.loc[row.index, "external_prior_runs"].max()) if len(row) else None})
    pd.DataFrame(summaries).to_csv(directory / "coverage_by_horse.csv", index=False)
    report = {"scope": "Retrospective source coverage only; no outcome-based membership or prediction-gain claim",
        "registered_horses": len(files), "profile_records": len(past), "profile_records_before_first_local": len(earlier),
        "horses_with_usable_external_history": int(local.loc[populated, "horse_id"].nunique()),
        "local_rows_with_usable_external_history": int(populated.sum()),
        "first_local_rows_with_usable_external_history": int((populated & local.horse_prev_races.eq(0)).sum()),
        "prior_courses": earlier.course_name.value_counts().to_dict(),
        "cohort_sha256": sha(directory / "cohort.json"), "projection_sha256": sha(target),
        "profile_sha256": {p.name: sha(p) for p in files},
        "source_sha256": {n: sha(ROOT / n) for n in ["src/public_history_features.py", "scripts/collect_missing_history_results.py",
            "scripts/collect_prior_horse_results.py", "scripts/audit_missing_history.py"]},
        "production_or_local_results_changed": False}
    (directory / "feature_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "profile_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
