"""Compose the Flat and Cheltenham public-history pilots for one research test."""
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
from src.missing_history_snapshot import combine_histories


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-major-jumps", action="store_true")
    args = parser.parse_args()
    output = ROOT / "data/audit/feature_review_018"
    suffix = "_v2" if args.include_major_jumps else ""
    destination = output / f"combined_external_features{suffix}.parquet"
    if destination.exists():
        raise ValueError("Do not overwrite a frozen source projection")
    pilots = [ROOT / "data/research/public_prior_results", ROOT / "data/research/public_prior_results_cheltenham"]
    if args.include_major_jumps:
        pilots.append(ROOT / "data/research/public_prior_results_jumps_class1")
    frames, evidence, files = [], {}, {}
    for pilot in pilots:
        receipt = json.loads((pilot / "feature_audit.json").read_text())
        hashes = receipt.get("source_files_sha256", receipt.get("profile_sha256"))
        for name, expected in hashes.items():
            path = pilot / "horses" / name
            if sha(path) != expected:
                raise ValueError(f"Frozen source profile changed: {path}")
            files[str(path.relative_to(ROOT))] = expected
        frames.append(load_cached_results(pilot / "horses"))
        evidence[str(pilot.relative_to(ROOT))] = sha(pilot / "feature_audit.json")
    past = combine_histories(frames)
    local = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet",
        columns=["race_id", "horse_id", "horse_name", "race_date", "distance_furlongs"])
    values = external_history_features(local, past, local)
    projection = pd.concat([local[["race_id", "horse_id", "horse_name", "race_date"]], values], axis=1)
    temporary = destination.with_suffix(".parquet.tmp")
    projection.to_parquet(temporary, index=False, compression="zstd"); temporary.replace(destination)
    known = values.external_prior_runs.gt(0)
    receipt = {"scope": "Retrospective research; no current ratings/lifetime totals/partial-field Elo",
        "source_pilots": evidence, "source_profiles_sha256": files,
        "unique_external_performances": len(past), "rows_with_prior_external_results": int(known.sum()),
        "horses_with_prior_external_results": int(local.loc[known, "horse_id"].nunique()),
        "projection_sha256": sha(destination),
        "source_sha256": {n: sha(ROOT/n) for n in ["src/public_history_features.py", "src/missing_history_snapshot.py",
                                                     "scripts/build_missing_history_projection.py"]},
        "eligibility_rule": "earliest independently registered pre-race cohort membership for that horse; strict earlier performance dates; local results excluded"}
    (output / f"combined_external_receipt{suffix}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_profiles_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
