"""Build a separate dated feature projection after the fixed public pilot ends."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.public_history_features import external_history_features, load_cached_results

OUTPUT = ROOT / "data/research/public_prior_results"


def main():
    receipt = json.loads((OUTPUT / "coverage_report.json").read_text())
    if receipt["stage"] != "complete" or receipt["completed"] != receipt["horses"]:
        raise ValueError("Finish the registered pilot before auditing coverage")
    local = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet",
        columns=["horse_id", "race_id", "horse_name", "race_date", "distance_furlongs"])
    prior = load_cached_results(OUTPUT / "horses")
    features = external_history_features(local, prior, local)
    projection = pd.concat([local[["race_id", "horse_id", "horse_name", "race_date"]], features], axis=1)
    path = OUTPUT / "features.parquet"
    projection.to_parquet(path, index=False, compression="zstd")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUTPUT / "horses").glob("*.json"))}
    population = features.external_prior_runs.gt(0)
    report = {"source_profiles": len(hashes), "source_results": len(prior),
        "local_rows_with_known_external_history": int(population.sum()),
        "local_horses_with_known_external_history": int(local.loc[population, "horse_id"].nunique()),
        "by_year_rows": {str(k): int(v) for k, v in population.groupby(pd.to_datetime(local.race_date).dt.year).sum().items()},
        "source_snapshot_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_files_sha256": hashes, "projection_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feature_source_sha256": hashlib.sha256((ROOT / "src/public_history_features.py").read_bytes()).hexdigest(),
        "selected_model_features_changed": False,
        "scope": "Coverage and date-strict feature construction only; no outcome-based predictive claim"}
    report["cohort_eligibility"] = "The legacy cache key first_local_date is the first cohort-eligible row, not necessarily the first ever DB row. Features are unavailable before that date."
    (OUTPUT / "feature_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "source_files_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
