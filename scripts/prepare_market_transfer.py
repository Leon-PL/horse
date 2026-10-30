"""Audit one fixed transfer/recency proposal against the installed033 inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.deployment_validation import file_sha256 as sha
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.market_transfer_research import FEATURES, market_transfer_features
from src.prior_market_research import past_market_observations
from src.run_store import get_production_run_id

OUTPUT = ROOT / "data/audit/market_transfer_034"
REFERENCE = ROOT / "data/audit/prior_market_feasibility_033/installation"
KEY = ["race_id", "horse_name", "race_date"]
SOURCE_PATHS = ["src/market_transfer_research.py", "src/prior_market_research.py",
                "src/established_form_features.py", "src/feature_semantics.py",
                "src/forecast_evaluation.py", "src/result_status.py",
                "scripts/prepare_market_transfer.py", "scripts/screen_market_transfer.py"]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "preparation.json").exists():
        raise ValueError("Preserve the completed preparation; use the registered screen or a new case")
    original = json.loads((REFERENCE / "manifest.json").read_text())
    baseline = original["features"]
    before = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    if len(baseline) != 154 or before["fe_signature"] != original["fe_signature"]:
        raise ValueError("Use the installed033 generation for this fixed proposal")
    source_hashes = {p: sha(ROOT / p) for p in SOURCE_PATHS}
    frame = pd.read_parquet(OUTPUT / "source_inputs.parquet")
    added = market_transfer_features(frame)
    earlier = frame.race_date.le("2025-09-23")
    prefix, state = market_transfer_features(frame.loc[earlier], return_state=True)
    pd.testing.assert_frame_equal(prefix, added.loc[earlier])
    pd.testing.assert_frame_equal(market_transfer_features(frame.loc[~earlier], state=state), added.loc[~earlier])
    projection = pd.concat([frame[KEY], added], axis=1)
    projection.to_parquet(OUTPUT / "projection.parquet", index=False)

    # Cohort context is metadata, not a replacement of any existing predictor.
    context = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet",
        columns=KEY + ["horse_prev_races", "handicap", "horse_prior_market_support_3"],
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    joined = frame.merge(context, on=KEY, validate="one_to_one")
    joined = pd.concat([joined, added.reset_index(drop=True)], axis=1)
    valid = valid_race_mask(joined)
    established = joined.groupby("race_id").horse_prev_races.transform("min").ge(5)
    handicap = joined.groupby("race_id").handicap.transform("min").eq(1)
    cohorts = {}
    for label, start, end in (("Q4_screen", "2025-10-01", "2025-12-31"),
                              ("Q1_description", "2026-01-01", "2026-03-31")):
        part = joined.loc[valid & established & handicap & joined.race_date.between(start, end)]
        missing = part.horse_prior_market_support_3.eq(0)
        recovered = missing & part[FEATURES[2]].gt(0)
        cohorts[label] = {"races": int(part.race_id.nunique()), "runners": len(part),
            "same_scale_zero_support": int(missing.sum()), "transferable_price_history": int(recovered.sum()),
            "recovered_from_flat_jump_change": int((recovered & part[FEATURES[4]].eq(1)).sum()),
            "recovered_races": int(part.loc[recovered, "race_id"].nunique()),
            "trend_known_runners": int(part[FEATURES[1]].notna().sum())}
    cases = pd.read_csv(ROOT / "data/audit/prior_market_feasibility_033/error_case_inputs.csv", parse_dates=["race_date"])
    case_inputs = cases[KEY + ["case_group"]].merge(joined[KEY + list(FEATURES)], on=KEY, validate="one_to_one")
    case_inputs.to_csv(OUTPUT / "fixed_case_inputs.csv", index=False)
    observations = past_market_observations(frame).logrel
    history_rows = []
    for case in cases.itertuples():
        identity = frame.loc[frame.race_id.eq(case.race_id) & frame.horse_name.eq(case.horse_name), "horse_id"].item()
        previous = joined.loc[joined.horse_id.eq(identity) & joined.race_date.lt(case.race_date)
                              & joined.race_date.ge(case.race_date - pd.Timedelta(days=365))].sort_values("race_date").tail(3)
        for index, row in previous.iterrows():
            history_rows.append({"case_race_id": case.race_id, "horse_name": case.horse_name,
                **row[["race_id", "race_date", "race_name", "race_type", "surface", "handicap", "odds", "official_rating"]].to_dict(),
                "past_log_field_relative_price": observations.iloc[index]})
    pd.DataFrame(history_rows).to_csv(OUTPUT / "fixed_case_source_history.csv", index=False)
    del joined, projection, prefix, state
    gc.collect()

    # The old151-input baseline is no longer the installed recipe. Preserve
    # all corrected033 input columns exactly, adding only the missing context.
    directory = OUTPUT / "baseline"
    directory.mkdir(exist_ok=True)
    previous = context.set_index(KEY).horse_prev_races
    input_path = REFERENCE / "corrected_inputs.parquet"
    source = pq.ParquetFile(input_path)
    schema = source.schema_arrow.append(pa.field("horse_prev_races", pa.float64()))
    total = 0
    with pq.ParquetWriter(directory / "inputs.parquet", schema, compression="snappy") as writer:
        for batch in source.iter_batches(batch_size=32768):
            original_table = pa.Table.from_batches([batch])
            index = pd.MultiIndex.from_frame(original_table.select(KEY).to_pandas())
            counts = previous.reindex(index)
            if counts.isna().any():
                raise ValueError("Current cohort context misses frozen033 rows")
            extended = original_table.append_column("horse_prev_races", pa.array(counts.to_numpy(), type=pa.float64()))
            if not extended.drop(["horse_prev_races"]).equals(original_table, check_metadata=False):
                raise ValueError("Existing model inputs changed while adding cohort metadata")
            writer.write_table(extended)
            total += len(extended)
    if total != len(frame):
        raise ValueError("Baseline and proposal populations differ")
    teacher = ROOT / "data/processed/betfair_bsp.parquet"
    reference = {"baseline_features": baseline, "inputs_sha256": sha(directory / "inputs.parquet"),
        "teacher_sha256": sha(teacher), "original_corrected_inputs_sha256": sha(input_path),
        "original_columns_preserved_exactly": True, "added_metadata": ["horse_prev_races"],
        "source_case": "033", "source_manifest_sha256": sha(REFERENCE / "manifest.json")}
    if reference["teacher_sha256"] != original["teacher_inputs_sha256"]["data/processed/betfair_bsp.parquet"]:
        raise ValueError("Keep the qualified033 teaching observations")
    write(directory / "manifest.json", reference)
    (directory / "features.txt").write_text("\n".join(baseline) + "\n", encoding="utf-8")
    candidate_path = ROOT / "feature_sets/market_transfer_034.txt"
    candidate_path.write_text("# Research only: fixed cross-scale price history and recency034\n" +
                              "\n".join(baseline + list(FEATURES)) + "\n", encoding="utf-8")
    after = {"production": get_production_run_id(), "fe_signature": fe_signature()}
    if before != after or any(sha(ROOT / p) != digest for p, digest in source_hashes.items()):
        raise ValueError("Source or installed generation changed during preparation")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "baseline_features": len(baseline), "candidate_features": len(baseline) + len(FEATURES),
        "features": list(FEATURES), "source_sha256": source_hashes,
        "input_sha256": {"source_inputs.parquet": sha(OUTPUT / "source_inputs.parquet"),
            "projection.parquet": sha(OUTPUT / "projection.parquet"),
            "baseline/manifest.json": sha(directory / "manifest.json")},
        "candidate_file_sha256": sha(candidate_path), "real_prefix_and_seeded_append_exact": True,
        "coverage": {c: int(added[c].notna().sum()) for c in FEATURES},
        "quantiles": added.quantile([0, .01, .5, .99, 1]).to_dict(), "target_cohorts": cohorts,
        "installed": after, "elapsed_seconds": time.monotonic() - started,
        "screen_rule": "One preset400-tree pair against corrected154 inputs; overall and established-handicap race log loss must improve with neither runner Brier worse. No retuning or automatic promotion.",
        "scope": "Source/feature audit only; no fitting. Previously inspected Q4development for the screen; Q1coverage is descriptive only.",
        "limitations": ["Past prices are conditional on the original opposition and race conditions, not an absolute ability scale.",
                        "A price trend can reflect a class/discipline change, not just improving ability.",
                        "No official-rating units are pooled across scales. Five new inputs are one prespecified family."]}
    write(OUTPUT / "preparation.json", report)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
