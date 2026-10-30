"""Preserve known historical batch context and rounding during a copy-only release.

This is a closed, audit-backed migration, not a general signature override.
Unchanged model columns must remain exact; the only permitted selected-column
variation is the documented last-append rounding in two prize-history inputs.
"""
from pathlib import Path
import gc
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.deployment_validation import file_sha256
from src.feature_store import FeatureStore
from src.track_config import get_track_config
from scripts.install_live_feature_speed import verify_sources

OUT = ROOT / "data/audit/live_feature_speed_20261029/installation"
diagnostic_path = OUT / "raw_rebuild_differences.json"
diagnostic = json.loads(diagnostic_path.read_text())
plan = json.loads((OUT / "plan.json").read_text())
verify_sources(plan)
old, rebuilt = FeatureStore(str(OUT / "previous_store")), FeatureStore(str(OUT / "rebuilt_store"))
atr = {"atr_speed_score", "atr_speed_score_rank_pct_in_race", "atr_speed_score_vs_best_in_race",
       "atr_speed_score_vs_field", "atr_speed_score_z_in_race", "atr_stride_x_gallop",
       "atr_stride_x_gallop_vs_best_in_race", "atr_stride_x_gallop_vs_field", "atr_stride_x_gallop_z_in_race"}
prize = {"horse_avg_prize_contested", "prize_money_change_vs_avg"}
differences = diagnostic["differences"]
if {item["column"] for item in differences} != atr | prize:
    raise ValueError("Unexpected raw rebuild differences; do not preserve automatically")
for item in differences:
    if not item["missingness_equal"] or item["before_dtype"] != item["after_dtype"]:
        raise ValueError("Historical missingness or types changed")
    if item["column"] in atr and item["selected"]:
        raise ValueError("Batch-global ATR columns must be absent from the installed model")
    if item["column"] in prize and (not item["float32_exact"] or item["max_error"] > 1e-10
                                    or item["first_date"][:10] != "2026-10-28"):
        raise ValueError("Prize differences are not the documented append-rounding case")
if diagnostic["state"]["ratings"]["different_families"]:
    raise ValueError("Raw rating state changed")
changed_state = set(diagnostic["state"]["counts"]["different_families"])
if any(name != "_safe_cum_prize" and not name.startswith("tw::") for name in changed_state):
    raise ValueError("Unexpected carried-state changes")

# Independently reproduce both batch-global composites under the ORIGINAL
# final-build / one-day-append partition, establishing the cause of differences.
columns = ["race_date", "track", "atr_max_speed_avg_3", "atr_finishing_speed_pct_avg_3",
           "atr_peak_stride_length_avg_3", "atr_speed_score", "atr_stride_x_gallop"]
late = pd.read_parquet(old.featured_path, columns=columns,
                       filters=[("race_date", ">=", pd.Timestamp("2026-07-30"))])
for appended in (False, True):
    block = late.loc[pd.to_datetime(late.race_date).eq("2026-10-28") == appended]
    z = [(block[name] - block[name].mean()) / block[name].std()
         for name in ("atr_max_speed_avg_3", "atr_finishing_speed_pct_avg_3")]
    np.testing.assert_array_equal(((z[0] + z[1]) / 2).to_numpy(), block.atr_speed_score.to_numpy())
    circumference = block.track.astype(str).map(lambda value: get_track_config(value).get("circumference_f", np.nan))
    gallop = block.atr_peak_stride_length_avg_3 * ((circumference - circumference.mean()) / circumference.std())
    np.testing.assert_array_equal(gallop.to_numpy(), block.atr_stride_x_gallop.to_numpy())
del late
gc.collect()

before, after = old.load_counts_state(), rebuilt.load_counts_state()
window_proofs = {}
for name in sorted(changed_state - {"_safe_cum_prize"}):
    parts = name.split("::")
    if len(parts) != 4 or int(parts[-1]) not in (14, 30, 365):
        raise ValueError("Unexpected time-window state format")
    cutoff = pd.Timestamp(old.last_date()).value - int(parts[-1]) * 86_400_000_000_000

    def usable(mapping):
        result = {}
        for entity, values in mapping.items():
            active = [(int(day), float(value)) for day, value in values if int(day) >= cutoff]
            if active:
                result[entity] = active
        return result

    left, right = usable(before[name]), usable(after[name])
    if left.keys() != right.keys():
        raise ValueError(f"Usable time-window identities differ: {name}")
    for entity in left:
        for position, dtype in ((0, "int64"), (1, "float64")):
            x = np.array([row[position] for row in left[entity]], dtype=dtype)
            y = np.array([row[position] for row in right[entity]], dtype=dtype)
            if not np.array_equal(x, y, equal_nan=True):
                raise ValueError(f"Usable time-window values differ: {name}")
    window_proofs[name] = {"usable_rows_exact": True, "cutoff_ns": cutoff,
        "usable_rows": sum(map(len, left.values())),
        "reason": "Only expired/empty carried buffers differ under the original separate append; all future-reachable rows agree"}
    del left, right
a, b = before["_safe_cum_prize"], after["_safe_cum_prize"]
if a.keys() != b.keys():
    raise ValueError("Prize accumulator identities differ")
x = np.array([a[key] for key in a], dtype=float)
y = np.array([b[key] for key in a], dtype=float)
if not np.isfinite(x).all() or not np.isfinite(y).all():
    raise ValueError("Prize accumulators must be finite")
ulp = np.abs(np.spacing(np.maximum(np.abs(x), np.abs(y))))
if np.any(np.abs(x - y) > 4 * ulp):
    raise ValueError("Prize accumulation changed beyond floating-point append order")
rounding = {"changed_entities": int(np.sum(x != y)), "max_absolute_difference": float(np.max(np.abs(x-y))),
            "within_four_float64_ulps": True}
del before, after, a, b, x, y
gc.collect()


def tables(path, columns, size=16384):
    pieces, rows = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=columns):
        offset = 0
        while offset < batch.num_rows:
            count = min(size - rows, batch.num_rows - offset)
            pieces.append(batch.slice(offset, count))
            rows += count
            offset += count
            if rows == size:
                yield pa.Table.from_batches(pieces)
                pieces, rows = [], 0
    if pieces:
        yield pa.Table.from_batches(pieces)


preserved = sorted(atr | prize)
schema = pq.read_schema(rebuilt.featured_path).remove_metadata()
raw = FeatureStore(str(OUT / "raw_rebuilt_store"))
with rebuilt._access():
    for name in rebuilt._FILES:
        os.link(Path(rebuilt.root) / name, Path(raw.root) / name)
    with rebuilt._staged_update() as pending:
        rows = 0
        with pq.ParquetWriter(pending.featured_path, schema) as writer:
            for current, original in zip(tables(rebuilt.featured_path, schema.names),
                                         tables(old.featured_path, ["race_id", "horse_name", *preserved])):
                if current.num_rows != original.num_rows or any(
                    not current[name].equals(original[name]) for name in ("race_id", "horse_name")):
                    raise ValueError("Historical row identities differ during preservation")
                arrays = [(original[name] if name in preserved else current[name]).cast(schema.field(name).type)
                          for name in schema.names]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                rows += current.num_rows
                if rows % 131072 == 0:
                    print(f"Preserved historical precision: {rows:,} rows", flush=True)
        if rows != old.meta()["featured_rows"]:
            raise ValueError("Historical preservation lost rows")
        os.link(rebuilt.processed_path, pending.processed_path)
        os.link(old.state_path, pending.state_path)
        os.link(old.counts_state_path, pending.counts_state_path)
        metadata = rebuilt.meta()
        metadata["historical_boundary_preservation"] = {
            "columns": preserved, "original_append_date": "2026-10-28", "prize_rounding": rounding,
            "time_window_proofs": window_proofs,
            "raw_rebuild_diagnostic_sha256": file_sha256(diagnostic_path),
            "carried_state_bytes_preserved": True,
            "reason": "Retain prior batch-global research values and exact append precision; numerical feature algorithms unchanged"}
        Path(pending.meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
report = {"passed": True, "rows": rows, "columns_preserved": preserved, "original_atr_partition_replayed_exactly": True,
          "prize_rounding": rounding, "time_window_proofs": window_proofs, "old_carried_state_bytes_preserved": True,
          "scope": "Candidate preservation only; exhaustive strict requalification and activation still required"}
(OUT / "precision_preservation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
