"""Combine the independently qualified source batches before one rebuild."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.deployment_validation import file_sha256 as sha
from src.result_amendment_repair import corrected_rows

CASE = ROOT / "data/audit/result_amendments_042"
OUTPUT = CASE / "consolidated"


def main():
    OUTPUT.mkdir(exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError("Preserve the completed or partially written consolidated batch")
    source_dirs = [CASE, CASE / "official_review", CASE / "official_last_places"]
    proposed_parts, original_parts, source_hashes = [], [], {}
    for directory in source_dirs:
        proposed = pd.read_parquet(directory / "proposed_repairs.parquet")
        before = pd.read_parquet(directory / "before_db_rows.parquet").drop(columns="_source_horse_id", errors="ignore")
        # Raw declared counts can include subsequently withdrawn runners.
        # Both independent fields have already passed exact starter-set checks.
        proposed["num_runners"] = proposed.groupby("race_id").race_id.transform("size")
        corrected_rows(before, proposed)
        proposed_parts.append(proposed)
        original_parts.append(before)
        for name in ("proposed_repairs.parquet", "before_db_rows.parquet"):
            path = directory / name
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        for path in directory.glob("*.json"):
            source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    proposed = pd.concat(proposed_parts, ignore_index=True, sort=False)
    before = pd.concat(original_parts, ignore_index=True, sort=False)
    for part in (proposed, before):
        part["race_id"] = part.race_id.astype(str)
    # The processed cache stores numeric IDs/dates while SQLite returns text.
    # Canonicalise the proposal's factual identity types before parquet;
    # preserve the original database rows and their SQL types unchanged.
    proposed["horse_id"] = pd.to_numeric(proposed.horse_id, errors="raise").astype("int64")
    proposed["race_date"] = pd.to_datetime(proposed.race_date, errors="raise")
    if proposed.groupby("race_id").size().shape[0] != sum(p.race_id.nunique() for p in proposed_parts):
        raise ValueError("Independent repair batches overlap")
    after = corrected_rows(before, proposed)
    proposed["corrected_finish_position"] = proposed.source_position.astype("int64")
    proposed["corrected_won"] = proposed.source_position.eq(1).astype(int)
    proposed["corrected_lengths_behind"] = np.where(proposed.corrected_won.eq(1), 0., np.nan)
    proposed.to_parquet(OUTPUT / "proposed_repairs.parquet", index=False)
    before.to_parquet(OUTPUT / "before_db_rows.parquet", index=False)
    restored = corrected_rows(pd.read_parquet(OUTPUT / "before_db_rows.parquet"),
                              pd.read_parquet(OUTPUT / "proposed_repairs.parquet"))
    pd.testing.assert_frame_equal(restored, after, check_dtype=False, check_exact=True)
    changed = proposed.finish_position.ne(proposed.source_position)
    changed_winners = proposed.won.ne(proposed.corrected_won)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "eligible_fields": int(proposed.race_id.nunique()),
        "eligible_field_runners": len(proposed), "eligible_position_changes": int(changed.sum()),
        "eligible_changed_winners": int(proposed.loc[changed_winners, "race_id"].nunique()),
        "provider_field_counts": {"atr_complete_orders": int(proposed_parts[0].race_id.nunique()),
                                  "direct_official_amendments": sum(int(p.race_id.nunique()) for p in proposed_parts[1:])},
        "source_batches": [p.relative_to(ROOT).as_posix() for p in source_dirs],
        "canonical_rows_modified": 0, "new_model_fits": 0,
        "limitations": ["Three remaining winner disagreements have no official-source amendment; no automatic change.",
            "ATR partial/nonunique order disagreements remain outside the automatic repair.",
            "Final official placings are retrospective; original publication times are not reconstructed.",
            "New fitted models must use rebuilt corrected histories. Original research scores and their label versions remain immutable."]}
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "input_sha256": source_hashes,
        "implementation_sha256": {name: sha(ROOT / name) for name in ("scripts/consolidate_result_repairs.py", "src/result_amendment_repair.py")},
        "scope": "One consolidated mandatory source correction; never select facts by predictive score"}
    for name, value in (("audit.json", report), ("audit_plan.json", plan)):
        with (OUTPUT / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
