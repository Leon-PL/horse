"""Check that non-completions survive storage as actual historical starts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import pyarrow.parquet as pq
from src.feature_engineer import _add_result_history_flags


def main():
    source = ROOT / "data/processed/feature_store/featured.parquet"
    columns = ["race_id", "race_date", "off_time", "horse_id", "horse_name", "finish_position", "won",
               "race_type", "horse_prev_races"]
    has_labels = "finish_pos_label" in pq.read_schema(source).names
    if has_labels:
        columns.append("finish_pos_label")
    f = pd.read_parquet(source, columns=columns).sort_values(["race_date", "off_time", "race_id"], kind="stable").reset_index(drop=True)
    labelled = _add_result_history_flags(f.copy())
    settled = f.finish_position.gt(0).groupby(f.race_id, sort=False).transform("any")
    actual_nonfinishers = f.finish_position.eq(0) & settled
    prior_starts = f.groupby("horse_id", sort=False).cumcount()
    # Horses should rarely run twice on a day; use the first count that day.
    prior_starts = prior_starts.groupby([f.horse_id, f.race_date], sort=False).transform("min")
    understated = f.horse_prev_races.lt(prior_starts)
    report = {"stored_finish_label_column": has_labels,
        "actual_nonfinishers_in_settled_fields": int(actual_nonfinishers.sum()),
        "recognised_nonfinisher_flags": int(labelled.loc[actual_nonfinishers, "_non_finisher_result"].sum()),
        "nonfinishers_treated_as_pending_by_history": int((actual_nonfinishers & labelled._result_start.eq(0)).sum()),
        "rows_with_understated_known_prior_starts": int(understated.sum()),
        "maximum_prior_start_undercount": int((prior_starts - f.horse_prev_races).max()),
        "by_race_type": {str(k): {"runners": int(len(g)), "nonfinishers": int(actual_nonfinishers.loc[g.index].sum())}
                         for k, g in f.groupby("race_type", sort=False)},
        "scope": "Original persisted snapshot; explicit non-runners are excluded by the scraper. A zero position in a settled field is a non-completion, not a pending card."}
    out = ROOT / "data/audit/model_improvement/015_nonfinisher_history"
    out.mkdir(parents=True, exist_ok=True)
    (out / "discovery.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
