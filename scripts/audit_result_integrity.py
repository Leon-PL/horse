"""Measure source-margin, identity, label and market-teacher integrity."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.betfair import attach_bsp, load_bsp, market_prob_bsp_preferred
from src.result_margins import cumulative_beaten_lengths

OUTPUT = ROOT / "data/audit/model_improvement/014_margin_repair"


def audit():
    columns = ["race_id", "race_date", "horse_id", "horse_name", "finish_position", "won",
               "lengths_behind", "age", "num_runners", "horse_prev_races", "odds", "track", "off_time"]
    f = pd.read_parquet(ROOT / "data/processed/feature_store/featured.parquet", columns=columns)
    fin = f.finish_position.ge(1)
    total = cumulative_beaten_lengths(f)
    ordered = f.loc[fin].sort_values(["race_id", "finish_position"])
    decreases = ordered.groupby("race_id", sort=False).lengths_behind.diff().lt(-1e-6)
    races = f.groupby("race_id", sort=False).agg(rows=("horse_name", "size"), declared=("num_runners", "first"), wins=("won", "sum"))
    birth = pd.to_datetime(f.race_date).dt.year - f.age
    ages = f.assign(birth=birth).groupby("horse_id").agg(
        rows=("age", "size"), dates=("race_date", "nunique"), births=("birth", "nunique"))
    report = {"rows": len(f), "races": int(f.race_id.nunique()),
              "date_min": str(f.race_date.min())[:10], "date_max": str(f.race_date.max())[:10],
              "horse_id_missing": int(f.horse_id.isna().sum()),
              "duplicate_race_horse_ids": int(f.duplicated(["race_id", "horse_id"]).sum()),
              "invalid_prices": int((~np.isfinite(f.odds) | f.odds.le(1)).sum()),
              "fields_without_one_winner": int(races.wins.ne(1).sum()),
              "fields_with_wrong_declared_size": int(races.rows.ne(races.declared).sum()),
              "negative_previous_run_counts": int(f.horse_prev_races.lt(0).sum()),
              "age_birthyear_inconsistent_horses": int((ages.births.gt(1) & ages.rows.ge(3)).sum()),
              "margins": {"nonmonotonic_source_rows": int(decreases.sum()),
                          "nonmonotonic_source_races": int(ordered.loc[decreases].race_id.nunique()),
                          "finishers": int(fin.sum()), "corrected_known_finishers": int(total.loc[fin].notna().sum()),
                          "changed_measured_margins": int((fin & total.notna() & ~np.isclose(total, f.lengths_behind, equal_nan=True)).sum()),
                          "false_close_finishes_at_2_lengths": int((fin & f.lengths_behind.le(2) & total.gt(2)).sum()),
                          "raw_gap_median_finishers": float(f.loc[fin, "lengths_behind"].median()),
                          "corrected_total_median_finishers": float(total.loc[fin].median())}}
    prices = attach_bsp(f, load_bsp(), cols=("bsp",))
    # Explicitly reconstruct the old runner-by-runner mixture for this receipt.
    ids = prices.race_id
    def old_normalise(values):
        raw = 1 / pd.to_numeric(values, errors="coerce").where(values.gt(1))
        return raw / raw.groupby(ids, sort=False).transform("sum")
    old_bsp = old_normalise(prices.bsp)
    old = old_bsp.where(old_bsp.notna(), old_normalise(prices.odds))
    old_mass = old.groupby(ids, sort=False).sum()
    prices["teacher"] = market_prob_bsp_preferred(prices)
    bsp_coverage = prices.bsp.notna().groupby(ids, sort=False).agg(["any", "all"])
    new_mass = prices.teacher.groupby(ids, sort=False).sum(min_count=1)
    report["market_teacher"] = {"complete_bsp_fields": int(bsp_coverage["all"].sum()),
        "partial_bsp_fields": int((bsp_coverage["any"] & ~bsp_coverage["all"]).sum()),
        "old_fields_with_probability_mass_error": int(old_mass.sub(1).abs().gt(1e-8).sum()),
        "old_max_probability_total": float(old_mass.max()),
        "new_available_fields": int(new_mass.notna().sum()),
        "new_max_probability_mass_error": float(new_mass.sub(1).abs().max())}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "data_integrity.json").write_text(json.dumps(report, indent=2))
    sample = f.loc[f.race_id.astype(str).eq("588358")].copy()
    sample["beaten_lengths"] = total.loc[sample.index]
    sample.to_csv(OUTPUT / "source_margin_example.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    audit()
