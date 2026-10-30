"""No-fit feasibility audit of strictly earlier complete-field SP information.

Current-race prices are never emitted. Counts describe source coverage, not
predictive performance. Source SP is distinct from executable or timed odds.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.deployment_validation import file_sha256
from src.forecast_evaluation import valid_race_mask

OUTPUT = ROOT / "data/audit/prior_market_feasibility_033"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/audit/trip_suitability_032/previous_generation/feature_store/processed.parquet"
    f = pd.read_parquet(source, columns=["race_id", "horse_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "odds"],
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    f.horse_id = pd.to_numeric(f.horse_id, errors="raise").astype("Int64").astype(str)
    price = pd.to_numeric(f.odds, errors="coerce")
    available = price.ge(1.01) & np.isfinite(price) & valid_race_mask(f)
    complete = available.groupby(f.race_id, sort=False).transform("all")
    implied = (1/price).where(complete)
    f["past_field_probability"] = implied/implied.groupby(f.race_id, sort=False).transform("sum")
    f = f.sort_values(["race_date", "race_id"], kind="stable")
    history = {str(k):(g.race_date.to_numpy(dtype="datetime64[D]"), g.past_field_probability.to_numpy())
               for k,g in f.groupby("horse_id", sort=False)}
    cohort_path = ROOT / "data/audit/handicap_weight_review_030/cohort_context.parquet"
    cohort = pd.read_parquet(cohort_path, columns=["race_id", "horse_id", "horse_name", "race_date"])
    rows = []
    for row in cohort.itertuples(index=False):
        days, values = history[str(row.horse_id)]
        current = np.datetime64(row.race_date.date(), "D")
        end = int(np.searchsorted(days, current, side="left"))
        recent = values[max(0,end-3):end]
        known = recent[np.isfinite(recent)]
        last_date = str(days[end-1]) if end else None
        if last_date is not None and pd.Timestamp(last_date) >= row.race_date:
            raise ValueError("Current/future price entered prior-history audit")
        rows.append({"race_id":str(row.race_id), "horse_id":str(row.horse_id), "horse_name":row.horse_name,
            "race_date":str(row.race_date)[:10], "local_prior_starts":end, "last_local_date":last_date,
            "known_prices_in_last_3_starts":len(known),
            "mean_probability_last_3_starts":float(known.mean()) if len(known) else None})
    result = pd.DataFrame(rows)
    result.to_parquet(OUTPUT / "coverage_context.parquet", index=False)
    report = {"scope":"Source feasibility only; no feature fit, model training or predictive-gain claim",
        "cohort":"Saved951 complete Q1 established-handicap fields, every horse with5+ local starts",
        "runners":len(result), "fields":int(result.race_id.nunique()),
        "historical_complete_price_fields":int(f.loc[f.past_field_probability.notna(),"race_id"].nunique()),
        "prior_price_support_counts":result.known_prices_in_last_3_starts.value_counts().sort_index().to_dict(),
        "share_with_all_3_prior_prices":float(result.known_prices_in_last_3_starts.eq(3).mean()),
        "source_sha256":file_sha256(source), "cohort_sha256":file_sha256(cohort_path),
        "script_sha256":file_sha256(Path(__file__)), "production_changed":False,
        "next_hypothesis":"Past market assessments relative to the opposition may contain information absent from result-only ratings. Audit a causal cached projection before one capped test; current-race prices must remain excluded."}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
