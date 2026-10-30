"""Compare cached result winners with complete independent exchange fields.

This is a source-integrity census, not a predictive evaluation. Exchange
settlement alone cannot establish an amended official order, so disagreements
are leads for source review, never automatic replacement labels.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.betfair import attach_bsp, bf_track_key
from src.deployment_validation import file_sha256 as sha

CASE = ROOT / "data/audit/result_amendments_042"
OUTPUT = CASE / "winner_census"
HISTORY = ROOT / "data/audit/live_feature_speed_20261029/installation/previous_store/processed.parquet"
EXCHANGE = ROOT / "data/processed/betfair_bsp.parquet"


def main():
    OUTPUT.mkdir(exist_ok=False)
    sources = {name: sha(ROOT / name) for name in
               ("scripts/audit_result_winners.py", "src/betfair.py", "src/utils.py")}
    plan = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sources, "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                            (HISTORY, EXCHANGE, CASE / "proposed_repairs.parquet")},
        "scope": "Source integrity through28October; no prediction scoring, fitting, source collection or canonical mutation",
        "eligibility": "Identical complete starter population; one exchange market; source reported and observed sizes agree; all winner labels observed; one winner in each source",
        "policy": "Report differences before/after the already-reviewed70-field proposal. Never infer an official order from settlement alone."}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    history = pd.read_parquet(HISTORY, columns=["race_id", "horse_id", "race_date", "track", "off_time",
        "horse_name", "num_runners", "finish_position", "finish_pos_label", "won", "race_type"])
    history["race_id"] = history.race_id.astype(str)
    book = pd.read_parquet(EXCHANGE, columns=["race_date", "track", "off_key", "horse_key", "market_id", "n_runners", "won"])
    book["race_date"] = pd.to_datetime(book.race_date).dt.strftime("%Y-%m-%d")
    book["_course"] = book.track.map(bf_track_key)
    grouped = book.groupby(["race_date", "_course", "off_key"], dropna=False)
    book["exchange_field_size"] = grouped.horse_key.transform("nunique")
    book["exchange_market_count"] = grouped.market_id.transform("nunique")
    book = book.rename(columns={"won": "exchange_won", "n_runners": "exchange_reported_size",
                                "market_id": "exchange_market_id"})
    joined = attach_bsp(history, bsp=book, cols=("exchange_won", "exchange_reported_size",
                       "exchange_field_size", "exchange_market_count", "exchange_market_id"))
    group = joined.groupby("race_id", sort=False)
    n = group.race_id.transform("size")
    eligible_row = (joined.num_runners.eq(n) & joined.exchange_field_size.eq(n)
        & joined.exchange_reported_size.eq(n) & joined.exchange_market_count.eq(1)
        & joined.exchange_won.isin([0, 1]) & joined.won.isin([0, 1]))
    eligible = eligible_row.groupby(joined.race_id).transform("all")
    eligible &= group.exchange_market_id.transform("nunique").eq(1)
    eligible &= group.exchange_won.transform("sum").eq(1) & group.won.transform("sum").eq(1)
    different = joined.won.ne(joined.exchange_won).groupby(joined.race_id).transform("any") & eligible
    repairs = pd.read_parquet(CASE / "proposed_repairs.parquet", columns=["race_id", "horse_id", "corrected_won"])
    repairs["race_id"] = repairs.race_id.astype(str)
    # Match by actual source identity, not display spelling or another
    # provider's horse ID namespace.
    for frame in (joined, repairs):
        frame["_source_horse_id"] = pd.to_numeric(frame.horse_id, errors="raise").astype("Int64")
    joined = joined.merge(repairs[["race_id", "_source_horse_id", "corrected_won"]],
                          on=["race_id", "_source_horse_id"], how="left", sort=False, validate="one_to_one")
    corrected = joined.corrected_won.fillna(joined.won)
    unresolved = corrected.ne(joined.exchange_won).groupby(joined.race_id).transform("any") & eligible.to_numpy()
    joined["eligible_complete_comparison"] = eligible.to_numpy()
    joined["disagrees_before_repair"] = different.to_numpy()
    joined["disagrees_after_reviewed_repair"] = unresolved
    mismatches = joined.loc[different.to_numpy()].copy()
    decisions = pd.read_csv(CASE / "field_decisions.csv", dtype={"race_id": str})
    mismatches = mismatches.merge(decisions[["race_id", "decision"]], on="race_id", how="left", validate="many_to_one")
    mismatches["decision"] = mismatches.decision.fillna("outside_atr_disagreement_roster")
    mismatches.to_parquet(OUTPUT / "disagreement_fields.parquet", index=False)
    fields = mismatches.groupby("race_id", sort=False).agg(race_date=("race_date", "first"),
        track=("track", "first"), off_time=("off_time", "first"), runners=("horse_name", "size"),
        race_type=("race_type", "first"), decision=("decision", "first"),
        still_disagrees=("disagrees_after_reviewed_repair", "first")).reset_index()
    fields.to_csv(OUTPUT / "disagreement_fields.csv", index=False)
    if any(sha(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError("Source changed during the census")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "history_runners": len(joined),
        "history_fields": int(joined.race_id.nunique()), "eligible_fields": int(joined.loc[eligible.to_numpy(), "race_id"].nunique()),
        "eligible_runners": int(eligible.sum()), "winner_disagreement_fields_before": len(fields),
        "winner_disagreement_fields_after_reviewed_repair": int(fields.still_disagrees.sum()),
        "resolved_by_reviewed_batch": int((~fields.still_disagrees).sum()),
        "residual_decisions": {str(k): int(v) for k, v in fields.loc[fields.still_disagrees, "decision"].value_counts().items()},
        "new_model_fits": 0, "canonical_writes": False, "plan_sha256": sha(OUTPUT / "plan.json"),
        "limitations": ["Exchange settlement and official retrospective results may differ legitimately after later rulings.",
            "Unmatched/partial fields and dead heats are excluded from this one-winner comparison.",
            "Residual disagreements require independent official-source review; no labels were inferred from prices or settlement."]}
    with (OUTPUT / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
