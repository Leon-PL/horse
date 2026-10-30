"""Audit final-position disagreements and prepare a conservative repair batch.

No database mutation or prediction evaluation. Use complete unambiguous cached
ATR official orders, and require complete agreeing exchange winner labels.
Malformed/partial/contradictory fields remain quarantined with reasons.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from src.betfair import attach_bsp
from src.database import DB_PATH
from src.deployment_validation import file_sha256 as sha
from src.utils import normalise_horse_key, normalise_off_time_key, normalise_track_key

CASE = ROOT / "data/audit/result_amendments_042"
HISTORY = ROOT / "data/audit/live_feature_speed_20261029/installation/previous_store/processed.parquet"
ATR = ROOT / "data/atr_cache/atr_sectionals.parquet"
EXCHANGE = ROOT / "data/processed/betfair_bsp.parquet"
KEY = ["_date", "_track", "_time", "_horse"]
EVENT = KEY[:3]


def keys(frame):
    frame = frame.copy()
    frame["_date"] = pd.to_datetime(frame.race_date, errors="raise").dt.strftime("%Y-%m-%d")
    frame["_track"] = frame.track.map(normalise_track_key)
    frame["_time"] = frame.off_time.map(normalise_off_time_key)
    frame["_horse"] = frame.horse_name.map(normalise_horse_key)
    return frame


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    if (CASE / "audit.json").exists():
        raise ValueError("Preserve the completed source audit")
    source_hashes = {name: sha(ROOT / name) for name in ["scripts/audit_result_amendments.py", "src/data_scraper.py",
        "src/database.py", "src/betfair.py", "src/utils.py"]}
    plan = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_contract": "Same date/course/time/horse keys; full identical starter sets; original/final orders each a complete1..Npermutation; one ATR event; complete agreeing exchange winner labels.",
        "repair_policy": "Final official positions/win labels; withhold physical margins on reordered fields, new winner origin0. Keep all original rows and evidence. Never infer final orders from commentary or partial fields.",
        "source_sha256": source_hashes,
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [HISTORY, ATR, EXCHANGE, CASE / "goodwood_source.json"]},
        "scope": "Source integrity/maintenance through28October2026; no predictive score, no reserved-period model evaluation, no canonical mutation."}
    write(CASE / "audit_plan.json", plan)
    columns = ["race_id", "horse_id", "horse_name", "race_date", "track", "off_time", "num_runners",
               "finish_position", "finish_pos_label", "won", "lengths_behind"]
    history = keys(pd.read_parquet(HISTORY, columns=columns))
    history["race_id"] = history.race_id.astype(str)
    if history.duplicated(KEY).any():
        raise ValueError("Ambiguous canonical result keys")
    cache = keys(pd.read_parquet(ATR, columns=["race_date", "track", "off_time", "horse_name", "atr_raceid",
        "atr_official_position", "atr_in_running_comment"], filters=[("race_date", "<=", "2026-10-28")]))
    cache["source_position"] = pd.to_numeric(cache.atr_official_position, errors="coerce")
    relevant = cache[KEY + ["atr_raceid", "source_position"]].drop_duplicates()
    conflicting = relevant.duplicated(KEY, keep=False)
    bad_keys = pd.MultiIndex.from_frame(relevant.loc[conflicting, KEY])
    cache = cache.loc[~pd.MultiIndex.from_frame(cache[KEY]).isin(bad_keys)].drop_duplicates(KEY)
    cache["source_event_size"] = cache.groupby(EVENT).horse_name.transform("size")
    cache["source_event_ids"] = cache.groupby(EVENT).atr_raceid.transform("nunique")
    joined = history.merge(cache[KEY + ["source_position", "atr_raceid", "source_event_size", "source_event_ids", "atr_in_running_comment"]],
                           on=KEY, how="left", validate="one_to_one")
    mismatch = joined.source_position.notna() & joined.source_position.ne(joined.finish_position)
    ids = joined.loc[mismatch, "race_id"].unique()
    fields = joined.loc[joined.race_id.isin(ids)].copy()
    book = pd.read_parquet(EXCHANGE).rename(columns={"won": "exchange_won"})
    fields = attach_bsp(fields, bsp=book, cols=("exchange_won",))
    decisions, eligible = [], []
    for race_id, group in fields.groupby("race_id", sort=False):
        n = len(group)
        order = list(range(1, n + 1))
        reason = "eligible"
        if group.num_runners.nunique() != 1 or not group.num_runners.eq(n).all():
            reason = "canonical_partial_field"
        elif group.source_position.isna().any() or not group.source_event_size.eq(n).all():
            reason = "source_partial_or_different_starters"
        elif not group.source_event_ids.eq(1).all() or group.atr_raceid.nunique() != 1:
            reason = "source_ambiguous_event"
        elif sorted(group.finish_position.tolist()) != order or sorted(group.source_position.tolist()) != order:
            reason = "not_two_complete_unique_orders"
        elif not np.array_equal(group.won.to_numpy(), group.finish_position.eq(1).to_numpy()):
            reason = "canonical_win_label_inconsistent"
        elif group.exchange_won.isna().any() or not group.exchange_won.isin([0, 1]).all() or group.exchange_won.sum() != 1:
            reason = "exchange_winner_evidence_incomplete"
        elif not np.array_equal(group.source_position.eq(1).to_numpy(), group.exchange_won.to_numpy()):
            reason = "official_order_vs_exchange_winner_conflict"
        winner_changed = bool(not group.finish_position.eq(1).equals(group.source_position.eq(1)))
        decisions.append({"race_id": race_id, "race_date": str(group.race_date.iloc[0])[:10],
            "track": group.track.iloc[0], "off_time": group.off_time.iloc[0], "runners": n,
            "position_changes": int(group.finish_position.ne(group.source_position).sum()),
            "winner_changed": winner_changed, "decision": reason})
        if reason == "eligible":
            eligible.append(group)
    decision = pd.DataFrame(decisions)
    decision.to_csv(CASE / "field_decisions.csv", index=False)
    fields.to_parquet(CASE / "source_disagreements.parquet", index=False)
    proposed = pd.concat(eligible, ignore_index=True) if eligible else fields.iloc[:0].copy()
    proposed["corrected_finish_position"] = proposed.source_position.astype("Int64")
    proposed["corrected_won"] = proposed.source_position.eq(1).astype(int)
    proposed["corrected_lengths_behind"] = np.where(proposed.corrected_won.eq(1), 0., np.nan)
    proposed.to_parquet(CASE / "proposed_repairs.parquet", index=False)
    # Compare the entire eligible field with the current DB before proposing a
    # transaction. IDs are scoped; never load or rewrite unrelated current data.
    with sqlite3.connect(f"file:{Path(DB_PATH).as_posix()}?mode=ro", uri=True) as connection:
        rows = []
        for race_id in proposed.race_id.unique():
            rows.extend(pd.read_sql_query("SELECT * FROM results WHERE race_id=?", connection, params=[str(race_id)]).to_dict("records"))
    stored = pd.DataFrame(rows)
    if len(stored):
        stored["race_id"] = stored.race_id.astype(str)
        # Processing standardises horse-name spelling/capitalisation. Stable
        # source IDs, not display spelling, identify rows in the source DB.
        for part in (stored, proposed):
            ids = pd.to_numeric(part.horse_id, errors="raise")
            if ids.isna().any() or ids.le(0).any() or ids.mod(1).ne(0).any():
                raise ValueError("Historical repair needs valid source horse IDs")
            part["_source_horse_id"] = ids.astype("int64")
        comparison = proposed.merge(stored, on=["race_id", "_source_horse_id"], suffixes=("_frozen", "_db"), how="outer", validate="one_to_one", indicator=True)
        if not comparison.pop("_merge").eq("both").all():
            raise ValueError("Current DB and frozen eligible field identities differ")
        if not comparison.horse_name_frozen.map(normalise_horse_key).equals(comparison.horse_name_db.map(normalise_horse_key)):
            raise ValueError("Source ID matched a different normalized horse name")
        for column in ("finish_position", "won", "lengths_behind"):
            np.testing.assert_array_equal(comparison[column + "_frozen"], comparison[column + "_db"])
        stored.to_parquet(CASE / "before_db_rows.parquet", index=False)
    if any(sha(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("Source changed during audit")
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "canonical_rows": len(history),
        "cache_rows": len(cache), "conflicting_source_keys": int(conflicting.sum()),
        "matched_comment_rows": int(joined.atr_in_running_comment.fillna("").str.strip().ne("").sum()),
        "position_disagreement_rows": int(mismatch.sum()), "disagreement_fields": len(decision),
        "decisions": {str(k): int(v) for k, v in decision.decision.value_counts().items()},
        "eligible_fields": int(proposed.race_id.nunique()), "eligible_field_runners": len(proposed),
        "eligible_position_changes": int(proposed.finish_position.ne(proposed.source_position).sum()),
        "eligible_changed_winners": int(decision.loc[decision.decision.eq("eligible"), "winner_changed"].sum()),
        "current_db_rows_match": True, "canonical_rows_modified": 0,
        "earliest_repair_date": str(proposed.race_date.min())[:10] if len(proposed) else None,
        "source_sha256": source_hashes, "plan_sha256": sha(CASE / "audit_plan.json"),
        "limitations": ["No claim that unmatched or unsupported amended races are correct; quarantine is preserved.",
                        "Exchange winner evidence corroborates the final winner, not every intermediate placing.",
                        "ATR official positions are retrospective; actual publication times are not recovered."]}
    write(CASE / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
