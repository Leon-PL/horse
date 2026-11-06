"""Reconcile overfull result fields against the local public Betfair archive.

Default: produce a repair plan without changing results. --apply removes only
extra non-winning rows when the two independent feeds agree on the complete
starter field size and winners. All original race rows are backed up first.
No network access, feature rebuild, model fitting or promotion occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.betfair import bf_horse_key, bf_track_key
from src.utils import normalise_off_time_key

RESULT_COLUMNS = ["race_id", "race_date", "track", "off_time", "horse_name",
                  "num_runners", "finish_position", "won"]
KEYS = ["race_date", "track_key", "off_key"]


def keyed(frame):
    frame = frame.copy()
    frame["race_id"] = frame["race_id"].astype(str) if "race_id" in frame else ""
    frame["race_date"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["track_key"] = frame["track"].map(bf_track_key)
    frame["off_key"] = frame["off_time"].map(normalise_off_time_key)
    frame["horse_key"] = frame["horse_name"].map(bf_horse_key)
    return frame


def fingerprint(frame):
    columns = sorted(RESULT_COLUMNS)
    stable = frame[columns].copy().sort_values("horse_name").reset_index(drop=True)
    for col in ("num_runners", "finish_position", "won"):
        stable[col] = pd.to_numeric(stable[col], errors="coerce").astype(float)
    stable["race_id"] = stable["race_id"].astype(str)
    data = stable.to_json(orient="records", double_precision=15)
    return hashlib.sha256(data.encode()).hexdigest()


def repair_plan(results, reference):
    """Conservative, race-level evidence; missing or ambiguous data abstains."""
    results, reference = keyed(results), keyed(reference)
    count = results.groupby("race_id")["race_id"].transform("size")
    reported = pd.to_numeric(results.num_runners, errors="coerce")
    affected = results.loc[count > reported].race_id.unique()
    reference_groups = {key: group for key, group in reference.groupby(KEYS, sort=False)}
    ambiguous = results.groupby(KEYS).race_id.nunique()
    ambiguous = set(ambiguous[ambiguous > 1].index)
    repairs, skipped = [], Counter()
    groups = results[results.race_id.isin(affected)].groupby("race_id", sort=False)
    for race_id, race in groups:
        keys = race[KEYS].drop_duplicates()
        if len(keys) != 1 or keys.isna().any().any():
            skipped["ambiguous_race_metadata"] += 1
            continue
        key = tuple(keys.iloc[0])
        field = reference_groups.get(key)
        if key in ambiguous or field is None:
            skipped["no_unique_reference_race"] += 1
            continue
        counts = pd.to_numeric(race.num_runners, errors="coerce")
        if counts.isna().any() or counts.nunique() != 1 or counts.iloc[0] != len(field):
            skipped["field_sizes_do_not_agree"] += 1
            continue
        if "event_id" not in field or field.event_id.isna().any() or field.event_id.nunique() != 1:
            skipped["ambiguous_reference_event"] += 1
            continue
        if (race.horse_key.eq("").any() or field.horse_key.eq("").any()
                or race.horse_key.duplicated().any() or field.horse_key.duplicated().any()):
            skipped["ambiguous_horse_names"] += 1
            continue
        labels = pd.to_numeric(field.won, errors="coerce")
        source_labels = pd.to_numeric(race.won, errors="coerce")
        if not labels.isin([0, 1]).all() or not source_labels.isin([0, 1]).all() or labels.sum() < 1:
            skipped["invalid_winner_labels"] += 1
            continue
        if not set(field.horse_key) <= set(race.horse_key):
            skipped["reference_runner_missing_from_results"] += 1
            continue
        winners = set(field.loc[labels.eq(1), "horse_key"])
        if winners != set(race.loc[source_labels.eq(1), "horse_key"]):
            skipped["winner_disagreement"] += 1
            continue
        keep = race.horse_key.isin(field.horse_key)
        positions = pd.to_numeric(race.finish_position, errors="coerce")
        if not positions.eq(1).equals(source_labels.eq(1)):
            skipped["result_label_disagreement"] += 1
            continue
        extras = race.loc[~keep]
        if extras.empty or extras.won.ne(0).any():
            skipped["unsafe_extra_rows"] += 1
            continue
        repairs.append({"race_id": race_id, "race_date": key[0],
                        "track_key": key[1], "off_key": key[2],
                        "event_id": str(field.event_id.iloc[0]),
                        "before": len(race), "after": len(field),
                        "remove": extras.horse_name.tolist(),
                        "source_fingerprint": fingerprint(race)})
    return {"affected_races": len(affected), "repairable_races": len(repairs),
            "rows_to_remove": sum(len(r["remove"]) for r in repairs),
            "skipped": dict(skipped), "repairs": repairs}


def apply_plan(conn, plan, backup):
    """Optimistically validate, durably back up, then commit all deletions."""
    if Path(backup).exists():
        raise FileExistsError(f"Refusing to overwrite backup: {backup}")
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        all_rows = []
        for repair in plan["repairs"]:
            race = pd.read_sql_query("SELECT * FROM results WHERE race_id = ?", conn,
                                     params=[repair["race_id"]])
            if fingerprint(race) != repair["source_fingerprint"]:
                raise RuntimeError(f"Race {repair['race_id']} changed after planning; no repair applied")
            if len(set(repair["remove"])) != repair["before"] - repair["after"]:
                raise ValueError("Invalid repair size")
            targets = race[race.horse_name.isin(repair["remove"])]
            if len(targets) != len(repair["remove"]) or not targets.won.eq(0).all():
                raise ValueError("Unsafe deletion target")
            all_rows.extend(race.to_dict(orient="records"))
        Path(backup).parent.mkdir(parents=True, exist_ok=True)
        with open(backup, "x", encoding="utf-8") as fh:
            json.dump(all_rows, fh, default=str, allow_nan=True)
            fh.flush()
            os.fsync(fh.fileno())
        for repair in plan["repairs"]:
            from src.identity_schema import result_rowid
            conn.executemany("DELETE FROM results WHERE rowid = ?",
                             [(result_rowid(conn,{"race_id":repair["race_id"],"horse_name":name}),) for name in repair["remove"]])
            actual = conn.execute("SELECT COUNT(*) FROM results WHERE race_id = ?",
                                   (repair["race_id"],)).fetchone()[0]
            if actual != repair["after"]:
                raise RuntimeError("Repair count mismatch; transaction rolled back")
    return len(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/races.db")
    parser.add_argument("--reference", default="data/processed/betfair_sp_public.parquet")
    parser.add_argument("--output", default="data/audit/20261025/field_reconciliation")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not Path(args.db).is_file():
        parser.error(f"Results database does not exist: {args.db}")
    with sqlite3.connect(args.db) as conn:
        print("Loading result keys...", flush=True)
        results = pd.read_sql_query(f"SELECT {', '.join(RESULT_COLUMNS)} FROM results", conn)
        reference = pd.read_parquet(args.reference, columns=["race_date", "track", "off_time",
                                                            "horse_name", "won", "event_id"])
        print("Reconciling independent fields...", flush=True)
        plan = repair_plan(results, reference)
        plan["reference_path"] = str(Path(args.reference).resolve())
        plan["applied"] = False
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in plan.items() if k != "repairs"}), flush=True)
        if args.apply and plan["repairs"]:
            apply_plan(conn, plan, folder / "original_races.json")
            plan["applied"] = True
            (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            print("Repair committed. Export full results, then rebuild features before training.", flush=True)


if __name__ == "__main__":
    main()
