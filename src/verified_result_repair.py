"""Transactional complete-field repair with original-row receipts and rollback.

Inputs are independently reviewed SL/RTV full-field snapshots. Factual event and
horse identities remain fixed; only source-confirmed nonstarters can be removed.
No model fitting, store replacement or source collection happens here.
"""
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from src.runner_identity import runner_index, runner_keys
from src.identity_schema import ensure_repair_keys, result_rowid

TARGETS = ("finish_position", "finish_pos_label", "won", "lengths_behind")
KEYS = ["race_id", "horse_name"]


def _ordered(frame):
    out = frame.copy()
    out["race_id"] = out.race_id.astype(str)
    if not runner_index(out).is_unique:
        raise ValueError("Duplicate repair identity")
    out["_runner_key"] = runner_keys(out)
    return out.sort_values([*KEYS, "_runner_key"]).drop(columns="_runner_key").reset_index(drop=True)


def _indexed(frame):
    out = frame.copy()
    out.index = runner_index(out)
    return out


def _rows(frame):
    return [{k: None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
            for row in frame.to_dict("records")]


def _current(connection, ids):
    return pd.read_sql_query("SELECT * FROM results WHERE race_id IN ("+",".join("?" for _ in ids)+")", connection, params=list(ids))


def validate(before, after, additional, evidence):
    before, after = _ordered(before), _ordered(after)
    if set(before.race_id) != set(after.race_id) or set(before) != set(after):
        raise ValueError("Repair must include complete original/replacement fields")
    indexed = _indexed(before)
    remaining = _indexed(after)
    if not remaining.index.isin(indexed.index).all():
        raise ValueError("Existing repair cannot invent starters")
    ordinary = [c for c in before if c not in TARGETS]
    pd.testing.assert_frame_equal(indexed.loc[remaining.index, ordinary], remaining[ordinary], check_dtype=False, check_exact=True)
    removed = indexed.loc[~indexed.index.isin(remaining.index)]
    approved_removals = {tuple(x) for x in evidence.get("confirmed_nonstarters", [])}
    approved_ids = {(str(race), "sl:"+str(horse)) for race,horse in evidence.get("confirmed_nonstarter_ids", [])}
    name_counts = before.groupby(KEYS).size()
    approved = all(key in approved_ids or ((str(row.race_id),row.horse_name) in approved_removals
        and name_counts.loc[(row.race_id,row.horse_name)] == 1) for key,row in removed.iterrows())
    if (not approved or not removed.finish_position.eq(0).all() or not removed.won.eq(0).all()):
        raise ValueError("Nonstarter removal lacks exact reviewed evidence")
    if not additional.empty and set(additional.race_id.astype(str)) & set(before.race_id):
        raise ValueError("New and corrected fields overlap")
    for frame in (after, additional):
        if frame.empty:
            continue
        for race_id, group in frame.groupby("race_id"):
            positions = pd.to_numeric(group.finish_position, errors="raise")
            if (not group.num_runners.eq(len(group)).all() or positions.isna().any() or positions.lt(0).any()
                    or positions.mod(1).ne(0).any() or not group.won.eq(positions.eq(1).astype(int)).all()
                    or group.won.sum() < 1 or group.horse_id.isna().any() or group.horse_id.duplicated().any()):
                raise ValueError("Invalid complete verified field: "+str(race_id))
    for name in ("review_roster_sha256", "verification_sha256"):
        if len(evidence.get(name, "")) != 64:
            raise ValueError("Missing reviewed source provenance")
    return before, after


def apply_verified_results(connection, before, after, additional, *, repair_id, evidence, manage_transaction=True):
    before, after = validate(before, after, additional, evidence)
    if connection.in_transaction == manage_transaction:
        raise ValueError("Repair transaction ownership mismatch")
    additional = additional.copy()
    if not additional.empty:
        additional["race_id"] = additional.race_id.astype(str)
    content = json.dumps({"before": _rows(before), "after": _rows(after), "additional": _rows(additional), "evidence": evidence}, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(content.encode()).hexdigest()
    with connection if manage_transaction else nullcontext():
        if manage_transaction:
            connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE IF NOT EXISTS verified_result_batches (repair_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL, observed_at TEXT NOT NULL, evidence_json TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS verified_result_rows (repair_id TEXT, race_id TEXT, horse_name TEXT, before_json TEXT, after_json TEXT, PRIMARY KEY (repair_id,race_id,horse_name))")
        ensure_repair_keys(connection, "verified_result_rows", "before_json", "after_json")
        prior = connection.execute("SELECT content_sha256 FROM verified_result_batches WHERE repair_id=?", (repair_id,)).fetchone()
        if prior:
            if prior[0] != digest:
                raise ValueError("Repair ID belongs to different evidence")
            pd.testing.assert_frame_equal(_ordered(_current(connection, after.race_id.unique()))[after.columns], after, check_dtype=False, check_exact=True)
            if not additional.empty:
                actual_new = _current(connection, additional.race_id.unique())
                if len(actual_new) != len(additional) or set(runner_index(actual_new)) != set(runner_index(additional)):
                    raise ValueError("A previously inserted field changed")
            return {"already_applied": True, "content_sha256": digest}
        pd.testing.assert_frame_equal(_ordered(_current(connection, before.race_id.unique()))[before.columns], before, check_dtype=False, check_exact=True)
        if not additional.empty and not _current(connection, additional.race_id.unique()).empty:
            raise ValueError("A missing field appeared after review")
        connection.execute("INSERT INTO verified_result_batches VALUES (?,?,?,?)", (repair_id, digest, datetime.now(timezone.utc).isoformat(), json.dumps(evidence, sort_keys=True)))
        old = _indexed(before)
        new = _indexed(after)
        for key, record in old.iterrows():
            replacement = new.loc[[key]] if key in new.index else pd.DataFrame()
            connection.execute("INSERT INTO verified_result_rows (repair_id,race_id,horse_name,before_json,after_json,runner_key) VALUES (?,?,?,?,?,?)",
                (repair_id, record.race_id, record.horse_name, json.dumps(_rows(old.loc[[key]])[0], sort_keys=True),
                 json.dumps(_rows(replacement)[0], sort_keys=True) if not replacement.empty else None, key[1]))
            rowid = result_rowid(connection, record)
            if replacement.empty:
                connection.execute("DELETE FROM results WHERE rowid=?", (rowid,))
            else:
                values = [replacement.iloc[0][c] for c in TARGETS]
                values = [None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v for v in values]
                connection.execute("UPDATE results SET finish_position=?,finish_pos_label=?,won=?,lengths_behind=? WHERE rowid=?", (*values, rowid))
        if not additional.empty:
            from src.database import _insert_results
            _insert_results(connection, additional)
            if "scraped_at" in additional:
                if pd.to_datetime(additional.scraped_at, errors="coerce").isna().any():
                    raise ValueError("Invalid reviewed source observation timestamp")
                connection.executemany("UPDATE results SET scraped_at=? WHERE rowid=?",
                    [(str(row["scraped_at"]), result_rowid(connection,row)) for row in additional.to_dict("records")])
            for row in _rows(additional):
                connection.execute("INSERT INTO verified_result_rows (repair_id,race_id,horse_name,before_json,after_json,runner_key) VALUES (?,?,?,?,?,?)",
                    (repair_id, row["race_id"], row["horse_name"], None, json.dumps(row, sort_keys=True), runner_keys(pd.DataFrame([row])).iloc[0]))
        completed_ids = list(after.race_id.unique()) + (list(additional.race_id.unique()) if not additional.empty else [])
        connection.executemany("UPDATE result_sync_races SET completed=1,last_error=NULL WHERE race_id=?", [(str(x),) for x in completed_ids])
        actual = _ordered(_current(connection, after.race_id.unique()))
        pd.testing.assert_frame_equal(actual[after.columns], after, check_dtype=False, check_exact=True)
    return {"already_applied": False, "content_sha256": digest, "repaired_fields": int(after.race_id.nunique()),
        "repaired_field_rows": len(after), "removed_nonstarters": len(before)-len(after),
        "new_fields": int(additional.race_id.nunique()) if not additional.empty else 0, "new_starters": len(additional)}
