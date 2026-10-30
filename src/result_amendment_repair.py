"""Conservative, transactional maintenance of source-verified final placings.

This does not engineer features or choose model settings. Original complete
rows are retained in the same transaction as the correction. All existing
rows must still match the reviewed snapshot; a partial or changed field aborts
the entire batch, rather than making a best-effort subset of repairs.
"""
from datetime import datetime, timezone
from contextlib import nullcontext
import hashlib
import json

import numpy as np
import pandas as pd

from src.utils import normalise_horse_key, normalise_off_time_key, normalise_track_key
from src.result_status import NONSTARTER_LABELS, result_labels

KEY = ["race_id", "_horse_id"]
TARGETS = ["finish_position", "finish_pos_label", "won", "lengths_behind"]


def _identified(frame):
    frame = frame.drop(columns="_source_horse_id", errors="ignore").copy()
    frame["race_id"] = frame.race_id.astype(str)
    ids = pd.to_numeric(frame.horse_id, errors="raise")
    if ids.isna().any() or ids.le(0).any() or ids.mod(1).ne(0).any():
        raise ValueError("Amendments require valid source horse IDs")
    frame["_horse_id"] = ids.astype("int64")
    if frame.duplicated(KEY).any() or frame.duplicated(["race_id", "horse_name"]).any():
        raise ValueError("Duplicate correction identity")
    return frame.sort_values(KEY).reset_index(drop=True)


def corrected_rows(before, proposed):
    """Validate full source fields and return only the reviewed replacements."""
    before, proposed = _identified(before), _identified(proposed)
    if before.empty or not before[KEY].equals(proposed[KEY]):
        raise ValueError("Correction must include the identical complete starter population")
    for col in ("finish_position", "won", "lengths_behind"):
        np.testing.assert_array_equal(before[col], proposed[col])
    if not before.horse_name.map(normalise_horse_key).equals(proposed.horse_name.map(normalise_horse_key)):
        raise ValueError("Source horse IDs disagree with horse names")
    for col, normalize in (("track", normalise_track_key), ("off_time", normalise_off_time_key)):
        if not before[col].map(normalize).equals(proposed[col].map(normalize)):
            raise ValueError("Source event metadata disagrees")
    if not pd.to_datetime(before.race_date).equals(pd.to_datetime(proposed.race_date)):
        raise ValueError("Source dates disagree")
    for race_id, indexes in before.groupby("race_id", sort=False).groups.items():
        group = proposed.loc[indexes]
        n = len(group)
        order = list(range(1, n + 1))
        if not group.num_runners.eq(n).all():
            raise ValueError("Correction source is not a complete field")
        provider = group.get("source_provider", pd.Series("atr_cache", index=group.index)).fillna("atr_cache")
        if provider.nunique() != 1:
            raise ValueError("Mixed official providers within one field")
        if provider.iloc[0] == "atr_cache":
            if sorted(group.finish_position) != order or sorted(group.source_position) != order:
                raise ValueError("Only two complete unique finishing orders can be repaired automatically")
            if group.atr_raceid.isna().any() or group.atr_raceid.nunique() != 1:
                raise ValueError("Ambiguous official source event")
        elif provider.iloc[0] == "sporting_life_amendment":
            if (not group.source_event_id.astype(str).eq(str(race_id)).all()
                    or not group.source_stage.eq("WEIGHEDIN").all()
                    or not group.source_receipt_sha256.fillna("").str.fullmatch(r"[a-f0-9]{64}").all()
                    or pd.to_datetime(group.source_observed_at, utc=True, format="ISO8601", errors="coerce").isna().any()):
                raise ValueError("Missing settled official-source provenance")
            if not group.source_original_finish_position.eq(group.finish_position).all():
                raise ValueError("Stored order differs from the observed original source order")
            if group.result_amendment_json.isna().any():
                raise ValueError("Official source supplies no amendment evidence")
            # Full started fields can contain genuine non-finishers at0.
            # Positive finishers still need a complete, unique ranking.
            for physical, positions in ((True, group.finish_position), (False, group.source_position)):
                positive = positions.loc[positions.gt(0)].tolist()
                ordinary = sorted(positive) == list(range(1, len(positive)+1))
                # The provider sometimes puts a disqualified first-past-post
                # winner last among ALL starters, including non-finishers.
                # Its explicit final position is N, leaving a legitimate gap.
                demoted = group.source_position.eq(n) & group.finish_position.eq(1)
                last_of_starters = (not physical and len(positive) < n and demoted.sum() == 1
                    and sorted(positive) == [*range(1, len(positive)), n])
                if (positions.isna().any() or positions.lt(0).any() or positions.mod(1).ne(0).any()
                        or not (ordinary or last_of_starters)):
                    raise ValueError("Unsupported official finishing order")
            nf = group.source_position.eq(0)
            if group.loc[nf, "source_finish_pos_label"].fillna("").str.upper().isin(NONSTARTER_LABELS).any():
                raise ValueError("A nonstarter cannot enter a complete started result field")
        else:
            raise ValueError("Unreviewed official-source provider")
        if not group.source_position.eq(1).equals(group.exchange_won.eq(1)):
            raise ValueError("Final winner disagrees with exchange evidence")
        if not group.exchange_won.isin([0, 1]).all() or group.exchange_won.sum() != 1:
            raise ValueError("Incomplete exchange winner evidence")
        if not group.won.eq(1).equals(group.finish_position.eq(1)):
            raise ValueError("Original labels are inconsistent")
        if group.finish_position.eq(group.source_position).all():
            raise ValueError("An unchanged field is outside this amendment batch")
    after = before.copy()
    after["finish_position"] = proposed.source_position.astype("int64")
    after["finish_pos_label"] = ""
    nonfinish = after.finish_position.eq(0)
    if nonfinish.any():
        # Use the ordinary source contract: position0 in a complete, settled
        # started field is DNF when its particular reason was not reported.
        observed = pd.DataFrame({"race_id": proposed.race_id, "finish_position": proposed.source_position,
            "num_runners": proposed.num_runners, "finish_pos_label": proposed.source_finish_pos_label})
        after.loc[nonfinish, "finish_pos_label"] = result_labels(observed).loc[nonfinish]
    after["won"] = after.finish_position.eq(1).astype("int64")
    # Arrival gaps cannot be reassigned to a new finishing sequence.
    after["lengths_behind"] = np.where(after.won.eq(1), 0., np.nan)
    return after.drop(columns="_horse_id")


def _json_rows(frame):
    # Keep full round-trip float precision in the original-row receipt.
    return [{key: None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
             for key, value in row.items()} for row in frame.to_dict("records")]


def _current_rows(connection, identities):
    ids = identities.race_id.unique().tolist()
    return pd.read_sql_query("SELECT * FROM results WHERE race_id IN (" + ",".join("?" for _ in ids) + ")",
                             connection, params=ids)


def apply_amendments(connection, before, proposed, *, repair_id, evidence, manage_transaction=True):
    """Apply once, or verify a prior identical application without writing.

    By default the connection must be outside a transaction. A caller can own
    the transaction explicitly when coupling DB, CSV, store and model commits;
    then it must commit/roll back the complete operation itself. Unrelated source
    updates cannot slip between checking the snapshot and committing the batch.
    No deletion, fitting, store publication or production promotion occurs.
    """
    if manage_transaction and connection.in_transaction:
        raise ValueError("Amendment repair must own its transaction")
    if not manage_transaction and not connection.in_transaction:
        raise ValueError("Caller-managed amendment repair requires an active transaction")
    expected = _identified(before).drop(columns="_horse_id")
    after = corrected_rows(before, proposed)
    manifest = {"repair_id": str(repair_id), "evidence": evidence,
                "before": _json_rows(expected), "after": _json_rows(after)}
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    with connection if manage_transaction else nullcontext():
        if manage_transaction:
            connection.execute("BEGIN IMMEDIATE")
        connection.execute("""CREATE TABLE IF NOT EXISTS result_amendment_batches (
            repair_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL,
            observed_at TEXT NOT NULL, evidence_json TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS result_amendment_rows (
            repair_id TEXT NOT NULL, race_id TEXT NOT NULL, horse_name TEXT NOT NULL,
            original_row_json TEXT NOT NULL, corrected_values_json TEXT NOT NULL,
            PRIMARY KEY (repair_id, race_id, horse_name))""")
        applied = connection.execute("SELECT content_sha256 FROM result_amendment_batches WHERE repair_id=?",
                                     (str(repair_id),)).fetchone()
        if applied and applied[0] != digest:
            raise ValueError("This repair ID already records a different evidence batch")
        current = _current_rows(connection, expected)
        current = _identified(current).drop(columns="_horse_id")
        target = after if applied else expected
        if set(current) != set(target):
            raise ValueError("Canonical result schema changed after the source audit")
        pd.testing.assert_frame_equal(current[target.columns], target, check_dtype=False, check_exact=True)
        if applied:
            return {"already_applied": True, "updated_rows": 0, "fields": int(after.race_id.nunique()),
                    "content_sha256": digest}
        observed = datetime.now(timezone.utc).isoformat()
        connection.execute("INSERT INTO result_amendment_batches VALUES (?,?,?,?)", (str(repair_id), digest,
                           observed, json.dumps(evidence, sort_keys=True, allow_nan=False)))
        original = _json_rows(expected)
        changed = _json_rows(after[["race_id", "horse_name", *TARGETS]])
        for old, new in zip(original, changed):
            connection.execute("INSERT INTO result_amendment_rows VALUES (?,?,?,?,?)", (str(repair_id),
                old["race_id"], old["horse_name"], json.dumps(old, sort_keys=True, allow_nan=False),
                json.dumps(new, sort_keys=True, allow_nan=False)))
            values = [new[c] for c in TARGETS]
            cursor = connection.execute("""UPDATE results SET finish_position=?, finish_pos_label=?, won=?, lengths_behind=?
                WHERE race_id=? AND horse_name=?""", (*values, old["race_id"], old["horse_name"]))
            if cursor.rowcount != 1:
                raise ValueError("Correction target changed; rolling back the entire batch")
        actual = _current_rows(connection, expected)
        actual = _identified(actual).drop(columns="_horse_id")
        pd.testing.assert_frame_equal(actual[after.columns], after, check_dtype=False, check_exact=True)
    return {"already_applied": False, "updated_rows": len(after), "fields": int(after.race_id.nunique()),
            "content_sha256": digest, "observed_at": observed}
