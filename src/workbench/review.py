"""Persistent review decisions and validated correction proposals."""
from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from src.workbench.data import fingerprint, records

RACE_FIELDS = {"going", "surface", "distance_furlongs", "prize_money", "race_class", "race_type"}
RUNNER_FIELDS = {"jockey", "trainer", "weight_lbs", "draw", "official_rating", "age", "headgear", "finish_position", "won", "finish_pos_label", "lengths_behind"}
NUMBERS = {"distance_furlongs", "prize_money", "weight_lbs", "draw", "official_rating", "age", "finish_position", "won", "lengths_behind"}
INTEGERS = {"draw", "official_rating", "age", "finish_position", "won"}


class ReviewStore:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.executescript("""
            CREATE TABLE IF NOT EXISTS notes(race_id TEXT PRIMARY KEY,note TEXT NOT NULL,bookmarked INTEGER NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS decisions(issue_id TEXT PRIMARY KEY,state TEXT NOT NULL,note TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY,race_id TEXT NOT NULL,status TEXT NOT NULL,body TEXT NOT NULL,created_at TEXT NOT NULL);
        """)
        try:
            with con:
                yield con
        finally:
            con.close()

    def read(self):
        if not self.path.exists():
            return {"notes": [], "decisions": [], "proposals": []}
        with self.connect() as con:
            return {"notes": [dict(r) for r in con.execute("SELECT * FROM notes ORDER BY updated_at DESC LIMIT 500")],
                    "decisions": [dict(r) for r in con.execute("SELECT * FROM decisions LIMIT 2000")],
                    "proposals": [dict(r) for r in con.execute("SELECT id,race_id,status,created_at FROM proposals ORDER BY created_at DESC LIMIT 100")]}

    def note(self, race_id, note, bookmarked):
        if not race_id or len(note) > 10000:
            raise ValueError("A race ID and a note of at most 10,000 characters are required")
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO notes VALUES (?,?,?,?)", (race_id, note, int(bool(bookmarked)), datetime.now(timezone.utc).isoformat()))
        return {"saved": True}

    def decide(self, issue_id, state, note):
        if not issue_id or state not in {"open", "acknowledged", "investigating"} or len(note) > 10000:
            raise ValueError("Invalid review decision")
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)", (issue_id, state, note, datetime.now(timezone.utc).isoformat()))
        return {"saved": True}

    def propose(self, catalog, race_id, changes, reason, evidence, expected_fingerprint):
        if not reason.strip() or not evidence.strip() or len(reason)+len(evidence) > 20000:
            raise ValueError("A reason and source evidence are required")
        if not isinstance(changes, list) or not 1 <= len(changes) <= 100:
            raise ValueError("Provide between one and 100 field changes")
        frame = catalog.canonical(race_id)
        original = records(frame)
        if fingerprint(original) != expected_fingerprint:
            raise ValueError("Race changed while being reviewed; reload it before proposing a correction")
        corrected, preview = frame.copy(), []
        for change in changes:
            field, value = change.get("field"), change.get("value")
            if field not in RACE_FIELDS | RUNNER_FIELDS:
                raise ValueError("Field is not supported for correction proposals")
            if field in NUMBERS and value is not None:
                value = float(value)
                if not pd.notna(value) or abs(value) == float("inf"):
                    raise ValueError("Correction must be a finite number")
                if field in INTEGERS:
                    if value != int(value):
                        raise ValueError("This field requires an integer")
                    value = int(value)
            if value is not None and field not in NUMBERS and (not isinstance(value, str) or len(value) > 1000):
                raise ValueError("Text correction is invalid")
            if field in RACE_FIELDS:
                mask = pd.Series(True, index=corrected.index)
            else:
                identity = str(change.get("horse_id", ""))
                mask = corrected.horse_id.astype(str).eq(identity)
                if mask.sum() != 1:
                    raise ValueError("Runner corrections require a unique canonical horse ID")
            preview.extend({"horse_id": row.get("horse_id"), "horse_name": row.get("horse_name"), "field": field,
                            "before": row.get(field), "after": value} for row in records(corrected.loc[mask]))
            corrected.loc[mask, field] = value
        findings = catalog.validate_race(corrected)
        proposal_id = uuid4().hex
        body = {"id": proposal_id, "race_id": race_id, "reason": reason, "evidence": evidence,
                "base_fingerprint": expected_fingerprint, "changes": preview, "before": original,
                "after": records(corrected), "findings": findings,
                "validation_passed": not any(f["severity"] == "error" for f in findings),
                "application": "Review proposal only. Canonical data is unchanged; apply through the project's targeted repair owner after review, then publish a new dataset."}
        with self.connect() as con:
            con.execute("INSERT INTO proposals VALUES (?,?,?,?,?)", (proposal_id, race_id, "proposed", json.dumps(body, allow_nan=False), datetime.now(timezone.utc).isoformat()))
        return body

    def proposal(self, identifier):
        with self.connect() as con:
            row = con.execute("SELECT body FROM proposals WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Correction proposal was not found")
        return json.loads(row[0])
