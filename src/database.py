"""
Historical Database Module
==========================
SQLite-backed persistent storage for scraped race results: the schema, reads,
writes and the published ``data/raw/race_results.parquet``.

Collection lives in ``src.ingestion.results``, which records every fetch in the
ledger tables defined here, so a partial scrape can be resumed safely.

Usage::

    from src.database import load_from_database, db_stats

    # Load everything (or a window) as a DataFrame
    df = load_from_database()                # all records
    df = load_from_database(days_back=30)    # last 30 days only

    # Quick summary
    print(db_stats())
"""

import logging
import hashlib
import json
import os
import sys
import sqlite3
import tempfile
from pathlib import Path
from uuid import uuid4
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from src.ingestion.storage import atomic_json, atomic_parquet, read_json, writer_lock
from src.utils import normalise_horse_key
from src.runner_identity import horse_ids, runner_keys, runner_positions, runner_index
from src.identity_schema import rebuild_without_key, ensure_source_keys, ensure_repair_keys, adopt_stored_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(config.DATA_DIR, "races.db")

# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    race_id           TEXT    NOT NULL,
    race_date         TEXT    NOT NULL,
    off_time          TEXT,
    track             TEXT,
    region            TEXT,
    race_name         TEXT,
    race_class        TEXT,
    race_type         TEXT,
    distance_furlongs REAL,
    going             TEXT,
    prize_money       REAL,
    num_runners       INTEGER,
    horse_name        TEXT    NOT NULL,
    horse_id          TEXT,
    jockey            TEXT,
    trainer           TEXT,
    age               INTEGER,
    sex               TEXT,
    headgear          TEXT,
    wind_surgery      INTEGER,
    weight_lbs        REAL,
    reported_jockey_claim_lbs REAL,
    draw              INTEGER,
    form              TEXT,
    days_since_last_run INTEGER,
    odds              REAL,
    official_rating   INTEGER,
    finish_position   INTEGER,
    finish_pos_label  TEXT,
    won               INTEGER,
    lengths_behind    REAL,
    horse_runs        INTEGER,
    horse_wins        INTEGER,
    horse_places      INTEGER,
    surface           TEXT,
    handicap          INTEGER,
    sire              TEXT,
    dam               TEXT,
    damsire           TEXT,
    weather_temp_max  REAL,
    weather_temp_min  REAL,
    weather_precip_mm REAL,
    weather_wind_kmh  REAL,
    weather_precip_prev3 REAL,
    scraped_at        TEXT    DEFAULT (datetime('now'))
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_results_date ON results(race_date);
"""

# New source evidence is append-only and separate from the canonical model
# matrix. Adding optional metadata must not fabricate old observations or
# silently invalidate all existing feature histories by adding empty columns.
_CREATE_CONTEXT = """
CREATE TABLE IF NOT EXISTS result_context_observations (
    race_id TEXT NOT NULL, horse_name TEXT NOT NULL, horse_id TEXT,
    race_date TEXT NOT NULL, declared_weight_lbs REAL,
    reported_jockey_claim_lbs REAL, weight_penalty_lbs REAL,
    ride_description TEXT, nonfinish_reason TEXT, result_amendment_json TEXT,
    content_sha256 TEXT NOT NULL, observed_at TEXT NOT NULL,
    PRIMARY KEY (race_id, horse_name, content_sha256)
);
"""
_CONTEXT_FIELDS = ("reported_jockey_claim_lbs", "weight_penalty_lbs", "ride_description", "nonfinish_reason", "result_amendment_json")

# Pre-race text (Sporting Life runner commentary and race verdict), recorded
# every time it is seen on a racecard or a result page. A text feature may use
# it only once the card and result copies are shown to agree, i.e. the result
# page does not rewrite it after the race.
_CREATE_RUNNER_TEXT = """
CREATE TABLE IF NOT EXISTS runner_text_observations (
    race_id TEXT NOT NULL, horse_name TEXT NOT NULL, horse_id TEXT,
    race_date TEXT NOT NULL, phase TEXT NOT NULL,
    commentary TEXT, verdict TEXT,
    content_sha256 TEXT NOT NULL, observed_at TEXT NOT NULL
);
"""
_TEXT_FIELDS = {"sl_commentary": "commentary", "sl_verdict": "verdict"}

_CREATE_PEDIGREE = """
CREATE TABLE IF NOT EXISTS horse_pedigree (
    horse_id   TEXT PRIMARY KEY,
    horse_name TEXT,
    sire       TEXT,
    dam        TEXT,
    damsire    TEXT,
    foaled     TEXT,
    colour     TEXT,
    fetched_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_SYNC_DATES = """
CREATE TABLE IF NOT EXISTS result_sync_dates (
    race_date TEXT PRIMARY KEY,
    listing_succeeded INTEGER NOT NULL DEFAULT 0,
    listing_attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    last_error TEXT
);
"""

_CREATE_SYNC_RACES = """
CREATE TABLE IF NOT EXISTS result_sync_races (
    race_date TEXT NOT NULL,
    race_id TEXT NOT NULL,
    url TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    last_error TEXT,
    PRIMARY KEY (race_date, race_id)
);
"""


# -----------------------------------------------------------------------
# Connection helpers
# -----------------------------------------------------------------------

@contextmanager
def _get_conn():
    """Commit or roll back an operation, then release its database handles."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        with conn:
            yield conn
    finally:
        # sqlite3.Connection.__exit__ manages the transaction, not lifetime.
        # Leaving handles to GC retained WAL files and Windows file locks.
        conn.close()


def init_db() -> None:
    """Create the results table and indexes if they don't already exist."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.execute(_CREATE_CONTEXT)
        # Context remains separate from canonical training columns. An explicit
        # insert column list below supports old tables whose new column appends.
        context_columns = {row[1] for row in conn.execute("PRAGMA table_info(result_context_observations)")}
        if "result_amendment_json" not in context_columns:
            conn.execute("ALTER TABLE result_context_observations ADD COLUMN result_amendment_json TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_context_date ON result_context_observations(race_date)")
        conn.execute(_CREATE_RUNNER_TEXT)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runner_text_date ON runner_text_observations(race_date)")
        conn.execute(_CREATE_SYNC_DATES)
        conn.execute(_CREATE_SYNC_RACES)
        ensure_source_keys(conn, "results")
        ensure_source_keys(conn, "runner_text_observations", ("phase", "content_sha256"))
        # Migrate: add columns that may not exist in older databases
        for col, col_type in [
            ("sire", "TEXT"),
            ("dam", "TEXT"),
            ("damsire", "TEXT"),
            ("weather_temp_max", "REAL"),
            ("weather_temp_min", "REAL"),
            ("weather_precip_mm", "REAL"),
            ("weather_wind_kmh", "REAL"),
            ("weather_precip_prev3", "REAL"),
            # NULL = not yet backfilled; 0/1 once scraped/backfilled
            # (scripts/backfill_wind_surgery.py).
            ("wind_surgery", "INTEGER"),
            ("finish_pos_label", "TEXT"),
            ("reported_jockey_claim_lbs", "REAL"),
            # No longer collected, but older rows hold it and the history carries it.
            ("opening_odds", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE results ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError as error:
                if "duplicate column name" not in str(error).lower():
                    raise
        _init_source_state(conn)
    logger.info(f"Database ready at {DB_PATH}")


def migrate_runner_identity():
    """Upgrade legacy keys atomically, preserving source values and revisions."""
    with writer_lock(Path(DB_PATH).with_suffix(".source.lock")) as acquired:
        if not acquired:
            raise RuntimeError("Results refresh is already running")
        init_db()
        with _get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = {"results": rebuild_without_key(conn, "results", ["race_id", "horse_name"]),
                       "runner_text_observations": rebuild_without_key(conn, "runner_text_observations",
                           ["race_id", "horse_name", "phase", "content_sha256"])}
            ensure_source_keys(conn, "results")
            ensure_source_keys(conn, "runner_text_observations", ("phase", "content_sha256"))
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, original, replacement in (("verified_result_rows", "before_json", "after_json"),
                                                  ("result_amendment_rows", "original_row_json", "corrected_values_json")):
                if table in tables:
                    ensure_repair_keys(conn, table, original, replacement)
            _init_source_state(conn)
        return changed


def _init_source_state(conn):
    """Track every factual SQL edit, including edits by older repair scripts."""
    conn.execute("CREATE TABLE IF NOT EXISTS result_source_state (id INTEGER PRIMARY KEY CHECK(id=1), "
                 "database_id TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0, columns_hash TEXT)")
    conn.execute("INSERT OR IGNORE INTO result_source_state(id,database_id) VALUES (1,?)", (uuid4().hex,))
    conn.execute("CREATE TABLE IF NOT EXISTS result_source_changes (revision INTEGER PRIMARY KEY, "
                 "race_date TEXT, race_id TEXT, operation TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS result_race_observations (race_id TEXT, race_date TEXT, "
                 "content_sha256 TEXT, observed_at TEXT, payload_json TEXT, PRIMARY KEY(race_id,content_sha256))")
    columns = [r[1] for r in conn.execute("PRAGMA table_info(results)") if r[1] != "scraped_at"]
    digest = hashlib.sha256(json.dumps(columns).encode()).hexdigest()
    if conn.execute("SELECT columns_hash FROM result_source_state WHERE id=1").fetchone()[0] == digest:
        return
    for operation, ref in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
        name = "result_revision_" + operation.lower()
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        condition = " WHEN " + " OR ".join(f'NEW."{c}" IS NOT OLD."{c}"' for c in columns) if operation == "UPDATE" else ""
        conn.execute(f"CREATE TRIGGER {name} AFTER {operation} ON results{condition} BEGIN "
                     "UPDATE result_source_state SET revision=revision+1 WHERE id=1; "
                     f"INSERT INTO result_source_changes SELECT revision,{ref}.race_date,{ref}.race_id,'{operation}' "
                     "FROM result_source_state WHERE id=1; END")
    conn.execute("UPDATE result_source_state SET columns_hash=? WHERE id=1", (digest,))


# Every column of the published history, in table order. Declaring them means a
# stray value is an error at publication, where inferring types from the values
# would quietly turn the whole column into something else.
RESULTS_SNAPSHOT_COLUMNS = {
    "race_id": "whole", "race_date": "text", "off_time": "text", "track": "text",
    "region": "text", "race_name": "text", "race_class": "real", "race_type": "text",
    "distance_furlongs": "real", "going": "text", "prize_money": "real",
    "num_runners": "whole", "horse_name": "text", "horse_id": "whole", "jockey": "text",
    "trainer": "text", "age": "whole", "sex": "text", "headgear": "text",
    "weight_lbs": "real", "draw": "whole", "form": "text", "days_since_last_run": "whole",
    "odds": "real", "official_rating": "whole", "finish_position": "whole", "won": "whole",
    "lengths_behind": "real", "horse_runs": "whole", "horse_wins": "whole",
    "horse_places": "whole", "surface": "text", "handicap": "whole", "sire": "text",
    "dam": "text", "damsire": "text", "weather_temp_max": "real", "weather_temp_min": "real",
    "weather_precip_mm": "real", "weather_wind_kmh": "real",
    "weather_precip_prev3": "real", "opening_odds": "real",
    "wind_surgery": "real", "finish_pos_label": "text", "reported_jockey_claim_lbs": "real",
}


def _snapshot_text(series: pd.Series) -> pd.Series:
    """Text with one missing marker: a blank is missing, and missing is NaN."""
    values = series.astype(object)
    return values.where(values.notna() & values.ne(""), np.nan)


def _snapshot_number(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        return pd.to_numeric(_snapshot_text(series), errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"results.{series.name} must hold numbers: {error}") from error


def _snapshot_whole(series: pd.Series) -> pd.Series:
    """Integers, or floats while any row is missing, which keeps them numeric."""
    numbers = _snapshot_number(series)
    return numbers if numbers.hasnans else numbers.astype("int64")


def _snapshot_real(series: pd.Series) -> pd.Series:
    return _snapshot_number(series).astype("float64")


_SNAPSHOT_KINDS = {"text": (_snapshot_text, "string"), "whole": (_snapshot_whole, "int64"),
                   "real": (_snapshot_real, "float64")}


def results_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The published history's columns and types, whichever route the rows took."""
    different = set(frame.columns) ^ set(RESULTS_SNAPSHOT_COLUMNS)
    # Older databases can be published before the next ingestion migrates them.
    if different == {"reported_jockey_claim_lbs"} and "reported_jockey_claim_lbs" not in frame:
        frame = frame.assign(reported_jockey_claim_lbs=np.nan)
    elif different:
        raise ValueError(f"Results columns differ from the declared snapshot: {sorted(different)}")
    return pd.DataFrame({name: _SNAPSHOT_KINDS[kind][0](frame[name])
                         for name, kind in RESULTS_SNAPSHOT_COLUMNS.items()})


def results_snapshot_path() -> Path:
    return Path(config.RAW_DATA_DIR) / "race_results.parquet"


def read_results_snapshot(path=None, columns=None) -> pd.DataFrame:
    """Read the published history, with the same missing marker as a fresh frame."""
    frame = pd.read_parquet(path or results_snapshot_path(), columns=columns)
    for name in frame.columns:
        if RESULTS_SNAPSHOT_COLUMNS[name] == "text":
            frame[name] = _snapshot_text(frame[name])
    return frame


def publish_results_snapshot(*, force=False):
    """Publish only a changed canonical revision, at its declared column types."""
    path = results_snapshot_path()
    receipt_path = path.with_suffix(".source.json")
    with _get_conn() as conn:
        conn.execute("BEGIN")
        database_id, revision, columns_hash = conn.execute(
            "SELECT database_id,revision,columns_hash FROM result_source_state WHERE id=1").fetchone()
        try:
            receipt, stat = read_json(receipt_path), path.stat()
        except (OSError, ValueError):
            receipt, stat = {}, None
        if (not force and stat and receipt.get("database_id") == database_id
                and receipt.get("revision") == revision and receipt.get("columns_hash") == columns_hash
                and receipt.get("bytes") == stat.st_size
                and receipt.get("mtime_ns") == stat.st_mtime_ns):
            return {**receipt, "state": "cached"}
        frame = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", conn).drop(columns="scraped_at", errors="ignore")
        _export_results_snapshot(frame, str(path))
        stat = path.stat()
        receipt = {"database_id": database_id, "revision": revision, "columns_hash": columns_hash, "rows": len(frame),
                   "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "path": str(path),
                   "published_at": datetime.now().isoformat()}
        atomic_json(receipt_path, receipt)
    return {**receipt, "state": "published"}


def publish_after_repair(db_path):
    """Existing targeted tools publish only when they edited this canonical DB."""
    if Path(db_path).resolve() != Path(DB_PATH).resolve():
        return None
    with writer_lock(Path(DB_PATH).with_suffix(".source.lock")) as acquired:
        if not acquired:
            raise RuntimeError("Results refresh is running; publish the repaired snapshot after it finishes")
        init_db()
        return publish_results_snapshot()


# -----------------------------------------------------------------------
# Insert / upsert
# -----------------------------------------------------------------------

_COLUMNS = [
    "race_id", "race_date", "off_time", "track", "region",
    "race_name", "race_class", "race_type", "distance_furlongs",
    "going", "prize_money", "num_runners", "horse_name", "horse_id",
    "jockey", "trainer", "age", "sex", "headgear", "wind_surgery",
    "weight_lbs", "reported_jockey_claim_lbs",
    "draw", "form", "days_since_last_run", "odds",
    "official_rating",
    "finish_position", "finish_pos_label", "won", "lengths_behind", "horse_runs",
    "horse_wins", "horse_places", "surface", "handicap",
    "sire", "dam", "damsire",
    "weather_temp_max", "weather_temp_min", "weather_precip_mm",
    "weather_wind_kmh", "weather_precip_prev3",
]


def insert_results(df: pd.DataFrame) -> int:
    """
    Insert a DataFrame of race results into the database.

    Rows whose source race/horse identity already exists are silently
    skipped (``INSERT OR IGNORE``).

    Returns:
        Number of *new* rows inserted.
    """
    if df is None or df.empty:
        return 0

    # Defensive guard: historical results should contain at least one
    # settled finisher per race.  Sporting Life occasionally serves a
    # non-settled page shape on result URLs; those batches tend to have
    # all finish_position values at 0 and should not enter the results DB.
    if {"race_id", "finish_position"}.issubset(df.columns):
        _finish_pos = pd.to_numeric(df["finish_position"], errors="coerce").fillna(0)
        _race_has_finisher = (_finish_pos > 0).groupby(df["race_id"]).transform("any")
        _bad_races = int((~_race_has_finisher).groupby(df["race_id"]).first().sum())
        if _bad_races:
            logger.warning(
                "Skipping %d unresolved result races with no settled finishers",
                _bad_races,
            )
            df = df.loc[_race_has_finisher].copy()

    if df.empty:
        return 0

    init_db()

    with _get_conn() as conn:
        inserted = _insert_results(conn, df)
    logger.info("  Inserted %d new rows", inserted)
    return inserted


def _insert_results(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert within the caller's transaction (also used by the sync ledger)."""

    observations = df
    df = df.loc[~runner_index(df).duplicated(keep="first")].copy()
    if horse_ids(df).isna().any():
        missing_races = df.loc[horse_ids(df).isna(), "race_id"].astype(str).unique()
        references = []
        for offset in range(0,len(missing_races),500):
            ids = missing_races[offset:offset+500].tolist()
            references.append(pd.read_sql_query("SELECT race_id,horse_id,horse_name FROM results WHERE race_id IN ("+
                ",".join("?" for _ in ids)+")", conn, params=ids))
        if references:
            df = adopt_stored_ids(df, pd.concat(references,ignore_index=True))
    # Keep only columns that exist in both the DF and the schema
    cols = [c for c in _COLUMNS if c in df.columns]
    subset = df[cols].copy()
    if "horse_id" in subset:
        subset["horse_id"] = horse_ids(subset)
    subset = subset.astype(object).where(pd.notna(subset), None)

    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO results ({col_names}) VALUES ({placeholders})"

    cursor = conn.executemany(sql, subset.values.tolist())
    inserted = cursor.rowcount  # excludes revision/evidence trigger side effects
    _insert_result_context(conn, observations)
    _insert_runner_text(conn, observations, "result")
    return inserted


def _insert_result_context(conn, frame):
    """Persist supplied result evidence in the same transaction as its field."""
    if not any(c in frame for c in _CONTEXT_FIELDS):
        return
    fields = ("race_id", "horse_name", "horse_id", "race_date", "weight_lbs", *_CONTEXT_FIELDS)
    observed = datetime.now().astimezone(timezone.utc).isoformat(timespec="microseconds")
    rows = []
    for record in frame.reindex(columns=fields).to_dict("records"):
        record = {k: (None if pd.isna(v) else v) for k,v in record.items()}
        if not any(record[c] is not None for c in _CONTEXT_FIELDS):
            continue
        # Adding optional metadata must not duplicate unchanged legacy evidence.
        content_record = {k: v for k, v in record.items() if k != "result_amendment_json" or v is not None}
        content = json.dumps(content_record, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
        digest = hashlib.sha256(content.encode()).hexdigest()
        rows.append(tuple(record[c] for c in fields)+(digest, observed))
    columns = ["declared_weight_lbs" if c == "weight_lbs" else c for c in fields] + ["content_sha256", "observed_at"]
    conn.executemany("INSERT OR IGNORE INTO result_context_observations (" + ",".join(columns) + ") VALUES (" +
                     ",".join("?" for _ in columns) + ")", rows)


def _insert_runner_text(conn, frame, phase):
    """Append each newly seen commentary/verdict, keyed by its content."""
    if not any(c in frame for c in _TEXT_FIELDS):
        return
    observed = datetime.now().astimezone(timezone.utc).isoformat(timespec="microseconds")
    text = frame.reindex(columns=["race_id", "horse_name", "horse_id", "race_date", *_TEXT_FIELDS]).rename(columns=_TEXT_FIELDS)
    rows = []
    for record in text.to_dict("records"):
        record = {k: (None if pd.isna(v) else str(v)) for k, v in record.items()}
        if record["commentary"] is None and record["verdict"] is None:
            continue
        content = json.dumps([record["commentary"], record["verdict"]], separators=(",", ":"))
        rows.append((record["race_id"], record["horse_name"], record["horse_id"], record["race_date"], phase,
                     record["commentary"], record["verdict"], hashlib.sha256(content.encode()).hexdigest(), observed))
    conn.executemany("INSERT OR IGNORE INTO runner_text_observations (race_id, horse_name, horse_id, race_date, phase, "
                     "commentary, verdict, content_sha256, observed_at) VALUES (?,?,?,?,?,?,?,?,?)", rows)


def record_runner_text(frame: pd.DataFrame, phase: str) -> None:
    """Record racecard text outside a results transaction."""
    if frame is None or frame.empty:
        return
    init_db()
    with _get_conn() as conn:
        _insert_runner_text(conn, frame, phase)


def load_runner_text(*, phase=None, start=None, end=None) -> pd.DataFrame:
    """Every observation (earliest first); research code chooses which copy to trust."""
    init_db()
    clauses, parameters = [], []
    for clause, value in (("phase = ?", phase), ("race_date >= ?", start), ("race_date <= ?", end)):
        if value is not None:
            clauses.append(clause)
            parameters.append(str(value))
    query = "SELECT * FROM runner_text_observations"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    with _get_conn() as conn:
        return pd.read_sql_query(query + " ORDER BY observed_at, race_id, horse_name", conn, params=parameters)


def load_result_context(*, start=None, end=None):
    """Latest observed dated context, kept separate from default training inputs."""
    init_db()
    clauses, parameters = [], []
    for operator, value in ((">=", start), ("<=", end)):
        if value is not None:
            clauses.append("race_date "+operator+" ?")
            parameters.append(str(value))
    query = "SELECT * FROM result_context_observations"
    if clauses:
        query += " WHERE "+" AND ".join(clauses)
    with _get_conn() as conn:
        frame = pd.read_sql_query(query+" ORDER BY observed_at, content_sha256", conn, params=parameters)
    frame["_runner_key"] = runner_keys(frame)
    return frame.drop_duplicates(["race_id", "_runner_key"], keep="last").drop(columns="_runner_key").reset_index(drop=True)


# -----------------------------------------------------------------------
# Query helpers
# -----------------------------------------------------------------------

def get_dates_in_db() -> set[str]:
    """Return the set of distinct race_date strings already stored."""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT race_date FROM results ORDER BY race_date"
        ).fetchall()
    return {r[0] for r in rows}


def get_latest_date() -> str | None:
    """Return the most recent race_date in the DB, or None if empty."""
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(race_date) FROM results"
        ).fetchone()
    return row[0] if row and row[0] else None


def load_from_database(days_back: int | None = None) -> pd.DataFrame:
    """
    Load results from the database as a DataFrame.

    Args:
        days_back: If given, only return the last *N* days.
                   If ``None``, return everything.

    Returns:
        DataFrame of scraped runner rows.
    """
    init_db()

    if days_back is not None:
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = "SELECT * FROM results WHERE race_date >= ? ORDER BY race_date"
        params: tuple = (cutoff,)
    else:
        query = "SELECT * FROM results ORDER BY race_date"
        params = ()

    with _get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    # Drop the internal scraped_at column if present
    if "scraped_at" in df.columns:
        df = df.drop(columns=["scraped_at"])

    logger.info(f"Loaded {len(df)} records from database" +
                (f" (last {days_back} days)" if days_back else ""))
    return df


# -----------------------------------------------------------------------
# Sync — incremental scrape + store
# -----------------------------------------------------------------------

def _validate_race_results(df: pd.DataFrame, date_str: str, race_id: str) -> None:
    """Reject empty, unresolved, misrouted, or visibly truncated race pages."""
    required = {"race_id", "race_date", "horse_name", "finish_position", "num_runners", "won"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError("Empty or incomplete result response")
    if not (df["race_id"].astype(str).eq(race_id).all()
            and df["race_date"].eq(date_str).all()):
        raise ValueError("Result response belongs to another race or date")
    names = df["horse_name"]
    if names.isna().any() or names.astype(str).str.strip().eq("").any():
        raise ValueError("Missing runner names")
    if "horse_id" in df:
        identities = horse_ids(df)
        if identities.isna().any() or identities.duplicated().any():
            raise ValueError("Missing or duplicate source horse identities")
    elif names.map(normalise_horse_key).duplicated().any():
        raise ValueError("Missing or duplicate runner names")
    # Non-finishers (including position 99) are legitimate runners. Retain
    # them, but require a winner before declaring the whole result settled.
    positions = pd.to_numeric(df["finish_position"], errors="coerce")
    if positions.isna().any() or positions.lt(0).any() or positions.mod(1).ne(0).any():
        raise ValueError("Missing or invalid finish position")
    if not positions.eq(1).any():
        raise ValueError("Result response has no settled winner")
    if "won" in df.columns:
        won = pd.to_numeric(df["won"], errors="coerce")
        if not won.isin([0, 1]).all() or not won.eq(1).equals(positions.eq(1)):
            raise ValueError("Winner labels disagree with finish positions")
    if "num_runners" in df.columns:
        counts = pd.to_numeric(df["num_runners"], errors="coerce")
        if counts.isna().any() or counts.le(0).any() or counts.nunique() != 1:
            raise ValueError("Missing or inconsistent field size")
        expected = counts.iloc[0]
        if len(df) != expected:
            raise ValueError(f"Incomplete or inconsistent field: {len(df)} of {expected:g} runners")


def _replace_race_results(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Replace a validated complete field inside the caller's transaction.

    Updating supplied columns preserves separate weather/pedigree enrichment.
    Runners absent from the authoritative field are removed, so rescraping an
    old fragment can also correct obsolete non-runner rows. The caller must
    validate the complete response before invoking this function.
    """
    race_id = str(df["race_id"].iloc[0])
    previous = pd.read_sql_query("SELECT rowid AS _rowid,race_id,horse_id,horse_name FROM results WHERE race_id=?", conn, params=(race_id,))
    df = adopt_stored_ids(df, previous)
    _protect_reviewed_results(conn, df)
    keys = runner_keys(previous)
    if keys.duplicated().any():
        raise ValueError("Stored race has ambiguous source horse identities")
    rowids = dict(zip(keys, previous._rowid))
    incoming = runner_keys(df)
    df = df.copy()
    if "horse_id" in df:
        df["horse_id"] = horse_ids(df)
    cols = [c for c in _COLUMNS if c in df.columns]
    values = df[cols].astype(object).where(pd.notna(df[cols]), None)
    update_cols = [col for col in cols if col != "race_id"]
    updates, additions = [], []
    for key, record in zip(incoming, values.to_dict("records")):
        if key in rowids:
            supplied = [record[col] for col in update_cols]
            updates.append(supplied+[int(rowids[key])]+supplied)
        else:
            additions.append([record[col] for col in cols])
    incoming_keys = set(incoming)
    conn.executemany("DELETE FROM results WHERE rowid=?", [(int(rowid),) for key,rowid in rowids.items() if key not in incoming_keys])
    conn.executemany("UPDATE results SET "+",".join(col+"=?" for col in update_cols)+
        ",scraped_at=datetime('now') WHERE rowid=? AND ("+" OR ".join(col+" IS NOT ?" for col in update_cols)+")", updates)
    conn.executemany(f"INSERT INTO results ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", additions)
    _insert_result_context(conn, df)
    _insert_runner_text(conn, df, "result")
    _insert_race_observation(conn, df)
    return len(additions)


def _protect_reviewed_results(conn, frame):
    """A routine provider refresh cannot undo independently reviewed result facts."""
    race_id = str(frame.race_id.iloc[0])
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    reviewed = any(table in tables and conn.execute(f"SELECT 1 FROM {table} WHERE race_id=? LIMIT 1", (race_id,)).fetchone()
                   for table in ("verified_result_rows", "result_amendment_rows"))
    if not reviewed:
        return
    cols = [c for c in ("finish_position", "won", "lengths_behind", "finish_pos_label") if c in frame]
    current = pd.read_sql_query("SELECT * FROM results WHERE race_id=?", conn, params=(race_id,))
    try:
        if len(current) != len(frame):
            raise ValueError("Reviewed starter population changed")
        positions = runner_positions(frame,current)
        pd.testing.assert_frame_equal(current.iloc[positions][cols].reset_index(drop=True), frame[cols].reset_index(drop=True), check_dtype=False, check_exact=True)
    except (AssertionError,ValueError) as error:
        raise ValueError(f"Review required: source disagrees with accepted correction for race {race_id}") from error


def _insert_race_observation(conn, frame):
    columns = [c for c in ("winning_time_secs", "off_time_actual", "prize_currency") if c in frame]
    if not columns:
        return
    values = {c: None if pd.isna(frame[c].iloc[0]) else frame[c].iloc[0] for c in columns}
    content = json.dumps(values, sort_keys=True, default=str, allow_nan=False)
    conn.execute("INSERT OR IGNORE INTO result_race_observations VALUES (?,?,?,?,?)",
                 (str(frame.race_id.iloc[0]), str(frame.race_date.iloc[0]), hashlib.sha256(content.encode()).hexdigest(),
                  datetime.now().isoformat(), content))


def ingest_results(frame, *, publish=True):
    """Shared complete-field writer for network collectors and historical repairs."""
    if frame is None or frame.empty:
        return 0
    fields = []
    for (day, race_id), field in frame.groupby(["race_date", "race_id"], sort=False):
        _validate_race_results(field, str(day), str(race_id))
        fields.append(field)
    init_db()
    with writer_lock(Path(DB_PATH).with_suffix(".source.lock")) as acquired:
        if not acquired:
            raise RuntimeError("Results refresh is already running")
        with _get_conn() as conn:
            inserted = sum(_replace_race_results(conn, field) for field in fields)
        if publish:
            publish_results_snapshot()
    return inserted


def _export_results_snapshot(frame: pd.DataFrame, path: str) -> None:
    """Publish the whole history without exposing a partially written file."""
    import pyarrow as pa
    schema = pa.schema([(name, _SNAPSHOT_KINDS[kind][1]) for name, kind in RESULTS_SNAPSHOT_COLUMNS.items()])
    atomic_parquet(results_snapshot_frame(frame), path, schema=schema)


def write_csv_atomically(frame: pd.DataFrame, path: str) -> None:
    """Write a CSV without exposing a partially written file."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=folder, suffix=".tmp", delete=False) as fh:
            temporary = fh.name
            frame.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)


def queue_history_audit(start: str | None = None, end: str | None = None) -> int:
    """Queue stored legacy dates for a one-off completeness relist.

    Dates stored before the sync ledger existed are trusted by default, but
    the old date-level sync could leave a day incomplete without ever saying
    so: it kept only the first 50 listed races per day, and a race page that
    failed to fetch was logged and skipped while its date counted as present.
    Queued dates are relisted by the next results collection whose window
    covers them; races already stored complete are adopted, not refetched.
    Returns the number of dates queued.
    """
    init_db()
    with _get_conn() as conn:
        dates = [r[0] for r in conn.execute("""
            SELECT DISTINCT race_date FROM results
            WHERE race_date >= COALESCE(?, '') AND race_date <= COALESCE(?, '9999')
              AND race_date NOT IN (SELECT race_date FROM result_sync_dates)
        """, (start, end))]
        conn.executemany(
            "INSERT INTO result_sync_dates (race_date) VALUES (?)",
            [(d,) for d in dates],
        )
    logger.info("Queued %d legacy dates for a completeness relist", len(dates))
    return len(dates)


def result_rows(start: str, end: str, columns, db_path=None) -> pd.DataFrame:
    """Read-only window of canonical rows; never opens a write transaction."""
    path = Path(db_path or DB_PATH).resolve()
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as con:
        return pd.read_sql_query(f"SELECT {','.join(columns)} FROM results WHERE race_date BETWEEN ? AND ?",
                                 con, params=(start, end))


def init_pedigree_table() -> None:
    with _get_conn() as conn:
        conn.execute(_CREATE_PEDIGREE)
        conn.execute("CREATE TABLE IF NOT EXISTS pedigree_fetch_state (horse_id TEXT PRIMARY KEY, "
                     "status TEXT NOT NULL, attempts INTEGER NOT NULL, next_retry_at TEXT, last_error TEXT)")


def load_pedigree() -> pd.DataFrame:
    init_pedigree_table()
    with _get_conn() as conn:
        return pd.read_sql_query(
            "SELECT horse_id, sire, dam, damsire, foaled FROM horse_pedigree", conn
        )


# -----------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------

def db_stats() -> dict:
    """Return a summary of the database contents."""
    init_db()
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        dates = conn.execute(
            "SELECT COUNT(DISTINCT race_date) FROM results"
        ).fetchone()[0]
        tracks = conn.execute(
            "SELECT COUNT(DISTINCT track) FROM results"
        ).fetchone()[0]
        races = conn.execute(
            "SELECT COUNT(DISTINCT race_id) FROM results"
        ).fetchone()[0]
        min_date = conn.execute(
            "SELECT MIN(race_date) FROM results"
        ).fetchone()[0]
        max_date = conn.execute(
            "SELECT MAX(race_date) FROM results"
        ).fetchone()[0]
        cutoff = (datetime.now()-timedelta(days=14)).date().isoformat()
        pending, recent_pending = conn.execute("SELECT COUNT(*),COALESCE(SUM(race_date>=?),0) "
                                               "FROM result_sync_races WHERE completed=0", (cutoff,)).fetchone()
        unverified = conn.execute("SELECT COUNT(*) FROM result_sync_dates WHERE listing_succeeded=0 AND race_date>=?", (cutoff,)).fetchone()[0]

    return {
        "total_runners": total,
        "total_dates": dates,
        "total_tracks": tracks,
        "total_races": races,
        "earliest_date": min_date or "—",
        "latest_date": max_date or "—",
        "pending_races": pending,
        "recent_pending_races": recent_pending,
        "recent_unverified_dates": unverified,
        "db_path": DB_PATH,
        "db_size_mb": round(
            os.path.getsize(DB_PATH) / (1024 * 1024), 2
        ) if os.path.exists(DB_PATH) else 0,
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # This CLI's progress output contains emoji, and on Windows a redirected
    # stdout defaults to cp1252, which cannot encode them. The resulting
    # UnicodeEncodeError is raised by the very first print — before any
    # work starts — so a redirected command exited non-zero having silently done
    # nothing at all. Force UTF-8 with replacement so a console encoding can
    # never abort it.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(
        description="Manage the historical race database"
    )
    parser.add_argument(
        "--audit-history",
        action="store_true",
        help="Queue every stored legacy date for a completeness relist by the "
             "next results collection whose window covers it",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print database statistics",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export the full database to a CSV file",
    )
    args = parser.parse_args()

    if args.stats or (not args.audit_history and not args.export):
        s = db_stats()
        print("\n📦  Race Database Statistics")
        print("=" * 40)
        for k, v in s.items():
            print(f"  {k:20s}: {v}")
        print()

    if args.audit_history:
        print(f"  Queued {queue_history_audit()} legacy dates for a completeness relist.")
        print("  Run python -m src.ingestion --sync DAYS --sources sl over those dates.")

    if args.export:
        df = load_from_database()
        write_csv_atomically(df, args.export)
        print(f"  Exported {len(df)} records to {args.export}")
