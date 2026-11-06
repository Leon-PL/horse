"""
Historical Database Module
==========================
SQLite-backed persistent storage for scraped race results.

Instead of re-scraping months of data every time you retrain, this module
stores results in a local ``data/races.db`` file and tracks unfinished
race fetches so a partial scrape can be resumed safely.

Usage::

    from src.database import sync_database, load_from_database, db_stats

    # Resume missing/incomplete races (incremental)
    sync_database(days_back=90)

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

import pandas as pd

import config
from src.source_storage import atomic_json, read_json, writer_lock
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

# Legacy rows have no scrape manifest. Recheck a small recent window without
# unexpectedly re-scraping years of existing history on the first upgraded run.
RECENT_RELIST_DAYS = 7

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


def publish_results_snapshot(*, force=False):
    """Publish only a changed canonical revision, retaining the CSV dtype contract."""
    path = Path(config.RAW_DATA_DIR) / "race_results.csv"
    receipt_path = path.with_suffix(".source.json")
    with _get_conn() as conn:
        conn.execute("BEGIN")
        database_id, revision = conn.execute("SELECT database_id,revision FROM result_source_state WHERE id=1").fetchone()
        try:
            receipt, stat = read_json(receipt_path), path.stat()
        except (OSError, ValueError):
            receipt, stat = {}, None
        if (not force and stat and receipt.get("database_id") == database_id
                and receipt.get("revision") == revision and receipt.get("bytes") == stat.st_size
                and receipt.get("mtime_ns") == stat.st_mtime_ns):
            return {**receipt, "state": "cached"}
        frame = pd.read_sql_query("SELECT * FROM results ORDER BY race_date", conn).drop(columns="scraped_at", errors="ignore")
        _export_results_snapshot(frame, str(path))
        stat = path.stat()
        receipt = {"database_id": database_id, "revision": revision, "rows": len(frame),
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
    "weight_lbs",
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
        DataFrame matching the same schema as ``scrape_results()``.
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
    """Publish a whole-history CSV without exposing a partially written file."""
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
    Queued dates are relisted by the next ``sync_database`` whose window
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


def sync_database(days_back: int = 90, *, enrich: bool = False, return_report=False,
                  start=None, end=None, recheck=False, revision_interval_seconds=21600,
                  pedigree_limit=100, weather=False, offline=False, backlog_max_races=0):
    """Resume result ingestion and atomically export the complete history.

    Keep one idle connection for the operation's lifetime. Transactions use
    their own short-lived handles; the idle handle prevents SQLite from
    checkpointing the WAL after every individual race's connection closes.
    It never opens a read or write transaction across network requests.
    Set ``enrich=False`` to collect results without optional source backfills.
    """
    if days_back < 0:
        raise ValueError("days_back must be non-negative")
    if backlog_max_races < 0:
        raise ValueError("Historical recovery budget must be non-negative")
    with writer_lock(Path(DB_PATH).with_suffix(".source.lock")) as acquired:
        if not acquired:
            raise RuntimeError("Results refresh is already running")
        init_db()
        with _get_conn() as idle:
            idle.execute("SELECT race_id FROM results LIMIT 1").fetchall()
            report = _sync_database(days_back, enrich=enrich, start=start, end=end, recheck=recheck,
                revision_interval_seconds=revision_interval_seconds, pedigree_limit=pedigree_limit, weather=weather, offline=offline,
                backlog_max_races=backlog_max_races)
        atomic_json(Path(DB_PATH).with_suffix(".refresh.json"), report)
        if return_report:
            return report
        frame = load_from_database()
        frame.attrs["source_refresh"] = report
        return frame


def _sync_database(days_back: int, *, enrich=False, start=None, end=None, recheck=False,
                   revision_interval_seconds=21600, pedigree_limit=100, weather=False, offline=False,
                   backlog_max_races=0):
    """Resume incomplete race scrapes and return/export the full history.

    Each listed unfinished race is attempted once per invocation, within the
    requested window. Recent dates are relisted to discover delayed results.
    Older legacy dates without a manifest are preserved, not assumed audited.
    Empty listings remain retryable: the scraper also returns [] on parse
    failures, so they cannot certify that a day had no races.
    """
    from src.data_scraper import SportingLifeScraper

    if days_back < 0:
        raise ValueError("days_back must be non-negative")
    init_db()
    existing_dates = get_dates_in_db()
    today = datetime.now()
    all_dates = [
        (today - timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(days_back)
    ]
    if start is not None or end is not None:
        last = datetime.fromisoformat(end).date() if end else (today-timedelta(days=1)).date()
        first = datetime.fromisoformat(start).date() if start else last-timedelta(days=max(0, days_back-1))
        if first > last or last >= today.date():
            raise ValueError("Results range must end before today and start no later than its end")
        all_dates = [(last-timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
    date_window = set(all_dates)
    recent_dates = {d for d in all_dates if d >= (today-timedelta(days=RECENT_RELIST_DAYS)).date().isoformat()}
    with _get_conn() as conn:
        before_revision = conn.execute("SELECT revision FROM result_source_state WHERE id=1").fetchone()[0]
        managed_dates = {r[0] for r in conn.execute("SELECT race_date FROM result_sync_dates")}
        incomplete_dates = {r[0] for r in conn.execute("""
            SELECT d.race_date FROM result_sync_dates d
            WHERE d.listing_succeeded = 0 OR EXISTS (
                SELECT 1 FROM result_sync_races r
                WHERE r.race_date = d.race_date AND r.completed = 0
            )
        """)}
    legacy_dates = (existing_dates & date_window) - managed_dates - recent_dates
    if legacy_dates:
        logger.info("Preserving %d older legacy dates without a completeness audit", len(legacy_dates))
    listing_dates = sorted(
        (date_window - existing_dates - managed_dates)
        | (incomplete_dates & date_window) | recent_dates
    )
    if recheck:
        listing_dates = sorted(date_window)
    scraper = SportingLifeScraper()
    scraper.raw_store_path = Path(DB_PATH).parent / "sl_cache/raw.sqlite"
    scraper.offline = offline

    # Persist the listing attempt and every expected race before fetching any
    # result. An interruption or partial failure therefore cannot hide a gap.
    for date_str in listing_dates:
        with _get_conn() as conn:
            conn.execute("""
                INSERT INTO result_sync_dates
                    (race_date, listing_attempts, last_attempt_at) VALUES (?, 1, datetime('now'))
                ON CONFLICT(race_date) DO UPDATE SET listing_succeeded = 0,
                    listing_attempts = listing_attempts + 1, last_attempt_at = datetime('now')
            """, (date_str,))
        try:
            links = scraper.get_results_urls(date_str, uk_only=True)
            if not links and date_str not in getattr(scraper, "empty_result_dates", set()):
                raise ValueError("Empty or unreadable results listing")
            race_links = {str(info.get("race_id") or ""): info["url"] for info in links}
            if any(not race_id or not url for race_id, url in race_links.items()):
                raise ValueError("Listing has a missing race ID or URL")
            with _get_conn() as conn:
                for race_id, url in race_links.items():
                    # Adopt complete legacy races without refetching their
                    # pages; a stored fragment must still be retried.
                    stored = conn.execute("""
                        SELECT COUNT(*), MAX(num_runners), MAX(finish_position = 1),
                               MIN(num_runners)
                        FROM results WHERE race_date = ? AND race_id = ?
                    """, (date_str, race_id)).fetchone()
                    complete = int(bool(stored[0] and stored[2]
                                        and stored[0] == stored[1] == stored[3]))
                    conn.execute("""
                        INSERT INTO result_sync_races (race_date, race_id, url, completed)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(race_date, race_id) DO UPDATE SET
                            url = excluded.url,
                            completed = CASE
                                WHEN result_sync_races.last_error LIKE 'not run:%'
                                THEN result_sync_races.completed
                                ELSE excluded.completed END
                    """, (date_str, race_id, url, complete))
                conn.execute("""
                    UPDATE result_sync_dates SET listing_succeeded = 1, last_error = NULL
                    WHERE race_date = ?
                """, (date_str,))
        except Exception as exc:
            logger.warning("Failed to verify listing for %s: %s", date_str, exc)
            with _get_conn() as conn:
                conn.execute("UPDATE result_sync_dates SET last_error = ? WHERE race_date = ?",
                             (str(exc), date_str))

    # Persisted URLs remain usable even when a subsequent listing request
    # fails or omits the race. Never truncate the number of races in a day.
    with _get_conn() as conn:
        candidates = conn.execute("SELECT race_date,race_id,url,completed,last_attempt_at,last_error "
                                  "FROM result_sync_races ORDER BY race_date,race_id").fetchall()
        jobs = []
        for ds, race_id, url, closed, attempted, error in candidates:
            if ds not in date_window:
                continue
            due = (attempted is None or (today-datetime.fromisoformat(attempted)).total_seconds() >= revision_interval_seconds)
            if not closed or ((recheck or (ds in recent_dates and due)) and not str(error or "").startswith("not run:")):
                jobs.append((ds, race_id, url))
        backlog_jobs = []
        if backlog_max_races:
            cutoff = (today-timedelta(days=1)).isoformat()
            candidates = conn.execute("SELECT race_date,race_id,url FROM result_sync_races "
                "WHERE completed=0 AND race_date<? AND (last_attempt_at IS NULL OR last_attempt_at<?) "
                "AND COALESCE(last_error,'') NOT LIKE 'Review required:%' "
                "AND COALESCE(last_error,'') NOT LIKE 'Missing or duplicate%' "
                "ORDER BY attempts,COALESCE(last_attempt_at,''),race_date,race_id",
                (today.date().isoformat(), cutoff)).fetchall()
            backlog_jobs = [job for job in candidates if job[0] not in date_window][:backlog_max_races]
            jobs.extend(backlog_jobs)
        conn.executemany("""
            UPDATE result_sync_races SET attempts = attempts + 1, last_attempt_at = ?
            WHERE race_date = ? AND race_id = ?
        """, [(today.isoformat(), ds, race_id) for ds, race_id, _ in jobs])

    # Existing URLs identify the recovery targets. A listing can prove a
    # terminal state for many of them in one request; unseen or conflicting
    # entries still need their individual page. This does not declare the
    # rest of an old date complete or add unexamined races to its ledger.
    backlog_listing_failures = 0
    backlog_blocked = False
    backlog_listing_loads = 0
    for ds in sorted({job[0] for job in backlog_jobs}):
        if ds in getattr(scraper, "result_listing_evidence", {}):
            continue
        try:
            backlog_listing_loads += 1
            scraper.get_results_urls(ds, uk_only=True)
        except Exception as exc:
            backlog_listing_failures += 1
            logger.warning("Historical listing unavailable on %s: %s", ds, exc)
            if getattr(getattr(exc, "response", None), "status_code", None) in {403, 429, 503}:
                backlog_blocked = True
                break
    listing_closed = _close_listing_terminals(scraper, jobs, today)
    listing_closed_ids = set(listing_closed)
    jobs = [job for job in jobs if (job[0], job[1]) not in listing_closed_ids]
    backlog_ids = {(ds, identity) for ds, identity, _ in backlog_jobs}
    if backlog_blocked:
        jobs = [job for job in jobs if (job[0], job[1]) not in backlog_ids]
    backlog_completed = len(backlog_ids & listing_closed_ids)
    inserted = 0
    completed = 0
    not_run = 0
    pages_finished = 0
    inserted_dates: set[str] = set()
    # A DORMANT page inside the relist window may simply not be updated yet;
    # only older ones are known never to have run.
    settled_before = (today - timedelta(days=RECENT_RELIST_DAYS)).strftime("%Y-%m-%d")
    if jobs:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data_scraper import MAX_WORKERS, RaceNotRun

        logger.info("Fetching %d unfinished race pages (%d workers)", len(jobs), MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(scraper.scrape_race_result, url, date_str=ds,
                            raise_if_not_run=True): (ds, race_id)
                for ds, race_id, url in jobs
            }
            for future in as_completed(futures):
                ds, race_id = futures[future]
                closed_this_job = False
                try:
                    frame = pd.DataFrame(future.result())
                    _validate_race_results(frame, ds, race_id)
                    with _get_conn() as conn:
                        added = _replace_race_results(conn, frame)
                        conn.execute("""
                            UPDATE result_sync_races SET completed = 1, last_error = NULL
                            WHERE race_date = ? AND race_id = ?
                        """, (ds, race_id))
                    inserted += added
                    completed += 1
                    closed_this_job = True
                    if added:
                        inserted_dates.add(ds)
                except RaceNotRun as exc:
                    # Abandoned, void, never run or Arabian: close it with the
                    # reason, so "pending" keeps meaning "failed unexplained".
                    terminal = exc.reason != "DORMANT" or ds < settled_before
                    with _get_conn() as conn:
                        conn.execute("""
                            UPDATE result_sync_races SET completed = ?, last_error = ?
                            WHERE race_date = ? AND race_id = ?
                        """, (int(terminal), f"not run: {exc.reason}", ds, race_id))
                    if terminal:
                        completed += 1
                        not_run += 1
                        closed_this_job = True
                except Exception as exc:
                    logger.warning("Race %s on %s remains pending: %s", race_id, ds, exc)
                    with _get_conn() as conn:
                        conn.execute("""
                            UPDATE result_sync_races SET completed = 0, last_error = ?
                            WHERE race_date = ? AND race_id = ?
                        """, (str(exc), ds, race_id))
                if (ds, race_id) in backlog_ids:
                    backlog_completed += int(closed_this_job)
                pages_finished += 1
                if pages_finished % 25 == 0 or pages_finished == len(jobs):
                    logger.info("Result recovery: %d races closed; %d of %d pages finished",
                                completed+len(listing_closed), pages_finished, len(jobs))
    logger.info("Inserted %d new runner rows; %d races closed as not run; "
                "%d race requests remain pending", inserted, not_run, len(jobs) - completed)

    enrichment = {}
    if weather and date_window and not offline:
        try:
            from src.weather import backfill_weather_in_db
            enrichment["weather"] = {"state": "ok", "updated": backfill_weather_in_db(dates=sorted(date_window))}
        except Exception as exc:
            enrichment["weather"] = {"state": "error", "error": str(exc)}
    if enrich and not offline:
        try:
            from src.pedigree_backfill import backfill_pedigree
            enrichment["pedigree"] = {"state": "ok", "stored": backfill_pedigree(limit=pedigree_limit)}
        except Exception as exc:
            enrichment["pedigree"] = {"state": "error", "error": str(exc)}

    # Save a CSV copy for the pipeline. This MUST be the full history, not the
    # scrape window: feature engineering computes cumulative/rolling features
    # across each horse's whole career, and get_featured_dataset() rebuilds the
    # store from this CSV. Writing only `days_back` here previously clobbered
    # the feature store down to the scrape window.
    snapshot = publish_results_snapshot()
    lo, hi = min(all_dates, default="9999"), max(all_dates, default="0000")
    with _get_conn() as conn:
        pending = conn.execute("SELECT COUNT(*) FROM result_sync_races WHERE completed=0 AND race_date BETWEEN ? AND ?", (lo, hi)).fetchone()[0]
        listings = conn.execute("SELECT COUNT(*) FROM result_sync_dates WHERE listing_succeeded=0 AND race_date BETWEEN ? AND ?", (lo, hi)).fetchone()[0]
        total_pending = conn.execute("SELECT COUNT(*) FROM result_sync_races WHERE completed=0").fetchone()[0]
        change = conn.execute("SELECT MIN(race_date),COUNT(DISTINCT race_id) FROM result_source_changes WHERE revision>?", (before_revision,)).fetchone()
        latest = conn.execute("SELECT MAX(race_date) FROM results").fetchone()[0]
        current_revision = conn.execute("SELECT revision FROM result_source_state WHERE id=1").fetchone()[0]
        backlog_counts = {}
        if backlog_max_races:
            review = "(COALESCE(last_error,'') LIKE 'Review required:%' OR COALESCE(last_error,'') LIKE 'Missing or duplicate%')"
            count, needs_review, due = conn.execute(
                f"SELECT COUNT(*),COALESCE(SUM({review}),0),COALESCE(SUM(NOT {review} "
                "AND (last_attempt_at IS NULL OR last_attempt_at<?)),0) FROM result_sync_races "
                "WHERE completed=0 AND race_date<? AND (race_date<? OR race_date>?)",
                ((today-timedelta(days=1)).isoformat(), today.date().isoformat(), lo, hi)).fetchone()
            backlog_counts = {"review_required_races": needs_review, "due_races": due,
                              "retry_waiting_races": count-needs_review-due}
    incomplete = (pending or listings or current_revision != snapshot["revision"]
                  or any(v["state"] == "error" for v in enrichment.values())
                  or (backlog_max_races and total_pending-pending))
    return {"source": "sporting_life", "state": "partial" if incomplete else "ok",
            "start": min(all_dates, default=None), "end": max(all_dates, default=None), "latest_date": latest,
            "rows": snapshot["rows"], "new_rows": inserted, "changed_races": change[1], "changed_from": change[0],
            "requests": len(jobs), "pending_races": pending, "unverified_dates": listings,
            "listing_resolved_races": len(listing_closed),
            **({"backlog": {"state": "partial" if total_pending-pending else "ok",
                             "budget": backlog_max_races, "attempted": len(backlog_jobs),
                             "completed": backlog_completed, "unresolved_attempts": len(backlog_jobs)-backlog_completed,
                             "listing_loads": backlog_listing_loads,
                             "listing_failures": backlog_listing_failures, "aborted_blocked": backlog_blocked,
                             "remaining_races": total_pending-pending, **backlog_counts}}
               if backlog_max_races else {}),
            "older_pending_races": total_pending-pending, "snapshot": snapshot, "enrichment": enrichment}


def _close_listing_terminals(scraper, jobs, today):
    """Close only dated, unambiguous source exclusions with no accepted field."""
    if not jobs:
        return []
    from src.data_scraper import NOT_RUN_STAGES
    evidence = getattr(scraper, "result_listing_evidence", {})
    settled_before = (today-timedelta(days=RECENT_RELIST_DAYS)).date().isoformat()
    closed = []
    observed_at = datetime.now().isoformat()
    with _get_conn() as conn:
        _ensure_terminal_evidence(conn)
        for ds, identity, _ in jobs:
            item = evidence.get(ds, {}).get(identity, {})
            reason = item.get("reason")
            if (item.get("source_date") != ds or not item.get("source_url")
                    or len(str(item.get("payload_sha256") or "")) != 64
                    or reason not in NOT_RUN_STAGES | {"ARABIAN"}
                    or item.get("has_results") and reason != "ARABIAN"
                    or reason == "DORMANT" and ds >= settled_before):
                continue
            if conn.execute("SELECT 1 FROM results WHERE race_id=? LIMIT 1", (identity,)).fetchone():
                continue
            conn.execute("UPDATE result_sync_races SET completed=1,last_error=? WHERE race_date=? AND race_id=?",
                         (f"not run: {reason}", ds, identity))
            conn.execute("INSERT OR REPLACE INTO result_terminal_evidence "
                         "(race_date,race_id,reason,source_url,observed_at,payload_sha256) VALUES (?,?,?,?,?,?)",
                         (ds, identity, reason, item["source_url"], observed_at, item["payload_sha256"]))
            closed.append((ds, identity))
    if closed:
        logger.info("Resolved %d terminal races from dated source listings", len(closed))
    return closed


def _ensure_terminal_evidence(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS result_terminal_evidence ("
                 "race_date TEXT, race_id TEXT, reason TEXT, source_url TEXT, observed_at TEXT, "
                 "payload_sha256 TEXT, PRIMARY KEY(race_date,race_id))")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(result_terminal_evidence)")}
    if "payload_sha256" not in columns:
        conn.execute("ALTER TABLE result_terminal_evidence ADD COLUMN payload_sha256 TEXT")


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
    # scraping starts — so `python -m src.database --sync N > log.txt` exited
    # non-zero having silently done nothing at all. Force UTF-8 with
    # replacement so a console encoding can never abort a long scrape.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(
        description="Manage the historical race database"
    )
    parser.add_argument(
        "--sync",
        type=int,
        metavar="DAYS",
        help="Sync missing/incomplete races within DAYS, rechecking recent listings",
    )
    parser.add_argument(
        "--audit-history",
        action="store_true",
        help="Before syncing, queue every stored legacy date for a completeness "
             "relist (use with a --sync window that covers the history)",
    )
    parser.add_argument(
        "--results-only", action="store_true",
        help="Skip optional weather and pedigree backfills during results sync",
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

    if args.stats or (not args.sync and not args.export):
        s = db_stats()
        print("\n📦  Race Database Statistics")
        print("=" * 40)
        for k, v in s.items():
            print(f"  {k:20s}: {v}")
        print()

    if args.audit_history:
        if not args.sync:
            parser.error("--audit-history needs --sync DAYS covering the dates to relist")
        print(f"  Queued {queue_history_audit()} legacy dates for a completeness relist")

    if args.sync:
        print(f"\n🔄 Syncing database (last {args.sync} days)...")
        from src.source_pipeline import refresh_sources
        report = refresh_sources(days_back=args.sync, enrich=False)
        print(json.dumps(report, indent=2, default=str))
        if report["state"] != "ok":
            raise SystemExit(1)

    if args.export:
        df = load_from_database()
        _export_results_snapshot(df, args.export)
        print(f"  Exported {len(df)} records to {args.export}")
