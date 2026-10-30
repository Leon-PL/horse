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
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pandas as pd

import config

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
    scraped_at        TEXT    DEFAULT (datetime('now')),

    UNIQUE(race_id, horse_name)
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
        conn.execute(_CREATE_SYNC_DATES)
        conn.execute(_CREATE_SYNC_RACES)
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
    logger.info(f"Database ready at {DB_PATH}")


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

    Rows whose ``(race_id, horse_name)`` already exist are silently
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

    # Keep only columns that exist in both the DF and the schema
    cols = [c for c in _COLUMNS if c in df.columns]
    subset = df[cols].copy()

    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO results ({col_names}) VALUES ({placeholders})"

    changes_before = conn.total_changes
    conn.executemany(sql, subset.values.tolist())
    inserted = conn.total_changes - changes_before
    _insert_result_context(conn, df)
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
    return frame.drop_duplicates(["race_id", "horse_name"], keep="last").reset_index(drop=True)


# -----------------------------------------------------------------------
# Query helpers
# -----------------------------------------------------------------------

def _count_rows() -> int:
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]


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
    required = {"race_id", "race_date", "horse_name", "finish_position"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError("Empty or incomplete result response")
    if not (df["race_id"].astype(str).eq(race_id).all()
            and df["race_date"].eq(date_str).all()):
        raise ValueError("Result response belongs to another race or date")
    names = df["horse_name"]
    if (names.isna().any() or names.astype(str).str.strip().eq("").any()
            or names.duplicated().any()):
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
    previous = {row[0] for row in conn.execute(
        "SELECT horse_name FROM results WHERE race_id = ?", (race_id,),
    )}
    names = set(df["horse_name"])
    cols = [c for c in _COLUMNS if c in df.columns]
    updates = [f"{col} = excluded.{col}" for col in cols
               if col not in {"race_id", "horse_name"}]
    updates.append("scraped_at = datetime('now')")
    sql = (f"INSERT INTO results ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' for _ in cols)}) "
           f"ON CONFLICT(race_id, horse_name) DO UPDATE SET {', '.join(updates)}")
    conn.executemany(sql, df[cols].values.tolist())
    obsolete = previous - names
    conn.executemany("DELETE FROM results WHERE race_id = ? AND horse_name = ?",
                     [(race_id, name) for name in obsolete])
    _insert_result_context(conn, df)
    return len(names - previous)


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


def sync_database(days_back: int = 90, *, enrich: bool = True) -> pd.DataFrame:
    """Resume result ingestion and atomically export the complete history.

    Keep one idle connection for the operation's lifetime. Transactions use
    their own short-lived handles; the idle handle prevents SQLite from
    checkpointing the WAL after every individual race's connection closes.
    It never opens a read or write transaction across network requests.
    Set ``enrich=False`` to collect results without optional source backfills.
    """
    if days_back < 0:
        raise ValueError("days_back must be non-negative")
    init_db()
    with _get_conn() as idle:
        # Touch the initialized database so this connection actually joins
        # the WAL; merely opening a connection does not open its WAL pager.
        idle.execute("SELECT race_id FROM results LIMIT 1").fetchall()
        return _sync_database(days_back, enrich=enrich)


def _sync_database(days_back: int, *, enrich: bool = True) -> pd.DataFrame:
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
    date_window = set(all_dates)
    recent_dates = set(all_dates[:RECENT_RELIST_DAYS])
    with _get_conn() as conn:
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
    scraper = SportingLifeScraper()

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
            if not links:
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
        jobs = [r for r in conn.execute("""
            SELECT race_date, race_id, url FROM result_sync_races
            WHERE completed = 0 ORDER BY race_date, race_id
        """) if r[0] in date_window]
        conn.executemany("""
            UPDATE result_sync_races SET attempts = attempts + 1, last_attempt_at = datetime('now')
            WHERE race_date = ? AND race_id = ?
        """, [(ds, race_id) for ds, race_id, _ in jobs])

    inserted = 0
    completed = 0
    not_run = 0
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
                except Exception as exc:
                    logger.warning("Race %s on %s remains pending: %s", race_id, ds, exc)
                    with _get_conn() as conn:
                        conn.execute("""
                            UPDATE result_sync_races SET last_error = ?
                            WHERE race_date = ? AND race_id = ?
                        """, (str(exc), ds, race_id))
    logger.info("Inserted %d new runner rows; %d races closed as not run; "
                "%d race requests remain pending", inserted, not_run, len(jobs) - completed)

    if inserted > 0 and enrich:
        try:
            from src.weather import backfill_weather_in_db
            backfill_weather_in_db(dates=sorted(inserted_dates))
        except Exception as exc:
            logger.warning("Weather backfill for new rows failed: %s", exc)
        try:
            from src.pedigree_backfill import backfill_pedigree
            backfill_pedigree()
        except Exception as exc:
            logger.warning("Pedigree backfill for new horses failed: %s", exc)

    # Save a CSV copy for the pipeline. This MUST be the full history, not the
    # scrape window: feature engineering computes cumulative/rolling features
    # across each horse's whole career, and get_featured_dataset() rebuilds the
    # store from this CSV. Writing only `days_back` here previously clobbered
    # the feature store down to the scrape window.
    full_df = load_from_database()
    csv_path = os.path.join(config.RAW_DATA_DIR, "race_results.csv")
    _export_results_snapshot(full_df, csv_path)
    logger.info(f"  Saved full CSV snapshot ({len(full_df):,} rows) → {csv_path}")

    return full_df


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

    return {
        "total_runners": total,
        "total_dates": dates,
        "total_tracks": tracks,
        "total_races": races,
        "earliest_date": min_date or "—",
        "latest_date": max_date or "—",
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
        df = sync_database(days_back=args.sync, enrich=not args.results_only)
        print(f"  Done — {len(df)} records available\n")

    if args.export:
        df = load_from_database()
        df.to_csv(args.export, index=False)
        print(f"  Exported {len(df)} records to {args.export}")
