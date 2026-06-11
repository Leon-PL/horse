"""Pedigree backfill from Sporting Life horse profile pages.

The results/racecard APIs never include breeding, so the ``sire``/``dam``/
``damsire`` columns in ``results`` have always been empty and the pedigree
features computed on them were inert. Profile pages
(``/racing/profiles/horse/{id}``) do carry sire, dam, damsire, foaling
date and colour — one request per horse, cached forever in the
``horse_pedigree`` table.

Note: the profile JSON's nested ``horse_reference.id`` values for
sire/dam are buggy (they echo the horse's own id), so pedigree is stored
and joined by NAME strings.

Usage::

    from src.pedigree_backfill import backfill_pedigree, apply_pedigree
    backfill_pedigree(limit=500)        # fetch 500 unknown horses (recent first)
    df = apply_pedigree(df)             # fill sire/dam/damsire on a results frame
"""

import logging
import sqlite3
import time
from datetime import datetime

import pandas as pd

import config
from src.database import DB_PATH, _get_conn

logger = logging.getLogger(__name__)

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


def init_pedigree_table() -> None:
    with _get_conn() as conn:
        conn.execute(_CREATE_PEDIGREE)


def _profile_url(horse_id: str) -> str:
    from src.data_scraper import SPORTING_LIFE_BASE

    return f"{SPORTING_LIFE_BASE}/racing/profiles/horse/{horse_id}"


def fetch_horse_pedigree(scraper, horse_id: str) -> dict | None:
    """Fetch one horse profile; returns a row dict or None on failure.

    A fetch that succeeds but has no breeding info still returns a row
    (with None fields) so the horse is not retried forever.
    """
    data = scraper._fetch_next_data(_profile_url(horse_id))
    if not data:
        return None
    profile = data.get("props", {}).get("pageProps", {}).get("profile") or {}
    if not profile:
        return None

    def _name(node) -> str | None:
        if isinstance(node, dict):
            return node.get("name") or None
        return None

    return {
        "horse_id": str(horse_id),
        "horse_name": profile.get("name"),
        "sire": _name(profile.get("sire")),
        "dam": _name(profile.get("dam")),
        "damsire": _name(profile.get("damsire")),
        "foaled": profile.get("foaled"),
        "colour": profile.get("colour"),
    }


def pending_horse_ids(limit: int | None = None) -> list[tuple[str, str]]:
    """Horses in results with no pedigree row yet, most recent runners first."""
    init_pedigree_table()
    query = """
        SELECT r.horse_id, MAX(r.race_date) AS last_seen
        FROM results r
        LEFT JOIN horse_pedigree p ON p.horse_id = r.horse_id
        WHERE r.horse_id IS NOT NULL AND r.horse_id != ''
          AND p.horse_id IS NULL
        GROUP BY r.horse_id
        ORDER BY last_seen DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with _get_conn() as conn:
        return [(row[0], row[1]) for row in conn.execute(query)]


def backfill_pedigree(
    limit: int | None = None,
    delay: float = 0.0,
    log_every: int = 100,
) -> int:
    """Fetch pedigree for horses that don't have one yet. Returns count stored.

    The scraper already enforces ``config.REQUEST_DELAY`` between requests,
    so *delay* is extra sleep on top (default none).
    """
    from src.data_scraper import SportingLifeScraper

    scraper = SportingLifeScraper()
    todo = pending_horse_ids(limit)
    if not todo:
        logger.info("Pedigree backfill: nothing to do")
        return 0
    _delay = float(delay)
    logger.info("Pedigree backfill: %d horses pending (extra delay %.1fs)", len(todo), _delay)

    stored = failures = 0
    conn = sqlite3.connect(DB_PATH)
    try:
        for n, (horse_id, _last_seen) in enumerate(todo, 1):
            try:
                row = fetch_horse_pedigree(scraper, horse_id)
            except Exception:
                logger.exception("profile fetch failed for %s", horse_id)
                row = None
            if row is None:
                failures += 1
                # Park unfetchable horses so the queue drains; retried only
                # if the placeholder row is deleted.
                row = {"horse_id": str(horse_id), "horse_name": None, "sire": None,
                       "dam": None, "damsire": None, "foaled": None, "colour": None}
            conn.execute(
                "INSERT OR REPLACE INTO horse_pedigree "
                "(horse_id, horse_name, sire, dam, damsire, foaled, colour, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (row["horse_id"], row["horse_name"], row["sire"], row["dam"],
                 row["damsire"], row["foaled"], row["colour"],
                 datetime.now().isoformat(timespec="seconds")),
            )
            stored += 1
            if n % 50 == 0:
                conn.commit()
            if n % log_every == 0:
                logger.info("  %d/%d fetched (%d failures)", n, len(todo), failures)
            if _delay > 0:
                time.sleep(_delay)
        conn.commit()
    finally:
        conn.close()
    logger.info("Pedigree backfill done: %d stored, %d failures", stored, failures)
    return stored


def load_pedigree() -> pd.DataFrame:
    init_pedigree_table()
    with _get_conn() as conn:
        return pd.read_sql_query(
            "SELECT horse_id, sire, dam, damsire, foaled FROM horse_pedigree", conn
        )


def apply_pedigree(df: pd.DataFrame) -> pd.DataFrame:
    """Fill sire/dam/damsire (and foaled) on a results frame from the cache."""
    if "horse_id" not in df.columns:
        return df
    ped = load_pedigree()
    if ped.empty:
        return df
    ped["horse_id"] = ped["horse_id"].astype(str)
    # Plain mapping avoids merge suffix headaches with existing null columns.
    idx = ped.set_index("horse_id")
    hid = df["horse_id"].astype(str)
    for col in ("sire", "dam", "damsire"):
        mapped = hid.map(idx[col])
        if col in df.columns:
            df[col] = df[col].where(df[col].notna() & (df[col] != ""), mapped)
        else:
            df[col] = mapped
    if "foaled" not in df.columns:
        df["foaled"] = hid.map(idx["foaled"])
    coverage = df["sire"].notna().mean()
    logger.info("Pedigree applied: %.1f%% of rows have a sire", 100 * coverage)
    return df
