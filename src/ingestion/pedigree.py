"""Sporting Life pedigree collection and Racing TV foaling-date enrichment.

The results/racecard APIs never include breeding, so the ``sire``/``dam``/
``damsire`` columns in ``results`` have always been empty and the pedigree
features computed on them were inert. Profile pages
(``/racing/profiles/horse/{id}``) do carry sire, dam, damsire, foaling
date and colour — one request per horse, cached forever in the
``horse_pedigree`` table.

Note: the profile JSON's nested ``horse_reference.id`` values for
sire/dam are buggy (they echo the horse's own id), so pedigree is stored
and joined by NAME strings. ``src.database`` owns the table and
``src.data_processor.apply_pedigree`` joins it onto a results frame.
"""

import logging
import sqlite3
import time
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import config
from src import database
from src.ingestion import racing_tv
from src.ingestion.storage import writer_lock
from src.utils import normalise_horse_key

logger = logging.getLogger(__name__)

_RTV_EVIDENCE = [
    "foaled_rtv", "rtv_horse_id", "rtv_foaled_observed_at", "rtv_profile_url",
    "rtv_country_of_foaling", "rtv_coat_colour", "rtv_sire_country", "rtv_dam_country",
    "rtv_damsire", "rtv_damsire_country",
    "rtv_profile_overseas_starts", "rtv_profile_last_overseas_start",
]


_PROFILE_CANONICAL = ["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]
_PROFILE_RUNNERS = ["race_id", "runner_id", "horse_id", "horse_name", "in_results",
            "is_nonrunner", "res_horse__slug", "card_horse__slug", "card_horse_id"]
_PROFILE_RACES = ["race_id", "race_date", "track_name", "off_time_utc", "is_abandoned", "is_arab_race"]

def _overseas_form(profile, captured_at):
    """Count distinct, dated overseas starts already settled before capture day."""
    capture_day = date.fromisoformat(captured_at[:10])
    starts = {}
    for run in profile.get("performances") or []:
        country = ((run.get("track") or {}).get("country_code") or {}).get("racing")
        run_id = run.get("id")
        if (run.get("non_runner") is not False or not isinstance(run_id, int) or run_id <= 0
                or not isinstance(country, str) or len(country) != 3 or not country.isascii() or not country.isalpha()
                or country.upper() in {"UNK", "XXX"}):
            continue
        try:
            day = date.fromisoformat(run["meeting_date"])
        except (KeyError, ValueError, TypeError):
            continue
        if day < capture_day:
            starts.setdefault(run_id, set()).add((day, country.upper()))
    distinct = [next(iter(records)) for records in starts.values() if len(records) == 1]
    dates = [day for day, country in distinct if country not in {"GBR", "IRE"}]
    return (len(dates), max(dates).isoformat()) if dates else (None, None)


def racing_tv_foaling_observations() -> pd.DataFrame:
    """Read profile facts and observed overseas form with their capture times."""
    from src.ingestion.racing_tv import SITE_BASE, RawStore, horse_profile_path

    path = Path(config.DATA_DIR) / "rtv_cache" / "rtv_raw.sqlite"
    columns = ["horse_name", "sire", "dam", *_RTV_EVIDENCE]
    rows = []
    if path.is_file():
        store = RawStore(str(path), readonly=True)
        try:
            for _, fetched_at, profile in store.horse_profiles():
                slug = profile.get("slug")
                country = profile.get("country_code") or {}
                colours = profile.get("display_colors") or []
                rows.append((profile.get("name"), profile.get("sire_name"), profile.get("dam_name"),
                             profile.get("foaling_date"), profile.get("id"), fetched_at,
                             SITE_BASE + horse_profile_path(slug) if isinstance(slug, str) and slug else None,
                             country.get("racing") if isinstance(country, dict) else None,
                             ", ".join(colours) if isinstance(colours, list) and all(isinstance(c, str) for c in colours) else None,
                             profile.get("sire_country_code"), profile.get("dam_country_code"),
                             profile.get("dam_sire_name"), profile.get("dam_sire_country_code"),
                             *_overseas_form(profile, fetched_at)))
        finally:
            store.close()
    return pd.DataFrame(rows, columns=columns)


def _parse_foaling_date(values: pd.Series) -> pd.Series:
    values = values.astype("string").str.strip()
    iso = values.str.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T00:00:00(?:\.0+)?(?:Z|\+00:00)?)?")
    return pd.to_datetime(values.where(iso).str[:10], format="%Y-%m-%d", errors="coerce")


def resolve_foaling_dates(pedigree: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Link unique name/parentage/year matches and retain both reported dates."""
    from src.ingestion.racing_tv import _pedigree_key

    resolved = pedigree.copy().rename(columns={"foaled": "foaled_sporting_life"})
    observations = profiles.copy()
    keys = ["_horse", "_sire", "_dam", "_year"]
    for frame, date_col in ((resolved, "foaled_sporting_life"), (observations, "foaled_rtv")):
        for name in ("horse", "sire", "dam"):
            values = frame["horse_name" if name == "horse" else name].astype("string").str.strip()
            normalise = normalise_horse_key if name == "horse" else _pedigree_key
            frame["_" + name] = values.map(normalise, na_action="ignore").replace("", pd.NA)
        parsed = _parse_foaling_date(frame[date_col])
        frame["_year"] = parsed.dt.year
        frame[date_col] = parsed.dt.strftime("%Y-%m-%d").astype("string")

    observations["rtv_horse_id"] = observations.rtv_horse_id.astype("string").str.replace(r"\.0$", "", regex=True)
    observations = observations.dropna(subset=[*keys, "rtv_horse_id"])
    observations = observations.loc[observations.rtv_horse_id.str.fullmatch(r"[1-9]\d*")]
    observations = observations.sort_values("rtv_foaled_observed_at").drop_duplicates(
        [*keys, "rtv_horse_id", "foaled_rtv"], keep="last")
    observations = observations.loc[~observations.duplicated("rtv_horse_id", keep=False)
                                    & ~observations.duplicated(keys, keep=False)]
    eligible = resolved.dropna(subset=keys)
    links = eligible.loc[~eligible.duplicated(keys, keep=False), ["horse_id", *keys]].merge(
        observations, on=keys, validate="one_to_one")
    resolved = resolved.drop(columns=keys).merge(
        links.reindex(columns=["horse_id", *_RTV_EVIDENCE]),
        on="horse_id", how="left", validate="one_to_one")

    sl = resolved.foaled_sporting_life.where(~resolved.foaled_sporting_life.str.endswith("-01-01", na=False))
    rtv = resolved.foaled_rtv.where(~resolved.foaled_rtv.str.endswith("-01-01", na=False))
    conflict = sl.notna() & rtv.notna() & sl.ne(rtv).fillna(False)
    resolved["foaled"] = rtv.fillna(sl).mask(conflict)
    resolved["foaling_date_source"] = pd.Series(pd.NA, index=resolved.index, dtype="string")
    resolved.loc[sl.notna() & rtv.isna(), "foaling_date_source"] = "sporting_life"
    resolved.loc[rtv.notna() & ~conflict, "foaling_date_source"] = "racing_tv"
    resolved.loc[conflict, "foaling_date_source"] = "conflict"
    return resolved


def _profile_url(horse_id: str) -> str:
    from src.ingestion.sporting_life import SPORTING_LIFE_BASE

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
    database.init_pedigree_table()
    query = """
        SELECT r.horse_id, MAX(r.race_date) AS last_seen
        FROM results r
        LEFT JOIN horse_pedigree p ON p.horse_id = r.horse_id
        LEFT JOIN pedigree_fetch_state s ON s.horse_id = r.horse_id
        WHERE r.horse_id IS NOT NULL AND r.horse_id != ''
          AND (p.horse_id IS NULL OR (s.horse_id IS NULL AND p.horse_name IS NULL
               AND p.sire IS NULL AND p.dam IS NULL AND p.damsire IS NULL))
          AND (s.next_retry_at IS NULL OR s.next_retry_at <= ?)
        GROUP BY r.horse_id
        ORDER BY last_seen DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    with database._get_conn() as conn:
        return [(row[0], row[1]) for row in conn.execute(query, (datetime.now().isoformat(),))]


def backfill_pedigree(
    limit: int | None = None,
    delay: float = 0.0,
    log_every: int = 100,
) -> int:
    """Fetch pedigree for horses that don't have one yet. Returns count stored.

    The scraper already enforces ``config.REQUEST_DELAY`` between requests,
    so *delay* is extra sleep on top (default none).
    """
    from src.ingestion.sporting_life import SportingLifeScraper

    scraper = SportingLifeScraper()
    scraper.raw_store_path = Path(database.DB_PATH).parent/"sl_cache/raw.sqlite"
    todo = pending_horse_ids(limit)
    if not todo:
        logger.info("Pedigree backfill: nothing to do")
        return 0
    _delay = float(delay)
    logger.info("Pedigree backfill: %d horses pending (extra delay %.1fs)", len(todo), _delay)

    stored = failures = 0
    conn = sqlite3.connect(database.DB_PATH)
    try:
        for n, (horse_id, _last_seen) in enumerate(todo, 1):
            try:
                row = fetch_horse_pedigree(scraper, horse_id)
            except Exception:
                logger.exception("profile fetch failed for %s", horse_id)
                row = None
            if row is None:
                failures += 1
                previous = conn.execute("SELECT attempts FROM pedigree_fetch_state WHERE horse_id=?", (str(horse_id),)).fetchone()
                attempts = previous[0]+1 if previous else 1
                retry = datetime.now()+timedelta(hours=min(24, 2**min(attempts-1, 5)))
                conn.execute("INSERT OR REPLACE INTO pedigree_fetch_state VALUES (?,?,?,?,?)",
                             (str(horse_id), "retryable_error", attempts, retry.isoformat(), "Profile unavailable or invalid"))
                conn.commit()
                continue
            conn.execute(
                "INSERT OR REPLACE INTO horse_pedigree "
                "(horse_id, horse_name, sire, dam, damsire, foaled, colour, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (row["horse_id"], row["horse_name"], row["sire"], row["dam"],
                 row["damsire"], row["foaled"], row["colour"],
                 datetime.now().isoformat(timespec="seconds")),
            )
            stored += 1
            conn.execute("INSERT INTO pedigree_fetch_state VALUES (?,?,1,NULL,NULL) "
                         "ON CONFLICT(horse_id) DO UPDATE SET status=excluded.status, "
                         "attempts=pedigree_fetch_state.attempts+1,next_retry_at=NULL,last_error=NULL",
                         (str(horse_id), "available" if row.get("sire") else "confirmed_empty"))
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


def rtv_profile_targets(start=None, end=None):
    """Use accepted RTV race observations and unique canonical runner links."""
    from src.ingestion.pipeline import match_runners
    from src.ingestion.rtv_observations import observation_tables

    db = Path(config.DATA_DIR) / "races.db"
    if not db.exists():
        return []
    with closing(sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True)) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='horse_pedigree'").fetchone():
            return []
        first, last = con.execute("SELECT min(race_date), max(race_date) FROM results").fetchone()
        pending = pd.read_sql_query(
            "SELECT horse_id FROM horse_pedigree WHERE substr(foaled,6,5)='01-01' "
            "AND sire IS NOT NULL AND dam IS NOT NULL AND horse_name IS NOT NULL", con)
    start, end = start or first, end or last
    if not start or not end or pending.empty:
        return []
    canonical = database.result_rows(start, end, _PROFILE_CANONICAL, db_path=db)
    if canonical.empty or not canonical.horse_id.astype(str).isin(pending.horse_id).any():
        return []
    tables = observation_tables(start, end, names=("runners",),
                                column_overrides={"runners": _PROFILE_RUNNERS, "races": _PROFILE_RACES})
    runners = tables["runners"].reindex(columns=_PROFILE_RUNNERS)
    runners = runners.loc[runners.in_results.eq(True) & ~runners.is_nonrunner.eq(True)]
    links = match_runners(tables["races"], runners, canonical)
    if links.empty:
        return []
    links = links.merge(runners, left_on=["rtv_race_id", "rtv_runner_id"],
                        right_on=["race_id", "runner_id"], validate="one_to_one")
    links["rtv_horse_id"] = pd.to_numeric(links.horse_id, errors="coerce")
    links = links.dropna(subset=["rtv_horse_id", "sl_horse_id"])
    pairs = links[["sl_horse_id", "rtv_horse_id"]].drop_duplicates()
    pairs = pairs.loc[~pairs.sl_horse_id.duplicated(keep=False) & ~pairs.rtv_horse_id.duplicated(keep=False)]
    pairs = pairs.loc[pairs.sl_horse_id.astype(str).isin(pending.horse_id)]
    links = links.merge(pairs, on=["sl_horse_id", "rtv_horse_id"], validate="many_to_one")
    card_slug = links.card_horse__slug.where(pd.to_numeric(links.card_horse_id, errors="coerce").eq(links.rtv_horse_id))
    links["slug"] = links.res_horse__slug.fillna(card_slug).astype("string").str.strip("/")
    links = links.loc[links.slug.str.fullmatch(r"[a-z0-9][a-z0-9-]*", na=False)]
    counts = links.groupby("slug").rtv_horse_id.transform("nunique")
    links = links.loc[counts.eq(1)].sort_values(["race_date", "slug"], ascending=[False, True])
    return links.drop_duplicates("rtv_horse_id").slug.tolist()


def backfill_rtv_foaling_profiles(*, limit=None, start=None, end=None, progress=None):
    """Resume cached lookups with bounded batches and the shared RTV rate cap."""
    if limit is not None and limit < 1:
        raise ValueError("The profile budget must be positive")
    root = Path(config.DATA_DIR) / "rtv_cache"
    result = {"state": "running", "candidates": 0, "pending": 0, "requested": 0, "ok": 0,
              "unavailable": 0, "retryable": 0}

    def report():
        if progress is not None:
            progress(dict(result))

    with writer_lock(root / ".horse_profiles.lock") as acquired:
        if not acquired:
            return {**result, "state": "busy"}
        report()
        targets = rtv_profile_targets(start, end)
        if not targets:
            return {**result, "state": "cached"}
        with closing(racing_tv.RawStore(str(root / "rtv_raw.sqlite"))) as store:
            pending = [slug for slug in targets
                       if store.status(racing_tv.horse_profile_path(slug)) not in racing_tv.FINAL_STATUSES]
            result.update(candidates=len(targets), pending=len(pending))
            report()
            client = racing_tv.RTVClient()
            wanted = pending[:limit]
            for offset in range(0, len(wanted), 100):
                batch = wanted[offset:offset+100]
                try:
                    outcome = racing_tv.scrape_horse_profiles(batch, client, store)
                except (racing_tv.RTVError, requests.RequestException) as exc:
                    statuses = [store.status(racing_tv.horse_profile_path(slug)) for slug in batch]
                    terminal = sum(status in racing_tv.FINAL_STATUSES for status in statuses)
                    result["ok"] += statuses.count(200)
                    result["unavailable"] += statuses.count(404)
                    result["requested"] += terminal
                    result["pending"] -= terminal
                    logger.warning("Racing TV horse profile collection stopped: %s", exc)
                    result.update(state="partial", error=str(exc))
                    break
                for key in ("requested", "ok", "unavailable", "retryable"):
                    result[key] += outcome[key]
                result["pending"] -= outcome["ok"] + outcome["unavailable"]
                report()
                if outcome["retryable"] == len(batch):
                    result.update(state="partial", error="Racing TV returned no usable response for the batch")
                    break
    if result["state"] == "running":
        result["state"] = "partial" if result["retryable"] else "ok" if result["requested"] else "cached"
    report()
    return result


def backfill_rtv_foaling_and_publish(*, limit=None, progress=None):
    """Standalone profile-only operation using the common dataset publication."""
    from src.ingestion.publication import capture_pedigree_dataset

    result = backfill_rtv_foaling_profiles(limit=limit, progress=progress)
    if result["state"] != "busy":
        if progress is not None:
            progress({**result, "state": "publishing"})
        manifest = capture_pedigree_dataset()
        result["dataset_manifest_id"] = manifest["id"]
        result["foaling_coverage"] = manifest["sources"]["pedigree"]["provenance"]["foaling_coverage"]
    return result
