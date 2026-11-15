"""Collect complete French and UAE race results from Sporting Life.

Every runner of every accepted race is kept, not only horses already known in
the UK/Irish archive. Selecting runners by later British or Irish appearances
would give training rows overseas form that a live card for the same raider
could not have, and rating updates need the whole field.

Sporting Life horse IDs are shared with the domestic archive, so a raider's
overseas starts join its UK/Irish career without name matching. Racing TV was
dropped as a second feed: it added almost no meetings, and its runners could
only be identified by matching them to horses already seen in the UK/Ireland.

These races are history only. They feed horse form, trip and rating sweeps;
they never become training or evaluation rows.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import logging
from pathlib import Path
import re

import pandas as pd
import requests

import config
from src import database
from src.ingestion.sporting_life import MAX_WORKERS, RaceNotRun, SportingLifeScraper
from src.ingestion.storage import atomic_parquet, writer_lock
from src.result_margins import cumulative_beaten_lengths
from src.result_status import result_labels
from src.utils import canonical_track_key

logger = logging.getLogger(__name__)
SL_COUNTRIES = {"france": "fr", "united arab emirates": "ae"}
COUNTRIES = frozenset(SL_COUNTRIES.values())
SCHEMA_VERSION = 2
COLUMNS = (
    "horse_id", "horse_name", "race_date", "off_time", "country", "track", "race_type", "surface",
    "distance_furlongs", "finish_position", "finish_pos_label", "num_runners", "lengths_behind",
    "beaten_lengths", "provider", "native_race_id", "source_url", "available_date",
)
_EXCLUDED = re.compile(r"\b(?:attel[eé]|mont[eé]|trot(?:ting)?|purebred\s+arabian|"
                       r"arab\s+race|trial\s+\d+)\b", re.I)
_SURFACES = {"turf": "Turf", "polytrack": "Polytrack", "tapeta": "Tapeta", "fibresand": "Fibresand",
             "sand": "Sand", "dirt": "Dirt", "all weather": "Allweather", "allweather": "Allweather"}
_FAILURES = (requests.RequestException, ValueError, TypeError, KeyError, FileNotFoundError)
# Sporting Life keeps some foreign result pages permanently empty. After this
# many days an empty page is recorded as unavailable instead of failing every
# refresh; a recent one stays a failure because it may still be published.
EMPTY_RESULT_GRACE_DAYS = 3


def native_path() -> Path:
    return Path(config.DATA_DIR) / "foreign_form" / f"races_v{SCHEMA_VERSION}.parquet"


def _track(name):
    key = canonical_track_key(name)
    return {"parislongchamp": "longchamp", "paris-longchamp": "longchamp"}.get(key, key)


def _surface(value):
    if not isinstance(value, str):
        return None
    return _SURFACES.get(value.strip().casefold())


def race_rows(frame: pd.DataFrame, info: dict, day: str) -> pd.DataFrame:
    """Validate one parsed Sporting Life result and return every runner."""
    database._validate_race_results(frame, day, info["race_id"])
    if not frame.region.eq("INT").all():
        raise ValueError("Foreign listing returned a domestic result")
    if frame.horse_id.isna().any() or frame.horse_id.duplicated().any():
        raise ValueError("Foreign result has a missing or repeated horse ID")
    frame = frame.copy()
    frame["finish_pos_label"] = result_labels(frame)
    # Domestic processed rows carry non-finishers as position 0 with a label.
    finished = frame.finish_pos_label.fillna("").eq("")
    frame["finish_position"] = pd.to_numeric(frame.finish_position, errors="raise").where(finished, 0)
    frame["beaten_lengths"] = cumulative_beaten_lengths(frame)
    available = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    return pd.DataFrame({
        "horse_id": frame.horse_id.astype("int64"),
        "horse_name": frame.horse_name.astype("string"),
        "race_date": day,
        "off_time": frame.off_time.astype("string"),
        "country": SL_COUNTRIES[info["country"].strip().lower()],
        "track": frame.track.map(_track),
        "race_type": frame.race_type.astype("string"),
        "surface": frame.surface.map(_surface).astype("string"),
        "distance_furlongs": pd.to_numeric(frame.distance_furlongs, errors="coerce"),
        "finish_position": pd.to_numeric(frame.finish_position, errors="raise").astype("int64"),
        "finish_pos_label": frame.finish_pos_label.astype("string"),
        "num_runners": pd.to_numeric(frame.num_runners, errors="raise").astype("int64"),
        "lengths_behind": pd.to_numeric(frame.lengths_behind, errors="coerce"),
        "beaten_lengths": pd.to_numeric(frame.beaten_lengths, errors="coerce"),
        "provider": "sl",
        "native_race_id": str(info["race_id"]),
        "source_url": info["url"],
        "available_date": available,
    }, columns=list(COLUMNS))


def _sl_race(scraper, info, day):
    try:
        frame = pd.DataFrame(scraper.scrape_race_result(info["url"], day, raise_if_not_run=True))
    except RaceNotRun:
        return None
    return race_rows(frame, info, day)


def refresh_foreign_form(start, end, *, replay=False):
    """Fetch every French and UAE flat/jumps result in a date range and upsert it atomically."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or last >= date.today():
        raise ValueError("Foreign form requires a valid range ending before today")
    path = native_path()
    with writer_lock(path.with_suffix(".lock")) as acquired:
        if not acquired:
            raise RuntimeError("Another foreign-form collection is running")
        scraper = SportingLifeScraper()
        scraper.raw_store_path = Path(config.DATA_DIR) / "sl_cache" / "raw.sqlite"
        scraper.offline = replay
        races, failures, unavailable = [], [], []
        days = [(first + timedelta(days=offset)).isoformat() for offset in range((last - first).days + 1)]
        # One pool for the whole range: the scraper's own limiter still caps
        # concurrency and spacing, but workers no longer idle between days.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            listings = {day: pool.submit(scraper.get_results_urls, day, countries=SL_COUNTRIES) for day in days}
            jobs, collected = [], set()
            for day, job in listings.items():
                try:
                    listing = job.result()
                    if day not in scraper.result_listing_evidence:
                        raise ValueError("Unreadable Sporting Life listing")
                except _FAILURES as exc:
                    failures.append(f"sl {day}: {exc}")
                    continue
                collected.add(day)
                listing = [info for info in listing if not _EXCLUDED.search(info["race_name"])]
                jobs += [(day, info, pool.submit(_sl_race, scraper, info, day)) for info in listing]
                logger.info("Foreign form %s: %d races listed", day, len(listing))
            for day, info, job in jobs:
                try:
                    race = job.result()
                except _FAILURES as exc:
                    settled = date.fromisoformat(day) <= date.today() - timedelta(days=EMPTY_RESULT_GRACE_DAYS)
                    if settled and str(exc) == "Empty or incomplete result response":
                        unavailable.append(f"sl {info['race_id']} ({day})")
                        continue
                    failures.append(f"sl {info['race_id']}: {exc}")
                    collected.discard(day)
                    continue
                if race is not None and not race.empty:
                    races.append(race)
        collected_days = sorted(collected)

        previous = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        current = pd.concat(races, ignore_index=True) if races else pd.DataFrame(columns=COLUMNS)
        keys = ["provider", "native_race_id", "horse_id"]
        # A re-collected race replaces its previous runners as a whole, so a
        # corrected field never keeps a runner the source has since removed.
        if not current.empty and not previous.empty:
            previous = previous.loc[~previous.native_race_id.isin(set(current.native_race_id))]
        frames = [f for f in (previous, current) if not f.empty]
        combined = pd.concat(frames, ignore_index=True) if frames else previous
        combined = combined.drop_duplicates(keys, keep="last").sort_values(
            ["race_date", "native_race_id", "finish_position"]).reset_index(drop=True)
        atomic_parquet(combined, path)
        accepted = accepted_foreign_form(combined)
        return {"state": "partial" if failures else "ok", "start": start, "end": end,
                "races": int(current.native_race_id.nunique()) if not current.empty else 0,
                "runners": len(current), "complete_days": len(collected_days),
                "retained_runners": len(accepted),
                "retained_races_by_country": accepted.drop_duplicates("native_race_id")
                .country.value_counts().to_dict(),
                "unavailable": unavailable, "failures": failures}


def accepted_foreign_form(frame):
    """Keep complete races; withhold a horse-day that appears in two races."""
    frame = frame.loc[frame.country.isin(COUNTRIES) & frame.provider.eq("sl")]
    keys = ["horse_id", "race_date"]
    clash = frame.loc[frame.duplicated(keys, keep=False), "native_race_id"]
    # A horse cannot run twice on one day; both races are suspect, so both go.
    frame = frame.loc[~frame.native_race_id.isin(set(clash))]
    return frame.sort_values(["race_date", "native_race_id", "finish_position"]).reset_index(drop=True)
