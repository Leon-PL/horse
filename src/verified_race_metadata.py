"""Immutable, source-confirmed corrections to published historical metadata."""
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd
import config


@lru_cache(maxsize=4)
def _read(path, modified_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))["races"]


def apply_verified_race_types(frame):
    """Correct complete fields by event identity, never by winner/price/name guesses."""
    if "race_id" not in frame or "race_type" not in frame:
        return frame
    path = Path(config.BASE_DIR) / "reference/verified_race_types.json"
    records = _read(str(path), path.stat().st_mtime_ns)
    identifiers = frame.race_id.astype("string").str.replace(r"\.0$", "", regex=True)
    known = identifiers.isin(records)
    if not known.any():
        return frame
    for identity in identifiers.loc[known].unique():
        correction = records[identity]
        rows = identifiers.eq(identity)
        dates = pd.to_datetime(frame.loc[rows, "race_date"], errors="raise").dt.normalize()
        tracks = frame.loc[rows, "track"].astype("string").str.strip().str.casefold()
        if not dates.eq(pd.Timestamp(correction["date"])).all() or not tracks.eq(correction["track"].casefold()).all():
            raise ValueError(f"Verified race {identity} disagrees with its source date/course")
        frame.loc[rows, "race_type"] = correction["race_type"]
    return frame


def apply_verified_runner_ratings(frame):
    """Restore independently verified race-day marks by event AND horse ID.

    Some published source fields contain single-digit marks instead of their
    actual ratings. Never reconstruct a missing digit arithmetically or replace
    other low ratings: the catalog records explicit independent source values.
    """
    if not {"race_id", "horse_id", "official_rating"}.issubset(frame.columns):
        return frame
    path = Path(config.BASE_DIR) / "reference/verified_runner_ratings.json"
    records = _read(str(path), path.stat().st_mtime_ns)
    race_ids = frame.race_id.astype("string").str.replace(r"\.0$", "", regex=True)
    known = race_ids.isin(records)
    if not known.any():
        return frame
    horse_ids = frame.horse_id.astype("string").str.replace(r"\.0$", "", regex=True)
    from src.utils import normalise_horse_key
    for identity in race_ids.loc[known].unique():
        correction = records[identity]
        event = race_ids.eq(identity)
        dates = pd.to_datetime(frame.loc[event, "race_date"], errors="raise").dt.normalize()
        tracks = frame.loc[event, "track"].astype("string").str.strip().str.casefold()
        if not dates.eq(pd.Timestamp(correction["date"])).all() or not tracks.eq(correction["track"].casefold()).all():
            raise ValueError(f"Verified rating event {identity} disagrees with its source date/course")
        for horse_id, values in correction["runners"].items():
            rows = event & horse_ids.eq(horse_id)
            if not rows.any():
                continue
            if not frame.loc[rows, "horse_name"].map(normalise_horse_key).eq(normalise_horse_key(values["horse_name"])).all():
                raise ValueError(f"Verified rating horse {horse_id} disagrees with its identity")
            frame.loc[rows, "official_rating"] = values["official_rating"]
            # Do not create a partially populated indicator on fresh raw data:
            # the processor will derive the complete column after correction.
            if "has_official_rating" in frame:
                frame.loc[rows, "has_official_rating"] = 1
    return frame
