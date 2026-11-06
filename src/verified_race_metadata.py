"""Immutable, source-confirmed corrections to published historical metadata."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
import config


@lru_cache(maxsize=4)
def _read(path, modified_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))["races"]


@lru_cache(maxsize=4)
def _aliases(path, modified_ns, normalise):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tables = {}
    for role in ("jockey", "trainer"):
        names = pd.Series(list(payload.get(role, {}).items()), dtype=object)
        if names.empty:
            continue
        old, new = normalise(names.str[0]), normalise(names.str[1])
        mapping = dict(zip(old[old.ne(new)], new[old.ne(new)]))
        if set(mapping) & set(mapping.values()):
            raise ValueError(f"{role} aliases chain: {sorted(set(mapping) & set(mapping.values()))}")
        tables[role] = mapping
    return tables


def apply_connection_aliases(frame, normalise):
    """Give each jockey and trainer one name across Sporting Life's renamings.

    ``normalise`` is the caller's name cleaner, so reference names are compared
    in the frame's own form. Names absent from the table are left unchanged.
    """
    path = Path(config.BASE_DIR) / "reference/connection_aliases.json"
    for role, mapping in _aliases(str(path), path.stat().st_mtime_ns, normalise).items():
        if role in frame:
            hit = frame[role].isin(mapping)
            if hit.any():
                frame.loc[hit, role] = frame.loc[hit, role].map(mapping)
    return frame


def _rows_by_event(identifiers, known):
    """Row positions of each corrected event, in order of first appearance."""
    hits = np.flatnonzero(known.to_numpy())
    events = identifiers.to_numpy(dtype=object)[hits]
    return {event: hits[at] for event, at in pd.Series(events).groupby(events, sort=False).indices.items()}


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
    race_type = frame.columns.get_loc("race_type")
    for identity, rows in _rows_by_event(identifiers, known).items():
        correction = records[identity]
        dates = pd.to_datetime(frame["race_date"].iloc[rows], errors="raise").dt.normalize()
        tracks = frame["track"].iloc[rows].astype("string").str.strip().str.casefold()
        if not dates.eq(pd.Timestamp(correction["date"])).all() or not tracks.eq(correction["track"].casefold()).all():
            raise ValueError(f"Verified race {identity} disagrees with its source date/course")
        frame.iloc[rows, race_type] = correction["race_type"]
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
    for identity, event in _rows_by_event(race_ids, known).items():
        correction = records[identity]
        dates = pd.to_datetime(frame["race_date"].iloc[event], errors="raise").dt.normalize()
        tracks = frame["track"].iloc[event].astype("string").str.strip().str.casefold()
        if not dates.eq(pd.Timestamp(correction["date"])).all() or not tracks.eq(correction["track"].casefold()).all():
            raise ValueError(f"Verified rating event {identity} disagrees with its source date/course")
        event_rows = pd.Series(False, index=frame.index)
        event_rows.iloc[event] = True
        for horse_id, values in correction["runners"].items():
            rows = event_rows & horse_ids.eq(horse_id)
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
