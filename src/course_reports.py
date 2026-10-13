"""Per-race course reports: stalls position and rail movements (GB, BHA).

ACTION_PLAN 5a-bis. Two facts every GB going report publishes and the results
database lacks:

* **Stalls position** by distance ("5f & 6f - Stands side; remainder -
  Inside"). Draw bias depends on which rail the stalls are against, and at
  many courses that changes from one meeting to the next, so a static
  per-course draw bias cannot express it.
* **Rail movements**, as the yards they add to each race. The database stores
  the advertised trip, which never changes between meetings: Chester's
  1m2f70y on 2025-05-07 was run over 1m2f96y.

Both come from the BHA API, collected by ``scripts/fetch_bha_going.py`` into
``data/processed/bha_going.parquet`` (fixture-level text) and
``data/processed/bha_races.parquet`` (race-level ``distanceChange``).

:func:`attach_course_reports` joins them onto results inside
``data_processor.process_data``, so the values are part of the processed data
and covered by ``featured_dataset.data_signature``: a later backfill of these
files invalidates the feature store exactly like any other history change.
Irish races, jumps races (no stalls) and fixtures the BHA did not publish get
missing values, never a guess.
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd

import config
from src.utils import canonical_track_key, normalise_off_time_key

logger = logging.getLogger(__name__)

GOING_PATH = os.path.join(config.PROCESSED_DATA_DIR, "bha_going.parquet")
RACES_PATH = os.path.join(config.PROCESSED_DATA_DIR, "bha_races.parquet")

#: Stalls positions, as parsed. ``stands``/``far`` describe straight-course
#: starts; ``inside``/``outside`` round-course starts; ``centre`` either.
POSITIONS = ("inside", "outside", "centre", "stands", "far")

#: BHA course names ``canonical_track_key`` does not map to the results' names.
_BHA_ALIAS = {"great yarmouth": "yarmouth", "royal ascot": "ascot"}

#: Longest start on each course's straight, in furlongs (approximate, public
#: course guides). Used only when a report says "straight" or "round" without
#: naming the race's distance; absent courses have no straight course.
STRAIGHT_MAX_F = {
    "ascot": 8, "ayr": 6, "bath": 0, "beverley": 5, "carlisle": 6,
    "catterick": 5, "chepstow": 8, "doncaster": 8, "epsom downs": 5,
    "ffos las": 8, "goodwood": 6, "hamilton": 6, "haydock": 6,
    "leicester": 7, "lingfield": 7.7, "musselburgh": 5, "newbury": 8,
    "newcastle": 8, "newmarket": 10, "nottingham": 6, "redcar": 8,
    "ripon": 6, "salisbury": 8, "sandown": 5, "southwell": 5, "thirsk": 6,
    "windsor": 6, "yarmouth": 8, "york": 6,
}

#: How far (furlongs) a distance named in a report may sit from the race's
#: advertised distance and still be that start. Reports round ("7f" for a
#: 7f36y start); 0.5f (110 yards) keeps distinct starts apart.
_MATCH_TOL_F = 0.5


def bha_track_key(name) -> str:
    """Join key for BHA fixtures: apply to BOTH sides (BHA courses and DB tracks)."""
    key = canonical_track_key(name)
    return _BHA_ALIAS.get(key, key)


# ── Stalls text parser ──────────────────────────────────────────────────────

_POS_PATTERNS = [
    ("stands", r"(?:grand\s*)?stands?'?\s*(?:side|rail)|near\s*side|\bstands\b"),
    ("far", r"far\s*(?:side|rail)|\bfar\b"),
    ("inside", r"\binside|\binner"),
    ("outside", r"\boutside|\bouter"),
    ("centre", r"centre|center|middle|central|\bmid\b"),
]
_GROUP_PATTERNS = [
    ("all", r"remainders?|\brest\b|all\s+others?(?:\s+races)?|all\s+other\s+races"
            r"|all\s+races|\bothers\b|\ball\b"),
    # "Back straight starts" are round-course starts, not the straight course.
    ("back_straight", r"back\s+straight(?:\s+starts?)?"),
    ("straight", r"straight(?:\s+course)?"),
    ("sprint", r"sprints?"),
    ("round_mile", r"round\s+mile"),
    ("round", r"round(?:\s+course)?"),
]
_M = r"(\d+)\s*(?:miles?|m)(?![a-z])"
_F = r"(\d+(?:\.\d+)?)\s*(?:furlongs?|f)(?![a-z])"
_Y = r"(\d+)\s*(?:yards?|yds?|y)(?![a-z])"
_DIST = rf"(?:{_M}\s*(?:{_F}\s*)?(?:{_Y})?|{_F}\s*(?:{_Y})?)"
_M_RE, _F_RE, _Y_RE = re.compile(_M), re.compile(_F), re.compile(_Y)
_TOKEN_RE = re.compile(
    "|".join(
        [rf"(?P<pos_{k}>{p})" for k, p in _POS_PATTERNS]
        + [rf"(?P<grp_{k}>{p})" for k, p in _GROUP_PATTERNS]
        + [rf"(?P<dist>{_DIST})", r"(?P<bare>\b\d{1,2}\b)", r"(?P<except>\bexcept\b)"]
    )
)
#: Newmarket names the course in use ("Far Side Course"); that is not a stalls
#: position and would otherwise read as one.
_COURSE_NAME_RE = re.compile(r"\b(?:far|stands?'?|stand)\s+side\s+course\b")


_CONJUNCTION_RE = re.compile(r"\band\b|&|,|/|\+")


def _tokens(text: str) -> list[tuple[str, object, bool]]:
    """``(kind, value, joined)`` tokens; *joined* = a conjunction precedes it."""
    s = str(text).lower().replace("½", ".5")
    s = re.sub(r"(\d)\s*1/2", r"\1.5", s)
    s = _COURSE_NAME_RE.sub(" ", s)
    out: list[tuple[str, object, bool]] = []
    prev_end = 0
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup or ""
        joined = bool(_CONJUNCTION_RE.search(s[prev_end:m.start()]))
        prev_end = m.end()
        if kind.startswith("pos_"):
            out.append(("pos", kind[4:], joined))
        elif kind == "grp_round_mile":
            out.append(("dist", 8.0, joined))
        elif kind == "grp_back_straight":
            out.append(("grp", "round", joined))
        elif kind.startswith("grp_"):
            out.append(("grp", kind[4:], joined))
        elif kind == "except":
            out.append(("except", None, joined))
        elif kind == "dist":
            text_dist = m.group("dist")
            mm, ff, yy = _M_RE.search(text_dist), _F_RE.search(text_dist), _Y_RE.search(text_dist)
            furlongs = (8.0 * float(mm.group(1)) if mm else 0.0) \
                + (float(ff.group(1)) if ff else 0.0) \
                + (float(yy.group(1)) / 220.0 if yy else 0.0)
            if 4.0 <= furlongs <= 36.0:
                out.append(("dist", round(furlongs, 3), joined))
        elif kind == "bare":
            # "5,6,7,1m Inner": bare numbers in a distance list are furlongs.
            value = float(m.group("bare"))
            if 4.0 <= value <= 24.0:
                out.append(("dist", value, joined))
    # "Outside/Stand side - 6f": consecutive positions name one start; keep the last.
    merged: list[tuple[str, object, bool]] = []
    for tok in out:
        if merged and tok[0] == "pos" and merged[-1][0] == "pos":
            merged[-1] = tok
        else:
            merged.append(tok)
    return merged


def _strip_qualifiers(specs: list[tuple[str, object, bool]]) -> list[tuple[str, object]]:
    """Drop a straight/round group that only qualifies the distances after it.

    "Straight Course: 5F: Stands side" assigns stands side to 5f, not to every
    straight start; "Straight Course and 2 miles - Stand side" assigns it to
    both, which the conjunction tells apart.
    """
    kept = []
    for i, (kind, spec, _joined) in enumerate(specs):
        nxt = specs[i + 1] if i + 1 < len(specs) else None
        if (kind == "grp" and spec in ("straight", "round") and nxt
                and nxt[0] == "dist" and not nxt[2]):
            continue
        kept.append((kind, spec))
    return kept


def parse_stalls(text) -> list[tuple[str, object, str]]:
    """Parse a stalls report into ``[(kind, spec, position), ...]``.

    ``kind`` is ``"dist"`` (spec = furlongs) or ``"grp"`` (spec = ``all``,
    ``straight``, ``round`` or ``sprint``). Reports come in two orders:
    distances first ("5f - Centre; remainder - Inside", about 93%) and
    position first ("Centre: 5f, 6f. Inside: 1m", "Inside, except 1m2f
    outside"); the first meaningful token decides which.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    toks = _tokens(text)
    if not toks:
        return []
    out: list[tuple[str, object, str]] = []
    positions = [t for t in toks if t[0] == "pos"]
    specs_present = any(t[0] in ("dist", "grp") for t in toks)
    if len(positions) == 1 and not specs_present:
        return [("grp", "all", positions[0][1])]

    def emit(specs, position):
        for kind, spec in _strip_qualifiers(specs):
            out.append((kind, spec, position))

    i = 0
    if toks[0][0] == "pos":
        # Position first: each position owns the specs that follow it, until
        # "except" turns the open position into the default and the rest of the
        # report reverts to distances-first.
        current, owned = None, []
        while i < len(toks):
            kind, value, joined = toks[i]
            if kind == "pos":
                if current is not None:
                    emit(owned, current) if owned else out.append(("grp", "all", current))
                current, owned = value, []
            elif kind == "except":
                if current is not None and not owned:
                    out.append(("grp", "all", current))
                    current = None
                i += 1
                break
            else:
                owned.append((kind, value, joined))
            i += 1
        else:
            if current is not None:
                emit(owned, current) if owned else out.append(("grp", "all", current))
            return out
        if current is not None and owned:
            emit(owned, current)
    pending: list[tuple[str, object, bool]] = []
    for kind, value, joined in toks[i:]:
        if kind == "pos":
            if pending:
                emit(pending, value)
                pending = []
        elif kind in ("dist", "grp"):
            pending.append((kind, value, joined))
    return out


def stalls_for_race(parsed, distance_f, track_key: str):
    """The stalls position for one race, or ``None`` when the report is silent.

    A distance named in the report wins; then "straight"/"round"/"sprint"
    groups (by the course's straight length); then the report's default.
    """
    if not parsed or distance_f is None or not np.isfinite(distance_f):
        return None
    dists = [(abs(spec - distance_f), pos) for kind, spec, pos in parsed if kind == "dist"]
    if dists:
        gap, pos = min(dists, key=lambda t: t[0])
        if gap <= _MATCH_TOL_F:
            return pos
    groups = {spec: pos for kind, spec, pos in parsed if kind == "grp"}
    straight_max = STRAIGHT_MAX_F.get(track_key, 0)
    on_straight = distance_f <= straight_max + 0.25
    if on_straight and "straight" in groups:
        return groups["straight"]
    if not on_straight and "round" in groups:
        return groups["round"]
    if distance_f <= 7.5 and "sprint" in groups:
        return groups["sprint"]
    return groups.get("all")


# ── Join onto results ───────────────────────────────────────────────────────

def _read(path: str, columns: list[str]) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except (OSError, ValueError, KeyError):
        logger.warning("Could not read %s; course reports left missing", path, exc_info=True)
        return None


def attach_course_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``stalls_position`` and ``rail_distance_change_yards`` per race.

    ``stalls_position`` is one of :data:`POSITIONS` for GB flat races whose
    report names it, else missing. ``rail_distance_change_yards`` is the BHA's
    ``distanceChange`` for GB races it lists (0 = no rail movement), else
    missing.
    """
    df = df.copy()
    df["stalls_position"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df["rail_distance_change_yards"] = np.nan
    needed = {"race_date", "track", "race_id"}
    if df.empty or not needed.issubset(df.columns):
        return df

    dates = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    tkey = df["track"].map(bha_track_key)

    going = _read(GOING_PATH, ["race_date", "course", "fixture_type", "stalls", "abandoned"])
    if going is not None and "distance_furlongs" in df.columns:
        going = going[(going["abandoned"].fillna(0) == 0) & going["stalls"].notna()].copy()
        going["track_key"] = going["course"].map(bha_track_key)
        going["race_date"] = going["race_date"].astype(str)
        # One report per course-day; a second session the same day is rare and
        # its stalls text is the same course's.
        going = going.drop_duplicates(["race_date", "track_key"])
        parsed = {(r.race_date, r.track_key): parse_stalls(r.stalls)
                  for r in going.itertuples(index=False)}
        is_flat = df.get("race_type", pd.Series("Flat", index=df.index)).astype(str).eq("Flat")
        races = pd.DataFrame({
            "race_id": df["race_id"], "d": dates, "t": tkey,
            "dist": pd.to_numeric(df["distance_furlongs"], errors="coerce"), "flat": is_flat,
        }).drop_duplicates("race_id")
        races = races[races["flat"]]
        pos = {
            r.race_id: stalls_for_race(parsed.get((r.d, r.t)), r.dist, r.t)
            for r in races.itertuples(index=False)
        }
        df["stalls_position"] = df["race_id"].map(pos).astype("string")

    bha_races = _read(RACES_PATH, ["race_date", "race_time", "course", "distance_change_yards",
                                    "abandoned"])
    if bha_races is not None and "off_time" in df.columns:
        bha_races = bha_races[bha_races["abandoned"].fillna(0) == 0].copy()
        bha_races["k"] = (bha_races["race_date"].astype(str) + "|"
                          + bha_races["course"].map(bha_track_key) + "|"
                          + bha_races["race_time"].map(normalise_off_time_key))
        # Divisions share a time only when the BHA lists them separately with
        # the same start; their distances are the same race's.
        change = bha_races.drop_duplicates("k").set_index("k")["distance_change_yards"]
        key = dates + "|" + tkey + "|" + df["off_time"].map(normalise_off_time_key)
        df["rail_distance_change_yards"] = key.map(change).astype("float64")
    return df
