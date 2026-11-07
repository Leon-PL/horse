"""Pure per-run feature derivations from At The Races sectionals.

These helpers turn one runner's *raw post-race* ATR sectional record (the
columns added by :func:`merge_atr_sectionals`) into a
set of **per-run scalars and tags**.  They describe how the horse ran in
*that* race — they are NOT model features on their own.

``src.feature_engineer.add_atr_features`` consumes these, then rolls each one
over a horse's *prior* runs (leakage-safe ``shift(1)``) to build the actual
features.  Keeping the derivations here (pure, row-local, no history) makes
them unit-testable in isolation and keeps the leakage boundary crisp.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import re
from functools import lru_cache
from itertools import chain

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from src.ingestion.atr_tabs import ALL_ATR_COLS, load_atr_cache
from src.utils import (
    normalise_horse_key as _normalise_horse_key,
    normalise_off_time_key as _normalise_off_time_key,
    normalise_track_key as _normalise_track_key,
)

logger = logging.getLogger(__name__)

# ── In-running comment tagger ──────────────────────────────────────────
# ATR comments use a stable race-reading vocabulary.  Each tag is a 0/1
# flag per run; downstream we roll them into recent *rates* per horse.
# Patterns are deliberately specific to avoid false positives.
_COMMENT_PATTERNS: dict[str, str] = {
    # Running style
    "atr_cmt_front": r"\b(?:led|made all|made virtually all|disputed (?:the )?lead|"
                     r"prominent|racing prominently|raced prominently|chased (?:the )?leaders?)\b",
    "atr_cmt_held_up": r"\b(?:held up|in rear|in last|towards (?:the )?rear|behind|"
                       r"settled (?:in )?rear|dropped (?:to the )?rear)\b",
    # Trouble in running (next-time positive — ran into trouble, excuse)
    "atr_cmt_trouble": r"\b(?:hampered|short of room|no room|not clear run|denied (?:a )?clear run|"
                       r"badly drawn|squeezed|checked|blocked|nowhere to go|forced wide|carried wide|"
                       r"short of (?:racing )?room|brought down)\b",
    # Stamina / galloped to the line (positive)
    "atr_cmt_stayed_on": r"\b(?:stayed on|kept on|ran on|rallied|finished well|"
                         r"stayed on well|kept on well|ran on well|good late)\b",
    # Didn't see it out (negative)
    "atr_cmt_weakened": r"\b(?:weakened|faded|tired|no extra|found nothing|lost touch|"
                        r"beaten (?:over|before)|weakened (?:quickly|over))\b",
    # Outpaced / never travelling (negative, often trip/class too short or long)
    "atr_cmt_outpaced": r"\b(?:outpaced|struggling|always behind|never (?:on terms|dangerous|nearer))\b",
    # Greenness / quirks (juvenile-improver / quirk signal)
    "atr_cmt_green": r"\b(?:green|unruly|awkward|reluctant|hung (?:left|right|badly)|"
                     r"edged (?:left|right)|wandered|flashed (?:its |the )?tail|carried head)\b",
    # National Hunt jumping errors (replaces RTV's jump metrics)
    "atr_cmt_nh_error": r"\b(?:mistake|blundered?|not fluent|untidy|sketchy|fiddled|pecked|"
                        r"hit \d|jumped (?:slowly|left|right|poorly|deliberately))\b",
    # Eased when beaten — its closing sectional is UNRELIABLE (the trap).
    "atr_cmt_eased": r"\b(?:eased|not knocked about|never knocked about|not extended|"
                     r"eased down|never (?:asked|nearer))\b",
    # Won/finished with plenty in hand (genuine winning impression)
    "atr_cmt_comfortable": r"\b(?:comfortably|readily|easily|cosily|going away|ridden out|"
                           r"in command|hard held)\b",
}
_COMMENT_RES: dict[str, re.Pattern] = {
    k: re.compile(v, re.IGNORECASE) for k, v in _COMMENT_PATTERNS.items()
}

COMMENT_TAGS: list[str] = list(_COMMENT_PATTERNS)


def tag_comment(text) -> dict[str, int]:
    """Return 0/1 flags for each comment tag in *text*."""
    s = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    return {tag: int(bool(rx.search(s))) for tag, rx in _COMMENT_RES.items()}


# ── Sectional-array parsing ────────────────────────────────────────────

def _as_list(v) -> list[float]:
    """Decode a JSON-string / list cell into a list of floats (or [])."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        seq = list(v)
    elif isinstance(v, str):
        if not v.strip():
            return []
        try:
            seq = json.loads(v)
        except (ValueError, TypeError):
            return []
    else:
        return []
    out: list[float] = []
    for x in seq:
        try:
            f = float(x)
        except (ValueError, TypeError):
            continue
        if not np.isnan(f):
            out.append(f)
    return out


def _sequence(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def _speed_sections(names: list) -> tuple[bool, ...]:
    """Which labelled sections hold a speed over a stretch of the course."""
    return tuple(
        bool(re.search(r'\bto\b', str(name), re.IGNORECASE)) and 'mph' not in str(name).casefold()
        for name in names
    )


@lru_cache(maxsize=4096)
def _speed_sections_of(labels: str) -> tuple[bool, ...]:
    return _speed_sections(_sequence(labels))


def _sectional_speeds(values, labels) -> list[float]:
    """Older caches may mix instantaneous speeds and acceleration seconds in.

    Respect source labels when present; never interpret a 0–30 mph time as
    a section's closing speed. Unlabelled legacy three-section inputs retain
    their existing contract.
    """
    names = _sequence(labels)
    raw = _sequence(values)
    if names:
        if len(names) != len(raw):
            return []
        raw = [value for value, keep in zip(raw, _speed_sections(names)) if keep]
    return _as_list(raw)


def _decode_cells(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """All JSON array cells as one float array plus row offsets.

    Joining the cells has the decoder read every row in one call. Items keep
    their positions, with nulls as NaN. Returns None when some cell needs the
    row-by-row handling of :func:`_as_list`.
    """
    texts = []
    for cell in cells:
        if isinstance(cell, (list, tuple, np.ndarray)):
            return None
        texts.append(cell if isinstance(cell, str) and cell.strip() else "[]")
    try:
        rows = json.loads("[" + ",".join(texts) + "]")
        if len(rows) != len(texts) or not all(type(row) is list for row in rows):
            return None
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.fromiter(map(len, rows), dtype=np.int64, count=len(rows)))
        return np.fromiter(chain.from_iterable(rows), dtype=np.float64, count=offsets[-1]), offsets
    except (TypeError, ValueError):
        return None


def _flatten(rows: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.fromiter(map(len, rows), dtype=np.int64, count=len(rows)))
    return np.fromiter(chain.from_iterable(rows), dtype=np.float64, count=offsets[-1]), offsets


def _speed_cells(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(speeds, offsets, keep): every section, and which of them are speeds."""
    decoded = _decode_cells(values)
    if decoded is None or not all(isinstance(cell, str) or not isinstance(cell, (list, tuple, np.ndarray))
                                  for cell in labels):
        flat, offsets = _flatten([_sectional_speeds(speeds, names) for speeds, names in zip(values, labels)])
        return flat, offsets, np.ones(len(flat), dtype=bool)
    flat, offsets = decoded
    keep = np.ones(len(flat), dtype=bool)
    for row, cell in enumerate(labels):
        sections = _speed_sections_of(cell) if isinstance(cell, str) else ()
        if sections:
            start, stop = offsets[row], offsets[row + 1]
            keep[start:stop] = sections if len(sections) == stop - start else False
    return flat, offsets, keep


@nb.njit(cache=True)
def _nb_speed_metrics(values, keep, offsets, early, top, closing, surge):
    """First, highest and last speed, and the largest rise between sections."""
    for row in range(len(offsets) - 1):
        count, first, last, peak, jump = 0, 0.0, 0.0, 0.0, 0.0
        for at in range(offsets[row], offsets[row + 1]):
            value = values[at]
            if not keep[at] or value != value:
                continue
            if count == 0:
                first, peak = value, value
            else:
                if value > peak:
                    peak = value
                step = value - last
                if count == 1 or step > jump:
                    jump = step
            last = value
            count += 1
        if count > 0:
            early[row], top[row], closing[row] = first, peak, last
        if count >= 2:
            surge[row] = jump if jump > 0.0 else 0.0


@nb.njit
def _numpy_sum(values, n):
    """``numpy.sum`` of a short array: the same block order, so the same bits."""
    if n < 8:
        total = -0.0
        for i in range(n):
            total += values[i]
        return 0.0 + total
    blocks = values[:8].copy()
    i = 8
    while i < n - n % 8:
        for j in range(8):
            blocks[j] += values[i + j]
        i += 8
    total = ((blocks[0] + blocks[1]) + (blocks[2] + blocks[3])) + ((blocks[4] + blocks[5]) + (blocks[6] + blocks[7]))
    while i < n:
        total += values[i]
        i += 1
    return 0.0 + total


@nb.njit(cache=True)
def _nb_energy_gap(energy, energy_offsets, optimum, optimum_offsets, out):
    """Minus the mean distance from the optimum over the shared sections."""
    gaps = np.empty(128)
    for row in range(len(energy_offsets) - 1):
        e, e_stop = energy_offsets[row], energy_offsets[row + 1]
        o, o_stop = optimum_offsets[row], optimum_offsets[row + 1]
        count = 0
        while True:
            while e < e_stop and energy[e] != energy[e]:
                e += 1
            while o < o_stop and optimum[o] != optimum[o]:
                o += 1
            if e == e_stop or o == o_stop:
                break
            if count == len(gaps):
                gaps = np.concatenate((gaps, gaps))
            gaps[count] = abs(energy[e] - optimum[o])
            count += 1
            e += 1
            o += 1
        if count > 0:
            out[row] = -(_numpy_sum(gaps, count) / count)


def _energy_gap(energy_cells: np.ndarray, optimum_cells: np.ndarray) -> np.ndarray:
    energy = _decode_cells(energy_cells) or _flatten([_as_list(cell) for cell in energy_cells])
    optimum = _decode_cells(optimum_cells) or _flatten([_as_list(cell) for cell in optimum_cells])
    out = np.full(len(energy_cells), np.nan)
    _nb_energy_gap(energy[0], energy[1], optimum[0], optimum[1], out)
    return out


def _comment_tags(comments: list[str]) -> dict[str, np.ndarray]:
    """0/1 tag columns. RE2 reads ASCII comments far faster than ``re``.

    The two engines agree on ASCII text. They can differ on word boundaries and
    case folding beyond it, so any other comment stays with ``re``.
    """
    text = pa.array(comments, type=pa.string())
    other = np.flatnonzero(~pc.string_is_ascii(text).to_numpy(zero_copy_only=False))
    tags = {}
    for tag, pattern in _COMMENT_PATTERNS.items():
        flags = pc.match_substring_regex(text, pattern, ignore_case=True)
        tags[tag] = flags.to_numpy(zero_copy_only=False).astype(np.float64)
        for row in other:
            tags[tag][row] = float(_COMMENT_RES[tag].search(comments[row]) is not None)
    return tags


def efficiency_error_scores(frame: pd.DataFrame) -> np.ndarray:
    """Native ATR efficiency error: lower is better, with no upper grade cutoff.

    Unrated source rows sometimes contain a zero placeholder and no profile.
    Such a zero is missing information, not a perfectly efficient performance.
    Legacy inputs without a grade column retain their numeric contract.
    """
    source = frame.get("atr_efficiency_score", pd.Series(np.nan, index=frame.index))
    values = pd.to_numeric(source, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)
    values[(~np.isfinite(values)) | (values < 0)] = np.nan
    if "atr_efficiency_grade" in frame:
        grade = frame.atr_efficiency_grade.fillna("").astype(str).str.strip().str.upper()
        ungraded = ~grade.isin(["A++", "A+", "A", "B", "C", "D", "E", "F"]).to_numpy()
        for row in np.flatnonzero((values == 0) & ungraded):
            energy = frame.atr_energy_dist.iloc[row] if "atr_energy_dist" in frame else None
            optimum = frame.atr_optimum_energy.iloc[row] if "atr_optimum_energy" in frame else None
            if not _as_list(energy) or not _as_list(optimum):
                values[row] = np.nan
    return values


# Per-run continuous metrics and whether higher is better (drives best/min,
# field-relative sign and the composite score).
RUN_METRICS: list[tuple[str, bool]] = [
    ("atr_finishing_speed_pct", True),   # closing strength (% of avg)
    ("atr_early_pace", True),            # first-segment speed → front-running
    ("atr_max_speed", True),             # top sectional speed
    ("atr_closing_speed", True),         # last-segment speed
    ("atr_turn_of_foot", True),          # biggest segment-to-segment acceleration
    ("atr_efficiency_score", False),     # deviation/error score: lower is better
    ("atr_energy_eff", True),            # -mean|energy-optimum| (closer to ideal)
    ("atr_peak_stride_length", True),    # biomechanics
    ("atr_peak_stride_freq", True),
    ("atr_upgrade_clean", True),         # finished worse than sectionals imply (unlucky)
]

# Plausible bands — values outside are NaN'd (parse can pick up junk).
_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "atr_finishing_speed_pct": (40.0, 160.0),
    "atr_early_pace": (10.0, 60.0),
    "atr_max_speed": (10.0, 60.0),
    "atr_closing_speed": (10.0, 60.0),
    "atr_turn_of_foot": (0.0, 30.0),
    "atr_efficiency_score": (0.0, np.inf),  # D/E/F scores are valid observations
    "atr_peak_stride_length": (10.0, 35.0),
    "atr_peak_stride_freq": (1.0, 4.0),
    "atr_closing_in_hand": (40.0, 160.0),
}


def compute_atr_run_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-run scalars + comment tags for every row of *df*.

    Expects the raw ATR columns (post-merge) plus ``won`` / ``lengths_behind``
    for cleaning the "unlucky" signal.  Returns a DataFrame aligned to
    ``df.index`` holding only the new per-run columns (all describing the
    current race — the caller rolls them to prior runs).
    """
    n = len(df)
    out: dict[str, np.ndarray] = {}

    def col(name):
        return df[name].to_numpy() if name in df.columns else np.full(n, np.nan, dtype=object)

    speeds_c = col("atr_sectional_speeds")
    speed_labels_c = col('atr_sectional_speed_labels')
    energy_c = col("atr_energy_dist")
    optimum_c = col("atr_optimum_energy")
    fs_c = pd.to_numeric(df.get("atr_finishing_speed_pct", np.nan), errors="coerce").to_numpy()
    eff_c = efficiency_error_scores(df)
    psl_c = pd.to_numeric(df.get("atr_peak_stride_length", np.nan), errors="coerce").to_numpy()
    psf_c = pd.to_numeric(df.get("atr_peak_stride_freq", np.nan), errors="coerce").to_numpy()
    pos_c = pd.to_numeric(df.get("atr_official_position", np.nan), errors="coerce").to_numpy()
    adj_c = pd.to_numeric(df.get("atr_adj_placing", np.nan), errors="coerce").to_numpy()
    won_c = pd.to_numeric(df.get("won", np.nan), errors="coerce").to_numpy()
    lb_c = pd.to_numeric(df.get("lengths_behind", np.nan), errors="coerce").to_numpy()

    # Comment tags. With no comment the tags are unknown, not zero: the published
    # tags stay missing, so the rolled rates skip that run rather than counting
    # it as a run where the comment said nothing. The gating flags below read
    # the unmasked tags, where a missing comment simply raises no flag.
    cmt_series = df.get("atr_in_running_comment", pd.Series([""] * n, index=df.index)).fillna("").astype(str)
    has_comment = cmt_series.str.strip().ne("").to_numpy()
    tags = _comment_tags(cmt_series.tolist())

    early = np.full(n, np.nan)
    mx = np.full(n, np.nan)
    closing = np.full(n, np.nan)
    tof = np.full(n, np.nan)

    eased_a = tags["atr_cmt_eased"] == 1
    stayed_a = tags["atr_cmt_stayed_on"] == 1
    comf_a = tags["atr_cmt_comfortable"] == 1

    speeds, speed_offsets, speed_keep = _speed_cells(speeds_c, speed_labels_c)
    _nb_speed_metrics(speeds, speed_keep, speed_offsets, early, mx, closing, tof)
    energy_eff = _energy_gap(energy_c, optimum_c)

    # Sectional upgrade (finished worse than sectionals imply), CLEANED:
    # discard when the horse was eased while beaten (its fast finishing
    # sectional is an artefact of not being asked), or tailed off. No reliable
    # signal means the horse ran to form.
    beaten = np.isnan(won_c) | (won_c == 0)
    unreliable_up = (eased_a & beaten) | ((lb_c > 12) & beaten)
    upgrade = np.where(unreliable_up, 0.0, np.clip(pos_c - adj_c, -6.0, 6.0))
    upgrade[np.isnan(pos_c) | np.isnan(adj_c)] = np.nan

    # Closing speed "in hand": credit a strong finishing % only when the
    # finish was genuine (won/ran-on) and not eased-when-beaten.
    genuine = (won_c == 1) | stayed_a | comf_a
    unreliable = eased_a & (won_c == 0)
    in_hand = np.where(genuine & ~unreliable, fs_c, np.nan)

    out["atr_finishing_speed_pct"] = fs_c.astype(float)
    out["atr_early_pace"] = early
    out["atr_max_speed"] = mx
    out["atr_closing_speed"] = closing
    out["atr_turn_of_foot"] = tof
    out["atr_efficiency_score"] = eff_c.astype(float)
    out["atr_energy_eff"] = energy_eff
    out["atr_peak_stride_length"] = psl_c.astype(float)
    out["atr_peak_stride_freq"] = psf_c.astype(float)
    out["atr_upgrade_clean"] = upgrade
    out["atr_closing_in_hand"] = in_hand
    for t in COMMENT_TAGS:
        out[t] = np.where(has_comment, tags[t], np.nan)

    res = pd.DataFrame(out, index=df.index)

    # Sanitise out-of-band continuous values.
    for c, (lo, hi) in _PLAUSIBLE.items():
        if c in res.columns:
            v = res[c]
            res.loc[v.notna() & ((~np.isfinite(v)) | (v < lo) | (v > hi)), c] = np.nan
    return res


# ---------------------------------------------------------------------------
# Join the accepted cache onto a race frame (used by feature_engineer)
# ---------------------------------------------------------------------------

_ATR_MERGE_KEYS = ["_rd", "_trk", "_ot", "_hn"]


def _atr_merge_table(cache: pd.DataFrame) -> pd.DataFrame:
    """The cache as unique merge keys plus every ATR column."""
    cache = cache.copy()
    cache["_rd"] = pd.to_datetime(cache["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    cache["_trk"] = cache["track"].map(_normalise_track_key)
    cache["_ot"] = cache["off_time"].map(_normalise_off_time_key)
    cache["_hn"] = cache["horse_name"].map(_normalise_horse_key)

    for c in ALL_ATR_COLS:
        if c not in cache.columns:
            cache[c] = float("nan")

    merge_cols = _ATR_MERGE_KEYS + ALL_ATR_COLS
    keys = _ATR_MERGE_KEYS
    cache_slim = cache[merge_cols].drop_duplicates(subset=keys+["atr_horse_id"], keep="last")
    return cache_slim.loc[~cache_slim.duplicated(keys,keep=False)]


_merge_table_scope: list | None = None


@contextmanager
def atr_merge_scope():
    """Prepare the cache once for a build that merges it chunk by chunk."""
    global _merge_table_scope
    _merge_table_scope = []
    try:
        yield
    finally:
        _merge_table_scope = None


def _merge_table() -> pd.DataFrame | None:
    """The prepared cache, or None when there is nothing to join."""
    if _merge_table_scope:
        return _merge_table_scope[0]
    cache = load_atr_cache()
    table = None if cache.empty else _atr_merge_table(cache)
    if _merge_table_scope is not None:
        _merge_table_scope.append(table)
    return table


def merge_atr_sectionals(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join cached ATR sectionals onto a race DataFrame.

    Matches on ``(race_date, track, off_time, horse_name)`` using the
    shared normalisers.  All ATR columns are added (NaN where absent);
    per-furlong series remain JSON strings (see :data:`ATR_ARRAY_COLS`).
    Only the keys take part in the join, so a wide frame is not copied.
    """
    cache_slim = _merge_table()
    if cache_slim is None:
        for c in ALL_ATR_COLS:
            df[c] = float("nan")
        return df

    keys = _ATR_MERGE_KEYS
    df["_rd"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["_trk"] = df["track"].map(_normalise_track_key)
    df["_ot"] = df["off_time"].map(_normalise_off_time_key)
    df["_hn"] = df["horse_name"].map(_normalise_horse_key)
    ambiguous_left = df.duplicated(keys,keep=False).to_numpy()

    joined = df[keys].merge(
        cache_slim.loc[cache_slim["_rd"].isin(df["_rd"].unique())],
        on=keys, how="left", indicator="_atr_merge", validate="many_to_one",
    )
    merged = pd.concat(
        [df.drop(columns=keys).reset_index(drop=True), joined.drop(columns=keys)], axis=1,
    )

    for c in ALL_ATR_COLS:
        if c not in merged.columns:
            merged[c] = float("nan")
    merged.loc[ambiguous_left,ALL_ATR_COLS] = float("nan")

    matched = (merged["_atr_merge"] == "both") & ~ambiguous_left
    logger.info(
        "ATR merge: key matches %d / %d (%.1f%%)",
        int(matched.sum()), len(merged), 100 * matched.sum() / max(1, len(merged)),
    )
    return merged.drop(columns=["_atr_merge"], errors="ignore")
