"""Pure per-run feature derivations from At The Races sectionals.

These helpers turn one runner's *raw post-race* ATR sectional record (a row
of the selected dataset's ``atr/sectionals`` table) into a set of **per-run
scalars and tags**.  They describe how the horse ran in *that* race — they
are NOT model features on their own.

:func:`published_atr_rolling` joins those rows to the canonical results by
their saved race and horse IDs, then rolls each one over a horse's *prior*
starts that were available at the decision.  Keeping the derivations pure,
row-local and history-free makes them unit-testable in isolation and keeps
the leakage boundary crisp.
"""

from __future__ import annotations

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

from src.ingestion.atr_tabs import ALL_ATR_COLS
from src.utils import normalise_off_time_key as _normalise_off_time_key

logger = logging.getLogger(__name__)

# ── In-running comment tagger ──────────────────────────────────────────
# ATR comments use a stable race-reading vocabulary.  Each tag is a 0/1
# flag per run; downstream we roll them into recent *rates* per horse.
# Patterns are deliberately specific to avoid false positives.
_COMMENT_PATTERNS: dict[str, str] = {
    # Running style, read from the early clause only (see _early_clause)
    "atr_cmt_front": r"\b(?:led|made all|made virtually all|disputed (?:the )?lead|"
                     r"prominent|racing prominently|raced prominently|chased (?:the )?(?:leaders?|winner)|"
                     r"track(?:ed|ing) (?:the )?leaders?|pressed (?:the )?leaders?|close[- ]up|handy)\b",
    "atr_cmt_held_up": r"\b(?:held up|in rear|in last|towards (?:the )?rear|behind|"
                       r"settled (?:in )?rear|dropped (?:to the )?rear|waited with|off the pace)\b",
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
    # Not "never nearer": that is a closer still making ground at the line.
    "atr_cmt_eased": r"\b(?:eased|not knocked about|never knocked about|not extended|"
                     r"eased down|never asked)\b",
    # Won/finished with plenty in hand (genuine winning impression)
    "atr_cmt_comfortable": r"\b(?:comfortably|readily|easily|cosily|going away|ridden out|"
                           r"in command|hard held)\b",
}
_COMMENT_RES: dict[str, re.Pattern] = {
    k: re.compile(v, re.IGNORECASE) for k, v in _COMMENT_PATTERNS.items()
}

COMMENT_TAGS: list[str] = list(_COMMENT_PATTERNS)

# Running style is where a horse raced early. A comment narrates the whole
# race, so "held up in rear, ..., led inside final furlong" is a hold-up ride
# that won; matched anywhere, "led" made the front tag mostly a win flag.
# Style tags therefore read only the first clause, of the first three, that
# states a position; clauses before it describe the start or the preliminaries.
_STYLE_TAGS = frozenset({"atr_cmt_front", "atr_cmt_held_up"})
_EARLY_CLAUSES = 3
_POSITION = (r"\b(?:led|made (?:virtually )?all|disputed|prominent|prominently|chased|track(?:ed|ing)|pressed|"
             r"close[- ]up|handy|with (?:the )?leaders?|in touch|mid-?division|midfield|held up|waited with|rear|last|"
             r"behind|near leaders|off the pace|detached|tailed off|raced in (?:second|third|fourth|2nd|3rd|4th))\b")
_POSITION_RE = re.compile(_POSITION, re.IGNORECASE)

# Phrases rewritten before tagging, in both engines (RE2 has no look-ahead):
# "behind leaders" places a horse with the leaders rather than held up, and
# "ridden out wide" describes a route, not a comfortable finish.
_REWRITES: tuple[tuple[str, str], ...] = (
    (r"(?i)\bbehind (?:the )?lead(?:ers?|ing)\b", "near leaders"),
    (r"(?i)\bridden out wide\b", "ridden wide"),
)
_REWRITE_RES = tuple((re.compile(pattern), replacement) for pattern, replacement in _REWRITES)


def _rewrite(text: str) -> str:
    for regex, replacement in _REWRITE_RES:
        text = regex.sub(replacement, text)
    return text


def _early_clause(text: str) -> str:
    """The first clause, of the first three, that states a running position ('' if none)."""
    for clause in text.split(",", _EARLY_CLAUSES)[:_EARLY_CLAUSES]:
        if _POSITION_RE.search(clause):
            return clause
    return ""


def tag_comment(text) -> dict[str, int]:
    """Return 0/1 flags for each comment tag in *text*."""
    s = "" if text is None or (isinstance(text, float) and np.isnan(text)) else _rewrite(str(text))
    early = _early_clause(s)
    tags = {tag: int(bool(rx.search(early if tag in _STYLE_TAGS else s))) for tag, rx in _COMMENT_RES.items()}
    # "Close up in rear": a bunched field with the horse at the back is a hold-up ride.
    tags["atr_cmt_front"] &= 1 - tags["atr_cmt_held_up"]
    return tags


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


def _early_clauses(text: pa.Array) -> pa.Array:
    """:func:`_early_clause` for every comment, in Arrow (equal on ASCII text)."""
    clauses = pc.list_slice(pc.split_pattern(text, ",", max_splits=_EARLY_CLAUSES), 0, _EARLY_CLAUSES)
    flat = pc.list_flatten(clauses)
    parents = pc.list_parent_indices(clauses).to_numpy(zero_copy_only=False)
    hits = np.flatnonzero(pc.match_substring_regex(flat, _POSITION, ignore_case=True).to_numpy(zero_copy_only=False))
    first = hits[np.r_[True, parents[hits][1:] != parents[hits][:-1]]] if len(hits) else hits
    chosen = np.full(len(text), -1, dtype=np.int64)
    chosen[parents[first]] = first
    found = chosen >= 0
    if not found.any():
        return pa.array([""] * len(text), type=pa.string())
    picked = flat.take(pa.array(np.where(found, chosen, 0)))
    return pc.if_else(pa.array(found), picked, pa.scalar("", type=pa.string()))


def _comment_tags(comments: list[str]) -> dict[str, np.ndarray]:
    """0/1 tag columns. RE2 reads ASCII comments far faster than ``re``.

    The two engines agree on ASCII text. They can differ on word boundaries and
    case folding beyond it, so any other comment stays with ``re``.
    """
    text = pa.array(comments, type=pa.string())
    other = np.flatnonzero(~pc.string_is_ascii(text).to_numpy(zero_copy_only=False))
    for pattern, replacement in _REWRITES:
        text = pc.replace_substring_regex(text, pattern, replacement)
    early = _early_clauses(text)
    tags = {}
    for tag, pattern in _COMMENT_PATTERNS.items():
        flags = pc.match_substring_regex(early if tag in _STYLE_TAGS else text, pattern, ignore_case=True)
        tags[tag] = flags.to_numpy(zero_copy_only=False).astype(np.float64)
    tags["atr_cmt_front"] = tags["atr_cmt_front"] * (1 - tags["atr_cmt_held_up"])
    for row in other:
        for tag, flag in tag_comment(comments[row]).items():
            tags[tag][row] = float(flag)
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
# Prior-run history from the selected dataset (used by feature_engineer)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _published_history(results_version, atr_version, foreign_version):
    """Small scalar history, with every canonical start retained as a window slot.

    Saved per (dataset versions, this module's code) in ``src.fe_cache``: every process and live card reads it
    back instead of rescoring all published sectionals (~30 s).
    """
    from src import fe_cache
    from src.ingestion import atr_tabs
    key = [results_version, atr_version, foreign_version, fe_cache.source_hash(__file__),
           fe_cache.source_hash(atr_tabs.__file__)]
    history = fe_cache.frame("atr_history", key, lambda: _compute_published_history(foreign_version))
    history[["sl_race_id", "sl_horse_id"]] = history[["sl_race_id", "sl_horse_id"]].astype(object)
    return history


def _compute_published_history(foreign_version):
    from src.source_dataset import read_table
    canonical = read_table("sl", "results")
    raw = read_table("atr", "sectionals", linked=True)
    timing = read_table("atr", "observations_sectionals")
    raw["_available"] = pd.to_datetime(timing.available_at, utc=True).to_numpy()
    raw = raw.loc[raw.match_status.eq("matched")].copy()
    keys = ["sl_race_id", "sl_horse_id"]
    if raw.duplicated(keys).any():
        raise ValueError("Accepted ATR identities are not unique")
    canonical = canonical.copy()
    canonical["sl_race_id"] = canonical.race_id.astype("string").str.replace(r"\.0$", "", regex=True)
    canonical["sl_horse_id"] = canonical.horse_id.astype("string").str.replace(r"\.0$", "", regex=True)
    context = canonical.merge(raw.reindex(columns=[*keys, *ALL_ATR_COLS, "_available"]), on=keys, how="inner", validate="one_to_one")
    run = compute_atr_run_features(context)
    columns = list(dict.fromkeys([*(name for name, _ in RUN_METRICS), "atr_closing_in_hand", *COMMENT_TAGS]))
    scalars = pd.concat([context[keys + ["_available"]].reset_index(drop=True), run.reindex(columns=columns).reset_index(drop=True)], axis=1)
    history = canonical[["sl_race_id", "sl_horse_id", "race_date", "off_time"]].merge(scalars, on=keys, how="left", validate="one_to_one")
    history["_event"] = pd.to_datetime(history.race_date.astype(str).str[:10] + " " + history.off_time.map(_normalise_off_time_key), utc=True)
    if foreign_version is not None:
        from src.foreign_form import history as foreign_history
        foreign = foreign_history()
        if not foreign.empty:
            slots = pd.DataFrame({
                "sl_horse_id": foreign.horse_id.astype("string"),
                "sl_race_id": "foreign:" + foreign.provider + ":" + foreign.native_race_id,
                "_event": pd.to_datetime(foreign._event_dt, utc=True),
            })
            history = pd.concat([history, slots], ignore_index=True)
    history = history.sort_values(["sl_horse_id", "_event", "sl_race_id"], kind="stable").reset_index(drop=True)
    history["_position"] = np.arange(len(history))
    # Plain object IDs, like the target's, however the links were read back
    # (a dataset whose ATR rows all failed to link keeps pandas strings here).
    history[keys] = history[keys].astype(object)
    return history


def published_atr_rolling(frame):
    """Prior starts whose observations were available at the target decision.

    Windows retain untracked starts as missing slots. The full immutable scalar
    history is shared across chunks, so chunk boundaries cannot change support.
    """
    from src.source_dataset import require_manifest
    manifest = require_manifest()
    if "atr" not in manifest["sources"]:
        output = {}
        for metric, _ in RUN_METRICS:
            for suffix in ("avg_3", "avg_5", "best_3", "trend"):
                output[f"{metric}_{suffix}"] = np.full(len(frame), np.nan)
        for count in (3, 5):
            output[f"atr_closing_in_hand_avg_{count}"] = np.full(len(frame), np.nan)
            for tag in COMMENT_TAGS:
                if tag not in {"atr_cmt_eased", "atr_cmt_comfortable"}:
                    output[f"{tag}_rate_{count}"] = np.full(len(frame), np.nan)
        output["atr_observation_count_5"] = np.zeros(len(frame))
        output["atr_days_since_observation"] = np.full(len(frame), np.nan)
        return output
    from src.foreign_form import version as foreign_version
    history = _published_history(manifest["sources"]["sl"]["version"],
                                 manifest["sources"]["atr"]["version"], foreign_version())
    target = pd.DataFrame({"_row": np.arange(len(frame)),
        "sl_horse_id": frame.horse_id.astype("string").str.replace(r"\.0$", "", regex=True).to_numpy()})
    target["_event"] = pd.to_datetime(frame.race_date.astype(str).str[:10] + " " + frame.off_time.map(_normalise_off_time_key), utc=True).to_numpy()
    if "prediction_as_of" in frame:
        decision = pd.to_datetime(frame.prediction_as_of, utc=True, errors="raise").to_numpy()
        target["_event"] = target._event.where(target._event.le(decision), decision)
    linked = pd.merge_asof(target.sort_values("_event"), history[["_event", "sl_horse_id", "_position"]].sort_values("_event"),
                           on="_event", by="sl_horse_id", direction="backward", allow_exact_matches=False).sort_values("_row")
    positions = linked._position.fillna(-1).to_numpy(dtype="int64")[:, None] - np.arange(5)
    safe = np.clip(positions, 0, max(len(history)-1, 0))
    if history.empty:
        return {}
    horses = history.sl_horse_id.to_numpy()[safe]
    available = pd.to_datetime(history._available, utc=True).to_numpy(dtype="datetime64[ns]")[safe]
    decisions = pd.to_datetime(target._event, utc=True).to_numpy(dtype="datetime64[ns]")[:, None]
    valid = (positions >= 0) & (horses == target.sl_horse_id.to_numpy()[:, None]) & (available <= decisions)
    output = {}

    def window(column, count, operation="mean"):
        values = pd.to_numeric(history[column], errors="coerce").to_numpy(float)[safe[:, :count]]
        values = np.where(valid[:, :count], values, np.nan)
        support = np.isfinite(values).sum(axis=1)
        if operation == "mean":
            return np.divide(np.nansum(values, axis=1), support, out=np.full(len(frame), np.nan), where=support > 0)
        filled = np.where(np.isfinite(values), values, -np.inf if operation == "max" else np.inf)
        result = filled.max(axis=1) if operation == "max" else filled.min(axis=1)
        return np.where(support, result, np.nan)

    for metric, higher in RUN_METRICS:
        short = metric.removeprefix("atr_")
        for count in (3, 5):
            output[f"atr_{short}_avg_{count}"] = window(metric, count)
        output[f"atr_{short}_best_3"] = window(metric, 3, "max" if higher else "min")
        output[f"atr_{short}_trend"] = output[f"atr_{short}_avg_3"] - output[f"atr_{short}_avg_5"]
    for count in (3, 5):
        output[f"atr_closing_in_hand_avg_{count}"] = window("atr_closing_in_hand", count)
        for tag in COMMENT_TAGS:
            if tag not in {"atr_cmt_eased", "atr_cmt_comfortable"}:
                output[f"{tag}_rate_{count}"] = window(tag, count)
    support = valid & np.isfinite(pd.to_numeric(history.atr_finishing_speed_pct, errors="coerce").to_numpy(float)[safe])
    output["atr_observation_count_5"] = support.sum(axis=1).astype(float)
    events = history._event.to_numpy(dtype="datetime64[ns]")[safe]
    ages = np.where(support, (decisions-events) / np.timedelta64(1, "D"), np.inf).min(axis=1)
    output["atr_days_since_observation"] = np.where(np.isfinite(ages), ages, np.nan)
    return output
