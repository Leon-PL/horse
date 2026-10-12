"""Pure per-run feature derivations from At The Races sectionals.

These helpers turn one runner's *raw post-race* ATR sectional record (the
columns added by :func:`src.attheraces_scraper.merge_atr_sectionals`) into a
set of **per-run scalars and tags**.  They describe how the horse ran in
*that* race — they are NOT model features on their own.

``src.feature_engineer.add_atr_features`` consumes these, then rolls each one
over a horse's *prior* runs (leakage-safe ``shift(1)``) to build the actual
features.  Keeping the derivations here (pure, row-local, no history) makes
them unit-testable in isolation and keeps the leakage boundary crisp.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

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


# Per-run continuous metrics and whether higher is better (drives best/min,
# field-relative sign and the composite score).
RUN_METRICS: list[tuple[str, bool]] = [
    ("atr_finishing_speed_pct", True),   # closing strength (% of avg)
    ("atr_early_pace", True),            # first-segment speed → front-running
    ("atr_max_speed", True),             # top sectional speed
    ("atr_closing_speed", True),         # last-segment speed
    ("atr_turn_of_foot", True),          # biggest segment-to-segment acceleration
    ("atr_efficiency_score", True),      # TPD even-pace efficiency
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
    "atr_efficiency_score": (0.0, 40.0),
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
    energy_c = col("atr_energy_dist")
    optimum_c = col("atr_optimum_energy")
    fs_c = pd.to_numeric(df.get("atr_finishing_speed_pct", np.nan), errors="coerce").to_numpy()
    eff_c = pd.to_numeric(df.get("atr_efficiency_score", np.nan), errors="coerce").to_numpy()
    psl_c = pd.to_numeric(df.get("atr_peak_stride_length", np.nan), errors="coerce").to_numpy()
    psf_c = pd.to_numeric(df.get("atr_peak_stride_freq", np.nan), errors="coerce").to_numpy()
    pos_c = pd.to_numeric(df.get("atr_official_position", np.nan), errors="coerce").to_numpy()
    adj_c = pd.to_numeric(df.get("atr_adj_placing", np.nan), errors="coerce").to_numpy()
    won_c = pd.to_numeric(df.get("won", np.nan), errors="coerce").to_numpy()
    lb_c = pd.to_numeric(df.get("lengths_behind", np.nan), errors="coerce").to_numpy()

    # Comment tags — vectorised regex (much faster than per-row on 500k+ rows).
    cmt_series = df.get("atr_in_running_comment", pd.Series([""] * n, index=df.index)).fillna("").astype(str)
    tags = {
        t: cmt_series.str.contains(rx).to_numpy(dtype=np.float64)
        for t, rx in _COMMENT_RES.items()
    }

    early = np.full(n, np.nan)
    mx = np.full(n, np.nan)
    closing = np.full(n, np.nan)
    tof = np.full(n, np.nan)
    energy_eff = np.full(n, np.nan)
    upgrade = np.full(n, np.nan)
    in_hand = np.full(n, np.nan)

    eased_a = tags["atr_cmt_eased"]
    stayed_a = tags["atr_cmt_stayed_on"]
    comf_a = tags["atr_cmt_comfortable"]

    for i in range(n):
        sp = _as_list(speeds_c[i])
        if sp:
            early[i] = sp[0]
            mx[i] = max(sp)
            closing[i] = sp[-1]
            if len(sp) >= 2:
                tof[i] = max(0.0, max(b - a for a, b in zip(sp[:-1], sp[1:])))
        e, o = _as_list(energy_c[i]), _as_list(optimum_c[i])
        if e and o:
            m = min(len(e), len(o))
            energy_eff[i] = -float(np.mean([abs(e[j] - o[j]) for j in range(m)]))

        # Sectional upgrade (finished worse than sectionals imply), CLEANED:
        # discard when the horse was eased while beaten (its fast finishing
        # sectional is an artefact of not being asked), or tailed off.
        if not np.isnan(pos_c[i]) and not np.isnan(adj_c[i]):
            raw_up = pos_c[i] - adj_c[i]
            beaten = (won_c[i] == 0) if not np.isnan(won_c[i]) else True
            tailed_off = (not np.isnan(lb_c[i])) and lb_c[i] > 12 and beaten
            if (eased_a[i] == 1 and beaten) or tailed_off:
                upgrade[i] = 0.0   # no reliable signal → ran to form
            else:
                upgrade[i] = float(np.clip(raw_up, -6.0, 6.0))

        # Closing speed "in hand": credit a strong finishing % only when the
        # finish was genuine (won/ran-on) and not eased-when-beaten.
        if not np.isnan(fs_c[i]):
            genuine = (won_c[i] == 1) or stayed_a[i] == 1 or comf_a[i] == 1
            unreliable = (eased_a[i] == 1) and (won_c[i] == 0)
            if genuine and not unreliable:
                in_hand[i] = fs_c[i]

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
        out[t] = tags[t]

    res = pd.DataFrame(out, index=df.index)

    # Sanitise out-of-band continuous values.
    for c, (lo, hi) in _PLAUSIBLE.items():
        if c in res.columns:
            v = res[c]
            res.loc[v.notna() & ((v < lo) | (v > hi)), c] = np.nan
    return res
