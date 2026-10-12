"""Features derived from BHA + IHRB stewards reports.

Pipeline:  raw report text  ->  (horse, date, tag) event table  ->  strictly
past per-horse features joined onto the race frame.

**Leakage (the only thing that matters here).**  A stewards report about race
R is a *result of* R.  Every feature this module emits is a strictly-past
aggregate: for a race on date D it only counts events dated *before* D
(``merge_asof`` with ``allow_exact_matches=False``).  Never join a report to
the race it describes.  See CLAUDE.md hard rule #1.

The text parser is deliberately regex-based (fast, transparent, unit-tested).
Horse identity is resolved two ways:

* BHA report bodies write runners in BLOCK CAPS, optionally ``(IRE)`` -- these
  are detected directly.
* When a list of known runners is supplied (e.g. the runners of that
  course/date from ``race_results``) we match those names instead, which is
  what IHRB needs because its Title-Case names are not distinguishable from
  ordinary prose.

Build the event table once (offline), persist it, then call
``add_stewards_features`` inside the FeatureStore stage.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

import pandas as pd

from src.utils import canonical_track_key, normalise_horse_key, normalise_track_key


def _to_dt(series_or_scalar):
    """Parse dates tolerant of the mixed ``YYYY-MM-DD`` / ``... 00:00:00`` forms
    that appear in the saved report files (a plain ``to_datetime`` infers one
    format from the first row and NaTs the rest)."""
    return pd.to_datetime(series_or_scalar, format="mixed", errors="coerce")


# Minimum similarity for a misspelt report name to be snapped to an actual
# runner of the same fixture (recovers source typos like "Amalifi Bay").
_FUZZY_THRESHOLD = 0.86

# ── Tag taxonomy ──────────────────────────────────────────────────────
# Each tag maps to a compiled, case-insensitive pattern.  Patterns are matched
# against a *window* of text around a horse mention, so they stay short.
# Forward-looking tags (gear permissions) are flagged separately because they
# describe a *future* run and are therefore legitimately known pre-race.

_TAG_PATTERNS: dict[str, str] = {
    # "had an excuse" / trouble-in-running -> bounce-back angle (+CLV)
    "trouble_in_running": (
        r"hamper|interferen|short of room|no clear run|denied (?:a )?clear run|"
        r"badly away|slowly away|slowly into stride|missed the break|impeded|"
        r"squeezed|carried (?:wide|out)|became unbalanced|short of racing room"
    ),
    "lost_shoe": r"lost\s+(?:its|a|the|her|his)?[\w\s-]{0,18}?shoe",
    # soundness / veterinary (negative, often recurring)
    "lame": r"\blame\b|lost (?:its|her|his)? ?action|jarred up",
    "bled": r"\bbled\b|epistaxis|burst a blood vessel|broke a blood vessel|blood vessel",
    "breathing": r"gurgl|\bchok|made a (?:respiratory )?noise|respiratory|abnormal breathing|soft palate",
    "distressed": r"distress|collapsed|finished weak",
    # jumping reliability (jumps form)
    "jump_error": r"made a mistake|mistake at the|blunder|pecked on landing|hit the",
    "fell": r"\bfell\b",
    "unseated": r"unseated",
    "brought_down": r"brought down",
    "pulled_up": r"pulled up",
    "refused": r"refused to (?:race|jump|start)|reluctant to (?:race|start)|whipped round|ran out",
    # temperament / inexperience
    "hung": r"\bhung (?:left|right|badly|under|towards)|hanging (?:left|right)|edged (?:left|right)",
    "green": r"\bgreen\b|inexperienc|reluctant to go|ran in snatches",
    "unruly": r"unruly|reared|fractious|withdrawn (?:at the start|under orders)|refused to enter",
    # integrity
    "non_trier": (
        r"not (?:appear to be )?running on (?:its|his|her|their)? ?merits|"
        r"failed to run on (?:its|his|her|their)? ?merits|"
        r"not run on (?:its|his|her|their)? ?merits|"
        r"could (?:not|n't) be ridden out|ride.*?with greater vigour"
    ),
    # forward-looking declaration (legitimately known before the *next* run).
    # The "(?:also )?" + wider gap catch "Permission was also given for X to
    # wear..." and "...to be fitted with..." which the strict form missed.
    "gear_permission": (
        r"permission (?:was )?(?:also )?given .{0,60}?"
        r"(?:wear|use|carry|fitted with|fit a).{0,40}?"
        r"(?:hood|visor|cheekpiece|tongue[- ]?(?:tie|strap)|blinker|eye[- ]?shield)"
    ),
}

FORWARD_LOOKING_TAGS = frozenset({"gear_permission"})
ALL_TAGS = tuple(_TAG_PATTERNS)

_COMPILED = {tag: re.compile(pat, re.IGNORECASE) for tag, pat in _TAG_PATTERNS.items()}

# BLOCK-CAPS runner mention, e.g. "AVOID DE MASTER (IRE)" / "FUBAR".
_CAPS_HORSE = re.compile(r"\b([A-Z][A-Z'’.]+(?:\s+(?:[A-Z'’.]{2,}|DE|DU|DES|LA|LE|EL|VON))*)\b(?:\s*\([A-Z]{2,3}\))?")
_COUNTRY_SUFFIX = re.compile(r"\s*\([A-Z]{2,3}\)\s*$")

# Words that are CAPS but never horse names (race/clerk boilerplate).
_CAPS_STOPWORDS = frozenset({
    "RACE", "CLASS", "GBB", "IRE", "GB", "FR", "USA", "THE", "AND", "HANDICAP",
    "HURDLE", "CHASE", "STAKES", "MAIDEN", "NOVICES", "STEEPLE", "FILLIES",
    "VETERINARY", "OFFICER", "STEWARDS", "RACEDAY", "TV", "RACINGTV", "BHA",
})

# If a BLOCK-CAPS span contains any of these it is a race *title*, not a horse
# (e.g. "THE BRISCO HANDICAP STAKES"); discard the whole mention.
_RACE_TITLE_WORDS = frozenset({
    "HANDICAP", "STAKES", "HURDLE", "CHASE", "MAIDEN", "NOVICES", "NOVICE",
    "STEEPLE", "FILLIES", "MARES", "TROPHY", "CUP", "SERIES", "CONDITIONS",
    "CLAIMING", "SELLING", "NURSERY", "APPRENTICE", "AMATEUR", "QUALIFIER",
    "FINAL", "DIVISION", "PLATE", "BUMPER", "JUVENILE",
})

# Apostrophe variants that must collapse to a straight quote before the merge
# key is built, so "Ken's Boy" and "Ken’s Boy" resolve identically.
_APOS = str.maketrans({"’": "'", "‘": "'", "´": "'", "`": "'"})

# How many characters either side of a horse mention to scan for tag patterns.
_WINDOW = 140


def _clean_horse(raw: str) -> str:
    """Strip the country suffix and normalise to the canonical merge key."""
    cleaned = _COUNTRY_SUFFIX.sub("", raw).translate(_APOS).strip()
    return normalise_horse_key(cleaned)


def _caps_mentions(text: str) -> list[tuple[int, str]]:
    """Locate BLOCK-CAPS runner mentions -> list of (start_index, horse_key)."""
    out: list[tuple[int, str]] = []
    for m in _CAPS_HORSE.finditer(text):
        token = m.group(1).strip()
        words = token.split()
        # Skip pure boilerplate, race titles, and single short tokens.
        if all(w in _CAPS_STOPWORDS for w in words):
            continue
        if words[0] == "THE" or any(w in _RACE_TITLE_WORDS for w in words):
            continue
        if len(token) < 3:
            continue
        out.append((m.start(), _clean_horse(m.group(0))))
    return out


def _known_mentions(text: str, known: Iterable[str]) -> list[tuple[int, str]]:
    """Locate occurrences of supplied runner names -> (start_index, horse_key)."""
    out: list[tuple[int, str]] = []
    for name in known:
        if not name or len(str(name)) < 3:
            continue
        for m in re.finditer(re.escape(str(name)), text, re.IGNORECASE):
            out.append((m.start(), normalise_horse_key(str(name).translate(_APOS))))
    return out


def _resolve_mentions(text: str, known_runners: Iterable[str]) -> list[tuple[int, str]]:
    """Fixture-anchored identity resolution -> (index, horse_key).

    Every returned ``horse_key`` is one of ``known_runners`` (i.e. a horse that
    actually ran), so downstream joins are 100% by construction. Names are
    found by exact (case-insensitive) match first; any BLOCK-CAPS mention that
    is *not* an exact runner is fuzzily snapped to the closest runner of the
    same fixture, which recovers source misspellings.
    """
    known = [str(n) for n in known_runners if n and len(str(n)) >= 3]
    if not known:
        return []
    key_by_norm = {normalise_horse_key(n.translate(_APOS)): n for n in known}
    runner_keys = list(key_by_norm)

    mentions = _known_mentions(text, known)
    seen_spans = {(s, h) for s, h in mentions}

    # Fuzzy-recover caps mentions that did not match a runner exactly.
    for start, caps_key in _caps_mentions(text):
        if caps_key in key_by_norm:
            continue
        match = difflib.get_close_matches(caps_key, runner_keys, n=1, cutoff=_FUZZY_THRESHOLD)
        if match and (start, match[0]) not in seen_spans:
            mentions.append((start, match[0]))
            seen_spans.add((start, match[0]))
    return mentions


def tag_report(text: str, known_runners: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """Return distinct ``(horse_key, tag)`` pairs found in one report.

    A tag is attributed to the nearest preceding horse mention within
    ``_WINDOW`` chars (falling back to the nearest following mention). When
    ``known_runners`` is supplied, identity is resolved against that fixture's
    actual runners (exact + fuzzy), so emitted horses always exist in results.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    mentions = _resolve_mentions(text, known_runners) if known_runners else _caps_mentions(text)
    if not mentions:
        return []
    mentions.sort()
    starts = [s for s, _ in mentions]

    found: set[tuple[str, str]] = set()
    for tag, rx in _COMPILED.items():
        for tm in rx.finditer(text):
            pos = tm.start()
            # nearest mention starting at or before the tag, within the window
            cand = [(s, h) for s, h in mentions if s <= pos and pos - s <= _WINDOW]
            if cand:
                _, horse = cand[-1]
            else:
                # else nearest following mention within the window
                fwd = [(s, h) for s, h in mentions if s > pos and s - pos <= _WINDOW]
                if not fwd:
                    continue
                _, horse = fwd[0]
            found.add((horse, tag))
    return sorted(found)


# ── Event table ───────────────────────────────────────────────────────

def build_runners_by_fixture(
    results: pd.DataFrame, *, track_col: str = "track",
    date_col: str = "race_date", horse_col: str = "horse_name",
) -> dict[tuple[str, str], list[str]]:
    """``{(canonical_track_key, iso_date): [runner names]}`` from a results frame."""
    r = results[[track_col, date_col, horse_col]].dropna(subset=[horse_col]).copy()
    r["_tk"] = r[track_col].map(canonical_track_key)
    r["_d"] = _to_dt(r[date_col]).dt.date.astype(str)
    return {k: g[horse_col].tolist() for k, g in r.groupby(["_tk", "_d"])}


def build_event_table(
    bha: pd.DataFrame | None = None,
    ihrb: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    runners_by_fixture: dict[tuple[str, str], list[str]] | None = None,
) -> pd.DataFrame:
    """Parse report frames into a long event table.

    Columns: ``date, track_key, horse_key, tag, source, forward_looking,
    fixture_matched``. Supply ``results`` (or a prebuilt ``runners_by_fixture``)
    to enable fixture-anchored identity resolution: every horse then exists in
    results (``fixture_matched=True``). Reports whose fixture is absent from
    results fall back to BLOCK-CAPS detection (``fixture_matched=False``) and
    are flagged so the unjoinable residue can be measured, never silently mixed.
    """
    if runners_by_fixture is None and results is not None:
        runners_by_fixture = build_runners_by_fixture(results)

    rows: list[dict] = []

    def _emit(date, track, text, source):
        date = _to_dt(date)
        if pd.isna(date):
            return
        ckey = canonical_track_key(track) if track else ""
        known = None
        if runners_by_fixture is not None:
            known = runners_by_fixture.get((ckey, str(date.date())))
        matched = known is not None
        for horse, tag in tag_report(text, known):
            rows.append({
                "date": date.normalize(),
                "track_key": ckey,
                "horse_key": horse,
                "tag": tag,
                "source": source,
                "forward_looking": tag in FORWARD_LOOKING_TAGS,
                "fixture_matched": matched,
            })

    if bha is not None and not bha.empty:
        for _, r in bha.iterrows():
            _emit(r.get("date"), r.get("course"), r.get("enquiry_text"), "bha")

    if ihrb is not None and not ihrb.empty:
        for _, r in ihrb.iterrows():
            _emit(r.get("report_date"), _track_from_ihrb_title(r.get("title")),
                  r.get("content_text"), "ihrb")

    cols = ["date", "track_key", "horse_key", "tag", "source",
            "forward_looking", "fixture_matched"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).drop_duplicates().reset_index(drop=True)


_IHRB_TITLE_TRACK = re.compile(r"^([A-Za-z .'-]+?)\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}")
# Title keywords that mark a non-raceday item (referral / appeal / rule /
# regulatory notice) rather than a course's daily enquiries.
_IHRB_NON_RACEDAY = re.compile(
    r"referral|appeal|regulatory|notice|rule change|prohibited|substance|"
    r"\(rider\)|\(trainer\)|\(owner\)|committee|disqualif|inquiry into",
    re.IGNORECASE,
)


def _track_from_ihrb_title(title) -> str:
    """'Fairyhouse 1st January 2025' -> 'Fairyhouse'; non-raceday titles -> ''."""
    if not isinstance(title, str):
        return ""
    if _IHRB_NON_RACEDAY.search(title):
        return ""
    m = _IHRB_TITLE_TRACK.match(title.strip())
    return m.group(1).strip() if m else ""


# ── Features (strictly past) ──────────────────────────────────────────

def add_stewards_features(
    races: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horse_col: str = "horse_name",
    date_col: str = "race_date",
    tags: Iterable[str] = ALL_TAGS,
) -> pd.DataFrame:
    """Attach per-horse, strictly-past stewards features to ``races``.

    For each tag emits ``stew_<tag>_career`` = number of prior occurrences for
    that horse *before* the race date.  Backward-looking tags use
    ``allow_exact_matches=False`` so a same-day report can never leak.  Forward-
    looking tags (gear permission) also use the most recent prior occurrence —
    the declaration precedes the next run, which is what we want to know.
    """
    out = races.copy()
    out["_hk"] = out[horse_col].astype(str).str.translate(_APOS).map(normalise_horse_key)
    out["_d"] = _to_dt(out[date_col]).dt.normalize()
    out["_orig"] = range(len(out))
    # merge_asof requires the left frame globally sorted by the `on` column.
    left = out.sort_values("_d")

    tags = list(tags)
    feat_cols = [f"stew_{t}_career" for t in tags]
    for col in feat_cols:
        out[col] = 0

    ev_all = events.dropna(subset=["horse_key", "date"]).copy()
    ev_all["date"] = _to_dt(ev_all["date"]).dt.normalize()

    for tag, col in zip(tags, feat_cols):
        ev = ev_all[ev_all["tag"] == tag]
        if ev.empty:
            continue
        # cumulative count of this tag per horse over time
        ev = (
            ev.groupby(["horse_key", "date"]).size().rename("n").reset_index()
            .sort_values("date")
        )
        ev["cum"] = ev.groupby("horse_key")["n"].cumsum()
        # One grouped as-of join: for each race, the running count of this tag
        # strictly before the race date (allow_exact_matches=False = no leak).
        m = pd.merge_asof(
            left[["_orig", "_hk", "_d"]], ev[["horse_key", "date", "cum"]],
            left_on="_d", right_on="date", left_by="_hk", right_by="horse_key",
            allow_exact_matches=False, direction="backward",
        )
        vals = pd.Series(m["cum"].fillna(0).to_numpy(), index=m["_orig"].to_numpy())
        out[col] = vals.reindex(out["_orig"]).to_numpy()

    out[feat_cols] = out[feat_cols].fillna(0).astype("int32")
    # Composite "had an excuse, ever" convenience flag (career>0 of the
    # bounce-back tags). The recency-aware version lives in the rolling features.
    excuse = [f"stew_{t}_career" for t in ("trouble_in_running", "lost_shoe") if f"stew_{t}_career" in out]
    if excuse:
        out["stew_excuse_career"] = out[excuse].sum(axis=1).astype("int32")
    return out.drop(columns=["_hk", "_d", "_orig"])


def add_stewards_rolling_features(
    races: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horse_col: str = "horse_name",
    date_col: str = "race_date",
    windows: Iterable[int] = (3, 5),
    tags: Iterable[str] = ALL_TAGS,
) -> pd.DataFrame:
    """Strictly-past rolling stewards features over a horse's last *N* runs.

    Recency matters far more than career totals (a horse that bled two runs ago
    is a very different proposition from one that bled forty runs ago). For each
    tag and window N emits ``stew_<tag>_last<N>`` = times that tag occurred in
    the horse's *previous* N runs.

    Leakage-safe by the CLAUDE.md rule #1 pattern: each (horse, run) is marked
    with the tags from *that* run, then per horse we take
    ``rolling(N).sum().shift(1)`` — the ``shift(1)`` drops the current run, so a
    same-race report can never enter its own feature.
    """
    tags = list(tags)
    windows = list(windows)
    out = races.copy()
    out["_hk"] = out[horse_col].astype(str).str.translate(_APOS).map(normalise_horse_key)
    out["_d"] = _to_dt(out[date_col]).dt.normalize()

    # Per-(horse, run-date) indicator for each tag (1 if that tag was recorded
    # for the horse in that race). One run per horse per day is assumed.
    ev = events.dropna(subset=["horse_key", "date"]).copy()
    new_cols: list[str] = []
    if not ev.empty:
        ev["date"] = _to_dt(ev["date"]).dt.normalize()
        ev = ev[ev["tag"].isin(tags)]
        ind = (
            ev.assign(v=1)
            .pivot_table(index=["horse_key", "date"], columns="tag", values="v",
                         aggfunc="max", fill_value=0)
            .reset_index()
        )
        ind.columns.name = None
        present_tags = [t for t in tags if t in ind.columns]
        out = out.merge(
            ind.rename(columns={"horse_key": "_hk", "date": "_d"}),
            on=["_hk", "_d"], how="left",
        )
        for t in present_tags:
            out[t] = out[t].fillna(0)
    else:
        present_tags = []

    # Rolling within each horse, excluding the current run. Done in one grouped
    # rolling per window over ALL tag columns at once (vectorised C rolling — a
    # per-tag transform(lambda) is ~30x slower on the full results frame).
    if present_tags:
        gs = out.sort_values(["_hk", "_d"])
        grp = gs.groupby("_hk", sort=False)
        for n in windows:
            rolled = (
                grp[present_tags].rolling(n, min_periods=1).sum()
                .reset_index(level=0, drop=True)
            )
            # shift(1) within each horse drops the current run (no leakage).
            rolled = rolled.groupby(gs["_hk"], sort=False).shift(1)
            for t in present_tags:
                col = f"stew_{t}_last{n}"
                new_cols.append(col)
                out.loc[rolled.index, col] = rolled[t]
    # Tags absent from the data still get zero columns for a stable schema.
    for t in tags:
        for n in windows:
            col = f"stew_{t}_last{n}"
            if col not in out.columns:
                out[col] = 0.0
                new_cols.append(col)

    if new_cols:
        out[new_cols] = out[new_cols].fillna(0).astype("int32")
    # Recency-aware "had an excuse in the last N runs" composite.
    for n in windows:
        parts = [f"stew_{t}_last{n}" for t in ("trouble_in_running", "lost_shoe")
                 if f"stew_{t}_last{n}" in out]
        if parts:
            out[f"stew_excuse_last{n}"] = out[parts].sum(axis=1).astype("int32")
    drop = ["_hk", "_d"] + [t for t in present_tags if t in out.columns]
    return out.drop(columns=drop)


# ── Persisted artifact + pipeline integration ─────────────────────────

DEFAULT_EVENTS_PATH = "data/processed/stewards_events.parquet"


def build_stewards_event_artifact(
    *,
    bha_path: str = "data/raw/bha_stewards_reports.csv",
    ihrb_path: str = "data/raw/ihrb_daily_enquiries.csv",
    results_path: str = "data/raw/race_results.csv",
    out_path: str = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    """Parse the raw report files into the persisted event table (run offline).

    Keeps the expensive text parsing out of the hot FE path: build this once
    (and on each new scrape), then :func:`enrich_with_stewards` just does cheap
    as-of / rolling joins against it.
    """
    from pathlib import Path

    bha = pd.read_csv(bha_path) if Path(bha_path).exists() else None
    ihrb = pd.read_csv(ihrb_path) if Path(ihrb_path).exists() else None
    results = pd.read_csv(
        results_path, usecols=["race_date", "track", "horse_name"], low_memory=False
    ) if Path(results_path).exists() else None

    events = build_event_table(bha=bha, ihrb=ihrb, results=results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out_path, index=False)
    return events


def enrich_with_stewards(
    df: pd.DataFrame,
    *,
    events_path: str = DEFAULT_EVENTS_PATH,
    windows: Iterable[int] = (3, 5),
) -> pd.DataFrame:
    """Attach career + rolling stewards features from the persisted artifact.

    No-op (returns ``df`` unchanged) when the artifact is missing, so it is safe
    to drop into the FE pipeline before the first scrape/build. This is the
    single entry point the walk-forward harness should call to evaluate the
    stewards features before they are promoted to the model allowlist.
    """
    from pathlib import Path

    if not Path(events_path).exists():
        return df
    events = pd.read_parquet(events_path)
    if events.empty:
        return df
    out = add_stewards_features(df, events)
    out = add_stewards_rolling_features(out, events, windows=windows)
    return out


def add_stewards_features_stage(df: pd.DataFrame) -> pd.DataFrame:
    """``engineer_features``-compatible stage wrapper (``df -> df``).

    Drop into the ``stages`` list in feature_engineer.py to activate, once the
    features have been validated under the locked walk-forward. Artifact-gated,
    so it is inert until ``build_stewards_event_artifact`` has been run.
    """
    return enrich_with_stewards(df)


def main() -> None:
    import argparse
    import json
    import logging

    p = argparse.ArgumentParser(description="Build the stewards event-table artifact")
    p.add_argument("--bha", default="data/raw/bha_stewards_reports.csv")
    p.add_argument("--ihrb", default="data/raw/ihrb_daily_enquiries.csv")
    p.add_argument("--results", default="data/raw/race_results.csv")
    p.add_argument("--out", default=DEFAULT_EVENTS_PATH)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    ev = build_stewards_event_artifact(
        bha_path=args.bha, ihrb_path=args.ihrb,
        results_path=args.results, out_path=args.out,
    )
    matched = float(ev["fixture_matched"].mean()) if not ev.empty else 0.0
    print(json.dumps({
        "events": int(len(ev)),
        "fixture_matched_pct": round(100 * matched, 1),
        "by_source": ev["source"].value_counts().to_dict() if not ev.empty else {},
        "out": args.out,
    }))


if __name__ == "__main__":
    main()
