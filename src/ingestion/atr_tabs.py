"""The ATR source contract and extra tab parsers, without features.

Collection and feature building both read this module: the tabs, the accepted
columns, the cache location and the cache reader. It imports no feature code.
"""
from datetime import datetime
import math
import os
import re
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup
import pandas as pd

import config

COLLECTION_VERSION = 3
PERFORMANCE_TABS = ("times", "tools", "stride", "jumpsdata", "distanceran")
TERMINAL_TAB_STATES = {"data", "empty", "not_found"}
PRE_RACE_TABS = {"card-core", "card-quick-form", "card-pinsticker", "card-pace",
                 "card-formscan", "card-tourquotes", "card-printouts"}
EXTRA_METRICS = [
    "atr_jump_speed_lost_early_pct", "atr_jump_speed_lost_middle_pct",
    "atr_jump_speed_lost_late_pct", "atr_jump_speed_lost_avg_pct",
    "atr_jump_consistency_pct", "atr_jump_course_winner_speed_lost_pct",
    "atr_distance_ran_yards", "atr_distance_diff_vs_median_lengths",
    "atr_race_median_distance_ran_yards",
]
EXTRA_TEXT = ["atr_jump_fluency_grade", "atr_jumps_par_updated_date",
              "atr_distance_par_updated_date", "atr_distance_ran_display",
              "atr_collection_version", "atr_jumps_source_reference", "atr_distance_source_status"]
EXTRA_ARRAYS = ["atr_jumps_source_metrics", "atr_jumps_quality_flags", "atr_jumps_help",
                "atr_tab_manifest"]

ATR_CACHE_DIR = os.path.join(config.DATA_DIR, "atr_cache")
ATR_CACHE_FILE = os.path.join(ATR_CACHE_DIR, "atr_sectionals.parquet")

# Scalar numeric feature columns (NaN when absent) exposed by the merge.
ATR_METRIC_COLS = [
    "atr_final_time",            # total race time for the runner (seconds)
    "atr_finishing_speed_pct",   # finishing speed as % of average (TPD)
    "atr_efficiency_score",      # TPD efficiency score
    "atr_peak_stride_length",    # peak stride length (TPD units)
    "atr_peak_stride_freq",      # peak stride frequency (strides/sec)
    "atr_n_sections",            # number of furlong splits captured
    "atr_adj_placing",           # sectional-adjusted finishing placing
    "atr_top_speed_mph",
    "atr_final_speed_mph",       # instantaneous finish speed, not final section
    "atr_acceleration_0_30_seconds",
    "atr_race_finishing_speed_pct",
    "atr_first_section_furlongs",
    "atr_field_expected_runners",
    "atr_field_provided_runners",
]

# Text / identifier columns carried through the merge.
ATR_TEXT_COLS = [
    "atr_raceid",
    "atr_horse_id",
    "atr_saddle_cloth",
    "atr_draw",
    "atr_official_position",
    "atr_efficiency_grade",
    "atr_in_running_comment",
    "atr_race_early_pace",
    "atr_race_early_pace_flag",
    "atr_race_finishing_pace_flag",
    "atr_parser_version",
    "atr_source_race_date",
    "atr_source_off_time_local",
    "atr_field_coverage",
]

# Variable-length per-furlong series, stored as JSON strings so the cache
# stays a flat parquet table. Source labels/units and unknown chart series
# are retained as well as the standard numeric arrays.
ATR_ARRAY_COLS = [
    "atr_section_markers",       # ["5f","4f",...,"Fin."]
    "atr_split_times",           # per-furlong split time (s)
    "atr_running_totals",        # cumulative time at each marker (s)
    "atr_pace_flags",            # per-split pace class (even / fast / slow ...)
    "atr_sectional_speeds",      # segment speeds (mph)
    "atr_sectional_speed_labels",
    "atr_energy_dist",           # energy distribution % per marker
    "atr_optimum_energy",        # optimum energy distribution % per marker
    "atr_stride_length",         # stride length per marker
    "atr_stride_frequency",      # strides/second per marker
    "atr_stride_markers",
    "atr_stride_frequency_markers",
    "atr_stride_length_reference",
    "atr_stride_frequency_reference",
    "atr_front_flags",
    "atr_section_notes",
    "atr_race_pace_profile",      # published colour stops; no inferred speeds
    "atr_source_metrics",        # labelled values with units, including extras
    "atr_source_charts",         # full labelled stride chart inputs
    "atr_source_payloads",       # fragment URL/status/hash provenance
    "atr_field_missing_runners", # canonical runners absent from source tables
]

ATR_METRIC_COLS += EXTRA_METRICS
ATR_TEXT_COLS += EXTRA_TEXT
ATR_ARRAY_COLS += EXTRA_ARRAYS
ALL_ATR_COLS = ATR_METRIC_COLS + ATR_TEXT_COLS + ATR_ARRAY_COLS


def load_atr_cache() -> pd.DataFrame:
    """Load cached ATR sectionals (or return empty frame)."""
    if os.path.exists(ATR_CACHE_FILE):
        return pd.read_parquet(ATR_CACHE_FILE)
    return pd.DataFrame()


def current_collection_rows(cache: pd.DataFrame) -> pd.DataFrame:
    """A previously cached runner is not proof that newer tabs were collected."""
    if 'atr_collection_version' not in cache:
        return cache.iloc[:0]
    return cache.loc[cache.atr_collection_version.astype(str).eq(str(COLLECTION_VERSION))]


def inventory(html):
    """Classify every link in the actual race's tab strip; flag new ones."""
    soup = BeautifulSoup(html, "html.parser")
    anchor = soup.select_one('a[data-id="tab-full-result"], a[data-id="card-core"]')
    if anchor is None:
        return {"state": "unresolved", "tabs": [], "unknown_tabs": [], "reason": "No race tab strip"}
    strip = anchor.find_parent("ul")
    tabs = []
    for link in strip.select("a[href]") if strip else []:
        href = link.get("href", "")
        path = urlsplit(href).path
        tab_id = link.get("data-id")
        kind = None
        match = re.fullmatch(r"/ajax/sectionals-razor/(\d+)/([^/]+)", path)
        if match and match[2] in PERFORMANCE_TABS:
            policy, kind = "collect", match[2]
        elif tab_id == "tab-full-result":
            policy = "canonical_results"
        elif tab_id == "tab-future-form" and re.fullmatch(r"/ajax/future-form-content/\d+", path):
            policy = "exclude_future_outcomes"
        elif tab_id in PRE_RACE_TABS:
            policy = "pre_race_snapshot_only"
        else:
            policy = "unknown"
        tabs.append({"label": link.get_text(" ", strip=True), "href": href,
                     "id": tab_id, "kind": kind, "policy": policy})
    unknown = [row for row in tabs if row["policy"] == "unknown"]
    return {"state": "needs_review" if unknown else "verified", "tabs": tabs, "unknown_tabs": unknown}


def distance_yards(display):
    """Parse the source's rounded m/f/y distance; never infer metres/lengths."""
    value = re.sub(r"\s+", "", display)
    if value in {"", "-", "—", "–"}:
        return None
    match = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)f)?(?:(\d+(?:\.\d+)?)y)?", value)
    if not match or not any(match.groups()):
        raise ValueError(f"Unknown distance units: {display!r}")
    return 1760 * float(match[1] or 0) + 220 * float(match[2] or 0) + float(match[3] or 0)


def signal_runner_count(kind, records):
    series = {'times':('atr_split_times',), 'tools':('atr_sectional_speeds','atr_energy_dist'),
              'stride':('atr_stride_length','atr_stride_frequency')}.get(kind,())
    scalars = {'tools':('atr_top_speed_mph','atr_final_speed_mph'),
               'jumpsdata':('atr_jump_speed_lost_avg_pct',),
               'distanceran':('atr_distance_ran_yards',)}.get(kind,())
    return sum(any(isinstance(value,(int,float)) and math.isfinite(value)
                   for value in [*(row.get(k) for k in scalars),
                                 *(v for k in series for v in row.get(k,[]) or [])])
               for row in records.values())


def parse_distance(html, race_id):
    from src.ingestion.attheraces import _runner_meta, _sectional_entries
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    if "Distance Ran compared to Race Median Average" not in text:
        raise ValueError("Unrecognised distance tab")
    median = re.search(r"Median Distance Ran\s+((?:\d+(?:\.\d+)?[mfy]\s*)+)", text)
    context = {"atr_race_median_distance_ran_yards": distance_yards(median[1]) if median else None,
               "atr_distance_source_status":"beta" if 'Beta Data' in text else 'unspecified'}
    revision = re.search(r"Par data was last updated on\s+(\d+)(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", text)
    if revision:
        context["atr_distance_par_updated_date"] = datetime.strptime(
            " ".join(revision.groups()), "%d %B %Y").date().isoformat()
    rows, seen = [], set()
    for entry in _sectional_entries(soup):
        row = _runner_meta(entry)
        node = entry.select_one(".card-cell--distancerandistran")
        diff = entry.select_one(".card-cell--distanceranvmedian")
        if node is None or diff is None:
            # Some untracked runners have empty split placeholders instead of
            # distance cells, even when another runner has a measured distance.
            identity_classes = {'card-cell--pos-no','card-cell--silk','card-cell--horse-basic'}
            cells = entry.select('.card-cell')
            identity = [cell for cell in cells if identity_classes.intersection(cell.get('class', []))]
            extra = [cell for cell in cells if not identity_classes.intersection(cell.get('class', []))]
            empty_splits = extra and all('card-cell--sectional-time' in cell.get('class', [])
                                         and cell.get_text(strip=True) in {'', '-', '—', '–'} for cell in extra)
            empty_row = len(identity) == 3 and (empty_splits or (not extra and median is None))
            if not (node is None and diff is None and empty_row):
                raise ValueError("Missing labelled distance cells")
        link = entry.select_one("a.horse__link")
        linked = parse_qs(urlsplit(link.get("href", "") if link else "").query).get("raceid", [])
        if linked != [str(race_id)] or not row.get("atr_horse_id") or row["atr_horse_id"] in seen:
            raise ValueError("Distance runner/race identity mismatch")
        seen.add(row["atr_horse_id"])
        display = node.get_text(" ", strip=True) if node else ''
        yards = distance_yards(display)
        difference = diff.get_text(" ", strip=True) if diff else ''
        if difference == "No Change":
            delta = 0.0 if yards is not None else None
        elif difference in {"", "-", "—", "–"}:
            delta = None
        elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s+lengths?", difference):
            delta = float(difference.split()[0])
        else:
            raise ValueError("Unknown distance difference units")
        row.update(context, atr_distance_ran_display=display, atr_distance_ran_yards=yards,
                   atr_distance_diff_vs_median_lengths=delta)
        rows.append(row)
    return rows, context


def parse_extra(kind, html, race_id):
    if kind == "distanceran":
        rows, context = parse_distance(html, race_id)
    elif kind == "jumpsdata":
        from src.ingestion.atr_jumps import parse_jumps_data
        result = parse_jumps_data(html, race_id)
        if result.state == "unresolved" and result.reason == "No labelled jumps runners":
            rows, context = [], result.context
        elif result.state != "data":
            raise ValueError(result.reason or "Unrecognised jumps data")
        else:
            context = {key: value for key, value in result.context.items()
                       if key in EXTRA_METRICS + EXTRA_TEXT + EXTRA_ARRAYS}
            rows = [{**row, **context} for row in result.rows]
    else:
        raise ValueError(f"Unknown extra tab {kind}")
    return {row["horse_name"]: row for row in rows}, context if rows else {}
