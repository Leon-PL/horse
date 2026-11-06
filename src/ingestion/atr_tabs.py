"""ATR tab inventory and additional raw performance parsers, without features."""
from datetime import datetime
import math
import re
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup

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
    from src.attheraces_scraper import _runner_meta, _sectional_entries
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
            # ATR's unsupported-race template repeats identities without any
            # measurement cells or median. Accept only that exact empty shape.
            identity_classes = {'card-cell--pos-no','card-cell--silk','card-cell--horse-basic'}
            cells = entry.select('.card-cell')
            identity_only = len(cells) == 3 and all(identity_classes.intersection(cell.get('class',[]))
                                                   for cell in cells)
            if not (node is None and diff is None and median is None and identity_only):
                raise ValueError("Missing labelled distance cells")
        link = entry.select_one("a.horse__link")
        linked = parse_qs(urlsplit(link.get("href", "") if link else "").query).get("raceid", [])
        if linked != [str(race_id)] or not row.get("atr_horse_id") or row["atr_horse_id"] in seen:
            raise ValueError("Distance runner/race identity mismatch")
        seen.add(row["atr_horse_id"])
        display = node.get_text(" ", strip=True) if node else ''
        difference = diff.get_text(" ", strip=True) if diff else ''
        if difference == "No Change":
            delta = 0.0
        elif difference in {"", "-", "—", "–"}:
            delta = None
        elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s+lengths?", difference):
            delta = float(difference.split()[0])
        else:
            raise ValueError("Unknown distance difference units")
        row.update(context, atr_distance_ran_display=display, atr_distance_ran_yards=distance_yards(display),
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
