"""Collect ATR's separate Jumps Data tab; independent of sectional rebuilding.

This stores observations, not prediction inputs. In particular the course par
and A--E fluency grade may be revised after a race. Their revision date must be
considered before historical use. Nothing here runs at import time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup

from src.ingestion.attheraces import _Fetcher, _runner_meta, _sectional_entries
from src.ingestion.storage import atomic_json

PARSER_VERSION = 1
LABELS = {
    "speed lost jumping early": "atr_jump_speed_lost_early_pct",
    "speed lost jumping middle": "atr_jump_speed_lost_middle_pct",
    "speed lost jumping late": "atr_jump_speed_lost_late_pct",
    "speed lost jumping average": "atr_jump_speed_lost_avg_pct",
    "jumping fluency grade": "atr_jump_fluency_grade",
    "speed lost jumping consistency": "atr_jump_consistency_pct",
}
GRADE_FIELD = "atr_jump_fluency_grade"


@dataclass
class JumpsObservation:
    state: str
    rows: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    reason: str | None = None


def _text(value: str) -> str:
    return " ".join(value.split())


def _percent(value: str) -> float | None:
    value = _text(value)
    if value in {"", "-", "–", "—", "?", "N/A"}:
        return None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%", value):
        raise ValueError(f"Unrecognised percentage: {value!r}")
    result = float(value.removesuffix("%").strip())
    if not math.isfinite(result):
        raise ValueError("Non-finite percentage")
    return result


def parse_jumps_data(
    html: str, race_id: str, *, expected_horse_ids: list[str] | None = None,
) -> JumpsObservation:
    """Parse labelled pills, preserving unknown metrics and partial coverage.

    Duplicate layout columns are ignored. Race IDs inside horse links must agree
    with the requested native ID; an unfamiliar page is never recorded as absent.
    Percentages stay in percentage units (13.6, not 0.136).
    """
    race_id = str(race_id)
    if not re.fullmatch(r"\d+", race_id):
        raise ValueError("ATR race ID must be numeric")
    soup = BeautifulSoup(html, "html.parser")
    context = {"atr_raceid": race_id, "atr_jumps_parser_version": PARSER_VERSION}
    heading = soup.select_one(".sectional-analysis__header")
    if heading is None or "jumps data" not in heading.get_text(" ", strip=True).lower():
        return JumpsObservation("unresolved", context=context, reason="unrecognised_jumps_page")
    context["atr_jumps_source_reference"] = (soup.select_one(".sectional-analysis") or {}).get("id")
    par = re.search(r"course winners\):\s*([\d.]+\s*%)", heading.get_text(" ", strip=True), re.I)
    if par:
        try:
            context["atr_jump_course_winner_speed_lost_pct"] = _percent(par[1])
        except ValueError:
            return JumpsObservation("unresolved", context=context, reason="invalid_course_par")
    revision = re.search(
        r"Par data was last updated on\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})",
        soup.get_text(" ", strip=True), re.I,
    )
    if revision:
        try:
            context["atr_jumps_par_updated_date"] = datetime.strptime(
                " ".join(revision.groups()), "%d %B %Y",
            ).date().isoformat()
        except ValueError:
            return JumpsObservation("unresolved", context=context, reason="invalid_par_revision_date")
    context["atr_jumps_help"] = [
        _text(x.get("title", "")) for x in soup.select(".tutorial__item[title]")
    ]
    rows = []
    seen = set()
    try:
        for entry in _sectional_entries(soup):
            pills = entry.select(".text-pill")
            if not pills:
                continue
            row = _runner_meta(entry)
            horse_id = row.get("atr_horse_id")
            if not horse_id or horse_id in seen:
                raise ValueError("Missing or duplicate native horse ID")
            link = entry.select_one("a.horse__link")
            linked_races = parse_qs(urlsplit(link.get("href", "")).query).get("raceid", [])
            if linked_races != [race_id]:
                raise ValueError("Source horse link does not identify the requested race")
            metrics = []
            for pill in pills:
                title = _text(pill.get("title", ""))
                display = _text(pill.get_text(" ", strip=True))
                label, separator, _ = title.partition(":")
                name = LABELS.get(label.lower()) if separator else None
                metrics.append({"label": label if separator else title,
                                "display": display, "title": title, "field": name})
                if name is None:
                    continue
                if name in row:
                    raise ValueError("Duplicate jumps measurement")
                row[name] = (None if display in {"", "-", "–", "—", "?"} else display) \
                    if name == GRADE_FIELD else _percent(display)
            if not set(LABELS.values()).issubset(row):
                raise ValueError("Missing labelled jumps columns")
            row["atr_jumps_source_metrics"] = metrics
            row["atr_jumps_quality_flags"] = []
            if row[GRADE_FIELD] is not None and row[GRADE_FIELD] not in {"A", "B", "C", "D", "E"}:
                row["atr_jumps_quality_flags"].append("unknown_fluency_grade")
            consistency = row["atr_jump_consistency_pct"]
            if consistency is not None and not 0 <= consistency <= 100:
                row["atr_jumps_quality_flags"].append("consistency_outside_0_100")
            seen.add(horse_id)
            rows.append(row)
        if not rows:
            raise ValueError("No labelled jumps runners")
        if expected_horse_ids is not None:
            expected = {str(x) for x in expected_horse_ids}
            if not seen.issubset(expected):
                raise ValueError("Source contains unexpected native horse IDs")
            context["atr_jumps_expected_runners"] = len(expected)
            context["atr_jumps_missing_horse_ids"] = sorted(expected - seen)
            context["atr_jumps_coverage"] = "complete" if seen == expected else "partial"
        else:
            context["atr_jumps_coverage"] = "unchecked"
        context["atr_jumps_provided_runners"] = len(rows)
    except ValueError as exc:
        return JumpsObservation("unresolved", context=context, reason=str(exc))
    return JumpsObservation("data", rows, context)


def fetch_jumps_data(
    race_id: str, *, racecard_url: str, expected_horse_ids: list[str] | None = None,
    fetcher=None,
) -> JumpsObservation:
    """Fetch one fragment; the shared client retains the exact raw response.

    This bounded collector does not launch another backfill or change its queue.
    HTTP errors and challenges stop this fetch; callers must honour retry timing.
    """
    race_id = str(race_id)
    if not re.fullmatch(r"\d+", race_id):
        raise ValueError("ATR race ID must be numeric")
    page = urlsplit(racecard_url)
    if (page.scheme != "https" or page.hostname not in {"www.attheraces.com", "m.attheraces.com"}
            or not re.fullmatch(r"/racecard/[^/]+/\d{2}-[A-Za-z]+-\d{4}/\d{4}", page.path)):
        raise ValueError("An exact HTTPS ATR racecard URL is required")
    client = fetcher if fetcher is not None else _Fetcher(3)
    url = f"https://www.attheraces.com/ajax/sectionals-razor/{race_id}/jumpsdata"
    response = client.get(url, referer=racecard_url)
    provenance = {
        "atr_raceid": race_id, "requested_racecard_url": racecard_url,
        "source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "request": dict(getattr(client, "last_response", {})),
    }
    if response is None:
        return JumpsObservation("blocked", context=provenance, reason="transport_error")
    status, html = response
    provenance["http_status"] = status
    provenance["sha256"] = provenance["request"].get("sha256") or hashlib.sha256(
        html.encode("utf-8")).hexdigest()
    if status == 404:
        return JumpsObservation("not_found", context=provenance, reason="http_404")
    if status != 200 or "Client Challenge" in html[:4000]:
        return JumpsObservation("blocked", context=provenance, reason="http_error_or_challenge")
    result = parse_jumps_data(html, race_id, expected_horse_ids=expected_horse_ids)
    result.context.update(provenance)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--racecard-url", required=True)
    parser.add_argument("--expected-horse-ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True,
                        help="Observation JSON; kept outside the accepted sectional cache")
    args = parser.parse_args(argv)
    result = fetch_jumps_data(args.race_id, racecard_url=args.racecard_url,
                              expected_horse_ids=args.expected_horse_ids)
    atomic_json(args.output, asdict(result))
    print(f"{result.state}: {len(result.rows)} runners; {args.output}")
    return 0 if result.state == "data" else 1


if __name__ == "__main__":
    raise SystemExit(main())
