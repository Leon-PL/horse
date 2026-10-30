"""Fixed baseline-pilot readout readiness, without outcome-based stopping."""
from __future__ import annotations

from datetime import datetime, time, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from src.forward_store import ROOT, publish_json, timestamp, utcnow
from src.deployment_validation import file_sha256

LONDON = ZoneInfo("Europe/London")


def register_plan(study, path, *, now=None):
    study.verify()
    created = timestamp(now or utcnow())
    start = max(study.protocol["first_eligible_date"],
                (created.astimezone(LONDON).date()+timedelta(days=1)).isoformat())
    value = {"schema_version": 1, "created_at": created.isoformat(),
        "study_protocol_sha256": study.protocol_sha256, "study_path": str(study.path),
        "model_id": study.protocol["model_id"], "model_sha256": study.protocol["model_sha256"],
        "eligible_start_date": start, "minimum_calendar_days": 84,
        "minimum_published_pairs": 1000, "minimum_race_dates": 60, "settlement_grace_days": 7,
        "formal_confirmation": False,
        "purpose": "Baseline coverage, timing, score-gap and variance pilot; not powered market-parity confirmation",
        "comparisons": ["Pinned independent model", "Same-time market M0 identity reference"],
        "stopping_rule": "Close at end of first full UK calendar day satisfying elapsed days, published eligible pair count and distinct race dates; include all eligible pairs through that day; wait seven full calendar days for results",
        "outcome_rule": "Readout timing and inclusion never use wins, losses, score differences or settlement availability; preserve pending/changed fields and frozen forecasts",
        "interim_rule": "Operational/descriptive reports may be viewed; no early parity declaration, sample-size shortening, candidate tuning or promotion on pilot results",
        "next_stage": "Pilot data become development evidence; any formal confirmation requires a separately frozen candidate, sample-size assumptions and fresh start",
        "source_sha256": {name: file_sha256(ROOT/name) for name in (
            "src/forward_readout.py", "scripts/check_forward_readout.py", "src/forward_store.py")}}
    if study.protocol.get("record_place_probabilities",False):
        value["comparisons"].append("Pinned place refinement versus retained legacy place calculation, using the issued top-k definition")
    path = Path(path)
    publish_json(path, value)
    publish_json(path.with_suffix(".sha256.json"), {"sha256": file_sha256(path)})
    return value


def load_plan(path):
    path = Path(path)
    anchor = json.loads(path.with_suffix(".sha256.json").read_text())
    if file_sha256(path) != anchor["sha256"]:
        raise ValueError("Pilot readout plan changed after registration")
    plan = json.loads(path.read_text())
    if any(file_sha256(ROOT/name) != expected for name, expected in plan.get("source_sha256", {}).items()):
        raise ValueError("Registered pilot implementation changed; use its frozen reader")
    return plan


def readout_readiness(study, plan, *, now=None, records=None):
    moment = timestamp(now or utcnow())
    if (plan["study_protocol_sha256"] != study.protocol_sha256
            or plan["model_id"] != study.protocol["model_id"]
            or plan["model_sha256"] != study.protocol["model_sha256"]
            or plan["formal_confirmation"] is not False):
        raise ValueError("Readout plan and baseline study do not match")
    for name in ("minimum_calendar_days", "minimum_published_pairs", "minimum_race_dates", "settlement_grace_days"):
        if type(plan[name]) is not int or plan[name] < 1:
            raise ValueError("Pilot readout thresholds must be positive integers")
    if timestamp(plan["created_at"]) > moment:
        raise ValueError("Readout plan is from the future")
    start = datetime.fromisoformat(plan["eligible_start_date"]).date()
    first_full_day = timestamp(plan["created_at"]).astimezone(LONDON).date()+timedelta(days=1)
    if start < max(first_full_day, datetime.fromisoformat(study.protocol["first_eligible_date"]).date()):
        raise ValueError("Pilot must start on a fresh full day after registration")
    earliest_end = start+timedelta(days=plan["minimum_calendar_days"]-1)
    observed = []
    seen = set()
    for record in study.selected_records() if records is None else records:
        if record["event_id"] in seen:
            raise ValueError("Duplicate event in selected pilot pairs")
        seen.add(record["event_id"])
        if not record["benchmark_eligible"] or not record["paired_available"]:
            raise ValueError("Selected pilot record is not a complete eligible pair")
        day = timestamp(record["first_start"]).astimezone(LONDON).date()
        if day >= start and timestamp(record["published_at"]) <= moment:
            observed.append((day, record["event_id"], record["capture_id"]))
    observed.sort()
    counts = {}
    for day, _, _ in observed:
        counts[day] = counts.get(day, 0)+1
    cumulative = 0
    qualification_date = None
    for dates, day in enumerate(sorted(counts), 1):
        cumulative += counts[day]
        if cumulative >= plan["minimum_published_pairs"] and dates >= plan["minimum_race_dates"]:
            qualification_date = day
            break
    close_date = max(earliest_end, qualification_date) if qualification_date else None
    earliest_readout = (datetime.combine(close_date+timedelta(days=1+plan["settlement_grace_days"]), time.min, LONDON)
                        if close_date else None)
    phase = "collecting"
    if close_date and moment.astimezone(LONDON).date() > close_date:
        phase = "settlement_grace" if moment < earliest_readout else "ready_for_descriptive_readout"
    included = [row for row in observed if close_date and row[0] <= close_date]
    return {"checked_at": moment.isoformat(), "study_protocol_sha256": study.protocol_sha256,
        "phase": phase, "formal_confirmation_assessed": False,
        "eligible_start_date": start.isoformat(), "published_eligible_pairs": len(observed),
        "distinct_race_dates": len(counts), "earliest_sample_close_date": earliest_end.isoformat(),
        "sample_close_date": close_date.isoformat() if close_date else None,
        "earliest_readout_at": earliest_readout.isoformat() if earliest_readout else None,
        "readout_sample_pairs": len(included),
        "readout_capture_ids": [row[2] for row in included],
        "scope": "Readiness uses publication metadata only; no result labels or automatic performance decision"}
