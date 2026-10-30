from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from src.forward_readout import register_plan, load_plan, readout_readiness


@pytest.fixture
def context(tmp_path):
    study = SimpleNamespace(path=tmp_path, protocol_sha256="protocol", verify=lambda: None,
        protocol={"model_id": "pinned", "model_sha256": "artifact", "first_eligible_date": "2026-10-27"})
    plan = {"study_protocol_sha256": "protocol", "model_id": "pinned", "model_sha256": "artifact",
        "formal_confirmation": False, "created_at": "2026-10-26T19:00:00Z", "eligible_start_date": "2026-10-27",
        "minimum_calendar_days": 3, "minimum_published_pairs": 3, "minimum_race_dates": 2, "settlement_grace_days": 2}
    def record(identifier, day):
        start = f"2026-10-{day:02d}T15:00:00Z"
        return {"event_id": str(identifier), "capture_id": f"capture_{identifier}", "first_start": start,
                "published_at": start.replace("15:00", "14:55"), "benchmark_eligible": True, "paired_available": True}
    records = [record(1, 27), record(2, 27), record(3, 28), record(4, 29), record(5, 30)]
    return study, plan, records


def test_readout_waits_for_full_calendar_window_and_keeps_entire_last_day(context):
    study, plan, records = context
    before = readout_readiness(study, plan, records=records, now="2026-10-29T18:00:00Z")
    assert before["phase"] == "collecting" and before["sample_close_date"] == "2026-10-29"
    assert before["published_eligible_pairs"] == 4
    assert before["readout_capture_ids"] == ["capture_1", "capture_2", "capture_3", "capture_4"]
    assert not before["formal_confirmation_assessed"]
    grace = readout_readiness(study, plan, records=records, now="2026-10-30T12:00:00Z")
    assert grace["phase"] == "settlement_grace"
    assert grace["published_eligible_pairs"] == 4  # The fifth race has not published yet.
    ready = readout_readiness(study, plan, records=records, now="2026-11-01T00:00:00Z")
    assert ready["phase"] == "ready_for_descriptive_readout" and ready["readout_sample_pairs"] == 4
    assert ready["published_eligible_pairs"] == 5  # Later records cannot expand the closed sample.


def test_count_and_date_requirements_prevent_premature_close(context):
    study, plan, records = context
    plan["minimum_race_dates"] = 4
    report = readout_readiness(study, plan, records=records[:4], now="2027-01-01T00:00:00Z")
    assert report["phase"] == "collecting" and report["sample_close_date"] is None
    assert report["readout_sample_pairs"] == 0


@pytest.mark.parametrize("problem", ["duplicate", "ineligible", "other_model", "formal", "future_plan", "backdated"])
def test_inconsistent_plans_or_records_fail(context, problem):
    study, plan, records = context
    if problem == "duplicate": records.append(deepcopy(records[0]))
    elif problem == "ineligible": records[0]["paired_available"] = False
    elif problem == "other_model": plan["model_sha256"] = "changed"
    elif problem == "formal": plan["formal_confirmation"] = True
    elif problem == "future_plan": plan["created_at"] = "2028-01-01T00:00:00Z"
    else: plan["eligible_start_date"] = "2026-10-26"
    with pytest.raises(ValueError):
        readout_readiness(study, plan, records=records, now="2027-01-01T00:00:00Z")


def test_registration_is_exclusive_anchored_and_uses_next_full_uk_day(context, tmp_path):
    study, _, _ = context
    path = tmp_path / "pilot.json"
    # July UTC23:30 is already the next UK calendar day; eligible day is July3.
    study.protocol["first_eligible_date"] = "2026-07-01"
    register_plan(study, path, now="2026-07-01T23:30:00Z")
    value = load_plan(path)
    assert value["eligible_start_date"] == "2026-07-03"
    assert value["minimum_calendar_days"] == 84 and value["minimum_published_pairs"] == 1000
    with pytest.raises(FileExistsError): register_plan(study, path)
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="changed"):
        load_plan(path)
