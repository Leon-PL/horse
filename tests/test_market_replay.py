import copy
import json

import numpy as np
import pytest

from src.market_replay import ReplayRejected, replay_market, split_market_streams


def messages():
    # 2026-07-01 14:00 UTC; UK local time 15:00.
    off = 1782914400000
    runners = [{"id": i, "name": f"{i}. Horse {i} (IRE)", "status": "ACTIVE"} for i in [1, 2, 3]]
    md = {"marketType": "WIN", "countryCode": "GB", "eventTypeId": "7",
          "marketTime": "2026-07-01T14:00:00.000Z", "eventName": "Newcastle 1st Jul",
          "status": "OPEN", "inPlay": False, "numberOfActiveRunners": 3, "runners": runners}
    final = copy.deepcopy(md)
    final.update(status="CLOSED", inPlay=True, numberOfActiveRunners=0)
    for r, status in zip(final["runners"], ["WINNER", "LOSER", "LOSER"]):
        r["status"] = status
    return [
        {"pt": off - 600000, "mc": [{"id": "1.123", "marketDefinition": md,
            "rc": [{"id": 1, "ltp": 2.0}, {"id": 2, "ltp": 4.0}, {"id": 3, "ltp": 4.0}]}]},
        {"pt": off - 300000, "mc": [{"id": "1.123", "rc": [{"id": 1, "ltp": 3.0}]}]},
        {"pt": off - 299999, "mc": [{"id": "1.123", "rc": [{"id": 2, "ltp": 1000.0}]}]},
        {"pt": off + 600000, "mc": [{"id": "1.123", "marketDefinition": final}]},
    ]


def encode(msgs):
    return b"\n".join(json.dumps(m).encode() for m in msgs)


def test_replay_cutoff_inclusive_and_change_only_prices_carried_forward():
    result = replay_market(encode(messages()))
    assert [r["market_price"] for r in result] == [3, 4, 4]
    np.testing.assert_allclose([r["market_probability"] for r in result], [.4, .3, .3])
    assert result[0]["off_key"] == "14:00"
    assert result[0]["off_key_local"] == "15:00"
    assert result[1]["price_changed_ms"] < result[1]["observed_ms"] <= result[1]["decision_ms"]


def test_future_information_changes_only_labels():
    original = replay_market(encode(messages()))
    changed = messages()
    md = changed[-1]["mc"][0]["marketDefinition"]
    md["marketTime"] = "2026-07-01T18:30:00.000Z"
    md["eventName"] = "Wrong future track"
    md["runners"][2].update(status="REMOVED", name="Renamed in future")
    changed[2]["mc"][0]["rc"][0]["ltp"] = 1.01
    replay = replay_market(encode(changed))
    labels = {"won", "settlement_status", "withdrawn_after_decision"}
    assert [{k: v for k, v in r.items() if k not in labels} for r in original] == [
        {k: v for k, v in r.items() if k not in labels} for r in replay]
    assert len(replay) == 3 and replay[2]["withdrawn_after_decision"]
    assert replay[2]["won"] == 0


def test_removed_before_decision_is_excluded_from_that_field():
    msgs = messages()
    md = copy.deepcopy(msgs[0]["mc"][0]["marketDefinition"])
    md["runners"][2]["status"] = "REMOVED"
    md["numberOfActiveRunners"] = 2
    msgs[1]["mc"][0]["marketDefinition"] = md
    msgs[-1]["mc"][0]["marketDefinition"]["runners"][2]["status"] = "REMOVED"
    assert len(replay_market(encode(msgs))) == 2


def test_event_file_demultiplexes_independent_definitions_and_prices():
    first, second = messages(), copy.deepcopy(messages())
    for m in second:
        m["mc"][0]["id"] = "1.456"
    second[0]["mc"][0]["rc"][0]["ltp"] = 10
    combined = [{"pt": a["pt"], "mc": a["mc"] + b["mc"]} for a, b in zip(first, second)]
    split = list(split_market_streams(encode(combined)))
    assert len(split) == 2
    results = [replay_market(raw) for _, raw in split]
    assert {r[0]["market_id"] for r in results} == {"1.123", "1.456"}
    assert results[0][1]["market_price"] == results[1][1]["market_price"] == 4


@pytest.mark.parametrize("problem, reason", [
    ("missing_price", "missing_or_invalid_active_price"),
    ("partial_field", "incomplete_active_field"),
    ("inplay", "not_open_preplay_at_decision"),
    ("late_archive", "archive_begins_after_decision"),
    ("out_of_order", "missing_or_nonmonotonic_timestamp"),
    ("dead_heat", "not_single_winner_settled"),
    ("bad_json", "malformed_json"),
])
def test_reject_whole_market(problem, reason):
    msgs = messages()
    if problem == "missing_price":
        msgs[0]["mc"][0]["rc"].pop()
    elif problem == "partial_field":
        msgs[0]["mc"][0]["marketDefinition"]["numberOfActiveRunners"] = 4
    elif problem == "inplay":
        msgs[0]["mc"][0]["marketDefinition"]["inPlay"] = True
    elif problem == "late_archive":
        msgs[0]["pt"] = msgs[1]["pt"] + 1
    elif problem == "out_of_order":
        msgs[2]["pt"] = msgs[0]["pt"] - 1
    elif problem == "dead_heat":
        msgs[-1]["mc"][0]["marketDefinition"]["runners"][1]["status"] = "WINNER"
    raw = encode(msgs) + (b"\nnot json" if problem == "bad_json" else b"")
    with pytest.raises(ReplayRejected, match=reason):
        replay_market(raw)
