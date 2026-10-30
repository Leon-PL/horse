"""Forward records must be complete, timely, immutable and outcome-independent."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import forward_capture as capture
from src.forward_store import ForwardStudy, publish_json, create_study
from src.forward_assessment import assess_records

NOW = datetime(2026, 10, 28, 16, 54, 45, tzinfo=timezone.utc)
EVENT_ID = "34688173775100023"
MARKET_ID = "34688173878503072"


@pytest.fixture
def context(tmp_path, monkeypatch):
    protocol = {
        "model_id": "pinned-run", "model_sha256": "fixed-model", "first_eligible_date": "2026-10-27",
        "horizon_seconds": 300, "capture_window_seconds": 30,
        "max_runner_price_age_seconds": 300, "max_market_age_seconds": 90,
        "max_card_age_seconds": 90, "clock_skew_tolerance_seconds": 2,
        "max_history_lag_days": 2, "minimum_available_gbp": 2.0,
        "reference": "Matchbook normalised best-back GBP", "assessment_scope": "Monitoring only",
    }
    root = tmp_path / "study"
    root.mkdir()
    publish_json(root / "protocol.json", protocol)
    publish_json(root / "protocol.sha256.json", {"sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()})
    study = ForwardStudy(root)
    monkeypatch.setattr(study, "verify", lambda: None)  # Pure orchestration fixture; native pin tested separately.
    event = {"id": int(EVENT_ID), "name": "17:00 Lingfield", "start": "2026-10-28T17:00:00Z",
             "status": "open", "in-running-flag": False,
             "meta-tags": [{"type": "COUNTRY", "name": "UK"}]}
    market = {"id": int(MARKET_ID), "event-id": int(EVENT_ID), "name": "WIN", "status": "open",
              "in-running-flag": False, "live": False, "number-of-winners": 1,
              "type": "multirunner", "product": "EXCHANGE", "runners": []}
    for index, (name, odds) in enumerate(zip(["Alpha", "Bravo", "Charlie"], [3.0, 4.0, 8.0])):
        market["runners"].append({"id": 34688173879100072+index, "market-id": int(MARKET_ID),
            "name": f"{index+1}. {name}", "status": "open", "withdrawn": False,
            "last-price-update-time": (NOW-timedelta(seconds=3)).isoformat(),
            "prices": [{"side": "back", "odds": odds, "available-amount": 5.0,
                        "currency": "GBP", "odds-type": "DECIMAL"}]})
    cards = pd.DataFrame({"race_id": ["race-1"]*3, "horse_name": ["Alpha", "Bravo", "Charlie"],
        "race_date": ["2026-10-28"]*3, "track": ["Lingfield"]*3, "off_time": ["17:00"]*3,
        "num_runners": [3]*3, "handicap": [0]*3, "won": [0]*3, "finish_position": [0]*3,
        "odds": [3.5, 3.5, 7.0], "signal": [1., 2., 3.]})
    class Client:
        def get_event(self, event_id):
            assert str(event_id) == EVENT_ID
            return deepcopy(event)
        def get_event_markets(self, event_id, **kwargs):
            assert kwargs["names"] == "WIN" and kwargs["currency"] == "GBP"
            return {"markets": [deepcopy(market)]}
    class Predictor:
        feature_cols = ["signal", "num_runners", "handicap"]
        def predict_races(self, frame, *, win_only):
            assert win_only
            return frame.assign(win_probability=frame.horse_name.map({"Alpha": .5, "Bravo": .3, "Charlie": .2}))
    from src import live_prediction, feature_store
    store = SimpleNamespace(meta=lambda: {"last_date": "2026-10-27", "fe_signature": "fixed"})
    monkeypatch.setattr(feature_store, "FeatureStore", lambda: store)
    monkeypatch.setattr(live_prediction, "feature_store_signature", lambda store: "stable-vintage")
    monkeypatch.setattr(live_prediction, "features_for_live_cards", lambda frame, *a, **k: frame.copy())
    monkeypatch.setattr(capture, "_load_model", lambda study: Predictor())
    study.observe_event(event, NOW-timedelta(hours=1))
    def fetch(event):
        return cards.copy(), {"url": "https://example.test/card", "started_at": NOW.isoformat(),
                              "received_at": NOW.isoformat(), "payload": {"unissued_labels": False}}
    return SimpleNamespace(study=study, protocol=protocol, event=event, market=market,
                           cards=cards, client=Client(), fetch=fetch)


def issue(ctx, *, now=NOW):
    return capture.capture_event(ctx.study, EVENT_ID, client=ctx.client, card_fetcher=ctx.fetch,
                                 clock=lambda: now)


def test_complete_pair_preserves_exact_ids_and_first_success(context):
    result = issue(context)
    assert result["paired_available"] and result["selected_for_benchmark"]
    record = context.study.verify_record(result["capture_id"])
    frame = pd.read_parquet(context.study.path / "records" / result["capture_id"] / "forecasts.parquet")
    assert frame.runner_id.iloc[0] == "34688173879100072"
    np.testing.assert_allclose(frame.market_probability, np.array([1/3, 1/4, 1/8])/(1/3+1/4+1/8))
    assert frame.model_probability.tolist() == [.5, .3, .2]
    assert record["market_id"] == MARKET_ID
    assert len(list(context.study.selected_records())) == 1
    with pytest.raises(ValueError, match="already recorded"):
        issue(context)


def test_place_refinement_and_legacy_control_are_saved_before_results(context,monkeypatch):
    context.study.protocol["record_place_probabilities"] = True
    context.cards["race_type"] = "Flat"
    class Predictor:
        feature_cols = ["signal","num_runners","handicap"]
        place_refinement = object()
        def predict_races(self,frame,*,win_only):
            assert not win_only
            return frame.assign(win_probability=frame.horse_name.map({"Alpha":.5,"Bravo":.3,"Charlie":.2}),
                place_probability=frame.horse_name.map({"Alpha":.9,"Bravo":.8,"Charlie":.7}) if self.place_refinement is not None else 1.)
    monkeypatch.setattr(capture,"_load_model",lambda study:Predictor())
    result = issue(context)
    assert result["selected_for_benchmark"]
    frame = pd.read_parquet(context.study.path/"records"/result["capture_id"]/"forecasts.parquet")
    assert frame.model_place_probability.tolist() == [.9,.8,.7]
    assert frame.legacy_place_probability.eq(1).all() and frame.forecast_places.eq(3).all()
    labels = context.cards.assign(won=[1,0,0],finish_position=[1,2,0])
    report,_ = assess_records(context.study,labels,now=NOW+timedelta(days=1))
    assert report["place_comparison"]["model_brier"] == pytest.approx(.18)
    assert report["place_comparison"]["legacy_brier"] == pytest.approx(1/3)


@pytest.mark.parametrize("problem,expected", [
    ("missing", "incomplete_back_prices"), ("lay_only", "incomplete_back_prices"),
    ("stale", "stale_or_future_runner_price"), ("future", "stale_or_future_runner_price"),
    ("currency", "unexpected_price_currency"), ("inplay", "market_not_open_preplay"),
    ("suspended", "runner_state_unknown"), ("duplicate", "duplicate_market_runner"),
    ("thin", "incomplete_back_prices"),
])
def test_invalid_market_never_substitutes_or_drops_a_runner(context, problem, expected):
    runner = context.market["runners"][0]
    if problem == "missing": runner["prices"] = []
    elif problem == "lay_only": runner["prices"][0]["side"] = "lay"
    elif problem == "stale": runner["last-price-update-time"] = (NOW-timedelta(seconds=301)).isoformat()
    elif problem == "future": runner["last-price-update-time"] = (NOW+timedelta(seconds=3)).isoformat()
    elif problem == "currency": runner["prices"][0]["currency"] = "USD"
    elif problem == "inplay": context.market["in-running-flag"] = True
    elif problem == "suspended": runner["status"] = "suspended"
    elif problem == "thin": runner["prices"][0]["available-amount"] = 1.99
    else: context.market["runners"].append(deepcopy(runner))
    result = issue(context)
    assert not result["selected_for_benchmark"]
    assert expected in result["errors"]["market"]
    frame = pd.read_parquet(context.study.path / "records" / result["capture_id"] / "forecasts.parquet")
    assert len(frame) == 3 and frame.market_probability.isna().all()


def test_changed_field_keeps_both_forecasts_without_partial_renormalisation(context):
    context.market["runners"].pop()
    result = issue(context)
    assert not result["paired_available"] and not result["selected_for_benchmark"]
    folder = context.study.path / "records" / result["capture_id"]
    assert len(pd.read_parquet(folder / "market_forecast.parquet")) == 2
    assert len(pd.read_parquet(folder / "independent_forecast.parquet")) == 3
    assert "field_mismatch" in result["errors"]["market"]


def test_labelled_racecard_cannot_supply_independent_forecast(context):
    context.cards.loc[0, ["won", "finish_position"]] = 1
    result = issue(context)
    assert "contains_outcomes" in result["errors"]["independent"]
    assert not result["paired_available"]
    frame = pd.read_parquet(context.study.path / "records" / result["capture_id"] / "forecasts.parquet")
    assert frame.model_probability.isna().all() and frame.market_probability.notna().all()


def test_card_change_after_slow_native_inference_rejects_independent_probability(context):
    original_fetch = context.fetch
    calls = []
    def changing_fetch(event):
        cards, receipt = original_fetch(event)
        if calls:
            cards.loc[0, "odds"] = 20.0
        calls.append(True)
        return cards, receipt
    context.fetch = changing_fetch
    result = issue(context)
    assert "racecard_changed_during_forecast" in result["errors"]["independent"]
    assert not result["paired_available"] and not result["selected_for_benchmark"]
    folder = context.study.path / "records" / result["capture_id"]
    assert (folder / "racecard_refresh.parquet").exists()
    frame = pd.read_parquet(folder / "forecasts.parquet")
    assert frame.model_probability.isna().all() and frame.market_probability.notna().all()


def test_exact_fresh_card_recheck_can_follow_slow_native_inference(context, monkeypatch):
    done = []
    original_loader = capture._load_model
    def load(study):
        done.append(True)
        return original_loader(study)
    monkeypatch.setattr(capture, "_load_model", load)
    def clock():
        return NOW if done else NOW-timedelta(seconds=150)
    def fetch(event):
        observed = clock().isoformat()
        return context.cards.copy(), {"started_at": observed, "received_at": observed}
    result = capture.capture_event(context.study, EVENT_ID, client=context.client,
                                   card_fetcher=fetch, clock=clock)
    assert result["paired_available"] and result["selected_for_benchmark"]
    assert result["card_observed_at"] == NOW.isoformat()
    assert result["started_at"] == (NOW-timedelta(seconds=150)).isoformat()


@pytest.mark.parametrize("change,reason", [("schedule", "schedule_changed"),
    ("late_discovery", "schedule_first_seen_too_late"), ("early", "outside_five_minute_window")])
def test_fixed_schedule_and_timing_gates(context, change, reason):
    if change == "schedule":
        context.event["start"] = "2026-10-28T17:05:00Z"
        context.cards["off_time"] = "17:05"
    elif change == "late_discovery":
        path = context.study.path / "events" / f"{EVENT_ID}.json"
        value = json.loads(path.read_text()); value["first_seen_at"] = NOW.isoformat()
        path.write_text(json.dumps(value))
    now = NOW-timedelta(seconds=60) if change == "early" else NOW
    if change == "early":
        context.fetch = lambda event: (context.cards.copy(), {"started_at": now.isoformat(), "received_at": now.isoformat()})
        for row in context.market["runners"]: row["last-price-update-time"] = now.isoformat()
    result = issue(context, now=now)
    assert reason in result["reasons"] and not result["selected_for_benchmark"]


def test_actual_publication_time_controls_eligibility(context):
    def delayed_publication():
        return NOW+timedelta(seconds=20) if list((context.study.path / "records").glob("*/capture.json")) else NOW
    result = capture.capture_event(context.study, EVENT_ID, client=context.client,
        card_fetcher=context.fetch, clock=delayed_publication)
    assert not result["selected_for_benchmark"]
    assert "published_after_window" in result["reasons"]
    assert context.study.selected(EVENT_ID) is None


def test_discovery_pages_preserve_first_schedule_and_exact_ids(context):
    other = deepcopy(context.event)
    other["id"] += 17
    old_start = context.event["start"]
    context.event["start"] = "2026-10-28T17:02:00Z"
    class Pages:
        offsets = []
        def get_events(self, **kwargs):
            self.offsets.append(kwargs["offset"])
            return {"events": [context.event] if kwargs["offset"] == 0 else [other], "total": 2}
    client = Pages()
    receipt = capture.discover_events(context.study, client=client, clock=lambda: NOW)
    assert client.offsets == [0, 1] and not receipt["truncated"]
    assert receipt["events"][0]["id"] == EVENT_ID
    assert receipt["events"][0]["first_start"] == old_start.replace("Z", "+00:00")
    assert receipt["events"][0]["start"] == context.event["start"]
    assert len(list((context.study.path / "discovery").glob("*.json"))) == 1


def test_repeating_discovery_page_is_bounded_and_marked_incomplete(context):
    class Repeated:
        calls = 0
        def get_events(self, **kwargs):
            self.calls += 1
            return {"events": [context.event], "total": 500}
    client = Repeated()
    receipt = capture.discover_events(context.study, client=client, clock=lambda: NOW)
    assert client.calls == 2 and receipt["truncated"]
    assert len(receipt["events"]) == 1


def test_empty_evidence_cannot_claim_an_eligible_record(context):
    capture_id, _ = context.study.begin_attempt()
    with pytest.raises(ValueError, match="complete paired evidence"):
        context.study.finish_attempt(capture_id, {"event_id": EVENT_ID, "benchmark_eligible": True})


@pytest.mark.parametrize("changed_card", [False, True])
def test_timed_capture_waits_then_refetches_card_and_prices(context, changed_card):
    from tests.test_forward_timing import Clock
    clock = Clock(NOW-timedelta(seconds=60))
    observations = []
    def fetch(event):
        observations.append(clock.read())
        frame = context.cards.copy()
        if changed_card and len(observations) > 1:
            frame.loc[0, "odds"] = 20.
        return frame, {"started_at": clock.read().isoformat(), "received_at": clock.read().isoformat()}
    original = context.client.get_event_markets
    def markets(*a, **kwargs):
        for runner in context.market["runners"]:
            runner["last-price-update-time"] = (clock.read()-timedelta(seconds=1)).isoformat()
        return original(*a, **kwargs)
    context.client.get_event_markets = markets
    result = capture.capture_event(context.study, EVENT_ID, client=context.client, card_fetcher=fetch,
        clock=clock.read, wait_for_issue_window=True, sleeper=clock.sleep, monotonic=clock.steady)
    assert observations[1] > observations[0]
    assert result["capture_mode"] == "timed"
    folder = context.study.path / "records" / result["capture_id"]
    receipt = json.loads((folder / "timing_receipt.json").read_text())
    assert receipt["wait_seconds"] == 47
    assert result["selected_for_benchmark"] is (not changed_card)
    if changed_card:
        assert "racecard_changed_during_forecast" in result["errors"]["independent"]


def test_history_change_while_waiting_invalidates_computed_forecast(context, monkeypatch):
    from tests.test_forward_timing import Clock
    from src import live_prediction
    clock = Clock(NOW-timedelta(seconds=60))
    monkeypatch.setattr(live_prediction, "feature_store_signature", lambda store:
        "initial" if clock.elapsed == 0 else "changed")
    def fetch(event):
        return context.cards.copy(), {"started_at": clock.read().isoformat(), "received_at": clock.read().isoformat()}
    result = capture.capture_event(context.study, EVENT_ID, client=context.client, card_fetcher=fetch,
        clock=clock.read, wait_for_issue_window=True, sleeper=clock.sleep, monotonic=clock.steady)
    assert "history_changed_during_forecast" in result["errors"]["independent"]
    assert not result["selected_for_benchmark"]


def test_interrupted_timed_capture_never_selects_a_partial_record(context):
    from tests.test_forward_timing import Clock
    clock = Clock(NOW-timedelta(seconds=60))
    def fetch(event):
        return context.cards.copy(), {"started_at": clock.read().isoformat(), "received_at": clock.read().isoformat()}
    def interrupt(seconds):
        raise KeyboardInterrupt("manual cancellation")
    with pytest.raises(KeyboardInterrupt):
        capture.capture_event(context.study, EVENT_ID, client=context.client, card_fetcher=fetch,
            clock=clock.read, wait_for_issue_window=True, sleeper=interrupt, monotonic=clock.steady)
    assert context.study.selected(EVENT_ID) is None
    assert list((context.study.path / "records").glob("*/timing_intent.json"))
    assert not list((context.study.path / "records").glob("*/capture.json"))


def test_native_pinned_artifact_matches_saved_registry_model(tmp_path):
    from src.run_store import get_production_run_id, load_run_predictor
    from src.feature_store import FeatureStore
    from src.live_prediction import feature_store_signature
    run_id = get_production_run_id()
    if run_id is None:
        pytest.skip("No local production artifact installed")
    study = create_study(tmp_path / "native-study")
    study.verify()
    assert study.protocol["model_id"] == run_id
    model = capture._load_model(study)
    actual_store = FeatureStore()
    cols = list(dict.fromkeys(model.feature_cols + ["race_id", "horse_name", "odds"]))
    # Native inference verification only; historical inputs never become a
    # forward capture or an eligible study record.
    meta = pd.read_parquet(actual_store.featured_path, columns=["race_id"])
    ids = meta.race_id.drop_duplicates().iloc[-3:].tolist()
    frame = pd.read_parquet(actual_store.featured_path, columns=cols, filters=[("race_id", "in", ids)])
    result = model.predict_races(frame.sort_values(["race_id", "horse_name"]), win_only=True)
    expected = load_run_predictor(run_id).predict_races(frame.sort_values(["race_id", "horse_name"]), win_only=True)
    np.testing.assert_array_equal(result.win_probability, expected.win_probability)
    np.testing.assert_allclose(result.groupby("race_id").win_probability.sum(), 1, atol=1e-12)
    assert list(study.selected_records()) == []
    assert feature_store_signature(actual_store)


def test_protocol_and_record_tampering_fail_closed(context):
    result = issue(context)
    folder = context.study.path / "records" / result["capture_id"]
    with (folder / "forecasts.parquet").open("ab") as stream: stream.write(b"tampered")
    with pytest.raises(ValueError, match="inputs changed"):
        list(context.study.selected_records())
    path = context.study.path / "protocol.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="protocol changed"):
        ForwardStudy(context.study.path)


def test_assessment_retains_dnfs_and_excludes_later_field_changes(context):
    result = issue(context)
    labels = context.cards.copy()
    labels["won"] = [1, 0, 0]
    labels["finish_position"] = [1, 2, 0]  # Actual DNF retained as a loser.
    report, paired = assess_records(context.study, labels, now=NOW+timedelta(hours=1))
    assert report["races"] == 1 and len(paired) == 3
    assert not report["confirmation_assessed"]
    changed = labels.iloc[:2].copy(); changed["num_runners"] = 2
    report, paired = assess_records(context.study, changed)
    assert paired.empty and report["coverage"]["changed_field_or_identity"] == 1
    assert len(pd.read_parquet(context.study.path / "records" / result["capture_id"] / "forecasts.parquet")) == 3
