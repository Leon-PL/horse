import pytest

from src.ingestion.sporting_life import SportingLifeScraper


@pytest.mark.parametrize("method", ["scrape_race_result", "scrape_racecard"])
@pytest.mark.parametrize("name,source_code,expected", [
    ("Purchase Hospitality Mares' Novices' Hurdle", "", "Hurdle"),
    ("JenningsBet Chipchase Stakes (Group 3)", "", "Flat"),
    ("Ladbrokes Punchestown Gold Cup (Grade 1)", "", "Chase"),
    ("Sponsor Novices' Hurdle", "CHASE", "Chase"),
])
def test_result_and_card_ingestion_use_verified_type_parser(monkeypatch, method, name, source_code, expected):
    scraper = SportingLifeScraper()
    payload = {"props": {"pageProps": {"isUkorIreMeeting": True, "race": {
        "race_summary": {"race_summary_reference": {"id": 1234}, "name": name,
            "course_name": "Punchestown", "date": "2026-10-28", "distance": "2m",
            "time": "14:00", "ride_count": 2, "run_type": source_code},
        "rides": [{"id": 1}, {"id": 2}]}}}}
    monkeypatch.setattr(scraper, "_fetch_next_data", lambda url: payload)
    def parse_ride(ride, **context):
        return {**context, "horse_name": f"Horse {ride['id']}", "finish_position": ride["id"]}
    monkeypatch.setattr(scraper, "_ride_to_dict", parse_ride)
    result = getattr(scraper, method)("https://example.test/race/1234")
    assert len(result) == 2
    assert {row["race_type"] for row in result} == {expected}


@pytest.mark.parametrize("method", ["scrape_race_result", "scrape_racecard"])
@pytest.mark.parametrize("reason,label", [
    ("PulledUp", "PU"), ("Fell", "F"), ("UnseatedRider", "UR"), ("Refused", "R"),
])
def test_withdrawn_horses_are_excluded_but_started_nonfinishers_remain(monkeypatch, method, reason, label):
    scraper = SportingLifeScraper()
    def ride(identity, name, status, position, casualty=None):
        return {"horse": {"horse_reference": {"id": identity}, "name": name, "age": 5},
            "ride_status": status, "finish_position": position, "handicap": "11-7",
            "casualty": casualty or {}}
    payload = {"props": {"pageProps": {"isUkorIreMeeting": True, "race": {
        "race_summary": {"race_summary_reference": {"id": 1234}, "name": "Handicap Hurdle",
            "course_name": "Punchestown", "date": "2026-10-28", "distance": "2m",
            "time": "14:00", "ride_count": 2},
        "rides": [ride(1, "Finisher", "RUNNER", 1),
                  ride(2, "Did Not Finish", "RUNNER", 0, {"reason": reason}),
                  ride(3, "Withdrawn Entry", "WITHDRAWN", 0),
                  ride(4, "Nonrunner Entry", "NONRUNNER", 0)]}}}}
    monkeypatch.setattr(scraper, "_fetch_next_data", lambda url: payload)
    rows = getattr(scraper, method)("https://example.test/race/1234")
    assert {row["horse_name"] for row in rows} == {"Finisher", "Did Not Finish"}
    assert all(row["num_runners"] == 2 for row in rows)
    if method == "scrape_race_result":
        nonfinisher = next(row for row in rows if row["horse_name"] == "Did Not Finish")
        assert nonfinisher["finish_position"] == 0
        assert nonfinisher["finish_pos_label"] == label
        assert nonfinisher["won"] == 0
    else:
        assert all(row["finish_pos_label"] == "" for row in rows)


@pytest.fixture
def parse_result_ride():
    def parse(position, casualty, *, is_result=True):
        return SportingLifeScraper()._ride_to_dict(
            {"finish_position": position, "casualty": casualty, "ride_status": "RUNNER"},
            race_id="1234", date_str="2026-10-28", off_time="14:00", course_name="Bath",
            region="England", race_name="Handicap", race_class="4", race_type="Flat",
            distance_f=8., going="Good", prize_money=1000., num_runners=2,
            surface_str="Turf", has_handicap=True, is_result=is_result)
    return parse


@pytest.mark.parametrize("position", [0, 0., "0", "0.0", " 0 ", None, ""])
def test_numeric_or_missing_position_recovers_structured_reason(parse_result_ride, position):
    row = parse_result_ride(position, {"reason": "UnseatedRider"})
    assert (row["finish_position"], row["finish_pos_label"], row["won"]) == (0, "UR", 0)


@pytest.mark.parametrize("label", ["PU", "F", "BD", "RR", "WO"])
def test_explicit_result_label_has_precedence(parse_result_ride, label):
    row = parse_result_ride(label, {"reason": "Fell"})
    assert row["finish_pos_label"] == label


@pytest.mark.parametrize("casualty", [None, {}, "Fell", {"reason": None},
    {"reason": []}, {"reason": "UnrecognisedReason"}])
def test_unknown_casualty_does_not_invent_a_reason(parse_result_ride, casualty):
    row = parse_result_ride(0, casualty)
    assert (row["finish_position"], row["finish_pos_label"], row["won"]) == (0, "", 0)


@pytest.mark.parametrize("position", [1, 2])
def test_positive_finish_has_precedence_over_casualty(parse_result_ride, position):
    row = parse_result_ride(position, {"reason": "Fell"})
    assert (row["finish_position"], row["finish_pos_label"], row["won"]) == (position, "", int(position == 1))


def test_pending_card_does_not_receive_any_result_signal(parse_result_ride):
    row = parse_result_ride("PU", {"reason": "PulledUp"}, is_result=False)
    assert (row["finish_position"], row["finish_pos_label"], row["won"]) == (0, "", 0)
