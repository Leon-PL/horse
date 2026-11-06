"""Regression tests for the Racing TV (RaceiQ) scraper.

Fixtures in ``tests/fixtures/rtv`` are real API payloads captured on
2026-10-29 (trimmed of image/video links): a flat race in BST (Kempton
2025-08-13), an Irish flat race with a disqualification and a non-runner
(Gowran Park 2025-08-13), a race with joint favourites (Ffos Las
2025-08-13, results only) and a GMT hurdle with pulled-up runners
(Stratford 2026-10-29). No network access is required.
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ingestion import racing_tv as rtv

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "rtv")


def load(name):
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def race_payloads(name):
    return {kind: load(f"{name}_{kind}.json")
            for kind in ("results", "racecard", "iq_metrics", "iq_sectional")}


def parsed_tables(name):
    p = race_payloads(name)
    parsed = rtv.parse_race(p["results"], p["racecard"], p["iq_metrics"], p["iq_sectional"])
    tables = {k: pd.DataFrame(v) for k, v in parsed.items()}
    tables["races"] = rtv._derive_races(tables["races"])
    tables["runners"] = rtv._derive_runners(tables["runners"])
    rtv._attach_race_context(tables)
    return tables


def runner(tables, horse):
    ru = tables["runners"]
    return ru.loc[ru.horse_name == horse].iloc[0]


def test_numeric_tail_order_cannot_turn_a_casualty_into_a_finisher():
    raw = pd.DataFrame({"race_id": [1, 1, 1], "res_finish_position": [1, 9, 8],
        "res_status__state": ["finished", "casualty", "disqualified"],
        "card_withdrawn": [False, False, False]})
    result = rtv._derive_runners(raw)
    assert result.is_finisher.tolist() == [True, False, True]
    assert not result.is_nonrunner.iloc[1]
    assert result.finish_position.iloc[1] == 9  # preserve the source number


# ── value parsers ─────────────────────────────────────────────────────

@pytest.mark.parametrize("text,lengths", [
    ("2 1/2 l", 2.5), ("1/2 l", 0.5), ("35 l", 35.0), ("1 l", 1.0),
    ("Neck", 0.3), ("Nose", 0.05), ("Short Head", 0.1), ("Head", 0.2),
    ("Dead Heat", 0.0), ("1¾", 1.75),
])
def test_parse_margin(text, lengths):
    assert rtv.parse_margin(text) == pytest.approx(lengths)


@pytest.mark.parametrize("text", [None, "", "Dist", "a long way", "1 2 3 l"])
def test_parse_margin_unknown_is_nan_not_zero(text):
    assert np.isnan(rtv.parse_margin(text))


@pytest.mark.parametrize("text,secs", [
    ("1m, 28.11s", 88.11), ("5m, 31.9s", 331.9), ("59.3s", 59.3), ("23.39s", 23.39),
])
def test_parse_race_time(text, secs):
    assert rtv.parse_race_time(text) == pytest.approx(secs)


def test_parse_race_time_garbage():
    assert np.isnan(rtv.parse_race_time("n/a"))
    assert np.isnan(rtv.parse_race_time(None))


def test_parse_weight_and_equipment():
    assert rtv.parse_weight_lbs("9-2") == 128
    assert rtv.parse_weight_lbs("11-10") == 164
    assert np.isnan(rtv.parse_weight_lbs("128"))
    assert rtv.parse_equipment("tp<sup>1</sup>") == ("tp", 1)
    assert rtv.parse_equipment("b") == ("b", 0)
    assert rtv.parse_equipment("") == (None, None)


@pytest.mark.parametrize("text,symbols,first,ambiguous,wind,decoded", [
    ("0", "p", None, None, 0, True), ("z<sup>1</sup>", "t", "t", None, 0, True),
    ("+<sup>1</sup>", "p,t", None, "+", 0, True), ("B", "b", None, None, 0, True),
    ("b", "b,t", None, None, 0, True), ("D", "b,e/s", None, None, 0, True),
    ("-<sup>1</sup>", "h,p", None, "-", 0, True), ("E<sup>1</sup>", "e/s", "e/s", None, 0, True),
    ("w<sup>1</sup> z<sup>1</sup>", "t", "t", None, 1, True), ("w<sup>1</sup>", None, None, None, 1, True),
    ("Q", None, None, None, 0, False), ("", None, None, None, None, True),
])
def test_decode_equipment(text, symbols, first, ambiguous, wind, decoded):
    d = rtv.decode_equipment(text)
    assert (d["symbols"], d["first_time"], d["first_time_ambiguous"], d["wind_op"], d["decoded"]) == \
        (symbols, first, ambiguous, wind, decoded)


def test_every_glyph_seen_in_the_sample_is_decoded():
    seen = ["0", "z", "B", "+", "H", "V", "b", "h", "w", "v", "-", "E", "A", "D", "e", "g", "*", "d", "C", "O",
            "}", "j", "J", "a", "=", "o", "G", "F", "c", "L"]
    assert all(rtv.decode_equipment(g)["decoded"] for g in seen)


@pytest.mark.parametrize("raw,clean", [
    ("J. A. Powell", "J A Powell"), ("P. Mulligan, Ireland", "P Mulligan"),
    ("J & T Gosden", "J & T Gosden"), ("Sir Mark Prescott", "Sir Mark Prescott"),
])
def test_clean_person(raw, clean):
    assert rtv._clean_person(raw) == clean


def test_name_forms_pick_initials_by_shape():
    df = pd.DataFrame({"a": ["James Andrew Fahey", "A M Balding"], "b": ["J Fahey", "Andrew Balding"]})
    short, full = rtv._name_forms(df, ["a", "b"])
    assert short.tolist() == ["J Fahey", "A M Balding"]
    assert full.tolist() == ["James Andrew Fahey", "Andrew Balding"]


def test_times_from_scheduled_start():
    # The URL uses local time; races.db uses UTC.
    assert rtv.local_hhmm("2025-08-13T18:15:00+01:00") == "1815"
    assert rtv.utc_off_time("2025-08-13T18:15:00+01:00") == "17:15"
    assert rtv.utc_off_time("2026-10-29T13:20:00+00:00") == "13:20"
    assert rtv.local_hhmm("not a time") is None


def test_distance_text():
    assert rtv._distance_yards_from_text("2m 6f 7y") == 4847
    assert rtv._distance_yards_from_text("7f") == 1540
    assert rtv._sector_furlongs("2m 5f") == pytest.approx(21.0)


def test_flatten_keeps_field_and_object_with_same_stem():
    # Results carry trainer_name (initials) AND trainer.name (full name).
    out = rtv._flatten({"trainer_name": "A M Balding", "trainer": {"name": "Andrew Balding"}}, "res")
    assert out == {"res_trainer_name": "A M Balding", "res_trainer__name": "Andrew Balding"}


# ── day list ──────────────────────────────────────────────────────────

def test_list_races_filters_countries_and_uses_local_time():
    races = rtv.list_races(load("day_list_2025-08-13.json"))
    assert {r["country_iso"] for r in races} == {"gb", "ie"}
    kempton = [r for r in races if r["track_slug"] == "kempton-park"]
    assert kempton[0]["hhmm"] == "1815"
    everything = rtv.list_races(load("day_list_2025-08-13.json"), countries=None)
    assert any(r["track_slug"] == "scottsville" for r in everything)


# ── parsing ───────────────────────────────────────────────────────────

def test_kempton_race_fields():
    t = parsed_tables("kempton_flat_bst")
    race = t["races"].iloc[0]
    assert race.race_id == 608747
    assert race.off_time_local == "18:15" and race.off_time_utc == "17:15"
    assert race.distance_yards == 1540 and race.distance_furlongs == pytest.approx(7.0)
    assert race.win_time_s == pytest.approx(88.11)
    assert race.first_prize == 4320 and race.race_class == "5"
    assert race.surface == "Polytrack" and race.going == "Standard / Slow"
    # Known API inconsistency: the sectional endpoint's start is an hour on in BST.
    assert race.sect_schedule_shift_min == 60
    assert race.sect_time_index_score == pytest.approx(5.1)


def test_kempton_runner_fields():
    t = parsed_tables("kempton_flat_bst")
    assert len(t["runners"]) == 9
    w = runner(t, "Pacific Mission")
    assert w.final_position == 1 and w.sp_decimal == 6.5 and w.weight_lbs == 128
    assert w.draw == 1 and w.cloth_number == 7
    assert w.equipment == "z" and w.equipment_first_time == 1
    assert w.trainer == "A M Balding" and w.trainer_full == "Andrew Balding"
    assert w.sect_fsp == pytest.approx(107.65)
    # Scrape-time pedigree-page fields are renamed, never left as race-day values.
    assert "card_horse__age" not in t["runners"]
    assert "scrape_card_horse__age" in t["runners"]
    assert runner(t, "Maneuver").sp_favourite_code == 1


def test_beaten_lengths_from_gap_to_previous_finisher():
    t = parsed_tables("kempton_flat_bst")
    assert runner(t, "Maneuver").margin_to_prev == 2.5
    assert runner(t, "Mayfair Market").beaten_lengths == pytest.approx(4.75)
    assert runner(t, "Querida Zarah").beaten_lengths == pytest.approx(24.8)


def test_metrics_and_sectionals_long_tables():
    t = parsed_tables("kempton_flat_bst")
    m = t["metrics"]
    assert len(m) == 90  # 9 runners x (4 speed + 6 stride)
    assert set(m["group"]) == {"speed", "stride"}
    fsp = m[(m.metric == "fsp")].set_index("runner_id")["value"]
    assert fsp.max() == pytest.approx(107.65)
    s = t["sectionals"]
    assert len(s) == 63  # 9 runners x 7 furlongs
    w = runner(t, "Pacific Mission").runner_id
    last = s[(s.runner_id == w)].sort_values("sector_number").iloc[-1]
    assert last.cumulative_sector_time == pytest.approx(88.11)
    assert last.sector_furlongs == pytest.approx(7.0)


def test_disqualification_keeps_physical_and_official_order():
    t = parsed_tables("gowran_dsq")
    dq = runner(t, "Yes Oui Si")
    assert dq.is_disqualified and dq.finish_position == 11 and dq.final_position == 13
    assert runner(t, "Kaptain Bay").final_position == 11
    # Margins chain in physical order, through the disqualified horse.
    assert runner(t, "Thilos").beaten_lengths == pytest.approx(19.4)
    nr = runner(t, "Hurricane Helen")
    assert nr.is_nonrunner and not nr.is_finisher
    assert t["races"].iloc[0].race_class is None  # Irish races have no class


def test_jumps_race_casualties_and_jumping_metrics():
    t = parsed_tables("stratford_hurdle_gmt")
    race = t["races"].iloc[0]
    assert race.sect_schedule_shift_min == 0  # GMT: no shift
    assert race.distance_furlongs == pytest.approx(4847 / 220)
    pu = runner(t, "Vajra")
    assert pu.state == "casualty" and not pu.is_finisher and np.isnan(pu.draw)
    assert set(t["metrics"]["group"]) == {"race_metrics", "jumping", "stride"}
    s = t["sectionals"]
    trail = s[s.runner_id == pu.runner_id].sort_values("sector_number")
    assert trail["sector_time"].iloc[0] > 0 and pd.isna(trail["sector_time"].iloc[-1])


# ── quality checks ────────────────────────────────────────────────────

def test_quality_checks_clean_race_has_only_known_api_shift():
    flags = rtv.quality_checks(parsed_tables("kempton_flat_bst"))
    assert set(flags["check"]) == {"sectional_schedule_shift"}


def test_quality_checks_joint_favourites_are_consistent():
    p = race_payloads("ffos_las_joint_fav")
    parsed = rtv.parse_race(p["results"])
    tables = {k: pd.DataFrame(v) for k, v in parsed.items()}
    tables["races"] = rtv._derive_races(tables["races"])
    tables["runners"] = rtv._derive_runners(tables["runners"])
    codes = tables["runners"].set_index("horse_name")["sp_favourite_code"]
    assert codes["Mcmurray"] == 2 and codes["Crest of Fire"] == 2
    flags = rtv.quality_checks(tables)
    assert "favourite_code_inconsistent" not in set(flags["check"])


def test_quality_checks_catch_planted_errors():
    t = parsed_tables("kempton_flat_bst")
    s = t["sectionals"]
    w = runner(t, "Maneuver").runner_id
    # A tracking glitch: 3s moved from sector 2 to sector 3.
    i2 = s.index[(s.runner_id == w) & (s.sector_number == 2)][0]
    i3 = s.index[(s.runner_id == w) & (s.sector_number == 3)][0]
    s.loc[i2, "sector_time"] += 3.0
    s.loc[i3, "sector_time"] -= 3.0
    m = t["metrics"]
    m.loc[m.index[(m.metric == "fsp")][0], "value"] = 15000.0
    t["races"].loc[0, "win_time_s"] = 90.0
    checks = set(rtv.quality_checks(t)["check"])
    assert {"sector_glitch", "metric_out_of_range", "winner_time_mismatch"} <= checks


# ── scraping with a fake client ───────────────────────────────────────

class FakeClient:
    """Serves fixture payloads by API path and records requests."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.n_requests = 0

    def get(self, path):
        self.calls.append(path)
        self.n_requests += 1
        if path in self.routes:
            return 200, json.dumps(self.routes[path]).encode("utf-8")
        return 404, b'{"meta":{"status":404}}'


def _routes_for_day():
    day = load("day_list_2025-08-13.json")
    day["meetings"] = [m for m in day["meetings"] if m["track"]["slug"] == "/kempton-park"]
    first = day["meetings"][0]["completed_races"][0]
    hhmm = rtv.local_hhmm(first["start_time_scheduled"])
    paths = rtv.race_paths("2025-08-13", "kempton-park", hhmm)
    p = race_payloads("kempton_flat_bst")
    return day, {rtv.day_list_path("2025-08-13"): day, paths["results"]: p["results"],
                 paths["racecard"]: p["racecard"], paths["iq_metrics"]: p["iq_metrics"],
                 paths["iq_sectional"]: p["iq_sectional"]}


def test_scrape_day_caches_final_answers(tmp_path):
    _, routes = _routes_for_day()
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    client = FakeClient(routes)
    summary = rtv.scrape_day("2025-08-13", client, store, today=date(2026, 10, 29), discover=False)
    assert summary["races"] == 2 and summary["done"] == 2
    first_run = len(client.calls)
    rtv.scrape_day("2025-08-13", client, store, today=date(2026, 10, 29), discover=False)
    # Second run: every answer (200 and 404) is final, the day is not recent.
    assert len(client.calls) == first_run
    assert store.day_status("2025-08-13") == "complete"
    tables = rtv.build_tables(store=store)
    assert 608747 in set(tables["races"]["race_id"])
    store.close()


def test_recent_incomplete_raceiq_is_refetched_once_per_run(tmp_path):
    _, routes = _routes_for_day()
    res_path = next(p for p in routes if p.startswith("/results/2025"))
    routes[res_path]["race"]["race_iq_state"]["metrics_complete"] = False
    metrics_path = res_path.replace("/results/", "/iq-results/") + "/metrics"
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    client = FakeClient(routes)
    recent = date(2025, 8, 15)
    rtv.scrape_day("2025-08-13", client, store, today=recent, discover=False)
    assert client.calls.count(res_path) == 1  # not fetched twice in one run
    rtv.scrape_day("2025-08-13", client, store, today=recent, discover=False)
    assert client.calls.count(res_path) == 2 and client.calls.count(metrics_path) == 2
    # Once the race is old, incomplete flags no longer trigger refetches.
    rtv.scrape_day("2025-08-13", client, store, today=date(2026, 10, 29), discover=False)
    assert client.calls.count(res_path) == 2
    store.close()


def test_build_by_year_round_trip(tmp_path):
    _, routes = _routes_for_day()
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    rtv.scrape_day("2025-08-13", FakeClient(routes), store, today=date(2026, 10, 29), discover=False)
    counts = rtv.build_by_quarter(store=store, directory=str(tmp_path / "tables"))
    assert counts["races"] == 1 and counts["sectionals"] == 63
    runners = rtv.load_table("runners", directory=str(tmp_path / "tables"))
    assert len(runners) == 9 and runners["horse_name"].iloc[0]
    store.close()


def test_scrape_day_skips_raceiq_when_results_say_none(tmp_path):
    day, routes = _routes_for_day()
    second = day["meetings"][0]["completed_races"][1]
    paths = rtv.race_paths("2025-08-13", "kempton-park", rtv.local_hhmm(second["start_time_scheduled"]))
    no_iq = json.loads(json.dumps(routes[next(p for p in routes if p.startswith("/results/2025"))]))
    no_iq["race"]["race_iq_state"] = {"show_metrics": False, "show_sectionals": False}
    routes[paths["results"]] = no_iq
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    client = FakeClient(routes)
    rtv.scrape_day("2025-08-13", client, store, today=date(2026, 10, 29), discover=False)
    assert paths["iq_metrics"] not in client.calls
    store.close()


class FlakyClient(FakeClient):
    def get(self, path):
        raise rtv.RTVTransientError(path)


def test_transient_failure_is_not_cached(tmp_path):
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    with pytest.raises(rtv.RTVTransientError):
        rtv.scrape_day("2025-08-13", FlakyClient({}), store, today=date(2026, 10, 29), discover=False)
    assert store.status(rtv.day_list_path("2025-08-13")) is None
    store.close()


# ── HTTP client ───────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status, body=b"{}", headers=None, text=""):
        self.status_code = status
        self.content = body
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.urls = []

    def get(self, url, headers=None, timeout=None):
        self.urls.append((url, dict(headers or {})))
        return self.responses.pop(0)


def test_client_retries_rate_limits(monkeypatch):
    monkeypatch.setattr(rtv.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(429), FakeResponse(503), FakeResponse(200, b'{"ok":1}')])
    client = rtv.RTVClient(request_delay=0, api_key="k", session=session)
    assert client.get("/results/x") == (200, b'{"ok":1}')
    assert all(h.get("API-KEY") == "k" for _, h in session.urls)


def test_client_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(rtv.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse(429)] * 3)
    client = rtv.RTVClient(request_delay=0, max_retries=2, api_key="k", session=session)
    with pytest.raises(rtv.RTVTransientError):
        client.get("/results/x")


def test_client_rediscovers_key_once_on_403(monkeypatch):
    monkeypatch.setattr(rtv.time, "sleep", lambda s: None)
    bundle = 'x={\\"extra\\":{\\"API_ENV\\":\\"production\\",\\"API_KEY\\":\\"0f0f0f0f-1e46-4f82-9d18-d81b4803537d\\"}}'
    session = FakeSession([
        FakeResponse(403),
        FakeResponse(200, text='<script src="/_expo/static/js/web/index-abc123.js"></script>'),
        FakeResponse(200, text=bundle),
        FakeResponse(200, b'{"ok":1}'),
    ])
    client = rtv.RTVClient(request_delay=0, api_key="stale", session=session)
    assert client.get("/iq-results/x/metrics") == (200, b'{"ok":1}')
    assert session.urls[-1][1]["API-KEY"] == "0f0f0f0f-1e46-4f82-9d18-d81b4803537d"


def test_client_403_with_fresh_key_is_an_error(monkeypatch):
    monkeypatch.setattr(rtv.time, "sleep", lambda s: None)
    client = rtv.RTVClient(request_delay=0, api_key="k", session=FakeSession([FakeResponse(403)]))
    client._key_fresh = True
    with pytest.raises(rtv.RTVForbiddenError):
        client.get("/iq-results/x/metrics")


# ── odds, tips, tipsters, race-day snapshots, profiles ────────────────

def test_parse_odds_one_row_per_runner_and_bookmaker():
    rows = pd.DataFrame(rtv.parse_odds(load("kempton_flat_bst_odds.json"), 608747))
    assert len(rows) == 22 and rows["runner_id"].nunique() == 2
    assert rows["bookmaker"].notna().all() and rows["price_decimal"].gt(1).all()
    assert set(rows["fluctuation"].dropna()) <= {"shortening", "drifting"}
    assert rows["each_way_places"].notna().any()


def test_parse_race_tips():
    rows = rtv.parse_race_tips(load("kempton_flat_bst_tips.json"), 608747)
    assert len(rows) == 1 and rows[0]["horse_name"] == "Pacific Mission"


def test_tipster_and_track_tables(tmp_path):
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    store.put("/tips/2025-08-13/tipsters", "tipsters", "2025-08-13", 200,
              json.dumps(load("tipsters_2025-08-13.json")).encode())
    tips = rtv._tipster_tips_table(store, None, None)
    assert len(tips) == 2 and tips["tipster"].iloc[0] == "Johnny Ward"
    assert {"tip_advised_price", "tip_each_way", "tip_result__state"} <= set(tips.columns)
    store.close()


def test_snapshot_tables_join_pace_maps_to_race(tmp_path):
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    paths = rtv.pre_race_paths("2026-10-30", "newmarket", "1135")
    for kind, name in (("pre_racecard", "racecard"), ("pre_pace_maps", "pace_maps"), ("pre_prompts", "prompts")):
        store.put_snapshot(paths[kind], kind, "2026-10-30", 200,
                           json.dumps(load(f"newmarket_20261030_1135_{name}.json")).encode(), "2026-10-29T20:00:00+00:00")
    t = rtv.build_snapshot_tables(store=store)
    pm = t["pace_maps"]
    assert len(pm) == 6 and (pm["race_id"] == 728672).all()
    assert pm["pm_pace_map__position"].dtype.kind == "f"
    assert set(t["pace_course_stats"]["stat_position_category"]) >= {"Held Up", "Led"}
    assert t["prompts"]["prompt"].str.contains("Dialstone").any()
    assert len(t["pre_runners"]) == 6
    store.close()


def test_profile_tables_mark_scrape_time_fields(tmp_path):
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    store.put("/horses/pacific-mission-united-kingdom", "horse", "2026-10-29", 200,
              json.dumps(load("horse_pacific_mission.json")).encode())
    t = rtv.build_profile_tables(store)
    p = t["horse_profiles"].iloc[0]
    assert p["scrape_horse_foaling_date"] == "2023-02-27" and p["scrape_horse_sire_name"] == "Lope De Vega"
    assert len(t["horse_performances"]) == 3
    assert not any("replay_videos" in c for c in t["horse_performances"].columns)
    store.close()


def test_discover_unlisted_races_from_results_db(tmp_path):
    import sqlite3
    db = tmp_path / "races.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE results (race_date TEXT, track TEXT, off_time TEXT)")
    con.executemany("INSERT INTO results VALUES (?,?,?)", [
        ("2021-07-10", "Newmarket", "12:30"), ("2021-07-10", "Newmarket", "13:05"),
        ("2021-07-10", "Kempton", "18:00"), ("2021-07-10", "Royal Ascot", "13:31")])
    con.commit(); con.close()
    listed = [{"track_name": "Newmarket", "track_slug": "newmarket", "off_time_utc": "13:05", "hhmm": "1405"},
              # Two minutes off the results DB time: the same race.
              {"track_name": "Ascot", "track_slug": "ascot", "off_time_utc": "13:30", "hhmm": "1430"}]
    found = rtv.discover_unlisted_races("2021-07-10", listed, db_path=str(db))
    got = {(r["track_slug"], r["hhmm"]) for r in found}
    assert got == {("newmarket", "1330"), ("kempton-park", "1900")}  # BST: UTC+1


def test_day_status_from_an_older_scraper_is_outdated(tmp_path):
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    store.mark_day("2025-08-13", "complete", 1, 1, 1)
    assert store.day_status("2025-08-13") == "complete"
    store.con.execute("UPDATE days SET version = 1")
    assert store.day_status("2025-08-13") == "outdated"
    store.close()


class CountingSession(FakeSession):
    def __init__(self):
        super().__init__([])
        import threading
        self.lock = threading.Lock()
        self.n = 0

    def get(self, url, headers=None, timeout=None):
        with self.lock:
            self.n += 1
        return FakeResponse(200, json.dumps({"url": url}).encode())


def test_run_tasks_concurrently_stores_every_answer(tmp_path):
    session = CountingSession()
    client = rtv.RTVClient(max_rps=float("inf"), api_key="k", session=session)
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    tasks = [rtv._Task(f"/results/x/{i}", f"/results/x/{i}", "results", "2025-08-13") for i in range(25)]
    statuses = rtv._run_tasks(client, store, tasks, workers=6)
    assert len(statuses) == 25 and session.n == 25 and client.n_requests == 25
    assert all(store.status(t.key) == 200 for t in tasks)
    store.close()


def test_rate_cap_spaces_requests():
    client = rtv.RTVClient(max_rps=50, api_key="k", session=CountingSession())
    import time as _t
    t0 = _t.monotonic()
    for _ in range(11):
        client.get("/x")
    assert _t.monotonic() - t0 >= 0.19  # 10 gaps of 1/50 s


def test_parse_race_time_whole_minutes():
    assert rtv.parse_race_time("4m") == 240.0


@pytest.mark.parametrize("a,b", [
    ("M Weatherer", "Mark Weatherer"), ("M & D Easterby", "Michael & David Easterby"),
    ("Sir Mark Prescott", "Sir Mark Prescott Bt"), ("Mrs A J Perrett", "Amanda Perrett"),
])
def test_person_key_matches_name_forms(a, b):
    assert rtv._person_key(a) == rtv._person_key(b)


def test_learn_glyphs_needs_agreeing_evidence(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setattr(rtv, "LEARNED_GLYPHS_FILE", str(tmp_path / "learned.json"))
    monkeypatch.setattr(rtv, "_learned_glyphs", None)
    db = tmp_path / "races.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE results (race_date TEXT, horse_name TEXT, headgear TEXT)")
    hood_visor = "[{'symbol': 'h', 'name': 'Hood', 'count': 2}, {'symbol': 'v', 'name': 'Visor', 'count': 1}]"
    con.executemany("INSERT INTO results VALUES (?,?,?)", [
        ("2025-01-01", "Alpha", hood_visor), ("2025-01-02", "Beta", hood_visor),
        ("2025-01-03", "Gamma", "[{'symbol': 'p', 'name': 'Cheek pieces', 'count': 1}]"),
        ("2025-01-04", "Delta", hood_visor)])
    con.commit(); con.close()
    store = rtv.RawStore(str(tmp_path / "raw.sqlite"))
    for d, horse, eq in (("2025-01-01", "Alpha", "Q"), ("2025-01-02", "Beta", "Q<sup>1</sup>"),
                         ("2025-01-03", "Gamma", "X"), ("2025-01-04", "Delta", "X")):
        body = {"race": {"id": 1, "runners": [{"horse_name": horse, "equipment_text": eq}]}}
        store.put(f"/results/{d}/t/1200", "results", d, 200, json.dumps(body).encode())
    out = rtv.learn_equipment_glyphs(store, db_path=str(db))
    assert out["learned"] == {"Q": {"symbols": ["h", "v"], "count": 2}}
    assert out["unresolved"]["X"]["seen_as"] == ["h,v", "p"]  # disagreeing evidence: not learned
    assert rtv.decode_equipment("Q")["symbols"] == "h,v"
    monkeypatch.setattr(rtv, "_learned_glyphs", None)
    store.close()


def test_placeholder_weight_is_missing():
    assert np.isnan(rtv.parse_weight_lbs("0-0"))


def test_full_backfill_checks_on_planted_values():
    t = parsed_tables("stratford_hurdle_gmt")
    t["races"].loc[0, "win_time_s"] = 120.0  # 2m 6f in 2 minutes: impossible
    t["races"]["win_speed_mph"] = t["races"]["distance_yards"] / t["races"]["win_time_s"] * 3600 / 1760
    t["runners"].loc[t["runners"].index[0], "weight_lbs"] = 131.0  # under the 10-0 jumps minimum
    flags = rtv.quality_checks(t)
    got = set(zip(flags["check"], flags["severity"]))
    assert ("win_time_implausible", "error") in got
    assert ("winner_time_mismatch", "error") in got
    assert ("weight_long_handicap", "info") in got


def test_arab_races_are_marked():
    races = rtv._derive_races(pd.DataFrame([
        {"res_title_full": "Emirates Breeders Handicap (Arab Race) (Class 1)"},
        {"res_title_full": "Carbonetzero Purebred Arabian Maiden Stakes"},
        {"res_title_full": "Hh Sheika Fatima Bint Mubarak Cup (Arab Group 2)"},
        {"res_title_full": "Rapido Racing LTD Conditions Arab Stakes"},
        {"res_title_full": "Arabian Horse Bahrain Equine Products UK Handicap Chase"},  # a sponsor
        {"res_title_full": "Unibet Novice Stakes (Class 5)"}]))
    assert races["is_arab_race"].tolist() == [True, True, True, True, False, False]
