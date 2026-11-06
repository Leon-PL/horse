import pandas as pd

from src import database
from src.ingestion import sporting_life as data_scraper
from src.ingestion.sporting_life import SportingLifeScraper, _verdict_text

VERDICT = "<p><b>PACHMENA</b> will find this easier &amp; is preferred to <b>Magnetic North</b>.</p>"


def ride(name="A", commentary="Second on hurdles debut.", *, result=True, verdict=VERDICT):
    return SportingLifeScraper()._ride_to_dict({"horse": {"name": name}, "handicap": "11-0",
        "commentary": commentary, "finish_position": 1 if name == "A" else 2}, race_id="r",
        date_str="2026-01-01", off_time="12:00", course_name="Test", region="UK", race_name="Test Hurdle",
        race_class="Class3", race_type="Hurdle", distance_f=16., going="Good", prize_money=1000.,
        num_runners=2, surface_str="Turf", has_handicap=False, is_result=result,
        verdict=_verdict_text({"verdict": verdict}))


def test_verdict_is_plain_text_and_card_and_result_both_carry_commentary():
    assert _verdict_text({"verdict": VERDICT}) == "PACHMENA will find this easier & is preferred to Magnetic North."
    assert _verdict_text({"verdict": "<p>One.</p><p>Two.</p>"}) == "One. Two."
    assert _verdict_text({"verdict": "<p> </p>"}) is _verdict_text({}) is None
    for result in (True, False):
        row = ride(result=result)
        assert row["sl_commentary"] == "Second on hurdles debut."
        assert row["sl_verdict"].startswith("PACHMENA")


def test_text_is_appended_per_phase_and_kept_out_of_results(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "races.db"))
    frame = pd.DataFrame([ride(), ride("B", None)])
    assert database.insert_results(frame) == 2
    database.insert_results(frame)                       # unchanged text: no new rows
    database.record_runner_text(frame, "racecard")
    with database._get_conn() as conn:
        database._replace_race_results(conn, frame.assign(sl_commentary="Rewritten after the race."))
    seen = database.load_runner_text()
    assert seen.groupby("phase").size().to_dict() == {"racecard": 2, "result": 4}
    assert seen.loc[seen.horse_name.eq("B") & seen.phase.eq("racecard"), "commentary"].isna().all()
    assert set(database.load_runner_text(phase="result").commentary.dropna()) == {
        "Second on hurdles debut.", "Rewritten after the race."}
    core = database.load_from_database()
    assert not {"sl_commentary", "sl_verdict", "commentary", "verdict"} & set(core.columns)


def test_fresh_racecards_record_text_but_cache_cards_without_it(tmp_path, monkeypatch):
    card = pd.DataFrame([ride(result=False)])
    recorded, cached = [], []
    monkeypatch.setattr(data_scraper, "load_cached_racecards", lambda date_str=None: None)
    monkeypatch.setattr(data_scraper, "scrape_todays_racecards", lambda **_: card)
    monkeypatch.setattr(database, "record_runner_text", lambda frame, phase: recorded.append((phase, frame.copy())))
    monkeypatch.setattr(data_scraper, "save_racecards_cache", lambda frame, date_str=None: cached.append(frame))
    monkeypatch.setattr(data_scraper, "enrich_racecard_pedigree", lambda frame: frame)
    out = data_scraper.get_scraped_racecards(date_str="2026-01-01")
    assert recorded[0][0] == "racecard" and recorded[0][1].sl_commentary.notna().all()
    assert "sl_commentary" not in cached[0] and "sl_commentary" not in out
