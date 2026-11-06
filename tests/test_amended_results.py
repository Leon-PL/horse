import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import database
from src.ingestion.sporting_life import SportingLifeScraper
from src.result_margins import cumulative_beaten_lengths


def payload():
    path = Path(__file__).parent / "fixtures/amended_goodwood_20230827.json"
    return json.loads(path.read_text(encoding="utf-8"))["payload"]


def parse(page, method="scrape_race_result"):
    scraper = SportingLifeScraper()
    scraper._fetch_next_data = lambda url: page
    return pd.DataFrame(getattr(scraper, method)("https://fixture.test/754950", "2023-08-27"))


def test_real_amended_result_uses_final_winner_and_keeps_physical_evidence_separate():
    frame = parse(payload()).set_index("horse_name")
    assert len(frame) == 5 and "Bless Him" not in frame.index
    assert frame.loc["The Gatekeeper", "finish_position"] == 1
    assert frame.loc["The Gatekeeper", "won"] == 1
    assert frame.loc["Thunder Ball", "finish_position"] == 2
    assert frame.loc["Thunder Ball", "won"] == 0
    assert frame.won.sum() == 1
    assert frame.num_runners.eq(5).all()
    assert frame.winning_time_secs.isna().all()
    assert frame.loc["The Gatekeeper", "lengths_behind"] == 0
    assert frame.drop(index="The Gatekeeper").lengths_behind.isna().all()
    margins = cumulative_beaten_lengths(frame.reset_index())
    assert margins.iloc[0] == 0 and margins.iloc[1:].isna().all()
    evidence = json.loads(frame.loc["The Gatekeeper", "result_amendment_json"])
    assert evidence["original_finish_position"] == 2
    assert evidence["amended_position"] == 1
    assert evidence["original_finish_distance"] == "nse"
    assert evidence["original_winning_time_secs"] == pytest.approx(102.74)


def test_missing_or_unchanged_amendments_preserve_original_measurements():
    page = payload()
    for ride in page["props"]["pageProps"]["race"]["rides"]:
        ride.pop("amended_position", None)
    first = parse(page)
    assert first.set_index("horse_name").loc["Thunder Ball", "won"] == 1
    np.testing.assert_allclose(first.winning_time_secs, 102.74, rtol=0, atol=1e-12)
    assert first.result_amendment_json.isna().all()
    for ride in page["props"]["pageProps"]["race"]["rides"]:
        if ride["finish_position"] > 0:
            ride["amended_position"] = ride["finish_position"]
    pd.testing.assert_frame_equal(parse(page), first, check_exact=True)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "unknown", float("nan"), float("inf"), 6])
def test_malformed_nonempty_amendment_cannot_silently_restore_wrong_winner(value):
    page = payload()
    page["props"]["pageProps"]["race"]["rides"][0]["amended_position"] = value
    with pytest.raises(ValueError, match="amended"):
        parse(page)


@pytest.mark.parametrize("value", [None, ""])
def test_absent_amendment_value_preserves_legacy_result_contract(value):
    page = payload()
    for ride in page["props"]["pageProps"]["race"]["rides"]:
        ride["amended_position"] = value
    frame = parse(page)
    assert frame.set_index("horse_name").loc["Thunder Ball", "won"] == 1


def test_string_integral_amendments_are_supported():
    page = payload()
    for ride in page["props"]["pageProps"]["race"]["rides"]:
        if "amended_position" in ride:
            ride["amended_position"] = str(ride["amended_position"])
    assert parse(page).set_index("horse_name").loc["The Gatekeeper", "won"] == 1


def test_pending_cards_ignore_result_amendments_and_clear_metadata():
    page = payload()
    page["props"]["pageProps"]["race"]["rides"][0]["amended_position"] = "invalid pending result"
    frame = parse(page, "scrape_racecard")
    assert frame.won.eq(0).all() and frame.finish_position.eq(0).all()
    assert frame.result_amendment_json.isna().all() and frame.lengths_behind.isna().all()


@pytest.mark.parametrize("stage", ["FINISHED", "RESULT", "STEWARDSENQUIRY", "GOINGDOWN", "UNRECOGNISED"])
def test_explicit_unsettled_stages_cannot_publish_a_provisional_winner(stage):
    page = payload()
    page["props"]["pageProps"]["race"]["race_summary"]["race_stage"] = stage
    assert parse(page).empty


def test_absent_legacy_stage_retains_its_result_contract():
    page = payload()
    page["props"]["pageProps"]["race"]["race_summary"].pop("race_stage")
    assert parse(page).set_index("horse_name").loc["The Gatekeeper", "won"] == 1


def test_real_amended_hurdle_result_preserves_nonfinishers_and_revises_six_placings():
    path = Path(__file__).parent / "fixtures/amended_doncaster_20201211.json"
    page = json.loads(path.read_text())["payload"]
    scraper = SportingLifeScraper()
    scraper._fetch_next_data = lambda url: page
    frame = pd.DataFrame(scraper.scrape_race_result("fixture", "2020-12-11")).set_index("horse_name")
    assert len(frame) == 8 and frame.won.sum() == 1
    assert frame.loc["Ecco", "finish_position"] == 1 and frame.loc["Eaton Hill", "finish_position"] == 6
    assert frame.loc[["Clarendon Street", "Doctor Dex"], "finish_position"].eq(0).all()
    assert frame.loc[["Clarendon Street", "Doctor Dex"], "finish_pos_label"].eq("PU").all()
    assert frame.loc[frame.won.eq(0), "lengths_behind"].isna().all()
    assert frame.winning_time_secs.isna().all() and frame.result_amendment_json.notna().all()


def test_context_migration_preserves_original_evidence_without_exposing_it_to_training(tmp_path, monkeypatch):
    path = tmp_path / "races.db"
    monkeypatch.setattr(database, "DB_PATH", str(path))
    # Existing038 table has eleven columns; the new optional field appends.
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE result_context_observations (
            race_id TEXT NOT NULL, horse_name TEXT NOT NULL, horse_id TEXT,
            race_date TEXT NOT NULL, declared_weight_lbs REAL,
            reported_jockey_claim_lbs REAL, weight_penalty_lbs REAL,
            ride_description TEXT, nonfinish_reason TEXT,
            content_sha256 TEXT NOT NULL, observed_at TEXT NOT NULL,
            PRIMARY KEY (race_id, horse_name, content_sha256))""")
    frame = parse(payload())
    assert database.insert_results(frame) == 5
    assert database.insert_results(frame) == 0
    core = database.load_from_database()
    assert "result_amendment_json" not in core
    assert core.set_index("horse_name").loc["The Gatekeeper", "won"] == 1
    source = database.load_result_context()
    assert len(source) == 5
    detail = json.loads(source.set_index("horse_name").loc["Thunder Ball", "result_amendment_json"])
    assert detail["original_finish_position"] == 1 and detail["amended_position"] == 2
    assert source.content_sha256.str.len().eq(64).all()


def test_optional_metadata_does_not_duplicate_unchanged_legacy_context(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "races.db"))
    database.init_db()
    frame = parse(payload()).iloc[[0]].copy()
    frame["result_amendment_json"] = None
    frame["ride_description"] = "Led, stayed on"
    fields = ("race_id", "horse_name", "horse_id", "race_date", "weight_lbs",
              "reported_jockey_claim_lbs", "weight_penalty_lbs", "ride_description", "nonfinish_reason")
    record = frame.reindex(columns=fields).to_dict("records")[0]
    record = {k: None if pd.isna(v) else v for k, v in record.items()}
    content = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    digest = hashlib.sha256(content.encode()).hexdigest()
    with database._get_conn() as connection:
        connection.execute("""INSERT INTO result_context_observations
            (race_id,horse_name,horse_id,race_date,declared_weight_lbs,
            reported_jockey_claim_lbs,weight_penalty_lbs,ride_description,nonfinish_reason,
            content_sha256,observed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            tuple(record[c] for c in fields) + (digest, "2026-10-29T10:00:00+00:00"))
        database._insert_result_context(connection, frame)
    assert len(database.load_result_context()) == 1
