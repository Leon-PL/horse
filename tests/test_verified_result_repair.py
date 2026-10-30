import sqlite3
import pandas as pd
import pytest

from src import database
from src.verified_result_repair import apply_verified_results


def case(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path/"results.db"))
    database.init_db()
    con = sqlite3.connect(database.DB_PATH)
    frame = pd.DataFrame({"race_id": ["1"]*3, "horse_name": ["A", "B", "Nonstarter"],
        "horse_id": ["1", "2", "3"], "race_date": ["2025-01-01"]*3, "num_runners": [2]*3,
        "finish_position": [1, 2, 0], "won": [1, 0, 0], "lengths_behind": [0., 1., None]})
    database._insert_results(con, frame)
    con.commit()
    before = pd.read_sql_query("SELECT * FROM results", con)
    after = before.iloc[:2].copy()
    after["finish_position"] = [2, 1]
    after["won"] = [0, 1]
    after["lengths_behind"] = [None, 0.]
    evidence = {"review_roster_sha256": "a"*64, "verification_sha256": "b"*64,
        "confirmed_nonstarters": [["1", "Nonstarter"]]}
    return con, before, after, evidence


def test_full_field_repair_removes_only_confirmed_nonstarters_and_is_idempotent(tmp_path, monkeypatch):
    con, before, after, evidence = case(tmp_path, monkeypatch)
    result = apply_verified_results(con, before, after, pd.DataFrame(), repair_id="test", evidence=evidence)
    assert result["removed_nonstarters"] == 1
    assert con.execute("SELECT horse_name FROM results WHERE won=1").fetchone()[0] == "B"
    assert con.execute("SELECT COUNT(*) FROM verified_result_rows").fetchone()[0] == 3
    assert apply_verified_results(con, before, after, pd.DataFrame(), repair_id="test", evidence=evidence)["already_applied"]
    con.close()


@pytest.mark.parametrize("fault", ["changed_source", "unapproved_removal", "unrelated_edit", "bad_field"])
def test_bad_repair_cannot_partially_change_source(tmp_path, monkeypatch, fault):
    con, before, after, evidence = case(tmp_path, monkeypatch)
    if fault == "changed_source":
        con.execute("UPDATE results SET odds=999 WHERE horse_name='A'")
        con.commit()
    elif fault == "unapproved_removal":
        evidence["confirmed_nonstarters"] = []
    elif fault == "unrelated_edit":
        after.loc[0, "age"] = 20
    else:
        after["num_runners"] = 99
    original = pd.read_sql_query("SELECT * FROM results", con)
    with pytest.raises((AssertionError, ValueError)):
        apply_verified_results(con, before, after, pd.DataFrame(), repair_id="test", evidence=evidence)
    pd.testing.assert_frame_equal(pd.read_sql_query("SELECT * FROM results", con), original)
    con.close()


def test_reviewed_insert_timestamp_and_caller_rollback(tmp_path, monkeypatch):
    con, before, after, evidence = case(tmp_path, monkeypatch)
    extra = pd.DataFrame({"race_id": ["2"], "horse_name": ["New"], "horse_id": ["4"],
        "race_date": ["2025-01-02"], "num_runners": [1], "finish_position": [1],
        "won": [1], "scraped_at": ["2026-10-30 10:29:00"]})
    con.execute("BEGIN IMMEDIATE")
    apply_verified_results(con, before, after, extra, repair_id="rollback", evidence=evidence,
                           manage_transaction=False)
    assert con.execute("SELECT scraped_at FROM results WHERE race_id='2'").fetchone()[0] == extra.scraped_at.iloc[0]
    con.rollback()
    pd.testing.assert_frame_equal(pd.read_sql_query("SELECT * FROM results", con), before)
    result = apply_verified_results(con, before, after, extra, repair_id="commit", evidence=evidence)
    assert result["new_starters"] == 1
    assert apply_verified_results(con, before, after, extra, repair_id="commit", evidence=evidence)["already_applied"]
    con.close()
