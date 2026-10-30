"""A field repair requires independent complete fields, and is reversible."""
import json
import sqlite3

import pandas as pd
import pytest

from scripts.reconcile_result_fields import apply_plan, repair_plan


def fields():
    base = {"race_id": "a", "race_date": "2025-11-24", "track": "Newcastle",
            "off_time": "14:50", "num_runners": 2, "weather_temp_max": 17}
    results = pd.DataFrame([
        {**base, "horse_name": "Winner", "finish_position": 1, "won": 1},
        {**base, "horse_name": "Faller", "finish_position": 0, "won": 0},
        {**base, "horse_name": "Withdrawn", "finish_position": 3, "won": 0},
    ])
    reference = results.iloc[:2].copy()
    reference["event_id"] = "event-a"
    return results, reference


@pytest.mark.parametrize("dnf_position", [0, 99])
def test_remove_only_extra_runner_while_retaining_actual_dnf(dnf_position):
    results, reference = fields()
    results.loc[1, "finish_position"] = dnf_position
    plan = repair_plan(results, reference)
    assert plan["repairable_races"] == 1
    assert plan["rows_to_remove"] == 1
    assert plan["repairs"][0]["remove"] == ["Withdrawn"]


@pytest.mark.parametrize("problem", ["field_size", "winner", "missing_runner", "event", "horse_collision"])
def test_ambiguous_or_incomplete_evidence_abstains(problem):
    results, reference = fields()
    if problem == "field_size":
        results["num_runners"] = 1
    elif problem == "winner":
        reference["won"] = [0, 1]
    elif problem == "missing_runner":
        reference.loc[1, "horse_name"] = "Different horse"
    elif problem == "event":
        reference["event_id"] = ["event-a", "event-b"]
    elif problem == "horse_collision":
        results.loc[2, "horse_name"] = "Winner!"
    plan = repair_plan(results, reference)
    assert plan["affected_races"] == 1
    assert plan["repairs"] == []
    assert sum(plan["skipped"].values()) == 1


def test_two_result_races_with_same_join_key_are_not_repaired():
    results, reference = fields()
    duplicate = results.copy()
    duplicate["race_id"] = "b"
    plan = repair_plan(pd.concat([results, duplicate], ignore_index=True), reference)
    assert plan["repairable_races"] == 0
    assert plan["skipped"] == {"no_unique_reference_race": 2}


def make_database(results):
    conn = sqlite3.connect(":memory:")
    results.to_sql("results", conn, index=False)
    return conn


def test_apply_backs_up_all_columns_and_preserves_remaining_rows(tmp_path):
    results, reference = fields()
    plan = repair_plan(results, reference)
    with make_database(results) as conn:
        backup = tmp_path / "original.json"
        assert apply_plan(conn, plan, backup) == 3
        saved = json.loads(backup.read_text())
        assert len(saved) == 3
        assert all(row["weather_temp_max"] == 17 for row in saved)
        actual = pd.read_sql_query("SELECT * FROM results", conn)
        assert actual.horse_name.tolist() == ["Winner", "Faller"]
        assert actual.weather_temp_max.eq(17).all()


def test_changed_race_aborts_before_backup_and_deletion(tmp_path):
    results, reference = fields()
    plan = repair_plan(results, reference)
    with make_database(results) as conn:
        conn.execute("UPDATE results SET num_runners = 3")
        conn.commit()
        backup = tmp_path / "original.json"
        with pytest.raises(RuntimeError, match="changed after planning"):
            apply_plan(conn, plan, backup)
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3
        assert not backup.exists()


def test_interrupted_delete_rolls_back_and_retains_backup(tmp_path):
    results, reference = fields()
    plan = repair_plan(results, reference)
    with make_database(results) as conn:
        conn.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON results "
                     "BEGIN SELECT RAISE(ABORT, 'injected deletion failure'); END")
        backup = tmp_path / "original.json"
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            apply_plan(conn, plan, backup)
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3
        assert len(json.loads(backup.read_text())) == 3


def test_backup_is_never_overwritten(tmp_path):
    results, reference = fields()
    backup = tmp_path / "original.json"
    backup.write_text("keep")
    with make_database(results) as conn:
        with pytest.raises(FileExistsError):
            apply_plan(conn, repair_plan(results, reference), backup)
        assert backup.read_text() == "keep"
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 3
