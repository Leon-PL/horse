import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.result_amendment_repair import apply_amendments, corrected_rows


def source():
    before = pd.DataFrame({"race_id": ["one"] * 3, "horse_id": ["101", "102", "103"],
        "horse_name": ["First", "Second", "Third"], "race_date": ["2025-01-01"] * 3,
        "track": ["Kempton"] * 3, "off_time": ["14:00"] * 3, "num_runners": [3] * 3,
        "finish_position": [1, 2, 3], "finish_pos_label": [None] * 3,
        "won": [1, 0, 0], "lengths_behind": [0., .1, 2.], "official_rating": [80, 81, 82],
        "weather_temp_max": [9.] * 3, "scraped_at": ["2025-01-02"] * 3})
    proposed = before.copy()
    proposed["horse_name"] = proposed.horse_name.str.upper()
    proposed["source_position"] = [2., 1., 3.]
    proposed["exchange_won"] = [0., 1., 0.]
    proposed["atr_raceid"] = "source-one"
    return before, proposed


def test_corrected_official_labels_withhold_arrival_margins_and_preserve_other_values():
    before, proposed = source()
    # The DB can carry a declared count, while processing has the started count.
    before["num_runners"] = 4
    after = corrected_rows(before.sample(frac=1), proposed.sample(frac=1))
    assert after.finish_position.tolist() == [2, 1, 3]
    assert after.won.tolist() == [0, 1, 0]
    assert after.lengths_behind.tolist()[1] == 0
    assert after.lengths_behind.iloc[[0, 2]].isna().all()
    pd.testing.assert_frame_equal(after.drop(columns=["finish_position", "finish_pos_label", "won", "lengths_behind"]),
                                before.drop(columns=["finish_position", "finish_pos_label", "won", "lengths_behind"]))


def test_transaction_preserves_original_rows_and_repeated_application_is_a_verified_noop():
    before, proposed = source()
    with sqlite3.connect(":memory:") as connection:
        before.to_sql("results", connection, index=False)
        first = apply_amendments(connection, before, proposed, repair_id="test", evidence={"source": "fixed"})
        assert first["updated_rows"] == 3
        originals = connection.execute("SELECT original_row_json FROM result_amendment_rows ORDER BY horse_name").fetchall()
        assert len(originals) == 3 and json.loads(originals[0][0])["won"] == 1
        changes = connection.total_changes
        second = apply_amendments(connection, before, proposed, repair_id="test", evidence={"source": "fixed"})
        assert second["already_applied"] and connection.total_changes == changes
        with pytest.raises(ValueError, match="different evidence"):
            apply_amendments(connection, before, proposed, repair_id="test", evidence={"source": "changed"})


@pytest.mark.parametrize("mutation", ["winner", "partial", "extra", "enrichment"])
def test_changed_current_field_aborts_all_updates(mutation):
    before, proposed = source()
    with sqlite3.connect(":memory:") as connection:
        before.to_sql("results", connection, index=False)
        if mutation == "winner":
            connection.execute("UPDATE results SET won=0 WHERE horse_id='101'")
        elif mutation == "partial":
            connection.execute("DELETE FROM results WHERE horse_id='103'")
        elif mutation == "extra":
            connection.execute("INSERT INTO results SELECT race_id,'104','Fourth',race_date,track,off_time,num_runners,4,'',0,1,70,9,scraped_at FROM results LIMIT 1")
        else:
            connection.execute("UPDATE results SET weather_temp_max=11")
        connection.commit()
        snapshot = pd.read_sql_query("SELECT * FROM results", connection)
        with pytest.raises((ValueError, AssertionError)):
            apply_amendments(connection, before, proposed, repair_id="test", evidence={})
        pd.testing.assert_frame_equal(snapshot, pd.read_sql_query("SELECT * FROM results", connection))
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='result_amendment_batches'").fetchall()


@pytest.mark.parametrize("column,value", [("source_position", [1, 1, 3]), ("source_position", [2, 1, 0]),
    ("exchange_won", [1, 0, 0]), ("exchange_won", [np.nan, 1, 0]),
    ("atr_raceid", ["one", "two", "one"]), ("horse_name", ["Wrong", "Second", "Third"]),
    ("num_runners", [4, 4, 4]), ("horse_id", [101, 101, 103])])
def test_unreliable_source_cannot_enter_automatic_batch(column, value):
    before, proposed = source()
    proposed[column] = value
    with pytest.raises((ValueError, AssertionError)):
        corrected_rows(before, proposed)


def test_second_field_failure_rolls_back_the_first_and_its_evidence():
    before, proposed = source()
    other = before.assign(race_id="two")
    before = pd.concat([before, other], ignore_index=True)
    proposed = pd.concat([proposed, proposed.assign(race_id="two")], ignore_index=True)
    with sqlite3.connect(":memory:") as connection:
        before.to_sql("results", connection, index=False)
        connection.executescript("""CREATE TRIGGER fail_second BEFORE UPDATE ON results
            WHEN NEW.race_id='two' BEGIN SELECT RAISE(ABORT, 'injected failure'); END;""")
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            apply_amendments(connection, before, proposed, repair_id="test", evidence={})
        pd.testing.assert_frame_equal(before, pd.read_sql_query("SELECT * FROM results", connection))
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='result_amendment_rows'").fetchall()


def test_original_evidence_preserves_full_float_precision():
    before, proposed = source()
    value = np.nextafter(9., 10.)
    before["weather_temp_max"] = value
    with sqlite3.connect(":memory:") as connection:
        before.to_sql("results", connection, index=False)
        apply_amendments(connection, before, proposed, repair_id="precision", evidence={})
        original = json.loads(connection.execute("SELECT original_row_json FROM result_amendment_rows LIMIT 1").fetchone()[0])
        assert original["weather_temp_max"] == value


def official_source():
    before, proposed = source()
    before.loc[2, ["finish_position", "finish_pos_label"]] = [0, "DNF"]
    proposed.loc[2, "finish_position"] = 0
    proposed["source_position"] = [2, 1, 0]
    proposed["source_provider"] = "sporting_life_amendment"
    proposed["source_event_id"] = "one"
    proposed["source_stage"] = "WEIGHEDIN"
    proposed["source_receipt_sha256"] = "a" * 64
    proposed["source_observed_at"] = "2026-10-29T22:00:00+00:00"
    proposed["source_original_finish_position"] = [1, 2, 0]
    proposed["source_finish_pos_label"] = ["", "", "PU"]
    proposed["result_amendment_json"] = "{}"
    return before, proposed


def test_observed_official_amendment_can_include_genuine_nonfinishers():
    before, proposed = official_source()
    after = corrected_rows(before, proposed)
    assert after.finish_position.tolist() == [2, 1, 0]
    assert after.finish_pos_label.tolist() == ["", "", "PU"]
    assert after.won.tolist() == [0, 1, 0]
    assert after.lengths_behind.iloc[[0, 2]].isna().all()


@pytest.mark.parametrize("column,value", [("source_stage", "RESULT"), ("source_event_id", "wrong"),
    ("source_receipt_sha256", "missing"), ("source_observed_at", "invalid"),
    ("source_original_finish_position", 4), ("result_amendment_json", None), ("source_provider", "unreviewed")])
def test_direct_official_repair_rejects_missing_or_conflicting_evidence(column, value):
    before, proposed = official_source()
    proposed[column] = value
    with pytest.raises((ValueError, AssertionError)):
        corrected_rows(before, proposed)


@pytest.mark.parametrize("final_position,accepted", [(4, True), (3, False), (5, False)])
def test_explicit_disqualified_winner_can_be_last_of_all_starters(final_position, accepted):
    before, proposed = official_source()
    before = pd.concat([before, before.iloc[[2]].assign(horse_id="104", horse_name="Fourth")], ignore_index=True)
    proposed = pd.concat([proposed, proposed.iloc[[2]].assign(horse_id="104", horse_name="Fourth")], ignore_index=True)
    before["num_runners"] = proposed["num_runners"] = 4
    proposed.loc[0, "source_position"] = final_position
    if accepted:
        after = corrected_rows(before, proposed)
        assert after.finish_position.tolist() == [4, 1, 0, 0]
        assert after.won.tolist() == [0, 1, 0, 0]
    else:
        with pytest.raises(ValueError, match="finishing order"):
            corrected_rows(before, proposed)


def test_a_reported_nonfinisher_without_a_specific_reason_stays_generic():
    before, proposed = official_source()
    proposed.loc[2, "source_finish_pos_label"] = ""
    assert corrected_rows(before, proposed).loc[2, "finish_pos_label"] == "DNF"
    proposed.loc[2, "source_finish_pos_label"] = "NR"
    with pytest.raises(ValueError, match="nonstarter"):
        corrected_rows(before, proposed)


def test_caller_can_rollback_the_entire_uncommitted_correction_and_ledger():
    before, proposed = source()
    with sqlite3.connect(":memory:") as connection:
        before.to_sql("results", connection, index=False)
        connection.execute("BEGIN IMMEDIATE")
        apply_amendments(connection, before, proposed, repair_id="outer", evidence={}, manage_transaction=False)
        assert connection.in_transaction
        assert connection.execute("SELECT won FROM results WHERE horse_id='102'").fetchone()[0] == 1
        connection.rollback()
        pd.testing.assert_frame_equal(before, pd.read_sql_query("SELECT * FROM results", connection))
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='result_amendment_rows'").fetchall()
        with pytest.raises(ValueError, match="active transaction"):
            apply_amendments(connection, before, proposed, repair_id="outer", evidence={}, manage_transaction=False)


def test_caller_can_commit_a_verified_amendment_together_with_other_publication_state():
    before, proposed = source()
    with sqlite3.connect(":memory:") as connection:
        before.to_sql("results", connection, index=False)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            apply_amendments(connection, before, proposed, repair_id="outer", evidence={}, manage_transaction=False)
            connection.execute("CREATE TABLE publication (state TEXT)")
            connection.execute("INSERT INTO publication VALUES ('ready')")
        assert connection.execute("SELECT won FROM results WHERE horse_id='102'").fetchone()[0] == 1
        assert connection.execute("SELECT state FROM publication").fetchone()[0] == "ready"
