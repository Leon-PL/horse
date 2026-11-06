"""The published results history: declared types and one missing marker, by every route."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import database
from src.database import (RESULTS_SNAPSHOT_COLUMNS, _export_results_snapshot, read_results_snapshot,
                          results_snapshot_frame)


def history(rows=2, **changes):
    example = {"text": "x", "whole": 1, "real": 1.5}
    frame = pd.DataFrame({name: [example[kind]] * rows for name, kind in RESULTS_SNAPSHOT_COLUMNS.items()})
    for name, values in changes.items():
        frame[name] = values
    return frame


def test_every_database_column_is_declared(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "races.db"))
    database.init_db()
    with sqlite3.connect(tmp_path / "races.db") as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(results)")]
    assert set(columns) - {"scraped_at"} == set(RESULTS_SNAPSHOT_COLUMNS)


def test_the_published_file_reads_back_as_the_frame_built_from_the_database(tmp_path):
    frame = history(3, race_id=["7", "8", "9"], going=["Good", "", None], age=[3, None, 5],
                    race_class=["", "4", None], draw=[1, 2, 3], sire=[None, None, None])
    path = tmp_path / "history.parquet"
    _export_results_snapshot(frame, path)
    pd.testing.assert_frame_equal(read_results_snapshot(path), results_snapshot_frame(frame))


def test_a_column_can_be_read_alone(tmp_path):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(going=["Good", ""]), path)
    assert list(read_results_snapshot(path, columns=["going"]).columns) == ["going"]


def test_numbers_are_typed_by_declaration_not_by_their_values(tmp_path):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(race_id=["7", "8"], race_class=["", "4"], age=[3, 5], form=["1", "2"]), path)
    back = read_results_snapshot(path)
    assert back.race_id.tolist() == [7, 8] and back.race_id.dtype == "int64"
    assert back.race_class.dtype == "float64" and back.race_class.isna().tolist() == [True, False]
    assert back.age.dtype == "int64"
    assert back.form.tolist() == ["1", "2"]


@pytest.mark.parametrize("blank", ["", None])
def test_missing_text_is_always_nan(tmp_path, blank):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(going=["Good", blank]), path)
    for frame in (read_results_snapshot(path), results_snapshot_frame(history(going=["Good", blank]))):
        missing = frame.going.iloc[1]
        assert isinstance(missing, float) and np.isnan(missing)


def test_a_missing_whole_number_stays_numeric(tmp_path):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(age=[3, None]), path)
    back = read_results_snapshot(path)
    assert back.age.dtype == "float64" and back.age.iloc[0] == 3 and np.isnan(back.age.iloc[1])


def test_text_in_a_number_column_stops_publication_and_keeps_the_last_file(tmp_path):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(), path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="race_class"):
        _export_results_snapshot(history(race_class=["4", "Listed"]), path)
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


def test_interrupted_publication_keeps_the_last_file(tmp_path, monkeypatch):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(), path)
    before = path.read_bytes()

    def fail_replace(*args):
        raise OSError("injected publish failure")

    monkeypatch.setattr(database.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        _export_results_snapshot(history(3), path)
    assert path.read_bytes() == before
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


@pytest.mark.parametrize("change", ["added", "removed"])
def test_a_column_that_is_not_declared_stops_publication(change):
    frame = history(unexpected=1) if change == "added" else history().drop(columns="going")
    with pytest.raises(ValueError, match="unexpected" if change == "added" else "going"):
        results_snapshot_frame(frame)


def test_a_long_decimal_keeps_every_bit_of_the_database_value(tmp_path):
    path = tmp_path / "history.parquet"
    _export_results_snapshot(history(weather_precip_prev3=[0.30000000000000004, np.nan]), path)
    back = read_results_snapshot(path).weather_precip_prev3
    assert back.iloc[0] == 0.30000000000000004 and np.isnan(back.iloc[1])
