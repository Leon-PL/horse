"""The batched build's direct store write, rating worker and copy-free helpers."""
import os
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from contextlib import closing

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import config
from src import store_build as batched
from src.data_processor import process_data
from src.established_form_features import _log1p
from src.feature_engineer import _with_columns, _without, add_race_relative_signal_features
from src.feature_store import FeatureStore
from src.parquet_io import RollingParquetWriter, combine_parquet_parts
from src.rtv_text import replaced_columns
from src.runner_identity import identity_columns, join_runner_columns
from tests.test_live_parity import _synthetic_raw


def state_differences(left, right, *, limit=20) -> list[str]:
    """Where two carried states differ. Python and numpy numbers may mix.

    Copied from the retired scripts/compare_feature_stores.py (2026-11-07)."""
    found: list[str] = []
    stack = [(left, right, "state")]
    while stack and len(found) < limit:
        a, b, at = stack.pop()
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                found.append(f"{at}: {len(a.keys() ^ b.keys())} keys in only one state")
            else:
                stack.extend((a[key], b[key], f"{at}[{key!r}]") for key in a)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                found.append(f"{at}: lengths {len(a)} and {len(b)}")
            else:
                stack.extend((x, y, f"{at}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            x, y = np.asarray(a), np.asarray(b)
            if not np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f"):
                found.append(f"{at}: arrays differ")
        elif a != b and not (a != a and b != b):
            found.append(f"{at}: {a!r} and {b!r}")
    return found


def test_frames_with_one_schema_stream_into_one_file(tmp_path):
    with RollingParquetWriter(tmp_path, batch_rows=2) as writer:
        writer.write(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", None, "z"]}))
        writer.write(pd.DataFrame({"a": [4.0], "b": ["y"]}, index=[9]))
    assert len(writer.paths) == 1 and writer.rows == 4
    table = pq.read_table(writer.paths[0])
    assert table.schema.metadata is None
    assert table.column_names == ["a", "b"]
    assert table["a"].to_pylist() == [1.0, 2.0, 3.0, 4.0]
    assert table["b"].to_pylist() == ["x", None, "z", "y"]
    groups = pq.ParquetFile(writer.paths[0]).metadata
    assert [groups.row_group(i).num_rows for i in range(groups.num_row_groups)] == [2, 1, 1]


def test_a_changed_schema_starts_a_file_that_reconciles_into_row_order(tmp_path):
    with RollingParquetWriter(tmp_path) as writer:
        writer.write(pd.DataFrame({"a": [1, 2]}))
        writer.write(pd.DataFrame({"a": [3.5], "extra": [7.0]}))
        writer.write(pd.DataFrame({"a": [4]}))
    assert len(writer.paths) == 3
    target = tmp_path / "whole.parquet"
    assert combine_parquet_parts(writer.paths, target) == 4
    table = pq.read_table(target)
    assert table["a"].to_pylist() == [1.0, 2.0, 3.5, 4.0]
    assert table["extra"].to_pylist() == [None, None, 7.0, None]


def test_one_streamed_file_becomes_the_store_as_it_stands(tmp_path):
    with RollingParquetWriter(tmp_path) as writer:
        writer.write(pd.DataFrame({"a": [1.0]}))
    target = tmp_path / "featured.parquet"
    assert batched._finish_featured(writer.paths, writer.rows, str(target)) == 1
    assert pq.read_table(target)["a"].to_pylist() == [1.0]
    assert not any("part_" in path.name for path in tmp_path.iterdir())


@pytest.mark.parametrize("name, value", [("DATA_DIR", "elsewhere"), ("TRUESKILL_C", -1.0)])
def test_settings_cover_every_config_value(monkeypatch, name, value):
    settings = batched._settings()
    assert batched._settings_match(settings)
    monkeypatch.setattr(config, name, value)
    assert not batched._settings_match(settings)


def test_the_build_stays_here_when_helpers_would_import_other_settings(monkeypatch):
    monkeypatch.setattr(config, "TRUESKILL_C", config.TRUESKILL_C + 1.0)
    with batched._helper_processes(True) as (rater, finisher):
        assert rater is None and finisher is None


def test_the_build_stays_here_when_no_helper_can_start(monkeypatch):
    class Broken:
        def __init__(self, **_kwargs):
            pass

        def submit(self, *_args):
            failed = Future()
            failed.set_exception(BrokenProcessPool("no helper"))
            return failed

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(batched, "ProcessPoolExecutor", Broken)
    with batched._helper_processes(True) as (rater, finisher):
        assert rater is None and finisher is None


def test_no_helper_is_started_unless_wanted(monkeypatch):
    monkeypatch.setattr(batched, "ProcessPoolExecutor", None)
    with batched._helper_processes(False) as (rater, finisher):
        assert rater is None and finisher is None


def test_second_half_state_merges_only_when_it_holds_other_entities():
    state = {"_tg_wins": {"trainer||going": 2.0}, "first": {"a": 1.0}}
    batched._add_second_half_state(state, {"_tg_wins": {"horse|track": 1.0}, "second": {"b": 1.0}})
    assert state == {"_tg_wins": {"trainer||going": 2.0, "horse|track": 1.0},
                     "first": {"a": 1.0}, "second": {"b": 1.0}}
    with pytest.raises(RuntimeError, match="same state: first"):
        batched._add_second_half_state(state, {"first": {"a": 2.0}})


def _history():
    processed = process_data(_synthetic_raw(n_days=24, races_per_day=3, field=6))
    processed["race_date"] = pd.to_datetime(processed["race_date"])
    return processed.sort_values("race_date", kind="stable").reset_index(drop=True)


def _rated(chunk, keyed):
    return join_runner_columns(chunk, keyed, [c for c in keyed if c not in identity_columns(keyed)])


def test_helpers_rate_and_finish_chunks_exactly_as_one_process_does(tmp_path):
    processed = _history()
    cut = processed["race_date"].iloc[len(processed) // 2]
    chunks = [processed[processed["race_date"] < cut], processed[processed["race_date"] >= cut]]
    store = FeatureStore.__new__(FeatureStore)

    alone = tmp_path / "alone"
    alone.mkdir()
    carried, counts = None, {}
    with closing(batched._StoreTail(str(alone), counts)) as tail:
        for chunk in chunks:
            keyed, _columns, carried = store._seeded_rating_frame(chunk, carried)
            tail.write(batched.engineer_features(_rated(chunk, keyed), save=False, skip_ratings=True,
                                                 counts_state=counts))
        paths, rows, _ = tail.close()
    expected = [pq.read_table(path) for path in paths]

    shared = tmp_path / "shared"
    shared.mkdir()
    first_half = {}
    with batched._helper_processes(True) as (rater, finisher):
        assert rater is not None and finisher is not None
        finisher.submit(batched._open_tail, str(shared)).result()
        jobs = [rater.submit(batched._rate_ahead, chunk) for chunk in chunks]
        for chunk, job in zip(chunks, jobs):
            partial = batched.engineer_features(_rated(chunk, job.result()), save=False, skip_ratings=True,
                                                counts_state=first_half, half="first")
            finisher.submit(batched._finish_chunk, partial).result()
        state_path = str(shared / "second_half_state.pkl")
        paths, shared_rows = finisher.submit(batched._close_tail, state_path).result()
        state, seconds = rater.submit(batched._worker_result).result()
    assert seconds > 0 and shared_rows == rows
    batched._add_second_half_state(first_half, batched._load_second_half_state(state_path))
    assert not os.path.exists(state_path)

    assert all(pq.read_table(path).equals(table) for path, table in zip(paths, expected))
    assert state_differences(first_half, counts) == []
    assert state_differences(state, carried) == []


def test_without_matches_drop_and_ignores_missing_columns():
    frame = pd.DataFrame({"a": [1.0], "_tmp": [2.0], "b": ["x"], "_other": [3]})
    expected = frame.drop(columns=["_tmp", "_other", "absent"], errors="ignore")
    pd.testing.assert_frame_equal(_without(frame.copy(), ["_tmp", "_other", "absent"]), expected)


def test_with_columns_matches_concat():
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=[4, 7])
    new = pd.DataFrame({"b": [3, 4], "c": ["x", "y"]}, index=[4, 7])
    pd.testing.assert_frame_equal(_with_columns(frame, new), pd.concat([frame, new], axis=1))


def test_replaced_columns_keeps_order_and_leaves_the_input_alone():
    frame = pd.DataFrame({"a": [1.0, 2.0], "stale": [8.0, 9.0], "b": [3.0, 4.0]})
    output = pd.DataFrame({"stale": [5.0, 6.0], "new": [7.0, 8.0]})
    columns = list(frame.columns)
    result = replaced_columns(frame, ["stale", "absent"], output)
    assert list(frame.columns) == columns
    pd.testing.assert_frame_equal(
        result, pd.concat([frame.drop(columns=["stale"]), output], axis=1))


def test_margin_over_the_best_rival_is_a_float_in_every_chunk():
    # Whole-number bases gave integers in some chunks and floats in others,
    # which made the chunks' parquet schemas differ for no reason.
    frame = pd.DataFrame({"race_id": [1, 1, 2, 2], "horse_prev_races": [3, 5, 4, 4]})
    out = add_race_relative_signal_features(frame)
    assert out["horse_prev_races_vs_best_in_race"].dtype == np.float64
    assert out["horse_prev_races_vs_best_in_race"].tolist() == [-2.0, 2.0, 0.0, 0.0]


@pytest.mark.parametrize("value", [0, 1, 7, 365, (1 << 15) - 1, 1 << 15, 40_000])
def test_log1p_lookup_is_numpy_log1p(value):
    assert _log1p(value) == float(np.log1p(value))
