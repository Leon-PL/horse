"""Fault injection: rows and carried history must advance as one generation."""

import json
import os
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src import feature_store
from src.feature_store import FeatureStore


@pytest.fixture
def simple_features(monkeypatch):
    """Cheap stateful FE: retries expose double-counting as wrong feature values."""
    def engineer(frame, *, counts_state, **kwargs):
        counter = counts_state.setdefault("counter", {"rows": 0})
        start = counter["rows"]
        counter["rows"] += len(frame)
        return frame.assign(history_count=range(start, start + len(frame)))

    def ratings(self, frame, seed=None):
        count = (seed or {}).get("elo", {}).get("rows", 0) + len(frame)
        return {"elo": {"rows": count}, "glicko": {"rows": count}}

    def keyed(self, frame, seed):
        return frame[["race_id", "horse_name"]].copy(), [], ratings(self, frame, seed)

    monkeypatch.setattr(feature_store, "engineer_features", engineer)
    monkeypatch.setattr(FeatureStore, "_compute_ratings_state", ratings)
    monkeypatch.setattr(FeatureStore, "_seeded_rating_frame", keyed)


@pytest.fixture
def rows():
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-02"] * 2),
        "race_id": [1, 1, 2, 2],
        "horse_name": ["A", "B", "A", "B"],
    })


@pytest.fixture
def store(tmp_path, simple_features, rows):
    result = FeatureStore(str(tmp_path / "store"))
    result.build(rows.iloc[:2])
    return result


def _snapshot(store):
    return {name: (Path(store.root) / name).read_bytes() for name in store._FILES}


def _assert_complete(store, rows):
    pd.testing.assert_frame_equal(store.load_processed(), rows)
    assert store.load_featured()["history_count"].tolist() == list(range(len(rows)))
    assert store.load_counts_state() == {"counter": {"rows": len(rows)}}
    assert store.load_state() == {"elo": {"rows": len(rows)}, "glicko": {"rows": len(rows)}}
    assert store.meta()["featured_rows"] == len(rows)
    assert store.meta()["processed_rows"] == len(rows)
    assert store.last_date() == rows["race_date"].max()
    assert not Path(store._journal_path).exists()
    assert not list(Path(store.root).glob(".commit-*"))


def _fail_serialization(monkeypatch, filename):
    """Fail after some bytes are written, not just before opening the output."""
    if filename.endswith(".parquet"):
        original = pd.DataFrame.to_parquet

        def broken(frame, path, *args, **kwargs):
            result = original(frame, path, *args, **kwargs)
            if Path(path).name == filename:
                raise OSError("injected serialization failure")
            return result

        monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)
    else:
        module = pickle if filename.endswith(".pkl") else json
        original = module.dump

        def broken(obj, fh, *args, **kwargs):
            result = original(obj, fh, *args, **kwargs)
            if Path(fh.name).name == filename:
                raise OSError("injected serialization failure")
            return result

        monkeypatch.setattr(module, "dump", broken)


@pytest.mark.parametrize("operation", ["build", "append"])
@pytest.mark.parametrize("filename", FeatureStore._FILES)
def test_failed_write_preserves_previous_generation_and_retry(
    store, rows, monkeypatch, operation, filename,
):
    original = _snapshot(store)
    incoming = rows if operation == "build" else rows.iloc[2:]
    with monkeypatch.context() as patch:
        _fail_serialization(patch, filename)
        with pytest.raises(OSError, match="injected serialization failure"):
            getattr(store, operation)(incoming)
    assert _snapshot(store) == original
    _assert_complete(store, rows.iloc[:2])

    getattr(store, operation)(incoming)
    _assert_complete(store, rows)
    assert store.append(rows.iloc[2:]).empty
    _assert_complete(store, rows)


def test_failed_first_build_can_be_retried(tmp_path, simple_features, rows, monkeypatch):
    store = FeatureStore(str(tmp_path))
    with monkeypatch.context() as patch:
        _fail_serialization(patch, "counts_state.pkl")
        with pytest.raises(OSError, match="injected serialization failure"):
            store.build(rows)
    assert not store.exists()
    assert store.load_counts_state() is None
    store.build(rows)
    _assert_complete(store, rows)


@pytest.mark.parametrize("operation", ["build", "append"])
@pytest.mark.parametrize("stop_after", range(6))
def test_interrupted_commit_recovers_once_on_reopen(
    store, rows, monkeypatch, operation, stop_after,
):
    """Interrupt just after the journal or any of the five atomic replacements."""
    original_replace = os.replace
    replacements = 0

    def interrupted(source, target):
        nonlocal replacements
        original_replace(source, target)
        if Path(target).parent == Path(store.root):
            current = replacements
            replacements += 1
            if current == stop_after:
                raise KeyboardInterrupt("simulated process interruption")

    incoming = rows if operation == "build" else rows.iloc[2:]
    with monkeypatch.context() as patch:
        patch.setattr(feature_store.os, "replace", interrupted)
        with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
            getattr(store, operation)(incoming)
    assert Path(store._journal_path).exists()

    reopened = FeatureStore(store.root)
    # Even a metadata/status read must recover before exposing a generation.
    assert reopened.meta()["processed_rows"] == len(rows)
    assert reopened.append(rows.iloc[2:]).empty
    _assert_complete(reopened, rows)


def test_blocked_commit_never_exposes_partial_generation(store, rows, monkeypatch):
    original_replace = os.replace

    def blocked(source, target):
        if Path(target) == Path(store.processed_path):
            raise PermissionError("injected locked file")
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(feature_store.os, "replace", blocked)
        with pytest.raises(PermissionError, match="injected locked file"):
            store.append(rows.iloc[2:])
        reopened = FeatureStore(store.root)
        for read in (reopened.exists, reopened.meta, reopened.load_featured,
                     reopened.load_processed, reopened.load_state, reopened.load_counts_state):
            with pytest.raises(PermissionError, match="injected locked file"):
                read()
    assert store.append(rows.iloc[2:]).empty
    _assert_complete(store, rows)


def test_second_store_cannot_access_active_writer(store):
    other = FeatureStore(store.root)
    with store._access():
        # Calls on the owning instance are reentrant.
        assert store.exists()
        with pytest.raises(RuntimeError, match="in use by another operation"):
            other.load_counts_state()
    assert other.exists()


def test_incomplete_staging_cannot_replace_store(store):
    original = _snapshot(store)
    with pytest.raises(FileNotFoundError):
        with store._staged_update() as pending:
            pending._save_counts_state({"counter": {"rows": 999}})
    assert _snapshot(store) == original


def test_batched_rebuild_failure_preserves_store(store, rows, monkeypatch):
    from scripts import build_feature_store_batched as batched

    original = _snapshot(store)
    monkeypatch.setattr(batched, "FeatureStore", lambda: store)
    monkeypatch.setattr(batched, "process_data", lambda **kwargs: rows.copy())
    monkeypatch.setattr(batched, "engineer_features", feature_store.engineer_features)
    monkeypatch.setattr(batched, "fe_signature", lambda: "current-fe")
    monkeypatch.setattr(batched, "data_signature", lambda *args, **kwargs: "current-data")
    monkeypatch.setattr(batched, "atr_cache_signature", lambda cutoff: "current-atr")
    monkeypatch.setattr("sys.argv", ["build_feature_store_batched.py", "--chunk-rows", "2"])

    with monkeypatch.context() as patch:
        def broken(*args, **kwargs):
            raise OSError("injected consolidation failure")
        patch.setattr(batched, "_finish_featured", broken)
        with pytest.raises(OSError, match="injected consolidation failure"):
            batched.main()
    assert _snapshot(store) == original
    batched.main()
    _assert_complete(store, rows)
    assert store.meta()["fe_signature"] == "current-fe"
    assert store.meta()["data_signature"] == "current-data"


def test_batched_rebuild_can_use_separate_store(store, rows, monkeypatch, tmp_path):
    from scripts import build_feature_store_batched as batched

    original = _snapshot(store)
    destination = tmp_path / "candidate"
    monkeypatch.setattr(batched, "process_data", lambda **kwargs: rows.copy())
    monkeypatch.setattr(batched, "engineer_features", feature_store.engineer_features)
    monkeypatch.setattr(batched, "fe_signature", lambda: "candidate-fe")
    monkeypatch.setattr(batched, "data_signature", lambda *args, **kwargs: "same-data")
    monkeypatch.setattr(batched, "atr_cache_signature", lambda cutoff: "same-atr")
    monkeypatch.setattr("sys.argv", ["build_feature_store_batched.py", "--chunk-rows", "2", "--store-root", str(destination)])
    batched.main()
    assert _snapshot(store) == original
    candidate = FeatureStore(str(destination))
    _assert_complete(candidate, rows)
    assert candidate.meta()["fe_signature"] == "candidate-fe"


def test_batched_rebuild_refuses_to_stamp_code_that_changed_mid_build(store, rows, monkeypatch):
    """The store in use on 2026-10-12 was stamped after FE files were edited
    mid-build, so its signature matched neither generation (ledger 12)."""
    from scripts import build_feature_store_batched as batched

    original = _snapshot(store)
    signatures = iter(["code-at-start", "code-after-an-edit"])
    monkeypatch.setattr(batched, "FeatureStore", lambda: store)
    monkeypatch.setattr(batched, "process_data", lambda **kwargs: rows.copy())
    monkeypatch.setattr(batched, "engineer_features", feature_store.engineer_features)
    monkeypatch.setattr(batched, "fe_signature", lambda: next(signatures))
    monkeypatch.setattr(batched, "data_signature", lambda *args, **kwargs: "current-data")
    monkeypatch.setattr(batched, "atr_cache_signature", lambda cutoff: "current-atr")
    monkeypatch.setattr("sys.argv", ["build_feature_store_batched.py", "--chunk-rows", "2"])

    with pytest.raises(RuntimeError, match="changed while the store was building"):
        batched.main()
    assert _snapshot(store) == original


@pytest.fixture
def dataset(store, rows, monkeypatch):
    from src import featured_dataset

    monkeypatch.setattr(featured_dataset, "FeatureStore", lambda: store)
    monkeypatch.setattr(featured_dataset, "fe_signature", lambda: "test-fe")
    monkeypatch.setattr(featured_dataset, "_atr_signature", lambda cutoff: {
        "cutoff": str(pd.Timestamp(cutoff).date()), "sig": "test-atr",
    })
    featured_dataset._stamp(store, "test-fe", featured_dataset.data_signature(rows.iloc[:2]))
    return featured_dataset


@pytest.mark.parametrize("rebuild", [False, True])
def test_dataset_lock_spans_decision_commit_stamp_and_load(
    dataset, store, rows, monkeypatch, rebuild,
):
    """A second writer cannot replace the generation between public methods."""
    competing = FeatureStore(store.root)
    checkpoints = []

    def check_competing_writer(phase):
        checkpoints.append(phase)
        with pytest.raises(RuntimeError, match="in use by another operation"):
            competing.append(rows)

    def signature():
        check_competing_writer("decision")
        return "test-fe"

    original_data_signature = dataset.data_signature
    original_stamp = dataset._stamp

    def data_signature(*args, **kwargs):
        check_competing_writer("source-signature")
        return original_data_signature(*args, **kwargs)

    def stamp(*args, **kwargs):
        check_competing_writer("after-row-commit")
        return original_stamp(*args, **kwargs)

    def columns(*args, **kwargs):
        check_competing_writer("before-final-load")
        return None

    monkeypatch.setattr(dataset, "fe_signature", signature)
    monkeypatch.setattr(dataset, "data_signature", data_signature)
    monkeypatch.setattr(dataset, "_stamp", stamp)
    monkeypatch.setattr(dataset, "_model_load_columns", columns)
    featured = dataset.get_featured_dataset(processed=rows, rebuild=rebuild)

    assert featured["history_count"].tolist() == list(range(len(rows)))
    assert {"decision", "source-signature", "after-row-commit"} <= set(checkpoints)
    if not rebuild:
        assert "before-final-load" in checkpoints
    _assert_complete(store, rows)
    assert store.meta()["data_signature"] == original_data_signature(rows)
    assert competing.append(rows).empty  # lock is released after the whole operation


@pytest.mark.parametrize("failure", ["serialization", "flush", "replace"])
def test_failed_signature_stamp_keeps_complete_metadata(
    dataset, store, rows, monkeypatch, failure,
):
    original = _snapshot(store)

    def broken_dump(obj, fh, **kwargs):
        fh.write('{"incomplete":')
        raise OSError("injected stamp failure")

    def broken_operation(*args, **kwargs):
        raise OSError("injected stamp failure")

    with monkeypatch.context() as patch:
        if failure == "serialization":
            patch.setattr(dataset.json, "dump", broken_dump)
        else:
            patch.setattr(dataset.os, "fsync" if failure == "flush" else "replace", broken_operation)
        with pytest.raises(OSError, match="injected stamp failure"):
            dataset._stamp(store, "new-fe", "new-data")
    assert _snapshot(store) == original
    assert not list(Path(store.root).glob(".meta-*.json"))
    _assert_complete(store, rows.iloc[:2])
    dataset._stamp(store, "new-fe", "new-data")
    assert store.meta()["fe_signature"] == "new-fe"
    assert store.meta()["data_signature"] == "new-data"
    assert store.meta()["last_date"] == "2026-01-01"


def test_signature_stamp_holds_lock_while_replacing_metadata(dataset, store, monkeypatch):
    competing = FeatureStore(store.root)
    original_replace = os.replace
    checked = []

    def replace(source, target):
        with pytest.raises(RuntimeError, match="in use by another operation"):
            competing.load_counts_state()
        checked.append(target)
        return original_replace(source, target)

    monkeypatch.setattr(dataset.os, "replace", replace)
    dataset._stamp(store, "next-fe", "next-data")
    assert checked == [store.meta_path]
