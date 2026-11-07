"""Streaming refresh preserves native values and commits rows/state together."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import store_refresh
from src.feature_store import FeatureStore
from src.featured_dataset import _stamp, data_signature, fe_signature
from src.parquet_io import combine_parquet_parts
from tests.test_feature_store_transactions import simple_features, rows


@pytest.fixture(autouse=True)
def fixed_source_versions(monkeypatch):
    # Transaction fault injection must not read or race the live backfills.
    from src import featured_dataset
    for module in (featured_dataset, store_refresh):
        monkeypatch.setattr(module, "_atr_signature", lambda cutoff: {"cutoff": str(pd.Timestamp(cutoff).date()), "sig": "atr-fixture"})
        monkeypatch.setattr(module, "_rtv_signature", lambda cutoff: {"cutoff": str(pd.Timestamp(cutoff).date()), "sig": {"runs": "rtv-fixture"}})


def test_streamed_parquet_preserves_rows_missing_values_and_numeric_widening(tmp_path):
    first = pd.DataFrame({"id": [1, 2], "value": [np.nan, 4.0], "name": ["a", None]})
    second = pd.DataFrame({"id": [3.0], "name": ["b"], "later": [2.5]})
    paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for frame, path in zip([first, second], paths):
        frame.to_parquet(path, index=False)
    output = tmp_path / "combined.parquet"
    assert combine_parquet_parts(paths, output, batch_rows=1) == 3
    pd.testing.assert_frame_equal(pd.read_parquet(output), pd.concat([first, second], ignore_index=True))


@pytest.mark.parametrize("fail", [False, True])
def test_streaming_commit_and_retry_preserve_rows_and_carried_state(tmp_path, monkeypatch, simple_features, rows, fail):
    store = FeatureStore(str(tmp_path / "store"))
    history = rows.iloc[:2].copy()
    store.build(history)
    _stamp(store, fe_signature(), data_signature(history, cutoff=history.race_date.max()))
    original = {name: hashlib.sha256((Path(store.root) / name).read_bytes()).hexdigest()
                for name in store._FILES}
    combine = store_refresh.combine_parquet_parts
    if fail:
        def broken(paths, destination):
            Path(destination).write_bytes(b"partial")
            raise OSError("injected consolidation failure")
        monkeypatch.setattr(store_refresh, "combine_parquet_parts", broken)
        with pytest.raises(OSError, match="injected"):
            store_refresh.append_current_store(rows, store=store)
        assert original == {name: hashlib.sha256((Path(store.root) / name).read_bytes()).hexdigest()
                            for name in store._FILES}
        monkeypatch.setattr(store_refresh, "combine_parquet_parts", combine)
    monkeypatch.setattr(store, "load_featured", lambda *a, **k: pytest.fail("Historical features must not be materialised"))
    assert store_refresh.append_current_store(rows, store=store)
    actual = pd.read_parquet(store.featured_path)
    assert actual.history_count.tolist() == [0, 1, 2, 3]
    assert store.load_counts_state()["counter"]["rows"] == 4
    assert store.load_state()["elo"]["rows"] == 4
    assert store.meta()["featured_rows"] == store.meta()["processed_rows"] == 4
    assert store_refresh.append_current_store(rows, store=store)  # no double counting
    assert store.load_counts_state()["counter"]["rows"] == 4
