"""State ownership and exact incremental parity for live FE optimisations."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import feature_engineer as fe, feature_store
from src.feature_store import FeatureStore


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("initial", [{}, {"counter": {"rows": 7}}])
def test_live_state_is_loaded_once_and_never_persisted(tmp_path, monkeypatch, fail, initial):
    store = FeatureStore(str(tmp_path / "store"))
    card = pd.DataFrame({"race_id": ["r1"], "horse_name": ["A"],
                         "race_date": pd.to_datetime(["2026-01-02"])})
    card.to_parquet(store.featured_path, index=False)
    card.to_parquet(store.processed_path, index=False)
    Path(store.meta_path).write_text("{}", encoding="utf-8")
    store._save_state({"elo": None, "glicko": None})
    store._save_counts_state(initial)
    before = {name: (Path(store.root) / name).read_bytes() for name in store._FILES}
    load = store.load_counts_state
    loaded = []

    def counted_load():
        state = load()
        loaded.append(state)
        return state

    def engineer(frame, *, counts_state, **kwargs):
        counter = counts_state.setdefault("counter", {"rows": 0})
        previous = counter["rows"]
        counter["rows"] += len(frame)
        if fail:
            raise ValueError("injected FE failure")
        return frame.assign(prior_rows=previous)

    monkeypatch.setattr(store, "load_counts_state", counted_load)
    monkeypatch.setattr(store, "_seeded_rating_frame",
                        lambda frame, seed: (frame[["race_id", "horse_name"]], [], seed))
    monkeypatch.setattr(feature_store, "engineer_features", engineer)
    for _ in range(2):
        if fail:
            with pytest.raises(ValueError, match="injected FE failure"):
                store.featurize_live(card)
        else:
            result = store.featurize_live(card)
            assert result.prior_rows.tolist() == [initial.get("counter", {}).get("rows", 0)]
        assert {name: (Path(store.root) / name).read_bytes() for name in store._FILES} == before
    assert len(loaded) == 2
    assert loaded[0] is not loaded[1]
    assert pickle.loads(before["counts_state.pkl"]) == initial


def test_live_store_without_carried_state_still_requires_rebuild(tmp_path):
    store = FeatureStore(str(tmp_path))
    Path(store.featured_path).touch()
    Path(store.meta_path).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks carried state"):
        store.featurize_live(pd.DataFrame())


def test_expanding_updates_preserve_seed_and_match_full_history():
    frame = pd.DataFrame({"k": ["A", "B", "A", "B", "A", "C", "A", "B"],
                          "v": [1., 9., np.nan, 4., 3., 8., 5., 7.]})
    expected = fe._expanding_prev(frame, "k", "v", ["median", "std"])
    state = {}
    with fe.counts_state_scope(state):
        fe._expanding_prev(frame.iloc[:4], "k", "v", ["median", "std"])
    seed = state["exp::k::v"]
    before = pickle.dumps(seed)
    pieces = []
    for lo, hi in ((4, 6), (6, 8)):
        with fe.counts_state_scope(state):
            pieces.append(fe._expanding_prev(frame.iloc[lo:hi], "k", "v", ["median", "std"]))
    assert pickle.dumps(seed) == before
    for stat in expected:
        pd.testing.assert_series_equal(pd.concat([part[stat] for part in pieces]),
                                       expected[stat].iloc[4:], check_exact=True, check_names=False)
    assert state["exp::k::v"]["B"] == [9., 4., 7.]


def test_distinct_updates_preserve_seed_and_match_full_history():
    frame = pd.DataFrame({"entity": ["A", "B", "A", "B", "A", "C", "A", "B"],
                          "value": ["X", "Y", "X", "Z", None, "Z", "W", "Y"],
                          "race_date": pd.date_range("2026-01-01", periods=8),
                          "race_id": [f"r{i}" for i in range(8)]})
    expected = fe._distinct_count(frame, "entity", "value")
    state = {}
    with fe.counts_state_scope(state):
        fe._distinct_count(frame.iloc[:4], "entity", "value")
    seed = state["nuniq::entity::value"]
    before = pickle.dumps(seed)
    pieces = []
    for lo, hi in ((4, 6), (6, 8)):
        with fe.counts_state_scope(state):
            pieces.append(fe._distinct_count(frame.iloc[lo:hi], "entity", "value"))
    assert pickle.dumps(seed) == before
    pd.testing.assert_series_equal(pd.concat(pieces), expected.iloc[4:], check_exact=True)
    assert state["nuniq::entity::value"] == {"A": {"X", "W"}, "B": {"Y", "Z"}, "C": {"Z"}}
