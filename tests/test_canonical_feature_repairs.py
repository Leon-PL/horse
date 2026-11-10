"""Correctness contracts must hold through the ordinary preprocessing path."""
import numpy as np
import pandas as pd
import pytest

from src.data_processor import clean_data
from src.each_way import adjust_place_probs_for_race
from src.model import get_feature_columns
from src.prize_currency_features import normalise_prize_money


def test_normal_preprocessing_corrects_published_race_kind_for_the_whole_field():
    frame = pd.DataFrame({"race_id": [622145]*2, "race_date": ["2021-04-28"]*2,
        "horse_name": ["A", "B"], "track": ["Punchestown"]*2, "race_type": ["Flat"]*2})
    assert clean_data(frame).race_type.eq("Chase").all()
    frame["race_date"] = "2022-04-28"
    with pytest.raises(ValueError, match="source date/course"):
        clean_data(frame)


def test_normal_currency_path_is_idempotent_and_preserves_evidence():
    source = pd.DataFrame({"race_date": pd.to_datetime(["2025-01-06"]*3),
        "track": ["CURRAGH", "Ascot", "Unverified venue"], "prize_money": [10000., 10000., 10000.]})
    result = normalise_prize_money(source)
    assert 0 < result.prize_money.iloc[0] < 10000.
    assert result.prize_money.iloc[1] == 10000.
    assert np.isnan(result.prize_money.iloc[2])
    assert result.prize_money_original.tolist() == [10000.]*3
    assert result.prize_fx_date.iloc[0] < source.race_date.iloc[0]
    pd.testing.assert_frame_equal(result, normalise_prize_money(result))


def test_unconverted_raw_money_never_reenters_an_all_feature_model(monkeypatch):
    import config
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    assert get_feature_columns(pd.DataFrame({"signal": [1.], "prize_money_original": [1000.]})) == ["signal"]


@pytest.mark.parametrize("explicit_rebuild", [False, True])
def test_ordinary_training_rebuild_receives_corrections_without_research_flags(monkeypatch, tmp_path, explicit_rebuild):
    from contextlib import nullcontext
    from src import store_refresh
    from src.data_processor import process_data
    from src import store_build as builder
    from tests.test_live_parity import _synthetic_raw
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=4)
    raw = raw.assign(race_id=622145, race_date="2021-04-28", track="Punchestown", race_type="Flat",
                     going="Yielding (Good To Yielding In Places)", prize_money=10000.)
    processed = process_data(raw)
    import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    captured = []
    class Store:
        _rebuild_reason = "fixture_rebuild"
        def _access(self): return nullcontext()
        def _staged_update(self): return nullcontext(self)
        def meta(self): return {}
        def load_featured(self, **kwargs): return pd.DataFrame({"signal": [1.]})
    monkeypatch.setattr(store_refresh, "FeatureStore", Store)
    monkeypatch.setattr(store_refresh, "process_data", lambda **kwargs: processed)
    monkeypatch.setattr(store_refresh, "append_current_store", lambda *args, **kwargs: False)
    monkeypatch.setattr(store_refresh, "_model_load_columns", lambda store: ["signal"])
    def build(store, chunk, **kwargs):
        captured.append(kwargs["processed"])
        assert chunk == 60000
    monkeypatch.setattr(builder, "build_store", build)
    result = store_refresh.get_training_dataset(rebuild=explicit_rebuild)
    assert result.signal.tolist() == [1.] and len(captured) == 1
    assert captured[0].race_type.eq("Chase").all()
    assert captured[0].going.eq("Yielding").all()
    assert captured[0].prize_currency.eq("EUR").all()
    assert captured[0].prize_money.lt(10000.).all()


def test_place_floor_does_not_invent_additional_placed_horses():
    win = np.array([.8, .05, .05, .05, .05])
    result = adjust_place_probs_for_race(np.full(5, .6), win, 2)
    np.testing.assert_allclose(result, [.8, .3, .3, .3, .3], atol=1e-10, rtol=0)


@pytest.mark.parametrize("size,places", [(2, 3), (4, 3), (6, 2), (12, 3), (20, 4)])
def test_place_constraints_survive_extreme_probabilities_and_runner_order(size, places):
    rng = np.random.default_rng(size)
    for _ in range(15):
        win = rng.dirichlet(np.full(size, .15))
        raw = rng.choice([1e-15, .01, .5, .99, 1-1e-15], size)
        result = adjust_place_probs_for_race(raw, win, places)
        assert np.all(result >= win-1e-12) and np.all(result <= 1)
        assert result.sum() == pytest.approx(min(size, places), abs=1e-10)
        order = rng.permutation(size)
        permuted = adjust_place_probs_for_race(raw[order], win[order], places)
        np.testing.assert_allclose(result[order], permuted, atol=1e-10, rtol=0)
