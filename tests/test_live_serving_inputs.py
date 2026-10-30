import numpy as np
import pandas as pd
import pytest

import src.live_prediction as live


@pytest.mark.parametrize("code,sex", [("Flat", "Male"), ("Hurdle", "Female"), ("Chase", "Female")])
def test_single_category_indicators_have_training_meanings(code, sex):
    source = pd.DataFrame({"race_type": [code]*3, "sex": [sex]*3, "official_rating": [90., np.nan, 100.]})
    result = live.stable_card_indicators(source)
    for label in ("Flat", "Hurdle", "Chase"):
        assert result[f"race_type_{label}"].eq(int(code == label)).all()
    assert result.sex_Female.eq(int(sex == "Female")).all()
    pd.testing.assert_series_equal(result.official_rating, source.official_rating)
    assert list(source.columns) == ["race_type", "sex", "official_rating"]


def test_fallback_keeps_missing_measurements_and_only_completes_binary_schema(monkeypatch):
    history = pd.DataFrame({"race_id": [1], "race_date": ["2026-01-01"], "horse_name": ["A"],
        "official_rating": [np.nan], "weight_lbs": [np.nan], "race_type": ["Chase"],
        "sex": ["Female"], "going": ["Soft"], "going_Soft": [1], "going_numeric": [1.],
        "season_Winter": [1]})
    card = pd.DataFrame({"race_id": [2], "race_date": ["2026-01-02"], "horse_name": ["B"],
        "official_rating": [np.nan], "weight_lbs": [np.nan], "race_type": ["Flat"], "sex": ["Male"],
        "going": ["Good"], "going_numeric": [np.nan]})
    captured = []
    def engineer(frame, **kwargs):
        captured.append(frame.copy())
        return frame
    monkeypatch.setattr(live, "engineer_features", engineer)
    result = live.feature_engineer_with_history_core(history, card)
    assert captured[0].official_rating.isna().all()
    assert captured[0].weight_lbs.isna().all()
    assert result.going_Soft.eq(0).all()
    assert result.race_type_Chase.eq(0).all()
    assert result.sex_Female.eq(0).all()
    assert result.going_numeric.isna().all()
    assert result.season_Winter.eq(0).all()


def test_live_cache_keys_include_preparation_source(monkeypatch, tmp_path):
    class Store:
        def meta(self):
            return {"fe_signature": "test"}
    store = Store()
    for name in ("featured_path", "processed_path", "state_path", "counts_state_path"):
        path = tmp_path/name; path.write_bytes(b"unchanged")
        setattr(store, name, str(path))
    monkeypatch.setattr(live, "_serving_input_signature", lambda: "old")
    first = live.feature_store_signature(store)
    first_history = live.history_source_signature(pd.DataFrame({"race_date": ["2026-01-01"]}))
    monkeypatch.setattr(live, "_serving_input_signature", lambda: "new")
    assert live.feature_store_signature(store) != first
    assert live.history_source_signature(pd.DataFrame({"race_date": ["2026-01-01"]})) != first_history


def test_public_live_entry_completes_indicators_before_feature_stage(monkeypatch):
    from src import data_processor, featured_dataset
    card = pd.DataFrame({"horse_id": [1, 2], "race_id": [3, 3], "race_date": ["2026-01-02"]*2,
                         "horse_name": ["A", "B"]})
    monkeypatch.setattr(featured_dataset, "fe_signature", lambda: "test")
    monkeypatch.setattr(live, "feature_store_signature", lambda store: "stable")
    monkeypatch.setattr(live, "_require_current_history", lambda *args: None)
    def process(**kwargs):
        return kwargs["df"].assign(race_type="Flat", sex="Male")
    monkeypatch.setattr(data_processor, "process_data", process)
    class Store:
        def exists(self): return True
        def meta(self): return {"fe_signature": "test"}
        def last_date(self): return pd.Timestamp("2026-01-01")
        def featurize_live(self, processed):
            assert processed.race_type_Chase.eq(0).all()
            assert processed.race_type_Hurdle.eq(0).all()
            assert processed.sex_Female.eq(0).all()
            return processed
    result = live.features_for_live_cards(card, "2026-01-02", store=Store(), use_cache=False)
    assert len(result) == 2
