import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from src import feature_store
from src.feature_schema import categorical_columns, complete_categorical_columns, require_numeric_features
from src.feature_store import FeatureStore
from src.model import RacePredictor, _IdentityScaler, get_feature_columns


def test_categories_come_from_fields_instead_of_numeric_prefixes():
    history = pd.DataFrame({"going": ["Good", "Soft"], "race_type": ["Flat", "Flat"],
        "sex": ["Male", "Male"], "race_date": ["2025-01-01", "2025-01-02"]})
    declared = categorical_columns(history)
    assert {"going_Good", "going_Soft", "race_type_Chase", "sex_Female", "season_Summer"} <= set(declared)
    source = pd.DataFrame({"going_Good": [1], "going_numeric": [np.nan]})
    expected = ["going_Good", "going_Soft", "going_numeric", "going_unknown_measurement"]
    result = complete_categorical_columns(source, expected, declared)
    assert result.going_Soft.tolist() == [0]
    assert result.going_numeric.isna().all()
    assert "going_unknown_measurement" not in result
    assert "going_Soft" not in source


@pytest.mark.parametrize("metadata", [True, False])
def test_store_completes_only_declared_categorical_columns(tmp_path, monkeypatch, metadata):
    store = FeatureStore(str(tmp_path))
    prepared = pd.DataFrame({"race_id": [1], "horse_name": ["A"], "race_date": ["2025-01-01"],
        "going": ["Soft"], "going_Soft": [1], "going_numeric": [2.]})
    prepared.to_parquet(store.processed_path, index=False)
    prepared.to_parquet(store.featured_path, index=False)
    Path(store.meta_path).write_text(json.dumps({"categorical_columns": categorical_columns(prepared)} if metadata else {}))
    monkeypatch.setattr(store, "load_counts_state", lambda: {})
    monkeypatch.setattr(store, "load_state", lambda: {})
    monkeypatch.setattr(store, "_seeded_rating_frame", lambda f, s: (f[["race_id", "horse_name"]], [], s))
    monkeypatch.setattr(feature_store, "engineer_features", lambda f, **kw: f)
    card = prepared.drop(columns=["going_Soft", "going_numeric"]).assign(going="Good")
    actual = store.featurize_live(card)
    assert actual.going_Soft.eq(0).all()
    assert "going_numeric" not in actual


def test_training_requires_the_whole_allowlist_but_candidates_are_explicit(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["known", "missing"])
    frame = pd.DataFrame({"known": [1.]})
    with pytest.raises(ValueError, match="FEATURE_ALLOWLIST.*missing"):
        get_feature_columns(frame)
    assert get_feature_columns(frame, apply_allowlist=False) == ["known"]


@pytest.mark.parametrize("values", [["1"], [1+2j]])
def test_selected_features_must_be_numeric(values, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal"])
    with pytest.raises(ValueError, match="nonnumeric: signal"):
        get_feature_columns(pd.DataFrame({"signal": values}))


def test_prediction_validates_saved_feature_names_and_numeric_types():
    predictor = RacePredictor()
    predictor.feature_cols = ["signal", "other"]
    predictor.scaler = _IdentityScaler()
    with pytest.raises(ValueError, match="missing: other"):
        predictor._prepare_prediction_frame(pd.DataFrame({"signal": [1.]}))
    with pytest.raises(ValueError, match="nonnumeric: other"):
        predictor._prepare_prediction_frame(pd.DataFrame({"signal": [1.], "other": ["2"]}))
    _, matrix = predictor._prepare_prediction_frame(pd.DataFrame({"other": [np.nan], "signal": [1.]}))
    assert list(matrix) == predictor.feature_cols
    assert np.isnan(matrix.other.iloc[0])


def test_duplicated_features_are_not_a_valid_schema():
    with pytest.raises(ValueError, match="duplicated: signal"):
        require_numeric_features(pd.DataFrame([[1., 2.]], columns=["signal", "signal"]),
                                 ["signal"], context="Test")
