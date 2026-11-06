"""The collection reorganization must preserve shared state and cache locations."""
import importlib

import pytest


@pytest.mark.parametrize("retired", [
    "source_pipeline", "source_auxiliary", "rtv_storage", "data_scraper", "rtv_scraper",
    "pedigree_backfill", "rtv_feature_collection", "weather", "bha_stewards_scraper",
    "source_storage", "attheraces_scraper", "atr_rebuild",
])
def test_retired_collection_names_stay_retired(retired):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"src.{retired}")


def test_racecard_cache_uses_configured_data_directory(tmp_path, monkeypatch):
    import config
    from src.ingestion.sporting_life import _racecard_cache_path

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    assert _racecard_cache_path("2026-11-04") == str(tmp_path / "racecards_cache/racecards_2026-11-04.csv")


def test_processing_does_not_create_an_unconsumed_export(tmp_path, monkeypatch):
    import config
    import pandas as pd
    from src import data_processor
    from src.data_processor import process_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_processor, "apply_pedigree", lambda frame: frame)
    raw = pd.DataFrame([{
        "race_id": "42", "horse_id": "11", "horse_name": "A",
        "race_date": "2026-11-04", "track": "York", "off_time": "14:00",
        "finish_position": 1, "odds": 3.0, "age": 4, "weight_lbs": 126,
        "num_runners": 8, "race_class": 3,
    }])
    result = process_data(raw)
    assert len(result) == 1
    assert list(tmp_path.iterdir()) == []
