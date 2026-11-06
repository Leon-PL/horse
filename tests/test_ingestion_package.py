"""The collection reorganization must preserve shared state and cache locations."""
import importlib

import pytest


@pytest.mark.parametrize("legacy,current", [
    ("source_pipeline", "pipeline"),
    ("source_auxiliary", "auxiliary"),
    ("source_storage", "storage"),
    ("rtv_storage", "rtv_storage"),
    ("data_scraper", "sporting_life"),
    ("rtv_scraper", "racing_tv"),
])
def test_compatibility_imports_share_the_implementation(legacy, current):
    assert importlib.import_module(f"src.{legacy}") is importlib.import_module(f"src.ingestion.{current}")


def test_racecard_cache_uses_configured_data_directory(tmp_path, monkeypatch):
    import config
    from src.ingestion.sporting_life import _racecard_cache_path

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    assert _racecard_cache_path("2026-11-04") == str(tmp_path / "racecards_cache/racecards_2026-11-04.csv")


def test_processing_does_not_create_an_unconsumed_export(tmp_path, monkeypatch):
    import config
    import pandas as pd
    from src import pedigree_backfill
    from src.data_processor import process_data

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pedigree_backfill, "apply_pedigree", lambda frame: frame)
    raw = pd.DataFrame([{
        "race_id": "42", "horse_id": "11", "horse_name": "A",
        "race_date": "2026-11-04", "track": "York", "off_time": "14:00",
        "finish_position": 1, "odds": 3.0, "age": 4, "weight_lbs": 126,
        "num_runners": 8, "race_class": 3,
    }])
    result = process_data(raw)
    assert len(result) == 1
    assert list(tmp_path.iterdir()) == []
