import json

import pandas as pd
import pytest

import config
from src.data_processor import _normalise_person_names, clean_data, process_data
from src.verified_race_metadata import apply_connection_aliases
from tests.test_live_parity import _synthetic_raw


def test_renamed_people_get_one_name_and_recleaning_is_identical():
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=4)
    raw["jockey"] = ["M P Sheehy", "Mikey Sheehy", "m.p. sheehy", "Oisin Murphy"]
    raw["trainer"] = ["Miss E Doyle", "Liz Doyle", "B J Murphy", "B T Murphy"]
    first = clean_data(raw)
    assert first.jockey.tolist() == ["Mikey Sheehy", "Mikey Sheehy", "Mikey Sheehy", "Oisin Murphy"]
    # Two trainers Racing TV filed under one ID stay apart.
    assert first.trainer.tolist() == ["Liz Doyle", "Liz Doyle", "B J Murphy", "B T Murphy"]
    again = clean_data(first)
    for col in ("jockey", "trainer"):
        pd.testing.assert_series_equal(first[col], again[col])


def test_live_card_processing_maps_names():
    card = _synthetic_raw(n_days=1, races_per_day=1, field=3)
    card["jockey"] = ["M P Sheehy", "H Cobden", "Rossa Ryan"]
    out = process_data(df=card)
    assert out.jockey.tolist() == ["Mikey Sheehy", "Harry Cobden", "Rossa Ryan"]


def test_reference_names_are_compared_in_cleaned_form(tmp_path, monkeypatch):
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "connection_aliases.json").write_text(
        json.dumps({"jockey": {"C McGivern": "Corey McGivern"}}), encoding="utf-8")
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    frame = pd.DataFrame({"jockey": _normalise_person_names(pd.Series(["C McGivern", "Other"]))})
    assert apply_connection_aliases(frame, _normalise_person_names).jockey.tolist() == ["Corey Mcgivern", "Other"]


def test_alias_chains_are_rejected(tmp_path, monkeypatch):
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "connection_aliases.json").write_text(
        json.dumps({"trainer": {"A Smith": "Alan Smith", "Alan Smith": "Al Smith"}}), encoding="utf-8")
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="chain"):
        apply_connection_aliases(pd.DataFrame({"trainer": ["A Smith"]}), _normalise_person_names)
