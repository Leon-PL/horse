"""ATR efficiency direction, retained poor runs and unknown placeholders."""
import numpy as np
import pytest

from src import atr_features as af
from test_atr_features import _make_history


def test_efficiency_keeps_poor_finite_runs_and_selects_lowest_as_best(monkeypatch, publish_dataset):
    from src import feature_engineer as fe
    from src.source_dataset import read_table

    def poor_runs(sectionals):
        sectionals["atr_efficiency_score"] = [147., 72.7, 5.4]
    frame = _make_history(monkeypatch, publish_dataset, poor_runs)
    computed = af.compute_atr_run_features(read_table("atr", "sectionals").assign(won=0, lengths_behind=2.0))
    assert computed.atr_efficiency_score.tolist() == [147., 72.7, 5.4]
    features = fe.add_atr_features(frame)
    latest = features.iloc[-1]
    assert latest.atr_efficiency_score_best_3 == 5.4
    assert latest.atr_efficiency_score_avg_3 == pytest.approx((147+72.7+5.4)/3)
    assert dict(af.RUN_METRICS)["atr_efficiency_score"] is False


def test_efficiency_invalid_scores_and_unrated_zero_placeholders_are_unknown():
    import pandas
    frame = pandas.DataFrame({"atr_efficiency_score": [0., -1., np.inf, 0.],
        "atr_efficiency_grade": ["A++", "A", "F", None],
        "atr_energy_dist": ["[90.0, 100.0]"]*3 + [None], "atr_optimum_energy": ["[92.0, 101.0]"]*3 + [None]})
    values = af.efficiency_error_scores(frame)
    assert values[0] == 0
    assert np.isnan(values[1:]).all()
    assert frame.atr_efficiency_score.iloc[2] == np.inf  # does not mutate raw inputs


def record(day, score, *, grade="C", horse="101", race=None, **kwargs):
    return {"horse_id": horse, "race_id": str(day) if race is None else race,
            "race_date": f"2026-01-{day:02d}", "off_time": "14:00",
            "atr_efficiency_score": score, "atr_efficiency_grade": grade,
            "atr_finishing_speed_pct": 100+day, "atr_race_finishing_speed_pct": 100,
            **kwargs}


def test_collection_guard_ignores_feature_edits_but_still_detects_source_edits(tmp_path, monkeypatch):
    from pathlib import Path
    from src.ingestion import atr_rebuild
    project = Path(atr_rebuild.config.BASE_DIR)
    for name in (*atr_rebuild.CODE_FILES, "src/atr_features.py"):
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((project/name).read_bytes())
    monkeypatch.setattr(atr_rebuild.config, "BASE_DIR", str(tmp_path))
    before = atr_rebuild.code_hashes()
    with (tmp_path/"src/atr_features.py").open("a", encoding="utf-8") as stream:
        stream.write("\n_NEW_FEATURE = 1\n")
    assert atr_rebuild.code_hashes() == before
    with (tmp_path/"src/ingestion/attheraces.py").open("a", encoding="utf-8") as stream:
        stream.write("\n_NEW_PARSER_SETTING = 1\n")
    assert atr_rebuild.code_hashes() != before
    after_parser_edit = atr_rebuild.code_hashes()
    with (tmp_path/"src/ingestion/atr_rebuild.py").open("a", encoding="utf-8") as stream:
        stream.write("\n_NEW_ACCEPTANCE_SETTING = 1\n")
    assert atr_rebuild.code_hashes() != after_parser_edit
