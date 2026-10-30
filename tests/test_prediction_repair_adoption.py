import json

import pandas as pd
import pytest

from scripts import adopt_prediction_repair as adoption


def scores(loss, brier=.1, binary=.3):
    return {"race_log_loss": loss, "brier": brier, "binary_log_loss": binary}


@pytest.mark.parametrize("hard,teacher,gate,wanted", [
    (scores(1.89), scores(1.88), True, None),
    (scores(1.85), scores(1.86), True, "015_nonfinisher_history"),
    (scores(1.85), scores(1.84), True, "016_market_teacher"),
    (scores(1.85), scores(1.84, brier=.11), True, "015_nonfinisher_history"),
    (scores(1.85), scores(1.84), False, None),
])
def test_selection_requires_gain_over_best_completed_model_and_all_metric_guards(
        tmp_path, monkeypatch, hard, teacher, gate, wanted):
    monkeypatch.setattr(adoption, "BASE", tmp_path)
    monkeypatch.setattr(adoption, "write_results", lambda *args: None)
    for name in adoption.EXPERIMENTS:
        (tmp_path/name).mkdir()
        (tmp_path/name/"completion.json").write_text("{}")
    reports = {"014_margin_repair": {"pooled": {"candidate": scores(1.87)}},
        adoption.EXPERIMENTS[0]: {"candidate_passes_numeric_advancement": gate, "pooled": {"candidate": hard}},
        adoption.EXPERIMENTS[1]: {"candidate_passes_numeric_advancement": gate, "pooled": {"candidate": teacher}}}
    monkeypatch.setattr(adoption, "completed", lambda directory: reports[directory.name])
    frame = pd.DataFrame({"race_id": [1,1], "horse_name": ["A","B"],
        "race_date": pd.Timestamp("2025-07-01"), "won": [1,0],
        "baseline_probability": [.4,.6], "candidate_probability": [.6,.4]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame.copy())
    assert adoption.select()["selected"] == wanted


def test_changed_completed_artifact_prevents_selection(tmp_path):
    (tmp_path/"completion.json").write_text(json.dumps({"evidence_sha256": {"model.joblib": "wrong"}}))
    (tmp_path/"model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Completed evidence changed"):
        adoption.completed(tmp_path)


def test_secondary_comparison_rejects_a_missing_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(adoption, "BASE", tmp_path)
    for name in adoption.EXPERIMENTS:
        (tmp_path/name).mkdir()
        (tmp_path/name/"completion.json").write_text("{}")
    reports = {"pooled": {"candidate": scores(1.86)}, "candidate_passes_numeric_advancement": True}
    monkeypatch.setattr(adoption, "completed", lambda directory: reports)
    frame = pd.DataFrame({"race_id": [1,1], "horse_name": ["A","B"],
        "race_date": pd.Timestamp("2025-07-01"), "won": [1,0],
        "baseline_probability": [.4,.6], "candidate_probability": [.6,.4]})
    def read(path):
        return frame.copy() if path.parent.name==adoption.EXPERIMENTS[0] else frame.iloc[:1].copy()
    monkeypatch.setattr(pd, "read_parquet", read)
    with pytest.raises(ValueError, match="lost a field"):
        adoption.select()


def make_store(root, generation):
    from src.feature_store import FeatureStore
    store = FeatureStore(str(root))
    for name in store._FILES:
        value = json.dumps({"generation": generation}) if name == "meta.json" else f"{generation}:{name}"
        (root / name).write_text(value)
    return store


def test_source_zip_then_store_install_preserves_path_and_all_five_old_files(tmp_path):
    from zipfile import ZipFile
    current = make_store(tmp_path/"current", "original")
    candidate = make_store(tmp_path/"candidate", "corrected")
    recovery = tmp_path/"recovery"
    source = tmp_path/"example.py"; source.write_text("source example")
    adoption.capture_qualification_sources(tmp_path/"source.zip", ["example.py"], root=tmp_path)
    with ZipFile(tmp_path/"source.zip") as archive:
        assert archive.read("example.py") == b"source example"
    expected = {name: adoption.digest(tmp_path/"candidate"/name) for name in candidate._FILES}
    adoption.install_saved_store(candidate, current, recovery, expected)
    adoption.install_saved_store(candidate, current, recovery, expected)
    for name in current._FILES:
        assert adoption.digest(tmp_path/"current"/name) == expected[name]
        assert "original" in (recovery/name).read_text()


def test_changed_candidate_aborts_before_store_replacement(tmp_path):
    current = make_store(tmp_path/"current", "original")
    candidate = make_store(tmp_path/"candidate", "corrected")
    expected = {name: adoption.digest(tmp_path/"candidate"/name) for name in candidate._FILES}
    (tmp_path/"candidate"/"counts_state.pkl").write_bytes(b"unexpected different state")
    with pytest.raises(ValueError, match="Candidate generation changed"):
        adoption.install_saved_store(candidate, current, tmp_path/"recovery", expected)
    assert json.loads((tmp_path/"current"/"meta.json").read_text())["generation"] == "original"
