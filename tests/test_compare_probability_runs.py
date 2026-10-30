"""Comparisons align by horse, retain DNF starters and share one population."""
import pandas as pd
import pytest

from scripts.compare_probability_runs import compare


def frame():
    return pd.DataFrame({
        "race_id": ["a", "a", "a"], "race_date": ["2026-07-15"] * 3,
        "horse_name": ["winner", "faller", "last"], "num_runners": [3] * 3,
        "finish_position": [1, 0, 3], "won": [1, 0, 0],
    })


class Predictor:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_races(self, data):
        output = data[["race_id", "horse_name"]].copy()
        output["win_probability"] = output.horse_name.map(self.probabilities)
        return output.iloc[::-1]


def test_shared_field_alignment_and_score_gate(tmp_path):
    old = Predictor({"winner": .2, "faller": .4, "last": .4})
    new = Predictor({"winner": .6, "faller": .2, "last": .2})
    saved = tmp_path / "paired.parquet"
    report = compare(frame(), old, new, prediction_path=saved)
    assert report["runners"] == 3  # the genuine faller remains
    assert report["metrics"]["candidate"]["brier"] < report["metrics"]["baseline"]["brier"]
    assert report["non_worsening_probability_scores"]
    assert report["paired_delta_candidate_minus_baseline"]["race_log_loss"]["mean"] < 0
    paired = pd.read_parquet(saved).set_index("horse_name")
    assert paired.loc["faller", "finish_position"] == 0
    assert paired.loc["winner", "baseline_probability"] == .2
    assert paired.loc["winner", "candidate_probability"] == .6
    assert not compare(frame(), new, old)["non_worsening_probability_scores"]


def test_incomplete_fields_cannot_be_scored_as_complete():
    incomplete = frame().iloc[:2].copy()
    predictor = Predictor({"winner": .6, "faller": .4})
    with pytest.raises(ValueError, match="No complete"):
        compare(incomplete, predictor, predictor)


def test_invalid_model_probability_is_rejected():
    invalid = Predictor({"winner": .7, "faller": .7, "last": -.4})
    with pytest.raises(ValueError, match="invalid probabilities"):
        compare(frame(), invalid, invalid)
