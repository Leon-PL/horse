import numpy as np
import pandas as pd
import pytest

from src.probability_calibration import PlaceProbabilityRefinement, fit_recent_place_refinement


def test_recent_calibration_cannot_fit_a_current_or_future_calibration_label(monkeypatch):
    import config
    from src.model import RacePredictor
    dates = pd.date_range("2024-01-01",periods=250)
    groups = np.array([5]*len(dates))
    X = np.repeat(np.arange(len(dates)),5).reshape(-1,1).astype(np.float32)
    positions = np.tile([1,2,3,4,0],len(dates))
    calls = []
    class Constant:
        def predict_proba(self,matrix):
            return np.tile([.8,.2],(len(matrix),1))
    def fit(self,matrix,target,**kwargs):
        calls.append(float(matrix.max()))
        return Constant()
    monkeypatch.setattr(RacePredictor,"_train_win_classifier",fit)
    monkeypatch.setattr(RacePredictor,"_train_place_classifier",fit)
    p = RacePredictor()
    p.feature_cols = ["signal"]
    mapping,info = fit_recent_place_refinement(p,X,groups,(positions==1).astype(int),
        ((positions>=1)&(positions<=2)).astype(int),positions,dates,race_types=np.array(["Chase"]*len(X)))
    assert mapping is not None and len(calls)==2
    first = pd.Timestamp(info["first_date"])
    assert dates[int(max(calls))] <= first-pd.Timedelta(days=7)
    assert (pd.Timestamp(info["last_date"])-first).days < 90


def test_saved_refinement_preserves_old_models_and_replays_new_place_probabilities(tmp_path):
    from test_model_probability_pipeline import _predictor,_races
    from src.model import RacePredictor
    p = _predictor()
    baseline = p.predict_races(_races())
    p.place_refinement = PlaceProbabilityRefinement(1.02,.01,0.,{("Unknown",3):.02})
    before = p.predict_races(_races())
    path = tmp_path/"refined.joblib"
    p.save(str(path))
    restored = RacePredictor()
    restored.load(str(path))
    after = restored.predict_races(_races())
    np.testing.assert_array_equal(before.win_probability,baseline.win_probability)
    np.testing.assert_array_equal(after.place_probability,before.place_probability)
    assert restored.place_refinement is not None
