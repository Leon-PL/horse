import numpy as np
import pandas as pd

import config
from src.model import RacePredictor, _IdentityScaler


def test_scaler_preserves_missing_and_converts_integer_double_and_single_inputs():
    scaler = _IdentityScaler()
    raw = np.array([[1.00000003, np.nan], [7., 1/3]], dtype=np.float64)
    for operation in (scaler.fit_transform, scaler.transform):
        actual = operation(raw)
        assert actual.dtype == np.dtype("float32")
        np.testing.assert_array_equal(actual, raw.astype(np.float32))
        np.testing.assert_array_equal(operation(actual), actual)
        assert operation(np.array([[1,2]], dtype=np.int64)).dtype == np.dtype("float32")


def test_native_fit_and_saved_forecast_do_not_depend_on_frame_numeric_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal", "second"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 8)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", False)
    rng = np.random.default_rng(62)
    frame = pd.DataFrame({"race_id": np.repeat(np.arange(160),4),
        "race_date": np.repeat(pd.date_range("2023-01-01",periods=160),4),
        "horse_name": [f"h{i}" for i in range(640)], "num_runners": 4, "handicap": 0,
        "odds": np.tile([2.,4.,6.,8.],160), "won": np.tile([1,0,0,0],160),
        "finish_position": np.tile([1,2,3,4],160), "lengths_behind": np.tile([0.,1.,2.,3.],160),
        "signal": rng.normal(size=640), "second": rng.normal(size=640)})
    params = {"classifier":{"n_estimators":30,"min_child_samples":5,"min_child_weight":.01}}
    outputs=[]
    for dtype in (np.float32,np.float64):
        data=frame.copy();data[["signal","second"]]=data[["signal","second"]].astype(dtype)
        predictor=RacePredictor(random_seed=42);predictor._autotune_njobs=2
        predictor.fit_as_of(data,data.race_date.max(),params=params,win_only=True)
        outputs.append(predictor.predict_races(frame,win_only=True).win_probability.to_numpy())
        path=tmp_path/f"{np.dtype(dtype).name}.joblib";predictor.save(str(path))
        restored=RacePredictor();restored.load(str(path))
        _,matrix=restored._prepare_prediction_frame(frame)
        assert matrix.dtypes.eq(np.dtype("float32")).all()
        np.testing.assert_array_equal(restored.predict_races(frame,win_only=True).win_probability,outputs[-1])
    np.testing.assert_array_equal(*outputs)
