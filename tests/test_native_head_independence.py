"""Adding the place head must not change a locked win-only experiment."""
import numpy as np
import pandas as pd
import pytest

import config
from src.model import RacePredictor


@pytest.mark.parametrize("market_teacher", [False, True])
def test_native_full_model_has_identical_win_forecasts_to_win_only(monkeypatch, market_teacher):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", ["signal"])
    monkeypatch.setattr(config, "BURN_IN_MONTHS", 0)
    monkeypatch.setattr(config, "EARLY_STOPPING_ROUNDS", 8)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_DISTILL", market_teacher)
    monkeypatch.setattr(config, "WIN_SOFT_LABEL_BLEND", .5)
    rng = np.random.default_rng(50)
    races = 180
    frame = pd.DataFrame({"race_id": np.repeat(np.arange(races), 4),
        "race_date": np.repeat(pd.date_range("2023-01-01", periods=races), 4),
        "horse_name": [f"H{i}" for i in range(races*4)],
        "finish_position": np.tile([1, 2, 3, 4], races),
        "won": np.tile([1, 0, 0, 0], races), "num_runners": 4, "handicap": 0,
        "odds": np.tile([2., 4., 6., 8.], races),
        "lengths_behind": np.tile([0., 1., 2., 3.], races),
        "signal": rng.normal(size=races*4)})
    params = {role: {"n_estimators": 30, "min_child_samples": 5, "min_child_weight": .01}
              for role in ("classifier", "place")}
    probabilities = []
    for win_only in (True, False):
        model = RacePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
        model._autotune_njobs = 2
        model.fit_as_of(frame, frame.race_date.max(), params=params, win_only=win_only)
        probabilities.append(model.predict_races(frame, win_only=True).win_probability.to_numpy())
        assert (model.place_model is None) == win_only
    np.testing.assert_allclose(*probabilities, atol=1e-12, rtol=0)
