import numpy as np
import pytest

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from research.meta_learner.specialists import linear_combiner
from research.meta_learner.test_screening import synthetic


def test_linear_combiner_learns_each_correction_from_earlier_quarters_only():
    frame = synthetic(effect=0.6)
    frame["c_signal"] = frame.x * 0.3  # half the true effect, so the fitted weight should be near 2
    frame["c_noise"] = frame.noise * 0.1
    combiner = linear_combiner(["c_signal", "c_noise"])
    correction, folds = s.walk_forward(frame, combiner)
    assert [f["fold"] for f in combiner.info] == list(s.SCORED)
    assert all(f["converged"] for f in combiner.info)
    assert np.mean([f["c_signal"] for f in combiner.info]) == pytest.approx(2.0, abs=0.5)
    assert all(abs(f["c_noise"]) < 1.0 for f in combiner.info)
    scored = frame.quarter.isin(s.SCORED).to_numpy()
    assert np.isfinite(correction[scored]).all() and np.isnan(correction[~scored]).all()


def test_store_groups_never_include_excluded_columns():
    from src.model import EXCLUDE_COLUMNS
    blocked = ["atr_speed_score", "trainer_first_string_by_odds", "bsp"]
    assert set(blocked) <= set(EXCLUDE_COLUMNS)
    assert sc.allowed([*blocked, "rtv_late_gain_last"]) == ["rtv_late_gain_last"]
