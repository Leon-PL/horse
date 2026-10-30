import numpy as np
import pytest

from scripts.canonical_training_cache import cached_component, fit_key


def test_resume_uses_only_the_identical_completed_component(tmp_path):
    calls = []
    def fit(x, y, **kwargs):
        calls.append(1)
        return {"fitted": y.tolist()}
    x, y = np.array([[1.], [2.]]), np.array([0., 1.])
    key = fit_key("manifest", "win", ["signal"], (x, y), {"sample_weight": np.ones(2)})
    result = cached_component(tmp_path, key, fit, (x, y), {"sample_weight": np.ones(2)})
    recovered = cached_component(tmp_path, key, lambda *a, **k: pytest.fail("Completed fit was repeated"), (x, y), {})
    assert recovered == result and calls == [1]


def test_changed_training_inputs_weights_calibration_split_or_recipe_do_not_reuse():
    x, y = np.array([[1.], [2.]]), np.array([0., 1.])
    kwargs = {"sample_weight": np.ones(2), "soft_y": np.array([.2, .7]), "eval_set": [(x, y)]}
    base = fit_key("manifest", "win", ["signal"], (x, y), kwargs)
    variants = [
        fit_key("other", "win", ["signal"], (x, y), kwargs),
        fit_key("manifest", "place", ["signal"], (x, y), kwargs),
        fit_key("manifest", "win", ["other"], (x, y), kwargs),
        fit_key("manifest", "win", ["signal"], (x[::-1], y), kwargs),
        fit_key("manifest", "win", ["signal"], (x, y[::-1]), kwargs),
        fit_key("manifest", "win", ["signal"], (x, y), {**kwargs, "sample_weight": np.array([1., 2.])}),
        fit_key("manifest", "win", ["signal"], (x, y), {**kwargs, "soft_y": np.array([.3, .7])}),
        fit_key("manifest", "win", ["signal"], (x, y), {**kwargs, "eval_set": [(x+1, y)]}),
    ]
    assert base not in variants and len(set(variants)) == len(variants)
