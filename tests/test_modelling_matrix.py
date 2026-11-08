"""Explicit matrix inputs keep wide selection out of copied DataFrames."""
import numpy as np
import pytest

from src.modelling.win import WinModel
from tests.test_modelling import fields, recipe


@pytest.mark.parametrize("objective", ["exploded", "cross_entropy"])
@pytest.mark.parametrize("capacity", ["screen", "native"])
def test_matrix_path_matches_named_frames_and_does_not_retain_training_arrays(objective, capacity):
    frame = fields()
    spec = recipe(objective=objective)
    matrix = np.ascontiguousarray(frame[list(spec.features)].to_numpy(float))
    context = frame.drop(columns=list(spec.features))
    reference = WinModel(spec).fit(frame, capacity=capacity)
    model = WinModel(spec).fit(context, matrix=matrix, matrix_columns=spec.features, capacity=capacity)
    np.testing.assert_array_equal(model.predict(context, matrix=matrix, matrix_columns=spec.features), reference.predict(frame))
    assert model.temperature == reference.temperature
    assert not any(isinstance(v, np.ndarray) for v in vars(model).values())
    shuffled = context.sample(frac=1, random_state=6)
    actual = model.predict(shuffled, matrix=matrix[shuffled.index], matrix_columns=spec.features)
    np.testing.assert_array_equal(actual, reference.predict(frame)[shuffled.index])


def test_matrix_slices_share_the_input_and_context_stays_narrow():
    frame = fields()
    spec = recipe()
    matrix = np.ascontiguousarray(frame[list(spec.features)].to_numpy(float))
    context = frame.drop(columns=list(spec.features))
    model = WinModel(spec)
    original = model._fit_estimator
    seen = []
    def inspect(part, **kwargs):
        assert not set(spec.features) & set(part.columns)
        assert np.shares_memory(kwargs["matrix"], matrix)
        seen.append(len(part))
        return original(part, **kwargs)
    model._fit_estimator = inspect
    model.fit(context, matrix=matrix, matrix_columns=spec.features, capacity="screen")
    assert len(seen) == 2 and seen[0] < seen[1]


def test_matrix_columns_shape_and_numeric_contract_fail_closed():
    frame = fields()
    spec = recipe()
    context = frame.drop(columns=list(spec.features))
    matrix = frame[list(spec.features)].to_numpy(float)
    with pytest.raises(ValueError, match="column order"):
        WinModel(spec).fit(context, matrix=matrix, capacity="screen")
    with pytest.raises(ValueError, match="column order"):
        WinModel(spec).fit(context, matrix=matrix, matrix_columns=tuple(reversed(spec.features)), capacity="screen")
    with pytest.raises(ValueError, match="shape"):
        WinModel(spec).fit(context, matrix=matrix[:-1], matrix_columns=spec.features, capacity="screen")
    with pytest.raises(ValueError, match="numeric"):
        WinModel(spec).fit(context, matrix=matrix.astype(str), matrix_columns=spec.features, capacity="screen")
    bad = matrix.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinities"):
        WinModel(spec).fit(context, matrix=bad, matrix_columns=spec.features, capacity="screen")


def test_matrix_cannot_override_a_shared_context_feature():
    frame = fields()
    spec = recipe(features=("num_runners", "signal"))
    matrix = frame[list(spec.features)].to_numpy(float)
    matrix[:, 0] += 1
    with pytest.raises(ValueError, match="disagree"):
        WinModel(spec).fit(frame.drop(columns="signal"), matrix=matrix, matrix_columns=spec.features, capacity="screen")
