import numpy as np
import pytest

from src.market_correction_research import apply_correction, correction_objective, fit_correction


def arrays():
    market = np.array([.55, .3, .15, .35, .25, .25, .15])
    model = np.array([.4, .4, .2, .5, .2, .2, .1])
    return model, market, np.array([3, 4]), np.array([1, 0, 0, 1, 0, 0, 0])


def test_identity_is_exact_and_each_field_stays_normalized():
    model, market, groups, _ = arrays()
    np.testing.assert_array_equal(apply_correction(model, market, groups), market)
    corrected = apply_correction(model, market, groups, .4, 1.1)
    np.testing.assert_allclose([corrected[:3].sum(), corrected[3:].sum()], 1, atol=1e-14)
    assert np.all((corrected > 0) & (corrected < 1))


def test_analytic_gradient_matches_numerical_derivative():
    model, market, groups, won = arrays()
    parameters = np.array([.35, .9])
    _, gradient = correction_objective(parameters, model, market, groups, won)
    differences = []
    for direction in np.eye(2)*1e-6:
        above = correction_objective(parameters+direction, model, market, groups, won)[0]
        below = correction_objective(parameters-direction, model, market, groups, won)[0]
        differences.append((above-below)/2e-6)
    np.testing.assert_allclose(gradient, differences, rtol=1e-6, atol=1e-8)


def test_bounded_fit_does_not_worsen_training_objective_and_market_only_has_zero_model_weight():
    model, market, groups, won = arrays()
    for allowed in [False, True]:
        result = fit_correction(model, market, groups, won, model_allowed=allowed)
        assert result["accepted_fit"] and result["converged"]
        assert 0 <= result["alpha"] <= 1 and .5 <= result["beta"] <= 1.5
        assert result["objective"] <= result["market_objective"] + 1e-10
        if not allowed:
            assert result["alpha"] == 0


@pytest.mark.parametrize("change", ["partial_book", "negative", "empty_group", "zero_market", "wrong_winner"])
def test_invalid_fields_fail_instead_of_renormalizing_a_partial_population(change):
    model, market, groups, won = arrays()
    if change == "partial_book": model[0] = .3
    elif change == "negative": model[0] = -.1
    elif change == "empty_group": groups[0] = 0
    elif change == "zero_market": market[0] = 0
    else: won[0] = 0
    with pytest.raises(ValueError):
        correction_objective([0., 1.], model, market, groups, won)


def test_empty_input_and_negative_regularization_are_rejected():
    with pytest.raises(ValueError):
        apply_correction([], [], [])
    with pytest.raises(ValueError):
        correction_objective([0., 1.], *arrays(), ridge=-.1)
