import numpy as np
import pytest

from src.luck_weights import comment_severity, luck_weights, race_luck, stewards_severity


def test_comment_severity_separates_luck_from_habit():
    comments = [
        "tracked leaders, fell 2 out",
        "held up, badly hampered and lost all chance 3f out",
        "midfield, short of room over 1f out, kept on",
        "prominent, hampered 2f out, one pace",
        "slowly into stride, always behind",           # habit, not luck
        "not fluent 7th, weakened 3 out",              # jumping error, not luck
        "led, unseated rider 4th",
        None,
    ]
    assert comment_severity(comments).tolist() == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0]


def test_stewards_severity_takes_the_worst_tag():
    assert stewards_severity(["hung", "lost_shoe"]) == 0.5
    assert stewards_severity(["trouble_in_running", "lame"]) == 1.0
    assert stewards_severity(["gear_permission", "green"]) == 0.0
    assert stewards_severity([]) == 0.0


def test_race_luck_counts_only_troubled_losers_by_market_chance():
    severity = [1.0, 0.0, 0.5, 1.0, 0.0]
    market = [0.5, 0.3, 0.2, 0.6, 0.4]
    won = [0, 1, 0, 1, 0]
    luck = race_luck(severity, market, won, [3, 2])
    np.testing.assert_allclose(luck, [0.5 + 0.1, 0.0])


def test_unpriced_races_use_uniform_chances():
    luck = race_luck([1.0, 0.0, 0.0, 0.0], [np.nan, 0.2, 0.3, 0.5], [0, 1, 0, 0], [4])
    assert luck[0] == pytest.approx(0.25)


def test_weight_schemes():
    severity = [1.0, 0.0, 0.5, 1.0, 0.0]
    market = [0.5, 0.3, 0.2, 0.6, 0.4]
    won = [0, 1, 0, 1, 0]
    np.testing.assert_allclose(luck_weights("race_beta2", severity, market, won, [3, 2]), [0, 0, 0, 1, 1])
    np.testing.assert_allclose(luck_weights("race_beta4", severity, market, won, [3, 2]), [0, 0, 0, 1, 1])
    np.testing.assert_allclose(luck_weights("runner", severity, market, won, [3, 2]), [0, 1, 0.5, 1, 1])
    small = luck_weights("race_beta2", [0.5, 0, 0], [0.1, 0.5, 0.4], [0, 1, 0], [3])
    np.testing.assert_allclose(small, [0.9, 0.9, 0.9])
