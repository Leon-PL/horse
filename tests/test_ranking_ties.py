"""An uncertain ranking cannot improve its accuracy by dropping tied races."""
import numpy as np
import pandas as pd
import pytest

from src.model import RacePredictor


def evaluate(scores, positions):
    positions = np.asarray(positions)
    relevance = np.where(positions == 1, 5, 0)
    scores = np.asarray(scores, dtype=float)
    return RacePredictor()._evaluate_as_ranker(
        scores, relevance, np.array([len(scores)]),
        pd.DataFrame({"finish_position": positions}), "ties",
        calibrated_probs=scores / scores.sum(),
    )


def test_uniform_race_counts_and_receives_expected_credit():
    metrics = evaluate([.25] * 4, [1, 2, 3, 4])
    assert metrics["total_races"] == 1
    assert metrics["top1_accuracy"] == .25
    assert metrics["win_in_top3"] == .75
    assert metrics["ndcg_at_1"] == pytest.approx(.25)


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
def test_partial_ties_are_independent_of_result_order(order):
    scores = np.array([.4, .2, .2, .2, .0])[order]
    positions = np.array([2, 1, 3, 4, 5])[order]
    metrics = evaluate(scores, positions)
    assert metrics["top1_accuracy"] == 0
    assert metrics["win_in_top3"] == pytest.approx(2 / 3)


def test_top_tie_shares_win_credit():
    metrics = evaluate([.4, .4, .1, .1], [1, 2, 3, 4])
    assert metrics["top1_accuracy"] == .5
    assert metrics["win_in_top3"] == 1


def test_dead_heat_top_three_credit_is_probability_of_any_winner():
    metrics = evaluate([.2] * 5, [1, 1, 3, 4, 5])
    assert metrics["top1_accuracy"] == .4
    assert metrics["win_in_top3"] == pytest.approx(.9)


def test_unambiguous_ranking_keeps_binary_credit():
    metrics = evaluate([.6, .3, .1], [2, 1, 3])
    assert metrics["top1_accuracy"] == 0
    assert metrics["win_in_top3"] == 1
