"""Missing prices are unknown, never invented stakes, returns or substitutes."""
import numpy as np
import pandas as pd
import pytest

from src.bet_analysis import analyse_test_set, max_drawdown
from src.bet_settlement import valid_decimal_odds, value_bet_selection


def test_decimal_price_validation_rejects_non_prices():
    prices = [np.nan, np.inf, -1, 0, 0.5, 1, 1.01, 4]
    assert valid_decimal_odds(prices).tolist() == [False] * 6 + [True, True]
    selected = value_bet_selection(np.ones(8), prices, 0.01, implied_prob=np.zeros(8))
    assert selected["mask"].tolist() == [False] * 6 + [True, True]


@pytest.mark.parametrize("price", [np.nan, np.inf, 0.0, 1.0])
def test_testset_does_not_bet_on_unpriced_top_pick(price):
    frame = pd.DataFrame({
        "race_id": ["r", "r"], "race_date": pd.to_datetime(["2026-01-01"] * 2),
        "horse_name": ["winner", "other"], "finish_position": [1, 2],
        "won": [1, 0], "num_runners": [2, 2], "odds": [price, 4.0],
    })
    result = analyse_test_set(np.array([0.7, 0.3]), np.array([2]), frame,
                              value_threshold=0.99)
    assert result["bets"].empty
    assert result["stats"]["top_pick"]["bets"] == 0


def test_walkforward_keeps_prediction_metrics_without_priced_top_picks(monkeypatch):
    import src.backtester as bt
    days = pd.date_range("2025-01-01", "2025-03-31")
    frame = pd.DataFrame({
        "race_id": np.repeat(np.arange(len(days)), 2), "race_date": np.repeat(days, 2),
        "horse_name": np.tile(["winner", "other"], len(days)),
        "finish_position": np.tile([1, 2], len(days)), "won": np.tile([1, 0], len(days)),
        "num_runners": 2, "odds": np.tile([np.nan, 4.0], len(days)), "feature": 1.0,
    })

    class Predictor:
        def __init__(self, **kwargs):
            pass

        def train_on_fold(self, X_train, X_test, *args, **kwargs):
            return np.where(kwargs["test_df"]["horse_name"] == "winner", 0.7, 0.3), None

    monkeypatch.setattr(bt, "RacePredictor", Predictor)
    monkeypatch.setattr(bt, "get_feature_columns", lambda df: ["feature"])
    result = bt.walk_forward_validation(frame, min_train_months=2, value_threshold=0.99)
    assert len(result["folds"]) == 1
    assert result["folds"][0].top_pick_bets == 0
    assert np.isfinite(result["folds"][0].log_loss)
    assert result["bets"].empty


def test_drawdown_includes_losses_before_first_profit():
    assert max_drawdown(np.array([-1.0, -1.0, -1.0])) == 3.0
    assert max_drawdown(np.array([-2.0, 3.0, -4.0])) == 4.0
