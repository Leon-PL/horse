"""Tests for the place-market strategy maths (settlement + +EV evaluation)."""

import numpy as np
import pandas as pd
import pytest

from src.bet_settlement import settle_place_exchange_bet, settle_place_exchange_bets
from src.place_market_model import evaluate_place_market, _placed_target


class _ConstModel:
    """predict_proba returns a constant P(place)."""
    def __init__(self, p): self.p = p
    def predict_proba(self, X):
        n = len(X); return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def test_settle_place_exchange_scalar():
    # placed: (5-1)*0.98 = 3.92 ; not placed: -1
    assert settle_place_exchange_bet(5.0, True, commission=0.02) == pytest.approx(3.92)
    assert settle_place_exchange_bet(5.0, False, commission=0.02) == -1.0
    # zero commission
    assert settle_place_exchange_bet(3.0, True, commission=0.0) == pytest.approx(2.0)


def test_settle_place_exchange_vectorised():
    pnl = settle_place_exchange_bets([4.0, 4.0], [True, False], commission=0.0)
    assert list(pnl) == [3.0, -1.0]


def _eval_df():
    # 4 runners, P(place)=0.4 constant. place_bsp gives EVs: 0.6, -0.2, 0.2, 1.0
    return pd.DataFrame({
        "f1": [0.0, 0.0, 0.0, 0.0],                       # a (dummy) feature column
        "place_bsp": [4.0, 2.0, 3.0, 5.0],
        "bf_places_paid": [3, 3, 3, 3],
        "finish_position": [2, 4, 5, 1],                  # placed: A yes, B no, C no, D yes
        "placed_bf": [1, 0, 0, 1],
        "race_type": ["Hurdle", "Hurdle", "Flat", "Chase"],
    })


def test_evaluate_selects_positive_ev_and_books_pnl():
    res = evaluate_place_market(_eval_df(), _ConstModel(0.4), ["f1"],
                                ev_threshold=0.10, commission=0.0, n_boot=200)
    # A(+0.6), C(+0.2), D(+1.0) clear 0.10; B(-0.2) excluded
    assert res["n_bets"] == 3
    # PnL: A placed (4-1)=3 ; C not placed -1 ; D placed (5-1)=4  -> total 6 over 3 = +200%
    assert res["total_pnl"] == pytest.approx(6.0)
    assert res["roi_pct"] == pytest.approx(200.0)
    assert res["hit_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_evaluate_threshold_tightens():
    res = evaluate_place_market(_eval_df(), _ConstModel(0.4), ["f1"],
                                ev_threshold=0.70, commission=0.0, n_boot=50)
    assert res["n_bets"] == 1                              # only D (+1.0) clears 0.70


def test_evaluate_jumps_subset():
    res = evaluate_place_market(_eval_df(), _ConstModel(0.4), ["f1"],
                                ev_threshold=0.10, commission=0.0, n_boot=50)
    # jumps = A (Hurdle) + D (Chase); C is Flat -> excluded from jumps subset
    assert res["jumps"]["n_bets"] == 2


def test_evaluate_empty_selection_is_safe():
    res = evaluate_place_market(_eval_df(), _ConstModel(0.0), ["f1"],
                                ev_threshold=0.10, commission=0.0, n_boot=50)
    assert res["n_bets"] == 0 and res["roi_pct"] is None and res["total_pnl"] == 0.0


def test_placed_target_prefers_exchange_flag():
    df = pd.DataFrame({"placed_bf": [1, 0], "finish_position": [9, 1], "bf_places_paid": [3, 3]})
    assert list(_placed_target(df)) == [1, 0]              # uses placed_bf, not finish_position
