"""Tests for the shared bet-selection & settlement rules.

These rules drive the test-set simulation, the walk-forward backtester
and live Today's Picks settlement — a regression here silently skews
every PnL number in the app.
"""

import numpy as np
import pytest

from src.bet_settlement import (
    EW_MAX_ODDS,
    EW_MIN_ODDS,
    EW_STAKE_UNITS,
    EXCHANGE_COMMISSION,
    PLACE_EXCHANGE_COMMISSION,
    settle_win_exchange_bet,
    settle_win_exchange_bets,
    dynamic_value_threshold,
    ew_bet_selected,
    ew_odds_in_band,
    ew_placed_flag,
    settle_ew_bet,
    settle_win_bet,
    settle_win_bets,
    value_bet_selection,
)


class TestDynamicValueThreshold:
    def test_scalar_at_reference_odds(self):
        # At odds of 3.0 the threshold equals the base value.
        assert dynamic_value_threshold(0.05, 3.0) == pytest.approx(0.05)

    def test_scales_with_sqrt_of_odds(self):
        assert dynamic_value_threshold(0.05, 12.0) == pytest.approx(0.05 * 2.0)

    def test_odds_clipped_at_one(self):
        assert dynamic_value_threshold(0.05, 0.5) == pytest.approx(
            dynamic_value_threshold(0.05, 1.0)
        )

    def test_array_input(self):
        out = dynamic_value_threshold(0.1, np.array([3.0, 12.0]))
        assert out == pytest.approx([0.1, 0.2])


class TestValueBetSelection:
    def test_edge_above_threshold_selected(self):
        # odds 4.0 → implied 0.25, dyn threshold 0.05*sqrt(4/3)≈0.0577
        sel = value_bet_selection([0.35], [4.0], 0.05)
        assert sel["mask"][0]
        assert sel["edge"][0] == pytest.approx(0.10)
        assert sel["clv"][0] == pytest.approx(1.4)

    def test_edge_below_threshold_rejected(self):
        sel = value_bet_selection([0.28], [4.0], 0.05)
        assert not sel["mask"][0]

    def test_invalid_odds_rejected(self):
        sel = value_bet_selection([0.5, 0.5], [0.0, np.nan], 0.05)
        assert not sel["mask"].any()

    def test_explicit_implied_prob_overrides_raw(self):
        # Raw 1/odds would give edge 0.10 (selected); normalised implied
        # prob of 0.34 gives edge 0.01 (rejected).
        sel = value_bet_selection([0.35], [4.0], 0.05, implied_prob=[0.34])
        assert not sel["mask"][0]


class TestWinSettlement:
    def test_winner_pays_odds_minus_stake(self):
        assert settle_win_bet(5.0, True) == pytest.approx(4.0)

    def test_loser_costs_stake(self):
        assert settle_win_bet(5.0, False) == pytest.approx(-1.0)

    def test_kelly_stake_scales(self):
        assert settle_win_bet(3.0, True, stake=2.5) == pytest.approx(5.0)
        assert settle_win_bet(3.0, False, stake=2.5) == pytest.approx(-2.5)

    def test_vectorised_matches_scalar(self):
        odds = np.array([5.0, 5.0, 2.0])
        won = np.array([True, False, True])
        out = settle_win_bets(odds, won)
        expected = [settle_win_bet(o, w) for o, w in zip(odds, won)]
        assert out == pytest.approx(expected)


class TestEachWayRules:
    def test_band_inclusive(self):
        assert ew_odds_in_band(EW_MIN_ODDS)
        assert ew_odds_in_band(EW_MAX_ODDS)
        assert not ew_odds_in_band(EW_MIN_ODDS - 0.01)
        assert not ew_odds_in_band(EW_MAX_ODDS + 0.01)

    def test_band_rejects_nan(self):
        assert not ew_odds_in_band(float("nan"))

    def test_selection_requires_positive_ev(self):
        assert ew_bet_selected(0.5, 0.1, 4.0, 0.05)
        assert not ew_bet_selected(0.5, -0.1, 4.0, 0.05)
        assert not ew_bet_selected(0.01, 0.1, 4.0, 0.05)

    def test_placed_flag(self):
        assert ew_placed_flag(1, 3) == 1
        assert ew_placed_flag(3, 3) == 1
        assert ew_placed_flag(4, 3) == 0

    def test_non_finishers_never_place(self):
        # finish_position 0 / NaN / None encode DNF, falls, non-runners.
        assert ew_placed_flag(0, 3) == 0
        assert ew_placed_flag(float("nan"), 3) == 0
        assert ew_placed_flag(None, 3) == 0

    def test_winner_pays_both_legs(self):
        # 2 units staked; win odds 10, place odds 3.25
        pnl = settle_ew_bet(10.0, 3.25, won=True, placed=True)
        assert pnl == pytest.approx(10.0 + 3.25 - EW_STAKE_UNITS)

    def test_placed_only_pays_place_leg(self):
        pnl = settle_ew_bet(10.0, 3.25, won=False, placed=True)
        assert pnl == pytest.approx(3.25 - EW_STAKE_UNITS)

    def test_unplaced_loses_both_units(self):
        pnl = settle_ew_bet(10.0, 3.25, won=False, placed=False)
        assert pnl == pytest.approx(-EW_STAKE_UNITS)


class TestWinExchangeSettlement:
    """Win bets struck on the exchange, netted by commission.

    Four research harnesses in ``scripts/`` each carried their own
    ``COMM = 0.02`` and open-coded ``np.where(won, (odds-1)*(1-COMM), -1)``.
    They now call this function, so the walk-forward ROIs they report stay
    comparable with the backtester and live settlement (hard rule #2). These
    tests pin the formula to what those scripts computed, so the consolidation
    cannot silently move a published number.
    """

    @staticmethod
    def _legacy(odds, won, commission=0.02):
        """The expression the scripts used before consolidation."""
        return np.where(
            np.asarray(won).astype(bool),
            (np.asarray(odds, dtype=float) - 1.0) * (1.0 - commission),
            -1.0,
        )

    def test_matches_the_formula_the_scripts_used(self):
        odds = np.array([1.5, 2.0, 4.5, 11.0, 100.0])
        won = np.array([True, False, True, False, True])
        np.testing.assert_allclose(
            settle_win_exchange_bets(odds, won), self._legacy(odds, won),
        )

    def test_winner_keeps_profit_net_of_commission(self):
        # 5.0 decimal => 4.0 profit, less 2% commission on winnings.
        assert settle_win_exchange_bet(5.0, won=True) == pytest.approx(4.0 * 0.98)

    def test_loser_loses_exactly_the_stake(self):
        """Commission applies to winnings only, never to a losing stake."""
        assert settle_win_exchange_bet(5.0, won=False) == pytest.approx(-1.0)
        assert settle_win_exchange_bet(5.0, won=False, stake=2.5) == pytest.approx(-2.5)

    def test_stake_scales_both_outcomes(self):
        assert settle_win_exchange_bet(3.0, True, stake=10.0) == pytest.approx(2.0 * 10 * 0.98)
        odds, won = np.array([3.0, 3.0]), np.array([True, False])
        np.testing.assert_allclose(
            settle_win_exchange_bets(odds, won, stake=np.array([2.0, 4.0])),
            [2.0 * 2.0 * 0.98, -4.0],
        )

    def test_commission_is_overridable_but_shared_by_default(self):
        assert settle_win_exchange_bet(5.0, True, commission=0.0) == pytest.approx(4.0)
        # Win and place markets are charged the same rate; the place alias is
        # kept only so existing callers keep working.
        assert PLACE_EXCHANGE_COMMISSION == EXCHANGE_COMMISSION

    def test_evens_at_commission_is_a_small_loss(self):
        """A fair 2.0 shot is -EV on the exchange: the house edge is real."""
        odds = np.full(10_000, 2.0)
        won = np.arange(10_000) % 2 == 0  # exactly 50% strike rate
        assert settle_win_exchange_bets(odds, won).mean() == pytest.approx(-0.01)


def test_betfair_tick_down_uses_the_increment_below_the_price():
    from src.bet_settlement import betfair_tick_down
    got = betfair_tick_down([1.5, 2.0, 2.02, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 50.0, 1.01])
    assert got.tolist() == [1.49, 1.99, 2.0, 2.98, 3.95, 4.9, 9.8, 14.5, 19.5, 48.0, 1.01]
    assert betfair_tick_down([10.0], ticks=2).tolist() == [9.6]
    # An off-ladder BSP steps down within its band.
    assert betfair_tick_down([7.33]).tolist() == [7.13]


def test_apply_reduction_factor_cuts_decimal_odds_with_a_floor():
    from src.bet_settlement import apply_reduction_factor
    assert apply_reduction_factor([10.0, 1.05], [0.75, 0.5]).tolist() == [7.5, 1.01]
