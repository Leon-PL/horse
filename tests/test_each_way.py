"""Tests for each-way terms, value calculation and Kelly sizing."""

import pandas as pd
import pytest

from src.each_way import (
    ew_value,
    ew_value_bets,
    get_ew_terms,
    kelly_ew,
    place_odds_decimal,
)


class TestEwTerms:
    @pytest.mark.parametrize(
        "runners,is_handicap,places,eligible",
        [
            (4, False, 0, False),
            (5, False, 2, True),
            (7, False, 2, True),
            (8, False, 3, True),
            (15, False, 3, True),
            (16, False, 3, True),
            (16, True, 4, True),
        ],
    )
    def test_standard_uk_terms(self, runners, is_handicap, places, eligible):
        terms = get_ew_terms(runners, is_handicap=is_handicap)
        assert terms.eligible is eligible
        assert terms.places_paid == places

    def test_place_odds(self):
        # 10/1 at 1/4 terms → place leg pays 10/4 → decimal 3.5
        assert place_odds_decimal(11.0, 0.25) == pytest.approx(3.5)


class TestEwValue:
    def test_positive_place_edge(self):
        terms = get_ew_terms(12)
        result = ew_value(win_prob=0.10, place_prob=0.45, win_odds=11.0, ew_terms=terms)
        # implied place prob = 1/3.5 ≈ 0.2857 → edge ≈ 0.1643
        assert result["place_odds"] == pytest.approx(3.5)
        assert result["place_edge"] == pytest.approx(0.45 - 1 / 3.5)
        assert result["place_value"]

    def test_ineligible_race_returns_zeros(self):
        terms = get_ew_terms(4)
        result = ew_value(0.3, 0.6, 5.0, terms)
        assert result["ew_ev"] == 0.0
        assert not result["ew_value"]


class TestKellyEw:
    def test_matches_manual_kelly(self):
        terms = get_ew_terms(12)
        out = kelly_ew(win_prob=0.2, place_prob=0.5, win_odds=6.0, ew_terms=terms, fraction=1.0)
        # win leg: b=5, k=(5*0.2-0.8)/5 = 0.04
        assert out["win_kelly"] == pytest.approx(0.04)
        # place leg: p_odds=2.25, b=1.25, k=(1.25*0.5-0.5)/1.25 = 0.1
        assert out["place_kelly"] == pytest.approx(0.1)

    def test_quarter_kelly_scaling(self):
        terms = get_ew_terms(12)
        full = kelly_ew(0.2, 0.5, 6.0, terms, fraction=1.0)
        quarter = kelly_ew(0.2, 0.5, 6.0, terms, fraction=0.25)
        assert quarter["ew_kelly"] == pytest.approx(full["ew_kelly"] * 0.25)


class TestEwValueBets:
    def test_filters_band_and_edge(self):
        df = pd.DataFrame(
            {
                "horse_name": ["A", "B", "C", "D"],
                "ew_eligible": [True, True, True, True],
                "place_edge": [0.30, 0.30, 0.005, 0.30],
                "place_value": [True, True, True, False],
                "odds": [10.0, 2.0, 10.0, 10.0],
            }
        )
        out = ew_value_bets(df, min_place_edge=0.05)
        # B fails the odds band, C fails the edge, D fails place_value
        assert list(out["horse_name"]) == ["A"]
