"""Tests for the market-anchor (Benter combination) maths."""

import numpy as np
import pytest

from src.model import apply_market_anchor, fit_market_anchor, market_probs_from_odds


def _grouped_normalise(p, groups):
    out = []
    off = 0
    for g in groups:
        seg = p[off:off + g]
        out.append(seg / seg.sum())
        off += g
    return np.concatenate(out)


class TestMarketProbsFromOdds:
    def test_normalised_per_race(self):
        odds = np.array([2.0, 4.0, 4.0, 3.0, 6.0])
        groups = np.array([3, 2])
        mkt, valid = market_probs_from_odds(odds, groups)
        assert valid.all()
        assert mkt[:3].sum() == pytest.approx(1.0)
        assert mkt[3:].sum() == pytest.approx(1.0)
        # shorter odds -> higher prob
        assert mkt[0] > mkt[1]

    def test_race_with_missing_odds_flagged_invalid(self):
        odds = np.array([2.0, np.nan, 3.0, 6.0])
        groups = np.array([2, 2])
        _, valid = market_probs_from_odds(odds, groups)
        assert not valid[0]
        assert valid[1]


class TestApplyMarketAnchor:
    def test_pure_market_recovers_market(self):
        groups = np.array([3, 4])
        rng = np.random.default_rng(0)
        mdl = _grouped_normalise(rng.uniform(0.05, 0.9, 7), groups)
        mkt = _grouped_normalise(rng.uniform(0.05, 0.9, 7), groups)
        out = apply_market_anchor(mdl, mkt, groups, alpha=0.0, beta=1.0)
        assert out == pytest.approx(mkt, abs=1e-9)

    def test_pure_model_recovers_model(self):
        groups = np.array([3, 4])
        rng = np.random.default_rng(1)
        mdl = _grouped_normalise(rng.uniform(0.05, 0.9, 7), groups)
        mkt = _grouped_normalise(rng.uniform(0.05, 0.9, 7), groups)
        out = apply_market_anchor(mdl, mkt, groups, alpha=1.0, beta=0.0)
        assert out == pytest.approx(mdl, abs=1e-6)

    def test_invalid_race_keeps_original(self):
        groups = np.array([2, 2])
        mdl = np.array([0.7, 0.3, 0.6, 0.4])
        mkt = np.array([0.5, 0.5, 0.5, 0.5])
        valid = np.array([False, True])
        out = apply_market_anchor(mdl, mkt, groups, 0.0, 1.0, race_valid=valid)
        assert out[:2] == pytest.approx(mdl[:2])
        assert out[2:] == pytest.approx(mkt[2:])


class TestFitMarketAnchor:
    def test_recovers_market_when_winners_follow_market(self):
        rng = np.random.default_rng(42)
        groups_list, mdl_parts, mkt_parts, won_parts = [], [], [], []
        for _ in range(2000):
            n = int(rng.integers(5, 12))
            mkt = rng.dirichlet(np.ones(n) * 2)
            mdl = rng.dirichlet(np.ones(n) * 2)  # uninformative model
            winner = rng.choice(n, p=mkt)
            won = np.zeros(n, dtype=int)
            won[winner] = 1
            groups_list.append(n)
            mkt_parts.append(mkt)
            mdl_parts.append(mdl)
            won_parts.append(won)
        groups = np.array(groups_list)
        anchor = fit_market_anchor(
            np.clip(np.concatenate(mdl_parts), 1e-6, 1 - 1e-6),
            np.clip(np.concatenate(mkt_parts), 1e-6, 1 - 1e-6),
            groups,
            np.concatenate(won_parts),
        )
        assert anchor is not None
        # The model carries no information, so it should get ~zero weight
        # and the market ~full weight.
        assert abs(anchor["alpha"]) < 0.1
        assert 0.85 < anchor["beta"] < 1.15

    def test_too_few_races_returns_none(self):
        groups = np.array([3] * 10)
        p = np.full(30, 1 / 3)
        won = np.tile([1, 0, 0], 10)
        assert fit_market_anchor(p, p, groups, won) is None
