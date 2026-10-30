import numpy as np
import pandas as pd
import pytest

from src.betfair import bsp_implied_prob, market_prob_bsp_preferred


def test_partial_bsp_uses_sp_for_the_entire_field():
    f = pd.DataFrame({"race_id": [1]*3, "bsp": [2, 4, np.nan], "odds": [3, 4, 6]})
    expected = np.array([1/3, 1/4, 1/6]); expected /= expected.sum()
    np.testing.assert_allclose(market_prob_bsp_preferred(f), expected)
    assert bsp_implied_prob(f.bsp, f.race_id).tolist() == pytest.approx([np.nan]*3, nan_ok=True)


def test_complete_bsp_is_preferred_and_index_does_not_matter():
    f = pd.DataFrame({"race_id": [1]*3, "bsp": [2, 4, 8], "odds": [3, 4, 6]}, index=[10, 2, 7])
    np.testing.assert_allclose(market_prob_bsp_preferred(f), [4/7, 2/7, 1/7])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0., 1., -2.])
def test_no_complete_source_produces_no_teacher(invalid):
    f = pd.DataFrame({"race_id": [1]*3, "bsp": [2, 4, invalid], "odds": [3, invalid, 6]})
    assert np.isnan(market_prob_bsp_preferred(f)).all()


def test_missing_runners_and_inconsistent_declarations_are_rejected():
    f = pd.DataFrame({"race_id": [1, 1], "bsp": [2, 4], "odds": [3, 5], "num_runners": [3, 3]})
    assert np.isnan(market_prob_bsp_preferred(f)).all()
    f["num_runners"] = [2, 3]
    assert np.isnan(market_prob_bsp_preferred(f)).all()


def test_interleaved_races_choose_their_own_complete_source():
    f = pd.DataFrame({"race_id": [1, 2, 1, 2], "bsp": [2, 2, np.nan, 3], "odds": [3, 5, 6, 5]})
    p = market_prob_bsp_preferred(f)
    np.testing.assert_allclose(p, [2/3, .6, 1/3, .4])
    np.testing.assert_allclose(pd.Series(p).groupby(f.race_id).sum(), 1)


def test_absent_price_columns_do_not_break_hard_label_training():
    f = pd.DataFrame({"race_id": [1, 1]})
    assert np.isnan(market_prob_bsp_preferred(f)).all()
    f["bsp"] = [2, 4]
    np.testing.assert_allclose(market_prob_bsp_preferred(f), [2/3, 1/3])
