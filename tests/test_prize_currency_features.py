import numpy as np
import pandas as pd
import pytest

from src.prize_currency_features import convert_prizes, native_prize_history, FEATURES


def rates():
    return pd.DataFrame({"observation_date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"]),
                         "gbp_per_eur": [.8, .85, .9]})


def frame():
    return pd.DataFrame({"race_id": [1, 2, 3], "race_date": pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
        "horse_id": [42]*3, "horse_name": ["A"]*3, "horse_prev_races": [0, 1, 2],
        "track": ["IE", "GB", "GB"], "prize_money": [10000., 9000., 10000.]})


def test_converts_gbp_per_eur_in_correct_direction_and_keeps_gbp():
    f = frame()
    np.testing.assert_array_equal(convert_prizes(f, {"IE": "EUR", "GB": "GBP"}, rates()), [8000., 9000., 10000.])


def test_weekend_and_same_day_rates_are_not_looked_ahead():
    f = frame().iloc[[1]].copy()
    f["track"] = "IE"
    f["prize_money"] = 100.
    # Monday must use Friday, although Monday's afternoon observation is present.
    assert convert_prizes(f, {"IE": "EUR"}, rates()).iloc[0] == 85.


def test_future_rate_changes_do_not_change_earlier_prizes():
    f = frame().iloc[[0]]
    a = convert_prizes(f, {"IE": "EUR"}, rates())
    changed = rates(); changed.loc[1:, "gbp_per_eur"] = 1.9
    pd.testing.assert_series_equal(a, convert_prizes(f, {"IE": "EUR"}, changed))


@pytest.mark.parametrize("mapping", [{}, {"IE": "EUR", "GB": "USD"}])
def test_no_currency_guess_for_missing_or_unverified_venues(mapping):
    with pytest.raises(ValueError, match="explicitly verified"):
        convert_prizes(frame(), mapping, rates())


def test_no_future_rate_substitution_when_history_is_unavailable():
    f = frame().iloc[[0]].copy(); f["race_date"] = pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError, match="earlier published"):
        convert_prizes(f, {"IE": "EUR"}, rates())


def test_history_excludes_current_prize_and_fixes_false_cross_currency_drop():
    f = frame()
    corrected = native_prize_history(f, convert_prizes(f, {"IE": "EUR", "GB": "GBP"}, rates()))
    # A debutant has no history: missing, not 0.
    np.testing.assert_array_equal(corrected.horse_max_prize_contested, [np.nan, 8000., 9000.])
    np.testing.assert_array_equal(corrected.horse_avg_prize_contested, [np.nan, 8000., 8500.])
    # The change is a log ratio, not a £ difference.
    np.testing.assert_allclose(corrected.prize_money_change_vs_avg,
                               [np.nan, np.log1p(9000.) - np.log1p(8000.), np.log1p(10000.) - np.log1p(8500.)])
    original = native_prize_history(f, f.prize_money)
    assert original.prize_money_change_vs_avg.iloc[1] == pytest.approx(np.log1p(9000.) - np.log1p(10000.))
    assert original.prize_money_change_vs_avg.iloc[1] < 0 < corrected.prize_money_change_vs_avg.iloc[1]


def test_reconstruction_is_stable_under_row_permutation():
    f = frame(); shuffled = f.iloc[[2, 0, 1]]
    a = native_prize_history(f, f.prize_money)
    b = native_prize_history(shuffled, shuffled.prize_money).reindex(a.index)
    pd.testing.assert_frame_equal(a, b)


def test_preserves_zero_prizes_and_prior_start_denominator():
    f = frame(); f["prize_money"] = [0., 100., 0.]
    result = native_prize_history(f, f.prize_money)
    np.testing.assert_array_equal(result.horse_avg_prize_contested, [np.nan, 0., 50.])
    # An unknown (zero) purse has no ratio to the horse's typical purse.
    assert np.isnan(result.prize_money_change_vs_avg.iloc[2])
    assert tuple(result.columns) == FEATURES


@pytest.mark.parametrize("bad", [0., -1., np.nan])
def test_rejects_invalid_reference_rates(bad):
    r = rates(); r.loc[0, "gbp_per_eur"] = bad
    with pytest.raises(ValueError, match="positive"):
        convert_prizes(frame(), {"IE": "EUR", "GB": "GBP"}, r)
