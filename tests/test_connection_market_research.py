from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.connection_market_research import FEATURES, complete_market_residuals, connection_market_features


def field(day, identity, *, prices=(2., 2.), trainers=("Alpha", "Beta"), jockeys=("One", "Two"),
          positions=(1, 2)):
    return pd.DataFrame({"race_id": str(identity), "race_date": pd.Timestamp(day),
        "horse_name": [f"Horse{identity}a", f"Horse{identity}b"], "num_runners": 2,
        "trainer": trainers, "jockey": jockeys, "finish_position": positions,
        "won": [int(p == 1) for p in positions], "odds": prices, "finish_pos_label": ""})


def test_known_win_expectations_and_shrinkage():
    first = field("2025-01-01", 1, prices=(1.25, 5.))
    next_day = field("2025-01-02", 2)
    frame = pd.concat([first, next_day], ignore_index=True)
    obs = complete_market_residuals(first)
    np.testing.assert_allclose(obs.probability, [.8, .2])
    np.testing.assert_allclose(obs.residual, [.2, -.2])
    result = connection_market_features(frame)
    assert result.iloc[:2][list(FEATURES[:2])].isna().all().all()
    np.testing.assert_allclose(result.iloc[2:][FEATURES[0]], [.2/21, -.2/21])
    np.testing.assert_allclose(result.iloc[2:][FEATURES[2]], np.log(2))


def test_same_day_prices_and_outcomes_cannot_change_that_days_features():
    frame = pd.concat([field("2025-01-01", 1), field("2025-01-02", 2),
                       field("2025-01-02", 3)], ignore_index=True)
    before = connection_market_features(frame)
    changed = frame.copy()
    changed.loc[changed.race_date.eq("2025-01-02"), "odds"] = [10., 1.1, 2., 20.]
    changed.loc[changed.race_date.eq("2025-01-02"), "won"] = [0, 1, 0, 1]
    changed.loc[changed.race_date.eq("2025-01-02"), "finish_position"] = [2, 1, 2, 1]
    pd.testing.assert_frame_equal(before, connection_market_features(changed))


def test_whole_book_missing_prices_do_not_become_neutral_results():
    frame = pd.concat([field("2025-01-01", 1, prices=(2., np.nan)),
                       field("2025-01-02", 2)], ignore_index=True)
    assert complete_market_residuals(frame.iloc[:2]).isna().all().all()
    result = connection_market_features(frame)
    assert result[list(FEATURES[:2])].isna().all().all()
    assert result[list(FEATURES[2:])].eq(0).all().all()


def test_expiry_boundary_and_no_history_are_missing_not_zero():
    frame = pd.concat([field("2025-01-01", 1), field("2025-03-02", 2),
                       field("2025-03-03", 3, trainers=("New", "Beta"))], ignore_index=True)
    result = connection_market_features(frame)
    assert result.iloc[2][FEATURES[2]] == pytest.approx(np.log(2))  # 60 days inclusive
    assert result.iloc[4][FEATURES[2]] == 0
    assert np.isnan(result.iloc[4][FEATURES[0]])
    assert result.iloc[5][FEATURES[2]] == pytest.approx(np.log(2))


def test_multiple_entries_and_field_mass_are_counted_correctly():
    first = field("2025-01-01", 1, trainers=("Same", "Same"))
    next_day = field("2025-01-02", 2, trainers=("Same", "Other"))
    out, seed = connection_market_features(pd.concat([first, next_day], ignore_index=True), return_state=True)
    assert out.iloc[2][FEATURES[2]] == pytest.approx(np.log(3))
    assert out.iloc[2][FEATURES[0]] == pytest.approx(0.)  # all earlier win/book mass belongs to Same
    assert seed["history"][("trainer", "same")][0][3] == 2
    assert len(seed["history"][("trainer", "same")]) == 2


def test_unknown_connections_are_not_pooled():
    first = field("2025-01-01", 1, trainers=(None, "unknown"))
    next_day = field("2025-01-02", 2, trainers=("", "N/A"))
    result, seed = connection_market_features(pd.concat([first, next_day], ignore_index=True), return_state=True)
    assert result[FEATURES[0]].isna().all() and result[FEATURES[2]].eq(0).all()
    assert not any(kind == "trainer" for kind, name in seed["history"])


def test_prefix_append_permutation_and_caller_state_are_exact():
    frame = pd.concat([field("2025-01-01", 1), field("2025-01-02", 2),
                       field("2025-01-03", 3)], ignore_index=True)
    full = connection_market_features(frame)
    prefix, seed = connection_market_features(frame.iloc[:2], return_state=True)
    before = deepcopy(seed)
    pd.testing.assert_frame_equal(connection_market_features(frame.iloc[2:], state=seed), full.iloc[2:])
    pd.testing.assert_frame_equal(prefix, full.iloc[:2])
    assert seed == before
    pd.testing.assert_frame_equal(connection_market_features(frame.sample(frac=1, random_state=4)).sort_index(), full)
    with pytest.raises(ValueError, match="before"):
        connection_market_features(frame.iloc[:2], state=seed)


def test_pending_cards_never_advance_or_expire_seed_state():
    _, seed = connection_market_features(field("2025-01-01", 1), return_state=True)
    before = deepcopy(seed)
    card = field("2025-04-01", 2, positions=(0, 0))
    result, after = connection_market_features(card, state=seed, return_state=True)
    assert result[list(FEATURES[:2])].isna().all().all()
    assert seed == before == after


def test_nonfinishers_in_complete_fields_contribute_zero_wins():
    first = field("2025-01-01", 1, positions=(1, 0))
    first.loc[1, "finish_pos_label"] = "F"
    frame = pd.concat([first, field("2025-01-02", 2)], ignore_index=True)
    result = connection_market_features(frame)
    np.testing.assert_allclose(result.iloc[2:][FEATURES[0]], [.5/21, -.5/21])
