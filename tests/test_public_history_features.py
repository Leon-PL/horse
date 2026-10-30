import numpy as np
import pandas as pd
import pytest

from src.public_history_features import external_history_features


def data():
    cards = pd.DataFrame({"horse_id": [1, 1, 1, 2], "race_id": [10, 11, 12, 13],
        "race_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-05", "2025-01-05"]),
        "distance_furlongs": 8.})
    prior = pd.DataFrame({"horse_id": [1, 1, 1], "race_id": [1, 2, 3],
        "date": ["2025-01-01", "2025-01-03", "2025-01-04"],
        "position": [1, 4, 2], "runner_count": [5, 5, 5], "distance": ["1m", "7f", "1m 2f"],
        "casualty": [{}, {}, {}]})
    return cards, prior


def test_strict_date_known_form_and_unknown_history():
    cards, prior = data()
    out = external_history_features(cards, prior, cards)
    np.testing.assert_array_equal(out.external_prior_runs, [1, 1, 3, 0])
    np.testing.assert_allclose(out.external_prior_quality_last.iloc[:3], [1, 1, .75])
    np.testing.assert_allclose(out.external_prior_quality_mean_3.iloc[:3], [1, 1, 2/3])
    assert out.external_prior_distance_change.iloc[2] == -2
    assert out.iloc[3].drop("external_prior_runs").isna().all()


def test_local_records_are_never_counted_twice():
    cards, prior = data()
    local = pd.concat([cards, pd.DataFrame({"horse_id": [1], "race_id": [1]})], ignore_index=True)
    out = external_history_features(cards, prior, local)
    np.testing.assert_array_equal(out.external_prior_runs, [0, 0, 2, 0])


def test_future_outcomes_and_appends_do_not_change_earlier_features():
    cards, prior = data()
    earlier = cards.iloc[:2]
    before = external_history_features(earlier, prior, cards)
    changed = prior.copy()
    changed.loc[changed.date.ge("2025-01-03"), "position"] = 1
    changed = pd.concat([changed, changed.iloc[[0]].assign(race_id=99, date="2026-01-01")])
    pd.testing.assert_frame_equal(before, external_history_features(earlier, changed, cards))


def test_permutation_and_string_numeric_identity_are_equivalent():
    cards, prior = data()
    expected = external_history_features(cards, prior, cards)
    shuffled = cards.sample(frac=1, random_state=2).copy()
    shuffled.horse_id = shuffled.horse_id.astype(str)
    actual = external_history_features(shuffled, prior.sample(frac=1, random_state=3), cards).sort_index()
    pd.testing.assert_frame_equal(actual, expected)


def test_unknown_and_nonrunner_positions_are_not_invented_losers():
    cards, prior = data()
    prior["position"] = 0
    prior["casualty"] = [{"code": "NR"}, {}, {"code": "PU"}]
    out = external_history_features(cards, prior, cards)
    np.testing.assert_array_equal(out.external_prior_runs, [0, 0, 1, 0])
    assert out.external_prior_quality_last.iloc[2] == 0


@pytest.mark.parametrize("reason", ["PulledUp", "Fell", "UnseatedRider", "Refused"])
def test_structured_source_nonfinish_is_a_known_start_not_missing_history(reason):
    cards, prior = data()
    prior["position"] = np.nan
    prior["casualty"] = [{"reason": "NonRunner"}, {}, {"reason": reason}]
    out = external_history_features(cards, prior, cards)
    np.testing.assert_array_equal(out.external_prior_runs, [0, 0, 1, 0])
    assert out.external_prior_quality_last.iloc[2] == 0
    assert out.external_prior_days_since_last.iloc[2] == 1


def test_duplicate_conflicts_raise_and_missing_distance_stays_unknown():
    cards, prior = data()
    conflict = pd.concat([prior, prior.iloc[[0]].assign(position=3)])
    with pytest.raises(ValueError, match="Conflicting"):
        external_history_features(cards, conflict, cards)
    prior.distance = "unknown"
    assert external_history_features(cards, prior, cards).external_prior_distance_change.isna().all()


def test_later_cohort_eligibility_cannot_appear_on_earlier_local_rows():
    cards, prior = data()
    prior["eligible_from_date"] = "2025-01-04"
    out = external_history_features(cards, prior, cards)
    np.testing.assert_array_equal(out.external_prior_runs, [0, 0, 3, 0])
    assert out.iloc[:2].drop(columns="external_prior_runs").isna().all().all()
