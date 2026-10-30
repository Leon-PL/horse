import pandas as pd
import pytest

from src.missing_history_snapshot import combine_histories
from src.public_history_features import external_history_features


def records():
    return pd.DataFrame({"horse_id": [1, 2], "race_id": [10, 20], "date": ["2024-01-01"]*2,
        "position": [2, 1], "runner_count": [6, 5], "distance": ["2m", "1m"], "casualty": [{}, {}],
        "eligible_from_date": ["2025-01-01", "2025-03-01"]})


def test_cohort_union_never_counts_a_performance_twice_or_backdates_an_unseen_horse():
    earlier = records()
    later = records().assign(eligible_from_date="2025-04-01")
    combined = combine_histories([later, earlier])
    assert len(combined) == 2
    cards = pd.DataFrame({"horse_id": [1, 2], "race_id": [100, 200], "race_date": pd.to_datetime(["2025-02-01"]*2)})
    out = external_history_features(cards, combined, cards)
    assert out.external_prior_runs.tolist() == [1, 0]


def test_conflicting_source_performances_fail_instead_of_selecting_a_snapshot():
    original = records()
    with pytest.raises(ValueError, match="disagree"):
        combine_histories([original, original.assign(position=3)])
