"""A schema pilot must expose errors without changing racing semantics."""
import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pandera.pandas")

from src.data_contracts import validate_settled_races  # noqa: E402


def settled_field():
    return pd.DataFrame({
        "race_id": [101, 101, 101],
        "horse_name": ["Alpha", "Beta", "Gamma"],
        "race_date": pd.to_datetime(["2026-01-01"] * 3),
        "finish_position": [1, 2, 0],
        "won": [1, 0, 0],
        "num_runners": [3, 3, 3],
        "finish_pos_label": ["1", "2", "PU"],
        "optional_form": [np.nan, .2, .7],
    })


def test_nonfinishers_and_unknown_history_are_preserved():
    frame = settled_field()
    before = frame.copy(deep=True)
    result = validate_settled_races(frame)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result, before)


@pytest.mark.parametrize("fault", ["partial", "duplicate", "pending", "two_winners",
                                   "missing_column", "negative_position", "bad_date"])
def test_rejects_malformed_fields_without_fixing_or_mutating_them(fault):
    frame = settled_field()
    if fault == "partial":
        frame = frame.iloc[:2].copy()
    elif fault == "duplicate":
        frame.loc[1, "horse_name"] = "ALPHA"
    elif fault == "pending":
        frame[["won", "finish_position"]] = 0
    elif fault == "two_winners":
        frame.loc[1, ["won", "finish_position"]] = [1, 1]
    elif fault == "missing_column":
        frame = frame.drop(columns="num_runners")
    elif fault == "negative_position":
        frame.loc[1, "finish_position"] = -1
    elif fault == "bad_date":
        frame["race_date"] = ["bad", "2026-01-01", "2026-01-01"]
    before = frame.copy(deep=True)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_settled_races(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_empty_frame_is_not_a_successful_audit():
    with pytest.raises(ValueError, match="at least one"):
        validate_settled_races(settled_field().iloc[:0])
