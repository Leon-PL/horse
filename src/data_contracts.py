"""Opt-in settled-field validation; never fills, coerces or drops input rows.

The existing complete-field rule remains the semantic authority. This schema
provides reusable column checks and collected errors for audits, without
changing the frozen training or serving imports.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from src.forecast_evaluation import valid_race_mask

RESULT_COLUMNS = [
    "race_id", "horse_name", "race_date", "finish_position", "won", "num_runners",
]


def _nonempty(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype("string").str.strip().ne("").fillna(False)


def _nonnegative_integer(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce")
    return numbers.ge(0) & numbers.mod(1).eq(0)


def settled_race_schema() -> pa.DataFrameSchema:
    """Allow extra feature columns, including legitimately missing history."""
    return pa.DataFrameSchema(
        {
            "race_id": pa.Column(checks=pa.Check(_nonempty), nullable=False),
            "horse_name": pa.Column(str, checks=pa.Check(_nonempty), nullable=False),
            "race_date": pa.Column(
                checks=pa.Check(lambda s: pd.to_datetime(s, errors="coerce").notna()),
                nullable=False,
            ),
            "finish_position": pa.Column(
                checks=pa.Check(_nonnegative_integer), nullable=False,
            ),
            "won": pa.Column(checks=pa.Check.isin([0, 1]), nullable=False),
            "num_runners": pa.Column(
                checks=[pa.Check(_nonnegative_integer),
                        pa.Check(lambda s: pd.to_numeric(s, errors="coerce").ge(2))],
                nullable=False,
            ),
        },
        checks=pa.Check(valid_race_mask, name="complete_single_winner_fields"),
        strict=False,
        coerce=False,
        add_missing_columns=False,
        drop_invalid_rows=False,
        name="settled_complete_races",
    )


def validate_settled_races(frame: pd.DataFrame) -> pd.DataFrame:
    """Collect structural/field errors and return an unchanged valid copy.

    Explicit non-finishers with position zero are retained as actual starts.
    Pending cards, partial fields and ambiguous identities are rejected.
    """
    if frame.empty:
        raise ValueError("Expected at least one complete settled field")
    return settled_race_schema().validate(frame, lazy=True, inplace=False)
