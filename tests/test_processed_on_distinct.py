"""Cleaning by distinct value and finding corrected events give the row-by-row result."""
import numpy as np
import pandas as pd

from src import data_processor as dp
from src import verified_race_metadata as vm


def test_on_distinct_repeats_the_row_result_including_missing_values():
    values = pd.Series(["2-1", np.nan, "2-1", "P/3", np.nan], index=[5, 6, 7, 8, 9], name="form_str", dtype=object)
    out = dp._on_distinct(values, lambda distinct: distinct.apply(lambda x: f"<{x}>"))
    assert out.tolist() == [f"<{x}>" for x in values]
    assert out.index.equals(values.index) and out.name == "form_str"


def test_on_distinct_keeps_the_inferred_type():
    values = pd.Series(["21", "4", "21"], dtype=object)
    assert dp._on_distinct(values, lambda distinct: distinct.apply(dp._parse_form_avg)).dtype == np.float64
    assert dp._on_distinct(values, lambda distinct: distinct.apply(dp._parse_form_wins)).dtype == np.int64


def test_person_names_clean_as_before():
    names = pd.Series(["j.p. spencer", " J.P  Spencer ", None, "a.b.", "j.p. spencer"], index=[3, 4, 5, 6, 7])
    expected = (names.astype(str).str.strip().str.title()
                .str.replace(r"\.(?=[A-Z])", " ", regex=True).str.replace(r"\.\s*", " ", regex=True)
                .str.replace(r"\s{2,}", " ", regex=True).str.strip().str.title())
    pd.testing.assert_series_equal(dp._normalise_person_names(names), expected)


def test_rows_by_event_lists_positions_in_first_seen_order():
    identifiers = pd.Series(["9", "4", "9", "7", "4"], index=[1, 1, 2, 3, 3], dtype="string")
    known = identifiers.isin(["9", "4"])
    rows = vm._rows_by_event(identifiers, known)
    assert list(rows) == ["9", "4"]
    assert rows["9"].tolist() == [0, 2] and rows["4"].tolist() == [1, 4]
