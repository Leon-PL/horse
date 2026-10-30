"""Query equivalence is required before comparing speed or memory."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("psutil")

from scripts.benchmark_data_queries import (  # noqa: E402
    _since, assert_equivalent, summarize_duckdb, summarize_pandas,
)


def test_filtered_summary_matches_with_nulls_missing_dates_and_unordered_rows(tmp_path):
    frame = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-03",
                                     "2026-01-04", None]),
        "race_id": [5, 3, 5, 6, 7],
        "rating": [10., 999., np.nan, 30., 100.],
    })
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    before = path.read_bytes()
    left = summarize_pandas(path, "2026-01-03", ["rating"])
    right = summarize_duckdb(path, "2026-01-03", ["rating"])
    assert_equivalent(left, right)
    assert left.runners.tolist() == [2, 1]
    assert left.mean_rating.tolist() == [10., 30.]
    assert _since(path, 2) == "2026-01-03"
    assert path.read_bytes() == before


def test_query_identifiers_are_quoted_instead_of_interpolated_as_sql(tmp_path):
    name = 'rating"with quote'
    frame = pd.DataFrame({"race_date": pd.to_datetime(["2026-01-01"]),
                          "race_id": [1], name: [2.]})
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    assert_equivalent(summarize_pandas(path, "2026-01-01", [name]),
                      summarize_duckdb(path, "2026-01-01", [name]))
