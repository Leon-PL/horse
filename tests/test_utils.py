"""Tests for shared utility helpers (merge keys, Kelly, dtype compaction)."""

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    compact_numeric_dtypes,
    kelly_criterion,
    normalise_horse_key,
    normalise_off_time_key,
    normalise_track_key,
)


class TestMergeKeyNormalisers:
    def test_horse_key_title_case_and_whitespace(self):
        assert normalise_horse_key("  RED   rum ") == "Red Rum"

    def test_track_key_lower_case(self):
        assert normalise_track_key(" Kempton  Park ") == "kempton park"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1408", "14:08"),
            ("14:08", "14:08"),
            ("14:08:00", "14:08"),
            ("2:05", "02:05"),
            ("1408 BST", "14:08"),
            ("nan", ""),
            (None, ""),
        ],
    )
    def test_off_time_key(self, raw, expected):
        assert normalise_off_time_key(raw) == expected


class TestKellyCriterion:
    def test_positive_edge(self):
        # p=0.5 at evens+ (odds 3.0): b=2, k=(2*0.5-0.5)/2 = 0.25
        assert kelly_criterion(0.5, 3.0, fraction=1.0) == pytest.approx(0.25)

    def test_negative_edge_clamped_to_zero(self):
        assert kelly_criterion(0.1, 2.0, fraction=1.0) == 0.0

    def test_fractional_kelly(self):
        assert kelly_criterion(0.5, 3.0, fraction=0.25) == pytest.approx(0.0625)


class TestCompactNumericDtypes:
    def test_downcasts_floats_and_ints(self):
        df = pd.DataFrame(
            {
                "f": np.array([1.5, 2.5], dtype=np.float64),
                "i": np.array([-5, 10], dtype=np.int64),
                "u": np.array([5, 10], dtype=np.int64),
                "s": ["a", "b"],
            }
        )
        out = compact_numeric_dtypes(df, label="test")
        assert out["f"].dtype == np.float32
        assert out["i"].dtype.itemsize < 8
        assert out["u"].dtype.kind == "u"
        assert out["s"].dtype == object

    def test_values_preserved(self):
        df = pd.DataFrame({"f": [1.5, -2.25], "i": [3, -4]})
        out = compact_numeric_dtypes(df.copy())
        assert out["f"].tolist() == [1.5, -2.25]
        assert out["i"].tolist() == [3, -4]

    def test_none_and_empty_passthrough(self):
        assert compact_numeric_dtypes(None) is None
        empty = pd.DataFrame()
        assert compact_numeric_dtypes(empty) is empty
