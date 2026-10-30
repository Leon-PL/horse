import numpy as np
import pandas as pd
import pytest

from src.data_processor import clean_data
from src.verified_race_metadata import apply_verified_runner_ratings
from tests.test_live_parity import _synthetic_raw


def source():
    return pd.DataFrame({"race_id": [622382, 123], "horse_id": [990168, 1],
        "horse_name": ["Mellys Flyer", "Other Horse"], "race_date": ["2021-04-21", "2025-01-01"],
        "track": ["Catterick", "Catterick"], "official_rating": [6., 40.]})


def test_verified_values_replace_corruption_without_affecting_other_low_marks():
    out = apply_verified_runner_ratings(source())
    assert out.official_rating.tolist() == [64., 40.]
    assert "has_official_rating" not in out
    pd.testing.assert_frame_equal(out, apply_verified_runner_ratings(out.copy()))


def test_verified_missing_rating_can_be_observed_when_recleaning():
    f = source().iloc[:1].copy()
    f["horse_id"], f["horse_name"] = 967261, "Hope Probe"
    f["official_rating"], f["has_official_rating"] = 6., 0
    out = apply_verified_runner_ratings(f)
    assert out.official_rating.iloc[0] == 55.
    assert out.has_official_rating.iloc[0] == 1


@pytest.mark.parametrize("column,value", [("race_date", "2021-04-22"), ("track", "York"), ("horse_name", "Different Horse")])
def test_source_identity_guards_are_not_bypassed(column, value):
    f = source()
    f.loc[0, column] = value
    with pytest.raises(ValueError, match="source date/course|identity"):
        apply_verified_runner_ratings(f)


def test_same_named_horse_with_another_id_is_not_changed():
    f = source()
    f.loc[0, "horse_id"] = 999999
    assert apply_verified_runner_ratings(f).official_rating.iloc[0] == 6.


def test_normal_processor_observes_all_other_ratings_and_recleaning_is_identical():
    raw = _synthetic_raw(n_days=1, races_per_day=1, field=4)
    raw["race_id"], raw["race_date"], raw["track"] = 622382, "2021-04-21", "Catterick"
    raw["official_rating"] = 60
    raw.loc[0, ["horse_id", "horse_name", "official_rating"]] = [990168, "Mellys Flyer", 6]
    first = clean_data(raw)
    assert first.official_rating.tolist() == [64, 60, 60, 60]
    assert first.has_official_rating.eq(1).all()
    again = clean_data(first)
    for col in ("official_rating", "has_official_rating"):
        pd.testing.assert_series_equal(first[col], again[col])
