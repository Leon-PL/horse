"""Stalls position and rail movements from BHA going reports (course_reports)."""

import numpy as np
import pandas as pd
import pytest

from src import course_reports as cr


@pytest.mark.parametrize("text, races", [
    # Distances first, the common form.
    ("5f & 6f - Stands side; 1m - Centre; Remainder - Inside",
     [(5.0, "stands"), (6.04, "stands"), (8.1, "centre"), (12.0, "inside")]),
    ("7f 36y outside; remainder inside", [(7.16, "outside"), (9.47, "inside")]),
    ("1m 2f - centre; 51/2f - outside; remainder - inside rail.",
     [(5.5, "outside"), (10.0, "centre"), (7.0, "inside")]),
    ("5,6,7,1m Inner, 1m2f Outer", [(6.0, "inside"), (10.32, "outside")]),
    # Newmarket names the course in use; it is not a stalls position.
    ("Far Side Course 10f Centre Remainder Far Side", [(10.0, "centre"), (6.0, "far")]),
    # Position first, and "except".
    ("Centre: 5f, 6f, 1m2f & 1m4f Inside: 7f & 1m",
     [(6.0, "centre"), (7.0, "inside"), (12.0, "centre")]),
    ("Inside, except 1m 2f and 2m 2f Outside",
     [(10.0, "outside"), (8.0, "inside"), (18.0, "outside")]),
    ("Mid for 5f Inner for remainder", [(5.0, "centre"), (11.1, "inside")]),
    # Single position for the whole card.
    ("All: Inside", [(7.0, "inside"), (16.0, "inside")]),
    ("Inside all races", [(9.0, "inside")]),
])
def test_parse_and_assign(text, races):
    parsed = cr.parse_stalls(text)
    for dist, expected in races:
        assert cr.stalls_for_race(parsed, dist, "unknown course") == expected, (text, dist)


def test_straight_and_round_groups_use_the_course_straight():
    # Ascot's straight course runs to a mile; York's to 6f.
    parsed = cr.parse_stalls("Straight - Centre Round - Inner")
    assert cr.stalls_for_race(parsed, 8.0, "ascot") == "centre"
    assert cr.stalls_for_race(parsed, 8.0, "york") == "inside"
    assert cr.stalls_for_race(parsed, 12.0, "ascot") == "inside"


def test_group_followed_by_distance_is_a_qualifier_unless_joined():
    qualified = cr.parse_stalls("Straight Course: 5F: Stands side, 1m & 6F: Centre. Round Course: Inside")
    assert cr.stalls_for_race(qualified, 7.0, "ascot") is None  # straight 7f never named
    assert cr.stalls_for_race(qualified, 5.0, "ascot") == "stands"
    joined = cr.parse_stalls("Straight Course and 2 miles - Stand side Round Course - Inside rail.")
    assert cr.stalls_for_race(joined, 5.0, "ripon") == "stands"
    assert cr.stalls_for_race(joined, 16.0, "ripon") == "stands"
    assert cr.stalls_for_race(joined, 10.0, "ripon") == "inside"


def test_back_straight_starts_are_round_course():
    parsed = cr.parse_stalls("Straight course: Centre Back Straight starts: Inside")
    assert cr.stalls_for_race(parsed, 6.0, "newbury") == "centre"
    assert cr.stalls_for_race(parsed, 10.0, "newbury") == "inside"


def test_silent_report_gives_none():
    parsed = cr.parse_stalls("5f, 6f and 1m - Centre 1m 2f - Inside")
    assert cr.stalls_for_race(parsed, 14.0, "nottingham") is None
    assert cr.stalls_for_race([], 6.0, "york") is None
    assert cr.parse_stalls(None) == [] and cr.parse_stalls(float("nan")) == []


def _results():
    return pd.DataFrame({
        "race_id": ["A", "A", "B", "C", "D"],
        "race_date": ["2025-05-07"] * 4 + ["2025-05-08"],
        "off_time": ["13:30", "13:30", "14:05", "15:10", "13:30"],
        "track": ["Chester", "Chester", "Chester", "Chester", "Cork"],
        "race_type": ["Flat", "Flat", "Flat", "Hurdle", "Flat"],
        "distance_furlongs": [5.07, 5.07, 10.32, 16.0, 6.0],
    })


def _write_reports(tmp_path, monkeypatch):
    going = pd.DataFrame({
        "race_date": ["2025-05-07"], "course": ["Chester"], "fixture_type": ["FLAT"],
        "stalls": ["1m 2f - Outside Remainder - Inside"], "abandoned": [0],
    })
    races = pd.DataFrame({
        "race_date": ["2025-05-07", "2025-05-07", "2025-05-07"],
        "race_time": ["13:30:00", "14:05:00", "15:10:00"],
        "course": ["Chester"] * 3, "distance_change_yards": [20.0, 26.0, 0.0],
        "abandoned": [0, 0, 0],
    })
    going.to_parquet(tmp_path / "going.parquet", index=False)
    races.to_parquet(tmp_path / "races.parquet", index=False)
    monkeypatch.setattr(cr, "GOING_PATH", str(tmp_path / "going.parquet"))
    monkeypatch.setattr(cr, "RACES_PATH", str(tmp_path / "races.parquet"))


def test_attach_joins_by_course_day_and_off_time(tmp_path, monkeypatch):
    _write_reports(tmp_path, monkeypatch)
    out = cr.attach_course_reports(_results())
    assert out["stalls_position"].tolist()[:3] == ["inside", "inside", "outside"]
    assert pd.isna(out.loc[3, "stalls_position"])  # jumps: no stalls
    assert pd.isna(out.loc[4, "stalls_position"])  # Ireland: no BHA report
    assert out["rail_distance_change_yards"].tolist()[:4] == [20.0, 20.0, 26.0, 0.0]
    assert np.isnan(out.loc[4, "rail_distance_change_yards"])


def test_attach_without_files_adds_missing_columns():
    out = cr.attach_course_reports(_results())  # conftest hides the local files
    assert out["stalls_position"].isna().all()
    assert out["rail_distance_change_yards"].isna().all()
    assert len(out) == 5
