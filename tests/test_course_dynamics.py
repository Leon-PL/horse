"""Tests for add_course_dynamics_features (src/feature_engineer.py).

Covers the course-experience / handedness / pace-bias / travel / going-at-course
features. The stage is called in isolation on a hand-built frame (real track
names so the static config lookup yields meaningful quirk scores and lat/lon).
All features must be race-safe: each row sees only the horse's PRIOR runs.
"""

import numpy as np
import pandas as pd

from src.feature_engineer import add_course_dynamics_features, _haversine_km


def _frame():
    # Horse A: Ascot (Right) -> Chester (Left) -> Chester (Left)
    # Horse B: Chester (Left) -> York (Left)
    # Rows are in per-horse chronological order (what the pipeline guarantees).
    rows = [
        # horse_id, race_id, track, direction, shape, draw_bias, circ, htr, wr
        (1, "rA1", "Ascot",   1, 0, 1, 14.0, 0, 0.0),
        (1, "rA2", "Chester", 0, 1, 2, 8.0,  0, 0.0),
        (1, "rA3", "Chester", 0, 1, 2, 8.0,  1, 0.5),
        (2, "rB1", "Chester", 0, 1, 2, 8.0,  0, 0.0),
        (2, "rB2", "York",    0, 0, 1, 16.0, 0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=[
        "horse_id", "race_id", "track", "track_direction", "track_shape",
        "track_draw_bias", "track_circumference", "horse_track_runs",
        "horse_track_wr_shrunk",
    ])
    df["_result_start"] = 1.0
    df["_won_for_history"] = 0.0
    df["going"] = "Good"
    df["pace_style_index"] = [0.5, -0.3, 0.8, 0.0, 0.2]
    df["pace_front_runner_flag"] = [1, 0, 1, 0, 0]
    return df


def test_static_scores_in_unit_range():
    out = add_course_dynamics_features(_frame())
    for col in ("course_quirk_score", "course_favours_leaders"):
        assert out[col].between(0.0, 1.0).all()
    # Chester (tight, draw_bias 2, short circuit) is quirkier than Ascot (galloping).
    chester = out.loc[out["track"] == "Chester", "course_quirk_score"].iloc[0]
    ascot = out.loc[out["track"] == "Ascot", "course_quirk_score"].iloc[0]
    assert chester > ascot


def test_first_time_at_course_matches_prior_runs():
    out = add_course_dynamics_features(_frame())
    expected = (out["horse_track_runs"] == 0).astype(int)
    assert (out["first_time_at_course"] == expected).all()
    # course_form_x_quirk is the product of proven course form and quirk.
    np.testing.assert_allclose(
        out["course_form_x_quirk"].to_numpy(),
        (out["horse_track_wr_shrunk"] * out["course_quirk_score"]).to_numpy(),
    )


def test_handedness_is_race_safe():
    out = add_course_dynamics_features(_frame()).set_index("race_id")
    # Horse A's direction history: Right, Left, Left -> curdir prior counts 0,0,1.
    assert out.loc["rA1", "horse_curdir_runs"] == 0
    assert out.loc["rA2", "horse_curdir_runs"] == 0
    assert out.loc["rA3", "horse_curdir_runs"] == 1
    # Switching to an unfamiliar hand: A's 2nd run is its first ever left-handed
    # outing (career_runs=1>0, curdir_runs=0) -> flagged; debut & repeat are not.
    assert out.loc["rA1", "first_time_this_hand"] == 0   # career debut, not a switch
    assert out.loc["rA2", "first_time_this_hand"] == 1   # right -> left switch
    assert out.loc["rA3", "first_time_this_hand"] == 0   # already raced left


def test_travel_from_last_run():
    out = add_course_dynamics_features(_frame()).set_index("race_id")
    # First career run has no previous track -> NaN travel.
    assert np.isnan(out.loc["rA1", "travel_from_last_km"])
    # Ascot -> Chester is a long haul (>200 km); Chester -> Chester is ~0.
    assert out.loc["rA2", "travel_from_last_km"] > 200.0
    assert out.loc["rA2", "is_long_haul"] == 1
    assert out.loc["rA3", "travel_from_last_km"] < 1.0
    assert out.loc["rA3", "is_long_haul"] == 0


def test_going_at_course_race_safe():
    out = add_course_dynamics_features(_frame()).set_index("race_id")
    # Horse A runs Good ground at Chester twice: 2nd Chester run has 1 prior C+G.
    assert out.loc["rA2", "horse_track_going_runs"] == 0
    assert out.loc["rA3", "horse_track_going_runs"] == 1


def test_haversine_known_distance():
    # London -> Paris is ~344 km.
    d = _haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330.0 < float(d) < 360.0
