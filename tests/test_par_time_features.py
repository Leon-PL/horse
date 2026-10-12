"""Correctness + leakage tests for time-based going-corrected speed figures."""
import numpy as np
import pandas as pd
import pytest

from src.par_time_features import (
    parse_winning_time,
    add_time_speed_figure,
    add_past_time_speed_features,
)


def test_parse_winning_time():
    assert parse_winning_time("1m 13.24s") == pytest.approx(73.24)
    assert parse_winning_time("59.34s") == pytest.approx(59.34)
    assert parse_winning_time("2m 37.18s") == pytest.approx(157.18)
    assert parse_winning_time("3m 6.50s") == pytest.approx(186.5)
    assert np.isnan(parse_winning_time(""))
    assert np.isnan(parse_winning_time(None))
    assert np.isnan(parse_winning_time("junk"))
    # absurd values rejected
    assert np.isnan(parse_winning_time("0.5s"))


def _toy():
    # two races, same track/surface/distance; race B run on faster ground.
    rows = []
    # race A: winner 73.0s, runner-up 2 lengths back
    rows += [
        dict(race_id="A", track="Test", race_date="2024-01-01", surface="Turf",
             distance_furlongs=6.0, finish_position=1, lengths_behind=0.0, horse_name="Fast"),
        dict(race_id="A", track="Test", race_date="2024-01-01", surface="Turf",
             distance_furlongs=6.0, finish_position=2, lengths_behind=2.0, horse_name="Slow"),
    ]
    # race B: faster day, winner 71.0s
    rows += [
        dict(race_id="B", track="Test", race_date="2024-02-01", surface="Turf",
             distance_furlongs=6.0, finish_position=1, lengths_behind=0.0, horse_name="Slow"),
        dict(race_id="B", track="Test", race_date="2024-02-01", surface="Turf",
             distance_furlongs=6.0, finish_position=2, lengths_behind=4.0, horse_name="Fast"),
    ]
    wt = pd.DataFrame({"race_id": ["A", "B"], "winning_time_secs": [73.0, 71.0]})
    return pd.DataFrame(rows), wt


def test_figure_winner_beats_beaten_runner():
    df, wt = _toy()
    out = add_time_speed_figure(df, winning_times=wt)
    # within a race the winner (0 lengths) must score >= a beaten runner
    a = out[out.race_id == "A"].set_index("horse_name")["tspf"]
    assert a["Fast"] > a["Slow"]


def test_figure_present_and_finite_for_clocked_races():
    df, wt = _toy()
    out = add_time_speed_figure(df, winning_times=wt)
    assert out["tspf"].notna().all()


def test_missing_winning_time_yields_nan():
    df, wt = _toy()
    wt2 = wt[wt.race_id == "A"]  # race B has no clock
    out = add_time_speed_figure(df, winning_times=wt2)
    assert out.loc[out.race_id == "B", "tspf"].isna().all()
    assert out.loc[out.race_id == "A", "tspf"].notna().all()


def test_rolling_is_strictly_past_no_leak():
    df, wt = _toy()
    out = add_past_time_speed_features(add_time_speed_figure(df, winning_times=wt))
    # 'Fast' runs race A (first ever) then race B. Its rolling features in race A must be
    # NaN (no prior runs); in race B they must reflect ONLY race A.
    fast = out[out.horse_name == "Fast"].sort_values("race_date")
    first, second = fast.iloc[0], fast.iloc[1]
    assert first.race_id == "A"
    assert np.isnan(first["tspf_avg_5"]), "current race figure leaked into its own feature"
    assert first["tspf_n_past"] == 0
    # second race's past-avg equals the figure earned in race A
    a_fig = out[(out.race_id == "A") & (out.horse_name == "Fast")]["tspf"].iloc[0]
    assert second["tspf_avg_5"] == pytest.approx(a_fig)
    assert second["tspf_n_past"] == 1


def test_rolling_handles_missing_figures():
    df, wt = _toy()
    wt2 = wt[wt.race_id == "A"]
    out = add_past_time_speed_features(add_time_speed_figure(df, winning_times=wt2))
    # no crash; n_past counts only clocked past runs
    assert out["tspf_n_past"].max() <= 1
