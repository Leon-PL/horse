"""Tests for the intent-signal additions: wind surgery (scrape + FE),
trainer travel distance, and jockey-booking strength."""

import numpy as np
import pandas as pd
import pytest

from src.data_scraper import SportingLifeScraper
from src.feature_engineer import (
    add_medical_features,
    add_trainer_runner_intent_features,
)
from src.trainer_locations import (
    _surname_initial_key,
    haversine_km,
    normalise_trainer_name,
    resolve_trainer_latlon,
)


# ---------------------------------------------------------------------------
# Scraper: medication -> wind_surgery
# ---------------------------------------------------------------------------

def _ride_dict(medication):
    ride = {
        "ride_status": "",
        "horse": {"name": "Test Horse", "horse_reference": {"id": 1}},
        "medication": medication,
    }
    return SportingLifeScraper()._ride_to_dict(
        ride,
        race_id="1",
        date_str="2026-07-02",
        off_time="14:00",
        course_name="Ascot",
        region="UK",
        race_name="Test Stakes",
        race_class="4",
        race_type="Flat",
        distance_f=8.0,
        going="Good",
        prize_money=5000.0,
        num_runners=8,
        surface_str="Turf",
        has_handicap=False,
        is_result=False,
    )


class TestWindSurgeryParse:
    def test_ws_flagged(self):
        assert _ride_dict([{"short_name": "WS"}])["wind_surgery"] == 1

    def test_lowercase_ws_on_result_pages(self):
        assert _ride_dict([{"short_name": "ws"}])["wind_surgery"] == 1

    def test_null_medication(self):
        assert _ride_dict(None)["wind_surgery"] == 0

    def test_other_medication_not_ws(self):
        assert _ride_dict([{"short_name": "XX"}])["wind_surgery"] == 0


# ---------------------------------------------------------------------------
# FE: wind surgery history features
# ---------------------------------------------------------------------------

def _ws_history_df():
    # One horse, four runs: op before run 2 and run 4; a rival with no ops.
    rows = [
        dict(race_id="R1", race_date="2024-01-01", horse_name="Wheezy",
             sex="Gelding", wind_surgery=0),
        dict(race_id="R2", race_date="2024-02-01", horse_name="Wheezy",
             sex="Gelding", wind_surgery=1),
        dict(race_id="R3", race_date="2024-03-01", horse_name="Wheezy",
             sex="Gelding", wind_surgery=0),
        dict(race_id="R4", race_date="2024-04-01", horse_name="Wheezy",
             sex="Gelding", wind_surgery=1),
        dict(race_id="R4", race_date="2024-04-01", horse_name="Sound",
             sex="Mare", wind_surgery=np.nan),  # un-backfilled row
    ]
    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df


class TestWindSurgeryFeatures:
    def test_declared_flag_and_nan_treated_as_zero(self):
        out = add_medical_features(_ws_history_df())
        w = out[out.horse_name == "Wheezy"].set_index("race_id")
        assert list(w["wind_surgery_run"]) == [0, 1, 0, 1]
        s = out[out.horse_name == "Sound"]
        assert (s["wind_surgery_run"] == 0).all()

    def test_prior_count_excludes_current_run(self):
        out = add_medical_features(_ws_history_df())
        w = out[out.horse_name == "Wheezy"].set_index("race_id")
        assert list(w["wind_ops_prior"]) == [0, 0, 1, 1]

    def test_first_wind_surgery_only_on_first_op(self):
        out = add_medical_features(_ws_history_df())
        w = out[out.horse_name == "Wheezy"].set_index("race_id")
        assert list(w["first_wind_surgery"]) == [0, 1, 0, 0]

    def test_prev_run_flag_marks_second_run_after_op(self):
        out = add_medical_features(_ws_history_df())
        w = out[out.horse_name == "Wheezy"].set_index("race_id")
        assert list(w["prev_run_wind_surgery"]) == [0, 0, 1, 0]

    def test_works_without_sex_column(self):
        df = _ws_history_df().drop(columns=["sex"])
        out = add_medical_features(df)
        assert "wind_surgery_run" in out.columns


# ---------------------------------------------------------------------------
# Trainer locations: normalisation, matching, geometry
# ---------------------------------------------------------------------------

class TestTrainerNameMatching:
    def test_normalise_strips_titles_and_dots(self):
        assert normalise_trainer_name("Dr R. Newland") == "r newland"
        assert normalise_trainer_name("Mrs J.  Smith") == "j smith"

    def test_surname_initial_key(self):
        assert _surname_initial_key("n w alexander") == ("alexander", "n")
        assert _surname_initial_key("nick walter alexander") == ("alexander", "n")
        # joint licences use the first partner
        assert _surname_initial_key("r newland & j insole") == ("newland", "r")
        assert _surname_initial_key("smith") is None

    def test_resolve_exact_initial_and_ambiguous(self, tmp_path):
        csv = tmp_path / "trainers.csv"
        pd.DataFrame([
            dict(bha_id=1, bha_style="N. W. Alexander",
                 bha_fullname="Nick Walter Alexander",
                 licence_type="Combined", county="Fife", lat=56.25, lon=-3.15),
            # two J Smiths in different counties -> surname+initial ambiguous
            dict(bha_id=2, bha_style="J. Smith", bha_fullname="John Smith",
                 licence_type="Flat", county="Kent", lat=51.20, lon=0.70),
            dict(bha_id=3, bha_style="J. Smith", bha_fullname="James Smith",
                 licence_type="Flat", county="Fife", lat=56.25, lon=-3.15),
        ]).to_csv(csv, index=False)

        res = resolve_trainer_latlon(
            ["Nick Alexander", "N W Alexander", "Jake Smith", "Nobody Here"],
            path=csv,
        )
        # initial-form and full-form both hit the same yard
        assert res["N W Alexander"] == (56.25, -3.15)
        assert res["Nick Alexander"] == (56.25, -3.15)
        # ambiguous surname+initial must NOT resolve
        assert "Jake Smith" not in res
        assert "Nobody Here" not in res

    def test_haversine_sanity(self):
        # London -> Edinburgh is ~535 km
        d = haversine_km(51.5074, -0.1278, 55.9533, -3.1883)
        assert 500 < float(d) < 570


# ---------------------------------------------------------------------------
# Jockey-booking strength
# ---------------------------------------------------------------------------

class TestJockeyBooking:
    def _race(self):
        # One race; Yard A runs two horses, the better rider on the outsider.
        return pd.DataFrame([
            dict(race_id="R", trainer="Yard A", horse_name="Outsider",
                 jockey_elo=1800.0, odds=20.0, official_rating=70),
            dict(race_id="R", trainer="Yard A", horse_name="Fancied",
                 jockey_elo=1400.0, odds=3.0, official_rating=80),
            dict(race_id="R", trainer="Yard B", horse_name="Rival",
                 jockey_elo=1600.0, odds=2.0, official_rating=85),
        ])

    def test_stable_pecking_order_by_jockey(self):
        out = add_trainer_runner_intent_features(self._race())
        by_horse = out.set_index("horse_name")
        assert by_horse.loc["Outsider", "trainer_first_string_by_jockey"] == 1
        assert by_horse.loc["Fancied", "trainer_first_string_by_jockey"] == 0
        assert by_horse.loc["Fancied", "jockey_elo_vs_best_stablemate"] == -400.0
        # sole runner: is its own best booking
        assert by_horse.loc["Rival", "jockey_elo_vs_best_stablemate"] == 0.0

    def test_booking_vs_market_sign(self):
        out = add_trainer_runner_intent_features(self._race())
        by_horse = out.set_index("horse_name")
        # best jockey on the least-fancied horse -> strongly positive
        assert by_horse.loc["Outsider", "jockey_booking_vs_market"] > 0.5
        # market leader with the weakest rider -> negative
        assert by_horse.loc["Rival", "jockey_booking_vs_market"] < 0
