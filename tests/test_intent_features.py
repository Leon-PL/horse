"""Tests for the intent-signal additions: the wind-surgery scrape and
jockey-booking strength. Trainer travel distance (src/trainer_locations.py) was
refuted and removed with the research code on 2026-11-07; the wind-surgery
history features were removed from FE on 2026-11-07 (research 065) because no
feature set used them (refuted, docs/NEGATIVE_RESULTS.md). The scraped flag is
still collected."""

import pandas as pd

from src.ingestion.sporting_life import SportingLifeScraper
from src.features.connections import add_trainer_runner_intent_features


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

    def test_booking_vs_market_sign(self):
        out = add_trainer_runner_intent_features(self._race())
        by_horse = out.set_index("horse_name")
        # best jockey on the least-fancied horse -> strongly positive
        assert by_horse.loc["Outsider", "jockey_booking_vs_market"] > 0.5
        # market leader with the weakest rider -> negative
        assert by_horse.loc["Rival", "jockey_booking_vs_market"] < 0
