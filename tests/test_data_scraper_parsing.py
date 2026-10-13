"""Regression tests for Sporting Life parsing helpers.

These pin the contracts of the pure parsing helpers in ``src.data_scraper``.
Two of these tests guard live bugs found during the data-collection audit:

* ``_parse_lengths_behind`` matched named margins by substring with the wrong
  precedence, so ``"snk"`` returned ``nk``'s 0.3 and any numeric part was
  dropped.
* Arabian (different-breed) races were being ingested; ``_is_arabian_race``
  now filters them.
"""

import math

import pytest

from src.data_scraper import (
    _is_arabian_race,
    _parse_distance_to_furlongs,
    _parse_fractional_odds,
    _parse_lengths_behind,
    _parse_weight_to_lbs,
)


class TestLengthsBehind:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("snk", 0.2),   # short-neck — must NOT be shadowed by "nk"
            ("nk", 0.3),
            ("sh", 0.1),
            ("shd", 0.1),
            ("hd", 0.2),
            ("nse", 0.05),
            ("dht", 0.0),
            ("3", 3.0),
            ("8 ½", 8.5),
            ("1¼", 1.25),
            ("2¾", 2.75),
        ],
    )
    def test_values(self, raw, expected):
        assert _parse_lengths_behind(raw) == pytest.approx(expected)

    def test_empty_is_nan(self):
        assert math.isnan(_parse_lengths_behind(""))

    def test_named_code_not_matched_as_substring(self):
        # The old bug: "snk" returned 0.3 because "nk" was found inside it.
        assert _parse_lengths_behind("snk") != 0.3


class TestFractionalOdds:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("5/2", 3.5),
            ("11/4", 3.75),
            ("11/4f", 3.75),   # favourite marker stripped
            ("evs", 2.0),
            ("evens", 2.0),
            ("2", 2.0),        # already decimal
            ("7/2jf", 4.5),    # joint-fav marker stripped
        ],
    )
    def test_values(self, raw, expected):
        assert _parse_fractional_odds(raw) == pytest.approx(expected)

    def test_zero_denominator_does_not_crash(self):
        # Falls through to the default rather than dividing by zero.
        assert _parse_fractional_odds("5/0") == 5.0


class TestDistanceAndWeight:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2m", 16.0),
            ("1m 4f", 12.0),
            ("5f", 5.0),
            ("2m 7f 96y", 23.0 + 96 / 220.0),
        ],
    )
    def test_distance(self, raw, expected):
        assert _parse_distance_to_furlongs(raw) == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("11-9", 11 * 14 + 9),
            ("8-0", 112),
            ("10-7", 147),
        ],
    )
    def test_weight(self, raw, expected):
        assert _parse_weight_to_lbs(raw) == expected


class TestArabianGuard:
    def test_detects_by_horse_suffix(self):
        runners = [{"horse_name": "Toofaan (A)"}, {"horse_name": "Mazyoonah (A)"}]
        assert _is_arabian_race("Some Maiden Stakes", "Bath", runners)

    def test_detects_by_race_name(self):
        runners = [{"horse_name": "Galileo"}]
        assert _is_arabian_race("Emirates Breeders Handicap - Arab Race", "Bath", runners)

    def test_detects_by_extras_feed_track(self):
        assert _is_arabian_race("Anything", "RMG Extras", [{"horse_name": "X"}])

    def test_thoroughbred_not_flagged(self):
        # "Arabella" is a thoroughbred sponsor/horse name, not an Arabian race.
        runners = [{"horse_name": "Frankel"}, {"horse_name": "Enable"}]
        assert not _is_arabian_race(
            "Melissa, Arabella And Oriana Hawthorne Handicap", "Bath", runners
        )

    def test_results_listing_drops_arabian_races(self, monkeypatch):
        """Sporting Life files Arabian races under a "TRP Extras" pseudo-course
        that passes the UK/IRE filter (country England). Listed, they entered
        the sync ledger as races that can never complete."""
        from src.data_scraper import SportingLifeScraper

        def meeting(course, races):
            return {"meeting_summary": {"course": {
                        "name": course, "country": {"long_name": "England"}}},
                    "races": [{"name": n, "time": "14:30",
                               "race_summary_reference": {"id": i}} for i, n in races]}

        page = {"props": {"pageProps": {"meetings": [
            meeting("Wolverhampton", [(1, "Novice Stakes"), (2, "Handicap (Arab)")]),
            meeting("TRP Extras", [(3, "Arab Race - Wolverhampton - Novice Stakes")]),
        ]}}}
        scraper = SportingLifeScraper()
        monkeypatch.setattr(scraper, "_fetch_next_data", lambda url: page)
        assert [r["race_id"] for r in scraper.get_results_urls("2020-10-22")] == ["1"]
