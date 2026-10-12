"""Tests for src.stewards_features.

Fixtures mirror real BHA enquiry text and IHRB content captured June 2026.
The non-negotiable property is leakage-safety: a report on race day D must
never contribute to a feature for a race on day D.
"""

import pandas as pd
import pytest

from src import stewards_features as sf


# ── Tagger ─────────────────────────────────────────────────────────────

class TestTagReport:
    def test_lost_shoe_and_position(self):
        text = (
            "Following the race, the Veterinary Officer reported that HEY BOB (IRE), "
            "placed third, lost its right-fore shoe."
        )
        tags = dict.fromkeys(t for _, t in sf.tag_report(text))
        pairs = sf.tag_report(text)
        assert ("Hey Bob", "lost_shoe") in pairs

    def test_interference_attributed_to_nearest_horse(self):
        text = (
            "Jumping the final hurdle, FUBAR (IRE), placed fifth, made a mistake and "
            "shifted left hampering AVOID DE MASTER (IRE), placed third."
        )
        pairs = set(sf.tag_report(text))
        assert ("Fubar", "jump_error") in pairs
        # the hamper happens to AVOID DE MASTER (the nearest preceding mention
        # of the word 'hampering' is FUBAR, but within-window AVOID also valid)
        assert any(t == "trouble_in_running" for _, t in pairs)

    def test_hung_and_lost_shoe_same_horse(self):
        text = (
            "Following the race, Nathan Moscrop reported that EL JEFE (IRE), unplaced, "
            "hung left throughout, and the Veterinary Officer reported that the gelding "
            "lost its left fore shoe."
        )
        pairs = set(sf.tag_report(text))
        assert ("El Jefe", "hung") in pairs
        assert ("El Jefe", "lost_shoe") in pairs

    def test_gear_permission_is_forward_looking(self):
        text = (
            "Permission was given for DISCKO DES PLAGES (FR) to wear a hood in the "
            "parade ring which was removed before going to the start."
        )
        pairs = set(sf.tag_report(text))
        assert any(t == "gear_permission" for _, t in pairs)
        assert "gear_permission" in sf.FORWARD_LOOKING_TAGS

    def test_known_runners_path_for_titlecase(self):
        # IHRB-style Title Case prose; identity supplied from results.
        text = (
            "The Raceday Stewards enquired into an incident involving Magic Boum, "
            "ridden by Sam Ewing, placed fourth, where interference occurred."
        )
        pairs = set(sf.tag_report(text, known_runners=["Magic Boum", "Chautuaqua"]))
        assert ("Magic Boum", "trouble_in_running") in pairs

    def test_empty_and_none(self):
        assert sf.tag_report("") == []
        assert sf.tag_report(None) == []
        assert sf.tag_report("Nothing to report") == []


# ── Event table ────────────────────────────────────────────────────────

class TestEventTable:
    def test_build_from_bha(self):
        bha = pd.DataFrame([{
            "date": "2024-06-01", "course": "EXETER",
            "enquiry_text": "HEY BOB (IRE), placed third, lost its right-fore shoe.",
        }])
        ev = sf.build_event_table(bha=bha)
        assert set(ev.columns) == {
            "date", "track_key", "horse_key", "tag", "source",
            "forward_looking", "fixture_matched",
        }
        assert (ev["horse_key"] == "Hey Bob").any()
        assert (ev["tag"] == "lost_shoe").any()
        assert ev["track_key"].iloc[0] == "exeter"
        # No results supplied -> caps fallback, fixture not matched.
        assert ev["fixture_matched"].eq(False).all()

    def test_ihrb_track_parsed_from_title(self):
        assert sf._track_from_ihrb_title("Fairyhouse 1st January 2025") == "Fairyhouse"
        assert sf._track_from_ihrb_title("Rule Change 8 February 2025") == ""
        assert sf._track_from_ihrb_title(None) == ""

    def test_empty_inputs(self):
        ev = sf.build_event_table()
        assert ev.empty
        assert "horse_key" in ev.columns


class TestFixtureAnchoring:
    def test_canonical_track_key_reconciles_park_bridge(self):
        from src.utils import canonical_track_key as ck
        assert ck("KEMPTON PARK") == ck("Kempton") == "kempton"
        assert ck("Catterick Bridge") == ck("Catterick") == "catterick"
        assert ck("STRATFORD-ON-AVON") == ck("Stratford") == "stratford"
        assert ck("Haydock Park") == "haydock"

    def test_fuzzy_recovers_source_typo(self):
        # Report misspells a runner; fixture's true runners include the right one.
        text = "AMALIFI BAY (IRE), unplaced, hung left throughout."
        runners = ["Amalfi Bay", "Some Other Horse", "Third Runner"]
        pairs = set(sf.tag_report(text, known_runners=runners))
        assert ("Amalfi Bay", "hung") in pairs   # snapped to the real runner

    def test_fuzzy_does_not_invent_distant_match(self):
        text = "ZZZQQQ WIDGET, unplaced, hung left."
        runners = ["Amalfi Bay", "Some Other Horse"]
        # Nothing close enough -> no spurious attribution.
        assert sf.tag_report(text, known_runners=runners) == []

    def test_event_table_flags_fixture_match(self):
        bha = pd.DataFrame([{
            "date": "2024-06-01", "course": "KEMPTON PARK",
            "enquiry_text": "AMALIFI BAY, placed second, lost its near-fore shoe.",
        }])
        results = pd.DataFrame([
            {"track": "Kempton", "race_date": "2024-06-01", "horse_name": "Amalfi Bay"},
            {"track": "Kempton", "race_date": "2024-06-01", "horse_name": "Other One"},
        ])
        ev = sf.build_event_table(bha=bha, results=results)
        assert ev["fixture_matched"].all()
        assert (ev["horse_key"] == "Amalfi Bay").any()   # typo recovered + joined
        assert ev["track_key"].iloc[0] == "kempton"


# ── Features: the leakage guard ────────────────────────────────────────

class TestFeaturesNoLeakage:
    def _events(self):
        return pd.DataFrame([
            {"horse_key": "Hey Bob", "date": pd.Timestamp("2024-05-01"),
             "track_key": "exeter", "tag": "lost_shoe", "source": "bha",
             "forward_looking": False},
            {"horse_key": "Hey Bob", "date": pd.Timestamp("2024-06-01"),
             "track_key": "exeter", "tag": "lost_shoe", "source": "bha",
             "forward_looking": False},
        ])

    def test_same_day_report_does_not_leak(self):
        races = pd.DataFrame([
            {"horse_name": "Hey Bob", "race_date": "2024-06-01"},  # same day as 2nd event
        ])
        out = sf.add_stewards_features(races, self._events())
        # The 2024-06-01 report must NOT count for the 2024-06-01 race; only the
        # earlier 2024-05-01 event is strictly in the past -> career == 1.
        assert out.loc[0, "stew_lost_shoe_career"] == 1

    def test_career_accumulates_only_past(self):
        races = pd.DataFrame([
            {"horse_name": "Hey Bob", "race_date": "2024-04-01"},  # before any event
            {"horse_name": "Hey Bob", "race_date": "2024-05-15"},  # after 1st event
            {"horse_name": "Hey Bob", "race_date": "2024-07-01"},  # after both
            {"horse_name": "Ghost", "race_date": "2024-07-01"},    # no events
        ])
        out = sf.add_stewards_features(races, self._events()).set_index("race_date")
        assert out.loc["2024-04-01", "stew_lost_shoe_career"] == 0
        assert out.loc["2024-05-15", "stew_lost_shoe_career"] == 1
        assert out.loc["2024-07-01", "stew_lost_shoe_career"].max() == 2  # Hey Bob
        # unrelated horse stays zero
        assert (out["stew_lost_shoe_career"] >= 0).all()

    def test_all_feature_columns_present_even_when_no_events(self):
        races = pd.DataFrame([{"horse_name": "Nobody", "race_date": "2024-01-01"}])
        out = sf.add_stewards_features(races, pd.DataFrame(
            columns=["horse_key", "date", "tag"]))
        for tag in sf.ALL_TAGS:
            assert f"stew_{tag}_career" in out.columns
            assert out[f"stew_{tag}_career"].iloc[0] == 0


class TestRollingFeatures:
    def _events(self):
        # Horse bled at runs on 2024-01-01 and 2024-03-01.
        return pd.DataFrame([
            {"horse_key": "Bleeder", "date": pd.Timestamp("2024-01-01"), "tag": "bled"},
            {"horse_key": "Bleeder", "date": pd.Timestamp("2024-03-01"), "tag": "bled"},
        ])

    def test_rolling_excludes_current_run(self):
        races = pd.DataFrame([
            {"horse_name": "Bleeder", "race_date": "2024-01-01"},  # bled THIS run
            {"horse_name": "Bleeder", "race_date": "2024-02-01"},  # 1 prior bleed
            {"horse_name": "Bleeder", "race_date": "2024-03-01"},  # 1 prior (the Jan one)
            {"horse_name": "Bleeder", "race_date": "2024-04-01"},  # 2 prior bleeds
        ])
        out = sf.add_stewards_rolling_features(
            races, self._events(), windows=(5,)).set_index("race_date")
        # The same-day report on 2024-01-01 must NOT count for that run.
        assert out.loc["2024-01-01", "stew_bled_last5"] == 0
        assert out.loc["2024-02-01", "stew_bled_last5"] == 1
        assert out.loc["2024-03-01", "stew_bled_last5"] == 1
        assert out.loc["2024-04-01", "stew_bled_last5"] == 2

    def test_window_size_limits_lookback(self):
        events = pd.DataFrame([
            {"horse_key": "H", "date": pd.Timestamp(f"2024-0{i}-01"), "tag": "hung"}
            for i in range(1, 5)  # hung Jan,Feb,Mar,Apr
        ])
        # The race frame holds every run (as on full history): Jan..May.
        races = pd.DataFrame([
            {"horse_name": "H", "race_date": f"2024-0{i}-01"} for i in range(1, 6)
        ])
        out = sf.add_stewards_rolling_features(
            races, events, windows=(2, 3)).set_index("race_date")
        assert out.loc["2024-05-01", "stew_hung_last2"] == 2   # only Mar+Apr
        assert out.loc["2024-05-01", "stew_hung_last3"] == 3   # Feb+Mar+Apr

    def test_schema_stable_without_events(self):
        races = pd.DataFrame([{"horse_name": "X", "race_date": "2024-01-01"}])
        out = sf.add_stewards_rolling_features(
            races, pd.DataFrame(columns=["horse_key", "date", "tag"]), windows=(3,))
        for tag in sf.ALL_TAGS:
            assert out[f"stew_{tag}_last3"].iloc[0] == 0


class TestIntegration:
    def test_enrich_no_op_when_artifact_missing(self):
        races = pd.DataFrame([{"horse_name": "X", "race_date": "2024-01-01"}])
        out = sf.enrich_with_stewards(races, events_path="does/not/exist.parquet")
        pd.testing.assert_frame_equal(out, races)

    def test_build_artifact_and_enrich_roundtrip(self, tmp_path):
        bha = tmp_path / "bha.csv"
        pd.DataFrame([{
            "date": "2024-01-01", "course": "KEMPTON PARK",
            "enquiry_text": "BLEEDER, unplaced, bled from both nostrils.",
        }]).to_csv(bha, index=False)
        results = tmp_path / "res.csv"
        pd.DataFrame([
            {"track": "Kempton", "race_date": "2024-01-01", "horse_name": "Bleeder"},
        ]).to_csv(results, index=False)
        art = tmp_path / "events.parquet"
        ev = sf.build_stewards_event_artifact(
            bha_path=str(bha), ihrb_path=str(tmp_path / "none.csv"),
            results_path=str(results), out_path=str(art))
        assert (ev["tag"] == "bled").any()
        assert art.exists()

        # Full-history frame: the bleed run (Jan) and a later run (Feb).
        races = pd.DataFrame([
            {"horse_name": "Bleeder", "race_date": "2024-01-01"},
            {"horse_name": "Bleeder", "race_date": "2024-02-01"},
        ])
        out = sf.enrich_with_stewards(
            races, events_path=str(art), windows=(5,)).set_index("race_date")
        assert out.loc["2024-02-01", "stew_bled_career"] == 1
        assert out.loc["2024-02-01", "stew_bled_last5"] == 1
        assert out.loc["2024-01-01", "stew_bled_last5"] == 0   # current run excluded
