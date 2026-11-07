"""Regression tests for the At The Races sectionals scraper.

These pin the pure helpers and HTML parsers in
``src.ingestion.attheraces`` against fixtures mirroring the live ATR
``/ajax/sectionals/{raceid}/{kind}`` fragment structure (captured June
2026).  No network access is required.
"""

import json
import math

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.ingestion import attheraces as atr
from src.ingestion import atr_tabs
from src import atr_features


# ── URL / time helpers ────────────────────────────────────────────────

class TestSlugAndDate:
    @pytest.mark.parametrize(
        "track,slug",
        [
            ("Kempton", "Kempton"),
            ("Lingfield", "Lingfield"),            # NOT "Lingfield-Park"
            ("Chelmsford City", "Chelmsford-City"),
            ("Epsom Downs", "Epsom-Downs"),
            ("Newton Abbot", "Newton-Abbot"),
            ("Bangor-on-Dee", "Bangor-on-Dee"),    # existing hyphens/case kept
            ("Royal Ascot", "Ascot"),              # override
        ],
    )
    def test_track_to_slug(self, track, slug):
        assert atr._track_to_slug(track) == slug

    def test_date_to_atr(self):
        assert atr._date_to_atr("2026-06-10") == "10-June-2026"
        assert atr._date_to_atr("2026-03-01") == "01-March-2026"

    def test_date_to_atr_invalid(self):
        assert atr._date_to_atr("not-a-date") is None

    @pytest.mark.parametrize(
        "raw,hhmm", [("14:08", "1408"), ("14:08:00", "1408"), ("9:05", "0905")],
    )
    def test_off_time_to_hhmm(self, raw, hhmm):
        assert atr._off_time_to_hhmm(raw) == hhmm

    def test_bst_and_hour_shift(self):
        assert atr._is_bst("2026-06-10") is True    # summer
        assert atr._is_bst("2026-01-10") is False   # winter
        assert atr._add_one_hour("1810") == "1910"
        assert atr._add_one_hour("2340") == "0040"  # wraps midnight


class TestSmallHelpers:
    def test_to_float(self):
        assert atr._to_float("Running Total: 14.68") == pytest.approx(14.68)
        assert atr._to_float("-") is None
        assert atr._to_float(None) is None
        assert atr._to_float("1,234.5") == pytest.approx(1234.5)

    def test_extract_raceid_from_comment(self):
        assert atr._extract_raceid("<!-- RaceID: 1595773 -->") == "1595773"

    def test_extract_raceid_from_query(self):
        assert atr._extract_raceid('href="/form/horse/X/9?raceid=42"') == "42"

    def test_extract_raceid_missing(self):
        assert atr._extract_raceid("<html>no id here</html>") is None

    def test_parse_chart(self):
        chart = BeautifulSoup(
            '<div class="chart-energy-distribution" '
            "data-data='[[\"5f\", 82.86, 84.41], [\"Fin.\", 98.49, 100.44]]'></div>",
            "html.parser",
        ).div
        markers, s1, s2 = atr._parse_chart(chart)
        assert markers == ["5f", "Fin."]
        assert s1 == pytest.approx([82.86, 98.49])
        assert s2 == pytest.approx([84.41, 100.44])

    def test_parse_chart_none(self):
        assert atr._parse_chart(None) == ([], [], [])


# ── HTML parsers (fixtures mirror the live fragment structure) ─────────

def _wrap(entry_html: str) -> BeautifulSoup:
    """Wrap a sectional card-entry in the two-column card-body layout."""
    html = (
        '<div class="card-body"><div class="card-entry"></div></div>'
        f'<div class="card-body">{entry_html}</div>'
    )
    return BeautifulSoup(html, "html.parser")


_HORSE_HEAD = (
    '<span class="p--small visible--inline-block">8.</span>'
    '<a class="horse__link" href="/form/horse/Moonshine/3793401?raceid=1595773">Moonshine</a>'
    '<span class="p--x-small visible--inline-block">(1)</span>'
    '<span class="p--x-small tooltip" title="led inside final furlong, ran on well"></span>'
)


def _times_fixture():
    cells = ""
    for split, total, pace in [(14.68, 14.68, "even-to-slow"), (11.20, 25.88, "even-to-fast")]:
        cells += (
            '<div class="card-cell card-cell--sectional-time">'
            f'<div class="card-sectional card-sectional--{pace}">'
            f'<div class="card-sectional__inner tooltip" title="Running Total: {total}">'
            f'<span class="visible">{split}</span></div></div></div>'
        )
    # trailing finish marker carries no value
    cells += (
        '<div class="card-cell card-cell--sectional-time">'
        '<div class="card-sectional card-sectional--finish"></div></div>'
    )
    return _wrap(f'<div class="card-entry" data-officialplacing="1" data-adjplacing="6">{_HORSE_HEAD}{cells}</div>')


def _tools_fixture():
    body = (
        '<span class="text-pill text-pill--even tooltip" title="Start to 4f">34.78</span>'
        '<span class="text-pill text-pill--slow tooltip" title="2f to Finish">37.48</span>'
        '<span class="text-pill text-pill--medium text-pill--grey tooltip" title="Efficency Score: 17.18">C</span>'
        '<div class="width--24 p--medium">101.28</div>'
        '<div class="chart-energy-distribution" '
        "data-data='[[\"5f\", 82.86, 84.41], [\"Fin.\", 98.49, 100.44]]'></div>"
    )
    return _wrap(f'<div class="card-entry" data-officialplacing="1">{_HORSE_HEAD}{body}</div>')


def _stride_fixture():
    body = (
        '<div class="chart-energy-distribution" '
        "data-data='[[\"5f\", 20.12, 26.94], [\"Fin.\", 23.74, 26.94]]'></div>"
        '<div class="width--24">26.94 / 2.28</div>'
    )
    return _wrap(f'<div class="card-entry" data-officialplacing="1">{_HORSE_HEAD}{body}</div>')


class TestParsers:
    def test_parse_times(self):
        rec = atr._parse_times(_times_fixture())["Moonshine"]
        assert rec["atr_split_times"] == pytest.approx([14.68, 11.20])
        assert rec["atr_running_totals"] == pytest.approx([14.68, 25.88])
        assert rec["atr_pace_flags"] == ["even-to-slow", "even-to-fast"]
        assert rec["atr_final_time"] == pytest.approx(25.88)
        assert rec["atr_n_sections"] == 2.0
        assert rec["atr_official_position"] == "1"
        assert rec["atr_adj_placing"] == pytest.approx(6.0)
        assert rec["atr_horse_id"] == "3793401"
        assert rec["atr_saddle_cloth"] == "8"
        assert rec["atr_draw"] == "1"
        assert "ran on well" in rec["atr_in_running_comment"]

    def test_parse_tools(self):
        rec = atr._parse_tools(_tools_fixture())["Moonshine"]
        assert rec["atr_sectional_speeds"] == pytest.approx([34.78, 37.48])
        assert rec["atr_sectional_speed_labels"] == ["Start to 4f", "2f to Finish"]
        assert rec["atr_efficiency_grade"] == "C"
        assert rec["atr_efficiency_score"] == pytest.approx(17.18)
        assert rec["atr_finishing_speed_pct"] == pytest.approx(101.28)
        assert rec["atr_section_markers"] == ["5f", "Fin."]
        assert rec["atr_energy_dist"] == pytest.approx([82.86, 98.49])
        assert rec["atr_optimum_energy"] == pytest.approx([84.41, 100.44])

    def test_parse_stride(self):
        rec = atr._parse_stride(_stride_fixture())["Moonshine"]
        assert rec["atr_stride_length"] == pytest.approx([20.12, 23.74])
        assert rec["atr_peak_stride_length"] == pytest.approx(26.94)
        assert rec["atr_peak_stride_freq"] == pytest.approx(2.28)


# ── Cache encode + merge round-trip ────────────────────────────────────

class _FakeFetcher:
    """Routes ``get`` calls to canned ``(status, text)`` responses by URL substring."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, *, referer=None):
        self.calls.append(url)
        path = url.split("?", 1)[0]  # ignore query (the AJAX page= holds /racecard/)
        for needle, resp in self.routes.items():
            if needle in path:
                return resp
        return (404, "")


_RACECARD = ('<html><!-- RaceID: 1595773 --></html>')


class TestOutcomeClassification:
    """The critical guard: transient blocks must never look like no-data."""

    def test_racecard_blocked_is_blocked(self):
        f = _FakeFetcher({"/racecard/": (429, "")})
        assert atr._fetch_one(f, "Kempton", "10-June-2026", "1910")[0] == atr.BLOCKED

    def test_racecard_challenge_is_blocked(self):
        # _Fetcher.get already maps the challenge to 429; simulate that here.
        f = _FakeFetcher({"/racecard/": (429, "")})
        assert atr._fetch_one(f, "Kempton", "10-June-2026", "1910")[0] == atr.BLOCKED

    def test_racecard_404_is_not_found(self):
        f = _FakeFetcher({"/racecard/": (404, "")})
        assert atr._fetch_one(f, "Naas", "10-June-2026", "1430")[0] == atr.NOT_FOUND

    def test_racecard_200_no_raceid_is_retryable(self):
        f = _FakeFetcher({"/racecard/": (200, "<html>no id</html>")})
        assert atr._fetch_one(f, "Kempton", "10-June-2026", "1910")[0] == atr.BLOCKED

    def test_ajax_blocked_is_blocked_not_no_data(self):
        # Racecard OK with a RaceID, but every AJAX fragment is rate-limited.
        f = _FakeFetcher({"/racecard/": (200, _RACECARD), "/ajax/sectionals-razor/": (429, "")})
        outcome, rows = atr._fetch_one(f, "Kempton", "10-June-2026", "1910")
        assert outcome == atr.BLOCKED
        assert rows is None

    def test_ajax_404_fragment_is_not_found(self):
        # Racecard OK, but the sectional fragments genuinely don't exist (404).
        f = _FakeFetcher({"/racecard/": (200, _RACECARD), "/ajax/sectionals-razor/": (404, "")})
        assert atr._fetch_one(f, "Kempton", "10-June-2026", "1910")[0] == atr.NOT_FOUND

    def test_full_data_path(self):
        routes = {
            "/racecard/": (200, _RACECARD),
            "/times": (200, str(_times_fixture())),
            "/tools": (200, str(_tools_fixture())),
            "/stride": (200, str(_stride_fixture())),
        }
        outcome, rows = atr._fetch_one(_FakeFetcher(routes), "Kempton", "10-June-2026", "1910")
        assert outcome == atr.DATA
        assert len(rows) == 1
        assert rows[0]["horse_name"] == "Moonshine"
        assert rows[0]["atr_raceid"] == "1595773"

    def test_outcome_wrapper_never_retries_a_different_summer_race(self, monkeypatch):
        # UTC is explicit; falling back an hour can silently collect another field.
        monkeypatch.setattr(atr, "_is_bst", lambda d: True)
        seen = []

        def fake_fetch_one(fetcher, slug, atr_date, hhmm):
            seen.append(hhmm)
            return (atr.BLOCKED, None) if hhmm == "1910" else (atr.DATA, [{"horse_name": "X"}])

        monkeypatch.setattr(atr, "_fetch_one", fake_fetch_one)
        outcome, rows = atr.fetch_atr_sectionals_outcome("2026-06-10", "Kempton", "18:10", fetcher=object())
        assert outcome == atr.BLOCKED and rows is None
        assert seen == ["1910"]

    def test_outcome_wrapper_preserves_exact_race_absence(self, monkeypatch):
        monkeypatch.setattr(atr, "_is_bst", lambda d: True)
        # A missing correct time does not justify trying a different field.
        outcomes = iter([(atr.NOT_FOUND, None), (atr.BLOCKED, None)])
        monkeypatch.setattr(atr, "_fetch_one", lambda *a: next(outcomes))
        outcome, _ = atr.fetch_atr_sectionals_outcome("2026-06-10", "Kempton", "18:10", fetcher=object())
        assert outcome == atr.NOT_FOUND


class TestTrackExclusion:
    def test_irish_tracks_in_exclusion_set(self):
        for t in ("Dundalk", "Curragh", "Leopardstown", "Down Royal", "Limerick"):
            assert atr._normalise_track_key(t) in atr._NO_SECTIONALS_TRACKS

    def test_gb_tracks_not_excluded(self):
        for t in ("Kempton", "Wolverhampton", "Newcastle", "Lingfield", "Newmarket"):
            assert atr._normalise_track_key(t) not in atr._NO_SECTIONALS_TRACKS

    def test_backfill_skips_excluded_tracks(self, tmp_path, monkeypatch):
        # All-Irish race set: nothing should be fetched (no network calls).
        monkeypatch.setattr(atr, "ATR_CACHE_FILE", str(tmp_path / "c.parquet"))
        monkeypatch.setattr(atr_tabs, 'ATR_CACHE_FILE', str(tmp_path / "c.parquet"))
        monkeypatch.setattr(atr, "ATR_NO_DATA_FILE", str(tmp_path / "n.json"))

        def _boom(*a, **k):
            raise AssertionError("excluded track should never be fetched")

        monkeypatch.setattr(atr, "_Fetcher", _boom)
        races = pd.DataFrame([
            {"race_date": "2026-06-20", "track": "Down Royal", "off_time": "16:13"},
            {"race_date": "2026-06-20", "track": "Curragh", "off_time": "14:30"},
        ])
        summary = atr.backfill_atr_sectionals_for_races(races)
        assert summary["excluded_tracks_races"] == 2
        assert summary["missing_races"] == 0
        assert summary["fetched_races"] == 0


class TestCacheAndPublication:
    def test_cache_round_trip_and_publication_links_by_runner(self, tmp_path, monkeypatch, publish_dataset):
        from src.source_dataset import read_table
        cache = pd.DataFrame([{
            "race_date": "2026-06-10", "track": "Kempton", "off_time": "18:10",
            "horse_name": "Moonshine", "atr_final_time": 72.98,
            "atr_finishing_speed_pct": 101.28, "atr_efficiency_grade": "C",
            "atr_official_position": "1",
            "atr_split_times": [14.68, 11.20], "atr_pace_flags": ["even", "fast"],
        }])
        cache_file = tmp_path / "atr.parquet"
        monkeypatch.setattr(atr, "ATR_CACHE_FILE", str(cache_file))
        monkeypatch.setattr(atr_tabs, 'ATR_CACHE_FILE', str(cache_file))
        atr.save_atr_cache(cache)

        # Arrays must persist as JSON strings (flat parquet).
        reloaded = atr.load_atr_cache()
        assert reloaded["atr_split_times"].iloc[0] == json.dumps([14.68, 11.20])

        # Publication links the collector's name-keyed row to its canonical runner.
        canonical = pd.DataFrame([
            {"race_id": "r1", "horse_id": 1, "race_date": "2026-06-10", "track": "Kempton",
             "off_time": "18:10", "horse_name": "Moonshine"},
            {"race_id": "r1", "horse_id": 2, "race_date": "2026-06-10", "track": "Kempton",
             "off_time": "18:10", "horse_name": "Ghost Runner"},
        ])
        publish_dataset(canonical, atr={"sectionals": reloaded})
        linked = read_table("atr", "sectionals", linked=True)
        assert linked.match_status.tolist() == ["matched"]
        assert linked[["sl_race_id", "sl_horse_id"]].astype(str).values.tolist() == [["r1", "1"]]
        assert linked.loc[0, "atr_final_time"] == pytest.approx(72.98)
        assert linked.loc[0, "atr_efficiency_grade"] == "C"

    def test_dataset_without_atr_gives_missing_history(self):
        results = pd.DataFrame([
            {"race_id": "r1", "horse_id": 1, "race_date": "2026-06-10", "track": "Kempton",
             "off_time": "18:10", "horse_name": "Moonshine"},
        ])
        rolled = atr_features.published_atr_rolling(results)
        assert math.isnan(rolled["atr_finishing_speed_pct_avg_3"][0])
        assert rolled["atr_observation_count_5"][0] == 0
