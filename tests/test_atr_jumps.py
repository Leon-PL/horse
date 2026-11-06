"""Jumps tab contracts, checked against a retained live Chepstow response."""
from dataclasses import asdict
import json
from pathlib import Path

from bs4 import BeautifulSoup
import pytest

from src.ingestion import atr_jumps as jumps

HTML = (Path(__file__).parent / "fixtures/atr_current/jumpsdata.html").read_text(encoding="utf-8")
RACE_ID = "1629661"
URL = "https://www.attheraces.com/racecard/Chepstow/04-November-2026/1220"
EXPECTED = ["3399904", "3585720", "3508309", "3409723", "3586853"]


def modified(action):
    soup = BeautifulSoup(HTML, "html.parser")
    action(soup)
    return jumps.parse_jumps_data(str(soup), RACE_ID)


def test_real_runners_units_grade_and_course_par():
    result = jumps.parse_jumps_data(HTML, RACE_ID, expected_horse_ids=EXPECTED)
    assert result.state == "data"
    winner, second, third = result.rows
    assert winner["atr_horse_id"] == "3399904"
    assert winner["atr_saddle_cloth"] == "5"
    assert winner["atr_official_position"] == "1"
    assert [winner[x] for x in jumps.LABELS.values()] == [15.4, 15.1, 10.3, 13.6, "C", 83]
    assert second["atr_jump_fluency_grade"] == "A"
    assert third["atr_jump_consistency_pct"] == 94
    assert third["atr_jump_fluency_grade"] == "D"
    assert result.context["atr_jump_course_winner_speed_lost_pct"] == 11.42
    # The historical benchmark was revised AFTER this race. Preserve that date.
    assert result.context["atr_jumps_par_updated_date"] == "2026-11-05"
    assert len(result.context["atr_jumps_help"]) == 6
    assert "atr_efficiency_grade" not in winner


def test_missing_nonfinishers_are_coverage_not_perfect_zero_jumps():
    result = jumps.parse_jumps_data(HTML, RACE_ID, expected_horse_ids=EXPECTED)
    assert len(result.rows) == 3
    assert result.context["atr_jumps_coverage"] == "partial"
    assert result.context["atr_jumps_missing_horse_ids"] == ["3409723", "3586853"]
    assert result.context["atr_jumps_expected_runners"] == 5
    assert jumps.parse_jumps_data(HTML, RACE_ID).context["atr_jumps_coverage"] == "unchecked"


def test_column_order_does_not_define_measurements():
    def reverse(soup):
        for entry in soup.select(".card-entry"):
            pills = entry.select(".text-pill")
            if pills:
                parents = [p.parent for p in pills]
                for pill, parent in zip(reversed(pills), parents):
                    parent.append(pill.extract())
    result = modified(reverse)
    assert result.state == "data"
    assert result.rows[0]["atr_jump_speed_lost_early_pct"] == 15.4
    assert result.rows[0]["atr_jump_consistency_pct"] == 83


def test_missing_value_stays_missing():
    result = modified(lambda soup: setattr(soup.select_one('.text-pill'), 'string', '—'))
    assert result.state == "data"
    assert result.rows[0]["atr_jump_speed_lost_early_pct"] is None
    assert result.rows[0]["atr_jump_speed_lost_middle_pct"] == 15.1


def test_unknown_new_measurement_is_retained():
    def extra(soup):
        pill = soup.new_tag('span', attrs={'class': 'text-pill', 'title': 'New Metric: 1.2 yards'})
        pill.string = '1.2 yards'
        soup.select('.card-body')[1].select_one('.card-entry').append(pill)
    result = modified(extra)
    assert result.state == "data"
    assert result.rows[0]["atr_jumps_source_metrics"][-1] == {
        "label": "New Metric", "display": "1.2 yards", "title": "New Metric: 1.2 yards", "field": None,
    }


@pytest.mark.parametrize('display', ['13.6 mph', 'NaN%', 'infinity%', '13.6% changed'])
def test_unit_or_value_drift_is_retryable_not_absence(display):
    result = modified(lambda soup: setattr(soup.select_one('.text-pill'), 'string', display))
    assert result.state == "unresolved"
    assert result.rows == []


def test_unknown_grade_preserved_and_flagged():
    result = modified(lambda soup: setattr(soup.select('[title^="Jumping Fluency Grade:"]')[0], 'string', 'AB'))
    assert result.state == "data"
    assert result.rows[0]["atr_jump_fluency_grade"] == "AB"
    assert result.rows[0]["atr_jumps_quality_flags"] == ["unknown_fluency_grade"]


def test_wrong_race_cannot_be_written_under_requested_id():
    result = jumps.parse_jumps_data(HTML, "1629662")
    assert result.state == "unresolved" and not result.rows
    assert "requested race" in result.reason


def test_unexpected_runner_and_duplicate_identity_rejected():
    result = jumps.parse_jumps_data(HTML, RACE_ID, expected_horse_ids=EXPECTED[:1])
    assert result.state == "unresolved" and not result.rows
    def duplicate(soup):
        links = soup.select('.card-body')[1].select('a.horse__link')
        links[1]['href'] = links[0]['href']
    result = modified(duplicate)
    assert result.state == "unresolved" and not result.rows


def test_missing_semantic_label_is_not_positionally_guessed():
    def remove(soup):
        del soup.select_one('.text-pill')['title']
    assert modified(remove).state == "unresolved"


@pytest.mark.parametrize('html', ['', '<html>Client Challenge</html>', '<h1>Full Result</h1>'])
def test_unrecognised_response_is_not_a_terminal_no_data_decision(html):
    assert jumps.parse_jumps_data(html, RACE_ID).state == "unresolved"


class FakeFetcher:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.last_response = {"retry_after_seconds": 300}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_one_live_shape_fetch_has_provenance_and_no_discovery_requests():
    client = FakeFetcher((200, HTML))
    result = jumps.fetch_jumps_data(RACE_ID, racecard_url=URL, fetcher=client, expected_horse_ids=EXPECTED)
    assert result.state == "data"
    assert client.calls == [(f'https://www.attheraces.com/ajax/sectionals-razor/{RACE_ID}/jumpsdata',
                             {'referer': URL})]
    assert len(result.context['sha256']) == 64
    assert result.context['retrieved_at'].endswith('+00:00')
    assert result.context['requested_racecard_url'] == URL


@pytest.mark.parametrize('response,state', [(None, 'blocked'), ((429, ''), 'blocked'),
                                         ((503, ''), 'blocked'), ((200, 'Client Challenge'), 'blocked'),
                                         ((404, ''), 'not_found'), ((200, ''), 'unresolved')])
def test_fetch_classifies_errors_and_honours_retry_metadata(response, state):
    result = jumps.fetch_jumps_data(RACE_ID, racecard_url=URL, fetcher=FakeFetcher(response))
    assert result.state == state and not result.rows
    assert result.context['request']['retry_after_seconds'] == 300


def test_bounded_cli_writes_observation_only(tmp_path, monkeypatch):
    observation = jumps.parse_jumps_data(HTML, RACE_ID)
    monkeypatch.setattr(jumps, 'fetch_jumps_data', lambda *args, **kwargs: observation)
    target = tmp_path / 'observation.json'
    assert jumps.main(['--race-id', RACE_ID, '--racecard-url', URL, '--output', str(target)]) == 0
    assert json.loads(target.read_text()) == asdict(observation)


@pytest.mark.parametrize('url', ['http://www.attheraces.com/racecard/X/04-November-2026/1220',
                                'https://example.com/racecard/X/04-November-2026/1220',
                                'https://www.attheraces.com/results'])
def test_invalid_source_context_never_makes_request(url):
    client = FakeFetcher((200, HTML))
    with pytest.raises(ValueError):
        jumps.fetch_jumps_data(RACE_ID, racecard_url=url, fetcher=client)
    assert client.calls == []
