"""Current public ATR routes, exact native IDs and resumable collection."""
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.ingestion import attheraces as atr
from src.ingestion import atr_tabs
from src.ingestion import auxiliary
from test_attheraces_scraper import _FakeFetcher, _times_fixture, _tools_fixture, _stride_fixture


DATE = "02-November-2026"
CARD = f"/racecard/Wolverhampton/{DATE}/2030"


def article(path=CARD, raceid="1629208"):
    return (f'<article class="panel-content--medium"><h3><a href="{path}">20:30 race</a></h3>'
            f'<a href="/form/horse/Macarone/3544146?raceid={raceid}">Winner</a>'
            f'<a href="/form/trainer/X/5?raceid={raceid}">Trainer</a></article>')


def routes(index=None):
    return {"/results/": (200, article() if index is None else index),
            "/times": (200, str(_times_fixture())),
            "/tools": (200, str(_tools_fixture())),
            "/stride": (200, str(_stride_fixture()))}


def test_index_scopes_native_ids_to_their_exact_race_article():
    second = f"/racecard/Chelmsford-City/{DATE}/2015"
    html = '<a href="/form/horse/Unrelated/1?raceid=999">Nav</a>' + article() + article(second, "1628437")
    assert atr._parse_results_index(html) == {CARD: "1629208", second: "1628437"}


def test_conflicting_native_ids_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        atr._parse_results_index(article() + article(raceid="99"))
    conflicting = article().replace('</article>', '<a href="/form/horse/X/1?raceid=99">X</a></article>')
    with pytest.raises(ValueError, match="Conflicting"):
        atr._parse_results_index(conflicting)


def test_active_sectional_tab_precedes_unrelated_legacy_query_ids():
    html = '<a href="/form/horse/X/1?raceid=99">Nav</a>'
    html += '<a href="/ajax/sectionals-razor/1629208/times">Times</a>'
    assert atr._extract_raceid(html) == "1629208"
    assert atr._extract_raceid(html + '/ajax/sectionals-razor/123/tools') is None
    assert atr._extract_raceid('?raceid=12" ?raceid=34"') is None


@pytest.mark.parametrize("path", ["/form/horse/Macarone/GB/3544146?raceid=1629208",
                                  "/form/horse/Macarone/3544146?raceid=1629208",
                                  "/horse/Macarone/GB/3544146"])
def test_country_segment_in_current_horse_link_retains_native_id(path):
    entry = BeautifulSoup(f'<div><a class="horse__link" href="{path}">Macarone</a></div>', 'html.parser').div
    assert atr._runner_meta(entry)["atr_horse_id"] == "3544146"


def test_current_workflow_uses_one_index_per_day_and_never_needs_racecard():
    other = f"/racecard/Wolverhampton/{DATE}/2000"
    client = _FakeFetcher(routes(article() + article(other, "1629207")))
    for time in ("2030", "2000"):
        outcome, rows = atr._fetch_one(client, "Wolverhampton", DATE, time)
        assert outcome == atr.DATA and len(rows) == 1
    assert len(client.calls) == 11
    assert sum('/results/' in url for url in client.calls) == 1
    assert not any('/racecard/' in url for url in client.calls)
    assert all('/ajax/sectionals-razor/' in url for url in client.calls[1:])


def test_accepted_index_survives_new_session_and_challenged_index(tmp_path):
    first = _FakeFetcher(routes())
    first.index_dir = tmp_path
    assert atr._fetch_one(first, "Wolverhampton", DATE, "2030")[0] == atr.DATA
    receipt = json.loads((tmp_path/f"{DATE}.json").read_text())
    assert receipt['race_ids'][CARD] == "1629208"
    second = _FakeFetcher({**routes(), '/results/': (429, '')})
    second.index_dir = tmp_path
    assert atr._fetch_one(second, "Wolverhampton", DATE, "2030")[0] == atr.DATA
    assert len(second.calls) == 5 and not any('/results/' in url for url in second.calls)


def test_challenge_is_never_saved_as_a_missing_race_or_index(tmp_path):
    client = _FakeFetcher({'/results/': (200, '<title>Client Challenge</title>')})
    client.index_dir = tmp_path
    assert atr._fetch_one(client, "Wolverhampton", DATE, "2030")[0] == atr.BLOCKED
    assert not list(tmp_path.iterdir()) and len(client.calls) == 1
    client.routes = routes()
    assert atr._fetch_one(client, "Wolverhampton", DATE, "2030")[0] == atr.DATA


def test_transport_retains_real_http_status_and_body_before_classification(tmp_path, monkeypatch):
    from src.ingestion.storage import RawPayloadStore
    client = atr._Fetcher()
    client.raw_store_path = tmp_path/'raw.sqlite'
    monkeypatch.setattr(client, '_throttle', lambda: None)
    fake = SimpleNamespace(get=lambda *a, **kw: SimpleNamespace(status_code=200, text='<title>Client Challenge</title>'),
                           close=lambda: None)
    client.mode, client._req = 'requests', fake
    try:
        assert client.get('https://www.attheraces.com/results/test') == (429, '')
        store = RawPayloadStore(client.raw_store_path)
        assert store.latest('https://www.attheraces.com/results/test') == b'<title>Client Challenge</title>'
        with store.connection() as con:
            assert con.execute('SELECT status FROM observations').fetchone()[0] == 200
    finally:
        client.close()


def test_index_redirecting_to_a_different_day_is_retryable():
    client = _FakeFetcher(routes(article(CARD.replace('02-November', '05-November'))))
    assert atr._fetch_one(client, "Wolverhampton", DATE, "2030")[0] == atr.BLOCKED


def test_later_blocked_tab_does_not_publish_incomplete_race():
    client = _FakeFetcher({**routes(), '/tools': (429, '')})
    outcome, rows = atr._fetch_one(client, "Wolverhampton", DATE, "2030")
    assert outcome == atr.BLOCKED and rows is None
    assert not any('/stride' in url for url in client.calls)


def test_publication_window_uses_utc_off_time_and_exact_48_hour_boundary():
    frame = pd.DataFrame({'race_date': ['2026-11-03']*3,
                          'off_time': ['12:00:00', '1201', '11:59']})
    now = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
    assert atr._publication_ready(frame, now=now).tolist() == [True, False, True]


def test_coordinator_defers_young_races_and_keeps_them_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(auxiliary.config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(auxiliary, '_utc_now', lambda: datetime(2026, 11, 5, 12, tzinfo=timezone.utc))
    monkeypatch.setattr(atr, 'load_atr_cache', lambda: pd.DataFrame())
    monkeypatch.setattr(atr, '_load_no_data_race_keys', lambda: set())
    fields = pd.DataFrame({'race_date': ['2026-11-04', '2026-11-02'],
                           'track': ['Wolverhampton']*2, 'off_time': ['20:30']*2})
    monkeypatch.setattr(auxiliary, 'canonical_fields', lambda *a: fields)
    def backfill(frame, **kwargs):
        assert frame.race_date.tolist() == ['2026-11-02']
        return {'fetched_races': 1, 'resolved_races': 1}
    monkeypatch.setattr(atr, 'backfill_atr_sectionals_for_races', backfill)
    result = auxiliary.refresh_atr('2026-11-02', '2026-11-04')
    assert result['pending_races'] == result['deferred_recent_races'] == 1
    assert result['state'] == 'partial'
    fields.drop(index=1, inplace=True)
    monkeypatch.setattr(atr, 'backfill_atr_sectionals_for_races', lambda *a, **kw: pytest.fail('HTTP for young race'))
    result = auxiliary.refresh_atr('2026-11-02', '2026-11-04')
    assert result['fetched_races'] == 0 and result['pending_races'] == 1
    assert 'publication window' in result['reason']


def test_direct_backfill_also_defers_young_races_before_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(atr, 'ATR_CACHE_FILE', str(tmp_path/'atr.parquet'))
    monkeypatch.setattr(atr_tabs, 'ATR_CACHE_FILE', str(tmp_path/'atr.parquet'))
    monkeypatch.setattr(atr, 'ATR_NO_DATA_FILE', str(tmp_path/'absent.json'))
    monkeypatch.setattr(atr, '_Fetcher', lambda **kw: pytest.fail('HTTP for young race'))
    races = pd.DataFrame({'race_date': [datetime.now(timezone.utc).date().isoformat()],
                          'track': ['Wolverhampton'], 'off_time': ['12:00']})
    result = atr.backfill_atr_sectionals_for_races(races)
    assert result['deferred_recent_races'] == 1 and result['fetched_races'] == 0
    assert not (tmp_path/'absent.json').exists()
