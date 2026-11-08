"""BHA completeness, identity, revisions, pacing and regular-source integration."""
from datetime import date, timedelta
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from src.ingestion import bha_stewards as bha, bha_reports, pipeline
from src.ingestion.storage import RawPayloadStore, writer_lock

DAY = '2026-11-06'
PDF = 'https://crate.horseracing.software/stewardsreport/?fixtureId=1&year=2026'


def report(**changes):
    return {'fixtureId': 1, 'year': '2026', 'date': DAY, 'course': 'AINTREE',
            'enquiry': '<p>ALPHA lost a shoe.</p>', 'pdfReportUrl': PDF, **changes}


def document(**changes):
    return {'course': 'AINTREE', 'race_date': DAY, 'parser_version': 1, 'pages': 1,
            'text': 'Full report including going, non-runners and suspensions.',
            'sections': {'going': 'Good', 'rail': 'Rail moved'},
            'races': [{'race_number': 1, 'off_time_local': '12:40', 'text': '.\nTHE AINTREE HANDICAP\nNothing to report'}],
            'printed_publication_labels': ['Generated today'], 'replacement_character_count': 0, **changes}


def response(payload=None, *, url='https://api09.horseracing.software/test', status=200, headers=None, body=None):
    def check():
        if status >= 400:
            raise requests.HTTPError(str(status))
    return SimpleNamespace(status_code=status, headers=headers or {}, url=url,
        content=body if body is not None else json.dumps(payload).encode(),
        json=lambda: payload, raise_for_status=check)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(bha.config, 'DATA_DIR', str(tmp_path))
    monkeypatch.setattr(bha, 'parse_document', lambda body: document())
    calls, pages = [], {}
    def factory(con, root, **kwargs):
        class FakeClient:
            requests = 0
            seconds = wait_seconds = 0
            def get(self, url, *, params=None, document=False):
                calls.append((url, params, document))
                self.requests += 1
                if document:
                    r = response(url=url, body=b'%PDF retained original')
                else:
                    records = pages.get(params['ondate'], [report(date=params['ondate'])])
                    payload = {'current_page': params['pagenum'], 'last_page': 1,
                               'total': len(records), 'data': records}
                    r = response(payload, url=requests.Request('GET', url, params=params).prepare().url)
                sha = RawPayloadStore(root/'raw.sqlite').put(r.url, r.status_code, r.content)
                return r, sha
            def close(self):
                pass
        return FakeClient()
    return tmp_path/'bha', factory, calls, pages


def test_regular_refresh_retains_full_json_and_pdf_and_restart_does_no_http(env):
    root, factory, calls, pages = env
    pages[DAY] = [report(runnerNotes='<p>Breathing issue reported.</p>', newField={'a': 2},
                         nonRacedayIncidents='Incident outside a race')]
    first = bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    assert first['state'] == 'ok' and first['fetched_documents'] == 1
    tables = bha.read_tables(root)
    row = tables['reports'].iloc[0]
    assert json.loads(row.source_record_json)['newField'] == {'a': 2}
    assert row.runner_notes_text == 'Breathing issue reported.'
    assert row.non_race_incidents_text == 'Incident outside a race'
    assert row.available_at == row.observed_at and row.available_at[:10] >= DAY
    assert tables['documents'].iloc[0].text.startswith('Full report')
    assert tables['documents'].iloc[0].going_text == 'Good'
    assert len(tables['race_notes']) == 1
    assert len(calls) == 2
    assert bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)['state'] == 'cached'
    assert len(calls) == 2


def test_empty_day_has_durable_receipt(env):
    root, factory, calls, pages = env
    pages[DAY] = []
    assert bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)['state'] == 'ok'
    assert bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)['state'] == 'cached'
    assert len(calls) == 1
    with bha.connect(root) as con:
        assert tuple(con.execute('SELECT state,total FROM days').fetchone()) == ('complete', 0)


def test_amendment_replaces_current_report_but_preserves_both_revisions_and_pdf_recheck(env):
    root, factory, calls, pages = env
    bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    pages[DAY] = [report(enquiry='<p>Amended finding.</p>')]
    bha.refresh_reports(DAY, DAY, root=root, client_factory=factory, recheck=True)
    with bha.connect(root) as con:
        assert con.execute('SELECT count(*) FROM reports').fetchone()[0] == 1
        assert con.execute('SELECT count(*) FROM revisions').fetchone()[0] == 2
    assert len(calls) == 4
    assert bha.read_tables(root)['reports'].iloc[0].enquiry_text == 'Amended finding.'


def test_disappearing_report_is_retained_and_marked_absent(env):
    root, factory, calls, pages = env
    bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    pages[DAY] = []
    bha.refresh_reports(DAY, DAY, root=root, client_factory=factory, recheck=True)
    table = bha.read_tables(root)['reports']
    assert len(table) == 1 and table.iloc[0].presence_status == 'absent_from_latest_day'


def test_wrong_day_does_not_publish_or_complete_a_date(env):
    root, factory, calls, pages = env
    pages[DAY] = [report(date='2026-11-05')]
    result = bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    assert result['state'] == 'partial' and result['failed'] == 1
    assert bha.read_tables(root) == {}
    with bha.connect(root) as con:
        assert con.execute('SELECT state FROM days').fetchone()[0] == 'retry'


def test_truncated_pagination_retains_previous_good_day(tmp_path):
    with bha.connect(tmp_path) as con:
        con.execute('INSERT INTO days(day) VALUES (?)', (DAY,))
        bha.store_report(con, report(), source_url='original', source_sha256='original')
        client = SimpleNamespace(get=lambda *a, **k: (response({'current_page': 1, 'last_page': 1,
                'total': 2, 'data': [report(enquiry='changed')]}), 'new'))
        with pytest.raises(ValueError, match='truncated'):
            bha.fetch_day(con, client, DAY)
        assert con.execute('SELECT count(*) FROM revisions').fetchone()[0] == 1
        assert con.execute('SELECT state FROM days').fetchone()[0] == 'pending'


def test_bad_pdf_keeps_source_report_and_raw_bytes_retryable(env, monkeypatch):
    root, factory, calls, pages = env
    monkeypatch.setattr(bha, 'parse_document', lambda b: document(race_date='2026-11-05'))
    result = bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    assert result['pending_days'] == 0 and result['pending_documents'] == 1 and result['failed'] == 1
    assert set(bha.read_tables(root)) == {'reports'}
    assert RawPayloadStore(root/'raw.sqlite').latest(PDF) == b'%PDF retained original'


def test_legacy_import_preserves_mixed_dates_without_claiming_day_completeness(tmp_path):
    p = tmp_path/'old.csv'
    pd.DataFrame([{'fixture_id': 1, 'year': 2026, 'date': DAY, 'course': 'AINTREE',
                   'enquiry_html': '<p>Original</p>'},
                  {'fixture_id': 2, 'year': 2025, 'date': '2025-11-06 00:00:00', 'course': 'AINTREE',
                   'enquiry_html': '<p>Older</p>'}]).to_csv(p, index=False)
    original = p.read_bytes()
    with bha.connect(tmp_path/'bha') as con:
        bha.import_legacy(con, p)
        bha.import_legacy(con, p)
        assert con.execute('SELECT count(*) FROM reports').fetchone()[0] == 2
        assert con.execute('SELECT count(*) FROM days').fetchone()[0] == 0
    assert p.read_bytes() == original


def test_cooldown_survives_restart_and_token_is_not_sent_to_document_host(tmp_path, monkeypatch):
    monkeypatch.setattr(bha, '_extract_bearer_token', lambda session: 'public-test-token')
    monkeypatch.setattr(bha.time, 'sleep', lambda n: None)
    sent = []
    class Session:
        headers = {}
        def get(self, url, **kwargs):
            sent.append(kwargs)
            return response(status=429, url=url, headers={'Retry-After': '3600'})
    with bha.connect(tmp_path) as con:
        client = bha.Client(con, tmp_path, session=Session())
        with pytest.raises(bha.Deferred):
            client.get(PDF, document=True)
        assert sent[0]['headers']['Authorization'] is None
    with bha.connect(tmp_path) as con:
        with pytest.raises(bha.Deferred):
            bha.Client(con, tmp_path, session=Session()).get(bha.BHA_STEWARDS_API)
    assert len(sent) == 1


def test_html_legacy_notes_and_distinct_sections_are_preserved():
    row = bha_reports.normalise_report(report(enquiryComments='', enquiry='<p>Enquiry</p> ~ <p>Runner note</p>',
                        fixtureComments='<b>Fixture</b>', nonRacedayIncidents='<p>Other</p>'))
    assert row['enquiry_text'] == 'Enquiry' and row['runner_notes_text'] == 'Runner note'
    assert row['fixture_comments_text'] == 'Fixture' and row['non_race_incidents_text'] == 'Other'


def parse_text(monkeypatch, text):
    import pypdf
    page = SimpleNamespace(extract_text=lambda: text, get_contents=lambda: None)
    monkeypatch.setattr(pypdf, 'PdfReader', lambda _: SimpleNamespace(pages=[page], is_encrypted=False))
    return bha_reports.parse_document(b'%PDF input')


def test_pdf_extracts_context_and_races_without_backdating_generated_footer(monkeypatch):
    parsed = parse_text(monkeypatch,
        "AINTREE STEWARDS' REPORT\nFriday 06 November 2026\nStewards: Official\nGoing: Good\n"
        "Rail: Moved 10 yards\nNon-Runners\nALPHA: Infected foot\nRace 1 - 12:40pm.\nTHE AINTREE HANDICAP\n"
        "Nothing to report\nRace 2 - 1:15pm.\nTHE AINTREE HURDLE\nReported breathing issue.\n"
        "Suspensions\nRider: 2 days\nPublished: Today\n")
    assert parsed['race_date'] == DAY and parsed['printed_publication_labels'] == ['Today']
    assert parsed['sections']['non_runners'] == 'ALPHA: Infected foot'
    assert parsed['races'][1]['off_time_local'] == '13:15'
    assert 'Suspensions' not in parsed['races'][1]['text']
    assert 'available_at' not in parsed


def test_revised_off_times_are_not_race_notes(monkeypatch):
    # Shapes taken from real reports (2021:19714, 2022:1137, 2024:10126, 2024:50079).
    parsed = parse_text(monkeypatch,
        "AINTREE STEWARDS' REPORT\nFriday 06 November 2026\nStewards: Official\nGoing: Good\n"
        "Race 1 - 12:45pm\nRace 2 - 1:20pm (previously 1:15pm)\n"
        "Race Delay: At the request of the BHA Racing Department, the following races have been delayed:\n"
        "Race 1 - 12:40pm (+5) now 12:45pm\nNon-Runners\nALPHA: Infected foot\n"
        "Race 1 - 12:40pm.\nTHE AINTREE HANDICAP\nNothing to report\n"
        "Race 2 - 1:15pm.\nTHE AINTREE HURDLE\nReported breathing issue.\n")
    assert [(r['race_number'], r['off_time_local']) for r in parsed['races']] == [(1, '12:40'), (2, '13:15')]
    assert parsed['races'][0]['text'] == '.\nTHE AINTREE HANDICAP\nNothing to report'
    assert parsed['sections']['going'] == 'Good'
    assert parsed['sections']['non_runners'] == 'ALPHA: Infected foot'


def test_unrecognised_race_headings_are_retained_for_review(monkeypatch):
    with pytest.raises(ValueError, match='full stop'):
        parse_text(monkeypatch, "AINTREE STEWARDS' REPORT\nFriday 06 November 2026\nGoing: Good\n"
                                "Race 1 - 12:40pm\nTHE AINTREE HANDICAP\nNothing to report\n")


def test_stored_parser_1_revised_times_are_dropped_before_publication(env, monkeypatch):
    from src.ingestion import publication
    from src.source_dataset import registry_root
    root, factory, calls, pages = env
    # Parser 1 stored the revised time as a second note for race 1.
    monkeypatch.setattr(bha, 'parse_document', lambda body: document(races=[
        {'race_number': 1, 'off_time_local': '12:45', 'text': '(+5) now 12:45pm'},
        {'race_number': 1, 'off_time_local': '12:40', 'text': '.\nTHE AINTREE HANDICAP\nNothing to report'}]))
    bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    notes = bha.read_tables(root)['race_notes']
    assert notes[['race_number', 'off_time_local']].values.tolist() == [[1, '12:40']]
    monkeypatch.setattr(bha, 'default_root', lambda: root)
    canonical = pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'race_date', 'track', 'off_time'])
    receipt = publication.publish_bha(canonical)
    published = pd.read_parquet(registry_root()/'bha'/receipt['version']/receipt['tables']['race_notes']['file'])
    assert len(published) == 1


def test_regular_coordinator_and_command_include_bha(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.config, 'DATA_DIR', str(tmp_path))
    asked = []
    monkeypatch.setattr(bha, 'refresh_reports', lambda *a, **k: asked.append((a, k)) or {'state': 'ok'})
    day = (date.today()-timedelta(days=1)).isoformat()
    result = pipeline.refresh_sources(1, sources=('bha',))
    assert result['sources']['bha']['state'] == 'ok' and asked[0][0] == (day, day)
    assert 'bha' in pipeline.DEFAULT_SOURCES
    saved = json.loads((tmp_path/'source_refresh/status.json').read_text())
    assert saved['sources']['bha']['health']['attempt_state'] == 'ok'


def test_second_collector_cannot_make_requests(env):
    root, factory, calls, pages = env
    with writer_lock(root/'.collector.lock') as acquired:
        assert acquired
        assert bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)['state'] == 'busy'
    assert not calls


def test_stop_marker_prevents_requests_without_discarding_the_queue(env):
    root, factory, calls, pages = env
    root.mkdir()
    (root/'STOP').touch()
    with bha.connect(root) as con:
        bha.set_control(con, 'retry_after', bha.now().timestamp()+3600)
    assert bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)['state'] == 'stopped'
    assert not calls


@pytest.mark.parametrize('value,expected', [('bad date', 0), ('nan', 0), ('3600', 3600)])
def test_retry_after_parsing_cannot_cancel_provider_cooldown(value, expected):
    assert bha.retry_delay(value) == expected


@pytest.mark.parametrize('delay', [float('nan'), float('inf'), 0, -.1])
def test_invalid_request_gap_cannot_disable_pacing(tmp_path, delay):
    with bha.connect(tmp_path) as con:
        with pytest.raises(ValueError, match='delay'):
            bha.Client(con, tmp_path, request_delay=delay)


def test_fixture_source_publication_retains_recorded_availability_and_native_keys(env, monkeypatch):
    from src.ingestion import publication
    from src.source_dataset import registry_root, verify_source
    root, factory, calls, pages = env
    bha.refresh_reports(DAY, DAY, root=root, client_factory=factory)
    monkeypatch.setattr(bha, 'default_root', lambda: root)
    canonical = pd.DataFrame(columns=['race_id', 'horse_id', 'horse_name', 'race_date', 'track', 'off_time'])
    receipt = publication.publish_bha(canonical)
    assert set(receipt['reports']) == {'reports', 'documents', 'race_notes'}
    verify_source(receipt)
    base = registry_root()/'bha'/receipt['version']
    timing = pd.read_parquet(base/receipt['tables']['observations_reports']['file'])
    assert timing.availability_basis.eq('recorded').all()
    assert timing.available_at.iloc[0] >= pd.Timestamp(DAY, tz='UTC')
    assert publication.publish_bha(canonical)['version'] == receipt['version']


def test_drain_continues_other_dates_after_an_item_failure(monkeypatch):
    import sys
    answers = iter([{'state': 'partial', 'failed': 1, 'fetched_days': 0, 'fetched_documents': 0},
                    {'state': 'ok'}])
    calls = []
    monkeypatch.setattr(sys, 'argv', ['bha', '--drain'])
    monkeypatch.setattr(bha, 'refresh_reports', lambda *a, **k: calls.append(k) or next(answers))
    bha.main()
    assert len(calls) == 2


def test_drain_waits_without_spamming_during_provider_cooldown(monkeypatch):
    import sys
    answers = iter([{'state': 'waiting', 'retry_after': bha.now().timestamp()+3600}, {'state': 'ok'}])
    sleeps = []
    monkeypatch.setattr(sys, 'argv', ['bha', '--drain'])
    monkeypatch.setattr(bha, 'refresh_reports', lambda *a, **k: next(answers))
    monkeypatch.setattr(bha.time, 'sleep', sleeps.append)
    bha.main()
    assert sleeps == [30]
