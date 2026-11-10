"""A fresh backfill must not import bad rows or turn blocks into absence."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from src.ingestion import atr_rebuild as rebuild, attheraces as atr
from src.ingestion import atr_tabs
from src.ingestion.storage import read_json, writer_lock
from test_attheraces_scraper import _FakeFetcher
from test_atr_rich_collection import page


@pytest.fixture
def environment(tmp_path, monkeypatch):
    accepted = tmp_path/'accepted'
    accepted.mkdir()
    monkeypatch.setattr(rebuild.config,'DATA_DIR',str(tmp_path))
    monkeypatch.setattr(atr,'ATR_CACHE_DIR',str(accepted))
    monkeypatch.setattr(atr,'ATR_CACHE_FILE',str(accepted/'atr_sectionals.parquet'))
    monkeypatch.setattr(atr_tabs, 'ATR_CACHE_FILE', str(accepted/'atr_sectionals.parquet'))
    monkeypatch.setattr(atr,'ATR_INDEX_DIR',str(accepted/'index'))
    canonical = tmp_path/'canonical.sqlite'
    monkeypatch.setattr(rebuild.database,'DB_PATH',str(canonical))
    with sqlite3.connect(canonical) as con:
        con.execute('CREATE TABLE results(race_id TEXT,race_date TEXT,track TEXT,off_time TEXT,horse_name TEXT,horse_id TEXT,finish_position INTEGER)')
        con.executemany('INSERT INTO results VALUES (?,?,?,?,?,?,?)',[
            ('r1','2026-11-02','Wolverhampton','20:30','Macarone','sl-1',1),
            ('r2','2026-11-02','Dundalk','20:00','Irish Runner','sl-2',1)])
    pd.DataFrame([{'race_date':'2026-11-02','track':'Wolverhampton','off_time':'20:30',
                  'horse_name':'Macarone','atr_raceid':'1629208','atr_sectional_speeds':'[999]'}]).to_parquet(atr.ATR_CACHE_FILE)
    return tmp_path/'fresh', canonical


class Network(_FakeFetcher):
    def __init__(self, path, routes=None):
        super().__init__(routes or {f'/{kind}':(200,page(kind)) for kind in ('times','tools','stride')})
        self.raw_store_path = path
        self.last_response = {}
    def get(self,url,**kwargs):
        reply = super().get(url,**kwargs)
        self.last_response = {'url':url,'status':reply[0]}
        return reply
    def close(self):
        pass


def client_for(monkeypatch, root, routes=None):
    client = Network(root/'raw.sqlite',routes)
    monkeypatch.setattr(atr,'_Fetcher',lambda **kw:client)
    return client


def test_fresh_preparation_reuses_only_native_ids_and_excludes_unsupported_tracks(environment):
    root, _ = environment
    receipt = rebuild.prepare(root)
    assert receipt['queued_races'] == receipt['native_id_hints'] == 1
    assert receipt['excluded_unsupported_races'] == 1
    assert receipt['old_parsed_rows_imported'] == receipt['legacy_absence_decisions_imported'] == 0
    with rebuild.connect(root) as con:
        assert con.execute('SELECT count(*) FROM runners').fetchone()[0] == 0
        assert con.execute('SELECT native_id FROM jobs').fetchone()[0] == '1629208'
    with pytest.raises(RuntimeError,match='already prepared'):
        rebuild.prepare(root)


def test_worker_saves_verified_rich_records_and_resume_skips_completed_field(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    before = hashlib.sha256(Path(atr.ATR_CACHE_FILE).read_bytes()).hexdigest()
    client = client_for(monkeypatch,root)
    rebuild.run(root,max_races=1)
    assert len(client.calls) == 6  # five performance endpoints + bounded navigation audit
    with rebuild.connect(root) as con:
        row = json.loads(con.execute('SELECT payload FROM runners').fetchone()[0])
        assert row['sl_horse_id'] == 'sl-1'
        assert row['atr_sectional_speeds'] == [35.87,38.91,37.11]
        assert row['atr_stride_frequency'][0] == 2.46
    assert (root/'daily/2026-11-02.parquet').exists()
    assert hashlib.sha256(Path(atr.ATR_CACHE_FILE).read_bytes()).hexdigest() == before
    rebuild.run(root,max_races=1)
    assert len(client.calls) == 6
    assert read_json(root/'status.json')['state'] == 'complete'


def test_field_mismatch_is_reviewed_without_publishing_any_runners(environment,monkeypatch):
    root, canonical = environment
    with sqlite3.connect(canonical) as con:
        con.execute("UPDATE results SET horse_name='Different Horse' WHERE race_id='r1'")
    rebuild.prepare(root)
    client_for(monkeypatch,root)
    rebuild.run(root,max_races=1)
    status = read_json(root/'status.json')
    assert status['races'] == {'review':1} and status['saved_rows'] == 0


@pytest.mark.parametrize('finish_position,coverage',[(0,'finishers_only'),(2,'partial_tracking')])
def test_legitimate_source_omissions_keep_available_runners_with_explicit_coverage(environment,monkeypatch,finish_position,coverage):
    root, canonical = environment
    with sqlite3.connect(canonical) as con:
        con.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?)',
            ('r1','2026-11-02','Wolverhampton','20:30','Untracked Runner','sl-3',finish_position))
    rebuild.prepare(root)
    client_for(monkeypatch,root)
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        row = json.loads(con.execute('SELECT payload FROM runners').fetchone()[0])
        assert row['atr_field_coverage'] == coverage
        assert row['atr_field_expected_runners'] == 2 and row['atr_field_provided_runners'] == 1
        assert row['atr_field_missing_runners'] == [{'horse_name':'Untracked Runner','finish_position':finish_position}]
    assert read_json(root/'status.json')['races'] == {'data_partial':1}


def test_blocked_fragments_remain_pending_with_global_backoff(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    client = client_for(monkeypatch,root,{'/ajax/':(429,'')})
    rebuild.run(root,max_races=1)
    status = read_json(root/'status.json')
    assert status['races'] == {'retry':1} and status['saved_rows'] == 0
    assert status['global_retry_after'] > rebuild.utc_now().timestamp()
    assert len(client.calls) == 1


def test_unrecognised_html_is_not_recorded_as_no_data(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    client_for(monkeypatch,root,{'/ajax/':(200,'<html>new layout</html>')})
    rebuild.run(root,max_races=1)
    assert read_json(root/'status.json')['races'] == {'retry':1}


def test_confirmed_404_outcome_does_not_create_fake_runner_records(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    client_for(monkeypatch,root,{'/ajax/':(404,'')})
    rebuild.run(root,max_races=1)
    status = read_json(root/'status.json')
    assert status['races'] == {'no_data':1} and status['saved_rows'] == 0


def test_worker_keeps_waiting_after_a_block_and_obeys_stop_file(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    client = client_for(monkeypatch,root,{'/ajax/':(429,'')})
    monkeypatch.setattr(rebuild.time,'sleep',lambda seconds:(root/'STOP').write_text('stop'))
    rebuild.run(root)
    assert read_json(root/'status.json')['state'] == 'stopped'
    assert len(client.calls) == 1


def test_changed_collector_code_stops_before_more_requests(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        rebuild.set_control(con,'code_hashes',{'changed':'yes'})
    client = client_for(monkeypatch,root)
    rebuild.run(root)
    assert read_json(root/'status.json')['state'] == 'code_changed'
    assert not client.calls


def test_normal_collection_cannot_overlap_rebuild(environment):
    with writer_lock(Path(atr.ATR_CACHE_FILE).parent/'.collector.lock') as acquired:
        assert acquired
        with pytest.raises(RuntimeError,match='already running'):
            atr.backfill_atr_sectionals_for_races(pd.DataFrame())


def test_transient_service_failure_defers_one_item_without_an_hour_global_stall(environment,monkeypatch):
    root, _ = environment
    # Check the backoff policy, independent of time spent flushing fixture I/O
    # on a busy machine. The original two-second wall-clock window was flaky.
    frozen_now = rebuild.utc_now()
    monkeypatch.setattr(rebuild, 'utc_now', lambda: frozen_now)
    rebuild.prepare(root)
    client_for(monkeypatch,root,{'/ajax/':(503,'')})
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        now = rebuild.utc_now().timestamp()
        assert 58 < rebuild.control(con,'global_retry')-now <= 60
        assert con.execute('SELECT next_retry FROM jobs').fetchone()[0]-now > 298
        assert rebuild.control(con,'transient_streak') == 1


def test_transient_errors_escalate_but_throttling_and_challenges_keep_long_backoff():
    first = rebuild._retry_policy({'status':503},0,0)
    assert first['global_delay'] == 60 and first['item_delay'] == 300
    assert rebuild._retry_policy({'status':503},1,1)['global_delay'] == 120
    assert rebuild._retry_policy({'status':429},0,0)['global_delay'] == 3600
    assert rebuild._retry_policy({'status':403},0,0)['global_delay'] == 3600
    challenge = {'status':200,'reason':'client_challenge','url':atr.ATR_BASE+'/results/test'}
    assert rebuild._retry_policy(challenge,0,0)['scope'] == 'discovery'
    api_challenge = {'status':503,'reason':'client_challenge','url':atr.ATR_BASE+'/ajax/sectionals-razor/1/tools'}
    assert rebuild._retry_policy(api_challenge,0,0)['global_delay'] == 3600


def test_server_retry_after_is_never_shortened():
    policy = rebuild._retry_policy({'status':503,'retry_after_seconds':7200},0,0)
    assert policy['global_delay'] == policy['item_delay'] == 7200


def test_identical_race_endpoint_500_defers_only_that_race():
    response = {'status':500, 'url':atr.ATR_BASE+'/ajax/sectionals-razor/123/times', 'sha256':'a'*64}
    assert rebuild._retry_policy(response, 0, 0)['scope'] == 'global'
    assert rebuild._retry_policy(response, 1, 3, previous=response) == {
        'scope':'item', 'global_delay':0, 'item_delay':86400, 'transient':False}
    assert rebuild._retry_policy({**response, 'sha256':'b'*64}, 1, 3, previous=response)['scope'] == 'global'
    assert rebuild._retry_policy({**response, 'retry_after_seconds':7200}, 1, 3,
                                 previous=response)['global_delay'] == 7200


def test_a_successful_field_resets_transient_failure_escalation(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        rebuild.set_control(con,'transient_streak',3)
    client_for(monkeypatch,root)
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        assert rebuild.control(con,'transient_streak') == 0
