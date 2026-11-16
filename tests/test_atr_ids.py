"""Native ID discovery must be batched, dated and independent of telemetry."""
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from src.ingestion import atr_ids,atr_rebuild as rebuild,attheraces as atr
from src.ingestion.storage import RawPayloadStore,atomic_json,read_json
from test_atr_current_routes import article,DATE,CARD
from test_atr_rebuild import client_for
from test_atr_rich_collection import page
import test_atr_rebuild as rebuild_tests

environment = rebuild_tests.environment


def add_second(canonical):
    with sqlite3.connect(canonical) as con:
        con.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?)',
                    ('r3','2026-11-02','Wolverhampton','21:00','Other Runner','sl-3',1))


def test_one_index_saves_all_ids_even_when_the_first_tab_is_blocked(environment,monkeypatch):
    root,canonical = environment
    add_second(canonical)
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        con.execute('UPDATE jobs SET native_id=NULL')
    index = article()+article(CARD.replace('2030','2100'),'1629209')
    client = client_for(monkeypatch,root,{'/results/':(200,index),'/ajax/':(429,'')})
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        assert dict(con.execute('SELECT race_id,native_id FROM jobs'))=={'r1':'1629208','r3':'1629209'}
        assert con.execute('SELECT count(*) FROM runners').fetchone()[0]==0
    assert len(client.calls)==2 and sum('/results/' in url for url in client.calls)==1


def test_saved_index_is_recovered_without_http_during_discovery_cooldown(environment,monkeypatch):
    root,_ = environment
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        con.execute('UPDATE jobs SET native_id=NULL')
        rebuild.set_control(con,'discovery_retry',rebuild.utc_now().timestamp()+3600)
    atomic_json(Path(atr.ATR_INDEX_DIR)/f'{DATE}.json',{'source_url':f'{atr.ATR_BASE}/results/{DATE}',
                'source_sha256':'a'*64,'race_ids':{CARD:'1629208'}})
    client = client_for(monkeypatch,root)
    rebuild.run(root,max_races=1)
    assert len(client.calls)==5 and not any('/results/' in url for url in client.calls)
    assert read_json(root/'status.json')['saved_rows']==1


def test_saved_results_page_queues_exact_links_and_protects_existing_evidence(environment, tmp_path):
    root, _ = environment
    rebuild.prepare(root)
    saved = tmp_path/'Browser download.html'
    canonical = f'<link rel="canonical" href="{atr.ATR_BASE}/results/{DATE}">'
    saved.write_text(canonical + article(), encoding='utf-8')
    index = atr_ids.import_saved_results(saved, atr.ATR_INDEX_DIR)
    with rebuild.connect(root) as con:
        con.execute('UPDATE jobs SET native_id=NULL')
        seen = {}
        assert atr_ids.recover_cached(con, atr.ATR_INDEX_DIR, seen=seen)[0]['ids_added_or_corrected'] == 1
        assert atr_ids.recover_cached(con, atr.ATR_INDEX_DIR, seen=seen) == []
        assert con.execute('SELECT native_id FROM jobs').fetchone()[0] == '1629208'
    saved.write_text(canonical + article(CARD, '99999'), encoding='utf-8')
    with pytest.raises(ValueError, match='would remove or change'):
        atr_ids.import_saved_results(saved, atr.ATR_INDEX_DIR)
    assert read_json(Path(index['index']))['race_ids'] == {CARD: '1629208'}


def test_saved_replay_supplements_partial_results_without_overwriting_ids(environment, tmp_path):
    root, canonical = environment
    add_second(canonical)
    rebuild.prepare(root)
    other = CARD.replace('2030', '2100')
    saved = tmp_path/'replays.html'
    canonical_link = f'<link rel="canonical" href="{atr.ATR_BASE}/replays/{DATE}">'
    replay_link = f'/replays/{DATE.lower()}/wolverhampton/2100'
    replay_article = (f'<article class="article-preview"><a href="{other}">Result</a>'
                      f'<a href="{replay_link}" data-video-raceid="1629209">Full replay</a>'
                      f'<a href="{replay_link}?finish=yes" data-video-raceid="1629209">Finish</a></article>')
    existing = Path(atr.ATR_INDEX_DIR)/f'{DATE}.json'
    atomic_json(existing, {'race_ids':{CARD:'1629208'}})
    saved.write_text(canonical_link + replay_article, encoding='utf-8')
    imported = atr_ids.import_saved_results(saved, atr.ATR_INDEX_DIR)
    assert read_json(existing)['race_ids'] == {CARD:'1629208'}
    assert read_json(Path(imported['index']))['race_ids'] == {other:'1629209'}
    with rebuild.connect(root) as con:
        rebuild.set_control(con, 'discovery_retry', rebuild.utc_now().timestamp()+3600)
        reports = atr_ids.recover_cached(con, atr.ATR_INDEX_DIR)
        assert sum(report['ids_added_or_corrected'] for report in reports) == 1
        assert dict(con.execute('SELECT race_id,native_id FROM jobs')) == {'r1':'1629208','r3':'1629209'}
        assert con.execute("SELECT source_url FROM id_evidence WHERE race_id='r3'").fetchone()[0].endswith(f'/replays/{DATE}')
        conflict = atr_ids.apply_index(con, DATE, {other:'999'}, {'source_url':f'{atr.ATR_BASE}/replays/{DATE}'})
        assert conflict['conflicts'] == ['r3']
        assert con.execute("SELECT native_id FROM jobs WHERE race_id='r3'").fetchone()[0] == '1629209'
    saved.write_text(canonical_link + replay_article.replace('wolverhampton/2100','wolverhampton/2030'), encoding='utf-8')
    with pytest.raises(ValueError, match='Conflicting replay links'):
        atr_ids.import_saved_results(saved, atr.ATR_INDEX_DIR)


def test_fresh_known_race_precedes_a_newer_retry(environment):
    root, canonical = environment
    add_second(canonical)
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        con.execute("UPDATE jobs SET native_id='1629209',state='retry' WHERE race_id='r3'")
        job = rebuild.pick_job(con, rebuild.utc_now().timestamp(), '2026-11-05 00:00', discovery_allowed=False)
        assert job['race_id'] == 'r1'


def test_incomplete_saved_day_does_not_starve_other_missing_dates(environment,monkeypatch):
    root,canonical=environment
    add_second(canonical)
    rebuild.prepare(root)
    client=client_for(monkeypatch,root,{'/results/':(200,article())})
    with rebuild.connect(root) as con:
        con.execute('UPDATE jobs SET native_id=NULL')
        client.on_index=lambda day,mapping,receipt:atr_ids.apply_index(con,day,mapping,receipt)
        now=rebuild.utc_now().timestamp()
        report=rebuild.discover_ids(con,client,now,'2026-11-05 00:00')
        assert report['outcome']==atr.DATA and report['ids_added']==1
        assert con.execute('SELECT next_retry FROM id_discovery').fetchone()[0]>now+86300
        assert rebuild.discover_ids(con,client,now+61,'2026-11-05 00:00') is None
    assert len(client.calls)==1


def test_results_lookup_gets_the_window_when_cooldown_expires_mid_field(environment, monkeypatch):
    root, canonical = environment
    add_second(canonical)
    rebuild.prepare(root)
    routes = {'/1629209/times': (429, '')}
    routes.update({f'/{kind}':(200,page(kind)) for kind in ('times','tools','stride')})
    routes['/results/'] = (200, article()+article(CARD.replace('2030','2100'),'1629209'))
    client = client_for(monkeypatch, root, routes)
    clock = rebuild.utc_now()
    monkeypatch.setattr(rebuild, 'utc_now', lambda: clock)
    original_get = client.get

    def get(url, **kwargs):
        nonlocal clock
        if '/times' in url:
            clock += pd.Timedelta(seconds=10)
        return original_get(url, **kwargs)

    monkeypatch.setattr(client, 'get', get)
    with rebuild.connect(root) as con:
        rebuild.set_control(con, 'inventory_retry', clock.timestamp()+5)
        rebuild.set_control(con, 'discovery_retry', clock.timestamp()+5)
    rebuild.run(root, max_races=2)
    first_page = next(url for url in client.calls if '/ajax/' not in url)
    assert first_page.endswith(f'/results/{DATE}')
    with rebuild.connect(root) as con:
        assert rebuild.control(con, 'last_id_discovery')['ids_added'] == 1
        assert con.execute("SELECT native_id FROM jobs WHERE race_id='r3'").fetchone()[0] == '1629209'


def test_index_uses_exact_bst_adjusted_time_and_never_guesses_ids(environment):
    root,canonical = environment
    with sqlite3.connect(canonical) as con:
        con.execute("UPDATE results SET race_date='2026-06-10',off_time='18:10'")
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        report=atr_ids.apply_index(con,'10-June-2026',{'/racecard/Wolverhampton/10-June-2026/1910':'12345'})
        assert report['ids_added_or_corrected']==1
        assert con.execute('SELECT native_id FROM jobs').fetchone()[0]=='12345'
        assert atr_ids.apply_index(con,'10-June-2026',{'/racecard/Wolverhampton/10-June-2026/1810':'98765'})['ids_added_or_corrected']==0
        assert con.execute('SELECT native_id FROM jobs').fetchone()[0]=='12345'


def test_verified_field_is_not_reassigned_by_conflicting_index(environment):
    root,_=environment
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        con.execute("UPDATE jobs SET state='data'")
        report=atr_ids.apply_index(con,DATE,{CARD:'999'})
        assert report['conflicts']==['r1']
        assert con.execute('SELECT native_id FROM jobs').fetchone()[0]=='1629208'


def test_duplicate_source_id_for_two_card_paths_is_not_accepted(environment):
    root,canonical=environment
    add_second(canonical)
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        con.execute('UPDATE jobs SET native_id=NULL')
        report=atr_ids.apply_index(con,DATE,{CARD:'123',CARD.replace('2030','2100'):'123'})
        assert set(report['conflicts'])=={'r1','r3'}
        assert con.execute('SELECT count(*) FROM jobs WHERE native_id IS NOT NULL').fetchone()[0]==0


@pytest.mark.parametrize('bad',[{},[],{'/racecard/Wolverhampton/03-November-2026/2030':'123'}])
def test_malformed_or_wrong_day_index_does_not_change_ids(environment,bad):
    root,_=environment
    rebuild.prepare(root)
    with rebuild.connect(root) as con:
        with pytest.raises(ValueError):
            atr_ids.apply_index(con,DATE,bad)
        assert con.execute('SELECT native_id FROM jobs').fetchone()[0]=='1629208'


@pytest.mark.parametrize('verified_second',[False,True])
def test_legacy_one_hour_collision_keeps_supported_or_verified_owner(verified_second):
    jobs=[{'race_id':'a','race_date':'2025-06-01','track':'York','off_time':'14:00','native_id':'900',
           'state':'pending','expected':json.dumps([{'horse_key':'Alpha'},{'horse_key':'Bravo'}])},
          {'race_id':'b','race_date':'2025-06-01','track':'York','off_time':'15:00','native_id':'900',
           'state':'data' if verified_second else 'pending','expected':json.dumps([{'horse_key':'Other'}])}]
    # Use the same normalisation as preparation/field validation.
    for job in jobs:
        job['expected']=json.dumps([{'horse_key':atr._normalise_horse_key(r['horse_key'])} for r in json.loads(job['expected'])])
    cache=pd.DataFrame({'race_date':['2025-06-01']*2,'track':['York']*2,
                        'horse_name':['Alpha','Bravo'],'atr_raceid':['900']*2})
    plan=atr_ids.plan_legacy_hints(jobs,cache)
    assert plan['changes']==({'a':None} if verified_second else {'b':None})


def test_later_challenge_does_not_hide_retained_success(tmp_path):
    store=RawPayloadStore(tmp_path/'raw.sqlite')
    url=atr.ATR_BASE+'/ajax/sectionals-razor/123/times'
    store.put(url,200,b'<html>original source</html>')
    store.put(url,200,b'<title>Client Challenge</title>')
    client=rebuild.ReplayFetcher(store)
    assert client.get(url)==(200,'<html>original source</html>') and client.reused==1


def test_dirty_export_survives_restart_without_another_network_request(environment,monkeypatch):
    root,_=environment
    rebuild.prepare(root)
    client=client_for(monkeypatch,root)
    outcome,rows=atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    with rebuild.connect(root) as con:
        job=con.execute('SELECT * FROM jobs').fetchone()
        rebuild.accept(con,job,outcome,rows)
        assert con.execute('SELECT race_date FROM dirty_days').fetchone()[0]=='2026-11-02'
    resumed=client_for(monkeypatch,root)
    rebuild.run(root,max_races=0)
    assert (root/'daily/2026-11-02.parquet').exists() and not resumed.calls
    with rebuild.connect(root) as con:
        assert con.execute('SELECT count(*) FROM dirty_days').fetchone()[0]==0


def test_empty_fragments_do_not_confirm_absence_under_an_unverified_hint(environment,monkeypatch):
    root,_=environment
    cache=pd.read_parquet(atr.ATR_CACHE_FILE)
    cache['horse_name']='Wrong field behind a hint'
    cache.to_parquet(atr.ATR_CACHE_FILE,index=False)
    rebuild.prepare(root)
    client_for(monkeypatch,root,{'/ajax/':(404,'')})
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        job=con.execute('SELECT * FROM jobs').fetchone()
        assert job['state']=='retry' and job['native_id'] is None
        assert con.execute('SELECT count(*) FROM runners').fetchone()[0]==0
