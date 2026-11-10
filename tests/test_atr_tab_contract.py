"""Completion means all known performance tabs, with replay and upgrade safety."""
import json
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.ingestion import attheraces as atr, atr_rebuild as rebuild
from src.ingestion import atr_tabs as tabs
from src.ingestion.storage import RawPayloadStore
from test_attheraces_scraper import _FakeFetcher
from test_atr_rich_collection import page
import test_atr_rebuild as rebuild_tests

environment = rebuild_tests.environment
client_for = rebuild_tests.client_for

FIXTURES = Path(__file__).parent / 'fixtures/atr_current'


def test_real_distance_units_native_ids_and_rounded_source_values():
    rows, context = tabs.parse_distance(page('distanceran'), '1629208')
    assert len(rows) == 10
    assert context['atr_race_median_distance_ran_yards'] == 1339
    assert context['atr_distance_par_updated_date'] == '2026-11-03'
    assert context['atr_distance_source_status'] == 'beta'
    assert rows[0]['atr_distance_ran_yards'] == 1337
    assert rows[0]['atr_distance_diff_vs_median_lengths'] == -.63
    # The published lengths retain more precision than the rounded yards.
    assert rows[2]['atr_distance_ran_yards'] == 1339
    assert rows[2]['atr_distance_diff_vs_median_lengths'] == .04
    assert rows[3]['atr_distance_diff_vs_median_lengths'] == 0
    with pytest.raises(ValueError,match='identity'):
        tabs.parse_distance(page('distanceran'), '999')


@pytest.mark.parametrize('display,expected',[('1m 2f 17y',2217),('6f 17y',1337),('—',None)])
def test_distance_units(display,expected):
    assert tabs.distance_yards(display) == expected


def test_distance_unit_drift_is_not_silently_converted():
    with pytest.raises(ValueError):
        tabs.distance_yards('1200m metres')


def test_real_unavailable_distance_template_is_missing_not_zero():
    rows, context = tabs.parse_distance(page('unavailable/distanceran'),'1628888')
    assert len(rows)==4 and context['atr_race_median_distance_ran_yards'] is None
    assert all(r['atr_distance_ran_yards'] is None and r['atr_distance_diff_vs_median_lengths'] is None for r in rows)
    assert tabs.signal_runner_count('distanceran',{r['horse_name']:r for r in rows}) == 0


def test_distance_split_placeholders_keep_other_runners_measured():
    soup = BeautifulSoup(page('distanceran'), 'html.parser')
    first, second, *_ = atr._sectional_entries(soup)
    for cell in first.select('.card-cell')[3:]:
        cell.decompose()
    placeholder = soup.new_tag('div', attrs={'class': 'card-cell card-cell--sectional-time'})
    placeholder.string = '-'
    first.append(placeholder)
    second.select_one('.card-cell--distancerandistran').clear()
    second.select_one('.card-cell--distanceranvmedian').string = 'No Change'
    rows, _ = tabs.parse_distance(str(soup), '1629208')
    assert len(rows) == 10 and rows[2]['atr_distance_ran_yards'] == 1339
    assert all(row['atr_distance_ran_yards'] is None and row['atr_distance_diff_vs_median_lengths'] is None
               for row in rows[:2])
    placeholder.string = '12.34'
    with pytest.raises(ValueError, match='Missing labelled distance cells'):
        tabs.parse_distance(str(soup), '1629208')


def test_identity_only_empty_templates_do_not_claim_performance_data():
    client = _FakeFetcher({f'/{kind}':(200,(FIXTURES/'unavailable'/f'{kind}.html').read_text(encoding='utf-8'))
                           for kind in tabs.PERFORMANCE_TABS})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-04','Wexford','15:10',client,raceid='1628888')
    assert outcome == atr.NOT_FOUND and rows is None
    assert len(client.tab_receipts)==5
    assert all(r['state']=='empty' and r['signal_runner_count']==0 for r in client.tab_receipts)


def test_main_scraper_fetches_distance_and_jumps_with_provenance():
    client = _FakeFetcher({f'/{kind}':(200,page(kind)) for kind in ('times','tools','stride','distanceran')})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    assert outcome == atr.DATA and len(client.calls) == 5
    winner = next(row for row in rows if row['horse_name']=='Macarone')
    assert winner['atr_distance_ran_yards'] == 1337
    assert winner['atr_distance_diff_vs_median_lengths'] == -.63
    assert winner['atr_collection_version'] == str(tabs.COLLECTION_VERSION)
    assert {r['kind'] for r in winner['atr_tab_manifest']} == set(tabs.PERFORMANCE_TABS)


def test_main_scraper_captures_jumps_without_sectionals_and_preserves_missingness():
    client = _FakeFetcher({'/jumpsdata':(200,page('jumpsdata'))})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-04','Chepstow','12:20',client,raceid='1629661')
    assert outcome == atr.DATA and len(rows)==3
    assert rows[0]['atr_jump_fluency_grade'] == 'C'
    assert rows[0]['atr_jump_speed_lost_avg_pct'] == 13.6
    assert rows[0]['atr_jumps_par_updated_date'] == '2026-11-05'
    assert rows[0]['atr_source_race_date'] == '2026-11-04'
    assert 'atr_split_times' not in rows[0]


def test_blocked_new_tab_cannot_make_a_three_tab_field_complete():
    client = _FakeFetcher({**{f'/{k}':(200,page(k)) for k in ('times','tools','stride')},'/jumpsdata':(429,'')})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    assert outcome == atr.BLOCKED and rows is None
    assert client.tab_receipts[-1]['kind']=='jumpsdata'
    assert client.tab_receipts[-1]['state']=='blocked'


def test_real_speed_measurement_survives_missing_splits_and_curves():
    tools = BeautifulSoup(page('tools'),'html.parser')
    for chart in tools.select('.chart-energy-distribution'):
        chart.decompose()
    for pill in tools.select('.text-pill'):
        if ' to ' in pill.get('title',''):
            pill.decompose()
    client = _FakeFetcher({'/tools':(200,str(tools))})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    assert outcome==atr.DATA
    assert rows[0]['atr_top_speed_mph']==41.08
    assert not rows[0].get('atr_sectional_speeds') and not rows[0].get('atr_energy_dist')
    assert client.tab_receipts[1]['signal_runner_count']==1


def test_cross_tab_duplicate_native_identity_is_rejected():
    distance = BeautifulSoup(page('distanceran'),'html.parser')
    distance.select('.card-body')[1].select_one('a.horse__link').string = 'Another name'
    client = _FakeFetcher({**{f'/{k}':(200,page(k)) for k in ('times','tools','stride')},
                           '/distanceran':(200,str(distance))})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    assert outcome == atr.UNRESOLVED and rows is None
    assert client.last_response['reason']=='duplicate_native_horse_identity'


def test_raw_replay_fetches_only_two_missing_tabs(tmp_path):
    store = RawPayloadStore(tmp_path/'raw.sqlite')
    for kind in ('times','tools','stride'):
        store.put(f'{atr.ATR_BASE}/ajax/sectionals-razor/1629208/{kind}',200,page(kind).encode())
    network = _FakeFetcher({})
    network.last_response = {}
    replay = rebuild.ReplayFetcher(store,network)
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',replay,raceid='1629208')
    assert outcome == atr.DATA and rows
    assert replay.reused == 3
    assert [url.rsplit('/',1)[1] for url in network.calls] == ['jumpsdata','distanceran']


def test_existing_cache_is_not_completion_until_the_collection_contract_is_current():
    old = pd.DataFrame([{'horse_name':'A','atr_parser_version':'2'}])
    assert atr.current_collection_rows(old).empty
    old['atr_collection_version'] = str(tabs.COLLECTION_VERSION)
    assert len(atr.current_collection_rows(old)) == 1


def test_source_inventory_covers_all_real_result_and_pre_race_tabs():
    result = tabs.inventory((FIXTURES/'result_navigation.html').read_text())
    assert result['state']=='verified' and len(result['tabs'])==6
    assert {r['kind'] for r in result['tabs'] if r['policy']=='collect'} == {'times','tools','stride','distanceran'}
    assert result['tabs'][-1]['policy']=='exclude_future_outcomes'
    pre = tabs.inventory((FIXTURES/'pre_race_navigation.html').read_text())
    assert len(pre['tabs'])==7 and pre['state']=='verified'
    assert {r['id'] for r in pre['tabs']} == tabs.PRE_RACE_TABS
    assert all(r['policy']=='pre_race_snapshot_only' for r in pre['tabs'])


def test_unknown_tab_is_flagged_and_sidebar_links_are_ignored():
    html = (FIXTURES/'result_navigation.html').read_text()
    html = html.replace('</ul>','<li><a href="/ajax/sectionals-razor/1629208/newmetric">New Data</a></li></ul>')
    result = tabs.inventory(html+'<ul class="tabs__links"><li><a href="/outside">Unrelated</a></li></ul>')
    assert result['state']=='needs_review'
    assert [r['label'] for r in result['unknown_tabs']]==['New Data']


def test_upgrade_preserves_rows_and_reopens_three_tab_work(environment):
    root, _ = environment
    rebuild.prepare(root)
    client = _FakeFetcher({f'/{k}':(200,page(k)) for k in ('times','tools','stride')})
    _, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    row = rows[0]
    row.pop('atr_collection_version')
    row.pop('atr_tab_manifest')
    row['atr_source_payloads'] = [{k:r[k] for k in ('kind','url','status','sha256')}
                                  for r in row['atr_source_payloads'][:3]]
    payload = json.dumps(row)
    with rebuild.connect(root) as con:
        con.execute('INSERT INTO runners VALUES (?,?,?)',('r1','macarone',payload))
        con.execute("UPDATE jobs SET state='data',native_id=NULL")
        rebuild.set_control(con,'collection_version',2)
    receipt = rebuild.upgrade_tab_coverage(root)
    assert receipt['saved_rows_preserved']==1 and receipt['reopened_races']==1
    with rebuild.connect(root) as con:
        assert con.execute('SELECT payload FROM runners').fetchone()[0] == payload
        assert tuple(con.execute('SELECT state,native_id FROM jobs').fetchone()) == ('pending','1629208')
        assert dict(con.execute('SELECT kind,state FROM tab_coverage')) == {
            'times':'legacy_received','tools':'legacy_received','stride':'legacy_received',
            'jumpsdata':'pending','distanceran':'pending'}
    assert rebuild.upgrade_tab_coverage(root)['reopened_races']==0


def test_durable_ledger_records_new_tab_blocks_and_unattempted_work(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    client_for(monkeypatch,root,{**{f'/{k}':(200,page(k)) for k in ('times','tools','stride')},'/jumpsdata':(429,'')})
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        states = dict(con.execute('SELECT kind,state FROM tab_coverage'))
        assert states == {'times':'data','tools':'data','stride':'data','jumpsdata':'blocked','distanceran':'pending'}
        assert con.execute('SELECT state FROM jobs').fetchone()[0]=='retry'


def test_missing_runners_are_reported_per_tab(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    client_for(monkeypatch,root)
    rebuild.run(root,max_races=1)
    with rebuild.connect(root) as con:
        receipt = json.loads(con.execute("SELECT receipt FROM tab_coverage WHERE kind='times'").fetchone()[0])
        assert receipt['expected_runners']==1 and receipt['missing_canonical_runners']==[]


def test_absent_inventory_page_does_not_block_other_discovery(environment,monkeypatch):
    root, _ = environment
    rebuild.prepare(root)
    network = client_for(monkeypatch,root)
    with rebuild.connect(root) as con:
        job = con.execute('SELECT * FROM jobs').fetchone()
        rebuild.audit_navigation(con,network,job,root)
        assert rebuild.control(con,'tab_inventory')['state']=='unresolved'
        assert rebuild.control(con,'discovery_retry',0)==0
        assert rebuild.control(con,'global_retry',0)==0
