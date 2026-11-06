"""Real current-page shapes: preserve useful telemetry without mixing units."""
import json
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src import atr_features as features
from src.ingestion import attheraces as atr
from test_attheraces_scraper import _FakeFetcher

FIXTURES = Path(__file__).parent/'fixtures/atr_current'


def page(kind):
    return (FIXTURES/f'{kind}.html').read_text(encoding='utf-8')


def test_current_tools_separate_segment_speeds_instantaneous_speeds_and_seconds():
    row = atr._parse_tools(BeautifulSoup(page('tools'), 'html.parser'))['Macarone']
    assert row['atr_sectional_speeds'] == [35.87, 38.91, 37.11]
    assert row['atr_top_speed_mph'] == 41.08
    assert row['atr_final_speed_mph'] == 33.15
    assert row['atr_acceleration_0_30_seconds'] == 4.7
    assert row['atr_source_metrics'][-1]['unit'] == 'seconds'
    assert row['atr_sectional_speed_labels'] == ['Start to 4f', '4f to 2f', '2f to Finish']


def test_stride_length_frequency_reference_and_markers_are_all_retained():
    row = atr._parse_stride(BeautifulSoup(page('stride'), 'html.parser'))['Macarone']
    assert row['atr_stride_length'] == [19.34,25.29,24.81,24.26,22.92,22.92]
    assert row['atr_stride_frequency'] == [2.46,2.35,2.34,2.31,2.45,2.31]
    assert row['atr_stride_frequency_reference'] == [2.45]*6
    assert row['atr_stride_frequency_markers'] == ['5f','4f','3f','2f','1f','Fin.']
    assert len(row['atr_source_charts']) == 2


def test_chart_labels_rather_than_column_order_identify_frequency():
    soup = BeautifulSoup(page('stride'), 'html.parser')
    charts = soup.select('.chart-energy-distribution')
    charts[0]['data-line1-title'], charts[1]['data-line1-title'] = charts[1]['data-line1-title'], charts[0]['data-line1-title']
    charts[0]['data-data'], charts[1]['data-data'] = charts[1]['data-data'], charts[0]['data-data']
    assert atr._parse_stride(soup)['Macarone']['atr_stride_frequency'][0] == 2.46


def test_front_flags_and_missing_totals_preserve_section_alignment():
    soup = BeautifulSoup(page('times'), 'html.parser')
    soup.select('.card-sectional__inner[title]')[1].attrs.pop('title')
    row = atr._parse_times(soup)['Macarone']
    assert row['atr_running_totals'] == [15.12,None,37.56,49.35,61.12,73.6]
    assert row['atr_front_flags'] == [False,False,False,False,True,True]
    assert row['atr_section_notes'][-1] == ['Front']
    assert row['atr_final_time'] == 73.6


def test_race_context_is_retained_separately_from_runner_context():
    row = atr._parse_race_analysis(BeautifulSoup(page('tools'), 'html.parser'))
    assert row['atr_race_early_pace'] == 'Even'
    assert row['atr_race_early_pace_flag'] == 'even'
    assert row['atr_race_finishing_speed_pct'] == 98.74
    assert row['atr_first_section_furlongs'] == 1.09
    assert len(row['atr_race_pace_profile']) == 6


def test_native_id_path_avoids_html_discovery_and_retains_provenance():
    client = _FakeFetcher({f'/{kind}':(200,page(kind)) for kind in ('times','tools','stride')})
    outcome, rows = atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')
    assert outcome == atr.DATA and len(rows) == 1
    row = rows[0]
    assert row['atr_parser_version'] == '2'
    assert row['atr_finishing_speed_pct'] == 99.67
    assert row['atr_race_finishing_speed_pct'] == 98.74
    assert len(row['atr_source_payloads']) == 5
    assert all(len(p['sha256']) == 64 for p in row['atr_source_payloads'])
    assert len(client.calls) == 5
    restored = atr._encode_arrays(pd.DataFrame(rows)).iloc[0]
    assert json.loads(restored.atr_stride_frequency)[0] == 2.46


def test_unexpected_success_html_is_not_permanent_no_data():
    client = _FakeFetcher({'/ajax/':(200,'<html>Changed template</html>')})
    assert atr.fetch_atr_sectionals_outcome('2026-11-02','Wolverhampton','20:30',client,raceid='1629208')[0] == atr.UNRESOLVED


def test_native_id_hint_for_another_date_or_time_is_rejected():
    client = _FakeFetcher({f'/{kind}':(200,page(kind)) for kind in ('times','tools','stride')})
    assert atr.fetch_atr_sectionals_outcome('2026-11-01','Wolverhampton','20:30',client,raceid='1629208')[0] == atr.UNRESOLVED
    assert client.last_response['reason'] == 'source_identity_mismatch'
    assert len(client.calls) == 1


def test_unknown_tool_metric_is_preserved_without_becoming_a_speed():
    soup = BeautifulSoup(page('tools').replace('Top Speed','New Metric'), 'html.parser')
    row = atr._parse_tools(soup)['Macarone']
    assert row['atr_sectional_speeds'] == [35.87,38.91,37.11]
    unknown = next(x for x in row['atr_source_metrics'] if x['label'] == 'New Metric')
    assert unknown['value'] == 41.08 and unknown['unit'] is None


def test_legacy_labelled_speed_arrays_do_not_turn_seconds_into_closing_speed():
    frame = pd.DataFrame([{'atr_sectional_speeds':json.dumps([35.87,38.91,37.11,41.08,33.15,4.7]),
        'atr_sectional_speed_labels':json.dumps(['Start to 4f','4f to 2f','2f to Finish','Top Speed','Final Speed','0-30mph']),
        'atr_finishing_speed_pct':99.67,'atr_efficiency_score':12.96,'atr_peak_stride_length':25.05,
        'atr_peak_stride_freq':2.45,'atr_official_position':1,'atr_adj_placing':1,'won':1,'lengths_behind':0}])
    row = features.compute_atr_run_features(frame).iloc[0]
    assert row.atr_closing_speed == 37.11
    assert row.atr_max_speed == 38.91
    assert row.atr_turn_of_foot == pytest.approx(3.04)


def test_retry_after_accepts_seconds_and_http_dates_without_invalid_values():
    assert atr._retry_after_seconds('120') == 120
    now = datetime(2026,11,5,17,59,tzinfo=timezone.utc).timestamp()
    assert atr._retry_after_seconds('Thu, 05 Nov 2026 18:00:00 GMT',now=now) == 60
    assert atr._retry_after_seconds('not a date') is None
    assert atr._retry_after_seconds('nan') is None
    assert atr._retry_after_seconds(None) is None
