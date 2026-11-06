"""Real artifact path: canonical history, read-only sources and both families."""
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

import config
from scripts.derive_atr_rich_features import derive
from test_atr_new_tabs_features import record


@pytest.fixture
def snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DATA_DIR',str(tmp_path))
    root=tmp_path/'atr_cache/rebuild_v2'
    root.mkdir(parents=True)
    rows=[('1','h1','2026-01-01','14:00',1),('2','h1','2026-01-05','14:00',0),
          ('3','h1','2026-01-09','14:00',1),('4','h1','2026-01-15','14:00',1)]
    with sqlite3.connect(tmp_path/'races.db') as con:
        con.execute('CREATE TABLE results(race_id,horse_id,race_date,off_time,finish_position,race_type,track,distance_furlongs)')
        con.executemany('INSERT INTO results VALUES (?,?,?,?,?,?,?,?)',[(*r,'Chase','Chepstow',20.) for r in rows])
    with sqlite3.connect(root/'rebuild.sqlite') as con:
        con.execute('CREATE TABLE jobs(race_id,expected)')
        con.execute('CREATE TABLE runners(race_id,horse_key,payload)')
        for race,horse,day,off,finish in (rows[0],rows[2]):
            payload=record(day,race,horse)
            payload['atr_field_coverage']='complete'
            payload['atr_finishing_speed_pct']=105.
            payload['atr_race_finishing_speed_pct']=100.
            con.execute('INSERT INTO jobs VALUES (?,?)',(race,json.dumps([{'horse_key':'horse','horse_id':horse}])))
            con.execute('INSERT INTO runners VALUES (?,?,?)',(race,'horse',json.dumps(payload)))
    (root/'status.json').write_text(json.dumps({'state':'running','saved_rows':2}))
    return tmp_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_tab_export_includes_untracked_target_races_and_never_writes_sources(snapshot):
    source=snapshot/'atr_cache/rebuild_v2/rebuild.sqlite'
    canonical=snapshot/'races.db'
    before=(digest(source),digest(canonical))
    out=snapshot/'output'
    report=derive(out,family='new-tabs')
    runs=pd.read_parquet(out/'run_metrics.parquet')
    prior=pd.read_parquet(out/'prior_candidates.parquet')
    assert len(runs)==2 and len(prior)==4
    final=prior.loc[prior.race_id.eq('4')].iloc[0]
    assert final.atr_jump_loss_pct_median_3 == 10
    assert final.atr_jump_observed_runs_3 == 2
    assert prior.loc[prior.race_id.eq('1'),'atr_jump_observed_runs_3'].iloc[0] == 0
    assert report['unmatched_source_rows']==0 and report['collector_hashes_unchanged']
    frozen=pd.read_parquet(out/'history_inputs.parquet')
    from src.research.atr_new_tabs_features import build_prior_features,ALL_FEATURES
    assert_frame = build_prior_features(frozen)
    pd.testing.assert_frame_equal(assert_frame,prior[list(ALL_FEATURES)])
    assert len(json.loads((out/'feature_names.json').read_text())['prior_predictors'])==17
    assert (digest(source),digest(canonical))==before


def test_original_rich_family_stays_compatible(snapshot):
    report=derive(snapshot/'rich',family='rich')
    prior=pd.read_parquet(snapshot/'rich/prior_candidates.parquet')
    assert len(prior)==2
    assert 'atr_rich_finish_vs_race_pp_median_3' in prior
    assert report['matched_canonical_runners']==2


def test_unknown_family_is_not_silently_used(snapshot):
    with pytest.raises(ValueError,match='Unknown ATR feature family'):
        derive(snapshot/'bad',family='other')
