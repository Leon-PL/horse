"""Source IDs survive namesakes, cache joins, schema changes and repairs."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import database, identity_schema
from src.runner_identity import join_runner_columns, runner_keys, runner_positions


def twins():
    return pd.DataFrame({"race_id": ["123", "123"], "race_date": ["2025-01-01"]*2,
        "horse_id": ["1", "2"], "horse_name": ["Sierra Nevada"]*2, "num_runners": [2,2],
        "finish_position": [1,2], "won": [1,0], "lengths_behind": [0.,1.], "finish_pos_label": ["",""]})


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(database,"DB_PATH",str(tmp_path/"races.db"))
    monkeypatch.setattr(database.config,"RAW_DATA_DIR",str(tmp_path))
    return tmp_path


def test_namesakes_match_by_source_id_without_expanding_or_reordering_rows():
    left = twins().iloc[::-1].copy()
    left.index = [7,7]
    right = twins()
    right["horse_id"] = [1.,2.]
    right["rating"] = [10.,20.]
    joined = join_runner_columns(left,right,["rating"])
    assert joined.rating.tolist() == [20.,10.]
    assert joined.index.tolist() == [7,7] and len(joined) == 2


def test_known_different_ids_never_match_by_name():
    left,right = twins().iloc[:1].copy(),twins().iloc[:1].copy()
    right["horse_id"] = "3"
    with pytest.raises(ValueError,match="identity match"):
        runner_positions(left,right)
    assert runner_positions(left,right,require_all=False).tolist() == [-1]


def test_missing_ids_can_only_fall_back_to_unambiguous_names():
    left = twins()
    right = twins().drop(columns="horse_id")
    with pytest.raises(ValueError,match="Duplicate runner identities"):
        runner_positions(left,right)
    left = left.iloc[:1]
    right = right.iloc[:1]
    assert runner_positions(left,right).tolist() == [0]


def test_empty_right_side_with_optional_matches_preserves_population():
    right = twins().iloc[:0].assign(rating=pd.Series(dtype=float))
    joined = join_runner_columns(twins(),right,["rating"],require_all=False)
    assert len(joined) == 2 and joined.rating.isna().all()


def test_skipping_duplicate_ids_keeps_original_name_ambiguity():
    right = pd.concat([twins().iloc[:1]]*2+[twins().iloc[1:]], ignore_index=True)
    left = twins().iloc[:1].drop(columns="horse_id")
    assert runner_positions(left, right, require_all=False, skip_ambiguous=True).tolist() == [-1]


def test_complete_fields_allow_namesakes_but_reject_repeated_source_ids():
    frame=twins()
    database._validate_race_results(frame,"2025-01-01","123")
    frame["horse_id"]=["1",1.]
    with pytest.raises(ValueError,match="duplicate source horse"):
        database._validate_race_results(frame,"2025-01-01","123")


def test_field_writer_retains_both_ids_handles_renames_and_deletes_only_obsolete_ids(isolated):
    frame=twins()
    assert database.ingest_results(frame) == 2
    frame.loc[0,"horse_name"]="Renamed Horse"
    assert database.ingest_results(frame) == 0
    loaded=database.load_from_database().set_index("horse_id")
    assert len(loaded)==2 and loaded.loc["1","horse_name"]=="Renamed Horse"
    reduced=frame.iloc[:1].assign(num_runners=1)
    database.ingest_results(reduced)
    assert database.load_from_database().horse_id.tolist()==["1"]


def test_identical_text_and_context_for_namesakes_remain_separate(isolated):
    frame=twins().assign(sl_commentary="Same comment",sl_verdict="Same verdict",reported_jockey_claim_lbs=0)
    database.ingest_results(frame)
    database.ingest_results(frame)
    assert len(database.load_runner_text())==2
    # Context content includes source IDs; latest-observation selection must too.
    assert len(database.load_result_context())==2


def legacy_schema():
    return {"_CREATE_TABLE":database._CREATE_TABLE.rsplit(");",1)[0].rstrip()+",UNIQUE(race_id,horse_name));",
            "_CREATE_RUNNER_TEXT":database._CREATE_RUNNER_TEXT.rsplit(");",1)[0].rstrip()+
                ",PRIMARY KEY(race_id,horse_name,phase,content_sha256));"}


def create_legacy(isolated):
    sql=legacy_schema()
    with sqlite3.connect(database.DB_PATH) as con:
        con.execute(sql["_CREATE_TABLE"])
        con.execute(sql["_CREATE_RUNNER_TEXT"])
        con.execute("INSERT INTO results(race_id,race_date,horse_name,horse_id,num_runners,finish_position,won) "
                    "VALUES ('123','2025-01-01','Sierra Nevada','1',2,1,1)")
    database.init_db()


def test_legacy_migration_preserves_rows_rowids_revisions_and_triggers(isolated):
    create_legacy(isolated)
    with sqlite3.connect(database.DB_PATH) as con:
        before=con.execute("SELECT rowid,* FROM results").fetchall()
        revision=con.execute("SELECT revision FROM result_source_state").fetchone()[0]
    assert database.migrate_runner_identity()["results"]
    assert not database.migrate_runner_identity()["results"]
    with sqlite3.connect(database.DB_PATH) as con:
        assert before==con.execute("SELECT rowid,* FROM results").fetchall()
        assert revision==con.execute("SELECT revision FROM result_source_state").fetchone()[0]
        plan=con.execute("EXPLAIN QUERY PLAN SELECT * FROM results WHERE race_id=?",("123",)).fetchall()
        assert any("results_race_lookup" in row[3] for row in plan)
    database.ingest_results(twins())
    with sqlite3.connect(database.DB_PATH) as con:
        assert con.execute("SELECT COUNT(*) FROM results").fetchone()[0]==2
        assert con.execute("SELECT revision FROM result_source_state").fetchone()[0]>revision


def test_failed_migration_rolls_back_original_table_and_values(isolated,monkeypatch):
    create_legacy(isolated)
    original=database.rebuild_without_key
    def interrupted(con,table,columns):
        original(con,table,columns)
        raise RuntimeError("interrupted migration")
    monkeypatch.setattr(database,"rebuild_without_key",interrupted)
    with pytest.raises(RuntimeError,match="interrupted"):
        database.migrate_runner_identity()
    with sqlite3.connect(database.DB_PATH) as con:
        assert con.execute("SELECT horse_name FROM results").fetchall()==[("Sierra Nevada",)]
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO results(race_id,race_date,horse_name,horse_id) VALUES ('123','2025-01-01','Sierra Nevada','2')")


def test_result_rowid_rejects_an_ambiguous_namesake(isolated):
    database.ingest_results(twins())
    with sqlite3.connect(database.DB_PATH) as con:
        with pytest.raises(ValueError,match="ambiguous"):
            identity_schema.result_rowid(con,{"race_id":"123","horse_name":"Sierra Nevada"})


def test_keys_keep_source_ids_separate_from_name_fallback():
    frame=twins()
    assert runner_keys(frame).tolist()==["sl:1","sl:2"]


def test_legacy_name_only_write_retains_known_ids_and_rejects_ambiguous_names(isolated):
    frame=twins().iloc[:1].assign(num_runners=1)
    database.ingest_results(frame)
    database.ingest_results(frame.drop(columns="horse_id"))
    assert database.load_from_database().horse_id.tolist()==["1"]
    database.ingest_results(twins())
    with pytest.raises(ValueError,match="Ambiguous legacy"):
        database.ingest_results(frame.drop(columns="horse_id"))


def test_legacy_append_stays_idempotent_for_duplicate_input_rows(isolated):
    legacy=twins().iloc[:1].drop(columns="horse_id")
    assert database.insert_results(pd.concat([legacy,legacy],ignore_index=True))==1
    assert database.insert_results(pd.concat([legacy,legacy],ignore_index=True))==0


@pytest.mark.parametrize("rows",[1,2])
def test_atr_does_not_identify_distinct_horses_by_a_shared_display_name(publish_dataset,rows):
    from src.atr_features import _published_history, published_atr_rolling
    from src.source_contracts import TableContract
    from src.source_dataset import read_table
    first=twins().assign(track="Kempton",off_time="12:00")
    later=first.assign(race_id="124",race_date="2025-01-10")
    sectionals=first.iloc[:rows].drop(columns=["race_id","horse_id"]).assign(
        atr_raceid="atr-race",atr_horse_id=["atr-1","atr-2"][:rows],atr_finishing_speed_pct=[100.,101.][:rows])
    _published_history.cache_clear()
    publish_dataset(pd.concat([first,later],ignore_index=True),
                    atr={"sectionals":(sectionals,TableContract("runner",(),delay_hours=48))})
    linked=read_table("atr","sectionals",linked=True)
    assert linked.match_status.eq("ambiguous").all() and linked.sl_horse_id.isna().all()
    rolled=published_atr_rolling(later)
    assert np.isnan(rolled["atr_finishing_speed_pct_avg_3"]).all()
    assert (rolled["atr_observation_count_5"]==0).all()


def test_rtv_name_snapshot_does_not_fill_both_namesakes(publish_dataset):
    from src import rtv_features as rtv
    from src.source_contracts import TableContract
    frame=twins().assign(track="Kempton",off_time="12:00")
    observation={"race_date":"2025-01-01","track":"Kempton","off_time":"12:00","horse_name":"Sierra Nevada",
                 "observed_at":"2025-01-01T11:50:00Z",**{name:1. for name in rtv.SNAPSHOT_FEATURES}}
    contract=TableContract("runner snapshot",(),role="snapshot")
    publish_dataset(frame,rtv_snapshots={"pre":(pd.DataFrame([observation]),contract)})
    result=rtv.add_pre_race_features(frame,as_of="2025-01-01T11:55:00Z")
    assert result[list(rtv.SNAPSHOT_FEATURES)].isna().all().all()


def test_feature_store_build_append_and_live_keep_namesakes_separate(tmp_path, monkeypatch):
    import config
    from src.data_processor import process_data
    from src.feature_store import FeatureStore
    from src.featured_dataset import _stamp, data_signature, fe_signature
    from src.store_refresh import append_current_store
    from tests.test_live_parity import _synthetic_raw

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    raw = _synthetic_raw(n_days=8, races_per_day=1, field=2)
    raw["horse_id"] = [1, 2]*8
    raw["horse_name"] = "Sierra Nevada"
    raw["finish_position"] = [1, 2]*8
    raw["won"] = [1, 0]*8
    proc = process_data(raw)
    dates = sorted(proc.race_date.unique())
    store = FeatureStore(str(tmp_path/"store"))
    history = proc[proc.race_date < dates[-2]]
    built = store.build(history)
    _stamp(store, fe_signature(), data_signature(history, cutoff=history.race_date.max()))
    # Append the next day the way training does.
    assert append_current_store(proc[proc.race_date <= dates[-2]], store=store)
    featured = store.load_featured()
    appended = featured[pd.to_datetime(featured.race_date) == dates[-2]]
    assert len(built) == 12 and len(appended) == 2
    assert len(featured) == 14
    card = proc[proc.race_date == dates[-1]]
    live = store.featurize_live(card)
    reversed_live = store.featurize_live(card.iloc[::-1])
    for frame in (appended, live, reversed_live):
        by_id = frame.set_index("horse_id")
        assert len(frame) == 2 and by_id.index.is_unique
        assert by_id.loc[1, "horse_elo"] > by_id.loc[2, "horse_elo"]
    pd.testing.assert_series_equal(live.set_index("horse_id").horse_elo.sort_index(),
                                   reversed_live.set_index("horse_id").horse_elo.sort_index())
