from types import SimpleNamespace
import numpy as np
import pandas as pd

from src.ingestion import rtv_observations as collection


def test_raw_sectional_position_column_is_normalised(tmp_path):
    tables = tmp_path/"tables"
    for name in ("races", "sectionals"):
        (tables/name).mkdir(parents=True)
    pd.DataFrame({"race_id": [1], "race_date": ["2025-08-14"], "track_name": ["Kempton Park"],
        "off_time_utc": ["11:30"], "is_abandoned": [False], "is_arab_race": [False], "distance_furlongs": [4.]}).to_parquet(tables/"races/2025-Q3.parquet")
    pd.DataFrame({"race_id": [1], "runner_id": [2], "sector_number": [1], "sector_time": [12.],
        "sector_length_yards": [220.], "sector_position": [3]}).to_parquet(tables/"sectionals/2025-Q3.parquet")
    result = collection.observation_tables("2025-08-01", "2025-08-31", source_root=tmp_path)
    assert result["sectionals"].race_position.tolist() == [3]


def test_pre_values_only_forward_fill_after_their_own_capture_and_bst_is_explicit(tmp_path, monkeypatch):
    key = "2025-08-14/kempton-park/1230"
    tables = {"pre_runners": pd.DataFrame({"race_key": [key]*2, "card_horse_name": ["A"]*2,
        "fetched_at": ["2025-08-14T09:00:00Z", "2025-08-14T09:10:00Z"],
        "card_timeform_rating": [80,82], "card_timeform_star_rating": [3,4],
        "card_jockey_name": ["J. Doe", "J Smith"], "card_format_weight": ["9-7", "9-7"],
        "card_apprentice_claim": [0, 5]}),
        "pace_maps": pd.DataFrame({"race_key": [key]*2, "pm_runner_name": ["A"]*2,
        "fetched_at": ["2025-08-14T09:05:00Z", "2025-08-14T11:29:00Z"],
        "pm_pace_map__position": [.3, 999.], "pm_draw__lengths_advantage": [.5, 999.]})}
    monkeypatch.setattr(collection.rtv, "RawStore", lambda *a: SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(collection.rtv, "build_snapshot_tables", lambda **kw:tables)
    path = tmp_path/"pre.parquet"
    collection.refresh_pre_observations(destination=path)
    actual = pd.read_parquet(path).sort_values("observed_at")
    assert actual.off_key.eq("11:30").all()
    assert actual.track_key.eq("kempton").all()
    assert len(actual) == 3
    assert actual.rtv_pre_timeform_rating.tolist() == [80.,80.,82.]
    assert np.isnan(actual.rtv_pre_pace_position.iloc[0])
    assert actual.rtv_pre_pace_position.iloc[1:].eq(.3).all()
    np.testing.assert_allclose(actual.rider_claim_lbs, [0, np.nan, 5], equal_nan=True)
    assert actual.rider_name.dropna().tolist() == ["j doe", "j smith"]


def test_snapshot_card_ratings_keep_their_timeform_symbols(tmp_path, monkeypatch):
    key = "2025-08-14/kempton-park/1230"
    tables = {"pre_runners": pd.DataFrame({"race_key": [key]*2, "card_horse_name": ["A", "B"],
        "fetched_at": ["2025-08-14T09:00:00Z"]*2, "card_timeform_rating": ["111p", "-"],
        "card_timeform_star_rating": [4, 1], "card_timeform_tipped_place": [1, None], "card_focus_horse": [True, False]}),
        "pace_maps": pd.DataFrame()}
    monkeypatch.setattr(collection.rtv, "RawStore", lambda *a: SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(collection.rtv, "build_snapshot_tables", lambda **kw:tables)
    path = tmp_path/"pre.parquet"
    collection.refresh_pre_observations(destination=path)
    actual = pd.read_parquet(path).sort_values("horse_key").reset_index(drop=True)
    assert actual.rtv_pre_timeform_rating.iloc[0] == 111
    assert actual.rtv_card_tf_improver.tolist() == [1., 0.]
    assert actual.rtv_card_tf_tipped.tolist() == [3., 0.]
    assert np.isnan(actual.rtv_card_tf_rating.iloc[1]) and actual.rtv_card_tf_stars.iloc[1] == 1



def test_historical_rider_claims_require_agreement_and_the_same_runner(tmp_path):
    tables = tmp_path / "tables"
    (tables / "races").mkdir(parents=True)
    (tables / "runners").mkdir()
    pd.DataFrame({"race_id": [101], "race_date": ["2025-08-14"], "track_name": ["Kempton Park"],
                  "off_time_utc": ["11:30"], "is_abandoned": [False], "is_arab_race": [False]}
                 ).to_parquet(tables / "races/2025-Q3.parquet")
    horses = list("ABCDEFG")
    pd.DataFrame({
        "race_id": 101, "runner_id": range(1, 8), "horse_name": horses,
        "in_results": True, "in_racecard": True, "is_nonrunner": False,
        "res_jockey_name": ["J. Doe", "J. Doe", "Another Rider", "J. Doe", "J. Doe", "J. Doe", "J. Doe"],
        "card_jockey_name": "J Doe", "res_jockey__id": 10, "card_jockey_id": [10, 10, 10, 99, 10, 10, 10],
        "res_apprentice_claim": [0, 5, 3, 5, 5, 5, None], "card_apprentice_claim": [0, 5, 3, 5, 3, 5, 0],
        "weight_lbs": [140, 135, 140, 140, 140, 142, 140],
    }).to_parquet(tables / "runners/2025-Q3.parquet")
    canonical = pd.DataFrame({
        "race_id": 200, "horse_id": range(11, 18), "race_date": "2025-08-14", "track": "Kempton",
        "off_time": "11:30", "horse_name": horses, "jockey": "J Doe", "weight_lbs": 140,
    })
    result = collection.historical_rider_claims(canonical, source_root=tmp_path)
    assert result.horse_id.tolist() == ["11", "12"]
    assert result.rider_claim_lbs.tolist() == [0, 5]
    assert result.rider_weight_lbs.tolist() == [140, 140]
