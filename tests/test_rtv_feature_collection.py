from types import SimpleNamespace
import numpy as np
import pandas as pd

from src import rtv_feature_collection as collection


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
        "card_timeform_rating": [80,82], "card_timeform_star_rating": [3,4]}),
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
