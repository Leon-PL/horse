from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src import rtv_features as rtv


def observations():
    frame = pd.DataFrame({"race_id": ["a", "b", "c", "d"], "horse_id": [1, 2, 1, 1],
        "race_date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-10", "2025-02-01"])})
    frame["available_date"] = frame.race_date+pd.Timedelta(days=2)
    for col in rtv.RUN_METRICS:
        frame[col] = [10., 20., 30., 90.]
    return frame


def cards():
    return pd.DataFrame({"horse_id": [1, 2, 1, 1, 3],
        "race_date": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-11", "2025-01-12", "2025-02-02"]),
        "race_id": ["a", "live-b", "live-c", "live-d", "e"], "horse_name": ["A", "B", "A", "A", "C"],
        "finish_position": [1, 1, 2, 1, 1], "odds": [2., 3., 5., 6., 2.]})


def test_current_race_and_publication_lag_are_excluded():
    out = rtv.add_history_features(cards(), runs=observations())
    assert np.isnan(out.loc[0, "rtv_fsp_mean3"])
    assert out.loc[1, "rtv_fsp_mean3"] == 20
    assert out.loc[2, "rtv_fsp_mean3"] == 10
    assert out.loc[3, "rtv_fsp_mean3"] == 20
    assert out.loc[3, "rtv_history_count3"] == 2
    assert out.loc[4, "rtv_history_count3"] == 0
    assert np.isnan(out.loc[4, "rtv_fsp_mean3"])


def test_future_current_labels_prices_and_row_order_cannot_change_prior_features():
    frame, runs = cards(), observations()
    baseline = rtv.add_history_features(frame, runs=runs)[list(rtv.HISTORY_FEATURES)]
    changed = runs.copy()
    changed.loc[changed.race_date.ge("2025-02-01"), list(rtv.RUN_METRICS)] = -999
    mutated = rtv.add_history_features(frame.assign(finish_position=99, odds=9999), runs=changed)
    pd.testing.assert_frame_equal(baseline, mutated[list(rtv.HISTORY_FEATURES)], check_exact=True)
    shuffled = frame.sample(frac=1, random_state=7)
    result = rtv.add_history_features(shuffled, runs=runs.sample(frac=1, random_state=4)).sort_index()
    pd.testing.assert_frame_equal(baseline, result[list(rtv.HISTORY_FEATURES)], check_exact=True)


def test_full_prefix_subset_and_append_match_without_cross_horse_state():
    frame, runs = cards(), observations()
    full = rtv.add_history_features(frame, runs=runs)
    for indexes in ([0, 1], [2], [3, 4]):
        subset = rtv.add_history_features(frame.iloc[indexes], runs=runs)
        pd.testing.assert_frame_equal(subset[list(rtv.HISTORY_FEATURES)], full.iloc[indexes][list(rtv.HISTORY_FEATURES)], check_exact=True)


def test_snapshot_requires_matching_event_and_pre_decision_timestamp():
    frame = pd.DataFrame({"race_id": ["x"], "race_date": ["2025-01-03"], "track": ["Kempton Park"],
        "off_time": ["13:00"], "horse_name": ["Test Horse"]})
    source = pd.DataFrame({"race_date": ["2025-01-03"]*3, "track_key": ["kempton"]*3,
        "off_key": ["13:00"]*3, "horse_key": ["Test Horse"]*3,
        "observed_at": ["2025-01-03T10:00:00Z", "2025-01-03T12:54:00Z", "2025-01-03T12:56:00Z"]})
    for col in rtv.PRE_FEATURES:
        source[col] = [5., 10., 999.]
    out = rtv.add_pre_race_features(frame, observations=source, as_of="2025-01-03T12:55:00Z")
    assert out.rtv_pre_pace_position.iloc[0] == 10
    early = rtv.add_pre_race_features(frame, observations=source, as_of="2025-01-03T11:00:00Z")
    assert early.rtv_pre_pace_position.iloc[0] == 5
    wrong = rtv.add_pre_race_features(frame.assign(off_time="14:00"), observations=source, as_of="2025-01-03T12:55:00Z")
    assert wrong[list(rtv.PRE_FEATURES)].isna().all().all()
    with pytest.raises(ValueError, match="timezone"):
        rtv.add_pre_race_features(frame, observations=source, as_of="2025-01-03 12:55")


def test_bad_metrics_nonfinishers_and_bad_sectionals_are_masked():
    races = pd.DataFrame({"race_id": [10]})
    runners = pd.DataFrame({"race_id": [10, 10], "runner_id": [11, 12], "is_finisher": [True, False], "is_nonrunner": [False, False]})
    links = pd.DataFrame({"rtv_race_id": [10, 10], "rtv_runner_id": [11, 12], "sl_race_id": ["sl", "sl"],
        "sl_horse_id": [1, 2], "race_date": ["2025-01-01"]*2})
    metrics = pd.DataFrame({"race_id": [10]*4, "runner_id": [11, 12, 11, 12],
        "metric": ["fsp", "fsp", "top_speed", "top_speed"], "value": [105., 20., 500., 39.]})
    result = rtv.build_run_observations(races, runners, metrics, pd.DataFrame(), pd.DataFrame(), links)
    assert result.loc[0, "fsp"] == 105
    assert np.isnan(result.loc[1, "fsp"])
    assert np.isnan(result.loc[0, "top_speed"])
    assert not any(c in result for c in ("finish_position", "odds", "rank", "timeform_rating"))


def test_complete_sectionals_and_flagged_or_missing_sectors():
    races = pd.DataFrame({"race_id": [10], "distance_furlongs": [4.]})
    runners = pd.DataFrame({"race_id": [10, 10], "runner_id": [11, 12], "is_finisher": [True, False], "is_nonrunner": [False, False]})
    links = pd.DataFrame({"rtv_race_id": [10], "rtv_runner_id": [11], "sl_race_id": ["sl"], "sl_horse_id": [1], "race_date": ["2025-01-01"]})
    sectors = pd.DataFrame({"race_id": [10]*4, "runner_id": [11]*4, "sector_number": [1,2,3,4],
        "sector_time": [10.]*4, "sector_length_yards": [220.]*4, "race_position": [2,2,1,1]})
    clean = rtv.build_run_observations(races, runners, pd.DataFrame(), sectors, pd.DataFrame(), links)
    assert clean.late_speed_ratio.iloc[0] == 1.
    assert clean.early_position.iloc[0] == 1.
    assert clean.late_gain.iloc[0] == 1.
    flags = pd.DataFrame({"race_id": [10], "runner_id": [11], "check": ["sector_glitch"]})
    masked = rtv.build_run_observations(races, runners, pd.DataFrame(), sectors, flags, links)
    assert np.isnan(masked.late_speed_ratio.iloc[0])
    incomplete = rtv.build_run_observations(races, runners, pd.DataFrame(), sectors.iloc[:3], pd.DataFrame(), links)
    assert np.isnan(incomplete.late_speed_ratio.iloc[0])


def test_external_fingerprint_uses_availability_date_and_detects_historical_edits(tmp_path, monkeypatch):
    from src.feature_inputs import rtv_cache_signature
    path = tmp_path/"runs.parquet"
    monkeypatch.setattr(rtv.config, "RTV_FEATURE_RUNS", str(path), raising=False)
    monkeypatch.setattr(rtv.config, "RTV_FEATURE_PRE", str(tmp_path/"missing.parquet"), raising=False)
    runs = observations()
    runs.to_parquet(path, index=False)
    first = rtv_cache_signature("2025-01-11")
    runs.loc[runs.race_date.eq("2025-01-10"), "fsp"] = 999.
    runs.to_parquet(path, index=False)
    assert rtv_cache_signature("2025-01-11") == first
    runs.loc[runs.race_date.eq("2025-01-01"), "fsp"] = 999.
    runs.to_parquet(path, index=False)
    assert rtv_cache_signature("2025-01-11") != first
