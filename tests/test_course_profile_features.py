"""Course-profile features (cp_*): prior-only, same-day safe, append-exact."""

import numpy as np
import pandas as pd
import pytest

import config
from src import course_reports as cr
from src import feature_engineer as fe
from src.data_processor import process_data
from tests.test_feature_store import _assert_append_matches_batch
from tests.test_live_parity import _synthetic_raw

_STALLS = ["6f - Stands side; remainder - Inside", "All: Far side",
           "Straight - Centre, Round - Outside", "7f & 8f - Inside; remainder - Centre"]


@pytest.fixture
def course_reports(tmp_path, monkeypatch):
    """BHA files covering the synthetic fixtures, with varied stalls and rail."""
    raw = _synthetic_raw(n_days=90, races_per_day=2, field=6, seed=23)
    days = raw[["race_date", "track"]].drop_duplicates().reset_index(drop=True)
    going = days.rename(columns={"track": "course"}).assign(
        fixture_type="FLAT", abandoned=0,
        stalls=[_STALLS[i % len(_STALLS)] for i in range(len(days))])
    races = raw[["race_date", "off_time", "track"]].drop_duplicates().reset_index(drop=True)
    races = races.rename(columns={"track": "course", "off_time": "race_time"}).assign(
        abandoned=0, distance_change_yards=[float((i * 13) % 40) for i in range(len(races))])
    going.to_parquet(tmp_path / "going.parquet", index=False)
    races.to_parquet(tmp_path / "races.parquet", index=False)
    monkeypatch.setattr(cr, "GOING_PATH", str(tmp_path / "going.parquet"))
    monkeypatch.setattr(cr, "RACES_PATH", str(tmp_path / "races.parquet"))
    return raw


def _cp(proc: pd.DataFrame) -> pd.DataFrame:
    df = proc.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["_event_dt"] = fe._event_sort_key(df)
    df = df.sort_values(["_event_dt", "race_id"]).copy()
    df = fe._add_result_history_flags(df)
    df = fe.add_track_config_features(df)
    df = fe.add_course_profile_features(df)
    return df.set_index(["race_id", "horse_name"]).sort_index()


def _cp_cols(df):
    return [c for c in df.columns if c.startswith("cp_")]


def test_reports_reach_the_features(course_reports):
    out = _cp(process_data(course_reports.copy(), save=False))
    flat = out[out["race_type"] == "Flat"]
    assert flat["cp_stalls_code"].notna().all()
    assert out.loc[out["race_type"] != "Flat", "cp_stalls_code"].isna().all()
    assert out["cp_rail_yards"].notna().all() and (out["cp_rail_yards"] > 0).any()
    assert flat["cp_draw_vs_field"].notna().all()
    assert len(_cp_cols(out)) >= 25


def test_a_race_never_sees_its_own_result(course_reports):
    proc = process_data(course_reports.copy(), save=False)
    # The final race only: synthetic horses can run twice a day, and an
    # earlier race that day is legitimately part of a later race's history.
    last = proc.sort_values(["race_date", "off_time"])["race_id"].iloc[-1]
    altered = proc.copy()
    idx = altered.index[altered["race_id"] == last]
    # Reverse its finishing order and its prices.
    pos = altered.loc[idx, "finish_position"].to_numpy()
    altered.loc[idx, "finish_position"] = pos.max() + 1 - pos
    altered.loc[idx, "won"] = (altered.loc[idx, "finish_position"] == 1).astype(int)
    altered.loc[idx, "odds"] = altered.loc[idx, "odds"].to_numpy()[::-1]
    a, b = _cp(proc), _cp(altered)
    cols = _cp_cols(a)
    assert (a.loc[last, "finish_position"].to_numpy() != b.loc[last, "finish_position"].to_numpy()).any()
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_course_statistics_ignore_earlier_races_the_same_day(course_reports):
    """Live picks are made before the day's results exist; training must match."""
    raw = course_reports.copy()
    # Put both of a day's races at the same course so they share course keys.
    raw["track"] = "Ascot"
    raw["race_type"] = "Flat"
    proc = process_data(raw, save=False)
    day = sorted(proc["race_date"].unique())[60]
    first, second = sorted(proc.loc[proc["race_date"] == day, "race_id"].unique())[:2]
    altered = proc.copy()
    idx = altered.index[altered["race_id"] == first]
    pos = altered.loc[idx, "finish_position"].to_numpy()
    altered.loc[idx, "finish_position"] = pos.max() + 1 - pos
    altered.loc[idx, "won"] = (altered.loc[idx, "finish_position"] == 1).astype(int)
    a, b = _cp(proc).loc[second], _cp(altered).loc[second]
    for col in ("cp_course_stamina_slope", "cp_weight_adv", "cp_age_adv"):
        np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope_a = (a["cp_draw_adv"] / a["cp_draw_vs_field"]).to_numpy(float)
        slope_b = (b["cp_draw_adv"] / b["cp_draw_vs_field"]).to_numpy(float)
    np.testing.assert_allclose(slope_a, slope_b, equal_nan=True)


def test_append_matches_full_build_with_reports(course_reports, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    proc = process_data(course_reports.copy(), save=False)
    proc["race_date"] = pd.to_datetime(proc["race_date"], errors="coerce")
    dates = np.sort(proc["race_date"].unique())
    boundary = dates[len(dates) * 2 // 3]
    hist = proc[proc["race_date"] < boundary].copy()
    new = proc[proc["race_date"] >= boundary].copy()
    _assert_append_matches_batch(tmp_path / "store", proc, hist, new, boundary)
