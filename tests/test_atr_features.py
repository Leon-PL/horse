"""Tests for ATR sectional feature derivation + leakage-safe rolling.

Covers the pure per-run helpers in ``src.atr_features`` and the rolling
wiring in ``src.feature_engineer.add_atr_features`` — in particular the
contract that a horse's feature at race N uses only races < N.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import atr_features as af
from src import feature_engineer as fe


# ── Pure helpers ───────────────────────────────────────────────────────

class TestCommentTagger:
    def test_trouble_and_style(self):
        t = af.tag_comment("held up in rear, short of room over 1f out, kept on well")
        assert t["atr_cmt_held_up"] == 1
        assert t["atr_cmt_trouble"] == 1
        assert t["atr_cmt_stayed_on"] == 1
        assert t["atr_cmt_front"] == 0

    def test_eased_and_front(self):
        t = af.tag_comment("led, ridden clear, eased close home")
        assert t["atr_cmt_front"] == 1
        assert t["atr_cmt_eased"] == 1

    def test_nh_error(self):
        assert af.tag_comment("prominent, blundered 3 out, weakened")["atr_cmt_nh_error"] == 1

    def test_empty(self):
        t = af.tag_comment(None)
        assert set(t.values()) == {0}


class TestRunScalars:
    def _row(self, **kw):
        base = dict(
            atr_sectional_speeds=json.dumps([34.0, 39.0, 37.0]),
            atr_energy_dist=json.dumps([90.0, 105.0, 100.0]),
            atr_optimum_energy=json.dumps([92.0, 104.0, 101.0]),
            atr_finishing_speed_pct=101.0,
            atr_efficiency_score=17.0,
            atr_peak_stride_length=26.0,
            atr_peak_stride_freq=2.3,
            atr_official_position=5,
            atr_adj_placing=2.0,
            atr_in_running_comment="held up, ran on well",
            won=0,
            lengths_behind=3.0,
        )
        base.update(kw)
        return base

    def test_velocity_scalars(self):
        out = af.compute_atr_run_features(pd.DataFrame([self._row()]))
        r = out.iloc[0]
        assert r["atr_early_pace"] == 34.0
        assert r["atr_max_speed"] == 39.0
        assert r["atr_closing_speed"] == 37.0
        assert r["atr_turn_of_foot"] == pytest.approx(5.0)   # 34->39
        assert r["atr_energy_eff"] < 0   # -mean|e-o|

    def test_upgrade_clean_counts_when_involved(self):
        # finished 5th, sectionals say 2nd, ran on, only 3L back -> unlucky upgrade
        out = af.compute_atr_run_features(pd.DataFrame([self._row()]))
        assert out.iloc[0]["atr_upgrade_clean"] == pytest.approx(3.0)

    def test_upgrade_discarded_when_eased_and_beaten(self):
        # The trap: tailed-off & eased horse posts a fast finish -> NOT an upgrade
        row = self._row(
            atr_official_position=20, atr_adj_placing=1.0, won=0,
            lengths_behind=40.0, atr_in_running_comment="tailed off, eased",
        )
        assert af.compute_atr_run_features(pd.DataFrame([row])).iloc[0]["atr_upgrade_clean"] == 0.0

    def test_closing_in_hand_gated(self):
        won = self._row(won=1, atr_in_running_comment="led, ridden out, comfortably")
        beaten_eased = self._row(won=0, atr_in_running_comment="eased when held")
        a = af.compute_atr_run_features(pd.DataFrame([won])).iloc[0]["atr_closing_in_hand"]
        b = af.compute_atr_run_features(pd.DataFrame([beaten_eased])).iloc[0]["atr_closing_in_hand"]
        assert a == 101.0          # genuine winning finish -> credited
        assert np.isnan(b)         # eased when beaten -> not credited

    def test_implausible_values_nand(self):
        row = self._row(atr_finishing_speed_pct=9999.0)
        assert np.isnan(af.compute_atr_run_features(pd.DataFrame([row])).iloc[0]["atr_finishing_speed_pct"])

    @pytest.mark.parametrize("missing", [None, np.nan, "", "   "])
    def test_no_comment_leaves_every_tag_unknown(self, missing):
        row = self._row(atr_in_running_comment=missing)
        out = af.compute_atr_run_features(pd.DataFrame([row])).iloc[0]
        assert out[af.COMMENT_TAGS].isna().all()

    def test_comment_without_the_phrase_is_a_real_zero(self):
        row = self._row(atr_in_running_comment="prominent throughout")
        out = af.compute_atr_run_features(pd.DataFrame([row])).iloc[0]
        assert out["atr_cmt_trouble"] == 0.0

    def test_mixed_rows_only_mask_the_rows_with_no_comment(self):
        frame = pd.DataFrame([self._row(atr_in_running_comment="led"), self._row(atr_in_running_comment=None)])
        tags = af.compute_atr_run_features(frame)["atr_cmt_front"]
        assert tags.iloc[0] == 1.0
        assert np.isnan(tags.iloc[1])

    def test_no_comment_raises_no_gating_flag(self):
        # With no comment there is no evidence of easing or of a genuine finish.
        beaten = af.compute_atr_run_features(pd.DataFrame([self._row(atr_in_running_comment=None)])).iloc[0]
        winner = af.compute_atr_run_features(
            pd.DataFrame([self._row(atr_in_running_comment=None, won=1)])).iloc[0]
        assert beaten["atr_upgrade_clean"] == pytest.approx(3.0)
        assert np.isnan(beaten["atr_closing_in_hand"])
        assert winner["atr_closing_in_hand"] == 101.0


# ── Rolling / leakage wiring ───────────────────────────────────────────

def _make_history(monkeypatch, publish_dataset, edit=None):
    """3 prior runs + 1 upcoming for one horse, with ATR published for the runs.

    *edit* may change the ATR rows before they are published.
    """
    from src.atr_features import _published_history
    from src.source_contracts import TableContract

    dates = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]
    canonical = pd.DataFrame(dict(
        race_id=["r0", "r1", "r2", "r9"], race_date=dates, track="Kempton", off_time="14:00",
        horse_name="Test Horse", horse_id="H1", won=[0, 0, 0, np.nan],
        lengths_behind=[2.0, 2.0, 2.0, np.nan], num_runners=8))
    rows = []
    for i, f in enumerate([100.0, 110.0, 120.0]):
        rows.append(dict(
            race_date=dates[i], track="Kempton", off_time="14:00", horse_name="Test Horse",
            atr_raceid=f"atr{i}", atr_horse_id="native1",
            atr_sectional_speeds=json.dumps([30.0 + i, 38.0, 36.0]),
            atr_energy_dist=json.dumps([90.0, 100.0]), atr_optimum_energy=json.dumps([92.0, 101.0]),
            atr_finishing_speed_pct=f, atr_efficiency_score=15.0 + i,
            atr_peak_stride_length=25.0, atr_peak_stride_freq=2.3,
            atr_official_position=3, atr_adj_placing=3.0,
            atr_in_running_comment="held up, ran on",
        ))
    sectionals = pd.DataFrame(rows)
    if edit is not None:
        edit(sectionals)
    # The upcoming race (r9) has no ATR row; its features come from history.
    _published_history.cache_clear()
    publish_dataset(canonical, atr={"sectionals": (sectionals, TableContract("runner", (), delay_hours=48))})
    return canonical


def test_rolling_is_prior_only(monkeypatch, publish_dataset):
    df = _make_history(monkeypatch, publish_dataset)
    out = fe.add_atr_features(df).sort_values("race_date").reset_index(drop=True)
    a3 = out["atr_finishing_speed_pct_avg_3"].tolist()
    # race1: no prior -> NaN; race2: mean(100)=100; race3: mean(100,110)=105;
    # upcoming(r9): mean(100,110,120)=110 (history only, never its own race)
    assert np.isnan(a3[0])
    assert a3[1] == pytest.approx(100.0)
    assert a3[2] == pytest.approx(105.0)
    assert a3[3] == pytest.approx(110.0)


def test_rates_skip_a_run_with_no_comment(monkeypatch, publish_dataset):
    def no_comment_on_r1(sectionals):
        sectionals.loc[sectionals.race_date.eq("2025-02-01"), "atr_in_running_comment"] = None
    df = _make_history(monkeypatch, publish_dataset, no_comment_on_r1)
    out = fe.add_atr_features(df).sort_values("race_date").reset_index(drop=True)
    rate = out["atr_cmt_held_up_rate_5"].tolist()
    # Both commented runs read "held up". The middle run must not dilute them.
    assert np.isnan(rate[0])
    assert rate[1:] == [1.0, 1.0, 1.0]


def test_rates_stay_missing_with_no_comment_history(monkeypatch, publish_dataset):
    def no_comments(sectionals):
        sectionals["atr_in_running_comment"] = None
    df = _make_history(monkeypatch, publish_dataset, no_comments)
    out = fe.add_atr_features(df)
    rates = [c for c in out.columns if c.startswith("atr_cmt_") and "_rate_" in c]
    assert rates
    assert out[rates].isna().all().all()
    # Sectional history is a separate fact and survives missing comments.
    assert out["atr_finishing_speed_pct_avg_3"].notna().sum() == 3


def test_raw_and_perrun_dropped(monkeypatch, publish_dataset):
    df = _make_history(monkeypatch, publish_dataset)
    # A frame that already carries this race's ATR record must not keep it.
    df = df.assign(atr_sectional_speeds="[30.0]", atr_official_position=3, atr_in_running_comment="led",
                   atr_finishing_speed_pct=100.0, atr_upgrade_clean=1.0, atr_cmt_trouble=0)
    out = fe.add_atr_features(df)
    # raw post-race + un-rolled per-run columns must not survive
    for leaked in ("atr_sectional_speeds", "atr_official_position", "atr_in_running_comment",
                   "atr_finishing_speed_pct", "atr_upgrade_clean", "atr_cmt_trouble"):
        assert leaked not in out.columns
    # rolled features + coverage flag must exist
    assert "atr_finishing_speed_pct_avg_3" in out.columns
    assert "atr_cmt_trouble_rate_5" in out.columns
    assert "has_atr_history" in out.columns


def test_upcoming_race_gets_history_features(monkeypatch, publish_dataset):
    df = _make_history(monkeypatch, publish_dataset)
    out = fe.add_atr_features(df)
    upcoming = out[out["race_id"] == "r9"].iloc[0]
    # the live/upcoming row has no own sectionals but inherits rolled history
    assert upcoming["has_atr_history"] == 1
    assert upcoming["atr_finishing_speed_pct_avg_3"] == pytest.approx(110.0)
