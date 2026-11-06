"""Efficiency direction, retained poor runs, grade semantics and repeatability."""
import numpy as np
import pandas as pd
import pytest

from src import atr_features as af
from src.research.atr_rich_features import derive_run_metrics, build_prior_features, CONSISTENCY
from test_atr_features import _make_history


def test_efficiency_keeps_poor_finite_runs_and_selects_lowest_as_best(monkeypatch):
    from src import feature_engineer as fe
    frame = _make_history(monkeypatch)
    frame["atr_efficiency_score"] = [147., 72.7, 5.4, np.nan]
    computed = af.compute_atr_run_features(frame)
    assert computed.atr_efficiency_score.iloc[:3].tolist() == [147., 72.7, 5.4]
    features = fe.add_atr_features(frame)
    latest = features.iloc[-1]
    assert latest.atr_efficiency_score_best_3 == 5.4
    assert latest.atr_efficiency_score_avg_3 == pytest.approx((147+72.7+5.4)/3)
    assert dict(af.RUN_METRICS)["atr_efficiency_score"] is False


def test_efficiency_invalid_scores_and_unrated_zero_placeholders_are_unknown(monkeypatch):
    frame = _make_history(monkeypatch)
    frame["atr_efficiency_score"] = [0., -1., np.inf, 0.]
    frame["atr_efficiency_grade"] = ["A++", "A", "F", None]
    frame.loc[3, ["atr_energy_dist", "atr_optimum_energy"]] = None
    values = af.efficiency_error_scores(frame)
    assert values[0] == 0
    assert np.isnan(values[1:]).all()
    assert frame.atr_efficiency_score.iloc[2] == np.inf  # does not mutate raw inputs


def record(day, score, *, grade="C", horse="101", race=None, **kwargs):
    return {"horse_id": horse, "race_id": str(day) if race is None else race,
            "race_date": f"2026-01-{day:02d}", "off_time": "14:00",
            "atr_efficiency_score": score, "atr_efficiency_grade": grade,
            "atr_finishing_speed_pct": 100+day, "atr_race_finishing_speed_pct": 100,
            **kwargs}


def test_grade_order_and_log_error_preserve_bad_runs():
    frame = pd.DataFrame([record(i+1, score, grade=grade) for i, (grade, score) in enumerate([
        ("A++", .03), ("A+", 1.5), ("A", 5), ("B", 12), ("C", 22), ("D", 42), ("E", 73), ("F", 147), ("?", None)])])
    result = derive_run_metrics(frame)
    assert result.atr_rich_efficiency_grade_rank.iloc[:8].tolist() == list(range(8))
    assert np.isnan(result.atr_rich_efficiency_grade_rank.iloc[-1])
    assert result.atr_rich_efficiency_log_error.iloc[7] == pytest.approx(np.log1p(147))


def test_profile_gaps_pair_the_same_positions_including_missing_cells():
    frame = pd.DataFrame([record(1, 42, atr_energy_dist=[100, None, 110, 80],
        atr_optimum_energy=[100, 120, 100, 100])])
    result = derive_run_metrics(frame).iloc[0]
    assert result.atr_rich_profile_worst_gap_pp == 20
    assert result.atr_rich_profile_irregularity_pp == 10  # residuals 0, +10, -20


def test_smooth_cadence_change_is_not_mistaken_for_irregularity():
    base = record(1, 10, atr_stride_markers=["3f", "2f", "1f", "Fin."],
        atr_stride_frequency_markers=["3f", "2f", "1f", "Fin."],
        atr_stride_length=[24, 24, 24, 24], atr_stride_frequency=[2, 2.1, 2.2, 2.3])
    frame = pd.DataFrame([base, {**base, "atr_stride_frequency": [2, 2.5, 1.8, 2.3]}])
    result = derive_run_metrics(frame)
    assert result.atr_rich_cadence_jitter_pct.iloc[0] == pytest.approx(0, abs=1e-12)
    assert result.atr_rich_cadence_jitter_pct.iloc[1] > 5


def test_consistency_needs_three_past_observations_and_excludes_current_values():
    frame = pd.DataFrame([record(1, np.expm1(1)), record(5, np.expm1(2)),
                          record(9, np.expm1(4)), record(13, 900), record(17, None)])
    result = build_prior_features(frame)
    name = "atr_rich_efficiency_mad_5"
    assert result[name].iloc[:3].isna().all()
    assert result[name].iloc[3] == pytest.approx(1)  # MAD of [1,2,4]
    changed = frame.copy()
    changed.loc[3, "atr_efficiency_score"] = .001
    assert build_prior_features(changed)[name].iloc[3] == result[name].iloc[3]
    assert set(CONSISTENCY).issubset(result.columns)


def test_independent_cadence_curve_does_not_need_a_stride_length_curve():
    frame = pd.DataFrame([record(1, 20, atr_stride_frequency=[2, 2.1, 2.2, 2.3],
                                 atr_stride_frequency_markers=["3f", "2f", "1f", "Fin."])])
    result = derive_run_metrics(frame).iloc[0]
    assert np.isfinite(result.atr_rich_late_cadence_change_pct)
    assert result.atr_rich_cadence_jitter_pct == pytest.approx(0, abs=1e-12)
    assert np.isnan(result.atr_rich_late_stride_length_change_pct)


def test_collection_guard_ignores_feature_edits_but_still_detects_source_edits(tmp_path, monkeypatch):
    from pathlib import Path
    from src.ingestion import atr_rebuild
    project = Path(atr_rebuild.config.BASE_DIR)
    for name in (*atr_rebuild.CODE_FILES, "src/atr_features.py"):
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((project/name).read_bytes())
    monkeypatch.setattr(atr_rebuild.config, "BASE_DIR", str(tmp_path))
    before = atr_rebuild.code_hashes()
    with (tmp_path/"src/atr_features.py").open("a", encoding="utf-8") as stream:
        stream.write("\n_NEW_FEATURE = 1\n")
    assert atr_rebuild.code_hashes() == before
    with (tmp_path/"src/ingestion/attheraces.py").open("a", encoding="utf-8") as stream:
        stream.write("\n_NEW_PARSER_SETTING = 1\n")
    assert atr_rebuild.code_hashes() != before
    after_parser_edit = atr_rebuild.code_hashes()
    with (tmp_path/"src/ingestion/atr_rebuild.py").open("a", encoding="utf-8") as stream:
        stream.write("\n_NEW_ACCEPTANCE_SETTING = 1\n")
    assert atr_rebuild.code_hashes() != after_parser_edit
