"""New-tab measurements, availability and canonical history isolation."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.research import atr_new_tabs_features as features


def record(day='2026-01-01', race='1', horse='h1', **extra):
    return dict(horse_id=horse,race_id=race,race_date=day,off_time='14:00',finish_position=1,
        atr_jump_speed_lost_avg_pct=10.,atr_jump_speed_lost_early_pct=8.,
        atr_jump_speed_lost_middle_pct=9.,atr_jump_speed_lost_late_pct=13.,
        atr_jump_consistency_pct=85.,atr_jump_fluency_grade='B',
        atr_jump_course_winner_speed_lost_pct=11.,atr_jumps_par_updated_date=day,
        atr_distance_ran_yards=1010.,atr_race_median_distance_ran_yards=1000.,
        atr_distance_diff_vs_median_lengths=3.7,**extra)


def target(day='2026-01-20', **extra):
    return dict(horse_id='h1',race_id='target',race_date=day,off_time='14:00',**extra)


def test_measurement_units_and_chepstow_values():
    row = record()
    row.update(atr_jump_speed_lost_avg_pct=13.6,atr_jump_speed_lost_early_pct=15.4,
               atr_jump_speed_lost_middle_pct=15.1,atr_jump_speed_lost_late_pct=10.3,
               atr_jump_consistency_pct=83,atr_jump_fluency_grade='C',
               atr_jump_course_winner_speed_lost_pct=11.42)
    result = features.derive_run_metrics(pd.DataFrame([row])).iloc[0]
    assert result.atr_jump_loss_pct == 13.6
    assert result.atr_jump_late_change_pp == pytest.approx(-5.1)
    assert result.atr_jump_phase_range_pp == pytest.approx(5.1)
    assert result.atr_jump_consistency_pct == 83
    assert result.atr_jump_grade_rank == 2
    assert result.atr_jump_vs_course_par_pp == pytest.approx(2.18)
    assert result.atr_path_extra_pct == pytest.approx(1.)
    assert result.atr_path_extra_lengths == 3.7
    assert result.atr_path_above_median == 1


def test_no_fixed_conversion_between_rounded_yards_and_published_lengths():
    row = record()
    row.update(atr_distance_ran_yards=1339.,atr_race_median_distance_ran_yards=1339.,
               atr_distance_diff_vs_median_lengths=.04)
    result=features.derive_run_metrics(pd.DataFrame([row])).iloc[0]
    assert result.atr_path_extra_pct == 0
    assert result.atr_path_extra_lengths == .04


@pytest.mark.parametrize('position',[0,np.nan,-1])
def test_nonfinishers_and_unknown_completion_cannot_earn_shortcut_credit(position):
    row=record()
    row.update(finish_position=position,atr_distance_ran_yards=200,atr_distance_diff_vs_median_lengths=-300)
    assert features.derive_run_metrics(pd.DataFrame([row])).isna().all().all()


def test_official_numeric_placing_can_supply_completion_when_canonical_is_absent():
    row=record()
    del row['finish_position']
    row['atr_official_position']='2'
    assert features.derive_run_metrics(pd.DataFrame([row])).atr_path_extra_pct.iloc[0] == pytest.approx(1.)
    row['atr_official_position']='PU'
    assert features.derive_run_metrics(pd.DataFrame([row])).isna().all().all()


def test_missing_middle_does_not_invent_phase_range():
    row=record()
    row['atr_jump_speed_lost_middle_pct']=None
    result=features.derive_run_metrics(pd.DataFrame([row])).iloc[0]
    assert np.isnan(result.atr_jump_phase_range_pp)
    assert result.atr_jump_late_change_pp == 5


def test_bad_values_and_unrated_zeros_stay_unknown_without_mutation():
    row=record()
    row.update(atr_jump_speed_lost_avg_pct=np.inf,atr_jump_consistency_pct=101,
               atr_distance_ran_yards=0,atr_jump_fluency_grade='A++',atr_jump_course_winner_speed_lost_pct=0)
    frame=pd.DataFrame([row])
    before=frame.copy(deep=True)
    result=features.derive_run_metrics(frame).iloc[0]
    assert np.isnan(result.atr_jump_loss_pct) and np.isnan(result.atr_jump_grade_rank)
    assert np.isnan(result.atr_jump_vs_course_par_pp) and np.isnan(result.atr_path_extra_lengths)
    assert_frame_equal(frame,before)
    row=record()
    row.update({name:0 for name in ['atr_jump_speed_lost_avg_pct','atr_jump_speed_lost_early_pct',
                                   'atr_jump_speed_lost_middle_pct','atr_jump_speed_lost_late_pct','atr_jump_consistency_pct']})
    row['atr_jump_fluency_grade']='?'
    assert features.derive_run_metrics(pd.DataFrame([row]))[list(features.JUMP_METRICS)].isna().all().all()
    row['atr_jump_fluency_grade']='A'
    assert features.derive_run_metrics(pd.DataFrame([row])).atr_jump_loss_pct.iloc[0] == 0


def test_exact_publication_boundary_is_excluded():
    history=pd.DataFrame([record()])
    targets=pd.DataFrame([target('2026-01-02'),target('2026-01-03'),
                          {**target('2026-01-03'),'off_time':'14:01'}])
    output=features.build_prior_features(history,targets)
    assert output.atr_jump_observed_runs_3.tolist() == [0,0,1]
    assert output.atr_jump_loss_pct_median_3.iloc[2] == 10
    assert output.atr_jump_grade_rank_median_3.iloc[2] == 1


def test_late_grade_revision_is_masked_per_cell_without_reordering_history():
    old=record()
    old.update(atr_jump_fluency_grade='A',atr_jumps_par_updated_date='2026-01-20')
    recent=record('2026-01-05',race='2')
    recent['atr_jump_fluency_grade']='E'
    history=pd.DataFrame([old,recent])
    targets=pd.DataFrame([target('2026-01-10'),target('2026-01-22')])
    output=features.build_prior_features(history,targets)
    assert output.atr_jump_grade_rank_median_3.tolist() == [4,2]
    assert output.atr_jump_grade_ab_rate_3.tolist() == [0,.5]
    assert output.atr_jump_loss_pct_median_3.tolist() == [10,10]


def test_unknown_revision_blocks_grade_and_course_par_but_not_raw_loss():
    row=record()
    row['atr_jumps_par_updated_date']=None
    output=features.build_prior_features(pd.DataFrame([row]),pd.DataFrame([target()])).iloc[0]
    assert np.isnan(output.atr_jump_grade_rank_median_3)
    assert np.isnan(output.atr_jump_vs_course_par_pp_median_3)
    assert output.atr_jump_loss_pct_median_3 == 10


def test_revision_day_end_handles_british_summer_time():
    frame=pd.DataFrame([{'atr_jumps_par_updated_date':'2026-03-29'}])
    assert features._revision_end(frame).iloc[0] == pd.Timestamp('2026-03-29T23:00Z')


def test_missing_appearances_occupy_slots_and_old_measurements_expire():
    history=pd.DataFrame([record(),
        {**target('2026-01-05'),'race_id':'2'},
        {**target('2026-01-09'),'race_id':'3'},
        {**target('2026-01-13'),'race_id':'4'}])
    output=features.build_prior_features(history,pd.DataFrame([target()])).iloc[0]
    assert output.atr_jump_observed_runs_3 == 0
    assert output.atr_path_observed_runs_3 == 0
    assert output[list(features.PRIOR_FEATURES)].isna().all()


def test_best_consistency_changes_and_counts_have_expected_direction():
    history=pd.DataFrame([record('2026-01-01','1'),record('2026-01-05','2'),record('2026-01-09','3')])
    history['atr_jump_speed_lost_avg_pct']=[5,10,20]
    history['atr_distance_diff_vs_median_lengths']=[-1,0,3]
    output=features.build_prior_features(history,pd.DataFrame([target()])).iloc[0]
    assert output.atr_jump_loss_pct_median_3 == 10
    assert output.atr_jump_loss_best_3 == 5
    assert output.atr_jump_loss_mad_5 == 5
    assert output.atr_path_extra_mad_5 == 1
    assert output.atr_jump_loss_recent_change == 12.5
    assert output.atr_path_extra_recent_change == 3.5
    assert output.atr_path_above_median_rate_3 == pytest.approx(1/3)
    assert output.atr_jump_observed_runs_3 == output.atr_path_observed_runs_3 == 3
    assert output.atr_jump_observation_age_days == 11


def test_spread_and_change_require_real_support():
    output=features.build_prior_features(pd.DataFrame([record()]),pd.DataFrame([target()])).iloc[0]
    assert np.isnan(output.atr_jump_loss_mad_5)
    assert np.isnan(output.atr_jump_loss_recent_change)


def test_current_race_labels_future_rows_and_other_horses_cannot_leak():
    history=pd.DataFrame([record(),record('2026-01-09','2'),record('2026-01-05','9','other')])
    original=features.build_prior_features(history)
    changed=history.copy(deep=True)
    changed.loc[1,['finish_position','atr_jump_speed_lost_avg_pct','atr_distance_ran_yards']]=[0,99,99999]
    changed['won']=[0,1,1]
    changed['bsp']=[500,1.01,1.01]
    future=record('2026-03-01','future')
    after=features.build_prior_features(pd.concat([changed,pd.DataFrame([future])],ignore_index=True))
    assert_frame_equal(original.iloc[:2],after.iloc[:2])
    assert original.atr_jump_observed_runs_3.iloc[2] == 0


def test_target_order_duplicate_index_and_processing_batches_are_stable():
    history=pd.DataFrame([record('2026-01-09','2'),record(),record('2026-01-05','x','other')])
    targets=pd.DataFrame([target(),target('2026-01-03'),{**target(),'horse_id':'absent'}],index=[8,2,8])
    first=features.build_prior_features(history,targets,batch_size=1)
    second=features.build_prior_features(history,targets,batch_size=1000)
    assert_frame_equal(first,second)
    assert first.index.tolist()==[8,2,8]
    assert first.atr_jump_observed_runs_3.tolist()==[2,0,0]


def test_bad_identity_time_or_duplicate_source_is_rejected_safely():
    history=pd.DataFrame([record()])
    targets=pd.DataFrame([{**target(),'horse_id':None},{**target(),'off_time':'25:00'}])
    output=features.build_prior_features(history,targets)
    assert output.atr_jump_observed_runs_3.eq(0).all()
    assert output[list(features.PRIOR_FEATURES)].isna().all().all()
    with pytest.raises(ValueError,match='Ambiguous'):
        features.build_prior_features(pd.concat([history,history]))


@pytest.mark.parametrize('lag',['0h','24h','NaT'])
def test_short_publication_assumption_is_not_allowed(lag):
    with pytest.raises(ValueError,match='48 hours'):
        features.build_prior_features(pd.DataFrame([record()]),availability_lag=lag)
