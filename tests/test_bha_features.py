"""BHA candidates must remain causal, attributable and independent of labels."""

import numpy as np
import pandas as pd
import pytest

from src import bha_events as events, bha_features as features
from src.source_dataset import use_manifest, publish_source
from src.source_contracts import TableContract


@pytest.fixture(autouse=True)
def recorded_policy_baseline(monkeypatch):
    # Keep the explicit capture-only regressions; separate tests below verify
    # the newly selected retrospective policy and its exact 48-hour boundary.
    import config
    monkeypatch.setattr(config, 'BHA_FEATURE_AVAILABILITY_POLICY', 'recorded')


def canonical(days=(1, 3, 5, 7, 9, 11)):
    return pd.DataFrame([{'race_id': f'r{day}', 'horse_id': str(h), 'horse_name': name,
        'race_date': f'2026-06-{day:02d}', 'track': 'Aintree', 'off_time': '14:00'}
        for day in days for h, name in ((1, 'Alpha'), (2, 'Beta'))])


def report(text='ALPHA made a respiratory noise.', *, available='2026-06-02T12:00:00Z', heading=True):
    return {'report_key': '2026:1', 'race_date': '2026-06-01', 'course': 'AINTREE',
            'available_at': available, 'presence_status': 'present', 'runner_notes_html': '',
            'enquiry_html': ('<p class="stewardsroom-head">Race 1 - 3:00pm</p>' if heading else '') + f'<p>{text}</p>'}


def bundle(text='ALPHA made a respiratory noise.', **kwargs):
    return events.build_observations(canonical(), {'reports': pd.DataFrame([report(text, **kwargs)])})


def target(day=3, horse='1'):
    return pd.DataFrame([{'race_id': f'r{day}', 'horse_id': horse,
                         'race_date': f'2026-06-{day:02d}', 'off_time': '14:00'}])


def test_incidents_are_attributed_to_their_horse_not_the_whole_paragraph():
    found, _ = events.attribute('ALPHA lost its right-hind shoe and BETA bled.', [(0, 'Alpha'), (1, 'Beta')], covered=True)
    assert found[0]['lost_shoe'] == 1 and found[0]['bled'] == 0
    assert found[1]['bled'] == 1 and found[1]['lost_shoe'] == 0


def test_negation_and_explicit_negative_examination_are_separate_from_an_issue():
    text = ('The Veterinary Officer examined ALPHA and found no evidence of bleeding. '
            'The gelding had nothing to report.')
    found, _ = events.attribute(text, [(0, 'Alpha'), (1, 'Beta')], covered=True)
    assert found[0]['bled'] == 0 and found[0]['vet_exam'] == 1 and found[0]['vet_clear'] == 1
    assert found[1]['vet_clear'] == 0 and found[1]['named_in_report'] == 0


def test_interference_causer_and_victim_are_not_interchanged():
    found, _ = events.attribute('ALPHA shifted left, hampering BETA.', [(0, 'Alpha'), (1, 'Beta')], covered=True)
    assert found[0]['interference_caused'] == 1 and found[0]['interference_suffered'] == 0
    assert found[1]['interference_suffered'] == 1 and found[1]['interference_caused'] == 0


def test_unknown_horse_does_not_transfer_its_finding_to_a_known_horse():
    found, _ = events.attribute('ALPHA finished well and FOREIGN HORSE bled.', [(0, 'Alpha')], covered=True)
    assert found[0]['bled'] == 0


def test_passive_interference_does_not_turn_the_victim_into_the_causer():
    found, _ = events.attribute('ALPHA was hampered by BETA.', [(0, 'Alpha'), (1, 'Beta')], covered=True)
    assert found[0]['interference_suffered'] == 1 and found[0]['interference_caused'] == 0
    assert found[1]['interference_caused'] == 1


def test_shared_preparation_permission_applies_to_the_named_horses():
    found, _ = events.attribute('Permission was given for ALPHA and BETA to wear a red hood to post.',
                                [(0, 'Alpha'), (1, 'Beta')], covered=True)
    assert found[0]['red_hood_to_post'] == found[1]['red_hood_to_post'] == 1


def test_ambiguous_names_are_not_linked():
    found, _ = events.attribute('ALPHA (IRE) bled.', [(0, 'Alpha'), (1, 'Alpha (IRE)')], covered=True)
    assert not found


def test_bst_uses_exact_local_time_and_does_not_guess_an_adjacent_hour():
    c = canonical((1,))
    c1, observed, _, _ = events.build_observations(c, {'reports': pd.DataFrame([report()])})
    assert len(observed) == 2 and c1._local.eq('15:00').all()
    wrong = report()
    wrong['enquiry_html'] = wrong['enquiry_html'].replace('3:00pm', '4:00pm')
    _, observed, _, _ = events.build_observations(c, {'reports': pd.DataFrame([wrong])})
    assert observed.empty


def test_case_sensitive_names_do_not_match_ordinary_english_words():
    found, _ = events.attribute('The horse had no problems.', [(0, 'The')], covered=False)
    assert not found


def test_later_observations_do_not_fill_earlier_predictions():
    c, observed, withdrawals, _ = bundle()
    early = features.compute_features(target(1), c, observed, withdrawals)
    later = features.compute_features(target(3), c, observed, withdrawals)
    assert early.bha_report_count_5.iloc[0] == 0 and np.isnan(early.bha_breathing_last.iloc[0])
    assert later.bha_breathing_last.iloc[0] == 1 and later.bha_report_count_5.iloc[0] == 1
    assert later.bha_breathing_days_since_5.iloc[0] == 2


def test_never_uses_current_race_report_even_if_its_timestamp_is_wrong():
    c, observed, withdrawals, _ = bundle(available='2026-06-01T10:00:00Z')
    values = features.compute_features(target(1), c, observed, withdrawals)
    assert values.bha_report_count_5.iloc[0] == 0


def test_prediction_cutoff_can_only_reduce_available_information():
    c, observed, withdrawals, _ = bundle()
    row = target(3)
    row['prediction_as_of'] = '2026-06-02T10:00:00Z'
    values = features.compute_features(row, c, observed, withdrawals)
    assert values.bha_report_count_5.iloc[0] == 0


def test_uncovered_starts_occupy_window_slots_and_are_not_zero_issues():
    c, observed, withdrawals, _ = bundle()
    values = features.compute_features(target(7), c, observed, withdrawals)
    assert np.isnan(values.bha_breathing_last.iloc[0])
    assert values.bha_breathing_rate_3.iloc[0] == 1 and values.bha_report_count_3.iloc[0] == 1
    expired = features.compute_features(target(13), c, observed, withdrawals)
    assert expired.bha_report_count_5.iloc[0] == 0 and np.isnan(expired.bha_breathing_rate_5.iloc[0])


def test_later_full_pdf_only_affects_predictions_after_its_capture():
    r = report('Nothing to report.')
    tables = {'reports': pd.DataFrame([r]), 'documents': pd.DataFrame([{
        'report_key': '2026:1', 'available_at': '2026-06-04T12:00:00Z',
        'going_text': 'Good to Soft (GSt 5.5)', 'rail_text': 'Rails moved out', 'non_runners_text': ''}]),
        'race_notes': pd.DataFrame([{'report_key': '2026:1', 'off_time_local': '15:00',
        'available_at': '2026-06-04T12:00:00Z', 'text': 'ALPHA made a respiratory noise.'}])}
    c, observed, withdrawals, _ = events.build_observations(canonical(), tables)
    early = features.compute_features(target(3), c, observed, withdrawals)
    later = features.compute_features(target(5), c, observed, withdrawals)
    assert early.bha_breathing_last.iloc[0] == 0 and np.isnan(early.bha_goingstick_last.iloc[0])
    assert later.bha_breathing_rate_5.iloc[0] == 1
    assert later.bha_goingstick_mean_3.iloc[0] == 5.5


def test_chunks_and_input_order_produce_identical_features():
    c, observed, withdrawals, _ = bundle()
    frame = pd.concat([target(day, horse) for day in (3, 5, 7, 9) for horse in ('1', '2')], ignore_index=True)
    full = features.compute_features(frame, c, observed, withdrawals)
    parts = pd.concat([features.compute_features(frame.iloc[:3], c, observed, withdrawals),
                       features.compute_features(frame.iloc[3:], c, observed, withdrawals)])
    pd.testing.assert_frame_equal(full, parts)
    shuffled = frame.sample(frac=1, random_state=2)
    pd.testing.assert_frame_equal(full, features.compute_features(shuffled, c, observed, withdrawals).sort_index())


def test_targets_labels_prices_and_current_outcomes_cannot_change_bha_features():
    c, observed, withdrawals, _ = bundle()
    frame = target(5)
    before = features.compute_features(frame, c, observed, withdrawals)
    frame['won'], frame['finish_position'], frame['odds'] = 1, 1, 1.01
    pd.testing.assert_frame_equal(before, features.compute_features(frame, c, observed, withdrawals))


def test_withdrawal_history_works_even_without_a_previous_run_report():
    c = canonical((3,))
    tables = {'reports': pd.DataFrame([report(heading=False)]), 'documents': pd.DataFrame([{
        'report_key': '2026:1', 'available_at': '2026-06-02T12:00:00Z',
        'non_runners_text': 'Race Horse and Trainer Reason\n3:00pm ALPHA (IRE), trained by\nTrainer Name\nSelf Cert (Infected Foot)'}])}
    prepared, observations, withdrawals, _ = events.build_observations(c, tables)
    assert observations.empty and len(withdrawals) == 1
    values = features.compute_features(target(3), prepared, observations, withdrawals)
    assert values.bha_health_withdrawals_365d.iloc[0] == 1 and values.bha_days_since_withdrawal.iloc[0] == 2


def test_missing_selected_source_cannot_fall_back_to_collectors(monkeypatch):
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('Unexpected source read'))
    with use_manifest({'sources': {}}):
        values = features.add_bha_features(target())
    assert values.bha_report_count_5.iloc[0] == 0 and np.isnan(values.bha_lame_last.iloc[0])


def test_explicit_unversioned_frames_do_not_silently_read_new_sources(monkeypatch):
    monkeypatch.setattr(features, 'load_bundle', lambda: pytest.fail('Unpinned BHA source read'))
    with use_manifest(None):
        values = features.add_bha_features(target())
    assert set(features.FEATURES).issubset(values)


def test_candidate_discovery_uses_new_columns_without_changing_the_live_allowlist(monkeypatch):
    import config
    from src.model import get_feature_columns
    from src.feature_selection import candidate_feature_columns
    monkeypatch.setattr(config, 'FEATURE_ALLOWLIST', ['existing'])
    frame = pd.DataFrame({'existing': [1.], 'bha_lame_last': [0.], 'won': [1], 'horse_id': ['h']})
    assert get_feature_columns(frame) == ['existing']
    assert candidate_feature_columns(frame) == ['existing', 'bha_lame_last']
    assert config.FEATURE_ALLOWLIST == ['existing']


def test_only_available_bha_content_invalidates_historical_signatures(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    rows = pd.DataFrame([report()])
    contract = TableContract('fixture report', ('report_key', 'available_at'), ('report_key',))
    first = publish_source('bha', {'reports': rows}, contracts={'reports': contract})
    with use_manifest({'sources': {'bha': first}}):
        old = features.cache_signature('2026-06-03')
    future = {**report(), 'report_key': '2026:2', 'available_at': '2026-07-01T12:00:00Z'}
    second = publish_source('bha', {'reports': pd.concat([rows, pd.DataFrame([future])])}, contracts={'reports': contract})
    with use_manifest({'sources': {'bha': second}}):
        assert features.cache_signature('2026-06-03') == old
    rows['enquiry_html'] = '<p>Amended historical observation.</p>'
    third = publish_source('bha', {'reports': rows}, contracts={'reports': contract})
    with use_manifest({'sources': {'bha': third}}):
        assert features.cache_signature('2026-06-03') != old


def test_observation_availability_requires_a_timezone():
    with pytest.raises(ValueError, match='timezone'):
        bundle(available='2026-06-02 12:00:00')


def test_two_day_policy_populates_backfilled_history_only_after_48_hours(monkeypatch):
    import config
    monkeypatch.setattr(config, 'BHA_FEATURE_AVAILABILITY_POLICY', 'race_plus_48h')
    c, observed, withdrawals, _ = bundle(available='2026-11-07T17:55:00Z')
    original = observed.copy(deep=True)
    before = target(3)
    before['prediction_as_of'] = '2026-06-03T13:59:59Z'
    assert features.compute_features(before, c, observed, withdrawals).bha_report_count_5.iloc[0] == 0
    exact = features.compute_features(target(3), c, observed, withdrawals)
    assert exact.bha_breathing_last.iloc[0] == 1 and exact.bha_report_count_5.iloc[0] == 1
    pd.testing.assert_frame_equal(observed, original)


def test_two_day_policy_keeps_current_race_and_future_starts_out():
    c, observed, withdrawals, _ = bundle(available='2026-11-07T17:55:00Z')
    current = features.compute_features(target(1), c, observed, withdrawals, policy='race_plus_48h')
    assert current.bha_report_count_5.iloc[0] == 0
    frame = target(3)
    frame['prediction_as_of'] = '2026-05-31T12:00:00Z'
    assert features.compute_features(frame, c, observed, withdrawals, policy='race_plus_48h').bha_report_count_5.iloc[0] == 0


def test_two_day_policy_applies_to_withdrawals_without_changing_the_capture_time():
    c = events.prepare_canonical(canonical((3,)))
    withdrawals = pd.DataFrame([{'horse_id': '1', 'event_at': '2026-06-01T14:00:00Z',
        'available_at': '2026-11-07T17:55:00Z', 'health': 1., 'going': 0.}])
    row = target(3)
    early = features.compute_features(row, c, pd.DataFrame(), withdrawals, policy='recorded')
    assumed = features.compute_features(row, c, pd.DataFrame(), withdrawals, policy='race_plus_48h')
    assert np.isnan(early.bha_health_withdrawals_365d.iloc[0])
    assert assumed.bha_health_withdrawals_365d.iloc[0] == 1
    assert withdrawals.available_at.iloc[0] == '2026-11-07T17:55:00Z'


def test_two_day_policy_is_order_and_chunk_invariant():
    c, observed, withdrawals, _ = bundle(available='2026-11-07T17:55:00Z')
    frame = pd.concat([target(d, h) for d in (1, 3, 5, 7) for h in ('1', '2')], ignore_index=True)
    full = features.compute_features(frame, c, observed, withdrawals, policy='race_plus_48h')
    parts = pd.concat([features.compute_features(frame.iloc[:3], c, observed, withdrawals, policy='race_plus_48h'),
                       features.compute_features(frame.iloc[3:], c, observed, withdrawals, policy='race_plus_48h')])
    pd.testing.assert_frame_equal(full, parts)
    shuffled = frame.sample(frac=1, random_state=5)
    pd.testing.assert_frame_equal(full, features.compute_features(shuffled, c, observed, withdrawals,
                                  policy='race_plus_48h').sort_index())


def test_two_day_assumption_is_in_feature_code_and_history_signatures(tmp_path, monkeypatch):
    import config
    from src.featured_dataset import fe_signature
    monkeypatch.setattr(config, 'DATA_DIR', str(tmp_path))
    rows = pd.DataFrame([report(available='2026-11-07T17:55:00Z')])
    contract = TableContract('fixture report', ('report_key', 'available_at'), ('report_key',))
    receipt = publish_source('bha', {'reports': rows}, contracts={'reports': contract})
    with use_manifest({'sources': {'bha': receipt}}):
        recorded = features.cache_signature('2026-06-03')
        signature = fe_signature()
        monkeypatch.setattr(config, 'BHA_FEATURE_AVAILABILITY_POLICY', 'race_plus_48h')
        assert features.cache_signature('2026-06-03') != recorded
        assert fe_signature() != signature


def test_historical_writer_records_the_policy_and_real_runner_keys(tmp_path, monkeypatch):
    import config
    import pyarrow.parquet as pq
    monkeypatch.setattr(config, 'BHA_FEATURE_AVAILABILITY_POLICY', 'race_plus_48h')
    c, observed, withdrawals, _ = bundle(available='2026-11-07T17:55:00Z')
    result = features._write_historical(c, observed, withdrawals, tmp_path)
    file = tmp_path/'historical_features.parquet'
    table = pd.read_parquet(file)
    assert result['rows'] == len(c) and result['rows_with_report_history'] > 0
    assert set(zip(table.race_id, table.horse_id)) == set(zip(c.race_id, c.horse_id))
    assert pq.read_metadata(file).metadata[b'bha_availability_policy'] == b'race_plus_48h'


def test_unknown_availability_policy_is_rejected():
    with pytest.raises(ValueError, match='policy'):
        features.availability_policy('guess_publication')
