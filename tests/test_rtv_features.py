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


def test_latest_metric_survives_gaps_within_three_observations():
    runs = observations()
    runs["late_speed_ratio"] = [1.05, 1.25, np.nan, np.nan]
    later = runs.iloc[[3]].assign(race_id="f", race_date=pd.Timestamp("2025-02-10"),
                                available_date=pd.Timestamp("2025-02-12"))
    runs = pd.concat([runs, later], ignore_index=True)
    frame = pd.DataFrame({"horse_id": [1, 2, 1, 1, 1, 3],
                          "race_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-12",
                                                       "2025-02-03", "2025-02-12", "2025-02-12"])})
    out = rtv.add_history_features(frame, runs=runs)
    np.testing.assert_allclose(out.rtv_late_speed_ratio_last, [np.nan, 1.25, 1.05, 1.05, np.nan, np.nan])
    pd.testing.assert_series_equal(out.rtv_late_speed_ratio_last.notna(), out.rtv_late_speed_ratio_mean3.notna(),
                                   check_names=False)


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


def linked_snapshots(race_ids, horse_ids, observed_at, status="matched"):
    """Snapshot rows as the selected dataset serves them: with their saved links."""
    return pd.DataFrame({"sl_race_id": pd.array(race_ids, dtype="string"),
        "sl_horse_id": pd.array(horse_ids, dtype="string"), "match_status": status, "observed_at": observed_at})


def test_snapshot_requires_matching_runner_and_pre_decision_timestamp():
    frame = pd.DataFrame({"race_id": ["x"], "horse_id": [1], "race_date": ["2025-01-03"], "track": ["Kempton Park"],
        "off_time": ["13:00"], "horse_name": ["Test Horse"]})
    source = linked_snapshots(["x"]*3, ["1"]*3,
        ["2025-01-03T10:00:00Z", "2025-01-03T12:54:00Z", "2025-01-03T12:56:00Z"])
    for col in rtv.PRE_FEATURES:
        source[col] = [5., 10., 999.]
    out = rtv.add_pre_race_features(frame, observations=source, as_of="2025-01-03T12:55:00Z")
    assert out.rtv_pre_pace_position.iloc[0] == 10
    early = rtv.add_pre_race_features(frame, observations=source, as_of="2025-01-03T11:00:00Z")
    assert early.rtv_pre_pace_position.iloc[0] == 5
    for wrong in (frame.assign(race_id="y"), frame.assign(horse_id=2)):
        result = rtv.add_pre_race_features(wrong, observations=source, as_of="2025-01-03T12:55:00Z")
        assert result[list(rtv.PRE_FEATURES)].isna().all().all()
    # A snapshot whose link was not accepted supplies nothing, even by name.
    ambiguous = source.assign(match_status="ambiguous")
    result = rtv.add_pre_race_features(frame, observations=ambiguous, as_of="2025-01-03T12:55:00Z")
    assert result[list(rtv.PRE_FEATURES)].isna().all().all()
    with pytest.raises(ValueError, match="canonical links"):
        rtv.add_pre_race_features(frame, observations=source.drop(columns="match_status"), as_of="2025-01-03T12:55:00Z")
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


def select_rtv(publish_dataset, **tables):
    """Publish Racing TV history tables (already keyed by SL IDs) and select them."""
    from src.source_contracts import TableContract
    return publish_dataset(rtv={name: (frame, TableContract("rtv history", ())) for name, frame in tables.items()})


def test_external_fingerprint_uses_availability_date_and_detects_historical_edits(publish_dataset):
    from src.feature_inputs import rtv_cache_signature
    runs = observations()
    select_rtv(publish_dataset, runs=runs)
    first = rtv_cache_signature("2025-01-11")
    assert first["runs"] is not None and first["pre"] is None
    runs.loc[runs.race_date.eq("2025-01-10"), "fsp"] = 999.
    select_rtv(publish_dataset, runs=runs)
    assert rtv_cache_signature("2025-01-11") == first
    runs.loc[runs.race_date.eq("2025-01-01"), "fsp"] = 999.
    select_rtv(publish_dataset, runs=runs)
    assert rtv_cache_signature("2025-01-11") != first


def test_derived_history_uses_only_matured_earlier_runs():
    runs = observations()
    runs["time_fig"] = [10., 20., np.nan, 90.]
    runs["avg_length_rel"] = [1., 2., 3., 4.]
    out = rtv.add_history_features(cards(), runs=runs)
    assert np.isnan(out.loc[0, "rtv_time_fig_mean3"])
    assert out.loc[2, "rtv_time_fig_best3"] == 10
    # Race c has no figure: the mean and best skip it, the last figure carries.
    assert out.loc[3, "rtv_time_fig_mean3"] == 10
    assert out.loc[3, "rtv_time_fig_best3"] == 10
    assert out.loc[3, "rtv_time_fig_last"] == 10
    assert out.loc[3, "rtv_avg_length_rel_mean3"] == 2
    changed = runs.assign(time_fig=runs.time_fig.where(runs.race_date.lt("2025-02-01"), -999.))
    pd.testing.assert_frame_equal(out[list(rtv.DERIVED_FEATURES)],
        rtv.add_history_features(cards(), runs=changed)[list(rtv.DERIVED_FEATURES)], check_exact=True)


class _FlatPar:
    def predict(self, frame):
        return np.full(len(frame), np.log(60.))


def test_time_figures_and_within_race_measures():
    races = pd.DataFrame({"race_id": [10], "race_date": ["2025-01-01"], "track_name": ["Kempton"], "surface": ["Polytrack"],
        "distance_yards": [1000.], "win_time_s": [60.], "distance_furlongs": [4.5]})
    runners = pd.DataFrame({"race_id": [10]*5, "runner_id": [11, 12, 13, 14, 15],
        "is_finisher": [True, True, False, True, True], "is_nonrunner": [False]*4+[True]})
    links = pd.DataFrame({"rtv_race_id": [10]*3, "rtv_runner_id": [11, 12, 13], "sl_race_id": ["sl"]*3,
        "sl_horse_id": [1, 2, 3], "race_date": ["2025-01-01"]*3})
    sectors = pd.DataFrame({"race_id": [10]*6, "runner_id": [11, 11, 12, 12, 13, 13], "sector_number": [1, 2]*3,
        "sector_time": [30.]*6, "sector_length_yards": [500.]*6, "race_position": [1, 1, 2, 2, 3, 3],
        "cumulative_sector_time": [30., 60., 27., 54., 31., 62.]})
    metrics = pd.DataFrame({"race_id": [10]*4, "runner_id": [11, 12, 13, 14], "metric": ["avg_length"]*4,
        "value": [7., 8., 9., 6.]})
    result = rtv.build_run_observations(races, runners, metrics, sectors, pd.DataFrame(), links, par=_FlatPar())
    assert result.time_fig.iloc[0] == pytest.approx(0.)
    assert result.time_fig.iloc[1] == pytest.approx(100*np.log(60/54))
    assert np.isnan(result.time_fig.iloc[2])  # a non-finisher gets no figure
    assert result.avg_length_rel.iloc[:3].tolist() == pytest.approx(
        ((np.array([7., 8., 9.])-7.5)/np.std([7., 8., 9., 6.], ddof=1)).tolist())
    too_few = rtv.build_run_observations(races, runners, metrics.iloc[:3], sectors, pd.DataFrame(), links, par=_FlatPar())
    assert too_few.avg_length_rel.isna().all()
    assert rtv.build_run_observations(races, runners, metrics, sectors, pd.DataFrame(), links).time_fig.isna().all()


def test_time_par_ignores_races_outside_the_par_window():
    days = pd.date_range(rtv.PAR_START, rtv.PAR_END, periods=600).strftime("%Y-%m-%d")
    distance = np.tile([1000., 1500., 2000., 2500.], 150)
    window = pd.DataFrame({"race_id": np.arange(600), "race_date": days, "track_name": "Kempton", "surface": "Polytrack",
        "going": "standard", "race_type": "Flat", "distance_yards": distance, "win_time_s": distance/15})
    # Older results timed by another method, and later races, must not move the par.
    outside = window.assign(race_id=window.race_id+1000, race_date="2021-07-10", win_time_s=distance/12)
    later = window.assign(race_id=window.race_id+2000, race_date="2025-01-01", win_time_s=distance/18)
    probe = window.head(8)
    expected = rtv.fit_time_par(window).predict(rtv._par_inputs(probe))
    actual = rtv.fit_time_par(pd.concat([outside, window, later], ignore_index=True)).predict(rtv._par_inputs(probe))
    assert actual.tolist() == expected.tolist()


def test_card_values_parse_symbols_and_unrated_cards():
    raw = pd.DataFrame({"card_timeform_rating": ["111p", "94?", "-", "0", None],
        "card_timeform_star_rating": [4, 2, 1, 3, None], "card_timeform_tipped_place": [1, None, 3, None, None],
        "card_focus_horse": [True, "False", "true", False, None]})
    values = rtv.card_values(raw)
    assert values.rtv_card_tf_rating.iloc[:2].tolist() == [111., 94.]
    assert values.rtv_card_tf_rating.iloc[2:].isna().all()
    assert values.rtv_card_tf_improver.iloc[:4].tolist() == [1., 0., 0., 0.]
    assert values.rtv_card_tf_doubtful.iloc[:4].tolist() == [0., 1., 0., 0.]
    assert values.rtv_card_tf_tipped.iloc[:4].tolist() == [3., 0., 1., 0.]
    assert values.rtv_card_tf_focus.iloc[:4].tolist() == [1., 0., 1., 0.]
    assert values.iloc[4].isna().all()  # no card published at all


def test_card_snapshot_wins_and_backfill_fills_only_unsnapshotted_rows():
    frame = pd.DataFrame({"race_id": ["x", "x", "live"], "horse_id": [1, 2, 3], "race_date": ["2025-01-03"]*3,
        "track": ["Kempton Park"]*3, "off_time": ["13:00"]*3, "horse_name": ["Test Horse", "Other", "Third"]})
    snapshot = linked_snapshots(["x"], ["1"], ["2025-01-03T10:00:00Z"])
    for col in rtv.SNAPSHOT_FEATURES:
        snapshot[col] = 5.
    backfill = pd.DataFrame({"race_id": ["x", "x"], "horse_id": [1, 2], "race_date": pd.to_datetime(["2025-01-03"]*2)})
    for col in rtv.CARD_FEATURES:
        backfill[col] = 7.
    out = rtv.add_pre_race_features(frame, observations=snapshot, as_of="2025-01-03T12:55:00Z")
    out = rtv.add_card_features(out, cards=backfill)
    assert out.loc[0, list(rtv.CARD_FEATURES)].eq(5.).all()
    assert out.loc[1, list(rtv.CARD_FEATURES)].eq(7.).all()
    assert out.loc[2, list(rtv.CARD_FEATURES)].isna().all()
    # A snapshot file written before the card family existed still joins.
    old = rtv.add_pre_race_features(frame, observations=snapshot.drop(columns=list(rtv.CARD_FEATURES)),
        as_of="2025-01-03T12:55:00Z")
    assert old.rtv_pre_pace_position.iloc[0] == 5 and old[list(rtv.CARD_FEATURES)].isna().all().all()


def test_card_fingerprint_covers_the_race_date_itself(publish_dataset):
    from src.feature_inputs import rtv_cache_signature
    cards = pd.DataFrame({"race_id": ["a", "b"], "horse_id": [1, 1], "race_date": pd.to_datetime(["2025-01-10", "2025-01-12"])})
    for col in rtv.CARD_FEATURES:
        cards[col] = 1.
    select_rtv(publish_dataset, cards=cards)
    first = rtv_cache_signature("2025-01-10")
    assert first["cards"] is not None and first["runs"] is None
    cards.loc[1, "rtv_card_tf_rating"] = 99.
    select_rtv(publish_dataset, cards=cards)
    assert rtv_cache_signature("2025-01-10") == first
    cards.loc[0, "rtv_card_tf_rating"] = 99.
    select_rtv(publish_dataset, cards=cards)
    assert rtv_cache_signature("2025-01-10") != first



def test_rider_claims_keep_explicit_zero_and_require_predecision_evidence(monkeypatch):
    frame = pd.DataFrame({
        "race_id": ["past"] * 3 + ["live"] * 4, "horse_id": range(1, 8),
        "race_date": "2026-01-03", "off_time": "13:00", "jockey": "J. Doe", "weight_lbs": 140,
        "finish_position": [1, 2, 0, 0, 0, 0, 0],
        "reported_jockey_claim_lbs": [None, 5, None, None, None, None, None],
        "prediction_as_of": "2026-01-03T12:50:00Z",
    })
    history = pd.DataFrame({
        "race_id": ["past", "past", "past", "live"], "horse_id": ["1", "2", "3", "7"],
        "race_date": "2026-01-03", "rider_name": "j doe", "rider_weight_lbs": 140,
        "rider_claim_lbs": [0, 7, 3, 5],
    })
    pre = linked_snapshots(["live"] * 5, ["4", "4", "5", "6", "6"], [
        "2026-01-03T12:45:00Z", "2026-01-03T12:53:00Z", "2026-01-03T12:51:00Z",
        "2026-01-03T12:45:00Z", "2026-01-03T12:46:00Z"])
    pre = pre.assign(rider_card_observed=1, rider_name=["j doe"] * 4 + ["s changed"],
                     rider_weight_lbs=140, rider_claim_lbs=[5, 7, 3, 5, 5])
    monkeypatch.setattr("src.source_dataset.read_table",
                        lambda provider, table, **kwargs: (history if table == "rider_claims" else pre).copy())
    result = rtv.add_rider_claims(frame)
    np.testing.assert_allclose(result.reported_jockey_claim_lbs,
                               [0, np.nan, 3, 5, np.nan, np.nan, np.nan], equal_nan=True)
    pd.testing.assert_series_equal(frame.weight_lbs, result.weight_lbs)
