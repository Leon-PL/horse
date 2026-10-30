import copy

import numpy as np
import pandas as pd
import pytest

from scripts.audit_handicap_weights import extract_weights, weight_mark_decomposition


def payload():
    return {"props": {"pageProps": {"race": {
        "race_summary": {"race_summary_reference": {"id": 123}, "date": "2026-01-01", "race_stage": "WEIGHEDIN"},
        "rides": [{"horse": {"horse_reference": {"id": 456}, "name": "Example",
            "previous_results": [{"date": "2026-10-01", "weight": "20-0"}], "age": 99},
            "handicap": "9-10", "jockey_claim": 3, "ride_status": "RUNNER"}],
    }}}}


def test_claim_is_separate_from_declared_weight_without_mutating_source():
    data = payload()
    original = copy.deepcopy(data)
    first = extract_weights(data, "123", "2026-01-01", ["456"])
    pd.testing.assert_frame_equal(first, extract_weights(data, "123", "2026-01-01", ["456"]))
    assert first.loc[0, "declared_weight_lbs"] == 136
    assert first.loc[0, "declared_less_reported_claim_lbs"] == 133
    assert "age" not in first
    assert data == original


def test_absent_claim_is_unknown_and_explicit_zero_remains_observed():
    data = payload()
    ride = data["props"]["pageProps"]["race"]["rides"][0]
    ride.pop("jockey_claim")
    absent = extract_weights(data, 123, "2026-01-01")
    assert np.isnan(absent.loc[0, "reported_claim_lbs"])
    assert not absent.loc[0, "claim_field_present"]
    ride["jockey_claim"] = 0
    known = extract_weights(data, 123, "2026-01-01")
    assert known.loc[0, "claim_field_present"]
    assert known.loc[0, "declared_less_reported_claim_lbs"] == 136


@pytest.mark.parametrize("claim", [None, "unknown", -3, 3.5, True, float("nan")])
def test_invalid_or_unavailable_claim_is_not_a_fabricated_allowance(claim):
    data = payload()
    data["props"]["pageProps"]["race"]["rides"][0]["jockey_claim"] = claim
    result = extract_weights(data, 123, "2026-01-01")
    assert np.isnan(result.loc[0, "reported_claim_lbs"])
    assert np.isnan(result.loc[0, "declared_less_reported_claim_lbs"])


@pytest.mark.parametrize("race_id,date,ids", [(124, "2026-01-01", ["456"]),
    (123, "2026-01-02", ["456"]), (123, "2026-01-01", ["457"])])
def test_wrong_event_date_or_complete_field_is_rejected(race_id, date, ids):
    with pytest.raises(ValueError):
        extract_weights(payload(), race_id, date, ids)


def test_nonrunner_claim_cannot_enter_the_competing_field():
    data = payload()
    removed = copy.deepcopy(data["props"]["pageProps"]["race"]["rides"][0])
    removed["horse"]["horse_reference"]["id"] = 789
    removed["ride_status"] = "NON-RUNNER"
    data["props"]["pageProps"]["race"]["rides"].append(removed)
    result = extract_weights(data, 123, "2026-01-01", ["456"])
    assert len(result) == 1


def test_reported_penalty_is_retained_without_adding_it_to_declared_weight_again():
    data = payload()
    data["props"]["pageProps"]["race"]["rides"][0]["weight_penalty"] = 5
    result = extract_weights(data, 123, "2026-01-01")
    assert result.loc[0, "declared_weight_lbs"] == 136
    assert result.loc[0, "declared_less_reported_claim_lbs"] == 133
    assert '"weight_penalty": 5' in result.loc[0, "other_weight_fields"]


def test_big_declared_weight_drop_is_not_called_a_big_rating_drop():
    frame = pd.DataFrame([dict(handicap=1,previous_handicap=1,mark_scale="Hurdle",previous_mark_scale="Hurdle",
        observed_mark=120.,previous_observed_mark=121.,weight_change=-11.,won=1,winner_loss_vs_market=1.)])
    selected, report = weight_mark_decomposition(frame)
    assert selected.loc[0,"mark_change_observed"] == -1
    assert selected.loc[0,"burden_move_minus_mark_move"] == -10
    assert report["contexts"]["weight_drop_at_least_7lb_but_mark_drop_at_most_2lb"]["runners"] == 1


def test_missing_marks_and_incomparable_scales_cannot_make_a_weight_rating_comparison():
    base = dict(handicap=1,previous_handicap=1,mark_scale="Flat|0",previous_mark_scale="Flat|0",
        observed_mark=60.,previous_observed_mark=60.,weight_change=-8.,won=0,winner_loss_vs_market=0.)
    frame = pd.DataFrame([base,{**base,"previous_observed_mark":np.nan},
        {**base,"previous_mark_scale":"Flat|1"},{**base,"previous_handicap":0}])
    selected, _ = weight_mark_decomposition(frame)
    assert selected.index.tolist() == [0]
