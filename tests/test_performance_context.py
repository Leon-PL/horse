from copy import deepcopy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import database
from src.data_scraper import SportingLifeScraper, _parse_reported_lbs
from src.performance_context import context_history, description_mentions


def ride(value=None, *, result=True):
    return SportingLifeScraper()._ride_to_dict({"horse": {"name": "A"}, "handicap": "11-0",
        "jockey_claim": value, "weight_penalty": 5, "ride_description": "Hampered, stayed on",
        "finish_position": 0, "casualty": {"reason": "Fell"}}, race_id="r", date_str="2026-01-01",
        off_time="12:00", course_name="Test", region="UK", race_name="Test Handicap",
        race_class="Class3", race_type="Hurdle", distance_f=16., going="Good", prize_money=1000.,
        num_runners=2, surface_str="Turf", has_handicap=True, is_result=result)


@pytest.mark.parametrize("value", [None, "", "unknown", True, -3, float("nan"), 2.5, 1000])
def test_unknown_claim_stays_unknown(value):
    assert _parse_reported_lbs(value) is None


def test_claim_and_penalty_preserve_allocated_weight_and_pending_card_clears_result_text():
    result = ride(5)
    assert result["weight_lbs"] == 154
    assert result["reported_jockey_claim_lbs"] == result["weight_penalty_lbs"] == 5
    assert result["nonfinish_reason"] == "Fell"
    assert result["finish_pos_label"] == "F"
    card = ride(5, result=False)
    assert card["ride_description"] is card["nonfinish_reason"] is None
    assert card["reported_jockey_claim_lbs"] == 5
    assert _parse_reported_lbs(0) == 0


def test_source_context_is_transactional_idempotent_and_separate_from_model_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path/"races.db"))
    frame = pd.DataFrame([ride(5), {**ride(None), "horse_name": "B", "finish_position": 1, "won": 1}])
    assert database.insert_results(frame) == 2
    assert database.insert_results(frame) == 0
    source = database.load_result_context()
    assert len(source) == 2 and source.content_sha256.str.len().eq(64).all()
    assert source.observed_at.str.endswith("+00:00").all()
    assert source.loc[source.horse_name.eq("B"), "reported_jockey_claim_lbs"].isna().all()
    core = database.load_from_database()
    assert "reported_jockey_claim_lbs" not in core and "ride_description" not in core
    with pytest.raises(RuntimeError):
        with database._get_conn() as conn:
            database._insert_result_context(conn, frame.assign(ride_description="new evidence"))
            raise RuntimeError("rollback")
    assert len(database.load_result_context()) == 2


def test_mentions_preserve_unknown_and_explicit_negation():
    assert np.isnan(description_mentions(None)["impeded"])
    assert description_mentions("Was not hampered, stayed on")["impeded"] == 0
    assert description_mentions("Badly hampered, stayed on")["impeded"] == 1
    assert description_mentions("No mistake at the last")["jump_error"] == 0


def history_frame():
    return pd.DataFrame({"race_id": ["a", "b", "c", "d", "e"], "horse_name": ["A"]*5,
        "race_date": pd.date_range("2026-01-01", periods=5), "finish_position": [1,0,0,2,0],
        "finish_pos_label": ["", "F", "DNF", "", ""],
        "ride_description": ["Hampered", "Mistake, fell", None, "Ran on", "Future text"],
        "nonfinish_reason": [None, "Fell", None, None, None]})


def test_context_history_is_prior_day_missing_aware_and_seed_invariant():
    frame = history_frame()
    full = context_history(frame)
    assert np.isnan(full.context_impeded_mention_rate_3.iloc[0])
    assert full.context_description_support_3.iloc[3] == 2
    assert full.context_reason_support_5.iloc[3] == 2  # generic DNF stays unknown
    _, seed = context_history(frame.iloc[:2], return_state=True)
    before = deepcopy(seed)
    resumed = context_history(frame.iloc[2:], state=seed)
    pd.testing.assert_frame_equal(resumed, full.iloc[2:])
    assert seed == before
    changed = frame.copy()
    changed.loc[2:, "ride_description"] = "Hampered, slowly away"
    pd.testing.assert_frame_equal(context_history(changed).iloc[:3], full.iloc[:3])
    with pytest.raises(ValueError):
        context_history(frame.iloc[:1], state=seed)


def test_unmapped_nonfinish_reason_is_not_known_negative_accident_evidence():
    frame = history_frame().iloc[:2].copy()
    frame.loc[0,["finish_position","finish_pos_label","nonfinish_reason"]] = [0,"DNF","UnmappedNewSourceEnum"]
    result = context_history(frame)
    assert result.context_reason_support_5.iloc[1] == 0
    assert np.isnan(result.context_accidental_nonfinish_rate_5.iloc[1])
