"""Tests for collateral-form features (prev-race rivals' next-out results).

The key property under test is the date guard: rival next-out results are
forward-looking per construction and must only be consumed when they
occurred strictly before the current race.
"""

import pandas as pd
import pytest

import config
from src.feature_engineer import add_collateral_form_features


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setattr(config, "COLLATERAL_FORM", True, raising=False)


def _df(rows):
    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"])
    return df.sort_values(["race_date", "race_id"]).reset_index(drop=True)


def test_key_race_signal():
    rows = [
        # Race 1 (Jan 1): A beats B and C
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "A", "finish_position": 1},
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "B", "finish_position": 2},
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "C", "finish_position": 3},
        # B and C both WIN their next starts (Jan 10) — r1 was a key race
        {"race_id": "r2", "race_date": "2025-01-10", "horse_name": "B", "finish_position": 1},
        {"race_id": "r2", "race_date": "2025-01-10", "horse_name": "X", "finish_position": 2},
        {"race_id": "r3", "race_date": "2025-01-10", "horse_name": "C", "finish_position": 1},
        {"race_id": "r3", "race_date": "2025-01-10", "horse_name": "Y", "finish_position": 2},
        # A's next run (Feb 1): collateral = B,C next-out results, both wins
        {"race_id": "r4", "race_date": "2025-02-01", "horse_name": "A", "finish_position": 1},
        {"race_id": "r4", "race_date": "2025-02-01", "horse_name": "Z", "finish_position": 2},
    ]
    out = add_collateral_form_features(_df(rows))
    a_row = out[(out["race_id"] == "r4") & (out["horse_name"] == "A")].iloc[0]
    assert a_row["collateral_n_rivals"] == 2
    assert a_row["collateral_next_win_rate"] == 1.0
    assert a_row["collateral_next_place_rate"] == 1.0


def test_date_guard_excludes_future_and_same_day():
    rows = [
        # Race 1: A beats B
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "A", "finish_position": 1},
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "B", "finish_position": 2},
        # A and B meet again Feb 1: B's "next run" IS this race —
        # its outcome must not be visible to A's features.
        {"race_id": "r2", "race_date": "2025-02-01", "horse_name": "A", "finish_position": 2},
        {"race_id": "r2", "race_date": "2025-02-01", "horse_name": "B", "finish_position": 1},
    ]
    out = add_collateral_form_features(_df(rows))
    a_row = out[(out["race_id"] == "r2") & (out["horse_name"] == "A")].iloc[0]
    assert a_row["collateral_n_rivals"] == 0
    assert pd.isna(a_row["collateral_next_win_rate"])


def test_flag_off_no_columns(monkeypatch):
    monkeypatch.setattr(config, "COLLATERAL_FORM", False, raising=False)
    rows = [
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "A", "finish_position": 1},
        {"race_id": "r1", "race_date": "2025-01-01", "horse_name": "B", "finish_position": 2},
    ]
    out = add_collateral_form_features(_df(rows))
    assert "collateral_next_win_rate" not in out.columns
