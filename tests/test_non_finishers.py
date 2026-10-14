"""Non-finishers (position 0) are losers, never missing (ledger 18).

Training and evaluation once dropped every runner with finish_position 0
(fell, pulled up, unseated): 24% of chase and 15% of hurdle runners. The test
set then held only horses known to have completed, so each-way bets showed
+79% at a closing-line value of 0.78. These tests keep them in, as losers that
never place.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src.bet_settlement import ew_placed_flag, placed_mask
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import prepare_multi_target_data, rps_per_race
from tests.test_live_parity import _synthetic_raw


def test_placed_mask_never_places_a_non_finisher():
    fp = np.array([0, 1, 2, 3, 4, np.nan])
    assert placed_mask(fp, 3).tolist() == [0, 1, 1, 1, 0, 0]
    assert placed_mask(fp, np.array([2, 2, 2, 2, 2, 2])).tolist() == [0, 1, 1, 0, 0, 0]
    # The vectorised rule agrees with the scalar settlement rule.
    for p in (0, 1, 3, 4):
        assert placed_mask(np.array([p]), 3)[0] == ew_placed_flag(p, 3)


def test_rps_orders_non_finishers_after_finishers():
    # A faller given 0.8 and the winner 0.2: the winner comes first in the
    # ranking, so F_1 = 0.2 and RPS = (0.2 - 1)^2 = 0.64. Sorting position 0
    # first (as if it won) would score 0.04.
    probs = np.array([0.8, 0.2])
    assert rps_per_race(probs, np.array([0, 1]), np.array([2])) == pytest.approx(0.64)


@pytest.fixture(scope="module")
def featured_with_fallers():
    raw = _synthetic_raw(n_days=320, races_per_day=2, field=6, seed=11)
    # Two non-winners in every jumps race fail to finish.
    jumps = raw["race_type"].ne("Flat")
    losers = raw["finish_position"].isin([5, 6]) & jumps
    raw.loc[losers, "finish_position"] = 0
    raw.loc[losers, "lengths_behind"] = np.nan
    return engineer_features(process_data(raw, save=False), save=False), int(losers.sum())


def test_training_frame_keeps_non_finishers_as_unplaced_losers(featured_with_fallers, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    featured, n_fallers = featured_with_fallers
    data = prepare_multi_target_data(featured)
    fp = np.concatenate([data["fp_train"], data["fp_test"]])
    # Burn-in trims the start of the history; within the races it keeps,
    # every non-finisher must still be there.
    kept_races = set(data["train_df"]["race_id"]) | set(data["test_df"]["race_id"])
    expected = int(((featured["finish_position"] == 0) & featured["race_id"].isin(kept_races)).sum())
    assert 0 < expected <= n_fallers
    assert int((fp == 0).sum()) == expected
    for part in ("train", "test"):
        part_dnf = data[f"fp_{part}"] == 0
        assert part_dnf.any()
        assert (data[f"y_{part}_placed"][part_dnf] == 0).all()
        assert (data[f"y_{part}_won"][part_dnf] == 0).all()
        assert (data[f"y_{part}_norm_pos"][part_dnf] == 1.0).all()
        assert (data[f"y_{part}_rel"][part_dnf] == 0).all()


def test_walk_forward_counts_a_bet_on_a_faller_as_a_loss(monkeypatch):
    import src.backtester as bt

    days = pd.date_range("2025-01-01", "2025-03-31")
    frame = pd.DataFrame({
        "race_id": np.repeat(np.arange(len(days)), 2), "race_date": np.repeat(days, 2),
        "horse_name": np.tile(["faller", "winner"], len(days)),
        "finish_position": np.tile([0, 1], len(days)), "won": np.tile([0, 1], len(days)),
        "num_runners": 2, "odds": np.tile([3.0, 3.0], len(days)), "feature": 1.0,
    })

    class Predictor:
        def __init__(self, **kwargs):
            pass

        def train_on_fold(self, X_train, X_test, *args, **kwargs):
            # The model always prefers the horse that will fall.
            return np.where(kwargs["test_df"]["horse_name"] == "faller", 0.7, 0.3), None

    monkeypatch.setattr(bt, "RacePredictor", Predictor)
    monkeypatch.setattr(bt, "get_feature_columns", lambda df: ["feature"])
    result = bt.walk_forward_validation(frame, min_train_months=2, value_threshold=0.0)
    bets = result["bets"]
    top = bets[bets["strategy"] == "top_pick"]
    assert not top.empty, "the faller's race vanished from the evaluation"
    assert (top["horse_name"] == "faller").all()
    assert (top["pnl"] < 0).all()
