"""Regression: ``trainer_freq`` must not leak same-race finish order.

A trainer with several runners in one race used to get consecutive cumcounts in
finishing-position order; that ordering leaked the result into ``trainer_freq``
and its race-relative derivatives (``trainer_freq_rank_pct_in_race``,
``trainer_freq_vs_best_in_race``) — the +56.8% ROI@BSP stablemate leak
(2026-06-26). Every stablemate must see one shared pre-race count.

Since 2026-11-07 (research 065) the count is made in the trainer stage on the
carried counts state, so an append or a live card continues it instead of
restarting at 1. The horse, jockey and track counts were dropped: no feature
used them.
"""
import numpy as np
import pandas as pd

from src.features.common import _add_result_history_flags
from src.features.connections import add_trainer_features
from src.features.history import counts_state_scope


def _prepared(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["track"] = df.get("track", "Ascot")
    df["num_runners"] = df.groupby("race_id")["race_id"].transform("size")
    df["finish_position"] = df.groupby("race_id").cumcount() + 1
    df["won"] = (df["finish_position"] == 1).astype(int)
    return _add_result_history_flags(df)


def _df():
    # Trainer T runs THREE horses in R2 (rows in finishing order: winner first).
    return _prepared([
        {"race_id": "R1", "race_date": "2020-01-01", "trainer": "T"},
        {"race_id": "R2", "race_date": "2020-01-02", "trainer": "T"},
        {"race_id": "R2", "race_date": "2020-01-02", "trainer": "T"},
        {"race_id": "R2", "race_date": "2020-01-02", "trainer": "T"},
        {"race_id": "R3", "race_date": "2020-01-03", "trainer": "T"},
    ])


def test_trainer_freq_shared_across_stablemates():
    out = add_trainer_features(_df())
    # One pre-race count per (race, trainer): 1 entry before R2, 4 before R3.
    assert out["trainer_freq"].tolist() == [1, 2, 2, 2, 5]


def test_no_within_block_finish_correlation():
    # Many multi-runner-trainer races; trainer_freq must be constant within each
    # (race, trainer) block, so it cannot correlate with who won.
    rng = np.random.default_rng(0)
    rows = []
    for r in range(200):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=r)
        for tr in rng.choice(["A", "B", "C"], size=rng.integers(2, 4)):
            rows.append({"race_id": f"R{r:03d}", "race_date": date, "trainer": tr})
    out = add_trainer_features(_prepared(rows))
    g = out.groupby(["race_id", "trainer"])["trainer_freq"].nunique()
    assert (g <= 1).all()


def test_carried_count_continues_across_an_append():
    rng = np.random.default_rng(1)
    rows = []
    for r in range(60):
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=r)
        for tr in rng.choice(["A", "B", "C", "D"], size=rng.integers(2, 6)):
            rows.append({"race_id": f"R{r:03d}", "race_date": date, "trainer": tr})
    frame = _prepared(rows)
    full = add_trainer_features(frame.copy())["trainer_freq"].to_numpy()

    cut = frame["race_id"] < "R040"
    state: dict = {}
    with counts_state_scope(state):
        add_trainer_features(frame[cut].copy())
    with counts_state_scope(state):
        appended = add_trainer_features(frame[~cut].copy())["trainer_freq"].to_numpy()
    np.testing.assert_array_equal(appended, full[~cut.to_numpy()])
