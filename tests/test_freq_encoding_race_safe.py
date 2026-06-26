"""Regression: `*_freq` frequency encodings must not leak same-race finish order.

A trainer with several runners in one race produces consecutive cumcounts in
finishing-position order; if not collapsed per (race, trainer) that ordering
leaks the result into `trainer_freq` and its race-relative derivatives
(`trainer_freq_rank_pct_in_race`, `trainer_freq_vs_best_in_race`) — the +56.8%
ROI@BSP stablemate leak (2026-06-26). encode_categorical must give every
stablemate one shared pre-race count.
"""
import pandas as pd

from src.data_processor import encode_categorical


def _df():
    # Trainer T runs THREE horses in R2 (rows in finishing order: winner first).
    return pd.DataFrame({
        "race_id": ["R1", "R2", "R2", "R2", "R3"],
        "race_date": pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-02", "2020-01-03"]),
        "trainer": ["T", "T", "T", "T", "T"],
        "jockey": ["J1", "J2", "J3", "J4", "J5"],
        "horse_name": ["h1", "h2", "h3", "h4", "h5"],
    })


def test_trainer_freq_shared_across_stablemates():
    out = encode_categorical(_df())
    r2 = out.loc[out["race_id"] == "R2", "trainer_freq"]
    # all three stablemates must see ONE identical pre-race count (no within-race
    # ordering that tracks who finished where).
    assert r2.nunique() == 1


def test_once_per_race_entities_unaffected():
    out = encode_categorical(_df())
    # jockey/horse appear once per race -> collapse is a no-op (still distinct
    # global cumcounts), so signal is preserved.
    assert out["jockey_freq"].tolist() == [1, 1, 1, 1, 1]
    assert out["horse_name_freq"].tolist() == [1, 1, 1, 1, 1]


def test_no_within_block_finish_correlation():
    # Many multi-runner-trainer races; trainer_freq must be constant within each
    # (race, trainer) block, so it cannot correlate with who won.
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for r in range(200):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=r)
        for tr in rng.choice(["A", "B", "C"], size=rng.integers(2, 4)):
            rows.append({"race_id": f"R{r}", "race_date": date, "trainer": tr,
                         "jockey": f"j{rng.integers(0,50)}",
                         "horse_name": f"h{len(rows)}"})
    out = encode_categorical(pd.DataFrame(rows))
    g = out.groupby(["race_id", "trainer"])["trainer_freq"].nunique()
    assert (g <= 1).all()
