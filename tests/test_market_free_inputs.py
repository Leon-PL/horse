"""No model input may depend on the current race's own price.

Market information enters only through the optional market anchor
(``src/model.py``, ``ODDS_DERIVED_COLUMNS``). A column built from the
current race's odds that is missing from that list is silently eligible for
selection; ``jockey_booking_vs_market`` was such a column. Prior-day market
features are allowed: they cannot see the final day's prices.
"""

import numpy as np
import pandas as pd
import pytest

import config
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.model import get_feature_columns
from tests.test_live_parity import _synthetic_raw


@pytest.fixture(autouse=True)
def every_eligible_feature(monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)


def test_no_eligible_input_reads_the_current_race_price():
    raw = _synthetic_raw(n_days=20)
    last = raw.race_date == raw.race_date.max()
    repriced = raw.copy()
    # Reverse the price order inside every final-day race.
    repriced.loc[last, "odds"] = (
        repriced.loc[last].groupby("race_id")["odds"].transform(lambda s: s.to_numpy()[::-1])
    )
    assert not np.array_equal(raw.loc[last, "odds"], repriced.loc[last, "odds"])

    def final_day(frame):
        featured = engineer_features(process_data(frame), save=False)
        day = featured[featured.race_date == featured.race_date.max()]
        return day.sort_values(["race_id", "horse_id"]).reset_index(drop=True)

    before, after = final_day(raw), final_day(repriced)
    columns = get_feature_columns(before)
    changed = [
        c for c in columns
        if not np.allclose(
            pd.to_numeric(before[c], errors="coerce").to_numpy(float),
            pd.to_numeric(after[c], errors="coerce").to_numpy(float),
            equal_nan=True,
        )
    ]
    assert changed == [], f"inputs built from the current race's price: {changed}"
