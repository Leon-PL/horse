from copy import deepcopy
import numpy as np
import pandas as pd

from src.data_processor import process_data
from src.feature_engineer import (
    _PRIOR_MARKET_STATE_KEY, add_prior_market_features, counts_state_scope, engineer_features,
)
from src.prior_market_features import FEATURES, prior_market_features
from src.result_status import pending_card
from tests.test_live_parity import _synthetic_raw


def test_ordinary_pipeline_emits_same_history_and_carries_live_seed():
    raw = _synthetic_raw(n_days=5,races_per_day=1,field=4)
    processed = process_data(raw)
    state = {}
    full = engineer_features(processed.copy(),save=False,counts_state=state)
    expected = prior_market_features(processed)
    keys = ["race_id","horse_name"]
    a = pd.concat([processed[keys],expected],axis=1).set_index(keys).sort_index()
    b = full.set_index(keys).sort_index()
    np.testing.assert_allclose(a[list(FEATURES)],b[list(FEATURES)],equal_nan=True,rtol=0,atol=1e-12)
    assert _PRIOR_MARKET_STATE_KEY in state
    split = processed.race_date.max()
    _, seed = prior_market_features(processed.loc[processed.race_date.lt(split)],return_state=True)
    card = pending_card(processed.loc[processed.race_date.eq(split)]).assign(odds=np.nan)
    copied = {_PRIOR_MARKET_STATE_KEY:deepcopy(seed)}
    with counts_state_scope(copied):
        live = add_prior_market_features(card.copy())
    expected_live = expected.loc[processed.race_date.eq(split)]
    np.testing.assert_allclose(live[list(FEATURES)],expected_live[list(FEATURES)],equal_nan=True,rtol=0,atol=1e-12)


def test_minimal_unpriced_unrated_card_preserves_unknowns_and_original_columns():
    frame = pd.DataFrame({"race_id":[1,1],"race_date":pd.to_datetime(["2025-01-01"]*2),
        "horse_name":["A","B"],"race_type":["Flat"]*2,"surface":["Turf"]*2})
    out = add_prior_market_features(frame.copy())
    assert out[list(FEATURES[:2])].isna().all().all()
    assert out[FEATURES[2]].eq(0).all()
    assert set(out)==set(frame)|set(FEATURES)
