"""attach_bsp must never replace a column the caller already owns.

It used to assign ``out[c] = merged[c]`` unconditionally, so a caller asking
for ``cols=("won",)`` had its own label swapped for Betfair's in silence. An
audit written that way compares the exchange with itself and always agrees
(ledger 2f). The guard makes that a loud error instead.
"""

import pandas as pd
import pytest

from src.betfair import attach_bsp, bf_horse_key
from src.utils import normalise_off_time_key


def _frames():
    ours = pd.DataFrame({
        "race_date": ["2025-03-01", "2025-03-01"],
        "track": ["Ascot", "Ascot"],
        "off_time": ["14:30", "14:30"],
        "horse_name": ["Alpha", "Bravo"],
        "won": [1, 0],
    })
    betfair = pd.DataFrame({
        "race_date": ["2025-03-01", "2025-03-01"],
        "track": ["Ascot", "Ascot"],
        "off_key": [normalise_off_time_key("14:30")] * 2,
        "horse_key": [bf_horse_key("Alpha"), bf_horse_key("Bravo")],
        "won": [0, 1],  # deliberately disagrees with ours
        "bsp": [3.5, 2.1],
    })
    return ours, betfair


def test_refuses_to_replace_an_existing_column():
    ours, betfair = _frames()
    with pytest.raises(ValueError, match="won"):
        attach_bsp(ours, betfair, cols=("won",))


def test_refuses_even_when_betfair_data_is_missing():
    ours, _ = _frames()
    with pytest.raises(ValueError, match="won"):
        attach_bsp(ours, pd.DataFrame(), cols=("won",))


def test_explicit_overwrite_is_allowed():
    ours, betfair = _frames()
    out = attach_bsp(ours, betfair, cols=("won",), overwrite=True)
    assert out["won"].tolist() == [0, 1]


def test_new_columns_join_and_input_is_untouched():
    ours, betfair = _frames()
    before = ours.copy()
    out = attach_bsp(ours, betfair, cols=("bsp",))
    assert out["bsp"].tolist() == [3.5, 2.1]
    assert "bsp" not in ours.columns
    pd.testing.assert_frame_equal(ours, before)
