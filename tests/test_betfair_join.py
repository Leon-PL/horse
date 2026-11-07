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


def test_default_bsp_cache_is_independent_of_working_directory(tmp_path, monkeypatch):
    from pathlib import Path
    from src import betfair
    assert Path(betfair.DEFAULT_BSP_PATH).is_absolute()
    cache = tmp_path / "cache.parquet"
    expected = pd.DataFrame({"bsp": [2.5, 3.5]})
    expected.to_parquet(cache, index=False)
    monkeypatch.setattr(betfair, "DEFAULT_BSP_PATH", str(cache))
    other = tmp_path / "another-folder"
    other.mkdir()
    monkeypatch.chdir(other)
    pd.testing.assert_frame_equal(betfair.load_bsp(), expected)
    assert betfair.load_bsp(str(tmp_path / "absent.parquet")) is None


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


def test_conflicting_normalised_price_matches_are_unknown():
    ours, betfair = _frames()
    conflict = pd.concat([betfair, betfair.iloc[[0]].assign(bsp=9.)])
    assert pd.isna(attach_bsp(ours, conflict, cols=("bsp",)).bsp.iloc[0])
    identical = pd.concat([betfair, betfair.iloc[[0]]])
    assert attach_bsp(ours, identical, cols=("bsp",)).bsp.tolist() == [3.5, 2.1]


def test_two_left_identities_collapsing_to_one_key_are_unknown():
    ours, betfair = _frames()
    ours = pd.concat([ours, ours.iloc[[0]]], ignore_index=True)
    out = attach_bsp(ours, betfair, cols=("bsp",))
    assert out.bsp.iloc[[0, 2]].isna().all()
    assert out.bsp.iloc[1] == 2.1


@pytest.mark.parametrize("same_price", [False, True])
def test_namesakes_require_exchange_identity_even_when_prices_agree(same_price):
    ours, right = _frames()
    ours["horse_name"] = "Sierra Nevada"
    ours["horse_id"] = [1040489, 1038101]
    right["horse_key"] = bf_horse_key("Sierra Nevada")
    right["bf_runner_id"] = [45802885, 620329]
    if same_price:
        right["bsp"] = 3.5
    assert attach_bsp(ours, right, cols=("bsp",)).bsp.isna().all()
    assert pd.isna(attach_bsp(ours.iloc[:1], right, cols=("bsp",)).bsp.iloc[0])
    ours["bf_runner_id"] = [45802885., 620329.]
    expected = [3.5, 3.5 if same_price else 2.1]
    assert attach_bsp(ours, right.iloc[::-1], cols=("bsp",)).bsp.tolist() == expected
    assert ours.horse_id.tolist() == [1040489, 1038101]


def test_same_name_different_known_exchange_selection_is_not_a_fallback():
    ours, right = _frames()
    ours["bf_runner_id"] = [99, 2]
    right["bf_runner_id"] = [1, 2]
    result = attach_bsp(ours, right, cols=("bsp",))
    assert pd.isna(result.bsp.iloc[0]) and result.bsp.iloc[1] == 2.1
