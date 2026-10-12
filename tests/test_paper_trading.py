"""Tests for the paper-trading selection + settlement maths (src/paper_trading.py)."""

import numpy as np
import pandas as pd
import pytest

from src.paper_trading import score_and_select, settle, log_selections, MARKET_FEATS


class _ConstModel:
    """predict_proba returns a constant P(place) so selection is driven by odds."""
    def __init__(self, p): self.p = p
    def predict_proba(self, X):
        n = len(X); return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _featured():
    return pd.DataFrame({
        "race_date": ["2026-06-23"] * 3,
        "track": ["Ascot"] * 3, "off_time": ["14:30"] * 3,
        "horse_name": ["Alpha", "Bravo", "Charlie"],
        "num_runners": [8, 8, 8], "race_type": ["Hurdle"] * 3,
    })


def _mb():
    return pd.DataFrame({
        "track": ["Ascot"] * 3, "off_time": ["14:30"] * 3,
        "horse_name": ["Alpha", "Bravo", "Charlie"],
        "mb_win_odds": [3.0, 6.0, 4.5],
        "mb_place_odds": [4.0, 2.0, 3.0],          # ev@p=0.4 -> +0.60, -0.20, +0.20
        "mb_place_available": [50.0, 120.0, 80.0],
        "places_paid": [3, 3, 3],
    })


def test_selects_only_positive_ev():
    sel = score_and_select(_featured(), _mb(), _ConstModel(0.4), MARKET_FEATS, ev_threshold=0.10)
    assert set(sel["horse_name"]) == {"Alpha", "Charlie"}      # Bravo (-0.20) excluded
    assert (sel["ev"] > 0.10).all()
    # captured the live price + available size for later settlement / liquidity
    assert sel.loc[sel["horse_name"] == "Alpha", "mb_place_available"].iloc[0] == 50.0


def test_threshold_tightens_selection():
    sel = score_and_select(_featured(), _mb(), _ConstModel(0.4), MARKET_FEATS, ev_threshold=0.50)
    assert set(sel["horse_name"]) == {"Alpha"}                 # only +0.60 clears 0.50


def test_jumps_only_filter():
    feat = _featured(); feat["race_type"] = "Flat"
    sel = score_and_select(feat, _mb(), _ConstModel(0.4), MARKET_FEATS, ev_threshold=0.10, jumps_only=True)
    assert sel.empty


def test_no_price_match_returns_empty():
    mb = _mb(); mb["horse_name"] = ["X", "Y", "Z"]             # no key overlap
    sel = score_and_select(_featured(), mb, _ConstModel(0.4), MARKET_FEATS)
    assert sel.empty


def test_settle_books_pnl(tmp_path):
    ledger = str(tmp_path / "ledger.csv")
    sel = score_and_select(_featured(), _mb(), _ConstModel(0.4), MARKET_FEATS, ev_threshold=0.10)
    log_selections(sel, ledger)
    # Alpha places (2nd of 3 paid), Charlie does not (5th)
    results = pd.DataFrame({
        "race_date": ["2026-06-23", "2026-06-23"], "track": ["Ascot", "Ascot"],
        "off_time": ["14:30", "14:30"], "horse_name": ["Alpha", "Charlie"],
        "finish_position": [2, 5],
    })
    summ = settle(results, ledger_path=ledger, commission=0.02)
    assert summ["total_settled"] == 2 and summ["placed"] == 1
    # Alpha: (4-1)*0.98 = 2.94 ; Charlie: -1 -> net 1.94
    assert summ["pnl"] == pytest.approx(1.94, abs=1e-6)
    led = pd.read_csv(ledger)
    assert int(led.loc[led["horse_name"] == "Alpha", "placed"].iloc[0]) == 1
    assert int(led.loc[led["horse_name"] == "Charlie", "placed"].iloc[0]) == 0


def test_settle_is_idempotent(tmp_path):
    ledger = str(tmp_path / "l.csv")
    log_selections(score_and_select(_featured(), _mb(), _ConstModel(0.4), MARKET_FEATS), ledger)
    results = pd.DataFrame({"race_date": ["2026-06-23"], "track": ["Ascot"],
                            "off_time": ["14:30"], "horse_name": ["Alpha"], "finish_position": [2]})
    s1 = settle(results, ledger_path=ledger)
    s2 = settle(results, ledger_path=ledger)            # re-settling must not double-count
    assert s2["newly_settled"] == 0
    assert s1["pnl"] == s2["pnl"]


def test_multiday_prices_and_market_normalisation_are_date_specific():
    today = _featured().iloc[:2].copy()
    tomorrow = today.copy()
    tomorrow["race_date"] = "2026-06-24"
    first_prices = _mb().iloc[:2].copy()
    first_prices["race_date"] = "2026-06-23"
    first_prices["mb_win_odds"] = [2.0, 4.0]
    second_prices = first_prices.copy()
    second_prices["race_date"] = "2026-06-24"
    second_prices["mb_win_odds"] = [8.0, 2.0]
    second_prices["mb_place_odds"] = [5.0, 6.0]

    class CapturingModel(_ConstModel):
        def predict_proba(self, X):
            self.scored = X.copy()
            return super().predict_proba(X)

    model = CapturingModel(0.8)
    result = score_and_select(
        pd.concat([today, tomorrow], ignore_index=True),
        pd.concat([first_prices, second_prices], ignore_index=True),
        model, MARKET_FEATS, ev_threshold=0.0,
    )
    assert len(result) == 4
    assert result.set_index(["race_date", "horse_name"]).loc[("2026-06-24", "Alpha"), "mb_place_odds"] == 5.0
    np.testing.assert_allclose(model.scored["mkt_win_prob"], [2/3, 1/3, 0.2, 0.8])
    assert model.scored["win_ltp300"].tolist() == [2.0, 4.0, 8.0, 2.0]


def test_multiday_selection_rejects_undated_market_prices():
    first = _featured()
    second = first.copy()
    second["race_date"] = "2026-06-24"
    with pytest.raises(ValueError, match="require race_date"):
        score_and_select(pd.concat([first, second]), _mb(), _ConstModel(0.8), MARKET_FEATS)
