"""Exercise the saved betting and price-timing views through their HTTP routes."""
# ruff: noqa: F811 -- shared pytest fixtures are injected by name
import json
from urllib.parse import urlencode

import pandas as pd
import pytest

from tests.test_workbench import catalog, http_server, request  # noqa: F401


def test_saved_run_uses_recorded_stakes_and_verified_clv(http_server):
    run_id = "20261112_134514_253097"
    folder = http_server.catalog.root/"runs"/run_id
    folder.mkdir(parents=True)
    (folder/"meta.json").write_text(json.dumps({
        "run_id": run_id, "name": "Saved report",
        "test_analysis": {"stats": {"each_way": {"bets": 2}}, "test_date_range": ["2025-12-31", "2026-01-01"]},
        "metrics": {"win_classifier": {"brier_score": .2, "reliability_bins": [
            None, {"mean_pred": .2, "obs_rate": .25, "count": 8},
        ]}},
        "training_config": {"evals": {"clv": {"by_strategy": {"each_way": {
            "n": 2, "median_clv": 1.1, "geometric_mean_clv": 1.08, "pct_beat_close": .5,
        }}}}},
    }), encoding="utf-8")
    pd.DataFrame([
        {"race_date": "2026-01-01", "race_id": "r1", "strategy": "each_way", "odds": 5, "stake": 2, "pnl": 3, "won": 0, "clv": 42},
        {"race_date": "2025-12-31", "race_id": "r2", "strategy": "each_way", "odds": 4, "stake": 2, "pnl": -2, "won": 0, "clv": 42},
        {"race_date": "2026-01-01", "race_id": "r1", "strategy": "top_pick", "odds": 4, "stake": 1, "pnl": 3, "won": 1, "clv": 42},
    ]).to_csv(folder/"bets.csv", index=False)
    with request(http_server, "/evaluation.js") as response:
        assert "javascript" in response.headers["Content-Type"]
    with request(http_server, "/api/modelling/evaluation") as response:
        assert json.load(response)["runs"] == [{"id": run_id, "name": "Saved report"}]
    url = f"/api/modelling/evaluation/{run_id}?strategy=each_way"
    with request(http_server, url) as response:
        report = json.load(response)
    bets = report["bets"]
    assert bets["summary"] == {"bets": 2, "staked": 4, "pnl": 1, "roi": 25, "strike_rate": 0, "drawdown": 2}
    assert [(q["quarter"], q["roi"]) for q in bets["quarterly"]] == [("2025Q4", -100), ("2026Q1", 150)]
    assert [(band["band"], band["bets"], band["staked"], band["roi"])
            for band in bets["odds_breakdown"] if band["bets"]] == [("3–5", 2, 4, 25)]
    assert bets["clv"] is None and report["clv"]["median_clv"] == 1.1
    assert report["forecast"]["bins"] == [{"predicted": .2, "observed": .25, "runners": 8}]
    assert all("clv" not in row for row in bets["rows"])
    assert bets["rows"][0]["race_name"] is None
    with request(http_server, url+"&"+urlencode({"quarter": "2026Q1", "band": "5–8"})) as response:
        filtered = json.load(response)
    assert filtered["bets"]["summary"]["bets"] == 0
    assert [(band["band"], band["bets"], band["roi"])
            for band in filtered["bets"]["odds_breakdown"] if band["bets"]] == [("3–5", 1, 150)]
    assert filtered["clv"]["n"] == 2
    assert not (http_server.catalog.root/"workbench").exists()


def test_price_timing_changes_settlement_and_keeps_forecast_sample(http_server):
    path = http_server.catalog.root/"research/t60_clv_105/frame.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"race_id": ["r1", "r1"], "horse_id": ["h1", "h2"], "race_date": ["2026-06-01"]*2,
        "won": [1, 0], "p_native": [.6, .4], "p_1100": [.75, .25], "mkt_1100": [.5, .5],
        "ltp_1100": [2., 2.], "rf_after_1100": [1., 1.], "bsp": [1.8, 2.2]}).to_parquet(path, index=False)
    url = "/api/modelling/evaluation/price-timing?cut=1100"
    with request(http_server, url) as response:
        recorded = json.load(response)
    with request(http_server, url+"&ticks=1") as response:
        worse = json.load(response)
    assert recorded["bets"]["summary"]["pnl"] == pytest.approx(.98)
    assert worse["bets"]["summary"]["pnl"] == pytest.approx(.9702)
    bet = worse["bets"]["rows"][0]
    assert bet["bsp"] == 1.8
    assert (bet["horse_name"], bet["track"], bet["race_name"]) == ("Name <script>alert(1)</script>", "Ascot", "Test field")
    assert worse["bets"]["clv"]["median_clv"] < recorded["bets"]["clv"]["median_clv"]
    band = worse["bets"]["odds_breakdown"][0]
    assert band["band"] == "1–3" and band["bets"] == 1
    assert band["clv"] == worse["bets"]["clv"]
    assert worse["forecast"] == recorded["forecast"]
    assert worse["forecast"]["races"] == 1
    assert worse["forecast"]["heads"]["win_classifier"]["race_log_loss"] == pytest.approx(.287682)
    with request(http_server, url+"&ticks=1&race_id=r1") as response:
        detail = json.load(response)["race"]
    assert detail["name"] == "Test field" and detail["track"] == "Ascot"
    assert detail["bets"] == 1 and detail["pnl"] == pytest.approx(.9702)
    first, second = detail["rows"]
    assert first["won"] == 1 and first["horse_id"] == "h1"
    assert [first[k] for k in ("market_price", "simulated_price", "base_probability", "corrected_probability")] == [2., 1.99, .6, .75]
    assert second["horse_id"] == "h2" and second["pnl"] is None
