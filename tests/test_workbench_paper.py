"""Workbench Paper record: the per-race field view reads the paper files without touching them."""
import json

import pandas as pd

from src.matchbook import record
from src.workbench.paper import Paper


def _forecast(day):
    return pd.DataFrame({
        "race_id": [1, 1, 2, 2], "horse_id": [10, 11, 20, 21], "race_date": day,
        "p_base": [0.6, 0.2, 0.5, 0.5], "horse_name": ["A", "B", "C", "D"],
        "track": ["Plumpton", "Plumpton", "Wolverhampton", "Wolverhampton"],
        "off_time": ["12:45", "12:45", "17:00", "17:00"], "runner_id": [100, 101, 200, 201],
        "p": [0.7, 0.3, 0.4, 0.6], "share": [0.6, 0.4, 0.5, 0.5], "edge": [0.1, -0.1, -0.1, 0.1],
        "back1": [1.6, 2.4, 2.0, 1.9], "back1_size": [50., 20., 5., 8.], "lay1": [1.7, 2.6, 2.1, 2.0],
        "lay1_size": [10., 10., 10., 10.], "price_0800": [1.8, 2.2, 2.0, 2.0], "price_1100": [1.65, 2.5, 2.05, 1.95],
        "eligible_race": True, "race_tight": [1., 1., .5, .5], "bet": [True, False, False, True],
        "fill_price": [1.6, None, None, 1.9], "fill_stake": [10., 0., 0., 8.]})


def test_day_lists_each_race_with_its_whole_field(tmp_path, monkeypatch):
    day = "2026-11-16"
    folder = tmp_path / day
    folder.mkdir()
    _forecast(day).to_parquet(folder / "forecast.parquet")
    (folder / "run.json").write_text(json.dumps({"win_run": "RUN"}))
    closes = pd.DataFrame({"runner_id": [100, 201], "back1": [1.5, 2.1], "lay1": [1.55, 2.2], "withdrawn": [False, True],
                           "taken_at": pd.to_datetime(["2026-11-16 12:40", "2026-11-16 16:55"]).tz_localize("UTC")})
    monkeypatch.setattr(record, "closes", lambda d: closes)
    out = Paper(tmp_path).day()
    json.dumps(out)
    assert out["date"] == day and out["stage"] == "forecast" and out["win_run"] == "RUN"
    first, second = out["races"]
    assert (first["off_time"], first["runners"], first["bets"]) == ("12:45", 2, 1)
    assert [r["horse_name"] for r in first["field"]] == ["A", "B"]           # by corrected forecast
    assert abs(first["field"][0]["p_model"] - 0.75) < 1e-9                    # win model normalised in race
    assert first["field"][0]["latest_back"] == 1.5 and first["field"][0]["latest_at"] == "12:40"
    assert second["field"][0]["withdrawn_now"] is True and second["tight"] == 0.5


def test_day_before_eleven_shows_prepared_forecast(tmp_path):
    day = "2026-11-17"
    (tmp_path / day).mkdir()
    _forecast(day)[["race_id", "horse_id", "race_date", "p_base", "horse_name", "track", "off_time"]].to_parquet(
        tmp_path / day / "prepared.parquet")
    out = Paper(tmp_path).day(day)
    assert out["stage"] == "prepared" and len(out["races"]) == 2
    assert set(out["races"][0]["field"][0]) == {"horse_name", "p_model"}
    assert Paper(tmp_path).day("2020-01-01")["races"] == []


def test_day_shows_sp_and_result_for_every_runner_once_settled(tmp_path, monkeypatch):
    day = "2026-11-16"
    (tmp_path / day).mkdir()
    _forecast(day).to_parquet(tmp_path / day / "forecast.parquet")
    pd.DataFrame({"race_id": [1, 1, 2, 2], "horse_id": [10, 11, 20, 21], "bsp": [1.7, 2.9, 2.2, 1.8],
                  "won": [0, 1, 0, 1]}).to_parquet(tmp_path / day / "field.parquet")
    pd.DataFrame({"race_id": [1], "horse_id": [10], "bsp": [1.7], "won": [0], "pnl_fill": [-10.], "clv": [0.94]}
                 ).to_parquet(tmp_path / day / "settled.parquet")
    monkeypatch.setattr(record, "closes", lambda d: pd.DataFrame())
    first = Paper(tmp_path).day()["races"][0]["field"]
    assert [(r["horse_name"], r["bsp"], r["won"]) for r in first] == [("A", 1.7, 0), ("B", 2.9, 1)]
    assert first[0]["pnl_fill"] == -10. and first[1]["pnl_fill"] is None
