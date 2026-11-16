"""Matchbook paper record: feed parsing, runner links, price inputs and fills (no network)."""
import numpy as np
import pandas as pd

from src.matchbook import feed, link, paper
from src.matchbook import model as mm


def _event(start, name, runners):
    return {"id": 1, "name": name, "sport-id": 24735152712200, "start": start, "status": "open",
            "markets": [{"id": 2, "name": "WIN", "status": "open", "volume": 50.0, "runners": runners},
                        {"id": 3, "name": "Place (3)", "runners": []}]}


def _runner(rid, name, back, lay):
    prices = [{"side": "back", "decimal-odds": o, "available-amount": a} for o, a in back]
    prices += [{"side": "lay", "decimal-odds": o, "available-amount": a} for o, a in lay]
    return {"id": rid, "name": name, "status": "open", "withdrawn": False, "volume": 1.0, "prices": prices}


def test_flatten_orders_quotes_and_strips_cloth():
    events = [_event("2026-06-01T13:15:00.000Z", "14:15 Plumpton",
                     [_runner(10, "4 Mysterland", [(1.6, 3), (2.44, 10)], [(6.6, 2), (6.4, 2)])])]
    rows = feed.flatten(events, "uk", pd.Timestamp("2026-06-01 10:00", tz="UTC"))
    r = rows.iloc[0]
    assert (r.horse_name, r.cloth, r.track, r.off_time) == ("Mysterland", 4, "Plumpton", "14:15")
    assert (r.back1, r.back1_size, r.back2, r.lay1, r.lay2) == (2.44, 10, 1.6, 6.4, 6.6)
    assert len(rows) == 1            # place markets are ignored


def test_reference_price_is_middle_of_a_tight_spread_else_back():
    out = feed.reference_price([4.0, 3.0, np.nan, 6.0], [4.4, np.nan, 5.0, 220.])
    assert np.isclose(out[0], np.sqrt(4.0 * 4.4)) and out[1] == 3.0 and np.isnan(out[2]) and out[3] == 6.0


def test_link_pairs_names_across_spellings_and_refuses_ambiguity():
    card = pd.DataFrame({"race_id": [1, 1], "horse_id": [11, 12], "horse_name": ["Blueking D'Oroux (FR)", "Ginolo"],
                         "track": ["Plumpton", "Plumpton"], "off_time": ["14:15", "14:15"]})
    book = pd.DataFrame({"event_id": ["e", "e"], "runner_id": ["r1", "r2"], "horse_name": ["Blueking Doroux", "Ginolo"],
                         "track": ["Plumpton", "Plumpton"], "off_time": ["14:15", "14:15"]})
    linked, report = link.link(card, book)
    assert linked.runner_id.tolist() == ["r1", "r2"] and report.status.tolist() == ["matched"]
    book.loc[1, "horse_name"] = "Blueking D'oroux"   # two book runners now share one loose key
    linked, report = link.link(card, book)
    assert linked.runner_id.tolist()[1] != "r2" or report.status.iloc[0] != "matched"


def test_price_inputs_match_research_definition():
    race = np.array([1, 1, 1, 2, 2])
    p = np.array([.5, .3, .2, .6, .4])
    p11 = np.array([2., 4., 4., 1.5, np.nan])
    p08 = np.array([2.2, 3.5, 4., 1.6, 3.])
    out = mm.price_inputs(race, p, p11, p08)
    assert list(out) == list(mm.PRICE_INPUTS)
    share = (1 / p11[:3]) / (1 / p11[:3]).sum()
    assert np.allclose(out.pm_logshare[:3], np.log(share))
    assert np.isclose(out.pm_gap[:3].mean(), 0, atol=1e-6)
    assert out.pm_rank.tolist()[:3] == [1., 2.5, 2.5]
    assert out.pm_field_priced.tolist() == [1., 1., 1., .5, .5]
    assert np.isnan(out.pm_logshare[4])


def test_fill_walks_the_back_levels():
    row = pd.Series({"back1": 5.0, "back1_size": 4.0, "back2": 4.8, "back2_size": 3.0, "back3": 4.5, "back3_size": 100.})
    price, stake = paper.fill(row, 10.)
    assert stake == 10. and np.isclose(price, (4 * 5 + 3 * 4.8 + 3 * 4.5) / 10)
    price, stake = paper.fill(pd.Series({**row, "back2": np.nan, "back3": np.nan}), 10.)
    assert stake == 4. and price == 5.


def _paper_day(tmp_path, monkeypatch):
    """A settled-ready paper day: two races of three, one bet, and Betfair's SP file for both."""
    from src.matchbook import record
    day = "2026-11-16"
    monkeypatch.setattr(paper, "PAPER", tmp_path)
    folder = tmp_path / day
    folder.mkdir()
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    forecast = pd.DataFrame({
        "race_id": [1, 1, 1, 2, 2, 2], "horse_id": range(10, 16), "race_date": day, "horse_name": names,
        "track": ["Plumpton"] * 3 + ["Wolverhampton"] * 3, "off_time": ["12:45"] * 3 + ["17:00"] * 3,
        "p_base": [.5, .3, .2, .4, .4, .2], "p": [.55, .3, .15, .3, .5, .2], "share": [.5, .3, .2, .4, .4, .2],
        "back1": [2.0, 3.3, 5.0, 2.5, 2.5, 5.0], "back1_size": 20., "lay1": [2.1, 3.5, 5.5, 2.6, 2.6, 5.5],
        "price_0800": 3., "price_1100": [2.05, 3.4, 5.2, 2.55, 2.55, 5.2], "eligible_race": True,
        "race_tight": 1., "runner_id": [f"m{i}" for i in range(6)], "event_id": "e"})
    forecast["edge"] = forecast.p - forecast.share
    forecast["bet"] = forecast.horse_id.isin([10, 14])
    forecast["fill_price"] = np.where(forecast.bet, forecast.back1, np.nan)
    forecast["fill_stake"] = np.where(forecast.bet, 10., 0.)
    forecast.to_parquet(folder / "forecast.parquet")
    forecast.loc[forecast.bet].reset_index(drop=True).to_parquet(folder / "bets.parquet")
    (folder / "run.json").write_text('{"date": "2026-11-16", "status": "forecast"}')
    bf = pd.DataFrame({"event_id": ["9"] * 3 + ["8"] * 3, "selection_id": [f"s{i}" for i in range(6)],
                       "track": forecast.track, "off_time": forecast.off_time, "horse_name": names,
                       "bsp": [1.8, 4.0, 6.0, 2.2, 3.0, 6.5], "won": [1, 0, 0, 0, 1, 0], "morningtradedvol": 100.})
    monkeypatch.setattr(paper, "betfair_day", lambda d: bf)
    monkeypatch.setattr(record, "closes", lambda d: pd.DataFrame())
    return day, folder


def test_settle_keeps_sp_and_result_for_the_whole_field(tmp_path, monkeypatch):
    day, folder = _paper_day(tmp_path, monkeypatch)
    bets = paper.settle(day)
    assert bets.status.eq("settled").all() and bets.won.tolist() == [1, 1]
    field = pd.read_parquet(folder / "field.parquet")
    assert len(field) == 6 and field.bsp.notna().all() and field.won.sum() == 2
    summary = paper.summary()["field"]
    assert summary["races"] == 2 and summary["runners"] == 6
    expected = -(np.log(.55) + np.log(.5)) / 2
    assert abs(summary["log_loss"]["forecast"] - expected) < 1e-9
    assert summary["clv_by_edge"]["forecast_above_share"]["n"] == 2


def test_a_field_fault_never_changes_the_bets(tmp_path, monkeypatch):
    import json
    day, folder = _paper_day(tmp_path, monkeypatch)
    monkeypatch.setattr(paper, "settle_field", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    bets = paper.settle(day)
    assert len(bets) == 2 and (folder / "settled.parquet").exists() and not (folder / "field.parquet").exists()
    run = json.loads((folder / "run.json").read_text())
    assert run["status"] == "settled" and "boom" in run["field_error"]
