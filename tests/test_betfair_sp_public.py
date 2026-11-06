"""The public Betfair SP files must join on the same keys as data/races.db.

Each case below is a trap that fails silently: a summer race that stays on
local time simply never matches, and nothing reports the hole.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mod():
    from src.ingestion import betfair_public
    return betfair_public


NEW_STYLE = (
    "event_id,menu_hint,event_name,event_dt,selection_id,selection_name,win_lose,bsp,ppwap,"
    "morningwap,ppmax,ppmin,ipmax,ipmin,morningtradedvol,pptradedvol,iptradedvol\n"
    "259562118,Ffos Las 30th Jun,7f Hcap,30-06-2026 19:39,6987961,Quilt,0,5.88710793,6.1571,"
    "9.5470,13.75,5.90,1000.00,3.80,1041.38,19870.90,2982.13\n"
)
OLD_STYLE = (
    "EVENT_ID,MENU_HINT,EVENT_NAME,EVENT_DT,SELECTION_ID,SELECTION_NAME,WIN_LOSE,BSP,PPWAP,"
    "MORNINGWAP,PPMAX,PPMIN,IPMAX,IPMIN,MORNINGTRADEDVOL,PPTRADEDVOL,IPTRADEDVOL\n"
    "173003114,GB / Thirsk  14th Sep,7f Hcap,14-09-2020 13:30,8709395,Zylan,0,28.73275303,"
    "30.10,20.38,38,18,1000,19.5,738.14,5614.72,1338.44\n"
    "173003115,GB / Thirsk  14th Dec,6f Hcap,14-12-2020 13:30,8709396,Zylan Two,1,4.5,"
    "4.6,5.0,6,4,1000,1.01,100.0,5000.0,800.0\n"
)


def test_summer_local_time_becomes_utc(mod, tmp_path):
    """The DB holds Quilt's race at 18:39 (UTC); the file says 19:39 BST."""
    path = tmp_path / "f.csv"
    path.write_text(NEW_STYLE, encoding="utf-8")
    row = mod.parse_file(str(path), "uk").iloc[0]
    assert (row.race_date, row.off_time, row.off_key) == ("2026-06-30", "18:39", "18:39")
    assert row.track == "Ffos Las" and row.region == "GB"
    assert row.horse_key == "quilt"
    assert row.bsp == pytest.approx(5.88710793)
    assert row.morningtradedvol == pytest.approx(1041.38)


def test_old_headers_and_country_prefix(mod, tmp_path):
    path = tmp_path / "f.csv"
    path.write_text(OLD_STYLE, encoding="utf-8")
    out = mod.parse_file(str(path), "uk")
    assert list(out.track) == ["Thirsk", "Thirsk"]
    # September is BST (shifted back an hour); December is GMT (unchanged).
    assert list(out.off_time) == ["12:30", "13:30"]
    assert list(out.won) == [0, 1]
