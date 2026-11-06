import pytest

from src.ingestion.sporting_life import _infer_race_type as infer_race_type


@pytest.mark.parametrize("name,expected", [
    ("Purchase Hospitality At Huntingdon Mares' Novices' Handicap Hurdle", "Hurdle"),
    ("Purchase By 3pm Today To WINABMW.ie Mares Maiden Hurdle", "Hurdle"),
    ("JenningsBet Chipchase Stakes (Group 3)", "Flat"),
    ("Chasemore Farm Brigadier Gerard Stakes (Group 3)", "Flat"),
    ("Vintage Acquisitions Whisky Chaser Handicap", "Flat"),
    ("'Peterborough Chase Day' Sunday 5th December Novices' Hurdle (GBB Race)", "Hurdle"),
    ("Book For Bobbyjo Chase Day 26th February Mares Maiden Hurdle (Div 2)", "Hurdle"),
    ("No Hurdles With Emplas Jump Jockeys Derby Handicap", "Flat"),
    ("Troytown Chase Day Early Bird Tickets On Sale Now Handicap", "Flat"),
    ("Purchase A Ticket Standard Open NH Flat Race", "Flat"),
    ("Ladbrokes Punchestown Gold Cup (Grade 1)", "Chase"),
    ("Paddy Power Irish Gold Cup (Grade 1)", "Chase"),
    ("Ascot Gold Cup (Group 1)", "Flat"),
    ("Gold Cup Handicap", "Flat"),
    ("Novices' Handicap Chase (GBB Race)", "Chase"),
    ("Connolly's RED MILLS Auction Hurdle Series Final", "Hurdle"),
    ("Champion I.N.H. Flat Race (Grade 1)", "Flat"),
    ("Castle Rock Harvest Pale Chase Maiden Stakes (GBB Race)", "Flat"),
    ("Chase The Dream At ctdracing.co.uk EBF Fillies' Restricted Novice Stakes (GBB Race)", "Flat"),
    ("West Chase British EBF Maiden Stakes (GBB Race)", "Flat"),
    ("Chase Distillery EBF Fillies' Handicap", "Flat"),
    ("Cruises Chase Handicap", "Flat"),
    ("Hullabaloos Chase Handicap", "Flat"),
    ("Flat Jockeys Can Jump Handicap Hurdle (To Be Ridden By Professional Flat Race Jockeys)", "Hurdle"),
    ("Guinness Galway Hurdle Handicap (Grade 3)", "Hurdle"),
    ("Chase Distillery Novices' Handicap Chase (GBB Race)", "Chase"),
])
def test_real_title_semantics(name, expected):
    assert infer_race_type(name) == expected


@pytest.mark.parametrize("code,expected", [("CHASE", "Chase"), ("HURDL", "Hurdle"),
    ("NH FLAT", "Flat"), ("N.H. FLAT", "Flat"), ("AW FLAT", "Flat")])
def test_structured_source_code_takes_precedence(code, expected):
    assert infer_race_type("Sponsor Hurdle Chase Bumper", code) == expected


def test_unknown_source_code_does_not_hide_a_clear_title():
    assert infer_race_type("Mares' Novices' Hurdle", "UNAVAILABLE") == "Hurdle"
