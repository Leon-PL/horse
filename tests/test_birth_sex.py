"""Sex must be the sex at birth: the stored code is scrape-time, not race day.

Results pages report a horse's sex as of the day they were scraped, so the
code encodes the horse's future - kept entire, gelded later, raced on past
four. Two-year-olds coded as mares returned +21.5% at BSP (ledger 15).
"""

import pandas as pd

from src.data_processor import _BIRTH_SEX, clean_data, encode_categorical


def test_every_code_collapses_to_birth_sex():
    assert set(_BIRTH_SEX.values()) == {"Female", "Male"}
    for code in ("F", "M", "Filly", "Mare"):
        assert _BIRTH_SEX[code] == "Female"
    for code in ("C", "G", "H", "R", "Colt", "Gelding", "Horse", "Rig"):
        assert _BIRTH_SEX[code] == "Male"


def test_processed_frame_has_no_race_day_sex_categories():
    raw = pd.DataFrame({
        "race_id": ["r1"] * 4, "race_date": ["2021-06-01"] * 4,
        "horse_name": ["A", "B", "C", "D"], "jockey": ["j"] * 4, "trainer": ["t"] * 4,
        "track": ["Ascot"] * 4, "going": ["Good"] * 4, "race_class": ["1"] * 4,
        "race_type": ["Flat"] * 4, "sex": ["m", "h", "f", "g"], "age": [2, 2, 3, 4],
        "finish_position": [1, 2, 3, 4], "won": [1, 0, 0, 0],
    })
    out = encode_categorical(clean_data(raw))
    assert set(out["sex"].dropna()) <= {"Female", "Male"}
    sex_dummies = sorted(c for c in out.columns if c.startswith("sex_"))
    assert sex_dummies == ["sex_Female", "sex_Male"]
