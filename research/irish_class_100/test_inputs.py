import numpy as np
import pandas as pd

from research.irish_class_100.inputs import build


def test_unclassed_race_gets_class_from_previous_year_prizes():
    rows = []
    for year, scale in ((2023, 1.0), (2024, 100.0)):   # 2024 prizes are much larger: must not be used for 2024
        for cls, prize in ((2, 20000), (4, 5000), (6, 2000)):
            rows.append((f"{year}{cls}", 1, f"{year}-03-0{cls}", "Flat", cls, prize * scale))
    rows.append(("x", 2, "2024-05-01", "Flat", None, 5000.0))     # Irish race, 2023's class 4 prize
    rows.append(("y", 2, "2024-06-01", "Flat", 2.0, 1.0))
    f = pd.DataFrame(rows, columns=["race_id", "horse_id", "race_date", "race_type", "race_class", "prize_money"])
    out = build(f).set_index("race_id")
    assert np.isclose(out.loc["x", "class_eff"], 4.0)
    assert np.isclose(out.loc["y", "class_change_eff"], 2.0 - 4.0)
