import numpy as np
import pandas as pd

from research.meta_learner.candidate_features import relatives_form, same_day_bias


def family_history():
    # Horses 1-3 share a dam; horse 4 is unrelated. One race per day, all four run.
    rows = []
    for day in range(6):
        for horse in range(1, 5):
            rows.append({"race_id": day, "horse_id": horse, "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                         "dam": "Mum" if horse < 4 else "Other", "damsire": "Grandad",
                         "won": int(horse == 1 + day % 3), "finish_position": 1 + (horse - 1 - day) % 4,
                         "num_runners": 4, "official_rating": 60. + horse})
    frame = pd.DataFrame(rows)
    frame["won"] = frame.finish_position.eq(1).astype(int)
    return frame


def test_relatives_form_uses_earlier_dates_and_other_offspring_only():
    history = family_history()
    base = relatives_form(history).set_index(["race_id", "horse_id"]).sort_index()
    changed = history.copy()
    later = changed.race_id.eq(3) & changed.horse_id.eq(2)
    changed.loc[later, ["finish_position", "official_rating"]] = [4, 10.]
    revised = relatives_form(changed).set_index(["race_id", "horse_id"]).sort_index()
    pd.testing.assert_frame_equal(base.loc[:3], revised.loc[:3])  # same-day and earlier rows unchanged
    assert not base.loc[4].equals(revised.loc[4])
    first_day = base.loc[0]
    assert (first_day.rel_runs == 0).all() and first_day.rel_finish_pct.isna().all()
    # Horse 1 on day 2 sees its two siblings' four earlier runs, never its own.
    assert base.loc[(2, 1), "rel_runs"] == 4 and base.loc[(2, 1), "rel_horses"] == 2
    assert base.loc[(2, 4), "rel_runs"] == 0  # unrelated dam


def card(offsets):
    rows = []
    for race, off in enumerate(offsets):
        for horse in range(8):
            rows.append({"race_id": race, "horse_id": race * 10 + horse, "race_date": pd.Timestamp("2025-05-01"),
                         "track": "Ascot", "off_time": f"{13 + off // 60:02d}:{off % 60:02d}", "race_type": "Flat",
                         "num_runners": 8, "finish_position": horse + 1, "draw": horse + 1,
                         "pace_style_index": float(horse)})
    return pd.DataFrame(rows)


def test_same_day_bias_ignores_races_off_within_the_lag_or_later():
    history = card([0, 30, 50, 90])
    base = same_day_bias(history).set_index("race_id")
    assert base.loc[0].day_races_before.eq(0).all()
    assert base.loc[2].day_races_before.eq(1).all()  # race at 13:30 is only 20 minutes earlier
    flipped = history.copy()
    rows = flipped.race_id.isin([1, 2])
    flipped.loc[rows, "finish_position"] = 9 - flipped.loc[rows, "finish_position"]
    revised = same_day_bias(flipped).set_index("race_id")
    pd.testing.assert_frame_equal(base.loc[[0, 1, 2]], revised.loc[[0, 1, 2]])
    assert not np.allclose(base.loc[3].day_draw_adv, revised.loc[3].day_draw_adv)
    # Low draws won the first race, so low draws get a positive interaction later.
    later = base.loc[3]
    assert later.day_draw_adv.iloc[0] > 0 and later.day_draw_x_low.iloc[0] > later.day_draw_x_low.iloc[-1]
