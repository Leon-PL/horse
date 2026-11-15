"""A class for races that have none (research 100).

Irish races carry no class in the results (92% blank since July 2025; only
Group and Listed races get class 1), so every class input is missing for Irish
runners, and for a British runner whose previous run was in Ireland. Among the
87 selected inputs, ``class_change_vs_field`` is missing for 95% of Irish
runners against 8% of British ones.

The effective class is the published class where there is one. Otherwise it
is read off the race's prize money (in pounds) against British races of the
same code and year: the class whose median log prize is nearest, interpolated
between neighbouring classes, so it is continuous. The scale is the previous
year's British races (the same year only for 2020, the first year).

- ``class_eff``: today's effective class (1 = best, 7 = lowest);
- ``class_change_eff``: today's effective class minus the previous run's
  (positive = dropping in class), the same sign as ``class_change``;
- ``class_change_eff_vs_field``: minus the race mean, as the store's
  ``class_change_vs_field``.
"""
import numpy as np
import pandas as pd

from src.feature_semantics import class_number

STORE_COLUMNS = ["race_id", "horse_id", "race_date", "race_type", "race_class", "prize_money"]
COLUMNS = ["class_eff", "class_change_eff", "class_change_eff_vs_field"]


def effective_class(frame: pd.DataFrame) -> np.ndarray:
    published = class_number(frame.race_class).astype(float)
    year = pd.to_datetime(frame.race_date).dt.year
    log_prize = np.log(pd.to_numeric(frame.prize_money, errors="coerce").where(lambda x: x > 0))
    out = published.to_numpy(dtype=float).copy()
    races = pd.DataFrame({"race_id": frame.race_id, "code": frame.race_type, "year": year, "cls": published,
                          "lp": log_prize}).drop_duplicates("race_id")
    known = races.dropna(subset=["cls", "lp"])
    same = known.groupby(["code", "year", "cls"]).lp.median()
    previous = known.assign(year=known.year + 1).groupby(["code", "year", "cls"]).lp.median()
    missing = np.isnan(out) & np.isfinite(log_prize.to_numpy())
    for (code, yr), idx in pd.Series(np.flatnonzero(missing)).groupby(
            [frame.race_type.to_numpy()[missing], year.to_numpy()[missing]]):
        table = None
        for scale in (previous, same if yr == year.min() else None):
            if scale is not None and (code, yr) in scale.index.droplevel("cls"):
                table = scale.loc[(code, yr)].sort_index()
                break
        if table is None:
            continue
        if len(table) < 2:
            continue
        # Class falls as prize rises: interpolate class against median log prize.
        lp, cls = table.to_numpy()[::-1], table.index.to_numpy(dtype=float)[::-1]
        order = np.argsort(lp)
        rows = idx.to_numpy()
        out[rows] = np.interp(log_prize.to_numpy()[rows], lp[order], cls[order])
    return out


def build(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame[STORE_COLUMNS].copy()
    f["race_date"] = pd.to_datetime(f.race_date)
    f["_row"] = np.arange(len(f))
    f["class_eff"] = effective_class(f)
    f = f.sort_values(["horse_id", "race_date", "race_id"], kind="mergesort")
    f["class_change_eff"] = f.class_eff - f.groupby("horse_id", sort=False).class_eff.shift()
    same_day = f.groupby("horse_id", sort=False).race_date.shift().eq(f.race_date)
    f.loc[same_day, "class_change_eff"] = np.nan
    f = f.sort_values("_row")
    f["class_change_eff_vs_field"] = f.class_change_eff - f.groupby("race_id").class_change_eff.transform("mean")
    return f[["race_id", "horse_id", *COLUMNS]].reset_index(drop=True)
