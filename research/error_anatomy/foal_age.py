"""Exact age from the foaling date.

The store holds a foaling date for 99.8% of runners and the model uses only
age in whole years. Among two- and three-year-olds a January foal can be four
months older than a May foal in the same race, which matters most where the
model is weakest: races of unraced and lightly raced young horses.

``foal-age``, declared before scoring: age in days at the race, the same
against the field's mean and its oldest runner, the foaling day of the year,
and the first two masked to runners aged two and three (missing for older
horses, where a few months cannot matter).

RESULT: a look-ahead leak, not an input. It scored -0.0015 alone and -0.0016
inside the joint correction at native capacity, but 22% of horses carry a
placeholder foaling date of 1 January. Every profile was fetched in 2026, the
placeholder is commoner the shorter the career (35% of one-run horses, 11% of
those with 20+ runs), and placeholder horses win 7-23% less often than BSP
implies. The date's precision encodes how the career went. Kept as a record;
do not build inputs from ``foaled``.
"""
import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS


def foal_age(frame, foaled, age):
    foaled = pd.to_datetime(pd.Series(np.asarray(foaled)), errors="coerce")
    days = (frame.race_date.reset_index(drop=True) - foaled).dt.days.astype(float)
    days = days.where(days.between(500, 8000))
    race = frame.race_id.to_numpy()
    out = frame[KEYS].reset_index(drop=True).copy()
    out["fa_days"] = days
    out["fa_days_vs_field"] = days - days.groupby(race).transform("mean")
    out["fa_days_vs_oldest"] = days - days.groupby(race).transform("max")
    out["fa_foal_day"] = foaled.dt.dayofyear.astype(float)
    young = pd.to_numeric(pd.Series(np.asarray(age)), errors="coerce").le(3)
    out["fa_young_days"] = days.where(young)
    out["fa_young_vs_field"] = out.fa_days_vs_field.where(young)
    columns = [c for c in out if c not in KEYS]
    out[columns] = out[columns].astype("float32")
    return out


def load(frame):
    store = sc.read_store(["foaled", "age"], start=frame.race_date.min()).drop(columns="race_date")
    store["race_id"] = store.race_id.astype(frame.race_id.dtype)
    work = frame[KEYS].merge(store, on=KEYS, how="left", validate="one_to_one")
    values = foal_age(frame, work.foaled.to_numpy(), work.age.to_numpy())
    return values, [c for c in values if c not in KEYS]


CLEAN_STORE = "data/processed/feature_store/featured.parquet"


def load_clean(frame):
    """``foal-age-clean``, declared 13 November before scoring.

    The rebuilt store takes foaling dates from Racing TV: no 1 January placeholders, but 11% of
    horses have no date and those win 12% less often than BSP implies, so a missing value still
    tells one runner from another. Here the inputs are given only in races where every runner has
    a date and are missing for the whole field otherwise; a value that is constant within a race
    cannot separate its runners.
    """
    store = pd.read_parquet(CLEAN_STORE, columns=[*KEYS, "foaled", "age"])
    store["race_id"] = store.race_id.astype(str)  # the stored forecasts key races by string
    store["horse_id"] = store.horse_id.astype(frame.horse_id.dtype)
    work = frame[KEYS].merge(store, on=KEYS, how="left", validate="one_to_one")
    values = foal_age(frame, work.foaled.to_numpy(), work.age.to_numpy())
    columns = [c for c in values if c not in KEYS]
    whole = values.fa_days.notna().groupby(frame.race_id.to_numpy()).transform("all").to_numpy()
    values.loc[~whole, columns] = np.nan
    print(f"foal-age-clean: {whole.mean():.3f} of runners are in fully dated races", flush=True)
    return values, columns
