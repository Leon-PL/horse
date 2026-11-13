"""The 11:00 exchange price of the race being forecast, as an input.

Every other input here is known the evening before. The user accepts a
forecast made at 11:00 on the day, and the 11:00 Betfair last traded price
covers the whole field in 88% of races back to 2020 (98% in 2026). This is
the first input that reads a price of the race itself, so a model that uses it
cannot forecast before 11:00.

``price-1100``, declared before scoring. From the latest published Betfair
``timing`` table, by its saved links to race and horse:

- the log of the runner's share of the field at 11:00 (1 / price, normalised
  over the starters that have a price, which also absorbs later withdrawals);
- that log share minus the model's log forecast, centred within the race;
- log price relative to the 11:00 favourite, and the rank by price;
- seconds since the runner last traded and the number of price changes so far;
- the move from 08:00 to 11:00, centred within the race;
- the share of the field with an 11:00 price.

Runners without an 11:00 price have every input missing.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from research.meta_learner import screening as s

KEYS = s.KEYS
ROOT = "data/sources/betfair"


def _table(name):
    manifest = json.load(open(f"{ROOT}/latest.json"))
    return pd.read_parquet(sorted(glob.glob(f"{ROOT}/*/{manifest['tables'][name]['file']}"), key=os.path.getmtime)[-1])


def prices():
    """11:00 and 08:00 prices with trading activity, one row per linked runner."""
    timing = _table("timing")[["ltp_0800", "ltp_1100", "age_1100", "nchg_1100"]].reset_index(names="source_row")
    links = _table("links_timing")
    links = links.loc[links.match_status.eq("matched"), ["source_row", "sl_race_id", "sl_horse_id"]]
    out = timing.merge(links, on="source_row", validate="one_to_one")
    out["race_id"] = out.sl_race_id.astype(str)
    out["horse_id"] = pd.to_numeric(out.sl_horse_id).astype("int64")
    out = out.drop(columns=["source_row", "sl_race_id", "sl_horse_id"])
    return out.loc[~out.duplicated(KEYS, keep=False)]


def price_inputs(frame):
    table = prices()
    table["horse_id"] = table.horse_id.astype(frame.horse_id.dtype)
    work = frame[[*KEYS, "p_single"]].merge(table, on=KEYS, how="left", validate="one_to_one")
    race = frame.race_id.to_numpy()
    price = work.ltp_1100.where(work.ltp_1100 > 1)
    inverse = 1 / price
    share = np.log(inverse / inverse.groupby(race).transform("sum"))
    out = frame[KEYS].copy()
    out["pm_logshare"] = share
    gap = share - np.log(work.p_single)
    out["pm_gap"] = gap - gap.groupby(race).transform("mean")
    out["pm_vs_favourite"] = np.log(price / price.groupby(race).transform("min"))
    out["pm_rank"] = price.groupby(race).rank(method="average")
    out["pm_age"] = np.log1p(pd.to_numeric(work.age_1100, errors="coerce").clip(lower=0)).where(price.notna())
    out["pm_changes"] = pd.to_numeric(work.nchg_1100, errors="coerce").where(price.notna())
    move = np.log(work.ltp_0800.where(work.ltp_0800 > 1) / price)
    out["pm_move_0800"] = move - move.groupby(race).transform("mean")
    out["pm_field_priced"] = price.notna().groupby(race).transform("mean").to_numpy()
    columns = [c for c in out if c not in KEYS]
    out[columns] = out[columns].astype("float32")
    return out


def load(frame):
    values = price_inputs(frame)
    return values, [c for c in values if c not in KEYS]
