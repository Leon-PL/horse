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


# --------------------------------------------------------------------------
# Price timing (declared 14 November before scoring; ERROR_ANATOMY next step 5).
# The 11:00 price is a different distance from the off in every race. Three arms, each judged against ``with-1100``:
#   ``price-1100-hours``: the eight ``price-1100`` inputs plus hours from 11:00 to the off (the move is from 08:00);
#   ``price-3600``: the same inputs one hour before the off, the main arm;
#   ``price-300``: the same inputs five minutes before the off, an upper reference (nearly the close).
# For the two fixed-offset arms the move is from the runner's first recorded price (18:00 the evening before, else
# 08:00, else 11:00 when 11:00 is more than an hour before the cut) to the cut, centred within the race, plus the
# race's median hours over which that move happened.


def timed_prices():
    """Prices at 08:00, 11:00 and one hour before the off, with activity and hours from 11:00 to the off."""
    timing = _table("timing")[["race_date", "off_ms", "ltp_eve1800", "ltp_0800", "ltp_1100", "age_1100",
                               "nchg_1100", "ltp_3600s", "age_3600s", "nchg_3600s", "ltp_300s", "age_300s",
                               "nchg_300s"]].reset_index(names="source_row")
    eleven = (pd.to_datetime(timing.race_date) + pd.Timedelta(hours=11)).dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    off = pd.to_datetime(timing.off_ms, unit="ms", utc=True)
    timing["hours_1100_to_off"] = (off - eleven).dt.total_seconds() / 3600.
    links = _table("links_timing")
    links = links.loc[links.match_status.eq("matched"), ["source_row", "sl_race_id", "sl_horse_id"]]
    out = timing.drop(columns=["race_date", "off_ms"]).merge(links, on="source_row", validate="one_to_one")
    out["race_id"] = out.sl_race_id.astype(str)
    out["horse_id"] = pd.to_numeric(out.sl_horse_id).astype("int64")
    out = out.drop(columns=["source_row", "sl_race_id", "sl_horse_id"])
    return out.loc[~out.duplicated(KEYS, keep=False)]


def _inputs_at(frame, work, when):
    race = frame.race_id.to_numpy()
    price = work[f"ltp_{when}"].where(work[f"ltp_{when}"] > 1)
    inverse = 1 / price
    share = np.log(inverse / inverse.groupby(race).transform("sum"))
    out = frame[KEYS].copy()
    out["pm_logshare"] = share
    gap = share - np.log(work.p_single)
    out["pm_gap"] = gap - gap.groupby(race).transform("mean")
    out["pm_vs_favourite"] = np.log(price / price.groupby(race).transform("min"))
    out["pm_rank"] = price.groupby(race).rank(method="average")
    out["pm_age"] = np.log1p(pd.to_numeric(work[f"age_{when}"], errors="coerce").clip(lower=0)).where(price.notna())
    out["pm_changes"] = pd.to_numeric(work[f"nchg_{when}"], errors="coerce").where(price.notna())
    if when == "1100":
        move = np.log(work.ltp_0800.where(work.ltp_0800 > 1) / price)
        out["pm_move_0800"] = move - move.groupby(race).transform("mean")
    else:
        cut_hours = work.hours_1100_to_off - (1. if when == "3600s" else 5 / 60)   # hours from 11:00 to the cut
        h = work.hours_1100_to_off
        candidates = [("ltp_eve1800", -17.), ("ltp_0800", -3.), ("ltp_1100", 0.)]   # hours relative to 11:00
        first = pd.Series(np.nan, index=work.index); first_at = pd.Series(np.nan, index=work.index)
        for column, at in candidates:
            usable = first.isna() & work[column].gt(1) & (cut_hours - at).gt(1)
            first = first.where(~usable, work[column]); first_at = first_at.where(~usable, at)
        move = np.log(first / price)
        out["pm_move_first"] = move - move.groupby(race).transform("mean")
        out["pm_move_hours"] = (cut_hours - first_at).where(move.notna()).groupby(race).transform("median")
    out["pm_field_priced"] = price.notna().groupby(race).transform("mean").to_numpy()
    return out


def load_timed(frame, *, when):
    table = timed_prices()
    table["horse_id"] = table.horse_id.astype(frame.horse_id.dtype)
    work = frame[[*KEYS, "p_single"]].merge(table, on=KEYS, how="left", validate="one_to_one")
    # 43 races since 2020 were off at or before 11:00, where the "11:00" price can be in-play; blank them.
    work.loc[work.hours_1100_to_off.le(0), "ltp_1100"] = np.nan
    if when == "1100-hours":
        out = _inputs_at(frame, work, "1100")
        hours = work.hours_1100_to_off.where(out.pm_logshare.notna())
        out["pm_hours_to_off"] = hours.groupby(frame.race_id.to_numpy()).transform("median")
    else:
        out = _inputs_at(frame, work, when)
    columns = [c for c in out if c not in KEYS]
    out[columns] = out[columns].astype("float32")
    return out, columns
