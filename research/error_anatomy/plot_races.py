"""Laying one out: do some trainers and owners win more than forecast when the handicap mark is the point?

The user's suggestion: some connections do not ask for a horse's best in
qualifying runs, to earn a lenient mark for a target race. Section 9 found the
generic patterns (handicap debut after runs at big prices, a mark well below a
recent high) priced on average across all runners. This asks whether they are
priced trainer by trainer and owner by owner.

``plot``, declared before scoring: the shrunk, 365-day-decayed residual and
market-gap means of ``entity_residuals`` (results and closing prices against
the model's forecast, strictly earlier dates), keyed by trainer and by owner,
each with

- the horse's handicap situation today: first handicap run; handicap off a
  mark below its last winning mark; handicap off a mark eased by 2lb or more
  since its last run; any other handicap; not a handicap;
- the preparation: whether the horse's last three runs were all at starting
  prices of 16/1 or longer (unfancied each time) or not;
- the switch today: down in class, a change of trip of two furlongs or more,
  both, or neither.

The existing inputs key a trainer by handicap or not, by code, by track and by
the horse's freshness, and never by any of these.
"""
import numba as nb
import numpy as np
import pandas as pd

from research.error_anatomy import owner as ow
from research.error_anatomy.newcomer import with_keys
from research.meta_learner import screening as s

KEYS = s.KEYS
STORE = "data/processed/feature_store/featured.parquet"
ENTITIES = {"trainer_mark": ["trainer", "_mark"], "trainer_prep": ["trainer", "_mark", "_prep"],
            "trainer_switch": ["trainer", "_switch"], "owner_mark": ["owner", "_mark"],
            "owner_prep": ["owner", "_mark", "_prep"]}


@nb.njit(cache=True)
def _unfancied(horse, day, price):
    """1 where the horse's last three runs on earlier dates were all at 17.0 or longer, 0 otherwise, NaN if fewer."""
    out = np.full(len(horse), np.nan)
    start = 0
    for i in range(len(horse)):
        if i == 0 or horse[i] != horse[i - 1]:
            start = i
        seen = 0
        long = 0
        for j in range(i - 1, start - 1, -1):
            if day[j] == day[i] or np.isnan(price[j]):
                continue
            seen += 1
            long += price[j] >= 17.
            if seen == 3:
                break
        if seen == 3:
            out[i] = 1. if long == 3 else 0.
    return out


def situations():
    store = pd.read_parquet(STORE, columns=[*KEYS, "race_date", "handicap", "handicap_discipline_runs_log",
                                            "or_vs_last_win_mark", "or_change", "class_change", "distance_change", "odds"])
    store["race_id"] = store.race_id.astype(str)
    handicap = pd.to_numeric(store.handicap, errors="coerce").eq(1)
    first = handicap & pd.to_numeric(store.handicap_discipline_runs_log, errors="coerce").eq(0)
    below = handicap & pd.to_numeric(store.or_vs_last_win_mark, errors="coerce").lt(0)
    eased = handicap & pd.to_numeric(store.or_change, errors="coerce").le(-2)
    store["_mark"] = np.select([~handicap, first, below, eased], ["none", "debut", "below_win", "eased"], "other")
    store = store.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index(drop=True)
    day = (store.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    unfancied = _unfancied(store.horse_id.to_numpy(np.int64), day, pd.to_numeric(store.odds, errors="coerce").to_numpy(float))
    store["_prep"] = np.where(unfancied == 1, "unfancied", "other")
    drop = pd.to_numeric(store.class_change, errors="coerce").gt(0)  # a higher class number is a lower grade
    trip = pd.to_numeric(store.distance_change, errors="coerce").abs().ge(2)
    store["_switch"] = np.select([drop & trip, drop, trip], ["both", "class", "trip"], "none")
    return store[[*KEYS, "_mark", "_prep", "_switch"]]


def plot_inputs(frame):
    extra = situations()
    extra["horse_id"] = extra.horse_id.astype(frame.horse_id.dtype)
    owners = pd.read_parquet(ow.SAVED)
    owners["horse_id"] = owners.horse_id.astype(frame.horse_id.dtype)
    work = frame[KEYS].merge(extra, on=KEYS, how="left", validate="one_to_one").merge(owners, on=KEYS, how="left",
                                                                                     validate="one_to_one")
    return with_keys(frame, {c: work[c].to_numpy(object) for c in ("_mark", "_prep", "_switch", "owner")}, ENTITIES, "pl")


def load(frame):
    values = plot_inputs(frame)
    return values, [c for c in values if c not in KEYS]
