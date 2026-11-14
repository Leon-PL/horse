"""Weight carried, with the trip and the ground, and against what the horse has carried before.

On jumps handicaps of 2025Q3-2026Q2 the stored forecast is about 5% too high
on the top third of the weights and 3% too low on the bottom third (11% too
low on the bottom third in chases of three miles or more), and the closing
price shows a weaker version of the same. The model has weight against the
field, but nothing joining weight to distance, discipline or ground, and
nothing about the weights a horse has carried before. Raised by the user after
the Cheltenham 12:35 of 14 November 2026.

``burden``, declared before scoring:

- today's weight against the field mean, as a rank, and below the top weight;
- that weight against the field multiplied by the trip (miles), the same on
  jumps only, on chases only, and on soft or heavy ground only;
- the trip, and flags for jumps, chase, handicap and soft ground, so a tree can
  condition on them;
- from the horse's runs on strictly earlier dates: today's weight minus the
  mean of its last three, minus the heaviest it has won under, and minus the
  heaviest it has carried; and the number of earlier runs under at least
  today's weight.
"""
import numba as nb
import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS


@nb.njit(cache=True)
def _history(horse, day, weight, won):
    """Per row, from earlier dates: mean of last three weights, heaviest won under, heaviest carried, runs at or above."""
    n = len(horse)
    out = np.full((n, 4), np.nan)
    start = 0
    for i in range(n):
        if i == 0 or horse[i] != horse[i - 1]:
            start = i
        w = weight[i]
        total = count = 0.
        best_won = top = -1.
        above = 0.
        seen = 0
        for j in range(i - 1, start - 1, -1):
            if day[j] >= day[i] or not np.isfinite(weight[j]):
                continue
            seen += 1
            if count < 3:
                total += weight[j]
                count += 1
            if weight[j] > top:
                top = weight[j]
            if won[j] > 0 and weight[j] > best_won:
                best_won = weight[j]
            if np.isfinite(w) and weight[j] >= w:
                above += 1
        if seen and np.isfinite(w):
            out[i, 0] = w - total / count
            out[i, 2] = w - top
            out[i, 3] = above
            if best_won > 0:
                out[i, 1] = w - best_won
    return out


def burden_inputs():
    store = sc.read_store(["weight_lbs", "distance_furlongs", "race_type", "handicap", "going", "finish_position"])
    store["_row"] = np.arange(len(store))
    race = store.race_id.to_numpy()
    weight = pd.to_numeric(store.weight_lbs, errors="coerce")
    weight = weight.where(weight.between(100, 190))
    trip = pd.to_numeric(store.distance_furlongs, errors="coerce") / 8.
    jump = store.race_type.isin(["Hurdle", "Chase"]).astype(float)
    chase = store.race_type.eq("Chase").astype(float)
    soft = store.going.astype(str).str.lower().str.contains("soft|heavy|yielding").astype(float)
    by_race = weight.groupby(race)
    relative = weight - by_race.transform("mean")
    out = store[KEYS + ["_row"]].copy()
    out["bd_weight_vs_field"] = relative
    out["bd_weight_rank"] = by_race.rank(pct=True)
    out["bd_weight_below_top"] = by_race.transform("max") - weight
    out["bd_weight_x_trip"] = relative * trip
    out["bd_weight_x_trip_jump"] = relative * trip * jump
    out["bd_weight_x_trip_chase"] = relative * trip * chase
    out["bd_weight_x_soft"] = relative * soft
    out["bd_trip"], out["bd_jump"], out["bd_chase"], out["bd_soft"] = trip, jump, chase, soft
    out["bd_handicap"] = pd.to_numeric(store.handicap, errors="coerce").fillna(0.)
    h = store.assign(_w=weight).sort_values(["horse_id", "race_date", "_row"], kind="stable")
    horse, _ = pd.factorize(h.horse_id)
    day = (h.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    won = pd.to_numeric(h.finish_position, errors="coerce").eq(1).to_numpy(np.float64)
    past = _history(horse.astype(np.int64), day, h._w.to_numpy(np.float64), won)
    past = pd.DataFrame(past, index=h._row.to_numpy(),
                        columns=["bd_vs_last3", "bd_vs_heaviest_win", "bd_vs_heaviest", "bd_runs_at_or_above"])
    out = out.join(past, on="_row").drop(columns="_row")
    columns = [c for c in out if c not in KEYS]
    out[columns] = out[columns].astype("float32")
    return out


def load(frame):
    values = burden_inputs()
    values["horse_id"] = values.horse_id.astype(frame.horse_id.dtype)
    values["race_id"] = values.race_id.astype(frame.race_id.dtype)
    values = frame[KEYS].merge(values, on=KEYS, how="left", validate="one_to_one")
    return values, [c for c in values if c not in KEYS]
