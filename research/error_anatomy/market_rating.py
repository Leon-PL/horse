"""A rating of each horse from its past closing prices, linked through shared rivals.

The horse-state inputs say how far the closing price sat from the model's
forecast on a horse's earlier runs, and exist only where a stored forecast
does (from 2022Q3). This asks a different question of the same prices and
needs no forecast: how good did the market think the horse was, on a scale
shared by all races?

Under a Luce model the closing probability of runner i is
``exp(theta_i) / sum_j exp(theta_j)``, so ``log p_i = theta_i + c`` with one
constant ``c`` per race. Races are processed in date order from the start of
the store. For a race, ``c`` is the precision-weighted mean of ``log p - m``
over the field (``m`` the runners' current ratings), and each runner's rating
then moves towards ``log p - c`` by a Kalman gain. A horse whose old rivals
have since been priced shorter enters its next race with those rivals rated
higher, so the constant of that race, and with it the horse's own update,
shifts: the link through rivals. Between runs the variance grows with days
off (twice as fast at three or younger), capped at the prior.

``market-rating``, declared before scoring. Prior variance ``P0`` is the mean
within-race variance of log closing probability before 2025-07-01; noise is
``0.3 * P0``; two speeds, ``q = P0 / 1000`` and ``P0 / 300`` per day. Inputs
per speed, all from strictly earlier dates: the rating, its standard
deviation, the rating against the field mean, the log of its Luce share of
today's field, and that log share minus the model's centred log forecast
(where past prices and the model disagree about today's field). Plus the
share of today's field with any rated run.
"""
import numba as nb
import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
SPEEDS = (1000., 300.)
NOISE = .3
MOMENTS_BEFORE = pd.Timestamp("2025-07-01")
YOUNG_FACTOR = 2.


@nb.njit(cache=True)
def _rate(horse, day, race_start, logp, young, n_horses, p0, r, q):
    """Prior rating, variance and rated-run count per row; rows sorted by date then race, races contiguous."""
    n = len(horse)
    mean = np.zeros(n_horses)
    var = np.full(n_horses, p0)
    last = np.full(n_horses, -1)
    runs = np.zeros(n_horses)
    out_mean, out_var, out_runs = np.zeros(n), np.zeros(n), np.zeros(n)
    races = len(race_start) - 1
    a = 0
    while a < races:
        b = a
        today = day[race_start[a]]
        while b < races and day[race_start[b]] == today:
            b += 1
        lo, hi = race_start[a], race_start[b]
        for i in range(lo, hi):  # every runner of the date reads its state before any race of the date is used
            h = horse[i]
            if last[h] >= 0 and last[h] != today:
                var[h] = min(p0, var[h] + q * (YOUNG_FACTOR if young[i] else 1.) * (today - last[h]))
            out_mean[i], out_var[i], out_runs[i] = mean[h], var[h], runs[h]
        for k in range(a, b):
            total = weight = 0.
            for i in range(race_start[k], race_start[k + 1]):
                if np.isfinite(logp[i]):
                    w = 1. / (var[horse[i]] + r)
                    total += w * (logp[i] - mean[horse[i]])
                    weight += w
            if weight == 0.:
                continue
            c = total / weight
            for i in range(race_start[k], race_start[k + 1]):
                h = horse[i]
                if np.isfinite(logp[i]):
                    gain = var[h] / (var[h] + r)
                    mean[h] += gain * (logp[i] - c - mean[h])
                    var[h] *= 1 - gain
                    runs[h] += 1.
        for i in range(lo, hi):
            last[horse[i]] = today
        a = b
    return out_mean, out_var, out_runs


def ratings():
    """One row per runner in the store with its prior ratings at each speed."""
    # Closing prices live in the study input (every runner from 2020-09-15), not in the feature store.
    store = sc.read_input(["race_date", "bsp", "odds", "age"], "2000-01-01")
    store["race_id"] = store.race_id.astype(str)
    store = store.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    race = store.race_id.to_numpy()
    p = np.full(len(store), np.nan)
    for column in ("odds", "bsp"):  # BSP where the whole field has one, else SP
        price = pd.to_numeric(store[column], errors="coerce")
        usable = (price > 1) & np.isfinite(price)
        whole = usable.groupby(store.race_id).transform("all").to_numpy()
        inverse = 1 / price.where(usable)
        share = (inverse / inverse.groupby(store.race_id).transform("sum")).to_numpy()
        p = np.where(whole, share, p)
    logp = np.log(p)
    centred = pd.Series(logp) - pd.Series(logp).groupby(race).transform("mean")
    p0 = float((centred[(store.race_date < MOMENTS_BEFORE).to_numpy()] ** 2).mean())
    print(f"market rating: prior variance {p0:.4f}, priced rows {np.isfinite(logp).mean():.3f}", flush=True)
    bounds = s.bounds_of(race)
    horse, uniques = pd.factorize(store.horse_id)
    day = (store.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    young = pd.to_numeric(store.age, errors="coerce").le(3).to_numpy(np.bool_)
    out = store[KEYS].copy()
    for speed in SPEEDS:
        mean, var, runs = _rate(horse.astype(np.int64), day, np.asarray(bounds, np.int64), logp, young, len(uniques),
                                p0, NOISE * p0, p0 / speed)
        tag = int(speed)
        rated = runs > 0
        out[f"mr_rating_{tag}"] = np.where(rated, mean, np.nan)
        out[f"mr_sd_{tag}"] = np.sqrt(var)
        level = pd.Series(mean)  # an unrated horse sits at zero, the level of an average runner in its races
        out[f"mr_vs_field_{tag}"] = (level - level.groupby(race).transform("mean")).to_numpy()
        peak = level.groupby(race).transform("max")
        total = np.log(np.exp(level - peak).groupby(race).transform("sum")) + peak
        out[f"mr_logshare_{tag}"] = (level - total).to_numpy()
    out["mr_runs"] = runs
    out["mr_field_rated"] = pd.Series(runs > 0).groupby(race).transform("mean").to_numpy()
    return out


def load(frame):
    values = ratings()
    values["horse_id"] = values.horse_id.astype(frame.horse_id.dtype)
    values = frame[[*KEYS, "p_single"]].merge(values, on=KEYS, how="left", validate="one_to_one")
    forecast = np.log(values.p_single)
    for speed in SPEEDS:
        tag = int(speed)
        values[f"mr_gap_{tag}"] = values[f"mr_logshare_{tag}"] - forecast
        values[f"mr_gap_{tag}"] -= values[f"mr_gap_{tag}"].groupby(frame.race_id.to_numpy()).transform("mean")
    values = values.drop(columns="p_single")
    columns = [c for c in values if c not in KEYS]
    values[columns] = values[columns].astype("float32")
    return values, columns
