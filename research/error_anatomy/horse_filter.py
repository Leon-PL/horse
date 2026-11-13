"""Filtered horse state: a running estimate per horse with its own uncertainty.

The horse-state inputs are fixed-weight averages of how far the closing price
sat from the forecast on the horse's earlier runs (last run, weights 0.5 and
0.25 per run). They weight a run 14 days ago and one 400 days ago alike, and a
two-year-old like a nine-year-old. A filter does neither.

``filter``, declared before scoring. For each horse a local-level filter of
the race-centred log gap between closing price and forecast. (The
rivals-beaten residual was declared too and dropped before any scoring: its
lag-one covariance between a horse's consecutive runs is -0.002 of a variance
of 0.084, so there is no level to filter.)

- the hidden level starts at zero with variance ``P0``;
- between runs it drifts: variance grows by ``q`` per day off, twice as fast
  for horses aged three or younger, and never above ``P0``;
- each run observes the level with noise ``r``.

``P0`` and ``r`` come from moments of the signal before 2025-07-01 (the
lag-one covariance of a horse's consecutive runs and the total variance), so
no scored race sets them. Three speeds, ``q = P0 / 1000, 300, 100`` per day.
Inputs per signal and speed: the predicted level, its standard deviation, and
the level against the field; plus the level divided by its standard deviation
at the middle speed. Strictly earlier dates only: a run on the same date is
not used.
"""
import numba as nb
import numpy as np
import pandas as pd

from research.error_anatomy import horse_state as hs
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
SPEEDS = (1000., 300., 100.)
MOMENTS_BEFORE = pd.Timestamp("2025-07-01")
YOUNG_FACTOR = 2.


@nb.njit(cache=True)
def _filter(horse, day, value, young, p0, r, q):
    """Predicted level and variance before each run; rows sorted by horse then day."""
    n = len(horse)
    mean = np.full(n, np.nan)
    var = np.full(n, np.nan)
    m, p, last = 0., p0, -1
    first = 0
    for i in range(n):
        if i == 0 or horse[i] != horse[i - 1]:
            m, p, last = 0., p0, -1
            first = i
        if i == first or day[i] != day[i - 1]:
            # Fold in the runs of the previous date, then drift to today.
            j = i - 1
            while j >= first and day[j] == day[i - 1] and i > first:
                if np.isfinite(value[j]):
                    gain = p / (p + r)
                    m += gain * (value[j] - m)
                    p *= 1 - gain
                j -= 1
            if last >= 0:
                rate = q * (YOUNG_FACTOR if young[i] else 1.)
                p = min(p0, p + rate * (day[i] - last))
            last = day[i]
        mean[i], var[i] = m, p
    return mean, var


def moments(frame, signal):
    """Level variance (lag-one covariance of consecutive runs) and noise variance, from races before the cut."""
    early = frame.loc[frame.race_date < MOMENTS_BEFORE, ["horse_id", "race_date"]].assign(v=signal)
    early = early.dropna(subset=["v"]).sort_values(["horse_id", "race_date"], kind="stable")
    previous = early.groupby("horse_id", sort=False).v.shift(1)
    both = previous.notna()
    level = float(np.cov(early.v[both], previous[both])[0, 1])
    total = float(early.v.var())
    if not 0 < level < total:
        raise ValueError(f"Unusable moments: level {level}, total {total}")
    return level, total - level


def filter_inputs(frame, probability="p_single", market="p_market"):
    signals = hs.signals(frame, probability, market)
    age = sc.read_store(["age"], start=frame.race_date.min()).drop(columns="race_date")
    age = s.attach(frame[KEYS].copy(), age).age
    work = pd.concat([frame[[*KEYS, "race_date"]], signals[["gap", "order"]]], axis=1)
    work["young"] = pd.to_numeric(age, errors="coerce").le(3).to_numpy()
    work["_row"] = np.arange(len(work))
    work = work.sort_values(["horse_id", "race_date", "_row"], kind="stable").reset_index(drop=True)
    horse = work.horse_id.to_numpy(np.int64)
    day = (work.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    young = work.young.to_numpy(np.bool_)
    made = {}
    for name in ("gap",):
        p0, r = moments(work, work[name])
        print(f"filter {name}: level variance {p0:.4f}, noise variance {r:.4f}", flush=True)
        for speed in SPEEDS:
            mean, var = _filter(horse, day, work[name].to_numpy(float), young, p0, r, p0 / speed)
            made[f"hf_{name}_{int(speed)}"] = mean
            made[f"hf_{name}_{int(speed)}_sd"] = np.sqrt(var)
        made[f"hf_{name}_z"] = made[f"hf_{name}_300"] / made[f"hf_{name}_300_sd"]
    out = pd.concat([work[[*KEYS, "_row"]], pd.DataFrame(made).astype("float32")], axis=1)
    out = out.sort_values("_row").drop(columns="_row").reset_index(drop=True)
    race = frame.race_id.to_numpy()
    for column in [c for c in out if c.startswith("hf_") and not c.endswith("_sd")]:
        out[f"{column}_vs_field"] = (out[column] - out[column].groupby(race).transform("mean")).astype("float32")
    return out


def load(frame):
    values = filter_inputs(frame)
    return values, [c for c in values if c not in KEYS]
