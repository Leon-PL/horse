"""Has the form worked out? Later runs of a horse's past rivals, against the model.

A rating system rates a race by its runners as they were on the day. A
handicapper also looks at what those runners did next: if the horses that
finished around ours have since run above expectations, the race was stronger
than it looked. Nothing in the store carries this.

For a runner today and each of its last three races R, take every other runner
in R and its runs strictly after R and strictly before today. Each such run has
residuals against the model's out-of-fold forecast (``entity_residuals``):
rivals beaten, win and the market gap. The inputs are shrunk means of those
residuals over the last race and over the last three, the number of rival runs
behind them, and the means relative to today's field.

Rivals are split by where they finished relative to our horse in R, because
the two halves say different things: horses that beat us franking the form
says the race was strong; horses we beat franking it says our beating them
was worth more.
"""
import numpy as np
import pandas as pd
from numba import njit

from research.error_anatomy.entity_residuals import KEYS, run_residuals

SHRINK = 4.  # rival runs
LOOKBACK = 3


@njit(cache=True)
def _frank(h_start, h_dates, cum, cum_n, race_of, pos_of_run, race_start, race_members, position, lookback):
    """For each run (in horse order), sums of rivals' later residuals over the last 1 and ``lookback`` races.

    ``cum``/``cum_n`` are per-horse running totals including the run itself, shape (runs, signals).
    Output columns: [last race, all lookback races] x [rivals ahead, rivals behind] x signals, then counts.
    """
    n, k = cum.shape
    sums = np.zeros((n, 2, 2, k))
    counts = np.zeros((n, 2, 2))
    for horse in range(len(h_start) - 1):
        a, b = h_start[horse], h_start[horse + 1]
        for t in range(a, b):
            today = h_dates[t]
            for back in range(1, lookback + 1):
                r = t - back
                if r < a:
                    break
                if h_dates[r] >= today:
                    continue
                race = race_of[r]
                for m in range(race_start[race], race_start[race + 1]):
                    j = race_members[m]            # the rival's run in that race, in horse order
                    if j == r:
                        continue
                    rival = pos_of_run[j]          # which horse block
                    ja, jb = h_start[rival], h_start[rival + 1]
                    # last run of the rival strictly before today
                    lo, hi = j, jb
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        if h_dates[mid] < today:
                            lo = mid
                        else:
                            hi = mid
                    if lo == j:
                        continue                   # the rival has not run since
                    side = 0 if position[j] < position[r] else 1
                    added = cum_n[lo] - cum_n[j]
                    if added <= 0:
                        continue
                    for s in range(k):
                        value = cum[lo, s] - cum[j, s]
                        sums[t, 1, side, s] += value
                        if back == 1:
                            sums[t, 0, side, s] += value
                    counts[t, 1, side] += added
                    if back == 1:
                        counts[t, 0, side] += added
    return sums, counts


def franking_inputs(history, forecasts, market=None):
    """``history``: every run (race_id, horse_id, race_date, finish_position, won), all dates.

    ``forecasts``: race-ordered frame of the runs that have an out-of-fold forecast, with ``p_single``
    and optionally a market probability column named by ``market``.
    """
    residual = run_residuals(forecasts)
    signals = pd.DataFrame({"order": residual.order, "win": residual.win}, index=forecasts.index)
    if market is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.log(forecasts[market].to_numpy(float)) - np.log(forecasts.p_single.to_numpy(float))
        signals["gap"] = np.clip(np.where(np.isfinite(gap), gap, np.nan), -3, 3)
    signals[KEYS[0]], signals[KEYS[1]] = forecasts.race_id.astype(str).to_numpy(), forecasts.horse_id.to_numpy()
    h = history[[*KEYS, "race_date", "finish_position"]].copy()
    h["race_id"] = h.race_id.astype(str)
    h = h.merge(signals, on=KEYS, how="left", validate="one_to_one")
    names = [c for c in ("order", "win", "gap") if c in h]
    h["_valid"] = h[names].notna().all(axis=1).astype(float)
    h = h.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index(drop=True)
    position = pd.to_numeric(h.finish_position, errors="coerce").to_numpy(float)
    position = np.where(np.isfinite(position) & (position >= 1), position, 999.)
    horse_code, _ = pd.factorize(h.horse_id)
    h_start = np.r_[np.flatnonzero(np.r_[True, horse_code[1:] != horse_code[:-1]]), len(h)].astype(np.int64)
    values = h[names].fillna(0.).to_numpy(float) * h._valid.to_numpy()[:, None]
    grouped = pd.DataFrame(values).groupby(horse_code)
    cum = grouped.cumsum().to_numpy(float)
    cum_n = h._valid.groupby(horse_code).cumsum().to_numpy(float)
    race_code, _ = pd.factorize(h.race_id)
    by_race = np.argsort(race_code, kind="stable").astype(np.int64)
    race_start = np.r_[0, np.cumsum(np.bincount(race_code))].astype(np.int64)
    dates = (h.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    sums, counts = _frank(h_start, dates, np.ascontiguousarray(cum), cum_n, race_code.astype(np.int64),
                          horse_code.astype(np.int64), race_start, by_race, position, LOOKBACK)
    out = h[KEYS].copy()
    for w, window in enumerate(("last", "three")):
        for side, label in enumerate(("ahead", "behind")):
            n = counts[:, w, side]
            out[f"fr_{window}_{label}_runs"] = n.astype("float32")
            for k, name in enumerate(names):
                out[f"fr_{window}_{label}_{name}"] = (sums[:, w, side, k] / (n + SHRINK)).astype("float32")
        total = counts[:, w].sum(1)
        for k, name in enumerate(names):
            out[f"fr_{window}_all_{name}"] = (sums[:, w, :, k].sum(1) / (total + SHRINK)).astype("float32")
    relative = [c for c in out if c.startswith("fr_three_") and not c.endswith("_runs")]
    for column in relative:
        out[f"{column}_vs_field"] = (out[column] - out[column].groupby(h.race_id).transform("mean")).astype("float32")
    return out


def load(frame):
    """Candidate loader: (values, columns) for the frame's races, from the long out-of-fold history."""
    from research.error_anatomy import entity_residuals as er
    from research.meta_learner import screen_candidates as sc
    history = sc.read_store(["finish_position", "won"])
    forecasts = er.long_frame()
    values = franking_inputs(history, forecasts, market="p_market")
    values = values.loc[values.race_id.isin(frame.race_id.astype(str).unique())]
    return values, [c for c in values if c not in KEYS]
