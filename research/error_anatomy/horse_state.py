"""Run-by-run state of each horse's errors, and short-memory stable form.

The third pass found that the corrected forecast's disagreement with the
market still repeats for the same horse (correlation 0.18 with its previous
run, 0.16 two runs back on the uncorrected forecast). The own-horse inputs of
``entity_residuals`` are one decayed mean per signal with a 365-day half-life,
which cannot tell a horse whose last run was the surprise from one whose
surprise is a year old.

Two groups, declared before either was scored:

- ``horse-state``: the horse's last five race-centred market gaps and last
  three rivals-beaten and win residuals, days back to each of the last three
  runs, run-weighted moving averages of each signal (weights 0.5 and 0.25 per
  run), and the horse's mean gap and rivals-beaten residual under today's
  code, surface and trip band.
- ``entity-recent``: the residual and market-gap means of ``entity_residuals``
  for trainer, jockey, trainer and jockey together, and sire, with half-lives
  of 30 and 90 days instead of 365.

Everything uses runs on strictly earlier dates, against the seed-42
out-of-fold forecast, with the earlier race's closing price.
"""
import numba
import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.meta_learner import screening as s

KEYS = s.KEYS
LAGS = {"gap": 5, "order": 3, "win": 3}
ALPHAS = (.5, .25)
CONDITIONS = ("_code", "_surface", "_trip")
RECENT = {"trainer": ["trainer"], "jockey": ["jockey"], "trainer_jockey": ["trainer", "jockey"], "sire": ["sire"]}
HALF_LIVES = (30., 90.)


@numba.njit(cache=True)
def _moving(group, values, alpha):
    """Run-weighted mean of earlier finite values within each group; rows are sorted by group then time."""
    out = np.full(len(values), np.nan)
    total = weight = 0.
    for i in range(len(values)):
        if i == 0 or group[i] != group[i - 1]:
            total = weight = 0.
        if weight > 0:
            out[i] = total / weight
        if np.isfinite(values[i]):
            total = (1 - alpha) * total + values[i]
            weight = (1 - alpha) * weight + 1.
        else:
            total *= 1 - alpha
            weight *= 1 - alpha
    return out


def signals(frame, probability="p_single", market="p_market"):
    residual = er.run_residuals(frame, probability)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(frame[market].to_numpy(float)) - np.log(frame[probability].to_numpy(float))
    gap = pd.Series(np.where(np.isfinite(gap), gap, np.nan), index=frame.index).clip(-3, 3)
    gap = gap - gap.groupby(frame.race_id).transform("mean")
    return pd.DataFrame({"gap": gap, "order": residual.order, "win": residual.win}, index=frame.index)


def horse_state(frame, probability="p_single", market="p_market"):
    """One row per runner of ``frame`` (race order kept) with the horse's run-by-run state."""
    work = pd.concat([frame[[*KEYS, "race_date"]], signals(frame, probability, market), er._keys(frame)[list(CONDITIONS)]],
                     axis=1)
    work["_row"] = np.arange(len(work))
    work = work.sort_values(["horse_id", "race_date", "_row"], kind="stable").reset_index(drop=True)
    horse = work.groupby("horse_id", sort=False)
    code = horse.ngroup().to_numpy()
    day = (work.race_date - pd.Timestamp("2000-01-01")).dt.days.astype(float)
    made = {}
    # A second run on the same date is not an earlier date: such rows get no state at all.
    same_day = (horse.race_date.shift(1) == work.race_date).to_numpy()
    for name, lags in LAGS.items():
        for k in range(1, lags + 1):
            made[f"hs_{name}_l{k}"] = horse[name].shift(k)
        for alpha in ALPHAS:
            made[f"hs_{name}_ma{int(alpha * 100)}"] = _moving(code, work[name].to_numpy(float), alpha)
    for k in (1, 2, 3):
        made[f"hs_days_l{k}"] = day - day.groupby(work.horse_id, sort=False).shift(k)
    made["hs_gap_change"] = made["hs_gap_l1"] - made["hs_gap_l2"]
    for condition in CONDITIONS:
        by = work.groupby(["horse_id", condition], sort=False)
        for name in ("gap", "order"):
            value = work[name]
            runs = value.notna().astype(float)
            total = value.fillna(0.)
            # Running totals minus the row itself: earlier rows only.
            prior_sum = total.groupby([work.horse_id, work[condition]], sort=False).cumsum() - total
            prior_runs = runs.groupby([work.horse_id, work[condition]], sort=False).cumsum() - runs
            made[f"hs_{name}_same{condition}"] = prior_sum / (prior_runs + 1.)
        made[f"hs_runs_same{condition}"] = by.cumcount().astype(float)
    out = pd.DataFrame(made, index=work.index).astype("float32")
    out[same_day] = np.nan
    out = pd.concat([work[[*KEYS, "_row"]], out], axis=1).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    relative = [c for c in out if c.startswith(("hs_gap_l1", "hs_gap_ma", "hs_order_ma"))]
    for column in relative:
        out[f"{column}_vs_field"] = (out[column] - out[column].groupby(frame.race_id.to_numpy()).transform("mean")
                                     ).astype("float32")
    return out


def entity_recent(frame, probability="p_single", market="p_market"):
    out = frame[KEYS].copy()
    for half_life in HALF_LIVES:
        values = er.entity_inputs(frame, probability, entities=RECENT, half_life=half_life, market=market,
                                  prefix=f"er{int(half_life)}")
        values = values.rename(columns={c: c.replace("mg_", f"mg{int(half_life)}_") for c in values if c.startswith("mg_")})
        out = out.merge(values, on=KEYS, validate="one_to_one")
    return out


# Ablations of ``horse-state``, to see which part carries it.
SUBSETS = {
    "horse-state-gap": ("hs_gap_",),                 # past closing prices against the forecast
    "horse-state-result": ("hs_win_", "hs_order_"),  # past results against the forecast
    "horse-state-lags": ("hs_gap_l", "hs_win_l", "hs_order_l", "hs_days_l"),  # no averages, no conditions
}


def load(name, frame):
    if name in SUBSETS:
        values = horse_state(frame)
        columns = [c for c in values if c.startswith(SUBSETS[name])]
        return values[[*KEYS, *columns]], columns
    values = {"horse-state": horse_state, "entity-recent": entity_recent}[name](frame)
    columns = [c for c in values if c not in KEYS]
    return values, columns
