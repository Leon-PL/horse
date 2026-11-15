"""How a horse runs fresh: its record on earlier reappearances (research 099).

Started from an analyst's read of Jonbon's Shloer Chase (13 November 2026):
some horses need their first run of a season, others go best fresh. The
store knows days since the last run and the trainer's record with returners
(the ``wave2`` freshness keys), but not this horse's own reappearance record.

``comeback``, declared before scoring. A comeback is a run 90 or more days
after the horse's previous recorded run. For each runner, from runs on
strictly earlier dates:

- ``cb_today``: today is a comeback; ``cb_days``: days since the last run;
- ``cb_n``: earlier comebacks;
- ``cb_pct``: finishing percentile on earlier comebacks (1 = won, 0 = last or
  did not finish), shrunk towards 0.5 by two runs; ``cb_pct_next``: the same on
  the run straight after each earlier comeback; ``cb_needs_run``:
  ``cb_pct - cb_pct_next`` (negative = needs the run);
- ``cb_pct_vs_career``: comeback percentile against the horse's other runs;
- ``cb_resid``, ``cb_resid_other``: mean of won minus the stored forecast on
  earlier comebacks and on other runs, shrunk by two runs (forecast rows only);
- ``cb_mkt_gap``: mean log closing price against the stored forecast on earlier
  comebacks (the market's view of the horse fresh), shrunk by two runs;
- the four ``cb_*`` above multiplied by ``cb_today``, so a correction can apply
  them only to a reappearance.
"""
import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
GAP_DAYS = 90
SHRINK = 2.


def _shrunk(total, count, centre):
    return (total + SHRINK * centre) / (count + SHRINK)


def comeback_inputs(frame):
    store = sc.read_store(["finish_position", "num_runners"])
    store["_row"] = np.arange(len(store))
    h = store.sort_values(["horse_id", "race_date", "_row"], kind="stable").reset_index(drop=True)
    by = h.horse_id
    horse = h.groupby("horse_id", sort=False)
    position = pd.to_numeric(h.finish_position, errors="coerce")
    runners = pd.to_numeric(h.num_runners, errors="coerce")
    pct = (1 - (position - 1) / (runners - 1).clip(lower=1)).where(position >= 1, 0.).clip(0, 1)
    days = (h.race_date - horse.race_date.shift(1)).dt.days
    is_cb = days.ge(GAP_DAYS)
    is_next = is_cb.groupby(by, sort=False).shift(1, fill_value=False) & days.lt(GAP_DAYS)

    # Forecast residual and market gap, joined by runner; missing outside the forecast window.
    f = frame[[*KEYS, "won", "p_single", "p_market"]].copy()
    f["race_id"] = f.race_id.astype(str)
    f["horse_id"] = f.horse_id.astype(h.horse_id.dtype)
    h = h.merge(f, on=KEYS, how="left", validate="one_to_one")
    resid = (h.won - h.p_single)
    gap = np.log(h.p_market / h.p_single)

    def before(values):
        v = values.fillna(0.).astype(float)
        return v.groupby(by, sort=False).cumsum() - v

    one = pd.Series(1., index=h.index)
    cbf, nxf = is_cb.astype(float), is_next.astype(float)
    has_r = resid.notna().astype(float)
    has_g = gap.replace([np.inf, -np.inf], np.nan).notna().astype(float)
    n_cb, s_cb = before(cbf), before(pct * cbf)
    n_nx, s_nx = before(nxf), before(pct * nxf)
    n_all, s_all = before(one), before(pct)
    n_rc, s_rc = before(cbf * has_r), before(resid.fillna(0) * cbf)
    n_ro, s_ro = before((1 - cbf) * has_r), before(resid.fillna(0) * (1 - cbf))
    n_gc, s_gc = before(cbf * has_g), before(gap.replace([np.inf, -np.inf], np.nan).fillna(0) * cbf)

    out = h[[*KEYS, "_row"]].copy()
    out["cb_today"] = cbf
    out["cb_days"] = days.astype(float)
    out["cb_n"] = n_cb
    out["cb_pct"] = _shrunk(s_cb, n_cb, .5).where(n_cb > 0)
    out["cb_pct_next"] = _shrunk(s_nx, n_nx, .5).where(n_nx > 0)
    out["cb_needs_run"] = out.cb_pct - out.cb_pct_next
    other = _shrunk(s_all - s_cb, n_all - n_cb, .5).where(n_all > n_cb)
    out["cb_pct_vs_career"] = out.cb_pct - other
    out["cb_resid"] = _shrunk(s_rc, n_rc, 0.).where(n_rc > 0)
    out["cb_resid_other"] = _shrunk(s_ro, n_ro, 0.).where(n_ro > 0)
    out["cb_mkt_gap"] = _shrunk(s_gc, n_gc, 0.).where(n_gc > 0)
    for c in ("cb_needs_run", "cb_pct_vs_career", "cb_resid", "cb_mkt_gap"):
        out[f"{c}_today"] = out[c] * out.cb_today
    # A second run on the same date is not an earlier date.
    same_day = (horse.race_date.shift(1) == h.race_date).to_numpy()
    columns = [c for c in out if c.startswith("cb_")]
    out.loc[same_day, [c for c in columns if c not in ("cb_today", "cb_days")]] = np.nan
    out[columns] = out[columns].astype("float32")
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True), columns


def load(frame, *, twin=False):
    values, columns = comeback_inputs(frame)
    values["race_id"] = values.race_id.astype(frame.race_id.dtype)
    values["horse_id"] = values.horse_id.astype(frame.horse_id.dtype)
    out = frame[[*KEYS, "quarter"]].merge(values, on=KEYS, how="left", validate="one_to_one")
    if twin:
        rng = np.random.default_rng(11)
        quarter = out.quarter.to_numpy()
        for label in pd.unique(quarter):
            idx = np.flatnonzero(quarter == label)
            out.loc[idx, columns] = out.loc[idx[rng.permutation(len(idx))], columns].to_numpy()
        out = out.rename(columns={c: f"{c}_twin" for c in columns})
        columns = [f"{c}_twin" for c in columns]
    return out[[*KEYS, *columns]], columns
