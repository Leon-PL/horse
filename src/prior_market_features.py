"""Prior-day market assessments; never current-race prices as predictors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.established_form_features import _horse_keys, mark_scale
from src.feature_semantics import observed_rating
from src.forecast_evaluation import valid_race_mask
from src.result_status import NONSTARTER_LABELS, result_labels

FEATURES = ("horse_prior_market_logrel_3_vs_field", "horse_prior_market_opposition_3_vs_field", "horse_prior_market_support_3")


def past_market_observations(frame):
    """Keep past relative market opinion and opposition quality in separate units.

    Log(N*p) compares opinion with a uniform field. Opponent quality uses
    observed race-day marks, requiring3 rated rivals; no imputed mark is used.
    The learner can estimate their relationship without an assumed conversion
    of log market probabilities into Elo points or handicap pounds.
    """
    price = pd.to_numeric(frame.odds, errors="coerce")
    known = valid_race_mask(frame) & price.ge(1.01) & np.isfinite(price)
    complete = known.groupby(frame.race_id, sort=False).transform("all")
    inverse = (1/price).where(complete)
    probability = inverse/inverse.groupby(frame.race_id, sort=False).transform("sum")
    logrel = np.log(probability*pd.to_numeric(frame.num_runners, errors="coerce"))
    marks = observed_rating(frame)
    count = marks.notna().groupby(frame.race_id, sort=False).transform("sum")-marks.notna().astype(int)
    total = marks.fillna(0).groupby(frame.race_id, sort=False).transform("sum")-marks.fillna(0)
    opposition = (total/count.where(count.ge(3))).where(valid_race_mask(frame))
    return pd.DataFrame({"logrel":logrel, "opposition":opposition}, index=frame.index)


def prior_market_features(frame, *, state=None, return_state=False):
    """Last3 actual same-scale starts/365 days, all current-day prices excluded.

    Missing complete books advance the window as unknown; NR/pending do not.
    Only past observations are stored. Input state remains unmodified.
    """
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    state = state or {}
    if dates.isna().any() or (len(frame) and state.get("last_date")
            and dates.min() <= pd.Timestamp(state["last_date"])):
        raise ValueError("Prior-market seed must end on an earlier complete date")
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    keys, scales = _horse_keys(frame), mark_scale(frame).to_numpy(dtype=object, na_value=None)
    obs = past_market_observations(frame)
    opinions, opposition = obs.logrel.to_numpy(), obs.opposition.to_numpy()
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    output = {c:np.full(len(frame), np.nan) for c in FEATURES}
    history = {k:list(v) for k,v in state.get("history", {}).items()}
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order]))+1, len(frame)]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        for i in rows:
            if keys[i] is None or scales[i] is None:
                continue
            recent = [(s,e) for day,s,e in history.get((keys[i], scales[i]), []) if days[i]-day <= 365]
            prices = [s for s,e in recent if np.isfinite(s)]
            rivals = [e for s,e in recent if np.isfinite(e)]
            output[FEATURES[2]][i] = len(prices)
            if prices:
                output[FEATURES[0]][i] = sum(prices)/len(prices)
            if rivals:
                output[FEATURES[1]][i] = sum(rivals)/len(rivals)
        for i in rows:
            if starts[i] and keys[i] is not None and scales[i] is not None:
                key = (keys[i], scales[i])
                history[key] = (history.get(key, [])+[(int(days[i]), float(opinions[i]), float(opposition[i]))])[-3:]
    out = pd.DataFrame(output, index=frame.index)
    for c in FEATURES[:2]:
        out[c] -= out[c].groupby(frame.race_id, sort=False).transform("mean")
    if return_state:
        return out, {"last_date":str(dates.max().date()) if len(frame) else state.get("last_date"), "history":history}
    return out
