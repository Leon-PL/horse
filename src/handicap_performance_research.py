"""Experimental prior-day form measured against rated handicap opposition.

Not registered in production feature engineering. These are mechanical proxies,
not BHA performance figures: no sectional, going/time or discretionary adjustment.
Declared weights intentionally include apprentice allowances, as BHA figures do.
Only mature horses are assessed, avoiding an invented weight-for-age correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.established_form_features import _horse_keys, mark_scale, race_discipline
from src.feature_semantics import observed_rating
from src.forecast_evaluation import valid_race_mask
from src.result_status import NONSTARTER_LABELS, result_labels

FEATURES = (
    "handicap_perf_net_mean_3_vs_field",
    "handicap_perf_net_best_3_vs_field",
    "handicap_perf_support_3",
)


def poundage(distance, discipline):
    """Fixed approximation to BHA's published typical-distance scales.

    Flat 17.05/f matches the stated 5f and 14f examples. Jumps linearly
    interpolate the published graph, extending its end slopes to 16f/36f.
    Unsupported distances/disciplines remain unknown. No fitted coefficients.
    https://www.britishhorseracing.com/regulation/performance-figures/
    """
    distance = np.asarray(distance, dtype=float)
    discipline = np.asarray(discipline, dtype=object)
    out = np.full(len(distance), np.nan)
    flat = (discipline == "Flat") & (distance >= 5) & (distance <= 20)
    jumps = np.isin(discipline, ["Hurdle", "Chase"]) & (distance >= 16) & (distance <= 36)
    out[flat] = 17.05 / distance[flat]
    out[jumps] = np.interp(distance[jumps],
        [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36],
        [1.07, 1.01, .95, .90, .85, .81, .78, .75, .72, .70, .68])
    return out


def performance_observations(frame):
    """Historical observations only; never feed these current outcomes to a model.

    Anchor on the median rating-implied level of the first three finishers.
    Require all three to have observed marks/weights and mature ages: Flat5+,
    jumps6+. Missing data or incomplete results do not create a performance.
    """
    code = race_discipline(frame).to_numpy(dtype=object, na_value=None)
    numeric = lambda c: pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
    distance, age, weight = (numeric(c) for c in ("distance_furlongs", "age", "weight_lbs"))
    known_weight = np.isfinite(weight) & (weight >= 50) & (weight <= 220)
    if "has_observed_weight" in frame:
        known_weight &= frame.has_observed_weight.eq(1).to_numpy()
    weight = np.where(known_weight, weight, np.nan)
    mature = ((code == "Flat") & (age >= 5)) | (np.isin(code, ["Hurdle", "Chase"]) & (age >= 6))
    finish, margin = numeric("finish_position"), numeric("beaten_lengths")
    lb = poundage(distance, code)
    marks = observed_rating(frame).to_numpy(float)
    eligible = (mature & frame.handicap.eq(1).to_numpy() & valid_race_mask(frame).to_numpy()
        & (finish >= 1) & (margin >= 0) & np.isfinite(margin) & np.isfinite(weight) & np.isfinite(lb))
    anchor_value = np.where(eligible & (finish <= 3), marks-weight+lb*margin, np.nan)
    work = pd.DataFrame({"race_id": frame.race_id.to_numpy(), "anchor": anchor_value})
    grouped = work.groupby("race_id", sort=False).anchor
    anchor = grouped.transform("median").where(grouped.transform("count").eq(3)).to_numpy()
    figure = np.where(eligible, anchor+weight-lb*margin, np.nan)
    return pd.DataFrame({"performance": figure, "pounds_per_length": lb,
        "race_anchor": anchor, "observed_weight": weight}, index=frame.index)


def handicap_performance_features(frame, *, state=None, return_state=False):
    """Only earlier days enter last-three-start histories, separately by scale.

    DNF/unrated/conditions starts advance the window with unknown performance.
    Nonstarters and pending cards do not advance it. Figures older than a year
    are not carried forward. State is bounded and can seed later live cards.
    """
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    state = state or {}
    if dates.isna().any() or (len(frame) and state.get("last_date")
            and dates.min() <= pd.Timestamp(state["last_date"])):
        raise ValueError("Seed history must end on an earlier complete date")
    obs = performance_observations(frame)
    scales = mark_scale(frame).to_numpy(dtype=object, na_value=None)
    keys = _horse_keys(frame)
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    history = {k: list(v) for k, v in state.get("history", {}).items()}
    means, bests, support = (np.full(len(frame), np.nan) for _ in range(3))
    # A horse normally starts once per date. A stable event key also makes
    # duplicate same-day starts independent of the incoming row permutation.
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), days))
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order]))+1, len(frame)]
    values, weights = obs.performance.to_numpy(), obs.observed_weight.to_numpy()
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        for i in rows:
            if keys[i] is None or scales[i] is None:
                continue
            recent = [v for day, v in history.get((keys[i], scales[i]), [])
                      if days[i]-day <= 365 and np.isfinite(v)]
            support[i] = len(recent)
            if recent and np.isfinite(weights[i]):
                means[i] = np.mean(recent)-weights[i]
                bests[i] = max(recent)-weights[i]
        for i in rows:
            if starts[i] and keys[i] is not None and scales[i] is not None:
                key = (keys[i], scales[i])
                history[key] = (history.get(key, [])+[(int(days[i]), float(values[i]))])[-3:]
    out = pd.DataFrame({FEATURES[0]: means, FEATURES[1]: bests, FEATURES[2]: support}, index=frame.index)
    for c in FEATURES[:2]:
        out[c] -= out[c].groupby(frame.race_id, sort=False).transform("mean")
    if return_state:
        return out, {"last_date": str(dates.max().date()) if len(frame) else state.get("last_date"), "history": history}
    return out
