"""Research-only handicap residual model in race-relative pounds."""
import logging

import numpy as np
import pandas as pd

from src.feature_selection import race_groups
from src.feature_semantics import CLAIM_ADJUSTED_WEIGHT_INPUTS, claim_adjusted_weight, observed_rating
from src.model import compute_recency_sample_weights
from src.modelling.data import numeric_matrix
from src.modelling.performance import PerformanceSpec, figure_rows, performance_targets
from src.race_objectives import CentredRegression, fit_boosted, race_bounds, race_softmax, raw_scores
from src.rtv_text import fit_scale


logger = logging.getLogger(__name__)
INPUTS = ("handicap", "official_rating", "has_official_rating", *CLAIM_ADJUSTED_WEIGHT_INPUTS)


def supported_fields(frame):
    observed = frame.handicap.eq(1) & np.isfinite(offset(frame))
    return observed.groupby(frame.race_id, sort=False).transform("all")


def offset(frame):
    return (observed_rating(frame) - claim_adjusted_weight(frame)).to_numpy(dtype=float)


def fit_predict(train, test, spec, dataset_start, capacity, threads):
    """Predict margin beyond the at-weights mark, then restore the known offset."""
    start = pd.Timestamp(dataset_start) + pd.DateOffset(months=spec.burn_in_months)
    train = train.loc[supported_fields(train) & train.race_date.ge(start)]
    groups = race_groups(train.race_id)
    if len(groups) < 50:
        raise ValueError("Insufficient rated handicap fields for a leeway fit")
    cal_first = train.race_date.iloc[np.r_[0, np.cumsum(groups)[:-1]][int(len(groups) * (1 - spec.calibration_fraction))]]
    earlier = train.loc[train.race_date.le(cal_first - pd.Timedelta(days=spec.purge_days))]
    held = train.loc[train.race_date.ge(cal_first)]
    margin_spec = PerformanceSpec("leeway-margin", spec.features)

    def fit(rows, *, refit):
        figures = performance_targets(rows, margin_spec)
        available = figure_rows(figures, rows.race_id)
        target = figures[available] - offset(rows)[available]
        rows = rows.iloc[np.flatnonzero(available)]
        weights = compute_recency_sample_weights(rows.race_date, half_life_days=spec.half_life_days,
            seasonal_boost=spec.seasonal_boost, decay_shape=spec.decay_shape)
        loss = CentredRegression(rows.race_id, target, weights)
        model = fit_boosted(numeric_matrix(rows, spec.features), target, rows.race_id, rows.race_date,
            objective=loss.objective, metric=loss.metric, params=dict(spec.params), screen_params=dict(spec.screen_params),
            capacity=capacity, n_jobs=threads, seed=spec.seed, early_stopping_rounds=spec.early_stopping_rounds,
            purge_days=spec.purge_days, refit=refit)
        return model

    logger.info("Leeway uses %d rated handicap fields; scale slab starts %s", len(groups), cal_first.date())
    temporary = fit(earlier, refit=False)
    held_scores = raw_scores(temporary, numeric_matrix(held, spec.features)) + offset(held)
    scale = fit_scale(held_scores, held.won.to_numpy(dtype=float), held.race_id)
    del temporary
    final = fit(train, refit=True)
    valid = supported_fields(test).to_numpy()
    p = np.full(len(test), np.nan)
    if valid.any():
        fields = test.loc[valid]
        score = raw_scores(final, numeric_matrix(fields, spec.features)) + offset(fields)
        p[valid] = race_softmax(scale * score, race_bounds(fields.race_id))
    return p, {"training_last_date": str(train.race_date.max().date()), "calibration_first_date": str(cal_first.date()),
               "calibration_fit_boundary": str(earlier.race_date.max().date()), "scale": scale,
               "training_races": len(groups), "supported_test_races": int(test.loc[valid, "race_id"].nunique())}
