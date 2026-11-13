"""Scored runners with the stored three-seed forecast, the market and race context.

Used by the error-anatomy scripts in this folder. Nothing here fits a model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

OUT = Path("data/research/error_anatomy_20261110")
KEYS = s.KEYS
CONTEXT = ["off_time", "track", "region", "race_name", "race_class", "race_type", "distance_furlongs", "going",
           "surface", "handicap", "prize_money", "jockey", "trainer", "age", "sex", "headgear", "weight_lbs", "draw",
           "official_rating", "finish_pos_label", "beaten_lengths", "opening_odds",
           "days_since_last_run", "race_is_bumper", "race_is_novice", "race_is_maiden", "prior_form_runs"]


def race_normalised(values, race_id):
    values = pd.Series(np.asarray(values, dtype=float))
    total = values.groupby(np.asarray(race_id)).transform("sum")
    return (values / total).to_numpy()


def scored_frame(extra=(), scored_only=True):
    """One row per runner in the scored quarters, in race order, with model and market probabilities."""
    frame = sc.base_frame()
    if scored_only:
        frame = frame.loc[frame.quarter.isin(s.SCORED)].reset_index(drop=True)
    store = sc.read_store([*CONTEXT, *extra], start=frame.race_date.min()).drop(columns="race_date")
    frame = s.attach(frame, store)
    prices = pd.read_parquet(s.STUDY / "input.parquet", columns=[*KEYS, "bsp", "odds"],
                             filters=[("race_date", ">=", frame.race_date.min())])
    prices.attrs = {}
    frame = s.attach(frame, prices)
    bsp = pd.to_numeric(frame.bsp, errors="coerce")
    frame["bsp_ok"] = (bsp > 1).groupby(frame.race_id).transform("all")
    frame["p_bsp"] = np.where(frame.bsp_ok, race_normalised(1 / bsp.where(bsp > 1), frame.race_id), np.nan)
    sp = pd.to_numeric(frame.odds, errors="coerce")
    frame["sp_ok"] = (sp > 1).groupby(frame.race_id).transform("all")
    frame["p_sp"] = np.where(frame.sp_ok, race_normalised(1 / sp.where(sp > 1), frame.race_id), np.nan)
    return frame


def race_table(frame, columns):
    """Per-race winner log loss for each probability column, with the race's first-row context."""
    bounds = s.bounds_of(frame.race_id.to_numpy())
    races = frame.iloc[bounds[:-1]].reset_index(drop=True).copy()
    won = frame.won.to_numpy(float)
    for column in columns:
        p = frame[column].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            races[f"loss_{column}"] = -np.add.reduceat(np.where(won > 0, np.log(p), 0.), bounds[:-1])
    races["winners"] = np.add.reduceat(won, bounds[:-1])
    races["rows"] = np.diff(bounds)
    return races


def _use_current_store_if_study_store_is_gone():
    """The dated study store (modelling_feature_store_20261109) was deleted on 12 November.

    The screens read only identity and card columns from it (trainer, sire, form and the like), keyed
    by race and horse. When it is absent, read those from the current feature store instead. Forecasts,
    labels and prices still come from the study's own files, so earlier results are unaffected.
    """
    from pathlib import Path
    from research.meta_learner import screen_candidates as sc
    try:
        missing = not sc.store_file().exists()
    except Exception:
        missing = True
    if missing:
        current = Path("data/processed/feature_store/featured.parquet")
        sc.store_file = lambda: current


_use_current_store_if_study_store_is_gone()
