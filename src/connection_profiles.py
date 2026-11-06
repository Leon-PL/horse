"""Learned style profiles for trainers, jockeys and sires (experiment 060).

LightGBM sees a trainer only through hand-built strike rates. This learns a
small vector per trainer (jockey, sire) from how its runners did against the
market in different situations, so a yard that "targets big handicaps" or
whose "debutants come on for the run" is described by where it beats or
misses expectation.

Model, fitted by alternating ridge least squares (numpy only)::

    won - market_prob  ~  b[e] + u[e] . sum_{c active} v[c]

``e`` is the entity, ``c`` the runner's context columns (discipline, debut /
second run / new yard / layoff, handicap, class, age, trip, field size,
season, going). Ridge shrinks small-sample entities to zero.

Factor coordinates rotate freely between fits, so raw vectors are not
comparable across quarterly refits. Only rotation-invariant quantities become
inputs: the profile's prediction for the runner's own context, the entity's
bias, named-context projections (e.g. this trainer's debutants) and the
entity's sample size.

Leakage: :func:`quarterly_features` refits for every quarter on races before
it, minus the purge gap (``wf_eval.expanding_period_folds``), and scores only
that quarter's runners, so no row's profile includes its own result. Research
only, not part of the feature build: if adopted it needs an FE stage
registered in ``_FE_SOURCE_FILES``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.wf_eval import expanding_period_folds

RANK = 4
RIDGE_ENTITY = 30.0      # in runner-equivalents at zero residual
RIDGE_CONTEXT = 10.0
ITERATIONS = 8
HALF_LIFE_DAYS = 730.0

CONTEXT_GROUPS = {
    "trainer": ["discipline", "state", "handicap", "class", "age", "trip", "field", "season"],
    "jockey": ["discipline", "handicap", "class", "age", "trip", "field"],
    "sire": ["discipline", "trip", "going", "age", "debut"],
}
#: Named-context projections offered as inputs (entity, context column).
PROJECTIONS = [("trainer", "state=debut"), ("trainer", "state=second_run"), ("trainer", "state=new_yard"),
               ("sire", "debut=debut")]


def context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Categorical context per runner from pre-race race and runner facts.

    *frame* needs race_type, surface, race_name, going, race_class, handicap,
    age, distance_furlongs, num_runners, race_date, horse_prev_races,
    days_since_last_race and new_yard (bool).
    """
    bumper = frame.race_name.fillna("").str.contains(r"NH Flat|Bumper|Flat Race", case=False, regex=True)
    flat = frame.race_type.eq("Flat")
    aw = ~frame.surface.fillna("Turf").eq("Turf")
    discipline = np.select([flat & bumper, flat & aw, flat, frame.race_type.eq("Hurdle"), frame.race_type.eq("Chase")],
                           ["nhf", "flat_aw", "flat_turf", "hurdle", "chase"], default="other")
    prev = pd.to_numeric(frame.horse_prev_races, errors="coerce")
    layoff = pd.to_numeric(frame.days_since_last_race, errors="coerce")
    state = np.select([prev.eq(0), frame.new_yard.astype(bool), prev.eq(1), layoff.ge(180), layoff.ge(60)],
                      ["debut", "new_yard", "second_run", "layoff_long", "layoff_mid"], default="fresh")
    cls = pd.to_numeric(frame.race_class.astype(str).replace("Nan", np.nan), errors="coerce")
    age = pd.to_numeric(frame.age, errors="coerce")
    trip = pd.to_numeric(frame.distance_furlongs, errors="coerce")
    field = pd.to_numeric(frame.num_runners, errors="coerce")
    going = frame.going.fillna("Unknown")
    return pd.DataFrame({
        "discipline": discipline,
        "state": state,
        "debut": np.where(prev.eq(0), "debut", np.where(prev.eq(1), "second_run", "raced")),
        "handicap": np.where(pd.to_numeric(frame.handicap, errors="coerce").eq(1), "hcp", "non_hcp"),
        "class": np.select([cls.le(2), cls.le(4), cls.ge(5)], ["c1_2", "c3_4", "c5_plus"], default="unknown"),
        "age": np.select([age.le(2), age.eq(3)], ["2yo", "3yo"], default="4plus"),
        "trip": np.select([trip.le(7), trip.le(10), trip.le(14), trip.le(20)], ["sprint", "mile", "middle", "staying"],
                          default="long"),
        "field": np.select([field.le(7), field.le(12)], ["small", "medium"], default="large"),
        "season": "q" + pd.to_datetime(frame.race_date).dt.quarter.astype(str),
        "going": np.select([going.isin(["Standard", "Slow"]) | aw, going.eq("Heavy"), going.isin(["Soft", "Yielding"])],
                           ["aw", "heavy", "soft"], default="good_fast"),
    }, index=frame.index)


def new_yard(frame: pd.DataFrame) -> pd.Series:
    """True where the horse's previous run (in the frame) was for another trainer."""
    order = frame.sort_values(["horse_id", "race_date"]).index
    previous = frame.loc[order].groupby("horse_id", sort=False).trainer.shift(1)
    flag = previous.notna() & previous.ne(frame.loc[order, "trainer"])
    return flag.reindex(frame.index).fillna(False).astype(bool)


def design(contexts: pd.DataFrame, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    """0/1 indicator matrix over every ``group=value`` column, and its names."""
    parts = [pd.get_dummies(contexts[g], prefix=g, prefix_sep="=", dtype=np.float32) for g in groups]
    matrix = pd.concat(parts, axis=1)
    return matrix.to_numpy(np.float32), list(matrix.columns)


@dataclass
class Profiles:
    names: list[str]          # context column names
    entities: pd.Index        # entity labels; position = code
    theta: np.ndarray         # (n_entities, 1 + rank): bias, then u
    context: np.ndarray       # (n_columns, rank): v
    runs: np.ndarray          # training rows per entity

    def codes(self, labels) -> np.ndarray:
        return self.entities.get_indexer(pd.Index(labels))

    def score(self, labels, active: np.ndarray) -> dict[str, np.ndarray]:
        code = self.codes(labels)
        known = code >= 0
        theta = np.where(known[:, None], self.theta[np.maximum(code, 0)], 0.0)
        z = active @ self.context
        return {"fit": theta[:, 0] + np.einsum("ij,ij->i", theta[:, 1:], z), "bias": theta[:, 0],
                "runs": np.where(known, np.log1p(self.runs[np.maximum(code, 0)]), 0.0), "theta": theta}

    def projection(self, theta: np.ndarray, column: str) -> np.ndarray:
        if column not in self.names:
            return np.zeros(len(theta))
        return theta[:, 0] + theta[:, 1:] @ self.context[self.names.index(column)]


def fit_profiles(labels, active: np.ndarray, names: list[str], residual, weights, *, rank: int = RANK,
                 ridge_entity: float = RIDGE_ENTITY, ridge_context: float = RIDGE_CONTEXT,
                 iterations: int = ITERATIONS, seed: int = 0, chunk: int = 50_000) -> Profiles:
    """Alternating ridge least squares for ``residual ~ b[e] + u[e] . (active @ V)``."""
    labels = pd.Series(np.asarray(labels, dtype=object))
    keep = labels.notna() & labels.astype(str).str.strip().ne("")
    entities = pd.Index(labels[keep].unique())
    codes = entities.get_indexer(labels.where(keep))
    w = np.where(codes >= 0, np.asarray(weights, dtype=np.float64), 0.0)
    codes = np.maximum(codes, 0)
    r = np.asarray(residual, dtype=np.float64)
    n_entities, n_cols = len(entities), active.shape[1]
    rng = np.random.default_rng(seed)
    context = rng.normal(0.0, 0.1, (n_cols, rank))
    theta = np.zeros((n_entities, rank + 1))
    eye_entity = np.eye(rank + 1) * ridge_entity
    for _ in range(iterations):
        x = np.column_stack([np.ones(len(r)), active @ context])
        normal = np.empty((n_entities, rank + 1, rank + 1))
        for a in range(rank + 1):
            for b in range(a, rank + 1):
                normal[:, a, b] = normal[:, b, a] = np.bincount(codes, w * x[:, a] * x[:, b], minlength=n_entities)
        rhs = np.column_stack([np.bincount(codes, w * x[:, a] * r, minlength=n_entities) for a in range(rank + 1)])
        theta = np.linalg.solve(normal + eye_entity, rhs[:, :, None])[:, :, 0]

        bias, u = theta[codes, 0], theta[codes, 1:]
        target = r - bias
        size = n_cols * rank
        gram, moment = np.zeros((size, size)), np.zeros(size)
        for lo in range(0, len(r), chunk):
            sl = slice(lo, lo + chunk)
            d = (active[sl, :, None] * u[sl, None, :]).reshape(-1, size)
            dw = d * w[sl, None]
            gram += d.T @ dw
            moment += dw.T @ target[sl]
        context = np.linalg.solve(gram + np.eye(size) * ridge_context, moment).reshape(n_cols, rank)
    runs = np.bincount(codes, (w > 0).astype(float), minlength=n_entities)
    return Profiles(list(names), entities, theta, context, runs)


def feature_names() -> list[str]:
    names = [f"{e}_prof_{k}" for e in CONTEXT_GROUPS for k in ("fit", "bias", "runs")]
    return names + [f"{e}_prof_{c.split('=')[1]}" for e, c in PROJECTIONS]


def quarterly_features(frame: pd.DataFrame, residual, *, log=None) -> pd.DataFrame:
    """Profile inputs for every runner from fits on earlier quarters only.

    *frame* is the full history in date order with the :func:`context_columns`
    inputs plus trainer, jockey, sire; *residual* is won minus the market
    probability (NaN where the race is unpriced: those rows do not fit).
    """
    log = log or (lambda *a: None)
    contexts = context_columns(frame)
    residual = np.asarray(residual, dtype=np.float64)
    dates = pd.to_datetime(frame.race_date).reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=frame.index, columns=feature_names())
    matrices = {e: design(contexts, groups) for e, groups in CONTEXT_GROUPS.items()}
    for fold in expanding_period_folds(dates, min_train_periods=1):
        fit_rows = fold.train & np.isfinite(residual)
        age = (dates[fold.train].max() - dates[fit_rows]).dt.days.to_numpy()
        weights = np.exp(-np.log(2) * age / HALF_LIFE_DAYS)
        test = np.flatnonzero(fold.test)
        for entity, (active, names) in matrices.items():
            labels = frame[entity].to_numpy(dtype=object)
            profiles = fit_profiles(labels[fit_rows], active[fit_rows], names, residual[fit_rows], weights)
            scored = profiles.score(labels[test], active[test])
            for key in ("fit", "bias", "runs"):
                out.iloc[test, out.columns.get_loc(f"{entity}_prof_{key}")] = scored[key]
            for e, column in PROJECTIONS:
                if e == entity:
                    out.iloc[test, out.columns.get_loc(f"{e}_prof_{column.split('=')[1]}")] = \
                        profiles.projection(scored["theta"], column)
        log("%s: profiles fitted on %d rows, scored %d", fold.label, int(fit_rows.sum()), len(test))
    return out
