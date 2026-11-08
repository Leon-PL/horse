"""Win-only adapter over the existing objectives, weights and calibration.

All statistical controls belong to ModelSpec, not process-global overrides.
Prediction uses trees and a saved temperature; it never fits or reads sources.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feature_selection import race_groups
from src.model import compute_recency_sample_weights
from src.betfair import market_prob_bsp_preferred
from src.probability_calibration import fit_temperature
from src.race_objectives import fit_boosted, raw_scores, race_bounds, race_softmax, training_objective
from src.modelling.data import numeric_matrix, ordered_frame
from src.modelling.spec import ModelSpec


def _teacher(frame):
    # Shared BSP-then-SP complete-field teacher. It reads columns only.
    return market_prob_bsp_preferred(frame).astype(np.float32)


@dataclass
class WinModel:
    spec: ModelSpec
    estimator: object = None
    temperature: float = 1.
    lineage: dict = None

    def _matrix_input(self, frame, matrix, matrix_columns):
        if matrix is None:
            if matrix_columns is not None:
                raise ValueError("Matrix column order supplied without a matrix")
            return None
        if matrix_columns is None or tuple(matrix_columns) != self.spec.features:
            raise ValueError("Explicit matrix column order must match the recipe")
        x = np.asarray(matrix)
        if x.shape != (len(frame), len(self.spec.features)):
            raise ValueError("Matrix shape differs from context rows or recipe features")
        if not (np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.bool_)) or np.iscomplexobj(x):
            raise ValueError("Matrix inputs must be real numeric values")
        x = x.astype(np.float64, copy=False)
        for first in range(0, len(x), 8192):
            if np.isinf(x[first:first + 8192]).any():
                raise ValueError("Model inputs may be missing, but cannot contain infinities")
        for column, name in enumerate(self.spec.features):
            if name in frame:
                if not pd.api.types.is_numeric_dtype(frame[name]):
                    raise ValueError(f"Matrix context feature {name} must be numeric")
                expected = frame[name].to_numpy(dtype=float, na_value=np.nan)
                if not np.array_equal(x[:, column], expected, equal_nan=True):
                    raise ValueError(f"Matrix values disagree with context feature {name}")
        return x

    def _fit_estimator(self, frame, *, capacity, threads, weights=None, refit=True, matrix=None):
        x = numeric_matrix(frame, self.spec.features) if matrix is None else matrix
        groups = race_groups(frame.race_id.to_numpy())
        y = frame.won.to_numpy(dtype=float)
        target = y.copy()
        if self.spec.soft_label_blend:
            teacher = _teacher(frame)
            target = np.where(teacher > 0, self.spec.soft_label_blend * teacher + (1 - self.spec.soft_label_blend) * y, y)
        if weights is None:
            weights = compute_recency_sample_weights(frame.race_date, half_life_days=self.spec.half_life_days,
                seasonal_boost=self.spec.seasonal_boost, decay_shape=self.spec.decay_shape)
        if self.spec.objective == "exploded":
            loss = training_objective(groups, frame.finish_position.to_numpy(), target, weights, self.spec.stage_weights)
            objective, metric, sample_weight = loss.objective, loss.metric, None
            label = y
        else:
            def objective(lo, hi):
                return "cross_entropy"
            label, sample_weight = target, weights
            def metric(lo, hi):
                g = race_groups(frame.race_id.to_numpy()[lo:hi])
                starts = np.r_[0, np.cumsum(g)[:-1]]
                def evaluate(_truth, p):
                    p = p / np.repeat(np.add.reduceat(p, starts), g)
                    return "race_log_loss", float(-np.sum(y[lo:hi] * np.log(np.clip(p, 1e-12, 1))) / len(g)), False
                return evaluate
        fitted = fit_boosted(x, label, frame.race_id.to_numpy(), frame.race_date.to_numpy(),
            objective=objective, metric=metric, sample_weight=sample_weight, capacity=capacity,
            params=dict(self.spec.params), seed=self.spec.seed, n_jobs=threads,
            early_stopping_rounds=self.spec.early_stopping_rounds, purge_days=self.spec.purge_days, refit=refit,
            screen_params=dict(self.spec.screen_params))
        # Strip custom loss closures and their training arrays before saving.
        if self.spec.objective == "exploded":
            fitted.objective = fitted._objective = "exploded_race_softmax"
        return fitted

    def fit(self, frame, *, dataset_start=None, capacity="native", threads=1, matrix=None, matrix_columns=None):
        self.estimator, self.lineage, self.temperature = None, None, 1.
        if threads < 1:
            raise ValueError("threads must be positive")
        matrix = self._matrix_input(frame, matrix, matrix_columns)
        ordered, positions = ordered_frame(frame)
        if matrix is not None and not np.array_equal(positions, np.arange(len(frame))):
            matrix = matrix[positions]
        beginning = pd.Timestamp(dataset_start if dataset_start is not None else ordered.race_date.min())
        cutoff = beginning + pd.DateOffset(months=self.spec.burn_in_months)
        warmed = ordered.race_date.ge(cutoff).to_numpy()
        if matrix is not None and warmed.any():
            matrix = matrix[int(np.flatnonzero(warmed)[0]):]
        ordered = ordered.loc[warmed].reset_index(drop=True)
        if ordered.empty:
            raise ValueError("Insufficient history after burn-in")
        if matrix is None:
            numeric_matrix(ordered, self.spec.features)
        groups = race_groups(ordered.race_id.to_numpy())
        starts = np.r_[0, np.cumsum(groups)[:-1]]
        bnd = int(len(groups) * (1 - self.spec.calibration_fraction))
        if not 0 < bnd < len(groups):
            raise ValueError("Insufficient history for chronological calibration")
        cal_start = ordered.race_date.iloc[starts[bnd]]
        train_rows = ordered.race_date.lt(cal_start) & ordered.race_date.le(cal_start - pd.Timedelta(days=self.spec.purge_days))
        cal_rows = ordered.race_date.ge(cal_start)
        if train_rows.sum() < 100 or cal_rows.sum() < 50:
            raise ValueError("Insufficient history for chronological calibration")
        weights = compute_recency_sample_weights(ordered.race_date, half_life_days=self.spec.half_life_days,
            seasonal_boost=self.spec.seasonal_boost, decay_shape=self.spec.decay_shape)
        earlier, calibration = ordered.loc[train_rows].reset_index(drop=True), ordered.loc[cal_rows].reset_index(drop=True)
        early_matrix = None if matrix is None else matrix[:int(train_rows.sum())]
        cal_matrix = None if matrix is None else matrix[int(np.flatnonzero(cal_rows)[0]):]
        temporary = self._fit_estimator(earlier, capacity=capacity, threads=threads, weights=weights[train_rows], refit=False,
                                        matrix=early_matrix)
        p = race_softmax(self._scores(temporary, calibration, matrix=cal_matrix), race_bounds(calibration.race_id))
        temperature = fit_temperature(p, race_groups(calibration.race_id), calibration.won.to_numpy())
        estimator = self._fit_estimator(ordered, capacity=capacity, threads=threads, weights=weights, matrix=matrix)
        self.estimator, self.temperature = estimator, temperature
        self.lineage = {"recipe": self.spec.to_dict(), "recipe_signature": self.spec.signature,
            "training_first_date": str(ordered.race_date.min().date()),
            "training_last_date": str(ordered.race_date.max().date()),
            "training_runners": len(ordered), "training_races": len(groups), "capacity": capacity,
            "calibration_first_date": str(calibration.race_date.min().date()),
            "calibration_last_date": str(calibration.race_date.max().date()),
            "calibration_fit_boundary": str(earlier.race_date.max().date()), "temperature": temperature,
            "selected_rounds": int(getattr(estimator, "n_estimators_", estimator.n_estimators))}
        return self

    def predict(self, frame, *, matrix=None, matrix_columns=None):
        if self.estimator is None:
            raise ValueError("Win model is not fitted")
        matrix = self._matrix_input(frame, matrix, matrix_columns)
        ordered, positions = ordered_frame(frame, labelled=False)
        if matrix is not None and not np.array_equal(positions, np.arange(len(frame))):
            matrix = matrix[positions]
        scores = self._scores(self.estimator, ordered, matrix=matrix)
        p = race_softmax(scores / self.temperature, race_bounds(ordered.race_id))
        out = np.empty(len(frame), dtype=float)
        out[positions] = p
        return out

    def _scores(self, estimator, frame, *, matrix=None):
        x = numeric_matrix(frame, self.spec.features) if matrix is None else matrix
        if self.spec.objective == "exploded":
            return raw_scores(estimator, x)
        return np.log(np.clip(estimator.predict(x), np.finfo(float).tiny, 1.))
