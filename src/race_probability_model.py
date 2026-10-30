"""Bounded research predictors trained on one winner distribution per race.

No market inputs or production promotion. Native data preparation, temporal
splits and serving calibration are shared with RacePredictor.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.forecast_evaluation import validate_probabilities
from src.model import RacePredictor, compute_recency_sample_weights, prepare_multi_target_data
from src.wf_eval import purge_days

logger = logging.getLogger(__name__)


def race_softmax(scores, groups):
    scores = np.asarray(scores, dtype=float)
    sizes = np.asarray(groups)
    if (scores.ndim != 1 or sizes.ndim != 1 or not len(sizes)
            or not np.isfinite(scores).all() or not np.isfinite(sizes).all()
            or np.any(sizes < 2) or np.any(sizes != np.floor(sizes)) or sizes.sum() != len(scores)):
        raise ValueError("Finite scores and complete integer race sizes >= 2 required")
    sizes = sizes.astype(int)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    centred = scores - np.repeat(np.maximum.reduceat(scores, starts), sizes)
    weights = np.exp(centred)
    return weights / np.repeat(np.add.reduceat(weights, starts), sizes)


def race_loss_gradient(scores, winners, groups, race_weights):
    """Weighted mean race cross entropy and its exact score gradient."""
    scores, winners, groups = np.asarray(scores), np.asarray(winners), np.asarray(groups, dtype=int)
    probabilities = race_softmax(scores, groups)
    _, _, _, starts = validate_probabilities(probabilities, winners, groups)
    weights = np.asarray(race_weights, dtype=float)
    if weights.shape != groups.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("One finite positive weight per race required")
    centred = scores - np.repeat(np.maximum.reduceat(scores, starts), groups)
    losses = np.log(np.add.reduceat(np.exp(centred), starts)) - np.add.reduceat(winners * centred, starts)
    gradient = (probabilities - winners) * np.repeat(weights, groups) / weights.sum()
    return float(np.dot(weights, losses) / weights.sum()), gradient


def race_objective(groups, race_weights):
    groups = np.asarray(groups, dtype=int)
    weights = np.repeat(np.asarray(race_weights, dtype=float), groups)

    def objective(scores, dataset):
        probabilities = race_softmax(scores, groups)
        # Custom objectives own weighting. A race's sum of label cross entropy
        # has equal weight regardless of field size; do not divide by runners.
        gradient = (probabilities - dataset.get_label()) * weights
        hessian = np.maximum(probabilities * (1 - probabilities), 1e-9) * weights
        return gradient, hessian

    return objective


class RaceProbabilityPredictor:
    def __init__(self, kind, feature_cols, *, params=None, threads=4, seed=42, linear_l2=.01):
        if kind not in ("conditional_logit", "race_softmax_gbm"):
            raise ValueError("Unknown fixed race-probability model")
        self.kind, self.feature_cols = kind, list(feature_cols)
        self.params = dict(config.CLASSIFIER_PARAMS if params is None else params)
        self.threads, self.seed, self.linear_l2 = int(threads), int(seed), float(linear_l2)
        self.model = None
        self.calibrator = RacePredictor(win_calibration_method="platt_isotonic", random_seed=seed)
        self.eval_split_info = {}

    @staticmethod
    def _weights(groups, dates):
        starts = np.r_[0, np.cumsum(groups)[:-1]]
        return compute_recency_sample_weights(pd.DatetimeIndex(dates)[starts], half_life_days=180,
                                             seasonal_boost=0, decay_shape="exp")

    def _fit_linear(self, X, y, groups, weights):
        transform = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True),
                                  StandardScaler())
        clean = np.asarray(X, dtype=float).copy()
        clean[~np.isfinite(clean)] = np.nan
        matrix = transform.fit_transform(clean)

        def objective(beta):
            loss, score_gradient = race_loss_gradient(matrix @ beta, y, groups, weights)
            return loss + .5 * self.linear_l2 * np.dot(beta, beta), matrix.T @ score_gradient + self.linear_l2 * beta

        fitted = minimize(objective, np.zeros(matrix.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6})
        if not fitted.success or not np.isfinite(fitted.fun):
            raise RuntimeError(f"Conditional-logit optimisation failed: {fitted.message}")
        logger.info("Conditional logit converged after %s iterations", fitted.nit)
        return {"transform": transform, "beta": fitted.x, "iterations": int(fitted.nit)}

    def _fit_base(self, X, y, groups, dates, *, refit, race_weights=None):
        starts = np.r_[0, np.cumsum(groups)[:-1]]
        race_dates = pd.DatetimeIndex(dates)[starts]
        split = RacePredictor._early_stop_split(groups, race_dates, purge_days=purge_days())
        end, validation = split if split is not None else (len(X), len(X))
        fit_end = len(X) if refit and self.kind == "conditional_logit" else end
        fit_groups = groups[np.cumsum(groups) <= fit_end]
        weights = self._weights(groups, dates) if race_weights is None else np.asarray(race_weights)
        if self.kind == "conditional_logit":
            model = self._fit_linear(X[:fit_end], y[:fit_end], fit_groups, weights[:len(fit_groups)])
            return model, {"training_runners": fit_end, "rounds": None,
                           "last_training_date": str(pd.Timestamp(dates[fit_end - 1]).date())}
        params = {key: value for key, value in self.params.items() if key != "n_estimators"}
        # Preserve complete fields in every tree fit; per-runner bagging would
        # change the race weighting. This is a declared training-design change.
        params.update(objective=race_objective(fit_groups, weights[:len(fit_groups)]), metric="None",
                      subsample=1., bagging_freq=0, num_threads=self.threads, seed=self.seed, verbosity=-1)
        maximum = int(self.params.get("n_estimators", 2000))
        dataset = lgb.Dataset(X[:fit_end], label=y[:fit_end])
        callbacks, validation_sets = [], None
        metric = None
        if split is not None:
            val_groups = groups[starts >= validation]
            validation_sets = [lgb.Dataset(X[validation:], label=y[validation:], reference=dataset)]

            def metric(scores, data):
                p = race_softmax(scores, val_groups)
                loss = -float(np.sum(data.get_label() * np.log(np.clip(p, 1e-12, 1))) / len(val_groups))
                return "race_log_loss", loss, False

            callbacks = [lgb.early_stopping(int(getattr(config, "EARLY_STOPPING_ROUNDS", 100)), verbose=False)]
        selected = lgb.train(params, dataset, num_boost_round=maximum, valid_sets=validation_sets,
                             feval=metric, callbacks=callbacks)
        rounds = selected.best_iteration or selected.current_iteration()
        del dataset, validation_sets
        if refit and split is not None:
            logger.info("Race-softmax refit: %s runners, %s selected trees", len(X), rounds)
            params["objective"] = race_objective(groups, weights)
            selected = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)
            fit_end = len(X)
        return selected, {"training_runners": fit_end, "rounds": int(rounds),
                          "last_training_date": str(pd.Timestamp(dates[fit_end - 1]).date())}

    def _raw(self, model, X, groups):
        if self.kind == "conditional_logit":
            clean = np.asarray(X, dtype=float).copy()
            clean[~np.isfinite(clean)] = np.nan
            scores = model["transform"].transform(clean) @ model["beta"]
        else:
            scores = model.predict(X, raw_score=True, num_threads=self.threads)
        return race_softmax(scores, groups)

    def fit_as_of(self, frame, as_of, *, dataset_start=None):
        cutoff = pd.Timestamp(as_of)
        if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
            raise ValueError("as_of must be an unambiguous calendar date")
        dates = pd.to_datetime(frame.race_date, errors="coerce")
        if dates.isna().any():
            raise ValueError("Missing race dates")
        data = prepare_multi_target_data(frame.loc[dates.dt.normalize() <= cutoff].copy(),
                                         dataset_start=dataset_start, training_only=True)
        if data["feature_cols"] != self.feature_cols:
            raise ValueError("Native data preparation changed registered features or order")
        X, y, groups = data["X_train"], data["y_train_won"], data["groups_train"]
        row_dates = pd.to_datetime(data["train_df"].race_date).to_numpy()
        starts = np.r_[0, np.cumsum(groups)[:-1]]
        validate_probabilities(race_softmax(np.zeros(len(y)), groups), y, groups)
        boundary = int(len(groups) * .8)
        if boundary < 20 or len(groups) - boundary < 5:
            raise ValueError("Insufficient chronological calibration history")
        cal_start = starts[boundary]
        allowed = row_dates[starts] <= pd.Timestamp(row_dates[cal_start]) - pd.Timedelta(days=purge_days())
        cal_train_groups = np.flatnonzero(allowed & (np.arange(len(groups)) < boundary))
        if not len(cal_train_groups):
            raise ValueError("No purged calibration training history")
        cal_end = int(np.cumsum(groups)[cal_train_groups[-1]])
        weights = self._weights(groups, row_dates)
        self.model, full_info = self._fit_base(X, y, groups, row_dates, refit=True, race_weights=weights)
        calibration_model, cal_info = self._fit_base(X[:cal_end], y[:cal_end], groups[:len(cal_train_groups)],
                                                     row_dates[:cal_end], refit=False,
                                                     race_weights=weights[:len(cal_train_groups)])
        cal_probs = self._raw(calibration_model, X[cal_start:], groups[boundary:])
        self.calibrator._fit_win_calibrator(cal_probs, groups[boundary:], data["fp_train"][cal_start:])
        self.eval_split_info = {"mode": "fit_as_of", "kind": self.kind, "as_of": str(cutoff.date()),
                               "training_runners": len(X), "training_races": len(groups),
                               "training_first_date": str(pd.Timestamp(row_dates[0]).date()),
                               "training_last_date": str(pd.Timestamp(row_dates[-1]).date()),
                               "full_fit": full_info, "calibration_base": cal_info,
                               "calibration_first_date": str(pd.Timestamp(row_dates[cal_start]).date()),
                               "calibration_last_date": str(pd.Timestamp(row_dates[-1]).date()),
                               "performance_claim": "None; separately score frozen later forecasts"}
        return dict(self.eval_split_info)

    def predict_races(self, frame, *, win_only=True):
        if self.model is None:
            raise ValueError("Fit or load a research model first")
        ordered = frame.sort_values(["race_id", "horse_name"], kind="stable").copy()
        if ordered.duplicated(["race_id", "horse_name"]).any():
            raise ValueError("Duplicate runner identities")
        groups = ordered.groupby("race_id", sort=False).size().to_numpy()
        raw = self._raw(self.model, ordered[self.feature_cols].to_numpy(dtype=np.float32), groups)
        ordered["win_probability"] = self.calibrator._calibrate_win_probs(raw, groups)
        return ordered[["race_id", "horse_name", "win_probability"]]

    def save(self, path):
        state = dict(self.__dict__)
        if self.kind == "race_softmax_gbm":
            state["model"] = self.model.model_to_string()
        joblib.dump(state, Path(path))

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        predictor = cls(state["kind"], state["feature_cols"])
        predictor.__dict__.update(state)
        if predictor.kind == "race_softmax_gbm":
            predictor.model = lgb.Booster(model_str=state["model"])
        return predictor
