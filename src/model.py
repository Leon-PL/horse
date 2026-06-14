"""
Model Module
=============
Win and Place classifiers for horse race prediction.

Supports:
- Win Classifier  — calibrated P(win) for value betting
- Place Classifier — calibrated P(place) for each-way betting
"""

import os
import logging
import warnings
import importlib
from typing import Optional, Callable

# Suppress sklearn validation warning when a saved model was fitted with
# feature names but prediction input is a plain numpy array.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    ndcg_score,
)
from sklearn.model_selection import GroupShuffleSplit
# Gradient-boosting frameworks are imported lazily: importing lightgbm
# alone can take 10-30 seconds on Windows (DLL scanning), and the
# Streamlit app imports this module at startup. Training/prediction
# entry points call _ensure_lightgbm()/_ensure_xgboost() before use.
LGBMRegressor = None
LGBMClassifier = None
LGBMRanker = None
XGBRegressor = None
XGBClassifier = None
CatBoostRegressor = None
CatBoostClassifier = None


def _ensure_lightgbm() -> None:
    global LGBMRegressor, LGBMClassifier, LGBMRanker
    if LGBMClassifier is None:
        import lightgbm as lgb
        LGBMRegressor = lgb.LGBMRegressor
        LGBMClassifier = lgb.LGBMClassifier
        LGBMRanker = lgb.LGBMRanker


def _ensure_xgboost() -> None:
    global XGBRegressor, XGBClassifier
    if XGBClassifier is None:
        import xgboost as xgb
        XGBRegressor = xgb.XGBRegressor
        XGBClassifier = xgb.XGBClassifier


def __getattr__(name: str):
    """PEP 562 lazy attributes — keeps ``from src.model import
    _FocalLGBMClassifier`` (used by pickled models) working without
    paying the lightgbm import at module load."""
    if name == "_FocalLGBMClassifier":
        return _get_focal_lgbm_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
Pool = None

import config
from src.bet_settlement import value_bet_selection as _value_bet_selection
# Re-exported for backwards compatibility — the betting simulation now
# lives in src.bet_analysis.
from src.bet_analysis import analyse_test_set, max_drawdown as _max_drawdown  # noqa: F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _off_time_to_seconds(off_time: pd.Series) -> pd.Series:
    """Parse off-time strings to seconds after midnight with safe fallbacks."""
    s = off_time.astype(str).str.strip()

    # Fast path for HH:MM or HH:MM:SS
    parts = s.str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?")
    h = pd.to_numeric(parts["h"], errors="coerce")
    m = pd.to_numeric(parts["m"], errors="coerce")
    sec = pd.to_numeric(parts["s"], errors="coerce").fillna(0)

    bad = h.isna() | m.isna()
    if bad.any():
        # Fallback for full datetime-like strings.
        try:
            dt = pd.to_datetime(s[bad], errors="coerce", format="mixed")
        except TypeError:
            dt = pd.to_datetime(s[bad], errors="coerce")
        h.loc[bad] = dt.dt.hour
        m.loc[bad] = dt.dt.minute
        sec.loc[bad] = dt.dt.second

    h = h.fillna(12).clip(0, 23)
    m = m.fillna(0).clip(0, 59)
    sec = sec.fillna(0).clip(0, 59)
    return h * 3600 + m * 60 + sec


class _IdentityScaler:
    """No-op scaler for tree-based models (keeps API compatibility)."""

    def fit_transform(self, X):
        return np.asarray(X)

    def transform(self, X):
        return np.asarray(X)

    def __reduce__(self):
        # Streamlit re-execs app.py via exec(), which creates shadow copies
        # of every imported name.  Pickle validates that the function object
        # stored in __reduce__ is identical (by id) to the one reachable via
        # its __qualname__ in its __module__.  Under Streamlit the ids
        # diverge → PicklingError.
        #
        # Fix: force-resolve the canonical function from the *real* module
        # at reduce-time so pickle's identity check always passes.
        import importlib as _il
        _mod = _il.import_module("src.model")
        return (getattr(_mod, "_rebuild_identity_scaler"), ())


def _rebuild_identity_scaler() -> "_IdentityScaler":
    """Top-level factory so pickle/joblib can reconstruct _IdentityScaler."""
    from src.model import _IdentityScaler as _Cls
    return _Cls()


def _is_truthy_flag(value) -> bool:
    """Interpret LightGBM-style param flags stored as bool/int/str."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _model_uses_linear_tree(model) -> bool:
    """Detect whether a model was trained with LightGBM linear leaves."""
    candidates = []

    try:
        if hasattr(model, "get_params"):
            params = model.get_params()
            if isinstance(params, dict):
                candidates.append(params)
    except Exception:
        pass

    for attr in ("_other_params", "params"):
        params = getattr(model, attr, None)
        if isinstance(params, dict):
            candidates.append(params)

    for booster_attr in ("booster_", "_Booster", "_booster"):
        booster = getattr(model, booster_attr, None)
        if booster is not None:
            params = getattr(booster, "params", None)
            if isinstance(params, dict):
                candidates.append(params)

    for params in candidates:
        if "linear_tree" in params:
            return _is_truthy_flag(params["linear_tree"])

    return False


def _predict_for_shap(model, X):
    """Return a 1-D output suitable for SHAP from classifiers."""
    if hasattr(model, "predict_proba"):
        pred = np.asarray(model.predict_proba(X))
        if pred.ndim == 2:
            return pred[:, -1]
        return pred.reshape(-1)

    pred = np.asarray(model.predict(X))
    if pred.ndim == 2:
        return pred[:, -1]
    return pred.reshape(-1)


def _compute_shap_matrix(
    model,
    X_scaled: pd.DataFrame,
    top_n_features: int,
    feature_cols: list[str],
) -> tuple[np.ndarray, str]:
    """Compute per-row SHAP values, using Kernel SHAP for linear-tree LGBM."""
    import shap

    if _model_uses_linear_tree(model):
        bg_n = min(len(X_scaled), 8)
        background = (
            X_scaled.sample(n=bg_n, random_state=config.RANDOM_SEED)
            if len(X_scaled) > bg_n
            else X_scaled.copy()
        )

        def _predict_fn(arr):
            arr_df = pd.DataFrame(arr, columns=feature_cols)
            return _predict_for_shap(model, arr_df)

        explainer = shap.KernelExplainer(_predict_fn, background.to_numpy())
        _shap_logger = logging.getLogger("shap")
        _prev_shap_level = _shap_logger.level
        _shap_logger.setLevel(logging.ERROR)
        try:
            try:
                shap_values = explainer.shap_values(
                    X_scaled.to_numpy(),
                    nsamples=min(512, max(128, top_n_features * 32)),
                    l1_reg=f"num_features({max(top_n_features * 2, 20)})",
                    silent=True,
                )
            except TypeError:
                shap_values = explainer.shap_values(
                    X_scaled.to_numpy(),
                    nsamples=min(512, max(128, top_n_features * 32)),
                    l1_reg=f"num_features({max(top_n_features * 2, 20)})",
                )
        finally:
            _shap_logger.setLevel(_prev_shap_level)
        method_label = "Kernel SHAP"
    else:
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            # Benign: SHAP warns that binary-classifier TreeExplainer output is
            # now a list of ndarray (one per class). The list/ndim-3 handling
            # below already selects the positive class, so this is just noise.
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer.*",
                category=UserWarning,
            )
            shap_values = explainer.shap_values(X_scaled)
        method_label = "Tree SHAP"

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    return shap_values, method_label


def _explain_race_with_shap(
    model,
    race_df: pd.DataFrame,
    feature_cols: list[str],
    scaler,
    top_n_features: int,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Shared SHAP explanation path for one race across model wrappers."""
    race_df = race_df.copy().reset_index(drop=True)

    missing = [c for c in feature_cols if c not in race_df.columns]
    for col in missing:
        race_df[col] = 0

    X = race_df[feature_cols].values
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols,
        index=race_df.index,
    )

    shap_values, method_label = _compute_shap_matrix(
        model, X_scaled, top_n_features, feature_cols,
    )

    explanations: dict[str, pd.DataFrame] = {}
    horse_names = race_df["horse_name"].values

    for i, name in enumerate(horse_names):
        sv = shap_values[i]
        fv = X_scaled.iloc[i]
        df_expl = pd.DataFrame({
            "feature": feature_cols,
            "shap_value": sv,
            "feature_value": fv,
        })
        df_expl["abs_shap"] = df_expl["shap_value"].abs()
        df_expl = df_expl.sort_values("abs_shap", ascending=False)
        explanations[name] = df_expl.head(top_n_features).drop(
            columns="abs_shap",
        )

    return explanations, method_label


def _require_catboost() -> None:
    global CatBoostRegressor, CatBoostClassifier, Pool
    if CatBoostRegressor is not None and CatBoostClassifier is not None and Pool is not None:
        return
    try:
        cb = importlib.import_module("catboost")
        CatBoostRegressor = cb.CatBoostRegressor
        CatBoostClassifier = cb.CatBoostClassifier
        Pool = cb.Pool
    except Exception as e:
        raise ImportError(
            "CatBoost is required for framework='cat'. Install with: pip install catboost"
        ) from e


def get_autotune_search_space(
    model_key: str,
    framework: str,
    *,
    include_recency: bool = True,
) -> list[dict]:
    """Return the Optuna search-space spec for a sub-model/framework pair."""
    framework = str(framework)
    specs: list[dict] = []

    if framework == "cat":
        specs.extend([
            {"name": "n_estimators", "kind": "int", "low": 100, "high": 1200, "step": 50},
            {"name": "depth", "kind": "int", "low": 4, "high": 10},
            {"name": "learning_rate", "kind": "float", "low": 0.01, "high": 0.3, "log": True},
            {"name": "l2_leaf_reg", "kind": "float", "low": 1e-3, "high": 20.0, "log": True},
            {"name": "random_strength", "kind": "float", "low": 0.0, "high": 2.0},
            {"name": "bagging_temperature", "kind": "float", "low": 0.0, "high": 2.0},
        ])
    else:
        specs.extend([
            {"name": "n_estimators", "kind": "int", "low": 100, "high": 3500, "step": 100},
            {"name": "max_depth", "kind": "int", "low": 3, "high": 12},
            {"name": "learning_rate", "kind": "float", "low": 0.005, "high": 0.3, "log": True},
            {"name": "subsample", "kind": "float", "low": 0.5, "high": 1.0, "step": 0.05},
            {"name": "colsample_bytree", "kind": "float", "low": 0.3, "high": 1.0, "step": 0.05},
            {"name": "reg_alpha", "kind": "float", "low": 1e-8, "high": 10.0, "log": True},
            {"name": "reg_lambda", "kind": "float", "low": 1e-8, "high": 10.0, "log": True},
        ])
        if framework == "xgb":
            specs.extend([
                {"name": "min_child_weight", "kind": "int", "low": 1, "high": 20},
                {"name": "gamma", "kind": "float", "low": 0.0, "high": 5.0, "step": 0.1},
            ])
        elif framework == "lgbm":
            specs.extend([
                {"name": "min_child_samples", "kind": "int", "low": 10, "high": 40},
                {"name": "num_leaves", "kind": "int", "low": 20, "high": 100},
                {"name": "linear_tree", "kind": "categorical", "choices": [True, False]},
            ])

    if include_recency and model_key in {"classifier", "place", "ranker"}:
        specs.extend([
            {"name": "recency_half_life_days", "kind": "int", "low": 60, "high": 360, "step": 30},
            {"name": "recency_seasonal_boost", "kind": "float", "low": 0.0, "high": 0.30, "step": 0.05},
            {"name": "recency_decay_shape", "kind": "categorical", "choices": ["exp", "poly", "linear"]},
        ])

    # Feature pruning params (shared across all frameworks)
    specs.extend([
        {"name": "FEATURE_PRUNE_FRACTION", "kind": "float", "low": 0.0, "high": 0.85, "step": 0.05},
        {"name": "FEATURE_CORR_THRESHOLD", "kind": "float", "low": 0.80, "high": 0.98, "step": 0.02},
    ])

    return specs


def _suggest_from_autotune_space(trial, specs: list[dict]) -> dict:
    """Sample a params dict from a search-space spec."""
    params: dict = {}
    for spec in specs:
        name = spec["name"]
        kind = spec["kind"]
        if kind == "fixed":
            params[name] = spec["value"]
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "int":
            kwargs = {}
            if spec.get("step") is not None:
                kwargs["step"] = spec["step"]
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], **kwargs)
        elif kind == "float":
            kwargs = {}
            if spec.get("step") is not None:
                kwargs["step"] = spec["step"]
            if spec.get("log"):
                kwargs["log"] = True
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
        else:
            raise ValueError(f"Unknown autotune spec kind: {kind}")
    return params


# ── Focal Loss ────────────────────────────────────────────────
def _focal_logloss_eval(y_true, y_pred):
    """Sigmoid-aware logloss for early stopping with custom objectives.

    LightGBM's built-in ``binary_logloss`` assumes predictions are
    already probabilities when the objective is ``custom``.  This eval
    function applies the sigmoid link first so the metric is correct.
    """
    p = 1.0 / (1.0 + np.exp(-np.asarray(y_pred, dtype=np.float64)))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    y = np.asarray(y_true, dtype=np.float64)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return "focal_logloss", loss, False  # name, value, is_higher_better


def _focal_binary_objective(y_true, y_pred):
    """Focal-loss objective for LightGBM binary classification.

    Down-weights easy negatives so the model focuses on ambiguous,
    competitive cases within each race.  Gradient is exact (Lin et al.
    2017); Hessian is the standard focal-weighted approximation for
    numerical stability.
    """
    gamma = getattr(config, "FOCAL_GAMMA", 2.0)
    _alpha_cfg = getattr(config, "FOCAL_ALPHA", "auto")
    if _alpha_cfg == "auto" or _alpha_cfg is None:
        # Up-weight the minority (positive) class: alpha = 1 - prevalence.
        # With ~10% winners, alpha ≈ 0.90 → minority gets 9× the weight,
        # compensating for the class imbalance.  Previously alpha was set
        # to prevalence itself (~0.10), which DOWN-weighted winners and
        # caused the model to converge to a constant baseline in 1 tree.
        _prev = float(np.mean(y_true))
        alpha = np.clip(1.0 - _prev, 0.5, 0.99)
    else:
        alpha = float(_alpha_cfg)

    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    log_p = np.log(p)
    log_1mp = np.log(1 - p)

    # Exact gradient
    grad_pos = alpha * (1 - p) ** gamma * (gamma * p * log_p + p - 1)
    grad_neg = (1 - alpha) * p ** gamma * (p - gamma * (1 - p) * log_1mp)
    grad = np.where(y_true == 1, grad_pos, grad_neg)

    # Approximate Hessian (stable)
    p_t = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    hess = alpha_t * (1 - p_t) ** gamma * p * (1 - p)
    hess = np.maximum(hess, 1e-7)

    return grad, hess


def _unpickle_focal_lgbm_classifier(state):
    """Stable reconstructor for _FocalLGBMClassifier.

    Pickle serialises this *function* by name (``src.model._unpickle_focal_lgbm_classifier``)
    rather than the class itself, so Streamlit module reloads — which create a new
    class object but keep the same module path — no longer cause a
    ``PicklingError: it's not the same object`` failure.
    """
    # Import fresh from the module so we always get the current class,
    # even if the module was reloaded since the instance was created.
    from src.model import _FocalLGBMClassifier as _cls  # noqa: PLC0415
    obj = object.__new__(_cls)
    if hasattr(obj, "__setstate__"):
        obj.__setstate__(state)
    else:
        obj.__dict__.update(state)
    return obj


_focal_lgbm_cls = None


def _get_focal_lgbm_class():
    """Build (once) the LGBMClassifier subclass with sigmoid-corrected
    ``predict_proba``.

    When a custom objective (e.g. focal loss) is used, the LightGBM
    booster returns raw logits; the subclass applies the sigmoid link so
    ``predict_proba`` returns valid probabilities. Defined lazily so
    importing src.model does not import lightgbm; unpickling reaches it
    through the module-level ``__getattr__``.
    """
    global _focal_lgbm_cls
    if _focal_lgbm_cls is not None:
        return _focal_lgbm_cls
    _ensure_lightgbm()

    class _FocalLGBMClassifier(LGBMClassifier):
        def predict_proba(self, X, **kwargs):
            # Go directly to the booster to avoid LGBMClassifier.predict →
            # predict_proba recursion loop.
            raw = self.booster_.predict(X, raw_score=True)
            p = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
            return np.column_stack([1 - p, p])

        def __copy__(self):
            # copy.copy() is called by LGBMClassifier.get_params() — must not
            # go through __reduce__ which triggers a module re-import chain.
            cls = type(self)
            obj = cls.__new__(cls)
            obj.__dict__.update(self.__dict__)
            return obj

        def __reduce__(self):
            # Use the stable module-level reconstructor instead of the class
            # reference directly — prevents PicklingError when Streamlit reloads
            # src.model between training and the joblib.dump call.
            state = self.__getstate__() if hasattr(self, "__getstate__") else self.__dict__.copy()
            return (_unpickle_focal_lgbm_classifier, (state,))

    _FocalLGBMClassifier.__module__ = __name__
    _FocalLGBMClassifier.__qualname__ = "_FocalLGBMClassifier"
    _focal_lgbm_cls = _FocalLGBMClassifier
    return _focal_lgbm_cls


# Columns to exclude from features (identifiers, targets, leaky features)
EXCLUDE_COLUMNS = [
    "race_id",
    "race_date",
    "horse_name",
    "jockey",
    "trainer",
    "track",
    "going",
    "race_class",
    "race_type",
    "finish_position",
    "won",
    "finish_time_secs",
    "lengths_behind",
    "season",
    # Raw odds column — never used as a feature directly, but kept
    # in the DataFrame for test-set betting analysis (value bets,
    # P&L, equity curves).  Derived odds features (implied_prob,
    # log_odds, etc.) are the actual model inputs.
    "odds",
    # Opening price from bet_movements — kept for market-drift / CLV
    # analysis, not a model input.
    "opening_odds",
    # Real-data identifier / text columns
    "horse_id",
    "jockey_id",
    "trainer_id",
    "off_time",
    "race_name",
    "distance_raw",
    "age_band",
    "form",
    "form_str",
    "sex",
    "headgear",
    "colour",
    "owner",
    "region",
    # Elo deltas encode the *current* race's outcome — they leak the
    # target for any model (ranking or classification).
    "horse_elo_delta",
    "jockey_elo_delta",
    "horse_margin_elo_delta",
    # year is almost a perfect proxy for train-vs-test in a temporal
    # split — it encodes *when* the row is, not anything about racing.
    "year",
    # trainer_elo_delta is not always present but exclude if it is
    "trainer_elo_delta",
    # API's days_since_last_run reflects scrape-time, not race-time;
    # the feature engineer computes a proper days_since_last_race instead.
    "days_since_last_run",
    # Form-string features parsed from the API's ``formsummary`` field.
    # On result pages the form string may include the CURRENT race's
    # result as the last digit, leaking finish_position directly.
    # Even if the API returns pre-race form, these are redundant with
    # the properly computed point-in-time rolling stats (horse_win_rate,
    # horse_avg_position, horse_wins_N, etc.).
    "form_last_pos",
    "form_avg",
    "form_wins_count",
    "form_length",
    "form_dnf_count",
    "form_has_break",
    "form_trend",
    # improving_flag is computed from the excluded form_trend,
    # laundering the leaky signal back into the feature set.
    "improving_flag",
    # jockey_elo_x_fav uses is_favourite which is derived from SP odds.
    # SP is the final market price at race-off — unavailable at bet time.
    "jockey_elo_x_fav",
    # Scrape-time lifetime totals from the results API — they describe
    # the horse AFTER the race (a debut winner shows run_count=1,
    # win_count=1) and leak the outcome.
    "horse_runs",
    "horse_wins",
    "horse_places",
    # Whole-dataset appearance counts: an entity's total row count
    # includes its FUTURE races, so high freq ⇒ long career ⇒ quality.
    "horse_name_freq",
    "jockey_freq",
    "trainer_freq",
    "track_freq",
]

# XGB-compatible hyperparameter names (used to filter params for
# classifier sub-models that may receive tuned dicts)
_XGB_VALID_HP = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda",
    "min_child_weight", "gamma",
}

# LGBM-compatible hyperparameter names
_LGBM_VALID_HP = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda",
    "min_child_samples", "num_leaves", "linear_tree",
}

# CatBoost-compatible hyperparameter names
_CAT_VALID_HP = {
    "n_estimators", "depth", "learning_rate",
    "l2_leaf_reg", "random_strength", "bagging_temperature",
}

# Default framework selection for each sub-model.
DEFAULT_FRAMEWORKS: dict[str, str] = {
    "classifier": "cat",
    "place": "cat",
    "ranker": "lgbm",
}


def make_relevance_labels(finish_positions: np.ndarray) -> np.ndarray:
    """Winner-heavy relevance labels used for ranking-quality metrics."""
    fp = np.asarray(finish_positions, dtype=int)
    return np.where(fp == 1, 5, np.where(fp == 2, 2, np.where(fp == 3, 1, 0)))


def normalise_implied_prob_by_race(df: pd.DataFrame) -> np.ndarray:
    """Return overround-normalised implied win probabilities per race."""
    if "odds" not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    raw_ip = (
        (1.0 / df["odds"].replace(0, np.nan))
        .fillna(0.0)
        .clip(0.0, 1.0)
        .values.astype(np.float32)
    )
    overround = df.groupby("race_id")["odds"].transform(
        lambda odds: (1.0 / odds).sum(),
    ).values.astype(np.float32)
    return raw_ip / np.maximum(overround, 1e-9)


def compute_recency_sample_weights(
    dates_like,
    half_life_days: float | None = None,
    seasonal_boost: float | None = None,
    decay_shape: str | None = None,
) -> np.ndarray:
    """Return normalised recency weights for dated rows.

    Supports three decay shapes (selected via *decay_shape*):
      * ``"exp"``    — exponential:  w = exp(-ln2 · t / hl)
      * ``"poly"``   — polynomial:   w = 1 / (1 + t / hl)  (heavier tail)
      * ``"linear"`` — linear ramp:  w = max(0, 1 - t / (2·hl))  (hard cutoff)

    All shapes share the same *half_life_days* parameter so that the
    meaning of "half-life" is comparable across shapes.
    """
    dates = pd.Series(pd.to_datetime(dates_like))
    if len(dates) == 0:
        return np.array([], dtype=np.float64)
    _hl = float(
        half_life_days
        if half_life_days is not None
        else getattr(config, "RECENCY_HALF_LIFE_DAYS", 180)
    )
    _hl = max(_hl, 1.0)
    _shape = (
        decay_shape
        if decay_shape is not None
        else getattr(config, "RECENCY_DECAY_SHAPE", "exp")
    )
    days_ago = (dates.max() - dates).dt.days.values.astype(np.float64)
    if _shape == "poly":
        weights = 1.0 / (1.0 + days_ago / _hl)
    elif _shape == "linear":
        weights = np.maximum(0.0, 1.0 - days_ago / (2.0 * _hl))
    else:  # "exp" (default)
        weights = np.exp(-np.log(2) * days_ago / _hl)
    _seasonal = float(
        seasonal_boost
        if seasonal_boost is not None
        else getattr(config, "RECENCY_SEASONAL_BOOST", 0.0)
    )
    if _seasonal > 0:
        _cur_month = dates.max().month
        _same_month = (dates.dt.month.values == _cur_month).astype(np.float64)
        weights *= (1.0 + _seasonal * _same_month)
    return weights / max(weights.mean(), 1e-12)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns (exclude targets and identifiers).

    Accepts every numeric dtype: the previous int64/float64/int32/float32
    allow-list silently dropped ~160 uint8/int8 columns (one-hot dummies,
    age, draw, handicap, pace/equipment flags ...) from the model.
    """
    return [
        col
        for col in df.columns
        if col not in EXCLUDE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
        and not pd.api.types.is_complex_dtype(df[col])
    ]


def _event_sort_key(df: pd.DataFrame) -> pd.Series:
    """Build a robust event-time key (race_date + off_time when available)."""
    race_dt = pd.to_datetime(df["race_date"], errors="coerce")
    if "off_time" not in df.columns:
        return race_dt

    off_secs = _off_time_to_seconds(df["off_time"]).astype(float)
    return race_dt + pd.to_timedelta(off_secs, unit="s")


def _drop_correlated_features(
    X: np.ndarray,
    feature_cols: list[str],
    importance: np.ndarray,
    threshold: float,
) -> list[str]:
    """Remove one feature from each highly-correlated pair.

    For every pair with |Pearson r| > *threshold*, the feature with
    **lower** pilot-model importance is dropped.  This avoids the
    classic pitfall where two near-identical features split importance
    between them, making both look unimportant to a subsequent pruning
    step.

    Runs in O(F²) with early-exit, which is fine for F < 500.
    """
    n_features = len(feature_cols)
    if n_features < 2 or threshold <= 0:
        return feature_cols

    # Rank features by importance (ascending) so we drop the weaker one
    imp_rank = np.argsort(importance)  # weakest first

    # Compute correlation matrix (on a subsample for speed)
    max_rows = min(len(X), 50_000)
    if max_rows < len(X):
        rng = np.random.RandomState(config.RANDOM_SEED)
        idx = rng.choice(len(X), max_rows, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Standardise columns to avoid numerical issues with constant cols
    std = X_sub.std(axis=0)
    std[std < 1e-12] = 1.0
    X_norm = (X_sub - X_sub.mean(axis=0)) / std
    corr = (X_norm.T @ X_norm) / max_rows
    np.fill_diagonal(corr, 0.0)  # ignore self-correlation

    drop = set()
    # Walk features weakest-first; if it's correlated with a stronger
    # feature, drop it.
    for idx in imp_rank:
        if idx in drop:
            continue
        partners = np.where(np.abs(corr[idx]) > threshold)[0]
        for p in partners:
            if p not in drop and importance[p] > importance[idx]:
                drop.add(idx)
                break
        # Also mark weaker partners of this feature
        if idx not in drop:
            for p in partners:
                if p not in drop and importance[p] <= importance[idx]:
                    drop.add(p)

    kept = [f for i, f in enumerate(feature_cols) if i not in drop]
    if drop:
        dropped_names = [feature_cols[i] for i in sorted(drop)]
        logger.info(
            "Correlation pruning (|r|>%.2f): %d → %d (dropped %d)",
            threshold, n_features, len(kept), len(drop),
        )
        logger.info("  Dropped: %s", dropped_names[:15])
    return kept


def _prune_features_quick(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    prune_fraction: float,
) -> list[str]:
    """Drop the lowest-importance features using a quick pilot model.

    Pipeline:
    1. Train a lightweight pilot LGBMClassifier.
    2. **Correlation pruning** — for each pair with |r| above
       ``FEATURE_CORR_THRESHOLD``, drop the less-important feature.
    3. **Importance pruning** — drop the bottom ``prune_fraction``
       of remaining features by split importance.
    """
    if len(train_df) < 2:
        logger.warning("Feature pruning skipped: not enough training samples (%d)", len(train_df))
        return feature_cols

    X = train_df[feature_cols].values
    y = train_df["won"].fillna(0).values.astype(int)

    _ensure_lightgbm()
    pilot = LGBMClassifier(
        n_estimators=60,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=30,
        objective="binary",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    pilot.fit(X, y)

    importance = pilot.feature_importances_

    # --- Step 1: correlation-aware pruning ---
    corr_thresh = float(getattr(config, "FEATURE_CORR_THRESHOLD", 0.0))
    if corr_thresh > 0:
        kept_after_corr = _drop_correlated_features(
            X, feature_cols, importance, corr_thresh,
        )
        if len(kept_after_corr) < len(feature_cols):
            # Re-index importance to match surviving columns
            keep_idx = [feature_cols.index(c) for c in kept_after_corr]
            importance = importance[keep_idx]
            X = X[:, keep_idx]
            feature_cols = kept_after_corr

    # --- Step 2: importance-based pruning ---
    n_drop = int(len(feature_cols) * prune_fraction)
    if n_drop < 1:
        return feature_cols

    sorted_idx = np.argsort(importance)          # ascending
    drop_set = set(sorted_idx[:n_drop].tolist())
    kept = [f for i, f in enumerate(feature_cols) if i not in drop_set]
    dropped = [feature_cols[i] for i in sorted_idx[:n_drop]]

    logger.info(
        f"Feature pruning: {len(feature_cols)} \u2192 {len(kept)} "
        f"(dropped {n_drop} lowest-importance features)"
    )
    logger.info(f"  Sample dropped: {dropped[:10]}")

    return kept


def _quick_prune_mask(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    prune_fraction: float,
    corr_threshold: float,
) -> np.ndarray:
    """Return a boolean column mask after quick pilot importance+correlation pruning."""
    n_feats = X.shape[1]
    mask = np.ones(n_feats, dtype=bool)

    if n_feats < 2 or len(X) < 2:
        return mask

    _ensure_lightgbm()
    pilot = LGBMClassifier(
        n_estimators=60, max_depth=5, learning_rate=0.1,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
        objective="binary", random_state=config.RANDOM_SEED,
        n_jobs=1, verbose=-1,
    )
    pilot.fit(X, y.astype(int))
    importance = pilot.feature_importances_

    # Correlation de-dup
    if corr_threshold > 0:
        kept = _drop_correlated_features(X, list(feature_cols), importance, corr_threshold)
        if len(kept) < n_feats:
            kept_set = set(kept)
            for i, c in enumerate(feature_cols):
                if c not in kept_set:
                    mask[i] = False
            # Re-index importance
            importance = importance[mask]

    # Importance pruning
    active_idx = np.where(mask)[0]
    n_drop = int(len(active_idx) * prune_fraction)
    if n_drop >= 1:
        sorted_imp = np.argsort(importance)
        drop_positions = sorted_imp[:n_drop]
        for dp in drop_positions:
            mask[active_idx[dp]] = False

    return mask


def prepare_multi_target_data(
    df: pd.DataFrame,
) -> dict:
    """
    Prepare a shared train/test split with **multiple** target arrays.

    Returns multiple targets per split:

    * ``y_*_won``    — binary ``won``         (for win classifier)
    * ``y_*_placed`` — binary placed          (for place classifier)
    * ``y_*_rel``    — relevance labels       (for ranking metrics)

    Returns:
        dict with keys ``X_train``, ``X_test``, ``y_train_rel``,
        ``y_test_rel``, ``y_train_won``,
        ``y_test_won``, ``groups_train``, ``groups_test``,
        ``feature_cols``, ``scaler``, ``test_df``.
    """
    logger.info("Preparing multi-target data …")

    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} features")

    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["_event_dt"] = _event_sort_key(df)
    # Sort by horse_name within each race to break the finish_position
    # ordering from the raw data — prevents row-position from leaking
    # the outcome when model scores are degenerate (all-equal).
    _sort_cols = ["_event_dt", "race_id"]
    if "horse_name" in df.columns:
        _sort_cols.append("horse_name")
    df = df.sort_values(_sort_cols).reset_index(drop=True)

    # Only keep rows with valid finish positions
    df = df[df["finish_position"].notna() & (df["finish_position"] > 0)].copy()

    # ── Degenerate race filter (training quality) ─────────────────────────
    _degenerate_removed_rows = 0
    _degenerate_removed_races = 0
    if "num_runners" in df.columns:
        _single_runner_ids = df.groupby("race_id")["num_runners"].first()
        _single_runner_ids = _single_runner_ids[_single_runner_ids <= 1].index
    else:
        _single_runner_ids = pd.Index([])
    _pos_spread = df.groupby("race_id")["finish_position"].transform("nunique")
    _identical_pos_ids = df.loc[_pos_spread <= 1, "race_id"].unique()
    _bad_odds_ids = pd.Index([])
    if "odds" in df.columns:
        _bad_odds_ids = df.loc[df["odds"] < 1.01, "race_id"].unique()
    _degenerate_ids = set(_single_runner_ids) | set(_identical_pos_ids) | set(_bad_odds_ids)
    if _degenerate_ids:
        _n_before = len(df)
        df = df[~df["race_id"].isin(_degenerate_ids)].copy()
        _degenerate_removed_rows = _n_before - len(df)
        _degenerate_removed_races = len(_degenerate_ids)
        logger.info(
            f"Removed {_degenerate_removed_rows} rows from {_degenerate_removed_races} "
            f"degenerate races (single-runner / identical positions / bad odds)"
        )

    # ── Burn-in exclusion (dataset-level, before split) ────────────────
    # Applied once from the absolute dataset start so that every
    # downstream split / CV fold sees the same burn-in boundary.
    _burn_months = getattr(config, "BURN_IN_MONTHS", 4)
    if _burn_months > 0 and len(df) > 0:
        _ds_start = pd.Timestamp(df["race_date"].min())
        _ds_end = pd.Timestamp(df["race_date"].max())
        _burn_cutoff = _ds_start + pd.DateOffset(months=_burn_months)
        if _burn_cutoff < _ds_end:
            _n_before = len(df)
            df = df[df["race_date"] >= _burn_cutoff].copy()
            logger.info(
                f"Burn-in: excluded {_n_before - len(df)} cold-start rows "
                f"(first {_burn_months} months, cutoff {_burn_cutoff.date()})"
            )
        else:
            logger.info(
                f"Burn-in skipped: dataset span "
                f"({(_ds_end - _ds_start).days} days) "
                f"does not exceed {_burn_months}-month burn-in period"
            )

    # Relevance labels: winner-heavy (1st=5, 2nd=2, 3rd=1, rest=0)
    fp = df["finish_position"].values.astype(int)
    df["relevance"] = make_relevance_labels(fp)

    # Time-based split (aligned to race boundary)
    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    split_race = df.iloc[split_idx]["race_id"]
    while split_idx < len(df) and df.iloc[split_idx]["race_id"] == split_race:
        split_idx += 1

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    train_df = train_df.drop(columns=["_event_dt"], errors="ignore")
    test_df = test_df.drop(columns=["_event_dt"], errors="ignore")

    # ── Purge gap ────────────────────────────────────────────────
    # Remove training rows within PURGE_DAYS of the test boundary
    # to prevent feature leakage from overlapping horse form.
    _purge_days = getattr(config, "PURGE_DAYS", 7)
    if _purge_days > 0:
        _test_start = test_df["race_date"].min()
        _purge_cutoff = _test_start - pd.Timedelta(days=_purge_days)
        _n_before = len(train_df)
        train_df = train_df[train_df["race_date"] <= _purge_cutoff].copy()
        _n_purged = _n_before - len(train_df)
        if _n_purged > 0:
            logger.info(
                f"Purged {_n_purged} training rows within {_purge_days} "
                f"days of test boundary ({_test_start.date()})"
            )

    if len(train_df) < 2:
        raise ValueError(
            f"Training data has only {len(train_df)} row(s) after filtering "
            f"(degenerate-race removal, {_purge_days}-day purge gap, "
            f"{_burn_months}-month burn-in). Add more data or reduce "
            f"BURN_IN_MONTHS / PURGE_DAYS in config."
        )

    # Save training dates for purged CV fold splitting in Phase 1
    train_race_dates = train_df["race_date"].values

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # ── Feature drift / PSI check ─────────────────────────────────
    # Population Stability Index measures how much the feature distribution
    # has shifted between training and test.  PSI > 0.2 = major drift,
    # > 0.1 = moderate drift.  Log warnings so users can investigate.
    _psi_thresh_warn = 0.1
    _psi_thresh_high = 0.2
    _psi_feats = feature_cols[:50] if len(feature_cols) > 50 else feature_cols
    _psi_results: list[tuple[str, float]] = []
    for _fi, _fc in enumerate(_psi_feats):
        _tr_arr = X_train[:, _fi].astype(np.float64)
        _te_arr = X_test[:, _fi].astype(np.float64)
        if not np.isfinite(_tr_arr).any() or not np.isfinite(_te_arr).any():
            continue
        # Use training deciles as bin boundaries
        _qtiles = np.percentile(_tr_arr[np.isfinite(_tr_arr)], np.linspace(0, 100, 11))
        _qtiles = np.unique(_qtiles)
        if len(_qtiles) < 3:
            continue  # degenerate (constant feature) — skip
        _qtiles[0] -= 1e-9 ; _qtiles[-1] += 1e-9
        _eps = 1e-9
        _tr_c = np.histogram(_tr_arr, bins=_qtiles)[0].astype(np.float64)
        _te_c = np.histogram(_te_arr, bins=_qtiles)[0].astype(np.float64)
        _tr_p = np.clip(_tr_c / max(_tr_c.sum(), _eps), _eps, None)
        _te_p = np.clip(_te_c / max(_te_c.sum(), _eps), _eps, None)
        _psi = float(np.sum((_te_p - _tr_p) * np.log(_te_p / _tr_p)))
        if _psi >= _psi_thresh_warn:
            _psi_results.append((_fc, _psi))
    if _psi_results:
        _psi_results.sort(key=lambda x: x[1], reverse=True)
        for _fc, _psi in _psi_results:
            _lvl = "HIGH" if _psi >= _psi_thresh_high else "moderate"
            logger.warning(
                f"Feature drift ({_lvl}): '{_fc}' PSI={_psi:.3f} "
                f"(train→test distribution shift)"
            )

    # ── Multiple targets ──────────────────────────────────────────
    y_train_rel = train_df["relevance"].values
    y_test_rel = test_df["relevance"].values

    # Non-finishers have NaN lengths_behind; impute with field-size
    # penalty (last + 10L) so the regressor treats them as poor outcomes
    # rather than as winners (0.0).
    def _impute_lb(df: pd.DataFrame) -> np.ndarray:
        lb = df["lengths_behind"].copy()
        if "finish_position" in df.columns:
            non_finisher = df["finish_position"].fillna(0).astype(int) < 1
        else:
            non_finisher = lb.isna()
        # For non-finishers: use max lb in race + 10, or 30 if unknown
        race_max_lb = df.groupby("race_id")["lengths_behind"].transform("max")
        lb = lb.where(~non_finisher, race_max_lb + 10.0)
        lb = lb.fillna(30.0)
        return lb.values.astype(np.float32)

    y_train_lb = _impute_lb(train_df)
    y_test_lb = _impute_lb(test_df)

    y_train_won = train_df["won"].fillna(0).values.astype(int)
    y_test_won = test_df["won"].fillna(0).values.astype(int)

    # Market-residual target: realised outcome minus *normalised* implied
    # win probability (overround removed).  Using raw 1/odds would inflate
    # the baseline by ~20% and introduce overround-dependent noise.
    ip_tr = normalise_implied_prob_by_race(train_df)
    ip_te = normalise_implied_prob_by_race(test_df)

    y_train_resid = y_train_won.astype(np.float32) - ip_tr
    y_test_resid = y_test_won.astype(np.float32) - ip_te

    # Place target — dynamic based on actual EW places paid per race
    # (2 for 5-7 runners, 3 for 8-15, 4 for 16+ handicaps)
    def _places_paid_vec(df):
        nr = df["num_runners"].values
        hcap = df.get("handicap", pd.Series(0, index=df.index)).values.astype(bool)
        return np.where(
            nr <= 4, 3,           # ineligible races fallback to 3
            np.where(nr <= 7, 2,
            np.where(nr <= 15, 3,
            np.where(hcap, 4, 3)))
        )

    _pp_train = _places_paid_vec(train_df)
    _pp_test = _places_paid_vec(test_df)
    y_train_placed = (train_df["finish_position"].values <= _pp_train).astype(int)
    y_test_placed = (test_df["finish_position"].values <= _pp_test).astype(int)

    # Normalised position target (0 = winner, 1 = last)
    nr_train = train_df["num_runners"].replace(0, 1).values.astype(np.float32)
    y_train_norm_pos = (
        (train_df["finish_position"].values.astype(np.float32) - 1)
        / np.maximum(nr_train - 1, 1)
    )
    nr_test = test_df["num_runners"].replace(0, 1).values.astype(np.float32)
    y_test_norm_pos = (
        (test_df["finish_position"].values.astype(np.float32) - 1)
        / np.maximum(nr_test - 1, 1)
    )

    fp_train = train_df["finish_position"].values.astype(np.float32)
    fp_test = test_df["finish_position"].values.astype(np.float32)

    # ── Recency sample weights ────────────────────────────────────
    # Recent races are more predictive — exponential decay over time.
    sample_weight_train = compute_recency_sample_weights(train_df["race_date"])

    # Group sizes
    groups_train = train_df.groupby("race_id", sort=False).size().values
    groups_test = test_df.groupby("race_id", sort=False).size().values

    # Scale features
    scaler = _IdentityScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        f"Training: {X_train.shape[0]} runners in {len(groups_train)} races"
    )
    logger.info(
        f"Test:     {X_test.shape[0]} runners in {len(groups_test)} races"
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_rel": y_train_rel,
        "y_test_rel": y_test_rel,
        "y_train_lb": y_train_lb,
        "y_test_lb": y_test_lb,
        "y_train_won": y_train_won,
        "y_test_won": y_test_won,
        "y_train_resid": y_train_resid,
        "y_test_resid": y_test_resid,
        "y_train_placed": y_train_placed,
        "y_test_placed": y_test_placed,
        "y_train_norm_pos": y_train_norm_pos,
        "y_test_norm_pos": y_test_norm_pos,
        "fp_train": fp_train,
        "fp_test": fp_test,
        "ip_train": ip_tr,
        "odds_train": train_df["odds"].fillna(0.0).values.astype(np.float32) if "odds" in train_df.columns else np.zeros(len(train_df), dtype=np.float32),
        "groups_train": groups_train,
        "groups_test": groups_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "test_df": test_df,
        "sample_weight_train": sample_weight_train,
        "train_race_dates": train_race_dates,
        "train_df": train_df,
        "degenerate_removed_rows": int(_degenerate_removed_rows),
        "degenerate_removed_races": int(_degenerate_removed_races),
    }


# ── Rank-Probability Score helpers ──────────────────────────────────────────


def _proba_to_logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities [0, 1] to logits (-∞, +∞) for softmax compatibility."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def _group_offsets(groups: np.ndarray) -> np.ndarray:
    """Return start-index of each group: [0, g0, g0+g1, ...]."""
    offsets = np.empty(len(groups), dtype=np.intp)
    offsets[0] = 0
    if len(groups) > 1:
        np.cumsum(groups[:-1], out=offsets[1:])
    return offsets


def _grouped_softmax(
    scores: np.ndarray,
    groups: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Vectorised per-group softmax using reduceat — ~20-50× faster than a
    Python loop for typical race counts."""
    scores = np.asarray(scores, dtype=np.float64)
    t = max(float(temperature), 1e-6)
    scaled = scores / t

    offsets = _group_offsets(groups)
    group_max = np.maximum.reduceat(scaled, offsets)
    gids = np.repeat(np.arange(len(groups)), groups)
    exp_vals = np.exp(scaled - group_max[gids])
    group_sum = np.add.reduceat(exp_vals, offsets)
    return exp_vals / np.maximum(group_sum[gids], 1e-12)


def _grouped_normalize(
    values: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Divide each element by its group sum (re-normalisation)."""
    offsets = _group_offsets(groups)
    group_sum = np.add.reduceat(values, offsets)
    gids = np.repeat(np.arange(len(groups)), groups)
    return values / np.maximum(group_sum[gids], 1e-12)


def rps_per_race(
    probs: np.ndarray,
    finish_positions: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Vectorised mean Rank-Probability Score over all races.

    RPS = 1/(N-1) * Σ_{k=1}^{N-1} (F_k − 1)² where F_k is the
    cumulative predicted probability for the actual top-k finishers.
    """
    probs = np.asarray(probs, dtype=np.float64)
    finish_positions = np.asarray(finish_positions, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.intp)
    n_groups = len(groups)
    if n_groups == 0:
        return float("nan")

    valid = groups >= 2
    if not valid.any():
        return float("nan")

    offsets = _group_offsets(groups)
    gids = np.repeat(np.arange(n_groups), groups)

    # Sort within each group by finish_position (lexsort keeps groups contiguous)
    order = np.lexsort((finish_positions, gids))
    sorted_probs = probs[order]

    # Grouped cumulative sum
    cumsum = np.cumsum(sorted_probs)
    bases = np.zeros(n_groups, dtype=np.float64)
    if n_groups > 1:
        bases[1:] = cumsum[offsets[1:] - 1]
    grouped_cumsum = cumsum - bases[gids]

    # (F − 1)²;  zero-out last element of each group (RPS uses F[:g-1])
    sq_diff = (grouped_cumsum - 1.0) ** 2
    sq_diff[np.cumsum(groups) - 1] = 0.0
    # Zero-out races with fewer than 2 runners
    sq_diff[np.repeat(~valid, groups)] = 0.0

    # Per-group mean = sum / (g − 1)
    group_sq_sum = np.add.reduceat(sq_diff, offsets)
    per_group_rps = group_sq_sum / np.maximum(groups - 1, 1)
    return float(np.mean(per_group_rps[valid]))


# ── Market anchor (Benter combination) ────────────────────────────────

def market_probs_from_odds(odds: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Overround-normalised market probabilities per race.

    Returns ``(market_probs, race_valid)`` where *race_valid* flags races
    in which every runner has usable odds — anchoring is only applied to
    those races.
    """
    odds_arr = np.asarray(odds, dtype=np.float64)
    offsets = _group_offsets(groups)
    runner_ok = np.isfinite(odds_arr) & (odds_arr > 1.01)
    race_valid = np.add.reduceat(runner_ok.astype(np.int64), offsets) == groups
    raw_ip = np.where(runner_ok, 1.0 / np.clip(odds_arr, 1.01, None), 1.0)
    sums = np.add.reduceat(raw_ip, offsets)
    mkt = raw_ip / np.repeat(np.maximum(sums, 1e-12), groups)
    return np.clip(mkt, 1e-6, 1.0 - 1e-6), race_valid


def apply_market_anchor(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    beta: float,
    race_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Per-race ``softmax(alpha*log(p_model) + beta*log(p_market))``.

    Rows in races where *race_valid* is False keep their original
    probabilities.
    """
    p = np.clip(np.asarray(model_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    z = alpha * np.log(p) + beta * np.log(market_probs)
    offsets = _group_offsets(groups)
    z = z - np.repeat(np.maximum.reduceat(z, offsets), groups)
    e = np.exp(z)
    anchored = e / np.repeat(np.add.reduceat(e, offsets), groups)
    if race_valid is not None:
        keep = ~np.repeat(race_valid, groups)
        anchored[keep] = np.asarray(model_probs, dtype=np.float64)[keep]
    return anchored


def fit_market_anchor(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    groups: np.ndarray,
    won: np.ndarray,
    race_valid: np.ndarray | None = None,
) -> dict | None:
    """Fit (alpha, beta) by maximum likelihood on the race winners.

    Returns ``{"alpha": float, "beta": float}`` or ``None`` when there is
    not enough usable data.
    """
    from scipy.optimize import minimize

    groups = np.asarray(groups, dtype=np.int64)
    won_arr = np.asarray(won, dtype=np.int64)
    offsets = _group_offsets(groups)
    one_winner = np.add.reduceat(won_arr, offsets) == 1
    usable = one_winner if race_valid is None else (one_winner & race_valid)
    if int(usable.sum()) < 50:
        return None

    row_mask = np.repeat(usable, groups)
    g_use = groups[usable]
    p_use = np.asarray(model_probs, dtype=np.float64)[row_mask]
    m_use = np.asarray(market_probs, dtype=np.float64)[row_mask]
    w_use = won_arr[row_mask].astype(bool)

    def neg_ll(params):
        anchored = apply_market_anchor(p_use, m_use, g_use, params[0], params[1])
        return -np.mean(np.log(np.clip(anchored[w_use], 1e-12, None)))

    res = minimize(neg_ll, x0=np.array([0.3, 0.8]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400})
    if not np.all(np.isfinite(res.x)):
        return None
    return {"alpha": float(res.x[0]), "beta": float(res.x[1])}


def get_feature_importance(
    model,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Get feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_cols))

    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False)

    return fi_df.head(top_n)


# =====================================================================


class RacePredictor:
    """
    Task-specific calibrated classifiers for horse race prediction.

    Models trained:

    1. **Win Classifier** — binary classifier → calibrated P(win).
       Drives the **Value Bet** and **Top Pick** strategies.
    2. **Place Classifier** — binary classifier → calibrated P(place).
       Drives the **Each-Way** strategy.
    3. **Race Ranker** (optional, ``config.TRAIN_RANKER``) — LambdaRank
       model trained and evaluated for **diagnostics only** (ranker/
       classifier agreement panels). Its scores are *not* blended into
       the win probabilities at inference time.
    4. **Linear baseline** — logistic regression fitted on the same
       features and reported in the metrics so tree-model lift is
       always measured against a simple reference. Not persisted.

    Each model is trained independently. Calibration (Platt + isotonic)
    is fitted on out-of-fold predictions from purged expanding-window
    CV, optionally refit on the most recent training slab.

    Historical note: this class was previously named
    ``TripleEnsemblePredictor``; a module-level alias is kept so older
    pickled models load unchanged. There is no stacking/blending — the
    "ensemble" naming was a leftover from a removed architecture.
    """

    def __init__(
        self,
        frameworks: dict[str, str] | None = None,
        **_ignored,
    ):
        self.clf_model = None
        self.place_model = None
        self.ranker_model = None
        self.scaler = None
        self.feature_cols = None
        self.frameworks = {
            "classifier": "cat", "place": "cat", "ranker": "lgbm",
            **(frameworks or {}),
        }
        # Win-classifier calibration (Platt)
        self.calibration_temp = 1.0  # kept for serialisation compat
        self.platt_a = 1.0
        self.platt_b = 0.0
        # Place-classifier calibration (Platt)
        self.place_platt_a = 1.0
        self.place_platt_b = 0.0
        # Isotonic calibration (fitted after Platt)
        self.win_iso_cal: IsotonicRegression | None = None
        self.place_iso_cal: IsotonicRegression | None = None
        # Market anchor (Benter combination) fitted on OOF predictions
        self.market_anchor: dict | None = None
        # n_jobs override — capped during autotune to leave CPU headroom
        self._autotune_njobs: int = -1
        self.metrics = None
        self.test_analysis = None
        self.last_explain_model_label = None

    # ── Framework-aware helpers ──────────────────────────────────
    def _clf_predict_proba(self, model, X) -> np.ndarray:
        """Return P(class=1) from a classifier (XGB or LGBM)."""
        return model.predict_proba(X)[:, 1]

    def _apply_market_anchor(
        self,
        probs: np.ndarray,
        groups: np.ndarray,
        df: pd.DataFrame | None,
    ) -> np.ndarray:
        """Blend calibrated win probs with the market when fitted/possible."""
        anchor = getattr(self, "market_anchor", None)
        if not anchor or df is None or "odds" not in df.columns:
            return probs
        odds = pd.to_numeric(df["odds"], errors="coerce").values
        mkt, race_valid = market_probs_from_odds(odds, groups)
        if not race_valid.any():
            return probs
        return apply_market_anchor(
            probs, mkt, groups, anchor["alpha"], anchor["beta"], race_valid,
        )

    @staticmethod
    def _early_stop_split(
        groups: np.ndarray,
        race_dates,
        frac: float = 0.9,
        purge_days: int = 7,
    ) -> tuple[int, int] | None:
        """Purged temporal split (row indices) for early stopping.

        Returns ``(train_end_row, val_start_row)`` or ``None`` when the
        slice is too small to split safely.
        """
        n_races = len(groups)
        if n_races < 40:
            return None
        cum = np.cumsum(groups)
        starts = np.concatenate([[0], cum[:-1]])
        bnd = max(1, int(n_races * frac))
        if bnd >= n_races:
            return None
        cutoff = race_dates[bnd] - pd.Timedelta(days=purge_days)
        tr_mask = race_dates[:bnd] <= cutoff
        if tr_mask.sum() < 20:
            return None
        tr_end_row = int(cum[int(np.where(tr_mask)[0][-1])])
        vl_start_row = int(starts[bnd])
        if tr_end_row <= 0 or (int(cum[-1]) - vl_start_row) < 50:
            return None
        return tr_end_row, vl_start_row

    # ── Training ─────────────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        save: bool = True,
        params: dict[str, dict] | None = None,
        progress_callback: "Callable[[str, float], None] | None" = None,
        value_config: dict | None = None,
        auto_tune: dict | None = None,
        **_ignored,
    ) -> dict:
        """Train two task-specific models:

        * **Win Classifier** — calibrated P(win) (Value Bets / Top Pick).
        * **Place Classifier** — calibrated P(place) (Each-Way).

        Phase 1: OOF CV → fit per-model calibration.
        Phase 1b (optional): Optuna auto-tune per model.
        Phase 2: Retrain on full data → evaluate → betting simulation.
        """
        data = prepare_multi_target_data(df)

        self.feature_cols = data["feature_cols"]
        self.scaler = data["scaler"]

        X_train = data["X_train"]
        X_test = data["X_test"]
        groups_train = data["groups_train"]
        groups_test = data["groups_test"]
        test_df = data["test_df"]

        X_eval = X_test
        groups_eval = groups_test
        y_eval_rel = data["y_test_rel"]
        eval_df = test_df

        self.eval_split_info = {
            "holdout_races": int(len(groups_test)),
            "validation_races": int(len(groups_eval)),
            "validation_runners": int(len(X_eval)),
            "degenerate_removed_rows": int(data.get("degenerate_removed_rows", 0)),
            "degenerate_removed_races": int(data.get("degenerate_removed_races", 0)),
        }

        def _cb(msg: str, pct: float = 0.0) -> None:
            if progress_callback is not None:
                progress_callback(msg, pct)

        # ── Phase 1: Purged expanding-window CV for calibration ──
        _cb("Phase 1 — OOF CV for calibration …", 0.0)
        n_cv_folds = getattr(config, "CV_N_FOLDS", 3)
        purge_gap = getattr(config, "PURGE_DAYS", 7)

        train_dates = pd.to_datetime(data["train_race_dates"])
        cum_g = np.cumsum(groups_train)
        race_starts_idx = np.concatenate([[0], cum_g[:-1]])
        race_dates_idx = pd.DatetimeIndex(train_dates.values[race_starts_idx])
        n_races = len(race_dates_idx)

        date_min = race_dates_idx.min()
        date_max = race_dates_idx.max()
        total_span = (date_max - date_min).days
        chunk_days = total_span / (n_cv_folds + 1)
        boundaries = [
            date_min + pd.Timedelta(days=int(chunk_days * i))
            for i in range(n_cv_folds + 2)
        ]
        boundaries[-1] = date_max + pd.Timedelta(days=1)

        _prune_frac = float(getattr(config, "FEATURE_PRUNE_FRACTION", 0.0))
        if _prune_frac > 0 and len(self.feature_cols) > 8:
            _anchor_cutoff = boundaries[1] - pd.Timedelta(days=purge_gap)
            _anchor_mask = pd.to_datetime(data["train_df"]["race_date"]) <= _anchor_cutoff
            _anchor_df = data["train_df"].loc[_anchor_mask].copy()
            if len(_anchor_df) >= 200:
                _kept_cols = _prune_features_quick(_anchor_df, self.feature_cols, _prune_frac)
                if len(_kept_cols) < len(self.feature_cols):
                    _keep_idx = [self.feature_cols.index(col) for col in _kept_cols]
                    self.feature_cols = _kept_cols
                    X_train = X_train[:, _keep_idx]
                    X_test = X_test[:, _keep_idx]
                    X_eval = X_test
                    data["feature_cols"] = _kept_cols
                    logger.info(
                        "Applied leak-safe feature pruning using %d anchor rows before OOF validation",
                        len(_anchor_df),
                    )
            else:
                logger.info(
                    "Feature pruning skipped: anchor slice too small (%d rows)",
                    len(_anchor_df),
                )

        oof_clf_logits_parts: list[np.ndarray] = []
        oof_place_probs_parts: list[np.ndarray] = []
        oof_ranker_scores_parts: list[np.ndarray] = []
        oof_groups_parts: list[np.ndarray] = []
        oof_fp_parts: list[np.ndarray] = []
        oof_placed_parts: list[np.ndarray] = []
        oof_df_parts: list[pd.DataFrame] = []

        _p: dict[str, dict] = params if isinstance(params, dict) else {}

        for fold_k in range(n_cv_folds):
            pct = 0.02 + fold_k * (0.24 / n_cv_folds)
            _cb(f"Phase 1 — CV fold {fold_k + 1}/{n_cv_folds} …", pct)

            val_start_dt = boundaries[fold_k + 1]
            val_end_dt = boundaries[fold_k + 2]
            purge_cutoff_dt = val_start_dt - pd.Timedelta(days=purge_gap)

            tr_mask = race_dates_idx <= purge_cutoff_dt
            vl_mask = (race_dates_idx >= val_start_dt) & (race_dates_idx < val_end_dt)
            tr_races = np.where(tr_mask)[0]
            vl_races = np.where(vl_mask)[0]

            if len(tr_races) < 20 or len(vl_races) < 5:
                logger.warning(f"  Fold {fold_k + 1}: insufficient data, skipping")
                continue

            tr_end = int(cum_g[tr_races[-1]])
            vl_beg = int(race_starts_idx[vl_races[0]])
            vl_end = int(cum_g[vl_races[-1]])

            X_f_tr = X_train[:tr_end]
            X_f_vl = X_train[vl_beg:vl_end]
            g_f_tr = groups_train[:tr_races[-1] + 1]
            g_f_vl = groups_train[vl_races[0]:vl_races[-1] + 1]
            sw_f = data["sample_weight_train"][:tr_end]
            fp_f_vl = data["fp_train"][vl_beg:vl_end]

            n_purged = vl_beg - tr_end
            logger.info(
                f"  Fold {fold_k + 1}/{n_cv_folds}: "
                f"{len(X_f_tr)} train ({len(g_f_tr)} races), "
                f"{len(X_f_vl)} val ({len(g_f_vl)} races), "
                f"{n_purged} purged"
            )

            # Train fold models (disposable — for OOF scores only).
            # A purged temporal tail of the fold's own training slice is
            # used for early stopping, mirroring the Phase-2 retrain —
            # without it each fold trains the full n_estimators, which
            # dominates total training time.
            y_won_f = data["y_train_won"][:tr_end]
            y_placed_f = data["y_train_placed"][:tr_end]
            _f_es = (
                self._early_stop_split(
                    g_f_tr, race_dates_idx[:tr_races[-1] + 1], purge_days=purge_gap,
                )
                if int(getattr(config, "EARLY_STOPPING_ROUNDS", 0)) > 0
                else None
            )
            if _f_es is not None:
                _f_tr_end, _f_vl_beg = _f_es
                _f_X, _f_sw = X_f_tr[:_f_tr_end], sw_f[:_f_tr_end]
                _f_eval_w = [(X_f_tr[_f_vl_beg:], y_won_f[_f_vl_beg:])]
                _f_eval_p = [(X_f_tr[_f_vl_beg:], y_placed_f[_f_vl_beg:])]
                _f_y_won, _f_y_placed = y_won_f[:_f_tr_end], y_placed_f[:_f_tr_end]
            else:
                _f_X, _f_sw = X_f_tr, sw_f
                _f_eval_w = _f_eval_p = None
                _f_y_won, _f_y_placed = y_won_f, y_placed_f

            n_pos_w = max(int(_f_y_won.sum()), 1)
            self.clf_model = self._train_win_classifier(
                _f_X, _f_y_won,
                scale_pos_weight=(len(_f_y_won) - n_pos_w) / n_pos_w,
                params=_p.get("classifier"), sample_weight=_f_sw,
                eval_set=_f_eval_w,
            )
            n_pp = max(int(_f_y_placed.sum()), 1)
            self.place_model = self._train_place_classifier(
                _f_X, _f_y_placed,
                scale_pos_weight=(len(_f_y_placed) - n_pp) / n_pp,
                params=_p.get("place"), sample_weight=_f_sw,
                eval_set=_f_eval_p,
            )
            if getattr(config, "TRAIN_RANKER", False):
                if _f_es is not None:
                    _f_g_tr_races = np.searchsorted(np.cumsum(g_f_tr), _f_tr_end, side="left") + 1
                    self.ranker_model = self._train_ranker(
                        _f_X,
                        data["y_train_rel"][:_f_tr_end],
                        g_f_tr[:_f_g_tr_races],
                        params=_p.get("ranker"),
                        sample_weight=_f_sw,
                    )
                else:
                    self.ranker_model = self._train_ranker(
                        X_f_tr,
                        data["y_train_rel"][:tr_end],
                        g_f_tr,
                        params=_p.get("ranker"),
                        sample_weight=_f_sw,
                    )
            else:
                self.ranker_model = None

            # Score validation fold
            clf_probs_vl = self.clf_model.predict_proba(X_f_vl)[:, 1]
            place_probs_vl = self.place_model.predict_proba(X_f_vl)[:, 1]
            ranker_scores_vl = self.ranker_model.predict(X_f_vl) if self.ranker_model is not None else None

            oof_clf_logits_parts.append(clf_probs_vl)
            oof_place_probs_parts.append(place_probs_vl)
            if ranker_scores_vl is not None:
                oof_ranker_scores_parts.append(np.asarray(ranker_scores_vl, dtype=np.float64))
            oof_groups_parts.append(g_f_vl)
            oof_fp_parts.append(fp_f_vl)
            oof_df_parts.append(data["train_df"].iloc[vl_beg:vl_end].copy())
            # Dynamic place target matching training labels
            _nr_vl = data["train_df"]["num_runners"].values[vl_beg:vl_end]
            _hc_vl = data["train_df"].get("handicap", pd.Series(0, index=data["train_df"].index)).values[vl_beg:vl_end].astype(bool)
            _pp_vl = np.where(_nr_vl <= 4, 3, np.where(_nr_vl <= 7, 2, np.where(_nr_vl <= 15, 3, np.where(_hc_vl, 4, 3))))
            oof_placed_parts.append((fp_f_vl <= _pp_vl).astype(int))

        # ── Fit calibration on concatenated OOF data ─────────────
        _cb("Phase 1 — Fitting calibration …", 0.28)

        _all_clf_logits = np.concatenate(oof_clf_logits_parts)
        _all_groups = np.concatenate(oof_groups_parts)
        _all_fp = np.concatenate(oof_fp_parts)
        _all_place_probs = np.concatenate(oof_place_probs_parts)
        _all_placed = np.concatenate(oof_placed_parts)
        _oof_df = pd.concat(oof_df_parts, ignore_index=True) if oof_df_parts else pd.DataFrame()

        # Win-classifier calibration: Platt only (classifier outputs
        # probabilities directly, no softmax/temperature needed)
        self.calibration_temp = 1.0  # unused — kept for serialisation compat
        self.platt_a, self.platt_b = self._optimise_platt_calibration(
            _all_clf_logits, _all_groups, _all_fp,
        )
        logger.info(f"  Win-clf Platt: a={self.platt_a:.4f}, b={self.platt_b:.4f}")

        # Banded Platt removed — it anchors probabilities to market
        # odds bands, washing out the model's genuine edge.

        # Place-classifier calibration: Platt on binary outcomes
        self.place_platt_a, self.place_platt_b = self._optimise_place_platt(
            _all_place_probs, _all_placed,
        )
        logger.info(
            f"  Place-clf Platt: a={self.place_platt_a:.4f}, "
            f"b={self.place_platt_b:.4f}"
        )

        # Isotonic calibration on OOF Platt-calibrated probs
        _oof_win_cal = self._calibrate_win_probs(_all_clf_logits, _all_groups)
        self.win_iso_cal = self._fit_isotonic(_oof_win_cal, (_all_fp == 1).astype(np.float64))
        _oof_place_cal = self._calibrate_place_probs(_all_place_probs)
        self.place_iso_cal = self._fit_isotonic(_oof_place_cal, _all_placed.astype(np.float64))
        logger.info("  Isotonic calibration fitted on OOF predictions.")

        # ── Phase 1b: Per-model Optuna auto-tune (optional) ──────
        if auto_tune is not None and params is None:
            _at_trials = auto_tune.get("n_trials", 30)
            _cb(f"Phase 1b — Auto-tuning 3 models ({_at_trials} trials each) …", 0.35)
            logger.info(f"Running per-model Optuna auto-tune ({_at_trials} trials × 3) …")

            _at_bnd = max(1, int(n_races * 0.8))
            _at_val_start = race_dates_idx[_at_bnd]
            _at_purge_cut = _at_val_start - pd.Timedelta(days=purge_gap)
            _at_tr_mask = race_dates_idx[:_at_bnd] <= _at_purge_cut
            if _at_tr_mask.sum() < 10:
                _at_tr_end_race = _at_bnd - 1
            else:
                _at_tr_end_race = int(np.where(_at_tr_mask)[0][-1])
            _at_tr_end = int(cum_g[_at_tr_end_race])
            _at_vl_beg = int(race_starts_idx[_at_bnd])

            _at_X_tr = X_train[:_at_tr_end]
            _at_X_vl = X_train[_at_vl_beg:]
            _at_g_tr = groups_train[:_at_tr_end_race + 1]
            _at_g_vl = groups_train[_at_bnd:]
            _at_sw = data["sample_weight_train"][:_at_tr_end]

            _at_targets_tr = {
                "rel": data["y_train_rel"][:_at_tr_end],
                "lb": data["y_train_lb"][:_at_tr_end],
                "won": data["y_train_won"][:_at_tr_end],
                "resid": data["y_train_resid"][:_at_tr_end],
                "placed": data["y_train_placed"][:_at_tr_end],
                "norm_pos": data["y_train_norm_pos"][:_at_tr_end],
                "fp": data["fp_train"][:_at_tr_end],
                "ip": data["ip_train"][:_at_tr_end],
                "odds": data["odds_train"][:_at_tr_end],
            }
            _at_targets_vl = {
                "rel": data["y_train_rel"][_at_vl_beg:],
                "lb": data["y_train_lb"][_at_vl_beg:],
                "won": data["y_train_won"][_at_vl_beg:],
                "resid": data["y_train_resid"][_at_vl_beg:],
                "placed": data["y_train_placed"][_at_vl_beg:],
                "norm_pos": data["y_train_norm_pos"][_at_vl_beg:],
                "fp": data["fp_train"][_at_vl_beg:],
                "ip": data["ip_train"][_at_vl_beg:],
                "odds": data["odds_train"][_at_vl_beg:],
            }

            _tuned_params: dict[str, dict] = {}
            _tune_models = {"classifier": "Win Classifier", "place": "Place Classifier"}
            _tune_metrics = {"classifier": "Hybrid", "place": "Hybrid"}
            if getattr(config, "TRAIN_RANKER", False):
                _tune_models["ranker"] = "Race Ranker"
                _tune_metrics["ranker"] = "Ranker Hybrid"
            _model_count = max(len(_tune_models), 1)
            for _m_i, (_mk, _m_label) in enumerate(sorted(_tune_models.items())):
                _m_pct = 0.36 + (_m_i / _model_count) * 0.25
                _m_metric = _tune_metrics[_mk]
                _cb(f"Phase 1b — Tuning {_m_label} ({_m_i + 1}/{_model_count}) …", _m_pct)

                def _at_cb(tnum, total, score, _lbl=_m_label, _base=_m_pct, _met=_m_metric):
                    _cb(
                        f"Tuning {_lbl} — trial {tnum}/{total} ({_met} {score:.4f})",
                        _base + (tnum / total) * (0.25 / _model_count),
                    )

                _at_result = self._auto_tune_model(
                    _mk, _at_X_tr, _at_X_vl,
                    _at_targets_tr, _at_targets_vl,
                    _at_g_tr, _at_g_vl,
                    sw_train=_at_sw,
                    train_dates=pd.to_datetime(data["train_df"]["race_date"].iloc[:_at_tr_end]),
                    n_trials=_at_trials,
                    callback=_at_cb,
                )
                _tuned_params[_mk] = _at_result["best_params"]
                logger.info(f"  {_m_label}: {_m_metric} {_at_result['best_score']:.6f}")
            params = _tuned_params
            _p = _tuned_params

            # Apply best pruning params to config for the full retrain
            # (take from any model's best params — same values across models)
            for _mk_params in _tuned_params.values():
                if "FEATURE_PRUNE_FRACTION" in _mk_params:
                    config.FEATURE_PRUNE_FRACTION = _mk_params.pop("FEATURE_PRUNE_FRACTION")
                if "FEATURE_CORR_THRESHOLD" in _mk_params:
                    config.FEATURE_CORR_THRESHOLD = _mk_params.pop("FEATURE_CORR_THRESHOLD")
                break  # same values in each model's params

        # ── Phase 2: Retrain on full training data ───────────────
        _cb("Phase 2 — Retraining on full data …", 0.62)
        all_metrics: dict = {}
        sw_full = data["sample_weight_train"]
        train_dates_full = pd.to_datetime(data["train_df"]["race_date"])

        def _weights_for_params(base_dates, model_params, default_weights):
            _hl, _seasonal, _shape = self._extract_recency_params(model_params)
            if _hl is None and _seasonal is None and _shape is None:
                return default_weights
            return compute_recency_sample_weights(
                base_dates,
                half_life_days=_hl,
                seasonal_boost=_seasonal,
                decay_shape=_shape,
            )

        logger.info("Retraining 3 task-specific models on full training data …")

        n_test = len(X_eval)

        # Internal temporal validation split for early stopping
        _es_rounds = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))
        if _es_rounds > 0:
            es_bnd = max(1, int(n_races * 0.9))
            if es_bnd >= n_races:
                es_bnd = n_races - 1
            es_val_start = race_dates_idx[es_bnd]
            es_purge_cut = es_val_start - pd.Timedelta(days=purge_gap)
            es_tr_mask = race_dates_idx[:es_bnd] <= es_purge_cut
            if es_tr_mask.sum() >= 20:
                es_tr_end_race = int(np.where(es_tr_mask)[0][-1])
            else:
                es_tr_end_race = es_bnd - 1
            es_tr_end = int(cum_g[es_tr_end_race])
            es_vl_beg = int(race_starts_idx[es_bnd])
            use_es = (es_vl_beg < len(X_train)) and (es_tr_end > 0)
        else:
            use_es = False

        if use_es:
            X_es_tr = X_train[:es_tr_end]
            X_es_vl = X_train[es_vl_beg:]
            g_es_tr = groups_train[:es_tr_end_race + 1]
            g_es_vl = groups_train[es_bnd:]
            sw_es_tr = sw_full[:es_tr_end]
            y_es_rel_tr = data["y_train_rel"][:es_tr_end]
            y_es_rel_vl = data["y_train_rel"][es_vl_beg:]
            y_es_won_tr = data["y_train_won"][:es_tr_end]
            y_es_won_vl = data["y_train_won"][es_vl_beg:]
            y_es_norm_tr = 1.0 - data["y_train_norm_pos"][:es_tr_end]
            y_es_norm_vl = 1.0 - data["y_train_norm_pos"][es_vl_beg:]
            y_es_placed_tr = data["y_train_placed"][:es_tr_end]
            y_es_placed_vl = data["y_train_placed"][es_vl_beg:]
            fp_es_tr = data["fp_train"][:es_tr_end]
            fp_es_vl = data["fp_train"][es_vl_beg:]
            ip_es_tr = data["ip_train"][:es_tr_end]
            ip_es_vl = data["ip_train"][es_vl_beg:]
            dates_es_tr = train_dates_full.iloc[:es_tr_end]
            logger.info(
                f"  Phase-2 early stopping split: {len(X_es_tr)} train, "
                f"{len(X_es_vl)} val ({es_vl_beg - es_tr_end} runners purged)"
            )
        elif _es_rounds > 0:
            logger.info("  Phase-2 early stopping disabled (insufficient split size)")
        else:
            logger.info("  Phase-2 early stopping disabled by config")

        # 1) Win Classifier (Value Bets / Top Pick model)
        _clf_fw = self.frameworks.get("classifier", "cat").upper()
        _cb(f"Phase 2 — Training Win Classifier ({_clf_fw}) …", 0.72)
        logger.info(f"Training Win Classifier ({_clf_fw}) — Value model …")
        _sw_clf_full = _weights_for_params(train_dates_full, _p.get("classifier"), sw_full)
        if use_es:
            n_pos_w = int(y_es_won_tr.sum())
            n_neg_w = len(y_es_won_tr) - n_pos_w
            _sw_clf_es = _weights_for_params(dates_es_tr, _p.get("classifier"), sw_es_tr)
            self.clf_model = self._train_win_classifier(
                X_es_tr, y_es_won_tr,
                scale_pos_weight=n_neg_w / max(n_pos_w, 1),
                params=_p.get("classifier"), sample_weight=_sw_clf_es,
                eval_set=[(X_es_vl, y_es_won_vl)],
            )
        else:
            n_pos_w = int(data["y_train_won"].sum())
            n_neg_w = len(data["y_train_won"]) - n_pos_w
            self.clf_model = self._train_win_classifier(
                X_train, data["y_train_won"],
                scale_pos_weight=n_neg_w / max(n_pos_w, 1),
                params=_p.get("classifier"), sample_weight=_sw_clf_full,
            )

        # 2) Place Classifier (EW model)
        _place_fw = self.frameworks.get("place", "cat").upper()
        _cb(f"Phase 2 — Training Place Classifier ({_place_fw}) …", 0.80)
        logger.info(f"Training Place Classifier ({_place_fw}) — EW model …")
        _sw_place_full = _weights_for_params(train_dates_full, _p.get("place"), sw_full)
        if use_es:
            n_pos_p = int(y_es_placed_tr.sum())
            n_neg_p = len(y_es_placed_tr) - n_pos_p
            _sw_place_es = _weights_for_params(dates_es_tr, _p.get("place"), sw_es_tr)
            self.place_model = self._train_place_classifier(
                X_es_tr, y_es_placed_tr,
                scale_pos_weight=n_neg_p / max(n_pos_p, 1),
                params=_p.get("place"), sample_weight=_sw_place_es,
                eval_set=[(X_es_vl, y_es_placed_vl)],
            )
        else:
            n_pos_p = int(data["y_train_placed"].sum())
            n_neg_p = len(data["y_train_placed"]) - n_pos_p
            self.place_model = self._train_place_classifier(
                X_train, data["y_train_placed"],
                scale_pos_weight=n_neg_p / max(n_pos_p, 1),
                params=_p.get("place"), sample_weight=_sw_place_full,
            )

        if getattr(config, "TRAIN_RANKER", False):
            _ranker_fw = self.frameworks.get("ranker", "lgbm").upper()
            _cb(f"Phase 2 — Training Race Ranker ({_ranker_fw}) …", 0.84)
            logger.info(f"Training Race Ranker ({_ranker_fw}) …")
            _sw_ranker_full = _weights_for_params(train_dates_full, _p.get("ranker"), sw_full)
            if use_es:
                _sw_ranker_es = _weights_for_params(dates_es_tr, _p.get("ranker"), sw_es_tr)
                self.ranker_model = self._train_ranker(
                    X_es_tr,
                    y_es_rel_tr,
                    g_es_tr,
                    params=_p.get("ranker"),
                    sample_weight=_sw_ranker_es,
                    eval_set=[(X_es_vl, y_es_rel_vl, g_es_vl)],
                )
            else:
                self.ranker_model = self._train_ranker(
                    X_train,
                    data["y_train_rel"],
                    groups_train,
                    params=_p.get("ranker"),
                    sample_weight=_sw_ranker_full,
                )
        else:
            self.ranker_model = None

        _used_training_slab_cal = self._fit_training_slab_calibrators(
            X_train=X_train,
            groups_train=groups_train,
            y_won=data["y_train_won"],
            y_placed=data["y_train_placed"],
            finish_positions=data["fp_train"],
            model_params=_p,
            sample_weight=sw_full,
            race_dates=race_dates_idx,
        )
        if _used_training_slab_cal:
            logger.info("  Calibration refit on latest training slab; holdout left untouched.")
        else:
            logger.info("  Training-slab calibration unavailable; using OOF calibration from Phase 1.")

        # ── Market anchor (Benter combination) fitted on OOF ─────
        self.market_anchor = None
        if (
            getattr(config, "MARKET_ANCHOR", False)
            and not _oof_df.empty
            and "odds" in _oof_df.columns
        ):
            _anchor_model_probs = self._calibrate_win_probs(
                _all_clf_logits, _all_groups, _oof_df,
            )
            _anchor_odds = pd.to_numeric(_oof_df["odds"], errors="coerce").values
            _anchor_mkt, _anchor_valid = market_probs_from_odds(_anchor_odds, _all_groups)
            self.market_anchor = fit_market_anchor(
                _anchor_model_probs,
                _anchor_mkt,
                _all_groups,
                (_all_fp == 1).astype(np.int64),
                race_valid=_anchor_valid,
            )
            if self.market_anchor:
                logger.info(
                    "  Market anchor fitted on OOF: alpha=%.4f (model), beta=%.4f (market)",
                    self.market_anchor["alpha"], self.market_anchor["beta"],
                )
            else:
                logger.info("  Market anchor skipped (insufficient OOF odds coverage).")

        # ── Score test set with each model ───────────────────────
        _cb("Evaluating models on holdout validation set …", 0.86)
        clf_test_probs_raw = self.clf_model.predict_proba(X_eval)[:, 1]
        place_test_probs_raw = self.place_model.predict_proba(X_eval)[:, 1]
        ranker_test_scores = self.ranker_model.predict(X_eval) if self.ranker_model is not None else None

        # Apply training-only calibration (+ market anchor when fitted)
        win_probs = self._calibrate_win_probs(clf_test_probs_raw, groups_eval, eval_df)
        win_probs = self._apply_market_anchor(win_probs, groups_eval, eval_df)
        place_probs = self._calibrate_place_probs(place_test_probs_raw)
        _vc = dict(value_config or {})
        if auto_tune is not None:
            # Tune the value threshold on OOF predictions, NOT the
            # holdout — tuning on the same set the betting simulation
            # reports on would make the headline ROI partly in-sample.
            if not _oof_df.empty:
                _vt_win = self._calibrate_win_probs(
                    _all_clf_logits, _all_groups, _oof_df,
                )
                _vt_win = self._apply_market_anchor(_vt_win, _all_groups, _oof_df)
                _vt_place = self._calibrate_place_probs(_all_place_probs)
                _vc = self._tune_value_threshold(
                    win_probs=_vt_win,
                    groups_eval=_all_groups,
                    eval_df=_oof_df,
                    value_config=_vc,
                    place_probs=_vt_place,
                )
            else:
                logger.warning(
                    "No OOF data for threshold tuning — falling back to "
                    "the holdout (reported ROI will be optimistic)."
                )
                _vc = self._tune_value_threshold(
                    win_probs=win_probs,
                    groups_eval=groups_eval,
                    eval_df=eval_df,
                    value_config=_vc,
                    place_probs=place_probs,
                )

        # ── Evaluate each model for its task ─────────────────────
        # Win Classifier: calibration + value-bet metrics
        # Rank and calibrate on the final deployed probabilities (incl.
        # market anchor) — same basis as the baselines.
        all_metrics["win_classifier"] = self._evaluate_as_ranker(
            win_probs, y_eval_rel, groups_eval, eval_df,
            "WIN_CLASSIFIER (Value)",
            calibrated_probs=win_probs,
            value_threshold=float(_vc.get("value_threshold", 0.05)),
        )
        # Place Classifier: place-specific metrics
        all_metrics["place_classifier"] = self._evaluate_place_model(
            place_probs, place_test_probs_raw, eval_df, groups_eval,
        )
        if ranker_test_scores is not None:
            all_metrics["ranker"] = self._evaluate_as_ranker(
                ranker_test_scores,
                y_eval_rel,
                groups_eval,
                eval_df,
                "RANKER",
            )

        # ── Linear baseline (sanity reference, not persisted) ─────
        # A logistic regression on the same features: if the tree
        # models don't clearly beat this, the extra complexity isn't
        # earning its keep on the current feature set.
        _cb("Training linear baseline (logistic regression) …", 0.88)
        try:
            _base_win = self._fit_linear_baseline(
                X_train, data["y_train_won"], sample_weight=sw_full,
            )
            _base_win_probs = self._baseline_predict_proba(_base_win, X_eval)
            all_metrics["baseline_win"] = self._evaluate_as_ranker(
                _base_win_probs, y_eval_rel, groups_eval, eval_df,
                "BASELINE_LOGREG_WIN",
                calibrated_probs=_base_win_probs,
                value_threshold=float(_vc.get("value_threshold", 0.05)),
            )
            _base_place = self._fit_linear_baseline(
                X_train, data["y_train_placed"], sample_weight=sw_full,
            )
            _base_place_probs = self._baseline_predict_proba(_base_place, X_eval)
            all_metrics["baseline_place"] = self._evaluate_place_model(
                _base_place_probs, _base_place_probs, eval_df, groups_eval,
            )
            logger.info(
                "Linear baseline (win): NDCG@1 %.4f / Brier %.6f  vs  tree: NDCG@1 %.4f / Brier %.6f",
                all_metrics["baseline_win"].get("ndcg_at_1", float("nan")),
                all_metrics["baseline_win"].get("brier_score", float("nan")),
                all_metrics["win_classifier"].get("ndcg_at_1", float("nan")),
                all_metrics["win_classifier"].get("brier_score", float("nan")),
            )
        except Exception as exc:
            logger.warning("Linear baseline skipped: %s", exc)

        # ── Market baseline (the benchmark that actually matters) ─
        # Overround-normalised implied probabilities from the odds.
        # If the win classifier doesn't beat this, the model knows
        # nothing the market doesn't already price in.
        try:
            if "odds" in eval_df.columns and eval_df["odds"].notna().any():
                _mkt_probs = normalise_implied_prob_by_race(eval_df)
                all_metrics["baseline_market"] = self._evaluate_as_ranker(
                    _mkt_probs, y_eval_rel, groups_eval, eval_df,
                    "BASELINE_MARKET",
                    calibrated_probs=_mkt_probs,
                )
                logger.info(
                    "Market baseline: NDCG@1 %.4f / Brier %.6f  vs  win classifier: NDCG@1 %.4f / Brier %.6f",
                    all_metrics["baseline_market"].get("ndcg_at_1", float("nan")),
                    all_metrics["baseline_market"].get("brier_score", float("nan")),
                    all_metrics["win_classifier"].get("ndcg_at_1", float("nan")),
                    all_metrics["win_classifier"].get("brier_score", float("nan")),
                )
        except Exception as exc:
            logger.warning("Market baseline skipped: %s", exc)

        self.metrics = all_metrics

        # ── OOF metrics (overfit diagnostics) ────────────────────
        _cb("Computing OOF diagnostics …", 0.89)
        logger.info("Computing OOF metrics for overfit diagnostics …")
        train_metrics: dict = {}
        if not _oof_df.empty:
            _oof_rel = _oof_df["relevance"].values
            _oof_win_probs = self._calibrate_win_probs(
                _all_clf_logits, _all_groups, _oof_df,
            )
            _oof_win_probs = self._apply_market_anchor(_oof_win_probs, _all_groups, _oof_df)
            train_metrics["win_classifier"] = self._evaluate_as_ranker(
                _oof_win_probs, _oof_rel, _all_groups,
                _oof_df, "WIN_CLASSIFIER_OOF",
                calibrated_probs=_oof_win_probs,
                value_threshold=float(_vc.get("value_threshold", 0.05)),
            )
            _oof_place_cal = self._calibrate_place_probs(_all_place_probs)
            train_metrics["place_classifier"] = self._evaluate_place_model(
                _oof_place_cal, _all_place_probs, _oof_df, _all_groups,
            )
            if oof_ranker_scores_parts:
                _all_ranker_scores = np.concatenate(oof_ranker_scores_parts)
                train_metrics["ranker"] = self._evaluate_as_ranker(
                    _all_ranker_scores,
                    _oof_rel,
                    _all_groups,
                    _oof_df,
                    "RANKER_OOF",
                )
            else:
                train_metrics["ranker"] = {}
        else:
            train_metrics["win_classifier"] = {}
            train_metrics["place_classifier"] = {}
            train_metrics["ranker"] = {}
        self.train_metrics = train_metrics

        # ── Betting simulation ───────────────────────────────────
        _cb("Analysing holdout validation betting performance …", 0.92)
        self.test_analysis = analyse_test_set(
            win_probs=win_probs,
            groups_test=groups_eval,
            test_df=eval_df,
            value_threshold=_vc.get("value_threshold", 0.05),
            staking_mode=_vc.get("staking_mode", "flat"),
            kelly_fraction=_vc.get("kelly_fraction", 0.25),
            bankroll=_vc.get("bankroll", 100.0),
            place_probs=place_probs,
            ew_min_place_edge=_vc.get("ew_min_place_edge"),
        )

        # Feature importance
        if self.clf_model is not None:
            fi = get_feature_importance(self.clf_model, self.feature_cols)
            logger.info(f"\nTop 15 Features:\n{fi.head(15).to_string()}")

        if save:
            _cb("Saving models …", 0.97)
            self.save()

        _cb("Training complete ✅", 1.0)
        return all_metrics

    # ── Sub-model factories ──────────────────────────────────────
    @staticmethod
    def _filter_params(params: dict | None, framework: str = "xgb") -> dict:
        """Keep only framework-compatible hyper-parameters."""
        if not params:
            return {}
        if framework == "lgbm":
            valid = _LGBM_VALID_HP
        elif framework == "cat":
            valid = _CAT_VALID_HP
        else:
            valid = _XGB_VALID_HP
        return {k: v for k, v in params.items() if k in valid}

    # ── Per-model Optuna auto-tuning ─────────────────────────────
    def _auto_tune_model(
        self,
        model_key: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        targets_train: dict,
        targets_val: dict,
        groups_train: np.ndarray,
        groups_val: np.ndarray,
        eval_df: pd.DataFrame | None = None,
        sw_train: np.ndarray | None = None,
        train_dates=None,
        n_trials: int = 30,
        callback=None,
        storage: str | None = None,
        study_name: str | None = None,
        load_if_exists: bool = False,
        folds: list[dict] | None = None,
    ) -> dict:
        """Run Optuna to find the best HP for a single sub-model.

        Args:
            model_key: One of ``_ALL_MODEL_KEYS``.
            targets_train / targets_val: dicts with keys
                ``"rel"``, ``"lb"``, ``"won"``, ``"resid"``, ``"placed"``,
                ``"norm_pos"``, ``"fp"``.
            callback: ``(trial_num, total, best_score)``
        Returns:
            dict with ``best_params``, ``best_score``, ``n_trials``.
        """
        import optuna
        import os as _os
        self._autotune_njobs = max(1, (_os.cpu_count() or 4) // 2)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        _metric_names = {
            "classifier": "Top1-Calibration Hybrid",
            "place": "Place Top-3 Hybrid",
            "ranker": "Ranker Top-1 Hybrid",
        }
        _metric_directions = {
            "classifier": "maximize",
            "place": "maximize",
            "ranker": "maximize",
        }

        fw = self.frameworks.get(model_key, "xgb")
        _feat_cols = self.feature_cols
        _search_space = get_autotune_search_space(
            model_key,
            fw,
            include_recency=train_dates is not None,
        )

        def _bounded_positive(value, *, low: float, high: float, default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                value_f = float(value)
            except Exception:
                return default
            if not np.isfinite(value_f):
                return default
            if high <= low:
                return default
            scaled = (value_f - low) / (high - low)
            return float(np.clip(scaled, 0.0, 1.0))

        def _bounded_negative_loss(value, *, upper: float, default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                value_f = float(value)
            except Exception:
                return default
            if not np.isfinite(value_f):
                return default
            return float(np.clip(1.0 - (value_f / max(upper, 1e-9)), 0.0, 1.0))

        def _bounded_symmetric(value, *, cap: float, default: float = 0.0) -> float:
            if value is None:
                return default
            try:
                value_f = float(value)
            except Exception:
                return default
            if not np.isfinite(value_f):
                return default
            return float(np.clip(value_f / max(cap, 1e-9), -1.0, 1.0))

        def _classifier_hybrid_score(metrics: dict[str, float | int | None]) -> float:
            top1 = _bounded_positive(metrics.get("top1_accuracy"), low=0.0, high=1.0)
            win_top3 = _bounded_positive(metrics.get("win_in_top3"), low=0.0, high=1.0)
            full_brier = _bounded_negative_loss(metrics.get("brier_score"), upper=0.25)
            value_brier = _bounded_negative_loss(
                metrics.get("value_bet_brier") if metrics.get("value_bet_brier") is not None else metrics.get("brier_score"),
                upper=0.30,
                default=full_brier,
            )
            value_bets = int(metrics.get("value_bets") or 0)
            support = float(np.clip(value_bets / 50.0, 0.0, 1.0))
            selected_quality = ((1.0 - support) * full_brier) + (support * value_brier)

            return float(
                0.45 * top1
                + 0.20 * win_top3
                + 0.175 * full_brier
                + 0.175 * selected_quality
            )

        def _place_hybrid_score(metrics: dict[str, float | int | None]) -> float:
            top3_accuracy = _bounded_positive(metrics.get("top3_accuracy"), low=0.0, high=1.0)
            place_precision = _bounded_positive(metrics.get("place_precision"), low=0.0, high=1.0)
            cal_brier = _bounded_negative_loss(metrics.get("brier_calibrated"), upper=0.25)
            logloss_quality = _bounded_negative_loss(metrics.get("log_loss"), upper=0.70)

            return float(
                0.35 * top3_accuracy
                + 0.25 * place_precision
                + 0.20 * cal_brier
                + 0.20 * logloss_quality
            )

        def _ranker_hybrid_score(metrics: dict[str, float | int | None]) -> float:
            ndcg1 = _bounded_positive(metrics.get("ndcg_at_1"), low=0.0, high=1.0)
            ndcg3 = _bounded_positive(metrics.get("ndcg_at_3"), low=0.0, high=1.0)
            top1 = _bounded_positive(metrics.get("top1_accuracy"), low=0.0, high=1.0)
            win_top3 = _bounded_positive(metrics.get("win_in_top3"), low=0.0, high=1.0)
            rps_quality = _bounded_negative_loss(metrics.get("rps"), upper=0.35)

            return float(
                0.30 * ndcg1
                + 0.20 * ndcg3
                + 0.25 * top1
                + 0.15 * win_top3
                + 0.10 * rps_quality
            )

        def _score_split(
            params: dict,
            split_X_train: np.ndarray,
            split_X_val: np.ndarray,
            split_targets_train: dict,
            split_targets_val: dict,
            split_groups_train: np.ndarray,
            split_groups_val: np.ndarray,
            split_sw_train: np.ndarray | None,
            split_train_dates,
            col_mask: np.ndarray | None = None,
        ) -> tuple[float, dict[str, float | int | None]]:

            # Apply column mask from trial-level pruning
            if col_mask is not None:
                split_X_train = split_X_train[:, col_mask]
                split_X_val = split_X_val[:, col_mask]

            def _win_metrics_eval(probs):
                won = split_targets_val["won"].astype(np.float32)
                eps = 1e-9
                probs = np.clip(probs, eps, 1 - eps)
                logloss_value = float(np.mean(won * np.log(probs) + (1 - won) * np.log(1 - probs)))
                brier = float(brier_score_loss(won, probs))

                top1_total = 0
                top1_hits = 0
                win3_hits = 0
                offset = 0
                for group_size in split_groups_val:
                    group_won = won[offset:offset + group_size]
                    group_probs = probs[offset:offset + group_size]
                    offset += group_size
                    if group_size < 2 or group_won.max() == group_won.min():
                        continue
                    top1_total += 1
                    if int(np.argmax(group_probs)) == int(np.argmax(group_won)):
                        top1_hits += 1
                    if int(np.argmax(group_won)) in set(np.argsort(group_probs)[-3:]):
                        win3_hits += 1

                metrics: dict[str, float | int | None] = {
                    "brier_score": round(brier, 6),
                    "log_loss": round(-logloss_value, 6),
                    "top1_accuracy": round(top1_hits / top1_total, 6) if top1_total else 0.0,
                    "win_in_top3": round(win3_hits / top1_total, 6) if top1_total else 0.0,
                    "total_races": int(top1_total),
                }

                if "odds" in split_targets_val:
                    odds_arr = np.asarray(split_targets_val["odds"], dtype=np.float64)
                    vb = _value_bet_selection(probs, odds_arr, float(getattr(config, "VALUE_THRESHOLD", 0.05)))
                    vb_mask = vb["mask"]
                    vb_count = int(vb_mask.sum())
                    metrics["value_bets"] = vb_count
                    if vb_count > 0:
                        vb_won = won[vb_mask]
                        vb_probs = probs[vb_mask]
                        vb_returns = np.where(vb_won == 1, odds_arr[vb_mask] - 1.0, -1.0)
                        metrics["value_bet_roi"] = round(float(vb_returns.mean()), 6)
                        metrics["avg_edge"] = round(float(vb["edge"][vb_mask].mean()), 6)
                        metrics["value_bet_brier"] = round(float(brier_score_loss(vb_won, vb_probs)), 6)
                        try:
                            metrics["value_bet_log_loss"] = round(float(log_loss(vb_won, vb_probs)), 6)
                        except Exception:
                            metrics["value_bet_log_loss"] = None
                    else:
                        metrics["value_bet_roi"] = None
                        metrics["avg_edge"] = None
                        metrics["value_bet_brier"] = None
                        metrics["value_bet_log_loss"] = None

                metrics["objective_score"] = round(_classifier_hybrid_score(metrics), 6)
                return float(metrics["objective_score"]), metrics

            def _place_metrics_eval(scores):
                placed = split_targets_val["placed"].astype(np.float32)
                finish_position = np.asarray(split_targets_val["fp"], dtype=np.float64)
                probs = 1.0 / (1.0 + np.exp(-scores))
                eps = 1e-9
                probs = np.clip(probs, eps, 1 - eps)
                logloss_value = float(np.mean(placed * np.log(probs) + (1 - placed) * np.log(1 - probs)))
                brier_cal = float(brier_score_loss(placed, probs))

                place_hits = 0
                total_placed = 0
                top3_hits = 0
                top3_total = 0
                offset = 0
                for group_size in split_groups_val:
                    fp_race = finish_position[offset:offset + group_size]
                    placed_race = placed[offset:offset + group_size]
                    group_probs = probs[offset:offset + group_size]
                    offset += group_size

                    if group_size <= 0:
                        continue

                    places_paid = int(np.clip(np.sum(placed_race == 1), 0, group_size))
                    predicted_place_idx = np.argsort(-group_probs)[:max(places_paid, 1)]
                    actual_placed = set(np.where(placed_race == 1)[0])
                    place_hits += len(set(predicted_place_idx) & actual_placed)
                    total_placed += len(actual_placed)

                    actual_top3 = set(np.where((fp_race >= 1) & (fp_race <= min(3, group_size)))[0])
                    if actual_top3:
                        predicted_top3 = np.argsort(-group_probs)[:min(3, group_size)]
                        top3_hits += len(set(predicted_top3) & actual_top3)
                        top3_total += len(actual_top3)

                metrics = {
                    "brier_calibrated": round(brier_cal, 6),
                    "log_loss": round(-logloss_value, 6),
                    "place_precision": round(place_hits / max(total_placed, 1), 6),
                    "top3_accuracy": round(top3_hits / max(top3_total, 1), 6),
                    "total_races": int(len(split_groups_val)),
                }
                metrics["objective_score"] = round(_place_hybrid_score(metrics), 6)
                return float(metrics["objective_score"]), metrics

            def _ranker_metrics_eval(scores):
                rel = np.asarray(split_targets_val["rel"], dtype=np.float64)
                finish_position = np.asarray(split_targets_val["fp"], dtype=np.float64)
                scores_arr = np.asarray(scores, dtype=np.float64)
                probs = _grouped_softmax(scores_arr, split_groups_val)
                probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
                y_won = (finish_position == 1).astype(np.float64)

                ndcg1_vals: list[float] = []
                ndcg3_vals: list[float] = []
                top1_hits = 0
                win3_hits = 0
                total = 0
                offset = 0
                for group_size in split_groups_val:
                    gl = rel[offset:offset + group_size]
                    gs = scores_arr[offset:offset + group_size]
                    offset += group_size
                    if group_size < 2 or gl.max() == gl.min() or gs.max() == gs.min():
                        continue
                    total += 1
                    try:
                        ndcg1_vals.append(float(ndcg_score([gl], [gs], k=1)))
                        ndcg3_vals.append(float(ndcg_score([gl], [gs], k=3)))
                    except ValueError:
                        pass
                    if int(np.argmax(gs)) == int(np.argmax(gl)):
                        top1_hits += 1
                    if int(np.argmax(gl)) in set(np.argsort(gs)[-3:]):
                        win3_hits += 1

                metrics = {
                    "rps": round(rps_per_race(probs, finish_position, split_groups_val), 6),
                    "brier_score": round(float(brier_score_loss(y_won, probs_clipped)), 6),
                    "log_loss": round(float(log_loss(y_won, probs_clipped)), 6),
                    "ndcg_at_1": round(float(np.mean(ndcg1_vals)) if ndcg1_vals else 0.0, 6),
                    "ndcg_at_3": round(float(np.mean(ndcg3_vals)) if ndcg3_vals else 0.0, 6),
                    "top1_accuracy": round(top1_hits / total, 6) if total else 0.0,
                    "win_in_top3": round(win3_hits / total, 6) if total else 0.0,
                    "total_races": int(total),
                }
                metrics["objective_score"] = round(_ranker_hybrid_score(metrics), 6)
                return float(metrics["objective_score"]), metrics

            trial_sw = split_sw_train
            if model_key in {"classifier", "place", "ranker"} and split_train_dates is not None:
                trial_sw = compute_recency_sample_weights(
                    split_train_dates,
                    half_life_days=params["recency_half_life_days"],
                    seasonal_boost=params["recency_seasonal_boost"],
                    decay_shape=params.get("recency_decay_shape", "exp"),
                )

            if model_key == "classifier":
                n_pw = max(int(split_targets_train["won"].sum()), 1)
                mdl = self._train_win_classifier(
                    split_X_train, split_targets_train["won"],
                    scale_pos_weight=(len(split_targets_train["won"]) - n_pw) / n_pw,
                    params=params, sample_weight=trial_sw,
                )
                scores = mdl.predict_proba(split_X_val)[:, 1]
            elif model_key == "place":
                n_pp = max(int(split_targets_train["placed"].sum()), 1)
                mdl = self._train_place_classifier(
                    split_X_train, split_targets_train["placed"],
                    scale_pos_weight=(len(split_targets_train["placed"]) - n_pp) / n_pp,
                    params=params, sample_weight=trial_sw,
                )
                scores = _proba_to_logit(mdl.predict_proba(split_X_val)[:, 1])
            elif model_key == "ranker":
                mdl = self._train_ranker(
                    split_X_train,
                    split_targets_train["rel"],
                    split_groups_train,
                    params=params,
                    sample_weight=trial_sw,
                )
                scores = mdl.predict(split_X_val)
            else:
                raise ValueError(f"Unknown model_key: {model_key}")

            if model_key == "classifier":
                return _win_metrics_eval(scores)
            if model_key == "ranker":
                return _ranker_metrics_eval(scores)
            return _place_metrics_eval(scores)

        def _aggregate_trial_metrics(split_metric_dicts: list[dict[str, float | int | None]]) -> dict[str, float | int]:
            aggregated: dict[str, float | int] = {}
            if not split_metric_dicts:
                return aggregated

            keys = set().union(*(m.keys() for m in split_metric_dicts))
            for key in keys:
                values = [m.get(key) for m in split_metric_dicts if isinstance(m.get(key), (int, float)) and m.get(key) is not None]
                if not values:
                    continue
                if key in {"value_bets", "total_races"}:
                    aggregated[key] = int(round(float(np.mean(values))))
                else:
                    aggregated[key] = round(float(np.mean(values)), 6)
            return aggregated

        def objective(trial):
            params = _suggest_from_autotune_space(trial, _search_space)

            # Extract feature-pruning params (don't pass to model constructors)
            _trial_prune_frac = params.pop("FEATURE_PRUNE_FRACTION", 0.0)
            _trial_corr_thresh = params.pop("FEATURE_CORR_THRESHOLD", 0.0)

            # Pre-compute column mask once for this trial (using first fold or main split)
            _trial_col_mask = None
            if _feat_cols is not None and _trial_prune_frac > 0:
                _ref_X = folds[0]["X_train"] if folds else X_train
                _ref_y = (folds[0]["targets_train"] if folds else targets_train)["won"]
                _trial_col_mask = _quick_prune_mask(
                    _ref_X, _ref_y, _feat_cols,
                    prune_fraction=_trial_prune_frac,
                    corr_threshold=_trial_corr_thresh,
                )

            if folds:
                split_scores = []
                split_metrics = []
                for fi, fold in enumerate(folds):
                    s, metrics = _score_split(
                        params,
                        fold["X_train"],
                        fold["X_val"],
                        fold["targets_train"],
                        fold["targets_val"],
                        fold["groups_train"],
                        fold["groups_val"],
                        fold.get("sw_train"),
                        fold.get("train_dates"),
                        col_mask=_trial_col_mask,
                    )
                    split_scores.append(s)
                    split_metrics.append(metrics)
                    # Report intermediate fold score so Optuna can prune
                    trial.report(float(np.mean(split_scores)), fi)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                score = float(np.mean(split_scores))
                trial_metrics = _aggregate_trial_metrics(split_metrics)
            else:
                score, trial_metrics = _score_split(
                    params,
                    X_train,
                    X_val,
                    targets_train,
                    targets_val,
                    groups_train,
                    groups_val,
                    sw_train,
                    train_dates,
                    col_mask=_trial_col_mask,
                )

            for key, value in trial_metrics.items():
                if isinstance(value, (int, float)):
                    trial.set_user_attr(key, value)

            if callback is not None:
                _mn = _metric_names[model_key]
                callback(trial.number + 1, n_trials, score)
            return score

        _study_direction = _metric_directions[model_key]
        _study_kwargs = {"direction": _study_direction}
        _study_kwargs["pruner"] = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=0,
        )
        if storage is not None:
            _study_kwargs["storage"] = storage
            _study_kwargs["study_name"] = study_name or f"autotune_{model_key}"
            _study_kwargs["load_if_exists"] = load_if_exists

        study = optuna.create_study(**_study_kwargs)
        study.set_user_attr("objective_name", _metric_names[model_key])
        study.set_user_attr("objective_direction", _study_direction)
        existing_trials = len(study.trials)
        remaining_trials = int(n_trials)
        if remaining_trials > 0:
            try:
                study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)
            finally:
                self._autotune_njobs = -1  # restore full parallelism

        _mn = _metric_names[model_key]
        logger.info(
            f"  Optuna auto-tune ({model_key}, {fw}) — {len(study.trials)} total trials\n"
            f"    Best {_mn}: {study.best_value:.6f}\n"
            f"    Best params: {study.best_params}"
        )
        return {
            "best_params": study.best_params,
            "best_score": round(study.best_value, 6),
            "n_trials": len(study.trials),
            "target_trials": len(study.trials),
            "metric_name": _mn,
            "framework": fw,
            "study_name": study.study_name,
            "storage": storage,
            "cv_folds": len(folds) if folds else 1,
        }

    # ── Linear baseline ──────────────────────────────────────────
    @staticmethod
    def _sanitise_for_linear(X) -> np.ndarray:
        """Tree models tolerate NaN/inf natively; linear models don't."""
        Xs = np.asarray(X, dtype=np.float64).copy()
        Xs[~np.isfinite(Xs)] = np.nan
        return Xs

    @classmethod
    def _fit_linear_baseline(cls, X, y, sample_weight=None):
        """Fit a logistic-regression baseline on the same feature matrix."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, C=1.0)),
        ])
        pipe.fit(cls._sanitise_for_linear(X), y, logreg__sample_weight=sample_weight)
        return pipe

    @classmethod
    def _baseline_predict_proba(cls, pipe, X) -> np.ndarray:
        return pipe.predict_proba(cls._sanitise_for_linear(X))[:, 1]

    def _train_classifier(self, X, y, scale_pos_weight=1.0, params=None, sample_weight=None, eval_set=None):
        _ensure_lightgbm()
        _ensure_xgboost()
        _es_rounds = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))
        fw = self.frameworks.get("classifier", "xgb")
        hp = {
            "n_estimators": config.CLASSIFIER_PARAMS.get("n_estimators", 500),
            "max_depth": config.CLASSIFIER_PARAMS.get("max_depth", 7),
            "learning_rate": config.CLASSIFIER_PARAMS.get("learning_rate", 0.05),
            "subsample": config.CLASSIFIER_PARAMS.get("subsample", 0.8),
            "colsample_bytree": config.CLASSIFIER_PARAMS.get("colsample_bytree", 0.8),
            "reg_alpha": config.CLASSIFIER_PARAMS.get("reg_alpha", 0.05),
            "reg_lambda": config.CLASSIFIER_PARAMS.get("reg_lambda", 1.0),
            "linear_tree": bool(config.CLASSIFIER_PARAMS.get("linear_tree", False)),
        }
        hp.update(self._filter_params(params, fw))
        if fw == "lgbm":
            logger.info(
                "Training win classifier (legacy) with linear_tree=%s",
                bool(hp.get("linear_tree", False)),
            )
            hp.setdefault("min_child_samples", config.CLASSIFIER_PARAMS.get("min_child_samples", 10))
            model = _get_focal_lgbm_class()(
                objective=_focal_binary_objective,
                **hp,
                subsample_freq=1,
                random_state=config.RANDOM_SEED,
                n_jobs=self._autotune_njobs,
                verbose=-1,
            )
            fit_kw: dict = {"sample_weight": sample_weight}
            if eval_set is not None:
                from lightgbm import early_stopping as _lgb_es
                fit_kw["eval_set"] = eval_set
                fit_kw["eval_metric"] = _focal_logloss_eval
                fit_kw["callbacks"] = [_lgb_es(_es_rounds, verbose=False)]
        elif fw == "cat":
            _require_catboost()
            hp_cat = {
                "n_estimators": config.CLASSIFIER_PARAMS.get("n_estimators", 500),
                "depth": config.CLASSIFIER_PARAMS.get("max_depth", 7),
                "learning_rate": config.CLASSIFIER_PARAMS.get("learning_rate", 0.05),
                "l2_leaf_reg": 3.0,
            }
            hp_cat.update(self._filter_params(params, "cat"))
            model = CatBoostClassifier(
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=config.RANDOM_SEED,
                verbose=False,
                **hp_cat,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                fit_kw["eval_set"] = eval_set
                fit_kw["use_best_model"] = True
                fit_kw["early_stopping_rounds"] = _es_rounds
        else:
            hp["min_child_weight"] = config.CLASSIFIER_PARAMS.get("min_child_weight", 3)
            model = XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                **hp,
                random_state=config.RANDOM_SEED,
                n_jobs=self._autotune_njobs,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                model.set_params(early_stopping_rounds=_es_rounds)
                fit_kw["eval_set"] = eval_set
                fit_kw["verbose"] = False
        model.fit(X, y, **fit_kw)
        logger.info(f"Win Classifier ({fw.upper()}) training complete")
        return model

    def _train_win_classifier(self, X, y, scale_pos_weight=1.0, params=None, sample_weight=None, eval_set=None):
        """Train a binary classifier predicting P(win).

        Uses standard binary logloss without class reweighting.
        scale_pos_weight / is_unbalance / focal loss all amplify
        minority-class gradients, which causes premature early stopping
        (8-16 trees) before the model can learn subtle patterns.
        Without reweighting the model trains 300-900 trees and achieves
        significantly better logloss (0.284 vs 0.328).
        """
        _ensure_lightgbm()
        _ensure_xgboost()
        _es_rounds = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))
        fw = self.frameworks.get("classifier", "lgbm")
        hp = {
            "n_estimators": config.CLASSIFIER_PARAMS.get("n_estimators", 500),
            "max_depth": config.CLASSIFIER_PARAMS.get("max_depth", 7),
            "learning_rate": config.CLASSIFIER_PARAMS.get("learning_rate", 0.05),
            "subsample": config.CLASSIFIER_PARAMS.get("subsample", 0.8),
            "colsample_bytree": config.CLASSIFIER_PARAMS.get("colsample_bytree", 0.8),
            "reg_alpha": config.CLASSIFIER_PARAMS.get("reg_alpha", 0.05),
            "reg_lambda": config.CLASSIFIER_PARAMS.get("reg_lambda", 1.0),
            "linear_tree": bool(config.CLASSIFIER_PARAMS.get("linear_tree", False)),
        }
        if "num_leaves" in config.CLASSIFIER_PARAMS:
            hp["num_leaves"] = config.CLASSIFIER_PARAMS["num_leaves"]
        hp.update(self._filter_params(params, fw))
        if fw == "lgbm":
            logger.info(
                "Training win classifier with linear_tree=%s",
                bool(hp.get("linear_tree", False)),
            )
            hp.setdefault("min_child_samples", config.CLASSIFIER_PARAMS.get("min_child_samples", 10))
            model = LGBMClassifier(
                objective="binary",
                **hp,
                subsample_freq=1,
                random_state=config.RANDOM_SEED,
                n_jobs=self._autotune_njobs,
                verbose=-1,
            )
            fit_kw: dict = {"sample_weight": sample_weight}
            if eval_set is not None:
                from lightgbm import early_stopping as _lgb_es
                fit_kw["eval_set"] = eval_set
                fit_kw["eval_metric"] = "binary_logloss"
                fit_kw["callbacks"] = [_lgb_es(_es_rounds, verbose=False)]
        elif fw == "cat":
            _require_catboost()
            hp_cat = {
                "n_estimators": config.CLASSIFIER_PARAMS.get("n_estimators", 500),
                "depth": config.CLASSIFIER_PARAMS.get("max_depth", 7),
                "learning_rate": config.CLASSIFIER_PARAMS.get("learning_rate", 0.05),
                "l2_leaf_reg": 3.0,
            }
            hp_cat.update(self._filter_params(params, "cat"))
            model = CatBoostClassifier(
                loss_function="Logloss",
                random_seed=config.RANDOM_SEED,
                verbose=False,
                **hp_cat,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                fit_kw["eval_set"] = eval_set
                fit_kw["use_best_model"] = True
                fit_kw["early_stopping_rounds"] = _es_rounds
        else:
            hp["min_child_weight"] = config.CLASSIFIER_PARAMS.get("min_child_weight", 3)
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **hp,
                random_state=config.RANDOM_SEED,
                n_jobs=self._autotune_njobs,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                model.set_params(early_stopping_rounds=_es_rounds)
                fit_kw["eval_set"] = eval_set
                fit_kw["verbose"] = False
        model.fit(X, y, **fit_kw)
        logger.info(f"Win Classifier ({fw.upper()}) training complete")
        return model

    def _train_place_classifier(self, X, y, scale_pos_weight=1.0, params=None, sample_weight=None, eval_set=None):
        """Train a classifier predicting top-3 finish (place).

        Uses standard binary logloss (not focal loss) because the place
        target is nearly balanced (~40% positive rate).  Focal loss
        down-weights easy examples which hurts performance when classes
        are balanced.
        """
        _ensure_lightgbm()
        _ensure_xgboost()
        _es_rounds = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))
        fw = self.frameworks.get("place", "xgb")
        hp = {
            "n_estimators": config.PLACE_CLASSIFIER_PARAMS.get("n_estimators", 500),
            "max_depth": config.PLACE_CLASSIFIER_PARAMS.get("max_depth", 6),
            "learning_rate": config.PLACE_CLASSIFIER_PARAMS.get("learning_rate", 0.05),
            "subsample": config.PLACE_CLASSIFIER_PARAMS.get("subsample", 0.8),
            "colsample_bytree": config.PLACE_CLASSIFIER_PARAMS.get("colsample_bytree", 0.8),
            "reg_alpha": config.PLACE_CLASSIFIER_PARAMS.get("reg_alpha", 0.1),
            "reg_lambda": config.PLACE_CLASSIFIER_PARAMS.get("reg_lambda", 1.0),
            "linear_tree": bool(config.PLACE_CLASSIFIER_PARAMS.get("linear_tree", False)),
        }
        hp.update(self._filter_params(params, fw))
        if fw == "lgbm":
            logger.info(
                "Training place classifier with linear_tree=%s",
                bool(hp.get("linear_tree", False)),
            )
            hp.setdefault("min_child_samples", config.PLACE_CLASSIFIER_PARAMS.get("min_child_samples", 10))
            model = LGBMClassifier(
                objective="binary",
                **hp,
                subsample_freq=1,
                random_state=config.RANDOM_SEED,
                n_jobs=self._autotune_njobs,
                verbose=-1,
            )
            fit_kw: dict = {"sample_weight": sample_weight}
            if eval_set is not None:
                from lightgbm import early_stopping as _lgb_es
                fit_kw["eval_set"] = eval_set
                fit_kw["eval_metric"] = "binary_logloss"
                fit_kw["callbacks"] = [_lgb_es(_es_rounds, verbose=False)]
        elif fw == "cat":
            _require_catboost()
            hp_cat = {
                "n_estimators": config.PLACE_CLASSIFIER_PARAMS.get("n_estimators", 500),
                "depth": config.PLACE_CLASSIFIER_PARAMS.get("max_depth", 6),
                "learning_rate": config.PLACE_CLASSIFIER_PARAMS.get("learning_rate", 0.05),
                "l2_leaf_reg": 3.0,
            }
            hp_cat.update(self._filter_params(params, "cat"))
            model = CatBoostClassifier(
                loss_function="Logloss",
                random_seed=config.RANDOM_SEED,
                verbose=False,
                **hp_cat,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                fit_kw["eval_set"] = eval_set
                fit_kw["use_best_model"] = True
                fit_kw["early_stopping_rounds"] = _es_rounds
        else:
            hp["min_child_weight"] = config.PLACE_CLASSIFIER_PARAMS.get("min_child_weight", 3)
            model = XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                **hp,
                random_state=config.RANDOM_SEED,
                n_jobs=self._autotune_njobs,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                model.set_params(early_stopping_rounds=_es_rounds)
                fit_kw["eval_set"] = eval_set
                fit_kw["verbose"] = False
        model.fit(X, y, **fit_kw)
        logger.info(f"Place Classifier ({fw.upper()}) training complete")
        return model

    def _train_ranker(self, X, y, groups, params=None, sample_weight=None, eval_set=None):
        """Train a LightGBM LambdaRank model on per-race relevance labels."""
        _ensure_lightgbm()
        fw = self.frameworks.get("ranker", "lgbm")
        if fw != "lgbm":
            logger.warning(
                "Ranker framework '%s' is not supported in this build; using LGBM LambdaRank.",
                fw,
            )
        _es_rounds = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))
        hp = {
            "n_estimators": config.RANKER_PARAMS.get("n_estimators", 800),
            "max_depth": config.RANKER_PARAMS.get("max_depth", 6),
            "learning_rate": config.RANKER_PARAMS.get("learning_rate", 0.05),
            "subsample": config.RANKER_PARAMS.get("subsample", 0.8),
            "colsample_bytree": config.RANKER_PARAMS.get("colsample_bytree", 0.8),
            "reg_alpha": config.RANKER_PARAMS.get("reg_alpha", 0.1),
            "reg_lambda": config.RANKER_PARAMS.get("reg_lambda", 1.0),
            "linear_tree": bool(config.RANKER_PARAMS.get("linear_tree", False)),
            "min_child_samples": config.RANKER_PARAMS.get("min_child_samples", 20),
            "num_leaves": config.RANKER_PARAMS.get("num_leaves", 63),
        }
        hp.update(self._filter_params(params, "lgbm"))
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1, 2, 4, 8, 16, 32],
            **hp,
            subsample_freq=1,
            random_state=config.RANDOM_SEED,
            n_jobs=self._autotune_njobs,
            verbose=-1,
        )
        fit_kw: dict = {
            "group": np.asarray(groups, dtype=np.int32).tolist(),
            "sample_weight": sample_weight,
        }
        if eval_set is not None:
            from lightgbm import early_stopping as _lgb_es

            _eval_X, _eval_y, _eval_group = eval_set[0]
            fit_kw["eval_set"] = [(_eval_X, _eval_y)]
            fit_kw["eval_group"] = [np.asarray(_eval_group, dtype=np.int32).tolist()]
            fit_kw["eval_at"] = [1, 3]
            fit_kw["callbacks"] = [_lgb_es(_es_rounds, verbose=False)]

        model.fit(X, y, **fit_kw)
        logger.info("Race Ranker (LGBM) training complete")
        return model

    def _evaluate_as_ranker(self, scores, y_test, groups_test, test_df, name,
                            calibrated_probs=None, value_threshold: float | None = None):
        """Evaluate scores using ranking + calibration metrics.

        *scores* are used for ranking metrics (NDCG, top-k accuracy).
        *calibrated_probs*, when provided, are used for calibration
        metrics (RPS, Brier, log-loss) and value-bet simulation.  If
        omitted, raw softmax(scores, T=1) is used as a fallback.
        """
        from sklearn.metrics import ndcg_score as _ndcg

        ndcg1, ndcg3 = [], []
        top1, win3, total = 0, 0, 0

        # Use pre-calibrated probabilities when available.
        if calibrated_probs is not None:
            all_probs = calibrated_probs
        else:
            all_probs = _grouped_softmax(scores, groups_test, 1.0)

        # NDCG / top-k still needs per-race iteration (sklearn API)
        offset = 0
        for g in groups_test:
            gl = y_test[offset:offset + g]
            gs = scores[offset:offset + g]
            offset += g
            # Skip races with no label variation (no clear ground-truth winner)
            # OR where all scores are identical (model has no opinion).
            # Counting argmax ties as "correct" would inflate top-1 accuracy.
            if g < 2 or gl.max() == gl.min() or gs.max() == gs.min():
                continue
            total += 1
            try:
                ndcg1.append(_ndcg([gl], [gs], k=1))
                ndcg3.append(_ndcg([gl], [gs], k=3))
            except ValueError:
                pass
            if np.argmax(gs) == np.argmax(gl):
                top1 += 1
            if np.argmax(gl) in set(np.argsort(gs)[-3:]):
                win3 += 1

        # Binary win labels for calibration metrics
        if "finish_position" in test_df.columns:
            y_won = (test_df["finish_position"].values[:len(scores)] == 1).astype(int)
        elif "won" in test_df.columns:
            y_won = test_df["won"].values[:len(scores)].astype(int)
        else:
            y_won = (y_test == y_test.max()).astype(int)  # fallback

        # Clip probabilities for numerical stability in log_loss
        probs_clipped = np.clip(all_probs, 1e-15, 1 - 1e-15)
        try:
            brier = float(brier_score_loss(y_won, probs_clipped))
        except Exception:
            brier = float('nan')
        try:
            logloss = float(log_loss(y_won, probs_clipped))
        except Exception:
            logloss = float('nan')

        # RPS — full-ranking calibration metric
        if "finish_position" in test_df.columns:
            _fp_ev2 = test_df["finish_position"].values[:len(scores)].astype(np.float32)
            rps = rps_per_race(all_probs, _fp_ev2, groups_test)
        else:
            rps = float("nan")

        # ── ECE + reliability diagram (10 equal-width bins) ────
        _n_bins = 10
        _bin_edges = np.linspace(0.0, 1.0, _n_bins + 1)
        _bin_ids = np.digitize(probs_clipped, _bin_edges, right=False) - 1
        _bin_ids = np.clip(_bin_ids, 0, _n_bins - 1)
        _reliability: list[dict | None] = []
        _ece = 0.0
        _n_total = len(probs_clipped)
        for _b in range(_n_bins):
            _mask = _bin_ids == _b
            _cnt = int(_mask.sum())
            if _cnt == 0:
                _reliability.append(None)
                continue
            _mean_pred = float(probs_clipped[_mask].mean())
            _obs = float(y_won[_mask].mean())
            _reliability.append({
                "bin_lo": round(float(_bin_edges[_b]), 2),
                "bin_hi": round(float(_bin_edges[_b + 1]), 2),
                "mean_pred": round(_mean_pred, 4),
                "obs_rate": round(_obs, 4),
                "count": _cnt,
            })
            _ece += abs(_mean_pred - _obs) * (_cnt / _n_total)

        # ── Per-decile calibration (10 equal-count bins) ─────────
        _decile_bins: list[dict] = []
        if _n_total >= 20:
            _sort_idx = np.argsort(probs_clipped)
            _chunk = _n_total // 10
            for _d in range(10):
                _lo = _d * _chunk
                _hi = (_d + 1) * _chunk if _d < 9 else _n_total
                _d_probs = probs_clipped[_sort_idx[_lo:_hi]]
                _d_won = y_won[_sort_idx[_lo:_hi]]
                _decile_bins.append({
                    "decile": _d + 1,
                    "prob_lo": round(float(_d_probs.min()), 4),
                    "prob_hi": round(float(_d_probs.max()), 4),
                    "mean_pred": round(float(_d_probs.mean()), 4),
                    "obs_rate": round(float(_d_won.mean()), 4),
                    "count": int(len(_d_probs)),
                })

        metrics = {
            "rps": round(rps, 6),
            "brier_score": round(brier, 6),
            "log_loss": round(logloss, 4),
            "ece": round(_ece, 6),
            "ndcg_at_1": np.mean(ndcg1) if ndcg1 else 0,
            "ndcg_at_3": np.mean(ndcg3) if ndcg3 else 0,
            "top1_accuracy": top1 / total if total else 0,
            "win_in_top3": win3 / total if total else 0,
            "total_races": total,
            "reliability_bins": _reliability,
            "decile_calibration": _decile_bins,
        }

        # ── Betting-relevant metrics ─────────────────────────────
        # These help the user see which model config leads to better
        # betting outcomes, not just better ranking.
        _has_odds = "odds" in test_df.columns
        if _has_odds:
            _odds = test_df["odds"].values[:len(scores)].astype(np.float64)
            _vt = float(value_threshold if value_threshold is not None else getattr(config, "VALUE_THRESHOLD", 0.05))
            _vb = _value_bet_selection(all_probs, _odds, _vt)
            _is_vb = _vb["mask"]
            _n_vb = int(_is_vb.sum())
            if _n_vb > 0:
                _vb_won = y_won[_is_vb]
                _vb_odds = _odds[_is_vb]
                _vb_probs = np.clip(all_probs[_is_vb], 1e-15, 1 - 1e-15)
                _vb_sr = float(_vb_won.mean())
                # Flat-stake ROI: (sum of returns - stakes) / stakes
                _vb_returns = np.where(_vb_won == 1, _vb_odds - 1.0, -1.0)
                _vb_roi = float(_vb_returns.mean())  # per-bet avg P&L
                _vb_avg_edge = float(_vb["edge"][_is_vb].mean())
                _vb_avg_clv = float(_vb["clv"][_is_vb].mean())
                _vb_exp_roi = float(_vb["expected_roi"][_is_vb].mean())
                metrics["value_bets"] = _n_vb
                metrics["value_bet_sr"] = round(_vb_sr, 4)
                metrics["value_bet_roi"] = round(_vb_roi, 4)
                metrics["avg_edge"] = round(_vb_avg_edge, 4)
                metrics["value_bet_avg_clv"] = round(_vb_avg_clv, 4)
                metrics["value_bet_exp_roi_pct"] = round(_vb_exp_roi * 100.0, 2)
                try:
                    metrics["value_bet_brier"] = round(float(brier_score_loss(_vb_won, _vb_probs)), 6)
                except Exception:
                    metrics["value_bet_brier"] = None
                try:
                    metrics["value_bet_log_loss"] = round(float(log_loss(_vb_won, _vb_probs)), 4)
                except Exception:
                    metrics["value_bet_log_loss"] = None
            else:
                metrics["value_bets"] = 0
                metrics["value_bet_sr"] = None
                metrics["value_bet_roi"] = None
                metrics["avg_edge"] = None
                metrics["value_bet_avg_clv"] = None
                metrics["value_bet_exp_roi_pct"] = None
                metrics["value_bet_brier"] = None
                metrics["value_bet_log_loss"] = None
        else:
            metrics["value_bets"] = None
            metrics["value_bet_sr"] = None
            metrics["value_bet_roi"] = None
            metrics["avg_edge"] = None
            metrics["value_bet_avg_clv"] = None
            metrics["value_bet_exp_roi_pct"] = None
            metrics["value_bet_brier"] = None
            metrics["value_bet_log_loss"] = None

        logger.info(f"\n{'='*50}")
        logger.info(f"  {name} Evaluation Results")
        logger.info(f"{'='*50}")
        logger.info(f"  RPS:             {metrics['rps']:.6f}")
        logger.info(f"  Brier Score:     {metrics['brier_score']:.6f}")
        logger.info(f"  ECE:             {metrics['ece']:.6f}")
        logger.info(f"  Log Loss:        {metrics['log_loss']:.4f}")
        for k in ["ndcg_at_1", "ndcg_at_3", "top1_accuracy", "win_in_top3"]:
            logger.info(f"  {k}: {metrics[k]:.4f}")
        if metrics.get("value_bets"):
            logger.info(f"  Value Bets:      {metrics['value_bets']}")
            logger.info(f"  VB Strike Rate:  {metrics['value_bet_sr']:.4f}")
            logger.info(f"  VB ROI (flat):   {metrics['value_bet_roi']:+.4f}")
            logger.info(f"  Avg Edge:        {metrics['avg_edge']:+.4f}")
        logger.info(f"  total_races: {metrics['total_races']}")
        logger.info(f"{'='*50}")
        return metrics

    @staticmethod
    def _extract_recency_params(
        params: dict | None,
    ) -> tuple[float | None, float | None, str | None]:
        if not params:
            return None, None, None
        half_life = params.get("recency_half_life_days")
        seasonal = params.get("recency_seasonal_boost")
        shape = params.get("recency_decay_shape")
        return (
            float(half_life) if half_life is not None else None,
            float(seasonal) if seasonal is not None else None,
            str(shape) if shape is not None else None,
        )

    def _tune_value_threshold(
        self,
        win_probs: np.ndarray,
        groups_eval: np.ndarray,
        eval_df: pd.DataFrame,
        value_config: dict | None,
        place_probs: np.ndarray | None = None,
    ) -> dict:
        """Grid-search the value threshold on validation data only."""
        base_cfg = dict(value_config or {})
        base_vt = float(base_cfg.get("value_threshold", 0.05))
        candidates = sorted({0.02, 0.03, 0.05, 0.07, 0.10, 0.15, round(base_vt, 3)})

        best_cfg = dict(base_cfg)
        best_score = -np.inf
        best_stats = None
        for vt in candidates:
            trial_cfg = {**base_cfg, "value_threshold": float(vt)}
            analysis = analyse_test_set(
                win_probs=win_probs,
                groups_test=groups_eval,
                test_df=eval_df,
                value_threshold=float(vt),
                staking_mode=trial_cfg.get("staking_mode", "flat"),
                kelly_fraction=trial_cfg.get("kelly_fraction", 0.25),
                bankroll=trial_cfg.get("bankroll", 100.0),
                place_probs=place_probs,
                ew_min_place_edge=trial_cfg.get("ew_min_place_edge"),
            )
            stats = analysis.get("stats", {}).get("value", {})
            n_bets = int(stats.get("bets", 0))
            roi = float(stats.get("roi", 0.0))
            total_pnl = float(stats.get("pnl", 0.0))
            # Prefer profitable thresholds that still place a non-trivial number of bets.
            score = total_pnl + 0.05 * n_bets + 0.1 * roi
            if n_bets < 5:
                score -= 5.0
            if score > best_score:
                best_score = score
                best_cfg = trial_cfg
                best_stats = stats

        if best_cfg.get("value_threshold") != base_vt:
            logger.info(
                "Validation tuned value threshold: %.3f -> %.3f%s",
                base_vt,
                best_cfg["value_threshold"],
                f" (bets={best_stats.get('bets', 0)}, pnl={best_stats.get('pnl', 0):+.2f})" if best_stats else "",
            )
        return best_cfg

    @staticmethod
    def _optimise_calibration_temperature(
        scores: np.ndarray,
        groups: np.ndarray,
        finish_positions: np.ndarray,
    ) -> float:
        """Find a scalar temperature minimising log-loss on OOF data.

        Uses ``scipy.optimize.minimize_scalar`` for a precise continuous
        optimisation.  Log-loss measures P(win) accuracy without
        anchoring to the full finishing distribution, so the model
        retains genuine divergence from market odds.
        """
        from scipy.optimize import minimize_scalar

        won = (finish_positions == 1).astype(np.float64)
        _eps = 1e-12

        def _logloss(temp: float) -> float:
            probs = _grouped_softmax(scores, groups, temp)
            probs = np.clip(probs, _eps, 1 - _eps)
            return -float(np.mean(won * np.log(probs) + (1 - won) * np.log(1 - probs)))

        result = minimize_scalar(_logloss, bounds=(0.2, 6.0), method="bounded")
        return float(result.x)

    @staticmethod
    def _softmax_with_temp(
        scores: np.ndarray,
        groups: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Per-race softmax at a given temperature, returns flat probability array."""
        return _grouped_softmax(scores, groups, temperature)

    @staticmethod
    def _optimise_platt_calibration(
        probs: np.ndarray,
        groups: np.ndarray,
        finish_positions: np.ndarray,
    ) -> tuple[float, float]:
        """Fit Platt (logistic) calibration: p_cal = σ(a·logit(p) + b).

        Applied element-wise then renormalised per race.  Minimises
        log-loss on the binary win outcome so calibration corrects
        P(win) accuracy without anchoring to the market.

        Returns:
            (a, b) — scale and bias of the logistic transform.
            a=1, b=0 is the identity (no correction).
        """
        from scipy.optimize import minimize

        won = (finish_positions == 1).astype(np.float64)
        eps = 1e-9
        _eps_ll = 1e-12
        logit_p = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1))

        _groups = np.asarray(groups, dtype=np.intp)

        def _logloss_platt(params: np.ndarray) -> float:
            a, b = float(params[0]), float(params[1])
            raw = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
            cal = _grouped_normalize(raw, _groups)
            cal = np.clip(cal, _eps_ll, 1 - _eps_ll)
            return -float(np.mean(won * np.log(cal) + (1 - won) * np.log(1 - cal)))

        result = minimize(
            _logloss_platt,
            x0=np.array([1.0, 0.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 1000},
        )
        a, b = float(result.x[0]), float(result.x[1])
        # Clamp to prevent extreme distortions
        a = float(np.clip(a, 0.1, 5.0))
        b = float(np.clip(b, -3.0, 3.0))
        return a, b

    # ── Place calibration ────────────────────────────────────────
    def _optimise_place_platt(self, probs: np.ndarray, placed: np.ndarray):
        """Fit Platt (a, b) on binary place outcomes minimising Brier score."""
        from scipy.optimize import minimize

        _eps = 1e-9

        def _brier(ab):
            a, b = ab
            lp = np.log(np.clip(probs, _eps, 1 - _eps)
                        / np.clip(1 - probs, _eps, 1))
            cal = 1.0 / (1.0 + np.exp(-(a * lp + b)))
            return float(np.mean((cal - placed) ** 2))

        res = minimize(
            _brier, x0=np.array([1.0, 0.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 600},
        )
        a = float(np.clip(res.x[0], 0.1, 5.0))
        b = float(np.clip(res.x[1], -3.0, 3.0))
        return a, b

    # ── Isotonic calibration ──────────────────────────────────────
    @staticmethod
    def _fit_isotonic(
        probs: np.ndarray, targets: np.ndarray,
    ) -> IsotonicRegression | None:
        """Fit isotonic regression calibrator on probabilities.

        Returns None if there are too few samples for a meaningful fit.
        """
        probs_arr = np.asarray(probs, dtype=np.float64)
        targets_arr = np.asarray(targets, dtype=np.float64)
        if len(probs_arr) < 50:
            return None
        if np.unique(targets_arr).size < 2:
            return None
        if np.unique(np.round(probs_arr, 6)).size < 10:
            return None
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs_arr, targets_arr)
        return iso

    def _fit_training_slab_calibrators(
        self,
        X_train: np.ndarray,
        groups_train: np.ndarray,
        y_won: np.ndarray,
        y_placed: np.ndarray,
        finish_positions: np.ndarray,
        model_params: dict[str, dict] | None,
        sample_weight: np.ndarray | None,
        race_dates=None,
    ) -> bool:
        """Fit Platt + isotonic calibrators on a training-only slab.

        A disposable pair of models is trained on the earliest 80 % of races,
        then calibrated on the latest 20 % of races from the training split.
        This keeps evaluation data untouched while providing time-local
        calibration that is closer to live deployment than old OOF folds.
        """
        n_races = len(groups_train)
        if n_races < 25:
            return False

        cal_bnd = max(1, int(n_races * 0.8))
        if cal_bnd >= n_races:
            return False

        cum_g = np.cumsum(groups_train)
        cal_row_start = int(cum_g[cal_bnd - 1]) if cal_bnd > 0 else 0
        if cal_row_start <= 0 or cal_row_start >= len(X_train):
            return False

        cal_groups = groups_train[cal_bnd:]
        if len(cal_groups) < 5:
            return False

        X_cal_tr = X_train[:cal_row_start]
        X_cal_vl = X_train[cal_row_start:]
        if len(X_cal_tr) < 100 or len(X_cal_vl) < 50:
            return False

        sw_cal = sample_weight[:cal_row_start] if sample_weight is not None else None
        y_won_cal_tr = y_won[:cal_row_start]
        y_placed_cal_tr = y_placed[:cal_row_start]
        won_cal_vl = y_won[cal_row_start:].astype(np.float64)
        placed_cal_vl = y_placed[cal_row_start:].astype(np.float64)
        fp_cal_vl = finish_positions[cal_row_start:]

        if np.unique(won_cal_vl).size < 2 or np.unique(placed_cal_vl).size < 2:
            return False

        clf_params = (model_params or {}).get("classifier") if isinstance(model_params, dict) else None
        place_params = (model_params or {}).get("place") if isinstance(model_params, dict) else None

        # Purged early-stopping split inside the slab-train portion so the
        # disposable models don't train the full n_estimators.
        _slab_es = (
            self._early_stop_split(
                groups_train[:cal_bnd], race_dates[:cal_bnd],
                purge_days=int(getattr(config, "PURGE_DAYS", 7)),
            )
            if race_dates is not None and int(getattr(config, "EARLY_STOPPING_ROUNDS", 0)) > 0
            else None
        )
        if _slab_es is not None:
            _s_tr_end, _s_vl_beg = _slab_es
            _s_X, _s_sw = X_cal_tr[:_s_tr_end], (sw_cal[:_s_tr_end] if sw_cal is not None else None)
            _s_eval_w = [(X_cal_tr[_s_vl_beg:], y_won_cal_tr[_s_vl_beg:])]
            _s_eval_p = [(X_cal_tr[_s_vl_beg:], y_placed_cal_tr[_s_vl_beg:])]
            _s_y_won, _s_y_placed = y_won_cal_tr[:_s_tr_end], y_placed_cal_tr[:_s_tr_end]
        else:
            _s_X, _s_sw = X_cal_tr, sw_cal
            _s_eval_w = _s_eval_p = None
            _s_y_won, _s_y_placed = y_won_cal_tr, y_placed_cal_tr

        n_pos_w = max(int(_s_y_won.sum()), 1)
        cal_clf = self._train_win_classifier(
            _s_X,
            _s_y_won,
            scale_pos_weight=(len(_s_y_won) - n_pos_w) / n_pos_w,
            params=clf_params,
            sample_weight=_s_sw,
            eval_set=_s_eval_w,
        )
        n_pos_p = max(int(_s_y_placed.sum()), 1)
        cal_place = self._train_place_classifier(
            _s_X,
            _s_y_placed,
            scale_pos_weight=(len(_s_y_placed) - n_pos_p) / n_pos_p,
            params=place_params,
            sample_weight=_s_sw,
            eval_set=_s_eval_p,
        )

        cal_clf_probs = cal_clf.predict_proba(X_cal_vl)[:, 1]
        cal_place_raw = cal_place.predict_proba(X_cal_vl)[:, 1]

        self.calibration_temp = 1.0
        self.platt_a, self.platt_b = self._optimise_platt_calibration(
            cal_clf_probs,
            cal_groups,
            fp_cal_vl,
        )
        self.place_platt_a, self.place_platt_b = self._optimise_place_platt(
            cal_place_raw,
            placed_cal_vl,
        )

        cal_win_probs = self._calibrate_win_probs(cal_clf_probs, cal_groups)
        self.win_iso_cal = self._fit_isotonic(cal_win_probs, won_cal_vl)
        cal_place_probs = self._calibrate_place_probs(cal_place_raw)
        self.place_iso_cal = self._fit_isotonic(cal_place_probs, placed_cal_vl)
        return True

    # ── Calibration application ──────────────────────────────────
    def _calibrate_win_probs(
        self, raw_probs: np.ndarray, groups: np.ndarray,
        test_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply Platt scaling to win classifier probabilities → calibrated P(win).

        The win classifier outputs probabilities directly via predict_proba,
        so no softmax/temperature step is needed — just Platt + per-race normalise.
        """
        probs = raw_probs.copy()

        # Global Platt
        if self.platt_a != 1.0 or self.platt_b != 0.0:
            _eps = 1e-9
            lp = np.log(np.clip(probs, _eps, 1 - _eps)
                        / np.clip(1 - probs, _eps, 1))
            raw = 1.0 / (1.0 + np.exp(-(self.platt_a * lp + self.platt_b)))
            probs = _grouped_normalize(raw, groups)
        else:
            probs = _grouped_normalize(probs, groups)

        # Isotonic refinement (preserves ranking, adjusts calibration curve)
        if self.win_iso_cal is not None:
            probs = self.win_iso_cal.predict(probs.astype(np.float64))
            probs = np.clip(probs, 1e-9, 1.0)
            probs = _grouped_normalize(probs, groups)

        return probs

    def _calibrate_place_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply Platt + isotonic calibration to place probabilities."""
        if self.place_platt_a == 1.0 and self.place_platt_b == 0.0:
            probs = raw_probs.copy()
        else:
            _eps = 1e-9
            lp = np.log(np.clip(raw_probs, _eps, 1 - _eps)
                         / np.clip(1 - raw_probs, _eps, 1))
            probs = 1.0 / (1.0 + np.exp(-(self.place_platt_a * lp + self.place_platt_b)))

        # Isotonic refinement
        if self.place_iso_cal is not None:
            probs = self.place_iso_cal.predict(probs.astype(np.float64))
            probs = np.clip(probs, 1e-9, 1.0)

        return probs

    def _prepare_prediction_frame(self, race_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        predict_df = race_df.reset_index(drop=True).copy()
        missing = [c for c in self.feature_cols if c not in predict_df.columns]
        if missing:
            # Zero-filling silently degrades predictions; make schema
            # drift between training FE and live FE loud.
            logger.warning(
                "Prediction frame is missing %d of %d model features "
                "(zero-filled): %s%s",
                len(missing), len(self.feature_cols), missing[:10],
                " …" if len(missing) > 10 else "",
            )
        for col in missing:
            predict_df[col] = 0

        X = predict_df[self.feature_cols].values
        X_scaled_np = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_np, columns=self.feature_cols, index=predict_df.index)
        return predict_df, X_scaled

    def _compute_place_probabilities(
        self,
        predict_df: pd.DataFrame,
        X_scaled: pd.DataFrame,
        win_probs: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        total_rows = len(predict_df)
        if total_rows == 0:
            return np.empty(0, dtype=np.float64)

        place_probs = np.zeros(total_rows, dtype=np.float64)
        if self.place_model is not None:
            raw_place = self.place_model.predict_proba(X_scaled)[:, 1]
            base_place = self._calibrate_place_probs(raw_place)
        else:
            base_place = None

        offsets = np.concatenate([[0], np.cumsum(groups[:-1])]) if len(groups) else np.empty(0, dtype=np.int64)
        for off, group_size in zip(offsets, groups):
            sl = slice(int(off), int(off + group_size))
            n = int(group_size)
            win_slice = win_probs[sl]
            if base_place is not None:
                race_df = predict_df.iloc[sl]
                is_hcap = bool(race_df["handicap"].iloc[0]) if "handicap" in race_df.columns else False
                from src.each_way import adjust_place_probs_for_race, get_ew_terms as _get_ew

                ew_terms = _get_ew(n, is_handicap=is_hcap)
                places_paid = ew_terms.places_paid if ew_terms.eligible else 3
                place_probs[sl] = adjust_place_probs_for_race(base_place[sl], win_slice, places_paid)
            else:
                places_paid = 3 if n >= 8 else (2 if n >= 5 else 0)
                if places_paid > 0 and n > places_paid:
                    k = n / places_paid
                    fallback = 1.0 - (1.0 - win_slice) ** k
                    fallback = np.maximum(fallback, win_slice)
                    total = fallback.sum()
                    if total > 0:
                        fallback = np.clip(fallback * (places_paid / total), 0.0, 1.0)
                    place_probs[sl] = fallback
                else:
                    place_probs[sl] = np.zeros(n, dtype=np.float64)
        return place_probs

    def predict_races(self, featured_df: pd.DataFrame, ew_fraction: float | None = None) -> pd.DataFrame:
        if self.clf_model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        if "race_id" not in featured_df.columns:
            raise ValueError("predict_races requires a race_id column.")

        predict_df, X_scaled = self._prepare_prediction_frame(featured_df)
        race_ids = pd.Index(pd.unique(predict_df["race_id"]))
        groups = predict_df.groupby("race_id", sort=False).size().values.astype(np.int64)
        offsets = _group_offsets(groups)
        n = len(predict_df)

        clf_probs_raw = self.clf_model.predict_proba(X_scaled)[:, 1] if self.clf_model is not None else np.repeat(1.0 / np.maximum(groups, 1), groups)
        win_probs = self._calibrate_win_probs(clf_probs_raw, groups, predict_df)
        win_probs = self._apply_market_anchor(win_probs, groups, predict_df)
        place_probs = self._compute_place_probabilities(predict_df, X_scaled, win_probs, groups)

        result_cols = [col for col in ["race_id", "track", "off_time", "race_name", "horse_name", "jockey", "trainer", "odds", "num_runners", "handicap"] if col in predict_df.columns]
        results = predict_df[result_cols].copy()
        results["win_probability"] = win_probs
        results["place_probability"] = place_probs
        results["predicted_rank"] = results.groupby("race_id", sort=False)["win_probability"].rank(ascending=False, method="min").astype(int)

        if "odds" in predict_df.columns:
            raw_ip = 1.0 / predict_df["odds"].values
            overround = np.add.reduceat(raw_ip, offsets)
            results["implied_prob"] = raw_ip / np.maximum(np.repeat(overround, groups), 1e-9)
            results["value_score"] = results["win_probability"] - results["implied_prob"]

        if "odds" in predict_df.columns:
            try:
                from src.each_way import compute_ew_columns

                results = compute_ew_columns(results, fraction_override=ew_fraction)
            except Exception as e:
                logger.debug("EW columns skipped: %s", e)

        results["_race_order"] = pd.Categorical(results["race_id"], categories=race_ids, ordered=True)
        results = results.sort_values(["_race_order", "predicted_rank"], kind="stable")
        return results.drop(columns=["_race_order"], errors="ignore")

    # ── Place model evaluation ───────────────────────────────────
    @staticmethod
    def _evaluate_place_model(place_probs, raw_probs, test_df, groups):
        """Evaluate place classifier with place-specific metrics."""
        # Use actual EW places paid per race (matches training target)
        nr = test_df["num_runners"].values
        hcap = test_df.get("handicap", pd.Series(0, index=test_df.index)).values.astype(bool)
        pp = np.where(nr <= 4, 3, np.where(nr <= 7, 2, np.where(nr <= 15, 3, np.where(hcap, 4, 3))))
        placed = (test_df["finish_position"].values <= pp).astype(int)

        # Brier score (calibration quality)
        brier_cal = float(np.mean((place_probs - placed) ** 2))
        brier_raw = float(np.mean((raw_probs - placed) ** 2))

        # Place precision: of model's top-3 picks per race, how many actually placed
        off = 0
        place_hits = 0
        total_placed = 0
        for g in groups:
            fp_race = test_df["finish_position"].values[off:off + g]
            pp_vec = pp[off:off + g]
            pp_race = place_probs[off:off + g]
            _k = int(pp_vec[0]) if len(pp_vec) > 0 else 3
            top3_idx = np.argsort(-pp_race)[:min(_k, g)]
            actual_placed = set(np.where(fp_race <= _k)[0])
            place_hits += len(set(top3_idx) & actual_placed)
            total_placed += len(actual_placed)
            off += g

        place_precision = place_hits / max(total_placed, 1)

        logger.info(
            f"  Place Model — Brier(cal): {brier_cal:.4f}, "
            f"Brier(raw): {brier_raw:.4f}, "
            f"Place precision: {place_precision:.3f}"
        )

        # ── ECE + reliability for place model ────────────────
        _pp_clipped = np.clip(place_probs, 1e-15, 1 - 1e-15)
        _p_n_bins = 10
        _p_edges = np.linspace(0.0, 1.0, _p_n_bins + 1)
        _p_bids = np.clip(np.digitize(_pp_clipped, _p_edges, right=False) - 1, 0, _p_n_bins - 1)
        _p_rel: list[dict | None] = []
        _p_ece = 0.0
        _p_total = len(_pp_clipped)
        for _b in range(_p_n_bins):
            _m = _p_bids == _b
            _c = int(_m.sum())
            if _c == 0:
                _p_rel.append(None)
                continue
            _mp = float(_pp_clipped[_m].mean())
            _ob = float(placed[_m].mean())
            _p_rel.append({"bin_lo": round(float(_p_edges[_b]), 2), "bin_hi": round(float(_p_edges[_b + 1]), 2), "mean_pred": round(_mp, 4), "obs_rate": round(_ob, 4), "count": _c})
            _p_ece += abs(_mp - _ob) * (_c / _p_total)

        return {
            "brier_calibrated": round(brier_cal, 4),
            "brier_raw": round(brier_raw, 4),
            "place_precision": round(place_precision, 3),
            "ece": round(_p_ece, 6),
            "reliability_bins": _p_rel,
        }

    # ── Prediction ───────────────────────────────────────────────
    def predict_race(self, race_df: pd.DataFrame, ew_fraction: float | None = None) -> pd.DataFrame:
        if self.clf_model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        race_df, X_scaled = self._prepare_prediction_frame(race_df)
        n = len(X_scaled)
        groups = np.array([n])

        # ── Score with each model ────────────────────────────────
        clf_probs_raw = self.clf_model.predict_proba(X_scaled)[:, 1] if self.clf_model is not None else np.full(n, 1.0 / n)

        # Calibrated win probabilities (+ market anchor when fitted)
        win_probs = self._calibrate_win_probs(clf_probs_raw, groups, race_df)
        win_probs = self._apply_market_anchor(win_probs, groups, race_df)

        # Place probabilities
        _place_probs = self._compute_place_probabilities(race_df, X_scaled, win_probs, groups)

        # ── Build results ────────────────────────────────────────
        results = race_df[["horse_name"]].copy()
        if "jockey" in race_df.columns:
            results["jockey"] = race_df["jockey"]
        if "trainer" in race_df.columns:
            results["trainer"] = race_df["trainer"]
        if "odds" in race_df.columns:
            results["odds"] = race_df["odds"]

        results["win_probability"] = win_probs
        results["place_probability"] = _place_probs
        results["predicted_rank"] = pd.Series(win_probs).rank(
            ascending=False, method="min",
        ).astype(int).values

        if "odds" in race_df.columns:
            _raw_ip = 1.0 / race_df["odds"].values
            _overround = _raw_ip.sum()
            results["implied_prob"] = _raw_ip / max(_overround, 1e-9)
            results["value_score"] = results["win_probability"] - results["implied_prob"]

        if "odds" in race_df.columns:
            try:
                from src.each_way import compute_ew_columns
                if "num_runners" in race_df.columns:
                    results["num_runners"] = race_df["num_runners"].values
                if "handicap" in race_df.columns:
                    results["handicap"] = race_df["handicap"].values
                results = compute_ew_columns(results, fraction_override=ew_fraction)
            except Exception as e:
                logger.debug("EW columns skipped: %s", e)

        return results.sort_values("predicted_rank")

    # ── SHAP explanation ────────────────────────────────────────
    def explain_race(
        self,
        race_df: pd.DataFrame,
        top_n_features: int = 10,
    ) -> dict:
        explain_model = None
        explain_label = None
        if self.clf_model is not None:
            explain_model = self.clf_model
            explain_label = "Win Classifier"
        elif self.place_model is not None:
            explain_model = self.place_model
            explain_label = "Place Classifier"
        elif self.ranker_model is not None:
            explain_model = self.ranker_model
            explain_label = "Race Ranker"

        if explain_model is None:
            raise ValueError(
                "Model not trained. Call train() or load() first."
            )

        explanations, method_label = _explain_race_with_shap(
            explain_model,
            race_df,
            self.feature_cols,
            self.scaler,
            top_n_features,
        )
        self.last_explain_model_label = f"{explain_label} ({method_label})"
        return explanations

    # ── Persistence ──────────────────────────────────────────────
    def save(self):
        path = os.path.join(config.MODELS_DIR, "triple_ensemble_models.joblib")
        joblib.dump(
            {
                "clf_model": self.clf_model,
                "place_model": self.place_model,
                "ranker_model": self.ranker_model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "calibration_temp": self.calibration_temp,
                "platt_a": self.platt_a,
                "platt_b": self.platt_b,
                "place_platt_a": self.place_platt_a,
                "place_platt_b": self.place_platt_b,
                "win_iso_cal": self.win_iso_cal,
                "place_iso_cal": self.place_iso_cal,
                "market_anchor": self.market_anchor,
                "frameworks": self.frameworks,
            },
            path,
        )
        logger.info(f"Models saved to {path}")

    def load(self):
        path = os.path.join(config.MODELS_DIR, "triple_ensemble_models.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No models found at {path}")
        data = joblib.load(path)
        self.clf_model = data.get("clf_model")
        self.place_model = data.get("place_model")
        self.ranker_model = data.get("ranker_model")
        self.scaler = data.get("scaler") or _IdentityScaler()
        self.feature_cols = data["feature_cols"]
        self.calibration_temp = float(data.get("calibration_temp", 1.0))
        self.platt_a = float(data.get("platt_a", 1.0))
        self.platt_b = float(data.get("platt_b", 0.0))
        self.place_platt_a = float(data.get("place_platt_a", 1.0))
        self.place_platt_b = float(data.get("place_platt_b", 0.0))
        self.win_iso_cal = data.get("win_iso_cal")
        self.place_iso_cal = data.get("place_iso_cal")
        self.market_anchor = data.get("market_anchor")
        self.frameworks = data.get("frameworks", {"classifier": "cat", "place": "cat", "ranker": "lgbm"})
        logger.info(f"Models loaded (frameworks: {self.frameworks})")

    # ── Walk-forward fold helper ─────────────────────────────────
    def train_on_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        train_df: pd.DataFrame,
        groups_train: np.ndarray,
        groups_test: np.ndarray,
        feature_cols: list[str],
        params: dict[str, dict] | None = None,
        return_place_probs: bool = False,
        fast_fold: bool = False,
        test_df: pd.DataFrame | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train 3 models on a pre-split fold with calibration.

        Uses the last 20 % of the training data as a held-out slab to
        fit temperature + Platt calibration (same pipeline as full
        ``train()``).  Returns calibrated win/place probabilities.

        When *fast_fold* is True, tree counts are halved for speed
        (suitable for walk-forward validation where many folds are
        trained and the models are disposable).
        """
        self.feature_cols = feature_cols
        fold_params = params or {}

        # Fast-fold overrides: halve n_estimators + cap depth for speed
        _saved_params: dict | None = None
        if fast_fold:
            _saved_params = {
                "clf_n_est": config.CLASSIFIER_PARAMS.get("n_estimators", 500),
                "clf_depth": config.CLASSIFIER_PARAMS.get("max_depth", 7),
                "xgb_n_est": config.XGBOOST_PARAMS.get("n_estimators", 500),
                "xgb_depth": config.XGBOOST_PARAMS.get("max_depth", 6),
                "lgb_n_est": config.LIGHTGBM_PARAMS.get("n_estimators", 500),
                "lgb_depth": config.LIGHTGBM_PARAMS.get("max_depth", 6),
            }
            config.CLASSIFIER_PARAMS["n_estimators"] = max(100, _saved_params["clf_n_est"] // 2)
            config.CLASSIFIER_PARAMS["max_depth"] = min(_saved_params["clf_depth"], 5)
            config.XGBOOST_PARAMS["n_estimators"] = max(100, _saved_params["xgb_n_est"] // 2)
            config.XGBOOST_PARAMS["max_depth"] = min(_saved_params["xgb_depth"], 5)
            config.LIGHTGBM_PARAMS["n_estimators"] = max(100, _saved_params["lgb_n_est"] // 2)
            config.LIGHTGBM_PARAMS["max_depth"] = min(_saved_params["lgb_depth"], 5)

        fp = train_df["finish_position"].values.astype(np.float32)
        y_won = train_df["won"].fillna(0).values.astype(int)

        _prune_frac = float(getattr(config, "FEATURE_PRUNE_FRACTION", 0.0))
        if _prune_frac > 0 and len(self.feature_cols) > 8:
            n_races_fold = len(groups_train)
            anchor_races = max(1, int(n_races_fold * 0.6))
            anchor_rows = int(np.cumsum(groups_train)[anchor_races - 1])
            anchor_df = train_df.iloc[:anchor_rows].copy()
            if len(anchor_df) >= 200:
                kept_cols = _prune_features_quick(anchor_df, self.feature_cols, _prune_frac)
                if len(kept_cols) < len(self.feature_cols):
                    keep_idx = [self.feature_cols.index(col) for col in kept_cols]
                    self.feature_cols = kept_cols
                    feature_cols = kept_cols
                    X_train = X_train[:, keep_idx]
                    X_test = X_test[:, keep_idx]
                    logger.info(
                        "Applied leak-safe fold pruning using %d anchor rows",
                        len(anchor_df),
                    )
            else:
                logger.info(
                    "Fold feature pruning skipped: anchor slice too small (%d rows)",
                    len(anchor_df),
                )

        # Dynamic place target based on actual EW places paid per race
        _nr_fold = train_df["num_runners"].values
        _hc_fold = train_df.get("handicap", pd.Series(0, index=train_df.index)).values.astype(bool)
        _pp_fold = np.where(_nr_fold <= 4, 3, np.where(_nr_fold <= 7, 2, np.where(_nr_fold <= 15, 3, np.where(_hc_fold, 4, 3))))
        y_placed = (fp <= _pp_fold).astype(int)

        # Recency sample weights
        dates = pd.to_datetime(train_df["race_date"])
        sw = compute_recency_sample_weights(dates).copy()

        if not self._fit_training_slab_calibrators(
            X_train=X_train,
            groups_train=groups_train,
            y_won=y_won,
            y_placed=y_placed,
            finish_positions=fp,
            model_params=fold_params,
            sample_weight=sw,
        ):
            # Too few races — skip calibration
            self.calibration_temp = 1.0
            self.platt_a, self.platt_b = 1.0, 0.0
            self.place_platt_a, self.place_platt_b = 1.0, 0.0
            self.win_iso_cal = None
            self.place_iso_cal = None

        # ── Train final models on ALL training data ──────────────
        n_pw = max(int(y_won.sum()), 1)
        self.clf_model = self._train_win_classifier(
            X_train, y_won,
            scale_pos_weight=(len(y_won) - n_pw) / n_pw,
            params=fold_params.get("classifier"),
            sample_weight=sw,
        )
        n_pp = max(int(y_placed.sum()), 1)
        self.place_model = self._train_place_classifier(
            X_train, y_placed,
            scale_pos_weight=(len(y_placed) - n_pp) / n_pp,
            params=fold_params.get("place"),
            sample_weight=sw,
        )

        # ── Score test set with calibration ──────────────────────
        clf_probs_raw = self.clf_model.predict_proba(X_test)[:, 1]
        place_probs_raw = self.place_model.predict_proba(X_test)[:, 1]

        win_probs = self._calibrate_win_probs(clf_probs_raw, groups_test)
        place_probs = self._calibrate_place_probs(place_probs_raw)

        # Restore config if fast_fold
        if fast_fold and _saved_params is not None:
            config.CLASSIFIER_PARAMS["n_estimators"] = _saved_params["clf_n_est"]
            config.CLASSIFIER_PARAMS["max_depth"] = _saved_params["clf_depth"]
            config.XGBOOST_PARAMS["n_estimators"] = _saved_params["xgb_n_est"]
            config.XGBOOST_PARAMS["max_depth"] = _saved_params["xgb_depth"]
            config.LIGHTGBM_PARAMS["n_estimators"] = _saved_params["lgb_n_est"]
            config.LIGHTGBM_PARAMS["max_depth"] = _saved_params["lgb_depth"]

        if return_place_probs:
            return win_probs, place_probs
        return win_probs


# Backwards-compatible alias — older pickled models (run snapshots,
# models/triple_ensemble_models.joblib) reference this name.
TripleEnsemblePredictor = RacePredictor
