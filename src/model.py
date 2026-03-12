"""
Model Module
=============
Learning-to-Rank models for horse race prediction.

Supports:
- XGBRanker (listwise ranking objective)
- LGBMRanker (LambdaRank objective)
- Rank Ensemble (weighted blend of both rankers)

Ranks horses **within** each race and converts raw scores to
pseudo-probabilities via softmax for downstream consumption.
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
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    ndcg_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker, XGBRegressor, XGBClassifier
from lightgbm import LGBMRanker, LGBMRegressor, LGBMClassifier
CatBoostRanker = None
CatBoostRegressor = None
CatBoostClassifier = None
Pool = None

import config

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


def _require_catboost() -> None:
    global CatBoostRanker, CatBoostRegressor, CatBoostClassifier, Pool
    if CatBoostRanker is not None and CatBoostRegressor is not None and CatBoostClassifier is not None and Pool is not None:
        return
    try:
        cb = importlib.import_module("catboost")
        CatBoostRanker = cb.CatBoostRanker
        CatBoostRegressor = cb.CatBoostRegressor
        CatBoostClassifier = cb.CatBoostClassifier
        Pool = cb.Pool
    except Exception as e:
        raise ImportError(
            "CatBoost is required for framework='cat'. Install with: pip install catboost"
        ) from e


def _groups_to_group_id(groups: np.ndarray) -> np.ndarray:
    """Expand group sizes into per-row group-id array."""
    return np.repeat(np.arange(len(groups), dtype=np.int32), groups.astype(int))


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


class _FocalLGBMClassifier(LGBMClassifier):
    """LGBMClassifier with sigmoid-corrected ``predict_proba``.

    When a custom objective (e.g. focal loss) is used, the LightGBM
    booster returns raw logits.  This subclass applies the sigmoid
    link so ``predict_proba`` returns valid probabilities.
    """

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
]

# Model types that use the ranking objective
RANKER_MODELS = {"xgb_ranker", "lgbm_ranker", "cat_ranker", "rank_ensemble"}
ALL_MODELS = RANKER_MODELS | {"triple_ensemble"}

# XGB-compatible hyperparameter names (used to filter params for
# regressor / classifier sub-models that may receive ranker-tuned dicts)
_XGB_VALID_HP = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda",
    "min_child_weight", "gamma",
}

# LGBM-compatible hyperparameter names
_LGBM_VALID_HP = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "reg_alpha", "reg_lambda",
    "min_child_samples", "num_leaves",
}

# CatBoost-compatible hyperparameter names
_CAT_VALID_HP = {
    "n_estimators", "depth", "learning_rate",
    "l2_leaf_reg", "random_strength", "bagging_temperature",
}

# Default framework selection for each sub-model in the ensemble.
DEFAULT_FRAMEWORKS: dict[str, str] = {
    "ltr": "lgbm",
    "regressor": "lgbm",
    "classifier": "cat",
    "place": "cat",
    "norm_pos": "lgbm",
    "residual": "cat",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns (exclude targets and identifiers)."""
    return [
        col
        for col in df.columns
        if col not in EXCLUDE_COLUMNS and df[col].dtype in ["int64", "float64", "int32", "float32"]
    ]


def _event_sort_key(df: pd.DataFrame) -> pd.Series:
    """Build a robust event-time key (race_date + off_time when available)."""
    race_dt = pd.to_datetime(df["race_date"], errors="coerce")
    if "off_time" not in df.columns:
        return race_dt

    off_secs = _off_time_to_seconds(df["off_time"]).astype(float)
    return race_dt + pd.to_timedelta(off_secs, unit="s")


def _prune_features_quick(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    prune_fraction: float,
) -> list[str]:
    """Drop the lowest-importance features using a quick pilot model.

    Trains a lightweight LGBMClassifier (100 trees) on the training
    data to rank features by split-based importance, then removes the
    bottom ``prune_fraction`` of features.
    """
    if len(train_df) < 2:
        logger.warning("Feature pruning skipped: not enough training samples (%d)", len(train_df))
        return feature_cols

    X = train_df[feature_cols].values
    y = train_df["won"].fillna(0).values.astype(int)

    pilot = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    pilot.fit(X, y)

    importance = pilot.feature_importances_
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


def prepare_ranking_data(
    df: pd.DataFrame,
) -> tuple:
    """
    Prepare data for Learning-to-Rank models.

    Instead of a binary win/loss target, LTR uses **relevance labels**
    derived from finishing position.  Higher label = better horse.
    A ``group`` array tells the ranker which rows belong to the same race.

    Returns:
        X_train, X_test, y_train, y_test, groups_train, groups_test,
        feature_columns, scaler, test_df
    """
    logger.info("Preparing ranking data...")

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
    # Remove entire races that contain systematic data errors:
    #   • Single-runner races  — no competition, relevance labels are meaningless
    #   • All-identical finish positions within a race — scraping artefact
    #   • Any runner with odds < 1.01       — impossible market, data error
    # These races are removed from training only; they never affect prediction.
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
        logger.info(
            f"Removed {_n_before - len(df)} rows from {len(_degenerate_ids)} "
            f"degenerate races (single-runner / identical positions / bad odds)"
        )

    # Relevance label: higher is better, ordinal.
    # Winner-heavy scheme so LambdaRank's NDCG gradient focuses on
    # getting the winner right: 1st=5, 2nd=2, 3rd=1, 4th+=0.
    # With NDCG gain = 2^label - 1: winner=31, 2nd=3, 3rd=1
    # → winner is 10× more important than 2nd place.
    fp = df["finish_position"].values.astype(int)
    df["relevance"] = np.where(fp == 1, 5, np.where(fp == 2, 2, np.where(fp == 3, 1, 0)))

    # Time-based split
    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    # Ensure we split at a race boundary
    split_race = df.iloc[split_idx]["race_id"]
    while split_idx < len(df) and df.iloc[split_idx]["race_id"] == split_race:
        split_idx += 1

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    train_df = train_df.drop(columns=["_event_dt"], errors="ignore")
    test_df = test_df.drop(columns=["_event_dt"], errors="ignore")

    # ── Purge gap ────────────────────────────────────────────────
    _purge_days = getattr(config, "PURGE_DAYS", 7)
    if _purge_days > 0:
        _test_start = test_df["race_date"].min()
        _purge_cutoff = _test_start - pd.Timedelta(days=_purge_days)
        train_df = train_df[train_df["race_date"] <= _purge_cutoff].copy()

    # ── Burn-in exclusion ────────────────────────────────────────
    _burn_months = getattr(config, "BURN_IN_MONTHS", 4)
    if _burn_months > 0 and len(train_df) > 0:
        _train_start = pd.Timestamp(train_df["race_date"].min())
        _train_end = pd.Timestamp(train_df["race_date"].max())
        _burn_cutoff = _train_start + pd.DateOffset(months=_burn_months)
        if _burn_cutoff < _train_end:
            train_df = train_df[train_df["race_date"] >= _burn_cutoff].copy()
        else:
            logger.info(
                f"Burn-in skipped: dataset span "
                f"({(_train_end - _train_start).days} days) "
                f"does not exceed {_burn_months}-month burn-in period"
            )

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df["relevance"].values
    y_test = test_df["relevance"].values

    # Group sizes: number of runners per race (order must match row order)
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
    logger.info(
        f"Test set: most recent data from {test_df['race_date'].min().date()} "
        f"to {test_df['race_date'].max().date()}"
    )

    return (
        X_train, X_test, y_train, y_test,
        groups_train, groups_test,
        feature_cols, scaler, test_df,
    )


def prepare_multi_target_data(
    df: pd.DataFrame,
) -> dict:
    """
    Prepare a shared train/test split with **multiple** target arrays.

    Identical preprocessing and temporal split to
    :func:`prepare_ranking_data`, but returns three targets per split:

    * ``y_*_rel``  — non-linear relevance labels  (for LTR ranker)
    * ``y_*_lb``   — ``lengths_behind``            (for regressor)
    * ``y_*_won``  — binary ``won``                (for classifier)

    Returns:
        dict with keys ``X_train``, ``X_test``, ``y_train_rel``,
        ``y_test_rel``, ``y_train_lb``, ``y_test_lb``, ``y_train_won``,
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
        logger.info(
            f"Removed {_n_before - len(df)} rows from {len(_degenerate_ids)} "
            f"degenerate races (single-runner / identical positions / bad odds)"
        )

    # Relevance labels: winner-heavy (1st=5, 2nd=2, 3rd=1, rest=0)
    fp = df["finish_position"].values.astype(int)
    df["relevance"] = np.where(fp == 1, 5, np.where(fp == 2, 2, np.where(fp == 3, 1, 0)))

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

    # ── Burn-in exclusion ────────────────────────────────────────────
    # Drop rows from the very START of the training window.  Feature
    # engineering already ran on the full history so cumulative/rolling
    # stats are correct — but these earliest rows have cold-start counts
    # (horse_prev_races ≈ 0, Elo ≈ 1500) that are unusable as training
    # signal and actively hurt gradient quality.
    _burn_months = getattr(config, "BURN_IN_MONTHS", 4)
    if _burn_months > 0 and len(train_df) > 0:
        _train_start = pd.Timestamp(train_df["race_date"].min())
        _train_end = pd.Timestamp(train_df["race_date"].max())
        _burn_cutoff = _train_start + pd.DateOffset(months=_burn_months)
        if _burn_cutoff < _train_end:
            _n_before = len(train_df)
            train_df = train_df[train_df["race_date"] >= _burn_cutoff].copy()
            _n_burned = _n_before - len(train_df)
            if _n_burned > 0:
                logger.info(
                    f"Burn-in: excluded {_n_burned} cold-start rows "
                    f"(first {_burn_months} months of training window)"
                )
        else:
            logger.info(
                f"Burn-in skipped: dataset span "
                f"({(_train_end - _train_start).days} days) "
                f"does not exceed {_burn_months}-month burn-in period"
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

    # Optional feature pruning (quick pilot model → drop bottom N %)
    _prune_frac = getattr(config, "FEATURE_PRUNE_FRACTION", 0.0)
    if _prune_frac > 0:
        feature_cols = _prune_features_quick(train_df, feature_cols, _prune_frac)

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

    y_train_lb = train_df["lengths_behind"].fillna(0).values.astype(np.float32)
    y_test_lb = test_df["lengths_behind"].fillna(0).values.astype(np.float32)

    y_train_won = train_df["won"].fillna(0).values.astype(int)
    y_test_won = test_df["won"].fillna(0).values.astype(int)

    # Market-residual target: realised outcome minus *normalised* implied
    # win probability (overround removed).  Using raw 1/odds would inflate
    # the baseline by ~20% and introduce overround-dependent noise.
    if "odds" in train_df.columns:
        _raw_ip_tr = (1.0 / train_df["odds"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0).values.astype(np.float32)
        _overround_tr = train_df.groupby("race_id")["odds"].transform(lambda o: (1.0 / o).sum()).values.astype(np.float32)
        ip_tr = _raw_ip_tr / np.maximum(_overround_tr, 1e-9)
    else:
        ip_tr = np.zeros(len(train_df), dtype=np.float32)
    if "odds" in test_df.columns:
        _raw_ip_te = (1.0 / test_df["odds"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0).values.astype(np.float32)
        _overround_te = test_df.groupby("race_id")["odds"].transform(lambda o: (1.0 / o).sum()).values.astype(np.float32)
        ip_te = _raw_ip_te / np.maximum(_overround_te, 1e-9)
    else:
        ip_te = np.zeros(len(test_df), dtype=np.float32)

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
    max_date = train_df["race_date"].max()
    days_ago = (max_date - train_df["race_date"]).dt.days.values.astype(np.float64)
    _hl = getattr(config, "RECENCY_HALF_LIFE_DAYS", 180)
    sample_weight_train = np.exp(-np.log(2) * days_ago / _hl)
    # Seasonal boost: same calendar month gets an uplift
    _seasonal = getattr(config, "RECENCY_SEASONAL_BOOST", 0.0)
    if _seasonal > 0:
        _cur_month = max_date.month
        _same_month = (train_df["race_date"].dt.month.values == _cur_month).astype(np.float64)
        sample_weight_train *= (1.0 + _seasonal * _same_month)
    # Normalise so mean weight ≈ 1 (keeps effective sample size stable)
    sample_weight_train = sample_weight_train / sample_weight_train.mean()

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


def _rps_objective_factory(
    groups: np.ndarray,
    finish_pos: np.ndarray,
):
    """Create a LightGBM custom ``fobj`` (gradient + Hessian) for RPS loss.

    Rank-Probability Score for a race of N runners:

        RPS = 1/(N-1) * sum_{k=1}^{N-1} (F_k - 1)^2

    The target CDF is 1 at every position (the winner is always in the
    top-k for all k >= 1).

    Gradient derivation for horse i with actual rank j_i (0-indexed):

        ∂RPS/∂s_i = 2·p_i/(N-1) · Σ_k  (F_k - 1) · (1_{j_i ≤ k} - F_k)
                  = 2·p_i/(N-1) · (suffix[j_i] - prefix[j_i])

    where:
        suffix[j] = Σ_{k=j}^{N-2}  residuals[k] · (1 - F_k)
        prefix[j] = Σ_{k=0}^{j-1}  residuals[k] · F_k
        residuals[k] = F_k - 1

    Hessian: diagonal approximation H_i = p_i·(1-p_i)·2/(N-1).

    Args:
        groups: Array of group sizes (runners per race).
        finish_pos: Actual finish positions (1 = winner), aligned with groups.

    Returns:
        A ``fobj(preds, dataset)`` callable set via ``params["objective"]``.
    """
    cum_g = np.concatenate([[0], np.cumsum(groups)]).astype(np.intp)
    fp_arr = finish_pos.astype(np.float32)
    n_races = len(groups)

    def rps_obj(preds: np.ndarray, train_data) -> tuple:  # LightGBM always calls fobj(preds, dataset)
        # train_data (Dataset) is unused — fp_arr from closure provides ground-truth ordering
        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)
        for r in range(n_races):
            start = int(cum_g[r])
            end = int(cum_g[r + 1])
            N = end - start
            if N < 2:
                hess[start:end] = 1.0
                continue

            # Softmax within race (numerically stable)
            s = preds[start:end].astype(np.float64)
            s = s - s.max()
            exp_s = np.exp(s)
            p = exp_s / exp_s.sum()

            # Sort by actual finish position (ascending → winner first)
            order = np.argsort(fp_arr[start:end])
            inv_order = np.empty(N, dtype=np.intp)
            inv_order[order] = np.arange(N, dtype=np.intp)

            # Cumulative predicted probability along actual ranking
            F = np.cumsum(p[order])                               # shape (N,)
            residuals = F[:N - 1] - 1.0

            # Vectorised suffix / prefix sums (avoid Python inner loops)
            rf_comp = residuals * (1.0 - F[:N - 1])              # r[k]·(1-F[k])
            suffix = np.zeros(N, dtype=np.float64)
            suffix[:N - 1] = np.cumsum(rf_comp[::-1])[::-1]      # suffix[j..N-2]

            rf_self = residuals * F[:N - 1]                       # r[k]·F[k]
            prefix = np.zeros(N, dtype=np.float64)
            prefix[1:] = np.cumsum(rf_self)                       # prefix[0..j-1]

            scale = 2.0 / (N - 1)
            j_arr = inv_order                                      # actual rank per runner
            grad[start:end] = p * scale * (suffix[j_arr] - prefix[j_arr])
            hess[start:end] = np.maximum(p * (1.0 - p) * scale, 1e-6)

        return grad, hess

    return rps_obj


def _value_objective_factory(
    groups: np.ndarray,
    finish_pos: np.ndarray,
    implied_probs: np.ndarray,
):
    """Create a LightGBM custom objective that optimises for betting value.

    Per-race softmax cross-entropy weighted by log-odds of the winner:

        L_r = -w_r · log(p_winner)
        w_r = max(1, log(1 / implied_prob_winner))

    This focuses model capacity on races where the winner was at higher
    odds (bigger overlay potential).  A 20/1 winner contributes ~3x the
    gradient of a 2/1 favourite, making the model prioritise exactly the
    edge-finding skill that drives positive ROI.

    Gradient:
        g_i = w_r · (p_i - y_i)     (standard softmax CE scaled by race weight)

    Hessian:
        H_i = w_r · p_i · (1 - p_i) (exact, no diagonal approximation)
    """
    cum_g = np.concatenate([[0], np.cumsum(groups)]).astype(np.intp)
    fp_arr = finish_pos.astype(np.float32)
    ip_arr = implied_probs.astype(np.float64)
    n_races = len(groups)

    # Pre-compute per-race weight from winner's implied probability
    race_weights = np.ones(n_races, dtype=np.float64)
    for r in range(n_races):
        start = int(cum_g[r])
        end = int(cum_g[r + 1])
        winner_mask = fp_arr[start:end] == 1
        if winner_mask.any():
            ip_winner = float(ip_arr[start + int(np.argmax(winner_mask))])
            # log(1/ip) ≈ log(decimal_odds);  clamp ip to avoid log(inf)
            ip_winner = max(ip_winner, 0.01)
            race_weights[r] = max(1.0, -np.log(ip_winner))

    def value_obj(preds: np.ndarray, train_data) -> tuple:
        grad = np.zeros_like(preds, dtype=np.float64)
        hess = np.zeros_like(preds, dtype=np.float64)
        for r in range(n_races):
            start = int(cum_g[r])
            end = int(cum_g[r + 1])
            N = end - start
            if N < 2:
                hess[start:end] = 1.0
                continue

            s = preds[start:end].astype(np.float64)
            s = s - s.max()
            exp_s = np.exp(s)
            p = exp_s / exp_s.sum()

            y = np.zeros(N, dtype=np.float64)
            winner_mask = fp_arr[start:end] == 1
            if winner_mask.any():
                y[np.argmax(winner_mask)] = 1.0

            w = race_weights[r]
            grad[start:end] = w * (p - y)
            hess[start:end] = np.maximum(w * p * (1.0 - p), 1e-6)

        return grad, hess

    return value_obj


def _unpickle_lgbm_rps_ranker(state):
    """Stable reconstructor for _LGBMRpsRanker.

    Pickle serialises this *function* by name so Streamlit module reloads
    no longer cause a ``PicklingError: it's not the same object`` failure.
    """
    from src.model import _LGBMRpsRanker as _cls  # noqa: PLC0415
    obj = object.__new__(_cls)
    obj.__setstate__(state)
    return obj


class _LGBMRpsRanker:
    """Sklearn-compatible wrapper around a native LightGBM booster trained
    with the custom Rank-Probability Score (RPS) objective.

    Provides ``.predict(X)`` returning raw scores (higher → more likely to
    win), fully compatible with the rest of the ``EnsembleModel`` pipeline.
    """

    def __init__(self, booster) -> None:
        self._booster = booster
        try:
            self.feature_importances_ = booster.feature_importance(
                importance_type="gain",
            )
        except Exception:
            self.feature_importances_ = None

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._booster.predict(X)

    def __repr__(self) -> str:
        try:
            return f"_LGBMRpsRanker(n_trees={self._booster.num_trees()})"
        except Exception:
            return "_LGBMRpsRanker()"

    # ── Pickle / joblib serialisation ───────────────────────────────────
    def __getstate__(self):
        return {
            "model_str": self._booster.model_to_string(),
            "feature_importances_": self.feature_importances_,
        }

    def __setstate__(self, state):
        import lightgbm as _lgb
        self._booster = _lgb.Booster(model_str=state["model_str"])
        self.feature_importances_ = state.get("feature_importances_")

    def __reduce__(self):
        return (_unpickle_lgbm_rps_ranker, (self.__getstate__(),))


def _build_monotone_constraints(
    feature_cols: list[str] | None,
) -> list[int] | None:
    """Build a LightGBM ``monotone_constraints`` vector (1 = mono-positive).

    Returns *None* if no ``feature_cols`` are provided (e.g. when train_ranker
    is called from auto-tune without column names known), which means LightGBM
    will apply no constraints.
    """
    if not feature_cols:
        return None
    _mono_feats = set(getattr(config, "MONOTONE_FEATURES", []))
    if not _mono_feats:
        return None
    constraints = [1 if c in _mono_feats else 0 for c in feature_cols]
    n_pos = sum(constraints)
    if n_pos > 0:
        logger.debug(
            f"Monotonic constraints: {n_pos} features constrained to be "
            f"monotone-positive ({[c for c in feature_cols if c in _mono_feats]})"
        )
    return constraints if any(constraints) else None


def train_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    model_type: str = "xgb_ranker",
    params: dict | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    groups_val: np.ndarray | None = None,
    finish_pos: np.ndarray | None = None,
    finish_pos_val: np.ndarray | None = None,
    implied_prob: np.ndarray | None = None,
    implied_prob_val: np.ndarray | None = None,
    feature_cols: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> object:
    """
    Train a Learning-to-Rank model.

    Args:
        X_train: Training features
        y_train: Relevance labels (higher = better)
        groups_train: Array of group sizes (runners per race)
        model_type: ``"xgb_ranker"`` or ``"lgbm_ranker"``
        params: Optional dict of hyperparameters. If *None*, falls back
                to ``config.XGBOOST_PARAMS`` / ``config.LIGHTGBM_PARAMS``.
        X_val / y_val / groups_val: Optional validation set for early
                stopping.  When provided, training stops if the
                validation metric does not improve for 50 rounds.

    Returns:
        Trained ranker model
    """
    logger.info(f"Training {model_type} ranker...")
    _es_rounds = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))

    if model_type == "xgb_ranker":
        defaults = {
            "n_estimators": config.XGBOOST_PARAMS["n_estimators"],
            "max_depth": config.XGBOOST_PARAMS["max_depth"],
            "learning_rate": config.XGBOOST_PARAMS["learning_rate"],
            "subsample": config.XGBOOST_PARAMS["subsample"],
            "colsample_bytree": config.XGBOOST_PARAMS["colsample_bytree"],
        }
        hp = {**defaults, **(params or {})}
        model = XGBRanker(
            objective="rank:ndcg",
            **hp,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        fit_kw: dict = {"group": groups_train}
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None and groups_val is not None:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["eval_group"] = [groups_val]
            fit_kw["verbose"] = False
            model.set_params(early_stopping_rounds=_es_rounds)
        model.fit(X_train, y_train, **fit_kw)

    elif model_type == "lgbm_ranker":
        defaults = {
            "n_estimators": config.LTR_PARAMS.get("n_estimators", 500),
            "max_depth": config.LTR_PARAMS.get("max_depth", 6),
            "learning_rate": config.LTR_PARAMS.get("learning_rate", 0.05),
            "subsample": config.LTR_PARAMS.get("subsample", 0.8),
            "colsample_bytree": config.LTR_PARAMS.get("colsample_bytree", 0.8),
            "min_child_samples": config.LTR_PARAMS.get("min_child_samples", 10),
            "reg_alpha": config.LTR_PARAMS.get("reg_alpha", 0.1),
            "reg_lambda": config.LTR_PARAMS.get("reg_lambda", 1.0),
        }
        hp = {**defaults, **(params or {})}
        _mono = _build_monotone_constraints(feature_cols)
        _can_es = (
            X_val is not None and groups_val is not None
            and finish_pos_val is not None and _es_rounds > 0
        )
        if _can_es:
            # Use native LightGBM API with custom RPS eval for early
            # stopping — RPS is smooth and reliable on small groups,
            # unlike NDCG.
            import lightgbm as _lgb_native  # noqa: PLC0415
            n_est = int(hp.pop("n_estimators", 500))
            lgb_params: dict = {
                "boosting_type": "gbdt",
                "objective": "lambdarank",
                "num_leaves": int(hp.pop("num_leaves", config.LTR_PARAMS.get("num_leaves", 31))),
                "max_depth": int(hp.pop("max_depth", -1)),
                "learning_rate": float(hp.pop("learning_rate", 0.05)),
                "feature_fraction": float(hp.pop("colsample_bytree", 0.8)),
                "bagging_fraction": float(hp.pop("subsample", 0.8)),
                "bagging_freq": 1,
                "min_child_samples": int(hp.pop("min_child_samples", 10)),
                "lambda_l1": float(hp.pop("reg_alpha", 0.1)),
                "lambda_l2": float(hp.pop("reg_lambda", 1.0)),
                "verbose": -1,
                "seed": config.RANDOM_SEED,
                "num_threads": -1,
                "metric": "None",
                "eval_at": [1],
            }
            if _mono is not None:
                lgb_params["monotone_constraints"] = _mono
            train_data = _lgb_native.Dataset(
                X_train, label=y_train.astype(np.float32),
                group=groups_train, free_raw_data=False,
            )
            val_data = _lgb_native.Dataset(
                X_val, label=y_val.astype(np.float32),
                group=groups_val, reference=train_data, free_raw_data=False,
            )
            _fp_val = finish_pos_val.astype(np.float32)
            _g_val = groups_val.copy()

            def _rps_eval_ranker(preds, _dataset):
                """Custom RPS feval for LGBMRanker early stopping."""
                probs = np.zeros(len(preds), dtype=np.float64)
                off = 0
                for g in _g_val:
                    sl = slice(off, off + g)
                    s = preds[sl].astype(np.float64)
                    s = s - s.max()
                    e = np.exp(s)
                    probs[sl] = e / max(e.sum(), 1e-12)
                    off += g
                rps = rps_per_race(probs, _fp_val, _g_val)
                return "rps", rps, False

            from lightgbm import early_stopping as _lgb_es, log_evaluation as _lgb_log
            callbacks = [
                _lgb_es(_es_rounds, verbose=False),
                _lgb_log(period=0),
            ]
            booster = _lgb_native.train(
                lgb_params, train_data, num_boost_round=n_est,
                valid_sets=[train_data, val_data],
                valid_names=["training", "valid"],
                feval=_rps_eval_ranker,
                callbacks=callbacks,
            )
            _best = getattr(booster, "best_iteration", n_est)
            if _best < n_est:
                logger.info(f"  lgbm_ranker early-stopped at round {_best}/{n_est}")
            model = _LGBMRpsRanker(booster)
        else:
            model = LGBMRanker(
                objective="lambdarank",
                **hp,
                subsample_freq=1,
                monotone_constraints=_mono,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            )
            fit_kw = {"group": groups_train, "eval_at": [1]}
            if sample_weight is not None:
                fit_kw["sample_weight"] = sample_weight
            model.fit(X_train, y_train, **fit_kw)

    elif model_type == "cat_ranker":
        _require_catboost()
        defaults = {
            "n_estimators": config.LTR_PARAMS.get("n_estimators", 500),
            "depth": config.LTR_PARAMS.get("max_depth", 6),
            "learning_rate": config.LTR_PARAMS.get("learning_rate", 0.05),
            "l2_leaf_reg": 3.0,
        }
        hp = {**defaults, **(params or {})}
        model = CatBoostRanker(
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=1",
            random_seed=config.RANDOM_SEED,
            verbose=False,
            **hp,
        )
        g_train = _groups_to_group_id(groups_train)
        train_pool = Pool(X_train, y_train, group_id=g_train,
                          weight=sample_weight)
        fit_kw: dict = {}
        if X_val is not None and y_val is not None and groups_val is not None:
            g_val = _groups_to_group_id(groups_val)
            fit_kw["eval_set"] = Pool(X_val, y_val, group_id=g_val)
            fit_kw["use_best_model"] = True
            fit_kw["early_stopping_rounds"] = _es_rounds
        model.fit(train_pool, **fit_kw)

    elif model_type == "lgbm_rps":
        # ── LightGBM with custom Rank-Probability Score objective ──────────
        import lightgbm as _lgb_native  # noqa: PLC0415
        if finish_pos is None:
            raise ValueError(
                "finish_pos must be supplied when model_type='lgbm_rps'"
            )
        _lp: dict = {**config.LTR_PARAMS, **(params or {})}
        n_est = int(_lp.get("n_estimators", 500))
        lgb_params: dict = {
            "boosting_type": "gbdt",
            "num_leaves": int(_lp.get("num_leaves", 31)),
            "max_depth": int(_lp.get("max_depth", -1)),
            "learning_rate": float(_lp.get("learning_rate", 0.05)),
            "feature_fraction": float(_lp.get("colsample_bytree", 0.8)),
            "bagging_fraction": float(_lp.get("subsample", 0.8)),
            "bagging_freq": 5,
            "min_child_samples": int(_lp.get("min_child_samples", 10)),
            "lambda_l1": float(_lp.get("reg_alpha", 0.1)),
            "lambda_l2": float(_lp.get("reg_lambda", 1.0)),
            "verbose": -1,
            "seed": config.RANDOM_SEED,
            "num_threads": -1,
        }
        lgb_params["objective"] = _rps_objective_factory(groups_train, finish_pos)
        _mono = _build_monotone_constraints(feature_cols)
        if _mono is not None:
            lgb_params["monotone_constraints"] = _mono
        train_data = _lgb_native.Dataset(
            X_train,
            label=finish_pos.astype(np.float32),
            group=groups_train,
            free_raw_data=False,
        )

        # Build validation set + custom RPS eval for early stopping
        valid_sets = [train_data]
        valid_names = ["training"]
        callbacks = []
        if (X_val is not None and groups_val is not None
                and finish_pos_val is not None and _es_rounds > 0):
            val_data = _lgb_native.Dataset(
                X_val,
                label=finish_pos_val.astype(np.float32),
                group=groups_val,
                reference=train_data,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

            _fp_val = finish_pos_val.astype(np.float32)
            _g_val = groups_val.copy()

            def _rps_eval(preds, _dataset):
                """Custom LightGBM feval: race-level RPS on validation."""
                probs = np.zeros(len(preds), dtype=np.float64)
                off = 0
                for g in _g_val:
                    sl = slice(off, off + g)
                    s = preds[sl].astype(np.float64)
                    s = s - s.max()
                    e = np.exp(s)
                    probs[sl] = e / max(e.sum(), 1e-12)
                    off += g
                rps = rps_per_race(probs, _fp_val, _g_val)
                return "rps", rps, False  # name, value, is_higher_better

            lgb_params["metric"] = "None"  # disable default metric
            from lightgbm import early_stopping as _lgb_es, log_evaluation as _lgb_log
            callbacks.append(_lgb_es(_es_rounds, verbose=False))
            callbacks.append(_lgb_log(period=0))  # suppress per-round log

        booster = _lgb_native.train(
            lgb_params,
            train_data,
            num_boost_round=n_est,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=_rps_eval if len(valid_sets) > 1 else None,
            callbacks=callbacks if callbacks else None,
        )
        _best = getattr(booster, "best_iteration", n_est)
        if _best < n_est:
            logger.info(f"  lgbm_rps early-stopped at round {_best}/{n_est}")
        model = _LGBMRpsRanker(booster)

    elif model_type == "lgbm_value":
        # ── LightGBM with value-weighted cross-entropy objective ───────────
        # Softmax CE weighted by log-odds of the winner — focuses model
        # capacity on identifying high-odds winners, the skill that
        # directly drives positive ROI.
        import lightgbm as _lgb_native  # noqa: PLC0415
        if finish_pos is None:
            raise ValueError(
                "finish_pos must be supplied when model_type='lgbm_value'"
            )
        if implied_prob is None:
            raise ValueError(
                "implied_prob must be supplied when model_type='lgbm_value'"
            )
        _lp: dict = {**config.LTR_PARAMS, **(params or {})}
        n_est = int(_lp.get("n_estimators", 500))
        lgb_params: dict = {
            "boosting_type": "gbdt",
            "num_leaves": int(_lp.get("num_leaves", 31)),
            "max_depth": int(_lp.get("max_depth", -1)),
            "learning_rate": float(_lp.get("learning_rate", 0.05)),
            "feature_fraction": float(_lp.get("colsample_bytree", 0.8)),
            "bagging_fraction": float(_lp.get("subsample", 0.8)),
            "bagging_freq": 5,
            "min_child_samples": int(_lp.get("min_child_samples", 10)),
            "lambda_l1": float(_lp.get("reg_alpha", 0.1)),
            "lambda_l2": float(_lp.get("reg_lambda", 1.0)),
            "verbose": -1,
            "seed": config.RANDOM_SEED,
            "num_threads": -1,
        }
        lgb_params["objective"] = _value_objective_factory(
            groups_train, finish_pos, implied_prob,
        )
        _mono = _build_monotone_constraints(feature_cols)
        if _mono is not None:
            lgb_params["monotone_constraints"] = _mono
        train_data = _lgb_native.Dataset(
            X_train,
            label=finish_pos.astype(np.float32),
            group=groups_train,
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["training"]
        callbacks = []
        if (X_val is not None and groups_val is not None
                and finish_pos_val is not None and _es_rounds > 0):
            val_data = _lgb_native.Dataset(
                X_val,
                label=finish_pos_val.astype(np.float32),
                group=groups_val,
                reference=train_data,
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

            _fp_val_v = finish_pos_val.astype(np.float32)
            _g_val_v = groups_val.copy()

            def _rps_eval_value(preds, _dataset):
                """Custom feval: RPS on validation (objective-agnostic)."""
                probs = np.zeros(len(preds), dtype=np.float64)
                off = 0
                for g in _g_val_v:
                    sl = slice(off, off + g)
                    s = preds[sl].astype(np.float64)
                    s = s - s.max()
                    e = np.exp(s)
                    probs[sl] = e / max(e.sum(), 1e-12)
                    off += g
                rps = rps_per_race(probs, _fp_val_v, _g_val_v)
                return "rps", rps, False

            lgb_params["metric"] = "None"
            from lightgbm import early_stopping as _lgb_es, log_evaluation as _lgb_log
            callbacks.append(_lgb_es(_es_rounds, verbose=False))
            callbacks.append(_lgb_log(period=0))

        booster = _lgb_native.train(
            lgb_params,
            train_data,
            num_boost_round=n_est,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=_rps_eval_value if len(valid_sets) > 1 else None,
            callbacks=callbacks if callbacks else None,
        )
        _best = getattr(booster, "best_iteration", n_est)
        if _best < n_est:
            logger.info(f"  lgbm_value early-stopped at round {_best}/{n_est}")
        model = _LGBMRpsRanker(booster)

    else:
        raise ValueError(f"Unknown ranker type: {model_type}")

    logger.info(f"{model_type} training complete")
    return model


def auto_tune_ranker(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    model_type: str = "xgb_ranker",
    n_trials: int = 30,
    callback=None,
) -> dict:
    """
    Automatic hyperparameter optimisation via Optuna.

    Uses **RPS** (Rank Probability Score) on a validation fold as the
    objective.  RPS is the natural metric for ranked outcomes: it
    penalises the full predicted finishing distribution, not just
    the binary win probability.
    Lower RPS = better calibrated probabilities = better value detection.

    Args:
        X_train / y_train / groups_train: Training split.
        X_val / y_val / groups_val: Validation split (held-out from
            training data — *not* the final test set).
        model_type: ``"xgb_ranker"``, ``"lgbm_ranker"``, or ``"cat_ranker"``.
        n_trials: Number of Optuna trials.
        callback: Optional ``callable(trial_number, total, best_score)``
                  for progress updates.

    Returns:
        dict with ``best_params``, ``best_score``,
        ``trials`` (list of dicts), ``n_trials``.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Derive finish positions from relevance labels (higher = better)
    fp_val = np.zeros(len(y_val), dtype=np.float32)
    offset = 0
    for g in groups_val:
        gl = y_val[offset:offset + g]
        # Convert relevance to rank: highest relevance = finish pos 1
        order = np.argsort(-gl)
        for rank, idx in enumerate(order):
            fp_val[offset + idx] = rank + 1
        offset += g

    temperature = getattr(config, "SOFTMAX_TEMPERATURE", 1.0)

    def _rps_score(model, X, groups):
        """RPS via per-race softmax probabilities."""
        scores = model.predict(X)
        probs = np.zeros(len(scores))
        offset = 0
        for g in groups:
            gs = scores[offset:offset + g]
            scaled = gs / max(temperature, 1e-6)
            exp_s = np.exp(scaled - scaled.max())
            probs[offset:offset + g] = exp_s / exp_s.sum()
            offset += g
        return rps_per_race(probs, fp_val, groups)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        if model_type == "xgb_ranker":
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)
            params["gamma"] = trial.suggest_float("gamma", 0.0, 5.0, step=0.1)
        elif model_type == "lgbm_ranker":
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 255)
        else:
            params["depth"] = trial.suggest_int("depth", 4, 10)
            params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True)

        model = train_ranker(X_train, y_train, groups_train, model_type, params=params)
        score = _rps_score(model, X_val, groups_val)

        if callback is not None:
            callback(trial.number + 1, n_trials, score)

        return score

    study = optuna.create_study(direction="minimize")  # lower RPS = better
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_log = []
    for t in study.trials:
        trials_log.append({
            "number": t.number,
            "params": t.params,
            "score": t.value,
        })

    logger.info(
        f"\nOptuna auto-tune ({model_type}) — {n_trials} trials\n"
        f"  Best RPS: {study.best_value:.6f}\n"
        f"  Best params: {study.best_params}"
    )

    return {
        "best_params": study.best_params,
        "best_score": round(study.best_value, 4),
        "trials": trials_log,
        "n_trials": n_trials,
    }


def evaluate_ranker(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    test_df: pd.DataFrame,
    model_name: str = "Ranker",
    feature_cols: list[str] | None = None,
) -> dict:
    """
    Evaluate a ranking model with ranking + calibration metrics.

    Metrics:
        - RPS — full-ranking calibration quality (lower = better)
        - Brier Score — binary calibration quality (lower = better)
        - Log Loss — penalises confident wrong predictions
        - NDCG@1, NDCG@3 — how well the model ranks the top finishers
        - Top-1 accuracy — how often the model's #1 pick actually won
        - Win-in-top-3 — how often the actual winner is in the model's top 3
    """
    if feature_cols is not None and isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=feature_cols)
    scores = model.predict(X_test)

    # Split scores back into per-race groups
    ndcg_1_list, ndcg_3_list = [], []
    top1_correct, win_in_top3, total_races = 0, 0, 0
    temperature = getattr(config, "SOFTMAX_TEMPERATURE", 1.0)
    all_probs = np.zeros(len(scores))
    offset = 0

    for g_size in groups_test:
        g_scores = scores[offset : offset + g_size]
        g_labels = y_test[offset : offset + g_size]

        # Per-race softmax for calibration metrics
        scaled = g_scores / max(temperature, 1e-6)
        exp_s = np.exp(scaled - scaled.max())
        all_probs[offset : offset + g_size] = exp_s / exp_s.sum()
        offset += g_size

        if g_size < 2 or g_labels.max() == g_labels.min():
            continue

        total_races += 1

        # NDCG (sklearn wants 2-D arrays)
        try:
            ndcg_1_list.append(ndcg_score([g_labels], [g_scores], k=1))
            ndcg_3_list.append(ndcg_score([g_labels], [g_scores], k=3))
        except ValueError:
            pass

        # Top-1 accuracy: did the model's top pick actually won?
        model_top = np.argmax(g_scores)
        actual_top = np.argmax(g_labels)
        if model_top == actual_top:
            top1_correct += 1

        # Winner in model's top 3?
        model_top3 = set(np.argsort(g_scores)[-3:])
        if actual_top in model_top3:
            win_in_top3 += 1

    # Calibration metrics
    if "won" in test_df.columns:
        y_won = test_df["won"].values[:len(scores)].astype(int)
    elif "finish_position" in test_df.columns:
        y_won = (test_df["finish_position"].values[:len(scores)] == 1).astype(int)
    else:
        y_won = (y_test == y_test.max()).astype(int)

    probs_clipped = np.clip(all_probs, 1e-15, 1 - 1e-15)
    try:
        brier = float(brier_score_loss(y_won, probs_clipped))
    except Exception:
        brier = float('nan')
    try:
        logloss = float(log_loss(y_won, probs_clipped))
    except Exception:
        logloss = float('nan')

    # RPS — full-ranking calibration metric; needs finish positions
    if "finish_position" in test_df.columns:
        fp_eval = test_df["finish_position"].values[:len(scores)].astype(np.float32)
        rps = rps_per_race(all_probs, fp_eval, groups_test)
    else:
        rps = float("nan")

    metrics = {
        "rps": round(rps, 6),
        "brier_score": round(brier, 6),
        "log_loss": round(logloss, 4),
        "ndcg_at_1": np.mean(ndcg_1_list) if ndcg_1_list else 0.0,
        "ndcg_at_3": np.mean(ndcg_3_list) if ndcg_3_list else 0.0,
        "top1_accuracy": top1_correct / total_races if total_races else 0.0,
        "win_in_top3": win_in_top3 / total_races if total_races else 0.0,
        "total_races": total_races,
    }

    # ── Reliability diagram (10-bin ECE) ──────────────────────────
    # Each bin: (mean_predicted_prob, observed_win_rate, n_runners).
    # ECE = Σ |mean_pred - obs_win_rate| × (bin_count / total).
    _bins = np.linspace(0.0, 1.0, 11)  # 10 bins of width 0.1
    _bin_ids = np.digitize(probs_clipped, _bins, right=False) - 1
    _bin_ids = np.clip(_bin_ids, 0, 9)
    _reliability_bins = []
    _ece_sum = 0.0
    _n_total = len(probs_clipped)
    for _b in range(10):
        _mask = _bin_ids == _b
        _cnt = int(_mask.sum())
        if _cnt == 0:
            _reliability_bins.append(None)
            continue
        _mean_pred = float(probs_clipped[_mask].mean())
        _obs = float(y_won[_mask].mean())
        _reliability_bins.append((_mean_pred, _obs, _cnt))
        _ece_sum += abs(_mean_pred - _obs) * (_cnt / _n_total)
    metrics["ece"] = round(_ece_sum, 6)
    metrics["reliability_bins"] = _reliability_bins  # list of (mean_pred, obs_frac, n) or None

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Evaluation Results")
    logger.info(f"{'='*50}")
    logger.info(f"  RPS:             {metrics['rps']:.6f}")
    logger.info(f"  Brier Score:     {metrics['brier_score']:.6f}")
    logger.info(f"  ECE:             {metrics['ece']:.6f}")
    logger.info(f"  Log Loss:        {metrics['log_loss']:.4f}")
    logger.info(f"  NDCG@1:          {metrics['ndcg_at_1']:.4f}")
    logger.info(f"  NDCG@3:          {metrics['ndcg_at_3']:.4f}")
    logger.info(f"  Top-1 Accuracy:  {metrics['top1_accuracy']:.4f}")
    logger.info(f"  Winner in Top 3: {metrics['win_in_top3']:.4f}")
    logger.info(f"  Races evaluated: {metrics['total_races']}")
    logger.info(f"{'='*50}")

    return metrics


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


def analyse_test_set(
    ltr_scores: np.ndarray,
    win_probs: np.ndarray,
    groups_test: np.ndarray,
    test_df: pd.DataFrame,
    value_threshold: float = 0.05,
    staking_mode: str = "flat",
    kelly_fraction: float = 0.25,
    bankroll: float = 100.0,
    place_probs: np.ndarray | None = None,
    ew_min_place_edge: float | None = None,
) -> dict:
    """
    Run betting simulation on the held-out test set.

    Args:
        ltr_scores: Raw LTR ranking scores (for top_pick selection).
        win_probs: Pre-calibrated P(win) from the win classifier.
        groups_test: Runners-per-race arrays.
        test_df: Test DataFrame (needs race_id, race_date, horse_name, odds, won).
        value_threshold: Minimum edge for value bets.
        staking_mode: "flat" or "kelly".
        kelly_fraction: Fraction of full Kelly.
        bankroll: Starting bankroll (Kelly only).
        place_probs: Pre-calibrated P(place) from place classifier.

    Returns:
        dict with bets, curves, stats, calibration, etc.
    """
    analysis_df = test_df.copy().reset_index(drop=True)

    # Probabilities are already calibrated — use directly
    analysis_df["model_prob"] = win_probs
    analysis_df["rank_score"] = ltr_scores

    has_odds = "odds" in analysis_df.columns
    has_won = "won" in analysis_df.columns

    if has_odds:
        # Use raw implied probability (1/odds) — consistent with strategy
        # calibrator.  Overround correction was inflating apparent edge, making
        # marginal bets look like value and producing lower real-world ROI.
        analysis_df["implied_prob"] = 1.0 / analysis_df["odds"]
        analysis_df["value_score"] = (
            analysis_df["model_prob"] - analysis_df["implied_prob"]
        )
    else:
        analysis_df["implied_prob"] = 0.0
        analysis_df["value_score"] = 0.0

    if not has_won:
        analysis_df["won"] = (
            analysis_df["finish_position"] == 1
        ).astype(int)

    # Attach place probability if supplied
    if place_probs is not None and len(place_probs) == len(analysis_df):
        analysis_df["place_prob"] = place_probs

    # ── Betting simulation ────────────────────────────────────────
    from src.utils import kelly_criterion as _kelly_fn
    from src.each_way import get_ew_terms, ew_value as _ew_value_fn

    use_kelly = staking_mode == "kelly"
    _current_bankroll = float(bankroll) if use_kelly else 0.0

    all_bets: list[dict] = []

    # Sort races chronologically so Kelly bankroll accumulates correctly
    _race_order = (
        analysis_df.groupby("race_id")["race_date"]
        .first()
        .sort_values()
        .index
    )

    for race_id in _race_order:
        race_group = analysis_df[analysis_df["race_id"] == race_id]

        # Strategy 1 — Top Pick: use model_prob (value model) when LTR
        # scores are unavailable (all-zero), otherwise use rank_score.
        _rs = race_group["rank_score"]
        _pick_col = "rank_score" if _rs.max() != _rs.min() else "model_prob"
        # Skip top pick when the model cannot genuinely distinguish
        # runners (all scores identical) — otherwise idxmax() picks by
        # row order, inflating strike rate via data-ordering artefacts.
        if race_group[_pick_col].max() == race_group[_pick_col].min():
            continue
        best = race_group.loc[race_group[_pick_col].idxmax()]
        odds_val = float(best["odds"]) if has_odds else 0.0
        pnl = (odds_val - 1.0) if best["won"] == 1 else -1.0

        rd = best["race_date"]
        rd_str = str(rd.date()) if hasattr(rd, "date") else str(rd)[:10]

        all_bets.append({
            "race_date": rd_str,
            "race_id": race_id,
            "track": best.get("track", ""),
            "strategy": "top_pick",
            "horse_name": best["horse_name"],
            "model_prob": round(float(best["model_prob"]), 4),
            "odds": round(odds_val, 2),
            "won": int(best["won"]),
            "stake": 1.0,
            "pnl": round(pnl, 2),
            "clv": round(float(best["model_prob"]) * odds_val, 4) if has_odds else None,
        })

        # Strategy 2 — Value Bets (odds-dependent threshold)
        # Tighter threshold at short odds (where calibration is better),
        # looser at long odds (where small edges are masked by variance).
        # Uses raw implied probability (1/odds) — same as strategy calibrator.
        # Any bet passing this check already has CLV > 1, so no extra gate needed.
        if has_odds:
            edge = race_group["value_score"]
            odds_col = race_group["odds"]
            dyn_thresh = value_threshold * np.sqrt(odds_col / 3.0)
            value_picks = race_group[edge > dyn_thresh]
            for _, vp in value_picks.iterrows():
                vp_odds = float(vp["odds"])
                vp_prob = float(vp["model_prob"])

                # CLV > 1 is guaranteed by positive raw edge + threshold;
                # no separate gate needed.
                vp_clv = round(vp_prob * vp_odds, 4)

                # Determine stake
                if use_kelly and _current_bankroll > 0:
                    k_frac = _kelly_fn(vp_prob, vp_odds, fraction=kelly_fraction)
                    stake = round(max(k_frac * _current_bankroll, 0), 4)
                    stake = min(stake, _current_bankroll)
                else:
                    stake = 1.0

                if stake < 0.01:
                    continue  # skip negligible bets

                pnl_v = stake * (vp_odds - 1.0) if vp["won"] == 1 else -stake

                if use_kelly:
                    _current_bankroll += pnl_v
                    _current_bankroll = max(_current_bankroll, 0)  # can't go negative

                vd = vp["race_date"]
                vd_str = str(vd.date()) if hasattr(vd, "date") else str(vd)[:10]
                all_bets.append({
                    "race_date": vd_str,
                    "race_id": race_id,
                    "track": vp.get("track", ""),
                    "strategy": "value",
                    "horse_name": vp["horse_name"],
                    "model_prob": round(vp_prob, 4),
                    "odds": round(vp_odds, 2),
                    "won": int(vp["won"]),
                    "stake": round(stake, 4),
                    "pnl": round(pnl_v, 4),
                    "clv": vp_clv,
                })

            # Strategy 3 — Each-Way Value Bets
            if has_odds and "place_prob" in race_group.columns:
                _nr = int(race_group["num_runners"].iloc[0]) if "num_runners" in race_group.columns else len(race_group)
                _hcap = bool(race_group.get("handicap", pd.Series(0)).iloc[0]) if "handicap" in race_group.columns else False
                ew_terms = get_ew_terms(_nr, is_handicap=_hcap)
                if ew_terms.eligible:
                    # Adjust P(top-3) → P(top-places_paid) + per-race normalise
                    from src.each_way import adjust_place_probs_for_race as _adj_pp
                    _raw_pp = race_group["place_prob"].values
                    _win_pp = race_group["model_prob"].values
                    _adj_place = _adj_pp(_raw_pp, _win_pp, ew_terms.places_paid)

                    for _ew_i, (_, ep) in enumerate(race_group.iterrows()):
                        if ep["odds"] < 4.0 or ep["odds"] > 51.0:
                            continue
                        ev_result = _ew_value_fn(
                            ep["model_prob"], float(_adj_place[_ew_i]),
                            ep["odds"], ew_terms,
                        )
                        # Use dedicated EW edge threshold (sidebar "Min place edge")
                        # with the same dynamic odds scaling as value bets.
                        _ew_base = ew_min_place_edge if ew_min_place_edge is not None else value_threshold
                        _ew_dyn_thresh = _ew_base * np.sqrt(ep["odds"] / 3.0)
                        if ev_result["place_edge"] > _ew_dyn_thresh and ev_result["place_ev"] > 0:
                            # EW bet = 2 units (1 win + 1 place)
                            fp_val = int(ep["finish_position"]) if "finish_position" in ep.index else 0
                            won_flag = int(ep["won"])
                            placed_flag = int(fp_val > 0 and fp_val <= ew_terms.places_paid)
                            pnl_ew = -2.0  # cost: 2 units
                            if won_flag:
                                pnl_ew += ep["odds"]  # win leg returns
                                pnl_ew += ev_result["place_odds"]  # place leg returns
                            elif placed_flag:
                                pnl_ew += ev_result["place_odds"]  # place leg only

                            ed = ep["race_date"]
                            ed_str = str(ed.date()) if hasattr(ed, "date") else str(ed)[:10]
                            all_bets.append({
                                "race_date": ed_str,
                                "race_id": race_id,
                                "track": ep.get("track", ""),
                                "strategy": "each_way",
                                "horse_name": ep["horse_name"],
                                "model_prob": round(float(ep["model_prob"]), 4),
                                "odds": round(float(ep["odds"]), 2),
                                "won": won_flag,
                                "placed": int(placed_flag or won_flag),
                                "stake": 2.0,
                                "pnl": round(pnl_ew, 4),
                                "clv": round(float(ep["model_prob"]) * float(ep["odds"]), 4),
                            })

    bets_df = pd.DataFrame(all_bets) if all_bets else pd.DataFrame()

    # ── Cumulative P&L curves ────────────────────────────────────
    curve_parts: list[pd.DataFrame] = []
    for strategy in ["top_pick", "value", "each_way"]:
        strat = bets_df[bets_df["strategy"] == strategy].copy() if not bets_df.empty else pd.DataFrame()
        if strat.empty:
            continue
        strat = strat.sort_values("race_date").reset_index(drop=True)
        strat["cum_pnl"] = strat["pnl"].cumsum()
        strat["cum_staked"] = strat["stake"].cumsum()
        strat["bet_number"] = range(1, len(strat) + 1)
        strat["cum_roi_pct"] = (
            strat["cum_pnl"] / strat["cum_staked"] * 100
        ).fillna(0)
        curve_parts.append(strat)
    curves_df = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()

    # ── Summary stats per strategy ───────────────────────────────
    stats: dict = {}
    for strategy in ["top_pick", "value", "each_way"]:
        strat = (
            bets_df[bets_df["strategy"] == strategy]
            if not bets_df.empty else pd.DataFrame()
        )
        n_bets = len(strat)
        n_wins = int(strat["won"].sum()) if n_bets else 0
        n_placed = int(strat["placed"].sum()) if (n_bets and "placed" in strat.columns) else n_wins
        total_pnl = float(strat["pnl"].sum()) if n_bets else 0.0
        total_staked = float(strat["stake"].sum()) if n_bets else 0.0
        strike_rate = n_wins / n_bets * 100 if n_bets else 0.0
        place_rate = n_placed / n_bets * 100 if n_bets else 0.0
        roi = total_pnl / total_staked * 100 if total_staked > 0 else 0.0
        avg_odds_win = float(strat[strat["won"] == 1]["odds"].mean()) if n_wins else 0.0
        avg_odds_all = float(strat["odds"].mean()) if n_bets else 0.0
        avg_stake = total_staked / n_bets if n_bets else 0.0
        max_dd = _max_drawdown(strat["pnl"].values) if n_bets else 0.0
        # CLV = model_prob × odds; > 1 means we beat the closing market
        avg_clv = float(strat["clv"].mean()) if (n_bets and "clv" in strat.columns and strat["clv"].notna().any()) else None

        stats[strategy] = {
            "bets": n_bets,
            "winners": n_wins,
            "placed": n_placed,
            "total_staked": round(total_staked, 2),
            "pnl": round(total_pnl, 2),
            "strike_rate": round(strike_rate, 1),
            "place_rate": round(place_rate, 1),
            "roi": round(roi, 1),
            "avg_stake": round(avg_stake, 4),
            "avg_odds_winners": round(avg_odds_win, 2),
            "avg_odds_all": round(avg_odds_all, 2),
            "max_drawdown": round(max_dd, 2),
            "avg_clv": round(avg_clv, 4) if avg_clv is not None else None,
        }

    # ── Daily P&L (for bar chart) ────────────────────────────────
    for strategy in ["top_pick", "value", "each_way"]:
        strat = (
            bets_df[bets_df["strategy"] == strategy].copy()
            if not bets_df.empty else pd.DataFrame()
        )
        if not strat.empty:
            daily = strat.groupby("race_date").agg(
                daily_pnl=("pnl", "sum"),
                daily_bets=("pnl", "count"),
                daily_wins=("won", "sum"),
            ).reset_index()
            daily["cum_pnl"] = daily["daily_pnl"].cumsum()
            stats[f"{strategy}_daily"] = daily.to_dict("records")

    # ── Value bets by odds band ──────────────────────────────────
    if not bets_df.empty and has_odds:
        value_bets = bets_df[bets_df["strategy"] == "value"].copy()
        if not value_bets.empty:
            bins = [0, 3, 5, 8, 12, 20, 9999]
            labels = ["1-3", "3-5", "5-8", "8-12", "12-20", "20+"]
            value_bets["odds_band"] = pd.cut(
                value_bets["odds"], bins=bins, labels=labels,
            )
            band = value_bets.groupby("odds_band", observed=True).agg(
                bets=("pnl", "count"),
                winners=("won", "sum"),
                pnl=("pnl", "sum"),
            ).reset_index()
            band["strike_rate"] = (
                band["winners"] / band["bets"] * 100
            ).round(1)
            band["roi"] = (
                band["pnl"] / band["bets"] * 100
            ).round(1)
            stats["value_by_odds_band"] = band.to_dict("records")

    # ── Calibration: model prob bucket vs actual win rate ────────
    cal_bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    cal_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]
    analysis_df["prob_bucket"] = pd.cut(
        analysis_df["model_prob"], bins=cal_bins, labels=cal_labels,
    )
    cal = analysis_df.groupby("prob_bucket", observed=True).agg(
        runners=("won", "count"),
        actual_wins=("won", "sum"),
        avg_model_prob=("model_prob", "mean"),
    ).reset_index()
    cal["actual_win_rate"] = (
        cal["actual_wins"] / cal["runners"] * 100
    ).round(1)
    cal["avg_model_pct"] = (cal["avg_model_prob"] * 100).round(1)
    calibration = cal.to_dict("records")

    # ── Test-set date range ──────────────────────────────────────
    test_dates = pd.to_datetime(analysis_df["race_date"])
    date_range = (
        str(test_dates.min().date()),
        str(test_dates.max().date()),
    )

    logger.info(
        f"\nTest-Set Analysis ({date_range[0]} → {date_range[1]}):\n"
        f"  Top-Pick: {stats.get('top_pick', {}).get('bets', 0)} bets, "
        f"SR {stats.get('top_pick', {}).get('strike_rate', 0):.1f}%, "
        f"ROI {stats.get('top_pick', {}).get('roi', 0):+.1f}%, "
        f"P&L {stats.get('top_pick', {}).get('pnl', 0):+.2f}\n"
        f"  Value:    {stats.get('value', {}).get('bets', 0)} bets, "
        f"SR {stats.get('value', {}).get('strike_rate', 0):.1f}%, "
        f"ROI {stats.get('value', {}).get('roi', 0):+.1f}%, "
        f"P&L {stats.get('value', {}).get('pnl', 0):+.2f}\n"
        f"  Each-Way: {stats.get('each_way', {}).get('bets', 0)} bets, "
        f"SR {stats.get('each_way', {}).get('strike_rate', 0):.1f}%, "
        f"ROI {stats.get('each_way', {}).get('roi', 0):+.1f}%, "
        f"P&L {stats.get('each_way', {}).get('pnl', 0):+.2f}"
    )

    _value_config = {
        "value_threshold": value_threshold,
        "staking_mode": staking_mode,
        "kelly_fraction": kelly_fraction,
        "starting_bankroll": bankroll,
        "final_bankroll": round(_current_bankroll, 2) if use_kelly else None,
    }

    # ── Slim predictions cache (for strategy calibration) ────────
    # Save the per-runner model/place probabilities so the calibration
    # page can skip re-running predict_race() on subsequent visits.
    _pred_cols = ["race_id", "horse_name"]
    for _c in ["odds", "jockey", "trainer", "race_date"]:
        if _c in analysis_df.columns:
            _pred_cols.append(_c)
    _pred_cols.append("model_prob")
    if "place_prob" in analysis_df.columns:
        _pred_cols.append("place_prob")
    predictions_df = analysis_df[_pred_cols].copy()

    return {
        "bets": bets_df,
        "curves": curves_df,
        "predictions": predictions_df,
        "stats": stats,
        "calibration": calibration,
        "test_date_range": date_range,
        "test_races": analysis_df["race_id"].nunique(),
        "test_runners": len(analysis_df),
        "value_config": _value_config,
    }


def _max_drawdown(pnl_series: np.ndarray) -> float:
    """Calculate maximum drawdown from a P&L array."""
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


# =====================================================================
#  Learning-to-Rank Predictors
# =====================================================================


class RankingPredictor:
    """
    Predictor using a Learning-to-Rank model (XGBRanker or LGBMRanker).

    Instead of predicting win probabilities, it produces a *relevance
    score* for each runner.  Scores are only meaningful **relative to
    other horses in the same race**.

    The predict_race method normalises scores into pseudo-probabilities
    (softmax over the race) so the output format is compatible with the
    rest of the application.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.model_type = None
        self.metrics = None
        self.test_analysis = None

    def train(
        self,
        df: pd.DataFrame,
        model_type: str = "xgb_ranker",
        save: bool = True,
        params: dict | None = None,
    ) -> dict:
        self.model_type = model_type

        (
            X_train, X_test, y_train, y_test,
            groups_train, groups_test,
            self.feature_cols, self.scaler, test_df,
        ) = prepare_ranking_data(df)

        self.model = train_ranker(X_train, y_train, groups_train, model_type, params=params)

        self.metrics = evaluate_ranker(
            self.model, X_test, y_test, groups_test, test_df,
            model_name=model_type.upper(),
        )

        # Test-set betting analysis (equity curves, P&L, value stats)
        test_scores = self.model.predict(X_test)
        _win_probs = _grouped_softmax(test_scores, groups_test, 1.0)
        self.test_analysis = analyse_test_set(
            ltr_scores=test_scores,
            win_probs=_win_probs,
            groups_test=groups_test,
            test_df=test_df,
        )

        # Feature importance
        fi = get_feature_importance(self.model, self.feature_cols)
        logger.info(f"\nTop 15 Features:\n{fi.head(15).to_string()}")

        if save:
            self.save()

        return self.metrics

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        missing = [c for c in self.feature_cols if c not in race_df.columns]
        for col in missing:
            race_df[col] = 0

        X = race_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Raw ranking scores (higher = better)
        raw_scores = self.model.predict(X_scaled)

        # Convert to pseudo-probabilities via softmax
        exp_scores = np.exp(raw_scores - raw_scores.max())  # numerical stability
        win_probs = exp_scores / exp_scores.sum()

        results = race_df[["horse_name"]].copy()
        if "jockey" in race_df.columns:
            results["jockey"] = race_df["jockey"]
        if "trainer" in race_df.columns:
            results["trainer"] = race_df["trainer"]
        if "odds" in race_df.columns:
            results["odds"] = race_df["odds"]

        results["win_probability"] = win_probs
        results["rank_score"] = raw_scores
        results["predicted_rank"] = results["win_probability"].rank(
            ascending=False, method="min"
        ).astype(int)

        if "odds" in race_df.columns:
            results["implied_prob"] = 1.0 / race_df["odds"].values
            results["value_score"] = results["win_probability"] - results["implied_prob"]

        return results.sort_values("predicted_rank")

    def explain_race(
        self,
        race_df: pd.DataFrame,
        top_n_features: int = 10,
    ) -> dict:
        """
        Compute SHAP values for every runner in a race.

        Returns a dict mapping horse_name → DataFrame with columns
        ``feature``, ``shap_value``, ``feature_value``
        (sorted by absolute SHAP value, top *top_n_features*).
        """
        import shap

        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        missing = [c for c in self.feature_cols if c not in race_df.columns]
        for col in missing:
            race_df[col] = 0

        X = race_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        explanations: dict[str, pd.DataFrame] = {}
        horse_names = race_df["horse_name"].values

        for i, name in enumerate(horse_names):
            sv = shap_values[i]
            fv = X_scaled[i]
            df_expl = pd.DataFrame({
                "feature": self.feature_cols,
                "shap_value": sv,
                "feature_value": fv,
            })
            df_expl["abs_shap"] = df_expl["shap_value"].abs()
            df_expl = df_expl.sort_values("abs_shap", ascending=False)
            explanations[name] = df_expl.head(top_n_features).drop(
                columns="abs_shap",
            )

        return explanations

    def save(self):
        save_path = os.path.join(config.MODELS_DIR, "ranker_model.joblib")
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "model_type": self.model_type,
            },
            save_path,
        )
        logger.info(f"Ranking model saved to {save_path}")

    def load(self):
        load_path = os.path.join(config.MODELS_DIR, "ranker_model.joblib")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No ranking model found at {load_path}")
        data = joblib.load(load_path)
        self.model = data["model"]
        self.scaler = data.get("scaler") or _IdentityScaler()
        self.feature_cols = data["feature_cols"]
        self.model_type = data["model_type"]
        logger.info("Ranking model loaded successfully")


class RankEnsemblePredictor:
    """
    Ensemble that averages XGBRanker + LGBMRanker scores (and optionally
    blends with a classifier's probabilities).
    """

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_cols = None
        self.weights = {"xgb_ranker": 0.5, "lgbm_ranker": 0.5}
        self.metrics = None
        self.test_analysis = None

    def train(self, df: pd.DataFrame, save: bool = True, params: dict | None = None) -> dict:
        (
            X_train, X_test, y_train, y_test,
            groups_train, groups_test,
            self.feature_cols, self.scaler, test_df,
        ) = prepare_ranking_data(df)

        all_metrics = {}
        for mt in ["xgb_ranker", "lgbm_ranker"]:
            self.models[mt] = train_ranker(X_train, y_train, groups_train, mt, params=params)
            m = evaluate_ranker(
                self.models[mt], X_test, y_test, groups_test, test_df,
                model_name=mt.upper(),
            )
            all_metrics[mt] = m

        # Evaluate ensemble: average the raw scores, then re-evaluate
        ens_scores = np.zeros(X_test.shape[0])
        for mt, model in self.models.items():
            ens_scores += self.weights[mt] * model.predict(X_test)

        # Re-run evaluation metrics on blended scores
        ens_metrics = self._evaluate_blended(
            ens_scores, y_test, groups_test, test_df
        )
        all_metrics["rank_ensemble"] = ens_metrics
        self.metrics = all_metrics

        # Test-set betting analysis (equity curves, P&L, value stats)
        _win_probs = _grouped_softmax(ens_scores, groups_test, 1.0)
        self.test_analysis = analyse_test_set(
            ltr_scores=ens_scores,
            win_probs=_win_probs,
            groups_test=groups_test,
            test_df=test_df,
        )

        if save:
            self.save()

        return all_metrics

    @staticmethod
    def _evaluate_blended(
        scores, y_test, groups_test, test_df
    ) -> dict:
        from sklearn.metrics import ndcg_score as _ndcg

        ndcg1, ndcg3, top1, win3, total = [], [], 0, 0, 0
        temperature = getattr(config, "SOFTMAX_TEMPERATURE", 1.0)
        all_probs = np.zeros(len(scores))
        offset = 0
        for g in groups_test:
            gs = scores[offset:offset + g]
            gl = y_test[offset:offset + g]
            scaled = gs / max(temperature, 1e-6)
            exp_s = np.exp(scaled - scaled.max())
            all_probs[offset:offset + g] = exp_s / exp_s.sum()
            offset += g
            if g < 2 or gl.max() == gl.min():
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

        if "won" in test_df.columns:
            y_won = test_df["won"].values[:len(scores)].astype(int)
        elif "finish_position" in test_df.columns:
            y_won = (test_df["finish_position"].values[:len(scores)] == 1).astype(int)
        else:
            y_won = (y_test == y_test.max()).astype(int)

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
            _fp_eval = test_df["finish_position"].values[:len(scores)].astype(np.float32)
            rps = rps_per_race(all_probs, _fp_eval, groups_test)
        else:
            rps = float("nan")

        metrics = {
            "rps": round(rps, 6),
            "brier_score": round(brier, 6),
            "log_loss": round(logloss, 4),
            "ndcg_at_1": np.mean(ndcg1) if ndcg1 else 0,
            "ndcg_at_3": np.mean(ndcg3) if ndcg3 else 0,
            "top1_accuracy": top1 / total if total else 0,
            "win_in_top3": win3 / total if total else 0,
            "total_races": total,
        }
        logger.info(f"\n{'='*50}")
        logger.info("  RANK ENSEMBLE Evaluation Results")
        logger.info(f"{'='*50}")
        logger.info(f"  RPS:             {metrics['rps']:.6f}")
        logger.info(f"  Brier Score:     {metrics['brier_score']:.6f}")
        logger.info(f"  Log Loss:        {metrics['log_loss']:.4f}")
        for k in ["ndcg_at_1", "ndcg_at_3", "top1_accuracy", "win_in_top3"]:
            logger.info(f"  {k}: {metrics[k]:.4f}")
        logger.info(f"  total_races: {metrics['total_races']}")
        logger.info(f"{'='*50}")
        return metrics

    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in race_df.columns]
        for col in missing:
            race_df[col] = 0

        X = race_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        raw_scores = np.zeros(X_scaled.shape[0])
        for mt, model in self.models.items():
            raw_scores += self.weights[mt] * model.predict(X_scaled)

        exp_scores = np.exp(raw_scores - raw_scores.max())
        win_probs = exp_scores / exp_scores.sum()

        results = race_df[["horse_name"]].copy()
        if "jockey" in race_df.columns:
            results["jockey"] = race_df["jockey"]
        if "trainer" in race_df.columns:
            results["trainer"] = race_df["trainer"]
        if "odds" in race_df.columns:
            results["odds"] = race_df["odds"]

        results["win_probability"] = win_probs
        results["rank_score"] = raw_scores
        results["predicted_rank"] = results["win_probability"].rank(
            ascending=False, method="min"
        ).astype(int)

        if "odds" in race_df.columns:
            results["implied_prob"] = 1.0 / race_df["odds"].values
            results["value_score"] = results["win_probability"] - results["implied_prob"]

        return results.sort_values("predicted_rank")

    def explain_race(
        self,
        race_df: pd.DataFrame,
        top_n_features: int = 10,
    ) -> dict:
        """
        Compute SHAP values for each runner using the **xgb_ranker**
        sub-model (dominant tree in the ensemble).
        """
        import shap

        # Pick whichever sub-model is available (prefer xgb)
        model = self.models.get("xgb_ranker") or next(iter(self.models.values()))

        missing = [c for c in self.feature_cols if c not in race_df.columns]
        for col in missing:
            race_df[col] = 0

        X = race_df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        explanations: dict[str, pd.DataFrame] = {}
        horse_names = race_df["horse_name"].values

        for i, name in enumerate(horse_names):
            sv = shap_values[i]
            fv = X_scaled[i]
            df_expl = pd.DataFrame({
                "feature": self.feature_cols,
                "shap_value": sv,
                "feature_value": fv,
            })
            df_expl["abs_shap"] = df_expl["shap_value"].abs()
            df_expl = df_expl.sort_values("abs_shap", ascending=False)
            explanations[name] = df_expl.head(top_n_features).drop(
                columns="abs_shap",
            )

        return explanations

    def save(self):
        path = os.path.join(config.MODELS_DIR, "rank_ensemble_models.joblib")
        joblib.dump(
            {
                "models": self.models,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "weights": self.weights,
            },
            path,
        )
        logger.info(f"Rank ensemble saved to {path}")

    def load(self):
        path = os.path.join(config.MODELS_DIR, "rank_ensemble_models.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No rank ensemble found at {path}")
        data = joblib.load(path)
        self.models = data["models"]
        self.scaler = data.get("scaler") or _IdentityScaler()
        self.feature_cols = data["feature_cols"]
        self.weights = data["weights"]
        logger.info("Rank ensemble loaded successfully")


# =====================================================================
#  Triple Ensemble (LTR + Regressor + Classifier)
# =====================================================================


class TripleEnsemblePredictor:
    """
    Three task-specific models, each optimised for a different betting strategy:

    1. **LTR Ranker** — Optimised for ranking horses correctly within each
       race (LambdaRank / NDCG).  Used for the **Top Pick** strategy.
    2. **Win Classifier** — Binary classifier → calibrated P(win) (Value Bets).
       Used for the **Value Bet** strategy (edge vs market).
    3. **Place Classifier** — Calibrated P(place) estimates.
       Used for the **Each-Way** strategy.

    Each model is trained independently — no blending or ensembling.
    Calibration (Platt scaling) is fitted per-model on OOF data.
    """

    def __init__(
        self,
        frameworks: dict[str, str] | None = None,
        **_ignored,
    ):
        self.ltr_model = None
        self.clf_model = None
        self.place_model = None
        self.scaler = None
        self.feature_cols = None
        self.frameworks = {
            "ltr": "lgbm", "classifier": "cat", "place": "cat",
            **(frameworks or {}),
        }
        # Win-classifier calibration (Platt)
        self.calibration_temp = 1.0  # kept for serialisation compat
        self.platt_a = 1.0
        self.platt_b = 0.0
        # Place-classifier calibration (Platt)
        self.place_platt_a = 1.0
        self.place_platt_b = 0.0
        self.metrics = None
        self.test_analysis = None

    # ── Framework-aware helpers ──────────────────────────────────
    def _clf_predict_proba(self, model, X) -> np.ndarray:
        """Return P(class=1) from a classifier (XGB or LGBM)."""
        return model.predict_proba(X)[:, 1]

    @property
    def _ltr_ranker_type(self) -> str:
        """Map framework key to train_ranker model_type string."""
        if self.frameworks["ltr"] == "cat":
            return "cat_ranker"
        if self.frameworks["ltr"] == "lgbm":
            obj = getattr(config, "LTR_OBJECTIVE", "value")
            if obj == "value":
                return "lgbm_value"
            if obj == "rps":
                return "lgbm_rps"
            return "lgbm_ranker"  # "lambdarank" or anything else
        return "xgb_ranker"

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
        """Train three task-specific models:

        * **LTR Ranker** — optimised for ranking (Top Pick).
        * **Win Classifier** — calibrated P(win) (Value Bets).
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

        oof_clf_logits_parts: list[np.ndarray] = []
        oof_place_probs_parts: list[np.ndarray] = []
        oof_groups_parts: list[np.ndarray] = []
        oof_fp_parts: list[np.ndarray] = []
        oof_placed_parts: list[np.ndarray] = []

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

            # Train fold models (disposable — for OOF scores only)
            # Win classifier (binary: won or not)
            y_won_f = data["y_train_won"][:tr_end]
            n_pos_w = max(int(y_won_f.sum()), 1)
            self.clf_model = self._train_win_classifier(
                X_f_tr, y_won_f,
                scale_pos_weight=(len(y_won_f) - n_pos_w) / n_pos_w,
                params=_p.get("classifier"), sample_weight=sw_f,
            )
            # Place classifier
            y_placed_f = data["y_train_placed"][:tr_end]
            n_pp = max(int(y_placed_f.sum()), 1)
            self.place_model = self._train_place_classifier(
                X_f_tr, y_placed_f,
                scale_pos_weight=(len(y_placed_f) - n_pp) / n_pp,
                params=_p.get("place"), sample_weight=sw_f,
            )

            # Score validation fold
            X_f_vl_df = pd.DataFrame(X_f_vl, columns=self.feature_cols)
            clf_probs_vl = self.clf_model.predict_proba(X_f_vl_df)[:, 1]
            place_probs_vl = self.place_model.predict_proba(X_f_vl_df)[:, 1]

            oof_clf_logits_parts.append(clf_probs_vl)
            oof_place_probs_parts.append(place_probs_vl)
            oof_groups_parts.append(g_f_vl)
            oof_fp_parts.append(fp_f_vl)
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
            }

            _tuned_params: dict[str, dict] = {}
            _tune_models = {"ltr": "LTR Ranker", "classifier": "Win Classifier", "place": "Place Classifier"}
            _tune_metrics = {"ltr": "NDCG@1", "classifier": "LogLoss", "place": "LogLoss"}
            for _m_i, (_mk, _m_label) in enumerate(sorted(_tune_models.items())):
                _m_pct = 0.36 + (_m_i / 3) * 0.25
                _m_metric = _tune_metrics[_mk]
                _cb(f"Phase 1b — Tuning {_m_label} ({_m_i + 1}/3) …", _m_pct)

                def _at_cb(tnum, total, score, _lbl=_m_label, _base=_m_pct, _met=_m_metric):
                    _cb(
                        f"Tuning {_lbl} — trial {tnum}/{total} ({_met} {score:.4f})",
                        _base + (tnum / total) * (0.25 / 3),
                    )

                _at_result = self._auto_tune_model(
                    _mk, _at_X_tr, _at_X_vl,
                    _at_targets_tr, _at_targets_vl,
                    _at_g_tr, _at_g_vl,
                    sw_train=_at_sw, n_trials=_at_trials, callback=_at_cb,
                )
                _tuned_params[_mk] = _at_result["best_params"]
                logger.info(f"  {_m_label}: {_m_metric} {_at_result['best_score']:.6f}")
            params = _tuned_params
            _p = _tuned_params

        # ── Phase 2: Retrain on full training data ───────────────
        _cb("Phase 2 — Retraining on full data …", 0.62)
        all_metrics: dict = {}
        sw_full = data["sample_weight_train"]

        logger.info("Retraining 3 task-specific models on full training data …")

        n_test = len(X_test)

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
            logger.info(
                f"  Phase-2 early stopping split: {len(X_es_tr)} train, "
                f"{len(X_es_vl)} val ({es_vl_beg - es_tr_end} runners purged)"
            )
        elif _es_rounds > 0:
            logger.info("  Phase-2 early stopping disabled (insufficient split size)")
        else:
            logger.info("  Phase-2 early stopping disabled by config")

        # 1) LTR Ranker (Top Pick model)
        # Skipped when config.TRAIN_RANKER is False — value model is used for
        # sorting instead.  Existing saved models that contain an ltr_model
        # will still use it at prediction time.
        if getattr(config, "TRAIN_RANKER", True):
            _ltr_fw = self.frameworks["ltr"].upper()
            _cb(f"Phase 2 — Training LTR Ranker ({_ltr_fw}) …", 0.64)
            logger.info(f"Training LTR Ranker ({_ltr_fw}) — Top Pick model …")
            _ltr_uses_es = use_es
            if _ltr_uses_es:
                self.ltr_model = train_ranker(
                    X_es_tr, y_es_rel_tr, g_es_tr,
                    self._ltr_ranker_type, params=_p.get("ltr"),
                    X_val=X_es_vl, y_val=y_es_rel_vl, groups_val=g_es_vl,
                    finish_pos=fp_es_tr, finish_pos_val=fp_es_vl,
                    implied_prob=ip_es_tr, implied_prob_val=ip_es_vl,
                    feature_cols=self.feature_cols, sample_weight=sw_es_tr,
                )
            else:
                self.ltr_model = train_ranker(
                    X_train, data["y_train_rel"], groups_train,
                    self._ltr_ranker_type, params=_p.get("ltr"),
                    finish_pos=data["fp_train"], implied_prob=data["ip_train"],
                    feature_cols=self.feature_cols, sample_weight=sw_full,
                )
        else:
            logger.info("Skipping LTR Ranker training (TRAIN_RANKER=False); predictions will use value model ordering.")
            self.ltr_model = None

        # 2) Win Classifier (Value Bets model)
        _clf_fw = self.frameworks.get("classifier", "cat").upper()
        _cb(f"Phase 2 — Training Win Classifier ({_clf_fw}) …", 0.72)
        logger.info(f"Training Win Classifier ({_clf_fw}) — Value model …")
        if use_es:
            n_pos_w = int(y_es_won_tr.sum())
            n_neg_w = len(y_es_won_tr) - n_pos_w
            self.clf_model = self._train_win_classifier(
                X_es_tr, y_es_won_tr,
                scale_pos_weight=n_neg_w / max(n_pos_w, 1),
                params=_p.get("classifier"), sample_weight=sw_es_tr,
                eval_set=[(X_es_vl, y_es_won_vl)],
            )
        else:
            n_pos_w = int(data["y_train_won"].sum())
            n_neg_w = len(data["y_train_won"]) - n_pos_w
            self.clf_model = self._train_win_classifier(
                X_train, data["y_train_won"],
                scale_pos_weight=n_neg_w / max(n_pos_w, 1),
                params=_p.get("classifier"), sample_weight=sw_full,
            )

        # 3) Place Classifier (EW model)
        _place_fw = self.frameworks.get("place", "cat").upper()
        _cb(f"Phase 2 — Training Place Classifier ({_place_fw}) …", 0.80)
        logger.info(f"Training Place Classifier ({_place_fw}) — EW model …")
        if use_es:
            n_pos_p = int(y_es_placed_tr.sum())
            n_neg_p = len(y_es_placed_tr) - n_pos_p
            self.place_model = self._train_place_classifier(
                X_es_tr, y_es_placed_tr,
                scale_pos_weight=n_neg_p / max(n_pos_p, 1),
                params=_p.get("place"), sample_weight=sw_es_tr,
                eval_set=[(X_es_vl, y_es_placed_vl)],
            )
        else:
            n_pos_p = int(data["y_train_placed"].sum())
            n_neg_p = len(data["y_train_placed"]) - n_pos_p
            self.place_model = self._train_place_classifier(
                X_train, data["y_train_placed"],
                scale_pos_weight=n_neg_p / max(n_pos_p, 1),
                params=_p.get("place"), sample_weight=sw_full,
            )

        # ── Score test set with each model ───────────────────────
        _cb("Evaluating models on test set …", 0.86)
        X_test_df = pd.DataFrame(X_test, columns=self.feature_cols)

        ltr_test_scores = (
            self.ltr_model.predict(X_test_df)
            if self.ltr_model is not None
            else np.zeros(len(X_test))
        )
        clf_test_probs_raw = self.clf_model.predict_proba(X_test_df)[:, 1]
        place_test_probs_raw = self.place_model.predict_proba(X_test_df)[:, 1]

        # Apply calibration
        win_probs = self._calibrate_win_probs(clf_test_probs_raw, groups_test, test_df)
        place_probs = self._calibrate_place_probs(place_test_probs_raw)

        # ── Evaluate each model for its task ─────────────────────
        # LTR: ranking metrics (Top Pick task) — skip when not trained.
        if self.ltr_model is not None:
            all_metrics["ltr_ranker"] = self._evaluate_as_ranker(
                ltr_test_scores, data["y_test_rel"], groups_test, test_df,
                "LTR_RANKER (Top Pick)",
            )
        else:
            all_metrics["ltr_ranker"] = {
                "ndcg_at_1": None, "ndcg_at_3": None,
                "top1_accuracy": None, "win_in_top3": None,
                "rps": None, "brier_score": None, "log_loss": None,
                "total_races": 0, "note": "LTR ranker not trained (TRAIN_RANKER=False)",
            }
        # Win Classifier: calibration + value-bet metrics
        # Pass the fully-calibrated win_probs so evaluation matches
        # what predict_race / analyse_test_set actually use.
        all_metrics["win_classifier"] = self._evaluate_as_ranker(
            clf_test_probs_raw, data["y_test_rel"], groups_test, test_df,
            "WIN_CLASSIFIER (Value)",
            calibrated_probs=win_probs,
        )
        # Place Classifier: place-specific metrics
        all_metrics["place_classifier"] = self._evaluate_place_model(
            place_probs, place_test_probs_raw, test_df, groups_test,
        )

        self.metrics = all_metrics

        # ── Train-set metrics (overfit diagnostics) ──────────────
        _cb("Computing overfit diagnostics …", 0.89)
        logger.info("Computing training-set metrics for overfit diagnostics …")
        X_train_df = pd.DataFrame(X_train, columns=self.feature_cols)
        train_df_full = data["train_df"]
        train_metrics: dict = {}

        ltr_train_scores = (
            self.ltr_model.predict(X_train_df)
            if self.ltr_model is not None
            else np.zeros(len(X_train))
        )
        if self.ltr_model is not None:
            train_metrics["ltr_ranker"] = self._evaluate_as_ranker(
                ltr_train_scores, data["y_train_rel"], groups_train,
                train_df_full, "LTR_RANKER_TRAIN",
            )
        else:
            train_metrics["ltr_ranker"] = {
                "ndcg_at_1": None, "ndcg_at_3": None,
                "top1_accuracy": None, "win_in_top3": None,
                "rps": None, "brier_score": None, "log_loss": None,
                "total_races": 0,
            }
        clf_train_probs_raw = self.clf_model.predict_proba(X_train_df)[:, 1]
        win_train_probs = self._calibrate_win_probs(
            clf_train_probs_raw, groups_train, train_df_full,
        )
        train_metrics["win_classifier"] = self._evaluate_as_ranker(
            clf_train_probs_raw, data["y_train_rel"], groups_train,
            train_df_full, "WIN_CLASSIFIER_TRAIN",
            calibrated_probs=win_train_probs,
        )
        place_train_raw = self.place_model.predict_proba(X_train)[:, 1]
        place_train_cal = self._calibrate_place_probs(place_train_raw)
        train_metrics["place_classifier"] = self._evaluate_place_model(
            place_train_cal, place_train_raw, train_df_full, groups_train,
        )
        self.train_metrics = train_metrics

        # ── Betting simulation ───────────────────────────────────
        _cb("Analysing test-set betting performance …", 0.92)
        _vc = value_config or {}
        self.test_analysis = analyse_test_set(
            ltr_scores=ltr_test_scores,
            win_probs=win_probs,
            groups_test=groups_test,
            test_df=test_df,
            value_threshold=_vc.get("value_threshold", 0.05),
            staking_mode=_vc.get("staking_mode", "flat"),
            kelly_fraction=_vc.get("kelly_fraction", 0.25),
            bankroll=_vc.get("bankroll", 100.0),
            place_probs=place_probs,
            ew_min_place_edge=_vc.get("ew_min_place_edge"),
        )

        # Feature importance
        _fi_model = self.ltr_model or self.clf_model
        if _fi_model is not None:
            fi = get_feature_importance(_fi_model, self.feature_cols)
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
        sw_train: np.ndarray | None = None,
        n_trials: int = 30,
        callback=None,
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
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Per-model evaluation helpers (aligned with each model's objective)
        fp_val = targets_val["fp"].astype(np.float32)
        rel_val = targets_val["rel"]

        def _ndcg1_eval(scores, groups):
            """Negative NDCG@1 — lower is better (for LTR)."""
            off, vals = 0, []
            for g in groups:
                if g < 2:
                    off += g
                    continue
                g_scores = scores[off:off + g]
                g_labels = rel_val[off:off + g]
                vals.append(ndcg_score([g_labels], [g_scores], k=1))
                off += g
            return -float(np.mean(vals)) if vals else 0.0

        def _win_logloss_eval(probs, groups):
            """Log-loss on P(win) — lower is better (for win classifier)."""
            won = targets_val["won"].astype(np.float32)
            eps = 1e-9
            probs = np.clip(probs, eps, 1 - eps)
            return -float(np.mean(won * np.log(probs) + (1 - won) * np.log(1 - probs)))

        def _logloss_eval(scores, groups):
            """Log-loss — lower is better (for place classifier)."""
            placed = targets_val["placed"].astype(np.float32)
            probs = 1.0 / (1.0 + np.exp(-scores))   # logits → probs
            eps = 1e-9
            probs = np.clip(probs, eps, 1 - eps)
            return -float(np.mean(placed * np.log(probs) + (1 - placed) * np.log(1 - probs)))

        _eval_fn = {
            "ltr": _ndcg1_eval,
            "classifier": _win_logloss_eval,
            "place": _logloss_eval,
        }
        _metric_names = {
            "ltr": "NDCG@1",
            "classifier": "LogLoss",
            "place": "LogLoss",
        }

        fw = self.frameworks.get(
            model_key, "lgbm" if model_key in {"ltr", "residual"} else "xgb",
        )
        _feat_cols = self.feature_cols

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 100, 1500, step=50,
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.3, log=True,
                ),
                "subsample": trial.suggest_float(
                    "subsample", 0.5, 1.0, step=0.05,
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.3, 1.0, step=0.05,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-8, 10.0, log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True,
                ),
            }
            if fw == "xgb":
                params["min_child_weight"] = trial.suggest_int(
                    "min_child_weight", 1, 20,
                )
                params["gamma"] = trial.suggest_float(
                    "gamma", 0.0, 5.0, step=0.1,
                )
            elif fw == "lgbm":
                params["min_child_samples"] = trial.suggest_int(
                    "min_child_samples", 10, 40,
                )
                params["num_leaves"] = trial.suggest_int(
                    "num_leaves", 20, 100,
                )
            else:  # catboost
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 20.0, log=True),
                    "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                }

            # Train and score
            if model_key == "ltr":
                mdl = train_ranker(
                    X_train, targets_train["rel"], groups_train,
                    self._ltr_ranker_type, params=params,
                    finish_pos=targets_train["fp"],
                    implied_prob=targets_train.get("ip"),
                    feature_cols=_feat_cols,
                )
                scores = mdl.predict(
                    pd.DataFrame(X_val, columns=_feat_cols)
                )
            elif model_key == "classifier":
                n_pw = max(int(targets_train["won"].sum()), 1)
                mdl = self._train_win_classifier(
                    X_train, targets_train["won"],
                    scale_pos_weight=(len(targets_train["won"]) - n_pw) / n_pw,
                    params=params, sample_weight=sw_train,
                )
                scores = mdl.predict_proba(
                    pd.DataFrame(X_val, columns=_feat_cols)
                )[:, 1]
            elif model_key == "place":
                n_pp = max(int(targets_train["placed"].sum()), 1)
                mdl = self._train_place_classifier(
                    X_train, targets_train["placed"],
                    scale_pos_weight=(len(targets_train["placed"]) - n_pp) / n_pp,
                    params=params, sample_weight=sw_train,
                )
                scores = _proba_to_logit(mdl.predict_proba(X_val)[:, 1])
            else:
                raise ValueError(f"Unknown model_key: {model_key}")

            score = _eval_fn[model_key](scores, groups_val)
            if callback is not None:
                _mn = _metric_names[model_key]
                callback(trial.number + 1, n_trials, score)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        _mn = _metric_names[model_key]
        logger.info(
            f"  Optuna auto-tune ({model_key}, {fw}) — {n_trials} trials\n"
            f"    Best {_mn}: {study.best_value:.6f}\n"
            f"    Best params: {study.best_params}"
        )
        return {
            "best_params": study.best_params,
            "best_score": round(study.best_value, 6),
            "n_trials": n_trials,
        }

    def _train_classifier(self, X, y, scale_pos_weight=1.0, params=None, sample_weight=None, eval_set=None):
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
        }
        hp.update(self._filter_params(params, fw))
        if fw == "lgbm":
            hp.setdefault("min_child_samples", config.CLASSIFIER_PARAMS.get("min_child_samples", 10))
            model = _FocalLGBMClassifier(
                objective=_focal_binary_objective,
                **hp,
                subsample_freq=1,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
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
                n_jobs=-1,
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
        }
        if "num_leaves" in config.CLASSIFIER_PARAMS:
            hp["num_leaves"] = config.CLASSIFIER_PARAMS["num_leaves"]
        hp.update(self._filter_params(params, fw))
        if fw == "lgbm":
            hp.setdefault("min_child_samples", config.CLASSIFIER_PARAMS.get("min_child_samples", 10))
            model = LGBMClassifier(
                objective="binary",
                **hp,
                subsample_freq=1,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
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
                n_jobs=-1,
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
        }
        hp.update(self._filter_params(params, fw))
        if fw == "lgbm":
            hp.setdefault("min_child_samples", config.PLACE_CLASSIFIER_PARAMS.get("min_child_samples", 10))
            model = LGBMClassifier(
                objective="binary",
                **hp,
                subsample_freq=1,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
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
                n_jobs=-1,
            )
            fit_kw = {"sample_weight": sample_weight}
            if eval_set is not None:
                model.set_params(early_stopping_rounds=_es_rounds)
                fit_kw["eval_set"] = eval_set
                fit_kw["verbose"] = False
        model.fit(X, y, **fit_kw)
        logger.info(f"Place Classifier ({fw.upper()}) training complete")
        return model

    def _evaluate_as_ranker(self, scores, y_test, groups_test, test_df, name,
                            calibrated_probs=None):
        """Evaluate scores using ranking + calibration metrics.

        *scores* are used for ranking metrics (NDCG, top-k accuracy).
        *calibrated_probs*, when provided, are used for calibration
        metrics (RPS, Brier, log-loss) and value-bet simulation.  If
        omitted, raw softmax(scores, T=1) is used as a fallback.
        """
        from sklearn.metrics import ndcg_score as _ndcg

        ndcg1, ndcg3 = [], []
        top1, win3, total = 0, 0, 0

        # Use pre-calibrated probabilities when available;
        # fall back to raw softmax for LTR / uncalibrated models.
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
            # OR where all scores are identical (model has no opinion — e.g.
            # all-zero LTR scores when ranker was not trained).  Counting
            # argmax ties as "correct" would inflate top-1 accuracy artificially.
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

        metrics = {
            "rps": round(rps, 6),
            "brier_score": round(brier, 6),
            "log_loss": round(logloss, 4),
            "ndcg_at_1": np.mean(ndcg1) if ndcg1 else 0,
            "ndcg_at_3": np.mean(ndcg3) if ndcg3 else 0,
            "top1_accuracy": top1 / total if total else 0,
            "win_in_top3": win3 / total if total else 0,
            "total_races": total,
        }

        # ── Betting-relevant metrics ─────────────────────────────
        # These help the user see which model config leads to better
        # betting outcomes, not just better ranking.
        _has_odds = "odds" in test_df.columns
        if _has_odds:
            _odds = test_df["odds"].values[:len(scores)].astype(np.float64)
            _raw_ip = np.where(_odds > 0, 1.0 / _odds, 0.0)
            # Normalise per race to strip bookmaker overround
            _ip_group_sum = np.add.reduceat(_raw_ip, _group_offsets(groups_test))
            _gids = np.repeat(np.arange(len(groups_test)), groups_test)
            _imp_prob = _raw_ip / np.maximum(_ip_group_sum[_gids], 1e-9)
            _edge = all_probs - _imp_prob
            _vt = getattr(config, "VALUE_THRESHOLD", 0.05)
            _dyn_thresh = _vt * np.sqrt(_odds / 3.0)
            _is_vb = _edge > _dyn_thresh
            _n_vb = int(_is_vb.sum())
            if _n_vb > 0:
                _vb_won = y_won[_is_vb]
                _vb_odds = _odds[_is_vb]
                _vb_sr = float(_vb_won.mean())
                # Flat-stake ROI: (sum of returns - stakes) / stakes
                _vb_returns = np.where(_vb_won == 1, _vb_odds - 1.0, -1.0)
                _vb_roi = float(_vb_returns.mean())  # per-bet avg P&L
                _vb_avg_edge = float(_edge[_is_vb].mean())
                metrics["value_bets"] = _n_vb
                metrics["value_bet_sr"] = round(_vb_sr, 4)
                metrics["value_bet_roi"] = round(_vb_roi, 4)
                metrics["avg_edge"] = round(_vb_avg_edge, 4)
            else:
                metrics["value_bets"] = 0
                metrics["value_bet_sr"] = None
                metrics["value_bet_roi"] = None
                metrics["avg_edge"] = None
        else:
            metrics["value_bets"] = None
            metrics["value_bet_sr"] = None
            metrics["value_bet_roi"] = None
            metrics["avg_edge"] = None

        logger.info(f"\n{'='*50}")
        logger.info(f"  {name} Evaluation Results")
        logger.info(f"{'='*50}")
        logger.info(f"  RPS:             {metrics['rps']:.6f}")
        logger.info(f"  Brier Score:     {metrics['brier_score']:.6f}")
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

        return probs

    def _calibrate_place_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply Platt calibration to place probabilities."""
        if self.place_platt_a == 1.0 and self.place_platt_b == 0.0:
            return raw_probs.copy()
        _eps = 1e-9
        lp = np.log(np.clip(raw_probs, _eps, 1 - _eps)
                     / np.clip(1 - raw_probs, _eps, 1))
        return 1.0 / (1.0 + np.exp(-(self.place_platt_a * lp + self.place_platt_b)))

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

        return {
            "brier_calibrated": round(brier_cal, 4),
            "brier_raw": round(brier_raw, 4),
            "place_precision": round(place_precision, 3),
        }

    # ── Prediction ───────────────────────────────────────────────
    def predict_race(self, race_df: pd.DataFrame, ew_fraction: float | None = None) -> pd.DataFrame:
        if self.ltr_model is None and self.clf_model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        race_df = race_df.reset_index(drop=True)
        missing = [c for c in self.feature_cols if c not in race_df.columns]
        for col in missing:
            race_df[col] = 0

        X = race_df[self.feature_cols].values
        X_scaled_np = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_np, columns=self.feature_cols, index=race_df.index)
        n = len(X_scaled)
        groups = np.array([n])

        # ── Score with each model ────────────────────────────────
        ltr_scores = self.ltr_model.predict(X_scaled) if self.ltr_model is not None else np.zeros(n)
        clf_probs_raw = self.clf_model.predict_proba(X_scaled)[:, 1] if self.clf_model is not None else np.full(n, 1.0 / n)

        # Calibrated win probabilities
        win_probs = self._calibrate_win_probs(clf_probs_raw, groups, race_df)

        # Place probabilities
        if self.place_model is not None:
            _place_raw = self.place_model.predict_proba(X_scaled)[:, 1]
            _place_probs = self._calibrate_place_probs(_place_raw)
            # Adjust for actual EW places paid
            from src.each_way import adjust_place_probs_for_race, get_ew_terms as _get_ew
            _is_hcap = bool(race_df["handicap"].iloc[0]) if "handicap" in race_df.columns else False
            _ew_t = _get_ew(n, is_handicap=_is_hcap)
            _pp = _ew_t.places_paid if _ew_t.eligible else 3
            _place_probs = adjust_place_probs_for_race(_place_probs, win_probs, _pp)
        else:
            # Harville fallback
            places_paid = 3 if n >= 8 else (2 if n >= 5 else 0)
            if places_paid > 0 and n > places_paid:
                k = n / places_paid
                _place_probs = 1.0 - (1.0 - win_probs) ** k
                _place_probs = np.maximum(_place_probs, win_probs)
                _s = _place_probs.sum()
                if _s > 0:
                    _place_probs = np.clip(_place_probs * (places_paid / _s), 0.0, 1.0)
            else:
                _place_probs = np.zeros(n)

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
        results["rank_score"] = ltr_scores
        # Always sort by win_probability (value model). LTR rank_score is kept
        # in the output for reference but is no longer used for ordering.
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

    # ── SHAP explanation (uses LTR sub-model) ────────────────────
    def explain_race(
        self,
        race_df: pd.DataFrame,
        top_n_features: int = 10,
    ) -> dict:
        import shap

        if self.ltr_model is None:
            raise ValueError(
                "Model not trained. Call train() or load() first."
            )

        race_df = race_df.reset_index(drop=True)

        missing = [c for c in self.feature_cols if c not in race_df.columns]
        for col in missing:
            race_df[col] = 0

        X = race_df[self.feature_cols].values
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_cols,
            index=race_df.index,
        )

        explainer = shap.TreeExplainer(self.ltr_model)
        shap_values = explainer.shap_values(X_scaled)

        explanations: dict[str, pd.DataFrame] = {}
        horse_names = race_df["horse_name"].values

        for i, name in enumerate(horse_names):
            sv = shap_values[i]
            fv = X_scaled.iloc[i]
            df_expl = pd.DataFrame({
                "feature": self.feature_cols,
                "shap_value": sv,
                "feature_value": fv,
            })
            df_expl["abs_shap"] = df_expl["shap_value"].abs()
            df_expl = df_expl.sort_values("abs_shap", ascending=False)
            explanations[name] = df_expl.head(top_n_features).drop(
                columns="abs_shap",
            )

        return explanations

    # ── Persistence ──────────────────────────────────────────────
    def save(self):
        path = os.path.join(config.MODELS_DIR, "triple_ensemble_models.joblib")
        joblib.dump(
            {
                "ltr_model": self.ltr_model,
                "clf_model": self.clf_model,
                "place_model": self.place_model,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "calibration_temp": self.calibration_temp,
                "platt_a": self.platt_a,
                "platt_b": self.platt_b,
                "place_platt_a": self.place_platt_a,
                "place_platt_b": self.place_platt_b,
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
        self.ltr_model = data.get("ltr_model")
        self.clf_model = data.get("clf_model")
        self.place_model = data.get("place_model")
        self.scaler = data.get("scaler") or _IdentityScaler()
        self.feature_cols = data["feature_cols"]
        self.calibration_temp = float(data.get("calibration_temp", 1.0))
        self.platt_a = float(data.get("platt_a", 1.0))
        self.platt_b = float(data.get("platt_b", 0.0))
        self.place_platt_a = float(data.get("place_platt_a", 1.0))
        self.place_platt_b = float(data.get("place_platt_b", 0.0))
        self.frameworks = data.get("frameworks", {"ltr": "lgbm", "classifier": "cat", "place": "cat"})
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
        return_place_probs: bool = False,
        fast_fold: bool = False,
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
        y_rel = np.maximum(0, 11 - fp.astype(int))
        y_won = train_df["won"].fillna(0).values.astype(int)

        # Dynamic place target based on actual EW places paid per race
        _nr_fold = train_df["num_runners"].values
        _hc_fold = train_df.get("handicap", pd.Series(0, index=train_df.index)).values.astype(bool)
        _pp_fold = np.where(_nr_fold <= 4, 3, np.where(_nr_fold <= 7, 2, np.where(_nr_fold <= 15, 3, np.where(_hc_fold, 4, 3))))
        y_placed = (fp <= _pp_fold).astype(int)

        if "implied_prob" in train_df.columns:
            ip = train_df["implied_prob"].fillna(0.0).clip(0.0, 1.0).values.astype(np.float32)
        elif "odds" in train_df.columns:
            ip = (1.0 / train_df["odds"].replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0).values.astype(np.float32)
        else:
            ip = np.zeros(len(train_df), dtype=np.float32)

        nr = train_df["num_runners"].replace(0, 1).values.astype(np.float32)
        y_norm = 1.0 - (fp - 1) / np.maximum(nr - 1, 1)

        # Recency sample weights
        dates = pd.to_datetime(train_df["race_date"])
        days_ago = (dates.max() - dates).dt.days.values.astype(np.float64)
        _hl = getattr(config, "RECENCY_HALF_LIFE_DAYS", 180)
        sw = np.exp(-np.log(2) * days_ago / _hl)
        _seasonal = getattr(config, "RECENCY_SEASONAL_BOOST", 0.0)
        if _seasonal > 0:
            _cur_month = dates.max().month
            _same_month = (dates.dt.month.values == _cur_month).astype(np.float64)
            sw *= (1.0 + _seasonal * _same_month)
        sw /= sw.mean()

        # ── Internal calibration split (last 20 % of races) ─────
        n_races = len(groups_train)
        cal_bnd = max(1, int(n_races * 0.8))
        cum_g = np.cumsum(groups_train)
        cal_row_start = int(cum_g[cal_bnd - 1]) if cal_bnd > 0 else 0
        cal_groups = groups_train[cal_bnd:]

        if len(cal_groups) >= 5:
            # Train disposable models on first 80 % for calibration scores
            X_cal_tr = X_train[:cal_row_start]
            X_cal_vl = X_train[cal_row_start:]
            sw_cal = sw[:cal_row_start]
            fp_cal_tr = fp[:cal_row_start]
            ip_cal_tr = ip[:cal_row_start]
            y_norm_cal_tr = y_norm[:cal_row_start]
            y_placed_cal_tr = y_placed[:cal_row_start]
            g_cal_tr = groups_train[:cal_bnd]
            fp_cal_vl = fp[cal_row_start:]

            _cal_ltr = train_ranker(
                X_cal_tr,
                np.maximum(0, 11 - fp_cal_tr.astype(int)),
                g_cal_tr, self._ltr_ranker_type,
                finish_pos=fp_cal_tr, implied_prob=ip_cal_tr,
                feature_cols=self.feature_cols,
            )
            y_won_cal_tr = y_won[:cal_row_start]
            _n_pw = max(int(y_won_cal_tr.sum()), 1)
            _cal_clf = self._train_win_classifier(
                X_cal_tr, y_won_cal_tr,
                scale_pos_weight=(len(y_won_cal_tr) - _n_pw) / _n_pw,
                sample_weight=sw_cal,
            )
            _n_pp = max(int(y_placed_cal_tr.sum()), 1)
            _cal_place = self._train_place_classifier(
                X_cal_tr, y_placed_cal_tr,
                scale_pos_weight=(len(y_placed_cal_tr) - _n_pp) / _n_pp,
                sample_weight=sw_cal,
            )

            X_cal_vl_df = pd.DataFrame(X_cal_vl, columns=feature_cols)
            _cal_clf_probs = _cal_clf.predict_proba(X_cal_vl_df)[:, 1]
            _cal_place_raw = _cal_place.predict_proba(X_cal_vl_df)[:, 1]
            # Dynamic place target for calibration slab
            _nr_cal = train_df["num_runners"].values[cal_row_start:]
            _hc_cal = train_df.get("handicap", pd.Series(0, index=train_df.index)).values[cal_row_start:].astype(bool)
            _pp_cal = np.where(_nr_cal <= 4, 3, np.where(_nr_cal <= 7, 2, np.where(_nr_cal <= 15, 3, np.where(_hc_cal, 4, 3))))
            _cal_placed_vl = (fp_cal_vl <= _pp_cal).astype(int)

            # Fit calibration on held-out slab
            self.calibration_temp = 1.0  # unused — kept for compat
            self.platt_a, self.platt_b = self._optimise_platt_calibration(
                _cal_clf_probs, cal_groups, fp_cal_vl,
            )
            self.place_platt_a, self.place_platt_b = self._optimise_place_platt(
                _cal_place_raw, _cal_placed_vl,
            )
        else:
            # Too few races — skip calibration
            self.calibration_temp = 1.0
            self.platt_a, self.platt_b = 1.0, 0.0
            self.place_platt_a, self.place_platt_b = 1.0, 0.0

        # ── Train final models on ALL training data ──────────────
        self.ltr_model = train_ranker(
            X_train, y_rel, groups_train, self._ltr_ranker_type,
            finish_pos=fp, implied_prob=ip,
            feature_cols=self.feature_cols,
        )
        n_pw = max(int(y_won.sum()), 1)
        self.clf_model = self._train_win_classifier(
            X_train, y_won,
            scale_pos_weight=(len(y_won) - n_pw) / n_pw,
            sample_weight=sw,
        )
        n_pp = max(int(y_placed.sum()), 1)
        self.place_model = self._train_place_classifier(
            X_train, y_placed,
            scale_pos_weight=(len(y_placed) - n_pp) / n_pp,
            sample_weight=sw,
        )

        # ── Score test set with calibration ──────────────────────
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        ltr_scores = self.ltr_model.predict(X_test_df)
        clf_probs_raw = self.clf_model.predict_proba(X_test_df)[:, 1]
        win_probs = self._calibrate_win_probs(clf_probs_raw, groups_test)
        place_probs_raw = self.place_model.predict_proba(X_test_df)[:, 1]
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
            return ltr_scores, win_probs, place_probs
        return ltr_scores
