"""
Configuration settings for the Horse Racing Prediction Application.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Data Collection Settings ---
# Number of historical races to collect (more = better model, slower collection)
MAX_RACES_TO_COLLECT = 2000

# User agent for web requests
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Request delay to be respectful to servers (seconds)
REQUEST_DELAY = 2.0

# --- Feature Engineering Settings ---
# Number of recent races to consider for form calculation
FORM_WINDOW = 5

# Rolling average windows
ROLLING_WINDOWS = [3, 5, 10, 20]

# When computing features for today's picks the pipeline prepends the full
# processed history so cumulative stats (win rates, Elo, speed figures …) are
# correct.  Loading and feature-engineering hundreds of thousands of historical
# rows is slow, but the largest rolling window is only 20 races, so anything
# older than LIVE_FE_HISTORY_MONTHS contributes nothing to live predictions.
# Increase if you notice stale-looking Elo/form values; decrease to go faster.
LIVE_FE_HISTORY_MONTHS = 30
LIVE_FEATURE_CACHE_VERSION = 1

# --- Model Settings ---
MODEL_FILE = os.path.join(MODELS_DIR, "horse_race_model.joblib")
SCALER_FILE = os.path.join(MODELS_DIR, "feature_scaler.joblib")
FEATURE_COLUMNS_FILE = os.path.join(MODELS_DIR, "feature_columns.joblib")

# Test set size
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# --- Classifier Hyperparameters ---
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# --- Per-Sub-Model Hyperparameters (Triple Ensemble) ---
# Each sub-model can have independent tuning.
# These override XGBOOST_PARAMS / LIGHTGBM_PARAMS where set.

CLASSIFIER_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.008,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 3,
    "min_child_samples": 50,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "num_leaves": 35,
}

PLACE_CLASSIFIER_PARAMS = {
    "n_estimators": 3000,
    "max_depth": 7,
    "learning_rate": 0.006,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "min_child_samples": 30,
    "reg_alpha": 0.4,
    "reg_lambda": 3.0,
}

# --- Softmax Temperature ---
# Controls sharpness of probability output: <1 = sharper, >1 = softer
SOFTMAX_TEMPERATURE = 1.0

# --- Sub-Model Framework Selection ---
# For each sub-model role, choose "xgb" (XGBoost), "lgbm" (LightGBM),
# or "cat" (CatBoost).
SUB_MODEL_FRAMEWORKS: dict[str, str] = {
    "classifier": "lgbm",  # Win classifier
    "place": "lgbm",       # Place classifier
}

# --- Elo Settings ---
ELO_K_BASE = 32.0   # K for horses with 0 prior runs
ELO_K_MIN = 8.0     # K floor for experienced horses
ELO_K_DECAY = 0.05  # exponential decay rate per run
MARGIN_ELO_SCALE = 5.0          # base lengths at which margin score ≈ 0.82
MARGIN_ELO_REF_DIST = 8.0       # reference distance (furlongs) for scale normalisation
MARGIN_ELO_DNF_PENALTY = 30.0   # virtual lb for non-finishers

# --- Recency Weighting ---
# Half-life in days for exponential recency sample weights.
# 180 = 6-month half-life (form is fleeting in racing).
RECENCY_HALF_LIFE_DAYS = 180
# Seasonal boost: uplift for training data from the same calendar month.
# Racing patterns (going, fitness cycles) repeat annually.
RECENCY_SEASONAL_BOOST = 0.15   # 15 % uplift for same-month data
# Decay shape: "exp" (exponential), "poly" (polynomial), or "linear".
# "exp"    — w = exp(-ln2 * t / half_life)
# "poly"   — w = 1 / (1 + t / half_life)            (heavier tail than exp)
# "linear" — w = max(0, 1 - t / (2 * half_life))   (hard cutoff)
RECENCY_DECAY_SHAPE = "exp"

# --- Focal Loss (Win & Place classifiers) ---
FOCAL_GAMMA = 2.0    # focusing parameter: higher → harder focus on ambiguous cases
FOCAL_ALPHA = "auto" # positive-class weight — "auto" computes from actual prevalence

# --- Feature Pruning ---
# Fraction of lowest-importance features to drop (0.0 = keep all).
FEATURE_PRUNE_FRACTION = 0.2
# Absolute Pearson correlation above which the less-important feature
# in a pair is dropped (before importance pruning). 0.0 = disabled.
FEATURE_CORR_THRESHOLD = 0.95

# --- Early Stopping ---
# Number of boosting rounds without improvement before stopping.
# 0 = disabled (train for full n_estimators on all training data).
# When enabled, Phase 2 holds out the last ~10 % of training data
# for validation — disabling lets the model see all data.
EARLY_STOPPING_ROUNDS = 80

# --- Purged Cross-Validation ---
# PURGE_DAYS: number of days to remove between train/test boundaries
# to prevent feature leakage from overlapping horse form.
PURGE_DAYS = 7
# BURN_IN_MONTHS: strip the first N months of the training window from the
# training split.  Feature engineering has already run on the full history
# so cumulative stats are correct, but these earliest rows have cold-start
# rolling counts (near-zero horse_prev_races, unwarmed Elo) that produce
# noisy gradients without contributing useful pattern learning.
# Set to 0 to disable.
BURN_IN_MONTHS = 4
# TE_WINDOW_DAYS: calendar window (days) for the additional windowed
# target-encoding features.  Alongside the full cumulative encodings,
# windowed encodings capture RECENT entity form — important under concept
# drift where career averages no longer reflect current-form baselines.
# Set to 0 to disable.
TE_WINDOW_DAYS = 365
# TE_EWMA_HALF_LIFE_RACES: half-life (in races) for the EWMA target encoding.
# ~10 races ≈ 90 days for a horse that runs roughly monthly; this gives a
# smooth exponential decay toward recent form without calendar cliff-edges.
# Set to 0 to disable.
TE_EWMA_HALF_LIFE_RACES = 10
# CV_N_FOLDS: expanding-window folds for Phase 1 meta-learner training.
# More folds = more OOF data for the meta-learner, but slower.
CV_N_FOLDS = 3
# META_LEARNER_INFERENCE: if True, the LGBM meta-learner is used directly at
# inference time. If False (default + recommended), the meta-learner is used
# only to discover optimal sub-model weights; inference then uses those weights
# via the linear blend path. This avoids the OOF distribution-shift problem
# (meta trained on weak fold-models, evaluated on stronger full-data models).
META_LEARNER_INFERENCE = False

# --- Ensemble Safety ---
# If True, when only a small number of sub-models are enabled,
# auto mode will select the best OOF single model instead of stacking.
ENSEMBLE_FORCE_BEST_FOR_SMALL_SETS = True
ENSEMBLE_SMALL_SET_MAX_MODELS = 3
# Additional guard for larger sets: if blend RPS is worse than best
# single-model RPS by this fraction (and top-1 is not better), fallback.
ENSEMBLE_GUARD_TOL = 0.0
# Number of Optuna trials for OOF ensemble weight optimisation.
WEIGHT_OPTUNA_TRIALS = 200

# --- Monotonic Constraints ---
# Features that should be monotone-positive in LightGBM classifiers.
# Higher values of these features should (all else equal) produce higher
# predicted scores — enforcing domain knowledge and preventing physically
# impossible inversions.
MONOTONE_FEATURES: list[str] = [
    "horse_elo",
    "official_rating",
    "jockey_elo",
    "horse_win_rate_shrunk",
    "trainer_win_rate_shrunk",
    "horse_track_wr_shrunk",
]


