"""
Configuration settings for the Horse Racing Prediction Application.
"""

import os

# Load a local .env (Matchbook creds etc.) so os.getenv below sees it. No-op if
# python-dotenv isn't installed or there's no .env. Real env vars still win.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

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
LIVE_FEATURE_CACHE_VERSION = 2

# Date-stamped cache entries (live feature cache, lookahead cache,
# racecards cache) older than this are deleted on app start-up.
CACHE_TTL_DAYS = 30

# --- Matchbook API Settings ---
MATCHBOOK_EDGE_URL = os.getenv("MATCHBOOK_EDGE_URL", "https://api.matchbook.com/edge/rest")
MATCHBOOK_BPAPI_URL = os.getenv("MATCHBOOK_BPAPI_URL", "https://api.matchbook.com/bpapi/rest")
MATCHBOOK_TIMEOUT_SECS = float(os.getenv("MATCHBOOK_TIMEOUT_SECS", "15"))
MATCHBOOK_HORSE_RACING_SPORT_ID = int(os.getenv("MATCHBOOK_HORSE_RACING_SPORT_ID", "24735152712200"))
MATCHBOOK_DEFAULT_CURRENCY = os.getenv("MATCHBOOK_DEFAULT_CURRENCY", "GBP")
MATCHBOOK_USERNAME = os.getenv("MATCHBOOK_USERNAME", "")
MATCHBOOK_PASSWORD = os.getenv("MATCHBOOK_PASSWORD", "")

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

# --- Sub-Model Framework Selection ---
# For each sub-model role, choose "xgb" (XGBoost), "lgbm" (LightGBM),
# or "cat" (CatBoost).
SUB_MODEL_FRAMEWORKS: dict[str, str] = {
    "classifier": "lgbm",  # Win classifier
    "place": "lgbm",       # Place classifier
}

# Win-probability calibration for newly trained models. The temperature option
# improved chronological audit scores, but remains opt-in until confirmed on a
# fully rebuilt feature store. Saved models preserve their fitted method.
# Values: "platt_isotonic", "race_temperature".
WIN_CALIBRATION_METHOD = "platt_isotonic"

# --- Market Anchor (Benter combination) ---
# After Platt/isotonic calibration, combine the win classifier with the
# market via softmax(alpha*log(p_model) + beta*log(p_market)) per race,
# with (alpha, beta) fitted by maximum likelihood on OOF predictions.
# beta > 1 also corrects the favourite-longshot bias. Races without
# complete odds fall back to the unanchored probabilities.
MARKET_ANCHOR = False

# --- Data-quality betting gates (src/quality_control.py) ---
# Pre-bet gates that SUPPRESS bets where the model's inputs are unreliable (a
# distinct concern from value/edge rules, which live in src/bet_settlement.py).
# Surfaced + tunable in the Picks app's Quality Control panel, so the rules stay
# visible and auditable rather than buried. Keys mirror QC_DEFAULTS in the
# module; anything set here overrides the module defaults.
#   thin_history: a foreign raider (e.g. a French horse at Cheltenham) has no
#   form in our DB, so the model under-rates it and the whole race is mispriced.
#   Block races containing a short-priced runner with too little DB history.
# enabled=False: a 2026-06-22 walk-forward-style quantification showed the
# thin-history gate does NOT improve betting results on the current (form-only,
# no-edge) model — it hurt top-pick ROI in every threshold tested and only
# marginally helped value bets. The real accuracy flaw it targets is not a
# betting flaw (the model already declines to back the horses it under-rates).
# Kept as documented, tunable infrastructure; turn on in the panel to use it.
QC_DEFAULTS = {
    "enabled": False,
    "thin_history_contender_max_odds": 8.0,   # "contender" = SP at/below this
    "thin_history_min_prev_races": 1,         # "thin" = <= this many prior DB runs
}

# --- Win-model market-probability distillation ---
# When True, the win classifier trains on the market-implied win
# probability (a low-noise teacher) via cross-entropy instead of the 0/1
# outcome, which improves calibration/log-loss. Odds are used only as the
# training *target* — never an inference feature, so the model stays
# odds-free at predict time. LGBM classifier only.
# WIN_SOFT_LABEL_BLEND: target = blend*market_prob + (1-blend)*won
#   1.0 = pure probability target; 0.5 = even blend (best-calibrated in tests).
# Validated gain on the full dataset was small (~-0.0013 log-loss) and
# improves accuracy only — it does NOT create a betting edge.
WIN_SOFT_LABEL_DISTILL = False
WIN_SOFT_LABEL_BLEND = 1.0

# --- Glicko Settings ---
GLICKO_ENABLED = True    # adds horse_glicko* features in feature engineering
GLICKO_RD_INIT = 350.0   # uncertainty for an unraced horse
GLICKO_RD_MIN = 50.0     # floor for well-evidenced horses
GLICKO_C = 70.0          # RD inflation per sqrt(month) idle
GLICKO_JOCKEY_TRAINER = False  # also rate jockeys/trainers (sweep 2026-06-11: hurt, keep off)
GLICKO_DIMENSIONAL = False     # per surface/race-type/dist-cat ratings (sweep 2026-06-11: hurt, keep off)
GLICKO_MARGIN = False          # beaten-lengths Glicko scores (sweep 2026-06-12: hurt, keep off)
GLICKO2_ENABLED = False        # Glicko-2 volatility (sweep 2026-06-12: neutral, keep off)
GLICKO2_TAU = 0.5              # volatility-change constraint
TRUESKILL_ENABLED = True       # TrueSkill mu/sigma (sweep 2026-06-12: best candidate, on)

# --- Collateral Form ---
# Prev-race rivals' next-out results ("key race" signal).
# Sweep 2026-06-12: unstable across folds (0.714/0.667), keep off.
COLLATERAL_FORM = False

# --- Elo Settings ---
ELO_K_BASE = 16.0   # K for horses with 0 prior runs (walk-forward swept 2026-06-11)
ELO_K_MIN = 8.0     # K floor for experienced horses
ELO_K_DECAY = 0.05  # exponential decay rate per run
MARGIN_ELO_SCALE = 3.0          # base lengths at which margin score ≈ 0.82 (walk-forward swept 2026-06-11)
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
# Held at 0.0 because FEATURE_ALLOWLIST below already fixes the feature set
# (top-150 by ablation); an extra per-run prune would re-cut it nondeterministically.
FEATURE_PRUNE_FRACTION = 0.0
# Absolute Pearson correlation above which the less-important feature
# in a pair is dropped (before importance pruning). 0.0 = disabled.
FEATURE_CORR_THRESHOLD = 0.95

# --- Feature allow-list ---
# When set, the model trains on ONLY these features (intersected with the
# numeric, non-excluded columns); everything else stays in the feature store
# but never enters the model. This makes a deterministic, reproducible cut —
# unlike FEATURE_PRUNE_FRACTION, which re-derives a different bottom-X% each
# run from a pilot model. None = use all features.
#
# Accepts a list of names or a path to a file (one name per line, '#'
# comments allowed, or a JSON list). Generate one from the ablation knee:
#   python scripts/model_lab.py --write-allowlist <K>
# then point this at the written file. Applied AFTER it, FEATURE_PRUNE_FRACTION
# still trims within the allow-list (set it to 0.0 to keep exactly the list).
# Versioned selection: fresh checkouts must use the same features as research
# runs. New experimental lists can still be supplied here explicitly.
FEATURE_ALLOWLIST = os.path.join(BASE_DIR, "feature_sets", "win.txt")

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

# --- Golden hold-out (pre-registered 2026-10-13) ---
# Races on or after this date are the hold-out ACTION_PLAN 7 requires: no
# research may tune, select or threshold on them, and a candidate is scored on
# them once. Chosen because no CLV or market-residual analysis in this project
# has ever seen them: every such harness drops rows without a Betfair price,
# and the hand-downloaded Betfair data stopped at 2026-06-10. Development folds
# must end PURGE_DAYS before it (src.wf_eval.development_mask). Do not move it
# later to reuse data after looking at a result.
GOLDEN_HOLDOUT_START = "2026-06-11"
# BURN_IN_MONTHS: strip the first N months of the training window from the
# training split.  Feature engineering has already run on the full history
# so cumulative stats are correct, but these earliest rows have cold-start
# rolling counts (near-zero horse_prev_races, unwarmed Elo) that produce
# noisy gradients without contributing useful pattern learning.
# Set to 0 to disable.
# 8 chosen from scripts/feature_audit.py: the debut share (first-seen horses)
# falls from ~91% at 2020-09 to its ~8% steady floor by ~2021-05 and stays
# there; was 4, which left several months of still-warming form features.
BURN_IN_MONTHS = 8
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
TE_EWMA_HALF_LIFE_RACES = 20  # walk-forward swept 2026-06-11 (was 10)
# CV_N_FOLDS: purged expanding-window folds for Phase 1 OOF predictions,
# used to fit the Platt/isotonic calibrators. More folds = more OOF data
# for calibration, but slower.
CV_N_FOLDS = 3



