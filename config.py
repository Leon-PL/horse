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
# User agent for web requests
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Request delay to be respectful to servers (seconds)
REQUEST_DELAY = 2.0

# --- Feature Engineering Settings ---
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
# racecards cache) older than this are deleted by
# src.utils.cleanup_stale_caches.
CACHE_TTL_DAYS = 30

# Test set size
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# --- Classifier Hyperparameters ---
# Win (CLASSIFIER_PARAMS) and place (PLACE_CLASSIFIER_PARAMS) models, each
# tuned independently.

CLASSIFIER_PARAMS = {
    "n_estimators": 4000,
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
# LightGBM ("lgbm") is the only supported framework; the per-role mapping is
# kept because saved runs and training recipes record it.
SUB_MODEL_FRAMEWORKS: dict[str, str] = {
    "classifier": "lgbm",  # Win classifier
    "place": "lgbm",       # Place classifier
}

# Win-model loss (experiment 057, docs/MODEL_IDEAS_057_060.md).
#   "exploded": a race softmax over the winner, then second from the runners
#     left, then third (src/race_objectives.py), with stage weights
#     WIN_EXPLODED_STAGE_WEIGHTS chosen on development quarters. Confirmed at
#     -0.0046 race log loss against "cross_entropy", 4 of 4 quarters.
#   "cross_entropy": the per-runner soft-label learner used until 057.
# The exploded model scores runners only relative to their race, so it needs
# "race_temperature" calibration.
WIN_OBJECTIVE = "exploded"
WIN_EXPLODED_STAGE_WEIGHTS = (1.0, 0.25, 0.0625)

# Win-probability calibration for newly trained models. Saved models preserve
# their fitted method. "race_temperature" is required by the exploded win
# objective; "platt_isotonic" was the default for the cross-entropy learner.
# Values: "platt_isotonic", "race_temperature".
WIN_CALIBRATION_METHOD = "race_temperature"
#038's qualified place-only refinement. Saved older models retain their exact
#legacy probabilities; normal future fits learn this layer from earlier data.
RECENT_PLACE_REFINEMENT = True

# --- Market Anchor (Benter combination) ---
# After Platt/isotonic calibration, combine the win classifier with the
# market via softmax(alpha*log(p_model) + beta*log(p_market)) per race,
# with (alpha, beta) fitted by maximum likelihood on OOF predictions.
# beta > 1 also corrects the favourite-longshot bias. Races without
# complete odds fall back to the unanchored probabilities.
MARKET_ANCHOR = False

# --- Win-model market-probability distillation ---
# When True, the win classifier trains on the market-implied win
# probability (a low-noise teacher) via cross-entropy instead of the 0/1
# outcome. Its effect requires a paired outcome-based comparison. Odds are used only as the
# training *target* — never an inference feature, so the model stays
# odds-free at predict time. LGBM classifier only.
# WIN_SOFT_LABEL_BLEND: target = blend*market_prob + (1-blend)*won
#   1.0 = pure probability target; 0.5 = even blend.
# Earlier gain comments had no recoverable equivalent native audit receipt.
# Corrected targets require a complete BSP field, else a complete SP field.
WIN_SOFT_LABEL_DISTILL = True
WIN_SOFT_LABEL_BLEND = 0.5

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
# Experiment 045 (scripts/screen_ratings.py, fit-period within-race log
# loss): K applies to the mean residual over opponents, so it is far larger
# than the old 16/8, which was swept on 2026-06-11, before the leak fixes.
ELO_K_BASE = 512.0  # K for horses with 0 prior runs
ELO_K_MIN = 128.0   # K floor for experienced horses
ELO_K_DECAY = 0.05  # exponential decay rate per run
ELO_ENTITY_K_BASE = 32.0   # value-added jockey/trainer offsets (16/2 and 64/8 worse)
ELO_ENTITY_K_MIN = 4.0
ELO_MOMENTUM_HALF_LIFE_DAYS = 30.0  # 365..30 improves slowly; flat below 30
RATING_DIM_PRIOR_RUNS = 3.0  # shrinkage for surface/race-type/distance edges (1..10 flat)
RATING_DNF_AS_LOSS = True    # non-finishers rank tied last
GLICKO_EVIDENCE_POWER = 0.25  # d² and scores weighted by n_opponents^-p (0 worse)
TRUESKILL_C = 2.0  # sigma inflation per month idle (0 → 2: 2.047 → 2.010 fit LL; 3+ worse)
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

# --- Feature Pruning ---
# Fraction of lowest-importance features to drop (0.0 = keep all).
# Held at 0.0 because FEATURE_ALLOWLIST below already fixes the feature set
# (walk-forward selection, experiment 048); an extra per-run prune would re-cut
# it nondeterministically.
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
# comments allowed, or a JSON list). win_selected_20261112.txt holds the 87
# inputs of the live win model (run 20261114_122027_967139, which the Matchbook
# paper record serves). Earlier lists are in git at tag pre-cleanup-2026-11.
# New selections come from `python -m src.modelling prepare` and `select`
# (docs/operations/MODELLING.md).
FEATURE_ALLOWLIST = os.path.join(BASE_DIR, "feature_sets", "win_selected_20261112.txt")

# BHA retrospective feature availability: an explicit assumption, not a claim
# that the original publication/revision timestamp was captured. Raw source
# capture times are retained unchanged. "recorded" restores capture-only use.
BHA_FEATURE_AVAILABILITY_POLICY = "race_plus_48h"

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



