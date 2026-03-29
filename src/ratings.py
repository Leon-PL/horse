"""
Horse Rating System (Elo-style)
================================
Computes dynamic Elo ratings for horses, jockeys, and trainers based on
head-to-head race results.

The key insight: if Horse A beats Horse B, and Horse B beats Horse C,
then Horse A should be rated above Horse C — *even if they never met*.
Traditional per-horse win-rate features cannot capture this transitive
strength.  Elo ratings propagate strength through the entire population.

Algorithm
---------
After each race, every pair of finishers is compared:

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))   # expected score for A vs B
    S_A = 1 if A beat B, 0.5 if tied, 0 if lost

    R_A_new = R_A + K * (S_A - E_A)

With N runners in a race, each horse is updated against all N-1 opponents.
The K-factor is divided by (N-1) to normalise the update magnitude so a
big-field race doesn't cause wild swings.

Usage::

    from src.ratings import compute_elo_features

    df = compute_elo_features(df)
    # Adds: horse_elo, jockey_elo, trainer_elo (ratings at the *start*
    # of each race, before that race's update).
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _off_time_to_seconds(off_time: pd.Series) -> pd.Series:
    """Parse off-time strings to seconds after midnight with safe fallbacks."""
    s = off_time.astype(str).str.strip()
    parts = s.str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?")
    h = pd.to_numeric(parts["h"], errors="coerce")
    m = pd.to_numeric(parts["m"], errors="coerce")
    sec = pd.to_numeric(parts["s"], errors="coerce").fillna(0)

    bad = h.isna() | m.isna()
    if bad.any():
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


def _event_sort_key(df: pd.DataFrame) -> pd.Series:
    """Build event-time key using race_date + off_time when available."""
    race_dt = pd.to_datetime(df["race_date"], errors="coerce")
    if "off_time" not in df.columns:
        return race_dt
    off_secs = _off_time_to_seconds(df["off_time"]).astype(float)
    return race_dt + pd.to_timedelta(off_secs, unit="s")


def _entity_keys(
    df: pd.DataFrame,
    *,
    id_col: str | None,
    name_col: str,
    prefix: str,
) -> np.ndarray:
    """Return stable entity keys, preferring IDs and falling back to names."""
    base = pd.Series(pd.NA, index=df.index, dtype="object")
    if name_col in df.columns:
        base = df[name_col].astype("string").where(df[name_col].notna(), pd.NA).astype("object")
    if id_col and id_col in df.columns and df[id_col].notna().any():
        id_vals = df[id_col].astype("string").where(df[id_col].notna(), pd.NA).astype("object")
        base = id_vals.where(id_vals.notna(), base)
    return pd.Series(base, index=df.index).map(
        lambda value: f"{prefix}:{value}" if pd.notna(value) else pd.NA
    ).to_numpy(dtype=object)

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_RATING = 1500.0
K_FACTOR = 16.0          # legacy fallback (per-opponent K)
MIN_RACES_FOR_RATING = 0  # include from the first race

# Adaptive K-factor settings (from config, with fallbacks)
K_BASE = getattr(config, "ELO_K_BASE", 32.0)
K_MIN = getattr(config, "ELO_K_MIN", 8.0)
K_DECAY = getattr(config, "ELO_K_DECAY", 0.05)
MOMENTUM_ALPHA = getattr(config, "ELO_MOMENTUM_ALPHA", 0.3)  # EMA smoothing for momentum


def _adaptive_k(n_races: int) -> float:
    """
    Compute an adaptive K-factor based on experience.

    Young / unexposed horses should have a higher K (their rating changes
    more with each result), while established horses need a lower K.

    Uses exponential decay:  K = K_MIN + (K_BASE - K_MIN) * exp(-K_DECAY * n)
    """
    return K_MIN + (K_BASE - K_MIN) * np.exp(-K_DECAY * n_races)


# ── Margin Elo config ─────────────────────────────────────────────────────
MARGIN_SCALE = getattr(config, "MARGIN_ELO_SCALE", 5.0)     # base lengths at which score ≈ 0.82
MARGIN_REF_DIST = getattr(config, "MARGIN_ELO_REF_DIST", 8.0)  # reference distance (furlongs)
DNF_PENALTY_LB = getattr(config, "MARGIN_ELO_DNF_PENALTY", 30.0)  # virtual lb for non-finishers

# Non-finisher labels: horses that started but didn't complete
_DNF_LABELS = frozenset({"PU", "F", "UR", "BD", "RR", "SU", "RO", "CO", "REF", "DSQ"})


# ── Core Elo helpers ──────────────────────────────────────────────────────

def _expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B given their ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_ratings_for_race(
    finish_order: list[tuple[str, int]],
    ratings: dict[str, float],
    k: float = K_FACTOR,
    race_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Update Elo ratings for all runners in a single race.

    Args:
        finish_order: list of (entity_id, finish_position) — position 1 is
                      best.  Entities with the same position are treated as
                      a draw.
        ratings: mutable dict of entity_id → current rating.
        k: K-factor per opponent pair (divided by field-1 internally).
            Used as fallback when race_counts is None.
        race_counts: Optional dict of entity_id → number of prior races.
            When provided, an adaptive K is computed per entity.

    Returns:
        dict mapping entity_id → rating change (delta) for this race.
    """
    n = len(finish_order)
    if n < 2:
        return {}

    deltas: dict[str, float] = defaultdict(float)

    for i, (id_a, pos_a) in enumerate(finish_order):
        r_a = ratings.get(id_a, DEFAULT_RATING)
        # Adaptive K: use entity's experience if available
        if race_counts is not None:
            k_entity = _adaptive_k(race_counts.get(id_a, 0))
        else:
            k_entity = k
        # Normalise K by number of opponents
        k_adj = k_entity / (n - 1)

        for j, (id_b, pos_b) in enumerate(finish_order):
            if i == j:
                continue
            r_b = ratings.get(id_b, DEFAULT_RATING)

            expected = _expected_score(r_a, r_b)

            if pos_a < pos_b:
                actual = 1.0
            elif pos_a == pos_b:
                actual = 0.5
            else:
                actual = 0.0

            deltas[id_a] += k_adj * (actual - expected)

    # Apply deltas
    for entity_id, delta in deltas.items():
        ratings[entity_id] = ratings.get(entity_id, DEFAULT_RATING) + delta

    return dict(deltas)


def _margin_actual_score(
    lb_a: float, lb_b: float, margin_scale: float = MARGIN_SCALE,
) -> float:
    """
    Continuous actual score for A vs B based on lengths-behind margin.

    Returns a value in (0, 1):
        - A beats B (lb_a < lb_b): score > 0.5, approaching 1.0 for big margins
        - A loses to B (lb_a > lb_b): score < 0.5, approaching 0.0 for big margins
        - Dead heat (lb_a == lb_b): exactly 0.5

    Uses ``1 - exp(-margin / margin_scale)`` to map the gap size
    to a [0, 0.5) bonus/penalty on top of the 0.5 baseline.

    *margin_scale* is distance-conditioned: longer races use a larger
    scale so that the same number of lengths produces a smaller Elo
    swing than in a sprint.
    """
    margin = lb_b - lb_a  # positive when A beat B
    if margin == 0.0:
        return 0.5
    sign = 1.0 if margin > 0 else -1.0
    mov = 0.5 * (1.0 - np.exp(-abs(margin) / margin_scale))
    return 0.5 + sign * mov


def _distance_margin_scale(distance_furlongs: float) -> float:
    """Return a margin scale proportional to race distance.

    scale = MARGIN_SCALE * (distance / REF_DIST)

    At REF_DIST (default 8f ≈ 1 mile) the scale equals the base
    MARGIN_SCALE.  A 5f sprint gets a tighter scale (3.1) so a
    1-length gap has *more* Elo impact.  A 24f chase gets a
    wider scale (15.0) reflecting that large lb margins are
    routine at staying distances.
    """
    if not np.isfinite(distance_furlongs) or distance_furlongs <= 0:
        return MARGIN_SCALE
    return MARGIN_SCALE * (distance_furlongs / MARGIN_REF_DIST)


def _update_margin_elo_for_race(
    runners: list[tuple[str, float]],
    ratings: dict[str, float],
    k: float = K_FACTOR,
    race_counts: dict[str, int] | None = None,
    margin_scale: float = MARGIN_SCALE,
) -> dict[str, float]:
    """
    Margin-weighted Elo update for one race.

    Like ``_update_ratings_for_race`` but uses continuous scores derived
    from the lengths-behind gap between each pair, rather than binary
    win/loss.

    Args:
        runners: list of (entity_id, effective_lengths_behind).
                 Non-finishers should already have a penalised lb value.
        ratings: mutable dict of entity_id → current rating.
        k: K-factor fallback.
        race_counts: optional entity_id → prior-race count for adaptive K.
        margin_scale: distance-conditioned scale for the margin → score
                      mapping (see ``_distance_margin_scale``).

    Returns:
        dict mapping entity_id → rating delta for this race.
    """
    n = len(runners)
    if n < 2:
        return {}

    deltas: dict[str, float] = defaultdict(float)

    for i, (id_a, lb_a) in enumerate(runners):
        r_a = ratings.get(id_a, DEFAULT_RATING)
        if race_counts is not None:
            k_entity = _adaptive_k(race_counts.get(id_a, 0))
        else:
            k_entity = k
        k_adj = k_entity / (n - 1)

        for j, (id_b, lb_b) in enumerate(runners):
            if i == j:
                continue
            r_b = ratings.get(id_b, DEFAULT_RATING)
            expected = _expected_score(r_a, r_b)
            actual = _margin_actual_score(lb_a, lb_b, margin_scale)
            deltas[id_a] += k_adj * (actual - expected)

    for entity_id, delta in deltas.items():
        ratings[entity_id] = ratings.get(entity_id, DEFAULT_RATING) + delta

    return dict(deltas)


# ── Public API ────────────────────────────────────────────────────────────

def compute_elo_features(
    df: pd.DataFrame,
    k: float = K_FACTOR,
) -> pd.DataFrame:
    """
    Add Elo rating features to the DataFrame.

    The function processes races **chronologically** and records each
    entity's rating *before* the race takes place (i.e. the rating the
    entity carried into the race).  This avoids any look-ahead leakage.

    New columns added:
        - ``horse_elo``   — horse's Elo rating entering this race
        - ``jockey_elo``  — jockey's rating entering this race
        - ``trainer_elo`` — trainer's rating entering this race
        - ``horse_elo_delta``  — rating change after this race
        - ``jockey_elo_delta`` — jockey rating change after this race

    Args:
        df: DataFrame with ``race_id``, ``race_date``, ``finish_position``,
            ``horse_name``, and optionally ``jockey`` and ``trainer``.
        k: K-factor (higher = more reactive).

    Returns:
        DataFrame with new Elo columns.
    """
    logger.info("Computing Elo ratings (horse, jockey, trainer)...")
    df["race_date"] = pd.to_datetime(df["race_date"])

    # The outer feature-engineering pipeline already sorts by _event_dt + race_id.
    # Reuse that order when available to avoid an extra full-frame sort/copy.
    if "_event_dt" in df.columns:
        event_dt = pd.to_datetime(df["_event_dt"], errors="coerce")
    else:
        event_dt = _event_sort_key(df)
        df["_event_dt"] = event_dt

    if not event_dt.is_monotonic_increasing:
        df.sort_values(["_event_dt", "race_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    has_jockey = "jockey" in df.columns
    has_trainer = "trainer" in df.columns

    # Rating dictionaries — persist across all races
    horse_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    jockey_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    trainer_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)

    # Race-count dictionaries for adaptive K-factor
    horse_race_counts: dict[str, int] = defaultdict(int)
    jockey_race_counts: dict[str, int] = defaultdict(int)
    trainer_race_counts: dict[str, int] = defaultdict(int)

    # Columns to populate
    horse_elo_col = np.full(len(df), np.nan)
    jockey_elo_col = np.full(len(df), np.nan)
    trainer_elo_col = np.full(len(df), np.nan)
    horse_delta_col = np.zeros(len(df))
    jockey_delta_col = np.zeros(len(df))

    # Momentum Elo: exponential moving average of recent deltas
    horse_momentum: dict[str, float] = defaultdict(float)
    horse_momentum_col = np.full(len(df), np.nan)

    # Group by race (preserving chronological order)
    # Pre-group indices: O(N) once instead of O(N) per-race boolean scan.
    race_ids_ordered = df.drop_duplicates("race_id", keep="first")["race_id"].values
    _race_groups = df.groupby("race_id", sort=False)

    # Pre-extract columns as numpy arrays for fast inner-loop access
    _horse_keys = _entity_keys(df, id_col="horse_id", name_col="horse_name", prefix="horse")
    _finish_pos = df["finish_position"].values if "finish_position" in df.columns else None
    _jockey_keys = _entity_keys(df, id_col="jockey_id", name_col="jockey", prefix="jockey") if has_jockey else None
    _trainer_keys = _entity_keys(df, id_col="trainer_id", name_col="trainer", prefix="trainer") if has_trainer else None

    # ── Dimensional Elo ──────────────────────────────────────────────
    # Separate ratings per surface, race_type, and distance category.
    # Falls back to global Elo when a horse has < MIN_DIM_RACES in
    # the dimension (cold-start protection).
    MIN_DIM_RACES = 3

    has_surface = "surface" in df.columns
    has_race_type = "race_type" in df.columns
    has_dist_cat = "dist_category" in df.columns

    # Rating dicts: key = (entity_name, dim_value)
    horse_surf_ratings: dict = defaultdict(lambda: DEFAULT_RATING)
    horse_rt_ratings: dict = defaultdict(lambda: DEFAULT_RATING)
    horse_dc_ratings: dict = defaultdict(lambda: DEFAULT_RATING)
    jockey_rt_ratings: dict = defaultdict(lambda: DEFAULT_RATING)

    horse_surf_counts: dict = defaultdict(int)
    horse_rt_counts: dict = defaultdict(int)
    horse_dc_counts: dict = defaultdict(int)
    jockey_rt_counts: dict = defaultdict(int)

    horse_elo_surf_col = np.full(len(df), np.nan)
    horse_elo_rt_col = np.full(len(df), np.nan)
    horse_elo_dc_col = np.full(len(df), np.nan)
    jockey_elo_rt_col = np.full(len(df), np.nan)

    _surfaces = df["surface"].values if has_surface else None
    _race_types = df["race_type"].values if has_race_type else None
    _dist_cats = df["dist_category"].values if has_dist_cat else None

    # ── Margin Elo state ─────────────────────────────────────────────
    # Separate rating pool: penalises for beaten lengths and DNF.
    horse_margin_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    horse_margin_race_counts: dict[str, int] = defaultdict(int)
    horse_margin_momentum: dict[str, float] = defaultdict(float)
    horse_margin_elo_col = np.full(len(df), np.nan)
    horse_margin_delta_col = np.zeros(len(df))
    horse_margin_momentum_col = np.full(len(df), np.nan)
    _lengths_behind = df["lengths_behind"].values if "lengths_behind" in df.columns else None
    _dist_furlongs = df["distance_furlongs"].values if "distance_furlongs" in df.columns else None

    for race_id in race_ids_ordered:
        idx = _race_groups.indices[race_id]

        # --- Record *pre-race* ratings (no look-ahead) ---
        # Must happen for ALL races including future/unfinished ones
        # so live racecards get their accumulated Elo ratings.
        for i in idx:
            _hk = _horse_keys[i]
            if pd.notna(_hk) and horse_race_counts[_hk] > 0:
                horse_elo_col[i] = horse_ratings[_hk]
                horse_momentum_col[i] = horse_momentum[_hk]
            # Margin Elo pre-race
            if pd.notna(_hk) and horse_margin_race_counts[_hk] > 0:
                horse_margin_elo_col[i] = horse_margin_ratings[_hk]
                horse_margin_momentum_col[i] = horse_margin_momentum[_hk]
            if has_jockey and pd.notna(_jockey_keys[i]) and jockey_race_counts[_jockey_keys[i]] > 0:
                jockey_elo_col[i] = jockey_ratings[_jockey_keys[i]]
            if has_trainer and pd.notna(_trainer_keys[i]) and trainer_race_counts[_trainer_keys[i]] > 0:
                trainer_elo_col[i] = trainer_ratings[_trainer_keys[i]]

            # ── Dimensional pre-race ratings ──
            if pd.isna(_hk):
                continue
            if _surfaces is not None and pd.notna(_surfaces[i]) and horse_race_counts[_hk] > 0:
                _sk = (_hk, _surfaces[i])
                horse_elo_surf_col[i] = (
                    horse_surf_ratings[_sk]
                    if horse_surf_counts[_sk] >= MIN_DIM_RACES
                    else horse_ratings[_hk]
                )
            if _race_types is not None and pd.notna(_race_types[i]) and horse_race_counts[_hk] > 0:
                _rk = (_hk, _race_types[i])
                horse_elo_rt_col[i] = (
                    horse_rt_ratings[_rk]
                    if horse_rt_counts[_rk] >= MIN_DIM_RACES
                    else horse_ratings[_hk]
                )
                if has_jockey and pd.notna(_jockey_keys[i]) and jockey_race_counts[_jockey_keys[i]] > 0:
                    _jrk = (_jockey_keys[i], _race_types[i])
                    jockey_elo_rt_col[i] = (
                        jockey_rt_ratings[_jrk]
                        if jockey_rt_counts[_jrk] >= MIN_DIM_RACES
                        else jockey_ratings[_jockey_keys[i]]
                    )
            if _dist_cats is not None and np.isfinite(_dist_cats[i]) and horse_race_counts[_hk] > 0:
                _dk = (_hk, int(_dist_cats[i]))
                horse_elo_dc_col[i] = (
                    horse_dc_ratings[_dk]
                    if horse_dc_counts[_dk] >= MIN_DIM_RACES
                    else horse_ratings[_hk]
                )

        # Skip races without finish positions (e.g. racecards / future)
        if _finish_pos is None:
            continue
        fp_race = _finish_pos[idx]
        valid_mask = np.isfinite(fp_race) & (fp_race > 0)
        if valid_mask.sum() < 2:
            continue

        valid_idx = idx[valid_mask]

        # --- Build finish order for Elo update (array access, no iterrows) ---
        horse_finish = [
            (_horse_keys[i], int(fp_race[valid_mask][j]))
            for j, i in enumerate(valid_idx)
            if pd.notna(_horse_keys[i])
        ]
        if len(horse_finish) < 2:
            continue

        # --- Update horse ratings ---
        h_deltas = _update_ratings_for_race(
            horse_finish, horse_ratings, k,
            race_counts=horse_race_counts,
        )
        for i in valid_idx:
            horse_name = _horse_keys[i]
            if pd.isna(horse_name):
                continue
            _delta = h_deltas.get(horse_name, 0.0)
            horse_delta_col[i] = _delta
            horse_race_counts[horse_name] += 1
            # Update momentum EMA: m_new = alpha * delta + (1 - alpha) * m_old
            horse_momentum[horse_name] = (
                MOMENTUM_ALPHA * _delta
                + (1.0 - MOMENTUM_ALPHA) * horse_momentum[horse_name]
            )

        # ── Margin Elo: includes non-finishers with penalty ──
        if _lengths_behind is not None:
            # Collect all horses that competed (finishers + non-finishers)
            # Finishers: fp > 0 → use their lengths_behind
            # Non-finishers: fp == 0 (PU/F/UR/BD) → penalised virtual lb
            # NR: not in the data (excluded by scraper)
            lb_race = _lengths_behind[idx]
            fp_full = fp_race

            # Max lb among finishers in this race
            finisher_lbs = lb_race[valid_mask]
            finite_lbs = finisher_lbs[np.isfinite(finisher_lbs)]
            max_lb = float(finite_lbs.max()) if len(finite_lbs) > 0 else 0.0

            margin_runners = []
            all_compete_idx = []
            for j_pos, i in enumerate(idx):
                hk = _horse_keys[i]
                if pd.isna(hk):
                    continue
                fp_i = fp_full[j_pos] if np.isfinite(fp_full[j_pos]) else 0
                if fp_i >= 1:
                    # Finisher: use actual lb (winner = 0.0)
                    lb_i = lb_race[j_pos]
                    if not np.isfinite(lb_i):
                        lb_i = 0.0 if fp_i == 1 else max_lb
                    margin_runners.append((hk, float(lb_i)))
                    all_compete_idx.append(i)
                elif fp_i == 0:
                    # Non-finisher who competed: penalise heavily
                    margin_runners.append((hk, max_lb + DNF_PENALTY_LB))
                    all_compete_idx.append(i)

            if len(margin_runners) >= 2:
                # Distance-conditioned margin scale
                _race_dist = float(_dist_furlongs[idx[0]]) if _dist_furlongs is not None else MARGIN_REF_DIST
                _m_scale = _distance_margin_scale(_race_dist)
                m_deltas = _update_margin_elo_for_race(
                    margin_runners, horse_margin_ratings, k,
                    race_counts=horse_margin_race_counts,
                    margin_scale=_m_scale,
                )
                for i in all_compete_idx:
                    hk = _horse_keys[i]
                    if pd.isna(hk):
                        continue
                    _md = m_deltas.get(hk, 0.0)
                    horse_margin_delta_col[i] = _md
                    horse_margin_race_counts[hk] += 1
                    horse_margin_momentum[hk] = (
                        MOMENTUM_ALPHA * _md
                        + (1.0 - MOMENTUM_ALPHA) * horse_margin_momentum[hk]
                    )

        # ── Update dimensional horse ratings ──
        _first_valid = valid_idx[0]
        if _surfaces is not None and pd.notna(_surfaces[_first_valid]):
            _rs = _surfaces[_first_valid]
            _sf = [
                ((_horse_keys[i], _rs), int(fp_race[valid_mask][j]))
                for j, i in enumerate(valid_idx)
                if pd.notna(_horse_keys[i])
            ]
            if len(_sf) >= 2:
                _update_ratings_for_race(
                    _sf, horse_surf_ratings, k, race_counts=horse_surf_counts,
                )
                for i in valid_idx:
                    if pd.notna(_horse_keys[i]):
                        horse_surf_counts[(_horse_keys[i], _rs)] += 1

        if _race_types is not None and pd.notna(_race_types[_first_valid]):
            _rr = _race_types[_first_valid]
            _rf = [
                ((_horse_keys[i], _rr), int(fp_race[valid_mask][j]))
                for j, i in enumerate(valid_idx)
                if pd.notna(_horse_keys[i])
            ]
            if len(_rf) >= 2:
                _update_ratings_for_race(
                    _rf, horse_rt_ratings, k, race_counts=horse_rt_counts,
                )
                for i in valid_idx:
                    if pd.notna(_horse_keys[i]):
                        horse_rt_counts[(_horse_keys[i], _rr)] += 1

        if _dist_cats is not None and np.isfinite(_dist_cats[_first_valid]):
            _rd = int(_dist_cats[_first_valid])
            _dcf = [
                ((_horse_keys[i], _rd), int(fp_race[valid_mask][j]))
                for j, i in enumerate(valid_idx)
                if pd.notna(_horse_keys[i])
            ]
            if len(_dcf) >= 2:
                _update_ratings_for_race(
                    _dcf, horse_dc_ratings, k, race_counts=horse_dc_counts,
                )
                for i in valid_idx:
                    if pd.notna(_horse_keys[i]):
                        horse_dc_counts[(_horse_keys[i], _rd)] += 1

        # --- Update jockey ratings ---
        if has_jockey:
            jockey_finish = [
                (_jockey_keys[i], int(_finish_pos[i]))
                for i in valid_idx
                if pd.notna(_jockey_keys[i])
            ]
            if len(jockey_finish) >= 2:
                j_deltas = _update_ratings_for_race(
                    jockey_finish, jockey_ratings, k,
                    race_counts=jockey_race_counts,
                )
                for i in valid_idx:
                    jname = _jockey_keys[i]
                    if pd.notna(jname):
                        jockey_delta_col[i] = j_deltas.get(jname, 0.0)
                        jockey_race_counts[jname] += 1

        # ── Update jockey dimensional (race type) ──
        if has_jockey and _race_types is not None:
            _rr_j = _race_types[_first_valid]
            if pd.notna(_rr_j):
                _jrf = [
                    ((_jockey_keys[i], _rr_j), int(_finish_pos[i]))
                    for i in valid_idx
                    if pd.notna(_jockey_keys[i])
                ]
                if len(_jrf) >= 2:
                    _update_ratings_for_race(
                        _jrf, jockey_rt_ratings, k,
                        race_counts=jockey_rt_counts,
                    )
                    for i in valid_idx:
                        _jn = _jockey_keys[i]
                        if pd.notna(_jn):
                            jockey_rt_counts[(_jn, _rr_j)] += 1

        # --- Update trainer ratings ---
        if has_trainer:
            trainer_finish = [
                (_trainer_keys[i], int(_finish_pos[i]))
                for i in valid_idx
                if pd.notna(_trainer_keys[i])
            ]
            if len(trainer_finish) >= 2:
                _update_ratings_for_race(
                    trainer_finish, trainer_ratings, k,
                    race_counts=trainer_race_counts,
                )
                for i in valid_idx:
                    tname = _trainer_keys[i]
                    if pd.notna(tname):
                        trainer_race_counts[tname] += 1

    race_ids = df["race_id"]
    horse_elo = pd.Series(horse_elo_col, index=df.index)
    jockey_elo = pd.Series(jockey_elo_col, index=df.index)
    trainer_elo = pd.Series(trainer_elo_col, index=df.index)

    # Relative-to-field features (race-aware)
    race_avg_elo = horse_elo.groupby(race_ids, sort=False).transform("mean")
    race_max_elo = horse_elo.groupby(race_ids, sort=False).transform("max")
    horse_elo_rank = horse_elo.groupby(race_ids, sort=False).rank(
        ascending=False, method="min"
    )

    combined_elo = horse_elo + 0.3 * jockey_elo if has_jockey else horse_elo.copy()
    combo_avg = combined_elo.groupby(race_ids, sort=False).transform("mean")

    # Momentum relative to field
    horse_mom = pd.Series(horse_momentum_col, index=df.index)
    mom_field_avg = horse_mom.groupby(race_ids, sort=False).transform("mean")

    elo_cols = {
        "horse_elo": horse_elo_col,
        "jockey_elo": jockey_elo_col,
        "trainer_elo": trainer_elo_col,
        "has_horse_elo": horse_elo.notna().astype(int).to_numpy(),
        "has_jockey_elo": jockey_elo.notna().astype(int).to_numpy(),
        "has_trainer_elo": trainer_elo.notna().astype(int).to_numpy(),
        "horse_elo_delta": horse_delta_col,
        "jockey_elo_delta": jockey_delta_col,
        "horse_elo_momentum": horse_momentum_col,
        "horse_elo_momentum_vs_field": (horse_mom - mom_field_avg).to_numpy(),
        "horse_elo_vs_field": (horse_elo - race_avg_elo).to_numpy(),
        "horse_elo_rank": horse_elo_rank.to_numpy(),
        "horse_elo_top": np.where(
            horse_elo.notna() & race_max_elo.notna(),
            (horse_elo == race_max_elo).astype(float),
            np.nan,
        ),
        "combined_elo": combined_elo.to_numpy(),
        "combined_elo_vs_field": (combined_elo - combo_avg).to_numpy(),
    }

    # ── Dimensional Elo features ──────────────────────────────────
    _horse_elo_surf = pd.Series(horse_elo_surf_col, index=df.index)
    _horse_elo_rtype = pd.Series(horse_elo_rt_col, index=df.index)
    _horse_elo_dcat = pd.Series(horse_elo_dc_col, index=df.index)
    _jockey_elo_rtype = pd.Series(jockey_elo_rt_col, index=df.index)

    elo_cols["horse_elo_surface"] = horse_elo_surf_col
    elo_cols["horse_elo_surface_edge"] = (_horse_elo_surf - horse_elo).to_numpy()
    elo_cols["horse_elo_rt"] = horse_elo_rt_col
    elo_cols["horse_elo_rt_edge"] = (_horse_elo_rtype - horse_elo).to_numpy()
    elo_cols["horse_elo_dist_cat"] = horse_elo_dc_col
    elo_cols["horse_elo_dist_cat_edge"] = (_horse_elo_dcat - horse_elo).to_numpy()
    elo_cols["jockey_elo_rt"] = jockey_elo_rt_col
    elo_cols["jockey_elo_rt_edge"] = (_jockey_elo_rtype - jockey_elo).to_numpy()

    # ── Margin Elo columns ────────────────────────────────────────
    _m_elo = pd.Series(horse_margin_elo_col, index=df.index)
    _m_mom = pd.Series(horse_margin_momentum_col, index=df.index)
    _m_avg = _m_elo.groupby(race_ids, sort=False).transform("mean")
    _m_max = _m_elo.groupby(race_ids, sort=False).transform("max")
    _m_rank = _m_elo.groupby(race_ids, sort=False).rank(ascending=False, method="min")
    _m_mom_avg = _m_mom.groupby(race_ids, sort=False).transform("mean")

    elo_cols["horse_margin_elo"] = horse_margin_elo_col
    elo_cols["horse_margin_elo_delta"] = horse_margin_delta_col
    elo_cols["horse_margin_elo_momentum"] = horse_margin_momentum_col
    elo_cols["horse_margin_elo_vs_field"] = (_m_elo - _m_avg).to_numpy()
    elo_cols["horse_margin_elo_rank"] = _m_rank.to_numpy()
    elo_cols["horse_margin_elo_top"] = np.where(
        _m_elo.notna() & _m_max.notna(),
        (_m_elo == _m_max).astype(float), np.nan,
    )
    elo_cols["horse_margin_elo_momentum_vs_field"] = (_m_mom - _m_mom_avg).to_numpy()
    elo_cols["margin_elo_edge"] = (_m_elo - horse_elo).to_numpy()

    if has_jockey:
        j_avg = jockey_elo.groupby(race_ids, sort=False).transform("mean")
        elo_cols["jockey_elo_vs_field"] = (jockey_elo - j_avg).to_numpy()

    if has_trainer:
        t_avg = trainer_elo.groupby(race_ids, sort=False).transform("mean")
        elo_cols["trainer_elo_vs_field"] = (trainer_elo - t_avg).to_numpy()

    elo_frame = pd.DataFrame(elo_cols, index=df.index)
    df = pd.concat([df.drop(columns=["_event_dt"], errors="ignore"), elo_frame], axis=1)

    n_horses = len(horse_ratings)
    n_jockeys = len(jockey_ratings) if has_jockey else 0
    n_trainers = len(trainer_ratings) if has_trainer else 0
    _n_dims = sum([has_surface, has_race_type, has_dist_cat])
    logger.info(
        f"Elo complete: {n_horses} horses, {n_jockeys} jockeys, "
        f"{n_trainers} trainers rated ({_n_dims} dimensional splits)"
    )

    return df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, config

    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.csv")
    featured_path = pq_path if os.path.exists(pq_path) else csv_path
    if not os.path.exists(featured_path):
        print("❌ No featured data found. Run training first.")
        raise SystemExit(1)

    df = pd.read_parquet(featured_path, engine="pyarrow") if featured_path.endswith(".parquet") else pd.read_csv(featured_path)
    df = compute_elo_features(df)

    print(f"\nRated {df['horse_elo'].nunique()} unique horse Elo values")
    print(f"Elo range: {df['horse_elo'].min():.0f} – {df['horse_elo'].max():.0f}")
    print(f"\nTop 10 horses by final Elo:")
    latest = df.sort_values("race_date").drop_duplicates("horse_name", keep="last")
    top = latest.nlargest(10, "horse_elo")[["horse_name", "horse_elo"]]
    print(top.to_string(index=False))
