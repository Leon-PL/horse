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


def _day_start_gather(values: np.ndarray, keys: list, dates: np.ndarray) -> np.ndarray:
    """Start-of-day view of per-row entity state.

    Returns ``values`` with every row replaced by the value at the
    entity's first row of that calendar day. Ratings recorded for an
    entity's later races on a day embed the day's earlier results —
    information that cannot exist at prediction time. Rows with a null
    key keep their own value.
    """
    n = len(values)
    pos = pd.Series(np.arange(n))
    key_series = [pd.Series(np.asarray(k, dtype=object)) for k in keys]
    day = pd.Series(pd.to_datetime(dates)).dt.normalize()
    first = pos.groupby([*key_series, day], sort=False, dropna=True).transform("min")
    first = first.fillna(pos).to_numpy().astype(np.int64)
    if (first == np.arange(n)).all():
        return values
    return np.asarray(values)[first]


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
    *,
    elo_state: dict | None = None,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
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

    # Optionally seed the rating state from a prior incremental run so the
    # chronological sweep continues from where it left off instead of
    # restarting from DEFAULT_RATING. The end-state is returned when
    # return_state=True so a feature store can persist it and re-feed it on
    # the next append — making the sweep process only the new rows while
    # producing byte-identical pre-race ratings.
    _seed_state = elo_state or {}

    def _seed(name: str, factory):
        d = defaultdict(factory)
        prior = _seed_state.get(name)
        if prior:
            d.update(prior)
        return d

    _R = lambda: DEFAULT_RATING  # noqa: E731 — rating default factory

    # Rating dictionaries — persist across all races
    horse_ratings: dict[str, float] = _seed("horse_ratings", _R)
    jockey_ratings: dict[str, float] = _seed("jockey_ratings", _R)
    trainer_ratings: dict[str, float] = _seed("trainer_ratings", _R)

    # Race-count dictionaries for adaptive K-factor
    horse_race_counts: dict[str, int] = _seed("horse_race_counts", int)
    jockey_race_counts: dict[str, int] = _seed("jockey_race_counts", int)
    trainer_race_counts: dict[str, int] = _seed("trainer_race_counts", int)

    # Columns to populate
    horse_elo_col = np.full(len(df), np.nan)
    jockey_elo_col = np.full(len(df), np.nan)
    trainer_elo_col = np.full(len(df), np.nan)
    horse_delta_col = np.zeros(len(df))
    jockey_delta_col = np.zeros(len(df))
    has_horse_elo_col = np.zeros(len(df), dtype=np.int8)
    has_jockey_elo_col = np.zeros(len(df), dtype=np.int8)
    has_trainer_elo_col = np.zeros(len(df), dtype=np.int8)

    # Momentum Elo: exponential moving average of recent deltas
    horse_momentum: dict[str, float] = _seed("horse_momentum", float)
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
    horse_surf_ratings: dict = _seed("horse_surf_ratings", _R)
    horse_rt_ratings: dict = _seed("horse_rt_ratings", _R)
    horse_dc_ratings: dict = _seed("horse_dc_ratings", _R)
    jockey_rt_ratings: dict = _seed("jockey_rt_ratings", _R)

    horse_surf_counts: dict = _seed("horse_surf_counts", int)
    horse_rt_counts: dict = _seed("horse_rt_counts", int)
    horse_dc_counts: dict = _seed("horse_dc_counts", int)
    jockey_rt_counts: dict = _seed("jockey_rt_counts", int)

    horse_elo_surf_col = np.full(len(df), np.nan)
    horse_elo_rt_col = np.full(len(df), np.nan)
    horse_elo_dc_col = np.full(len(df), np.nan)
    jockey_elo_rt_col = np.full(len(df), np.nan)

    _surfaces = df["surface"].values if has_surface else None
    _race_types = df["race_type"].values if has_race_type else None
    _dist_cats = df["dist_category"].values if has_dist_cat else None

    # ── Margin Elo state ─────────────────────────────────────────────
    # Separate rating pool: penalises for beaten lengths and DNF.
    horse_margin_ratings: dict[str, float] = _seed("horse_margin_ratings", _R)
    horse_margin_race_counts: dict[str, int] = _seed("horse_margin_race_counts", int)
    horse_margin_momentum: dict[str, float] = _seed("horse_margin_momentum", float)
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
            if pd.notna(_hk):
                horse_elo_col[i] = horse_ratings[_hk]
                horse_momentum_col[i] = horse_momentum[_hk]
                has_horse_elo_col[i] = np.int8(horse_race_counts[_hk] > 0)
            # Margin Elo pre-race
            if pd.notna(_hk):
                horse_margin_elo_col[i] = horse_margin_ratings[_hk]
                horse_margin_momentum_col[i] = horse_margin_momentum[_hk]
            if has_jockey and pd.notna(_jockey_keys[i]):
                jockey_elo_col[i] = jockey_ratings[_jockey_keys[i]]
                has_jockey_elo_col[i] = np.int8(jockey_race_counts[_jockey_keys[i]] > 0)
            if has_trainer and pd.notna(_trainer_keys[i]):
                trainer_elo_col[i] = trainer_ratings[_trainer_keys[i]]
                has_trainer_elo_col[i] = np.int8(trainer_race_counts[_trainer_keys[i]] > 0)

            # ── Dimensional pre-race ratings ──
            if pd.isna(_hk):
                continue
            if _surfaces is not None and pd.notna(_surfaces[i]):
                _sk = (_hk, _surfaces[i])
                horse_elo_surf_col[i] = (
                    horse_surf_ratings[_sk]
                    if horse_surf_counts[_sk] >= MIN_DIM_RACES
                    else horse_ratings[_hk]
                )
            if _race_types is not None and pd.notna(_race_types[i]):
                _rk = (_hk, _race_types[i])
                horse_elo_rt_col[i] = (
                    horse_rt_ratings[_rk]
                    if horse_rt_counts[_rk] >= MIN_DIM_RACES
                    else horse_ratings[_hk]
                )
                if has_jockey and pd.notna(_jockey_keys[i]):
                    _jrk = (_jockey_keys[i], _race_types[i])
                    jockey_elo_rt_col[i] = (
                        jockey_rt_ratings[_jrk]
                        if jockey_rt_counts[_jrk] >= MIN_DIM_RACES
                        else jockey_ratings[_jockey_keys[i]]
                    )
            if _dist_cats is not None and np.isfinite(_dist_cats[i]):
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

    # Date-strict view for entities that run several times a day:
    # all of a jockey's/trainer's rides on one day must carry the
    # rating recorded at their FIRST ride of the day — later rides'
    # ratings embed same-day results that don't exist at prediction
    # time. (Horses essentially never run twice a day.)
    _day_dates = pd.to_datetime(df["race_date"]).values
    if has_jockey:
        jockey_elo_col = _day_start_gather(jockey_elo_col, [_jockey_keys], _day_dates)
        has_jockey_elo_col = _day_start_gather(has_jockey_elo_col, [_jockey_keys], _day_dates)
        jockey_elo_rt_col = _day_start_gather(
            jockey_elo_rt_col,
            [_jockey_keys, _race_types if _race_types is not None else np.zeros(len(df))],
            _day_dates,
        )
    if has_trainer:
        trainer_elo_col = _day_start_gather(trainer_elo_col, [_trainer_keys], _day_dates)
        has_trainer_elo_col = _day_start_gather(has_trainer_elo_col, [_trainer_keys], _day_dates)

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
        "has_horse_elo": has_horse_elo_col,
        "has_jockey_elo": has_jockey_elo_col,
        "has_trainer_elo": has_trainer_elo_col,
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

    if return_state:
        out_state = {
            "horse_ratings": dict(horse_ratings),
            "jockey_ratings": dict(jockey_ratings),
            "trainer_ratings": dict(trainer_ratings),
            "horse_race_counts": dict(horse_race_counts),
            "jockey_race_counts": dict(jockey_race_counts),
            "trainer_race_counts": dict(trainer_race_counts),
            "horse_momentum": dict(horse_momentum),
            "horse_surf_ratings": dict(horse_surf_ratings),
            "horse_rt_ratings": dict(horse_rt_ratings),
            "horse_dc_ratings": dict(horse_dc_ratings),
            "jockey_rt_ratings": dict(jockey_rt_ratings),
            "horse_surf_counts": dict(horse_surf_counts),
            "horse_rt_counts": dict(horse_rt_counts),
            "horse_dc_counts": dict(horse_dc_counts),
            "jockey_rt_counts": dict(jockey_rt_counts),
            "horse_margin_ratings": dict(horse_margin_ratings),
            "horse_margin_race_counts": dict(horse_margin_race_counts),
            "horse_margin_momentum": dict(horse_margin_momentum),
        }
        return df, out_state

    return df


# ── Glicko-1 ratings ──────────────────────────────────────────────────────
# Glickman (1999). Adds a per-horse rating deviation (RD): uncertainty
# that shrinks with evidence and *grows during layoffs* — something the
# adaptive-K Elo above cannot express (K only decays with run count).
# Each race is treated as one rating period of n-1 pairwise games.

GLICKO_RATING_INIT = 1500.0
GLICKO_RD_INIT = getattr(config, "GLICKO_RD_INIT", 350.0)
GLICKO_RD_MIN = getattr(config, "GLICKO_RD_MIN", 50.0)
# RD inflation: after t months idle, RD -> sqrt(RD^2 + C^2 * t).
# C=70 returns a fully-known horse (RD 50) to ~max uncertainty in ~2 years.
GLICKO_C = getattr(config, "GLICKO_C", 70.0)

_GLICKO_Q = np.log(10.0) / 400.0


def _glicko_race_update(
    r: np.ndarray, rd: np.ndarray, fp: np.ndarray,
    exclude: np.ndarray | None = None,
    S_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Glicko-1 update for one race.

    Args:
        r: pre-race ratings (m,)
        rd: pre-race deviations (m,), already inflated for inactivity.
        fp: finish positions (m,) — lower is better, ties allowed.
        exclude: optional (m, m) bool matrix of extra pairs to ignore —
            stops an entity with several runners in one race (a trainer,
            occasionally a jockey across re-rides) playing itself.
        S_override: optional (m, m) actual-score matrix replacing the
            binary win/tie/loss scores (margin-aware variant).

    Returns:
        (new_ratings, new_rds)
    """
    m = len(r)
    q = _GLICKO_Q
    g = 1.0 / np.sqrt(1.0 + 3.0 * (q ** 2) * (rd ** 2) / (np.pi ** 2))

    # E[i, j] = expected score of i against j (uses opponent's g)
    diff = r[:, None] - r[None, :]
    E = 1.0 / (1.0 + 10.0 ** (-(g[None, :] * diff) / 400.0))
    # S[i, j] = actual score of i against j
    if S_override is not None:
        S = S_override
    else:
        S = np.where(fp[:, None] < fp[None, :], 1.0,
                     np.where(fp[:, None] == fp[None, :], 0.5, 0.0))

    off_diag = ~np.eye(m, dtype=bool)
    if exclude is not None:
        off_diag &= ~exclude
    g_sq_E = (g[None, :] ** 2) * E * (1.0 - E)
    d2_inv = (q ** 2) * np.where(off_diag, g_sq_E, 0.0).sum(axis=1)
    sum_term = np.where(off_diag, g[None, :] * (S - E), 0.0).sum(axis=1)

    denom = 1.0 / (rd ** 2) + d2_inv
    new_r = r + (q / denom) * sum_term
    new_rd = np.sqrt(1.0 / denom)
    return new_r, np.clip(new_rd, GLICKO_RD_MIN, GLICKO_RD_INIT)


def _glicko_pass(
    keys: list,
    fp: np.ndarray | None,
    dates: np.ndarray,
    race_ids_ordered: np.ndarray,
    race_group_indices: dict,
    margin: np.ndarray | None = None,
    *,
    state: dict | None = None,
    return_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """One chronological Glicko-1 sweep for an arbitrary entity keying.

    Args:
        keys: per-row hashable entity key (str or tuple), or None where
            unknown. Duplicate keys within a race (a trainer's multiple
            runners) never play against themselves; their post-race
            updates are averaged.
        fp: finish positions per row, or None (live rows — no updates).
        dates: race_date per row as datetime64.
        margin: optional per-row beaten lengths. When given, pairwise
            game scores become sigmoid((lb_j − lb_i) / MARGIN_SCALE)
            instead of binary win/lose — a 10-length thrashing moves
            ratings more than a nose verdict. Pairs with missing lengths
            fall back to the binary score.

    Returns:
        Per-row *pre-race* (rating, rd, has_prior_run, prior_run_count).
        When ``return_state`` is set, also returns the carried state dict so
        an incremental run can re-seed and sweep only new rows.
    """
    if state is not None:
        ratings: dict = dict(state.get("ratings", {}))
        rds: dict = dict(state.get("rds", {}))
        last_dt: dict = dict(state.get("last_dt", {}))
        counts: dict = dict(state.get("counts", {}))
    else:
        ratings = {}
        rds = {}
        last_dt = {}
        counts = {}

    n = len(keys)
    r_col = np.full(n, np.nan)
    rd_col = np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)
    cnt_col = np.zeros(n, dtype=np.int32)

    for race_id in race_ids_ordered:
        idx = race_group_indices[race_id]
        race_dt = dates[idx[0]]

        # Pre-race: inflate RD for inactivity, record values
        for i in idx:
            k = keys[i]
            if k is None:
                continue
            if k in ratings:
                idle_days = (race_dt - last_dt[k]) / np.timedelta64(1, "D")
                months = max(float(idle_days), 0.0) / 30.44
                rd_now = min(
                    float(np.sqrt(rds[k] ** 2 + (GLICKO_C ** 2) * months)),
                    GLICKO_RD_INIT,
                )
                rds[k] = rd_now
                r_col[i] = ratings[k]
                rd_col[i] = rd_now
                has_col[i] = 1
                cnt_col[i] = counts[k]
            else:
                r_col[i] = GLICKO_RATING_INIT
                rd_col[i] = GLICKO_RD_INIT

        # Post-race update (finishers only)
        if fp is None:
            continue
        fp_race = fp[idx]
        valid = np.isfinite(fp_race) & (fp_race > 0)
        upd = [
            (j, keys[idx[j]]) for j in range(len(idx))
            if valid[j] and keys[idx[j]] is not None
        ]
        ukeys = [k for _, k in upd]
        if len(set(ukeys)) < 2:
            continue
        m = len(upd)
        r_arr = np.array([ratings.get(k, GLICKO_RATING_INIT) for k in ukeys])
        rd_arr = np.array([rds.get(k, GLICKO_RD_INIT) for k in ukeys])
        fp_arr = np.array([float(fp_race[j]) for j, _ in upd])
        exclude = None
        if len(set(ukeys)) < m:
            karr = np.empty(m, dtype=object)
            karr[:] = ukeys
            exclude = karr[:, None] == karr[None, :]
        S_override = None
        if margin is not None:
            lb = np.array([float(margin[idx[j]]) for j, _ in upd])
            both = np.isfinite(lb[:, None]) & np.isfinite(lb[None, :])
            S_margin = 1.0 / (1.0 + np.exp(
                -np.clip((np.nan_to_num(lb)[None, :] - np.nan_to_num(lb)[:, None]) / MARGIN_SCALE, -60, 60)
            ))
            S_fp = np.where(fp_arr[:, None] < fp_arr[None, :], 1.0,
                            np.where(fp_arr[:, None] == fp_arr[None, :], 0.5, 0.0))
            S_override = np.where(both, S_margin, S_fp)
        new_r, new_rd = _glicko_race_update(
            r_arr, rd_arr, fp_arr, exclude=exclude, S_override=S_override
        )
        agg: dict = {}
        for k, nr, nrd in zip(ukeys, new_r, new_rd):
            agg.setdefault(k, []).append((nr, nrd))
        for k, vals in agg.items():
            ratings[k] = float(np.mean([v[0] for v in vals]))
            rds[k] = float(np.mean([v[1] for v in vals]))
            last_dt[k] = race_dt
            counts[k] = counts.get(k, 0) + 1

    if return_state:
        out = {"ratings": ratings, "rds": rds, "last_dt": last_dt, "counts": counts}
        return r_col, rd_col, has_col, cnt_col, out
    return r_col, rd_col, has_col, cnt_col


# ── Glicko-2 (Glickman 2001) ──────────────────────────────────────────────
# Adds a per-entity volatility σ: how erratic results have been. High σ
# horses swing between career-bests and duds — useful signal in itself.

_G2_SCALE = 173.7178
GLICKO2_SIGMA_INIT = 0.06


def _g2_solve_sigma(delta2: float, phi2: float, v: float, sigma: float, tau: float) -> float:
    """Illinois-method solve for the new volatility (Glickman's step 5)."""
    a = np.log(sigma ** 2)

    def f(x: float) -> float:
        ex = np.exp(x)
        return (
            ex * (delta2 - phi2 - v - ex) / (2.0 * (phi2 + v + ex) ** 2)
            - (x - a) / (tau ** 2)
        )

    A = a
    if delta2 > phi2 + v:
        B = np.log(delta2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0 and k < 50:
            k += 1
        B = a - k * tau
    fA, fB = f(A), f(B)
    for _ in range(60):
        if abs(B - A) <= 1e-6:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2.0
        B, fB = C, fC
    return float(np.exp(A / 2.0))


def _glicko2_pass(
    keys: list,
    fp: np.ndarray | None,
    dates: np.ndarray,
    race_ids_ordered: np.ndarray,
    race_group_indices: dict,
    *,
    state: dict | None = None,
    return_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Chronological Glicko-2 sweep. Returns per-row pre-race
    (rating, rd, volatility, has_prior_run) on the Glicko-1 scale."""
    tau = float(getattr(config, "GLICKO2_TAU", 0.5))
    phi_init = GLICKO_RD_INIT / _G2_SCALE
    phi_min = GLICKO_RD_MIN / _G2_SCALE

    if state is not None:
        mus: dict = dict(state.get("mus", {}))
        phis: dict = dict(state.get("phis", {}))
        sigmas: dict = dict(state.get("sigmas", {}))
        last_dt: dict = dict(state.get("last_dt", {}))
    else:
        mus = {}
        phis = {}
        sigmas = {}
        last_dt = {}

    n = len(keys)
    r_col = np.full(n, np.nan)
    rd_col = np.full(n, np.nan)
    vol_col = np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)

    for race_id in race_ids_ordered:
        idx = race_group_indices[race_id]
        race_dt = dates[idx[0]]

        for i in idx:
            k = keys[i]
            if k is None:
                continue
            if k in mus:
                idle_days = (race_dt - last_dt[k]) / np.timedelta64(1, "D")
                t = max(float(idle_days), 0.0) / 30.44  # rating periods ≈ months
                phi_now = min(
                    float(np.sqrt(phis[k] ** 2 + (sigmas[k] ** 2) * t)), phi_init
                )
                phis[k] = phi_now
                r_col[i] = GLICKO_RATING_INIT + _G2_SCALE * mus[k]
                rd_col[i] = _G2_SCALE * phi_now
                vol_col[i] = sigmas[k]
                has_col[i] = 1
            else:
                r_col[i] = GLICKO_RATING_INIT
                rd_col[i] = GLICKO_RD_INIT
                vol_col[i] = GLICKO2_SIGMA_INIT

        if fp is None:
            continue
        fp_race = fp[idx]
        valid = np.isfinite(fp_race) & (fp_race > 0)
        upd = [
            (j, keys[idx[j]]) for j in range(len(idx))
            if valid[j] and keys[idx[j]] is not None
        ]
        if len(upd) < 2:
            continue
        m = len(upd)
        mu = np.array([mus.get(k, 0.0) for _, k in upd])
        phi = np.array([phis.get(k, phi_init) for _, k in upd])
        sigma = np.array([sigmas.get(k, GLICKO2_SIGMA_INIT) for _, k in upd])
        fp_arr = np.array([float(fp_race[j]) for j, _ in upd])

        g = 1.0 / np.sqrt(1.0 + 3.0 * phi ** 2 / (np.pi ** 2))
        diff = mu[:, None] - mu[None, :]
        E = 1.0 / (1.0 + np.exp(-np.clip(g[None, :] * diff, -60, 60)))
        S = np.where(fp_arr[:, None] < fp_arr[None, :], 1.0,
                     np.where(fp_arr[:, None] == fp_arr[None, :], 0.5, 0.0))
        off = ~np.eye(m, dtype=bool)
        gE = (g[None, :] ** 2) * E * (1.0 - E)
        v = 1.0 / np.maximum(np.where(off, gE, 0.0).sum(axis=1), 1e-12)
        score_sum = np.where(off, g[None, :] * (S - E), 0.0).sum(axis=1)
        delta = v * score_sum

        new_sigma = np.array([
            _g2_solve_sigma(delta[i] ** 2, phi[i] ** 2, v[i], sigma[i], tau)
            for i in range(m)
        ])
        phi_star = np.sqrt(phi ** 2 + new_sigma ** 2)
        new_phi = np.clip(
            1.0 / np.sqrt(1.0 / phi_star ** 2 + 1.0 / v), phi_min, phi_init
        )
        new_mu = mu + (new_phi ** 2) * score_sum

        for (j, k), nm, nph, nsg in zip(upd, new_mu, new_phi, new_sigma):
            mus[k] = float(nm)
            phis[k] = float(nph)
            sigmas[k] = float(np.clip(nsg, 0.01, 0.2))
            last_dt[k] = race_dt

    if return_state:
        out = {"mus": mus, "phis": phis, "sigmas": sigmas, "last_dt": last_dt}
        return r_col, rd_col, vol_col, has_col, out
    return r_col, rd_col, vol_col, has_col


# ── TrueSkill ─────────────────────────────────────────────────────────────

def _trueskill_pass(
    keys: list,
    fp: np.ndarray | None,
    dates: np.ndarray,
    race_ids_ordered: np.ndarray,
    race_group_indices: dict,
    *,
    state: dict | None = None,
    return_state: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Chronological TrueSkill sweep: each race is an n-way free-for-all.

    Returns per-row pre-race (mu, sigma, has_prior_run). Time dynamics
    come from the environment's tau (added once per update), not layoffs.
    """
    import trueskill

    env = trueskill.TrueSkill(draw_probability=0.001)
    ts_ratings: dict = dict(state.get("ratings", {})) if state is not None else {}

    n = len(keys)
    mu_col = np.full(n, np.nan)
    sigma_col = np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)

    for race_id in race_ids_ordered:
        idx = race_group_indices[race_id]
        for i in idx:
            k = keys[i]
            if k is None:
                continue
            rating = ts_ratings.get(k)
            if rating is not None:
                mu_col[i] = rating.mu
                sigma_col[i] = rating.sigma
                has_col[i] = 1
            else:
                mu_col[i] = env.mu
                sigma_col[i] = env.sigma

        if fp is None:
            continue
        fp_race = fp[idx]
        valid = np.isfinite(fp_race) & (fp_race > 0)
        upd = [
            (j, keys[idx[j]]) for j in range(len(idx))
            if valid[j] and keys[idx[j]] is not None
        ]
        if len(upd) < 2:
            continue
        groups = [(ts_ratings.get(k, env.create_rating()),) for _, k in upd]
        ranks = [int(fp_race[j]) for j, _ in upd]
        try:
            new = env.rate(groups, ranks=ranks)
        except (FloatingPointError, ValueError):
            continue
        for (_, k), (rating,) in zip(upd, new):
            ts_ratings[k] = rating

    if return_state:
        return mu_col, sigma_col, has_col, {"ratings": ts_ratings}
    return mu_col, sigma_col, has_col


def compute_glicko_features(
    df: pd.DataFrame,
    *,
    glicko_state: dict | None = None,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Add Glicko-1 rating + deviation features.

    Processes races chronologically; records each entity's rating and RD
    *before* the race (no look-ahead), inflating RD for time idle.

    Horse columns (always):

    - ``horse_glicko``           — rating entering the race
    - ``horse_glicko_rd``        — rating deviation (uncertainty)
    - ``horse_glicko_lcb``       — conservative skill (rating − RD)
    - ``horse_glicko_vs_field``  — rating minus race average
    - ``horse_glicko_rank``      — rating rank within race
    - ``horse_glicko_rd_vs_field`` — uncertainty relative to the race

    With ``config.GLICKO_JOCKEY_TRAINER``: ``{jockey,trainer}_glicko``,
    ``_rd``, ``has_``, ``_vs_field``.

    With ``config.GLICKO_DIMENSIONAL``: per surface / race-type /
    distance-category horse ratings ``horse_glicko_{surface,rt,dist_cat}``
    plus ``_rd`` and ``_edge`` (dimensional minus global). Below
    3 prior runs in the dimension the global rating is used (edge = 0),
    mirroring the dimensional Elos' cold-start fallback.
    """
    logger.info("Computing Glicko ratings ...")
    df["race_date"] = pd.to_datetime(df["race_date"])
    if "_event_dt" in df.columns:
        event_dt = pd.to_datetime(df["_event_dt"], errors="coerce")
    else:
        event_dt = _event_sort_key(df)

    if not event_dt.is_monotonic_increasing:
        order = np.argsort(event_dt.values, kind="stable")
        df = df.iloc[order].reset_index(drop=True)

    _finish_pos = df["finish_position"].values if "finish_position" in df.columns else None
    _dates = pd.to_datetime(df["race_date"]).values
    race_ids_ordered = df.drop_duplicates("race_id", keep="first")["race_id"].values
    race_group_indices = df.groupby("race_id", sort=False).indices

    def _clean(arr) -> list:
        return [k if pd.notna(k) else None for k in arr]

    # Each chronological pass carries its own dict state. Thread per-pass
    # state (keyed by label) in and out so an incremental run re-seeds and
    # sweeps only the new rows. _run hides the variable return arity.
    _in = glicko_state or {}
    _out: dict = {}

    def _run(passfn, label, *args, **kwargs):
        res = passfn(
            *args, state=_in.get(label), return_state=return_state, **kwargs
        )
        if return_state:
            *cols_, st = res
            _out[label] = st
            return tuple(cols_)
        return res

    horse_keys = _clean(
        _entity_keys(df, id_col="horse_id", name_col="horse_name", prefix="horse")
    )
    glicko_col, rd_col, has_col, _ = _run(
        _glicko_pass, "horse",
        horse_keys, _finish_pos, _dates, race_ids_ordered, race_group_indices,
    )

    race_ids = df["race_id"]
    g_ser = pd.Series(glicko_col, index=df.index)
    rd_ser = pd.Series(rd_col, index=df.index)
    g_avg = g_ser.groupby(race_ids, sort=False).transform("mean")
    rd_avg = rd_ser.groupby(race_ids, sort=False).transform("mean")
    g_rank = g_ser.groupby(race_ids, sort=False).rank(ascending=False, method="min")

    cols = {
        "horse_glicko": glicko_col,
        "horse_glicko_rd": rd_col,
        "has_horse_glicko": has_col,
        "horse_glicko_lcb": glicko_col - rd_col,
        "horse_glicko_vs_field": (g_ser - g_avg).to_numpy(),
        "horse_glicko_rank": g_rank.to_numpy(),
        "horse_glicko_rd_vs_field": (rd_ser - rd_avg).to_numpy(),
    }

    if getattr(config, "GLICKO_JOCKEY_TRAINER", False):
        for ent, id_col, name_col in [
            ("jockey", "jockey_id", "jockey"),
            ("trainer", "trainer_id", "trainer"),
        ]:
            if name_col not in df.columns and id_col not in df.columns:
                continue
            keys = _clean(
                _entity_keys(df, id_col=id_col, name_col=name_col, prefix=ent)
            )
            r, rd, has, _ = _run(
                _glicko_pass, ent,
                keys, _finish_pos, _dates, race_ids_ordered, race_group_indices,
            )
            # Jockeys/trainers ride several races a day: start-of-day view
            r = _day_start_gather(r, [keys], _dates)
            rd = _day_start_gather(rd, [keys], _dates)
            has = _day_start_gather(has, [keys], _dates)
            r_s = pd.Series(r, index=df.index)
            cols[f"{ent}_glicko"] = r
            cols[f"{ent}_glicko_rd"] = rd
            cols[f"has_{ent}_glicko"] = has
            cols[f"{ent}_glicko_vs_field"] = (
                r_s - r_s.groupby(race_ids, sort=False).transform("mean")
            ).to_numpy()

    if getattr(config, "GLICKO_MARGIN", False) and "lengths_behind" in df.columns:
        lb = pd.to_numeric(df["lengths_behind"], errors="coerce").to_numpy(dtype=float)
        r, rd, has, _ = _run(
            _glicko_pass, "margin",
            horse_keys, _finish_pos, _dates, race_ids_ordered,
            race_group_indices, margin=lb,
        )
        r_s = pd.Series(r, index=df.index)
        cols["horse_glicko_margin"] = r
        cols["horse_glicko_margin_rd"] = rd
        cols["horse_glicko_margin_vs_field"] = (
            r_s - r_s.groupby(race_ids, sort=False).transform("mean")
        ).to_numpy()
        cols["horse_glicko_margin_edge"] = r - glicko_col

    if getattr(config, "GLICKO2_ENABLED", False):
        r, rd, vol, has = _run(
            _glicko2_pass, "glicko2",
            horse_keys, _finish_pos, _dates, race_ids_ordered, race_group_indices,
        )
        r_s = pd.Series(r, index=df.index)
        v_s = pd.Series(vol, index=df.index)
        cols["horse_glicko2"] = r
        cols["horse_glicko2_rd"] = rd
        cols["horse_glicko2_vol"] = vol
        cols["horse_glicko2_vs_field"] = (
            r_s - r_s.groupby(race_ids, sort=False).transform("mean")
        ).to_numpy()
        cols["horse_glicko2_vol_vs_field"] = (
            v_s - v_s.groupby(race_ids, sort=False).transform("mean")
        ).to_numpy()

    if getattr(config, "TRUESKILL_ENABLED", False):
        mu, sg, has = _run(
            _trueskill_pass, "trueskill",
            horse_keys, _finish_pos, _dates, race_ids_ordered, race_group_indices,
        )
        mu_s = pd.Series(mu, index=df.index)
        cols["horse_ts_mu"] = mu
        cols["horse_ts_sigma"] = sg
        cols["horse_ts_lcb"] = mu - 3.0 * sg
        cols["horse_ts_mu_vs_field"] = (
            mu_s - mu_s.groupby(race_ids, sort=False).transform("mean")
        ).to_numpy()

    if getattr(config, "GLICKO_DIMENSIONAL", False):
        MIN_DIM_RACES = 3  # below this, fall back to the global horse rating
        dims = [
            (suffix, df[col].values)
            for suffix, col in [
                ("surface", "surface"),
                ("rt", "race_type"),
                ("dist_cat", "dist_category"),
            ]
            if col in df.columns
        ]
        for suffix, vals in dims:
            keys = [
                (hk, v) if hk is not None and pd.notna(v) else None
                for hk, v in zip(horse_keys, vals)
            ]
            r, rd, _, cnt = _run(
                _glicko_pass, f"dim_{suffix}",
                keys, _finish_pos, _dates, race_ids_ordered, race_group_indices,
            )
            use_dim = (cnt >= MIN_DIM_RACES) & np.isfinite(r)
            r_eff = np.where(use_dim, r, glicko_col)
            rd_eff = np.where(use_dim, rd, rd_col)
            cols[f"horse_glicko_{suffix}"] = r_eff
            cols[f"horse_glicko_{suffix}_rd"] = rd_eff
            cols[f"horse_glicko_{suffix}_edge"] = r_eff - glicko_col

    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    logger.info("Glicko complete: %d feature columns", len(cols))
    if return_state:
        return df, _out
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
