"""
Horse Rating Systems
====================
Dynamic strength ratings for horses (Elo, margin Elo, Glicko-1/2, TrueSkill)
and value-added ratings for jockeys and trainers, computed from head-to-head
race results.

The key insight: if Horse A beats Horse B, and Horse B beats Horse C,
then Horse A should be rated above Horse C — *even if they never met*.
Traditional per-horse win-rate features cannot capture this transitive
strength.  Ratings propagate strength through the entire population.

Every column is the value an entity carried *into* a race: a row never sees
its own result.

Elo
---
After each race every pair of participants is compared::

    E_ij = 1 / (1 + 10^((R_j - R_i) / 400))   # expected score of i vs j
    S_ij = 1 if i beat j, 0.5 if tied, 0 if lost

    R_i_new = R_i + K_i * mean_j(S_ij - E_ij)

``K_i`` decays with the horse's experience. Its size must suit the
400-point scale: a K that is too small keeps every rating near its start,
so every expected score sits near 0.5 and the rating degenerates into a
tally of fields beaten instead of a strength estimate that adjusts for the
opposition.

Every horse starts at the same rating. Starting debutants relative to the
rated runners they meet was measured and rejected (experiment 045): maiden
fields are mostly recent debutants, so any field-relative start compounds
from cohort to cohort, with a fixed gap or a learned one.

Jockeys and trainers carry *offsets* added to the horse's rating, updated
from how their runners finish relative to what the horse ratings expected.
They measure value added, not the quality of the horses they are given.

Usage::

    from src.ratings import compute_elo_features

    df = compute_elo_features(df)
    # Adds: horse_elo, jockey_elo, trainer_elo (ratings at the *start*
    # of each race, before that race's update).
"""

import logging

import numpy as np
import pandas as pd

import config
from src.trueskill_kernel import rate_races

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
K_FACTOR = 16.0          # legacy signature default; adaptive K is always used
MIN_RACES_FOR_RATING = 0  # include from the first race

# Adaptive K-factor settings (from config, with fallbacks). Functions read
# these module bindings at call time so sweeps can patch them.
K_BASE = getattr(config, "ELO_K_BASE", 256.0)
K_MIN = getattr(config, "ELO_K_MIN", 128.0)
K_DECAY = getattr(config, "ELO_K_DECAY", 0.05)
MOMENTUM_ALPHA = getattr(config, "ELO_MOMENTUM_ALPHA", 0.3)  # EMA smoothing for momentum
# Momentum fades towards 0 while a horse is off the track.
MOMENTUM_HALF_LIFE_DAYS = getattr(config, "ELO_MOMENTUM_HALF_LIFE_DAYS", 180.0)
# Jockey/trainer value-added offsets (rating points added to the horse).
ENTITY_K_BASE = getattr(config, "ELO_ENTITY_K_BASE", 32.0)
ENTITY_K_MIN = getattr(config, "ELO_ENTITY_K_MIN", 4.0)
# Pseudo-runs shrinking dimension-specific edges towards 0 (must be > 0).
DIM_PRIOR_RUNS = getattr(config, "RATING_DIM_PRIOR_RUNS", 3.0)
# Non-finishers (fell, pulled up, ...) lose to every finisher and tie with
# each other in the Elo, Glicko and TrueSkill updates. Margin Elo always
# includes them with its own lengths penalty.
DNF_AS_LOSS = getattr(config, "RATING_DNF_AS_LOSS", True)

# Converts a mean score residual (actual minus expected, averaged over
# opponents) into rating points: the expected-score slope at evenly
# matched ratings is ln(10)/400 * 1/4.
RESID_TO_POINTS = 1600.0 / np.log(10.0)


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


def _adaptive_k(n_races, k_base: float | None = None, k_min: float | None = None):
    """
    Compute an adaptive K-factor based on experience.

    Young / unexposed horses should have a higher K (their rating changes
    more with each result), while established horses need a lower K.

    Uses exponential decay:  K = K_MIN + (K_BASE - K_MIN) * exp(-K_DECAY * n)
    """
    k_base = K_BASE if k_base is None else k_base
    k_min = K_MIN if k_min is None else k_min
    return k_min + (k_base - k_min) * np.exp(-K_DECAY * np.asarray(n_races, dtype=float))


# ── Margin Elo config ─────────────────────────────────────────────────────
MARGIN_SCALE = getattr(config, "MARGIN_ELO_SCALE", 5.0)     # base lengths at which score ≈ 0.82
MARGIN_REF_DIST = getattr(config, "MARGIN_ELO_REF_DIST", 8.0)  # reference distance (furlongs)
DNF_PENALTY_LB = getattr(config, "MARGIN_ELO_DNF_PENALTY", 30.0)  # virtual lb for non-finishers


# ── Core helpers ──────────────────────────────────────────────────────────

def _race_outcome(fp_race: np.ndarray, include_dnf: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows that take part in a rating update, and their effective positions.

    Finishers (position >= 1) always take part. With ``include_dnf``,
    non-finishers (position 0) in a race that has a finisher take part as a
    dead heat for last place. Pending cards (no finisher) never update.
    """
    fp_race = np.asarray(fp_race, dtype=float)
    fin = np.isfinite(fp_race) & (fp_race > 0)
    if include_dnf and fin.any():
        dnf = np.isfinite(fp_race) & (fp_race == 0)
        if dnf.any():
            return fin | dnf, np.where(dnf, fp_race[fin].max() + 1.0, fp_race)
    return fin, fp_race


def _pair_scores(pos: np.ndarray) -> np.ndarray:
    """S[i, j] = 1 if i finished ahead of j, 0.5 if level, 0 if behind."""
    a, b = pos[:, None], pos[None, :]
    return np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))


def _elo_expected(r: np.ndarray) -> np.ndarray:
    """E[i, j] = probability that i beats j given their ratings."""
    return 1.0 / (1.0 + 10.0 ** ((r[None, :] - r[:, None]) / 400.0))


def _mean_residual(S: np.ndarray, E: np.ndarray, same: np.ndarray | None = None) -> np.ndarray:
    """Mean actual-minus-expected score of each runner against its opponents.

    ``same`` marks pairs that are not opponents (two runners of one trainer).
    A row without any opponent gets NaN: the race says nothing about it.
    """
    opp = ~np.eye(len(S), dtype=bool)
    if same is not None:
        opp &= ~same
    n_opp = opp.sum(axis=1)
    total = np.where(opp, S - E, 0.0).sum(axis=1)
    return np.where(n_opp > 0, total / np.maximum(n_opp, 1), np.nan)


def _margin_score_matrix(lb: np.ndarray, margin_scale: float) -> np.ndarray:
    """
    Continuous actual score S[i, j] from lengths-behind margins.

    Returns values in (0, 1):
        - i beats j (lb_i < lb_j): score > 0.5, approaching 1.0 for big margins
        - i loses to j: score < 0.5, approaching 0.0 for big margins
        - Dead heat: exactly 0.5

    Uses ``1 - exp(-margin / margin_scale)`` to map the gap size
    to a [0, 0.5) bonus/penalty on top of the 0.5 baseline.

    *margin_scale* is distance-conditioned: longer races use a larger
    scale so that the same number of lengths produces a smaller Elo
    swing than in a sprint.
    """
    margin = lb[None, :] - lb[:, None]  # positive when i beat j
    mov = 0.5 * (1.0 - np.exp(-np.abs(margin) / margin_scale))
    return 0.5 + np.sign(margin) * mov


def _distance_margin_scale(distance_furlongs: float) -> float:
    """Return a margin scale proportional to race distance.

    scale = MARGIN_SCALE * (distance / REF_DIST)

    At REF_DIST (default 8f ≈ 1 mile) the scale equals the base
    MARGIN_SCALE.  A 5f sprint gets a tighter scale so a 1-length gap has
    *more* rating impact; a 24f chase gets a wider scale, reflecting that
    large lb margins are routine at staying distances.
    """
    if not np.isfinite(distance_furlongs) or distance_furlongs <= 0:
        return MARGIN_SCALE
    return MARGIN_SCALE * (distance_furlongs / MARGIN_REF_DIST)


def _decayed(value: float, last_dt, now, half_life_days: float) -> float:
    """``value`` faded by half for every ``half_life_days`` since ``last_dt``."""
    if last_dt is None:
        return value
    days = max(float((now - last_dt) / np.timedelta64(1, "D")), 0.0)
    return value * 0.5 ** (days / half_life_days)


class _DimensionEdge:
    """Shrunk dimension-specific edge (surface, race type, trip) for one keying.

        edge = RESID_TO_POINTS * (sum_d / (n_d + n0) - sum_all / (n_all + n0))

    ``sum_d``/``n_d`` accumulate an entity's mean score residuals (actual
    minus expected against the field, from its parent rating) in dimension
    value ``d``; ``sum_all``/``n_all`` over every run with a known value of
    that dimension. An entity that has only run under one value has an edge
    of exactly 0, and the edge grows smoothly with evidence — unlike a
    separate rating pool per dimension, which mostly duplicates the parent
    rating and has to fall back to it below some run count.
    """

    def __init__(self, state: dict | None = None):
        state = state or {}
        self.dim = dict(state.get("dim", {}))
        self.all = dict(state.get("all", {}))

    def edge(self, key, value) -> float:
        sd, nd = self.dim.get((key, value), (0.0, 0))
        sa, na = self.all.get(key, (0.0, 0))
        return RESID_TO_POINTS * (sd / (nd + DIM_PRIOR_RUNS) - sa / (na + DIM_PRIOR_RUNS))

    def add(self, key, value, resid: float) -> None:
        sd, nd = self.dim.get((key, value), (0.0, 0))
        self.dim[(key, value)] = (sd + resid, nd + 1)
        sa, na = self.all.get(key, (0.0, 0))
        self.all[key] = (sa + resid, na + 1)

    def state(self) -> dict:
        return {"dim": self.dim, "all": self.all}


def _dim_value(v):
    """Hashable dimension value, or None when unknown."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    return int(v) if isinstance(v, (float, np.floating)) else v


def _dimension_edge_sweep(
    keys: list, values, resid: np.ndarray,
    race_ids_ordered: np.ndarray, race_group_indices: dict,
    state: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Chronological pre-race :class:`_DimensionEdge` values for given residuals.

    ``resid`` holds each row's mean score residual from the parent rating
    pass (NaN where the row did not update it).
    """
    tracker = _DimensionEdge(state)
    out = np.full(len(keys), np.nan)
    for race_id in race_ids_ordered:
        idx = race_group_indices[race_id]
        for i in idx:
            v = _dim_value(values[i])
            if keys[i] is not None and v is not None:
                out[i] = tracker.edge(keys[i], v)
        for i in idx:
            v = _dim_value(values[i])
            if keys[i] is not None and v is not None and np.isfinite(resid[i]):
                tracker.add(keys[i], v, float(resid[i]))
    return out, tracker.state()


def _entity_residuals(R: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Per-row mean residual against runners of *other* entities.

    ``codes`` are integer entity ids (-1 = unknown, which never updates).
    Runners sharing an entity (a trainer's stablemates) are not opponents:
    their relative finish says nothing about that entity.
    """
    same = codes[:, None] == codes[None, :]
    res = _mean_residual(R, np.zeros_like(R), same)
    return np.where(codes >= 0, res, np.nan)


def _apply_entity_update(
    rows: np.ndarray, res: np.ndarray, keys: np.ndarray,
    offsets: dict, counts: dict, delta_col: np.ndarray | None,
) -> None:
    """Move each entity's offset by its runners' mean residual, once per race."""
    by_key: dict = {}
    for i, r in zip(rows, res):
        if np.isfinite(r):
            by_key.setdefault(keys[i], []).append((i, r))
    for key, items in by_key.items():
        n = counts.get(key, 0)
        k = _adaptive_k(n, ENTITY_K_BASE, ENTITY_K_MIN)
        delta = float(k * np.mean([r for _, r in items]))
        offsets[key] = offsets.get(key, 0.0) + delta
        counts[key] = n + 1
        if delta_col is not None:
            for i, _ in items:
                delta_col[i] = delta


# ── Public API ────────────────────────────────────────────────────────────

def _compute_elo_features(
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

    New columns added (selection):
        - ``horse_elo``   — horse's Elo rating entering this race
        - ``jockey_elo`` / ``trainer_elo`` — 1500 + the entity's value-added
          offset: rating points its runners finish above what their horse
          ratings expect
        - ``horse_elo_momentum`` — time-faded EMA of recent performances
          relative to the rating, in rating points
        - ``horse_elo_{surface,rt,dist_cat}`` and ``_edge`` — rating plus a
          shrunk dimension-specific edge (``jockey_elo_rt`` likewise)
        - ``combined_elo`` — horse rating plus jockey and trainer offsets
        - ``horse_elo_delta`` / ``jockey_elo_delta`` — change after this
          race (post-race; excluded from models)

    Args:
        df: DataFrame with ``race_id``, ``race_date``, ``finish_position``,
            ``horse_name``, and optionally ``jockey`` and ``trainer``.
        k: Unused; kept for call compatibility (K is adaptive).

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
    # restarting from scratch. The end-state is returned when
    # return_state=True so a feature store can persist it and re-feed it on
    # the next append — making the sweep process only the new rows while
    # producing byte-identical pre-race ratings.
    _seed_state = elo_state or {}

    def _seed(name: str) -> dict:
        return dict(_seed_state.get(name) or {})

    horse_ratings = _seed("horse_ratings")
    horse_race_counts = _seed("horse_race_counts")
    horse_momentum = _seed("horse_momentum")
    horse_last_dt = _seed("horse_last_dt")
    jockey_offsets = _seed("jockey_offsets")
    jockey_race_counts = _seed("jockey_race_counts")
    trainer_offsets = _seed("trainer_offsets")
    trainer_race_counts = _seed("trainer_race_counts")
    horse_margin_ratings = _seed("horse_margin_ratings")
    horse_margin_race_counts = _seed("horse_margin_race_counts")
    horse_margin_momentum = _seed("horse_margin_momentum")
    horse_margin_last_dt = _seed("horse_margin_last_dt")

    n_rows = len(df)
    horse_elo_col = np.full(n_rows, np.nan)
    jockey_off_col = np.full(n_rows, np.nan)
    trainer_off_col = np.full(n_rows, np.nan)
    horse_delta_col = np.zeros(n_rows)
    jockey_delta_col = np.zeros(n_rows)
    has_horse_elo_col = np.zeros(n_rows, dtype=np.int8)
    has_jockey_elo_col = np.zeros(n_rows, dtype=np.int8)
    has_trainer_elo_col = np.zeros(n_rows, dtype=np.int8)
    horse_momentum_col = np.full(n_rows, np.nan)
    horse_resid_col = np.full(n_rows, np.nan)
    jockey_resid_col = np.full(n_rows, np.nan)

    # Group by race (preserving chronological order)
    race_ids_ordered = df.drop_duplicates("race_id", keep="first")["race_id"].values
    race_groups = df.groupby("race_id", sort=False).indices
    dates = df["race_date"].values

    # Pre-extract columns as numpy arrays for fast inner-loop access
    _horse_keys = _entity_keys(df, id_col="horse_id", name_col="horse_name", prefix="horse")
    _h_ok = pd.notna(pd.Series(_horse_keys)).to_numpy()
    _finish_pos = (
        pd.to_numeric(df["finish_position"], errors="coerce").to_numpy(dtype=float)
        if "finish_position" in df.columns else None
    )
    _jockey_keys = _entity_keys(df, id_col="jockey_id", name_col="jockey", prefix="jockey") if has_jockey else None
    _trainer_keys = _entity_keys(df, id_col="trainer_id", name_col="trainer", prefix="trainer") if has_trainer else None
    # Integer ids (-1 = unknown) for same-entity tests inside a race.
    _jockey_codes = pd.factorize(pd.Series(_jockey_keys))[0] if has_jockey else None
    _trainer_codes = pd.factorize(pd.Series(_trainer_keys))[0] if has_trainer else None

    # ── Dimensional edges ────────────────────────────────────────────
    # Shrunk per-dimension edges on top of the global rating (see
    # _DimensionEdge), for surface, race type and distance category.
    dims = {
        name: (df[col].values, _DimensionEdge(_seed_state.get(f"horse_{name}_edge")),
               np.full(n_rows, np.nan))
        for name, col in [("surf", "surface"), ("rt", "race_type"), ("dc", "dist_category")]
        if col in df.columns
    }
    _race_types = df["race_type"].values if "race_type" in df.columns else None
    jockey_rt = _DimensionEdge(_seed_state.get("jockey_rt_edge"))
    jockey_rt_edge_col = np.full(n_rows, np.nan)
    # A jockey's edge subtracts their all-race-type mean, which every ride
    # moves, so a day-start gather per race type would still see the day's
    # earlier rides of other types. Updates are held until the day ends.
    jockey_rt_pending: list = []
    jockey_rt_day = None

    # ── Margin Elo state ─────────────────────────────────────────────
    # Separate rating pool: penalises for beaten lengths and DNF.
    horse_margin_elo_col = np.full(n_rows, np.nan)
    horse_margin_delta_col = np.zeros(n_rows)
    horse_margin_momentum_col = np.full(n_rows, np.nan)
    from src.result_margins import margin_values
    _lengths_behind = (
        pd.to_numeric(margin_values(df), errors="coerce").to_numpy(dtype=float)
        if "lengths_behind" in df.columns else None
    )
    _dist_furlongs = (
        pd.to_numeric(df["distance_furlongs"], errors="coerce").to_numpy(dtype=float)
        if "distance_furlongs" in df.columns else None
    )

    for race_id in race_ids_ordered:
        idx = race_groups[race_id]
        day = dates[idx[0]]
        known = idx[_h_ok[idx]]
        if jockey_rt_pending and day != jockey_rt_day:
            for args in jockey_rt_pending:
                jockey_rt.add(*args)
            jockey_rt_pending.clear()
        jockey_rt_day = day

        # --- Record *pre-race* ratings (no look-ahead) ---
        # Must happen for ALL races including future/unfinished ones
        # so live racecards get their accumulated ratings.
        for i in known:
            hk = _horse_keys[i]
            horse_elo_col[i] = horse_ratings.get(hk, DEFAULT_RATING)
            horse_margin_elo_col[i] = horse_margin_ratings.get(hk, DEFAULT_RATING)
            has_horse_elo_col[i] = np.int8(horse_race_counts.get(hk, 0) > 0)
            horse_momentum_col[i] = _decayed(
                horse_momentum.get(hk, 0.0), horse_last_dt.get(hk), day, MOMENTUM_HALF_LIFE_DAYS)
            horse_margin_momentum_col[i] = _decayed(
                horse_margin_momentum.get(hk, 0.0), horse_margin_last_dt.get(hk), day,
                MOMENTUM_HALF_LIFE_DAYS)
            for values, tracker, col in dims.values():
                v = _dim_value(values[i])
                if v is not None:
                    col[i] = tracker.edge(hk, v)
        for i in idx:
            if has_jockey and _jockey_codes[i] >= 0:
                jk = _jockey_keys[i]
                jockey_off_col[i] = jockey_offsets.get(jk, 0.0)
                has_jockey_elo_col[i] = np.int8(jockey_race_counts.get(jk, 0) > 0)
                v = _dim_value(_race_types[i]) if _race_types is not None else None
                if v is not None:
                    jockey_rt_edge_col[i] = jockey_rt.edge(jk, v)
            if has_trainer and _trainer_codes[i] >= 0:
                tk = _trainer_keys[i]
                trainer_off_col[i] = trainer_offsets.get(tk, 0.0)
                has_trainer_elo_col[i] = np.int8(trainer_race_counts.get(tk, 0) > 0)

        # Skip races without finish positions (e.g. racecards / future)
        if _finish_pos is None:
            continue
        fp_race = _finish_pos[idx]
        if not (np.isfinite(fp_race) & (fp_race > 0)).any():
            continue

        # ── Horse Elo, jockey/trainer offsets, dimensional edges ──
        part, pos = _race_outcome(fp_race, DNF_AS_LOSS)
        part &= _h_ok[idx]
        if part.sum() >= 2:
            p = idx[part]
            S = _pair_scores(pos[part])
            r = horse_elo_col[p]
            resid = _mean_residual(S, _elo_expected(r))
            n_prev = np.array([horse_race_counts.get(_horse_keys[i], 0) for i in p])
            delta = _adaptive_k(n_prev) * resid
            for i, rr, d, res, n in zip(p, r, delta, resid, n_prev):
                hk = _horse_keys[i]
                horse_ratings[hk] = float(rr + d)
                horse_race_counts[hk] = int(n) + 1
                horse_delta_col[i] = d
                horse_resid_col[i] = res
                # Momentum: EMA of performance relative to the rating, in
                # points, faded over the time since the last run.
                horse_momentum[hk] = (
                    MOMENTUM_ALPHA * res * RESID_TO_POINTS
                    + (1.0 - MOMENTUM_ALPHA) * horse_momentum_col[i]
                )
                horse_last_dt[hk] = day
                for values, tracker, _ in dims.values():
                    v = _dim_value(values[i])
                    if v is not None:
                        tracker.add(hk, v, float(res))

            # Jockey/trainer: residuals against the horse ratings plus the
            # current offsets, so each offset learns only what the horse
            # ratings and the other offset do not already explain.
            if has_jockey or has_trainer:
                s = r.copy()
                if has_jockey:
                    s += np.nan_to_num(np.array([jockey_offsets.get(_jockey_keys[i], 0.0)
                                                 if _jockey_codes[i] >= 0 else 0.0 for i in p]))
                if has_trainer:
                    s += np.nan_to_num(np.array([trainer_offsets.get(_trainer_keys[i], 0.0)
                                                 if _trainer_codes[i] >= 0 else 0.0 for i in p]))
                R = S - _elo_expected(s)
                if has_jockey:
                    jres = _entity_residuals(R, _jockey_codes[p])
                    jockey_resid_col[p] = jres
                    _apply_entity_update(p, jres, _jockey_keys, jockey_offsets,
                                         jockey_race_counts, jockey_delta_col)
                    if _race_types is not None:
                        for i, res in zip(p, jres):
                            v = _dim_value(_race_types[i])
                            if v is not None and np.isfinite(res):
                                jockey_rt_pending.append((_jockey_keys[i], v, float(res)))
                if has_trainer:
                    tres = _entity_residuals(R, _trainer_codes[p])
                    _apply_entity_update(p, tres, _trainer_keys, trainer_offsets,
                                         trainer_race_counts, None)

        # ── Margin Elo: includes non-finishers with penalty ──
        if (_lengths_behind is not None
                and ("beaten_lengths" not in df
                     or np.isfinite(_lengths_behind[idx][fp_race > 0]).all())):
            # Finishers use their lengths behind (winner = 0); non-finishers
            # (position 0 or unknown) get a penalised virtual margin.
            # Non-runners are not in the data (excluded by the scraper).
            lb_race = _lengths_behind[idx]
            finisher_lbs = lb_race[np.isfinite(fp_race) & (fp_race > 0)]
            finite_lbs = finisher_lbs[np.isfinite(finisher_lbs)]
            max_lb = float(finite_lbs.max()) if len(finite_lbs) > 0 else 0.0

            m_rows, m_lbs = [], []
            for j_pos, i in enumerate(idx):
                if not _h_ok[i]:
                    continue
                fp_i = fp_race[j_pos] if np.isfinite(fp_race[j_pos]) else 0
                if fp_i >= 1:
                    lb_i = lb_race[j_pos]
                    if not np.isfinite(lb_i):
                        lb_i = 0.0 if fp_i == 1 else max_lb
                    m_rows.append(i)
                    m_lbs.append(float(lb_i))
                elif fp_i == 0:
                    m_rows.append(i)
                    m_lbs.append(max_lb + DNF_PENALTY_LB)

            if len(m_rows) >= 2:
                m_rows = np.asarray(m_rows)
                _race_dist = float(_dist_furlongs[idx[0]]) if _dist_furlongs is not None else MARGIN_REF_DIST
                S = _margin_score_matrix(np.asarray(m_lbs), _distance_margin_scale(_race_dist))
                r = horse_margin_elo_col[m_rows]
                resid = _mean_residual(S, _elo_expected(r))
                n_prev = np.array([horse_margin_race_counts.get(_horse_keys[i], 0) for i in m_rows])
                delta = _adaptive_k(n_prev) * resid
                for i, rr, d, res, n in zip(m_rows, r, delta, resid, n_prev):
                    hk = _horse_keys[i]
                    horse_margin_ratings[hk] = float(rr + d)
                    horse_margin_race_counts[hk] = int(n) + 1
                    horse_margin_delta_col[i] = d
                    horse_margin_momentum[hk] = (
                        MOMENTUM_ALPHA * res * RESID_TO_POINTS
                        + (1.0 - MOMENTUM_ALPHA) * horse_margin_momentum_col[i]
                    )
                    horse_margin_last_dt[hk] = day

    for args in jockey_rt_pending:
        jockey_rt.add(*args)

    race_ids = df["race_id"]

    # Date-strict view for entities that run several times a day:
    # all of a jockey's/trainer's rides on one day must carry the
    # rating recorded at their FIRST ride of the day — later rides'
    # ratings embed same-day results that don't exist at prediction
    # time. (Horses essentially never run twice a day.)
    _day_dates = pd.to_datetime(df["race_date"]).values
    if has_jockey:
        jockey_off_col = _day_start_gather(jockey_off_col, [_jockey_keys], _day_dates)
        has_jockey_elo_col = _day_start_gather(has_jockey_elo_col, [_jockey_keys], _day_dates)
    if has_trainer:
        trainer_off_col = _day_start_gather(trainer_off_col, [_trainer_keys], _day_dates)
        has_trainer_elo_col = _day_start_gather(has_trainer_elo_col, [_trainer_keys], _day_dates)

    horse_elo = pd.Series(horse_elo_col, index=df.index)
    jockey_elo = pd.Series(DEFAULT_RATING + jockey_off_col, index=df.index)
    trainer_elo = pd.Series(DEFAULT_RATING + trainer_off_col, index=df.index)

    # Relative-to-field features (race-aware)
    race_avg_elo = horse_elo.groupby(race_ids, sort=False).transform("mean")
    race_max_elo = horse_elo.groupby(race_ids, sort=False).transform("max")
    horse_elo_rank = horse_elo.groupby(race_ids, sort=False).rank(
        ascending=False, method="min"
    )

    # Combined strength = the value-added model's own expectation input.
    combined_elo = horse_elo.copy()
    if has_jockey:
        combined_elo += np.nan_to_num(jockey_off_col)
    if has_trainer:
        combined_elo += np.nan_to_num(trainer_off_col)
    combo_avg = combined_elo.groupby(race_ids, sort=False).transform("mean")

    # Momentum relative to field
    horse_mom = pd.Series(horse_momentum_col, index=df.index)
    mom_field_avg = horse_mom.groupby(race_ids, sort=False).transform("mean")

    elo_cols = {
        "horse_elo": horse_elo_col,
        "jockey_elo": jockey_elo.to_numpy(),
        "trainer_elo": trainer_elo.to_numpy(),
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
    for name, out_name in [("surf", "surface"), ("rt", "rt"), ("dc", "dist_cat")]:
        edge = dims[name][2] if name in dims else np.full(n_rows, np.nan)
        elo_cols[f"horse_elo_{out_name}"] = horse_elo_col + edge
        elo_cols[f"horse_elo_{out_name}_edge"] = edge
    elo_cols["jockey_elo_rt"] = jockey_elo.to_numpy() + jockey_rt_edge_col
    elo_cols["jockey_elo_rt_edge"] = jockey_rt_edge_col

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

    logger.info(
        f"Elo complete: {len(horse_ratings)} horses, {len(jockey_offsets)} jockeys, "
        f"{len(trainer_offsets)} trainers rated ({len(dims)} dimensional splits)"
    )

    if return_state:
        out_state = {
            "horse_ratings": horse_ratings,
            "horse_race_counts": horse_race_counts,
            "horse_momentum": horse_momentum,
            "horse_last_dt": horse_last_dt,
            "jockey_offsets": jockey_offsets,
            "jockey_race_counts": jockey_race_counts,
            "trainer_offsets": trainer_offsets,
            "trainer_race_counts": trainer_race_counts,
            "horse_margin_ratings": horse_margin_ratings,
            "horse_margin_race_counts": horse_margin_race_counts,
            "horse_margin_momentum": horse_margin_momentum,
            "horse_margin_last_dt": horse_margin_last_dt,
            "jockey_rt_edge": jockey_rt.state(),
            **{f"horse_{name}_edge": tracker.state() for name, (_, tracker, _) in dims.items()},
        }
        return df, out_state

    return df


# ── Glicko-1 ratings ──────────────────────────────────────────────────────
# Glickman (1999). Adds a per-horse rating deviation (RD): uncertainty
# that shrinks with evidence and *grows during layoffs* — something the
# adaptive-K Elo above cannot express (K only decays with run count).
# Each race is one rating period of n-1 pairwise games. Those games are
# not independent — one performance decides all of them — so their
# evidence is discounted by (n-1)^-GLICKO_EVIDENCE_POWER; counted in full,
# a debut winner in a 20-runner field jumped ~400 points.

GLICKO_RATING_INIT = 1500.0
GLICKO_RD_INIT = getattr(config, "GLICKO_RD_INIT", 350.0)
GLICKO_RD_MIN = getattr(config, "GLICKO_RD_MIN", 50.0)
# RD inflation: after t months idle, RD -> sqrt(RD^2 + C^2 * t).
# C=70 returns a fully-known horse (RD 50) to ~max uncertainty in ~2 years.
GLICKO_C = getattr(config, "GLICKO_C", 70.0)
GLICKO_EVIDENCE_POWER = getattr(config, "GLICKO_EVIDENCE_POWER", 0.25)

_GLICKO_Q = np.log(10.0) / 400.0


def _evidence_weight(n_opp: np.ndarray) -> np.ndarray:
    """Weight per pairwise game so a race counts as n^(1-power) games."""
    return np.where(n_opp > 0, np.maximum(n_opp, 1) ** (-GLICKO_EVIDENCE_POWER), 0.0)


def _glicko_race_update(
    r: np.ndarray, rd: np.ndarray, fp: np.ndarray,
    exclude: np.ndarray | None = None,
    S_override: np.ndarray | None = None,
    with_residual: bool = False,
):
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
        with_residual: also return each row's mean actual-minus-expected
            score (NaN without opponents).

    Returns:
        (new_ratings, new_rds[, residuals])
    """
    m = len(r)
    q = _GLICKO_Q
    g = 1.0 / np.sqrt(1.0 + 3.0 * (q ** 2) * (rd ** 2) / (np.pi ** 2))

    # E[i, j] = expected score of i against j (uses opponent's g)
    diff = r[:, None] - r[None, :]
    E = 1.0 / (1.0 + 10.0 ** (-(g[None, :] * diff) / 400.0))
    # S[i, j] = actual score of i against j
    S = S_override if S_override is not None else _pair_scores(np.asarray(fp, dtype=float))

    off_diag = ~np.eye(m, dtype=bool)
    if exclude is not None:
        off_diag &= ~exclude
    w = _evidence_weight(off_diag.sum(axis=1))
    g_sq_E = (g[None, :] ** 2) * E * (1.0 - E)
    d2_inv = w * (q ** 2) * np.where(off_diag, g_sq_E, 0.0).sum(axis=1)
    sum_term = w * np.where(off_diag, g[None, :] * (S - E), 0.0).sum(axis=1)

    denom = 1.0 / (rd ** 2) + d2_inv
    new_r = r + (q / denom) * sum_term
    new_rd = np.clip(np.sqrt(1.0 / denom), GLICKO_RD_MIN, GLICKO_RD_INIT)
    if with_residual:
        return new_r, new_rd, _mean_residual(S, E, None if exclude is None else exclude)
    return new_r, new_rd


def _glicko_pass(
    keys: list,
    fp: np.ndarray | None,
    dates: np.ndarray,
    race_ids_ordered: np.ndarray,
    race_group_indices: dict,
    margin: np.ndarray | None = None,
    *,
    distance: np.ndarray | None = None,
    resid_out: np.ndarray | None = None,
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
            game scores become sigmoid((lb_j − lb_i) / scale) instead of
            binary win/lose — a 10-length thrashing moves ratings more than
            a nose verdict. The scale grows with ``distance`` (furlongs)
            as in margin Elo. Pairs with missing lengths fall back to the
            binary score.
        resid_out: optional per-row array receiving each updated row's mean
            score residual (for dimensional edges).

    Returns:
        Per-row *pre-race* (rating, rd, has_prior_run, prior_run_count).
        When ``return_state`` is set, also returns the carried state dict so
        an incremental run can re-seed and sweep only new rows.
    """
    state = state or {}
    ratings: dict = dict(state.get("ratings", {}))
    rds: dict = dict(state.get("rds", {}))
    last_dt: dict = dict(state.get("last_dt", {}))
    counts: dict = dict(state.get("counts", {}))

    n = len(keys)
    r_col = np.full(n, np.nan)
    rd_col = np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)
    cnt_col = np.zeros(n, dtype=np.int32)

    for race_id in race_ids_ordered:
        idx = race_group_indices[race_id]
        race_dt = dates[idx[0]]

        # Pre-race: compute uncertainty from the last rated result. Do not
        # change carried RD until there is an actual rating update: otherwise
        # an unrated/non-finish appearance counts the same idle days again
        # next time. Multiple runners for an entity share one pre-race RD.
        for i in idx:
            k = keys[i]
            if k is None:
                continue
            if k in ratings:
                idle_days = (race_dt - last_dt[k]) / np.timedelta64(1, "D")
                months = max(float(idle_days), 0.0) / 30.44
                r_col[i] = ratings[k]
                rd_col[i] = min(
                    float(np.sqrt(rds[k] ** 2 + (GLICKO_C ** 2) * months)),
                    GLICKO_RD_INIT,
                )
                has_col[i] = 1
                cnt_col[i] = counts[k]
            else:
                r_col[i] = GLICKO_RATING_INIT
                rd_col[i] = GLICKO_RD_INIT

        # Post-race update
        if fp is None:
            continue
        part, pos = _race_outcome(fp[idx], DNF_AS_LOSS)
        upd =[j for j in range(len(idx)) if part[j] and keys[idx[j]] is not None]
        ukeys = [keys[idx[j]] for j in upd]
        if len(set(ukeys)) < 2:
            continue
        m = len(upd)
        rows = idx[upd]
        r_arr = r_col[rows]
        rd_arr = rd_col[rows]
        fp_arr = pos[upd]
        exclude = None
        if len(set(ukeys)) < m:
            karr = np.empty(m, dtype=object)
            karr[:] = ukeys
            exclude = karr[:, None] == karr[None, :]
        S_override = None
        if margin is not None:
            lb = margin[rows].astype(float)
            both = np.isfinite(lb[:, None]) & np.isfinite(lb[None, :])
            scale = (_distance_margin_scale(float(distance[idx[0]]))
                     if distance is not None else MARGIN_SCALE)
            S_margin = 1.0 / (1.0 + np.exp(
                -np.clip((np.nan_to_num(lb)[None, :] - np.nan_to_num(lb)[:, None]) / scale, -60, 60)
            ))
            S_override = np.where(both, S_margin, _pair_scores(fp_arr))
        new_r, new_rd, resid = _glicko_race_update(
            r_arr, rd_arr, fp_arr, exclude=exclude, S_override=S_override, with_residual=True
        )
        if resid_out is not None:
            resid_out[rows] = resid
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
    (rating, rd, volatility, has_prior_run) on the Glicko-1 scale.

    Rating periods are months: idle time inflates φ by σ²·t once, with the
    volatility the update itself produces (Glickman's step 6 generalised to
    t periods) — not once before the race and again inside the update.
    """
    tau = float(getattr(config, "GLICKO2_TAU", 0.5))
    phi_init = GLICKO_RD_INIT / _G2_SCALE
    phi_min = GLICKO_RD_MIN / _G2_SCALE

    state = state or {}
    mus: dict = dict(state.get("mus", {}))
    phis: dict = dict(state.get("phis", {}))
    sigmas: dict = dict(state.get("sigmas", {}))
    last_dt: dict = dict(state.get("last_dt", {}))

    n = len(keys)
    r_col = np.full(n, np.nan)
    rd_col = np.full(n, np.nan)
    vol_col = np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)

    for race_id in race_ids_ordered:
        idx = race_group_indices[race_id]
        race_dt = dates[idx[0]]

        # Keep the persisted deviation tied to last_dt until an observed
        # result updates both. Pending and non-finish rows are read-only.
        race_phis, race_months = {}, {}
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
                race_phis[k] = phi_now
                race_months[k] = t
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
        part, pos = _race_outcome(fp[idx], DNF_AS_LOSS)
        upd = [(j, keys[idx[j]]) for j in range(len(idx))
               if part[j] and keys[idx[j]] is not None]
        if len(upd) < 2:
            continue
        m = len(upd)
        mu = np.array([mus.get(k, 0.0) for _, k in upd])
        phi = np.array([race_phis.get(k, phi_init) for _, k in upd])
        phi_base = np.array([phis.get(k, phi_init) for _, k in upd])
        months = np.array([race_months.get(k, 0.0) for _, k in upd])
        sigma = np.array([sigmas.get(k, GLICKO2_SIGMA_INIT) for _, k in upd])
        fp_arr = pos[[j for j, _ in upd]]

        g = 1.0 / np.sqrt(1.0 + 3.0 * phi ** 2 / (np.pi ** 2))
        diff = mu[:, None] - mu[None, :]
        E = 1.0 / (1.0 + np.exp(-np.clip(g[None, :] * diff, -60, 60)))
        S = _pair_scores(fp_arr)
        off = ~np.eye(m, dtype=bool)
        w = _evidence_weight(off.sum(axis=1))
        gE = (g[None, :] ** 2) * E * (1.0 - E)
        v = 1.0 / np.maximum(w * np.where(off, gE, 0.0).sum(axis=1), 1e-12)
        score_sum = w * np.where(off, g[None, :] * (S - E), 0.0).sum(axis=1)
        delta = v * score_sum

        new_sigma = np.array([
            _g2_solve_sigma(delta[i] ** 2, phi_base[i] ** 2, v[i], sigma[i], tau)
            for i in range(m)
        ])
        phi_star = np.minimum(np.sqrt(phi_base ** 2 + new_sigma ** 2 * months), phi_init)
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
# Each race is an n-way free-for-all. The library defaults assume outcomes
# far more deterministic than racing (performance noise beta = half the
# initial skill spread), so the parameters come from config. Uncertainty
# grows with time off the track, as in Glicko: without that, an established
# horse's sigma shrank until it barely moved (~0.14 mu per race after 20
# runs), so neither a layoff nor age-related decline could reopen it.

TRUESKILL_MU = 25.0
TRUESKILL_SIGMA = getattr(config, "TRUESKILL_SIGMA", 25.0 / 3.0)
TRUESKILL_BETA = getattr(config, "TRUESKILL_BETA", 25.0 / 6.0)
TRUESKILL_TAU = getattr(config, "TRUESKILL_TAU", 25.0 / 300.0)
# sigma -> sqrt(sigma^2 + C^2 * months idle), capped at TRUESKILL_SIGMA.
TRUESKILL_C = getattr(config, "TRUESKILL_C", 0.0)


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

    Returns per-row pre-race (mu, sigma, has_prior_run). The carried sigma
    only changes on an update; idle time inflates the pre-race view.
    """
    import trueskill

    env = trueskill.TrueSkill(
        mu=TRUESKILL_MU, sigma=TRUESKILL_SIGMA, beta=TRUESKILL_BETA,
        tau=TRUESKILL_TAU, draw_probability=0.001,
    )
    state = state or {}
    ts_ratings: dict = dict(state.get("ratings", {}))  # key -> (mu, sigma)
    last_dt: dict = dict(state.get("last_dt", {}))

    n = len(keys)
    mu_col = np.full(n, np.nan)
    sigma_col = np.full(n, np.nan)
    has_col = np.zeros(n, dtype=np.int8)

    codes: dict = {}
    entity = np.fromiter(
        (-1 if k is None else codes.setdefault(k, len(codes)) for k in keys),
        dtype=np.int64, count=n,
    )
    names = list(codes)
    state_mu = np.fromiter((ts_ratings.get(k, (0.0,))[0] for k in names), dtype=float, count=len(names))
    state_sigma = np.fromiter((ts_ratings.get(k, (0.0, 0.0))[1] for k in names), dtype=float, count=len(names))
    rated = np.fromiter((k in ts_ratings for k in names), dtype=bool, count=len(names))
    last_seen = np.array(
        [last_dt.get(k, np.datetime64("NaT")) for k in names], dtype=dates.dtype,
    ).astype(np.int64)

    race_rows = [race_group_indices[race_id] for race_id in race_ids_ordered]
    offsets = np.zeros(len(race_rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(rows) for rows in race_rows], dtype=np.int64)
    rows = np.concatenate(race_rows).astype(np.int64) if race_rows else np.empty(0, dtype=np.int64)

    failed = rate_races(
        entity, np.zeros(0) if fp is None else np.asarray(fp, dtype=float), fp is not None,
        dates.astype(np.int64), rows, offsets,
        float(np.timedelta64(1, "D") / np.timedelta64(1, np.datetime_data(dates.dtype)[0])),
        state_mu, state_sigma, rated, last_seen,
        TRUESKILL_MU, TRUESKILL_SIGMA, TRUESKILL_C ** 2, env.beta ** 2, env.tau ** 2,
        trueskill.calc_draw_margin(env.draw_probability, 2, env), trueskill.DELTA, DNF_AS_LOSS,
        mu_col, sigma_col, has_col, 2.0, -2.0,
    )
    if failed >= 0:
        raise ZeroDivisionError(f"TrueSkill update failed in race {race_ids_ordered[failed]}")
    seen = last_seen.astype(dates.dtype)
    for i in np.flatnonzero(rated):
        ts_ratings[names[i]] = (float(state_mu[i]), float(state_sigma[i]))
        last_dt[names[i]] = seen[i]

    if return_state:
        return mu_col, sigma_col, has_col, {"ratings": ts_ratings, "last_dt": last_dt}
    return mu_col, sigma_col, has_col


def _compute_glicko_features(
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

    With ``config.GLICKO_DIMENSIONAL``: ``horse_glicko_{surface,rt,dist_cat}``
    (rating plus a shrunk dimension-specific edge, see ``_DimensionEdge``),
    ``_rd`` (the global RD) and ``_edge``.
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

    _finish_pos = (
        pd.to_numeric(df["finish_position"], errors="coerce").to_numpy(dtype=float)
        if "finish_position" in df.columns else None
    )
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
    horse_resid = np.full(len(df), np.nan)
    glicko_col, rd_col, has_col, _ = _run(
        _glicko_pass, "horse",
        horse_keys, _finish_pos, _dates, race_ids_ordered, race_group_indices,
        resid_out=horse_resid,
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
        from src.result_margins import margin_values
        lb = pd.to_numeric(margin_values(df), errors="coerce").to_numpy(dtype=float)
        dist = (
            pd.to_numeric(df["distance_furlongs"], errors="coerce").to_numpy(dtype=float)
            if "distance_furlongs" in df.columns else None
        )
        r, rd, has, _ = _run(
            _glicko_pass, "margin",
            horse_keys, _finish_pos, _dates, race_ids_ordered,
            race_group_indices, margin=lb, distance=dist,
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
        for suffix, col in [("surface", "surface"), ("rt", "race_type"),
                            ("dist_cat", "dist_category")]:
            if col not in df.columns:
                continue
            label = f"dim_{suffix}"
            edge, st = _dimension_edge_sweep(
                horse_keys, df[col].values, horse_resid,
                race_ids_ordered, race_group_indices, _in.get(label),
            )
            if return_state:
                _out[label] = st
            cols[f"horse_glicko_{suffix}"] = glicko_col + edge
            cols[f"horse_glicko_{suffix}_rd"] = np.where(np.isfinite(edge), rd_col, np.nan)
            cols[f"horse_glicko_{suffix}_edge"] = edge

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


# ── French and UAE races (rating-only) ────────────────────────────────────
# Horses travel between Britain/Ireland and France or the UAE. Their overseas
# races are swept with the domestic ones so a raider arrives with a rating
# earned against its overseas opponents, and a British horse's French run
# updates it too. These rows only update ratings: they are removed before
# the frame is returned, so they never become training or evaluation rows.
FOREIGN_STATE_KEY = "foreign_races"  # {"through": ISO day the sweep has covered}


def _foreign_races(df: pd.DataFrame, state: dict | None):
    """French/UAE runners released before this frame's decision day and not
    already swept into *state* (``None`` when the selected dataset holds no
    foreign races), plus the day the sweep has now covered."""
    from src.source_dataset import current_manifest
    if current_manifest() is None or "horse_id" not in df.columns or df.empty:
        return None, None
    from src.foreign_form import history, version
    through = pd.to_datetime(df["race_date"]).max().normalize()
    if "prediction_as_of" in df.columns:
        decision = pd.to_datetime(df["prediction_as_of"], utc=True, errors="raise").dt.tz_localize(None)
        if pd.notna(decision.min()):
            through = min(through, decision.min().normalize())
    after = ((state or {}).get(FOREIGN_STATE_KEY) or {}).get("through")
    if version() is None:
        return None, through if after is None else max(through, pd.Timestamp(after))
    foreign = history(through=through, after=after)
    if state and after is None and not foreign.empty:
        raise ValueError("Rating state predates foreign races; rebuild the feature store")
    if after is not None:
        through = max(through, pd.Timestamp(after))
    return foreign, through


def _interleave_foreign(df: pd.DataFrame, foreign: pd.DataFrame) -> pd.DataFrame:
    """*df* (already in sweep order) with foreign runners placed by release time."""
    from src.feature_semantics import distance_category
    rows = pd.DataFrame({
        "race_id": ("foreign:" + foreign.provider + ":" + foreign.native_race_id).to_numpy(),
        "horse_id": foreign.horse_id.astype(df["horse_id"].dtype).to_numpy(),
        "horse_name": foreign.horse_name.to_numpy(),
        "race_date": pd.to_datetime(foreign.race_date).to_numpy(),
        "_event_dt": foreign._event_dt.to_numpy(),
        "finish_position": foreign.finish_position.to_numpy(),
        "lengths_behind": foreign.lengths_behind.to_numpy(),
        "beaten_lengths": foreign.beaten_lengths.to_numpy(),
        "distance_furlongs": foreign.distance_furlongs.to_numpy(),
        "race_type": foreign.race_type.to_numpy(),
        "surface": foreign.surface.to_numpy(),
        "num_runners": foreign.num_runners.to_numpy(),
    })
    rows["dist_category"] = distance_category(rows["distance_furlongs"])
    rows = rows.reindex(columns=df.columns).sort_values(["_event_dt", "race_id"], kind="stable")
    work = pd.concat([df, rows], ignore_index=True)
    work["_foreign_rating_row"] = np.r_[np.zeros(len(df), bool), np.ones(len(rows), bool)]
    order = np.argsort(pd.to_datetime(work["_event_dt"]).to_numpy(), kind="stable")
    return work.iloc[order].reset_index(drop=True)


def _with_foreign_races(core, df, state, return_state, *, sort_domestic, **kwargs):
    foreign, through = _foreign_races(df, state)
    if foreign is None or foreign.empty:
        result = core(df, **kwargs, return_state=return_state)
        if return_state and through is not None:
            result[1][FOREIGN_STATE_KEY] = {"through": through.date().isoformat()}
        return result
    had_event = "_event_dt" in df.columns
    domestic = sort_domestic(df)
    result = core(_interleave_foreign(domestic, foreign), **kwargs, return_state=return_state)
    rated, new_state = result if return_state else (result, None)
    keep = ~rated.pop("_foreign_rating_row").to_numpy(dtype=bool)
    rated = rated.loc[keep].reset_index(drop=True)
    if not had_event and "_event_dt" in rated.columns and "_event_dt" in domestic.columns:
        rated = rated.drop(columns="_event_dt")
    # Concatenating foreign rows can widen a domestic column's type.
    for col in rated.columns.intersection(domestic.columns):
        if rated[col].dtype != domestic[col].dtype:
            rated[col] = rated[col].astype(domestic[col].dtype)
    if return_state:
        new_state[FOREIGN_STATE_KEY] = {"through": through.date().isoformat()}
        return rated, new_state
    return rated


def _elo_order(df):
    out = df
    if "_event_dt" not in out.columns:
        out = out.copy()
        out["_event_dt"] = _event_sort_key(out)
    out["race_date"] = pd.to_datetime(out["race_date"])
    if not pd.to_datetime(out["_event_dt"], errors="coerce").is_monotonic_increasing:
        out = out.sort_values(["_event_dt", "race_id"]).reset_index(drop=True)
    return out


def _glicko_order(df):
    event = (pd.to_datetime(df["_event_dt"], errors="coerce") if "_event_dt" in df.columns
             else _event_sort_key(df))
    out = df.copy()
    out["_event_dt"] = event.to_numpy()
    if not event.is_monotonic_increasing:
        out = out.iloc[np.argsort(event.values, kind="stable")].reset_index(drop=True)
    return out


def compute_elo_features(df: pd.DataFrame, k: float = K_FACTOR, *, elo_state: dict | None = None,
                         return_state: bool = False):
    """Elo features (see ``_compute_elo_features``) swept with French/UAE races."""
    return _with_foreign_races(lambda frame, **kw: _compute_elo_features(frame, k, **kw), df,
                               elo_state, return_state, sort_domestic=_elo_order, elo_state=elo_state)


def compute_glicko_features(df: pd.DataFrame, *, glicko_state: dict | None = None,
                            return_state: bool = False):
    """Glicko/TrueSkill features (see ``_compute_glicko_features``) swept with French/UAE races."""
    return _with_foreign_races(_compute_glicko_features, df, glicko_state, return_state,
                               sort_domestic=_glicko_order,
                               glicko_state=glicko_state)
