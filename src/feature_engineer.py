"""
Feature Engineering Module
==========================
Creates predictive features from processed horse racing data.

Features include:
- Horse historical performance stats (win rate, avg position, form)
- Jockey and trainer statistics
- Elo ratings (horse, jockey, trainer — transitive head-to-head strength)
- Track/distance/going preferences
- Class performance
- Pace and speed figures
- Market indicators (odds-based)
"""

import logging

import numba as nb
import numpy as np
import pandas as pd

from src.ratings import compute_elo_features
from src.track_config import get_track_config, direction_code, shape_code

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


def _horse_key(df: pd.DataFrame) -> str:
    """Return best column for grouping horse history (prefer horse_id)."""
    if "horse_id" in df.columns and df["horse_id"].notna().sum() > 0:
        return "horse_id"
    return "horse_name"


# ── Numba-accelerated rolling helpers ────────────────────────────
# The pandas pattern  groupby(key).transform(lambda x: x.shift(1).rolling(w).mean())
# is extremely slow (~80s for 27K rows × 10K groups) because it creates a
# new Series per group.  These Numba kernels operate on pre-sorted numpy
# arrays with group-boundary labels, giving 50-100× speedup.

@nb.njit(cache=True)
def _nb_grouped_shift1(values, group_ids):
    """Shift values by 1 within each group (group_ids must be sorted)."""
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan
    for i in range(1, n):
        if group_ids[i] == group_ids[i - 1]:
            out[i] = values[i - 1]
        else:
            out[i] = np.nan
    return out


@nb.njit(cache=True)
def _nb_rolling_mean(shifted, group_ids, w):
    """Rolling mean of width `w` within each group (min_periods=1)."""
    n = len(shifted)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        c = 0
        for j in range(max(0, i - w + 1), i + 1):
            if group_ids[j] != group_ids[i]:
                continue
            if not np.isnan(shifted[j]):
                s += shifted[j]
                c += 1
        out[i] = s / c if c > 0 else np.nan
    return out


@nb.njit(cache=True)
def _nb_rolling_sum(shifted, group_ids, w):
    """Rolling sum of width `w` within each group (min_periods=1)."""
    n = len(shifted)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        c = 0
        for j in range(max(0, i - w + 1), i + 1):
            if group_ids[j] != group_ids[i]:
                continue
            if not np.isnan(shifted[j]):
                s += shifted[j]
                c += 1
        out[i] = s if c > 0 else np.nan
    return out


@nb.njit(cache=True)
def _nb_rolling_min(shifted, group_ids, w):
    """Rolling min of width `w` within each group (min_periods=1)."""
    n = len(shifted)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        mn = np.inf
        c = 0
        for j in range(max(0, i - w + 1), i + 1):
            if group_ids[j] != group_ids[i]:
                continue
            v = shifted[j]
            if not np.isnan(v):
                if v < mn:
                    mn = v
                c += 1
        out[i] = mn if c > 0 else np.nan
    return out


@nb.njit(cache=True)
def _nb_rolling_max(shifted, group_ids, w):
    """Rolling max of width `w` within each group (min_periods=1)."""
    n = len(shifted)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        mx = -np.inf
        c = 0
        for j in range(max(0, i - w + 1), i + 1):
            if group_ids[j] != group_ids[i]:
                continue
            v = shifted[j]
            if not np.isnan(v):
                if v > mx:
                    mx = v
                c += 1
        out[i] = mx if c > 0 else np.nan
    return out


@nb.njit(cache=True)
def _nb_rolling_std(shifted, group_ids, w, min_periods=2):
    """Rolling std (ddof=1) of width `w` within each group."""
    n = len(shifted)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        ss = 0.0
        c = 0
        for j in range(max(0, i - w + 1), i + 1):
            if group_ids[j] != group_ids[i]:
                continue
            v = shifted[j]
            if not np.isnan(v):
                s += v
                ss += v * v
                c += 1
        if c >= min_periods:
            mean = s / c
            var = (ss - c * mean * mean) / (c - 1) if c > 1 else 0.0
            out[i] = np.sqrt(max(var, 0.0))
        else:
            out[i] = np.nan
    return out


@nb.njit(cache=True)
def _nb_ewma(shifted, group_ids, alpha):
    """EWMA within each group on pre-shifted values."""
    n = len(shifted)
    out = np.empty(n, dtype=np.float64)
    ewma = np.nan
    prev_gid = -9999999  # sentinel
    for i in range(n):
        gid = group_ids[i]
        v = shifted[i]
        if gid != prev_gid:
            # New group — reset
            ewma = v if not np.isnan(v) else np.nan
            prev_gid = gid
        else:
            if not np.isnan(v):
                ewma = v if np.isnan(ewma) else alpha * v + (1.0 - alpha) * ewma
        out[i] = ewma
    return out


@nb.njit(cache=True)
def _nb_runs_since_win(won, group_ids):
    """Count consecutive non-winning runs since last win, per group."""
    n = len(won)
    out = np.empty(n, dtype=np.float64)
    counter = 0.0
    prev_gid = -9999999
    for i in range(n):
        gid = group_ids[i]
        if gid != prev_gid:
            # New group
            out[i] = 0.0
            counter = 0.0 if won[i] == 1 else 1.0
            prev_gid = gid
        else:
            out[i] = counter
            if won[i] == 1:
                counter = 0.0
            else:
                counter += 1.0
    return out


def _encode_groups(df: pd.DataFrame, key) -> np.ndarray:
    """Encode a groupby key into contiguous sorted integer IDs.

    Accepts a column name (str) or a list of column names for
    composite keys.  The DataFrame MUST already be sorted so that
    each group's rows are contiguous.
    """
    if isinstance(key, str):
        return df[key].astype("category").cat.codes.values.astype(np.int64)
    # Composite key: combine columns into a single category
    combined = df[key[0]].astype(str)
    for k in key[1:]:
        combined = combined + "||" + df[k].astype(str)
    return combined.astype("category").cat.codes.values.astype(np.int64)


def _fast_grouped_rolling(
    df: pd.DataFrame,
    group_key,
    value_col: str,
    operations: list[tuple[str, str, int | float]],
) -> dict[str, np.ndarray]:
    """Batch-compute shifted-then-rolled features for one group key.

    Internally sorts by ``(group_key, date, race_id)`` so that each
    entity's rows are contiguous and chronological — required for the
    Numba shift / rolling / EWMA kernels to operate correctly.
    Results are mapped back to the original row order.

    Args:
        df: DataFrame (any row order — sorting is handled internally).
        group_key: Column name (str) or list of column names for grouping.
        value_col: Column to compute rolling stats on.
        operations: List of (output_name, op_type, param) where op_type
            is one of 'mean', 'sum', 'min', 'max', 'std', 'ewma'
            and param is the window size (int) or alpha (float for ewma).

    Returns:
        Dict mapping output_name → numpy array aligned to ``df``'s row order.
    """
    n = len(df)
    # ── Build sort permutation: (group_key, date, race_id) ───────
    _sort_cols = [group_key] if isinstance(group_key, str) else list(group_key)
    for _dc in ("_event_dt", "race_date"):
        if _dc in df.columns:
            _sort_cols.append(_dc)
            break
    if "race_id" in df.columns:
        _sort_cols.append("race_id")

    _perm = (
        df[_sort_cols]
        .reset_index(drop=True)
        .sort_values(_sort_cols, kind="stable")
        .index.values
    )
    _inv = np.empty(n, dtype=np.intp)
    _inv[_perm] = np.arange(n)

    values = df[value_col].values[_perm].astype(np.float64)
    gids = _encode_groups(df, group_key)[_perm]
    shifted = _nb_grouped_shift1(values, gids)

    results = {}
    for out_name, op, param in operations:
        if op == "mean":
            arr = _nb_rolling_mean(shifted, gids, int(param))
        elif op == "sum":
            arr = _nb_rolling_sum(shifted, gids, int(param))
        elif op == "min":
            arr = _nb_rolling_min(shifted, gids, int(param))
        elif op == "max":
            arr = _nb_rolling_max(shifted, gids, int(param))
        elif op == "std":
            arr = _nb_rolling_std(shifted, gids, int(param))
        elif op == "ewma":
            arr = _nb_ewma(shifted, gids, float(param))
        else:
            raise ValueError(f"Unknown rolling op: {op}")
        results[out_name] = arr[_inv]
    return results


def _rolling_stats(
    group: pd.DataFrame,
    col: str,
    windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    """
    Calculate rolling statistics for a given column within a group.
    Uses shift(1) to prevent data leakage (only past data used).
    """
    for w in windows:
        group[f"{prefix}_mean_{w}"] = (
            group[col].shift(1).rolling(w, min_periods=1).mean()
        )
        group[f"{prefix}_std_{w}"] = (
            group[col].shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
    return group


def _bayesian_shrink(
    observed_rate: pd.Series,
    n_samples: pd.Series,
    prior_rate: float = 0.10,
    prior_strength: float = 8.0,
) -> pd.Series:
    """
    Bayesian shrinkage for noisy rate estimates.

    A horse with 1 run and 1 win has observed win_rate = 1.0 which is
    unreliable.  We shrink toward a population prior:

        p_hat = (n * observed + m * prior) / (n + m)

    Args:
        observed_rate: Raw observed rate (e.g. wins / runs).
        n_samples: Number of observations backing the rate.
        prior_rate: Population base-rate (≈0.10 for win, ≈0.30 for place).
        prior_strength: Pseudo-count for the prior (higher = more shrinkage).
    """
    n = n_samples.clip(lower=0)
    return (n * observed_rate + prior_strength * prior_rate) / (n + prior_strength)


def _race_safe_cumcount(df: pd.DataFrame, entity_col) -> pd.Series:
    """Like ``groupby(entity).cumcount()`` but excludes same-race peers.

    Standard ``cumcount()`` counts all prior rows for the entity.  When
    an entity (e.g. a trainer) has multiple runners in the same race,
    the second runner's count includes the first runner — data that
    wouldn't be available at prediction time (all runners are predicted
    simultaneously).  This version counts only **prior races**.
    """
    # Number of entity rows per race (e.g. trainer has 3 runners)
    same_race = df.groupby([entity_col, "race_id"]).cumcount()  # 0-based within (entity, race)
    naive_count = df.groupby(entity_col).cumcount()             # 0-based overall
    # Subtract the within-race index so all same-race runners see
    # the same prior-race count.
    return naive_count - same_race


def _race_safe_cumsum(df: pd.DataFrame, entity_col, value_col: str) -> pd.Series:
    """Like ``groupby(entity)[col].cumsum() - df[col]`` but race-safe.

    Excludes the ENTIRE current race's contribution, not just the current
    row.  For trainers with 3 runners in a race, none of those runners
    see any of the others' outcomes.
    """
    # Per-race total for this entity (e.g. trainer's total wins in this race)
    race_total = df.groupby([entity_col, "race_id"])[value_col].transform("sum")
    # Naive cumsum includes current row; subtract current row AND
    # everything from same-race peers.
    naive_cumsum = df.groupby(entity_col)[value_col].cumsum()
    # Within-race cumsum (partial sums of same-race peers seen so far)
    within_race_cumsum = df.groupby([entity_col, "race_id"])[value_col].cumsum()
    # Result: total accumulated BEFORE this race
    return naive_cumsum - within_race_cumsum


def _time_window_stats(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    value_col: str,
    window_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised calendar-window aggregation (O(N log N) per group).

    For each row, compute the sum of ``value_col`` and the count of rows
    by the same ``entity_col`` within the preceding ``window_days``
    calendar days (exclusive of the current row).

    Uses ``np.searchsorted`` on sorted dates within each group instead
    of an O(N²) inner Python loop.

    Returns:
        (value_sum, count) arrays aligned to ``df.index``.
    """
    val_out = np.zeros(len(df))
    cnt_out = np.zeros(len(df))

    delta = np.timedelta64(window_days, "D")

    # Work with 0-based positional indices to avoid out-of-bounds
    # when the DataFrame's index has gaps from prior filtering.
    df_reset = df[[entity_col, date_col, value_col]].reset_index(drop=True)

    for _, grp in df_reset.groupby(entity_col, sort=False):
        grp = grp.sort_values(date_col)
        idx = grp.index
        dates = grp[date_col].values
        vals = grp[value_col].values.astype(np.float64)
        n = len(grp)
        if n == 0:
            continue

        # Prefix-sum so we can compute range sums in O(1)
        cum_vals = np.empty(n + 1)
        cum_vals[0] = 0.0
        np.cumsum(vals, out=cum_vals[1:])

        # For each row find the earliest prior row inside the window
        cutoffs = dates - delta
        starts = np.searchsorted(dates, cutoffs, side="left")
        arange = np.arange(n)

        # Clamp: we only look at rows *before* the current one
        starts = np.minimum(starts, arange)

        val_out[idx] = cum_vals[arange] - cum_vals[starts]
        cnt_out[idx] = arange - starts

    return val_out, cnt_out


def add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate horse-specific historical features.

    For each horse at each race, these features represent its performance
    in PRIOR races only (no data leakage).
    """
    logger.info("Engineering horse features...")
    df = df.copy()

    hkey = _horse_key(df)

    # For cumulative stats, only use actual results (position > 0)
    # Racecards have finish_position == 0
    results_mask = df["finish_position"] > 0

    # --- Cumulative horse stats (shifted to avoid leakage) ---
    horse_groups = df.groupby(hkey)

    # Number of previous races
    df["horse_prev_races"] = horse_groups.cumcount()

    # Cumulative win rate (vectorised — avoids lambda overhead)
    df["horse_cum_wins"] = horse_groups["won"].cumsum() - df["won"]
    df["horse_win_rate"] = np.where(
        df["horse_prev_races"] > 0,
        df["horse_cum_wins"] / df["horse_prev_races"],
        0,
    )
    # Bayesian-shrunk win rate (reliable even with few runs)
    df["horse_win_rate_shrunk"] = _bayesian_shrink(
        df["horse_win_rate"], df["horse_prev_races"],
        prior_rate=0.10, prior_strength=8.0,
    )

    # Cumulative place rate (top 3)
    df["_placed"] = (df["finish_position"].between(1, 3)).astype(int)
    df["horse_cum_places"] = horse_groups["_placed"].cumsum() - df["_placed"]
    df["horse_place_rate"] = np.where(
        df["horse_prev_races"] > 0,
        df["horse_cum_places"] / df["horse_prev_races"],
        0,
    )
    # Bayesian-shrunk place rate
    df["horse_place_rate_shrunk"] = _bayesian_shrink(
        df["horse_place_rate"], df["horse_prev_races"],
        prior_rate=0.30, prior_strength=8.0,
    )

    # Override the static scraped lifetime stats with properly computed
    # point-in-time values so the model sees cumulative counts that
    # only reflect prior races, not future ones.
    df["horse_runs"] = df["horse_prev_races"].astype(int)
    df["horse_wins"] = df["horse_cum_wins"].astype(int)
    df["horse_places"] = df["horse_cum_places"].astype(int)

    # Cumulative average finishing position
    df["horse_cum_pos_sum"] = horse_groups["finish_position"].cumsum() - df["finish_position"]
    df["horse_avg_position"] = np.where(
        df["horse_prev_races"] > 0,
        df["horse_cum_pos_sum"] / df["horse_prev_races"],
        df["num_runners"] / 2,  # Default to middle for new horses
    )

    # --- Rolling form (Numba-accelerated) ---
    # Build all rolling operations on finish_position and won in one batch
    _pos_ops = []
    _won_ops = []
    for w in config.ROLLING_WINDOWS:
        _pos_ops.append((f"horse_avg_pos_{w}", "mean", w))
        _won_ops.append((f"horse_wins_{w}", "sum", w))

    # Best/worst/consistency (all on finish_position)
    _pos_ops.append(("horse_best_pos_5", "min", 5))
    _pos_ops.append(("horse_worst_pos_5", "max", 5))
    _pos_ops.append(("horse_pos_consistency", "std", 5))

    # EWMA on position and won
    for hl in [3, 7]:
        alpha = 1 - np.exp(-np.log(2) / hl)
        _pos_ops.append((f"horse_ewma_pos_{hl}", "ewma", alpha))
        _won_ops.append((f"horse_ewma_won_{hl}", "ewma", alpha))

    # Win pct last 3
    _won_ops.append(("horse_win_pct_last3", "mean", 3))

    # Execute batched rolling on finish_position
    for name, arr in _fast_grouped_rolling(df, hkey, "finish_position", _pos_ops).items():
        df[name] = arr
    # Execute batched rolling on won
    for name, arr in _fast_grouped_rolling(df, hkey, "won", _won_ops).items():
        df[name] = arr

    # Fill consistency NaN with default
    df["horse_pos_consistency"] = df["horse_pos_consistency"].fillna(3.0)
    # Fill win pct NaN with 0
    df["horse_win_pct_last3"] = df["horse_win_pct_last3"].fillna(0)

    # --- Days since last race ---
    horse_groups_dates = df.groupby(hkey)
    df["horse_last_race_date"] = horse_groups_dates["race_date"].shift(1)
    df["days_since_last_race"] = (
        (df["race_date"] - df["horse_last_race_date"]).dt.days.fillna(60)
    )

    # --- Days since last run — non-linear buckets ---
    df["days_log"] = np.log1p(df["days_since_last_race"])
    df["days_bucket_fresh"] = (df["days_since_last_race"] <= 14).astype(int)
    df["days_bucket_normal"] = df["days_since_last_race"].between(15, 28).astype(int)
    df["days_bucket_break"] = (df["days_since_last_race"] > 60).astype(int)

    # --- Race-type-aware horse features ---
    if "race_type" in df.columns:
        rt_groups = df.groupby([hkey, "race_type"])
        df["horse_rt_runs"] = rt_groups.cumcount()
        df["horse_rt_wins"] = rt_groups["won"].cumsum() - df["won"]
        df["horse_rt_win_rate"] = np.where(
            df["horse_rt_runs"] > 0,
            df["horse_rt_wins"] / df["horse_rt_runs"],
            0,
        )
        df = df.drop(columns=["horse_rt_wins"], errors="ignore")

    # --- Weight change from previous run ---
    if "weight_lbs" in df.columns:
        horse_groups = df.groupby(hkey)
        df["weight_change"] = horse_groups["weight_lbs"].diff().fillna(0)
        # Weight-change relative to the field: did THIS horse's burden shift
        # more or less than the race average?  Captures apprentice allowances
        # and handicapping adjustments that narrow/widen vs the competition.
        race_mean_wc = df.groupby("race_id")["weight_change"].transform("mean")
        df["weight_change_vs_field"] = (df["weight_change"] - race_mean_wc).fillna(0)
        # Binary flag: a drop of 3+ lbs is considered a meaningful relief.
        df["is_big_weight_drop"] = (df["weight_change"] < -3).astype(np.int8)

    # --- Age × form-trend interaction ---
    # Young horses (≤3yo) that are *improving* on recent form represent one
    # of the most reliable angles in racing.  We encode both the trend and
    # the interaction so the model can learn each component separately.
    if "age" in df.columns:
        _age = df["age"].fillna(4).values.astype(np.float32)
        _is_young = (_age <= 3).astype(np.float32)  # 1 for 2yo/3yo
        # form_slope > 0 means improving: avg position over last 10 is
        # *worse* (larger) than over last 3 → horse is finishing better recently
        if "horse_avg_pos_3" in df.columns and "horse_avg_pos_10" in df.columns:
            _slope = (
                df["horse_avg_pos_10"].fillna(0).values.astype(np.float32)
                - df["horse_avg_pos_3"].fillna(0).values.astype(np.float32)
            )
            df["form_slope"] = _slope
            df["age_x_form_slope"] = _is_young * _slope
        # Experience-adjusted age: young horse with many races is unusual
        # (can mean high-quality) vs young horse with few outings.
        df["age_x_prev_races"] = (
            _age * np.log1p(df["horse_prev_races"].fillna(0).values.astype(np.float32))
        )

    # --- Form-string features (if parsed in data_processor) ---
    for col in ["form_last_pos", "form_avg", "form_wins_count", "form_length"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Is maiden (has never won) ---
    df["is_maiden"] = (df["horse_cum_wins"] == 0).astype(int)

    # --- Runs since last win (Numba-accelerated) ---
    _gids = _encode_groups(df, hkey)
    df["runs_since_last_win"] = _nb_runs_since_win(
        df["won"].values.astype(np.float64), _gids,
    )

    # --- Days since last win ---
    df["_last_win_date"] = df["race_date"].where(df["won"] == 1)
    horse_groups = df.groupby(hkey)
    df["_last_win_date"] = horse_groups["_last_win_date"].ffill()
    df["_last_win_date"] = horse_groups["_last_win_date"].shift(1)
    df["days_since_last_win"] = (
        (df["race_date"] - df["_last_win_date"]).dt.days.fillna(365)
    )

    # --- Average lengths behind (Numba-accelerated) ---
    if "lengths_behind" in df.columns:
        _lb_ops = [
            ("horse_avg_lb_behind", "mean", 5),
        ]
        for hl in [3, 7]:
            alpha = 1 - np.exp(-np.log(2) / hl)
            _lb_ops.append((f"horse_ewma_lb_{hl}", "ewma", alpha))

        # Close finish rate
        df["_close_finish"] = (df["lengths_behind"] <= 2.0).astype(float)
        _cf_ops = [("close_finish_rate", "mean", 5)]

        for name, arr in _fast_grouped_rolling(df, hkey, "lengths_behind", _lb_ops).items():
            df[name] = arr
        for name, arr in _fast_grouped_rolling(df, hkey, "_close_finish", _cf_ops).items():
            df[name] = arr

        df["horse_avg_lb_behind"] = df["horse_avg_lb_behind"].fillna(3.0)
        df["close_finish_rate"] = df["close_finish_rate"].fillna(0)
        df = df.drop(columns=["_close_finish"], errors="ignore")

    # Clean up intermediate columns
    df = df.drop(
        columns=[
            "horse_cum_wins",
            "horse_cum_places",
            "horse_cum_pos_sum",
            "horse_last_race_date",
            "_placed",
            "_last_win_date",
        ],
        errors="ignore",
    )

    return df


def add_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate jockey-specific historical features."""
    logger.info("Engineering jockey features...")
    df = df.copy()

    jockey_groups = df.groupby("jockey")

    # Number of previous rides
    df["jockey_prev_rides"] = jockey_groups.cumcount()

    # Cumulative win rate (vectorised)
    df["jockey_cum_wins"] = jockey_groups["won"].cumsum() - df["won"]
    df["jockey_win_rate"] = np.where(
        df["jockey_prev_rides"] > 0,
        df["jockey_cum_wins"] / df["jockey_prev_rides"],
        0,
    )
    df["jockey_win_rate_shrunk"] = _bayesian_shrink(
        df["jockey_win_rate"], df["jockey_prev_rides"],
        prior_rate=0.10, prior_strength=10.0,
    )

    # Rolling form  (Numba-accelerated)
    _jock_won_ops = []
    _jock_pos_ops = []
    for w in [10, 20]:
        _jock_won_ops.append((f"jockey_wins_{w}", "sum", w))
        _jock_pos_ops.append((f"jockey_avg_pos_{w}", "mean", w))

    for col, ops in [("won", _jock_won_ops), ("finish_position", _jock_pos_ops)]:
        results = _fast_grouped_rolling(df, "jockey", col, ops)
        for name, arr in results.items():
            df[name] = arr

    # Place rate (top 3)
    df["_placed"] = (df["finish_position"] <= 3).astype(int)
    df["jockey_cum_places"] = jockey_groups["_placed"].cumsum() - df["_placed"]
    df["jockey_place_rate"] = np.where(
        df["jockey_prev_rides"] > 0,
        df["jockey_cum_places"] / df["jockey_prev_rides"],
        0,
    )
    df["jockey_place_rate_shrunk"] = _bayesian_shrink(
        df["jockey_place_rate"], df["jockey_prev_rides"],
        prior_rate=0.30, prior_strength=10.0,
    )

    # ── Time-based windows (14d, 30d) ────────────────────────────
    # Ride-count windows can span weeks for busy jockeys but months
    # for quiet ones.  Calendar windows capture "hot streaks" better.
    df["race_date"] = pd.to_datetime(df["race_date"])
    for days in [14, 30]:
        _wins_col = f"jockey_wins_{days}d"
        _rides_col = f"jockey_rides_{days}d"
        _wr_col = f"jockey_wr_{days}d"

        w_arr, r_arr = _time_window_stats(df, "jockey", "race_date", "won", days)
        df[_wins_col] = w_arr
        df[_rides_col] = r_arr
        with np.errstate(invalid="ignore"):
            df[_wr_col] = np.where(r_arr > 2, w_arr / r_arr, 0)
    return df


def add_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trainer-specific historical features."""
    logger.info("Engineering trainer features...")
    df = df.copy()

    trainer_groups = df.groupby("trainer")

    # Race-safe: trainers often have multiple runners in the same race.
    # All same-race runners must see identical prior-race stats.
    df["trainer_prev_runs"] = _race_safe_cumcount(df, "trainer")
    df["trainer_cum_wins"] = _race_safe_cumsum(df, "trainer", "won")
    df["trainer_win_rate"] = np.where(
        df["trainer_prev_runs"] > 0,
        df["trainer_cum_wins"] / df["trainer_prev_runs"],
        0,
    )
    df["trainer_win_rate_shrunk"] = _bayesian_shrink(
        df["trainer_win_rate"], df["trainer_prev_runs"],
        prior_rate=0.10, prior_strength=10.0,
    )

    # Rolling form  (Numba-accelerated)
    _trainer_won_ops = []
    for w in [10, 20]:
        _trainer_won_ops.append((f"trainer_wins_{w}", "sum", w))
    results = _fast_grouped_rolling(df, "trainer", "won", _trainer_won_ops)
    for name, arr in results.items():
        df[name] = arr

    # Place rate
    df["_placed"] = (df["finish_position"] <= 3).astype(int)
    df["trainer_cum_places"] = _race_safe_cumsum(df, "trainer", "_placed")
    df["trainer_place_rate"] = np.where(
        df["trainer_prev_runs"] > 0,
        df["trainer_cum_places"] / df["trainer_prev_runs"],
        0,
    )
    df["trainer_place_rate_shrunk"] = _bayesian_shrink(
        df["trainer_place_rate"], df["trainer_prev_runs"],
        prior_rate=0.30, prior_strength=10.0,
    )

    # ── Time-based windows (14d, 30d) ────────────────────────────
    df["race_date"] = pd.to_datetime(df["race_date"])
    for days in [14, 30]:
        _wins_col = f"trainer_wins_{days}d"
        _runs_col = f"trainer_runs_{days}d"
        _wr_col = f"trainer_wr_{days}d"

        w_arr, r_arr = _time_window_stats(df, "trainer", "race_date", "won", days)
        df[_wins_col] = w_arr
        df[_runs_col] = r_arr
        with np.errstate(invalid="ignore"):
            df[_wr_col] = np.where(r_arr > 2, w_arr / r_arr, 0)

    df = df.drop(columns=["trainer_cum_wins", "trainer_cum_places", "_placed"])

    # --- Trainer at track (race-safe) ---
    if "track" in df.columns:
        # Create a combined key so _race_safe helpers can group correctly
        df["_tt_key"] = df["trainer"].astype(str) + "||" + df["track"].astype(str)
        df["trainer_track_runs"] = _race_safe_cumcount(df, "_tt_key")
        df["trainer_track_wins"] = _race_safe_cumsum(df, "_tt_key", "won")
        df["trainer_track_win_rate"] = np.where(
            df["trainer_track_runs"] > 0,
            df["trainer_track_wins"] / df["trainer_track_runs"],
            0,
        )
        df = df.drop(columns=["trainer_track_wins", "_tt_key"], errors="ignore")

    return df


def add_jockey_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate jockey performance at specific tracks."""
    if "track" not in df.columns:
        return df

    logger.info("Engineering jockey-at-track features...")
    df = df.copy()

    jt_groups = df.groupby(["jockey", "track"])
    df["jockey_track_runs"] = jt_groups.cumcount()
    df["jockey_track_wins"] = jt_groups["won"].cumsum() - df["won"]
    df["jockey_track_win_rate"] = np.where(
        df["jockey_track_runs"] > 0,
        df["jockey_track_wins"] / df["jockey_track_runs"],
        0,
    )

    df = df.drop(columns=["jockey_track_wins"], errors="ignore")
    return df


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate track-specific features for each horse."""
    logger.info("Engineering track features...")
    df = df.copy()

    hkey = _horse_key(df)

    # Horse's track record
    track_groups = df.groupby([hkey, "track"])
    df["horse_track_runs"] = track_groups.cumcount()
    df["horse_track_wins"] = track_groups["won"].cumsum() - df["won"]
    df["horse_track_win_rate"] = np.where(
        df["horse_track_runs"] > 0,
        df["horse_track_wins"] / df["horse_track_runs"],
        0,
    )
    df["horse_track_wr_shrunk"] = _bayesian_shrink(
        df["horse_track_win_rate"], df["horse_track_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Horse's distance record
    dist_groups = df.groupby([hkey, "distance_furlongs"])
    df["horse_dist_runs"] = dist_groups.cumcount()
    df["horse_dist_wins"] = dist_groups["won"].cumsum() - df["won"]
    df["horse_dist_win_rate"] = np.where(
        df["horse_dist_runs"] > 0,
        df["horse_dist_wins"] / df["horse_dist_runs"],
        0,
    )
    df["horse_dist_wr_shrunk"] = _bayesian_shrink(
        df["horse_dist_win_rate"], df["horse_dist_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Horse's going record
    going_groups = df.groupby([hkey, "going"])
    df["horse_going_runs"] = going_groups.cumcount()
    df["horse_going_wins"] = going_groups["won"].cumsum() - df["won"]
    df["horse_going_win_rate"] = np.where(
        df["horse_going_runs"] > 0,
        df["horse_going_wins"] / df["horse_going_runs"],
        0,
    )
    df["horse_going_wr_shrunk"] = _bayesian_shrink(
        df["horse_going_win_rate"], df["horse_going_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    df = df.drop(
        columns=[
            "horse_track_wins",
            "horse_dist_wins",
            "horse_going_wins",
        ],
        errors="ignore",
    )
    return df


def add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate jockey–trainer combination statistics.

    Certain jockey-trainer partnerships outperform their individual stats.
    This captures the synergy (or lack of) between the two.
    """
    logger.info("Engineering jockey-trainer combo features...")
    df = df.copy()

    df["_jt_combo"] = df["jockey"].astype(str) + " | " + df["trainer"].astype(str)

    # Race-safe: a jockey-trainer combo can (rarely) have multiple
    # runners in the same race (different jockeys for same trainer,
    # but the *combo* key makes this less frequent).  Stay consistent.
    df["jt_prev_runs"] = _race_safe_cumcount(df, "_jt_combo")
    df["jt_cum_wins"] = _race_safe_cumsum(df, "_jt_combo", "won")
    df["jt_win_rate"] = np.where(
        df["jt_prev_runs"] > 0,
        df["jt_cum_wins"] / df["jt_prev_runs"],
        0,
    )
    df["jt_win_rate_shrunk"] = _bayesian_shrink(
        df["jt_win_rate"], df["jt_prev_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Combo place rate (top 3)
    df["_placed"] = (df["finish_position"] <= 3).astype(int)
    df["jt_cum_places"] = _race_safe_cumsum(df, "_jt_combo", "_placed")
    df["jt_place_rate"] = np.where(
        df["jt_prev_runs"] > 0,
        df["jt_cum_places"] / df["jt_prev_runs"],
        0,
    )
    df["jt_place_rate_shrunk"] = _bayesian_shrink(
        df["jt_place_rate"], df["jt_prev_runs"],
        prior_rate=0.30, prior_strength=5.0,
    )

    df = df.drop(
        columns=["_jt_combo", "jt_cum_wins", "jt_cum_places", "_placed"],
        errors="ignore",
    )

    # --- Is‑stable‑jockey booking signal ---
    # Trainers have a preferred jockey — the one they book most often.
    # When that jockey rides again it is a "stable jockey" booking, which
    # correlates with higher trainer intent (they chose their best pilot).
    # We derive this from prior ride history only (no look-ahead leakage).
    if "trainer" in df.columns and "jockey" in df.columns:
        # Count prior rides per (trainer, jockey) pair — race-safe so that
        # a trainer's multiple runners in the same race share identical priors.
        df["_tjkey"] = df["trainer"].astype(str) + "||" + df["jockey"].astype(str)
        df["_tj_prior_rides"] = _race_safe_cumcount(df, "_tjkey")

        # For each race row, find the jockey with the most prior rides for
        # this trainer.  We do this by: (a) for every row record the
        # (trainer, jockey, prior_rides) triple, then (b) for each trainer
        # take the rolling argmax prior to the current race.
        # Building a per-trainer "most‑used jockey so far" requires a
        # careful implementation to avoid leakage.  We achieve it by:
        #   1. groupby trainer, sort by race index
        #   2. expanding cummax of _tj_prior_rides within trainer
        #   3. is_stable_jockey = 1 when own prior rides == trainer's cummax
        trainer_max_tj = df.groupby("trainer")["_tj_prior_rides"].transform("cummax")
        df["is_stable_jockey"] = (
            (df["_tj_prior_rides"] == trainer_max_tj) & (df["_tj_prior_rides"] > 0)
        ).astype(np.int8)

        df = df.drop(columns=["_tjkey", "_tj_prior_rides"], errors="ignore")

    return df


def add_class_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect class changes between a horse's consecutive races.

    In UK racing, Class 1 is the highest and Class 7 the lowest.
    A horse *dropping* in class (higher number) is often a positive signal;
    being *raised* (lower number) suggests the connections think it's improved
    but the horse faces tougher competition.
    """
    logger.info("Engineering class change features...")
    df = df.copy()

    hkey = _horse_key(df)

    # Extract numeric class value  ("Class 3" → 3, "3" → 3)
    if "race_class" in df.columns:
        df["_class_num"] = pd.to_numeric(
            df["race_class"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        ).fillna(0)
    else:
        df["_class_num"] = 0

    horse_groups = df.groupby(hkey)
    df["_prev_class"] = horse_groups["_class_num"].shift(1)

    # Positive = dropping (easier), Negative = rising (harder)
    df["class_change"] = (df["_class_num"] - df["_prev_class"]).fillna(0)
    df["class_dropped"] = (df["class_change"] > 0).astype(int)
    df["class_raised"] = (df["class_change"] < 0).astype(int)

    # Official-rating change from previous run
    if "official_rating" in df.columns:
        df["or_change"] = horse_groups["official_rating"].diff().fillna(0)
    else:
        df["or_change"] = 0

    df = df.drop(columns=["_class_num", "_prev_class"], errors="ignore")
    return df


def add_course_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Course & Distance (C&D) winner flag and stats.

    A horse that has previously won at the same track AND distance is a
    proven C&D winner — one of the most respected angles in racing.
    """
    logger.info("Engineering course & distance features...")
    df = df.copy()

    hkey = _horse_key(df)

    # Round distance to nearest 0.5 f for grouping (minor variations)
    df["_dist_rounded"] = (df["distance_furlongs"] * 2).round() / 2

    # --- C&D (same track + same distance) ---
    cd_groups = df.groupby([hkey, "track", "_dist_rounded"])
    df["horse_cd_runs"] = cd_groups.cumcount()
    df["horse_cd_wins"] = cd_groups["won"].cumsum() - df["won"]
    df["horse_cd_winner"] = (df["horse_cd_wins"] > 0).astype(int)
    df["horse_cd_win_rate"] = np.where(
        df["horse_cd_runs"] > 0,
        df["horse_cd_wins"] / df["horse_cd_runs"],
        0,
    )

    # --- Course only (already partially in add_track_features, but
    #     add a "has won at course before" binary for clarity) ---
    c_groups = df.groupby([hkey, "track"])
    df["horse_course_wins"] = c_groups["won"].cumsum() - df["won"]
    df["horse_course_winner"] = (df["horse_course_wins"] > 0).astype(int)

    df = df.drop(
        columns=["_dist_rounded", "horse_cd_wins", "horse_course_wins"],
        errors="ignore",
    )
    return df


def add_distance_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features related to changes in trip distance between races.

    Stepping up or down in distance is one of the most important angles
    in racing — some horses improve dramatically with a trip change.
    """
    logger.info("Engineering distance change features...")
    df = df.copy()

    hkey = _horse_key(df)

    # --- Distance change from previous race ---
    horse_groups = df.groupby(hkey)
    df["_prev_dist"] = horse_groups["distance_furlongs"].shift(1)
    df["distance_change"] = (df["distance_furlongs"] - df["_prev_dist"]).fillna(0)
    df["stepping_up"] = (df["distance_change"] > 0.5).astype(int)
    df["stepping_down"] = (df["distance_change"] < -0.5).astype(int)

    # --- Distance category ---
    bins = [0, 6.5, 8.5, 11.5, 50]
    labels = [0, 1, 2, 3]  # Sprint / Mile / Middle / Staying
    df["dist_category"] = pd.cut(
        df["distance_furlongs"], bins=bins, labels=labels, include_lowest=True,
    ).astype(float).fillna(1)

    # --- Horse win rate at this distance category ---
    df["_dist_cat_str"] = df["dist_category"].astype(str)
    dc_groups = df.groupby([hkey, "_dist_cat_str"])
    df["horse_dist_cat_runs"] = dc_groups.cumcount()
    df["_dc_wins"] = dc_groups["won"].cumsum() - df["won"]
    df["horse_dist_cat_wr"] = np.where(
        df["horse_dist_cat_runs"] > 0,
        df["_dc_wins"] / df["horse_dist_cat_runs"],
        0,
    )
    df["horse_dist_cat_wr_shrunk"] = _bayesian_shrink(
        df["horse_dist_cat_wr"], df["horse_dist_cat_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # --- At preferred distance? (best win rate distance category) ---
    # For each horse, find the distance category where they've won most.
    # Use cummax() (not transform("max")) to avoid look-ahead: at each
    # point in time we only know the horse's wins up to NOW, not future.
    df["_cum_dist_cat_wins"] = df.groupby([hkey, "_dist_cat_str"])["won"].cumsum() - df["won"]
    # Running max of cumulative wins within each (horse, dist_cat) group
    df["_dc_running_max"] = df.groupby([hkey, "_dist_cat_str"])["_cum_dist_cat_wins"].cummax()
    # Running max across ALL distance categories for each horse
    df["_best_dc"] = df.groupby(hkey)["_dc_running_max"].cummax()
    df["at_preferred_dist"] = (
        (df["_cum_dist_cat_wins"] == df["_best_dc"]) & (df["_best_dc"] > 0)
    ).astype(int)

    df = df.drop(
        columns=["_prev_dist", "_dist_cat_str", "_dc_wins",
                 "_cum_dist_cat_wins", "_dc_running_max", "_best_dc"],
        errors="ignore",
    )
    return df


def add_headgear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from headgear information.

    Headgear (blinkers, visors, cheekpieces, etc.) can sharpen a horse's
    focus. *First-time headgear* is one of the most powerful signals in
    racing — it often produces immediate improvement.
    """
    if "headgear" not in df.columns:
        logger.info("No headgear column found — skipping headgear features.")
        # Still create the columns with zeros so the model doesn't break
        df["has_headgear"] = 0
        df["first_time_headgear"] = 0
        return df

    logger.info("Engineering headgear features...")
    df = df.copy()

    hg = df["headgear"].fillna("").astype(str).str.strip().str.lower()

    # Binary: wearing any headgear?
    df["has_headgear"] = (
        hg.ne("") & hg.ne("none") & hg.ne("0") & hg.ne("nan")
    ).astype(int)

    # First-time headgear detection
    hkey = _horse_key(df)
    horse_groups = df.groupby(hkey)
    df["_prev_hg"] = horse_groups["has_headgear"].shift(1).fillna(0)
    df["_ever_had_hg"] = horse_groups["has_headgear"].cumsum() - df["has_headgear"]
    # First-time = wearing headgear now AND never worn it in any prior race
    df["first_time_headgear"] = (
        (df["has_headgear"] == 1) & (df["_ever_had_hg"] == 0)
    ).astype(int)

    # Headgear removed (wore it last time but not now)
    df["headgear_off"] = (
        (df["has_headgear"] == 0) & (df["_prev_hg"] == 1)
    ).astype(int)

    df = df.drop(columns=["_prev_hg", "_ever_had_hg"], errors="ignore")
    return df


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse performance on the current surface (Turf vs AW/Polytrack)."""
    if "surface" not in df.columns:
        return df

    logger.info("Engineering surface features...")
    df = df.copy()

    hkey = _horse_key(df)

    surf_groups = df.groupby([hkey, "surface"])
    df["horse_surface_runs"] = surf_groups.cumcount()
    df["horse_surface_wins"] = surf_groups["won"].cumsum() - df["won"]
    df["horse_surface_win_rate"] = np.where(
        df["horse_surface_runs"] > 0,
        df["horse_surface_wins"] / df["horse_surface_runs"],
        0,
    )

    # First time on this surface type (e.g. turf horse switching to AW)
    df["first_time_surface"] = (df["horse_surface_runs"] == 0).astype(np.int8)

    # Surface switch from previous run — captures adaptation cost
    horse_groups = df.groupby(hkey)
    df["_prev_surface"] = horse_groups["surface"].shift(1)
    df["surface_switch"] = (
        df["_prev_surface"].notna() & (df["surface"] != df["_prev_surface"])
    ).astype(np.int8)
    df = df.drop(columns=["_prev_surface", "horse_surface_wins"], errors="ignore")
    return df


def add_jockey_intent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features capturing jockey switches, booking upgrades, and
    equipment-change signals.

    Connections booking a higher-rated jockey signals intent to win.
    Equipment changes (blinkers first time, tongue strap first time)
    are among the most predictive single-race signals in racing.
    """
    logger.info("Engineering jockey intent features...")
    df = df.copy()

    hkey = _horse_key(df)
    horse_groups = df.groupby(hkey)

    # --- Jockey switch (different jockey from last run) ---
    df["_prev_jockey"] = horse_groups["jockey"].shift(1)
    df["jockey_switch"] = (
        df["_prev_jockey"].notna() & (df["jockey"] != df["_prev_jockey"])
    ).astype(int)

    # --- Jockey upgrade (current jockey Elo > previous jockey Elo) ---
    if "jockey_elo" in df.columns:
        df["_prev_jockey_elo"] = horse_groups["jockey_elo"].shift(1)
        df["jockey_elo_change"] = (
            (df["jockey_elo"] - df["_prev_jockey_elo"]).fillna(0)
        )
        df["jockey_upgrade"] = (
            (df["jockey_switch"] == 1) & (df["jockey_elo_change"] > 50)
        ).astype(int)
        df = df.drop(columns=["_prev_jockey_elo"], errors="ignore")

    # --- Beaten favourite last time (bounce-back angle) ---
    if "is_favourite" in df.columns:
        df["_prev_fav"] = horse_groups["is_favourite"].shift(1).fillna(0)
        df["_prev_won"] = horse_groups["won"].shift(1).fillna(0)
        df["beaten_fav_last"] = (
            (df["_prev_fav"] == 1) & (df["_prev_won"] == 0)
        ).astype(int)
        df = df.drop(columns=["_prev_fav", "_prev_won"], errors="ignore")

    # ── Granular equipment signals ───────────────────────────────
    # The headgear column is stored as a JSON-like list of dicts:
    # [{"symbol": "b", "name": "Blinkers", "count": 1}, ...]
    # count=1 means first-time application — the strongest signal.
    if "headgear" in df.columns:
        import ast

        hg_raw = df["headgear"].fillna("").astype(str)

        # Parse the structured headgear data
        def _parse_hg(val):
            """Extract (set of symbols, dict of symbol→count)."""
            if not val or val in ("nan", "None", "", "0"):
                return set(), {}
            try:
                items = ast.literal_eval(val)
                if isinstance(items, list):
                    symbols = {d.get("symbol", "") for d in items if isinstance(d, dict)}
                    counts = {
                        d.get("symbol", ""): d.get("count", 999)
                        for d in items if isinstance(d, dict)
                    }
                    return symbols, counts
            except (ValueError, SyntaxError):
                pass
            return set(), {}

        parsed = hg_raw.apply(_parse_hg)
        symbols_series = parsed.apply(lambda x: x[0])
        counts_series = parsed.apply(lambda x: x[1])

        # Blinkers first time (count == 1)
        df["blinkers_first_time"] = [
            1 if "b" in syms and cts.get("b", 999) == 1 else 0
            for syms, cts in zip(symbols_series, counts_series)
        ]

        # Tongue strap first time
        df["tongue_strap_first_time"] = [
            1 if "t" in syms and cts.get("t", 999) == 1 else 0
            for syms, cts in zip(symbols_series, counts_series)
        ]

        # Cheekpieces first time
        df["cheekpieces_first_time"] = [
            1 if "p" in syms and cts.get("p", 999) == 1 else 0
            for syms, cts in zip(symbols_series, counts_series)
        ]

        # Visor first time
        df["visor_first_time"] = [
            1 if "v" in syms and cts.get("v", 999) == 1 else 0
            for syms, cts in zip(symbols_series, counts_series)
        ]

        # Any first-time equipment application
        df["any_first_time_equip"] = [
            1 if any(cts.get(s, 999) == 1 for s in syms) else 0
            for syms, cts in zip(symbols_series, counts_series)
        ]

        # Total number of equipment items worn
        df["equip_count"] = symbols_series.apply(len).astype(int)

    # --- Weight change direction ---
    if "weight_change" in df.columns:
        df["weight_dropped"] = (df["weight_change"] < 0).astype(int)
        df["weight_raised"] = (df["weight_change"] > 0).astype(int)

    df = df.drop(columns=["_prev_jockey"], errors="ignore")
    return df


def add_trainer_specialisation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trainer win rates by distance category and going.

    Some trainers specialise heavily — e.g. sprint specialists,
    trainers who target soft ground, etc.
    """
    logger.info("Engineering trainer specialisation features...")
    df = df.copy()

    # --- Trainer × distance category (race-safe) ---
    if "dist_category" in df.columns:
        df["_dc_str"] = df["dist_category"].astype(str)
        df["_tdc_key"] = df["trainer"].astype(str) + "||" + df["_dc_str"]
        df["trainer_dist_cat_runs"] = _race_safe_cumcount(df, "_tdc_key")
        df["_tdc_wins"] = _race_safe_cumsum(df, "_tdc_key", "won")
        df["trainer_dist_cat_wr"] = np.where(
            df["trainer_dist_cat_runs"] > 2,
            df["_tdc_wins"] / df["trainer_dist_cat_runs"],
            0,
        )
        df = df.drop(columns=["_dc_str", "_tdc_wins", "_tdc_key"], errors="ignore")

    # --- Trainer × going (race-safe) ---
    if "going" in df.columns:
        df["_tg_key"] = df["trainer"].astype(str) + "||" + df["going"].astype(str)
        df["trainer_going_runs"] = _race_safe_cumcount(df, "_tg_key")
        df["_tg_wins"] = _race_safe_cumsum(df, "_tg_key", "won")
        df["trainer_going_wr"] = np.where(
            df["trainer_going_runs"] > 2,
            df["_tg_wins"] / df["trainer_going_runs"],
            0,
        )
        df = df.drop(columns=["_tg_wins", "_tg_key"], errors="ignore")

    return df


def add_speed_figure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive proxy speed figures from ``lengths_behind`` and distance.

    Since raw finish times are unavailable, we normalise
    ``lengths_behind`` by distance to get a *lengths-per-furlong*
    performance metric (lower = faster).  Rolling averages of this
    metric across prior races serve as a horse's speed figure.

    All rolling windows use ``shift(1)`` to avoid data leakage.
    """
    if "lengths_behind" not in df.columns:
        return df

    logger.info("Engineering speed figure features...")
    df = df.copy()

    hkey = _horse_key(df)

    # Lengths per furlong — normalised for distance
    safe_dist = df["distance_furlongs"].replace(0, np.nan).fillna(8.0)
    df["_lb_per_furlong"] = df["lengths_behind"].fillna(0) / safe_dist

    # Per-race relative performance: horse vs race median (neg = better)
    race_median_lpf = df.groupby("race_id")["_lb_per_furlong"].transform("median")
    df["_rel_perf"] = df["_lb_per_furlong"] - race_median_lpf

    # --- Numba-accelerated rolling speed figures ---
    _speed_ops = [
        ("speed_fig_avg_5", "mean", 5),
        ("speed_fig_avg_10", "mean", 10),
        ("speed_fig_best_5", "min", 5),
    ]
    for name, arr in _fast_grouped_rolling(df, hkey, "_lb_per_furlong", _speed_ops).items():
        df[name] = arr

    # Trend: recent avg minus long-term avg (negative = improving)
    df["speed_fig_trend"] = df["speed_fig_avg_5"] - df["speed_fig_avg_10"]

    # Relative performance rolling avg (negative = above average)
    _rel_ops = [("speed_fig_rel_avg_5", "mean", 5)]
    for name, arr in _fast_grouped_rolling(df, hkey, "_rel_perf", _rel_ops).items():
        df[name] = arr

    # ── Class-adjusted speed figures ─────────────────────────────
    if "race_class" in df.columns:
        _class_num = pd.to_numeric(
            df["race_class"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        ).fillna(5)
        _quality = (1.0 - (_class_num - 1) * 0.1).clip(0.4, 1.0)
        df["_adj_lpf"] = df["_lb_per_furlong"].fillna(0) * (2.0 - _quality)

        _adj_ops = [
            ("speed_fig_adj_avg_5", "mean", 5),
            ("speed_fig_adj_best_5", "min", 5),
        ]
        for name, arr in _fast_grouped_rolling(df, hkey, "_adj_lpf", _adj_ops).items():
            df[name] = arr
        df = df.drop(columns=["_adj_lpf"], errors="ignore")

        # ── Class-relative Z-score speed figures ─────────────────
        _class_key = _class_num.astype(int).astype(str)
        # shift(1) so each row's own lb_per_furlong is excluded from
        # the class median / std (avoids current-race outcome leakage).
        _class_median = df.groupby(_class_key)["_lb_per_furlong"].transform(
            lambda x: x.shift(1).expanding().median()
        )
        _class_std = df.groupby(_class_key)["_lb_per_furlong"].transform(
            lambda x: x.shift(1).expanding().std()
        )
        _class_std = _class_std.replace(0, np.nan).fillna(1.0)
        df["_speed_z_class"] = (df["_lb_per_furlong"] - _class_median) / _class_std

        # Rolling Z-score averages (Numba-accelerated)
        _z_ops = [
            ("speed_z_class_avg_5", "mean", 5),
            ("speed_z_class_avg_10", "mean", 10),
            ("speed_z_class_best_5", "min", 5),
        ]
        for name, arr in _fast_grouped_rolling(df, hkey, "_speed_z_class", _z_ops).items():
            df[name] = arr
        df["speed_z_class_trend"] = (
            df["speed_z_class_avg_5"] - df["speed_z_class_avg_10"]
        )
        df = df.drop(columns=["_speed_z_class"], errors="ignore")

    df = df.drop(columns=["_lb_per_furlong", "_rel_perf"], errors="ignore")
    return df


def add_track_config_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge static track configuration data (direction, shape, gradients, etc.).

    These are fixed physical properties of each racecourse that provide
    genuinely new information not derivable from race results.
    """
    if "track" not in df.columns:
        return df

    logger.info("Engineering track configuration features...")
    df = df.copy()

    # Vectorised lookup — build columns from the config dict
    configs = df["track"].map(get_track_config)

    df["track_direction"] = configs.apply(lambda c: direction_code(c["direction"]))
    df["track_shape"] = configs.apply(lambda c: shape_code(c["shape"]))
    df["track_is_aw"] = configs.apply(lambda c: c["is_aw"])
    df["track_circumference"] = configs.apply(lambda c: c["circumference_f"])
    df["track_uphill_finish"] = configs.apply(lambda c: c["uphill_finish"])
    df["track_downhill"] = configs.apply(lambda c: c["downhill_section"])
    df["track_draw_bias"] = configs.apply(lambda c: c["draw_bias"])

    # --- Shape one-hot (more useful than ordinal for trees) ---
    for shape_name, code in [("Galloping", 0), ("Tight", 1), ("Sharp", 2),
                              ("Undulating", 3), ("Stiff", 4)]:
        df[f"track_shape_{shape_name.lower()}"] = (
            df["track_shape"] == code
        ).astype(int)

    # --- Draw × draw-bias interaction ---
    # Strong draw bias + unfavourable draw = big disadvantage
    if "draw_pct" in df.columns:
        _dp = df["draw_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
        df["draw_x_bias"] = _dp * df["track_draw_bias"]

    # --- Uphill finish × weight ---
    # Uphill finishes penalise heavier horses more
    if "weight_lbs" in df.columns:
        df["uphill_x_weight"] = df["track_uphill_finish"] * df["weight_lbs"]

    # --- Tight/sharp track × draw ---
    # On tight tracks, inside draws matter more
    if "draw_pct" in df.columns:
        df["tight_track_x_draw"] = (
            (df["track_shape_tight"] | df["track_shape_sharp"]).astype(float)
            * _dp
        )

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch & attach weather features for every race-day / track pair.

    Raw columns from weather module:
        weather_temp_max, weather_temp_min, weather_precip_mm,
        weather_wind_kmh, weather_precip_prev3

    Derived columns:
        weather_temp_range      – diurnal swing (stress indicator)
        weather_is_wet          – race-day precipitation >2 mm
        weather_heavy_rain_prev – 3-day prior precip >10 mm (soft/heavy going)
        weather_wind_x_dist     – wind × distance (stamina drain)
        weather_cold            – temperature below 5 °C
    """
    from src.weather import get_weather_for_races

    df = get_weather_for_races(df)

    # ── Derived weather features ─────────────────────────────────
    tmax = df["weather_temp_max"].fillna(14.0)
    tmin = df["weather_temp_min"].fillna(7.0)
    precip = df["weather_precip_mm"].fillna(1.5)
    wind = df["weather_wind_kmh"].fillna(18.0)
    prev3 = df["weather_precip_prev3"].fillna(4.5)

    df["weather_temp_range"] = tmax - tmin
    df["weather_is_wet"] = (precip > 2.0).astype(int)
    df["weather_heavy_rain_prev"] = (prev3 > 10.0).astype(int)
    df["weather_cold"] = (tmax < 5.0).astype(int)

    # Wind × distance interaction: wind matters more over longer trips
    dist = df["distance_f"].fillna(8.0) if "distance_f" in df.columns else 8.0
    df["weather_wind_x_dist"] = wind * dist / 100.0  # scaled

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit interaction features that capture well-known racing angles.

    Tree models can discover interactions via splits, but explicit
    features help them learn faster and more reliably.
    """
    logger.info("Engineering interaction features...")
    df = df.copy()

    # 1) days_since_last_race × class_dropped
    #    Fresh horse dropping in class — a classic trainer angle.
    if "days_since_last_race" in df.columns and "class_dropped" in df.columns:
        df["fresh_x_dropped"] = (
            df["days_log"] * df["class_dropped"]
        )

    # 2) horse_going_win_rate × going match indicator
    #    Proven ground preference on the right surface.
    if "horse_going_win_rate" in df.columns:
        df["going_wr_x_runs"] = (
            df["horse_going_win_rate"]
            * np.log1p(df.get("horse_going_runs", 0))
        )

    # 3) jockey_elo × is_favourite
    #    Strong rider booking confirmed by market.
    if "jockey_elo" in df.columns and "is_favourite" in df.columns:
        # Centre jockey_elo around baseline so the interaction is meaningful
        df["jockey_elo_x_fav"] = (
            (df["jockey_elo"] - 1500) * df["is_favourite"]
        )

    # 4) horse_elo × class_change
    #    High-rated horse dropping in class is a strong signal.
    if "horse_elo" in df.columns and "class_change" in df.columns:
        df["elo_x_class_drop"] = (
            (df["horse_elo"] - 1500) * df["class_dropped"]
        )

    # 5) speed_fig × distance match
    #    Good speed figure is more predictive at the proven distance.
    if "speed_fig_avg_5" in df.columns and "horse_dist_win_rate" in df.columns:
        df["speed_x_dist_wr"] = (
            (1.0 - df["speed_fig_avg_5"].clip(0, 2))  # invert: lower = better
            * df["horse_dist_win_rate"]
        )

    # 6) OR × handicap — official rating matters most in handicaps
    if "official_rating" in df.columns and "handicap" in df.columns:
        race_avg_or = df.groupby("race_id")["official_rating"].transform("mean")
        df["or_vs_field_x_hcap"] = (
            (df["official_rating"] - race_avg_or) * df["handicap"]
        )

    # 7) age × distance — young horses prefer sprints, older handle distance
    if "age" in df.columns and "distance_furlongs" in df.columns:
        df["age_x_distance"] = df["age"] * df["distance_furlongs"]

    # 8) age × going — younger horses struggle more on soft/heavy
    if "age" in df.columns and "going_Soft" in df.columns:
        soft_heavy = df.get("going_Soft", 0)
        if "going_Heavy" in df.columns:
            soft_heavy = soft_heavy + df["going_Heavy"]
        df["age_x_soft_going"] = df["age"] * soft_heavy

    # 9) days_since_last_race² — captures U-shape (very fresh & stale = bad)
    if "days_since_last_race" in df.columns:
        df["days_since_race_sq"] = df["days_since_last_race"] ** 2 / 1000.0

    # 10) Improving flag — both speed and form trending better
    if "speed_fig_trend" in df.columns and "form_trend" in df.columns:
        df["improving_flag"] = (
            (df["speed_fig_trend"] < 0) & (df["form_trend"] < 0)
        ).astype(int)

    # 11) Trainer intent score — class drop + jockey upgrade + first-time headgear
    intent = pd.Series(0, index=df.index)
    if "class_dropped" in df.columns:
        intent = intent + df["class_dropped"]
    if "jockey_upgrade" in df.columns:
        intent = intent + df["jockey_upgrade"]
    if "first_time_headgear" in df.columns:
        intent = intent + df["first_time_headgear"]
    df["trainer_intent_score"] = intent

    # 12) distance_change × class_change — trip + class shift combo
    if "distance_change" in df.columns and "class_change" in df.columns:
        df["dist_x_class_change"] = df["distance_change"] * df["class_change"]

    # 13) jockey_upgrade × class_dropped — booking a star for a drop
    if "jockey_upgrade" in df.columns and "class_dropped" in df.columns:
        df["jockey_up_x_dropped"] = df["jockey_upgrade"] * df["class_dropped"]

    # 14) close_finish_rate × speed_fig_avg_5 — competitive + fast
    if "close_finish_rate" in df.columns and "speed_fig_avg_5" in df.columns:
        df["close_x_speed"] = (
            df["close_finish_rate"] * (1.0 - df["speed_fig_avg_5"].clip(0, 2))
        )

    # ── Market-residual interactions ─────────────────────────────
    # These help the model learn *when* the market misprices a horse
    # by pairing market signals with fundamental ability indicators.

    # 15) norm_implied_prob × horse_win_rate_shrunk — market vs proven ability
    if "norm_implied_prob" in df.columns and "horse_win_rate_shrunk" in df.columns:
        df["mkt_x_win_rate"] = df["norm_implied_prob"] - df["horse_win_rate_shrunk"]

    # 16) log_odds × horse_elo — price level vs ability rating
    if "log_odds" in df.columns and "horse_elo" in df.columns:
        df["logodds_x_elo"] = df["log_odds"] * (df["horse_elo"] - 1500) / 400

    # 17) odds_vs_field × class_dropped — overlooked class droppers
    if "odds_vs_field" in df.columns and "class_dropped" in df.columns:
        df["odds_field_x_dropped"] = df["odds_vs_field"] * df["class_dropped"]

    # 18) norm_implied_prob × speed_fig_avg_5 — market vs recent speed
    if "norm_implied_prob" in df.columns and "speed_fig_avg_5" in df.columns:
        df["mkt_x_speed"] = df["norm_implied_prob"] - (1.0 - df["speed_fig_avg_5"].clip(0, 2))

    # 19) odds_vs_field × jockey_elo — price vs rider quality
    if "odds_vs_field" in df.columns and "jockey_elo" in df.columns:
        df["odds_field_x_jock_elo"] = df["odds_vs_field"] * (df["jockey_elo"] - 1500) / 400

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from betting odds (market indicators).

    Odds encode crowd wisdom about a horse's chances.

    .. warning::

       For historical results the ``odds`` column is the **Starting Price
       (SP)** — the final market price at race-off.  SP is significantly
       more informative than the early/ante-post odds available when a
       punter would actually place a bet.  This means backtest results
       using odds-derived features will be **optimistic** relative to
       live performance where only early morning or pre-race odds are
       available.
    """
    logger.info("Engineering market features...")
    df = df.copy()

    if "odds" not in df.columns:
        return df

    # Implied probability from odds
    df["implied_prob"] = 1.0 / df["odds"]

    # Normalised probability within race
    race_prob_sum = df.groupby("race_id")["implied_prob"].transform("sum")
    df["norm_implied_prob"] = df["implied_prob"] / race_prob_sum

    # Odds rank within race (1 = favourite)
    df["odds_rank"] = df.groupby("race_id")["odds"].rank(method="min")

    # Is favourite?
    df["is_favourite"] = (df["odds_rank"] == 1).astype(int)

    # Log odds (more linear relationship with probability)
    df["log_odds"] = np.log1p(df["odds"])

    # Odds relative to field average
    race_avg_odds = df.groupby("race_id")["odds"].transform("mean")
    df["odds_vs_field"] = df["odds"] / race_avg_odds

    # Market overround (sum of implied probs — higher = less efficient market)
    df["overround"] = race_prob_sum

    return df


def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features about the race context."""
    logger.info("Engineering race context features...")
    df = df.copy()

    # Draw advantage (normalized by field size)
    df["draw_pct"] = df["draw"] / df["num_runners"]

    # ── Track-specific draw bias ─────────────────────────────────
    # Per-track draw-position win-rate (point-in-time, no leakage).
    # Tracks like Chester, Beverley, Ascot have extreme draw biases.
    if "track" in df.columns:

        # Create draw-position bins (low / mid / high third)
        df["_draw_third"] = pd.cut(
            df["draw_pct"].clip(0.01, 0.99),
            bins=[0, 0.33, 0.66, 1.0],
            labels=["low", "mid", "high"],
            include_lowest=True,
        ).astype(str)

        for draw_pos in ["low", "mid", "high"]:
            mask = df["_draw_third"] == draw_pos
            # Track × draw-third win-rate (cumulative, shifted)
            td_groups = df[mask].groupby("track")
            df.loc[mask, f"_td_wins_{draw_pos}"] = td_groups["won"].cumsum() - df.loc[mask, "won"]
            df.loc[mask, f"_td_runs_{draw_pos}"] = td_groups.cumcount()

        # Recombine: the horse's draw-third win-rate at this track
        df["track_draw_wr_wins"] = 0.0
        df["track_draw_wr_runs"] = 0.0
        for draw_pos in ["low", "mid", "high"]:
            mask = df["_draw_third"] == draw_pos
            w_col = f"_td_wins_{draw_pos}"
            r_col = f"_td_runs_{draw_pos}"
            if w_col in df.columns:
                df.loc[mask, "track_draw_wr_wins"] = df.loc[mask, w_col]
                df.loc[mask, "track_draw_wr_runs"] = df.loc[mask, r_col]

        df["track_draw_win_rate"] = np.where(
            df["track_draw_wr_runs"] > 5,
            df["track_draw_wr_wins"] / df["track_draw_wr_runs"],
            0.0,
        )

        # Clean up
        drop_cols = ["_draw_third", "track_draw_wr_wins", "track_draw_wr_runs"]
        drop_cols += [c for c in df.columns if c.startswith("_td_")]
        df = df.drop(columns=drop_cols, errors="ignore")

    # ── Draw × distance × field size interactions ────────────────
    # Draw matters enormously in sprints (especially ≤6f) on tight
    # turning tracks with big fields, but barely at all over 2m+.
    _dp = df.get("draw_pct", pd.Series(0.5, index=df.index))
    _dp = _dp.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    _dist = df["distance_furlongs"].fillna(8.0)
    _nr = df["num_runners"].fillna(10)

    _is_sprint = (_dist <= 7.0).astype(float)   # sprints ≤ 7f
    _large = (_nr >= 14).astype(float)

    # Core interaction: draw advantage only matters for short distances
    # with big fields (where getting boxed in is a real risk).
    df["draw_x_sprint_x_large"] = _dp * _is_sprint * _large

    # Draw × inverse distance — draw impact fades with distance
    df["draw_x_inv_dist"] = _dp * (1.0 / _dist.clip(lower=4.0))

    # Draw × field size (normalized) — more runners = more draw impact
    df["draw_x_field_size"] = _dp * (_nr / 20.0).clip(0, 2)

    # Track bias × sprint — track draw bias is primarily a sprint phenomenon
    if "track_draw_bias" in df.columns:
        df["draw_bias_x_sprint"] = df["track_draw_bias"] * _is_sprint

    # ── Age-performance curve by race type ────────────────────────
    # Flat horses peak at 3-4yo; NH horses at 7-9yo. Modelling the
    # distance from peak age lets the model learn the decline curve.
    _age = df["age"].fillna(4).astype(float)
    _is_nh = 0.0
    if "race_type_Hurdle" in df.columns:
        _is_nh = df.get("race_type_Hurdle", 0) + df.get("race_type_Chase", 0)
        _is_nh = _is_nh.clip(0, 1).astype(float)
    elif "race_type" in df.columns:
        _is_nh = df["race_type"].isin(["Hurdle", "Chase"]).astype(float)

    # Peak age: 3.5 for Flat, 8.0 for NH
    _peak = np.where(_is_nh > 0.5, 8.0, 3.5)
    df["years_from_peak"] = _age - _peak
    df["years_from_peak_abs"] = np.abs(df["years_from_peak"])
    # Quadratic decline — penalty grows with distance from peak
    df["age_decline_curve"] = df["years_from_peak"] ** 2 / 25.0
    # Is the horse past peak? (different signal from approaching it)
    df["past_peak"] = (df["years_from_peak"] > 0).astype(int)
    # Age appropriateness for race type
    df["age_x_nh"] = _age * _is_nh   # older = more suited to NH
    # Young flat horse (2-3yo) — still improving
    df["young_flat"] = ((_age <= 3) & (_is_nh < 0.5)).astype(int)

    # Weight relative to field
    race_avg_weight = df.groupby("race_id")["weight_lbs"].transform("mean")
    df["weight_vs_field"] = df["weight_lbs"] - race_avg_weight

    # Age relative to field
    race_avg_age = df.groupby("race_id")["age"].transform("mean")
    df["age_vs_field"] = df["age"] - race_avg_age

    # Field size categories
    df["small_field"] = (df["num_runners"] <= 8).astype(int)
    df["large_field"] = (df["num_runners"] >= 14).astype(int)

    # Prize money rank (proxy for race quality)
    df["prize_log"] = np.log1p(df["prize_money"])

    # Weight per furlong — at longer distances weight matters more
    if "distance_furlongs" in df.columns:
        safe_dist = df["distance_furlongs"].replace(0, np.nan).fillna(8.0)
        df["weight_per_furlong"] = df["weight_lbs"] / safe_dist

    # Handicap features
    if "handicap" in df.columns:
        df["handicap"] = df["handicap"].fillna(0).astype(int)
        # OR x handicap interaction — OR is most meaningful in handicaps
        if "official_rating" in df.columns:
            df["or_x_handicap"] = df["official_rating"] * df["handicap"]

    # --- Odds coefficient of variation (field competitiveness) ---
    if "odds" in df.columns:
        race_std_odds = df.groupby("race_id")["odds"].transform("std").fillna(0)
        race_mean_odds = df.groupby("race_id")["odds"].transform("mean")
        df["odds_cv"] = np.where(
            race_mean_odds > 0, race_std_odds / race_mean_odds, 0
        )

    # --- Weight rank within race ---
    if "weight_lbs" in df.columns:
        df["weight_rank"] = df.groupby("race_id")["weight_lbs"].rank(
            ascending=False, method="min"
        )

    # --- OR vs top-rated in race ---
    if "official_rating" in df.columns:
        race_max_or = df.groupby("race_id")["official_rating"].transform("max")
        df["or_vs_top"] = df["official_rating"] - race_max_or

    # --- Age group buckets ---
    if "age" in df.columns:
        df["age_2yo"] = (df["age"] == 2).astype(int)
        df["age_3yo"] = (df["age"] == 3).astype(int)
        df["age_4_5yo"] = df["age"].between(4, 5).astype(int)
        df["age_6plus"] = (df["age"] >= 6).astype(int)

    # --- Horse Elo vs field average ---
    if "horse_elo" in df.columns:
        field_avg = df.groupby("race_id")["horse_elo"].transform("mean")
        df["horse_elo_vs_field"] = df["horse_elo"] - field_avg

    return df


def add_field_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features describing the quality and competitiveness of the field.

    These contextualise a horse's abilities relative to the strength of
    the opposition it faces — beating a Group 1 field is very different
    from beating a Class 6 seller.
    """
    logger.info("Engineering field quality features...")
    df = df.copy()

    # Average official rating of the race field
    if "official_rating" in df.columns:
        df["field_avg_or"] = df.groupby("race_id")["official_rating"].transform("mean")
        df["field_std_or"] = df.groupby("race_id")["official_rating"].transform("std").fillna(0)
        df["or_rank_in_race"] = df.groupby("race_id")["official_rating"].rank(
            ascending=False, method="min"
        )

    # Average Elo of the race field
    if "horse_elo" in df.columns:
        df["field_avg_elo"] = df.groupby("race_id")["horse_elo"].transform("mean")
        df["field_std_elo"] = df.groupby("race_id")["horse_elo"].transform("std").fillna(0)
        df["elo_pctile_in_race"] = df.groupby("race_id")["horse_elo"].rank(pct=True)

    return df


# ── Smoothed target encoding ────────────────────────────────────
def add_target_encoded_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smoothed target encoding for high-cardinality categorical entities.

    Encodes each entity's cumulative *mean normalised position*
    (0 = winner, 1 = last) with Bayesian shrinkage toward the global
    mean.  This captures entity-specific baselines that rolling stats
    and win-rate shrinkage don't fully cover — e.g. "this trainer's
    runners tend to finish in the top quarter regardless of conditions."

    All computations use prior-only (cumsum - current) to avoid leakage.
    """
    logger.info("Engineering target-encoded features …")
    hkey = _horse_key(df)

    # Normalised position: 0 = winner, 1 = last  (same as model's norm_pos)
    nr = df["num_runners"].replace(0, 1).values.astype(np.float32)
    df["_norm_pos"] = (
        (df["finish_position"].values.astype(np.float32) - 1)
        / np.maximum(nr - 1, 1)
    )

    # Use the expanding mean (prior-only) as the global prior so it
    # doesn't peek at future races.  The expanding mean converges
    # quickly (~0.45-0.50) so the prior is very stable.
    global_mean = 0.45  # neutral prior — close to the true dataset mean
    prior_strength = 15.0  # stronger shrinkage than win-rate (0.1 prior)

    # ---- Helper: target-encode one entity column ----
    def _te(entity_col: str, prefix: str, race_safe: bool = False):
        """Add ``{prefix}_te_norm_pos`` target-encoded feature."""
        if race_safe:
            cum_sum = _race_safe_cumsum(df, entity_col, "_norm_pos")
            cum_n = _race_safe_cumcount(df, entity_col)
        else:
            grp = df.groupby(entity_col)
            cum_sum = grp["_norm_pos"].cumsum() - df["_norm_pos"]
            cum_n = grp.cumcount()  # count of prior rows
        observed = np.full(np.shape(cum_sum), global_mean, dtype=float)
        np.divide(cum_sum, cum_n, out=observed, where=(cum_n > 0))
        # Bayesian shrinkage: (n * observed + m * prior) / (n + m)
        df[f"{prefix}_te_norm_pos"] = (
            (cum_n * observed
             + prior_strength * global_mean)
            / (cum_n + prior_strength)
        )

    # Core entities
    _te(hkey, "horse")
    if "jockey" in df.columns:
        _te("jockey", "jockey")
    if "trainer" in df.columns:
        _te("trainer", "trainer", race_safe=True)
    if "track" in df.columns:
        _te("track", "track", race_safe=True)
    if "going" in df.columns:
        _te("going", "going", race_safe=True)

    # Interaction: jockey × trainer combo
    if "jockey" in df.columns and "trainer" in df.columns:
        df["_jt_combo"] = df["jockey"].astype(str) + "_" + df["trainer"].astype(str)
        _te("_jt_combo", "jt", race_safe=True)
        df = df.drop(columns=["_jt_combo"], errors="ignore")

    # ── Windowed target encodings (recent-form baselines) ────────────────
    # These complement the lifetime cumulative encodings above.  Under
    # concept drift (long training windows) career averages shift away from
    # current form; a rolling 365-day window tracks the recent baseline.
    _te_w = getattr(config, "TE_WINDOW_DAYS", 365)
    if _te_w > 0 and "race_date" in df.columns:

        def _te_windowed(entity_col: str, prefix: str,
                         race_safe: bool = False) -> None:
            """Add ``{prefix}_te_norm_pos_{_te_w}d`` windowed encoding.

            When *race_safe* is True, aggregates to one observation per
            (entity, race_id) so that same-race runners sharing the
            entity value (e.g. all runners share the same going) cannot
            leak each other's outcomes.
            """
            if race_safe and "race_id" in df.columns:
                _agg = df.groupby([entity_col, "race_id"], sort=False).agg(
                    _np_sum=("_norm_pos", "sum"),
                    _np_cnt=("_norm_pos", "count"),
                    race_date=("race_date", "first"),
                ).reset_index()
                # Sum-of-individual-norm_pos in window
                _ws, _ = _time_window_stats(
                    _agg, entity_col, "race_date", "_np_sum", _te_w,
                )
                # Count-of-individual-runners in window
                _wc, _ = _time_window_stats(
                    _agg, entity_col, "race_date", "_np_cnt", _te_w,
                )
                _agg["_val_sum"] = _ws
                _agg["_val_cnt"] = _wc
                _lookup = _agg.set_index([entity_col, "race_id"])[
                    ["_val_sum", "_val_cnt"]
                ]
                _mi = pd.MultiIndex.from_arrays(
                    [df[entity_col], df["race_id"]]
                )
                _mapped = _lookup.reindex(_mi)
                val_sum = _mapped["_val_sum"].values.astype(float)
                val_cnt = _mapped["_val_cnt"].values.astype(float)
            else:
                val_sum, val_cnt = _time_window_stats(
                    df, entity_col, "race_date", "_norm_pos", _te_w,
                )
            observed = np.full(np.shape(val_sum), global_mean, dtype=float)
            np.divide(val_sum, val_cnt, out=observed, where=(val_cnt > 0))
            df[f"{prefix}_te_norm_pos_{_te_w}d"] = (
                (val_cnt * observed
                 + prior_strength * global_mean)
                / (val_cnt + prior_strength)
            )

        _te_windowed(hkey, "horse")
        if "jockey" in df.columns:
            _te_windowed("jockey", "jockey")
        if "trainer" in df.columns:
            _te_windowed("trainer", "trainer", race_safe=True)
        if "going" in df.columns:
            _te_windowed("going", "going", race_safe=True)

    # ── EWMA target encodings (smooth decay, no calendar cliff-edge) ─────
    # A race-count EWMA with half-life ≈ 10 races avoids the hard cutoff of
    # the 365d window while still down-weighting stale history — important
    # for horses/jockeys whose form drifts gradually over seasons.
    _te_hl = getattr(config, "TE_EWMA_HALF_LIFE_RACES", 10)
    if _te_hl > 0:
        _ewma_alpha = float(1 - np.exp(-np.log(2) / _te_hl))

        def _te_ewma(entity_col: str, prefix: str,
                     race_safe: bool = False) -> None:
            """Add ``{prefix}_te_ewma`` EWMA target-encoded feature.

            When *race_safe* is True, aggregates to one observation per
            (entity, race_id) first, computes the EWMA on that per-race
            series, then maps back — preventing same-race runners from
            leaking each other's outcomes.
            """
            if race_safe and "race_id" in df.columns:
                _dt_col = (
                    "_event_dt" if "_event_dt" in df.columns else "race_date"
                )
                _agg = (
                    df.groupby([entity_col, "race_id"], sort=False)
                    .agg(
                        _np_mean=("_norm_pos", "mean"),
                        _dt=(_dt_col, "first"),
                    )
                    .reset_index()
                    .sort_values([entity_col, "_dt", "race_id"])
                )
                _gids = _encode_groups(_agg, entity_col)
                _vals = _agg["_np_mean"].values.astype(np.float64)
                _shifted = _nb_grouped_shift1(_vals, _gids)
                _ewma_arr = _nb_ewma(_shifted, _gids, _ewma_alpha)
                # Map per-race EWMA back to individual runners
                _agg["_ewma"] = _ewma_arr
                _lookup = _agg.set_index([entity_col, "race_id"])["_ewma"]
                _mi = pd.MultiIndex.from_arrays(
                    [df[entity_col], df["race_id"]]
                )
                raw = _lookup.reindex(_mi).values
                cum_n = _race_safe_cumcount(df, entity_col).values
            else:
                ops = [(f"_{prefix}_te_ewma_raw", "ewma", _ewma_alpha)]
                results = _fast_grouped_rolling(
                    df, entity_col, "_norm_pos", ops,
                )
                raw = results[f"_{prefix}_te_ewma_raw"]
                cum_n = df.groupby(entity_col).cumcount()
            # Shrink toward prior for low-count entities
            df[f"{prefix}_te_ewma"] = (
                (cum_n * np.where(cum_n > 0, raw, global_mean)
                 + prior_strength * global_mean)
                / (cum_n + prior_strength)
            )

        _te_ewma(hkey, "horse")
        if "jockey" in df.columns:
            _te_ewma("jockey", "jockey")
        if "trainer" in df.columns:
            _te_ewma("trainer", "trainer", race_safe=True)
        if "going" in df.columns:
            _te_ewma("going", "going", race_safe=True)

    df = df.drop(columns=["_norm_pos"], errors="ignore")
    return df


def add_opposition_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strength-of-opposition features derived from Elo ratings.

    For each runner we compute three types of signal:

    1. **Current-race opposition** — average ``horse_elo`` of all
       co-runners *excluding* the horse itself.  This tells the model
       how tough *this* race is before it even starts.

    2. **Rolling prior-field quality** — per-horse rolling mean and EWMA
       of that per-race opposition Elo over its last N result races.
       This answers: "has this horse been competing in strong fields
       recently?"  A horse that finished 3rd against Group 1 quality
       is treated differently from one that won in a weak maiden.

    3. **Elo vs opposition** — horse's own Elo minus the field average
       (already computed by compute_elo_features as ``horse_elo_vs_field``,
       but here we add a *prior-race* EWMA version so the model can see
       whether the horse has been consistently above or below field level).

    Requires ``horse_elo`` to already be present (call after
    ``compute_elo_features``).
    """
    if "horse_elo" not in df.columns:
        return df

    logger.info("Engineering opposition-strength features…")
    df = df.copy()
    hkey = _horse_key(df)

    # ── 1. Per-race opposition mean Elo (excluding self) ──────────────
    # race_elo_sum / race_n give us the field total.
    # field_elo_excl_self = (sum - self) / (n - 1)
    race_elo_sum = df.groupby("race_id")["horse_elo"].transform("sum")
    race_n = df.groupby("race_id")["horse_elo"].transform("count")
    # Guard against single-runner races (n==1) to avoid division by zero
    df["field_avg_elo"] = np.where(
        race_n > 1,
        (race_elo_sum - df["horse_elo"]) / (race_n - 1),
        df["horse_elo"],   # fallback: own Elo (no opponents)
    )

    # ── 2. Rolling prior-field quality per horse ──────────────────────
    # field_avg_elo reflects THIS race's opposition, but for training
    # signal we need the PRIOR races' field quality (shifted by 1).
    # Use _fast_grouped_rolling which applies shift(1) internally.
    _opp_ops = [
        ("field_avg_elo_5",   "mean",  5),
        ("field_avg_elo_10",  "mean", 10),
    ]
    # Add EWMA half-life 7 (≈ last 10 races with decent weight)
    _ewma_hl = 7
    _ewma_alpha = 1.0 - np.exp(-np.log(2) / _ewma_hl)
    _opp_ops.append(("field_avg_elo_ewma", "ewma", _ewma_alpha))

    for name, arr in _fast_grouped_rolling(df, hkey, "field_avg_elo", _opp_ops).items():
        df[name] = arr

    # ── 3. Prior-race Elo-vs-field EWMA ──────────────────────────────
    # How consistently has this horse outrated / underrated its fields?
    df["_elo_vs_opp"] = df["horse_elo"] - df["field_avg_elo"]
    _vs_ops = [("elo_vs_field_ewma", "ewma", _ewma_alpha)]
    for name, arr in _fast_grouped_rolling(df, hkey, "_elo_vs_opp", _vs_ops).items():
        df[name] = arr
    df = df.drop(columns=["_elo_vs_opp"], errors="ignore")

    # Fill cold-start (first race: no prior field data) with neutral 1500
    for col in ("field_avg_elo_5", "field_avg_elo_10", "field_avg_elo_ewma"):
        df[col] = df[col].fillna(1500.0)
    df["elo_vs_field_ewma"] = df["elo_vs_field_ewma"].fillna(0.0)

    return df


# ── Cross-entity target encodings ────────────────────────────────
def add_cross_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target-encoded interaction features for entity × context combos
    not already covered by the per-entity functions.

    All use cumulative (prior-only) computation with shift(1) to
    avoid data leakage.
    """
    logger.info("Engineering cross-entity features...")
    hkey = _horse_key(df)

    # --- Jockey × going ---
    if "going" in df.columns:
        jg = df.groupby(["jockey", "going"])
        df["jockey_going_runs"] = jg.cumcount()
        df["_jg_w"] = jg["won"].cumsum() - df["won"]
        df["jockey_going_wr"] = np.where(
            df["jockey_going_runs"] > 2,
            df["_jg_w"] / df["jockey_going_runs"],
            0,
        )
        df = df.drop(columns=["_jg_w"], errors="ignore")

    # --- Jockey × distance category ---
    if "dist_category" in df.columns:
        df["_dc_str"] = df["dist_category"].astype(str)
        jdc = df.groupby(["jockey", "_dc_str"])
        df["jockey_dist_cat_runs"] = jdc.cumcount()
        df["_jdc_w"] = jdc["won"].cumsum() - df["won"]
        df["jockey_dist_cat_wr"] = np.where(
            df["jockey_dist_cat_runs"] > 2,
            df["_jdc_w"] / df["jockey_dist_cat_runs"],
            0,
        )
        df = df.drop(columns=["_dc_str", "_jdc_w"], errors="ignore")

    # --- Horse × going (win rate on this going type) ---
    if "going" in df.columns:
        hg = df.groupby([hkey, "going"])
        df["horse_going_runs"] = hg.cumcount()
        df["_hg_w"] = hg["won"].cumsum() - df["won"]
        df["horse_going_wr"] = np.where(
            df["horse_going_runs"] > 0,
            df["_hg_w"] / df["horse_going_runs"],
            0,
        )
        df = df.drop(columns=["_hg_w"], errors="ignore")

    # --- Horse × going × distance rounded (C&D&G specialisation) ---
    if "going" in df.columns and "distance_furlongs" in df.columns:
        df["_dist_r2"] = (df["distance_furlongs"] * 2).round() / 2
        hgd = df.groupby([hkey, "going", "_dist_r2"])
        df["horse_going_dist_runs"] = hgd.cumcount()
        df["_hgd_w"] = hgd["won"].cumsum() - df["won"]
        df["horse_going_dist_wr"] = np.where(
            df["horse_going_dist_runs"] > 0,
            df["_hgd_w"] / df["horse_going_dist_runs"],
            0,
        )
        df = df.drop(columns=["_dist_r2", "_hgd_w"], errors="ignore")

    # --- Trainer × race class (race-safe) ---
    if "race_class" in df.columns:
        df["_tc_key"] = df["trainer"].astype(str) + "||" + df["race_class"].astype(str)
        df["trainer_class_runs"] = _race_safe_cumcount(df, "_tc_key")
        df["_tc_w"] = _race_safe_cumsum(df, "_tc_key", "won")
        df["trainer_class_wr"] = np.where(
            df["trainer_class_runs"] > 2,
            df["_tc_w"] / df["trainer_class_runs"],
            0,
        )
        df = df.drop(columns=["_tc_w", "_tc_key"], errors="ignore")

    return df


# ── Conditional feature masking ──────────────────────────────────
# Some features are only meaningful for certain race types.
# Rather than building separate models, we zero-out irrelevant
# features so the tree splits ignore them naturally.
#
# Key insight: National Hunt (Hurdle / Chase) races have **no stall
# draws** — horses line up freely.  All draw-derived features are
# pure noise for NH and can actively hurt the model.

# Features that should be zeroed for NH races (Hurdle / Chase)
_FLAT_ONLY_FEATURES = [
    "draw_pct",
    "draw_x_bias",
    "draw_x_sprint_x_large",
    "draw_x_inv_dist",
    "draw_x_field_size",
    "draw_bias_x_sprint",
    "track_draw_win_rate",
    "tight_track_x_draw",
    "track_draw_bias",
    # AW (all-weather) is a Flat-only surface
    "track_is_aw",
]

# Features that should be zeroed for Flat races
_NH_ONLY_FEATURES = [
    # These are naturally ~0 for Flat due to how they're computed,
    # but explicit masking removes any residual noise.
    "age_x_nh",
]


def apply_conditional_feature_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-out features that are meaningless for certain race types.

    Draw-related features are set to zero for Hurdle / Chase because
    NH races have no stall draws.  This prevents the model from
    finding spurious correlations in random draw numbers assigned
    to NH runners in the data.

    Returns:
        DataFrame with masked features.
    """
    logger.info("Applying conditional feature masks (race-type gating)…")
    df = df.copy()

    # Determine NH flag
    if "race_type_Hurdle" in df.columns:
        is_nh = (
            df.get("race_type_Hurdle", 0).fillna(0)
            + df.get("race_type_Chase", 0).fillna(0)
        ).clip(0, 1).astype(bool)
    elif "race_type" in df.columns:
        is_nh = df["race_type"].isin(["Hurdle", "Chase"])
    else:
        logger.warning("Cannot determine race type — skipping masks")
        return df

    is_flat = ~is_nh
    n_nh = int(is_nh.sum())
    n_flat = int(is_flat.sum())

    masked_count = 0
    for col in _FLAT_ONLY_FEATURES:
        if col in df.columns:
            df.loc[is_nh, col] = 0.0
            masked_count += 1

    for col in _NH_ONLY_FEATURES:
        if col in df.columns:
            df.loc[is_flat, col] = 0.0
            masked_count += 1

    logger.info(
        f"  Masked {masked_count} features  "
        f"(Flat-only→zeroed for {n_nh:,} NH rows, "
        f"NH-only→zeroed for {n_flat:,} Flat rows)"
    )
    return df


def engineer_features(
    df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Processed DataFrame from data_processor
        save: Whether to save the feature-engineered data

    Returns:
        DataFrame with all engineered features
    """
    logger.info(f"Starting feature engineering on {len(df)} records...")

    # Ensure race_date is proper datetime (CSV round-trip turns it to str)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # Sort once — every sub-function needs chronological order for
    # cumsum / rolling / shift.  Sorting here avoids 12 redundant sorts.
    df["_event_dt"] = _event_sort_key(df)
    df = df.sort_values(["_event_dt", "race_id"]).copy()

    df = add_horse_features(df)
    df = add_jockey_features(df)
    df = add_trainer_features(df)
    df = add_jockey_trainer_features(df)
    df = add_jockey_track_features(df)
    df = add_track_features(df)
    df = add_course_distance_features(df)
    df = add_class_change_features(df)
    df = add_distance_change_features(df)
    df = add_target_encoded_features(df)  # smoothed entity baselines
    df = add_cross_entity_features(df)    # jockey×going, horse×going, etc.
    df = compute_elo_features(df)
    df = add_opposition_strength_features(df)  # needs horse_elo
    df = add_headgear_features(df)
    df = add_surface_features(df)
    df = add_trainer_specialisation_features(df)  # needs dist_category
    df = add_speed_figure_features(df)
    df = add_market_features(df)
    df = add_jockey_intent_features(df)     # needs is_favourite + jockey_elo
    df = add_race_context_features(df)
    df = add_field_quality_features(df)
    df = add_track_config_features(df)  # static venue data
    df = add_weather_features(df)       # Open-Meteo historical weather
    df = add_interaction_features(df)
    df = apply_conditional_feature_masks(df)  # zero irrelevant features

    # Defragment — the sub-functions above each add many columns
    # one-by-one, leaving the DataFrame highly fragmented in memory.
    df = df.copy()

    # ── Feature-specific NaN imputation ──────────────────────────
    # A blanket fillna(0) is harmful: Elo 0 means "terrible" (should be
    # 1500), speed figures 0 means "fastest ever" (should be median), etc.
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Elo columns: fill with default rating (unknown = average)
    elo_cols = [c for c in numeric_cols if "elo" in c.lower() and "delta" not in c.lower()]
    for c in elo_cols:
        df[c] = df[c].fillna(1500.0)

    # Speed figure columns: fill with column median (unknown = average)
    speed_cols = [c for c in numeric_cols if "speed_fig" in c.lower()]
    for c in speed_cols:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

    # Odds-based columns: fill with field median or neutral values
    odds_fill = {"implied_prob": 0.05, "norm_implied_prob": 0.05,
                 "log_odds": np.log1p(20.0), "odds_vs_field": 1.0,
                 "odds_rank": 5.0, "overround": 1.0}
    for c, fill_val in odds_fill.items():
        if c in df.columns:
            df[c] = df[c].fillna(fill_val)

    # Win-rate columns: shrunk versions already handle cold-start;
    # raw rates just need 0 (no history = no wins)
    # Days columns: fill with "no recent run"
    if "days_since_last_race" in df.columns:
        df["days_since_last_race"] = df["days_since_last_race"].fillna(60)
    if "days_log" in df.columns:
        df["days_log"] = df["days_log"].fillna(np.log1p(60))
    if "days_since_last_win" in df.columns:
        df["days_since_last_win"] = df["days_since_last_win"].fillna(365)
    if "horse_avg_lb_behind" in df.columns:
        df["horse_avg_lb_behind"] = df["horse_avg_lb_behind"].fillna(3.0)

    # Weather columns: fill with UK averages
    weather_defaults = {
        "weather_temp_max": 14.0, "weather_temp_min": 7.0,
        "weather_precip_mm": 1.5, "weather_wind_kmh": 18.0,
        "weather_precip_prev3": 4.5, "weather_temp_range": 7.0,
    }
    for c, val in weather_defaults.items():
        if c in df.columns:
            df[c] = df[c].fillna(val)

    # Everything else: fill with 0 (safe for counts, flags, rates)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Drop temporary sort helper.
    df = df.drop(columns=["_event_dt"], errors="ignore")

    if save:
        import os
        output_path = os.path.join(
            config.PROCESSED_DATA_DIR, "featured_races.parquet"
        )
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Saved feature-engineered data to {output_path}")

    logger.info(
        f"Feature engineering complete: {df.shape[1]} features, "
        f"{len(df)} records"
    )
    return df


if __name__ == "__main__":
    from src.data_processor import process_data

    processed = process_data(save=False)
    featured = engineer_features(processed)
    print(f"\nFeatured dataset shape: {featured.shape}")
    print(f"\nAll features:\n{list(featured.columns)}")
