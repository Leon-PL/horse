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
import ast
import os
from itertools import chain, repeat
from typing import NamedTuple

import numba as nb
import numpy as np
import pandas as pd

from src.ratings import compute_elo_features, compute_glicko_features
from src.fe_incremental import cumulative_counts, cumulative_counts_plain
from src.track_config import get_track_config, direction_code, shape_code
from src.rtv_features import add_rtv_features

import config
from src.source_dataset import dataset_build
from src.feature_context import counts_state_scope, current_context, current_counts_state
from src.feature_semantics import burden_weight, claim_adjusted_weight, distance_category
from src.foreign_form import history as foreign_history

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


# pandas warns past 100 blocks. Each consolidation copies the whole frame, so
# inside a stage it waits until only this much room is left.
_MID_STAGE_BLOCKS = 90


def _without(df: pd.DataFrame, columns) -> pd.DataFrame:
    """``df.drop(columns=columns, errors="ignore")`` without a whole-frame copy.

    ``drop`` rewrites every block, which costs real time at over a thousand
    columns. A working column sits in its own block, so deleting it in place
    leaves the rest of the frame alone. ``drop`` also consolidated the frame,
    which held long stages under pandas' block limit, so ``_defrag`` does that
    here, only when the frame is close to it.
    """
    for column in columns:
        if column in df.columns:
            del df[column]
    return _defrag(df, max_blocks=_MID_STAGE_BLOCKS)


def _with_columns(df: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add ``new`` beside ``df`` sharing their blocks instead of copying them."""
    return pd.concat([df, new], axis=1, copy=False)


def _defrag(df: pd.DataFrame, max_blocks: int = 48) -> pd.DataFrame:
    """Consolidate a fragmented DataFrame when needed.

    Each ``df[col] = ...`` adds one block to the BlockManager; the
    pipeline's stages add hundreds of columns this way. Past ~100 blocks
    pandas raises PerformanceWarning and every further insert degrades to
    O(n_blocks). Called between stages, this keeps the base block count
    low so no single stage's inserts can run the total up to the warning
    threshold mid-stage. Below *max_blocks* the whole-frame copy is skipped.
    """
    try:
        n_blocks = len(df._mgr.blocks)
    except Exception:
        return df
    return df.copy() if n_blocks > max_blocks else df


def _add_result_history_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate finished results, non-finishers, and pending racecards."""
    finish_pos = pd.to_numeric(df.get("finish_position", 0), errors="coerce").fillna(0)
    finished_mask = finish_pos > 0

    from src.result_status import NONSTARTER_LABELS, result_labels
    finish_labels = result_labels(df)
    non_finisher_mask = (~finished_mask) & finish_labels.ne("") & ~finish_labels.isin(NONSTARTER_LABELS)

    result_start_mask = finished_mask | non_finisher_mask

    df["_finished_result"] = finished_mask.astype(np.int8)
    df["_non_finisher_result"] = non_finisher_mask.astype(np.int8)
    df["_result_start"] = result_start_mask.astype(np.int8)
    df["_finish_pos_for_history"] = finish_pos.where(finished_mask)
    # Position as a share of the field (0 = winner, 1 = last), with a
    # non-finish counted as last. Raw positions scale with field size and
    # dropped non-finishes, so a faller looked like a horse with no run.
    field = (pd.to_numeric(df["num_runners"], errors="coerce") if "num_runners" in df
             else pd.Series(np.nan, index=df.index))
    field_share = ((finish_pos - 1) / (field - 1).clip(lower=1)).clip(0.0, 1.0)
    df["_norm_pos_for_history"] = field_share.where(
        finished_mask, pd.Series(np.where(non_finisher_mask, 1.0, np.nan), index=df.index)
    )
    df["_won_for_history"] = pd.to_numeric(df.get("won", 0), errors="coerce").where(result_start_mask)
    df["_placed_for_history"] = finish_pos.between(1, 3).astype(np.float32).where(result_start_mask)
    df["_result_start_for_history"] = pd.Series(
        np.where(result_start_mask, 1.0, np.nan),
        index=df.index,
        dtype=np.float32,
    )
    df["_non_finisher_for_history"] = pd.Series(
        np.where(result_start_mask, non_finisher_mask.astype(np.float32), np.nan),
        index=df.index,
        dtype=np.float32,
    )
    return df


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


@nb.njit(cache=True)
def _nb_time_window(dates_i8, values, group_ids, window_ns):
    """Numba-accelerated calendar-window sum & count.

    Args:
        dates_i8: int64 nanosecond timestamps, sorted within each group.
        values: float64 values to aggregate.
        group_ids: int64 contiguous group labels (must be sorted).
        window_ns: int64 window width in nanoseconds.

    Returns:
        (val_sum, count) — both float64 arrays of same length. Missing values
        are skipped in the sum (a non-start must not blank the whole window);
        count is every earlier row in the window.
    """
    n = len(dates_i8)
    val_sum = np.empty(n, dtype=np.float64)
    cnt = np.empty(n, dtype=np.float64)
    grp_start = 0  # first index of current group block
    for i in range(n):
        # Detect group boundary
        if i == 0 or group_ids[i] != group_ids[i - 1]:
            grp_start = i
        # Binary search for cutoff within [grp_start, i)
        cutoff = dates_i8[i] - window_ns
        lo = grp_start
        hi = i
        while lo < hi:
            mid = (lo + hi) >> 1
            if dates_i8[mid] < cutoff:
                lo = mid + 1
            else:
                hi = mid
        # Sum values in [lo, i)
        s = 0.0
        for j in range(lo, i):
            if not np.isnan(values[j]):
                s += values[j]
        val_sum[i] = s
        cnt[i] = float(i - lo)
    return val_sum, cnt


def _rolling_kernel(shifted, group_ids, op, param):
    """Run one rolling kernel over pre-shifted, group-sorted values."""
    if op == "mean":
        return _nb_rolling_mean(shifted, group_ids, int(param))
    if op == "sum":
        return _nb_rolling_sum(shifted, group_ids, int(param))
    if op == "min":
        return _nb_rolling_min(shifted, group_ids, int(param))
    if op == "max":
        return _nb_rolling_max(shifted, group_ids, int(param))
    if op == "std":
        return _nb_rolling_std(shifted, group_ids, int(param))
    if op == "ewma":
        return _nb_ewma(shifted, group_ids, float(param))
    raise ValueError(f"Unknown rolling op: {op}")


def _fast_grouped_rolling_core(
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

    # Race-safe broadcast (leak fix): a multi-runner entity (e.g. a trainer with
    # several runners in one race) has consecutive same-race rows after the sort,
    # so the per-row shift(1) only drops the current row and a later stablemate
    # would absorb an earlier stablemate's same-race result. Broadcast each
    # (entity, race) block's FIRST row — whose shift already points to the prior
    # race's last run — across the whole block, so every same-race runner sees
    # identical pre-race history. No-op for one-row blocks (a horse; a jockey:
    # one ride per race), the fix for multi-runner trainers. Routes through both
    # the full build and the seeded/incremental path (which calls this on `aug`).
    if "race_id" in df.columns and n > 0:
        _race_sorted = df["race_id"].values[_perm]
        _new_block = np.empty(n, dtype=bool); _new_block[0] = True
        _new_block[1:] = (gids[1:] != gids[:-1]) | (_race_sorted[1:] != _race_sorted[:-1])
        _block_first = np.maximum.accumulate(np.where(_new_block, np.arange(n), 0))
    else:
        _block_first = np.arange(n)

    results = {}
    for out_name, op, param in operations:
        arr = _rolling_kernel(shifted, gids, op, param)
        results[out_name] = arr[_block_first][_inv]   # race-safe broadcast, then back to row order
    return results


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
        prior_rate: Population base-rate (≈0.10 for win, ≈0.30 for place),
            or a per-row Series such as the entity's own overall rate.
        prior_strength: Pseudo-count for the prior (higher = more shrinkage).
    """
    n = n_samples.clip(lower=0)
    # A rate over no observations is undefined (NaN, see ``_rate``); it
    # carries no weight, so the estimate is the prior alone.
    observed = np.where(n > 0, observed_rate, 0.0)
    return (n * observed + prior_strength * prior_rate) / (n + prior_strength)


def _rate(numerator, denominator) -> np.ndarray:
    """``numerator / denominator``, missing where there is nothing to divide.

    A rate over zero runs is undefined, not 0: a zero reads as "never wins"
    for an entity that simply has no history yet.
    """
    return np.where(denominator > 0, numerator / denominator, np.nan)


def _shrink_to_entity(df: pd.DataFrame, rate_col: str, runs_col: str, entity: str) -> pd.Series:
    """Shrink a jockey/trainer × context win rate toward the entity's own rate.

    These context rates were raw and set to 0 below a few runs, so a thin
    context read as "never wins there". With 10 pseudo-runs at the entity's
    shrunk career rate, a thin context reads as the entity's usual rate and
    moves away from it only as runs in that context accumulate. Call after
    ``_entity_day_start_view`` so runs, rate and prior share the day start.
    """
    prior_col = f"{entity}_win_rate_shrunk"
    prior = df[prior_col] if prior_col in df.columns else 0.10
    return _bayesian_shrink(df[rate_col], df[runs_col], prior_rate=prior, prior_strength=10.0)


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


def _time_window_stats_core(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    value_col: str,
    window_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated calendar-window aggregation.

    For each row, compute the sum of ``value_col`` and the count of rows
    by the same ``entity_col`` within the preceding ``window_days``
    calendar days (exclusive of the current row).

    Returns:
        (value_sum, count) arrays aligned to ``df.index``.
    """
    # Sort by (entity, date) and encode into contiguous int groups
    sort_cols = [entity_col, date_col]
    ordered = df[sort_cols + [value_col]].copy()
    ordered["_orig_idx"] = np.arange(len(df))
    ordered = ordered.sort_values(sort_cols, kind="mergesort")

    gids = _encode_groups(ordered, entity_col)
    dates_i8 = pd.to_datetime(ordered[date_col]).values.astype("int64")
    vals = ordered[value_col].values.astype(np.float64)
    window_ns = np.int64(window_days) * np.int64(86_400_000_000_000)  # days→ns

    v_sum, v_cnt = _nb_time_window(dates_i8, vals, gids, window_ns)

    # Scatter results back to original row order
    orig_idx = ordered["_orig_idx"].values
    val_out = np.empty(len(df), dtype=np.float64)
    cnt_out = np.empty(len(df), dtype=np.float64)
    val_out[orig_idx] = v_sum
    cnt_out[orig_idx] = v_cnt
    return val_out, cnt_out


_DAY_NS = np.int64(86_400_000_000_000)


_NAT_I8 = np.iinfo(np.int64).min
_LATEST_I8 = np.iinfo(np.int64).max


def _nat_last(dates_i8: np.ndarray) -> np.ndarray:
    """Sort key putting missing dates last, as ``sort_values`` does."""
    return np.where(dates_i8 == _NAT_I8, _LATEST_I8, dates_i8)


def _time_window_with_seed(df, entity_col, date_col, value_col, window_days, seed):
    """Seed-aware ``_time_window_stats``: put each entity's carried recent rows
    (their REAL dates + values, within ``window_days``) ahead of its new rows so
    the calendar-window kernel runs unchanged on a new-only frame.

    ``seed``/``new_seed`` map entity -> list of ``(date_i8, value)`` within the
    last ``window_days`` of history — exactly what any later row's window can
    reach, so the append equals a full recompute. All work here is sized by the
    frame's entities, never by the whole carried state."""
    seed = seed or {}
    labels, uniques = pd.factorize(df[entity_col].astype(str).to_numpy())
    entities = uniques.tolist()
    dates_i8 = pd.to_datetime(df[date_col], errors="coerce").values.astype("int64")
    vals = df[value_col].to_numpy().astype(np.float64)

    tails = [seed.get(entity, ()) for entity in entities]
    sizes = np.fromiter(map(len, tails), np.int64, len(tails))
    carried = list(chain.from_iterable(tails))
    n_carried = len(carried)
    aug_label = np.concatenate([np.repeat(np.arange(len(tails)), sizes), labels])
    aug_date = np.concatenate(
        [np.fromiter((row[0] for row in carried), np.int64, n_carried), dates_i8])
    aug_val = np.concatenate(
        [np.fromiter((row[1] for row in carried), np.float64, n_carried), vals])

    # Stable, so ties keep carried rows first and new rows in frame order.
    order = np.lexsort((_nat_last(aug_date), aug_label))
    window_ns = np.int64(window_days) * _DAY_NS
    sorted_sum, sorted_count = _nb_time_window(
        aug_date[order], aug_val[order], aug_label[order], window_ns)
    v_sum = np.empty(len(order), dtype=np.float64)
    v_cnt = np.empty(len(order), dtype=np.float64)
    v_sum[order] = sorted_sum
    v_cnt[order] = sorted_count
    out = (v_sum[n_carried:], v_cnt[n_carried:])

    # Advance: keep each entity's rows within window_days of the frame's max date.
    cutoff = (aug_date.max() if len(aug_date) else np.int64(0)) - window_ns
    kept = np.flatnonzero(aug_date >= cutoff)
    kept = kept[np.argsort(aug_label[kept], kind="stable")]
    rows = list(zip(aug_date[kept].tolist(), aug_val[kept].tolist()))
    new_state = dict(seed)
    start = 0
    for entity, size in zip(entities, np.bincount(aug_label[kept], minlength=len(tails)).tolist()):
        if size:
            new_state[entity] = rows[start:start + size]
            start += size
    return out, new_state


def _time_window(df, entity_col, date_col, value_col, window_days):
    """Calendar-window stats, seed-aware under ``counts_state_scope`` (class B).

    Full build (no active state) calls the core directly — byte-identical.
    Under a seed scope it carries each entity's last ``window_days`` of rows so
    an append can run over new rows only. State keyed by (entity, value, days)."""
    if current_counts_state() is None:
        return _time_window_stats_core(df, entity_col, date_col, value_col, window_days)
    key = f"tw::{entity_col}::{value_col}::{window_days}"
    out, new_state = _time_window_with_seed(
        df, entity_col, date_col, value_col, window_days, current_counts_state().get(key)
    )
    current_counts_state()[key] = new_state
    return out


def _entity_day_start_view(
    df: pd.DataFrame,
    key_cols: list,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Make entity-history features date-strict: every run an entity has
    on one day receives the value computed at the entity's FIRST row of
    that day — which has seen strictly-earlier days only.

    Without this, a jockey's 14:30 features include his 13:30 result
    (and same-race stablemate rows, by row order). Those results do not
    exist at prediction time, so training would be systematically more
    informed than live serving. Horses essentially never run twice a
    day, so horse-keyed families don't need this; jockeys and trainers
    double constantly.

    ``key_cols`` must be the family's full history key (e.g.
    ``["jockey", "going"]`` for jockey×going stats) so dimensional
    features keep the current row's dimension.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols or "race_date" not in df.columns:
        return df
    keys = [df[k] if isinstance(k, str) else k for k in key_cols]
    day = pd.to_datetime(df["race_date"]).dt.normalize()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    first = pos.groupby([*keys, day], sort=False, dropna=True).transform("min")
    first = first.fillna(pos).to_numpy().astype(np.int64)
    if (first == np.arange(len(df))).all():
        return df
    for col in feature_cols:
        arr = df[col].to_numpy()
        df[col] = arr[first]
    return df


def _cumulative(keys, race_ids, metrics: dict) -> dict:
    """Seed-aware cumulative counts for a class-C stage.

    With no active seed (full build) this equals the inline
    ``groupby(key).cumsum() - flag`` formula. Under ``counts_state_scope`` it
    seeds from the carried state and writes the advanced state back, so the
    same stage code serves both build and append.
    """
    out, new_state = cumulative_counts(keys, race_ids, metrics, state=current_counts_state())
    if current_counts_state() is not None:
        current_counts_state().update(new_state)
    return out


def _cumulative_plain(keys, metrics: dict) -> dict:
    """Seed-aware **plain** cumulative counts (cumsum - current) for a class-C
    stage with at most one observation per entity per race. Mirrors
    ``_cumulative`` but threads ``PlainCountsAccumulator`` state."""
    out, new_state = cumulative_counts_plain(keys, metrics, state=current_counts_state())
    if current_counts_state() is not None:
        current_counts_state().update(new_state)
    return out


def _composite_key(df: pd.DataFrame, *cols) -> np.ndarray:
    """Single string key joining several columns for a per-(entity×dims) group.

    Used by the horse×dimension class-C stages so they can route through
    ``_cumulative`` (which keys on one entity array). The unit-separator never
    appears in the data, so this is collision-free vs a pandas multi-col groupby.
    """
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "\x1f" + df[c].astype(str)
    return s.to_numpy()


# Depth of per-entity raw-value tail carried as rolling state (class B). Bounded
# rolling ops (mean/sum/min/max/std, window ≤ this) are EXACT; EWMA (unbounded)
# is sufficient for bounded windows. Exponentially weighted means carry their
# actual accumulator separately: a truncated tail cannot represent long decay
# or an extended sequence of missing results.
_ROLL_SEED_CAP = 256


def _entity_str(df: pd.DataFrame, keys: list) -> np.ndarray:
    s = df[keys[0]].astype(str)
    for k in keys[1:]:
        s = s + "||" + df[k].astype(str)
    return s.to_numpy()


@nb.njit(cache=True)
def _nb_seeded_ewma_prior(values, group_ids, initial, alpha):
    """Emit the pre-row accumulator, then consume this row; retain terminal state."""
    n = len(values)
    prior = np.empty(n, dtype=np.float64)
    after = np.empty(n, dtype=np.float64)
    state = np.nan
    previous = -1
    for i in range(n):
        if group_ids[i] != previous:
            state = initial[i]
            previous = group_ids[i]
        prior[i] = state
        value = values[i]
        if not np.isnan(value):
            state = value if np.isnan(state) else alpha * value + (1.0 - alpha) * state
        after[i] = state
    return prior, after


class _Timeline(NamedTuple):
    """One entity grouping of a frame's rows, in history order."""
    sources: tuple            # the key, date and race arrays this was built from
    entities: list            # entity text by label
    labels: np.ndarray        # label of each row
    counts: np.ndarray        # rows per label
    order: np.ndarray         # rows by (entity, date, race id as text), stable
    history: np.ndarray       # the same with a missing date first: tail advance order
    sorted_labels: np.ndarray  # labels[order]
    opens_race: np.ndarray    # sorted row starts a new (entity, race) block


# Stages call ``_rolling`` dozens of times per frame on the same entity. The
# grouping is a pure function of ``sources``, which are compared by value, so
# this can only save work. It lives for one ``counts_state_scope``.


def _same_arrays(left: tuple, right: tuple) -> bool:
    return all(np.array_equal(a, b) for a, b in zip(left, right))


def _text_ranks(race_ids: np.ndarray) -> np.ndarray:
    """Rank of each id's string form: sort order of ``astype(str)``, unsorted rows."""
    labels, uniques = pd.factorize(race_ids, use_na_sentinel=False)
    text = np.array([str(value) for value in uniques.tolist()], dtype=object)
    ranks = np.empty(len(text), dtype=np.int64)
    ranks[np.argsort(text, kind="stable")] = np.arange(len(text))
    return ranks[labels]


def _timeline(df: pd.DataFrame, keys: list, date_col: str) -> _Timeline:
    key_arrays = [df[key].to_numpy() for key in keys]
    dates = pd.to_datetime(df[date_col], errors="coerce").to_numpy()
    n = len(df)
    races = df["race_id"].to_numpy() if "race_id" in df.columns else np.arange(n)
    sources = (*key_arrays, dates, races)
    slot = (tuple(keys), date_col)
    context = current_context()
    timelines = context.timelines if context is not None else {}
    known = timelines.get(slot)
    if known is not None and _same_arrays(known.sources, sources):
        return known

    labels, uniques = pd.factorize(_entity_str(df, keys))
    labels = labels.astype(np.int64)
    instants = dates.astype("datetime64[ns]").astype(np.int64)
    ranks = _text_ranks(races)
    order = np.lexsort((ranks, _nat_last(instants), labels))
    history = np.lexsort((ranks, instants, labels)) if (instants == _NAT_I8).any() else order
    sorted_labels = labels[order]
    sorted_ranks = ranks[order]
    opens_race = np.ones(n, dtype=bool)
    opens_race[1:] = (sorted_labels[1:] != sorted_labels[:-1]) | (sorted_ranks[1:] != sorted_ranks[:-1])
    line = _Timeline(tuple(np.array(source, copy=True) for source in sources), uniques.tolist(),
                     labels, np.bincount(labels, minlength=len(uniques)), order, history,
                     sorted_labels, opens_race)
    timelines[slot] = line
    return line


def _exact_seeded_ewmas(line, values, operations, seed):
    n = len(values)
    groups = line.sorted_labels
    first_block = np.maximum.accumulate(np.where(line.opens_race, np.arange(n), 0))
    last = np.ones(n, dtype=bool)
    last[:-1] = groups[1:] != groups[:-1]
    results, states = {}, {}
    for name, _, alpha in operations:
        alpha = float(alpha)
        if not 0 < alpha <= 1:
            raise ValueError("EWMA alpha must be in (0,1]")
        key = ("__exact_ewma__", name, alpha)
        if key not in seed and any(entity in seed for entity in line.entities):
            raise ValueError("Legacy truncated EWMA state cannot resume exactly; rebuild histories")
        carry = dict(seed.get(key, {}))
        start = np.fromiter(map(carry.get, line.entities, repeat(np.nan)), np.float64, len(line.entities))
        prior, after = _nb_seeded_ewma_prior(values, groups, start[groups], alpha)
        result = np.empty(n, dtype=np.float64)
        result[line.order] = prior[first_block]
        results[name] = result
        carry.update(zip(line.entities, after[last].tolist()))
        states[key] = carry
    return results, states


def _bounded_rolling(line, values, operations, seed):
    """Bounded windows over each entity's carried tail plus its new rows.

    Kernels re-read their window at every row, so a tail as long as the widest
    window gives the same bits as any longer one. Carried values sit directly
    ahead of the entity's rows, each its own race, as earlier history would.
    """
    n = len(values)
    depth = max(int(param) for _, _, param in operations)
    tails = [seed.get(entity) for entity in line.entities]
    taken = np.fromiter((min(len(tail), depth) if tail else 0 for tail in tails), np.int64, len(tails))
    carried = np.array(
        list(chain.from_iterable(tail[-size:] for tail, size in zip(tails, taken.tolist()) if size)),
        dtype=np.float64,
    )
    positions = np.arange(n) + np.cumsum(taken)[line.sorted_labels]
    total = n + carried.size
    is_row = np.zeros(total, dtype=bool)
    is_row[positions] = True
    aug_values = np.empty(total, dtype=np.float64)
    aug_values[positions] = values
    aug_values[~is_row] = carried
    groups = np.repeat(np.arange(len(tails)), taken + line.counts)
    opens = np.ones(total, dtype=bool)
    opens[positions] = line.opens_race
    first_block = np.maximum.accumulate(np.where(opens, np.arange(total), 0))[positions]
    shifted = _nb_grouped_shift1(aug_values, groups)
    results = {}
    for name, op, param in operations:
        result = np.empty(n, dtype=np.float64)
        result[line.order] = _rolling_kernel(shifted, groups, op, param)[first_block]
        results[name] = result
    return results


def _rolling_with_seed(df, group_key, value_col, operations, seed):
    """Seed-aware ``_fast_grouped_rolling``: resume each entity from its stored
    raw-value tail and return ``(results_in_df_order, new_seed)``.

    Exact across an append: new rows are strictly after history; a tail of
    ``_ROLL_SEED_CAP`` covers bounded windows. EWMA operations instead resume
    their exact per-entity accumulator, including across missing-result runs.
    ``seed`` maps entity strings to raw tails and reserved tuple keys to exact
    EWMA states; these are persisted in the existing counts-state pickle."""
    keys = [group_key] if isinstance(group_key, str) else list(group_key)
    seed = seed or {}
    if not len(df):
        return {name: np.empty(0, dtype=np.float64) for name, _, _ in operations}, dict(seed)
    date_col = "_event_dt" if "_event_dt" in df.columns else "race_date"
    line = _timeline(df, keys, date_col)
    raw = df[value_col].to_numpy()
    values = raw[line.order].astype(np.float64)

    bounded = [op for op in operations if op[1] != "ewma"]
    ewmas = [op for op in operations if op[1] == "ewma"]
    results = _bounded_rolling(line, values, bounded, seed) if bounded else {}

    # Advance state: last CAP raw values per entity, in chronological order.
    newest = raw[line.history].tolist()
    new_state = dict(seed)
    start = 0
    for entity, size in zip(line.entities, line.counts.tolist()):
        tail = seed.get(entity)
        recent = newest[start:start + size]
        new_state[entity] = (tail + recent if tail else recent)[-_ROLL_SEED_CAP:]
        start += size
    if ewmas:
        exact, states = _exact_seeded_ewmas(line, values, ewmas, seed)
        results.update(exact)
        new_state.update(states)
    return results, new_state


def _in_frame(seed: dict, keys: np.ndarray) -> dict:
    """The carried entries of this frame's entities.

    ``Series.map(dict)`` converts the whole dict on every call, which is what
    made each chunk slower than the last; only these entries can match.
    """
    return {key: seed[key] for key in pd.unique(keys).tolist() if key in seed}


def _last_rows(group, keys: np.ndarray) -> np.ndarray:
    """Mask of ``group.tail(1)``, without taking every column of a wide frame."""
    return (group.cumcount(ascending=False).to_numpy() == 0) & pd.notna(keys)


def _lag(df, entity_col, col):
    """Per-entity ``shift(1)`` (previous-row value), seed-aware under
    ``counts_state_scope`` (class L — bounded lookback of 1 row but unbounded in
    time). Full build calls ``groupby.shift(1)`` directly (byte-identical). Seeded:
    the first in-frame row of each entity takes the carried last value; state
    holds the last value per entity. Works for any dtype (numeric / date / str)."""
    g = df.groupby(entity_col, sort=False)
    shifted = g[col].shift(1)
    if current_counts_state() is None:
        return shifted
    key = f"lag::{entity_col}::{col}"
    seed = current_counts_state().get(key, {})
    entities = df[entity_col].to_numpy()
    carried = _in_frame(seed, entities)
    if carried:
        ent = pd.Series(entities, index=df.index)
        first_mask = pd.Series((g.cumcount() == 0).to_numpy(), index=df.index)
        shifted = shifted.where(~first_mask, ent.map(carried))
    last = _last_rows(g, entities)
    new_state = dict(seed)
    new_state.update(zip(entities[last], df[col].to_numpy()[last]))
    current_counts_state()[key] = new_state
    return shifted


def _running_extreme(df, entity_col, col, *, kind="max", prev=False):
    """Per-entity running max/min (optionally prior-only via grouped shift(1)),
    seed-aware (class C-like, unbounded). Full build is identical to
    ``groupby[col].cummax()/cummin()`` (then grouped ``.shift(1)`` if *prev*).
    State holds the running extreme per entity (numeric)."""
    g = df.groupby(entity_col, sort=False)
    cum = g[col].cummax() if kind == "max" else g[col].cummin()
    if prev:
        cum = cum.groupby(df[entity_col].to_numpy()).shift(1)
    if current_counts_state() is None:
        return cum

    # Seeded path. pandas ``cummax``/``cummin`` are skipna BUT keep NaN at NaN
    # input rows, and ``.shift(1)`` then propagates that NaN to the next row.
    # To match a full build exactly across a seam we carry two values per
    # entity: R = the running skipna extreme through end of history (to extend
    # the running max into new rows), and L = the cummax value AT the last
    # history row (which is NaN if that row's value was NaN) — that is what the
    # first new row's ``shift(1)`` must reproduce. Carrying only the skipna
    # extreme silently "revived" a career high that an intervening no-rating
    # race had nulled out in the full build.
    npf = np.fmax if kind == "max" else np.fmin
    state_key = f"ext::{kind}::{int(prev)}::{entity_col}::{col}"
    state = current_counts_state().get(state_key, {})  # entity -> (R, L)
    ent = df[entity_col]
    vals = df[col].to_numpy(dtype="float64")
    local = (g[col].cummax() if kind == "max" else g[col].cummin()).to_numpy(dtype="float64")
    carried = _in_frame(state, ent.to_numpy())
    R_row = ent.map({e: rl[0] for e, rl in carried.items()}).to_numpy(dtype="float64")
    # combined = full cummax restricted to this frame (NaN kept at NaN inputs).
    combined = np.where(np.isnan(vals), np.nan, npf(local, R_row))
    combined = pd.Series(combined, index=df.index)

    if prev:
        result = combined.groupby(df[entity_col].to_numpy()).shift(1)
        first = pd.Series((g.cumcount() == 0).to_numpy(), index=df.index)
        result = result.where(~first, ent.map({e: rl[1] for e, rl in carried.items()}))
    else:
        result = combined

    # Advance state: R' = extreme(R, frame skipna extreme); L' = combined at the
    # entity's last (chronological) row in this frame.
    frame_ext = g[col].max() if kind == "max" else g[col].min()
    is_last = (g.cumcount(ascending=False) == 0).to_numpy()
    last_combined = dict(zip(ent.to_numpy()[is_last], combined.to_numpy()[is_last]))
    new_state = dict(state)
    extremes = frame_ext.to_numpy(dtype="float64")
    frame_entities = frame_ext.index.tolist()
    prior = np.fromiter((carried.get(e, (np.nan,))[0] for e in frame_entities), np.float64, len(frame_entities))
    for e, extreme in zip(frame_entities, npf(prior, extremes).tolist()):
        new_state[e] = (extreme, last_combined.get(e, np.nan))
    current_counts_state()[state_key] = new_state
    return result


def _ffill_prev(df, entity_col, col):
    """Per-entity prior-only forward-fill (``groupby[col].ffill().shift(1)``),
    seed-aware (class L). State holds the last non-null value per entity (any
    dtype)."""
    g = df.groupby(entity_col, sort=False)
    filled = g[col].ffill()
    entities = df[entity_col].to_numpy()
    filled_prev = filled.groupby(entities).shift(1)
    if current_counts_state() is None:
        return filled_prev
    state_key = f"ffill::{entity_col}::{col}"
    seed = current_counts_state().get(state_key, {})
    carried = _in_frame(seed, entities)
    if carried:
        seed_vals = pd.Series(entities, index=df.index).map(carried)
        filled_prev = filled_prev.where(filled_prev.notna(), seed_vals)
    last = _last_rows(g, entities)
    new_state = dict(seed)
    for e, v in zip(entities[last], filled.to_numpy()[last]):
        if pd.notna(v):
            new_state[e] = v
    current_counts_state()[state_key] = new_state
    return filled_prev


@nb.njit(cache=True)
def _nb_expanding_prior(history, values):
    """Median and standard deviation of everything before each of ``values``.

    Reproduces ``x.shift(1).expanding()`` from the row after ``history``,
    bit for bit: infinities count as missing, the median is taken over the
    sorted values, and the deviation replays pandas' online Welford update
    with its Kahan term and its repeated-value rule. Cost is O(n log n) in
    history plus values, so a chunk never re-runs the history row by row.
    """
    n_history = len(history)
    n = len(values)
    pool = np.empty(n_history + n, dtype=np.float64)
    size = 0
    for source in (history, values):
        for value in source:
            if not (np.isnan(value) or np.isinf(value)):
                pool[size] = value
                size += 1
    ordered = np.sort(pool[:size])
    tree = np.zeros(size + 1, dtype=np.int64)
    top = 1
    while top * 2 <= size:
        top *= 2

    nobs = 0.0
    mean = 0.0
    squares = 0.0
    compensation = 0.0
    repeats = 0
    previous = np.nan
    medians = np.full(n, np.nan)
    deviations = np.full(n, np.nan)
    for i in range(n_history + n):
        at_row = i >= n_history
        value = values[i - n_history] if at_row else history[i]
        if at_row:
            row = i - n_history
            count = int(nobs)
            if count >= 1:
                wanted = count // 2 + 1
                found = 0
                for rank in (wanted, wanted - 1):
                    if rank < 1 or (count % 2 == 1 and rank != wanted):
                        continue
                    position = 0
                    remaining = rank
                    step = top
                    while step > 0:
                        probe = position + step
                        if probe <= size and tree[probe] < remaining:
                            position = probe
                            remaining -= tree[probe]
                        step >>= 1
                    if count % 2 == 1:
                        medians[row] = ordered[position]
                    elif found == 0:
                        medians[row] = ordered[position]
                        found = 1
                    else:
                        medians[row] = (medians[row] + ordered[position]) / 2
            if count > 1:
                variance = 0.0 if repeats >= count else squares / (nobs - 1.0)
                deviations[row] = np.sqrt(variance) if variance > 0.0 else 0.0
        if np.isnan(value) or np.isinf(value):
            continue
        nobs += 1.0
        repeats = repeats + 1 if value == previous else 1
        previous = value
        prior_mean = mean - compensation
        adjusted = value - compensation
        delta = adjusted - mean
        compensation = delta + mean - adjusted
        mean = mean + delta / nobs
        squares += (value - prior_mean) * (value - mean)
        index = int(np.searchsorted(ordered, value, side="left")) + 1
        while index <= size:
            tree[index] += 1
            index += index & (-index)
    return medians, deviations


def _expanding_prev(df, key_col, col, stats):
    """Per-key prior-only expanding stats (``shift(1).expanding().<stat>()``),
    seed-aware via FULL per-key history (option A — exact, unbounded state).
    *stats* ⊆ {'median','std'}. Returns ``{stat: Series in df order}``."""
    def _compute(gb):
        out = {}
        for st in stats:
            if st == "median":
                out[st] = gb.transform(lambda x: x.shift(1).expanding().median())
            else:
                out[st] = gb.transform(lambda x: x.shift(1).expanding().std())
        return out

    if current_counts_state() is None:
        return _compute(df.groupby(key_col, sort=False)[col])

    state_key = f"exp::{key_col}::{col}"
    seed = current_counts_state().get(state_key, {})  # key -> chronological list of values
    labels, uniques = pd.factorize(df[key_col].to_numpy(), use_na_sentinel=False)
    vals = df[col].to_numpy().astype("float64")
    medians = np.empty(len(df), dtype=np.float64)
    deviations = np.empty(len(df), dtype=np.float64)
    by_key = np.argsort(labels, kind="stable")
    ends = np.cumsum(np.bincount(labels, minlength=len(uniques))).tolist()
    # Unchanged histories can be shared: each touched list is replaced below.
    new_state = dict(seed)
    start = 0
    for key, end in zip(uniques.tolist(), ends):
        rows = by_key[start:end]
        start = end
        carried = seed.get(key, [])
        medians[rows], deviations[rows] = _nb_expanding_prior(
            np.array(carried, dtype=np.float64), vals[rows])
        new_state[key] = list(carried) + vals[rows].tolist()
    current_counts_state()[state_key] = new_state
    computed = {"median": medians, "std": deviations}
    return {st: pd.Series(computed[st], index=df.index) for st in stats}


def _distinct_count(df, entity_col, col):
    """Per-entity CAUSAL running distinct-value count: for each row, how many
    distinct non-null ``col`` values the entity has shown up to AND INCLUDING
    this race (chronological). This replaces the earlier whole-frame
    ``transform('nunique')``, which leaked the future — a row's count included
    values the entity would only show in LATER races, so it could not be
    computed incrementally and violated the no-leakage rule (CLAUDE.md #1).

    Order is made deterministic internally (sort by date + race_id) so the
    result is independent of the incoming row order, and identical whether
    computed in one full build or across appended chunks. Seed-aware under
    ``counts_state_scope``: state holds the set of values seen per entity, so an
    append continues the running count from where history left off."""
    n = len(df)
    date_col = "_event_dt" if "_event_dt" in df.columns else "race_date"
    sort_cols = [c for c in (date_col, "race_id") if c in df.columns]
    if sort_cols:
        perm = (
            df[sort_cols].reset_index(drop=True)
            .sort_values(sort_cols, kind="stable").index.to_numpy()
        )
    else:
        perm = np.arange(n)
    inv = np.empty(n, dtype=np.intp)
    inv[perm] = np.arange(n)

    ent_s = df[entity_col].to_numpy()[perm]
    val_s = df[col].to_numpy()[perm]
    notna_s = ~pd.isna(val_s)
    # First occurrence of each (entity, value) among non-null rows, in order.
    first_s = notna_s & (
        ~pd.DataFrame({"e": ent_s, "v": val_s}).duplicated(["e", "v"]).to_numpy()
    )

    state_key = f"nuniq::{entity_col}::{col}"
    seed = current_counts_state().get(state_key, {}) if current_counts_state() is not None else {}

    if seed:
        # A first-in-frame value only increments if not already carried.
        inc_s = first_s.copy()
        for row in np.flatnonzero(first_s).tolist():
            if val_s[row] in seed.get(ent_s[row], ()):
                inc_s[row] = False
        inc_s = inc_s.astype(np.int64)
        base = pd.Series(ent_s).map(
            {e: len(vs) for e, vs in _in_frame(seed, ent_s).items()}
        ).fillna(0).to_numpy()
    else:
        inc_s = first_s.astype(np.int64)
        base = 0

    run_s = base + pd.Series(inc_s).groupby(ent_s).cumsum().to_numpy()
    out = pd.Series(run_s[inv].astype("float64"), index=df.index)

    if current_counts_state() is not None:
        # Set union replaces touched entries without mutating the seed's sets.
        new_state = dict(seed)
        seen = pd.DataFrame({"e": ent_s[notna_s], "v": val_s[notna_s]}).drop_duplicates()
        grown: dict = {}
        for entity, value in zip(seen["e"].tolist(), seen["v"].tolist()):
            values = grown.get(entity)
            if values is None:
                values = grown[entity] = set(seed.get(entity, ()))
            values.add(value)
        new_state.update(grown)
        current_counts_state()[state_key] = new_state
    return out


def _rolling(df, group_key, value_col, operations) -> dict:
    """Rolling features, seed-aware under ``counts_state_scope`` (class B).

    Full build (no active state) calls ``_fast_grouped_rolling`` directly —
    byte-identical to before. Under a seed scope it carries a per-entity raw
    tail so an append can run over new rows only. State is keyed by the first
    output name (globally unique)."""
    if current_counts_state() is None:
        results = _fast_grouped_rolling_core(df, group_key, value_col, operations)
        # The persisted/seeded engine groups bounded operations before EWMA.
        # Give the reference sweep that same schema without changing values.
        ordered = [op for op in operations if op[1] != "ewma"] + [op for op in operations if op[1] == "ewma"]
        return {name: results[name] for name, _, _ in ordered}
    state_key = f"roll::{operations[0][0]}"
    results, new_state = _rolling_with_seed(
        df, group_key, value_col, operations, current_counts_state().get(state_key)
    )
    current_counts_state()[state_key] = new_state
    return results


def _with_foreign_starts(df, stage, history_columns):
    """Create a private horse timeline and project only domestic rows back."""
    if df.empty or _horse_key(df) != "horse_id":
        return df, None
    state = current_counts_state()
    key = f"foreign_form::{stage}"
    through = df.race_date.max()
    if "prediction_as_of" in df:
        decision_day = pd.to_datetime(df.prediction_as_of, utc=True, errors="raise").dt.tz_localize(None)
        # Historical training rows use race date; a live request can be
        # prepared in advance. Use the earliest decision day in that batch.
        earliest = decision_day.min()
        if pd.notna(earliest):
            through = min(through, earliest.normalize())
    foreign = foreign_history(through=through, after=state.get(key) if state is not None else None)
    if state is not None:
        state[key] = through
    if foreign.empty:
        return df, None
    from src.runner_identity import horse_ids
    domestic_ids = horse_ids(df)
    # An unseeded calculation only needs foreign histories for its target
    # horses. A seeded build also consumes runs for horses absent from this chunk.
    if state is None:
        foreign = foreign.loc[foreign.horse_id.isin(domestic_ids)]
    if foreign.empty:
        return df, None
    required = ["horse_id", "race_id", "race_date", *history_columns]
    domestic = df.reindex(columns=required).reset_index(drop=True)
    domestic["race_id"] = domestic.race_id.astype(str)
    domestic["_event_dt"] = df.get("_event_dt", df.race_date).to_numpy()
    foreign = foreign.copy()
    foreign["horse_id"] = foreign.horse_id.astype(df.horse_id.dtype)
    foreign["race_id"] = "foreign:" + foreign.provider + ":" + foreign.native_race_id
    foreign["won"] = foreign.finish_position.eq(1).astype(int)
    foreign = _add_result_history_flags(foreign)
    foreign = foreign.reindex(columns=domestic.columns)
    work = pd.concat([domestic, foreign], ignore_index=True).sort_values(["_event_dt", "race_id"])
    return work, domestic.index


def _roll_horse_starts(df, column, operations):
    work, positions = _with_foreign_starts(df, column, [column])
    values = _rolling(work, _horse_key(df), column, operations)
    if positions is None:
        return values
    return {name: pd.Series(value, index=work.index).loc[positions].to_numpy()
            for name, value in values.items()}


def add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["_result_start", "_finished_result", "_non_finisher_result", "_placed_for_history",
               "_norm_pos_for_history", "_won_for_history", "_non_finisher_for_history", "won",
               "race_type", "age", "weight_lbs", "reported_jockey_claim_lbs", "has_observed_weight",
               "lengths_behind", "beaten_lengths", "finish_position",
               "form_last_pos", "form_avg", "form_wins_count", "form_length"]
    work, positions = _with_foreign_starts(df, "form", [c for c in columns if c in df])
    if positions is None:
        return _add_horse_features(df)
    inputs = set(work.columns)
    engineered = _add_horse_features(work)
    updated = {"horse_runs", "horse_wins", "horse_places", "form_last_pos", "form_avg",
               "form_wins_count", "form_length"}
    for col in engineered:
        if col not in inputs or col in updated:
            df[col] = engineered.loc[positions, col].to_numpy()
    return df


def _add_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate horse-specific historical features.

    For each horse at each race, these features represent its performance
    in PRIOR races only (no data leakage).
    """
    logger.info("Engineering horse features...")

    hkey = _horse_key(df)

    # --- Cumulative horse stats (shifted to avoid leakage) ---
    result_start_flag = df["_result_start"]
    finished_flag = df["_finished_result"]
    non_finisher_flag = df["_non_finisher_result"]
    placed_hist = df["_placed_for_history"].fillna(0.0)
    norm_pos_hist = df["_norm_pos_for_history"].fillna(0.0)

    # Cumulative (prior-only, race-safe) career counts via the incremental
    # primitive. With state=None (a full build) this is identical to the
    # historical ``groupby(hkey)[v].cumsum() - v`` formula; the FeatureStore
    # append path can seed it from carried state so it never recomputes
    # history. Class C — see docs/architecture/FE_TAXONOMY.md. (Horses run once per race, so
    # race-safe == plain cumsum-minus-current here.)
    _counts = _cumulative(
        df[hkey].to_numpy(), df["race_id"].to_numpy(),
        {
            "horse_prev_races": result_start_flag.to_numpy(),
            "horse_prev_finishes": finished_flag.to_numpy(),
            "horse_cum_wins": df["won"].to_numpy(),
            "horse_cum_places": placed_hist.to_numpy(),
            "horse_cum_non_finishes": non_finisher_flag.to_numpy(),
            "horse_cum_npos_sum": norm_pos_hist.to_numpy(),
        },
    )
    df["horse_prev_races"] = _counts["horse_prev_races"].astype(np.int64)
    df["horse_prev_finishes"] = _counts["horse_prev_finishes"].astype(np.int64)

    # Cumulative win rate (vectorised — avoids lambda overhead)
    df["horse_cum_wins"] = _counts["horse_cum_wins"].astype(np.int64)
    df["horse_win_rate"] = _rate(df["horse_cum_wins"], df["horse_prev_races"])
    # Bayesian-shrunk win rate (reliable even with few runs)
    df["horse_win_rate_shrunk"] = _bayesian_shrink(
        df["horse_win_rate"], df["horse_prev_races"],
        prior_rate=0.10, prior_strength=8.0,
    )

    # Cumulative place rate (top 3)
    df["horse_cum_places"] = _counts["horse_cum_places"].astype(np.float32)
    df["horse_place_rate"] = _rate(df["horse_cum_places"], df["horse_prev_races"])
    # Bayesian-shrunk place rate
    df["horse_place_rate_shrunk"] = _bayesian_shrink(
        df["horse_place_rate"], df["horse_prev_races"],
        prior_rate=0.30, prior_strength=8.0,
    )

    df["horse_cum_non_finishes"] = _counts["horse_cum_non_finishes"].astype(np.int64)
    df["horse_non_finish_rate"] = _rate(df["horse_cum_non_finishes"], df["horse_prev_races"])
    df["horse_non_finish_rate_shrunk"] = _bayesian_shrink(
        df["horse_non_finish_rate"], df["horse_prev_races"],
        prior_rate=0.05, prior_strength=10.0,
    )
    df["horse_last_run_non_finish"] = (
        _lag(df, hkey, "_non_finisher_result").fillna(0).astype(np.int8)
    )

    # Override the static scraped lifetime stats with properly computed
    # point-in-time values so the model sees cumulative counts that
    # only reflect prior races, not future ones.
    df["horse_runs"] = df["horse_prev_races"].astype(int)
    df["horse_wins"] = df["horse_cum_wins"].astype(int)
    df["horse_places"] = df["horse_cum_places"].astype(int)

    # Career average position as a share of the field (0 = won, 1 = last,
    # non-finish = last); missing for a debutant.
    df["horse_cum_npos_sum"] = _counts["horse_cum_npos_sum"].astype(np.float64)
    df["horse_avg_position"] = _rate(df["horse_cum_npos_sum"], df["horse_prev_races"])

    # --- Rolling form (Numba-accelerated) ---
    # Build all rolling operations on finish_position and won in one batch
    _pos_ops = []
    _won_ops = []
    for w in config.ROLLING_WINDOWS:
        _pos_ops.append((f"horse_avg_pos_{w}", "mean", w))
        _won_ops.append((f"horse_wins_{w}", "sum", w))

    # Best/worst/consistency (all on the field-share position)
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
    _dnf_ops = [
        ("horse_non_finish_rate_5", "mean", 5),
        ("horse_non_finish_rate_10", "mean", 10),
    ]

    # Execute batched rolling on the field-share position
    for name, arr in _rolling(df,hkey, "_norm_pos_for_history", _pos_ops).items():
        df[name] = arr
    # Execute batched rolling on won
    for name, arr in _rolling(df,hkey, "_won_for_history", _won_ops).items():
        df[name] = arr
    for name, arr in _rolling(df,hkey, "_non_finisher_for_history", _dnf_ops).items():
        df[name] = arr

    # Rolling form over no prior runs stays missing: a placeholder (std 3.0,
    # win share 0) is indistinguishable from a real record.

    # --- Days since last race (missing for a debutant, not 60) ---
    df["horse_last_race_date"] = _lag(df, hkey, "race_date")
    df["days_since_last_race"] = (
        (df["race_date"] - df["horse_last_race_date"]).dt.days
    )

    # --- Days since last run — non-linear buckets ---
    df["days_log"] = np.log1p(df["days_since_last_race"])
    df["days_bucket_fresh"] = (df["days_since_last_race"] <= 14).astype(int)
    df["days_bucket_normal"] = df["days_since_last_race"].between(15, 28).astype(int)
    df["days_bucket_break"] = (df["days_since_last_race"] > 60).astype(int)

    # --- Race-type-aware horse features ---
    if "race_type" in df.columns:
        result_start_hist = df["_result_start"].fillna(0.0)
        won_hist = df["_won_for_history"].fillna(0.0)
        _rt = _cumulative(
            _composite_key(df, hkey, "race_type"), df["race_id"].to_numpy(),
            {"horse_rt_runs": result_start_hist.to_numpy(), "horse_rt_wins": won_hist.to_numpy()},
        )
        df["horse_rt_runs"] = _rt["horse_rt_runs"]
        df["horse_rt_wins"] = _rt["horse_rt_wins"]
        df["horse_rt_win_rate"] = _rate(df["horse_rt_wins"], df["horse_rt_runs"])
        df["horse_rt_wr_shrunk"] = _bayesian_shrink(
            df["horse_rt_win_rate"], df["horse_rt_runs"],
            prior_rate=0.10, prior_strength=6.0,
        )
        df["horse_rt_specialist_edge"] = (
            df["horse_rt_wr_shrunk"] - df["horse_win_rate_shrunk"]
        )
        df = _without(df, ["horse_rt_wins"])

    # --- Change in weight burden from the previous run ---
    if "weight_lbs" in df.columns:
        weight = claim_adjusted_weight(df)
        # Comparisons use the allocated weight where the claim is unverified (see burden_weight).
        burden = burden_weight(df)
        history = pd.DataFrame({hkey: df[hkey], "_burden_weight_lbs": burden}, index=df.index)
        change = burden - _lag(history, hkey, "_burden_weight_lbs")
        # Compare against the field's observed changes, with at least one rival.
        changes = change.groupby(df["race_id"], sort=False)
        race_mean = changes.transform("mean").where(changes.transform("count").ge(2))
        df = _with_columns(df, pd.DataFrame({
            "claim_adjusted_weight_lbs": weight,
            "weight_change": change,
            "weight_change_vs_field": change - race_mean,
            "is_big_weight_drop": change.lt(-3).astype(float).where(change.notna()),
        }, index=df.index))

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
                df["horse_avg_pos_10"].values.astype(np.float32)
                - df["horse_avg_pos_3"].values.astype(np.float32)
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

    # --- Runs since last win (missing for a horse that has never won) ---
    # Starts up to and including each win, carried forward to later runs.
    df["_starts_through_win"] = (df["horse_prev_races"] + 1).where(df["_won_for_history"].eq(1))
    df["runs_since_last_win"] = (
        df["horse_prev_races"] - _ffill_prev(df, hkey, "_starts_through_win")
    )

    # --- Days since last win (missing for a maiden, not 365) ---
    df["_last_win_date"] = df["race_date"].where(df["won"] == 1)
    df["_last_win_date"] = _ffill_prev(df, hkey, "_last_win_date")
    df["days_since_last_win"] = (
        (df["race_date"] - df["_last_win_date"]).dt.days
    )

    # --- Average lengths behind (Numba-accelerated) ---
    # Non-finishers (PU/F/UR) have NaN lengths_behind — exclude them
    # from rolling aggregates so they don't pollute form signals.
    if "lengths_behind" in df.columns:
        from src.result_margins import margin_values
        _lb_finished = margin_values(df).copy()
        # Mask non-finishers: finish_position <= 0 ⇒ NaN
        if "finish_position" in df.columns:
            _lb_finished = _lb_finished.where(
                df["finish_position"].fillna(0).astype(int) >= 1
            )

        df["_lb_finished"] = _lb_finished

        _lb_ops = [
            ("horse_avg_lb_behind", "mean", 5),
        ]
        for hl in [3, 7]:
            alpha = 1 - np.exp(-np.log(2) / hl)
            _lb_ops.append((f"horse_ewma_lb_{hl}", "ewma", alpha))

        # Close finish rate — only among finished races
        df["_close_finish"] = (_lb_finished <= 2.0).astype(float)
        df.loc[_lb_finished.isna(), "_close_finish"] = np.nan
        _cf_ops = [("close_finish_rate", "mean", 5)]

        for name, arr in _rolling(df,hkey, "_lb_finished", _lb_ops).items():
            df[name] = arr
        for name, arr in _rolling(df,hkey, "_close_finish", _cf_ops).items():
            df[name] = arr

        df = _without(df, ["_close_finish", "_lb_finished"])

    # Clean up intermediate columns
    df = _without(df, [
            "horse_cum_wins",
            "horse_cum_places",
            "horse_cum_non_finishes",
            "horse_cum_npos_sum",
            "horse_last_race_date",
            "_last_win_date",
            "_starts_through_win",
        ])

    return df


def add_established_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comparable marks and evidence from today's actual racing discipline."""
    from src.established_form_features import STATE_KEY, established_form_features
    seed = current_counts_state().get(STATE_KEY) if current_counts_state() is not None else None
    values, advanced = established_form_features(df, state=seed, return_state=True)
    for column in values:
        df[column] = values[column]
    if current_counts_state() is not None:
        current_counts_state()[STATE_KEY] = advanced
    return df


_PRIOR_MARKET_STATE_KEY = "prior_market::v1"


def add_prior_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Earlier market opinions and rated opposition, with no current-price input."""
    from src.prior_market_features import FEATURES, prior_market_features
    seed = current_counts_state().get(_PRIOR_MARKET_STATE_KEY) if current_counts_state() is not None else None
    # Direct minimal-frame callers can lack measurements. Keep those unknown;
    # do not add fabricated measurement columns to the caller's dataframe.
    missing = {c: np.nan for c in ("odds", "official_rating", "won", "finish_position", "num_runners") if c not in df}
    work = df.assign(**missing) if missing else df
    values, advanced = prior_market_features(work, state=seed, return_state=True)
    for column in FEATURES:
        df[column] = values[column]
    if current_counts_state() is not None:
        current_counts_state()[_PRIOR_MARKET_STATE_KEY] = advanced
    return df


def add_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate jockey-specific historical features."""
    logger.info("Engineering jockey features...")

    jockey_groups = df.groupby("jockey", sort=False)
    result_start_flag = df["_result_start"]
    placed_hist = df["_placed_for_history"].fillna(0.0)

    # Cumulative (prior-only, race-safe) counts via the incremental primitive
    # (state=None == inline cumsum-minus-current; seed-capable for append).
    # A jockey rides once per race, so race-safe == plain. Class C.
    _counts = _cumulative(
        df["jockey"].to_numpy(), df["race_id"].to_numpy(),
        {
            "jockey_prev_rides": result_start_flag.to_numpy(),
            "jockey_cum_wins": df["won"].to_numpy(),
            "jockey_cum_places": placed_hist.to_numpy(),
        },
    )

    # Number of previous rides
    df["jockey_prev_rides"] = _counts["jockey_prev_rides"].astype(np.int64)

    # Cumulative win rate (vectorised)
    df["jockey_cum_wins"] = _counts["jockey_cum_wins"].astype(np.int64)
    df["jockey_win_rate"] = _rate(df["jockey_cum_wins"], df["jockey_prev_rides"])
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

    for col, ops in [("_won_for_history", _jock_won_ops), ("_norm_pos_for_history", _jock_pos_ops)]:
        results = _rolling(df,"jockey", col, ops)
        for name, arr in results.items():
            df[name] = arr

    # Place rate (top 3)
    df["jockey_cum_places"] = _counts["jockey_cum_places"].astype(np.float32)
    df["jockey_place_rate"] = _rate(df["jockey_cum_places"], df["jockey_prev_rides"])
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
        _places_col = f"jockey_places_{days}d"
        _pr_col = f"jockey_pr_{days}d"

        w_arr, _ = _time_window(df, "jockey", "race_date", "_won_for_history", days)
        r_arr, _ = _time_window(df, "jockey", "race_date", "_result_start_for_history", days)
        p_arr, _ = _time_window(df, "jockey", "race_date", "_placed_for_history", days)
        df[_wins_col] = w_arr
        df[_rides_col] = r_arr
        df[_places_col] = p_arr
        with np.errstate(invalid="ignore"):
            df[_wr_col] = np.where(r_arr > 2, w_arr / r_arr, 0)
            df[_pr_col] = np.where(r_arr > 2, p_arr / r_arr, 0)

    for days in [14, 30]:
        df[f"jockey_wr_{days}d_delta"] = (
            df[f"jockey_wr_{days}d"] - df["jockey_win_rate_shrunk"]
        )
        df[f"jockey_pr_{days}d_delta"] = (
            df[f"jockey_pr_{days}d"] - df["jockey_place_rate_shrunk"]
        )

    df["jockey_hot_vs_30d"] = df["jockey_wr_14d"] - df["jockey_wr_30d"]
    df = _without(df, ["jockey_cum_wins", "jockey_cum_places"])
    df = _entity_day_start_view(df, ["jockey"], [
        "jockey_prev_rides", "jockey_win_rate", "jockey_win_rate_shrunk",
        "jockey_wins_10", "jockey_wins_20", "jockey_avg_pos_10", "jockey_avg_pos_20",
        "jockey_place_rate", "jockey_place_rate_shrunk",
        "jockey_wins_14d", "jockey_rides_14d", "jockey_places_14d",
        "jockey_wr_14d", "jockey_pr_14d",
        "jockey_wins_30d", "jockey_rides_30d", "jockey_places_30d",
        "jockey_wr_30d", "jockey_pr_30d",
        "jockey_wr_14d_delta", "jockey_pr_14d_delta",
        "jockey_wr_30d_delta", "jockey_pr_30d_delta", "jockey_hot_vs_30d",
    ])
    return df


def add_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trainer-specific historical features."""
    logger.info("Engineering trainer features...")

    trainer_groups = df.groupby("trainer", sort=False)
    placed_hist = df["_placed_for_history"].fillna(0.0)

    # Race-safe: trainers often have multiple runners in the same race.
    # All same-race runners must see identical prior-race stats. Routed through
    # the seed-aware cumulative primitive (class C; see docs/architecture/FE_TAXONOMY.md) so
    # the FeatureStore append can carry state instead of recomputing history.
    _tc = _cumulative(
        df["trainer"].to_numpy(), df["race_id"].to_numpy(),
        {
            "trainer_prev_runs": df["_result_start"].to_numpy(),
            "trainer_cum_wins": df["won"].to_numpy(),
            "trainer_cum_places": placed_hist.to_numpy(),
        },
    )
    df["trainer_prev_runs"] = _tc["trainer_prev_runs"]
    df["trainer_cum_wins"] = _tc["trainer_cum_wins"]
    df["trainer_win_rate"] = _rate(df["trainer_cum_wins"], df["trainer_prev_runs"])
    df["trainer_win_rate_shrunk"] = _bayesian_shrink(
        df["trainer_win_rate"], df["trainer_prev_runs"],
        prior_rate=0.10, prior_strength=10.0,
    )

    # Rolling form  (Numba-accelerated)
    _trainer_won_ops = []
    for w in [10, 20]:
        _trainer_won_ops.append((f"trainer_wins_{w}", "sum", w))
    results = _rolling(df,"trainer", "won", _trainer_won_ops)
    for name, arr in results.items():
        df[name] = arr

    # Place rate
    df["trainer_cum_places"] = _tc["trainer_cum_places"]
    df["trainer_place_rate"] = _rate(df["trainer_cum_places"], df["trainer_prev_runs"])
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
        _places_col = f"trainer_places_{days}d"
        _pr_col = f"trainer_pr_{days}d"

        w_arr, _ = _time_window(df, "trainer", "race_date", "_won_for_history", days)
        r_arr, _ = _time_window(df, "trainer", "race_date", "_result_start_for_history", days)
        p_arr, _ = _time_window(df, "trainer", "race_date", "_placed_for_history", days)
        df[_wins_col] = w_arr
        df[_runs_col] = r_arr
        df[_places_col] = p_arr
        with np.errstate(invalid="ignore"):
            df[_wr_col] = np.where(r_arr > 2, w_arr / r_arr, 0)
            df[_pr_col] = np.where(r_arr > 2, p_arr / r_arr, 0)

    for days in [14, 30]:
        df[f"trainer_wr_{days}d_delta"] = (
            df[f"trainer_wr_{days}d"] - df["trainer_win_rate_shrunk"]
        )
        df[f"trainer_pr_{days}d_delta"] = (
            df[f"trainer_pr_{days}d"] - df["trainer_place_rate_shrunk"]
        )

    df["trainer_hot_vs_30d"] = df["trainer_wr_14d"] - df["trainer_wr_30d"]

    df = _without(df, ["trainer_cum_wins", "trainer_cum_places"])
    df = _entity_day_start_view(df, ["trainer"], [
        "trainer_prev_runs", "trainer_win_rate", "trainer_win_rate_shrunk",
        "trainer_wins_10", "trainer_wins_20",
        "trainer_place_rate", "trainer_place_rate_shrunk",
        "trainer_wins_14d", "trainer_runs_14d", "trainer_places_14d",
        "trainer_wr_14d", "trainer_pr_14d",
        "trainer_wins_30d", "trainer_runs_30d", "trainer_places_30d",
        "trainer_wr_30d", "trainer_pr_30d",
        "trainer_wr_14d_delta", "trainer_pr_14d_delta",
        "trainer_wr_30d_delta", "trainer_pr_30d_delta", "trainer_hot_vs_30d",
    ])

    # --- Trainer at track (race-safe) ---
    if "track" in df.columns:
        # Create a combined key so _race_safe helpers can group correctly
        df["_tt_key"] = df["trainer"].astype(str) + "||" + df["track"].astype(str)
        _ttc = _cumulative(
            df["_tt_key"].to_numpy(), df["race_id"].to_numpy(),
            {"trainer_track_runs": df["_result_start"].to_numpy(), "trainer_track_wins": df["won"].to_numpy()},
        )
        df["trainer_track_runs"] = _ttc["trainer_track_runs"]
        df["trainer_track_wins"] = _ttc["trainer_track_wins"]
        df["trainer_track_win_rate"] = _rate(df["trainer_track_wins"], df["trainer_track_runs"])
        df = _entity_day_start_view(
            df, ["_tt_key"], ["trainer_track_runs", "trainer_track_win_rate"]
        )
        df["trainer_track_win_rate"] = _shrink_to_entity(
            df, "trainer_track_win_rate", "trainer_track_runs", "trainer"
        )
        df = _without(df, ["trainer_track_wins", "_tt_key"])

    return df


def add_jockey_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate jockey performance at specific tracks."""
    if "track" not in df.columns:
        return df

    logger.info("Engineering jockey-at-track features...")
    df["_jt_key"] = df["jockey"].astype(str) + "||" + df["track"].astype(str)
    _jtkc = _cumulative(
        df["_jt_key"].to_numpy(), df["race_id"].to_numpy(),
        {"jockey_track_runs": df["_result_start"].to_numpy(), "jockey_track_wins": df["won"].to_numpy()},
    )
    df["jockey_track_runs"] = _jtkc["jockey_track_runs"]
    df["jockey_track_wins"] = _jtkc["jockey_track_wins"]
    df["jockey_track_win_rate"] = _rate(df["jockey_track_wins"], df["jockey_track_runs"])
    df = _entity_day_start_view(
        df, ["_jt_key"], ["jockey_track_runs", "jockey_track_win_rate"]
    )
    df["jockey_track_win_rate"] = _shrink_to_entity(
        df, "jockey_track_win_rate", "jockey_track_runs", "jockey"
    )
    df = _without(df, ["jockey_track_wins", "_jt_key"])
    return df


def add_hidden_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag horses whose pre-race record shows prior runs NOT in our database.

    The classic case: a foreign raider (a French horse at Cheltenham) or a
    point-to-point graduate first appears with ``horse_prev_races == 0`` — so
    every form / rating feature is null and the model treats it as a blank-slate
    debutant and under-rates it (confirmed in the 2026-06 error analysis).

    **Leakage:** only PRE-RACE source fields are used. ``form`` is the record
    coming INTO the race — verified empirically that today's result is NOT in it
    (its most-recent character matches today's finishing position only at chance,
    ~12%). ``days_since_last_run`` is the raw source gap before today (the
    engineered ``days_since_last_race`` is DB-derived and defaults to 60 for
    DB-debutants, so it is unsuitable here). The raw lifetime ``horse_runs`` is
    deliberately NOT used — it is a *retroactive* career total (constant across a
    horse's rows; see line ~1087 where it is overwritten) and would leak the
    future. ``form`` parsing counts position characters (digits + P/F/U/… codes),
    requiring >=2 to clear the single-char placeholder that true debutants carry.
    """
    if "form" not in df.columns and "days_since_last_run" not in df.columns:
        return df
    logger.info("Engineering hidden-form (external-experience) features...")

    if "form" in df.columns:
        prior_form_runs = (df["form"].fillna("").astype(str)
                           .str.replace(r"[^0-9A-Za-z]", "", regex=True).str.len())
    else:
        prior_form_runs = pd.Series(0, index=df.index)
    df["prior_form_runs"] = prior_form_runs.astype(np.int16)

    src_days = (pd.to_numeric(df["days_since_last_run"], errors="coerce").fillna(0.0)
                if "days_since_last_run" in df.columns else pd.Series(0.0, index=df.index))
    db_prior = pd.to_numeric(df.get("horse_prev_races", 0), errors="coerce").fillna(0)

    # "Ran before per the source": a real gap since last run, or >=2 form chars
    # (>=2 avoids the single-char placeholder that true debutants carry).
    ran_before = (src_days > 0) | (df["prior_form_runs"] >= 2)
    # Appears as a DB debutant but has external history -> form we don't hold.
    df["has_hidden_form"] = ((db_prior == 0) & ran_before).astype(np.int8)
    # Recent runs the form book shows beyond our DB record (also catches
    # lightly-raced-but-experienced horses); capped by the ~6-run form window.
    excess = np.clip(df["prior_form_runs"].to_numpy() - db_prior.to_numpy(), 0, None)
    df["hidden_form_runs"] = np.where(ran_before.to_numpy(), excess, 0).astype(np.int16)
    # Experience corrected for hidden form, so raiders aren't treated as debutants.
    df["experience_with_hidden"] = (
        db_prior.to_numpy() + df["hidden_form_runs"].to_numpy()
    ).astype(np.int32)
    if "form" in df.columns:
        df = _add_hidden_form_figures(df)
    return df


_HIDDEN_FORM_FAIL = frozenset("PFURBCSDO")
_HIDDEN_FORM_FIGURES = ("wins", "places", "fails", "best", "mean", "last")


def hidden_form_figures(form, hidden_runs) -> np.ndarray:
    """Wins, places, failures to finish, best, mean and latest figure over the oldest
    ``hidden_runs`` characters of each form string; NaN where there are none.

    The form string runs oldest to newest and the database's own runs are the
    most recent, so the oldest ``hidden_form_runs`` figures are the ones the
    database does not account for. A "0" is tenth or worse; a letter is a
    failure to finish and counts as 12 for the latest figure only.
    """
    out = np.full((len(form), len(_HIDDEN_FORM_FIGURES)), np.nan)
    for i, (text, count) in enumerate(zip(form, hidden_runs)):
        if not count or count <= 0 or not isinstance(text, str):
            continue
        marks = [c for c in text.upper() if c.isascii() and c.isalnum()][:int(count)]
        if not marks:
            continue
        places = [10 if c == "0" else int(c) for c in marks if c.isdigit()]
        out[i, 0] = sum(p == 1 for p in places)
        out[i, 1] = sum(p <= 3 for p in places)
        out[i, 2] = sum(c in _HIDDEN_FORM_FAIL for c in marks)
        if places:
            out[i, 3], out[i, 4] = min(places), float(np.mean(places))
        last = marks[-1]
        out[i, 5] = (10 if last == "0" else int(last)) if last.isdigit() else 12.0
    return out


def _add_hidden_form_figures(df: pd.DataFrame) -> pd.DataFrame:
    """How the horse finished in the runs the database lacks, read from the card's form string.

    ``hidden_form_runs`` only counts those runs. Runners whose missing runs
    include a win won 499 races against 443 forecast on 2025Q3-2026Q2
    (docs/research/ERROR_ANATOMY_2026-11-10.md, section 9). Pre-race by the
    same argument as the count: only the card's ``form`` is read.
    """
    values = hidden_form_figures(df["form"].to_numpy(dtype=object), df["hidden_form_runs"].to_numpy())
    for j, name in enumerate(_HIDDEN_FORM_FIGURES):
        df[f"hidden_form_{name}"] = values[:, j].astype(np.float32)
    if "race_id" in df.columns:
        # Against the field, with a runner that has no such runs counted as no wins and a middling best.
        for name, neutral in (("hidden_form_wins", 0.0), ("hidden_form_best", 6.0)):
            filled = df[name].fillna(neutral)
            df[f"{name}_vs_field"] = (filled - filled.groupby(df["race_id"]).transform("mean")).astype(np.float32)
    return df


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate track-specific features for each horse."""
    logger.info("Engineering track features...")
    hkey = _horse_key(df)

    # Horse's track / distance / going records — class C, seed-aware so the
    # FeatureStore append carries state (see docs/architecture/FE_TAXONOMY.md). One row per
    # (horse, dim, race), so race-safe == plain cumsum-minus-current.
    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _rs = result_start_hist.to_numpy()
    _wh = won_hist.to_numpy()

    _trk = _cumulative(_composite_key(df, hkey, "track"), _race,
        {"horse_track_runs": _rs, "horse_track_wins": _wh})
    df["horse_track_runs"] = _trk["horse_track_runs"]
    df["horse_track_wins"] = _trk["horse_track_wins"]
    df["horse_track_win_rate"] = _rate(df["horse_track_wins"], df["horse_track_runs"])
    df["horse_track_wr_shrunk"] = _bayesian_shrink(
        df["horse_track_win_rate"], df["horse_track_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Distances carry yards (7f, 7.06f, 7.16f), so group on the nearest half
    # furlong, as the course-and-distance record does.
    df["_dist_half_f"] = (df["distance_furlongs"] * 2).round() / 2
    _dst = _cumulative(_composite_key(df, hkey, "_dist_half_f"), _race,
        {"horse_dist_runs": _rs, "horse_dist_wins": _wh})
    df["horse_dist_runs"] = _dst["horse_dist_runs"]
    df["horse_dist_wins"] = _dst["horse_dist_wins"]
    df["horse_dist_win_rate"] = _rate(df["horse_dist_wins"], df["horse_dist_runs"])
    df["horse_dist_wr_shrunk"] = _bayesian_shrink(
        df["horse_dist_win_rate"], df["horse_dist_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    _gng = _cumulative(_composite_key(df, hkey, "going"), _race,
        {"horse_going_runs": _rs, "horse_going_wins": _wh})
    df["horse_going_runs"] = _gng["horse_going_runs"]
    df["horse_going_wins"] = _gng["horse_going_wins"]
    df["horse_going_win_rate"] = _rate(df["horse_going_wins"], df["horse_going_runs"])
    df["horse_going_wr_shrunk"] = _bayesian_shrink(
        df["horse_going_win_rate"], df["horse_going_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    df = _without(df, [
            "horse_track_wins",
            "horse_dist_wins",
            "horse_going_wins",
            "_dist_half_f",
        ])
    return df


def add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate jockey–trainer combination statistics.

    Certain jockey-trainer partnerships outperform their individual stats.
    This captures the synergy (or lack of) between the two.
    """
    logger.info("Engineering jockey-trainer combo features...")
    df["_jt_combo"] = df["jockey"].astype(str) + " | " + df["trainer"].astype(str)
    placed_hist = df["_placed_for_history"].fillna(0.0)

    # Race-safe: a jockey-trainer combo can (rarely) have multiple
    # runners in the same race (different jockeys for same trainer,
    # but the *combo* key makes this less frequent).  Stay consistent.
    # Seed-aware cumulative primitive (class C; see docs/architecture/FE_TAXONOMY.md).
    _jtc = _cumulative(
        df["_jt_combo"].to_numpy(), df["race_id"].to_numpy(),
        {
            "jt_prev_runs": df["_result_start"].to_numpy(),
            "jt_cum_wins": df["won"].to_numpy(),
            "jt_cum_places": placed_hist.to_numpy(),
        },
    )
    df["jt_prev_runs"] = _jtc["jt_prev_runs"]
    df["jt_cum_wins"] = _jtc["jt_cum_wins"]
    df["jt_win_rate"] = _rate(df["jt_cum_wins"], df["jt_prev_runs"])
    df["jt_win_rate_shrunk"] = _bayesian_shrink(
        df["jt_win_rate"], df["jt_prev_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )

    # Combo place rate (top 3)
    df["jt_cum_places"] = _jtc["jt_cum_places"]
    df["jt_place_rate"] = _rate(df["jt_cum_places"], df["jt_prev_runs"])
    df["jt_place_rate_shrunk"] = _bayesian_shrink(
        df["jt_place_rate"], df["jt_prev_runs"],
        prior_rate=0.30, prior_strength=5.0,
    )

    df = _entity_day_start_view(df, ["_jt_combo"], [
        "jt_prev_runs", "jt_win_rate", "jt_win_rate_shrunk",
        "jt_place_rate", "jt_place_rate_shrunk",
    ])
    df = _without(df, ["_jt_combo", "jt_cum_wins", "jt_cum_places"])

    # --- Is‑stable‑jockey booking signal ---
    # Trainers have a preferred jockey — the one they book most often.
    # When that jockey rides again it is a "stable jockey" booking, which
    # correlates with higher trainer intent (they chose their best pilot).
    # We derive this from prior ride history only (no look-ahead leakage).
    if "trainer" in df.columns and "jockey" in df.columns:
        # Count prior rides per (trainer, jockey) pair — race-safe so that
        # a trainer's multiple runners in the same race share identical priors.
        df["_tjkey"] = df["trainer"].astype(str) + "||" + df["jockey"].astype(str)
        df["_tj_prior_rides"] = _cumulative(
            df["_tjkey"].to_numpy(), df["race_id"].to_numpy(),
            {"_tj_prior_rides": df["_result_start"].to_numpy()},
        )["_tj_prior_rides"]

        # For each race row, find the jockey with the most prior rides for
        # this trainer.  We do this by: (a) for every row record the
        # (trainer, jockey, prior_rides) triple, then (b) for each trainer
        # take the rolling argmax prior to the current race.
        # Building a per-trainer "most‑used jockey so far" requires a
        # careful implementation to avoid leakage.  We achieve it by:
        #   1. groupby trainer, sort by race index
        #   2. expanding cummax of _tj_prior_rides within trainer
        #   3. is_stable_jockey = 1 when own prior rides == trainer's cummax
        trainer_max_tj = _running_extreme(df, "trainer", "_tj_prior_rides", kind="max", prev=False)
        df["is_stable_jockey"] = (
            (df["_tj_prior_rides"] == trainer_max_tj) & (df["_tj_prior_rides"] > 0)
        ).astype(np.int8)

        df = _without(df, ["_tjkey", "_tj_prior_rides"])

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
    hkey = _horse_key(df)

    # Extract numeric class value  ("Class 3" → 3, "3" → 3)
    if "race_class" in df.columns:
        from src.feature_semantics import class_number
        df["_class_num"] = class_number(df["race_class"])
    else:
        df["_class_num"] = np.nan

    horse_groups = df.groupby(hkey)
    df["_prev_class"] = _lag(df, hkey, "_class_num")

    # Positive = dropping (easier), Negative = rising (harder)
    df["class_change"] = df["_class_num"] - df["_prev_class"]
    df["class_dropped"] = (df["class_change"] > 0).astype(int)
    df["class_raised"] = (df["class_change"] < 0).astype(int)

    # Official-rating change from previous run
    if "official_rating" in df.columns:
        from src.feature_semantics import observed_rating
        df["_observed_mark"] = observed_rating(df)
        df["or_change"] = df["_observed_mark"] - _lag(df, hkey, "_observed_mark")
        # Flat, hurdle and chase marks belong to separate rating scales.
        # Some Flat runners also have distinct turf and all-weather marks.
        # Without both published marks, a surface switch is not a proven rise.
        if "race_type" in df:
            from src.established_form_features import mark_scale
            df["_observed_mark_scale"] = mark_scale(df)
            previous_scale = _lag(df, hkey, "_observed_mark_scale")
            df["or_change"] = df["or_change"].where(df["_observed_mark_scale"].eq(previous_scale))
    else:
        df["or_change"] = np.nan

    df = _without(df, ["_class_num", "_prev_class", "_observed_mark", "_observed_mark_scale"])
    return df


def add_course_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Course & Distance (C&D) winner flag and stats.

    A horse that has previously won at the same track AND distance is a
    proven C&D winner — one of the most respected angles in racing.
    """
    logger.info("Engineering course & distance features...")
    hkey = _horse_key(df)

    # Round distance to nearest 0.5 f for grouping (minor variations)
    df["_dist_rounded"] = (df["distance_furlongs"] * 2).round() / 2

    # --- C&D (same track + same distance) ---
    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _cd = _cumulative(
        _composite_key(df, hkey, "track", "_dist_rounded"), _race,
        {"horse_cd_runs": result_start_hist.to_numpy(), "horse_cd_wins": won_hist.to_numpy()},
    )
    df["horse_cd_runs"] = _cd["horse_cd_runs"]
    df["horse_cd_wins"] = _cd["horse_cd_wins"]
    df["horse_cd_winner"] = (df["horse_cd_wins"] > 0).astype(int)
    df["horse_cd_win_rate"] = _rate(df["horse_cd_wins"], df["horse_cd_runs"])

    # --- Course only (already partially in add_track_features, but
    #     add a "has won at course before" binary for clarity) ---
    df["horse_course_wins"] = _cumulative(
        _composite_key(df, hkey, "track"), _race,
        {"horse_course_wins": won_hist.to_numpy()},
    )["horse_course_wins"]
    df["horse_course_winner"] = (df["horse_course_wins"] > 0).astype(int)

    df = _without(df, ["_dist_rounded", "horse_cd_wins", "horse_course_wins"])
    return df


def add_distance_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features related to changes in trip distance between races.

    Stepping up or down in distance is one of the most important angles
    in racing — some horses improve dramatically with a trip change.
    """
    logger.info("Engineering distance change features...")
    hkey = _horse_key(df)

    # --- Distance change from previous race ---
    horse_groups = df.groupby(hkey)
    trip_history, positions = _with_foreign_starts(df, "trip", ["distance_furlongs"])
    previous_trip = _lag(trip_history, hkey, "distance_furlongs")
    df["_prev_dist"] = previous_trip if positions is None else previous_trip.loc[positions].to_numpy()
    # Missing for a debutant: 0 would read as "same trip as last time".
    df["distance_change"] = df["distance_furlongs"] - df["_prev_dist"]
    df["stepping_up"] = (df["distance_change"] > 0.5).astype(int)
    df["stepping_down"] = (df["distance_change"] < -0.5).astype(int)

    # --- Distance category ---
    df["dist_category"] = distance_category(df["distance_furlongs"])

    # --- Horse win rate at this distance category ---
    df["_dist_cat_str"] = df["dist_category"].astype(str)
    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _dc = _cumulative(
        _composite_key(df, hkey, "_dist_cat_str"), _race,
        {"horse_dist_cat_runs": result_start_hist.to_numpy(), "_dc_wins": won_hist.to_numpy()},
    )
    df["horse_dist_cat_runs"] = _dc["horse_dist_cat_runs"]
    df["_dc_wins"] = _dc["_dc_wins"]
    df["horse_dist_cat_wr"] = _rate(df["_dc_wins"], df["horse_dist_cat_runs"])
    df["horse_dist_cat_wr_shrunk"] = _bayesian_shrink(
        df["horse_dist_cat_wr"], df["horse_dist_cat_runs"],
        prior_rate=0.10, prior_strength=5.0,
    )
    if "horse_win_rate_shrunk" in df.columns:
        df["horse_dist_cat_specialist_edge"] = (
            df["horse_dist_cat_wr_shrunk"] - df["horse_win_rate_shrunk"]
        )

    # Surface × distance-category record captures horses that are, for example,
    # good AW sprinters but mediocre on turf or over longer trips.
    if "surface" in df.columns:
        _sdc = _cumulative(
            _composite_key(df, hkey, "surface", "_dist_cat_str"), _race,
            {"horse_surface_dist_cat_runs": result_start_hist.to_numpy(), "_sdc_wins": won_hist.to_numpy()},
        )
        df["horse_surface_dist_cat_runs"] = _sdc["horse_surface_dist_cat_runs"]
        df["_sdc_wins"] = _sdc["_sdc_wins"]
        df["horse_surface_dist_cat_wr"] = _rate(df["_sdc_wins"], df["horse_surface_dist_cat_runs"])
        df["horse_surface_dist_cat_wr_shrunk"] = _bayesian_shrink(
            df["horse_surface_dist_cat_wr"], df["horse_surface_dist_cat_runs"],
            prior_rate=0.10, prior_strength=8.0,
        )
        if "horse_win_rate_shrunk" in df.columns:
            df["horse_surface_dist_cat_edge"] = (
                df["horse_surface_dist_cat_wr_shrunk"] - df["horse_win_rate_shrunk"]
            )

    # --- At preferred distance? (best win rate distance category) ---
    # For each horse, find the distance category where they've won most.
    # Use cummax() (not transform("max")) to avoid look-ahead: at each
    # point in time we only know the horse's wins up to NOW, not future.
    # Identical formula to _dc_wins above — reuse to avoid a second sweep and a
    # duplicate state key.
    df["_cum_dist_cat_wins"] = df["_dc_wins"]
    # Running max of cumulative wins within each (horse, dist_cat) group
    df["_hdc_key"] = _composite_key(df, hkey, "_dist_cat_str")
    df["_dc_running_max"] = _running_extreme(df, "_hdc_key", "_cum_dist_cat_wins", kind="max", prev=False)
    # Running max across ALL distance categories for each horse
    df["_best_dc"] = _running_extreme(df, hkey, "_dc_running_max", kind="max", prev=False)
    df = _without(df, ["_hdc_key"])
    df["at_preferred_dist"] = (
        (df["_cum_dist_cat_wins"] == df["_best_dc"]) & (df["_best_dc"] > 0)
    ).astype(int)

    df = _without(df, ["_prev_dist", "_dist_cat_str", "_dc_wins",
                 "_cum_dist_cat_wins", "_dc_running_max", "_best_dc", "_sdc_wins"])
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
    hg = df["headgear"].fillna("").astype(str).str.strip().str.lower()

    # Binary: wearing any headgear?
    df["has_headgear"] = (
        hg.ne("") & hg.ne("none") & hg.ne("0") & hg.ne("nan")
    ).astype(int)

    # First-time headgear detection
    hkey = _horse_key(df)
    horse_groups = df.groupby(hkey)
    df["_prev_hg"] = _lag(df, hkey, "has_headgear").fillna(0)
    df["_ever_had_hg"] = _cumulative(
        df[hkey].to_numpy(), df["race_id"].to_numpy(),
        {"_ever_had_hg": df["has_headgear"].to_numpy()},
    )["_ever_had_hg"]
    # First-time = wearing headgear now AND never worn it in any prior race
    df["first_time_headgear"] = (
        (df["has_headgear"] == 1) & (df["_ever_had_hg"] == 0)
    ).astype(int)

    # Headgear removed (wore it last time but not now)
    df["headgear_off"] = (
        (df["has_headgear"] == 0) & (df["_prev_hg"] == 1)
    ).astype(int)

    df = _without(df, ["_prev_hg", "_ever_had_hg"])
    return df


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horse performance on the current surface (Turf vs AW/Polytrack)."""
    if "surface" not in df.columns:
        return df

    logger.info("Engineering surface features...")
    hkey = _horse_key(df)

    result_start_hist = df["_result_start"].fillna(0.0)
    won_hist = df["_won_for_history"].fillna(0.0)
    _race = df["race_id"].to_numpy()
    _sf = _cumulative(
        _composite_key(df, hkey, "surface"), _race,
        {"horse_surface_runs": result_start_hist.to_numpy(), "horse_surface_wins": won_hist.to_numpy()},
    )
    df["horse_surface_runs"] = _sf["horse_surface_runs"]
    df["horse_surface_wins"] = _sf["horse_surface_wins"]
    df["horse_surface_win_rate"] = _rate(df["horse_surface_wins"], df["horse_surface_runs"])
    df["horse_surface_wr_shrunk"] = _bayesian_shrink(
        df["horse_surface_win_rate"], df["horse_surface_runs"],
        prior_rate=0.10, prior_strength=6.0,
    )
    if "horse_win_rate_shrunk" in df.columns:
        df["horse_surface_specialist_edge"] = (
            df["horse_surface_wr_shrunk"] - df["horse_win_rate_shrunk"]
        )

    # First time on this surface type (e.g. turf horse switching to AW)
    df["first_time_surface"] = (df["horse_surface_runs"] == 0).astype(np.int8)

    # Surface switch from previous run — captures adaptation cost
    horse_groups = df.groupby(hkey)
    df["_prev_surface"] = _lag(df, hkey, "surface")
    df["surface_switch"] = (
        df["_prev_surface"].notna() & (df["surface"] != df["_prev_surface"])
    ).astype(np.int8)

    # Combined horse context: surface × race type
    if "race_type" in df.columns:
        _srt = _cumulative(
            _composite_key(df, hkey, "surface", "race_type"), _race,
            {"horse_surface_rt_runs": result_start_hist.to_numpy(), "_surface_rt_wins": won_hist.to_numpy()},
        )
        df["horse_surface_rt_runs"] = _srt["horse_surface_rt_runs"]
        df["_surface_rt_wins"] = _srt["_surface_rt_wins"]
        df["horse_surface_rt_wr"] = _rate(df["_surface_rt_wins"], df["horse_surface_rt_runs"])
        df["horse_surface_rt_wr_shrunk"] = _bayesian_shrink(
            df["horse_surface_rt_wr"], df["horse_surface_rt_runs"],
            prior_rate=0.10, prior_strength=8.0,
        )
        if "horse_win_rate_shrunk" in df.columns:
            df["horse_surface_rt_edge"] = (
                df["horse_surface_rt_wr_shrunk"] - df["horse_win_rate_shrunk"]
            )
    df = _without(df, ["_prev_surface", "horse_surface_wins", "_surface_rt_wins"])
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
    hkey = _horse_key(df)
    horse_groups = df.groupby(hkey)

    # --- Jockey switch (different jockey from last run) ---
    df["_prev_jockey"] = _lag(df, hkey, "jockey")
    df["jockey_switch"] = (
        df["_prev_jockey"].notna() & (df["jockey"] != df["_prev_jockey"])
    ).astype(int)

    # --- Jockey upgrade (current jockey Elo > previous jockey Elo) ---
    if "jockey_elo" in df.columns:
        df["_prev_jockey_elo"] = _lag(df, hkey, "jockey_elo")
        df["jockey_elo_change"] = (
            (df["jockey_elo"] - df["_prev_jockey_elo"]).fillna(0)
        )
        df["jockey_upgrade"] = (
            (df["jockey_switch"] == 1) & (df["jockey_elo_change"] > 50)
        ).astype(int)
        df = _without(df, ["_prev_jockey_elo"])

    # --- Beaten favourite last time (bounce-back angle) ---
    if "is_favourite" in df.columns:
        df["_prev_fav"] = _lag(df, hkey, "is_favourite").fillna(0)
        df["_prev_won"] = _lag(df, hkey, "won").fillna(0)
        df["beaten_fav_last"] = (
            (df["_prev_fav"] == 1) & (df["_prev_won"] == 0)
        ).astype(int)
        df = _without(df, ["_prev_fav", "_prev_won"])

    # ── Granular equipment signals ───────────────────────────────
    # The headgear column is stored as a JSON-like list of dicts:
    # [{"symbol": "b", "name": "Blinkers", "count": 1}, ...]
    # count=1 means first-time application — the strongest signal.
    if "headgear" in df.columns:
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

        unique_hg = pd.Index(hg_raw.unique())
        parsed_map = {val: _parse_hg(val) for val in unique_hg}
        parsed = hg_raw.map(parsed_map)
        symbols_series = parsed.map(lambda x: x[0])
        counts_series = parsed.map(lambda x: x[1])

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
        df["equip_count"] = symbols_series.map(len).astype(int)

    # --- Weight change direction ---
    if "weight_change" in df.columns:
        change = df["weight_change"]
        df["weight_dropped"] = change.lt(0).astype(float).where(change.notna())
        df["weight_raised"] = change.gt(0).astype(float).where(change.notna())

    df = _without(df, ["_prev_jockey"])
    return df


def add_trainer_specialisation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trainer win rates by distance category and going.

    Some trainers specialise heavily — e.g. sprint specialists,
    trainers who target soft ground, etc.
    """
    logger.info("Engineering trainer specialisation features...")
    # --- Trainer × distance category (race-safe) ---
    if "dist_category" in df.columns:
        df["_dc_str"] = df["dist_category"].astype(str)
        df["_tdc_key"] = df["trainer"].astype(str) + "||" + df["_dc_str"]
        _tdcc = _cumulative(
            df["_tdc_key"].to_numpy(), df["race_id"].to_numpy(),
            {"trainer_dist_cat_runs": df["_result_start"].to_numpy(), "_tdc_wins": df["won"].to_numpy()},
        )
        df["trainer_dist_cat_runs"] = _tdcc["trainer_dist_cat_runs"]
        df["_tdc_wins"] = _tdcc["_tdc_wins"]
        df["trainer_dist_cat_wr"] = _rate(df["_tdc_wins"], df["trainer_dist_cat_runs"])
        df = _entity_day_start_view(
            df, ["_tdc_key"], ["trainer_dist_cat_runs", "trainer_dist_cat_wr"]
        )
        df["trainer_dist_cat_wr"] = _shrink_to_entity(
            df, "trainer_dist_cat_wr", "trainer_dist_cat_runs", "trainer"
        )
        df = _without(df, ["_dc_str", "_tdc_wins", "_tdc_key"])

    # --- Trainer × going (race-safe) ---
    if "going" in df.columns:
        df["_tg_key"] = df["trainer"].astype(str) + "||" + df["going"].astype(str)
        _tgc = _cumulative(
            df["_tg_key"].to_numpy(), df["race_id"].to_numpy(),
            {"trainer_going_runs": df["_result_start"].to_numpy(), "_tg_wins": df["won"].to_numpy()},
        )
        df["trainer_going_runs"] = _tgc["trainer_going_runs"]
        df["_tg_wins"] = _tgc["_tg_wins"]
        df["trainer_going_wr"] = _rate(df["_tg_wins"], df["trainer_going_runs"])
        df = _entity_day_start_view(
            df, ["_tg_key"], ["trainer_going_runs", "trainer_going_wr"]
        )
        df["trainer_going_wr"] = _shrink_to_entity(
            df, "trainer_going_wr", "trainer_going_runs", "trainer"
        )
        df = _without(df, ["_tg_wins", "_tg_key"])

    return df


WINNER_MARGIN_CAP = 20.0


def add_speed_figure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive proxy speed figures from ``lengths_behind`` and distance.

    These legacy inputs are margin-based performance proxies, not measured
    speed. Winning-time data exists separately. Use cumulative distance
    behind the winner, never the raw gap to the preceding finisher. A winner
    scores minus its winning distance, so lower is still better.

    All rolling windows use ``shift(1)`` to avoid data leakage.
    """
    if "lengths_behind" not in df.columns:
        return df

    logger.info("Engineering speed figure features...")
    # Only use lengths_behind from horses that actually finished
    from src.result_margins import margin_values
    _lb_for_speed = margin_values(df).copy()
    if "finish_position" in df.columns:
        _position = df["finish_position"].fillna(0).astype(int)
        _lb_for_speed = _lb_for_speed.where(_position >= 1)
        # Credit winners the distance back to the next finisher (negative), capped at
        # WINNER_MARGIN_CAP lengths. Scoring every winner as 0 made a wide-margin winner equal
        # to a nose winner and left 36% of ``speed_fig_best_5`` at exactly zero. Only cumulative
        # margins qualify: the legacy ``lengths_behind`` is the gap to the horse in front.
        if "beaten_lengths" in df.columns:
            _next_best = _lb_for_speed.where(_position > 1).groupby(df["race_id"], sort=False).transform("min")
            _credit = -_next_best.clip(upper=WINNER_MARGIN_CAP)
            _lb_for_speed = _lb_for_speed.mask((_position == 1) & _credit.notna(), _credit)

    # Lengths normalized for distance with non-linear power-law scaling (2c)
    # This reflects that margin dispersion does not scale linearly with distance;
    # 1 length in a 5f sprint requires more energy/speed output difference than in a 24f chase.
    safe_dist = df["distance_furlongs"].replace(0, np.nan).fillna(8.0)
    df["_lb_per_furlong"] = _lb_for_speed.fillna(np.nan) / np.power(safe_dist, 0.6)

    # Per-race relative performance: horse vs race median (neg = better)
    race_median_lpf = df.groupby("race_id")["_lb_per_furlong"].transform("median")
    df["_rel_perf"] = df["_lb_per_furlong"] - race_median_lpf

    # --- Numba-accelerated rolling speed figures ---
    # Foreign starts consume a slot, with the domestic performance proxy
    # and class scales unavailable for those starts.
    _speed_ops = [
        ("speed_fig_avg_5", "mean", 5),
        ("speed_fig_avg_10", "mean", 10),
        ("speed_fig_best_5", "min", 5),
    ]
    for name, arr in _roll_horse_starts(df, "_lb_per_furlong", _speed_ops).items():
        df[name] = arr

    # Trend: recent avg minus long-term avg (negative = improving)
    df["speed_fig_trend"] = df["speed_fig_avg_5"] - df["speed_fig_avg_10"]

    # Relative performance rolling avg (negative = above average)
    _rel_ops = [("speed_fig_rel_avg_5", "mean", 5)]
    for name, arr in _roll_horse_starts(df, "_rel_perf", _rel_ops).items():
        df[name] = arr

    # ── Class-adjusted speed figures ─────────────────────────────
    if "race_class" in df.columns:
        from src.feature_semantics import class_number
        _class_num = class_number(df["race_class"])
        _quality = (1.0 - (_class_num - 1) * 0.1).clip(0.4, 1.0)
        # Additive class-based penalty instead of multiplicative adjustment (2a)
        # Multiplicative adjustment leaves all winners (lengths_behind=0.0) at 0.0,
        # which fails to distinguish a Class 1 winner from a Class 7 winner.
        # We apply an additive penalty proportional to the quality gap (max penalty 0.5 units).
        # A non-finisher or unavailable margin is not a zero-length defeat.
        # Preserve missingness so it cannot become the horse's best speed run.
        df["_adj_lpf"] = df["_lb_per_furlong"] + (1.0 - _quality) * 0.5

        _adj_ops = [
            ("speed_fig_adj_avg_5", "mean", 5),
            ("speed_fig_adj_best_5", "min", 5),
        ]
        for name, arr in _roll_horse_starts(df, "_adj_lpf", _adj_ops).items():
            df[name] = arr
        df = _without(df, ["_adj_lpf"])

        # ── Class-relative Z-score speed figures ─────────────────
        # shift(1) so each row's own lb_per_furlong is excluded from the class
        # median / std (avoids current-race outcome leakage). Seed-aware via
        # _expanding_prev (full per-class history — option A, exact).
        df["_class_key"] = _class_num.astype("Int64").astype("string").fillna("Unknown")
        _exp = _expanding_prev(df, "_class_key", "_lb_per_furlong", ["median", "std"])
        # Every performance on a date uses the same prior-day class scale.
        # A row-wise shift alone lets earlier finishers from this very race
        # enter the scale, so shuffling source result rows changes future
        # horse features. Gather by position (not first non-null value).
        _positions = pd.Series(np.arange(len(df)), index=df.index)
        _first = _positions.groupby(
            [df["_class_key"], pd.to_datetime(df["race_date"]).dt.normalize()], sort=False,
        ).transform("min").to_numpy(dtype=int)
        _class_median = pd.Series(_exp["median"].to_numpy()[_first], index=df.index)
        _class_std = pd.Series(_exp["std"].to_numpy()[_first], index=df.index).replace(0, np.nan).fillna(1.0)
        df["_speed_z_class"] = ((df["_lb_per_furlong"] - _class_median) / _class_std).where(_class_num.notna())
        df = _without(df, ["_class_key"])

        # Rolling Z-score averages (Numba-accelerated)
        _z_ops = [
            ("speed_z_class_avg_5", "mean", 5),
            ("speed_z_class_avg_10", "mean", 10),
            ("speed_z_class_best_5", "min", 5),
        ]
        for name, arr in _roll_horse_starts(df, "_speed_z_class", _z_ops).items():
            df[name] = arr
        df["speed_z_class_trend"] = (
            df["speed_z_class_avg_5"] - df["speed_z_class_avg_10"]
        )
        df = _without(df, ["_speed_z_class"])

    df = _without(df, ["_lb_per_furlong", "_rel_perf"])
    return df


def add_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive historical features from At The Races (TPD) sectionals.

    ATR replaced RacingTV as the sectional source (RTV retired 2026-10):
    broader coverage and a far richer per-run record (full velocity curve,
    energy distribution, stride, sectional-adjusted placing and the in-running
    comment).  See :mod:`src.atr_features` for the per-run derivations.

    **Leakage prevention**: the selected dataset's ATR rows are *post-race
    outcomes*.  :func:`src.atr_features.published_atr_rolling` turns each run
    into per-run scalars and tags, joined by saved race and horse IDs, and
    rolls them over the horse's PRIOR starts whose observations were
    available at the decision.  Every raw/per-run column is then dropped, so
    nothing from the current race reaches the model.

    Feature families: (1) in-running comment rates, (2) cleaned sectional
    upgrade, (3) early-pace / closing run-style (feeds ``add_pace_style``),
    (4) closing speed "in hand", (5) stride biomechanics + energy efficiency.
    """
    from src.atr_features import ALL_ATR_COLS, COMMENT_TAGS, RUN_METRICS, published_atr_rolling

    logger.info("Engineering ATR sectional features...")
    new = published_atr_rolling(df)
    race_ids = df.race_id
    for metric, _ in RUN_METRICS:
        column = f"{metric}_avg_3"
        if column in new:
            values = pd.Series(new[column], index=df.index)
            new[f"{column}_vs_field"] = (values - values.groupby(race_ids).transform("mean")).to_numpy()
    # These unused legacy composites depended on batch-wide statistics.
    # Field-relative scaling is causal and invariant to chunk boundaries.
    zscores = []
    for column in ("atr_max_speed_avg_3", "atr_finishing_speed_pct_avg_3"):
        values = pd.Series(new.get(column, np.nan), index=df.index)
        deviation = values.groupby(race_ids).transform("std")
        zscores.append((values-values.groupby(race_ids).transform("mean")) / deviation.where(deviation.gt(0)))
    new["atr_speed_score"] = ((zscores[0]+zscores[1])/2).to_numpy()
    from src.track_config import get_track_config
    circumference = pd.to_numeric(df.track.map(lambda track: get_track_config(track).get("circumference_f", np.nan)), errors="coerce")
    # Circumference in miles is a fixed physical scale. Normalising a
    # race's course against its own field would make every value missing.
    new["atr_stride_x_gallop"] = np.asarray(new.get("atr_peak_stride_length_avg_3", np.nan)) * (circumference/8.).to_numpy()
    new["has_atr_history"] = pd.notna(new.get("atr_finishing_speed_pct_avg_3", np.full(len(df), np.nan))).astype(np.int8)
    raw = set(ALL_ATR_COLS) | {name for name, _ in RUN_METRICS} | set(COMMENT_TAGS) | {"atr_closing_in_hand"}
    return _with_columns(_without(df, [c for c in raw if c in df]), pd.DataFrame(new, index=df.index))


def add_pace_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build pace-style proxies from historical ATR sectional metrics.

    Early-pace history (first-section speed vs field) drives the front-runner
    ↔ closer axis; finishing-speed % history is the balance/closing component.
    Built on the ATR sectional history (see :func:`add_atr_features`).
    """
    logger.info("Engineering pace-style features...")

    pace_components = [
        ("atr_early_pace_avg_3_vs_field", 1.00),    # faster early → front-runner
        ("atr_max_speed_avg_3_vs_field", 0.50),
    ]
    balance_components = [
        ("atr_finishing_speed_pct_avg_3_vs_field", 1.00),  # strong finish → closer
    ]
    trend_components = [
        ("atr_early_pace_trend", 1.00),
        ("atr_max_speed_trend", 0.50),
        ("atr_finishing_speed_pct_trend", 0.50),
    ]

    def _weighted_history_score(components: list[tuple[str, float]]) -> pd.Series:
        total = np.zeros(len(df), dtype=np.float64)
        weight_sum = np.zeros(len(df), dtype=np.float64)
        for col, weight in components:
            if col not in df.columns:
                continue
            raw = pd.to_numeric(df[col], errors="coerce")
            valid = raw.notna().to_numpy(dtype=np.float64)
            values = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
            total += values * weight
            weight_sum += valid * abs(weight)
        score = np.divide(
            total,
            weight_sum,
            out=np.full(len(df), np.nan, dtype=np.float64),
            where=weight_sum > 0,
        )
        return pd.Series(score, index=df.index)

    pace_base = _weighted_history_score(pace_components)
    pace_finish_bias = _weighted_history_score(balance_components)
    pace_trend = _weighted_history_score(trend_components)

    # Expose pace history coverage explicitly (2d)
    df["has_pace_style_history"] = pace_base.notna().astype(np.int8)

    df["pace_style_index"] = pace_base
    df["pace_finish_bias"] = pace_finish_bias
    df["pace_style_balance"] = pace_base - pace_finish_bias
    df["pace_style_trend"] = pace_trend

    race_mean = df.groupby("race_id")["pace_style_index"].transform("mean")
    race_std = df.groupby("race_id")["pace_style_index"].transform("std").fillna(0.0)
    race_max = df.groupby("race_id")["pace_style_index"].transform("max")
    race_min = df.groupby("race_id")["pace_style_index"].transform("min")

    front_threshold = race_mean + 0.35 * race_std
    closer_threshold = race_mean - 0.35 * race_std

    df["pace_style_vs_field"] = df["pace_style_index"] - race_mean
    df["pace_finish_bias_vs_field"] = df["pace_finish_bias"] - df.groupby("race_id")["pace_finish_bias"].transform("mean")
    df["pace_front_runner_flag"] = (
        (df["pace_style_index"] >= front_threshold)
        & (df["pace_style_index"] > 0)
    ).astype(np.int8)
    df["pace_closer_flag"] = (
        (df["pace_style_index"] <= closer_threshold)
        & (df["pace_finish_bias"] > 0)
    ).astype(np.int8)

    field_pace_count = df.groupby("race_id")["pace_front_runner_flag"].transform("sum")
    field_closer_count = df.groupby("race_id")["pace_closer_flag"].transform("sum")
    field_size = df.get("num_runners", pd.Series(1.0, index=df.index)).replace(0, 1).fillna(1.0)

    df["field_pace_pressure"] = field_pace_count.astype(float)
    df["field_pace_pressure_share"] = field_pace_count / field_size
    df["field_closer_share"] = field_closer_count / field_size
    df["hot_pace_flag"] = (df["field_pace_pressure"] >= 3).astype(np.int8)
    df["lone_speed_flag"] = (
        (df["pace_front_runner_flag"] == 1)
        & (df["field_pace_pressure"] == 1)
    ).astype(np.int8)

    # Pace duel dynamics
    df["is_pace_duel"] = df["hot_pace_flag"]
    df["front_runner_vulnerability"] = df["pace_front_runner_flag"] * df["field_pace_pressure"]
    df["closer_setup_intensity"] = df["pace_closer_flag"] * df["field_pace_pressure"]

    df["pace_rank"] = df.groupby("race_id")["pace_style_index"].rank(
        ascending=False,
        method="min",
    )
    df["pace_rank_pct"] = np.where(
        field_size > 1,
        (field_size - df["pace_rank"]) / (field_size - 1),
        0.5,
    )
    df["pace_gap_to_leader"] = df["pace_style_index"] - race_max
    df["pace_gap_to_slowest"] = df["pace_style_index"] - race_min

    df["front_runner_pressure"] = (
        df["pace_style_index"].clip(lower=0.0)
        * df["field_pace_pressure_share"]
    )
    df["closer_pace_setup"] = (
        df["pace_finish_bias"].clip(lower=0.0)
        * df["field_pace_pressure_share"]
    )

    if "distance_furlongs" in df.columns:
        is_sprint = (df["distance_furlongs"].fillna(8.0) <= 7.0).astype(float)
        df["pace_style_x_sprint"] = df["pace_style_index"] * is_sprint
        df["is_pace_duel_x_distance"] = df["is_pace_duel"].astype(float) * df["distance_furlongs"].fillna(8.0)

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
    unique_tracks = pd.Index(df["track"].astype(str).unique())
    config_map = {track: get_track_config(track) for track in unique_tracks}

    df["track_direction"] = df["track"].map(
        lambda t: direction_code(config_map[str(t)]["direction"])
    )
    df["track_shape"] = df["track"].map(
        lambda t: shape_code(config_map[str(t)]["shape"])
    )
    df["track_is_aw"] = df["track"].map(lambda t: config_map[str(t)]["is_aw"])
    df["track_circumference"] = df["track"].map(
        lambda t: config_map[str(t)]["circumference_f"]
    )
    df["track_uphill_finish"] = df["track"].map(
        lambda t: config_map[str(t)]["uphill_finish"]
    )
    df["track_downhill"] = df["track"].map(
        lambda t: config_map[str(t)]["downhill_section"]
    )
    df["track_draw_bias"] = df["track"].map(
        lambda t: config_map[str(t)]["draw_bias"]
    )

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

        # New Draw × soft ground turf interaction: outside draws favored in heavy/soft turf
        _is_soft_going = df["going"].fillna("Good").isin(["Soft", "Heavy", "Yielding"]).astype(float)
        _is_turf = (~df["track_is_aw"].fillna(0).astype(bool)).astype(float)
        _is_soft_turf = _is_soft_going * _is_turf
        df["draw_pct_x_soft_turf"] = _dp * _is_soft_turf
        df["draw_bias_x_soft_turf"] = df["track_draw_bias"] * _is_soft_turf

    # --- Uphill finish × weight ---
    # Uphill finishes penalise heavier horses more
    if "weight_lbs" in df.columns:
        df["uphill_x_weight"] = df["track_uphill_finish"] * burden_weight(df)

    # --- Tight/sharp track × draw ---
    # On tight tracks, inside draws matter more
    if "draw_pct" in df.columns:
        df["tight_track_x_draw"] = (
            (df["track_shape_tight"] | df["track_shape_sharp"]).astype(float)
            * _dp
        )

    return df


# Static per-shape "quirkiness": how much an idiosyncratic layout rewards
# course-craft / experience. Galloping, fair tracks ≈ 0; tight/sharp ≈ 1.
# Codes match shape_code(): Galloping 0, Tight 1, Sharp 2, Undulating 3, Stiff 4.
_SHAPE_QUIRK = {0: 0.0, 1: 0.8, 2: 1.0, 3: 0.4, 4: 0.5}
# How much a layout favours front-runners (hard to make up ground): tight/sharp
# and short circuits favour leaders; long galloping tracks do not.
_SHAPE_LEADERS = {0: 0.1, 1: 0.7, 2: 0.9, 3: 0.4, 4: 0.3}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km (NaN-safe)."""
    R = 6371.0
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlat = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_course_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Course angles that go beyond the static venue config.

    All features are strictly race-safe (prior runs only, via the seed-aware
    cumulative / lag primitives). Themes:

    * **Course experience × quirkiness** — a track-debutant is penalised far
      more at an idiosyncratic course (tight Chester, switchback Goodwood) than
      at a fair galloping one. Plain ``horse_track_runs`` cannot express this;
      the interaction with a static quirk score can. Includes a pragmatic
      course-*specialisation* term (proven course form weighted by quirk) — a
      fully data-driven per-track specialisation score is leak-prone, so we let
      the model lean on existing race-safe course form where craft matters.
    * **Handedness** — some horses demonstrably prefer one turning direction;
      experience and success at *today's* hand, plus a flag for switching to an
      unfamiliar hand after a career of the other way.
    * **Pace bias** — front-runners are favoured on tight/short courses where
      it is hard to come from behind; interact run-style with a leader-bias.
    * **Travel** — distance shipped from the last run: mild fatigue but also a
      strong trainer-intent signal (you don't van a horse 300 km on a whim).
    * **Going-at-this-course** — ground preference is partly course-specific.

    Must run after ``add_track_config_features`` (needs track_direction/shape/
    draw_bias/circumference), ``add_track_features`` (horse_track_runs / wr) and
    ``add_pace_style_features`` (pace_style_index).
    """
    if "track" not in df.columns:
        return df
    logger.info("Engineering course-dynamics features...")
    hkey = _horse_key(df)
    _race = df["race_id"].to_numpy()
    _rs = df["_result_start"].fillna(0.0).to_numpy()
    _wh = df["_won_for_history"].fillna(0.0).to_numpy()

    # Career runs (prior only) — denominator for "share of career at this hand".
    career_runs = _cumulative(
        df[hkey].astype(str).to_numpy(), _race, {"_career_runs": _rs}
    )["_career_runs"]

    # ── Static course descriptors (0..1) ─────────────────────────
    shape = df.get("track_shape")
    draw_bias = pd.to_numeric(df.get("track_draw_bias"), errors="coerce").fillna(0.0) \
        if "track_draw_bias" in df.columns else pd.Series(0.0, index=df.index)
    circ = pd.to_numeric(df.get("track_circumference"), errors="coerce").fillna(10.0) \
        if "track_circumference" in df.columns else pd.Series(10.0, index=df.index)
    if shape is not None:
        quirk = shape.map(_SHAPE_QUIRK).fillna(0.0).astype(float)
        leaders = shape.map(_SHAPE_LEADERS).fillna(0.2).astype(float)
    else:
        quirk = pd.Series(0.0, index=df.index)
        leaders = pd.Series(0.2, index=df.index)
    short_circ = ((circ > 0) & (circ < 9.0)).astype(float)
    df["course_quirk_score"] = (quirk + 0.15 * draw_bias + 0.2 * short_circ).clip(0.0, 1.0)
    df["course_favours_leaders"] = (leaders + 0.2 * short_circ).clip(0.0, 1.0)

    # ── #1 Course experience × quirk ─────────────────────────────
    if "horse_track_runs" in df.columns:
        runs = df["horse_track_runs"].fillna(0.0)
        df["first_time_at_course"] = (runs == 0).astype(np.int8)
        df["course_inexperience"] = 1.0 / (1.0 + runs)
        df["course_debut_x_quirk"] = df["first_time_at_course"] * df["course_quirk_score"]
        df["course_inexp_x_quirk"] = df["course_inexperience"] * df["course_quirk_score"]
        # Field-relative course experience (race-safe: prior-run counts only).
        df["course_exp_vs_field"] = runs - df.groupby("race_id")["horse_track_runs"].transform("mean")
    if "horse_track_wr_shrunk" in df.columns:
        df["course_form_x_quirk"] = (
            df["horse_track_wr_shrunk"].fillna(0.0) * df["course_quirk_score"]
        )

    # ── #2 Handedness (left/right experience & success) ──────────
    if "track_direction" in df.columns:
        _hd = _cumulative(_composite_key(df, hkey, "track_direction"), _race,
            {"_curdir_runs": _rs, "_curdir_wins": _wh})
        cur_runs = _hd["_curdir_runs"]
        cur_wins = _hd["_curdir_wins"]
        df["horse_curdir_runs"] = cur_runs
        cur_wr = np.where(cur_runs > 0, cur_wins / np.maximum(cur_runs, 1e-9), 0.0)
        df["horse_curdir_wr_shrunk"] = _bayesian_shrink(
            pd.Series(cur_wr, index=df.index), pd.Series(cur_runs, index=df.index),
            prior_rate=0.10, prior_strength=5.0)
        # Experienced horse meeting an unfamiliar turning direction today.
        df["first_time_this_hand"] = ((cur_runs == 0) & (career_runs > 0)).astype(np.int8)
        df["hand_experience_ratio"] = np.where(
            career_runs > 0, cur_runs / np.maximum(career_runs, 1e-9), 0.0)

    # ── #5 Pace / run-style bias ─────────────────────────────────
    if "pace_style_index" in df.columns:
        df["pace_x_course_leaders"] = (
            df["pace_style_index"].fillna(0.0) * df["course_favours_leaders"])
    if "pace_front_runner_flag" in df.columns:
        df["frontrunner_at_bias_track"] = (
            df["pace_front_runner_flag"].fillna(0).astype(float) * df["course_favours_leaders"])

    # ── #6 Travel shipped from the last run ──────────────────────
    cfg = {t: get_track_config(str(t)) for t in df["track"].astype(str).unique()}
    df["track_lat"] = df["track"].astype(str).map(lambda t: cfg[t]["lat"])
    df["track_lon"] = df["track"].astype(str).map(lambda t: cfg[t]["lon"])
    prev_lat = _lag(df, hkey, "track_lat")
    prev_lon = _lag(df, hkey, "track_lon")
    df["travel_from_last_km"] = _haversine_km(
        prev_lat.to_numpy(), prev_lon.to_numpy(),
        df["track_lat"].to_numpy(), df["track_lon"].to_numpy())
    df["is_long_haul"] = (df["travel_from_last_km"] > 200.0).astype(np.int8)
    df = _without(df, ["track_lat", "track_lon"])

    # ── #7 Going at this specific course ─────────────────────────
    if "going" in df.columns:
        _tg = _cumulative(_composite_key(df, hkey, "track", "going"), _race,
            {"_tg_runs": _rs, "_tg_wins": _wh})
        tg_runs = _tg["_tg_runs"]
        df["horse_track_going_runs"] = tg_runs
        tg_wr = np.where(tg_runs > 0, _tg["_tg_wins"] / np.maximum(tg_runs, 1e-9), 0.0)
        df["horse_track_going_wr_shrunk"] = _bayesian_shrink(
            pd.Series(tg_wr, index=df.index), pd.Series(tg_runs, index=df.index),
            prior_rate=0.10, prior_strength=5.0)

    return df


# add_weather_features and add_trainer_travel_features were removed on
# 2026-10-13. Both families measured dead (ledger 2g: every weather signal
# scores 0.0000 because weather is race-level; ledger 6: travel is noise), and
# both made the store depend on more than the database: the weather stage
# called the Open-Meteo API at build time for rows with no stored weather and
# silently filled failures with constants, and the travel stage read an
# unsigned CSV. A parity check on 106,467 rows showed the removal drops
# exactly their 10 columns and leaves the other 1,058 bit-identical.


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add explicit interaction features that capture well-known racing angles.

    Tree models can discover interactions via splits, but explicit
    features help them learn faster and more reliably.
    """
    logger.info("Engineering interaction features...")
    interaction_cols: dict[str, pd.Series | np.ndarray | int | float] = {}

    # 1) days_since_last_race × class_dropped
    #    Fresh horse dropping in class — a classic trainer angle.
    if "days_since_last_race" in df.columns and "class_dropped" in df.columns:
        interaction_cols["fresh_x_dropped"] = (
            df["days_log"] * df["class_dropped"]
        )

    # 2) horse_going_win_rate × going match indicator
    #    Proven ground preference on the right surface.
    if "horse_going_win_rate" in df.columns:
        interaction_cols["going_wr_x_runs"] = (
            df["horse_going_win_rate"]
            * np.log1p(df.get("horse_going_runs", 0))
        )

    # 3) jockey_elo × is_favourite
    #    Strong rider booking confirmed by market.
    if "jockey_elo" in df.columns and "is_favourite" in df.columns:
        # Centre jockey_elo around baseline so the interaction is meaningful
        interaction_cols["jockey_elo_x_fav"] = (
            (df["jockey_elo"] - 1500) * df["is_favourite"]
        )

    # 4) horse_elo × class_change
    #    High-rated horse dropping in class is a strong signal.
    if "horse_elo" in df.columns and "class_change" in df.columns:
        interaction_cols["elo_x_class_drop"] = (
            (df["horse_elo"] - 1500) * df["class_dropped"]
        )

    # 5) speed_fig × distance match
    #    Good speed figure is more predictive at the proven distance.
    # speed_fig_avg_5 is lengths behind (0 = always won, median ~2), so a
    # clip at 2 gave half the field the same speed; 1/(1+x) keeps the order.
    if "speed_fig_avg_5" in df.columns:
        _speed_quality = 1.0 / (1.0 + df["speed_fig_avg_5"])
    if "speed_fig_avg_5" in df.columns and "horse_dist_win_rate" in df.columns:
        interaction_cols["speed_x_dist_wr"] = _speed_quality * df["horse_dist_win_rate"]

    # 6) OR × handicap — official rating matters most in handicaps.
    # Observed marks only: an unrated runner's race-median stand-in is not a
    # mark, and averaging it in pulls the field mean toward the median.
    if "official_rating" in df.columns and "handicap" in df.columns:
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        race_avg_or = _mark.groupby(df["race_id"]).transform("mean")
        interaction_cols["or_vs_field_x_hcap"] = (
            ((_mark - race_avg_or) * df["handicap"]).where(df["handicap"].ne(0), 0.0)
        )

    # 7) age × distance — young horses prefer sprints, older handle distance
    if "age" in df.columns and "distance_furlongs" in df.columns:
        interaction_cols["age_x_distance"] = df["age"] * df["distance_furlongs"]

    # 8) age × going — younger horses struggle more on soft/heavy
    if "age" in df.columns and "going_Soft" in df.columns:
        soft_heavy = df.get("going_Soft", 0)
        if "going_Heavy" in df.columns:
            soft_heavy = soft_heavy + df["going_Heavy"]
        interaction_cols["age_x_soft_going"] = df["age"] * soft_heavy

    # 9) days_since_last_race² — captures U-shape (very fresh & stale = bad)
    if "days_since_last_race" in df.columns:
        interaction_cols["days_since_race_sq"] = df["days_since_last_race"] ** 2 / 1000.0

    # 10) Improving flag — both speed and form trending better
    #     form_slope > 0 means improving (see add_horse_features);
    #     speed_fig_trend < 0 means improving (lower = faster).
    if "speed_fig_trend" in df.columns and "form_slope" in df.columns:
        interaction_cols["improving_flag"] = (
            (df["speed_fig_trend"] < 0) & (df["form_slope"] > 0)
        ).astype(int)

    # 11) Trainer intent score — class drop + jockey upgrade + first-time headgear
    intent = pd.Series(0, index=df.index)
    if "class_dropped" in df.columns:
        intent = intent + df["class_dropped"]
    if "jockey_upgrade" in df.columns:
        intent = intent + df["jockey_upgrade"]
    if "first_time_headgear" in df.columns:
        intent = intent + df["first_time_headgear"]
    interaction_cols["trainer_intent_score"] = intent

    # 12) distance_change × class_change — trip + class shift combo
    if "distance_change" in df.columns and "class_change" in df.columns:
        interaction_cols["dist_x_class_change"] = df["distance_change"] * df["class_change"]

    # 13) jockey_upgrade × class_dropped — booking a star for a drop
    if "jockey_upgrade" in df.columns and "class_dropped" in df.columns:
        interaction_cols["jockey_up_x_dropped"] = df["jockey_upgrade"] * df["class_dropped"]

    # 14) close_finish_rate × speed_fig_avg_5 — competitive + fast
    if "close_finish_rate" in df.columns and "speed_fig_avg_5" in df.columns:
        interaction_cols["close_x_speed"] = df["close_finish_rate"] * _speed_quality

    # ── Market-residual interactions ─────────────────────────────
    # These help the model learn *when* the market misprices a horse
    # by pairing market signals with fundamental ability indicators.

    # 15) norm_implied_prob × horse_win_rate_shrunk — market vs proven ability
    if "norm_implied_prob" in df.columns and "horse_win_rate_shrunk" in df.columns:
        interaction_cols["mkt_x_win_rate"] = df["norm_implied_prob"] - df["horse_win_rate_shrunk"]

    # 16) log_odds × horse_elo — price level vs ability rating
    if "log_odds" in df.columns and "horse_elo" in df.columns:
        interaction_cols["logodds_x_elo"] = df["log_odds"] * (df["horse_elo"] - 1500) / 400

    # 17) odds_vs_field × class_dropped — overlooked class droppers
    if "odds_vs_field" in df.columns and "class_dropped" in df.columns:
        interaction_cols["odds_field_x_dropped"] = df["odds_vs_field"] * df["class_dropped"]

    # 18) norm_implied_prob × speed_fig_avg_5 — market vs recent speed
    if "norm_implied_prob" in df.columns and "speed_fig_avg_5" in df.columns:
        interaction_cols["mkt_x_speed"] = df["norm_implied_prob"] - _speed_quality

    # 19) odds_vs_field × jockey_elo — price vs rider quality
    if "odds_vs_field" in df.columns and "jockey_elo" in df.columns:
        interaction_cols["odds_field_x_jock_elo"] = df["odds_vs_field"] * (df["jockey_elo"] - 1500) / 400

    # 20) Context-adjusted Elo interactions — let the model learn when a high
    # overall Elo should be trusted only in the horse's preferred conditions.
    if "horse_elo" in df.columns:
        _elo_centered = (df["horse_elo"] - 1500) / 400
        if "horse_surface_specialist_edge" in df.columns:
            interaction_cols["elo_x_surface_edge"] = _elo_centered * df["horse_surface_specialist_edge"]
        if "horse_rt_specialist_edge" in df.columns:
            interaction_cols["elo_x_rt_edge"] = _elo_centered * df["horse_rt_specialist_edge"]
        if "horse_dist_cat_specialist_edge" in df.columns:
            interaction_cols["elo_x_dist_cat_edge"] = _elo_centered * df["horse_dist_cat_specialist_edge"]
        if "horse_surface_rt_edge" in df.columns:
            interaction_cols["elo_x_surface_rt_edge"] = _elo_centered * df["horse_surface_rt_edge"]
        if "horse_surface_dist_cat_edge" in df.columns:
            interaction_cols["elo_x_surface_dist_edge"] = _elo_centered * df["horse_surface_dist_cat_edge"]

    # ── Additional strategic interactions ─────────────────────────

    # 21) Elo × recent form — high-rated horse currently in good form
    if "horse_elo" in df.columns and "horse_avg_pos_3" in df.columns:
        _elo_c = (df["horse_elo"] - 1500) / 400
        # avg_pos_3 is a field share (0 = won, 1 = last) → invert so positive = good
        _form_quality = 1.0 - df["horse_avg_pos_3"]
        interaction_cols["elo_x_recent_form"] = _elo_c * _form_quality

    # 22) Jockey-trainer synergy — both connections in hot form
    if "jockey_wr_14d" in df.columns and "trainer_wr_14d" in df.columns:
        interaction_cols["jockey_trainer_synergy"] = df["jockey_wr_14d"] * df["trainer_wr_14d"]

    # 23) Going specialist edge — performs better on this going than usual.
    # Shrunk minus shrunk, like the other specialist edges: a raw going rate
    # gave a one-from-one horse +0.9.
    if "horse_going_wr_shrunk" in df.columns and "horse_win_rate_shrunk" in df.columns:
        interaction_cols["going_specialist_edge"] = (
            df["horse_going_wr_shrunk"] - df["horse_win_rate_shrunk"]
        )

    # 24) Form slope × post-break — freshened horses with improving trend
    if "form_slope" in df.columns and "days_since_last_race" in df.columns:
        _is_fresh = (df["days_since_last_race"] > 30).astype(np.float32)
        interaction_cols["form_slope_x_fresh"] = df["form_slope"] * _is_fresh

    # 25) Jockey hot streak × horse experience — hot jockey on experienced horse
    if "jockey_hot_vs_30d" in df.columns and "horse_prev_races" in df.columns:
        interaction_cols["jockey_hot_x_experience"] = (
            df["jockey_hot_vs_30d"] * np.log1p(df["horse_prev_races"])
        )

    # 26) Distance specialist × going specialist — double-specialist edge
    if "horse_dist_win_rate" in df.columns and "horse_going_win_rate" in df.columns:
        interaction_cols["dist_x_going_specialist"] = (
            df["horse_dist_win_rate"] * df["horse_going_win_rate"]
        )

    if interaction_cols:
        interaction_frame = pd.DataFrame(interaction_cols, index=df.index)
        overlapping_cols = [col for col in interaction_frame.columns if col in df.columns]
        if overlapping_cols:
            df = _without(df, overlapping_cols)
        df = _with_columns(df, interaction_frame)

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


def add_pedigree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sire, dam, and damsire historical stats (Bayesian shrunk)."""
    logger.info("Engineering pedigree features...")
    df = df.copy()

    # Pre-compute condition masks for specialty pedigree features
    is_soft = df["going"].fillna("Good").isin(["Soft", "Heavy", "Yielding"])
    from src.surface_features import synthetic_surface
    is_aw = synthetic_surface(df["surface"]).eq(1)
    dist_f = df["distance_furlongs"].fillna(8.0)
    is_sprint = dist_f <= 7.0
    is_stay_route = dist_f >= 11.5

    result_start_flag = df["_result_start"].fillna(0.0)
    won_flag = df["won"].fillna(0.0)
    placed_flag = df["_placed_for_history"].fillna(0.0)

    for col in ["sire", "dam", "damsire"]:
        if col not in df.columns:
            continue

        # Race-safe metrics tracking each sire's/dam's progeny performance
        # We must exclude the current race to prevent data leakage from stable-mates
        # competing in the same race.
        # Use existing race-safe cumulative functions from the module!
        _ped = _cumulative(
            df[col].to_numpy(), df["race_id"].to_numpy(),
            {
                # Use the fillna(0) flags (as the conditional path below does) —
                # the raw _placed_for_history is NaN on result-less rows (live
                # cards, non-finishers), which propagated NaN into _safe_*_places
                # and left {col}_place_rate_shrunk NaN for every live runner while
                # _wins (from `won`, set to 0 on cards) stayed fine.
                f"_safe_{col}_runs": result_start_flag.to_numpy(),
                f"_safe_{col}_wins": won_flag.to_numpy(),
                f"_safe_{col}_places": placed_flag.to_numpy(),
            },
        )
        df[f"_safe_{col}_runs"] = _ped[f"_safe_{col}_runs"]
        df[f"_safe_{col}_wins"] = _ped[f"_safe_{col}_wins"]
        df[f"_safe_{col}_places"] = _ped[f"_safe_{col}_places"]

        df[f"{col}_win_rate"] = _rate(df[f"_safe_{col}_wins"], df[f"_safe_{col}_runs"])
        # Bayesian shrink towards typical population prior
        df[f"{col}_win_rate_shrunk"] = _bayesian_shrink(
            df[f"{col}_win_rate"], df[f"_safe_{col}_runs"],
            prior_rate=0.10, prior_strength=20.0,
        )

        df[f"{col}_place_rate"] = _rate(df[f"_safe_{col}_places"], df[f"_safe_{col}_runs"])
        df[f"{col}_place_rate_shrunk"] = _bayesian_shrink(
            df[f"{col}_place_rate"], df[f"_safe_{col}_runs"],
            prior_rate=0.30, prior_strength=20.0,
        )

        # Cleanup basic stats temporary columns
        temp_cols = [f"_safe_{col}_runs", f"_safe_{col}_wins", f"_safe_{col}_places", 
                     f"{col}_win_rate", f"{col}_place_rate"]

        # Only compute detailed conditional preferences for sire and damsire to avoid noise/fragmentation
        if col != "dam":
            conds = {
                "soft": is_soft,
                "aw": is_aw,
                "sprint": is_sprint,
                "stay": is_stay_route
            }
            for cond_name, mask in conds.items():
                temp_runs = f"_safe_{col}_{cond_name}_runs"
                temp_wins = f"_safe_{col}_{cond_name}_wins"
                temp_places = f"_safe_{col}_{cond_name}_places"

                # Conditional (going/surface/trip) progeny counts, race-safe and
                # seed-aware via _cumulative; metric keys are unique per (col, cond).
                _pc = _cumulative(
                    df[col].to_numpy(), df["race_id"].to_numpy(),
                    {
                        temp_runs: result_start_flag.where(mask, 0.0).to_numpy(),
                        temp_wins: won_flag.where(mask, 0.0).to_numpy(),
                        temp_places: placed_flag.where(mask, 0.0).to_numpy(),
                    },
                )
                df[temp_runs] = _pc[temp_runs]
                df[temp_wins] = _pc[temp_wins]
                df[temp_places] = _pc[temp_places]

                wr_raw = np.where(df[temp_runs] > 0, df[temp_wins] / df[temp_runs], 0.0)
                pr_raw = np.where(df[temp_runs] > 0, df[temp_places] / df[temp_runs], 0.0)

                # Shrunk rates
                df[f"{col}_{cond_name}_win_rate_shrunk"] = _bayesian_shrink(
                    wr_raw, df[temp_runs],
                    prior_rate=0.10, prior_strength=10.0,
                )
                df[f"{col}_{cond_name}_place_rate_shrunk"] = _bayesian_shrink(
                    pr_raw, df[temp_runs],
                    prior_rate=0.30, prior_strength=10.0,
                )

                # Specialisation edge (relative to overall class rating)
                df[f"{col}_{cond_name}_edge"] = (
                    df[f"{col}_{cond_name}_win_rate_shrunk"] - df[f"{col}_win_rate_shrunk"]
                )

                temp_cols.extend([temp_runs, temp_wins, temp_places])
                df = _defrag(df, max_blocks=_MID_STAGE_BLOCKS)

            # Drop temporary conditional indicator columns
            df = _without(df, ["_temp_runs_cond", "_temp_wins_cond", "_temp_places_cond"])

        df = _without(df, temp_cols)
        # All cards for a day are predicted before any of that day's results
        # exist. Progeny in later races must share the day's opening history.
        pedigree_cols = [c for c in df.columns if c.startswith(f"{col}_")
                         and (c.endswith("_shrunk") or c.endswith("_edge"))]
        df = _entity_day_start_view(df, [col], pedigree_cols)

    return df


def add_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive medical procedure flags: first run since a wind operation.

    A first-time-gelded flag lived here until 2026-10-13. It compared the sex
    code against "gelding" while the data holds single letters, so it was
    always 0 - and fixing the comparison would have created a leak: the stored
    sex is the horse's sex on the day the page was scraped, not on race day
    (ledger 15), so a gelding transition cannot be dated from history.
    """
    logger.info("Engineering medical procedure features (wind ops)...")
    hkey = _horse_key(df)

    # ── Wind surgery (declared operation since last run) ─────────────
    # ``wind_surgery`` = 1 when this run is the FIRST since a declared wind
    # operation (BHA racecard declaration; NULL on rows the backfill hasn't
    # reached → treated as 0).
    if "wind_surgery" in df.columns:
        ws = (
            pd.to_numeric(df["wind_surgery"], errors="coerce")
            .fillna(0)
            .astype(np.int8)
        )
        df["wind_surgery_run"] = ws
        # Declared ops strictly BEFORE this race (race-safe cumulative;
        # repeat ops signal a chronic breathing problem).
        df["wind_ops_prior"] = _cumulative(
            df[hkey].to_numpy(), df["race_id"].to_numpy(),
            {"wind_ops_prior": ws.to_numpy(np.float64)},
        )["wind_ops_prior"]
        df["first_wind_surgery"] = (
            (ws == 1) & (df["wind_ops_prior"] <= 0)
        ).astype(np.int8)
        # Second run after an op — fitness often improves on the first
        df["prev_run_wind_surgery"] = (
            _lag(df, hkey, "wind_surgery_run").fillna(0).astype(np.int8)
        )
    return df


def add_prize_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate prize money features capturing class intent and competition level."""
    if "prize_money" not in df.columns:
        return df

    logger.info("Engineering prize money features...")
    hkey = _horse_key(df)

    # Fill NaN with 0 (before grouping so the groupby sees clean values)
    df["prize_money"] = df["prize_money"].fillna(0)
    horse_groups = df.groupby(hkey)

    # Maximum purse the horse has CONTESTED, excluding the current race;
    # missing for a debutant.
    # NB: the shift MUST be inside transform — a bare
    # groupby().cummax().shift(1) shifts globally across horses and, with
    # raw rows in finish order, leaks finish position (CLAUDE.md rule 1).
    df["horse_max_prize_contested"] = _running_extreme(
        df, hkey, "prize_money", kind="max", prev=True
    )
    
    # Cumsum / count to avoid expanding().mean() issues:
    df["_safe_cum_prize"] = _cumulative(
        df[hkey].to_numpy(), df["race_id"].to_numpy(),
        {"_safe_cum_prize": df["prize_money"].to_numpy()},
    )["_safe_cum_prize"]
    hp_races = df.get("horse_prev_races", pd.Series(0, index=df.index))
    df["horse_avg_prize_contested"] = _rate(df["_safe_cum_prize"], hp_races)

    # Change vs typical class on a log scale. Purses are heavily skewed
    # (median about £5k, 99th percentile about £94k), so a £ difference is
    # dominated by the few big races. An unknown purse (0) leaves it missing.
    current = df["prize_money"].where(df["prize_money"] > 0)
    typical = df["horse_avg_prize_contested"].where(df["horse_avg_prize_contested"] > 0)
    df["prize_money_change_vs_avg"] = np.log1p(current) - np.log1p(typical)
    
    # Flag for severe class jumps based on money
    df["stepping_up_in_purse"] = (
        (hp_races > 0) & 
        (df["prize_money"] > df["horse_max_prize_contested"] * 1.5)
    ).astype(np.int8)
    
    df = _without(df, ["_safe_cum_prize"])
    return df


def add_collateral_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collateral form: how did the rivals from this horse's previous race
    perform in their own NEXT start?

    A horse beaten in a race whose other runners went on to win next time
    out ran better than the bare result suggests ("key race"). Uses only
    rival next-out results dated strictly BEFORE the current race, so no
    future information leaks despite the per-horse forward shift.

    Columns:
    - ``collateral_n_rivals``        — rivals with a known next-out result
    - ``collateral_next_win_rate``   — their next-out win rate
    - ``collateral_next_place_rate`` — their next-out top-3 rate
    - ``collateral_next_win_rate_vs_field`` — relative to today's field
    """
    if not getattr(config, "COLLATERAL_FORM", True):
        return df
    if "finish_position" not in df.columns or "race_id" not in df.columns:
        return df
    logger.info("Engineering collateral-form features...")

    hkey = _horse_key(df)
    groups = df.groupby(hkey, sort=False)

    # Each runner row annotated with that horse's NEXT run outcome.
    # Forward-looking by construction — only consumed under a date guard.
    next_date = groups["race_date"].shift(-1)
    next_fp = pd.to_numeric(groups["finish_position"].shift(-1), errors="coerce")
    rivals = pd.DataFrame({
        "race_id": df["race_id"].values,
        "rival_key": df[hkey].values,
        "next_date": next_date.values,
        "next_won": (next_fp == 1).astype(float).values,
        "next_placed": next_fp.between(1, 3).astype(float).values,
    })

    prev_race_id = groups["race_id"].shift(1)
    cur = pd.DataFrame({
        "row": np.arange(len(df)),
        "prev_race_id": prev_race_id.values,
        "self_key": df[hkey].values,
        "cur_date": df["race_date"].values,
    }).dropna(subset=["prev_race_id"])

    merged = cur.merge(rivals, left_on="prev_race_id", right_on="race_id")
    merged = merged[
        (merged["rival_key"] != merged["self_key"])
        & merged["next_date"].notna()
        & (merged["next_date"] < merged["cur_date"])
    ]
    agg = merged.groupby("row").agg(
        collateral_n_rivals=("next_won", "size"),
        collateral_next_win_rate=("next_won", "mean"),
        collateral_next_place_rate=("next_placed", "mean"),
    )

    out = agg.reindex(np.arange(len(df)))
    n_rivals = out["collateral_n_rivals"].fillna(0).astype(np.float32)
    win_rate = out["collateral_next_win_rate"].astype(np.float32)
    place_rate = out["collateral_next_place_rate"].astype(np.float32)

    race_ids = df["race_id"].values
    wr_ser = pd.Series(win_rate.values, index=df.index)
    wr_field = wr_ser.groupby(race_ids).transform("mean")

    df = pd.concat([df, pd.DataFrame({
        "collateral_n_rivals": n_rivals.values,
        "collateral_next_win_rate": win_rate.values,
        "collateral_next_place_rate": place_rate.values,
        "collateral_next_win_rate_vs_field": (wr_ser - wr_field).to_numpy(),
    }, index=df.index)], axis=1)
    return df


def add_trainer_runner_intent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trainer multi-runner and first-string intent features."""
    logger.info("Engineering trainer multi-runner intent features...")
    df = df.copy()

    required = {"race_id", "trainer"}
    if not required.issubset(df.columns):
        return df

    trainer_runners = df.groupby(["race_id", "trainer"])["trainer"].transform("count")
    df["same_trainer_runners"] = trainer_runners
    df["trainer_has_multiple_runners"] = (trainer_runners > 1).astype(np.int8)

    if "official_rating" in df.columns:
        # Observed marks only (an unrated runner has no place in the OR order).
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        _stable = [df["race_id"], df["trainer"]]
        df["trainer_runner_rank_by_or"] = _mark.groupby(_stable).rank(
            ascending=False,
            method="min",
        )
        df["trainer_first_string_by_or"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_or"] == 1)
        ).astype(np.int8)
        stable_best_or = _mark.groupby(_stable).transform("max")
        df["trainer_or_vs_best_stablemate"] = _mark - stable_best_or

    if "horse_elo" in df.columns:
        df["trainer_runner_rank_by_elo"] = df.groupby(["race_id", "trainer"])["horse_elo"].rank(
            ascending=False,
            method="min",
        )
        df["trainer_first_string_by_elo"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_elo"] == 1)
        ).astype(np.int8)
        stable_best_elo = df.groupby(["race_id", "trainer"])["horse_elo"].transform("max")
        df["trainer_elo_vs_best_stablemate"] = df["horse_elo"] - stable_best_elo

    if "odds" in df.columns:
        stable_group = df.groupby(["race_id", "trainer"], sort=False)
        df["trainer_runner_rank_by_odds"] = stable_group["odds"].rank(
            ascending=True,
            method="min",
        )
        df["trainer_first_string_by_odds"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_odds"] == 1)
        ).astype(np.int8)

        stable_min_odds = stable_group["odds"].transform("min")
        df["trainer_odds_vs_stablemate_best"] = df["odds"] - stable_min_odds

        stable_implied = (1.0 / df["odds"].replace(0, np.nan)).fillna(0.0)
        stable_total_implied = stable_implied.groupby([df["race_id"], df["trainer"]]).transform("sum")
        df["trainer_stable_market_share"] = np.divide(
            stable_implied,
            stable_total_implied,
            out=np.zeros(len(df), dtype=np.float64),
            where=stable_total_implied > 0,
        )

        if "official_rating" in df.columns:
            df["trainer_first_string_odds_or_agree"] = (
                (df["trainer_first_string_by_odds"] == 1)
                & (df.get("trainer_first_string_by_or", 0) == 1)
            ).astype(np.int8)
            df["trainer_string_disagreement_or_odds"] = (
                df["trainer_runner_rank_by_odds"]
                - df["trainer_runner_rank_by_or"]
            ).abs()

        if "horse_elo" in df.columns:
            df["trainer_first_string_odds_elo_agree"] = (
                (df["trainer_first_string_by_odds"] == 1)
                & (df.get("trainer_first_string_by_elo", 0) == 1)
            ).astype(np.int8)
            df["trainer_string_disagreement_elo_odds"] = (
                df["trainer_runner_rank_by_odds"]
                - df["trainer_runner_rank_by_elo"]
            ).abs()

        if "is_stable_jockey" in df.columns:
            df["trainer_first_string_stable_jockey"] = (
                df["trainer_first_string_by_odds"] * df["is_stable_jockey"]
            ).astype(np.int8)

    # ── Jockey-booking strength ──────────────────────────────────────
    # Which stablemate got the better rider, and is the booked jockey
    # stronger than the horse's market position implies ("top jockey on
    # an outsider" — connections don't waste a first-choice booking).
    if "jockey_elo" in df.columns:
        stable_jgroup = df.groupby(["race_id", "trainer"], sort=False)
        df["trainer_runner_rank_by_jockey"] = stable_jgroup["jockey_elo"].rank(
            ascending=False,
            method="min",
        )
        df["trainer_first_string_by_jockey"] = (
            (df["trainer_has_multiple_runners"] == 1)
            & (df["trainer_runner_rank_by_jockey"] == 1)
        ).astype(np.int8)
        stable_best_jelo = stable_jgroup["jockey_elo"].transform("max")
        df["jockey_elo_vs_best_stablemate"] = df["jockey_elo"] - stable_best_jelo

        if "odds" in df.columns:
            # Percentile ranks within the race: 1.0 = best jockey / most
            # fancied horse. Positive gap = rider outranks the horse's
            # market slot.
            jockey_pct = df.groupby("race_id")["jockey_elo"].rank(
                ascending=True, pct=True, method="average"
            )
            implied = 1.0 / df["odds"].replace(0, np.nan)
            market_pct = implied.groupby(df["race_id"]).rank(
                ascending=True, pct=True, method="average"
            )
            df["jockey_booking_vs_market"] = (jockey_pct - market_pct).fillna(0.0)

    return df


def add_race_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features about the race context."""
    logger.info("Engineering race context features...")
    _rs = (df["_result_start"] if "_result_start" in df
           else _add_result_history_flags(df.copy())["_result_start"]).to_numpy()

    # Draw advantage as a share of the highest stall drawn in the race.
    # Jumps/bumpers have no starting stalls, so draw == 0 is "no draw", NOT the
    # innermost stall (real flat draws are >= 1). Mark these missing so the
    # model isn't told every jumps horse is best-drawn, and so the per-track
    # draw-third stats below are computed over stalls races only (NaN draw_pct
    # falls outside the low/mid/high bins). Dividing by ``num_runners`` (the
    # field after non-runners) put draws above 1 in 6.8% of drawn rows.
    _stall = pd.to_numeric(df["draw"], errors="coerce")
    _stall = _stall.where(_stall > 0)
    df["draw_pct"] = _stall / _stall.groupby(df["race_id"]).transform("max")

    # ── Track-specific draw bias ─────────────────────────────────
    # Per-track draw-position win-rate (point-in-time, no leakage).
    # Tracks like Chester, Beverley, Ascot have extreme draw biases.
    # No-draw races (NaN draw_pct) are excluded from the bins, so these
    # stats reflect flat-with-stalls form only and don't pollute mixed
    # flat/jumps tracks (Ascot, Newcastle, Doncaster, ...).
    if "track" in df.columns:

        # Create draw-position bins (low / mid / high third)
        df["_draw_third"] = pd.cut(
            df["draw_pct"].clip(0.01, 0.99),
            bins=[0, 0.33, 0.66, 1.0],
            labels=["low", "mid", "high"],
            include_lowest=True,
        ).astype(str)

        for draw_pos in ["low", "mid", "high"]:
            mask = (df["_draw_third"] == draw_pos).to_numpy()
            # Exclude every runner in the current race, including earlier
            # rows (results are commonly stored in finishing order).
            _td = _cumulative(
                df["track"].to_numpy()[mask],
                df["race_id"].to_numpy()[mask],
                {
                    f"_td_wins_{draw_pos}": df["won"].to_numpy()[mask],
                    f"_td_runs_{draw_pos}": _rs[mask],
                },
            )
            df.loc[mask, f"_td_wins_{draw_pos}"] = _td[f"_td_wins_{draw_pos}"]
            df.loc[mask, f"_td_runs_{draw_pos}"] = _td[f"_td_runs_{draw_pos}"]

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

        # Missing below 6 prior runs: a 0 would read as "this draw never wins".
        df["track_draw_win_rate"] = np.where(
            df["track_draw_wr_runs"] > 5,
            df["track_draw_wr_wins"] / df["track_draw_wr_runs"],
            np.nan,
        )

        # Match the pre-race card view: exclude earlier races on this day.
        df = _entity_day_start_view(
            df, ["track", "_draw_third"], ["track_draw_win_rate"]
        )

        # Clean up
        drop_cols = ["_draw_third", "track_draw_wr_wins", "track_draw_wr_runs"]
        drop_cols += [c for c in df.columns if c.startswith("_td_")]
        df = _without(df, drop_cols)

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
    from src.feature_semantics import jumps_mask
    _is_nh = jumps_mask(df).astype(float)

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

    # Compare weight burdens (claim-adjusted where verified) when at least one rival is measured.
    weight = burden_weight(df)
    field_weight = weight.groupby(df["race_id"], sort=False)
    comparable_weight = field_weight.transform("count").ge(2)
    df["weight_vs_field"] = (weight - field_weight.transform("mean")).where(comparable_weight)

    # Age relative to field
    race_avg_age = df.groupby("race_id")["age"].transform("mean")
    df["age_vs_field"] = df["age"] - race_avg_age

    # Field size categories
    df["small_field"] = (df["num_runners"] <= 8).astype(int)
    df["large_field"] = (df["num_runners"] >= 14).astype(int)

    # ── Field-size-conditioned base rates ─────────────────────────
    # Explicit prior: a random runner wins 1/n and places k/n.
    # Gives the model an anchor that shifts with field size so it can
    # learn calibration offsets rather than raw rates.
    _nr_safe = _nr.clip(lower=1)
    df["base_win_rate"] = 1.0 / _nr_safe

    # places_paid follows standard UK EW rules
    _hcap = df.get("handicap", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    _pp = np.where(
        _nr <= 4, np.minimum(_nr, 3),  # match the trained <=4-runner top-three target
        np.where(_nr <= 7, 2,
        np.where(_nr <= 15, 3,
        np.where(_hcap, 4, 3)))
    )
    df["base_place_rate"] = _pp / _nr_safe.values

    # How far above baseline the market thinks this horse is (log scale)
    if "implied_prob" in df.columns:
        _ip = df["implied_prob"].clip(lower=1e-4)
        df["implied_prob_vs_base"] = np.log(_ip / df["base_win_rate"])

    # Prize money rank (proxy for race quality)
    df["prize_log"] = np.log1p(df["prize_money"])

    # Weight per furlong — at longer distances weight matters more
    if "distance_furlongs" in df.columns:
        safe_dist = df["distance_furlongs"].replace(0, np.nan).fillna(8.0)
        df["weight_per_furlong"] = weight / safe_dist

    # Handicap features
    if "handicap" in df.columns:
        df["handicap"] = df["handicap"].fillna(0).astype(int)
        # OR x handicap interaction — OR is most meaningful in handicaps.
        # Missing for an unrated handicapper, not the race-median stand-in.
        if "official_rating" in df.columns:
            from src.feature_semantics import observed_rating
            df["or_x_handicap"] = (
                (observed_rating(df) * df["handicap"]).where(df["handicap"].ne(0), 0.0)
            )

    # --- Odds coefficient of variation (field competitiveness) ---
    if "odds" in df.columns:
        race_std_odds = df.groupby("race_id")["odds"].transform("std").fillna(0)
        race_mean_odds = df.groupby("race_id")["odds"].transform("mean")
        df["odds_cv"] = np.where(
            race_mean_odds > 0, race_std_odds / race_mean_odds, 0
        )

    # --- Weight-burden rank among measured runners ---
    if "weight_lbs" in df.columns:
        df["weight_rank"] = field_weight.rank(ascending=False, method="min").where(comparable_weight)

    # --- OR vs top-rated in race (observed marks only) ---
    if "official_rating" in df.columns:
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        race_max_or = _mark.groupby(df["race_id"]).transform("max")
        df["or_vs_top"] = _mark - race_max_or

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

    # Official-rating context of the race field, from observed marks only.
    # data_processor gives an unrated runner the race-median OR; ranking and
    # spreading those stand-ins described the field as rated when it is not.
    # Missing where the race has too few rated runners to say.
    if "official_rating" in df.columns:
        from src.feature_semantics import observed_rating
        _mark = observed_rating(df)
        _by_race = _mark.groupby(df["race_id"])
        df["field_avg_or"] = _by_race.transform("mean")
        df["field_std_or"] = _by_race.transform("std")
        race_max_or = _by_race.transform("max")
        race_min_or = _by_race.transform("min")
        _or_rnk = _by_race.rank(ascending=False, method="first")
        race_second_or = _mark.where(_or_rnk.eq(2)).groupby(df["race_id"]).transform("max")
        df["or_rank_in_race"] = _by_race.rank(ascending=False, method="min")
        df["field_or_range"] = race_max_or - race_min_or
        df["field_top2_or_gap"] = race_max_or - race_second_or
        df["or_vs_second_top"] = _mark - race_second_or

    # Elo spread of the race field. ``field_avg_elo`` itself is the
    # opposition mean excluding the horse (add_opposition_strength_features);
    # it is not recomputed here, so the column and its rolling history agree.
    if "horse_elo" in df.columns:
        df["field_std_elo"] = df.groupby("race_id")["horse_elo"].transform("std").fillna(0)
        race_max_elo = df.groupby("race_id")["horse_elo"].transform("max")
        race_min_elo = df.groupby("race_id")["horse_elo"].transform("min")
        # Vectorised second-largest Elo
        _elo_rnk = df.groupby("race_id")["horse_elo"].rank(ascending=False, method="first")
        _second_elo_map = df.loc[_elo_rnk == 2, ["race_id", "horse_elo"]]\
            .drop_duplicates("race_id").set_index("race_id")["horse_elo"]
        _max_elo_map = df.groupby("race_id")["horse_elo"].max()
        race_second_elo = df["race_id"].map(
            _second_elo_map.reindex(_max_elo_map.index).fillna(_max_elo_map)
        )
        df["elo_pctile_in_race"] = df.groupby("race_id")["horse_elo"].rank(pct=True)
        df["field_elo_range"] = race_max_elo - race_min_elo
        df["field_top2_elo_gap"] = race_max_elo - race_second_elo
        df["elo_vs_second_top"] = df["horse_elo"] - race_second_elo

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

    # Normalised position: 0 = winner, 1 = last, a non-finish counted as
    # last (see ``_add_result_history_flags``). Leaving non-finishes out made
    # a jockey or trainer whose runners keep falling look better than one
    # whose runners finish down the field.
    if "_norm_pos_for_history" in df.columns:
        df["_norm_pos"] = df["_norm_pos_for_history"].astype(np.float32)
    else:
        nr = df["num_runners"].replace(0, 1).values.astype(np.float32)
        df["_norm_pos"] = (
            (df["finish_position"].values.astype(np.float32) - 1)
            / np.maximum(nr - 1, 1)
        )
    df["_norm_pos_safe"] = df["_norm_pos"].fillna(0.0)
    df["_norm_pos_weight"] = df["_norm_pos"].notna().astype(np.float32)

    # Use the expanding mean (prior-only) as the global prior so it
    # doesn't peek at future races.  The expanding mean converges
    # quickly (~0.45-0.50) so the prior is very stable.
    global_mean = 0.45  # neutral prior — close to the true dataset mean
    prior_strength = 15.0  # stronger shrinkage than win-rate (0.1 prior)

    # ---- Helper: target-encode one entity column ----
    def _te(entity_col: str, prefix: str, race_safe: bool = False):
        """Add ``{prefix}_te_norm_pos`` target-encoded feature.

        Lifetime (class-C) encoding routed through the seed-aware _cumulative
        primitive so the FeatureStore append can carry its state.
        cumulative_counts is always race-safe, which equals the plain
        cumsum-minus-current for one-row-per-race entities (horse/jockey); the
        ``race_safe`` flag is retained for call-site clarity but no longer
        changes the result.
        """
        _c = _cumulative(
            df[entity_col].to_numpy(), df["race_id"].to_numpy(),
            {
                f"{prefix}_te_sum": df["_norm_pos_safe"].to_numpy(),
                f"{prefix}_te_n": df["_norm_pos_weight"].to_numpy(),
            },
        )
        cum_sum = _c[f"{prefix}_te_sum"]
        cum_n = _c[f"{prefix}_te_n"]
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
        df = _without(df, ["_jt_combo"])

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
                    _np_sum=("_norm_pos_safe", "sum"),
                    _np_cnt=("_norm_pos_weight", "sum"),
                    race_date=("race_date", "first"),
                ).reset_index()
                # Sum-of-individual-norm_pos in window
                _ws, _ = _time_window(
                    _agg, entity_col, "race_date", "_np_sum", _te_w,
                )
                # Count-of-individual-runners in window
                _wc, _ = _time_window(
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
                val_sum, _ = _time_window(
                    df, entity_col, "race_date", "_norm_pos_safe", _te_w,
                )
                val_cnt, _ = _time_window(
                    df, entity_col, "race_date", "_norm_pos_weight", _te_w,
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
                        _event_dt=(_dt_col, "first"),
                    )
                    .reset_index()
                )
                # Seed-aware EWMA over the per-(entity, race) series via the
                # rolling primitive (carries an EWMA-converging tail), then map
                # the per-race value back onto each runner.
                _eo = _rolling(
                    _agg, entity_col, "_np_mean",
                    [(f"_{prefix}_te_ewma_raw", "ewma", _ewma_alpha)],
                )
                _agg["_ewma"] = _eo[f"_{prefix}_te_ewma_raw"]
                _lookup = _agg.set_index([entity_col, "race_id"])["_ewma"]
                _mi = pd.MultiIndex.from_arrays(
                    [df[entity_col], df["race_id"]]
                )
                raw = _lookup.reindex(_mi).values
            else:
                results = _rolling(
                    df, entity_col, "_norm_pos",
                    [(f"_{prefix}_te_ewma_raw", "ewma", _ewma_alpha)],
                )
                raw = results[f"_{prefix}_te_ewma_raw"]
            # Prior weighted-run count (class C, seed-aware) drives the shrink.
            cum_n = _cumulative(
                df[entity_col].to_numpy(), df["race_id"].to_numpy(),
                {f"_{prefix}_te_ewma_n": df["_norm_pos_weight"].to_numpy()},
            )[f"_{prefix}_te_ewma_n"]
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

    # Date-strict views: entities that appear several times per day
    # (jockeys/trainers/combos — and shared context keys like track and
    # going) must not see same-day earlier results in their encodings.
    _te_w_str = str(getattr(config, "TE_WINDOW_DAYS", 365))
    _suffixes = ["_te_norm_pos", f"_te_norm_pos_{_te_w_str}d", "_te_ewma"]

    def _te_cols(prefix: str) -> list[str]:
        return [f"{prefix}{s}" for s in _suffixes]

    if "jockey" in df.columns:
        df = _entity_day_start_view(df, ["jockey"], _te_cols("jockey"))
    if "trainer" in df.columns:
        df = _entity_day_start_view(df, ["trainer"], _te_cols("trainer"))
    if "jockey" in df.columns and "trainer" in df.columns:
        df["_jt_combo"] = df["jockey"].astype(str) + "_" + df["trainer"].astype(str)
        df = _entity_day_start_view(df, ["_jt_combo"], _te_cols("jt"))
        df = _without(df, ["_jt_combo"])
    if "track" in df.columns:
        df = _entity_day_start_view(df, ["track"], _te_cols("track"))
    if "going" in df.columns:
        df = _entity_day_start_view(df, ["going"], _te_cols("going"))

    df = _without(df, ["_norm_pos", "_norm_pos_safe", "_norm_pos_weight"])
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

    for name, arr in _rolling(df,hkey, "field_avg_elo", _opp_ops).items():
        df[name] = arr

    # ── 3. Prior-race Elo-vs-field EWMA ──────────────────────────────
    # How consistently has this horse outrated / underrated its fields?
    df["_elo_vs_opp"] = df["horse_elo"] - df["field_avg_elo"]
    _vs_ops = [("elo_vs_field_ewma", "ewma", _ewma_alpha)]
    for name, arr in _rolling(df,hkey, "_elo_vs_opp", _vs_ops).items():
        df[name] = arr
    df = _without(df, ["_elo_vs_opp"])

    # Leave cold-start opposition features missing. Tree learners can
    # branch on missingness directly, which is preferable to inventing a
    # synthetic neutral field-quality history for unraced horses.

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
    # Shared inputs for the class-C cross-entity counts (seed-aware _cumulative).
    _rs = df["_result_start"].fillna(0.0).to_numpy()
    _wh = df["_won_for_history"].fillna(0.0).to_numpy()
    _race = df["race_id"].to_numpy()

    # --- Jockey × going ---
    if "going" in df.columns:
        _jg = _cumulative(
            _composite_key(df, "jockey", "going"), _race,
            {"jockey_going_runs": _rs, "_jg_w": _wh},
        )
        df["jockey_going_runs"] = _jg["jockey_going_runs"]
        df["_jg_w"] = _jg["_jg_w"]
        df["jockey_going_wr"] = _rate(df["_jg_w"], df["jockey_going_runs"])
        df = _entity_day_start_view(
            df, ["jockey", "going"], ["jockey_going_runs", "jockey_going_wr"]
        )
        df["jockey_going_wr"] = _shrink_to_entity(
            df, "jockey_going_wr", "jockey_going_runs", "jockey"
        )
        df = _without(df, ["_jg_w"])

    # --- Jockey × distance category ---
    if "dist_category" in df.columns:
        df["_dc_str"] = df["dist_category"].astype(str)
        _jdc = _cumulative(
            _composite_key(df, "jockey", "_dc_str"), _race,
            {"jockey_dist_cat_runs": _rs, "_jdc_w": _wh},
        )
        df["jockey_dist_cat_runs"] = _jdc["jockey_dist_cat_runs"]
        df["_jdc_w"] = _jdc["_jdc_w"]
        df["jockey_dist_cat_wr"] = _rate(df["_jdc_w"], df["jockey_dist_cat_runs"])
        df = _entity_day_start_view(
            df, ["jockey", "_dc_str"], ["jockey_dist_cat_runs", "jockey_dist_cat_wr"]
        )
        df["jockey_dist_cat_wr"] = _shrink_to_entity(
            df, "jockey_dist_cat_wr", "jockey_dist_cat_runs", "jockey"
        )
        df = _without(df, ["_dc_str", "_jdc_w"])

    # --- Horse × going (win rate on this going type) ---
    if "going" in df.columns:
        # Tagged state key — this recomputes horse_going_runs (also produced in
        # add_track_features); a distinct key keeps the two seed states separate.
        _hg = _cumulative(
            _composite_key(df, hkey, "going"), _race,
            {"horse_going_runs__xent": _rs, "_hg_w": _wh},
        )
        df["horse_going_runs"] = _hg["horse_going_runs__xent"]
        df["_hg_w"] = _hg["_hg_w"]
        df["horse_going_wr"] = _rate(df["_hg_w"], df["horse_going_runs"])
        df = _without(df, ["_hg_w"])

    # --- Horse × going × distance rounded (C&D&G specialisation) ---
    if "going" in df.columns and "distance_furlongs" in df.columns:
        df["_dist_r2"] = (df["distance_furlongs"] * 2).round() / 2
        _hgd = _cumulative(
            _composite_key(df, hkey, "going", "_dist_r2"), _race,
            {"horse_going_dist_runs": _rs, "_hgd_w": _wh},
        )
        df["horse_going_dist_runs"] = _hgd["horse_going_dist_runs"]
        df["_hgd_w"] = _hgd["_hgd_w"]
        df["horse_going_dist_wr"] = _rate(df["_hgd_w"], df["horse_going_dist_runs"])
        df = _without(df, ["_dist_r2", "_hgd_w"])

    # --- Trainer × race class (race-safe) ---
    if "race_class" in df.columns:
        df["_tc_key"] = df["trainer"].astype(str) + "||" + df["race_class"].astype(str)
        _tc = _cumulative(
            df["_tc_key"].to_numpy(), _race,
            {"trainer_class_runs": df["_result_start"].to_numpy(), "_tc_w": df["won"].to_numpy()},
        )
        df["trainer_class_runs"] = _tc["trainer_class_runs"]
        df["_tc_w"] = _tc["_tc_w"]
        df["trainer_class_wr"] = _rate(df["_tc_w"], df["trainer_class_runs"])
        df = _entity_day_start_view(
            df, ["_tc_key"], ["trainer_class_runs", "trainer_class_wr"]
        )
        df["trainer_class_wr"] = _shrink_to_entity(
            df, "trainer_class_wr", "trainer_class_runs", "trainer"
        )
        df = _without(df, ["_tc_w", "_tc_key"])

    return df


# ── Conditional feature masking ──────────────────────────────────
# Some features are only meaningful for certain race types.
# We only gate the hard structural case here: National Hunt
# (Hurdle / Chase) races have no stall draws, so draw-derived
# features are not merely low-signal there, they are inapplicable.
#
# Tree models in this project can already branch on missingness, so
# use NaN for not-applicable continuous features instead of inventing
# a numeric value that might carry unintended meaning.

# Draw-derived features that are inapplicable for NH races.
_NH_INAPPLICABLE_DRAW_FEATURES = [
    "draw_pct",
    "draw_x_bias",
    "draw_x_sprint_x_large",
    "draw_x_inv_dist",
    "draw_x_field_size",
    "draw_bias_x_sprint",
    "track_draw_win_rate",
    "tight_track_x_draw",
    "track_draw_bias",
    "draw_pct_x_soft_turf",
    "draw_bias_x_soft_turf",
]


def apply_conditional_feature_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Mask structurally inapplicable features for certain race types.

    Draw-related features are set to NaN for Hurdle / Chase because
    NH races have no stall draws. This prevents the model from
    learning from random or backfilled draw artefacts while preserving
    the information that the feature is not applicable for that row.

    Returns:
        DataFrame with masked features.
    """
    logger.info("Applying conditional feature masks (race-type gating)…")
    df = df.copy()

    from src.feature_semantics import jumps_mask
    is_nh = jumps_mask(df)
    no_stall = pd.to_numeric(df["draw"], errors="coerce").isna() | df["draw"].le(0) if "draw" in df else is_nh
    draw_inapplicable = is_nh | no_stall

    is_flat = ~is_nh
    n_nh = int(is_nh.sum())

    masked_count = 0
    for col in _NH_INAPPLICABLE_DRAW_FEATURES:
        if col in df.columns:
            df.loc[draw_inapplicable, col] = np.nan
            masked_count += 1

    logger.info(
        f"  Masked {masked_count} draw-derived features "
        f"(set to NaN for {n_nh:,} NH rows)"
    )
    return df


def add_supplementary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional derived features: consistency, class trajectory,
    campaign intensity, seasonal, field composition, market intelligence."""
    logger.info("Engineering supplementary features...")
    hkey = _horse_key(df)

    # ── Consistency / Volatility ─────────────────────────────────
    _range_ops = [
        ("_pos_max_10", "max", 10),
        ("_pos_min_10", "min", 10),
        ("horse_pos_stddev_10", "std", 10),
    ]
    # On the field-share position (non-finish = last); missing without
    # enough prior runs rather than a placeholder spread.
    _range_results = _rolling(
        df, hkey, "_norm_pos_for_history", _range_ops,
    )
    df["horse_pos_range_10"] = (
        _range_results["_pos_max_10"] - _range_results["_pos_min_10"]
    )
    df["horse_pos_stddev_10"] = _range_results["horse_pos_stddev_10"]

    # Unlucky loser rate: places but doesn't win
    if "horse_place_rate_shrunk" in df.columns and "horse_win_rate_shrunk" in df.columns:
        df["horse_unlucky_rate"] = (
            df["horse_place_rate_shrunk"] - df["horse_win_rate_shrunk"]
        )

    # ── Class Trajectory ────────────────────────────────────────
    if "race_class" in df.columns:
        from src.feature_semantics import class_number
        _class_num = class_number(df["race_class"]).to_numpy()
        df["_class_num_tmp"] = _class_num

        # Class trend: short-term vs long-term avg class
        # Negative = horse recently aimed at higher (tougher) races
        _class_results = _rolling(
            df, hkey, "_class_num_tmp",
            [("_class_avg_3", "mean", 3), ("_class_avg_10", "mean", 10)],
        )
        df["horse_class_trend"] = (
            _class_results["_class_avg_3"] - _class_results["_class_avg_10"]
        )
        df["horse_class_trend"] = df["horse_class_trend"].fillna(0)

        # Best (lowest) class ever won at
        df["_win_class"] = np.where(
            (df["won"] == 1) & (_class_num > 0), _class_num, np.nan,
        )
        # Vectorised expanding min: fill NaN with inf so cummin ignores
        # non-winning races, then shift(1) for prior-only.
        df["_wc_filled"] = df["_win_class"].fillna(np.inf)
        df["horse_best_class_won"] = _running_extreme(df, hkey, "_wc_filled", kind="min", prev=False)
        df.loc[df["horse_best_class_won"] == np.inf, "horse_best_class_won"] = np.nan
        df["horse_best_class_won"] = _lag(df, hkey, "horse_best_class_won")
        df["horse_best_class_won"] = df["horse_best_class_won"].fillna(7)
        df = _without(df, ["_wc_filled"])

        # Current class vs best proven level (negative = below proven)
        df["class_vs_best_won"] = _class_num - df["horse_best_class_won"]

        df = _without(df, ["_class_num_tmp", "_win_class"])

    # ── Campaign Intensity ─────────────────────────────────────
    if "race_date" in df.columns:
        count_source = "_result_start_for_history" if "_result_start_for_history" in df.columns else "_ones"
        if count_source == "_ones":
            df["_ones"] = 1.0
        # Sum the start flags rather than count rows: a non-start is not a run.
        starts_14, _ = _time_window(df, hkey, "race_date", count_source, 14)
        starts_30, _ = _time_window(df, hkey, "race_date", count_source, 30)
        df["races_in_last_14d"] = starts_14
        df["races_in_last_30d"] = starts_30
        df = _without(df, ["_ones"])

    # ── Trainer Switch ──
    if "trainer" in df.columns:
        horse_groups = df.groupby(hkey)
        df["_prev_trainer"] = _lag(df, hkey, "trainer")
        df["trainer_switch"] = (
            df["_prev_trainer"].notna()
            & (df["trainer"] != df["_prev_trainer"])
        ).astype(np.int8)
        df = _without(df, ["_prev_trainer"])

    # ── Same-trainer runners in race ──
    if "trainer" in df.columns:
        df["same_trainer_runners"] = df.groupby(
            ["race_id", "trainer"]
        )["trainer"].transform("count")

    # ── Beaten-lengths trend ──
    if "horse_ewma_lb_3" in df.columns and "horse_ewma_lb_7" in df.columns:
        # Negative = improving (getting closer to winner)
        df["lb_trend"] = df["horse_ewma_lb_3"] - df["horse_ewma_lb_7"]
        df["lb_trend"] = df["lb_trend"].fillna(0)

    # ── Field Composition ──
    if "horse_wins" in df.columns:
        df["_has_won_before"] = (df["horse_wins"] > 0).astype(float)
        df["n_prev_winners_in_field"] = df.groupby("race_id")[
            "_has_won_before"
        ].transform("sum")
        df = _without(df, ["_has_won_before"])

    if "horse_prev_races" in df.columns:
        _med_runs = df.groupby("race_id")["horse_prev_races"].transform("median")
        df["field_experience_gap"] = df["horse_prev_races"] - _med_runs

    # ── Seasonal / Cyclical ──
    if "race_date" in df.columns:
        _month = pd.to_datetime(df["race_date"]).dt.month
        df["month_sin"] = np.sin(2 * np.pi * _month / 12)
        df["month_cos"] = np.cos(2 * np.pi * _month / 12)

    # ── Market vs Elo mispricing ──
    if "odds_rank" in df.columns and "horse_elo_rank" in df.columns:
        # Positive = market rates horse lower than Elo suggests
        df["odds_vs_elo_rank"] = df["horse_elo_rank"] - df["odds_rank"]

    # ── Short-window consistency (5-run) ─────────────────────────
    _cons_ops = [
        ("horse_pos_stddev_5", "std", 5),
        ("_pos_max_5", "max", 5),
        ("_pos_min_5", "min", 5),
    ]
    _cons_results = _rolling(
        df, hkey, "_norm_pos_for_history", _cons_ops,
    )
    df["horse_pos_stddev_5"] = _cons_results["horse_pos_stddev_5"]
    df["horse_pos_range_5"] = _cons_results["_pos_max_5"] - _cons_results["_pos_min_5"]

    # ── Beaten-lengths consistency ───────────────────────────────
    if "lengths_behind" in df.columns:
        _lb_cons = _rolling(
            df, hkey, "beaten_lengths" if "beaten_lengths" in df else "lengths_behind",
            [("horse_lb_stddev_5", "std", 5)],
        )
        df["horse_lb_stddev_5"] = _lb_cons["horse_lb_stddev_5"]

    # ── Distance versatility ─────────────────────────────────────
    # Std of finish positions across different distance categories.
    # Low value = consistent across distances; high = distance-dependent.
    if "dist_category" in df.columns:
        _dist_runs = df.groupby([hkey, "dist_category"])["finish_position"].transform("count")
        _n_dist_cats = _distinct_count(df, hkey, "dist_category")
        df["horse_n_dist_cats"] = _n_dist_cats.fillna(1).astype(np.int8)

    return df


def add_race_relative_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add extensive within-race relative versions of strong base signals."""
    logger.info("Engineering race-relative signal features...")

    if "race_id" not in df.columns:
        return df

    race_ids = df["race_id"]
    relative_cols: dict[str, pd.Series | np.ndarray] = {}

    def _add_relative_family_batch(
        cols: list[str],
        *,
        higher_is_better: bool,
    ) -> None:
        available = [col for col in cols if col in df.columns]
        if not available:
            return

        numeric_frame = (
            df[available]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        valid_cols = list(numeric_frame.columns)
        if not valid_cols:
            return
        numeric_frame = numeric_frame[valid_cols]

        grouped = numeric_frame.groupby(race_ids)
        race_mean = grouped.transform("mean")
        race_std = grouped.transform("std").fillna(0.0)

        # ``_vs_best_in_race`` compares each runner with the best *other*
        # runner, so the leader keeps its margin over the next best instead
        # of reading 0 like everyone level with it.
        oriented = numeric_frame if higher_is_better else -numeric_frame
        race_top = oriented.groupby(race_ids).transform("max")
        is_top = oriented.eq(race_top)
        sole_top = is_top & is_top.groupby(race_ids).transform("sum").eq(1)
        runner_up = oriented.where(~is_top).groupby(race_ids).transform("max")
        best_other = race_top.where(~sole_top, runner_up)

        signed_vs_field = numeric_frame - race_mean if higher_is_better else race_mean - numeric_frame
        signed_vs_best = (oriented - best_other).astype("float64")
        rank_pct = numeric_frame.groupby(race_ids).rank(
            ascending=not higher_is_better,
            pct=True,
            method="average",
        )

        # A missing base value stays missing in every transform; filling it
        # with 0 made "no data" read as field average or level with the best.
        for col in valid_cols:
            relative_cols[f"{col}_vs_field"] = signed_vs_field[col]
            relative_cols[f"{col}_z_in_race"] = pd.Series(
                np.where(
                    race_std[col] > 1e-12,
                    signed_vs_field[col] / race_std[col],
                    0.0,
                ),
                index=df.index,
            ).where(numeric_frame[col].notna())
            relative_cols[f"{col}_rank_pct_in_race"] = rank_pct[col]
            relative_cols[f"{col}_vs_best_in_race"] = signed_vs_best[col]

    higher_better_cols = [
        "horse_win_rate_shrunk",
        "horse_place_rate_shrunk",
        "horse_prev_races",
        "close_finish_rate",
        "horse_dist_runs",
        "horse_dist_win_rate",
        "horse_going_runs",
        "horse_going_win_rate",
        "horse_going_wr_shrunk",
        "horse_dist_cat_wr_shrunk",
        "horse_surface_wr_shrunk",
        "horse_surface_rt_wr_shrunk",
        "horse_rt_specialist_edge",
        "jockey_prev_rides",
        "jockey_elo",
        "jockey_win_rate_shrunk",
        "jockey_place_rate_shrunk",
        "jockey_wr_14d",
        "jockey_pr_14d",
        "jockey_wr_30d",
        "jockey_pr_30d",
        "jockey_pr_30d_delta",
        "jockey_track_runs",
        "jockey_track_win_rate",
        "jockey_dist_cat_runs",
        "jockey_elo_rt_edge",
        "jt_prev_runs",
        "trainer_prev_runs",
        "trainer_freq",
        "trainer_elo",
        "trainer_win_rate_shrunk",
        "trainer_place_rate_shrunk",
        "trainer_wr_14d",
        "trainer_pr_14d",
        "trainer_wr_30d",
        "trainer_pr_30d",
        "trainer_pr_30d_delta",
        "trainer_track_runs",
        "trainer_track_win_rate",
        "trainer_class_runs",
        "trainer_class_wr",
        "trainer_dist_cat_runs",
        "trainer_hot_vs_30d",
        "trainer_dist_cat_wr",
        "trainer_going_runs",
        "trainer_going_wr",
        "trainer_intent_score",
        "class_change",
        "field_avg_elo_5",
        "field_avg_elo_10",
        "field_avg_elo_ewma",
        # Bases are raw values, never ``*_vs_field`` columns: centring an
        # already-centred column reproduces it exactly.
        "horse_elo_momentum",
        "horse_margin_elo_momentum",
        "combined_elo",
        "horse_elo_surface_edge",
        "horse_elo_rt_edge",
        "horse_elo_dist_cat_edge",
        # ── ATR (TPD) sectional history (replaces RTV) ──
        "atr_finishing_speed_pct_avg_3",
        "atr_finishing_speed_pct_avg_5",
        "atr_finishing_speed_pct_best_3",
        "atr_finishing_speed_pct_trend",
        "atr_max_speed_avg_3",
        "atr_max_speed_avg_5",
        "atr_max_speed_best_3",
        "atr_max_speed_trend",
        "atr_closing_speed_avg_3",
        "atr_closing_speed_avg_5",
        "atr_closing_speed_best_3",
        "atr_closing_speed_trend",
        "atr_turn_of_foot_avg_3",
        "atr_turn_of_foot_avg_5",
        "atr_turn_of_foot_best_3",
        "atr_efficiency_score_avg_3",
        "atr_efficiency_score_avg_5",
        "atr_efficiency_score_best_3",
        "atr_energy_eff_avg_3",
        "atr_energy_eff_avg_5",
        "atr_peak_stride_length_avg_3",
        "atr_peak_stride_length_avg_5",
        "atr_peak_stride_freq_avg_3",
        "atr_upgrade_clean_avg_3",
        "atr_upgrade_clean_avg_5",
        "atr_upgrade_clean_best_3",
        "atr_closing_in_hand_avg_3",
        "atr_closing_in_hand_avg_5",
        "atr_early_pace_avg_3",
        "atr_speed_score",
        "atr_stride_x_gallop",
        # Comment-tag recent rates
        "atr_cmt_trouble_rate_5",
        "atr_cmt_stayed_on_rate_5",
        "atr_cmt_weakened_rate_5",
        "atr_cmt_outpaced_rate_5",
        "atr_cmt_front_rate_5",
        "atr_cmt_held_up_rate_5",
        "atr_cmt_green_rate_5",
        "atr_cmt_nh_error_rate_5",
    ]
    _add_relative_family_batch(higher_better_cols, higher_is_better=True)

    lower_better_cols = [
        "horse_avg_position",
        "horse_avg_pos_3",
        "horse_avg_pos_5",
        "horse_avg_pos_10",
        "horse_ewma_pos_3",
        "horse_ewma_pos_7",
        "horse_avg_lb_behind",
        "horse_ewma_lb_3",
        "horse_ewma_lb_7",
        "speed_fig_avg_5",
        "speed_fig_avg_10",
        "speed_fig_best_5",
        "speed_fig_adj_avg_5",
        "speed_fig_adj_best_5",
        "speed_fig_rel_avg_5",
        "speed_z_class_avg_5",
        "speed_z_class_avg_10",
        "speed_z_class_best_5",
        "speed_fig_trend",
        "horse_class_trend",
        "lb_trend",
        "days_since_last_race",
        "days_log",
        "days_since_last_win",
        "years_from_peak_abs",
        "age_decline_curve",
        "horse_te_norm_pos",
        "horse_te_ewma",
        "jockey_te_norm_pos",
        "jockey_te_ewma",
        "trainer_te_norm_pos",
        "trainer_te_ewma",
        "jt_te_norm_pos",
    ]
    _te_w = int(getattr(config, "TE_WINDOW_DAYS", 365))
    if _te_w > 0:
        lower_better_cols.extend([
            f"horse_te_norm_pos_{_te_w}d",
            f"jockey_te_norm_pos_{_te_w}d",
            f"trainer_te_norm_pos_{_te_w}d",
        ])
    _add_relative_family_batch(lower_better_cols, higher_is_better=False)

    # Signed context shifts are often useful relative to the field, but
    # do not have a universal "higher is better" direction.
    context_cols = [col for col in ["distance_change", "or_change", "jockey_elo_change"] if col in df.columns]
    if context_cols:
        context_frame = (
            df[context_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        valid_context_cols = list(context_frame.columns)
        if valid_context_cols:
            context_frame = context_frame[valid_context_cols]
            grouped = context_frame.groupby(race_ids)
            race_mean = grouped.transform("mean")
            race_std = grouped.transform("std").fillna(0.0)
            signed_vs_field = context_frame - race_mean
            abs_rank = context_frame.abs().groupby(race_ids).rank(
                ascending=False,
                pct=True,
                method="average",
            )
            for col in valid_context_cols:
                relative_cols[f"{col}_vs_field"] = signed_vs_field[col]
                relative_cols[f"{col}_z_in_race"] = pd.Series(
                    np.where(
                        race_std[col] > 1e-12,
                        signed_vs_field[col] / race_std[col],
                        0.0,
                    ),
                    index=df.index,
                ).where(context_frame[col].notna())
                relative_cols[f"{col}_abs_rank_pct_in_race"] = abs_rank[col]

    if relative_cols:
        overlap = [col for col in relative_cols if col in df.columns]
        if overlap:
            df = _without(df, overlap)
        df = _with_columns(df, pd.DataFrame(relative_cols, index=df.index))

    composite_cols: dict[str, pd.Series] = {}

    def _composite(name: str, cols: list[str]) -> None:
        available = [c for c in cols if c in df.columns]
        if not available:
            return
        composite_cols[name] = df[available].mean(axis=1)

    _composite(
        "horse_form_relative_score",
        [
            "horse_win_rate_shrunk_z_in_race",
            "horse_place_rate_shrunk_z_in_race",
            "horse_avg_position_z_in_race",
            "horse_avg_lb_behind_z_in_race",
            "speed_fig_avg_5_z_in_race",
            "speed_fig_trend_z_in_race",
        ],
    )
    _composite(
        "connections_relative_score",
        [
            "jockey_elo_z_in_race",
            "jockey_wr_14d_z_in_race",
            "trainer_wr_14d_z_in_race",
            "trainer_win_rate_shrunk_z_in_race",
            "jockey_track_win_rate_z_in_race",
            "trainer_track_win_rate_z_in_race",
        ],
    )
    _composite(
        "suitability_relative_score",
        [
            "horse_dist_win_rate_z_in_race",
            "horse_going_win_rate_z_in_race",
            "horse_going_wr_shrunk_z_in_race",
            "horse_surface_wr_shrunk_z_in_race",
            "horse_surface_rt_wr_shrunk_z_in_race",
        ],
    )
    _composite(
        "freshness_relative_score",
        [
            "days_since_last_race_z_in_race",
            "days_since_last_win_z_in_race",
            "years_from_peak_abs_z_in_race",
            "age_decline_curve_z_in_race",
        ],
    )
    _composite(
        "stable_connections_relative_score",
        [
            "trainer_intent_score_z_in_race",
            "trainer_prev_runs_z_in_race",
            "trainer_wr_14d_z_in_race",
            "jockey_wr_14d_z_in_race",
        ],
    )

    if composite_cols:
        df = _with_columns(df, pd.DataFrame(composite_cols, index=df.index))

    return df


def _use_day_order_after_ratings(df):
    """Preserve the existing day-order history policy after the rating stages."""
    return _without(df, ["_event_dt"])


@dataset_build
def engineer_features(
    df: pd.DataFrame,
    save: bool = True,
    *,
    skip_ratings: bool = False,
    counts_state: dict | None = None,
    defer_global_imputation: bool = False,
    half: str | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Processed DataFrame from data_processor
        save: Whether to save the feature-engineered data
        skip_ratings: When True, the Elo/Glicko rating *sweeps* are not run
            and the rating columns are expected to already be present on
            ``df`` (attached by the caller from persisted state). Used by
            the incremental feature store to avoid re-sweeping ratings over
            the full history on every append (the sweeps are ~47% of FE).
            Downstream rating-dependent stages then read those columns
            directly.
        half: Lets two processes share a build. ``"first"`` stops after
            ``add_atr_features`` and returns the working frame; ``"second"``
            takes that frame and finishes it. Each half carries its own
            ``counts_state``, so no entity may be tracked on both sides.

    Returns:
        DataFrame with all engineered features
    """
    if half != "second":
        logger.info(f"Starting feature engineering on {len(df)} records...")

        # Ensure race_date is proper datetime (CSV round-trip turns it to str)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

        # Sort once — every sub-function needs chronological order for
        # cumsum / rolling / shift.  Sorting here avoids 12 redundant sorts.
        df["_event_dt"] = _event_sort_key(df)
        df = df.sort_values(["_event_dt", "race_id"]).copy()
        df = _add_result_history_flags(df)

    # Pipeline stages run in order; each must see the columns its
    # predecessors added (dependency notes kept beside the entries).
    # _defrag between stages keeps the BlockManager from fragmenting as
    # each stage inserts its columns one-by-one.
    stages = [
        add_horse_features,
        add_established_form_features, # same-scale marks and discipline evidence
        add_hidden_form_features,      # external/unseen prior form (foreign raiders)
        add_pedigree_features,         # sire/dam/damsire edge
        add_medical_features,          # gelding
        add_prize_money_features,      # class shifts / purse jumps
        add_collateral_form_features,  # prev-race rivals' next-out form
        add_jockey_features,
        add_trainer_features,
        add_jockey_trainer_features,
        add_jockey_track_features,
        add_track_features,
        add_course_distance_features,
        add_class_change_features,
        add_distance_change_features,
        add_target_encoded_features,   # smoothed entity baselines
        add_cross_entity_features,     # jockey×going, horse×going, etc.
    ]
    if not skip_ratings:
        stages.append(compute_elo_features)
        if getattr(config, "GLICKO_ENABLED", True):
            stages.append(compute_glicko_features)  # rating + uncertainty (RD)
    # Retain the established downstream day-level ordering in every mode.
    # Its lifetime is an engine policy, independent of rating implementation.
    stages.append(_use_day_order_after_ratings)
    stages += [
        add_opposition_strength_features,  # needs horse_elo
        add_headgear_features,
        add_surface_features,
        add_trainer_specialisation_features,  # needs dist_category
        add_speed_figure_features,
        add_atr_features,
        add_market_features,
        add_trainer_runner_intent_features,  # stable pecking-order / intent
        add_jockey_intent_features,        # needs is_favourite + jockey_elo
        add_race_context_features,
        add_pace_style_features,           # needs ATR history + draw/field context
        add_field_quality_features,
        add_track_config_features,         # static venue data
        add_course_dynamics_features,      # course experience/handedness/pace-bias/travel
        add_supplementary_features,        # consistency, class, campaign, seasonal
        add_race_relative_signal_features,
        add_interaction_features,
        apply_conditional_feature_masks,   # zero irrelevant features
        add_prior_market_features,         # independent prior-day history, appended after existing stages
        add_rtv_features,                  # quality-filtered mature prior runs and observed pre-race context
    ]
    if half is not None:
        handover = stages.index(add_atr_features) + 1
        stages = stages[:handover] if half == "first" else stages[handover:]
    _profile = os.environ.get("FE_PROFILE")
    # A seeded counts_state (FeatureStore append) makes class-C stages resume
    # from carried state instead of recomputing history; None = full build.
    with counts_state_scope(counts_state):
        if _profile:
            import time as _t
            _timings = []
            for stage in stages:
                _t0 = _t.perf_counter()
                df = stage(df)
                df = _defrag(df)
                _timings.append((stage.__name__, _t.perf_counter() - _t0))
            _timings.sort(key=lambda x: -x[1])
            _total = sum(t for _, t in _timings)
            logger.warning("FE stage profile (rows=%d, total=%.2fs):", len(df), _total)
            for _name, _sec in _timings:
                logger.warning("  %6.2fs  %5.1f%%  %s", _sec, 100 * _sec / _total, _name)
        else:
            for stage in stages:
                df = stage(df)
                df = _defrag(df)

    if half == "first":
        return df

    # Final consolidation before the NaN-handling block below.
    df = _defrag(df, max_blocks=0)

    # ── Feature-specific NaN handling ────────────────────────────
    # Prefer semantic defaults where they truly exist, and otherwise leave
    # missing values intact so the tree models can learn from cold starts
    # and unknown context directly.
    # Unknown speed history stays missing. Full-frame medians include future
    # results and make old rows change when history is appended; tree models
    # can use missingness directly. The legacy defer_global_imputation argument
    # remains accepted for callers, but no global imputation is performed.

    # Odds-based columns: fill with field median or neutral values
    odds_fill = {"implied_prob": 0.05, "norm_implied_prob": 0.05,
                 "log_odds": np.log1p(20.0), "odds_vs_field": 1.0,
                 "odds_rank": 5.0, "overround": 1.0,
                 "trainer_runner_rank_by_odds": 1.0,
                 "trainer_odds_vs_stablemate_best": 0.0,
                 "trainer_stable_market_share": 1.0,
                 "trainer_string_disagreement_or_odds": 0.0,
                 "trainer_string_disagreement_elo_odds": 0.0}
    for c, fill_val in odds_fill.items():
        if c in df.columns:
            df[c] = df[c].fillna(fill_val)

    # History columns (days since a run or win, raw rates, form) stay
    # missing for a horse without that history: a placeholder such as 60
    # days or a 0 rate reads as a real, specific record.

    # Weather columns: fill with UK averages
    weather_defaults = {
        "weather_temp_max": 14.0, "weather_temp_min": 7.0,
        "weather_precip_mm": 1.5, "weather_wind_kmh": 18.0,
        "weather_precip_prev3": 4.5, "weather_temp_range": 7.0,
    }
    for c, val in weather_defaults.items():
        if c in df.columns:
            df[c] = df[c].fillna(val)

    # Drop temporary sort helper.
    df = _without(df, [
        "_event_dt",
        "_finished_result",
        "_non_finisher_result",
        "_result_start",
        "_finish_pos_for_history",
        "_norm_pos_for_history",
        "_won_for_history",
        "_placed_for_history",
        "_result_start_for_history",
        "_non_finisher_for_history",
    ])

    # Keep only numeric features something reads (src/store_columns.py, research 109): every stage still
    # runs, so later stages see their inputs, but unread columns are not stored or served.
    from src.store_columns import prune
    df = prune(df)

    if save:
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

    processed = process_data()
    featured = engineer_features(processed)
    print(f"\nFeatured dataset shape: {featured.shape}")
    print(f"\nAll features:\n{list(featured.columns)}")
