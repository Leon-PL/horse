"""History primitives: grouped rolling, cumulative, lag and time-window
calculations that resume from carried state (``counts_state_scope``).

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

from contextlib import contextmanager
from itertools import chain, repeat
from typing import NamedTuple

import numba as nb
import numpy as np
import pandas as pd

from src.fe_incremental import cumulative_counts, cumulative_counts_plain


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
        _new_block = np.empty(n, dtype=bool)
        _new_block[0] = True
        _new_block[1:] = (gids[1:] != gids[:-1]) | (_race_sorted[1:] != _race_sorted[:-1])
        _block_first = np.maximum.accumulate(np.where(_new_block, np.arange(n), 0))
    else:
        _block_first = np.arange(n)

    results = {}
    for out_name, op, param in operations:
        arr = _rolling_kernel(shifted, gids, op, param)
        results[out_name] = arr[_block_first][_inv]   # race-safe broadcast, then back to row order
    return results


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
    if _COUNTS_STATE is None:
        return _time_window_stats_core(df, entity_col, date_col, value_col, window_days)
    key = f"tw::{entity_col}::{value_col}::{window_days}"
    out, new_state = _time_window_with_seed(
        df, entity_col, date_col, value_col, window_days, _COUNTS_STATE.get(key)
    )
    _COUNTS_STATE[key] = new_state
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


# ── Incremental cumulative-state threading ─────────────────────────────────
# A seeded run (FeatureStore append) sets a flat {column: accumulator_state}
# dict here; class-C stages seed their cumulative_counts from it and write back
# their advanced state, so an append never recomputes history. A full build
# leaves it None -> stages compute from scratch (byte-identical). See
# docs/architecture/FE_REFACTOR_PLAN.md.
_COUNTS_STATE: dict | None = None


@contextmanager
def counts_state_scope(state: dict | None):
    """Activate a seed/collector dict for cumulative-count stages."""
    global _COUNTS_STATE
    prev = _COUNTS_STATE
    _COUNTS_STATE = state
    try:
        yield
    finally:
        _COUNTS_STATE = prev
        _TIMELINES.clear()


def _carried(key: str):
    """State carried under ``key`` by the enclosing ``counts_state_scope``, else None.

    Stage modules call this rather than importing ``_COUNTS_STATE``: an imported
    name keeps the value it had at import time, not the scope's rebinding.
    """
    return None if _COUNTS_STATE is None else _COUNTS_STATE.get(key)


def _carry(key: str, state) -> None:
    """Store ``state`` under ``key`` when a ``counts_state_scope`` is open."""
    if _COUNTS_STATE is not None:
        _COUNTS_STATE[key] = state


def _cumulative(keys, race_ids, metrics: dict) -> dict:
    """Seed-aware cumulative counts for a class-C stage.

    With no active seed (full build) this equals the inline
    ``groupby(key).cumsum() - flag`` formula. Under ``counts_state_scope`` it
    seeds from the carried state and writes the advanced state back, so the
    same stage code serves both build and append.
    """
    out, new_state = cumulative_counts(keys, race_ids, metrics, state=_COUNTS_STATE)
    if _COUNTS_STATE is not None:
        _COUNTS_STATE.update(new_state)
    return out


def _cumulative_plain(keys, metrics: dict) -> dict:
    """Seed-aware **plain** cumulative counts (cumsum - current) for a class-C
    stage with at most one observation per entity per race. Mirrors
    ``_cumulative`` but threads ``PlainCountsAccumulator`` state."""
    out, new_state = cumulative_counts_plain(keys, metrics, state=_COUNTS_STATE)
    if _COUNTS_STATE is not None:
        _COUNTS_STATE.update(new_state)
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
_TIMELINES: dict = {}


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
    known = _TIMELINES.get(slot)
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
    _TIMELINES[slot] = line
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
    if _COUNTS_STATE is None:
        return shifted
    key = f"lag::{entity_col}::{col}"
    seed = _COUNTS_STATE.get(key, {})
    entities = df[entity_col].to_numpy()
    carried = _in_frame(seed, entities)
    if carried:
        ent = pd.Series(entities, index=df.index)
        first_mask = pd.Series((g.cumcount() == 0).to_numpy(), index=df.index)
        shifted = shifted.where(~first_mask, ent.map(carried))
    last = _last_rows(g, entities)
    new_state = dict(seed)
    new_state.update(zip(entities[last], df[col].to_numpy()[last]))
    _COUNTS_STATE[key] = new_state
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
    if _COUNTS_STATE is None:
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
    state = _COUNTS_STATE.get(state_key, {})  # entity -> (R, L)
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
    _COUNTS_STATE[state_key] = new_state
    return result


def _ffill_prev(df, entity_col, col):
    """Per-entity prior-only forward-fill (``groupby[col].ffill().shift(1)``),
    seed-aware (class L). State holds the last non-null value per entity (any
    dtype)."""
    g = df.groupby(entity_col, sort=False)
    filled = g[col].ffill()
    entities = df[entity_col].to_numpy()
    filled_prev = filled.groupby(entities).shift(1)
    if _COUNTS_STATE is None:
        return filled_prev
    state_key = f"ffill::{entity_col}::{col}"
    seed = _COUNTS_STATE.get(state_key, {})
    carried = _in_frame(seed, entities)
    if carried:
        seed_vals = pd.Series(entities, index=df.index).map(carried)
        filled_prev = filled_prev.where(filled_prev.notna(), seed_vals)
    last = _last_rows(g, entities)
    new_state = dict(seed)
    for e, v in zip(entities[last], filled.to_numpy()[last]):
        if pd.notna(v):
            new_state[e] = v
    _COUNTS_STATE[state_key] = new_state
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

    if _COUNTS_STATE is None:
        return _compute(df.groupby(key_col, sort=False)[col])

    state_key = f"exp::{key_col}::{col}"
    seed = _COUNTS_STATE.get(state_key, {})  # key -> chronological list of values
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
    _COUNTS_STATE[state_key] = new_state
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
    seed = _COUNTS_STATE.get(state_key, {}) if _COUNTS_STATE is not None else {}

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

    if _COUNTS_STATE is not None:
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
        _COUNTS_STATE[state_key] = new_state
    return out


def _rolling(df, group_key, value_col, operations) -> dict:
    """Rolling features, seed-aware under ``counts_state_scope`` (class B).

    Full build (no active state) calls ``_fast_grouped_rolling`` directly —
    byte-identical to before. Under a seed scope it carries a per-entity raw
    tail so an append can run over new rows only. State is keyed by the first
    output name (globally unique)."""
    if _COUNTS_STATE is None:
        return _fast_grouped_rolling_core(df, group_key, value_col, operations)
    state_key = f"roll::{operations[0][0]}"
    results, new_state = _rolling_with_seed(
        df, group_key, value_col, operations, _COUNTS_STATE.get(state_key)
    )
    _COUNTS_STATE[state_key] = new_state
    return results
