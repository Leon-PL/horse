"""Shared helpers for feature stages: event order, keys, frame and rate helpers.

Moved out of ``src/feature_engineer.py`` on 2026-11-07 (research 065).
"""

import numpy as np
import pandas as pd


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


#: Distance bands in furlongs: Sprint / Mile / Middle / Staying.
DIST_CATEGORY_BINS = (0, 6.5, 8.5, 11.5, 50)


def dist_category(distance_furlongs: pd.Series) -> pd.Series:
    """Distance band 0-3 (Sprint / Mile / Middle / Staying); unknown is Mile.

    The one definition. The dimensional rating sweeps read it, so the store's
    standalone sweep and the full pipeline must bin identically.
    """
    return pd.cut(
        distance_furlongs, bins=list(DIST_CATEGORY_BINS), labels=[0, 1, 2, 3], include_lowest=True,
    ).astype(float).fillna(1)


def _horse_key(df: pd.DataFrame) -> str:
    """The column every horse-history stage groups by: always ``horse_id``.

    Names are not unique (race 688054 has two horses called Sierra Nevada), so
    a frame without a complete ``horse_id`` is an error rather than a cue to
    group by name.
    """
    if "horse_id" not in df.columns:
        raise ValueError("Horse history needs a horse_id column; names are not unique.")
    missing = int(df["horse_id"].isna().sum())
    if missing:
        raise ValueError(f"{missing:,} rows have no horse_id; horse history cannot be grouped by name.")
    return "horse_id"


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
