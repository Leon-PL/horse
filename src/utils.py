"""
Utility Functions
=================
Helper functions used across the horse racing prediction application.
"""

import logging
import os
import re
import shutil
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def cleanup_stale_caches(max_age_days: int | None = None) -> int:
    """Delete date-stamped cache entries older than *max_age_days*.

    Covers the live feature cache, lookahead cache and racecards cache —
    all keyed by a ``YYYY-MM-DD`` date in the file/folder name.  These
    are pure caches and are regenerated on demand.  Returns the number
    of paths removed.
    """
    import config

    if max_age_days is None:
        max_age_days = int(getattr(config, "CACHE_TTL_DAYS", 30))
    cutoff = datetime.now() - timedelta(days=max_age_days)
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")

    cache_dirs = [
        os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache"),
        os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache", "baseline"),
        os.path.join(config.PROCESSED_DATA_DIR, "lookahead_cache"),
        os.path.join(config.DATA_DIR, "racecards_cache"),
    ]

    removed = 0
    for cache_dir in cache_dirs:
        if not os.path.isdir(cache_dir):
            continue
        for entry in os.listdir(cache_dir):
            m = date_re.search(entry)
            if not m:
                continue
            try:
                entry_date = datetime.strptime(m.group(1), "%Y-%m-%d")
            except ValueError:
                continue
            if entry_date >= cutoff:
                continue
            path = os.path.join(cache_dir, entry)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except OSError as exc:
                logger.warning("Could not remove stale cache %s: %s", path, exc)
    if removed:
        logger.info("Removed %d stale cache entries (older than %d days)", removed, max_age_days)
    return removed


# ── Merge-key normalisers ─────────────────────────────────────────────
# Shared by the RTV scraper, Matchbook signal builder and paper-trade
# store so that horse/track/off-time joins behave identically everywhere.

def normalise_off_time_key(off_time) -> str:
    """Normalise off-time values to a stable ``HH:MM`` merge key."""
    s = str(off_time).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""
    # 1408 -> 14:08
    if re.fullmatch(r"\d{4}", s):
        return f"{s[:2]}:{s[2:]}"
    # 14:08[:SS] -> 14:08
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    # Last-resort digit extraction (e.g. "1408 BST")
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 4:
        return f"{digits[:2]}:{digits[2:4]}"
    return s


def normalise_track_key(track) -> str:
    """Normalise track names for robust joins."""
    return re.sub(r"\s+", " ", str(track).strip().lower())


def normalise_horse_key(name) -> str:
    """Normalise horse names for robust joins."""
    return re.sub(r"\s+", " ", str(name).strip().title())


def compact_numeric_dtypes(df: pd.DataFrame | None, *, label: str = "dataset") -> pd.DataFrame | None:
    """Downcast numeric columns to reduce dataset size in memory and on disk."""
    if df is None or df.empty:
        return df

    before_bytes = int(df.memory_usage(deep=False).sum())
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if not numeric_cols:
        return df

    for col in numeric_cols:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_float_dtype(series):
            downcasted = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_integer_dtype(series):
            downcast = "unsigned" if series.min(skipna=True) >= 0 else "integer"
            downcasted = pd.to_numeric(series, downcast=downcast)
        else:
            continue
        if downcasted.dtype != series.dtype:
            df[col] = downcasted

    after_bytes = int(df.memory_usage(deep=False).sum())
    if after_bytes < before_bytes:
        logger.info(
            "Compacted %s numeric dtypes: %.1f MB -> %.1f MB",
            label,
            before_bytes / (1024 * 1024),
            after_bytes / (1024 * 1024),
        )
    return df


def format_odds(odds: float) -> str:
    """Convert decimal odds to fractional display."""
    if odds <= 1:
        return "EVS"
    numerator = odds - 1
    # Common fractional odds
    common_fracs = [
        (0.5, "1/2"), (0.67, "4/6"), (0.8, "4/5"),
        (1.0, "EVS"), (1.5, "6/4"), (2.0, "2/1"),
        (2.5, "5/2"), (3.0, "3/1"), (4.0, "4/1"),
        (5.0, "5/1"), (6.0, "6/1"), (8.0, "8/1"),
        (10.0, "10/1"), (12.0, "12/1"), (14.0, "14/1"),
        (16.0, "16/1"), (20.0, "20/1"), (25.0, "25/1"),
        (33.0, "33/1"), (50.0, "50/1"), (100.0, "100/1"),
    ]
    closest = min(common_fracs, key=lambda x: abs(x[0] - numerator))
    return closest[1]


def kelly_criterion(
    prob: float,
    odds: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly Criterion bet size.

    Args:
        prob: Estimated win probability
        odds: Decimal odds offered
        fraction: Fraction of full Kelly to use (default 1/4 Kelly)

    Returns:
        Recommended stake as fraction of bankroll
    """
    b = odds - 1  # Net odds
    q = 1 - prob

    kelly = (b * prob - q) / b
    kelly = max(0, kelly)  # Never recommend negative bets

    return kelly * fraction


def print_race_prediction(results: pd.DataFrame):
    """Pretty-print race prediction results."""
    print("\n" + "=" * 70)
    print("  RACE PREDICTION")
    print("=" * 70)
    print(
        f"{'Rank':<6}{'Horse':<25}{'Win Prob':<12}"
        f"{'Odds':<10}{'Value':<10}"
    )
    print("-" * 70)

    for _, row in results.iterrows():
        rank = int(row["predicted_rank"])
        name = row["horse_name"][:24]
        prob = f"{row['win_probability']:.1%}"
        odds = f"{row.get('odds', 'N/A')}"
        value = f"{row.get('value_score', 0):+.3f}" if "value_score" in row else "N/A"

        marker = " ⭐" if rank == 1 else ""
        print(f"{rank:<6}{name:<25}{prob:<12}{odds:<10}{value:<10}{marker}")

    print("=" * 70)
