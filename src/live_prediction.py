from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import date, datetime, timedelta

import pandas as pd

import config
from src.feature_engineer import engineer_features

logger = logging.getLogger(__name__)


LIVE_FEATURE_CACHE_DIR = os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache")
LIVE_FEATURE_BASELINE_DIR = os.path.join(LIVE_FEATURE_CACHE_DIR, "baseline")
LOOKAHEAD_CACHE_DIR = os.path.join(config.PROCESSED_DATA_DIR, "lookahead_cache")
for _dir in (LIVE_FEATURE_CACHE_DIR, LIVE_FEATURE_BASELINE_DIR, LOOKAHEAD_CACHE_DIR):
    os.makedirs(_dir, exist_ok=True)


def history_source_signature(
    history_df: pd.DataFrame | None,
    *,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> str:
    payload: dict[str, object] = {
        "version": int(getattr(config, "LIVE_FEATURE_CACHE_VERSION", 1)),
        "history_months": int(getattr(config, "LIVE_FE_HISTORY_MONTHS", 30)),
        "source_path": os.path.abspath(source_path) if source_path else None,
        "source_mtime": float(source_mtime) if source_mtime is not None else None,
    }
    if source_path and os.path.exists(source_path):
        payload["source_size"] = int(os.path.getsize(source_path))
    if history_df is not None and not history_df.empty:
        race_dates = pd.to_datetime(history_df["race_date"], errors="coerce")
        payload.update({
            "rows": int(len(history_df)),
            "cols": int(len(history_df.columns)),
            "min_date": str(race_dates.min()),
            "max_date": str(race_dates.max()),
        })
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def gap_fill_signature(last_hist_date: date | None, target_date_str: str) -> str:
    if last_hist_date is None:
        return "no-history"
    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    gap_start = last_hist_date + timedelta(days=1)
    gap_end = target - timedelta(days=1)
    payload = {
        "gap_start": str(gap_start),
        "gap_end": str(gap_end),
        "days": max((gap_end - gap_start).days + 1, 0),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def cards_signature(cards_df: pd.DataFrame) -> str:
    if cards_df is None or cards_df.empty:
        return "empty"
    sort_cols = [col for col in ("race_id", "horse_name", "off_time") if col in cards_df.columns]
    ordered = cards_df.sort_values(sort_cols, kind="stable").reset_index(drop=True) if sort_cols else cards_df.reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha1(hashed).hexdigest()[:20]


def build_live_feature_cache_key(
    *,
    target_date_str: str,
    cards_sig: str,
    history_sig: str,
    gap_sig: str,
) -> str:
    payload = {
        "version": int(getattr(config, "LIVE_FEATURE_CACHE_VERSION", 1)),
        "target_date": target_date_str,
        "cards_sig": cards_sig,
        "history_sig": history_sig,
        "gap_sig": gap_sig,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def get_live_feature_cache_paths(target_date_str: str, cache_key: str) -> dict[str, str]:
    date_dir = os.path.join(LIVE_FEATURE_CACHE_DIR, target_date_str)
    os.makedirs(date_dir, exist_ok=True)
    file_stem = f"features_{cache_key}"
    return {
        "parquet": os.path.join(date_dir, f"{file_stem}.parquet"),
        "meta": os.path.join(date_dir, f"{file_stem}.json"),
        "baseline": os.path.join(LIVE_FEATURE_BASELINE_DIR, f"{target_date_str}_{file_stem}.parquet"),
    }


def live_feature_cache_exists(target_date_str: str, cache_key: str) -> bool:
    return os.path.exists(get_live_feature_cache_paths(target_date_str, cache_key)["parquet"])


def load_live_feature_cache(target_date_str: str, cache_key: str) -> pd.DataFrame | None:
    path = get_live_feature_cache_paths(target_date_str, cache_key)["parquet"]
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path, engine="pyarrow")


def save_live_feature_cache(
    target_date_str: str,
    cache_key: str,
    featured_df: pd.DataFrame,
    *,
    metadata: dict[str, object] | None = None,
    preserve_baseline: bool = False,
) -> dict[str, str]:
    paths = get_live_feature_cache_paths(target_date_str, cache_key)
    featured_df.to_parquet(paths["parquet"], index=False, engine="pyarrow")
    meta = {"cache_key": cache_key, **(metadata or {})}
    with open(paths["meta"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    if preserve_baseline and not os.path.exists(paths["baseline"]):
        shutil.copyfile(paths["parquet"], paths["baseline"])
    return paths


def feature_engineer_with_history_core(
    history_df: pd.DataFrame | None,
    live_processed: pd.DataFrame,
    *,
    extra_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature-engineer live rows using processed historical context."""
    hist = None if history_df is None else history_df.copy()

    if extra_history is not None and not extra_history.empty:
        if hist is not None and not hist.empty:
            hist = pd.concat([hist, extra_history], ignore_index=True, sort=False)
        else:
            hist = extra_history.copy()

    if hist is None or hist.empty:
        logger.warning("No historical processed data found — features will be built from live data only.")
        return engineer_features(live_processed, save=False)

    hist = hist.copy()
    live = live_processed.copy()
    hist["_is_live"] = 0
    live["_is_live"] = 1

    combined = pd.concat([hist, live], ignore_index=True, sort=False)

    no_coerce = {
        "race_id", "horse_name", "jockey", "trainer", "track",
        "race_name", "race_date", "off_time", "region", "form",
        "going", "race_type", "race_class", "surface", "headgear", "sex",
    }
    for col in combined.columns:
        if col.startswith("_") or col in no_coerce:
            continue
        if combined[col].dtype == "object":
            converted = pd.to_numeric(combined[col], errors="coerce")
            if converted.notna().sum() > combined[col].notna().sum() * 0.5:
                combined[col] = converted

    race_level_vars = {"track"}
    for col, freq_col in [
        ("horse_name", "horse_name_freq"),
        ("jockey", "jockey_freq"),
        ("trainer", "trainer_freq"),
        ("track", "track_freq"),
    ]:
        if col in combined.columns:
            combined[freq_col] = combined.groupby(col).cumcount() + 1
            if col in race_level_vars:
                combined[freq_col] = combined.groupby("race_id")[freq_col].transform("min")

    weather_cols = {
        "weather_temp_max", "weather_temp_min", "weather_precip_mm",
        "weather_wind_kmh", "weather_precip_prev3",
    }
    for col in combined.columns:
        if combined[col].dtype in ("float64", "float32", "int64", "int32") and col not in weather_cols:
            combined[col] = combined[col].fillna(0)

    logger.info(
        "Feature engineering with history: %s historical + %s live = %s total rows",
        f"{len(hist):,}",
        f"{len(live):,}",
        f"{len(combined):,}",
    )

    featured = engineer_features(combined, save=False)
    live_featured = featured[featured["_is_live"] == 1].copy()
    return live_featured.drop(columns=["_is_live"], errors="ignore")


# ── Lookahead cache ──────────────────────────────────────────────────

def _lookahead_paths(date_str: str) -> dict[str, str]:
    return {
        "parquet": os.path.join(LOOKAHEAD_CACHE_DIR, f"{date_str}.parquet"),
        "meta": os.path.join(LOOKAHEAD_CACHE_DIR, f"{date_str}.json"),
    }


def save_lookahead_cache(
    date_str: str,
    featured_df: pd.DataFrame,
    cards_sig: str,
) -> None:
    paths = _lookahead_paths(date_str)
    featured_df.to_parquet(paths["parquet"], index=False, engine="pyarrow")
    meta = {
        "date": date_str,
        "cards_sig": cards_sig,
        "rows": len(featured_df),
        "cols": len(featured_df.columns),
        "built": datetime.now().isoformat(),
    }
    with open(paths["meta"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Lookahead cache saved for %s: %d rows", date_str, len(featured_df))


def load_lookahead_cache(
    date_str: str,
    current_cards_sig: str | None = None,
) -> pd.DataFrame | None:
    """Load pre-computed features for *date_str*.

    If *current_cards_sig* is given, only returns cached data when the
    racecard signature matches (runners haven't changed since cache was built).
    """
    paths = _lookahead_paths(date_str)
    if not os.path.exists(paths["parquet"]):
        return None
    if current_cards_sig is not None and os.path.exists(paths["meta"]):
        with open(paths["meta"], encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("cards_sig") != current_cards_sig:
            logger.info(
                "Lookahead cache stale for %s (cards changed)", date_str
            )
            return None
    try:
        return pd.read_parquet(paths["parquet"], engine="pyarrow")
    except Exception:
        return None


def lookahead_cache_valid(date_str: str, current_cards_sig: str | None = None) -> bool:
    """Lightweight check: is there a valid lookahead cache for *date_str*?"""
    paths = _lookahead_paths(date_str)
    if not os.path.exists(paths["parquet"]):
        return False
    if current_cards_sig is not None and os.path.exists(paths["meta"]):
        with open(paths["meta"], encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("cards_sig") != current_cards_sig:
            return False
    return True


def clear_lookahead_cache() -> int:
    """Remove all lookahead cache files.  Returns count of files deleted."""
    removed = 0
    if os.path.isdir(LOOKAHEAD_CACHE_DIR):
        for f in os.listdir(LOOKAHEAD_CACHE_DIR):
            fp = os.path.join(LOOKAHEAD_CACHE_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)
                removed += 1
    return removed


def build_lookahead_cache(
    processed_history: pd.DataFrame,
    days_ahead: int = 7,
    progress_fn=None,
) -> dict[str, int]:
    """Scrape racecards for the next *days_ahead* days and pre-compute features.

    Returns ``{date_str: num_rows}`` for each date that was cached.
    """
    from src.data_collector import get_scraped_racecards
    from src.data_processor import process_data

    today = datetime.now().date()
    results: dict[str, int] = {}

    for offset in range(1, days_ahead + 1):
        target = today + timedelta(days=offset)
        target_str = target.strftime("%Y-%m-%d")

        if progress_fn:
            progress_fn(offset, days_ahead, target_str)

        try:
            cards = get_scraped_racecards(date_str=target_str)
        except Exception as exc:
            logger.warning("Lookahead: failed to scrape racecards for %s: %s", target_str, exc)
            continue
        if cards is None or cards.empty:
            logger.info("Lookahead: no racecards for %s", target_str)
            continue

        # Prepare racecard rows (pre-race placeholders)
        cards["won"] = 0
        cards["finish_position"] = 0
        cards["finish_time_secs"] = 0.0
        cards["lengths_behind"] = float('nan')

        try:
            proc = process_data(df=cards, save=False)
        except Exception as exc:
            logger.warning("Lookahead: process_data failed for %s: %s", target_str, exc)
            continue

        try:
            featured = feature_engineer_with_history_core(
                processed_history, proc,
            )
        except Exception as exc:
            logger.warning("Lookahead: FE failed for %s: %s", target_str, exc)
            continue

        sig = cards_signature(cards)
        save_lookahead_cache(target_str, featured, sig)
        results[target_str] = len(featured)

    return results