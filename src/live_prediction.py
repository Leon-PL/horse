from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import date, datetime, timedelta

import pandas as pd

import config
from src.feature_engineer import engineer_features
from src.source_dataset import live_dataset

logger = logging.getLogger(__name__)


LIVE_FEATURE_CACHE_DIR = os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache")
LIVE_FEATURE_BASELINE_DIR = os.path.join(LIVE_FEATURE_CACHE_DIR, "baseline")
LOOKAHEAD_CACHE_DIR = os.path.join(config.PROCESSED_DATA_DIR, "lookahead_cache")
for _dir in (LIVE_FEATURE_CACHE_DIR, LIVE_FEATURE_BASELINE_DIR, LOOKAHEAD_CACHE_DIR):
    os.makedirs(_dir, exist_ok=True)


def _serving_input_signature() -> str:
    """Invalidate live vectors when their preparation changes, independently of FE."""
    with open(__file__, "rb") as stream:
        source = stream.read().replace(b"\r\n", b"\n")
    from src.prize_currency_features import reference_signature
    return hashlib.sha256(source + reference_signature().encode()).hexdigest()[:20]


def stable_card_indicators(processed: pd.DataFrame) -> pd.DataFrame:
    """Keep the training indicator meanings on cards containing one category.

    Completing these inputs before FE also preserves derived jumps/draw
    interactions. Filling absent indicator columns after FE is too late.
    """
    result = processed.copy()
    categories = {"race_type": ("Flat", "Hurdle", "Chase"), "sex": ("Male", "Female")}
    for name, labels in categories.items():
        if name in result:
            for label in labels:
                result[f"{name}_{label}"] = result[name].eq(label).fillna(False).astype("int8")
    return result


def history_source_signature(
    history_df: pd.DataFrame | None,
    *,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> str:
    payload: dict[str, object] = {
        "serving_inputs": _serving_input_signature(),
        "version": int(config.LIVE_FEATURE_CACHE_VERSION),
        "history_months": int(config.LIVE_FE_HISTORY_MONTHS),
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


def feature_engineer_with_history_core(
    history_df: pd.DataFrame | None,
    live_processed: pd.DataFrame,
    *,
    extra_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature-engineer live rows using processed historical context."""
    from src.result_status import pending_card
    live_processed = stable_card_indicators(pending_card(live_processed))
    history_frames: list[pd.DataFrame] = []
    if history_df is not None and not history_df.empty:
        history_frames.append(history_df)
    if extra_history is not None and not extra_history.empty:
        history_frames.append(extra_history)

    if not history_frames:
        logger.warning("No historical processed data found — features will be built from live data only.")
        return engineer_features(live_processed, save=False)

    history_rows = sum(len(frame) for frame in history_frames)
    live_rows = len(live_processed)
    combined = pd.concat([*history_frames, live_processed], ignore_index=True, sort=False, copy=False)
    combined["_is_live"] = 0
    if live_rows > 0:
        combined.iloc[-live_rows:, combined.columns.get_loc("_is_live")] = 1

    no_coerce = {
        "race_id", "horse_name", "jockey", "trainer", "track",
        "race_name", "race_date", "off_time", "region", "form",
        "going", "race_type", "race_class", "surface", "headgear", "sex",
    }
    object_cols = [
        col for col in combined.select_dtypes(include=["object"]).columns
        if not col.startswith("_") and col not in no_coerce
    ]
    for col in object_cols:
        non_null = combined[col].notna().sum()
        if non_null == 0:
            continue
        converted = pd.to_numeric(combined[col], errors="coerce")
        if converted.notna().sum() > non_null * 0.5:
            combined[col] = converted

    # Mirror data_processor.encode_categorical EXACTLY (training/serving parity):
    # collapse each freq per (race, entity) min so a multi-runner trainer's
    # stablemates share one pre-race count instead of leaking finish order via
    # sequential cumcounts. race-level vars collapse per race.
    race_level_vars = {"track"}
    for col, freq_col in [
        ("horse_name", "horse_name_freq"),
        ("jockey", "jockey_freq"),
        ("trainer", "trainer_freq"),
        ("track", "track_freq"),
    ]:
        if col in combined.columns:
            combined[freq_col] = combined.groupby(col).cumcount() + 1
            if "race_id" in combined.columns:
                grp = "race_id" if col in race_level_vars else ["race_id", col]
                combined[freq_col] = combined.groupby(grp)[freq_col].transform("min")

    # Unknown measurements retain the missing-value branches learned during
    # training. Only structural one-hot absences have a zero interpretation.
    # add_time_features drops its temporary season label after encoding it.
    # Include that explicit domain even when concatenation cannot recover it.
    supported = {"race_type": ("Flat", "Hurdle", "Chase"), "sex": ("Male", "Female"),
                 "season": ("Winter", "Spring", "Summer", "Autumn")}
    for category in ("going", "race_class", "race_type", "region", "sex", "surface", "season"):
        labels = set(supported.get(category, ()))
        if category in combined:
            labels.update(combined[category].dropna().unique())
        # Match actual encoded category names. A numeric measurement such as
        # going_numeric is not a dummy merely because its observed range is0/1.
        for label in labels:
            column = f"{category}_{label}"
            if column in combined and pd.api.types.is_numeric_dtype(combined[column]):
                observed = combined[column].dropna()
                if observed.isin([0, 1]).all():
                    combined[column] = combined[column].fillna(0)

    logger.info(
        "Feature engineering with history: %s historical + %s live = %s total rows",
        f"{history_rows:,}",
        f"{live_rows:,}",
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


def feature_store_signature(store=None) -> str:
    """Tie live features to both the persisted history and current FE implementation."""
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature

    store = store or FeatureStore()
    parts = {"fe_signature": fe_signature(), "meta": store.meta(),
             "serving_inputs": _serving_input_signature()}
    from src.feature_inputs import rtv_cache_signature
    parts["rtv_inputs"] = rtv_cache_signature()
    for name in ("featured_path", "processed_path", "state_path", "counts_state_path"):
        path = getattr(store, name)
        try:
            stat = os.stat(path)
            parts[name] = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            parts[name] = None
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def save_lookahead_cache(
    date_str: str,
    featured_df: pd.DataFrame,
    cards_sig: str,
    *,
    history_sig: str | None = None,
) -> None:
    paths = _lookahead_paths(date_str)
    # Publish complete files and bind metadata to this exact parquet generation.
    fd, temporary = tempfile.mkstemp(dir=LOOKAHEAD_CACHE_DIR, suffix=".parquet")
    os.close(fd)
    try:
        featured_df.to_parquet(temporary, index=False, engine="pyarrow")
        artifact = os.stat(temporary)
        os.replace(temporary, paths["parquet"])
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    meta = {
        "date": date_str,
        "artifact": [artifact.st_mtime_ns, artifact.st_size],
        "cards_sig": cards_sig,
        "history_sig": history_sig or feature_store_signature(),
        "version": int(config.LIVE_FEATURE_CACHE_VERSION),
        "rows": len(featured_df),
        "cols": len(featured_df.columns),
        "built": datetime.now().isoformat(),
    }
    fd, temporary = tempfile.mkstemp(dir=LOOKAHEAD_CACHE_DIR, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        os.replace(temporary, paths["meta"])
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    logger.info("Lookahead cache saved for %s: %d rows", date_str, len(featured_df))


def lookahead_cache_valid(
    date_str: str,
    current_cards_sig: str | None = None,
    *,
    history_sig: str | None = None,
) -> bool:
    """Reject incomplete, outdated or mismatched feature caches."""
    paths = _lookahead_paths(date_str)
    if not os.path.isfile(paths["parquet"]) or not os.path.isfile(paths["meta"]):
        return False
    try:
        with open(paths["meta"], encoding="utf-8") as fh:
            meta = json.load(fh)
        if int(meta.get("version", -1)) != int(config.LIVE_FEATURE_CACHE_VERSION):
            return False
        if meta.get("date") != date_str:
            return False
        artifact = os.stat(paths["parquet"])
        if meta.get("artifact") != [artifact.st_mtime_ns, artifact.st_size]:
            return False
        if meta.get("history_sig") != (history_sig or feature_store_signature()):
            return False
        if current_cards_sig is not None and meta.get("cards_sig") != current_cards_sig:
            return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


def load_lookahead_cache(
    date_str: str,
    current_cards_sig: str | None = None,
    *,
    history_sig: str | None = None,
) -> pd.DataFrame | None:
    """Load only a complete cache matching the current feature implementation/history."""
    if not lookahead_cache_valid(date_str, current_cards_sig, history_sig=history_sig):
        return None
    try:
        path = _lookahead_paths(date_str)["parquet"]
        before = os.stat(path)
        frame = pd.read_parquet(path, engine="pyarrow")
        after = os.stat(path)
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            return None
        if not lookahead_cache_valid(date_str, current_cards_sig, history_sig=history_sig):
            return None
        return frame
    except Exception:
        logger.warning("Unreadable lookahead cache for %s", date_str, exc_info=True)
        return None


def _match_store_id_types(card: pd.DataFrame) -> pd.DataFrame:
    """Give ``horse_id`` the integer form the store's carried history uses.

    The store's rating and counts state is keyed by integer horse ids (history
    reaches it through a CSV round trip). A card holding them as text - a
    fresh scrape, a hand-built what-if race - matched none of it, so 25 of the
    149 model features lost every horse's history without any error
    (2026-10-13). Ids that are not all numeric are left alone.
    """
    if "horse_id" not in card.columns:
        return card
    ids = card["horse_id"]
    numeric = pd.to_numeric(ids, errors="coerce")
    if numeric.notna().sum() == ids.notna().sum() and numeric.notna().all():
        card = card.copy()
        card["horse_id"] = numeric.astype("int64")
    return card


#: Longest routine gap between UK/IRE race days (22 Dec -> 26 Dec). History
#: ending further before a card than this means results are missing.
MAX_HISTORY_GAP_DAYS = 4


def _results_db_latest() -> pd.Timestamp | None:
    """Latest settled race date in the results database, or None if absent."""
    from src import database

    if not os.path.exists(database.DB_PATH):
        return None
    try:
        latest = database.get_latest_date()
    except Exception:  # pragma: no cover - unreadable DB is reported elsewhere
        logger.warning("Could not read the results database's latest date", exc_info=True)
        return None
    return pd.Timestamp(latest) if latest else None


def _require_current_history(store_last, date_str: str) -> None:
    """Refuse to price a card against history that is missing recent races.

    The serving app never refreshes the store itself. A store a week behind
    priced every horse that ran that week on stale form - days since its last
    run, its latest result, its rating - and nothing said so.
    """
    if store_last is None:
        return
    store_last = pd.Timestamp(store_last).normalize()
    db_last = _results_db_latest()
    if db_last is not None and db_last.normalize() > store_last:
        raise RuntimeError(
            f"The feature store ends {store_last:%Y-%m-%d} but the results database holds races "
            f"to {db_last.normalize():%Y-%m-%d}. Refresh the store on the training app's Data page "
            "first, or every horse that ran since is priced on stale form."
        )
    newest = max(store_last, db_last.normalize()) if db_last is not None else store_last
    gap = (pd.Timestamp(date_str).normalize() - newest).days
    if gap > MAX_HISTORY_GAP_DAYS:
        raise RuntimeError(
            f"Results end {newest:%Y-%m-%d}, {gap} days before {date_str}. Sync the database "
            "(python -m src.ingestion --sync 14) and refresh the store before pricing live cards."
        )


@live_dataset
def features_for_live_cards(
    cards: pd.DataFrame, date_str: str, *, store=None, use_cache: bool = True,
) -> pd.DataFrame:
    """Reuse an exact cache or rebuild this card using the bounded, read-only store path.

    Any runner, odds, going or other card change triggers the complete card FE
    pass. Patching odds onto old features leaves race-relative predictors stale.
    """
    from src.data_processor import process_data
    from src.feature_store import FeatureStore
    from src.featured_dataset import fe_signature

    store = store or FeatureStore()
    if not store.exists():
        raise RuntimeError("Build the feature store on the training app's Data page first.")
    stored_fe_sig = store.meta().get("fe_signature")
    if stored_fe_sig is None:
        # Used to skip the check entirely: an unsigned store was served unverified.
        raise RuntimeError("The feature store carries no feature signature; rebuild it on the training app's Data page.")
    if stored_fe_sig != fe_signature():
        raise RuntimeError("Feature code has changed; rebuild the feature store on the training app's Data page.")
    last = store.last_date()
    if last is not None and pd.Timestamp(date_str).normalize() <= pd.Timestamp(last).normalize():
        raise ValueError("Live prediction requires a race date after the stored results; use saved evaluations for historical races.")
    _require_current_history(last, date_str)
    from src.prize_currency_features import prepare_prize_references
    prepare_prize_references(cards)
    sig = cards_signature(cards)
    history_sig = feature_store_signature(store)
    if use_cache:
        cached = load_lookahead_cache(date_str, sig, history_sig=history_sig)
        if cached is not None and not cached.empty:
            return cached
    card = cards.copy()
    card = _match_store_id_types(card)
    from src.result_status import pending_card
    card = pending_card(card)
    processed = stable_card_indicators(process_data(df=card))
    featured = store.featurize_live(processed)
    # Never publish features from a mixture of concurrent history generations.
    if feature_store_signature(store) != history_sig:
        raise RuntimeError("The feature store changed while building picks; refresh once its update completes.")
    if use_cache:
        save_lookahead_cache(date_str, featured, sig, history_sig=history_sig)
    return featured


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
