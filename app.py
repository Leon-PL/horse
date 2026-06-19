"""
Streamlit Web Application — v3.0
================================
Interactive web interface for the Horse Racing Prediction system.

Features:
- Train models with hyperparameter tuning & presets
- Experiment tracking and comparison
- Predictions for upcoming races
- Walk-forward backtesting
- Data exploration and model insights

Run with:
    streamlit run app.py
"""

import hashlib
import json
import os

# Limit numpy/BLAS threads to prevent ReleaseSemaphore crashes in Streamlit threadpool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_collector import collect_data, get_scraped_racecards, scrape_todays_results
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.live_prediction import (
    build_lookahead_cache,
    cards_signature,
    clear_lookahead_cache,
    feature_engineer_with_history_core,
    gap_fill_signature,
    history_source_signature,
    load_lookahead_cache,
    lookahead_cache_valid,
    save_lookahead_cache,
)
from src.database import db_stats as _raw_db_stats
from src.model import (
    RacePredictor,
    get_autotune_search_space,
    get_feature_importance,
    get_feature_columns,
)
from src.autotune import (
    AUTOTUNE_MODEL_INFO,
    build_autotune_dataset,
    build_config_snippet,
    create_autotune_session,
    delete_autotune_session,
    list_autotune_sessions,
    load_autotune_session,
    load_optuna_study,
    run_autotune_session,
)
from src.rtv_scraper import (
    RTV_METRIC_COLS,
    RTV_RANK_COLS,
    _normalise_horse_key,
    _normalise_off_time_key,
    _normalise_track_key,
    backfill_rtv_metrics_for_races,
    load_rtv_cache,
)
from src.backtester import walk_forward_validation
from src.run_store import save_run, list_runs as _raw_list_runs, load_run as _raw_load_run, load_run_meta as _raw_load_run_meta, delete_run as _raw_delete_run, rename_run, get_latest_run_id, restore_run_model, run_has_model, get_run_processed_path, get_run_featured_path, prune_runs, run_disk_usage
from src.utils import format_odds, kelly_criterion, compact_numeric_dtypes as _compact_numeric_dtypes
from src.bet_settlement import dynamic_value_threshold, ew_placed_flag, settle_ew_bet, settle_win_bet
from src.each_way import compute_ew_columns, ew_value_bets, get_ew_terms, kelly_ew, EachWayTerms
from src.matchbook_client import MatchbookAPIError, MatchbookClient
from src.matchbook_signals import build_fake_prediction_frame, build_signal_frame
from src.paper_trade_store import append_paper_trades, build_paper_trades_from_signals, load_paper_trades, save_paper_trades, settle_paper_trades

logger = logging.getLogger(__name__)


from src.app_helpers import (  # noqa: F401
    EXPERIMENTS_FILE,
    _PACE_DISPLAY_COLUMNS,
    _PACE_REQUIRED_COLUMNS,
    _add_shortcomings_bands,
    _attach_pace_diagnostics,
    _attach_ranker_diagnostics,
    _bet_confidence_state,
    _build_concentration_charts,
    _build_generalization_frame,
    _build_generalization_trend_chart,
    _build_metric_snapshot_frame,
    _build_model_tradeoff_frame,
    _build_overfit_section_charts,
    _build_ranker_selector_explainer,
    _build_run_name,
    _build_shortcomings_correlation_table,
    _build_shortcomings_fold_frame,
    _build_shortcomings_run_frame,
    _build_trend_chart,
    _cached_db_stats,
    _cached_load_df,
    _calibration_metric_cards,
    _calibration_signature,
    _first_metric_value,
    _flatten_numeric_metrics,
    _fmt_metric,
    _format_duration_compact,
    _gap_status,
    _has_pace_diagnostics,
    _invalidate_run_caches,
    _is_diagnostic_metric,
    _load_experiments,
    _log_experiment,
    _metric_direction,
    _metric_family,
    _metric_label,
    _model_display_name,
    _ordinal,
    _pace_race_summary,
    _pace_runner_tags,
    _prepare_shortcomings_run_frames,
    _progress_timing_text,
    _ranker_consensus_state,
    _render_pace_panel,
    _render_ranker_consensus_badge,
    _render_ranker_disagreement_panel,
    _render_shap_explanation,
    _resolve_model_metric_payload,
    _save_experiments,
    _summarise_shortcomings_slice,
    _value_bet_mask,
    _value_odds_range,
    delete_run,
    list_runs,
    load_run,
    load_run_meta,
)



# ── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="🏇 Horse Race Predictor",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stMetric {background:#1e1e2e; padding:12px; border-radius:10px;}
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stExpander"] details summary p {font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ────────────────────────────────────────────────────
_DEFAULTS = {
    "predictor": None,
    "featured_data": None,
    "train_processed_data": None,
    "train_dataset_meta": None,
    "autotune_featured_data": None,
    "autotune_processed_data": None,
    "autotune_dataset_meta": None,
    "model_featured_data": None,
    "model_dataset_meta": None,
    "metrics": None,
    "bt_report": None,
    "test_analysis": None,
    "active_run_id": None,
    "value_config": {
        "staking_mode": "flat",
        "value_threshold": 0.05,
        "value_min_odds": 1.0,
        "value_max_odds": 101.0,
        "kelly_fraction": 0.25,
        "bankroll": 100.0,
        "ew_enabled": True,
        "ew_fraction": 0.20,
        "ew_min_place_edge": 0.15,
        "ew_min_odds": 4.0,
        "ew_max_odds": 51.0,
    },
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Stale-cache housekeeping (once per session) ──────────────────────
if "did_cache_cleanup" not in st.session_state:
    from src.utils import cleanup_stale_caches

    try:
        cleanup_stale_caches()
    except Exception as _exc:
        logger.warning("Cache cleanup failed: %s", _exc)
    st.session_state["did_cache_cleanup"] = True


# ── Helpers ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def _cached_load_model():
    """Load model from disk — cached so joblib deserialization only happens once."""
    try:
        p = RacePredictor()
        p.load()
        return p
    except FileNotFoundError:
        return None


def load_existing_model():
    p = _cached_load_model()
    if p is not None:
        st.session_state.predictor = p
        return True
    return False




def _cached_load_history_context(path: str, _mtime: float) -> tuple[pd.DataFrame, object]:
    """Load processed history and its last available race date.

    Not cached itself: _cached_load_df already caches the heavy read
    (an extra cache_data layer here used to deep-copy the whole history
    frame on every call), and the max-date scan is milliseconds.
    """
    hist = _cached_load_df(path, _mtime)
    if hist is None or hist.empty or "race_date" not in hist.columns:
        return hist, None

    race_dates = hist["race_date"]
    if not pd.api.types.is_datetime64_any_dtype(race_dates):
        race_dates = pd.to_datetime(race_dates, errors="coerce")
    last_hist = race_dates.max()
    return hist, last_hist.date() if pd.notna(last_hist) else None


@st.cache_resource(show_spinner="Rebuilding featured dataset for active run …")
def _cached_build_featured_from_processed(path: str, _mtime: float) -> pd.DataFrame:
    processed = _cached_load_df(path, _mtime)
    return engineer_features(processed.copy(), save=False)


def _predictor_runtime_token(predictor: RacePredictor | None) -> str:
    if predictor is None:
        return "no-predictor"
    model_path = os.path.join(config.MODELS_DIR, "triple_ensemble_models.joblib")
    model_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
    return f"{id(predictor)}:{model_mtime:.6f}"


def _frame_signature(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "empty"
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha1(hashed).hexdigest()[:20]


def _prediction_cache_get(cache_key: tuple) -> pd.DataFrame | None:
    cache = st.session_state.setdefault("_prediction_result_cache", {})
    cached = cache.get(cache_key)
    return cached.copy() if isinstance(cached, pd.DataFrame) else None


def _prediction_cache_set(cache_key: tuple, value: pd.DataFrame) -> pd.DataFrame:
    cache = st.session_state.setdefault("_prediction_result_cache", {})
    order = st.session_state.setdefault("_prediction_result_cache_order", [])
    cache[cache_key] = value.copy()
    if cache_key in order:
        order.remove(cache_key)
    order.append(cache_key)
    while len(order) > 4:
        old_key = order.pop(0)
        cache.pop(old_key, None)
    return value


def _predict_featured_frame(
    predictor: RacePredictor,
    feature_df: pd.DataFrame,
    *,
    ew_fraction: float | None = None,
) -> pd.DataFrame:
    cache_key = (
        _predictor_runtime_token(predictor),
        round(float(ew_fraction), 6) if ew_fraction is not None else None,
        _frame_signature(feature_df),
    )
    cached = _prediction_cache_get(cache_key)
    if cached is not None:
        return cached

    if hasattr(predictor, "predict_races") and "race_id" in feature_df.columns:
        preds = predictor.predict_races(feature_df, ew_fraction=ew_fraction)
        preds = _attach_pace_diagnostics(preds, feature_df)
        preds = _attach_ranker_diagnostics(preds, predictor, feature_df)
        return _prediction_cache_set(cache_key, preds)

    if "race_id" not in feature_df.columns:
        preds = predictor.predict_race(feature_df, ew_fraction=ew_fraction)
        preds = _attach_pace_diagnostics(preds, feature_df)
        preds = _attach_ranker_diagnostics(preds, predictor, feature_df)
        return _prediction_cache_set(cache_key, preds)

    all_preds: list[pd.DataFrame] = []
    for race_id in feature_df["race_id"].unique() if "race_id" in feature_df.columns else []:
        feat_slice = feature_df[feature_df["race_id"] == race_id].copy()
        if feat_slice.empty:
            continue
        preds = predictor.predict_race(feat_slice, ew_fraction=ew_fraction)
        preds = _attach_pace_diagnostics(preds, feat_slice)
        preds = _attach_ranker_diagnostics(preds, predictor, feat_slice)
        preds["race_id"] = race_id
        all_preds.append(preds)

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return _prediction_cache_set(cache_key, combined)


def _global_featured_dataset_path() -> str | None:
    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.csv")
    if os.path.exists(pq_path):
        return pq_path
    if os.path.exists(csv_path):
        return csv_path
    # Fall back to the most recently modified cached featured dataset
    cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "cache")
    if os.path.isdir(cache_dir):
        candidates = sorted(
            (f for f in os.scandir(cache_dir) if f.name.startswith("featured_") and f.name.endswith(".parquet")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].path
    return None


def _global_processed_dataset_path() -> str | None:
    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.csv")
    if os.path.exists(pq_path):
        return pq_path
    if os.path.exists(csv_path):
        return csv_path
    return None


def _fmt_age_from_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    age_seconds = max(time.time() - float(ts), 0.0)
    if age_seconds < 60:
        return f"{int(age_seconds)}s ago"
    if age_seconds < 3600:
        return f"{int(age_seconds // 60)}m ago"
    if age_seconds < 86400:
        return f"{int(age_seconds // 3600)}h ago"
    return f"{int(age_seconds // 86400)}d ago"


def _dir_health(path: str, *, suffix: str | None = None) -> dict[str, object]:
    if not os.path.isdir(path):
        return {"exists": False, "count": 0, "latest_mtime": None}
    files = [
        entry.path for entry in os.scandir(path)
        if entry.is_file() and (suffix is None or entry.name.endswith(suffix))
    ]
    latest_mtime = max((os.path.getmtime(fp) for fp in files), default=None)
    return {
        "exists": True,
        "count": len(files),
        "latest_mtime": latest_mtime,
    }


def _dataframe_missing_pct(df: pd.DataFrame | None, col: str) -> float | None:
    if df is None or df.empty or col not in df.columns:
        return None
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        missing = series.isna()
    else:
        missing = series.isna() | series.astype(str).str.strip().eq("")
    return float(missing.mean() * 100.0)


@st.cache_data(ttl=60, show_spinner=False)
def _build_prediction_data_health(featured_path: str | None, featured_mtime: float | None) -> dict:
    db = _cached_db_stats()
    processed_path = _global_processed_dataset_path()
    processed_mtime = os.path.getmtime(processed_path) if processed_path and os.path.exists(processed_path) else None
    racecards_cache = _dir_health(os.path.join(config.DATA_DIR, "racecards_cache"), suffix=".csv")
    live_feature_cache = _dir_health(os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache"))
    lookahead_cache = _dir_health(os.path.join(config.PROCESSED_DATA_DIR, "lookahead_cache"))
    return {
        "db": db,
        "featured_path": featured_path,
        "featured_mtime": featured_mtime,
        "processed_path": processed_path,
        "processed_mtime": processed_mtime,
        "racecards_cache": racecards_cache,
        "live_feature_cache": live_feature_cache,
        "lookahead_cache": lookahead_cache,
    }


def _render_prediction_data_health(model_df: pd.DataFrame | None) -> None:
    featured_path = _global_featured_dataset_path()
    featured_mtime = os.path.getmtime(featured_path) if featured_path and os.path.exists(featured_path) else None
    health = _build_prediction_data_health(featured_path, featured_mtime)
    db_latest = (health.get("db") or {}).get("latest_date")
    db_latest_delta = "—"
    if db_latest:
        try:
            db_latest_dt = pd.to_datetime(db_latest, errors="coerce")
            if pd.notna(db_latest_dt):
                db_latest_delta = f"{max((datetime.now().date() - db_latest_dt.date()).days, 0)}d lag"
        except Exception:
            db_latest_delta = "—"

    latest_race_date = None
    if model_df is not None and not model_df.empty and "race_date" in model_df.columns:
        race_dates = pd.to_datetime(model_df["race_date"], errors="coerce")
        if race_dates.notna().any():
            latest_race_date = race_dates.max()

    jockey_missing = _dataframe_missing_pct(model_df, "jockey")
    trainer_missing = _dataframe_missing_pct(model_df, "trainer")
    odds_missing = _dataframe_missing_pct(model_df, "odds")
    required_cols = ["race_id", "horse_name", "race_date"]
    missing_required = [col for col in required_cols if model_df is None or col not in model_df.columns]

    with st.expander("🩺 Data Health", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "DB Latest",
            str(db_latest or "—"),
            db_latest_delta,
        )
        c2.metric(
            "Featured Cache",
            "Present" if health.get("featured_path") else "Missing",
            _fmt_age_from_ts(health.get("featured_mtime")),
        )
        c3.metric(
            "Racecards Cache",
            str((health.get("racecards_cache") or {}).get("count", 0)),
            _fmt_age_from_ts((health.get("racecards_cache") or {}).get("latest_mtime")),
        )
        c4.metric(
            "Lookahead Cache",
            str((health.get("lookahead_cache") or {}).get("count", 0)),
            _fmt_age_from_ts((health.get("lookahead_cache") or {}).get("latest_mtime")),
        )

        if latest_race_date is not None:
            st.caption(f"Loaded featured data latest race date: {latest_race_date.date().isoformat()}")
        if missing_required:
            st.warning(f"Missing required prediction columns: {', '.join(missing_required)}")

        issues: list[str] = []
        if jockey_missing is not None and jockey_missing > 5.0:
            issues.append(f"Jockey missing: {jockey_missing:.1f}%")
        if trainer_missing is not None and trainer_missing > 5.0:
            issues.append(f"Trainer missing: {trainer_missing:.1f}%")
        if odds_missing is not None and odds_missing > 10.0:
            issues.append(f"Odds missing: {odds_missing:.1f}%")
        if not health.get("featured_path"):
            issues.append("No featured dataset found on disk")
        if not (health.get("racecards_cache") or {}).get("exists"):
            issues.append("Racecards cache directory missing")

        if issues:
            for issue in issues:
                st.caption(f"• {issue}")
        else:
            st.caption("No obvious data-health issues detected in the current prediction state.")


def _dataset_cache_paths(data_source: str, days_back: int | None) -> dict[str, str | None]:
    cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    if data_source == "sample" or days_back is None:
        return {"cache_key": None, "featured": None, "processed": None}
    cache_key = f"{data_source}_{int(days_back)}d"
    return {
        "cache_key": cache_key,
        "featured": os.path.join(cache_dir, f"featured_{cache_key}.parquet"),
        "processed": os.path.join(cache_dir, f"processed_{cache_key}.parquet"),
    }


def _dataset_meta_from_frame(
    df: pd.DataFrame | None,
    *,
    data_source: str | None = None,
    requested_days: int | None = None,
    cache_key: str | None = None,
    featured_path: str | None = None,
    processed_path: str | None = None,
    origin: str | None = None,
) -> dict[str, object]:
    dates = pd.to_datetime(df["race_date"], errors="coerce") if isinstance(df, pd.DataFrame) and "race_date" in df.columns else pd.Series(dtype="datetime64[ns]")
    date_min = dates.min() if not dates.empty else pd.NaT
    date_max = dates.max() if not dates.empty else pd.NaT
    span_days = None
    months = None
    if pd.notna(date_min) and pd.notna(date_max):
        span_days = int((date_max - date_min).days) + 1
        months = int(dates.dt.to_period("M").nunique())
    actual_days = span_days if span_days is not None else (int(requested_days) if requested_days is not None else None)
    return {
        "data_source": data_source,
        "requested_days": int(requested_days) if requested_days is not None else None,
        "actual_days": actual_days,
        "date_start": date_min.date().isoformat() if pd.notna(date_min) else None,
        "date_end": date_max.date().isoformat() if pd.notna(date_max) else None,
        "months": months,
        "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "cols": int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0,
        "cache_key": cache_key,
        "featured_path": featured_path,
        "processed_path": processed_path,
        "origin": origin,
    }


def _set_training_dataset(
    featured_df: pd.DataFrame | None,
    *,
    processed_df: pd.DataFrame | None = None,
    dataset_meta: dict[str, object] | None = None,
) -> None:
    featured_df = _compact_numeric_dtypes(featured_df, label="training featured")
    processed_df = _compact_numeric_dtypes(processed_df, label="training processed")
    st.session_state.featured_data = featured_df
    st.session_state.train_processed_data = processed_df
    st.session_state.train_dataset_meta = dataset_meta or None


def _build_rtv_missing_diagnostics(processed_df: pd.DataFrame) -> dict[str, object]:
    """Return RTV coverage diagnostics for the current processed dataset."""
    required = {"race_date", "track", "off_time", "horse_name"}
    if processed_df is None or processed_df.empty or not required.issubset(processed_df.columns):
        return {
            "ok": False,
            "message": "Processed dataset is missing required key columns for RTV diagnostics.",
        }

    cache = load_rtv_cache()
    if cache is None or cache.empty:
        return {
            "ok": False,
            "message": "RTV cache is empty. Run backfill first.",
        }

    df = processed_df.copy()
    c = cache.copy()

    df["_rd"] = pd.to_datetime(df["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["_trk"] = df["track"].map(_normalise_track_key)
    df["_ot"] = df["off_time"].map(_normalise_off_time_key)
    df["_hn"] = df["horse_name"].map(_normalise_horse_key)

    c["_rd"] = pd.to_datetime(c["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    c["_trk"] = c["track"].map(_normalise_track_key)
    c["_ot"] = c["off_time"].map(_normalise_off_time_key)
    c["_hn"] = c["horse_name"].map(_normalise_horse_key)

    key_cols = ["_rd", "_trk", "_ot", "_hn"]
    metric_cols = [col for col in RTV_METRIC_COLS if col in c.columns]
    c = c[key_cols + metric_cols].drop_duplicates(key_cols, keep="last")

    m = df.merge(c, on=key_cols, how="left", indicator=True)
    has_any_metric = m[metric_cols].notna().any(axis=1)

    m["reason"] = "key_match"
    m.loc[m["_merge"] == "left_only", "reason"] = "no_key_match"
    m.loc[(m["_merge"] == "both") & (~has_any_metric), "reason"] = "key_match_but_no_metrics"

    # Race-level summaries for manual review
    race_group_cols = ["_rd", "track", "off_time"]
    no_key_races = (
        m[m["reason"] == "no_key_match"]
        .groupby(race_group_cols, dropna=False)
        .size()
        .reset_index(name="missing_runners")
        .sort_values(["missing_runners", "_rd"], ascending=[False, False])
    )
    no_metric_races = (
        m[m["reason"] == "key_match_but_no_metrics"]
        .groupby(race_group_cols, dropna=False)
        .size()
        .reset_index(name="runners_without_metrics")
        .sort_values(["runners_without_metrics", "_rd"], ascending=[False, False])
    )

    reason_pct = (m["reason"].value_counts(normalize=True) * 100).round(2)
    by_track_no_key = (
        m[m["reason"] == "no_key_match"]["track"]
        .value_counts()
        .reset_index(name="rows")
        .rename(columns={"index": "track"})
        .head(20)
    )

    out = {
        "ok": True,
        "rows": int(len(m)),
        "key_match_pct": round(float((m["_merge"] == "both").mean() * 100), 2),
        "any_metric_pct": round(float(has_any_metric.mean() * 100), 2),
        "no_key_pct": round(float(reason_pct.get("no_key_match", 0.0)), 2),
        "no_metrics_pct": round(float(reason_pct.get("key_match_but_no_metrics", 0.0)), 2),
        "no_key_races": no_key_races,
        "no_metric_races": no_metric_races,
        "top_no_key_tracks": by_track_no_key,
        "no_key_rows": m[m["reason"] == "no_key_match"][
            ["race_date", "track", "off_time", "horse_name"]
        ].copy(),
        "no_metric_rows": m[m["reason"] == "key_match_but_no_metrics"][
            ["race_date", "track", "off_time", "horse_name"]
        ].copy(),
    }

    if "race_type" in m.columns:
        by_rt = (
            m.assign(no_key=m["reason"].eq("no_key_match"))
            .groupby("race_type", dropna=False)["no_key"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index(name="no_key_match_pct")
            .sort_values("no_key_match_pct", ascending=False)
        )
        out["by_race_type_no_key"] = by_rt

    return out


def _drop_degenerate_races_pre_fe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop degenerate historical races before feature engineering.

    Mirrors the model-side race quality filter, but applies earlier so the
    feature pipeline does less work on known-bad races.
    Future rows (``_is_future == 1``) are preserved.
    """
    if df is None or df.empty or "race_id" not in df.columns:
        return df, {"removed_rows": 0, "removed_races": 0}

    is_future = df.get("_is_future", pd.Series(0, index=df.index)).fillna(0).astype(int)
    hist = df[is_future == 0].copy()
    fut = df[is_future == 1].copy()

    if hist.empty:
        return df, {"removed_rows": 0, "removed_races": 0}

    # Evaluate degeneracy only on rows with valid finishing positions.
    work = hist[hist["finish_position"].notna() & (hist["finish_position"] > 0)].copy()
    if work.empty:
        combined = pd.concat([hist, fut], ignore_index=True, sort=False)
        return combined, {"removed_rows": 0, "removed_races": 0}

    single_runner_ids = pd.Index([])
    if "num_runners" in work.columns:
        single_runner_ids = work.groupby("race_id")["num_runners"].first()
        single_runner_ids = single_runner_ids[single_runner_ids <= 1].index

    pos_spread = work.groupby("race_id")["finish_position"].nunique()
    identical_pos_ids = pos_spread[pos_spread <= 1].index

    bad_odds_ids = pd.Index([])
    if "odds" in work.columns:
        bad_odds_ids = work.loc[work["odds"] < 1.01, "race_id"].unique()

    degenerate_ids = set(single_runner_ids) | set(identical_pos_ids) | set(bad_odds_ids)
    if not degenerate_ids:
        combined = pd.concat([hist, fut], ignore_index=True, sort=False)
        return combined, {"removed_rows": 0, "removed_races": 0}

    before_rows = len(hist)
    hist = hist[~hist["race_id"].isin(degenerate_ids)].copy()
    removed_rows = before_rows - len(hist)

    combined = pd.concat([hist, fut], ignore_index=True, sort=False)
    return combined, {
        "removed_rows": int(removed_rows),
        "removed_races": int(len(degenerate_ids)),
    }


def _set_autotune_dataset(
    featured_df: pd.DataFrame | None,
    *,
    processed_df: pd.DataFrame | None = None,
    dataset_meta: dict[str, object] | None = None,
) -> None:
    featured_df = _compact_numeric_dtypes(featured_df, label="autotune featured")
    processed_df = _compact_numeric_dtypes(processed_df, label="autotune processed")
    st.session_state.autotune_featured_data = featured_df
    st.session_state.autotune_processed_data = processed_df
    st.session_state.autotune_dataset_meta = dataset_meta or None


def _set_model_dataset(
    featured_df: pd.DataFrame | None,
    *,
    dataset_meta: dict[str, object] | None = None,
) -> None:
    featured_df = _compact_numeric_dtypes(featured_df, label="prediction featured")
    st.session_state.model_featured_data = featured_df
    st.session_state.model_dataset_meta = dataset_meta or None


def _load_run_featured_dataset(run_id: str) -> tuple[pd.DataFrame | None, dict[str, object]]:
    try:
        run_meta = load_run_meta(run_id)
    except Exception:
        run_meta = {}

    tc = run_meta.get("training_config", {}) if isinstance(run_meta.get("training_config", {}), dict) else {}
    data_source = run_meta.get("data_source") or tc.get("data_source")
    requested_days = tc.get("dataset_days_requested", tc.get("days_back"))

    snapshot_path = get_run_featured_path(run_id)
    if snapshot_path and os.path.exists(snapshot_path):
        mtime = os.path.getmtime(snapshot_path)
        featured = _cached_load_df(snapshot_path, mtime)
        meta = _dataset_meta_from_frame(
            featured,
            data_source=data_source,
            requested_days=requested_days,
            featured_path=snapshot_path,
            processed_path=get_run_processed_path(run_id),
            origin="run_featured_snapshot",
        )
        meta["run_id"] = run_id
        return featured, meta

    cache_path = tc.get("dataset_featured_cache_path")
    if (not cache_path or not os.path.exists(str(cache_path))) and data_source and requested_days is not None:
        cache_path = _dataset_cache_paths(str(data_source), int(requested_days)).get("featured")
    if cache_path and os.path.exists(str(cache_path)):
        mtime = os.path.getmtime(str(cache_path))
        featured = _cached_load_df(str(cache_path), mtime)
        meta = _dataset_meta_from_frame(
            featured,
            data_source=data_source,
            requested_days=requested_days,
            cache_key=tc.get("dataset_cache_key"),
            featured_path=str(cache_path),
            processed_path=tc.get("dataset_processed_cache_path") or get_run_processed_path(run_id),
            origin="dataset_cache",
        )
        meta["run_id"] = run_id
        return featured, meta

    processed_path = get_run_processed_path(run_id)
    if processed_path and os.path.exists(processed_path):
        mtime = os.path.getmtime(processed_path)
        featured = _cached_build_featured_from_processed(processed_path, mtime)
        meta = _dataset_meta_from_frame(
            featured,
            data_source=data_source,
            requested_days=requested_days,
            processed_path=processed_path,
            origin="run_processed_snapshot",
        )
        meta["run_id"] = run_id
        return featured, meta

    return None, {"run_id": run_id, "origin": "missing"}


def load_model_data(run_id: str | None = None, force: bool = False) -> bool:
    target_run_id = run_id if run_id is not None else st.session_state.get("active_run_id")
    current_meta = st.session_state.get("model_dataset_meta") or {}
    if (
        not force
        and target_run_id is not None
        and current_meta.get("run_id") == target_run_id
        and st.session_state.get("model_featured_data") is not None
    ):
        return True

    if target_run_id:
        featured, meta = _load_run_featured_dataset(target_run_id)
        if featured is not None:
            _set_model_dataset(featured, dataset_meta=meta)
            return True

    path = _global_featured_dataset_path()
    if path and os.path.exists(path):
        mtime = os.path.getmtime(path)
        featured = _cached_load_df(path, mtime)
        meta = _dataset_meta_from_frame(featured, featured_path=path, origin="global_featured")
        meta["run_id"] = target_run_id
        _set_model_dataset(featured, dataset_meta=meta)
        return True

    _set_model_dataset(None, dataset_meta={"run_id": target_run_id, "origin": "missing"})
    return False


def _load_processed_history() -> pd.DataFrame | None:
    hist, _ = _get_processed_history_context()
    return hist


def _get_processed_history_context() -> tuple[pd.DataFrame | None, dict[str, object]]:
    """Load the processed (pre-feature-engineering) historical data.

    If the active run has a processed-data snapshot, that is used in
    preference to the global file — ensuring feature engineering uses
    the exact same history the model was trained on.
    """
    _run_id = st.session_state.get("active_run_id")
    if _run_id:
        _snap = get_run_processed_path(_run_id)
        if _snap and os.path.exists(_snap):
            _mtime = os.path.getmtime(_snap)
            _hist, _last_hist_date = _cached_load_history_context(_snap, _mtime)
            return _hist, {
                "path": _snap,
                "mtime": _mtime,
                "last_hist_date": _last_hist_date,
            }

    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.csv")
    path = pq_path if os.path.exists(pq_path) else csv_path
    if not os.path.exists(path):
        return None, {"path": None, "mtime": None, "last_hist_date": None}
    mtime = os.path.getmtime(path)
    hist, last_hist_date = _cached_load_history_context(path, mtime)
    return hist, {
        "path": path,
        "mtime": mtime,
        "last_hist_date": last_hist_date,
    }


def feature_engineer_with_history(
    live_processed: pd.DataFrame,
    extra_history: pd.DataFrame | None = None,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    hist = history_df if history_df is not None else _load_processed_history()
    return feature_engineer_with_history_core(hist, live_processed, extra_history=extra_history)


def scrape_gap_fill(target_date_str: str, progress_fn=None) -> pd.DataFrame | None:
    """Scrape and process results for dates between stored history and *target_date_str*.

    Returns processed (but not featured) DataFrame of gap-fill results,
    or ``None`` if no gap exists.
    """
    hist = _load_processed_history()
    target = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    if hist is not None and not hist.empty:
        hist_dates = pd.to_datetime(hist["race_date"], errors="coerce")
        last_hist = hist_dates.max().date()
    else:
        # No history — nothing to gap-fill against
        return None

    # Build list of dates we need results for (day after last history up to
    # the day before the target date)
    gap_start = last_hist + timedelta(days=1)
    gap_end = target - timedelta(days=1)

    if gap_start > gap_end:
        return None  # no gap

    gap_dates = []
    d = gap_start
    while d <= gap_end:
        gap_dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    logger.info(
        f"Gap-fill: scraping results for {len(gap_dates)} date(s) "
        f"({gap_dates[0]} → {gap_dates[-1]})"
    )

    all_gap_runners: list[pd.DataFrame] = []
    for idx, ds in enumerate(gap_dates):
        if progress_fn:
            progress_fn(idx + 1, len(gap_dates), ds)
        try:
            day_df = scrape_todays_results(date_str=ds)
            if day_df is not None and not day_df.empty:
                all_gap_runners.append(day_df)
                logger.info(f"  Gap {ds}: {len(day_df)} runners")
        except Exception as e:
            logger.warning(f"  Gap {ds}: failed — {e}")

    if not all_gap_runners:
        return None

    gap_raw = pd.concat(all_gap_runners, ignore_index=True)
    gap_proc = process_data(df=gap_raw, save=False)
    logger.info(f"Gap-fill complete: {len(gap_proc)} processed rows")
    return gap_proc


def load_existing_data():
    featured_path = _global_featured_dataset_path()
    if featured_path and os.path.exists(featured_path):
        featured_mtime = os.path.getmtime(featured_path)
        featured = _cached_load_df(featured_path, featured_mtime)
        processed = None
        processed_path = _global_processed_dataset_path()
        if processed_path and os.path.exists(processed_path):
            processed = _cached_load_df(processed_path, os.path.getmtime(processed_path))
        meta = _dataset_meta_from_frame(
            featured,
            featured_path=featured_path,
            processed_path=processed_path,
            origin="global_featured",
        )
        _set_training_dataset(featured, processed_df=processed, dataset_meta=meta)
        return True
    return False


# ── Auto-restore latest run on first load ────────────────────────────
if st.session_state.active_run_id is None and st.session_state.metrics is None:
    _latest_id = get_latest_run_id()
    if _latest_id is not None:
        try:
            _run = load_run(_latest_id)
            st.session_state.active_run_id = _latest_id
            st.session_state["sidebar_model_switcher"] = _latest_id
            st.session_state.metrics = _run.get("metrics")
            _ta_meta = _run.get("test_analysis", {})
            if _ta_meta:
                st.session_state.test_analysis = {
                    "bets": _run.get("bets_df", pd.DataFrame()),
                    "curves": _run.get("curves_df", pd.DataFrame()),
                    "stats": _ta_meta.get("stats", {}),
                    "calibration": _ta_meta.get("calibration", []),
                    "test_date_range": _ta_meta.get("test_date_range", ("?", "?")),
                    "test_races": _ta_meta.get("test_races", 0),
                    "test_runners": _ta_meta.get("test_runners", 0),
                }
            load_existing_model()
            load_model_data(_latest_id, force=True)
        except Exception:
            pass  # no runs yet or corrupt — graceful fallback


if st.session_state.featured_data is None:
    try:
        load_existing_data()
    except Exception:
        pass


# ── Default hyperparameter dictionaries ──────────────────────────────
DEFAULT_HP = {
    "classifier": {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
}

# Per-model config defaults keyed by model_key
_MODEL_KEY_DEFAULTS = {
    "classifier": config.CLASSIFIER_PARAMS,
    "place": config.PLACE_CLASSIFIER_PARAMS,
}

PRESETS = {
    "Conservative": {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.01,
        "subsample": 0.7, "colsample_bytree": 0.7,
    },
    "Balanced (default)": {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
    "Aggressive": {
        "n_estimators": 1000, "max_depth": 8, "learning_rate": 0.1,
        "subsample": 0.9, "colsample_bytree": 0.9,
    },
}

_CUSTOM_PRESET_LABEL = "Custom"
_LR_OPTIONS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]


def _nearest_option(value: float, options: list[float]) -> float:
    """Return nearest valid option for slider/select controls."""
    if not options:
        return value
    try:
        v = float(value)
    except Exception:
        return options[0]
    return min(options, key=lambda x: abs(float(x) - v))


def _mark_hp_preset_custom(prefix: str) -> None:
    """Mark the preset as custom when any manual HP control is changed."""
    _preset_key = f"{prefix}_preset"
    if st.session_state.get(_preset_key) != _CUSTOM_PRESET_LABEL:
        st.session_state[_preset_key] = _CUSTOM_PRESET_LABEL


def _apply_hp_preset(
    model_key: str,
    framework: str,
    prefix: str,
    preset_name: str,
) -> None:
    """Copy preset values into session state so sliders visibly update."""
    if preset_name == _CUSTOM_PRESET_LABEL or preset_name not in PRESETS:
        return

    defaults = _MODEL_KEY_DEFAULTS.get(model_key, DEFAULT_HP.get("classifier", {}))
    preset = PRESETS[preset_name]

    st.session_state[f"{prefix}_n_est"] = int(
        preset.get("n_estimators", defaults.get("n_estimators", 500))
    )
    st.session_state[f"{prefix}_depth"] = int(
        preset.get("max_depth", defaults.get("max_depth", 6))
    )
    st.session_state[f"{prefix}_lr"] = _nearest_option(
        float(preset.get("learning_rate", defaults.get("learning_rate", 0.05))),
        _LR_OPTIONS,
    )
    st.session_state[f"{prefix}_subsample"] = float(
        preset.get("subsample", defaults.get("subsample", 0.8))
    )
    st.session_state[f"{prefix}_colsample"] = float(
        preset.get("colsample_bytree", defaults.get("colsample_bytree", 0.8))
    )

    if framework == "xgb":
        st.session_state[f"{prefix}_mcw"] = int(defaults.get("min_child_weight", 3))
    elif framework == "cat":
        st.session_state[f"{prefix}_depth_cat"] = int(
            defaults.get("depth", defaults.get("max_depth", 6))
        )
        st.session_state[f"{prefix}_l2"] = float(defaults.get("l2_leaf_reg", 3.0))
    else:
        st.session_state[f"{prefix}_mcs"] = int(defaults.get("min_child_samples", 20))

    # Regularisation
    st.session_state[f"{prefix}_reg_alpha"] = float(
        preset.get("reg_alpha", defaults.get("reg_alpha", 0.1))
    )
    st.session_state[f"{prefix}_reg_lambda"] = float(
        preset.get("reg_lambda", defaults.get("reg_lambda", 1.0))
    )
    if framework == "lgbm":
        st.session_state[f"{prefix}_num_leaves"] = int(
            preset.get("num_leaves", defaults.get("num_leaves", 31))
        )
        st.session_state[f"{prefix}_linear_tree"] = bool(
            preset.get("linear_tree", defaults.get("linear_tree", False))
        )


def _on_hp_preset_change(model_key: str, framework: str, prefix: str) -> None:
    """Streamlit callback: apply selected preset values into HP widgets."""
    _preset_name = st.session_state.get(f"{prefix}_preset", _CUSTOM_PRESET_LABEL)
    _apply_hp_preset(model_key, framework, prefix, _preset_name)


def _hp_widgets(model_key: str, framework: str = "lgbm", prefix: str = "hp") -> dict:
    """Render hyperparameter controls for a single sub-model.

    Args:
        model_key: One of ``"classifier"``, ``"place"``.
        framework: ``"lgbm"``, ``"xgb"``, or ``"cat"``.
        prefix: Unique key prefix for Streamlit widgets.
    """
    defaults = _MODEL_KEY_DEFAULTS.get(model_key, DEFAULT_HP.get("classifier", {}))
    _preset_key = f"{prefix}_preset"
    _preset_options = [_CUSTOM_PRESET_LABEL, *list(PRESETS.keys())]
    if _preset_key not in st.session_state:
        st.session_state[_preset_key] = _CUSTOM_PRESET_LABEL

    # Preset selector
    preset_name = st.selectbox(
        "Preset",
        _preset_options,
        index=_preset_options.index(st.session_state[_preset_key])
        if st.session_state[_preset_key] in _preset_options
        else 0,
        key=_preset_key,
        on_change=_on_hp_preset_change,
        args=(model_key, framework, prefix),
        help="Quick preset — values are copied into sliders below.",
    )
    preset = PRESETS.get(preset_name, {})

    st.caption(
        "Training uses the slider values shown below. "
        "Selecting a preset now updates these controls immediately."
    )

    hp: dict = {}

    # Guard against legacy/off-grid values (e.g. 0.04) that break select_slider.
    _lr_key = f"{prefix}_lr"
    _lr_default = float(preset.get("learning_rate", defaults.get("learning_rate", 0.05)))
    if _lr_key not in st.session_state:
        st.session_state[_lr_key] = _nearest_option(_lr_default, _LR_OPTIONS)
    else:
        st.session_state[_lr_key] = _nearest_option(st.session_state[_lr_key], _LR_OPTIONS)

    c1, c2 = st.columns(2)
    with c1:
        hp["n_estimators"] = st.slider(
            "n_estimators (trees)", 50, 2000,
            preset.get("n_estimators", defaults.get("n_estimators", 500)),
            step=50, key=f"{prefix}_n_est",
            on_change=_mark_hp_preset_custom,
            args=(prefix,),
        )
        hp["max_depth"] = st.slider(
            "max_depth", 2, 15,
            preset.get("max_depth", defaults.get("max_depth", 6)),
            key=f"{prefix}_depth",
            on_change=_mark_hp_preset_custom,
            args=(prefix,),
        )
        hp["learning_rate"] = st.select_slider(
            "learning_rate",
            options=_LR_OPTIONS,
            value=st.session_state[_lr_key],
            key=f"{prefix}_lr",
            on_change=_mark_hp_preset_custom,
            args=(prefix,),
        )
    with c2:
        hp["subsample"] = st.slider(
            "subsample (row sampling)", 0.5, 1.0,
            preset.get("subsample", defaults.get("subsample", 0.8)),
            step=0.05, key=f"{prefix}_subsample",
            on_change=_mark_hp_preset_custom,
            args=(prefix,),
        )
        hp["colsample_bytree"] = st.slider(
            "colsample_bytree (feature sampling)", 0.3, 1.0,
            preset.get("colsample_bytree", defaults.get("colsample_bytree", 0.8)),
            step=0.05, key=f"{prefix}_colsample",
            on_change=_mark_hp_preset_custom,
            args=(prefix,),
        )

        # Framework-specific extras
        if framework == "xgb":
            hp["min_child_weight"] = st.slider(
                "min_child_weight", 1, 20,
                defaults.get("min_child_weight", 3),
                key=f"{prefix}_mcw",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )
        elif framework == "cat":
            hp["depth"] = st.slider(
                "depth", 3, 12,
                defaults.get("depth", defaults.get("max_depth", 6)),
                key=f"{prefix}_depth_cat",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )
            hp["l2_leaf_reg"] = st.slider(
                "l2_leaf_reg", 1.0, 20.0,
                float(defaults.get("l2_leaf_reg", 3.0)),
                step=0.5,
                key=f"{prefix}_l2",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )
        else:
            hp["min_child_samples"] = st.slider(
                "min_child_samples", 1, 100,
                defaults.get("min_child_samples", 20),
                key=f"{prefix}_mcs",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )

    # ── Regularisation controls (all frameworks) ─────────────────
    with st.expander("Regularisation", expanded=False):
        r1, r2 = st.columns(2)
        with r1:
            hp["reg_alpha"] = st.slider(
                "reg_alpha (L1)", 0.0, 5.0,
                float(defaults.get("reg_alpha", 0.1)),
                step=0.05, key=f"{prefix}_reg_alpha",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )
        with r2:
            hp["reg_lambda"] = st.slider(
                "reg_lambda (L2)", 0.0, 10.0,
                float(defaults.get("reg_lambda", 1.0)),
                step=0.1, key=f"{prefix}_reg_lambda",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )
        if framework == "lgbm":
            hp["num_leaves"] = st.slider(
                "num_leaves", 8, 128,
                int(defaults.get("num_leaves", 31)),
                key=f"{prefix}_num_leaves",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
            )
            hp["linear_tree"] = st.checkbox(
                "Linear tree (fit linear model in each leaf)",
                value=bool(defaults.get("linear_tree", False)),
                key=f"{prefix}_linear_tree",
                on_change=_mark_hp_preset_custom,
                args=(prefix,),
                help=(
                    "When enabled, each leaf fits a linear regression "
                    "instead of a constant value. Can improve accuracy "
                    "for features with strong linear relationships but "
                    "increases training time."
                ),
            )

    return hp


def _sync_hp_from_state(framework: str, prefix: str, hp: dict) -> dict:
    """Rebuild hyperparameters from Streamlit state for deterministic training."""
    synced = dict(hp or {})

    _common_map = {
        "n_estimators": f"{prefix}_n_est",
        "max_depth": f"{prefix}_depth",
        "learning_rate": f"{prefix}_lr",
        "subsample": f"{prefix}_subsample",
        "colsample_bytree": f"{prefix}_colsample",
        "reg_alpha": f"{prefix}_reg_alpha",
        "reg_lambda": f"{prefix}_reg_lambda",
    }
    for _hp_key, _state_key in _common_map.items():
        if _state_key in st.session_state:
            synced[_hp_key] = st.session_state[_state_key]

    if framework == "xgb":
        _state_key = f"{prefix}_mcw"
        if _state_key in st.session_state:
            synced["min_child_weight"] = st.session_state[_state_key]
    elif framework == "cat":
        _depth_key = f"{prefix}_depth_cat"
        _l2_key = f"{prefix}_l2"
        if _depth_key in st.session_state:
            synced["depth"] = st.session_state[_depth_key]
        if _l2_key in st.session_state:
            synced["l2_leaf_reg"] = st.session_state[_l2_key]
        synced.pop("max_depth", None)
    else:
        _mcs_key = f"{prefix}_mcs"
        _leaves_key = f"{prefix}_num_leaves"
        _lt_key = f"{prefix}_linear_tree"
        if _mcs_key in st.session_state:
            synced["min_child_samples"] = st.session_state[_mcs_key]
        if _leaves_key in st.session_state:
            synced["num_leaves"] = st.session_state[_leaves_key]
        if _lt_key in st.session_state:
            synced["linear_tree"] = bool(st.session_state[_lt_key])

    return synced


def _apply_runtime_linear_tree_flags(
    frameworks: dict[str, str],
    custom_hp: dict[str, dict] | None,
) -> None:
    """Push UI linear-tree selections into runtime config before training."""
    _param_maps = {
        "classifier": getattr(config, "CLASSIFIER_PARAMS", None),
        "place": getattr(config, "PLACE_CLASSIFIER_PARAMS", None),
    }
    for _mk, _cfg in _param_maps.items():
        if not isinstance(_cfg, dict):
            continue
        _fw = frameworks.get(_mk)
        _hp = (custom_hp or {}).get(_mk, {}) if isinstance(custom_hp, dict) else {}
        _state_key = f"hp_{_mk}_linear_tree"
        if _fw != "lgbm":
            _cfg.pop("linear_tree", None)
            continue
        _val = _hp.get("linear_tree")
        if _state_key in st.session_state:
            _val = bool(st.session_state[_state_key])
        _cfg["linear_tree"] = bool(_val)


def _mask_token(token: str | None) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return token
    return f"{token[:4]}...{token[-4:]}"


def _matchbook_events_table(events: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for event in events:
        meta_tags = event.get("meta-tags")
        country = None
        if isinstance(meta_tags, dict):
            country = meta_tags.get("COUNTRY")
        elif isinstance(meta_tags, list):
            for item in meta_tags:
                if isinstance(item, dict):
                    if item.get("type") == "COUNTRY":
                        country = item.get("name") or item.get("value")
                        break
                    if "COUNTRY" in item:
                        country = item.get("COUNTRY")
                        break
        rows.append({
            "event_id": event.get("id"),
            "name": event.get("name"),
            "start": event.get("start"),
            "status": event.get("status"),
            "country": country,
        })
    return pd.DataFrame(rows)


def _matchbook_markets_table(markets: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for market in markets:
        runners = market.get("runners") or []
        top_runner = runners[0] if runners else {}
        prices = top_runner.get("prices") or []
        best_back = next((p for p in prices if str(p.get("side", "")).lower() in {"back", "win"}), None)
        best_lay = next((p for p in prices if str(p.get("side", "")).lower() in {"lay", "lose"}), None)
        rows.append({
            "market_id": market.get("id"),
            "market_name": market.get("name"),
            "market_type": market.get("market-type") or market.get("type"),
            "status": market.get("status"),
            "runner_count": len(runners),
            "sample_runner": top_runner.get("name"),
            "best_back": None if best_back is None else best_back.get("odds"),
            "best_lay": None if best_lay is None else best_lay.get("odds"),
        })
    return pd.DataFrame(rows)


def _build_matchbook_predictions_for_date(date_str: str) -> pd.DataFrame:
    if st.session_state.predictor is None:
        if os.path.exists(_ENSEMBLE_MODEL_PATH):
            load_existing_model()
            load_model_data(force=True)
        else:
            raise MatchbookAPIError("No trained model is available. Train or load a model first.")

    cache = st.session_state.setdefault("matchbook_prediction_cache", {})
    cached = cache.get(date_str)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached

    cards_df = get_scraped_racecards(date_str=date_str)
    if cards_df is None or cards_df.empty:
        raise MatchbookAPIError(f"No Sporting Life racecards were found for {date_str}.")

    cards = cards_df.copy()
    cards["won"] = 0
    cards["finish_position"] = 0
    cards["finish_time_secs"] = 0.0
    cards["lengths_behind"] = np.nan

    processed = process_data(df=cards, save=False)
    featured = feature_engineer_with_history(processed)
    preds = _predict_featured_frame(
        st.session_state.predictor,
        featured,
        ew_fraction=st.session_state.value_config.get("ew_fraction"),
    )
    cache[date_str] = preds
    st.session_state["matchbook_prediction_cache"] = cache
    return preds


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🏇 Horse Race Predictor")
st.sidebar.caption("Calibrated Win + Place classifiers")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🎓 Train & Tune",
        "🧭 Autotune",
        "🧪 Experiments",
        "💰 Today's Picks",
        "🔌 Matchbook API",
        "🔎 Shortcomings",
        "⚖️ Strategy Calibrator",
        "📈 Model Insights",
    ],
)

st.sidebar.markdown("---")

# ── Data freshness indicator ──────────────────────────────────────────
_db = _cached_db_stats()
if _db and _db.get("total_runners", 0) > 0:
    _latest = _db.get("latest_date", "—")
    _total_r = _db.get("total_runners", 0)
    _total_races = _db.get("total_races", 0)
    st.sidebar.markdown(
        f"📅 **Data:** {_total_r:,} runners · {_total_races:,} races\n\n"
        f"📆 **Latest:** {_latest}\n\n"
        f"💾 **DB size:** {_db.get('db_size_mb', 0):.1f} MB"
    )
else:
    st.sidebar.info("🌐 Data via **Sporting Life** scraper — no keys needed.")

_ENSEMBLE_MODEL_PATH = os.path.join(config.MODELS_DIR, "triple_ensemble_models.joblib")

# ── Active model selector ─────────────────────────────────────────────
# Consume any pending selectbox sync BEFORE the widget renders.
# Training / Experiments "Activate" write _pending_model_switch instead of
# the widget key directly (which raises StreamlitAPIException after render).
_pending_switch = st.session_state.pop("_pending_model_switch", None)
if _pending_switch:
    st.session_state["sidebar_model_switcher"] = _pending_switch

_sidebar_runs = list_runs()
_sidebar_runs_with_model = [r for r in _sidebar_runs if run_has_model(r["run_id"])]

if _sidebar_runs_with_model:
    st.sidebar.markdown("**🤖 Active Model**")
    _active_id = st.session_state.get("active_run_id")

    _run_labels = {
        r["run_id"]: f"{r.get('name', r['run_id'])}  ({r.get('timestamp', '')[:10]})"
        for r in _sidebar_runs_with_model
    }
    _run_ids = list(_run_labels.keys())

    # Current selection index
    _current_idx = 0
    if _active_id in _run_ids:
        _current_idx = _run_ids.index(_active_id)

    _chosen_id = st.sidebar.selectbox(
        "Switch run",
        _run_ids,
        index=_current_idx,
        # Bind the dict at definition time — `_run_labels` is reused as a
        # list by the Shortcomings page, and format_func can be invoked
        # after that section has overwritten the module-level name.
        format_func=lambda rid, _labels=_run_labels: _labels.get(rid, str(rid)),
        key="sidebar_model_switcher",
        label_visibility="collapsed",
    )

    # If user picked a different run, activate it
    if _chosen_id != _active_id:
        with st.sidebar:
            with st.spinner("Restoring model …"):
                if restore_run_model(_chosen_id):
                    _cached_load_model.clear()
                    load_existing_model()
                    _restored = load_run(_chosen_id)
                    st.session_state.active_run_id = _chosen_id
                    st.session_state.metrics = _restored.get("metrics")
                    _rta = _restored.get("test_analysis", {})
                    if _rta:
                        st.session_state.test_analysis = {
                            "bets": _restored.get("bets_df", pd.DataFrame()),
                            "curves": _restored.get("curves_df", pd.DataFrame()),
                            "stats": _rta.get("stats", {}),
                            "calibration": _rta.get("calibration", []),
                            "test_date_range": _rta.get("test_date_range", ("?", "?")),
                            "test_races": _rta.get("test_races", 0),
                            "test_runners": _rta.get("test_runners", 0),
                        }
                    load_model_data(_chosen_id, force=True)
                    _invalidate_run_caches()
                    st.rerun()
                else:
                    st.sidebar.error("Failed to restore model.")

    # Show status of the active model
    if st.session_state.predictor is not None:
        _p = st.session_state.predictor
        _mtype = type(_p).__name__
        _label = "Win + Place Pipeline" if _mtype in ("RacePredictor", "TripleEnsemblePredictor") else _mtype
        st.sidebar.success(f"✅ **{_label}**")
    else:
        # Model file exists on disk but isn't loaded yet
        if st.sidebar.button("Load Saved Model", key="sidebar_load_model"):
            if load_existing_model():
                load_model_data(force=True)
                st.rerun()

elif os.path.exists(_ENSEMBLE_MODEL_PATH):
    # Runs exist on disk but none have model snapshots — load from file
    if st.session_state.predictor is not None:
        _p = st.session_state.predictor
        _mtype = type(_p).__name__
        _label = "Win + Place Pipeline" if _mtype in ("RacePredictor", "TripleEnsemblePredictor") else _mtype
        st.sidebar.success(f"✅ **{_label}** loaded")
    elif st.sidebar.button("Load Saved Model", key="sidebar_load_legacy"):
        if load_existing_model():
            load_model_data(force=True)
            st.sidebar.success("✅ Model loaded!")
            st.rerun()
else:
    st.sidebar.warning("⚠️ No model trained yet")

# ── Feature mismatch warning ─────────────────────────────────────────
if st.session_state.predictor is not None and st.session_state.get("model_featured_data") is not None:
    _model_feats = set(getattr(st.session_state.predictor, "feature_cols", []) or [])
    _data_feats = set(get_feature_columns(st.session_state.model_featured_data))
    if _model_feats and _data_feats:
        _missing_in_data = _model_feats - _data_feats
        if len(_missing_in_data) > 3:
            st.sidebar.warning(
                f"⚠️ Model expects {len(_missing_in_data)} features not in data. "
                "Re-train to align."
            )

# Apply pending value-strategy updates before sidebar widgets are instantiated.
_pending_vc = st.session_state.pop("_pending_value_config", None)
if isinstance(_pending_vc, dict):
    st.session_state.value_config.update(_pending_vc)
    st.session_state["_g_staking_mode"] = st.session_state.value_config.get("staking_mode", "flat")
    st.session_state["_g_value_threshold"] = float(st.session_state.value_config.get("value_threshold", 0.05))
    st.session_state["_g_value_odds_range"] = _value_odds_range(st.session_state.value_config)
    st.session_state["_g_kelly_fraction"] = float(st.session_state.value_config.get("kelly_fraction", 0.25))
    st.session_state["_g_bankroll"] = float(st.session_state.value_config.get("bankroll", 100.0))
    st.session_state["_g_ew_enabled"] = bool(st.session_state.value_config.get("ew_enabled", True))
    st.session_state["_g_ew_fraction"] = float(st.session_state.value_config.get("ew_fraction", 0.20))
    st.session_state["_g_ew_min_edge"] = float(st.session_state.value_config.get("ew_min_place_edge", 0.15))
    st.session_state["_g_ew_odds_range"] = (
        float(st.session_state.value_config.get("ew_min_odds", 4.0)),
        float(st.session_state.value_config.get("ew_max_odds", 51.0)),
    )

# ── Global Value Strategy Config ──────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Value Strategy", expanded=False):
    _g_staking = st.selectbox(
        "Staking mode",
        ["flat", "kelly"],
        index=["flat", "kelly"].index(st.session_state.value_config["staking_mode"]),
        format_func=lambda x: "Flat 1-unit" if x == "flat" else "Fractional Kelly",
        key="_g_staking_mode",
        help=(
            "**Flat** — stake 1 unit on every value bet.  \n"
            "**Fractional Kelly** — size bets proportionally to "
            "edge and bankroll."
        ),
    )
    _g_threshold = st.slider(
        "Edge threshold",
        min_value=0.01, max_value=0.20,
        value=st.session_state.value_config["value_threshold"],
        step=0.01,
        key="_g_value_threshold",
        help=(
            "Minimum *model_prob − implied_prob* edge required "
            "before placing a value bet (base — dynamically scaled by odds)."
        ),
    )
    _g_value_odds_range = st.slider(
        "Win value odds range",
        min_value=1.0,
        max_value=101.0,
        value=_value_odds_range(st.session_state.value_config),
        step=1.0,
        key="_g_value_odds_range",
        help="Only flag win value bets in this odds range. This is a strong ROI lever for longshot control.",
    )
    if _g_staking == "kelly":
        _g_kelly = st.slider(
            "Kelly fraction",
            min_value=0.05, max_value=1.0,
            value=st.session_state.value_config["kelly_fraction"],
            step=0.05,
            key="_g_kelly_fraction",
            help="Fraction of full Kelly. ¼ Kelly (0.25) is a popular choice.",
        )
        _g_bankroll = st.number_input(
            "Starting bankroll (units)",
            min_value=10.0, max_value=10000.0,
            value=st.session_state.value_config["bankroll"],
            step=10.0,
            key="_g_bankroll",
            help="Simulated starting bankroll for Kelly sizing.",
        )
    else:
        _g_kelly = st.session_state.value_config["kelly_fraction"]
        _g_bankroll = st.session_state.value_config["bankroll"]

    st.markdown("---")
    st.markdown("**Each-Way Settings**")
    _g_ew_enabled = st.checkbox(
        "Enable each-way analysis",
        value=st.session_state.value_config.get("ew_enabled", True),
        key="_g_ew_enabled",
        help="Show each-way value bets alongside win value bets.",
    )
    _ew_frac_options = [0.20, 0.25, 0.33]
    _ew_frac_labels = {0.20: "1/5", 0.25: "1/4", 0.33: "1/3"}
    _g_ew_fraction = st.selectbox(
        "EW odds fraction",
        _ew_frac_options,
        index=_ew_frac_options.index(
            st.session_state.value_config.get("ew_fraction", 0.20)
        ) if st.session_state.value_config.get("ew_fraction", 0.20) in _ew_frac_options else 0,
        format_func=lambda x: _ew_frac_labels.get(x, f"{x:.2f}"),
        key="_g_ew_fraction",
        help=(
            "The fraction of win odds paid for the place leg.  "
            "Standard UK is **1/4** for most races, but some "
            "bookmaker promos offer **1/5** (worse for you) or "
            "enhanced **1/3** (better for you) terms."
        ),
    )
    _g_ew_min_edge = st.slider(
        "Min place edge",
        min_value=0.01, max_value=0.20,
        value=st.session_state.value_config.get("ew_min_place_edge", 0.15),
        step=0.01,
        key="_g_ew_min_edge",
        help="Minimum place_prob \u2212 implied_place_prob edge for EW bets.",
    )
    _ew_odds_range = st.slider(
        "EW odds range",
        min_value=2.0, max_value=101.0,
        value=(
            st.session_state.value_config.get("ew_min_odds", 4.0),
            st.session_state.value_config.get("ew_max_odds", 51.0),
        ),
        step=1.0,
        key="_g_ew_odds_range",
        help="Only flag EW value in this odds range (decimal). Sweet spot is 4\u201351.",
    )

    # Persist to session state
    st.session_state.value_config = {
        "staking_mode": _g_staking,
        "value_threshold": _g_threshold,
        "value_min_odds": _g_value_odds_range[0],
        "value_max_odds": _g_value_odds_range[1],
        "kelly_fraction": _g_kelly,
        "bankroll": _g_bankroll,
        "ew_enabled": _g_ew_enabled,
        "ew_fraction": _g_ew_fraction,
        "ew_min_place_edge": _g_ew_min_edge,
        "ew_min_odds": _ew_odds_range[0],
        "ew_max_odds": _ew_odds_range[1],
    }

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Built with Streamlit · XGBoost · LightGBM · CatBoost<br>"
    "⚠️ For educational purposes only</small>",
    unsafe_allow_html=True,
)


# =====================================================================
#  TRAIN & TUNE
# =====================================================================
if page == "🎓 Train & Tune":
    st.title("🎓 Train & Tune")

    # ── Model overview ───────────────────────────────────────────────
    with st.expander("ℹ️ About the Models", expanded=False):
        st.markdown(
            "Two **task-specific classifiers** drive all betting decisions, "
            "plus two reference models:\n\n"
            "| Model | Objective | Role |\n"
            "|-------|-----------|------|\n"
            "| **Win Classifier** | Log-loss / focal | Value Bets & Top Pick — calibrated P(win) |\n"
            "| **Place Classifier** | Log-loss / focal | Each-Way — calibrated P(place) |\n"
            "| **Race Ranker** | LambdaRank | Diagnostics only — ranker/classifier agreement panels; **not** blended into win probabilities |\n"
            "| **Linear Baseline** | Logistic regression | Sanity reference on the same features — tree models should clearly beat it |\n\n"
            "Each classifier is calibrated with Platt scaling + isotonic regression, "
            "fitted on out-of-fold predictions from purged expanding-window CV "
            "(optionally refit on the most recent training slab)."
        )

    # ── Data Settings ────────────────────────────────────────────────
    st.subheader("1️⃣ Data Source")
    dc1, dc2, dc3 = st.columns([2, 2, 1])

    with dc1:
        data_source = st.selectbox(
            "Source",
            ["database", "scrape", "sample"],
            help=(
                "**database** — SQLite DB, only scrapes new days (recommended).  \n"
                "**scrape** — full re-scrape from Sporting Life.  \n"
                "**sample** — synthetic data for testing."
            ),
        )
    with dc2:
        if data_source == "sample":
            num_races = st.slider("Races to generate", 500, 5000, 1500, 100)
            days_back = 90
        else:
            num_races = 1500
            default_days = 90 if data_source == "database" else 30
            # Derive max from actual data span
            _db_info = _cached_db_stats()
            _max_days = 2000
            try:
                from datetime import datetime as _dt
                _earliest = _db_info.get("earliest_date", "")
                _latest = _db_info.get("latest_date", "")
                if _earliest and _earliest != "—" and _latest and _latest != "—":
                    _max_days = max(
                        (_dt.strptime(_latest, "%Y-%m-%d") - _dt.strptime(_earliest, "%Y-%m-%d")).days,
                        1,
                    )
            except Exception:
                pass
            days_back = st.slider(
                "Days of history", 1, _max_days,
                min(default_days, _max_days), 7,
            )
    with dc3:
        wf_min_train = st.number_input(
            "Min train months", 2, 24, 3, 1, key="wf_min_train",
            help="Minimum months of data before the first walk-forward fold.",
        )

    wf_fast_fold = st.checkbox(
        "Fast fold (halve tree count for WF folds)",
        value=True,
        help=(
            "When enabled, WF folds train with half the n_estimators "
            "for speed. Disable to use full hyperparameters — slower "
            "but gives a fairer comparison to the final model."
        ),
    )

    include_odds = st.checkbox(
        "Blend market odds (market anchor)",
        value=False,
        help=(
            "Odds are **never** fed to the model as features — the "
            "LightGBM/CatBoost models are always purely form-based.  \n\n"
            "When **enabled**, the form model's probabilities are blended "
            "with the market price via the Benter market anchor "
            "(`softmax(α·log p_model + β·log p_market)`), fitted on "
            "out-of-fold predictions.  \n\n"
            "**⚠️ SP is only known at race-off** — anchored backtest "
            "metrics will look optimistic vs. live betting.  \n\n"
            "**Disable** for the pure form model with no market "
            "information at all — the honest view of predictive power."
        ),
    )

    # ── Dataset cache status ─────────────────────────────────────────
    _cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "cache")
    os.makedirs(_cache_dir, exist_ok=True)
    _cache_files = [f for f in os.listdir(_cache_dir) if f.endswith(".parquet")] if os.path.isdir(_cache_dir) else []
    _global_featured_path = _global_featured_dataset_path()
    _global_cache_label = None
    if _global_featured_path and os.path.exists(_global_featured_path):
        _global_age_h = (time.time() - os.path.getmtime(_global_featured_path)) / 3600
        _global_size_mb = os.path.getsize(_global_featured_path) / (1024 * 1024)
        _global_cache_label = (
            f"{os.path.basename(_global_featured_path)} "
            f"({_global_size_mb:.1f} MB, {_global_age_h:.1f}h ago)"
        )
    if _cache_files or _global_cache_label:
        _cache_labels = []
        for cf in sorted(_cache_files):
            _cf_path = os.path.join(_cache_dir, cf)
            _age_h = (time.time() - os.path.getmtime(_cf_path)) / 3600
            _size_mb = os.path.getsize(_cf_path) / (1024 * 1024)
            _label = cf.replace("featured_", "").replace(".parquet", "")
            _cache_labels.append(f"{_label} ({_size_mb:.1f} MB, {_age_h:.1f}h ago)")
        _cc1, _cc2 = st.columns([3, 1])
        with _cc1:
            if _cache_labels:
                st.caption(f"📦 Dataset cache snapshots: {', '.join(_cache_labels)}")
            if _global_cache_label:
                st.caption(f"📄 Global training dataset: {_global_cache_label}")
        with _cc2:
            if st.button("🗑️ Clear cache", key="clear_dataset_cache"):
                import shutil
                shutil.rmtree(_cache_dir, ignore_errors=True)
                os.makedirs(_cache_dir, exist_ok=True)
                st.rerun()

    # ── Prepare / load dataset ───────────────────────────────────────
    _prep_col1, _prep_col2 = st.columns([1, 2])
    with _prep_col1:
        _do_prepare = st.button(
            "📦 Prepare Data", type="secondary", width="stretch",
        )
    with _prep_col2:
        _has_cached = st.session_state.featured_data is not None
        if _has_cached:
            _cd = st.session_state.featured_data
            _train_meta = st.session_state.get("train_dataset_meta") or {}
            _cd_dates = pd.to_datetime(_cd["race_date"], errors="coerce")
            _cd_span = f"{_cd_dates.min().date()} → {_cd_dates.max().date()}" if not _cd_dates.isna().all() else "?"
            _cd_months = int(_cd_dates.dt.to_period("M").nunique()) if not _cd_dates.isna().all() else 0
            st.success(
                f"✅ Dataset ready: **{len(_cd):,} rows**, "
                f"**{_cd_months} months** ({_cd_span})"
            )
            _loaded_source = _train_meta.get("data_source")
            _loaded_days = _train_meta.get("actual_days")
            if _loaded_source or _loaded_days:
                st.caption(
                    f"Training dataset context: source={_loaded_source or '?'} · "
                    f"actual span={_loaded_days if _loaded_days is not None else '?'}d"
                )
            if (
                _loaded_source is not None and _loaded_source != data_source
            ) or (
                _train_meta.get("requested_days") is not None
                and data_source != "sample"
                and int(_train_meta.get("requested_days")) != int(days_back)
            ):
                st.warning(
                    "Loaded training dataset does not match the current source/days controls. "
                    "Click Prepare Data if you want to train on the currently selected window."
                )
        else:
            st.info("No dataset loaded — click **Prepare Data** to build one.")

    with st.expander("🔎 RTV Coverage Diagnostics", expanded=False):
        _proc_for_rtv = st.session_state.get("train_processed_data")
        if _proc_for_rtv is None or _proc_for_rtv.empty:
            st.caption("Load or prepare a dataset first to run RTV diagnostics.")
        else:
            _run_rtv_audit = st.button("Run RTV missingness audit", key="run_rtv_missing_audit")
            if _run_rtv_audit:
                with st.spinner("Analysing RTV key matches and missing metrics …"):
                    st.session_state["_rtv_missing_audit"] = _build_rtv_missing_diagnostics(_proc_for_rtv)

            _rtv_audit = st.session_state.get("_rtv_missing_audit")
            if isinstance(_rtv_audit, dict):
                if not _rtv_audit.get("ok", False):
                    st.warning(str(_rtv_audit.get("message", "RTV diagnostics unavailable.")))
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Key match %", f"{_rtv_audit['key_match_pct']:.2f}%")
                    c2.metric("Any metric %", f"{_rtv_audit['any_metric_pct']:.2f}%")
                    c3.metric("No key match %", f"{_rtv_audit['no_key_pct']:.2f}%")
                    c4.metric("Key match, no metrics %", f"{_rtv_audit['no_metrics_pct']:.2f}%")

                    st.markdown("**Top tracks with no key match (rows)**")
                    st.dataframe(_rtv_audit["top_no_key_tracks"], width="stretch", height=220)

                    if "by_race_type_no_key" in _rtv_audit:
                        st.markdown("**No-key-match % by race type**")
                        st.dataframe(_rtv_audit["by_race_type_no_key"], width="stretch", height=180)

                    st.markdown("**Races where all/most runners have no RTV key match**")
                    st.dataframe(_rtv_audit["no_key_races"].head(200), width="stretch", height=260)

                    st.markdown("**Races with RTV key match but no populated metrics**")
                    st.dataframe(_rtv_audit["no_metric_races"].head(200), width="stretch", height=260)

                    d1, d2 = st.columns(2)
                    with d1:
                        st.download_button(
                            "⬇️ Download no-key-match rows CSV",
                            data=_rtv_audit["no_key_rows"].to_csv(index=False),
                            file_name="rtv_no_key_match_rows.csv",
                            mime="text/csv",
                            key="dl_rtv_no_key_rows",
                        )
                    with d2:
                        st.download_button(
                            "⬇️ Download key-match-no-metrics rows CSV",
                            data=_rtv_audit["no_metric_rows"].to_csv(index=False),
                            file_name="rtv_key_match_no_metrics_rows.csv",
                            mime="text/csv",
                            key="dl_rtv_no_metric_rows",
                        )

    if _do_prepare:
        _prep_progress = st.progress(0, text="Starting data pipeline …")

        _prep_total_steps = 4 + (3 if data_source != "sample" else 0)
        _prep_started_at = time.time()
        _prep_state = {"step": None, "step_started_at": _prep_started_at}

        def _update_prep_progress(step_num: int, text: str, stage_progress: float = 0.0) -> None:
            _step = min(max(int(step_num), 1), _prep_total_steps)
            _stage = min(max(float(stage_progress), 0.0), 1.0)
            if _prep_state["step"] != _step:
                _prep_state["step"] = _step
                _prep_state["step_started_at"] = time.time()
            _overall = ((_step - 1) + _stage) / _prep_total_steps
            _timing = _progress_timing_text(_prep_started_at, _prep_state["step_started_at"], _stage)
            _prep_progress.progress(_overall, text=f"Step {_step}/{_prep_total_steps} · {text} · {_timing}")

        _cache_paths = _dataset_cache_paths(data_source, int(days_back) if data_source != "sample" else None)
        _cache_key = _cache_paths.get("cache_key")
        _cache_path = _cache_paths.get("featured")
        _processed_cache_path = _cache_paths.get("processed")
        _cache_hit = False

        if data_source != "sample" and _cache_path and os.path.exists(_cache_path):
            _update_prep_progress(1, "📦 Loading cached dataset …")
            with st.spinner("Loading cached dataset …"):
                featured = _cached_load_df(_cache_path, os.path.getmtime(_cache_path))
                processed = (
                    _cached_load_df(_processed_cache_path, os.path.getmtime(_processed_cache_path))
                    if _processed_cache_path and os.path.exists(_processed_cache_path)
                    else None
                )
            _cache_age_h = (time.time() - os.path.getmtime(_cache_path)) / 3600
            _train_meta = _dataset_meta_from_frame(
                featured,
                data_source=data_source,
                requested_days=int(days_back),
                cache_key=str(_cache_key) if _cache_key is not None else None,
                featured_path=_cache_path,
                processed_path=_processed_cache_path if _processed_cache_path and os.path.exists(_processed_cache_path) else None,
                origin="dataset_cache",
            )
            _set_training_dataset(featured, processed_df=processed, dataset_meta=_train_meta)
            st.success(
                f"✅ Loaded cached dataset ({featured.shape[0]:,} rows, "
                f"{featured.shape[1]} cols) — built {_cache_age_h:.1f}h ago"
            )
            _cache_hit = True

        if not _cache_hit:
            # ── Step 1: Collect historical data ──────────────────────
            _update_prep_progress(1, "📊 Collecting historical data …")
            with st.spinner("Collecting data …"):
                if data_source in ("database", "scrape"):
                    raw_data = collect_data(source=data_source, days_back=days_back)
                else:
                    raw_data = collect_data(source="sample", num_races=num_races)
            st.success(f"✅ Collected {len(raw_data):,} race entries")
            raw_data["_is_future"] = 0

            # ── Step 1b: Incremental RTV backfill for this window ───
            if data_source != "sample":
                _update_prep_progress(2, "🏇 Updating RTV cache for missing races …")
                with st.spinner("Checking/fetching missing RTV metrics for this window …"):
                    try:
                        _rtv_stats = backfill_rtv_metrics_for_races(
                            raw_data,
                            skip_existing=True,
                        )
                        if _rtv_stats.get("missing_races", 0) > 0:
                            st.caption(
                                "RTV cache updated: "
                                f"{_rtv_stats.get('fetched_races', 0)} missing races checked, "
                                f"{_rtv_stats.get('new_rows', 0)} new runner-metric rows added"
                            )
                            if _rtv_stats.get("skipped_known_missing", 0) > 0:
                                st.caption(
                                    "RTV skip-list: "
                                    f"{_rtv_stats.get('skipped_known_missing', 0)} known no-data races skipped"
                                )
                            if _rtv_stats.get("new_known_missing", 0) > 0:
                                st.caption(
                                    "RTV skip-list updated: "
                                    f"{_rtv_stats.get('new_known_missing', 0)} additional past races marked as no-data"
                                )
                        else:
                            st.caption("RTV cache already up to date for the selected history window")
                    except Exception as _rtv_exc:
                        logger.warning("RTV incremental backfill failed: %s", _rtv_exc)
                        st.caption("RTV incremental backfill skipped due to an error (continuing build)")

            # ── Step 2: Fetch racecards for the build day + next 7 days ──
            # Offset 0 (the build day) is included so the current date is
            # never a blind spot: the historical results scrape only has
            # settled races (≤ yesterday), and without offset 0 the lookahead
            # window would start at tomorrow, leaving today uncovered.
            _LOOKAHEAD_OFFSETS = range(0, 8)  # today … +7
            _n_offsets = len(_LOOKAHEAD_OFFSETS)
            _future_card_sigs: dict[str, str] = {}  # date_str → signature
            if data_source != "sample":
                _update_prep_progress(3, "🗓️ Fetching racecards for today + next 7 days …")
                _card_frames: list[pd.DataFrame] = []
                _today = datetime.now().date()
                for _i, _offset in enumerate(_LOOKAHEAD_OFFSETS, start=1):
                    _target = _today + timedelta(days=_offset)
                    _target_str = _target.strftime("%Y-%m-%d")
                    try:
                        _cards = get_scraped_racecards(date_str=_target_str)
                        if _cards is not None and not _cards.empty:
                            _future_card_sigs[_target_str] = cards_signature(_cards)
                            _cards["_is_future"] = 1
                            _cards["won"] = 0
                            _cards["finish_position"] = 0
                            _cards["finish_time_secs"] = 0.0
                            _cards["lengths_behind"] = float("nan")
                            _card_frames.append(_cards)
                    except Exception as _card_exc:
                        logger.warning("Failed to fetch racecards for %s: %s", _target_str, _card_exc)
                    _update_prep_progress(3, f"🗓️ Fetching racecards for today + next 7 days … ({_i}/{_n_offsets})", _i / _n_offsets)
                if _card_frames:
                    _future_raw = pd.concat(_card_frames, ignore_index=True)
                    # The build day (offset 0) can overlap races already scraped
                    # as settled history (if run after some races have results).
                    # Keep the settled copy; drop the duplicate pre-race card rows
                    # so the combined FE pass never sees a race twice.
                    if "race_id" in raw_data.columns and "race_id" in _future_raw.columns:
                        _existing_rids = set(raw_data["race_id"].astype(str))
                        _future_raw = _future_raw[~_future_raw["race_id"].astype(str).isin(_existing_rids)]
                    if not _future_raw.empty:
                        st.success(f"✅ Fetched {len(_future_raw):,} future racecard entries")
                        raw_data = pd.concat([raw_data, _future_raw], ignore_index=True, sort=False)

            # ── Step 3: Process combined dataset ─────────────────────
            _process_step = 4 if data_source != "sample" else 2
            _update_prep_progress(_process_step, "🔧 Processing combined dataset …")
            with st.spinner("Cleaning …"):
                combined_processed = process_data(df=raw_data, save=False)

            # Remove known-bad historical races before FE to improve speed
            # and avoid deriving features from corrupt race outcomes.
            combined_processed, _deg_stats = _drop_degenerate_races_pre_fe(combined_processed)
            if _deg_stats.get("removed_races", 0) > 0:
                st.caption(
                    "🧹 Pre-FE race quality filter: removed "
                    f"{_deg_stats['removed_rows']:,} rows across "
                    f"{_deg_stats['removed_races']:,} degenerate races"
                )
            st.success(f"✅ {combined_processed.shape[0]:,} records, {combined_processed.shape[1]} columns")

            # ── Step 4: Feature-engineer combined dataset (one pass) ─
            _feature_step = 5 if data_source != "sample" else 3
            _update_prep_progress(_feature_step, "⚙️ Engineering features …")
            with st.spinner("Feature engineering …"):
                combined_featured = engineer_features(combined_processed, save=False)

            # ── Step 5: Split historical vs future ───────────────────
            _finalise_step = 6 if data_source != "sample" else 4
            _update_prep_progress(_finalise_step, "🧾 Finalising training dataset …")
            if "_is_future" in combined_featured.columns:
                future_featured = combined_featured[combined_featured["_is_future"] == 1].copy()
                featured = combined_featured[combined_featured["_is_future"] == 0].copy()
                featured = featured.drop(columns=["_is_future"], errors="ignore")
                future_featured = future_featured.drop(columns=["_is_future"], errors="ignore")
            else:
                featured = combined_featured
                future_featured = pd.DataFrame()

            # RTV derived-feature coverage on future rows is the meaningful
            # pre-race signal quality metric (raw RTV race metrics are not
            # expected to exist before the race is run).
            if not future_featured.empty:
                _raw_rtv_cols = [c for c in (RTV_METRIC_COLS + RTV_RANK_COLS) if c in future_featured.columns]
                _rtv_derived_cols = [
                    c for c in future_featured.columns
                    if c.startswith("rtv_") and c not in _raw_rtv_cols
                ]
                if _rtv_derived_cols:
                    _has_any_derived = future_featured[_rtv_derived_cols].notna().any(axis=1)
                    _derived_cov = 100.0 * float(_has_any_derived.mean())
                    st.caption(
                        "📊 Future RTV derived-feature coverage: "
                        f"{int(_has_any_derived.sum()):,} / {len(future_featured):,} rows "
                        f"({_derived_cov:.1f}%)"
                    )
                else:
                    st.caption("📊 Future RTV derived-feature coverage: no derived RTV columns found")

            # Also split processed for session state (historical only)
            if "_is_future" in combined_processed.columns:
                processed = combined_processed[combined_processed["_is_future"] == 0].copy()
                processed = processed.drop(columns=["_is_future"], errors="ignore")
            else:
                processed = combined_processed

            _train_meta = _dataset_meta_from_frame(
                featured,
                data_source=data_source,
                requested_days=int(days_back) if data_source != "sample" else None,
                cache_key=str(_cache_key) if _cache_key is not None else None,
                featured_path=_cache_path,
                processed_path=_processed_cache_path,
                origin="fresh_build",
            )
            _set_training_dataset(featured, processed_df=processed, dataset_meta=_train_meta)

            if data_source != "sample" and _cache_path:
                featured.to_parquet(_cache_path, index=False)
                if _processed_cache_path:
                    processed.to_parquet(_processed_cache_path, index=False)
                st.caption(f"💾 Dataset cached as `{_cache_key}`")

            # ── Step 6: Store future rows as lookahead cache ─────────
            if data_source != "sample":
                _lookahead_step = 7
                _update_prep_progress(_lookahead_step, "🔮 Updating lookahead cache …")

            if not future_featured.empty:
                clear_lookahead_cache()
                _future_dates = pd.to_datetime(
                    future_featured["race_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d").unique()
                _sorted_future_dates = sorted(_future_dates)
                for _idx, _fd in enumerate(_sorted_future_dates, start=1):
                    _fd_mask = (
                        pd.to_datetime(future_featured["race_date"], errors="coerce")
                        .dt.strftime("%Y-%m-%d") == _fd
                    )
                    _fd_rows = future_featured[_fd_mask]
                    _fd_sig = _future_card_sigs.get(_fd, cards_signature(_fd_rows))
                    save_lookahead_cache(_fd, _fd_rows, _fd_sig)
                    if data_source != "sample":
                        _update_prep_progress(
                            _lookahead_step,
                            f"🔮 Saving lookahead cache … ({_idx}/{len(_sorted_future_dates)})",
                            _idx / max(len(_sorted_future_dates), 1),
                        )
                st.caption(f"🔮 Lookahead cached: {', '.join(_sorted_future_dates)}")
            elif data_source != "sample":
                st.caption("🔮 No upcoming racecards found for lookahead.")

        _prep_elapsed = _format_duration_compact(time.time() - _prep_started_at)
        _prep_progress.progress(1.0, text=f"✅ Data ready! · elapsed {_prep_elapsed}")
        st.rerun()

    st.markdown("---")

    model_type = "race_predictor"

    # ── Model Frameworks ─────────────────────────────────────────────
    st.subheader("2️⃣ Model Frameworks")

    _TASK_MODELS = {
        "classifier": "Win Classifier (Value)",
        "place": "Place Classifier (EW)",
    }

    st.caption("Select the ML framework for each task-specific model.")
    _framework_options = ["lgbm", "xgb", "cat"]
    _fw_defaults = dict(getattr(config, "SUB_MODEL_FRAMEWORKS", {}))
    _frameworks: dict[str, str] = {}
    _fw_cols = st.columns(2)
    for i, (_mk, _label) in enumerate(_TASK_MODELS.items()):
        with _fw_cols[i]:
            _def_fw = _fw_defaults.get(_mk, "cat")
            _frameworks[_mk] = st.selectbox(
                _label,
                options=_framework_options,
                index=_framework_options.index(_def_fw) if _def_fw in _framework_options else 0,
                key=f"fw_{_mk}",
            )
    _train_ranker_toggle = st.checkbox(
        "Also train diagnostic Race Ranker (LightGBM LambdaRank)",
        value=bool(getattr(config, "TRAIN_RANKER", False)),
        key="train_ranker_toggle",
        help=(
            "Adds roughly a third to training time. The ranker's scores are "
            "never used for betting — they only power the ranker/classifier "
            "agreement panels and a dedicated autotune study."
        ),
    )
    config.TRAIN_RANKER = bool(_train_ranker_toggle)

    st.markdown("---")

    # ── Hyperparameter Tuning ────────────────────────────────────────
    st.subheader("3️⃣ Hyperparameters")
    st.caption(
        "Set hyperparameters per sub-model manually, or let Optuna "
        "results from the dedicated Autotune page drive training."
    )
    if getattr(config, "TRAIN_RANKER", False):
        st.caption(
            "The race ranker has no manual hyperparameter controls. It is trained automatically alongside the classifier and place model for diagnostics (ranker/classifier agreement), and saved autotune sessions can tune it with a dedicated ranker objective. Its scores are not blended into win probabilities."
        )

    tune_mode = st.radio(
        "Tuning mode",
        ["⚙️ Manual", "📦 Saved Autotune"],
        horizontal=True,
        help=(
            "**Manual** — choose hyperparameters per sub-model with sliders.\n\n"
            "**Saved Autotune** — reuse persisted Optuna results from the dedicated Autotune page."
        ),
    )

    custom_hp: dict[str, dict] | None = None
    _saved_autotune_session: dict | None = None

    if tune_mode == "⚙️ Manual":
        # One tab per model
        _enabled_keys_sorted = list(_TASK_MODELS.keys())
        _tab_labels = [_TASK_MODELS[k] for k in _enabled_keys_sorted]
        custom_hp = {}

        if len(_enabled_keys_sorted) == 1:
            # Single model — no tabs needed
            _mk = _enabled_keys_sorted[0]
            with st.expander(
                f"⚙️ {_TASK_MODELS[_mk]} ({_frameworks.get(_mk, 'lgbm').upper()})", expanded=False,
            ):
                custom_hp[_mk] = _hp_widgets(
                    _mk, framework=_frameworks.get(_mk, "lgbm"), prefix=f"hp_{_mk}",
                )
        else:
            _hp_tabs = st.tabs(_tab_labels)
            for _tab, _mk in zip(_hp_tabs, _enabled_keys_sorted):
                with _tab:
                    custom_hp[_mk] = _hp_widgets(
                        _mk, framework=_frameworks.get(_mk, "lgbm"), prefix=f"hp_{_mk}",
                    )

        for _mk in _enabled_keys_sorted:
            custom_hp[_mk] = _sync_hp_from_state(
                framework=_frameworks.get(_mk, "lgbm"),
                prefix=f"hp_{_mk}",
                hp=custom_hp.get(_mk, {}),
            )
    else:
        _autotune_sessions = list_autotune_sessions()
        if not _autotune_sessions:
            st.warning(
                "No saved autotune sessions found yet. Use the 🧭 Autotune page first, or switch to Manual."
            )
        else:
            _labels = []
            _label_to_session = {}
            for _session in _autotune_sessions:
                _meta = _session.get("dataset_meta") or {}
                _label = (
                    f"{_session.get('name', _session.get('session_id'))} · "
                    f"{_session.get('status', 'unknown')} · "
                    f"{_meta.get('data_source', '?')} {_meta.get('actual_days') or '—'}d · "
                    f"{_session.get('updated_at', '')[:16]}"
                )
                _labels.append(_label)
                _label_to_session[_label] = _session
            _picked_label = st.selectbox(
                "Saved autotune session",
                _labels,
                key="_train_saved_autotune_session",
            )
            _saved_autotune_session = _label_to_session.get(_picked_label)
            if _saved_autotune_session is not None:
                _saved_frameworks = _saved_autotune_session.get("frameworks") or {}
                if _saved_frameworks:
                    _frameworks = {**_frameworks, **_saved_frameworks}
                    st.info(
                        "Training will use the frameworks stored in the selected autotune session so the tuned params remain valid."
                    )
                custom_hp = {
                    _mk: dict(_params)
                    for _mk, _params in (_saved_autotune_session.get("best_params") or {}).items()
                    if isinstance(_params, dict) and _params
                }
                _apply_runtime_linear_tree_flags(_frameworks, custom_hp)

                _saved_ds = _saved_autotune_session.get("dataset_meta") or {}
                _current_ds = st.session_state.get("train_dataset_meta") or {}
                if _saved_ds and _current_ds:
                    _same_source = _saved_ds.get("data_source") == _current_ds.get("data_source")
                    _same_span = _saved_ds.get("actual_days") == _current_ds.get("actual_days")
                    if not (_same_source and _same_span):
                        st.warning(
                            "The selected autotune session was fitted on a different dataset window than the currently loaded training dataset. "
                            "Best practice is to align the tuning and training history windows."
                        )

                with st.expander("Saved best parameters", expanded=False):
                    st.json(_saved_autotune_session.get("best_params") or {})

    # Experiment name
    _train_ds_meta = st.session_state.get("train_dataset_meta") or {}
    _name_source = _train_ds_meta.get("data_source") or data_source
    _name_days = _train_ds_meta.get("actual_days")
    if _name_days is None and data_source != "sample":
        _name_days = int(days_back)
    _default_exp_name = _build_run_name(
        data_source=str(_name_source),
        days_back=(_name_days if _name_days is not None else None),
        include_odds=bool(include_odds),
        tune_mode_label=tune_mode,
        auto_trials=0,
    )
    exp_name = st.text_input(
        "Experiment name (optional)",
        value=_default_exp_name,
        help=(
            "Leave blank to auto-name with training setup (source/history/odds/tuning/model-count)."
        ),
    )

    st.markdown("---")

    # ── Value Strategy (read from sidebar global config) ─────────────
    _value_config = st.session_state.value_config
    _vc_mode = _value_config["staking_mode"]
    _vc_label = (
        f"Fractional Kelly ({_value_config['kelly_fraction']:.0%}) · "
        f"Bankroll {_value_config['bankroll']:.0f}"
        if _vc_mode == "kelly" else "Flat 1-unit"
    )
    st.info(
        f"📐 **Value Strategy:** {_vc_label} · "
        f"Threshold {_value_config['value_threshold']:.2f}  \n"
        f"_Change in the **⚙️ Value Strategy** section of the sidebar._"
    )


    # ── Advanced Training Options ────────────────────────────────────
    with st.expander("🔧 Advanced Training Options", expanded=False):
        _prune_cfg = float(getattr(config, "FEATURE_PRUNE_FRACTION", 0.0))
        _prune_enabled = st.checkbox(
            "Enable feature pruning",
            value=_prune_cfg > 0.0,
            help=(
                "Runs a quick pilot model on the training split and drops the "
                "lowest-importance features before full training."
            ),
            key="_train_prune_enabled",
        )
        if _prune_enabled:
            # Step 1: Correlation de-duplication (runs first)
            _corr_cfg = float(getattr(config, "FEATURE_CORR_THRESHOLD", 0.0))
            _corr_enabled = st.checkbox(
                "Drop correlated features",
                value=_corr_cfg > 0.0,
                help=(
                    "Step 1: Remove one feature from each "
                    "highly-correlated pair (keeping the more important one). "
                    "Runs before importance pruning to prevent near-duplicate "
                    "features from splitting importance."
                ),
                key="_train_corr_enabled",
            )
            if _corr_enabled:
                _corr_thresh = st.slider(
                    "Correlation threshold |r|",
                    min_value=0.80,
                    max_value=0.99,
                    value=max(_corr_cfg, 0.95),
                    step=0.01,
                    format="%.2f",
                    key="_train_corr_thresh",
                    help="Feature pairs with |Pearson r| above this are de-duplicated.",
                )
                config.FEATURE_CORR_THRESHOLD = _corr_thresh
            else:
                config.FEATURE_CORR_THRESHOLD = 0.0

            # Step 2: Importance pruning (runs second)
            _prune_pct = st.slider(
                "Prune bottom feature %",
                min_value=5,
                max_value=50,
                value=max(int(round(_prune_cfg * 100)), 20),
                step=5,
                key="_train_prune_pct",
                help="Step 2: Percentage of lowest-importance features to remove (after correlation de-dup).",
            )
            config.FEATURE_PRUNE_FRACTION = _prune_pct / 100.0

            _summary_parts = []
            if _corr_enabled:
                _summary_parts.append(f"correlation de-dup at |r|>{config.FEATURE_CORR_THRESHOLD:.2f}")
            _summary_parts.append(f"dropping bottom **{_prune_pct}%**")
            st.caption(f"Feature pruning active: {', then '.join(_summary_parts)}.")
        else:
            config.FEATURE_PRUNE_FRACTION = 0.0
            config.FEATURE_CORR_THRESHOLD = 0.0

        _es_val = int(getattr(config, "EARLY_STOPPING_ROUNDS", 0))
        _es_enabled = st.checkbox(
            "Enable early stopping",
            value=_es_val > 0,
            help=(
                "When enabled, Phase 2 holds out the last ~10% of training "
                "data for validation and stops adding trees when the metric "
                "stops improving. **Disabling** lets the model train on ALL "
                "data for the full n_estimators — recommended if you want "
                "hyperparameters to have maximum effect."
            ),
            key="_train_es_enabled",
        )
        if _es_enabled:
            _es_rounds_ui = st.slider(
                "Patience (rounds without improvement)",
                min_value=10, max_value=200, value=max(_es_val, 50), step=10,
                key="_train_es_rounds",
                help="Number of boosting rounds without improvement before stopping.",
            )
            config.EARLY_STOPPING_ROUNDS = _es_rounds_ui
        else:
            config.EARLY_STOPPING_ROUNDS = 0

    st.markdown("---")

    skip_wf = st.checkbox(
        "Skip walk-forward validation",
        value=False,
        help=(
            "When enabled, skips the walk-forward backtest and jumps "
            "straight to training the final model on the train/test split. "
            "Much faster but you won't get per-fold evaluation metrics."
        ),
        key="_skip_wf",
    )

    _train_label = (
        "🚀 Start Training (Final Split Only)"
        if skip_wf
        else "🚀 Start Training (Walk-Forward)"
    )

    # ── Train ────────────────────────────────────────────────────────
    if st.button(_train_label, type="primary", width="stretch"):
        if st.session_state.featured_data is None:
            st.error("No dataset loaded. Click **📦 Prepare Data** first.")
            st.stop()

        featured = st.session_state.featured_data.copy()
        _train_ds_meta = st.session_state.get("train_dataset_meta") or {}

        # Odds are never model features (always excluded in get_feature_columns);
        # the toggle only controls whether the market-anchor blend is applied.
        if include_odds:
            st.info(
                "🎯 Market anchor **on** — form-model probabilities will be "
                "blended with market odds (odds are still not model features)."
            )
        else:
            st.info("📐 Pure form model — no market blend.")

        if isinstance(custom_hp, dict):
            for _mk in list(custom_hp.keys()):
                custom_hp[_mk] = _sync_hp_from_state(
                    framework=_frameworks.get(_mk, "lgbm"),
                    prefix=f"hp_{_mk}",
                    hp=custom_hp.get(_mk, {}),
                )
        _apply_runtime_linear_tree_flags(_frameworks, custom_hp)

        st.success(f"✅ {featured.shape[1]} features")

        t0 = time.time()
        progress = st.progress(0, text="Starting training …")
        _train_total_steps = 1 if skip_wf else 2
        _train_state = {"step": None, "step_started_at": t0}

        def _update_train_progress(step_num: int, text: str, stage_progress: float = 0.0) -> None:
            _step = min(max(int(step_num), 1), _train_total_steps)
            _stage = min(max(float(stage_progress), 0.0), 1.0)
            if _train_state["step"] != _step:
                _train_state["step"] = _step
                _train_state["step_started_at"] = time.time()
            _overall = ((_step - 1) + _stage) / _train_total_steps
            _timing = _progress_timing_text(t0, _train_state["step_started_at"], _stage)
            progress.progress(_overall, text=f"Step {_step}/{_train_total_steps} · {text} · {_timing}")

        if not skip_wf:
            # ── Walk-forward backtest ────────────────────────────────
            _update_train_progress(1, "🔁 Starting walk-forward validation …")
            _vc = _value_config or {}

            def _wf_progress_cb(msg: str, pct: float) -> None:
                _update_train_progress(1, f"🔁 {msg}", pct)

            wf_report = walk_forward_validation(
                featured,
                model_type="race_predictor",
                min_train_months=int(wf_min_train),
                value_threshold=_vc.get("value_threshold", 0.05),
                frameworks=_frameworks,
                params=custom_hp,
                progress_callback=_wf_progress_cb,
                fast_fold=wf_fast_fold,
                ew_min_place_edge=_vc.get("ew_min_place_edge"),
            )
            st.session_state.wf_report = wf_report

            n_folds = len(wf_report.get("folds", []))
            _update_train_progress(
                1,
                f"✅ Walk-forward complete ({n_folds} folds)",
                1.0,
            )
        else:
            st.session_state.wf_report = None
            _update_train_progress(1, "⏭️ Walk-forward skipped — starting final model calibration and training …")

        # ── Retrain final model on ALL data for live predictions ─────
        if tune_mode == "📦 Saved Autotune" and not custom_hp:
            st.error("The selected autotune session does not contain usable best parameters.")
            st.stop()

        def _training_cb(msg: str, pct: float) -> None:
            _train_step = 1 if skip_wf else 2
            _update_train_progress(_train_step, f"🤖 {msg}", pct)

        if not skip_wf:
            _update_train_progress(2, "🤖 Starting final model calibration and training …")

        predictor = RacePredictor(frameworks=_frameworks)
        metrics = predictor.train(
            featured, params=custom_hp, progress_callback=_training_cb,
            value_config=_value_config,
            blend_market_odds=bool(include_odds),
        )

        elapsed = time.time() - t0
        st.session_state.predictor = predictor
        st.session_state.metrics = metrics

        progress.progress(1.0, text=f"✅ Complete! · elapsed {_format_duration_compact(time.time() - t0)}")

        # ── Results ──────────────────────────────────────────────────
        st.markdown("### 📊 Results")
        st.caption(f"Trained in {elapsed:.1f}s · Frameworks: {predictor.frameworks}")
        _eval_split = getattr(predictor, "eval_split_info", None)
        if isinstance(_eval_split, dict):
            st.caption(
                "Evaluation split: holdout "
                f"({_eval_split.get('validation_races', 0)} races / "
                f"{_eval_split.get('validation_runners', 0)} runners)."
            )
            st.caption(
                "Model safety-net degenerate filter removed "
                f"{_eval_split.get('degenerate_removed_rows', 0):,} rows across "
                f"{_eval_split.get('degenerate_removed_races', 0):,} races."
            )

        # ── Walk-Forward Validation Results ──────────────────────────
        wf_report = st.session_state.get("wf_report")
        if wf_report:
            wf_summary = wf_report.get("summary", pd.DataFrame())
            wf_bets = wf_report.get("bets", pd.DataFrame())
            wf_curves = wf_report.get("curves", pd.DataFrame())

            st.markdown("#### 🔁 Walk-Forward Validation")

            if not wf_summary.empty:
                wm1, wm2, wm3, wm4 = st.columns(4)
                wm1.metric("Folds", int(len(wf_summary)))
                _avg_brier = wf_summary["brier_score"].mean() if "brier_score" in wf_summary.columns else 0
                wm2.metric("Avg Brier", f"{_avg_brier:.4f}")
                wm3.metric("Avg NDCG@1", f"{wf_summary['ndcg_at_1'].mean():.4f}")
                wm4.metric("Avg Top-1", f"{wf_summary['top1_accuracy'].mean():.1%}")

                wp1, wp2, wp3, wp4 = st.columns(4)
                _wf_tp = float(wf_summary["top_pick_pnl"].sum())
                _wf_vp = float(wf_summary["value_pnl"].sum())
                _wf_ew = float(wf_summary["ew_pnl"].sum())
                _wf_combined = _wf_tp + _wf_vp + _wf_ew
                wp1.metric("Top Pick P&L", f"{_wf_tp:+.2f}u")
                wp2.metric("Value P&L", f"{_wf_vp:+.2f}u")
                wp3.metric("EW P&L", f"{_wf_ew:+.2f}u")
                wp4.metric("Combined P&L", f"{_wf_combined:+.2f}u")

                with st.expander("📋 Per-Fold Summary", expanded=False):
                    st.dataframe(wf_summary, width="stretch", hide_index=True)

            if not wf_curves.empty:
                _wf_strat_labels = {"top_pick": "Top Pick", "value": "Value", "each_way": "Each-Way"}
                _wf_curves_plot = wf_curves.copy()
                _wf_curves_plot["strategy"] = _wf_curves_plot["strategy"].map(
                    lambda s: _wf_strat_labels.get(s, s)
                )
                _wf_fig = px.line(
                    _wf_curves_plot,
                    x="race_date", y="cum_pnl",
                    color="strategy",
                    title="Walk-Forward Cumulative P&L",
                    labels={"race_date": "Date", "cum_pnl": "P&L (units)"},
                )
                _wf_fig.add_hline(y=0, line_dash="dash", line_color="grey")
                _wf_fig.update_layout(height=420, legend_title_text="")
                st.plotly_chart(_wf_fig, width="stretch")

            # Download buttons
            if not wf_summary.empty or not wf_bets.empty:
                _wd1, _wd2 = st.columns(2)
                with _wd1:
                    if not wf_summary.empty:
                        st.download_button(
                            "📥 Download fold summary",
                            data=wf_summary.to_csv(index=False).encode("utf-8"),
                            file_name="wf_fold_summary.csv", mime="text/csv",
                            key="wf_dl_folds",
                        )
                with _wd2:
                    if not wf_bets.empty:
                        st.download_button(
                            "📥 Download all bets",
                            data=wf_bets.to_csv(index=False).encode("utf-8"),
                            file_name="wf_all_bets.csv", mime="text/csv",
                            key="wf_dl_bets",
                        )

            st.markdown("---")
            st.markdown(
                "#### 🏆 Final Model Performance (full retrain)"
            )
            st.caption(
                "The final model is retrained on **all data** for live predictions. "
                "Metrics below reflect the temporal holdout validation split "
                "of that final retrain."
            )

        _model_display = {
            "win_classifier": ("Win Classifier (Value)", predictor.frameworks.get("classifier", "?").upper()),
            "ranker": ("Race Ranker (LambdaRank)", "LGBM"),
            "place_classifier": ("Place Classifier (EW)", predictor.frameworks.get("place", "?").upper()),
        }

        _perf_rows = []
        for mk, (label, fw) in _model_display.items():
            m = metrics.get(mk, {})
            row = {"Model": label, "Framework": fw}
            if mk == "win_classifier":
                row["Brier Score"] = m.get("brier_score")
                row["ECE"] = m.get("ece")
                row["RPS"] = m.get("rps")
                row["Value Bets"] = m.get("value_bets")
                row["VB Strike Rate"] = m.get("value_bet_sr")
                row["VB ROI"] = m.get("value_bet_roi")
                row["VB Exp ROI %"] = m.get("value_bet_exp_roi_pct")
                row["VB Brier"] = m.get("value_bet_brier")
                row["Avg Edge"] = m.get("avg_edge")
            elif mk == "ranker":
                row["Brier Score"] = m.get("brier_score")
                row["ECE"] = m.get("ece")
                row["RPS"] = m.get("rps")
                row["NDCG@1"] = m.get("ndcg_at_1")
                row["Top-1"] = m.get("top1_accuracy")
            elif mk == "place_classifier":
                row["Brier (cal)"] = m.get("brier_calibrated")
                row["Brier (raw)"] = m.get("brier_raw")
                row["ECE"] = m.get("ece")
                row["Place Precision"] = m.get("place_precision")
            _perf_rows.append(row)

        import pandas as _pd_perf
        import numpy as _np_perf
        _perf_df = _pd_perf.DataFrame(_perf_rows)
        st.dataframe(_perf_df, width="stretch", hide_index=True)

        # ── Most important features ───────────────────────────────
        _fi_model = getattr(predictor, "clf_model", None)
        _fi_cols = getattr(predictor, "feature_cols", None)
        if _fi_model is not None and _fi_cols is not None:
            st.markdown("#### 🧠 Most Important Features")
            _fi_top_n = st.slider(
                "Top features to show",
                min_value=10,
                max_value=50,
                value=20,
                step=5,
                key="_fi_top_n_train",
            )
            fi_df = get_feature_importance(_fi_model, _fi_cols, top_n=int(_fi_top_n))
            st.dataframe(fi_df, width="stretch", hide_index=True)
            if not fi_df.empty:
                _fig_fi = px.bar(
                    fi_df.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top {int(_fi_top_n)} Feature Importances",
                )
                _fig_fi.update_layout(height=max(350, 16 * len(fi_df) + 120))
                st.plotly_chart(_fig_fi, width="stretch")

        # ── Holdout Validation Analysis (equity curves, value picks, P&L) ─────
        test_analysis = getattr(predictor, "test_analysis", None)
        if test_analysis:
            st.session_state.test_analysis = test_analysis
            st.markdown("---")
            st.markdown("### 📈 Holdout Validation Analysis")
            st.caption(
                f"Validation period: **{test_analysis['test_date_range'][0]}** → "
                f"**{test_analysis['test_date_range'][1]}** · "
                f"{test_analysis['test_races']} races · "
                f"{test_analysis['test_runners']} runners"
            )
            st.caption(
                "⚠️ Backtest uses **SP (Starting Price)** odds — the final "
                "market price at race-off. Live predictions use earlier "
                "odds, so real-world ROI will be lower than shown here."
            )

            ta_stats = test_analysis["stats"]

            # ── Strategy summary metrics ─────────────────────────────
            st1, st2, st3 = st.columns(3)

            with st1:
                st.markdown("#### 🎯 Top Pick (1 bet / race)")
                tp = ta_stats.get("top_pick", {})
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Bets", tp.get("bets", 0))
                tc2.metric("Strike Rate", f"{tp.get('strike_rate', 0):.1f}%")
                tc3.metric(
                    "ROI",
                    f"{tp.get('roi', 0):+.1f}%",
                    f"{tp.get('pnl', 0):+.1f} units",
                )
                tc4.metric("Max DD", f"{tp.get('max_drawdown', 0):.1f}")
                st.caption(
                    f"Avg odds — all: {tp.get('avg_odds_all', 0):.2f} · "
                    f"winners: {tp.get('avg_odds_winners', 0):.2f}"
                )

            with st2:
                st.markdown("#### 💎 Value Bets")
                vb = ta_stats.get("value", {})
                _vc_info = test_analysis.get("value_config", {})
                _is_kelly = _vc_info.get("staking_mode") == "kelly"
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.metric("Bets", vb.get("bets", 0))
                vc2.metric("Strike Rate", f"{vb.get('strike_rate', 0):.1f}%")
                vc3.metric(
                    "ROI",
                    f"{vb.get('roi', 0):+.1f}%",
                    f"{vb.get('pnl', 0):+.1f} units",
                )
                vc4.metric("Max DD", f"{vb.get('max_drawdown', 0):.1f}")
                _stake_label = (
                    f"Kelly {_vc_info.get('kelly_fraction', 0.25):.0%} · "
                    f"Avg stake {vb.get('avg_stake', 0):.2f} · "
                    f"Total staked {vb.get('total_staked', 0):.1f}"
                ) if _is_kelly else f"Flat 1-unit · Total staked {vb.get('total_staked', 0):.0f}"
                st.caption(
                    f"{_stake_label}  \n"
                    f"Avg odds — all: {vb.get('avg_odds_all', 0):.2f} · "
                    f"winners: {vb.get('avg_odds_winners', 0):.2f}"
                )
                _vb_diag = []
                if vb.get("avg_edge") is not None:
                    _vb_diag.append(f"Avg edge {vb.get('avg_edge', 0):+.3f}")
                if vb.get("avg_clv") is not None:
                    _vb_diag.append(f"Avg CLV {vb.get('avg_clv', 0):.3f}x")
                if vb.get("expected_roi") is not None:
                    _vb_diag.append(f"Exp ROI {vb.get('expected_roi', 0):+.1f}%")
                if vb.get("selected_brier") is not None:
                    _vb_diag.append(f"Sel Brier {vb.get('selected_brier', 0):.4f}")
                if _vb_diag:
                    st.caption(" · ".join(_vb_diag))
                if _is_kelly and _vc_info.get("final_bankroll") is not None:
                    _sb = _vc_info.get("starting_bankroll", 100)
                    _fb = _vc_info["final_bankroll"]
                    _growth = ((_fb / _sb) - 1) * 100 if _sb > 0 else 0
                    st.caption(
                        f"💰 Bankroll: {_sb:.0f} → {_fb:.1f} "
                        f"({_growth:+.1f}%)"
                    )

            with st3:
                st.markdown("#### 🐎 Each-Way Bets")
                ewb = ta_stats.get("each_way", {})
                ew1, ew2, ew3, ew4 = st.columns(4)
                ew1.metric("Bets", ewb.get("bets", 0))
                ew2.metric("Place Rate", f"{ewb.get('place_rate', 0):.1f}%")
                ew3.metric(
                    "ROI",
                    f"{ewb.get('roi', 0):+.1f}%",
                    f"{ewb.get('pnl', 0):+.1f} units",
                )
                ew4.metric("Max DD", f"{ewb.get('max_drawdown', 0):.1f}")
                st.caption(
                    f"2-unit stakes · Won {ewb.get('winners', 0)} · "
                    f"Placed {ewb.get('placed', 0)} / {ewb.get('bets', 0)}  \n"
                    f"Avg odds — all: {ewb.get('avg_odds_all', 0):.2f} · "
                    f"winners: {ewb.get('avg_odds_winners', 0):.2f}"
                )

            # ── Equity curves ────────────────────────────────────────
            curves = test_analysis.get("curves")
            if curves is not None and not curves.empty:
                eq1, eq2 = st.columns(2)
                _hover_cols = [c for c in ["bet_number", "horse_name", "odds", "stake", "pnl"] if c in curves.columns]
                _staking_label = (
                    f"Kelly {_vc_info.get('kelly_fraction', 0.25):.0%}"
                    if _is_kelly else "flat 1-unit stakes"
                )
                _curve_markers = len(curves) <= 400
                with eq1:
                    fig_pnl = px.line(
                        curves,
                        x="race_date", y="cum_pnl",
                        color="strategy",
                        markers=_curve_markers,
                        title=f"Cumulative P&L ({_staking_label})",
                        hover_data=_hover_cols,
                        labels={
                            "race_date": "Date",
                            "cum_pnl": "P&L (units)",
                            "bet_number": "Bet #",
                        },
                    )
                    fig_pnl.add_hline(
                        y=0, line_dash="dash", line_color="grey",
                    )
                    fig_pnl.update_layout(height=400)
                    st.plotly_chart(fig_pnl, width="stretch")

                with eq2:
                    fig_roi = px.line(
                        curves,
                        x="race_date", y="cum_roi_pct",
                        color="strategy",
                        markers=_curve_markers,
                        title="Cumulative ROI %",
                        hover_data=_hover_cols,
                        labels={
                            "race_date": "Date",
                            "cum_roi_pct": "ROI (%)",
                            "bet_number": "Bet #",
                        },
                    )
                    fig_roi.add_hline(
                        y=0, line_dash="dash", line_color="grey",
                    )
                    fig_roi.update_layout(height=400)
                    st.plotly_chart(fig_roi, width="stretch")

            # ── Value bets by odds band ──────────────────────────────
            band_data = ta_stats.get("value_by_odds_band")
            if band_data:
                st.markdown("#### 🎰 Value Bets by Odds Band")
                band_df = pd.DataFrame(band_data)
                bc1, bc2 = st.columns(2)

                with bc1:
                    fig_band_roi = px.bar(
                        band_df, x="odds_band", y="roi",
                        title="ROI % by Odds Range",
                        text="roi",
                        color="roi",
                        color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
                        color_continuous_midpoint=0,
                    )
                    fig_band_roi.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                    )
                    fig_band_roi.update_layout(
                        height=350, showlegend=False,
                    )
                    st.plotly_chart(fig_band_roi, width="stretch")

                with bc2:
                    fig_band_sr = px.bar(
                        band_df, x="odds_band", y="strike_rate",
                        title="Strike Rate % by Odds Range",
                        text="strike_rate",
                        color="strike_rate",
                        color_continuous_scale="Blues",
                    )
                    fig_band_sr.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                    )
                    fig_band_sr.update_layout(
                        height=350, showlegend=False,
                    )
                    st.plotly_chart(fig_band_sr, width="stretch")

                st.dataframe(
                    band_df, hide_index=True,
                    width="stretch",
                )

            # ── Calibration chart ────────────────────────────────────
            cal_data = test_analysis.get("calibration")
            if cal_data:
                st.markdown("#### 🎯 Model Calibration")
                cal_df = pd.DataFrame(cal_data)
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Bar(
                    x=cal_df["prob_bucket"],
                    y=cal_df["avg_model_pct"],
                    name="Model Prob %",
                    marker_color="#3b82f6",
                ))
                fig_cal.add_trace(go.Bar(
                    x=cal_df["prob_bucket"],
                    y=cal_df["actual_win_rate"],
                    name="Actual Win %",
                    marker_color="#22c55e",
                ))
                fig_cal.update_layout(
                    barmode="group",
                    title="Predicted Probability vs Actual Win Rate",
                    xaxis_title="Model Probability Bucket",
                    yaxis_title="%",
                    height=380,
                )
                st.plotly_chart(fig_cal, width="stretch")

            # ── Reliability curve + per-decile calibration ─────────
            _win_clf_m = metrics.get("win_classifier", {})
            _rel_bins = _win_clf_m.get("reliability_bins")
            _decile_data = _win_clf_m.get("decile_calibration")
            if _rel_bins:
                st.markdown("#### 📉 Reliability Curve (Win Classifier)")
                _rb = [b for b in _rel_bins if b is not None]
                if _rb:
                    _rel_df = pd.DataFrame(_rb)
                    fig_rel = go.Figure()
                    fig_rel.add_trace(go.Scatter(
                        x=_rel_df["mean_pred"], y=_rel_df["obs_rate"],
                        mode="markers+lines", name="Model",
                        marker=dict(size=8, color="#3b82f6"),
                        text=[f"n={r['count']}" for r in _rb],
                        hovertemplate="Pred: %{x:.3f}<br>Obs: %{y:.3f}<br>%{text}",
                    ))
                    fig_rel.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines", name="Perfect",
                        line=dict(dash="dash", color="#94a3b8"),
                    ))
                    _ece_val = _win_clf_m.get("ece", 0)
                    fig_rel.update_layout(
                        title=f"Reliability Diagram (ECE = {_ece_val:.4f})",
                        xaxis_title="Mean Predicted Probability",
                        yaxis_title="Observed Win Rate",
                        height=380,
                        xaxis=dict(range=[0, max(0.5, _rel_df["mean_pred"].max() * 1.2)]),
                        yaxis=dict(range=[0, max(0.5, _rel_df["obs_rate"].max() * 1.2)]),
                    )
                    st.plotly_chart(fig_rel, width="stretch")

            if _decile_data:
                st.markdown("#### Per-Decile Calibration (Win Classifier)")
                _dec_df = pd.DataFrame(_decile_data)
                _dec_df["gap"] = (_dec_df["mean_pred"] - _dec_df["obs_rate"]).round(4)
                _dec_df.columns = ["Decile", "Prob Lo", "Prob Hi",
                                   "Mean Pred", "Obs Win Rate", "Count", "Gap"]
                st.dataframe(_dec_df, hide_index=True, width="stretch")

            # ── Daily P&L bars ───────────────────────────────────────
            daily_data = ta_stats.get("top_pick_daily")
            if daily_data:
                st.markdown("#### 📅 Daily P&L (Top Pick)")
                daily_df = pd.DataFrame(daily_data)
                fig_daily = px.bar(
                    daily_df, x="race_date", y="daily_pnl",
                    title="Daily P&L",
                    color="daily_pnl",
                    color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
                    color_continuous_midpoint=0,
                    labels={
                        "race_date": "Date",
                        "daily_pnl": "P&L (units)",
                    },
                )
                fig_daily.update_layout(
                    height=350, showlegend=False,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_daily, width="stretch")

            # ── Full bet log ─────────────────────────────────────────
            bets = test_analysis.get("bets")
            if bets is not None and not bets.empty:
                with st.expander(
                    f"📋 Full Bet Log ({len(bets)} bets)"
                ):
                    display_bets = bets.copy()
                    display_bets["pnl"] = display_bets["pnl"].apply(
                        lambda v: f"{v:+.2f}",
                    )
                    st.dataframe(
                        display_bets,
                        hide_index=True,
                        width="stretch",
                    )

        # ── Log experiment ───────────────────────────────────────────
        # Flatten nested metrics dict for storage
        flat: dict = {}
        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                if isinstance(mv, dict):
                    for kk, vv in mv.items():
                        flat[f"{mk}/{kk}"] = vv
                else:
                    flat[mk] = mv

        def _effective_linear_tree(_model_key: str) -> bool:
            _fw = _frameworks.get(_model_key)
            if _fw != "lgbm":
                return False
            _manual_hp = (custom_hp or {}).get(_model_key, {}) if isinstance(custom_hp, dict) else {}
            if "linear_tree" in _manual_hp:
                return bool(_manual_hp.get("linear_tree"))
            if _model_key == "classifier":
                return bool(getattr(config, "CLASSIFIER_PARAMS", {}).get("linear_tree", False))
            if _model_key == "place":
                return bool(getattr(config, "PLACE_CLASSIFIER_PARAMS", {}).get("linear_tree", False))
            return False

        _linear_tree_flags = {
            _mk: _effective_linear_tree(_mk)
            for _mk in _TASK_MODELS
        }

        _training_config = {
            "days_back": _train_ds_meta.get("actual_days"),
            "dataset_days_requested": _train_ds_meta.get("requested_days", int(days_back) if data_source != "sample" else None),
            "data_source": _train_ds_meta.get("data_source", data_source),
            "dataset_date_start": _train_ds_meta.get("date_start"),
            "dataset_date_end": _train_ds_meta.get("date_end"),
            "dataset_months": _train_ds_meta.get("months"),
            "dataset_cache_key": _train_ds_meta.get("cache_key"),
            "dataset_featured_cache_path": _train_ds_meta.get("featured_path"),
            "dataset_processed_cache_path": _train_ds_meta.get("processed_path"),
            "walk_forward": not bool(skip_wf),
            "wf_min_train_months": int(wf_min_train),
            "wf_fast_fold": bool(wf_fast_fold),
            "include_odds": bool(include_odds),
            "tuning_mode": "saved" if tune_mode == "📦 Saved Autotune" else "manual",
            "auto_tune_trials": None,
            "saved_autotune_session_id": (_saved_autotune_session or {}).get("session_id"),
            "saved_autotune_session_name": (_saved_autotune_session or {}).get("name"),
            "frameworks": dict(_frameworks),
            "linear_tree": _linear_tree_flags,
            "feature_pruning_enabled": bool(getattr(config, "FEATURE_PRUNE_FRACTION", 0.0) > 0.0),
            "feature_prune_fraction": float(getattr(config, "FEATURE_PRUNE_FRACTION", 0.0)),
            "feature_corr_threshold": float(getattr(config, "FEATURE_CORR_THRESHOLD", 0.0)),
            "early_stopping_rounds": int(config.EARLY_STOPPING_ROUNDS),
            "burn_in_months": int(getattr(config, "BURN_IN_MONTHS", 0)),
            "purge_days": int(getattr(config, "PURGE_DAYS", 0)),
            "eval_split": getattr(predictor, "eval_split_info", None),
        }

        _run_name = (exp_name or "").strip() or _default_exp_name

        exp_entry = {
            "name": _run_name,
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "data_source": _train_ds_meta.get("data_source", data_source),
            "data_rows": len(featured),
            "n_features": len(get_feature_columns(featured)),
            "tuning_mode": "saved" if tune_mode == "📦 Saved Autotune" else "manual",
            "hyperparameters": custom_hp if custom_hp else {},
            "metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in flat.items()
            },
            "elapsed_seconds": round(elapsed, 1),
            "training_config": _training_config,
        }
        # Add WF summary to experiment entry
        _wf_rep = st.session_state.get("wf_report")
        if _wf_rep and not _wf_rep.get("summary", pd.DataFrame()).empty:
            _wf_s = _wf_rep["summary"]
            _training_config["wf_folds"] = int(len(_wf_s))
            exp_entry["walk_forward"] = {
                "n_folds": int(len(_wf_s)),
                "avg_brier": round(float(_wf_s["brier_score"].mean()), 6),
                "avg_ndcg1": round(float(_wf_s["ndcg_at_1"].mean()), 4),
                "avg_top1": round(float(_wf_s["top1_accuracy"].mean()), 4),
                "top_pick_pnl": round(float(_wf_s["top_pick_pnl"].sum()), 2),
                "value_pnl": round(float(_wf_s["value_pnl"].sum()), 2),
                "ew_pnl": round(float(_wf_s["ew_pnl"].sum()), 2),
            }
        if _saved_autotune_session is not None:
            exp_entry["saved_autotune"] = {
                "session_id": _saved_autotune_session.get("session_id"),
                "name": _saved_autotune_session.get("name"),
            }
        _log_experiment(exp_entry)

        # ── Persist full run snapshot ────────────────────────────────
        _auto_info = None
        if _saved_autotune_session is not None:
            _auto_info = {
                "mode": "saved",
                "session_id": _saved_autotune_session.get("session_id"),
                "name": _saved_autotune_session.get("name"),
            }

        _run_id = save_run(
            name=_run_name,
            model_type=model_type,
            data_source=str(_train_ds_meta.get("data_source", data_source)),
            data_rows=len(featured),
            n_features=len(get_feature_columns(featured)),
            elapsed_seconds=elapsed,
            hyperparameters=custom_hp if custom_hp else {},
            metrics=metrics,
            train_metrics=getattr(predictor, "train_metrics", None),
            test_analysis=test_analysis,
            auto_tune=_auto_info,
            training_config=_training_config,
            wf_report=st.session_state.get("wf_report"),
            featured_df=featured,
            processed_df=st.session_state.get("train_processed_data"),
        )
        st.session_state.active_run_id = _run_id
        st.session_state["_pending_model_switch"] = _run_id  # applied before selectbox next rerun
        st.session_state.test_analysis = test_analysis
        _set_model_dataset(
            featured.copy(),
            dataset_meta={**_train_ds_meta, "run_id": _run_id, "origin": "training_session"},
        )
        _invalidate_run_caches()
        _cached_load_model.clear()  # new model saved — bust cache

        st.success(
            f"Model saved · Run **{_run_name}** persisted "
            f"({elapsed:.1f}s) 🎉"
        )


# =====================================================================
#  AUTOTUNE
# =====================================================================
elif page == "🧭 Autotune":
    st.title("🧭 Autotune")
    st.caption(
        "Run Optuna studies separately from training, persist them to disk, and reuse the best params later from Train & Tune."
    )
    st.info(
        "This workflow is isolated from model training. It stores studies under data/autotune, resumes cleanly, and does not retrain a final production model."
    )

    _at_sessions = list_autotune_sessions()
    _complete_sessions = sum(1 for s in _at_sessions if s.get("status") == "complete")
    _m1, _m2, _m3 = st.columns(3)
    _m1.metric("Saved sessions", len(_at_sessions))
    _m2.metric("Completed", _complete_sessions)
    _m3.metric("Latest update", (_at_sessions[0].get("updated_at", "—")[:16] if _at_sessions else "—"))

    st.markdown("---")
    st.subheader("1️⃣ Dataset")
    _dataset_mode = st.radio(
        "Dataset source",
        ["Use current Train & Tune dataset", "Load latest featured dataset", "Build fresh dataset"],
        horizontal=True,
        key="_autotune_dataset_mode",
    )

    if _dataset_mode == "Use current Train & Tune dataset":
        if st.session_state.featured_data is not None:
            if st.button("Use loaded training dataset", key="_autotune_use_train_ds"):
                _set_autotune_dataset(
                    st.session_state.featured_data.copy(),
                    processed_df=st.session_state.get("train_processed_data"),
                    dataset_meta=dict(st.session_state.get("train_dataset_meta") or {}),
                )
                st.success("Current training dataset copied into the autotune workspace.")
        else:
            st.warning("No training dataset is loaded yet. Prepare one on Train & Tune, load the latest featured dataset, or build fresh here.")
    elif _dataset_mode == "Load latest featured dataset":
        _global_featured = _global_featured_dataset_path()
        if _global_featured and os.path.exists(_global_featured):
            if st.button("Load latest featured dataset from disk", key="_autotune_load_global"):
                _featured = _cached_load_df(_global_featured, os.path.getmtime(_global_featured))
                _set_autotune_dataset(
                    _featured,
                    dataset_meta=_dataset_meta_from_frame(
                        _featured,
                        data_source="disk",
                        requested_days=None,
                        featured_path=_global_featured,
                        origin="autotune_global_featured",
                    ),
                )
                st.success("Loaded the latest featured dataset from disk.")
        else:
            st.warning("No global featured dataset file found yet.")
    else:
        _ad1, _ad2 = st.columns([2, 2])
        with _ad1:
            _at_source = st.selectbox("Data source", ["database", "scrape", "sample"], key="_autotune_source")
        with _ad2:
            if _at_source == "sample":
                _at_num_races = st.slider("Sample races", 500, 5000, 1500, 100, key="_autotune_num_races")
                _at_days_back = None
            else:
                _at_num_races = 1500
                _at_days_back = st.slider("Days of history", 1, 2000, 90, 7, key="_autotune_days_back")
        if st.button("📦 Prepare autotune dataset", type="secondary", key="_autotune_prepare"):
            _prep = st.progress(0, text="Preparing autotune dataset …")
            _prep.progress(10, text="Collecting raw data …")
            _featured, _processed, _meta = build_autotune_dataset(
                data_source=_at_source,
                days_back=_at_days_back,
                num_races=_at_num_races,
            )
            _prep.progress(100, text="Dataset ready")
            _set_autotune_dataset(_featured, processed_df=_processed, dataset_meta=_meta)
            st.success(f"Prepared {_meta.get('rows', 0):,} rows for autotuning.")

    _at_featured = st.session_state.get("autotune_featured_data")
    _at_meta = st.session_state.get("autotune_dataset_meta") or {}
    if isinstance(_at_featured, pd.DataFrame):
        st.success(
            f"Autotune dataset ready: {_at_meta.get('rows', len(_at_featured)):,} rows · "
            f"{_at_meta.get('months') or '—'} months · {_at_meta.get('date_start') or '?'} → {_at_meta.get('date_end') or '?'}"
        )
    else:
        st.info("Load or prepare a dataset before launching a study.")

    st.markdown("---")
    st.subheader("2️⃣ Study Setup")
    _task_models = {"classifier": "Win Classifier", "place": "Place Classifier"}
    if getattr(config, "TRAIN_RANKER", False):
        _task_models["ranker"] = "Race Ranker"
        st.info(
            "The dedicated Autotune page can now tune the race ranker as its own LightGBM LambdaRank study. "
            "Because the objective family changed, older autotune sessions cannot be resumed into this build."
        )
    else:
        st.caption(
            "ℹ️ The diagnostic Race Ranker is disabled, so no ranker study is offered. "
            "Enable it via the Train page toggle (or `config.TRAIN_RANKER`) to tune it."
        )
    _at_name = st.text_input(
        "Study name",
        value=f"autotune_{datetime.now():%m%d_%H%M}",
        help="Used for the persisted study folder name.",
    )
    _at_models = st.multiselect(
        "Models to tune",
        options=list(_task_models.keys()),
        default=list(_task_models.keys()),
        format_func=lambda key: _task_models.get(key, key),
        key="_autotune_models",
    )
    _setup_c1, _setup_c2 = st.columns(2)
    with _setup_c1:
        _at_trials = st.slider("Trials per model", 1, 200, 40, 5, key="_autotune_trials")
    with _setup_c2:
        _at_folds = st.slider("Purged walk-forward folds", 1, 5, 2, 1, key="_autotune_folds")
    st.caption(
        "Autotune always optimises the **form-only** model — odds are never "
        "model features. The market-odds blend is a train-time choice on the "
        "**Train & Tune** page, not part of hyperparameter search."
    )
    st.caption("When starting a new session this is the initial trial count per model. When resuming, it adds this many extra trials per model.")
    if _at_folds == 1:
        st.caption("Single split: the training window is used as-is with a purged validation window at the end. Faster but less robust.")
    else:
        st.caption("Optuna scores each trial by averaging the objective over purged walk-forward folds built from the outer training split.")

    _at_frameworks: dict[str, str] = {}
    _fw_cols = st.columns(len(_task_models))
    _fw_defaults = dict(getattr(config, "SUB_MODEL_FRAMEWORKS", {}))
    for _idx, (_mk, _label) in enumerate(_task_models.items()):
        with _fw_cols[_idx]:
            if _mk == "ranker":
                _at_frameworks[_mk] = st.selectbox(
                    _label,
                    options=["lgbm"],
                    index=0,
                    key=f"_autotune_fw_{_mk}",
                    help="The ranker uses LightGBM LambdaRank.",
                )
            else:
                _def_fw = _fw_defaults.get(_mk, "lgbm")
                _at_frameworks[_mk] = st.selectbox(
                    _label,
                    options=["lgbm", "xgb", "cat"],
                    index=["lgbm", "xgb", "cat"].index(_def_fw) if _def_fw in ["lgbm", "xgb", "cat"] else 0,
                    key=f"_autotune_fw_{_mk}",
                )

    st.markdown("#### Search Space")
    st.caption(
        "These are the exact hyperparameters Optuna will test for the selected model/framework combinations. Fixed rows are inherited from config and are not searched."
    )

    _space_tabs = st.tabs([_task_models[k] for k in _task_models])
    for _tab, _mk in zip(_space_tabs, _task_models):
        with _tab:
            _fw = _at_frameworks.get(_mk, "lgbm")
            _space_specs = get_autotune_search_space(_mk, _fw, include_recency=True)
            _rows = []
            for _spec in _space_specs:
                _kind = _spec.get("kind")
                if _kind == "fixed":
                    _range = str(_spec.get("value"))
                    _dist = "fixed"
                elif _kind == "categorical":
                    _range = ", ".join(str(c) for c in _spec.get("choices", []))
                    _dist = "categorical"
                else:
                    _low = _spec.get("low")
                    _high = _spec.get("high")
                    _step = _spec.get("step")
                    _range = f"{_low} → {_high}"
                    if _step is not None:
                        _range += f" (step {_step})"
                    _dist = "log-uniform" if _spec.get("log") else "uniform"
                    if _kind == "int":
                        _dist = f"int {_dist}"
                _rows.append({
                    "Parameter": _spec.get("name"),
                    "Type": _kind,
                    "Distribution": _dist,
                    "Range / Value": _range,
                    "Notes": _spec.get("note", ""),
                })

            st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
            _objective_label = AUTOTUNE_MODEL_INFO.get(_mk, {}).get("training_metric", "Hybrid objective")
            _objective_summary = AUTOTUNE_MODEL_INFO.get(_mk, {}).get("objective_summary", "")
            if _at_folds == 1:
                st.caption(f"Objective for this model: {_objective_label} on a single purged validation split.")
            else:
                st.caption(f"Objective for this model: mean {_objective_label} across {_at_folds} purged walk-forward folds.")
            if _objective_summary:
                st.caption(_objective_summary)

    _progress_box = st.empty()
    _progress_bar = st.progress(0, text="Idle")

    _run_col1, _run_col2 = st.columns([1, 1])
    with _run_col1:
        _start_new = st.button("🚀 Start new autotune session", type="primary", width="stretch", key="_autotune_start")
    with _run_col2:
        _resume_options = [s.get("session_id") for s in _at_sessions]
        _resume_target = st.selectbox(
            "Resume existing session",
            options=[""] + _resume_options,
            format_func=lambda sid: "Select a session" if sid == "" else next((f"{s.get('name')} ({sid})" for s in _at_sessions if s.get('session_id') == sid), sid),
            key="_autotune_resume_target",
        )
        _resume_clicked = st.button("▶️ Resume selected session", width="stretch", key="_autotune_resume")

    def _render_autotune_progress(stage: str, payload: dict) -> None:
        payload = payload or {}
        if stage == "setup":
            _progress_bar.progress(0.05, text="Preparing purged walk-forward folds")
            _progress_box.info(
                f"{payload.get('message', 'Preparing autotune splits')} · target folds {payload.get('target_folds', _at_folds)} · +{payload.get('requested_trials', _at_trials)} trials/model"
            )
            return

        if stage == "model_start":
            _model_key = payload.get("model_key")
            _model_index = int(payload.get("model_index", 1))
            _model_total = int(payload.get("model_total", 1))
            _overall = 0.05 + ((_model_index - 1) / max(_model_total, 1)) * 0.90
            _progress_bar.progress(
                min(max(_overall, 0.0), 0.94),
                text=f"{AUTOTUNE_MODEL_INFO.get(_model_key, {}).get('label', _model_key)} · starting trials",
            )
            _progress_box.info(
                f"Starting {AUTOTUNE_MODEL_INFO.get(_model_key, {}).get('label', _model_key)} · model {_model_index}/{_model_total} · CV folds {payload.get('cv_folds', _at_folds)}"
            )
            return

        if stage == "trial":
            _model_key = payload.get("model_key")
            _model_index = int(payload.get("model_index", 1))
            _model_total = int(payload.get("model_total", 1))
            _trial_num = int(payload.get("trial_num", 0))
            _trial_total = int(payload.get("trial_total", 1))
            _score = float(payload.get("score", 0.0))
            _overall = 0.05 + (((_model_index - 1) + (_trial_num / max(_trial_total, 1))) / max(_model_total, 1)) * 0.90
            _progress_bar.progress(
                min(max(_overall, 0.0), 0.95),
                text=f"{AUTOTUNE_MODEL_INFO.get(_model_key, {}).get('label', _model_key)} · trial {_trial_num}/{_trial_total}",
            )
            _progress_box.info(
                f"Running {AUTOTUNE_MODEL_INFO.get(_model_key, {}).get('label', _model_key)} · model {_model_index}/{_model_total} · fold-avg objective {_score:.4f} across {payload.get('cv_folds', _at_folds)} folds"
            )
            return

        if stage == "complete":
            _progress_bar.progress(1.0, text="Autotune complete")
            _progress_box.success(
                f"Autotune finished · {len(payload.get('models', []))} models · {payload.get('cv_folds', _at_folds)} folds"
            )
            return

    if _start_new:
        if not isinstance(_at_featured, pd.DataFrame):
            st.error("Prepare an autotune dataset first.")
            st.stop()
        if not _at_models:
            st.error("Select at least one model to tune.")
            st.stop()
        _at_featured_run = _at_featured.copy()
        # Odds-derived columns are excluded by get_feature_columns, so the
        # study is always form-only — no explicit drop needed.
        _new_session = create_autotune_session(
            name=_at_name.strip() or f"autotune_{datetime.now():%Y%m%d_%H%M%S}",
            dataset_meta={**_at_meta, "include_odds": False},
            frameworks=_at_frameworks,
            models=_at_models,
            n_trials=int(_at_trials),
            n_folds=int(_at_folds),
        )
        _manifest = run_autotune_session(
            session_id=_new_session["session_id"],
            featured_df=_at_featured_run,
            frameworks=_at_frameworks,
            models=_at_models,
            n_trials=int(_at_trials),
            n_folds=int(_at_folds),
            progress_callback=_render_autotune_progress,
        )
        st.session_state["_autotune_focus_session_id"] = _manifest.get("session_id")
        _progress_bar.progress(1.0, text="Autotune complete")
        _progress_box.success(f"Saved autotune session {_manifest.get('name')} ({_manifest.get('session_id')}).")
        st.rerun()

    if _resume_clicked and _resume_target:
        if not isinstance(_at_featured, pd.DataFrame):
            st.error("Load the dataset that matches the session you want to resume.")
            st.stop()
        _resume_manifest = load_autotune_session(_resume_target)
        if _resume_manifest is None:
            st.error("Selected autotune session could not be loaded.")
            st.stop()
        _at_featured_run = _at_featured.copy()
        # Form-only by construction (odds excluded in get_feature_columns).
        _manifest = run_autotune_session(
            session_id=_resume_target,
            featured_df=_at_featured_run,
            frameworks=dict(_resume_manifest.get("frameworks") or {}),
            models=list(_resume_manifest.get("models") or []),
            n_trials=int(_at_trials),
            n_folds=int(_resume_manifest.get("target_folds") or _at_folds),
            progress_callback=_render_autotune_progress,
        )
        st.session_state["_autotune_focus_session_id"] = _manifest.get("session_id")
        _progress_bar.progress(1.0, text="Autotune resume complete")
        _progress_box.success(f"Resumed autotune session {_manifest.get('name')} ({_manifest.get('session_id')}).")
        st.rerun()

    st.markdown("---")
    st.subheader("3️⃣ Saved Sessions")
    _at_sessions = list_autotune_sessions()
    if not _at_sessions:
        st.info("No autotune sessions saved yet.")
    else:
        _session_lookup = {_session.get("session_id"): _session for _session in _at_sessions}
        _focus_session_id = st.session_state.pop("_autotune_focus_session_id", None)
        if _focus_session_id in _session_lookup:
            st.session_state["_autotune_session_picker"] = _focus_session_id
        elif st.session_state.get("_autotune_session_picker") not in _session_lookup:
            st.session_state["_autotune_session_picker"] = _at_sessions[0].get("session_id")

        _selected_session_id = st.selectbox(
            "Saved sessions",
            options=list(_session_lookup.keys()),
            format_func=lambda sid: (
                f"{_session_lookup[sid].get('name', sid)} · {_session_lookup[sid].get('status', 'unknown')} · "
                f"{(_session_lookup[sid].get('dataset_meta') or {}).get('data_source', '?')} "
                f"{((_session_lookup[sid].get('dataset_meta') or {}).get('actual_days') or '—')}d · "
                f"{_session_lookup[sid].get('updated_at', '')[:16]}"
            ),
            key="_autotune_session_picker",
        )
        _selected_session = _session_lookup[_selected_session_id]
        _sel_meta = _selected_session.get("dataset_meta") or {}
        _s1, _s2, _s3, _s4 = st.columns(4)
        _s1.metric("Status", _selected_session.get("status", "—"))
        _s2.metric("Rows", _sel_meta.get("rows", 0))
        _s3.metric("Months", _sel_meta.get("months") or "—")
        _s4.metric("Target trials", _selected_session.get("target_trials", "—"))

        _split_summary = _selected_session.get("split_summary") or {}
        if _split_summary:
            st.markdown("#### Split Summary")
            _ss1, _ss2, _ss3, _ss4, _ss5 = st.columns(5)
            _ss1.metric("Outer train races", _split_summary.get("outer_train_races", "—"))
            _ss2.metric("Outer test races", _split_summary.get("outer_test_races", "—"))
            _ss3.metric("Outer train rows", _split_summary.get("outer_train_rows", "—"))
            _ss4.metric("Outer test rows", _split_summary.get("outer_test_rows", "—"))
            _ss5.metric("CV folds", _split_summary.get("cv_folds", _selected_session.get("target_folds", "—")))

        _fold_summaries = _selected_session.get("cv_fold_summaries") or []
        if _fold_summaries:
            with st.expander("Walk-forward fold breakdown", expanded=False):
                st.dataframe(pd.DataFrame(_fold_summaries), width="stretch", hide_index=True)

        with st.expander("Best parameter snippet", expanded=False):
            st.code(build_config_snippet(_selected_session) or "No completed params yet.", language="python")

        _summary_rows = []
        for _mk, _summary in (_selected_session.get("summaries") or {}).items():
            _summary_rows.append({
                "Model": AUTOTUNE_MODEL_INFO.get(_mk, {}).get("label", _mk),
                "Framework": _summary.get("framework"),
                "Metric": _summary.get("metric_name"),
                "Best Score": _summary.get("best_score"),
                "Completed Trials": _summary.get("n_trials"),
                "Target Trials": _summary.get("target_trials"),
                "CV Folds": _summary.get("cv_folds"),
            })
        if _summary_rows:
            st.dataframe(pd.DataFrame(_summary_rows), width="stretch", hide_index=True)

        # ── Delete session ──────────────────────────────────────
        _del_col1, _del_col2 = st.columns([3, 1])
        with _del_col2:
            if st.button("🗑️ Delete session", key="_autotune_delete_session", type="secondary"):
                st.session_state["_confirm_delete_autotune"] = _selected_session.get("session_id")
        if st.session_state.get("_confirm_delete_autotune") == _selected_session.get("session_id"):
            st.warning(f"Are you sure you want to delete **{_selected_session.get('name', _selected_session.get('session_id'))}**? This cannot be undone.")
            _cd1, _cd2, _ = st.columns([1, 1, 4])
            with _cd1:
                if st.button("Yes, delete", key="_autotune_confirm_delete", type="primary"):
                    delete_autotune_session(_selected_session["session_id"])
                    st.session_state.pop("_confirm_delete_autotune", None)
                    st.success("Session deleted.")
                    st.rerun()
            with _cd2:
                if st.button("Cancel", key="_autotune_cancel_delete"):
                    st.session_state.pop("_confirm_delete_autotune", None)
                    st.rerun()

        _available_models = [m for m in (_selected_session.get("models") or []) if (_selected_session.get("summaries") or {}).get(m)]
        if _available_models:
            _viz_model = st.selectbox(
                "Visualise model",
                options=_available_models,
                format_func=lambda key: AUTOTUNE_MODEL_INFO.get(key, {}).get("label", key),
                key="_autotune_viz_model",
            )
            try:
                _study = load_optuna_study(_selected_session["session_id"], _viz_model)
                import optuna

                _trial_df = _study.trials_dataframe().drop(
                    columns=[c for c in _study.trials_dataframe().columns if c.startswith("system_attrs_")],
                    errors="ignore",
                )
                _completed_trials = _study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)
                _shared_search_space = optuna.search_space.intersection_search_space(_completed_trials)
                _shared_params = set(_shared_search_space.keys())
                _param_names = sorted({key for trial in _completed_trials for key in trial.params.keys()})
                _varying_params = []
                _constant_params = []
                for _param_name in _param_names:
                    _values = {
                        trial.params[_param_name]
                        for trial in _completed_trials
                        if _param_name in trial.params
                    }
                    if len(_values) >= 2:
                        _varying_params.append(_param_name)
                    else:
                        _constant_params.append(_param_name)
                _stable_varying_params = [p for p in _varying_params if p in _shared_params]
                _dynamic_only_params = [p for p in _varying_params if p not in _shared_params]

                _best_attrs = {
                    k: v for k, v in (_study.best_trial.user_attrs or {}).items()
                    if isinstance(v, (int, float))
                }
                _objective_name = _study.user_attrs.get("objective_name") or AUTOTUNE_MODEL_INFO.get(_viz_model, {}).get("training_metric", "Objective")

                st.markdown(f"#### {AUTOTUNE_MODEL_INFO.get(_viz_model, {}).get('label', _viz_model)} Study")
                _ov1, _ov2, _ov3, _ov4 = st.columns(4)
                _ov1.metric("Objective", _objective_name)
                _ov2.metric("Direction", str(_study.user_attrs.get("objective_direction", "maximize")).title())
                _ov3.metric("Trials", len(_study.trials))
                _ov4.metric("Best Score", f"{_study.best_value:.4f}")
                if _best_attrs:
                    _attrs_df = pd.DataFrame(
                        [{"Metric": _metric_label(k) if _is_diagnostic_metric(k) else k.replace("_", " ").title(), "Value": v} for k, v in _best_attrs.items()]
                    )
                    with st.expander("Best trial components", expanded=False):
                        st.dataframe(_attrs_df, width="stretch", hide_index=True)

                _vt1, _vt2 = st.tabs(["Charts", "Trials"])
                with _vt1:
                    st.plotly_chart(optuna.visualization.plot_optimization_history(_study), width="stretch")
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        if _stable_varying_params:
                            st.plotly_chart(
                                optuna.visualization.plot_param_importances(_study, params=_stable_varying_params),
                                width="stretch",
                            )
                        else:
                            st.info("Parameter importance becomes available once at least one shared tuned parameter varies across completed trials.")
                    with _c2:
                        if _stable_varying_params:
                            st.plotly_chart(
                                optuna.visualization.plot_parallel_coordinate(_study, params=_stable_varying_params),
                                width="stretch",
                            )
                        else:
                            st.info("Parallel-coordinate plots need at least one parameter with multiple tested values.")
                    if _constant_params:
                        st.caption(
                            "Omitted constant parameters from charts: "
                            + ", ".join(_constant_params[:12])
                            + (" ..." if len(_constant_params) > 12 else "")
                        )
                    if _dynamic_only_params:
                        st.caption(
                            "Omitted dynamic-space parameters from Optuna charts: "
                            + ", ".join(_dynamic_only_params[:12])
                            + (" ..." if len(_dynamic_only_params) > 12 else "")
                        )
                    if len(_trial_df) >= 2:
                        _c3, _c4 = st.columns(2)
                        with _c3:
                            if _stable_varying_params:
                                st.plotly_chart(optuna.visualization.plot_slice(_study, params=_stable_varying_params), width="stretch")
                        with _c4:
                            try:
                                if len(_stable_varying_params) >= 2:
                                    st.plotly_chart(
                                        optuna.visualization.plot_contour(_study, params=_stable_varying_params[:6]),
                                        width="stretch",
                                    )
                            except Exception:
                                pass
                        _c5, _c6 = st.columns(2)
                        with _c5:
                            try:
                                st.plotly_chart(optuna.visualization.plot_edf(_study), width="stretch")
                            except Exception:
                                pass
                        with _c6:
                            try:
                                st.plotly_chart(optuna.visualization.plot_timeline(_study), width="stretch")
                            except Exception:
                                pass
                with _vt2:
                    st.dataframe(_trial_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.warning(f"Could not render Optuna visualisations for this study: {exc}")


# =====================================================================
#  EXPERIMENTS  (Run Manager)
# =====================================================================
elif page == "🧪 Experiments":
    st.title("🧪 Run Manager")
    st.caption(
        "Every training run is automatically saved with its full "
        "metrics, equity curves, calibration data and bet log. "
        "Select a run to explore, or delete runs you no longer need."
    )

    saved_runs = list_runs()

    if not saved_runs:
        st.info(
            "No saved runs yet. Train a model on the "
            "**🎓 Train & Tune** page to create your first run."
        )
        st.stop()

    # ── Run summary table ────────────────────────────────────────────
    run_rows = []
    for r in saved_runs:
        ta = r.get("test_analysis", {})
        ta_stats = ta.get("stats", {})
        tp = ta_stats.get("top_pick", {})
        vb = ta_stats.get("value", {})
        ew = ta_stats.get("each_way", {})
        m = r.get("metrics", {})
        tc = r.get("training_config", {}) if isinstance(r.get("training_config", {}), dict) else {}
        _fw = tc.get("frameworks", {}) if isinstance(tc.get("frameworks", {}), dict) else {}
        _tune_mode = tc.get("tuning_mode")
        _auto_trials = tc.get("auto_tune_trials")

        if _tune_mode == "auto":
            _tune_label = f"Auto ({_auto_trials or '—'} trials)"
        elif _tune_mode == "saved":
            _saved_name = tc.get("saved_autotune_session_name") or tc.get("saved_autotune_session_id") or "Saved study"
            _tune_label = f"Saved ({_saved_name})"
        elif _tune_mode == "manual":
            _tune_label = "Manual"
        else:
            _tune_label = "—"

        _fw_label = (
            ", ".join(f"{k}:{v}" for k, v in sorted(_fw.items())) if _fw else "—"
        )
        _lt = tc.get("linear_tree", {}) if isinstance(tc.get("linear_tree", {}), dict) else {}
        _lt_enabled = [k for k, v in sorted(_lt.items()) if v]
        _lt_label = ", ".join(_lt_enabled) if _lt_enabled else "Off"
        _wf_enabled = tc.get("walk_forward")
        if _wf_enabled is True:
            _wf_label = "Yes"
        elif _wf_enabled is False:
            _wf_label = "No"
        else:
            _wf_label = "—"

        # Flatten metrics for headline extraction
        _flat: dict = {}
        if isinstance(m, dict):
            for mk, mv in m.items():
                if isinstance(mv, dict):
                    for kk, vv in mv.items():
                        _flat[f"{mk}/{kk}"] = vv
                else:
                    _flat[mk] = mv

        run_rows.append({
            "Name": r.get("name", r["run_id"]),
            "Date": r.get("timestamp", "")[:16].replace("T", " "),
            "Data": r.get("data_source", "?"),
            "Days": tc.get("days_back"),
            "Burn-In": tc.get("burn_in_months"),
            "Purge": tc.get("purge_days"),
            "Test %": tc.get("test_size_pct"),
            "WF": _wf_label,
            "WF Folds": tc.get("wf_folds"),
            "WF Fast": "On" if tc.get("wf_fast_fold") is True else ("Off" if tc.get("wf_fast_fold") is False else "—"),
            "Mkt Blend": "On" if tc.get("include_odds") is True else ("Off" if tc.get("include_odds") is False else "—"),
            "Tuning": _tune_label,
            "Linear Trees": _lt_label,
            "ES Rounds": tc.get("early_stopping_rounds"),
            "Frameworks": _fw_label,
            "Rows": r.get("data_rows", 0),
            "TP ROI%": tp.get("roi"),
            "Value ROI%": vb.get("roi"),
            "EW ROI%": ew.get("roi"),
            "Time (s)": r.get("elapsed_seconds", 0),
            "run_id": r["run_id"],
            "_flat_metrics": _flat,
        })

    run_df = pd.DataFrame(run_rows)
    display_df = run_df.drop(columns=["run_id", "_flat_metrics"], errors="ignore")

    format_dict = {
        "Days": "{:.0f}",
        "Burn-In": "{:.0f}",
        "Purge": "{:.0f}",
        "Test %": "{:.0f}",
        "WF Folds": "{:.0f}",
        "ES Rounds": "{:.0f}",
        "Rows": "{:,.0f}",
        "NDCG@1": "{:.3f}",
        "Top-1 Acc": "{:.1%}",
        "TP ROI%": "{:+.1f}",
        "Value ROI%": "{:+.1f}",
        "EW ROI%": "{:+.1f}",
        "Time (s)": "{:.1f}",
    }

    def _run_table_gradient(col_series, lower_better=False):
        """Return a list of background-color CSS strings for a column."""
        vals = pd.to_numeric(col_series, errors="coerce")
        valid = vals.dropna()
        if valid.empty or valid.nunique() < 2:
            return [""] * len(vals)
        lo, hi = valid.min(), valid.max()
        styles = []
        for v in vals:
            if pd.isna(v):
                styles.append("")
                continue
            t = (v - lo) / (hi - lo)   # 0 = worst raw, 1 = best raw
            if lower_better:
                t = 1.0 - t             # flip so 1 = best
            if t < 0.5:
                r, g = 220, int(80 + 140 * (t / 0.5))
            else:
                r, g = int(220 - 180 * ((t - 0.5) / 0.5)), 220
            styles.append(f"background-color: rgba({r},{g},60,0.28)")
        return styles

    def _roi_gradient(col_series):
        """Green for positive ROI, red for negative — absolute magnitude."""
        styles = []
        for v in pd.to_numeric(col_series, errors="coerce"):
            if pd.isna(v):
                styles.append("")
            elif v > 0:
                intensity = min(abs(v) / 20, 1.0)   # saturate at +20%
                styles.append(f"background-color: rgba(40,180,60,{0.12 + 0.28 * intensity:.2f})")
            elif v < 0:
                intensity = min(abs(v) / 20, 1.0)
                styles.append(f"background-color: rgba(220,50,50,{0.12 + 0.28 * intensity:.2f})")
            else:
                styles.append("")
        return styles

    def _style_run_table(styler):
        if "NDCG@1" in display_df.columns:
            styler = styler.apply(_run_table_gradient, lower_better=False, subset=["NDCG@1"])
        if "Top-1 Acc" in display_df.columns:
            styler = styler.apply(_run_table_gradient, lower_better=False, subset=["Top-1 Acc"])
        if "TP ROI%" in display_df.columns:
            styler = styler.apply(_roi_gradient, subset=["TP ROI%"])
        if "Value ROI%" in display_df.columns:
            styler = styler.apply(_roi_gradient, subset=["Value ROI%"])
        if "EW ROI%" in display_df.columns:
            styler = styler.apply(_roi_gradient, subset=["EW ROI%"])
        styler = styler.format(format_dict, na_rep="—")
        return styler

    st.dataframe(
        _style_run_table(display_df.style),
        width="stretch",
        hide_index=True,
    )

    # ── Color-coded full metric comparison ─────────────────────────
    st.markdown("#### 🎨 Color-Coded Performance Comparison")
    st.caption(
        "Per-model metrics across runs with best/worst highlighting. "
        "Green = better, red = worse (direction-aware for RPS/Brier)."
    )

    _metric_rows = []
    for _r in run_rows:
        _row = {
            "Name": _r.get("Name"),
            "Date": _r.get("Date"),
        }
        _flat = _r.get("_flat_metrics", {}) if isinstance(_r.get("_flat_metrics"), dict) else {}
        for _k, _v in _flat.items():
            if (
                isinstance(_v, (int, float))
                and pd.notna(_v)
                and "total_races" not in str(_k).lower()
            ):
                _row[_k] = float(_v)
        _metric_rows.append(_row)

    _all_metrics_df = pd.DataFrame(_metric_rows)
    _fixed_cols = [c for c in ["Name", "Date"] if c in _all_metrics_df.columns]
    _metric_cols = [
        c for c in _all_metrics_df.columns
        if c not in _fixed_cols and pd.api.types.is_numeric_dtype(_all_metrics_df[c])
    ]

    if _metric_cols:
        # Stable ordering: Win Clf → Place Clf, core metrics first
        _priority = [
            "win_classifier/rps",
            "win_classifier/brier_score",
            "win_classifier/ndcg_at_1",
            "win_classifier/top1_accuracy",
            "win_classifier/value_bet_roi",
            "win_classifier/value_bet_sr",
            "win_classifier/avg_edge",
            "ranker/rps",
            "ranker/brier_score",
            "ranker/ndcg_at_1",
            "ranker/top1_accuracy",
            "place_classifier/brier_calibrated",
            "place_classifier/brier_raw",
            "place_classifier/top3_accuracy",
            "place_classifier/place_precision",
        ]
        _ordered_metrics = [c for c in _priority if c in _metric_cols] + sorted(
            [c for c in _metric_cols if c not in _priority]
        )
        _all_metrics_df = _all_metrics_df[_fixed_cols + _ordered_metrics]

        def _is_lower_better(_col: str) -> bool:
            _cl = _col.lower()
            return ("rps" in _cl and "/" in _cl) or ("brier" in _cl) or ("mae" in _cl) or ("loss" in _cl) or ("drawdown" in _cl)

        def _cell_color(_val, _col, _vals):
            if _val is None or (isinstance(_val, float) and np.isnan(_val)):
                return ""
            _valid = [v for v in _vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if len(_valid) < 2:
                return ""
            _lo, _hi = min(_valid), max(_valid)
            if _hi == _lo:
                return ""
            _t = (_val - _lo) / (_hi - _lo)
            if _is_lower_better(_col):
                _t = 1.0 - _t
            if _t < 0.5:
                _r, _g = 220, int(80 + 140 * (_t / 0.5))
            else:
                _r, _g = int(220 - 180 * ((_t - 0.5) / 0.5)), 220
            return f"background-color: rgba({_r},{_g},60,0.25)"

        def _metric_formatter(_col: str):
            _cl = _col.lower()
            if any(k in _cl for k in ["roi", "acc", "accuracy", "strike", "_sr", "avg_edge"]):
                return lambda v: "—" if pd.isna(v) else f"{v:+.2%}"
            if "ndcg" in _cl:
                return lambda v: "—" if pd.isna(v) else f"{v:.4f}"
            if any(k in _cl for k in ["bets", "races", "runners"]):
                return lambda v: "—" if pd.isna(v) else f"{int(round(v)):,}"
            return lambda v: "—" if pd.isna(v) else f"{v:.4f}"

        _styler = _all_metrics_df.style
        for _mc in _ordered_metrics:
            _vals = _all_metrics_df[_mc].tolist()
            _styler = _styler.map(
                lambda v, _c=_mc, _cv=_vals: _cell_color(v, _c, _cv),
                subset=[_mc],
            )
        _fmt = {_mc: _metric_formatter(_mc) for _mc in _ordered_metrics}
        _styler = _styler.format(_fmt)

        st.dataframe(
            _styler,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No numeric run metrics found to compare yet.")

    # ── Select a run to inspect ──────────────────────────────────────
    st.markdown("---")
    run_options = {
        f"{r.get('name', r['run_id'])}  ({r.get('timestamp', '')[:16].replace('T', ' ')})": r["run_id"]
        for r in saved_runs
    }
    selected_label = st.selectbox(
        "Select a run to inspect",
        list(run_options.keys()),
    )
    selected_run_id = run_options[selected_label]

    # ── Quick-activate button (restore this run's model) ─────────────
    _is_active = st.session_state.active_run_id == selected_run_id
    _has_model = run_has_model(selected_run_id)

    col_act, col_info = st.columns([1, 3])
    with col_act:
        if _is_active:
            st.success("✅ Active")
        elif _has_model:
            if st.button("🔄 Activate This Model", type="primary", key="restore_model"):
                with st.spinner("Restoring model from run snapshot …"):
                    if restore_run_model(selected_run_id):
                        # Bust cached model so it reloads from disk
                        _cached_load_model.clear()
                        load_existing_model()
                        # Restore metrics / test_analysis from this run
                        _restored = load_run(selected_run_id)
                        st.session_state.active_run_id = selected_run_id
                        st.session_state.metrics = _restored.get("metrics")
                        _rta = _restored.get("test_analysis", {})
                        if _rta:
                            st.session_state.test_analysis = {
                                "bets": _restored.get("bets_df", pd.DataFrame()),
                                "curves": _restored.get("curves_df", pd.DataFrame()),
                                "stats": _rta.get("stats", {}),
                                "calibration": _rta.get("calibration", []),
                                "test_date_range": _rta.get("test_date_range", ("?", "?")),
                                "test_races": _rta.get("test_races", 0),
                                "test_runners": _rta.get("test_runners", 0),
                                "value_config": _rta.get("value_config", {}),
                            }
                        st.session_state["_pending_model_switch"] = selected_run_id
                        _invalidate_run_caches()
                        st.success(f"Model from run **{selected_run_id}** is now active!")
                        st.rerun()
                    else:
                        st.error("Failed to restore model.")
        else:
            st.caption("🚫 No model snapshot (older run)")
    with col_info:
        if not _is_active and _has_model:
            st.caption(
                "Clicking **Activate** restores this run's model as the "
                "active model for all predictions."
            )

    tab_overview, tab_equity, tab_bets, tab_hp, tab_delete = st.tabs(
        ["📊 Overview", "📈 Equity & Calibration", "📋 Bet Log", "🔧 Hyperparameters", "🗑️ Manage"]
    )

    # Load full run data
    try:
        run_data = load_run(selected_run_id)
    except FileNotFoundError:
        st.error("Run data not found on disk.")
        st.stop()

    r_meta = run_data
    r_ta = r_meta.get("test_analysis", {})
    r_stats = r_ta.get("stats", {})
    r_bets = run_data.get("bets_df")
    r_curves = run_data.get("curves_df")
    r_wf_summary = run_data.get("wf_summary_df")
    r_wf_curves = run_data.get("wf_curves_df")

    # ── Overview tab ─────────────────────────────────────────────────
    with tab_overview:
        st.subheader(f"📊 {r_meta.get('name', selected_run_id)}")
        with st.expander("✏️ Rename this run", expanded=False):
            _new_run_name = st.text_input(
                "New name",
                value=r_meta.get("name", selected_run_id),
                key=f"rename_input_{selected_run_id}",
            )
            if st.button("💾 Save name", key=f"rename_btn_{selected_run_id}"):
                if rename_run(selected_run_id, _new_run_name):
                    _invalidate_run_caches()
                    st.success(f"Renamed to **{_new_run_name}**")
                    st.rerun()
                else:
                    st.error("Rename failed — run not found.")
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Data Rows", f"{r_meta.get('data_rows', 0):,}")
        ic2.metric("Features", r_meta.get("n_features", 0))
        ic3.metric("Train Time", f"{r_meta.get('elapsed_seconds', 0):.1f}s")
        _dr = r_ta.get("test_date_range", ("?", "?"))
        ic4.metric("Test Period", f"{_dr[0]} → {_dr[1]}" if _dr[0] != "?" else "—")

        _tc_over = r_meta.get("training_config", {}) if isinstance(r_meta.get("training_config", {}), dict) else {}
        _fw_over = _tc_over.get("frameworks", {}) if isinstance(_tc_over.get("frameworks", {}), dict) else {}
        if _fw_over:
            st.caption(f"Frameworks: {', '.join(f'{k}:{v}' for k, v in sorted(_fw_over.items()))}")

        # Strategy summary
        tp = r_stats.get("top_pick", {})
        vb = r_stats.get("value", {})
        ewb_r = r_stats.get("each_way", {})
        if tp or vb or ewb_r:
            st.markdown("#### 🎯 Strategy Performance")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**Top Pick (1 bet / race)**")
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Bets", tp.get("bets", 0))
                t2.metric("Strike Rate", f"{tp.get('strike_rate', 0):.1f}%")
                t3.metric("ROI", f"{tp.get('roi', 0):+.1f}%")
                t4.metric("P&L", f"{tp.get('pnl', 0):+.1f}")
            with s2:
                st.markdown("**Value Bets**")
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Bets", vb.get("bets", 0))
                v2.metric("Strike Rate", f"{vb.get('strike_rate', 0):.1f}%")
                v3.metric("ROI", f"{vb.get('roi', 0):+.1f}%")
                v4.metric("P&L", f"{vb.get('pnl', 0):+.1f}")
                _avg_clv = vb.get("avg_clv")
                _evc = r_ta.get("value_config", {})
                _clv_str = f" · Avg CLV **{_avg_clv:.3f}x**" if _avg_clv is not None else ""
                if _evc.get("staking_mode") == "kelly":
                    st.caption(
                        f"Kelly {_evc.get('kelly_fraction', 0.25):.0%} · "
                        f"Avg stake {vb.get('avg_stake', 0):.2f} · "
                        f"Total staked {vb.get('total_staked', 0):.1f}{_clv_str}"
                    )
                elif _clv_str:
                    st.caption(_clv_str.strip(" ·"))
                _vb_diag = []
                if vb.get("avg_edge") is not None:
                    _vb_diag.append(f"Avg edge **{vb.get('avg_edge', 0):+.3f}**")
                if vb.get("expected_roi") is not None:
                    _vb_diag.append(f"Exp ROI **{vb.get('expected_roi', 0):+.1f}%**")
                if vb.get("selected_brier") is not None:
                    _vb_diag.append(f"Sel Brier **{vb.get('selected_brier', 0):.4f}**")
                if _vb_diag:
                    st.caption(" · ".join(_vb_diag))
            with s3:
                st.markdown("**Each-Way Bets**")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Bets", ewb_r.get("bets", 0))
                e2.metric("Place Rate", f"{ewb_r.get('place_rate', 0):.1f}%")
                e3.metric("ROI", f"{ewb_r.get('roi', 0):+.1f}%")
                e4.metric("P&L", f"{ewb_r.get('pnl', 0):+.1f}")
                if ewb_r.get("bets", 0):
                    st.caption(
                        f"Won {ewb_r.get('winners', 0)} · "
                        f"Placed {ewb_r.get('placed', 0)} / {ewb_r.get('bets', 0)}"
                    )

        # Full metrics
        r_metrics = r_meta.get("metrics", {})
        if r_metrics:
            with st.expander("📊 Full Metrics"):
                if isinstance(r_metrics, dict) and any(
                    isinstance(v, dict) for v in r_metrics.values()
                ):
                    mdf = pd.DataFrame(r_metrics).T
                    _num_cols = mdf.select_dtypes("number").columns.tolist()
                    st.dataframe(
                        mdf.style.format("{:.4f}", subset=_num_cols),
                        width="stretch",
                    )
                else:
                    st.json(r_metrics)

    # ── Equity & Calibration tab ─────────────────────────────────────
    with tab_equity:
        if r_curves is not None and not r_curves.empty:
            _r_hover = [c for c in ["bet_number", "horse_name", "odds", "stake", "pnl"] if c in r_curves.columns]
            _evc2 = r_ta.get("value_config", {})
            _ek = _evc2.get("staking_mode") == "kelly"
            _eq_title = (
                f"Cumulative P&L (Kelly {_evc2.get('kelly_fraction', 0.25):.0%})"
                if _ek else "Cumulative P&L (flat 1-unit stakes)"
            )
            # Per-point markers over thousands of bets render tens of
            # thousands of SVG nodes and hang the browser; only show them
            # for short curves.
            _eq_markers = len(r_curves) <= 400
            eq1, eq2 = st.columns(2)
            with eq1:
                fig_pnl = px.line(
                    r_curves, x="race_date", y="cum_pnl",
                    color="strategy", markers=_eq_markers,
                    title=_eq_title,
                    hover_data=_r_hover,
                    labels={"race_date": "Date", "cum_pnl": "P&L (units)", "bet_number": "Bet #"},
                )
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="grey")
                fig_pnl.update_layout(height=400)
                st.plotly_chart(fig_pnl, width="stretch")
            with eq2:
                fig_roi = px.line(
                    r_curves, x="race_date", y="cum_roi_pct",
                    color="strategy", markers=_eq_markers,
                    title="Cumulative ROI %",
                    hover_data=_r_hover,
                    labels={"race_date": "Date", "cum_roi_pct": "ROI (%)", "bet_number": "Bet #"},
                )
                fig_roi.add_hline(y=0, line_dash="dash", line_color="grey")
                fig_roi.update_layout(height=400)
                st.plotly_chart(fig_roi, width="stretch")
        else:
            st.info("No equity curve data for this run.")

        # ── Walk-Forward Validation Results ──────────────────────────
        if r_wf_summary is not None and not r_wf_summary.empty:
            st.markdown("---")
            st.markdown("#### 🔄 Walk-Forward Validation")

            # Summary metrics
            _wf_n = len(r_wf_summary)
            _wf_cols = st.columns(7)
            _wf_labels = [
                ("Folds", _wf_n, "d"),
                ("Avg Brier", r_wf_summary["brier_score"].mean(), ".4f"),
                ("Avg NDCG@1", r_wf_summary["ndcg_at_1"].mean(), ".3f"),
                ("Avg Top-1", r_wf_summary["top1_accuracy"].mean(), ".1%"),
                ("Top Pick P&L", r_wf_summary["top_pick_pnl"].sum(), "+.2f"),
                ("Value P&L", r_wf_summary["value_pnl"].sum(), "+.2f"),
                ("EW P&L", r_wf_summary["ew_pnl"].sum(), "+.2f"),
            ]
            for _wc, (_wl, _wv, _wfmt) in zip(_wf_cols, _wf_labels):
                _wc.metric(_wl, f"{_wv:{_wfmt}}")

            # Per-fold table
            with st.expander("Per-fold breakdown", expanded=False):
                _wf_display = r_wf_summary.copy()
                _fmt_map = {
                    "brier_score": "{:.4f}", "ndcg_at_1": "{:.3f}",
                    "top1_accuracy": "{:.1%}", "win_in_top3": "{:.1%}",
                    "top_pick_pnl": "{:+.2f}", "value_pnl": "{:+.2f}", "ew_pnl": "{:+.2f}",
                }
                for _fc, _ff in _fmt_map.items():
                    if _fc in _wf_display.columns:
                        _wf_display[_fc] = _wf_display[_fc].apply(lambda v, f=_ff: f.format(v))
                st.dataframe(_wf_display, hide_index=True, width="stretch")

            # WF equity curves
            if r_wf_curves is not None and not r_wf_curves.empty:
                _wf_markers = len(r_wf_curves) <= 400
                wf_eq1, wf_eq2 = st.columns(2)
                with wf_eq1:
                    _wf_hover = [c for c in ["bet_number", "horse_name", "odds", "pnl"] if c in r_wf_curves.columns]
                    fig_wf_pnl = px.line(
                        r_wf_curves, x="race_date", y="cum_pnl",
                        color="strategy", markers=_wf_markers,
                        title="Walk-Forward Cumulative P&L",
                        hover_data=_wf_hover,
                        labels={"race_date": "Date", "cum_pnl": "P&L (units)"},
                    )
                    fig_wf_pnl.add_hline(y=0, line_dash="dash", line_color="grey")
                    fig_wf_pnl.update_layout(height=400)
                    st.plotly_chart(fig_wf_pnl, width="stretch")
                with wf_eq2:
                    fig_wf_roi = px.line(
                        r_wf_curves, x="race_date", y="cum_roi_pct",
                        color="strategy", markers=_wf_markers,
                        title="Walk-Forward Cumulative ROI %",
                        hover_data=_wf_hover,
                        labels={"race_date": "Date", "cum_roi_pct": "ROI (%)"},
                    )
                    fig_wf_roi.add_hline(y=0, line_dash="dash", line_color="grey")
                    fig_wf_roi.update_layout(height=400)
                    st.plotly_chart(fig_wf_roi, width="stretch")

        # Value bets by odds band
        band_data = r_stats.get("value_by_odds_band")
        if band_data:
            st.markdown("#### 🎰 Value Bets by Odds Band")
            band_df = pd.DataFrame(band_data)
            bc1, bc2 = st.columns(2)
            with bc1:
                fig_br = px.bar(
                    band_df, x="odds_band", y="roi",
                    title="ROI % by Odds Range", text="roi",
                    color="roi",
                    color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
                    color_continuous_midpoint=0,
                )
                fig_br.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_br.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_br, width="stretch")
            with bc2:
                fig_bs = px.bar(
                    band_df, x="odds_band", y="strike_rate",
                    title="Strike Rate % by Odds Range", text="strike_rate",
                    color="strike_rate", color_continuous_scale="Blues",
                )
                fig_bs.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_bs.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bs, width="stretch")
            st.dataframe(band_df, hide_index=True, width="stretch")

        # Calibration
        cal_data = r_ta.get("calibration")
        if cal_data:
            st.markdown("#### 🎯 Model Calibration")
            cal_df = pd.DataFrame(cal_data)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Bar(
                x=cal_df["prob_bucket"], y=cal_df["avg_model_pct"],
                name="Model Prob %", marker_color="#3b82f6",
            ))
            fig_cal.add_trace(go.Bar(
                x=cal_df["prob_bucket"], y=cal_df["actual_win_rate"],
                name="Actual Win %", marker_color="#22c55e",
            ))
            fig_cal.update_layout(
                barmode="group",
                title="Predicted Probability vs Actual Win Rate",
                xaxis_title="Model Probability Bucket",
                yaxis_title="%", height=380,
            )
            st.plotly_chart(fig_cal, width="stretch")

        # Daily P&L
        daily_data = r_stats.get("top_pick_daily")
        if daily_data:
            st.markdown("#### 📅 Daily P&L (Top Pick)")
            daily_df = pd.DataFrame(daily_data)
            fig_daily = px.bar(
                daily_df, x="race_date", y="daily_pnl",
                title="Daily P&L", color="daily_pnl",
                color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"],
                color_continuous_midpoint=0,
                labels={"race_date": "Date", "daily_pnl": "P&L (units)"},
            )
            fig_daily.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_daily, width="stretch")

    # ── Bet Log tab ──────────────────────────────────────────────────
    with tab_bets:
        if r_bets is not None and not r_bets.empty:
            st.subheader(f"📋 Full Bet Log ({len(r_bets)} bets)")
            strat_filter = st.multiselect(
                "Strategy filter",
                r_bets["strategy"].unique().tolist(),
                default=r_bets["strategy"].unique().tolist(),
                key="run_bets_strat",
            )
            filtered_bets = r_bets[r_bets["strategy"].isin(strat_filter)].copy()
            filtered_bets["pnl"] = filtered_bets["pnl"].apply(lambda v: f"{v:+.2f}")
            st.dataframe(filtered_bets, hide_index=True, width="stretch")
        else:
            st.info("No bet data for this run.")

    # ── Hyperparameters tab ──────────────────────────────────────────
    with tab_hp:
        st.subheader("🔧 Hyperparameters Used")
        hp_data = r_meta.get("hyperparameters", {})
        if hp_data:
            hp_df = pd.DataFrame([hp_data]).T
            hp_df.columns = ["Value"]
            st.dataframe(hp_df, width="stretch")
        else:
            st.info("No hyperparameters recorded.")

        at_data = r_meta.get("auto_tune")
        if at_data:
            st.markdown("#### 🔍 Optuna Auto-Tune")
            a1, a2 = st.columns(2)
            a1.metric("Trials", at_data.get("n_trials", "?"))
            a2.metric("Best Score", f"{at_data.get('best_score', 0):.4f}")

    # ── Manage tab (delete) ──────────────────────────────────────────
    with tab_delete:
        st.subheader("🗑️ Delete Runs")
        st.warning(
            "Deleted runs cannot be recovered. The model file itself "
            "is **not** deleted — only the run's metrics and analysis data."
        )

        del_options = {
            f"{r.get('name', r['run_id'])}  ({r.get('timestamp', '')[:16]})": r["run_id"]
            for r in saved_runs
        }
        del_selection = st.multiselect(
            "Select runs to delete",
            list(del_options.keys()),
            key="del_runs",
        )

        _del_col1, _del_col2 = st.columns([1, 3])
        with _del_col1:
            _do_delete = st.button(
                f"🗑️ Delete ({len(del_selection)})" if del_selection else "🗑️ Delete",
                type="primary",
                disabled=len(del_selection) == 0,
                key="btn_delete_runs",
            )
        with _del_col2:
            if not del_selection:
                st.caption("Select one or more runs above to delete.")

        if _do_delete and del_selection:
            for label in del_selection:
                rid = del_options[label]
                delete_run(rid)
                if st.session_state.active_run_id == rid:
                    st.session_state.active_run_id = None
                    st.session_state.test_analysis = None
                    st.session_state.metrics = None
            st.success(f"Deleted {len(del_selection)} run(s).")
            st.rerun()

        # ── Disk usage + prune old runs ───────────────────────────────
        st.markdown("---")
        st.subheader("💽 Disk Usage & Pruning")
        _usage = run_disk_usage()
        _total_gb = sum(b for _, b in _usage) / 1024 ** 3
        st.caption(
            f"**{len(_usage)} run(s)** using **{_total_gb:.2f} GB** "
            "(dataset snapshots dominate — pruning keeps the newest runs)."
        )
        _pr_col1, _pr_col2 = st.columns([1, 1])
        with _pr_col1:
            _keep_n = st.number_input(
                "Keep newest N runs", min_value=1, max_value=50, value=5,
                key="prune_keep_n",
            )
        with _pr_col2:
            _n_prunable = max(0, len(_usage) - int(_keep_n))
            _do_prune = st.button(
                f"🧹 Prune {_n_prunable} old run(s)",
                disabled=_n_prunable == 0,
                key="btn_prune_runs",
            )
        if _do_prune:
            _pruned = prune_runs(keep_latest=int(_keep_n))
            list_runs.clear()
            load_run.clear()
            load_run_meta.clear()
            if st.session_state.active_run_id in _pruned:
                st.session_state.active_run_id = None
                st.session_state.test_analysis = None
                st.session_state.metrics = None
            st.success(f"Pruned {len(_pruned)} run(s).")
            st.rerun()

        # ── Processed-data snapshot migration ────────────────────────
        st.markdown("---")
        st.subheader("📦 Backfill Dataset Snapshots")
        _runs_missing_snap = [
            r for r in saved_runs
            if get_run_processed_path(r["run_id"]) is None
        ]
        if not _runs_missing_snap:
            st.success(
                f"✅ All {len(saved_runs)} run(s) have a processed-data snapshot. "
                "No migration needed."
            )
        else:
            st.info(
                f"**{len(_runs_missing_snap)} run(s)** were saved before per-run dataset "
                "snapshots were introduced. Without a snapshot, their feature engineering "
                "context shifts every time you retrain on a different date range, causing "
                "prediction drift.\n\n"
                "Clicking below stamps the **current** `processed_races.parquet` into each "
                "of those runs as a fixed baseline. It won't restore their original training "
                "data, but it will stop further drift.",
                icon="ℹ️",
            )
            _global_pq = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
            _global_csv = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.csv")
            _global_proc = _global_pq if os.path.exists(_global_pq) else (_global_csv if os.path.exists(_global_csv) else None)
            if _global_proc is None:
                st.warning("No global processed_races file found — train a model first to generate it.")
            else:
                if st.button(
                    f"📦 Backfill {len(_runs_missing_snap)} run(s)",
                    type="primary",
                    key="btn_backfill_snapshots",
                ):
                    import shutil as _shutil
                    _ext = ".parquet" if _global_proc.endswith(".parquet") else ".csv"
                    _dest_fname = f"processed_races{_ext}"
                    _patched = 0
                    from src.run_store import RUNS_DIR as _RUNS_DIR
                    for _mr in _runs_missing_snap:
                        _run_dir = os.path.join(_RUNS_DIR, _mr["run_id"])
                        try:
                            _shutil.copy2(_global_proc, os.path.join(_run_dir, _dest_fname))
                            _patched += 1
                        except Exception as _e:
                            st.warning(f"Skipped {_mr['run_id']}: {_e}")
                    st.success(f"✅ Backfilled {_patched} run(s) with the current processed dataset.")
                    st.rerun()

    # ── Side-by-side run comparison ────────────────────────────────────
    if len(saved_runs) >= 2:
        st.markdown("---")
        st.subheader("🔀 Compare Runs Side-by-Side")
        st.caption("Select two runs to compare their metrics and equity curves.")

        _cmp_options = {
            f"{r.get('name', r['run_id'])}  ({r.get('timestamp', '')[:16].replace('T', ' ')})": r["run_id"]
            for r in saved_runs
        }
        _cmp_keys = list(_cmp_options.keys())

        _cc1, _cc2 = st.columns(2)
        with _cc1:
            _cmp_a_label = st.selectbox("Run A", _cmp_keys, index=0, key="cmp_a")
        with _cc2:
            _cmp_b_label = st.selectbox("Run B", _cmp_keys, index=min(1, len(_cmp_keys) - 1), key="cmp_b")

        _cmp_a_id = _cmp_options[_cmp_a_label]
        _cmp_b_id = _cmp_options[_cmp_b_label]

        if _cmp_a_id != _cmp_b_id:
            try:
                _ra = load_run(_cmp_a_id)
                _rb = load_run(_cmp_b_id)
            except Exception:
                _ra, _rb = None, None

            if _ra and _rb:
                # ── Metrics comparison table ──────────────────────
                _ma = _ra.get("metrics", {})
                _mb = _rb.get("metrics", {})
                if _ma and _mb and isinstance(_ma, dict) and isinstance(_mb, dict):
                    _all_models = sorted(set(list(_ma.keys()) + list(_mb.keys())))
                    _cmp_rows = []
                    for _mk in _all_models:
                        _va = _ma.get(_mk, {})
                        _vb = _mb.get(_mk, {})
                        if not isinstance(_va, dict) or not isinstance(_vb, dict):
                            continue
                        for _metric in ["ndcg_at_1", "ndcg_at_3", "top1_accuracy", "win_in_top3"]:
                            _a_val = _va.get(_metric)
                            _b_val = _vb.get(_metric)
                            if _a_val is not None and _b_val is not None:
                                _diff = float(_b_val) - float(_a_val)
                                _cmp_rows.append({
                                    "Sub-Model": _mk.replace("_", " ").title(),
                                    "Metric": _metric,
                                    f"Run A": round(float(_a_val), 4),
                                    f"Run B": round(float(_b_val), 4),
                                    "Δ (B−A)": round(_diff, 4),
                                })
                    if _cmp_rows:
                        _cmp_df = pd.DataFrame(_cmp_rows)
                        # Color the delta column
                        st.dataframe(
                            _cmp_df.style.format(
                                {"Run A": "{:.4f}", "Run B": "{:.4f}", "Δ (B−A)": "{:+.4f}"}
                            ).applymap(
                                lambda v: "color: #22c55e" if isinstance(v, (int, float)) and v > 0
                                else ("color: #ef4444" if isinstance(v, (int, float)) and v < 0 else ""),
                                subset=["Δ (B−A)"],
                            ),
                            width="stretch", hide_index=True,
                        )

                # ── Strategy comparison ───────────────────────────
                _ta_a = _ra.get("test_analysis", {}).get("stats", {})
                _ta_b = _rb.get("test_analysis", {}).get("stats", {})
                if _ta_a and _ta_b:
                    _s1, _s2, _s3 = st.columns(3)
                    for _col, _label, _strat in [(_s1, "Top Pick", "top_pick"), (_s2, "Value Bets", "value"), (_s3, "Each-Way", "each_way")]:
                        with _col:
                            _sa = _ta_a.get(_strat, {})
                            _sb = _ta_b.get(_strat, {})
                            if _sa and _sb:
                                st.markdown(f"**{_label}**")
                                _m1, _m2, _m3 = st.columns(3)
                                _roi_a = _sa.get("roi", 0)
                                _roi_b = _sb.get("roi", 0)
                                _m1.metric("ROI A", f"{_roi_a:+.1f}%")
                                _m2.metric("ROI B", f"{_roi_b:+.1f}%")
                                _m3.metric("Δ", f"{_roi_b - _roi_a:+.1f}%")

                # ── Equity curve overlay ──────────────────────────
                _ca = _ra.get("curves_df")
                _cb = _rb.get("curves_df")
                if _ca is not None and _cb is not None and not _ca.empty and not _cb.empty:
                    st.markdown("#### 📈 Equity Curve Overlay")
                    _ca_plot = _ca.copy()
                    _cb_plot = _cb.copy()
                    _ca_plot["run"] = _cmp_a_label[:30]
                    _cb_plot["run"] = _cmp_b_label[:30]
                    _overlay = pd.concat([_ca_plot, _cb_plot], ignore_index=True)
                    _overlay["label"] = _overlay["strategy"] + " — " + _overlay["run"]

                    _ov_hover = [c for c in ["bet_number", "horse_name", "odds", "stake", "pnl"] if c in _overlay.columns]
                    fig_overlay = px.line(
                        _overlay, x="race_date", y="cum_pnl",
                        color="label", markers=len(_overlay) <= 400,
                        title="Cumulative P&L Comparison",
                        hover_data=_ov_hover,
                        labels={"race_date": "Date", "cum_pnl": "P&L (units)", "bet_number": "Bet #"},
                    )
                    fig_overlay.add_hline(y=0, line_dash="dash", line_color="grey")
                    fig_overlay.update_layout(height=450)
                    st.plotly_chart(fig_overlay, width="stretch")
        else:
            st.info("Select two **different** runs to compare.")

    # ── Legacy experiment comparison (from experiments.json) ─────────
    experiments = _load_experiments()
    if experiments:
        st.markdown("---")
        with st.expander("📜 Legacy Experiment Log"):
            leg_rows = []
            for exp in experiments:
                row = {
                    "Name": exp["name"],
                    "Model": exp["model_type"],
                    "Rows": exp.get("data_rows", 0),
                    "Time (s)": exp.get("elapsed_seconds", 0),
                }
                m = exp.get("metrics", {})
                for key in ("ndcg_at_1", "top1_accuracy"):
                    found = [v for k, v in m.items() if key in k and v is not None]
                    if found:
                        row[key] = max(found)
                row["Timestamp"] = exp.get("timestamp", "")[:19]
                leg_rows.append(row)
            st.dataframe(
                pd.DataFrame(leg_rows),
                width="stretch",
                hide_index=True,
            )
            if st.button("🗑️ Clear Legacy Log", type="secondary"):
                _save_experiments([])
                st.rerun()


# =====================================================================
#  TODAY'S PICKS
# =====================================================================
elif page == "💰 Today's Picks":
    st.title("💰 Value Picks")

    # ── Date picker ──────────────────────────────────────────────────
    _tp_dc1, _tp_dc2 = st.columns([2, 3])
    with _tp_dc1:
        _picks_date = st.date_input(
            "📅 Race date",
            value=datetime.now().date(),
            key="picks_race_date",
        )
    _picks_date_str = _picks_date.strftime("%Y-%m-%d")
    _picks_is_today = _picks_date == datetime.now().date()

    with _tp_dc2:
        st.caption(
            "Scrapes racecards for the selected date, runs the win/place "
            "models on every race, and surfaces horses where the model "
            "sees genuine value."
        )

    # ── ensure model is loaded ───────────────────────────────────────
    if st.session_state.predictor is None:
        if os.path.exists(_ENSEMBLE_MODEL_PATH):
            load_existing_model()
            load_model_data(force=True)
        else:
            st.warning(
                "⚠️ No model available. Train one on the "
                "**Train & Tune** page first."
            )
            st.stop()

    # ── Active model info ────────────────────────────────────────
    _pred = st.session_state.predictor
    _model_name = type(_pred).__name__
    _w = getattr(_pred, "weights", None)
    _rid = st.session_state.get("active_run_id")

    with st.expander("ℹ️ Active Model", expanded=False):
        mi1, mi2 = st.columns([1, 2])
        with mi1:
            st.markdown(f"**Type:** {_model_name}")
            if _rid:
                st.markdown(f"**Run:** `{_rid}`")
            else:
                st.markdown("**Run:** _not tracked_")
        with mi2:
            _fw_info = getattr(_pred, "frameworks", {})
            if _fw_info:
                parts = [f"{k}: {v}" for k, v in sorted(_fw_info.items())]
                st.markdown(f"**Frameworks:** {' · '.join(parts)}")
        st.caption(
            "This is the model used for all picks below. "
            "To switch, go to **\U0001f9ea Experiments** and click "
            "**\U0001f504 Activate This Model** on any saved run."
        )

    # ── settings bar ─────────────────────────────────────────────
    _tp_vc = st.session_state.value_config
    _tp_mode = _tp_vc["staking_mode"]
    _tp_label = (
        f"Fractional Kelly ({_tp_vc['kelly_fraction']:.0%}) · "
        f"Bankroll {_tp_vc['bankroll']:.0f}"
        if _tp_mode == "kelly" else "Flat 1-unit"
    )
    st.info(
        f"📐 **Value Strategy:** {_tp_label} · "
        f"Threshold {_tp_vc['value_threshold']:.2f}  \n"
        f"_Change in the **⚙️ Value Strategy** section of the sidebar._"
    )
    value_base_thresh = _tp_vc["value_threshold"]
    st.caption(
        "**How it works:** For each horse the threshold is "
        r"$T = T_{\text{base}} \times \sqrt{\text{odds}/3}$ "
        "— so a 2/1 shot needs a smaller edge than a 20/1 shot."
    )

    st.markdown("---")

    # Detect date change → clear stale picks
    if st.session_state.get("_picks_date_prev") != _picks_date_str:
        st.session_state["picks_cards"] = None
        st.session_state.pop("picks_preds", None)
        st.session_state.pop("picks_meta", None)
        st.session_state["_picks_date_prev"] = _picks_date_str

    # Ensure session-state dicts exist
    if "picks_featured" not in st.session_state:
        st.session_state["picks_featured"] = {}
    if "picks_featured_meta" not in st.session_state:
        st.session_state["picks_featured_meta"] = {}
    if "picks_explanations" not in st.session_state:
        st.session_state["picks_explanations"] = {}

    # ── Refresh Odds button ──────────────────────────────────────
    _odds_refresh = st.button(
        "🔄 Refresh Odds",
        type="secondary",
        key="btn_picks_refresh_odds",
        help="Re-scrape racecards from Sporting Life for latest odds.",
    )

    # ── Auto-load racecards (first visit or date change) ─────────
    if (
        "picks_cards" not in st.session_state
        or st.session_state["picks_cards"] is None
        or _odds_refresh
    ):
        with st.spinner(f"{'Refreshing' if _odds_refresh else 'Loading'} racecards for {_picks_date_str} …"):
            cards_df = get_scraped_racecards(
                date_str=_picks_date_str,
                force_refresh=_odds_refresh,
            )
        if cards_df is not None and not cards_df.empty:
            st.session_state["picks_cards"] = cards_df
            if _odds_refresh:
                # Odds changed — invalidate predictions so they re-run
                st.session_state.pop("picks_preds", None)
                st.session_state.pop("picks_meta", None)
        else:
            st.warning(
                f"No racecards found for {_picks_date_str}. "
                "Cards may not be published this far in advance."
            )
            st.stop()

    if (
        "picks_cards" not in st.session_state
        or st.session_state["picks_cards"] is None
    ):
        st.stop()

    cards_df = st.session_state["picks_cards"]

    # Fingerprint of the current model state — detect model changes
    _picks_model_fp = (
        st.session_state.get("active_run_id", ""),
    )
    _cached_pick_preds = st.session_state.get("picks_preds")
    _cached_pick_feat = st.session_state.get("picks_featured", {}).get(_picks_date_str)
    _has_preds = "picks_preds" in st.session_state and st.session_state["picks_preds"] is not None
    _preds_stale = _has_preds and st.session_state.get("picks_model_fp") != _picks_model_fp
    _pace_stale = (_has_preds and not _has_pace_diagnostics(_cached_pick_preds)) or (
        _cached_pick_feat is not None and not _cached_pick_feat.empty and not _has_pace_diagnostics(_cached_pick_feat)
    )

    # ── Determine if predictions need to run ─────────────────────
    _needs_preds = not _has_preds or _preds_stale or _odds_refresh or _pace_stale

    if _needs_preds:
        cards_df = cards_df.reset_index(drop=True)

        # ── Load featured data from the cache hierarchy ──────────
        # The predictions page must NEVER feature-engineer on the fly:
        # Prepare Data / Train pre-computes features for the next 7 days
        # (the lookahead cache) over the identical training code path.
        # Each tier records why it missed, so a total miss is surfaced as
        # a loud, actionable error rather than a silent dead end.
        _feat_cache = st.session_state.get("picks_featured", {})
        _has_lookahead = lookahead_cache_valid(_picks_date_str, current_cards_sig=None)

        all_feat = None
        _feat_source = None
        _cache_diag: list[str] = []
        progress = st.progress(10, text="Loading features …")

        # 1. In-memory cache (this session)
        if _odds_refresh:
            _cache_diag.append("in-memory cache: skipped (odds refresh forces a reload)")
        elif _picks_date_str in _feat_cache:
            if _has_pace_diagnostics(_feat_cache[_picks_date_str]):
                all_feat = _feat_cache[_picks_date_str]
                _feat_source = "in-memory cache (this session)"
                progress.progress(50, text="⚡ Using cached features …")
            else:
                _cache_diag.append("in-memory cache: present but predates pace diagnostics")
        else:
            _cache_diag.append("in-memory cache: nothing for this date this session")

        # 2. Lookahead cache (built by Prepare Data / Train, 7 days ahead)
        if all_feat is None:
            if _has_lookahead:
                _lookahead_feat = load_lookahead_cache(_picks_date_str, current_cards_sig=None)
                if _lookahead_feat is None or _lookahead_feat.empty:
                    _cache_diag.append("lookahead cache: file present but empty/unreadable")
                elif not _has_pace_diagnostics(_lookahead_feat):
                    _cache_diag.append("lookahead cache: present but predates pace diagnostics — rebuild needed")
                else:
                    all_feat = _lookahead_feat
                    _feat_source = "lookahead cache (7-day pre-build)"
                    progress.progress(50, text="⚡ Loaded lookahead cache …")
            else:
                _cache_diag.append(
                    "lookahead cache: no entry for this date "
                    "(outside the 7-day build window, or Prepare Data not run yet)"
                )

        # 3. Live feature cache on disk — scan for any cached file for this date
        if all_feat is None:
            _date_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache", _picks_date_str)
            _pq_files = []
            if os.path.isdir(_date_cache_dir):
                _pq_files = sorted(
                    (f for f in os.scandir(_date_cache_dir) if f.name.endswith(".parquet")),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
            if _pq_files:
                try:
                    _loaded = pd.read_parquet(_pq_files[0].path, engine="pyarrow")
                    if _loaded.empty or not _has_pace_diagnostics(_loaded):
                        _cache_diag.append("live feature cache: file present but empty or predates pace diagnostics")
                    else:
                        all_feat = _loaded
                        _feat_source = "live feature cache (disk fallback)"
                        progress.progress(50, text="⚡ Loaded live feature cache …")
                except Exception as _lfc_exc:
                    _cache_diag.append(f"live feature cache: failed to read ({_lfc_exc})")
            else:
                _cache_diag.append("live feature cache: no file on disk for this date")

        # 4. Global featured dataset (today/past only)
        if all_feat is None:
            if _picks_date > datetime.now().date():
                _cache_diag.append(
                    "global featured dataset: skipped (date is in the future — "
                    "only the lookahead cache covers future dates)"
                )
            else:
                _gfp = _global_featured_dataset_path()
                if not _gfp:
                    _cache_diag.append("global featured dataset: none built yet")
                else:
                    progress.progress(20, text="Loading featured dataset …")
                    _full_feat = _cached_load_df(_gfp, os.path.getmtime(_gfp))
                    if _full_feat is None or "race_date" not in _full_feat.columns:
                        _cache_diag.append("global featured dataset: unreadable or missing race_date")
                    else:
                        _date_mask = (
                            pd.to_datetime(_full_feat["race_date"], errors="coerce")
                            .dt.strftime("%Y-%m-%d") == _picks_date_str
                        )
                        _date_slice = _full_feat.loc[_date_mask]
                        if _date_slice.empty:
                            _cache_diag.append("global featured dataset: no rows for this date")
                        else:
                            all_feat = _date_slice.copy()
                            _feat_source = "global featured dataset"
                            progress.progress(50, text="⚡ Loaded from featured dataset …")

        if all_feat is None or all_feat.empty:
            progress.empty()
            st.error(
                f"🚩 **No featured data found for {_picks_date_str}.**\n\n"
                "This page never feature-engineers on the fly — it reads the "
                "cache that **Prepare Data / Train** builds (which pre-computes "
                "features for the next 7 days). A total miss means the cache is "
                "missing, stale, or this date is outside the build window."
            )
            with st.expander("🔍 Why each cache tier missed", expanded=True):
                for _d in _cache_diag:
                    st.caption(f"• {_d}")
                st.markdown(
                    "**Fix:** run **🎓 Train & Tune → Prepare Data** to rebuild "
                    "the featured dataset and refresh the 7-day lookahead cache, "
                    "then reload this page."
                )
            st.stop()

        st.session_state["picks_feat_source"] = _feat_source

        # ── Merge fresh odds from racecards ──────────────────────
        _match_cols = ["race_id", "horse_name"]
        if all(c in all_feat.columns for c in _match_cols) and all(c in cards_df.columns for c in _match_cols):
            # Align dtypes so the merge doesn't fail on int vs object
            all_feat["race_id"] = all_feat["race_id"].astype(str)
            cards_df["race_id"] = cards_df["race_id"].astype(str)
            _current_keys = cards_df[_match_cols].drop_duplicates()
            all_feat = all_feat.merge(_current_keys, on=_match_cols, how="inner")
            if "odds" in cards_df.columns:
                _odds_map = cards_df[_match_cols + ["odds"]].drop_duplicates(subset=_match_cols)
                all_feat = all_feat.drop(columns=["odds"], errors="ignore")
                all_feat = all_feat.merge(_odds_map, on=_match_cols, how="left")

        # Cache in memory
        st.session_state.setdefault("picks_featured", {})[_picks_date_str] = all_feat

        # ── Coerce numeric columns that may have arrived as object ──
        _numeric_cols = [
            "distance_furlongs", "prize_money", "num_runners", "age",
            "weight_lbs", "draw", "horse_runs", "odds", "finish_position",
            "finish_time_secs", "lengths_behind",
        ]
        for _nc in _numeric_cols:
            if _nc in all_feat.columns and all_feat[_nc].dtype == object:
                all_feat[_nc] = pd.to_numeric(all_feat[_nc], errors="coerce")

        # ── Run predictions ──────────────────────────────────────
        progress.progress(70, text="Running predictions …")

        race_meta_df = (
            cards_df.groupby("race_id", sort=False)
            .agg(
                track=("track", "first"),
                off_time=("off_time", "first"),
                race_name=("race_name", "first"),
                runners=("race_id", "size"),
            )
            .reset_index()
        )
        race_meta = race_meta_df.to_dict("records")
        _skip_reasons: list[str] = []

        try:
            if hasattr(st.session_state.predictor, "predict_races") and "race_id" in all_feat.columns:
                progress.progress(100, text=f"Predicting {race_meta_df.shape[0]} races …")
                full_preds = _predict_featured_frame(
                    st.session_state.predictor,
                    all_feat,
                    ew_fraction=st.session_state.value_config.get("ew_fraction"),
                )
            else:
                all_preds: list[pd.DataFrame] = []
                race_ids = all_feat["race_id"].unique() if "race_id" in all_feat.columns else cards_df["race_id"].unique()
                for idx, rid in enumerate(race_ids):
                    progress.progress(
                        70 + int(30 * (idx + 1) / len(race_ids)),
                        text=f"Predicting race {idx + 1}/{len(race_ids)} …",
                    )
                    feat_slice = all_feat[all_feat["race_id"] == rid].copy() if "race_id" in all_feat.columns else pd.DataFrame()
                    if feat_slice.empty:
                        _skip_reasons.append(f"race {rid}: feature slice empty after engineering")
                        continue
                    preds = _predict_featured_frame(
                        st.session_state.predictor,
                        feat_slice,
                        ew_fraction=st.session_state.value_config.get("ew_fraction"),
                    )
                    preds["race_id"] = rid
                    all_preds.append(preds)
                full_preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
        except Exception as e:
            _skip_reasons.append(str(e))
            full_preds = pd.DataFrame()

        progress.empty()

        if full_preds.empty:
            st.error("Could not analyse any races.")
            if _skip_reasons:
                with st.expander("🔍 Failure details", expanded=True):
                    for _r in _skip_reasons:
                        st.caption(f"• {_r}")
            st.stop()

        st.session_state["picks_preds"] = full_preds
        st.session_state["picks_meta"] = race_meta
        st.session_state["picks_thresh"] = value_base_thresh
        # Record which model state produced these predictions
        st.session_state["picks_model_fp"] = _picks_model_fp
        st.session_state["picks_model_label"] = "Win + Place models"

    # ── display results ─────────────────────────────────────────
    if "picks_preds" in st.session_state and st.session_state["picks_preds"] is not None:
        # Warn if predictions were made with a different model than what's currently active
        _current_fp = (
            st.session_state.get("active_run_id", ""),
        )
        if st.session_state.get("picks_model_fp") != _current_fp:
            # Model changed — predictions will auto-refresh on next rerun
            st.rerun()
        else:
            _used_label = st.session_state.get("picks_model_label", "Win + Place models")
            _feat_src = st.session_state.get("picks_feat_source")
            _src_str = f" · features from **{_feat_src}**" if _feat_src else ""
            st.caption(f"ℹ️ Predictions computed with: **{_used_label}**{_src_str}")

        full_preds = st.session_state["picks_preds"]
        race_meta = st.session_state["picks_meta"]
        base_thresh = st.session_state.value_config["value_threshold"]

        # ── fetch actual race results ────────────────────────────
        _fetch_results = st.button(
            "🏁 Fetch Results",
            type="secondary",
            key="btn_fetch_results",
            help="Scrape race results to settle completed races.",
        )
        if _fetch_results:
            with st.spinner(f"Scraping results for {_picks_date_str} …"):
                _res_df = scrape_todays_results(date_str=_picks_date_str)
            if _res_df is not None and not _res_df.empty:
                st.session_state["picks_results"] = _res_df
                st.success(
                    f"Fetched results for **{_res_df['race_id'].nunique()}** "
                    f"completed races ({len(_res_df)} runners)"
                )
            else:
                st.info("No results available yet — races may not have started.")

        # Pre-race non-participation codes — known before the off, so bets are void
        _NR_LABELS = {"NR", "W", "WD", "NS", "VOID", "WITHDREW", "NON-RUNNER"}

        # Merge results into predictions if available
        if "picks_results" in st.session_state and st.session_state["picks_results"] is not None:
            _res_df = st.session_state["picks_results"]
            # Build a lookup: (race_id, horse_name) → {finish_position, won, finish_pos_label}
            _res_df["_race_id_str"] = _res_df["race_id"].astype(str)
            _res_lookup = {}
            for _, _rr in _res_df.iterrows():
                _key = (str(_rr["race_id"]), str(_rr["horse_name"]).strip().lower())
                _res_lookup[_key] = {
                    "finish_position": int(_rr.get("finish_position", 0)),
                    "won": int(_rr.get("won", 0)),
                    "lengths_behind": float(_rr.get("lengths_behind", 0.0)),
                    "finish_pos_label": str(_rr.get("finish_pos_label", "") or ""),
                }
            # Settled race IDs (those with results)
            _settled_rids = set(_res_df["_race_id_str"].unique())

            # Map onto predictions
            _fp_list, _won_list, _settled_list, _fpl_list = [], [], [], []
            for _, _pr in full_preds.iterrows():
                _k = (str(_pr["race_id"]), str(_pr["horse_name"]).strip().lower())
                _match = _res_lookup.get(_k)
                if _match and str(_pr["race_id"]) in _settled_rids:
                    _fp_list.append(_match["finish_position"])
                    _won_list.append(_match["won"])
                    _settled_list.append(True)
                    _fpl_list.append(_match["finish_pos_label"])
                else:
                    _fp_list.append(0)
                    _won_list.append(0)
                    _settled_list.append(False)
                    _fpl_list.append("")
            full_preds["result_fp"] = _fp_list
            full_preds["result_won"] = _won_list
            full_preds["is_settled"] = _settled_list
            full_preds["result_fp_label"] = _fpl_list
            full_preds["result_is_nr"] = [
                lbl.upper() in _NR_LABELS for lbl in _fpl_list
            ]
        else:
            full_preds["result_fp"] = 0
            full_preds["result_won"] = 0
            full_preds["is_settled"] = False
            full_preds["result_fp_label"] = ""
            full_preds["result_is_nr"] = False

        # ── identify value bets (odds-dependent threshold) ─────
        if "value_score" in full_preds.columns and "odds" in full_preds.columns:
            full_preds["dyn_threshold"] = dynamic_value_threshold(base_thresh, full_preds["odds"])
            _value_min_odds, _value_max_odds = _value_odds_range(st.session_state.value_config)
            full_preds["is_value"] = _value_bet_mask(full_preds, st.session_state.value_config)
            full_preds["value_odds_in_range"] = full_preds["odds"].between(_value_min_odds, _value_max_odds, inclusive="both")
            value_df = full_preds[full_preds["is_value"]].copy()
        else:
            full_preds["is_value"] = False
            full_preds["value_odds_in_range"] = False
            value_df = pd.DataFrame()

        # ── identify EW value bets ────────────────────────────────
        _tp_ew_cfg = st.session_state.value_config
        if (
            _tp_ew_cfg.get("ew_enabled", True)
            and "ew_value" in full_preds.columns
        ):
            ew_df = ew_value_bets(
                full_preds,
                min_place_edge=_tp_ew_cfg.get("ew_min_place_edge", 0.05),
                min_odds=_tp_ew_cfg.get("ew_min_odds", 4.0),
                max_odds=_tp_ew_cfg.get("ew_max_odds", 51.0),
            )
            full_preds["is_ew_value"] = full_preds.index.isin(ew_df.index)
        else:
            full_preds["is_ew_value"] = False
            ew_df = pd.DataFrame()

        # ── headline banner ───────────────────────────────────────
        _any_settled = full_preds["is_settled"].any()
        _n_settled_races = full_preds.loc[full_preds["is_settled"], "race_id"].nunique() if _any_settled else 0
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("🏟️ Races Analysed", len(race_meta))
        h2.metric("🐴 Total Runners", len(full_preds))
        h3.metric("💰 Win Value Bets", len(value_df))
        h4.metric("🔀 EW Value Bets", len(ew_df))

        # ── settlement summary (only when results are available) ──
        if _any_settled:
            st.markdown("---")

            # --- Value bet returns (exclude NR/void) ---
            _val_settled = value_df[value_df["is_settled"] & ~value_df["result_is_nr"]].copy() if not value_df.empty else pd.DataFrame()
            _val_n = len(_val_settled)
            _val_wins = int(_val_settled["result_won"].sum()) if _val_n else 0
            _val_pnl = 0.0
            for _, _vs in _val_settled.iterrows():
                _val_pnl += settle_win_bet(_vs["odds"], _vs["result_won"])
            _val_roi = (_val_pnl / _val_n * 100) if _val_n else 0.0

            # --- EW bet returns (exclude NR/void) ---
            _ew_settled = ew_df[ew_df["is_settled"] & ~ew_df["result_is_nr"]].copy() if not ew_df.empty else pd.DataFrame()
            _ew_n = len(_ew_settled)
            _ew_wins = 0
            _ew_placed = 0
            _ew_pnl = 0.0
            for _, _es in _ew_settled.iterrows():
                _won = bool(_es["result_won"])
                _placed = bool(ew_placed_flag(_es["result_fp"], _es.get("ew_places", 3)))
                _ew_pnl += settle_ew_bet(
                    _es["odds"], _es.get("place_odds", 0), _won, _placed,
                )
                if _won:
                    _ew_wins += 1
                if _won or _placed:
                    _ew_placed += 1
            _ew_roi = (_ew_pnl / (_ew_n * 2) * 100) if _ew_n else 0.0

            # --- Top pick returns (exclude races where the top pick was NR) ---
            _tp_settled_rids = full_preds.loc[full_preds["is_settled"], "race_id"].unique()
            _tp_n = 0
            _tp_wins = 0
            _tp_pnl = 0.0
            for _tp_rid in _tp_settled_rids:
                _tp_race = full_preds[full_preds["race_id"] == _tp_rid]
                _tp_best = _tp_race.loc[_tp_race["win_probability"].idxmax()]
                # If our top pick was a non-runner, skip this race (void)
                if _tp_best.get("result_is_nr", False):
                    continue
                _tp_n += 1
                _tp_pnl += settle_win_bet(_tp_best["odds"], _tp_best["result_won"])
                if _tp_best["result_won"]:
                    _tp_wins += 1

            # Combined P&L = Value + EW only (top pick excluded)
            _combined_pnl = _val_pnl + _ew_pnl
            _combined_staked = _val_n + _ew_n * 2  # total units risked
            _combined_roi = (_combined_pnl / _combined_staked * 100) if _combined_staked else 0.0

            # Staked amounts per strategy
            _tp_staked = _tp_n       # 1u per top pick
            _val_staked = _val_n     # 1u per value bet
            _ew_staked = _ew_n * 2   # 2u per EW bet (win + place)

            st.markdown(
                f"### 🏁 Settlement — {_n_settled_races} races completed"
            )
            s1, s2, s3, s4 = st.columns(4)
            s1.metric(
                "🎯 Top Pick",
                f"{_tp_wins}/{_tp_n} won · {_tp_staked}u staked",
                f"{_tp_pnl:+.1f}u",
                delta_color="normal" if _tp_pnl >= 0 else "inverse",
            )
            s2.metric(
                "💰 Value Bets",
                f"{_val_wins}/{_val_n} won · {_val_staked}u staked",
                f"{_val_pnl:+.1f}u ({_val_roi:+.1f}% ROI)",
                delta_color="normal" if _val_pnl >= 0 else "inverse",
            )
            s3.metric(
                "🔀 EW Bets",
                f"{_ew_placed}/{_ew_n} placed · {_ew_staked}u staked",
                f"{_ew_pnl:+.1f}u ({_ew_roi:+.1f}% ROI)",
                delta_color="normal" if _ew_pnl >= 0 else "inverse",
            )
            s4.metric(
                "📊 Combined P&L",
                f"{_combined_pnl:+.1f}u · {_combined_staked}u staked",
                f"{_combined_roi:+.1f}% ROI",
                delta_color="normal" if _combined_pnl >= 0 else "inverse",
            )
            st.caption(
                f"Combined = Value + EW only (top pick excluded). "
                f"Value {_val_pnl:+.1f}u | EW {_ew_pnl:+.1f}u"
            )

        st.markdown("---")

        # ── helper: edge colour ──────────────────────────────────
        def _edge_color(edge_pct: float) -> str:
            """Return CSS color for an edge value (in percentage points)."""
            if edge_pct >= 15:
                return "#00c853"   # bright green – huge edge
            if edge_pct >= 8:
                return "#66bb6a"   # green
            if edge_pct >= 3:
                return "#a5d6a7"   # light green
            if edge_pct > 0:
                return "#c8e6c9"   # faint green
            if edge_pct > -3:
                return "#ffcdd2"   # faint red
            return "#ef5350"       # red – big negative edge

        def _edge_html(val: float, fmt: str = "+.1f") -> str:
            """Coloured edge cell."""
            col = _edge_color(val)
            sign = "+" if val >= 0 else ""
            return (
                f"<span style='color:{col};font-weight:600'>"
                f"{sign}{val:{fmt[1:]}}%</span>"
            )

        def _rank_emoji(rank: int) -> str:
            if rank == 1: return "🥇"
            if rank == 2: return "🥈"
            if rank == 3: return "🥉"
            return f"#{rank}"

        # ── shared CSS for race tables ───────────────────────────
        _tbl_css = """
        <style>
        .race-tbl {width:100%;border-collapse:collapse;font-size:0.85rem;margin-bottom:0.3rem}
        .race-tbl th {text-align:left;padding:4px 6px;border-bottom:2px solid #444;
                       font-weight:600;color:#888;font-size:0.75rem;text-transform:uppercase;white-space:nowrap}
        .race-tbl td {padding:5px 6px;border-bottom:1px solid #333;vertical-align:top;white-space:nowrap}
        .race-tbl tr.val-row {background:rgba(0,200,83,0.07)}
        .race-tbl tr.ew-row  {background:rgba(33,150,243,0.07)}
        .race-tbl tr.both-row {background:rgba(255,215,0,0.08)}
        .race-tbl .horse-name {font-weight:700;white-space:normal}
        .race-tbl td.horse-cell {max-width:160px;white-space:normal;word-break:break-word}
        .race-tbl .sub {color:#999;font-size:0.75rem}
        .race-tbl .badge {font-size:0.7rem;padding:1px 5px;border-radius:3px;margin-left:4px}
        .badge-val {background:#1b5e20;color:#c8e6c9}
        .badge-ew  {background:#0d47a1;color:#bbdefb}
        .badge-align {background:#1b5e20;color:#e8f5e9}
        .badge-warn {background:#8d6e63;color:#fff3e0}
        .badge-pace {background:#4e342e;color:#fbe9e7}
        .badge-closer {background:#263238;color:#cfd8dc}
        .pace-summary {font-size:0.82rem;color:#aaa;margin:0.2rem 0 0.5rem 0}
        .settled-won {color:#ffd600;font-weight:700}
        .settled-placed {color:#66bb6a}
        .settled-lost {color:#999}
        .settled-nr {color:#ff9800}
        </style>
        """
        st.markdown(_tbl_css, unsafe_allow_html=True)

        def _build_race_table_html(rows_df: pd.DataFrame, show_result: bool = False, show_race_col: bool = False) -> str:
            """Build an HTML table for a set of runners."""
            # Header
            hdr = (
                "<tr>"
            )
            if show_race_col:
                hdr += "<th>Race</th>"
            hdr += (
                "<th>#</th>"
                "<th>Horse / Jockey</th>"
                "<th>Odds</th>"
                "<th>Model%</th>"
                "<th>Mkt%</th>"
                "<th>Edge</th>"
                "<th>Stake</th>"
                "<th>Place%</th>"
                "<th>Mkt Pl%</th>"
                "<th>EW Edge</th>"
                "<th>EW EV</th>"
                "<th>Pace</th>"
                "<th>Flags</th>"
            )
            if show_result:
                hdr += "<th>Result</th>"
            hdr += "</tr>"

            body = []
            for _, r in rows_df.iterrows():
                _iv = bool(r.get("is_value", False))
                _ie = bool(r.get("is_ew_value", False))
                _row_cls = (
                    "both-row" if _iv and _ie else
                    "val-row" if _iv else
                    "ew-row" if _ie else ""
                )
                rank = int(r["predicted_rank"])
                odds = r["odds"] if pd.notna(r.get("odds")) else None
                model_pct = r["win_probability"] * 100
                mkt_pct = r["implied_prob"] * 100 if pd.notna(r.get("implied_prob")) else None
                edge = r["value_score"] * 100 if pd.notna(r.get("value_score")) else None
                pl_prob = float(r["place_probability"]) * 100 if pd.notna(r.get("place_probability")) else None
                pl_odds = float(r["place_odds"]) if pd.notna(r.get("place_odds")) and float(r.get("place_odds", 0)) > 0 else None
                mkt_pl_pct = (1.0 / pl_odds * 100) if pl_odds else None
                ew_edge = (pl_prob - mkt_pl_pct) if (pl_prob is not None and mkt_pl_pct is not None) else None
                ew_ev = float(r["ew_ev"]) * 100 if pd.notna(r.get("ew_ev")) else None

                # Horse + jockey sub-line
                jockey = str(r.get("jockey", "")) if pd.notna(r.get("jockey")) and str(r.get("jockey")) not in ("", "nan") else ""
                trainer = str(r.get("trainer", "")) if pd.notna(r.get("trainer")) and str(r.get("trainer")) not in ("", "nan") else ""
                sub_parts = []
                if jockey:
                    sub_parts.append(jockey)
                if trainer:
                    sub_parts.append(f"T: {trainer}")
                sub_html = f"<br><span class='sub'>{' · '.join(sub_parts)}</span>" if sub_parts else ""

                # Stake column + Flags / badges
                _is_kelly = _tp_vc.get("staking_mode") == "kelly"
                _bankroll = float(_tp_vc.get("bankroll", 100.0))
                _stake_html = "—"
                flags = []
                if _iv:
                    _kf = kelly_criterion(r['win_probability'], float(odds or 0), fraction=_tp_vc["kelly_fraction"])
                    if _is_kelly and _kf > 0.001:
                        _stake_html = f"{_kf*100:.1f}%"
                    else:
                        _stake_html = "1u"
                    flags.append("<span class='badge badge-val'>💰 Value</span>")
                if _ie:
                    if _is_kelly and odds:
                        _ew_terms = get_ew_terms(int(r.get("num_runners", 0)), float(odds))
                        _ew_k = kelly_ew(
                            float(r.get("win_probability", 0)),
                            float(r.get("place_probability", 0)),
                            float(odds),
                            _ew_terms,
                            fraction=_tp_vc["kelly_fraction"],
                        )
                        _ew_kf = _ew_k.get("ew_kelly", 0.0)
                        if _ew_kf > 0.001:
                            _ew_total = _ew_kf * 2 * 100
                            _stake_html = f"{_ew_total:.1f}% <span class='sub'>({_ew_kf*100:.1f}% x2)</span>"
                        else:
                            _stake_html = "—"
                        flags.append("<span class='badge badge-ew'>🔀 EW</span>")
                    else:
                        if not _iv:
                            _stake_html = "1u EW"
                        else:
                            _stake_html += " EW"
                        flags.append("<span class='badge badge-ew'>🔀 EW</span>")
                _conf_label, _conf_detail = _bet_confidence_state(r)
                if _conf_label == "✅ Aligned" or _conf_label == "✓ Supported":
                    flags.append(f"<span class='badge badge-align'>{_conf_label}</span>")
                elif _conf_label:
                    flags.append(f"<span class='badge badge-warn'>{_conf_label}</span>")
                if rank == 1:
                    flags.append("🎯")

                pace_bits = []
                if bool(r.get("lone_speed_flag", False)):
                    pace_bits.append("<span class='badge badge-pace'>Lone speed</span>")
                elif bool(r.get("pace_front_runner_flag", False)):
                    pace_bits.append("<span class='badge badge-pace'>Front</span>")
                if bool(r.get("pace_closer_flag", False)):
                    pace_bits.append("<span class='badge badge-closer'>Closer</span>")
                _cps = pd.to_numeric(pd.Series([r.get("closer_pace_setup")]), errors="coerce").iloc[0]
                if pd.notna(_cps) and float(_cps) >= 0.08:
                    pace_bits.append("<span class='badge badge-closer'>Setup</span>")
                pace_html = " ".join(pace_bits) if pace_bits else "—"

                # Build row
                row_html = f"<tr class='{_row_cls}'>"
                if show_race_col:
                    _race_label = f"{r.get('off_time', '')} {r.get('track', '')}"
                    row_html += f"<td><span class='sub'>{_race_label.strip()}</span></td>"
                row_html += f"<td>{_rank_emoji(rank)}</td>"
                row_html += f"<td class='horse-cell'><span class='horse-name'>{r['horse_name']}</span>{sub_html}</td>"
                row_html += f"<td>{'%.1f' % odds if odds else '—'}</td>"
                row_html += f"<td><b>{'%.1f' % model_pct}%</b></td>"
                row_html += f"<td>{'%.1f' % mkt_pct + '%' if mkt_pct is not None else '—'}</td>"
                row_html += f"<td>{_edge_html(edge) if edge is not None else '—'}</td>"
                row_html += f"<td>{_stake_html}</td>"
                row_html += f"<td>{'%.1f' % pl_prob + '%' if pl_prob is not None else '—'}</td>"
                row_html += f"<td>{'%.1f' % mkt_pl_pct + '%' if mkt_pl_pct is not None else '—'}</td>"
                row_html += f"<td>{_edge_html(ew_edge) if ew_edge is not None else '—'}</td>"
                row_html += f"<td>{_edge_html(ew_ev) if ew_ev is not None else '—'}</td>"
                row_html += f"<td>{pace_html}</td>"
                row_html += f"<td>{' '.join(flags)}</td>"

                if show_result:
                    _rc_fp = int(r.get("result_fp", 0))
                    _rc_nr = bool(r.get("result_is_nr", False))
                    _rc_label = str(r.get("result_fp_label", "") or "")
                    _rc_won = bool(r.get("result_won", False))
                    _nf_labels = {"PU": "Pulled Up", "F": "Fell", "UR": "Unseated", "BD": "Brought Down", "RR": "Refused", "WO": "Walkover", "NR": "Non-Runner"}
                    if _rc_nr:
                        row_html += "<td class='settled-nr'>🚫 NR</td>"
                    elif _rc_won:
                        row_html += "<td class='settled-won'>🏆 1st</td>"
                    elif _rc_fp == 2:
                        row_html += "<td class='settled-placed'>🥈 2nd</td>"
                    elif _rc_fp == 3:
                        row_html += "<td class='settled-placed'>🥉 3rd</td>"
                    elif _rc_fp > 0:
                        row_html += f"<td class='settled-lost'>{_ordinal(_rc_fp)}</td>"
                    elif _rc_label:
                        row_html += f"<td class='settled-lost'>{_nf_labels.get(_rc_label, _rc_label)}</td>"
                    else:
                        row_html += "<td>—</td>"

                row_html += "</tr>"
                body.append(row_html)

            return (
                "<div style='overflow-x:auto'>"
                f"<table class='race-tbl'>{hdr}{''.join(body)}</table>"
                "</div>"
            )

        # ══════════════════════════════════════════════════
        #  VALUE PICKS SUMMARY
        # ══════════════════════════════════════════════════
        if not value_df.empty:
            st.subheader("⭐ Value Picks")
            st.caption(
                f"{len(value_df)} selections where model probability exceeds "
                "the market-implied probability."
            )
            if "ranker_disagrees_top_pick" in value_df.columns:
                _split_n = int(value_df["ranker_disagrees_top_pick"].fillna(False).sum())
                if _split_n > 0:
                    st.caption(f"⚠ {_split_n} value pick(s) sit in races where the ranker and win model split on the top pick.")
            _val_display = value_df.sort_values(
                ["off_time", "value_score"], ascending=[True, False],
            )
            _val_show_result = _val_display["is_settled"].any()
            st.markdown(
                _build_race_table_html(_val_display, show_result=_val_show_result, show_race_col=True),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "🔍 No win value bets found at the current threshold. "
                "Try lowering the base threshold."
            )

        # ══════════════════════════════════════════════════
        #  EW VALUE PICKS SUMMARY
        # ══════════════════════════════════════════════════
        if not ew_df.empty:
            st.markdown("---")
            st.subheader("🔀 Each-Way Value Picks")
            st.caption(
                f"{len(ew_df)} selections where the place probability "
                "exceeds the implied place odds from the fixed EW fraction."
            )
            if "ranker_disagrees_top_pick" in ew_df.columns:
                _split_ew_n = int(ew_df["ranker_disagrees_top_pick"].fillna(False).sum())
                if _split_ew_n > 0:
                    st.caption(f"⚠ {_split_ew_n} EW pick(s) come from split-view races, so treat conviction more cautiously.")
            _ew_display = ew_df.sort_values(
                ["off_time", "place_edge"], ascending=[True, False],
            )
            _ew_show_result = _ew_display["is_settled"].any()
            st.markdown(
                _build_race_table_html(_ew_display, show_result=_ew_show_result, show_race_col=True),
                unsafe_allow_html=True,
            )

        # ══════════════════════════════════════════════════
        #  FULL RACE-BY-RACE BREAKDOWN
        # ══════════════════════════════════════════════════
        st.markdown("---")
        st.subheader("🏟️ Full Race Card")

        # Group by track for a cleaner layout
        tracks_in_order = []
        seen_tracks = set()
        for m in race_meta:
            if m["track"] not in seen_tracks:
                tracks_in_order.append(m["track"])
                seen_tracks.add(m["track"])

        for track_name in tracks_in_order:
            track_races = [m for m in race_meta if m["track"] == track_name]
            st.markdown(f"### 🏟️ {track_name}")

            for rm in sorted(track_races, key=lambda x: x["off_time"]):
                rid = rm["race_id"]
                race_preds = full_preds[
                    full_preds["race_id"] == rid
                ].sort_values("predicted_rank")

                _race_settled = race_preds["is_settled"].any()
                n_value = int(race_preds["is_value"].sum())
                n_ew = int(race_preds.get("is_ew_value", pd.Series(False)).sum())
                # EW places for this race
                _n_runners = rm["runners"]
                _is_hcap = bool(race_preds["handicap"].iloc[0]) if "handicap" in race_preds.columns else False
                _ew_t = get_ew_terms(_n_runners, is_handicap=_is_hcap)
                _ew_places_str = f"EW {_ew_t.places_paid} places" if _ew_t.eligible else "No EW"
                badges = []
                if _race_settled:
                    badges.append("🏁 Result")
                if n_value:
                    badges.append(f"💰 {n_value} value")
                if n_ew:
                    badges.append(f"🔀 {n_ew} EW")
                badge_str = f"  —  **{'  ·  '.join(badges)}**" if badges else ""
                header = (
                    f"{rm['off_time']} — {rm['race_name']} "
                    f"({rm['runners']} runners · {_ew_places_str}){badge_str}"
                )

                with st.expander(header, expanded=(n_value + n_ew) > 0):
                    if _race_settled:
                        _display_order = race_preds.sort_values(
                            ["result_fp", "predicted_rank"],
                            ascending=[True, True],
                        )
                    else:
                        _display_order = race_preds

                    _pace_summary = _pace_race_summary(_display_order)
                    if _pace_summary:
                        st.markdown(
                            f"<div class='pace-summary'>🏇 Pace setup: {_pace_summary}</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        _build_race_table_html(_display_order, show_result=_race_settled),
                        unsafe_allow_html=True,
                    )

                    _feat_source = st.session_state.get("picks_featured", {}).get(_picks_date_str)
                    _expl_key = f"{_picks_date_str}|{_picks_model_fp[0]}|{rid}"
                    _expl_cache = st.session_state.get("picks_explanations", {})
                    _has_expl = _expl_key in _expl_cache
                    _run_expl = st.button(
                        "🔍 Explain why these horses are favoured" if not _has_expl else "🔄 Rebuild explanation",
                        key=f"btn_picks_explain_{rid}",
                        help="Show SHAP-based feature attributions for this race.",
                    )

                    if _run_expl:
                        if _feat_source is None or _feat_source.empty:
                            st.warning("Race features are not cached for this date yet. Re-run the picks analysis first.")
                        else:
                            _race_feat = _feat_source[_feat_source["race_id"] == rid].copy()
                            if _race_feat.empty:
                                st.warning("No engineered features found for this race.")
                            else:
                                with st.spinner("Computing feature attributions …"):
                                    try:
                                        _expl = st.session_state.predictor.explain_race(_race_feat)
                                        _expl_cache[_expl_key] = {
                                            "explanations": _expl,
                                            "model_label": getattr(
                                                st.session_state.predictor,
                                                "last_explain_model_label",
                                                "Win Classifier",
                                            ),
                                        }
                                        st.session_state["picks_explanations"] = _expl_cache
                                        _has_expl = True
                                    except Exception as e:
                                        st.warning(f"Explanation unavailable: {e}")

                    if _has_expl:
                        _payload = _expl_cache[_expl_key]
                        _render_shap_explanation(
                            _payload["explanations"],
                            race_preds,
                            key_prefix=f"picks_shap_{rid}",
                            model_label=_payload.get("model_label"),
                        )

    # ── Other prediction tools (folded in from the former Predict page) ──
    st.markdown("---")
    st.markdown("### 🔧 Other prediction tools")
    _model_df = st.session_state.get("model_featured_data")
    if _model_df is None:
        load_model_data()
        _model_df = st.session_state.get("model_featured_data")
    _other_tabs = st.tabs(["📋 From Dataset", "✏️ Custom Entry"])
    with _other_tabs[0]:
        st.subheader("📋 Predict from Loaded Data")
        if _model_df is None or _model_df.empty:
            st.info("Activate a model run with a saved dataset to use this tool.")
            st.stop()
        df = _model_df.copy()
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

        races = (
            df.groupby("race_id")
            .agg(
                date=("race_date", "first"),
                track=("track", "first"),
                runners=("horse_name", "count"),
            )
            .reset_index()
            .sort_values("date", ascending=False)
        )

        selected_race = st.selectbox(
            "Select Race",
            races["race_id"].values,
            format_func=lambda x: (
                f"{x} — "
                f"{races.loc[races['race_id']==x, 'track'].values[0]}"
                f" — "
                f"{races.loc[races['race_id']==x, 'runners'].values[0]}"
                " runners"
            ),
        )

        if st.button("🔮 Predict", key="pred_data", type="primary"):
            race_data = df[df["race_id"] == selected_race].copy()
            predictions = _predict_featured_frame(
                st.session_state.predictor,
                race_data,
                ew_fraction=st.session_state.value_config.get("ew_fraction"),
            )

            if "finish_position" in race_data.columns:
                actual = race_data[
                    ["horse_name", "finish_position"]
                ].copy()
                predictions = predictions.merge(
                    actual, on="horse_name", how="left",
                )

            st.markdown("### 🏆 Predictions")
            _render_ranker_consensus_badge(predictions)
            _render_pace_panel(predictions, key="dataset_pace")
            _render_ranker_disagreement_panel(predictions, key="dataset_ranker")
            for _, row in predictions.iterrows():
                rank = int(row["predicted_rank"])
                emoji = (
                    "🥇" if rank == 1 else
                    "🥈" if rank == 2 else
                    "🥉" if rank == 3 else f"#{rank}"
                )
                c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 2, 2])
                c1.markdown(f"**{emoji}**")
                _tp_tag = " 🎯 TOP PICK" if rank == 1 else ""
                c2.markdown(f"**{row['horse_name']}**{_tp_tag}")
                if bool(row.get("ranker_disagrees_top_pick", False)) and rank == 1:
                    c2.caption("Split top pick")
                if pd.notna(row.get("rank_disagreement")) and float(row.get("rank_disagreement", 0)) >= 2:
                    c2.caption(
                        f"Ranker differs: win #{int(row['predicted_rank'])} vs ranker #{int(row['ranker_rank'])}"
                    )
                c3.metric("Win Prob", f"{row['win_probability']:.1%}")
                if "odds" in row:
                    c4.metric("Odds", f"{row['odds']:.1f}")
                if (
                    "finish_position" in row
                    and pd.notna(row["finish_position"])
                ):
                    c5.metric(
                        "Actual", f"#{int(row['finish_position'])}",
                    )

            if "value_score" in predictions.columns and "odds" in predictions.columns:
                _pvc = st.session_state.value_config
                vb = predictions[_value_bet_mask(predictions, _pvc)]
                if not vb.empty:
                    st.markdown("### 💰 Value Bets")
                    for _, row in vb.iterrows():
                        _kf = kelly_criterion(row['win_probability'], row['odds'], fraction=_pvc["kelly_fraction"])
                        _k_label = f"{_pvc['kelly_fraction']:.0%}"
                        _ks = f" · Kelly {_k_label} **{_kf*100:.1f}%**" if _kf > 0.001 else ""
                        _clv = row['win_probability'] * row['odds']
                        _clv_str = f" · CLV **{_clv:.3f}x**" if _clv > 1.0 else ""
                        _conf_label, _conf_detail = _bet_confidence_state(row)
                        _conf_str = f" · {_conf_label}" if _conf_label else ""
                        _conf_detail_str = f"  \n_{_conf_detail}_" if _conf_detail else ""
                        st.info(
                            f"**{row['horse_name']}** — "
                            f"Model: {row['win_probability']:.1%} vs "
                            f"Market: {row['implied_prob']:.1%} — "
                            f"Value: +{row['value_score']:.1%}{_ks}{_clv_str}{_conf_str}"
                            f"{_conf_detail_str}"
                        )

            # ── Each-Way Value Bets ──────────────────────────────
            _pvc2 = st.session_state.value_config
            if (
                _pvc2.get("ew_enabled", True)
                and "ew_value" in predictions.columns
            ):
                ew_bets = ew_value_bets(
                    predictions,
                    min_place_edge=_pvc2.get("ew_min_place_edge", 0.05),
                    min_odds=_pvc2.get("ew_min_odds", 4.0),
                    max_odds=_pvc2.get("ew_max_odds", 51.0),
                )
                if not ew_bets.empty:
                    st.markdown("### 🔀 Each-Way Value Bets")
                    for _, row in ew_bets.iterrows():
                        _ew_k = kelly_ew(
                            row["win_probability"], row["place_probability"],
                            row["odds"],
                            get_ew_terms(
                                int(row.get("num_runners", 8)),
                                is_handicap=bool(row.get("handicap", 0)),
                            ),
                            fraction=_pvc2["kelly_fraction"],
                        )
                        _place_str = (
                            f"Place prob: **{row['place_probability']:.1%}** vs "
                            f"implied {1/row['place_odds']:.1%}"
                        )
                        _ew_kelly_str = (
                            f" · EW Kelly {_pvc2['kelly_fraction']:.0%} **{_ew_k['ew_kelly']*100:.1f}%**"
                            if _ew_k["ew_kelly"] > 0.001 else ""
                        )
                        _conf_label, _conf_detail = _bet_confidence_state(row)
                        _conf_str = f" · {_conf_label}" if _conf_label else ""
                        _terms_str = f"{int(row['ew_places'])} places at {row['ew_fraction_str']}"
                        st.success(
                            f"**{row['horse_name']}** @ {row['odds']:.1f} — "
                            f"{_place_str} — "
                            f"Edge: +{row['place_edge']:.1%} · "
                            f"EW EV: {row['ew_ev']:+.1%}{_ew_kelly_str}{_conf_str}\n\n"
                            f"<small>Terms: {_terms_str} · "
                            f"Place odds: {row['place_odds']:.2f}"
                            f"{' · ' + _conf_detail if _conf_detail else ''}</small>",
                            icon="🔀",
                        )

            # ── SHAP explanation ─────────────────────────────────
            try:
                expl = st.session_state.predictor.explain_race(
                    race_data,
                )
                _render_shap_explanation(
                    expl, predictions, key_prefix="shap_dataset",
                )
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

    # ── Custom Entry ─────────────────────────────────────────────────
    with _other_tabs[1]:
        st.subheader("✏️ Enter race details manually")

        if _model_df is not None:
            _df = _model_df
            track_opts = sorted(
                _df["track"].dropna().unique().tolist(),
            )
            going_opts = (
                sorted(_df["going"].dropna().unique().tolist())
                if "going" in _df.columns
                else ["Good", "Soft", "Heavy", "Firm"]
            )
            class_opts = (
                sorted(_df["race_class"].dropna().unique().tolist())
                if "race_class" in _df.columns
                else [f"Class {i}" for i in range(1, 6)]
            )
            type_opts = (
                sorted(_df["race_type"].dropna().unique().tolist())
                if "race_type" in _df.columns
                else ["Flat", "Hurdle", "Chase"]
            )
        else:
            track_opts = [
                "Ascot", "Cheltenham", "Newmarket", "York", "Aintree",
            ]
            going_opts = [
                "Good", "Good To Firm", "Good To Soft",
                "Soft", "Heavy", "Firm",
            ]
            class_opts = [f"Class {i}" for i in range(1, 6)]
            type_opts = ["Flat", "Hurdle", "Chase", "NH Flat"]

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            track = st.selectbox("Track", track_opts)
            going = st.selectbox("Going", going_opts)
        with rc2:
            race_class = st.selectbox("Race Class", class_opts)
            race_type = st.selectbox("Race Type", type_opts)
        with rc3:
            distance = st.selectbox(
                "Distance (f)",
                [5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32],
            )
            num_runners = st.number_input("Runners", 2, 20, 8)

        # ── Race-level extras ────────────────────────────────────
        rx1, rx2 = st.columns(2)
        with rx1:
            is_handicap = st.checkbox("Handicap?", value=True, key="c_hcap")
            prize_money = st.number_input(
                "Prize money (£)", 1000, 1_000_000, 10_000, 1000, key="c_prize",
            )
        with rx2:
            surface = st.selectbox(
                "Surface", ["Turf", "All Weather"], key="c_surface",
            )

        st.markdown("#### Horse Details")
        horses_data = []
        for i in range(num_runners):
            with st.expander(f"Horse {i+1}", expanded=i < 3):
                hc1, hc2, hc3, hc4 = st.columns([2, 2, 2, 1.5])
                with hc1:
                    name = st.text_input(
                        "Name", f"Horse {i+1}", key=f"cn_{i}",
                    )
                    jockey = st.text_input(
                        "Jockey", f"Jockey {i+1}", key=f"cj_{i}",
                    )
                    trainer = st.text_input(
                        "Trainer", f"Trainer {i+1}", key=f"ct_{i}",
                    )
                with hc2:
                    age = st.number_input(
                        "Age", 2, 12, 4, key=f"ca_{i}",
                    )
                    weight = st.number_input(
                        "Weight (lbs)", 112, 175, 130, key=f"cw_{i}",
                    )
                    odds = st.number_input(
                        "Odds", 1.1, 200.0, 5.0, key=f"co_{i}",
                    )
                with hc3:
                    official_rating = st.number_input(
                        "Official Rating", 0, 180, 0, key=f"cor_{i}",
                        help="0 = unknown (will use race median)",
                    )
                    days_since = st.number_input(
                        "Days since last run", 0, 999, 30, key=f"cd_{i}",
                    )
                    form_str = st.text_input(
                        "Form", "", key=f"cf_{i}",
                        help="e.g. 1-3-2-5 (most recent last)",
                    )
                with hc4:
                    draw = st.number_input(
                        "Draw", 1, 30, i + 1, key=f"cdr_{i}",
                    )
                    sex = st.selectbox(
                        "Sex", ["Gelding", "Colt", "Filly", "Mare", "Horse"],
                        key=f"cs_{i}",
                    )
                    headgear = st.selectbox(
                        "Headgear", ["", "b", "v", "t", "p", "h"],
                        key=f"ch_{i}",
                        help="b=blinkers, v=visor, t=tongue tie, p=cheekpieces, h=hood",
                    )

                horses_data.append({
                    "horse_name": name,
                    "jockey": jockey,
                    "trainer": trainer,
                    "track": track,
                    "going": going,
                    "race_class": race_class,
                    "race_type": race_type,
                    "distance_furlongs": distance,
                    "num_runners": num_runners,
                    "age": age,
                    "weight_lbs": weight,
                    "odds": odds,
                    "draw": draw,
                    "prize_money": prize_money,
                    "official_rating": official_rating,
                    "days_since_last_run": days_since,
                    "form": form_str if form_str else "",
                    "headgear": headgear,
                    "sex": sex,
                    "surface": surface,
                    "handicap": 1 if is_handicap else 0,
                })

        if st.button("🔮 Predict Custom Race", type="primary"):
            custom_df = pd.DataFrame(horses_data)
            custom_df["race_id"] = "CUSTOM_001"
            custom_df["race_date"] = pd.Timestamp.now().strftime(
                "%Y-%m-%d",
            )
            custom_df["won"] = 0
            custom_df["finish_position"] = 0
            custom_df["finish_time_secs"] = 0
            custom_df["lengths_behind"] = np.nan

            processed_custom = process_data(df=custom_df, save=False)
            featured_custom = feature_engineer_with_history(
                processed_custom,
            )
            predictions = _predict_featured_frame(
                st.session_state.predictor,
                featured_custom,
                ew_fraction=st.session_state.value_config.get("ew_fraction"),
            )

            st.markdown("### 🏆 Custom Race Prediction")
            _render_ranker_consensus_badge(predictions)
            _render_pace_panel(predictions, key="custom_pace")
            _render_ranker_disagreement_panel(predictions, key="custom_ranker")
            for _, row in predictions.iterrows():
                rank = int(row["predicted_rank"])
                emoji = (
                    "🥇" if rank == 1 else
                    "🥈" if rank == 2 else
                    "🥉" if rank == 3 else f"#{rank}"
                )
                c1, c2, c3, c4 = st.columns([1, 3, 2, 2])
                c1.markdown(f"**{emoji}**")
                _tp_tag = " 🎯 TOP PICK" if rank == 1 else ""
                c2.markdown(f"**{row['horse_name']}**{_tp_tag}")
                if bool(row.get("ranker_disagrees_top_pick", False)) and rank == 1:
                    c2.caption("Split top pick")
                if pd.notna(row.get("rank_disagreement")) and float(row.get("rank_disagreement", 0)) >= 2:
                    c2.caption(
                        f"Ranker differs: win #{int(row['predicted_rank'])} vs ranker #{int(row['ranker_rank'])}"
                    )
                c3.metric("Win Prob", f"{row['win_probability']:.1%}")
                if "odds" in row:
                    c4.metric("Odds", f"{row['odds']:.1f}")

            # ── SHAP explanation ─────────────────────────────────
            try:
                expl = st.session_state.predictor.explain_race(
                    featured_custom,
                )
                _render_shap_explanation(
                    expl, predictions, key_prefix="shap_custom",
                )
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")


# =====================================================================
#  MATCHBOOK API
# =====================================================================
elif page == "🔌 Matchbook API":
    st.title("🔌 Matchbook API")
    st.caption(
        "Connectivity harness for Matchbook. This page only reads public market data and tests login/account access; it does not place offers."
    )

    with st.expander("ℹ️ Integration Notes", expanded=False):
        st.markdown(
            "Use this page to verify public endpoint reachability, horse-racing event/market parsing, and authenticated session access before any live-betting workflow is added."
        )
        st.code(
            "MATCHBOOK_USERNAME=...\n"
            "MATCHBOOK_PASSWORD=...\n"
            "MATCHBOOK_HORSE_RACING_SPORT_ID=24735152712200",
            language="bash",
        )

    with st.form("matchbook_public_probe"):
        c1, c2, c3 = st.columns(3)
        with c1:
            _mb_hours = st.number_input("Hours ahead", min_value=1, max_value=168, value=24, step=1)
        with c2:
            _mb_per_page = st.number_input("Events to fetch", min_value=1, max_value=50, value=8, step=1)
        with c3:
            _mb_price_depth = st.number_input("Price depth", min_value=1, max_value=10, value=3, step=1)
        _mb_public_submit = st.form_submit_button("Fetch horse-racing events", type="primary")

    if _mb_public_submit:
        try:
            client = MatchbookClient()
            public_payload = client.get_horse_racing_events(
                hours_ahead=int(_mb_hours),
                per_page=int(_mb_per_page),
                include_prices=False,
                price_depth=int(_mb_price_depth),
            )
            events = public_payload.get("events") or []
            first_event = events[0] if events else None
            markets_payload = (
                client.get_event_markets(
                    first_event["id"],
                    include_prices=True,
                    price_depth=int(_mb_price_depth),
                    per_page=10,
                    currency=config.MATCHBOOK_DEFAULT_CURRENCY,
                )
                if first_event is not None else {"markets": []}
            )
            st.session_state["matchbook_public_probe_result"] = {
                "events": events,
                "markets": markets_payload.get("markets") or [],
            }
            st.success(f"Fetched {len(events)} upcoming horse-racing event(s) from Matchbook.")
        except MatchbookAPIError as exc:
            st.session_state["matchbook_public_probe_result"] = None
            st.error(str(exc))
        except Exception as exc:
            st.session_state["matchbook_public_probe_result"] = None
            st.error(f"Unexpected Matchbook error: {exc}")

    _public_probe = st.session_state.get("matchbook_public_probe_result")
    if isinstance(_public_probe, dict):
        _events = _public_probe.get("events") or []
        _markets = _public_probe.get("markets") or []
        m1, m2, m3 = st.columns(3)
        m1.metric("Events", len(_events))
        m2.metric("Markets", len(_markets))
        m3.metric("Currency", config.MATCHBOOK_DEFAULT_CURRENCY)

        if _events:
            st.markdown("**Upcoming events**")
            st.dataframe(_matchbook_events_table(_events), hide_index=True, width="stretch")
        if _markets:
            st.markdown("**Markets from first returned event**")
            st.dataframe(_matchbook_markets_table(_markets), hide_index=True, width="stretch")

        if _events:
            _event_lookup = {str(event.get("id")): event for event in _events if event.get("id") is not None}
            _event_ids = list(_event_lookup.keys())
            if _event_ids:
                st.markdown("**Signal table**")
                se1, se2, se3, se4 = st.columns([3, 1, 1, 1])
                with se1:
                    _selected_event_id = st.selectbox(
                        "Matchbook event",
                        _event_ids,
                        format_func=lambda eid: f"{_event_lookup[eid].get('name', eid)}",
                        key="matchbook_signal_event_id",
                    )
                with se2:
                    _min_back_edge = st.number_input("Min back edge", min_value=0.0, max_value=0.5, value=0.03, step=0.01, key="matchbook_min_back_edge")
                with se3:
                    _min_lay_edge = st.number_input("Min lay edge", min_value=0.0, max_value=0.5, value=0.03, step=0.01, key="matchbook_min_lay_edge")
                with se4:
                    _min_liquidity = st.number_input("Min liquidity", min_value=0.0, max_value=10000.0, value=10.0, step=5.0, key="matchbook_min_liquidity")

                if st.button("Build signal table", type="primary", key="btn_matchbook_signal_table"):
                    try:
                        signal_client = MatchbookClient()
                        selected_event = _event_lookup[_selected_event_id]
                        selected_event_markets = signal_client.get_event_markets(
                            _selected_event_id,
                            include_prices=True,
                            price_depth=int(_mb_price_depth),
                            per_page=20,
                            currency=config.MATCHBOOK_DEFAULT_CURRENCY,
                        ).get("markets") or []
                        selected_preds = build_fake_prediction_frame(
                            selected_event,
                            selected_event_markets,
                        )
                        signal_df = build_signal_frame(
                            selected_preds,
                            selected_event,
                            selected_event_markets,
                            min_back_edge=float(_min_back_edge),
                            min_lay_edge=float(_min_lay_edge),
                            min_liquidity=float(_min_liquidity),
                        )
                        st.session_state["matchbook_signal_result"] = {
                            "event": selected_event,
                            "markets": selected_event_markets,
                            "signals": signal_df,
                            "source": "fake_model",
                        }
                    except MatchbookAPIError as exc:
                        st.session_state["matchbook_signal_result"] = None
                        st.error(str(exc))
                    except Exception as exc:
                        st.session_state["matchbook_signal_result"] = None
                        st.error(f"Signal table build failed: {exc}")

    _signal_result = st.session_state.get("matchbook_signal_result")
    if isinstance(_signal_result, dict):
        _signal_df = _signal_result.get("signals")
        _signal_event = _signal_result.get("event") or {}
        _signal_markets = _signal_result.get("markets") or []
        st.markdown(f"**Signals for {_signal_event.get('name', 'selected event')}**")
        if _signal_result.get("source") == "fake_model":
            st.caption("Using deterministic fake model output derived from Matchbook market runners for fast UI testing.")
        if isinstance(_signal_df, pd.DataFrame) and not _signal_df.empty:
            _signal_labels = _signal_df["signal"].fillna("").astype(str)
            sg1, sg2, sg3 = st.columns(3)
            sg1.metric("Matched runners", int(len(_signal_df)))
            sg2.metric("Back candidates", int(_signal_labels.str.contains("BACK", regex=False).sum()))
            sg3.metric("Lay candidates", int(_signal_labels.str.contains("LAY", regex=False).sum()))
            pt1, pt2, pt3, pt4 = st.columns([1, 1, 1, 2])
            with pt1:
                _paper_stake = st.number_input("Paper stake", min_value=0.1, max_value=1000.0, value=1.0, step=0.5, key="matchbook_paper_stake")
            with pt2:
                _log_backs = st.checkbox("Log backs", value=True, key="matchbook_log_backs")
            with pt3:
                _log_lays = st.checkbox("Log lays", value=True, key="matchbook_log_lays")
            with pt4:
                st.caption("Logs selected signal candidates locally as paper trades using the displayed Matchbook entry prices.")

            if st.button("Log paper trades", key="btn_log_matchbook_paper_trades"):
                new_trades = build_paper_trades_from_signals(
                    _signal_df,
                    event=_signal_event,
                    stake=float(_paper_stake),
                    source=str(_signal_result.get("source") or "unknown"),
                    log_backs=bool(_log_backs),
                    log_lays=bool(_log_lays),
                )
                appended = append_paper_trades(new_trades)
                if appended > 0:
                    st.success(f"Logged {appended} paper trade(s).")
                else:
                    st.warning("No matching BACK/LAY candidates were selected for logging.")

            st.dataframe(
                _signal_df.style.format({
                    "win_probability": "{:.1%}",
                    "fair_odds": "{:.2f}",
                    "best_back_odds": "{:.2f}",
                    "best_back_available": "{:.2f}",
                    "best_lay_odds": "{:.2f}",
                    "best_lay_available": "{:.2f}",
                    "back_edge_pct": "{:.1%}",
                    "lay_edge_pct": "{:.1%}",
                    "spread_pct": "{:.1%}",
                }),
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning(
                "No model runners matched the selected Matchbook event. This usually means the Sporting Life racecards and Matchbook event names did not align on track/off-time yet."
            )
        if _signal_markets:
            with st.expander("Show selected event markets", expanded=False):
                st.dataframe(_matchbook_markets_table(_signal_markets), hide_index=True, width="stretch")

    st.markdown("**Paper trade log**")
    _paper_trades = load_paper_trades()
    log1, log2, log3 = st.columns([1, 1, 2])
    with log1:
        _settle_date = st.date_input("Settle date", value=datetime.now().date(), key="matchbook_settle_date")
    with log2:
        if st.button("Settle open trades", key="btn_settle_paper_trades"):
            with st.spinner(f"Scraping results for {_settle_date.strftime('%Y-%m-%d')} …"):
                _results_df = scrape_todays_results(date_str=_settle_date.strftime("%Y-%m-%d"))
            updated = settle_paper_trades(_paper_trades, _results_df)
            save_paper_trades(updated)
            _paper_trades = updated
            _settled_now = int((updated["status"].fillna("") == "SETTLED").sum())
            st.success(f"Settlement run completed. Total settled trades in log: {_settled_now}.")
    with log3:
        if not _paper_trades.empty:
            _open_count = int((_paper_trades["status"].fillna("OPEN") == "OPEN").sum())
            _settled_count = int((_paper_trades["status"].fillna("") == "SETTLED").sum())
            st.caption(f"Open trades: {_open_count} · Settled trades: {_settled_count}")

    if isinstance(_paper_trades, pd.DataFrame) and not _paper_trades.empty:
        _display_cols = [
            col for col in [
                "logged_at", "race_date", "event_name", "horse_name", "side", "stake",
                "entry_odds", "win_probability", "fair_odds", "back_edge_pct", "lay_edge_pct",
                "status", "result_finish_position", "result_won", "pnl",
            ] if col in _paper_trades.columns
        ]
        st.dataframe(
            _paper_trades[_display_cols].sort_values(["logged_at", "event_name", "horse_name"], ascending=[False, True, True]).style.format({
                "stake": "{:.2f}",
                "entry_odds": "{:.2f}",
                "win_probability": "{:.1%}",
                "fair_odds": "{:.2f}",
                "back_edge_pct": "{:.1%}",
                "lay_edge_pct": "{:.1%}",
                "pnl": "{:.2f}",
            }),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download paper trades CSV",
            data=_paper_trades.to_csv(index=False).encode("utf-8"),
            file_name="matchbook_paper_trades.csv",
            mime="text/csv",
            key="dl_matchbook_paper_trades",
        )
    else:
        st.caption("No paper trades logged yet.")

    _default_user = config.MATCHBOOK_USERNAME or ""
    _default_pass = config.MATCHBOOK_PASSWORD or ""
    with st.form("matchbook_auth_probe"):
        a1, a2 = st.columns(2)
        with a1:
            _mb_user = st.text_input("Username", value=_default_user)
        with a2:
            _mb_pass = st.text_input("Password", value=_default_pass, type="password")
        _mb_auth_submit = st.form_submit_button("Test account session")

    if _mb_auth_submit:
        try:
            auth_client = MatchbookClient(username=_mb_user.strip(), password=_mb_pass)
            auth_client.login()
            account = auth_client.get_account()
            balance = auth_client.get_balance()
            st.session_state["matchbook_auth_probe_result"] = {
                "session_token": auth_client.session_token,
                "account": account,
                "balance": balance,
            }
            st.success("Matchbook login succeeded and authenticated account endpoints responded.")
        except MatchbookAPIError as exc:
            st.session_state["matchbook_auth_probe_result"] = None
            st.error(str(exc))
        except Exception as exc:
            st.session_state["matchbook_auth_probe_result"] = None
            st.error(f"Unexpected Matchbook auth error: {exc}")

    _auth_probe = st.session_state.get("matchbook_auth_probe_result")
    if isinstance(_auth_probe, dict):
        st.markdown("**Authenticated probe**")
        st.write(f"Session token: {_mask_token(_auth_probe.get('session_token'))}")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("Account")
            st.json(_auth_probe.get("account") or {})
        with b2:
            st.markdown("Balance")
            st.json(_auth_probe.get("balance") or {})


# =====================================================================
#  SHORTCOMINGS
# =====================================================================
elif page == "🔎 Shortcomings":
    st.title("🔎 Model Shortcomings")
    st.caption(
        "Inspect which evaluation metrics line up with ROI, and where the model "
        "performs best or worst by race conditions like track, race type, and field size."
    )

    saved_runs = list_runs()
    if not saved_runs:
        st.info("No saved runs yet. Train a model first to analyse shortcomings.")
        st.stop()

    run_level_df = _build_shortcomings_run_frame(saved_runs)
    fold_level_df = _build_shortcomings_fold_frame(tuple(r.get("run_id") for r in saved_runs if r.get("run_id")))

    st.subheader("1️⃣ Metric ↔ ROI Correlation")
    _corr_sources = []
    if not fold_level_df.empty:
        _corr_sources.append("Walk-forward folds")
    if not run_level_df.empty:
        _corr_sources.append("Saved runs")

    if not _corr_sources:
        st.info("No saved metrics are available yet.")
    else:
        cc1, cc2 = st.columns([1.2, 1.6])
        with cc1:
            corr_source = st.radio(
                "Correlation source",
                options=_corr_sources,
                help="Fold-level correlations are usually more informative because they provide more observations.",
            )
        corr_base = fold_level_df if corr_source == "Walk-forward folds" else run_level_df

        _roi_candidates = [
            c for c in ["top_pick_roi", "value_roi", "each_way_roi", "combined_roi"]
            if c in corr_base.columns and pd.to_numeric(corr_base[c], errors="coerce").notna().sum() >= 3
        ]
        if not _roi_candidates:
            st.info("Not enough variation yet to compute meaningful correlations.")
        else:
            with cc2:
                corr_target = st.selectbox(
                    "Target ROI metric",
                    options=_roi_candidates,
                    format_func=lambda c: c.replace("_", " ").title(),
                )

            corr_df = _build_shortcomings_correlation_table(corr_base, corr_target)
            if corr_df.empty:
                st.info("Not enough variation yet to compute meaningful correlations.")
            else:
                cmt1, cmt2, cmt3 = st.columns(3)
                cmt1.metric("Observations", f"{len(corr_base):,}")
                cmt2.metric("Metrics Tested", f"{len(corr_df):,}")
                cmt3.metric("Best |Spearman|", f"{corr_df.iloc[0]['abs_spearman']:.3f}")

                st.dataframe(
                    corr_df[["metric", "n", "pearson", "spearman"]].style.format(
                        {"pearson": "{:+.3f}", "spearman": "{:+.3f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

                _top_corr = corr_df.head(12).copy()
                fig_corr = px.bar(
                    _top_corr.sort_values("spearman"),
                    x="spearman",
                    y="metric",
                    orientation="h",
                    color="spearman",
                    color_continuous_scale=["#ef4444", "#f8fafc", "#22c55e"],
                    color_continuous_midpoint=0,
                    title=f"Top Metric Correlations with {corr_target.replace('_', ' ').title()}",
                )
                fig_corr.update_layout(height=max(420, len(_top_corr) * 28), yaxis_title="", xaxis_title="Spearman correlation")
                st.plotly_chart(fig_corr, width="stretch")

                _scatter_metric = st.selectbox(
                    "Inspect one metric vs ROI",
                    options=corr_df["metric"].tolist(),
                )
                fig_scatter = px.scatter(
                    corr_base,
                    x=_scatter_metric,
                    y=corr_target,
                    hover_name="name" if "name" in corr_base.columns else None,
                    hover_data=[c for c in ["run_id", "fold", "timestamp"] if c in corr_base.columns],
                    title=f"{_scatter_metric} vs {corr_target.replace('_', ' ').title()}",
                )
                fig_scatter.update_layout(height=430)
                st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("---")
    st.subheader("2️⃣ Race Condition Breakdown")

    run_options = {
        f"{r.get('name', r['run_id'])}  ({r.get('timestamp', '')[:16].replace('T', ' ')})": r["run_id"]
        for r in saved_runs
    }
    _run_labels = list(run_options.keys())
    _default_run = 0
    _active_rid = st.session_state.get("active_run_id")
    if _active_rid:
        for i, lbl in enumerate(_run_labels):
            if run_options[lbl] == _active_rid:
                _default_run = i
                break

    selected_run_label = st.selectbox(
        "Run to inspect",
        options=_run_labels,
        index=_default_run,
    )
    selected_run_id = run_options[selected_run_label]
    selected_run_meta = load_run_meta(selected_run_id)
    selected_ta = selected_run_meta.get("test_analysis", {}) if isinstance(selected_run_meta.get("test_analysis", {}), dict) else {}

    _frames = _prepare_shortcomings_run_frames(selected_run_id)
    _available_strategies = [
        ("top_pick", "Top Pick"),
        ("value", "Value"),
        ("each_way", "Each-Way"),
    ]
    _available_strategies = [(k, lbl) for k, lbl in _available_strategies if isinstance(_frames.get(k), pd.DataFrame) and not _frames.get(k).empty]

    if not _available_strategies:
        st.warning(
            "This run does not have enough saved prediction/bet data to slice by race conditions. "
            "Use a run saved after the recent run-store updates."
        )
    else:
        sr1, sr2, sr3 = st.columns([1.2, 1.2, 1.2])
        with sr1:
            strategy_key = st.selectbox(
                "Strategy",
                options=[k for k, _ in _available_strategies],
                format_func=lambda k: dict(_available_strategies)[k],
            )
        _group_options = {
            "track": "Track",
            "race_type": "Race Type",
            "surface": "Surface",
            "going": "Going",
            "field_size_band": "Field Size Band",
            "num_runners": "Exact Runner Count",
            "distance_band": "Distance Band",
            "handicap_label": "Handicap vs Non-Handicap",
            "month": "Month",
        }
        _strategy_df = _frames[strategy_key].copy()
        _valid_group_cols = [k for k in _group_options if k in _strategy_df.columns]
        if not _valid_group_cols:
            st.info("No groups met the minimum-bets filter.")
        else:
            with sr2:
                group_col = st.selectbox(
                    "Group by",
                    options=_valid_group_cols,
                    format_func=lambda k: _group_options[k],
                )
            with sr3:
                min_bets = st.slider(
                    "Minimum bets per slice",
                    min_value=1,
                    max_value=max(1, min(50, int(len(_strategy_df)))),
                    value=min(3, max(1, int(len(_strategy_df)))),
                    step=1,
                )

            slice_df = _summarise_shortcomings_slice(_strategy_df, group_col)
            slice_df = slice_df[slice_df["bets"] >= min_bets].reset_index(drop=True)

            if slice_df.empty:
                st.info("No groups met the minimum-bets filter.")
            else:
                st.caption(
                    f"Test window: **{selected_ta.get('test_date_range', ['?', '?'])[0]}** → "
                    f"**{selected_ta.get('test_date_range', ['?', '?'])[1]}** · "
                    f"{selected_ta.get('test_races', 0):,} races"
                )

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Groups Shown", f"{len(slice_df):,}")
                sc2.metric("Total Bets", f"{int(slice_df['bets'].sum()):,}")
                sc3.metric("Best ROI", f"{slice_df['roi'].max():+.1f}%")
                sc4.metric("Worst ROI", f"{slice_df['roi'].min():+.1f}%")

                st.caption(
                    "Complete picture across all shown groups. Use ROI together with bets/races before "
                    "making strategy decisions on whether to avoid a track or race type."
                )

                _chart_df = slice_df.sort_values(["roi", "bets"], ascending=[True, False]).copy()
                _chart_df[group_col] = _chart_df[group_col].astype(str)
                _chart_df["_roi_color"] = np.where(_chart_df["roi"] >= 0, "Profitable", "Loss-making")

                fig_slice = px.bar(
                    _chart_df,
                    x="roi",
                    y=group_col,
                    color="_roi_color",
                    orientation="h",
                    hover_data=["bets", "races", "pnl", "strike_rate", "place_rate", "avg_odds", "avg_model_prob"],
                    color_discrete_map={"Loss-making": "#ef4444", "Profitable": "#22c55e"},
                    title=f"ROI Across All {dict(_group_options)[group_col]} Segments · {dict(_available_strategies)[strategy_key]}",
                )
                fig_slice.add_vline(x=0, line_dash="dash", line_color="#94a3b8", opacity=0.8)
                fig_slice.update_layout(
                    height=max(420, len(_chart_df) * 26),
                    yaxis_title="",
                    xaxis_title="ROI (%)",
                    legend_title_text="Outcome",
                )
                st.plotly_chart(fig_slice, width="stretch")

                fig_volume = px.scatter(
                    _chart_df,
                    x="bets",
                    y="roi",
                    size="races",
                    color="strike_rate",
                    hover_name=group_col,
                    hover_data={
                        "bets": True,
                        "races": True,
                        "pnl": ':.1f',
                        "strike_rate": ':.1f',
                        "place_rate": ':.1f',
                        "avg_odds": ':.2f',
                        "avg_model_prob": ':.3f',
                    },
                    color_continuous_scale="RdYlGn",
                    title=f"ROI vs Bet Volume · {dict(_group_options)[group_col]}",
                )
                fig_volume.add_hline(y=0, line_dash="dash", line_color="#94a3b8", opacity=0.8)
                fig_volume.update_layout(xaxis_title="Bets", yaxis_title="ROI (%)", coloraxis_colorbar_title="Strike %")
                st.plotly_chart(fig_volume, width="stretch")

                st.dataframe(
                    slice_df.style.format(
                        {
                            "staked": "{:.1f}",
                            "pnl": "{:+.1f}",
                            "avg_odds": "{:.2f}",
                            "avg_model_prob": "{:.3f}",
                            "strike_rate": "{:.1f}%",
                            "place_rate": "{:.1f}%",
                            "roi": "{:+.1f}%",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )


# =====================================================================
#  STRATEGY CALIBRATOR
# =====================================================================
elif page == "⚖️ Strategy Calibrator":
    st.title("⚖️ Strategy Calibrator")
    st.caption(
        "Grid-search over betting parameters on the test set to find "
        "optimal thresholds, odds ranges, and Kelly fractions."
    )

    from src.strategy_calibrator import (
        run_grid_search, run_validated_grid_search,
        precompute_analysis, DEFAULT_GRID,
    )

    # ── Load model + data ────────────────────────────────────────
    if st.session_state.predictor is None:
        if os.path.exists(_ENSEMBLE_MODEL_PATH):
            load_existing_model()
            load_model_data(force=True)
        else:
            st.warning("No model available. Train one on the **Train & Tune** page first.")
            st.stop()

    _feat_df = st.session_state.get("model_featured_data")
    if _feat_df is None:
        load_model_data()
        _feat_df = st.session_state.get("model_featured_data")
    if _feat_df is None:
        st.warning("No featured data. Train a model first.")
        st.stop()

    _pred = st.session_state.predictor

    # Resolve test split for calibration from active run metadata when possible.
    _cal_test_frac = float(getattr(config, "TEST_SIZE", 0.2))
    _cal_split_source = "config.TEST_SIZE"
    _cal_use_date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None
    _active_run_meta: dict | None = None
    _active_rid_for_split = st.session_state.get("active_run_id")
    if _active_rid_for_split:
        for _r in list_runs():
            if _r.get("run_id") != _active_rid_for_split:
                continue
            _active_run_meta = _r
            _tc = _r.get("training_config", {}) if isinstance(_r.get("training_config", {}), dict) else {}
            _pct = _tc.get("test_size_pct")
            if isinstance(_pct, (int, float)) and 1 <= float(_pct) <= 95:
                _cal_test_frac = float(_pct) / 100.0
                _cal_split_source = f"re-split ({int(_pct)}% tail of full dataset)"

            # Prefer exact saved test date range from the active run when available.
            _ta = _r.get("test_analysis", {}) if isinstance(_r.get("test_analysis", {}), dict) else {}
            _dr = _ta.get("test_date_range")
            if isinstance(_dr, (list, tuple)) and len(_dr) == 2 and _dr[0] and _dr[1]:
                try:
                    _start = pd.to_datetime(_dr[0])
                    _end = pd.to_datetime(_dr[1])
                    if pd.notna(_start) and pd.notna(_end):
                        _cal_use_date_range = (_start.normalize(), _end.normalize())
                        _cal_split_source = "training test split (exact same dates)"
                except Exception:
                    pass
            break

    # Rebuild calibrator data if model/data context changed
    _prev_test_size = getattr(config, "TEST_SIZE", 0.2)
    config.TEST_SIZE = _cal_test_frac
    _current_sig = _calibration_signature(_pred, _feat_df)
    _cached_sig = st.session_state.get("cal_analysis_sig")
    _cal_schema_ver = 2
    _cached_ver = st.session_state.get("cal_analysis_schema_ver")
    if _cached_sig != _current_sig or _cached_ver != _cal_schema_ver:
        for _k in list(st.session_state.keys()):
            if _k in ("cal_analysis_df", "cal_results", "cal_vresult") or _k.startswith("cal_precomputed_"):
                st.session_state.pop(_k, None)
        st.session_state["cal_analysis_sig"] = _current_sig
        st.session_state["cal_analysis_schema_ver"] = _cal_schema_ver
        if _cached_sig is not None:
            st.info("Detected model/data change: refreshed calibration cache for the current run.")

    # ── Build test-set analysis df (cached in session_state) ─────
    if "cal_analysis_df" not in st.session_state:
        from src.model import _event_sort_key

        # Helper to build + filter the base test feature dataframe
        def _build_test_feat_df():
            feat = _feat_df.copy()
            feat["race_date"] = pd.to_datetime(feat["race_date"])
            feat["_event_dt"] = _event_sort_key(feat)
            feat = feat.sort_values(["_event_dt", "race_id"]).reset_index(drop=True)
            feat = feat[feat["finish_position"].notna() & (feat["finish_position"] > 0)].copy()

            if _cal_use_date_range is not None:
                _start_dt, _end_dt = _cal_use_date_range
                _mask = (
                    (feat["race_date"].dt.normalize() >= _start_dt)
                    & (feat["race_date"].dt.normalize() <= _end_dt)
                )
                tdf = feat[_mask].copy()
                if tdf.empty:
                    split_idx = int(len(feat) * (1 - _cal_test_frac))
                    split_race = feat.iloc[split_idx]["race_id"]
                    while split_idx < len(feat) and feat.iloc[split_idx]["race_id"] == split_race:
                        split_idx += 1
                    tdf = feat.iloc[split_idx:].copy()
            else:
                split_idx = int(len(feat) * (1 - _cal_test_frac))
                split_race = feat.iloc[split_idx]["race_id"]
                while split_idx < len(feat) and feat.iloc[split_idx]["race_id"] == split_race:
                    split_idx += 1
                tdf = feat.iloc[split_idx:].copy()

            missing = [c for c in _pred.feature_cols if c not in tdf.columns]
            for c in missing:
                tdf[c] = 0
            return tdf.reset_index(drop=True)

        # ── Fast path: reuse predictions saved at training time ──
        _saved_preds: pd.DataFrame | None = None
        if _active_rid_for_split:
            try:
                _run_data = _raw_load_run(_active_rid_for_split)
                _saved_preds = _run_data.get("predictions_df")
            except Exception:
                pass

        if _saved_preds is not None and not _saved_preds.empty and \
                "model_prob" in _saved_preds.columns:
            with st.spinner("Loading saved test-set predictions …"):
                test_df = _build_test_feat_df()
                _join_cols = [c for c in ["race_id", "horse_name", "odds", "jockey", "trainer"]
                              if c in test_df.columns and c in _saved_preds.columns]
                _pred_cols_needed = _join_cols + [
                    c for c in ["model_prob", "place_prob"]
                    if c in _saved_preds.columns
                ]
                # Coerce race_id to the same type to avoid object vs int64 merge error
                if "race_id" in _join_cols:
                    _target_type = test_df["race_id"].dtype
                    _saved_preds = _saved_preds.copy()
                    _saved_preds["race_id"] = _saved_preds["race_id"].astype(_target_type)
                _merged = test_df[_join_cols].merge(
                    _saved_preds[_pred_cols_needed].drop_duplicates(_join_cols),
                    on=_join_cols,
                    how="left",
                )
                test_df["model_prob"] = _merged["model_prob"].fillna(
                    1.0 / test_df.groupby("race_id")["race_id"].transform("count")
                ).values
                if "place_prob" in _merged.columns:
                    test_df["place_prob"] = _merged["place_prob"].fillna(
                        3.0 / test_df.groupby("race_id")["race_id"].transform("count")
                    ).values
                else:
                    test_df["place_prob"] = 3.0 / test_df.groupby("race_id")["race_id"].transform("count")
                if "won" not in test_df.columns:
                    test_df["won"] = (test_df["finish_position"] == 1).astype(int)
                st.session_state["cal_analysis_df"] = test_df

        else:
            # ── Slow path: re-run predict_race() for each test race ──
            with st.spinner("Running model inference on test set (one-time) …"):
                test_df = _build_test_feat_df()

                all_probs = np.zeros(len(test_df), dtype=np.float64)
                all_place = np.zeros(len(test_df), dtype=np.float64)

                for rid in test_df["race_id"].unique():
                    rmask = test_df["race_id"] == rid
                    ridx = np.where(rmask.values)[0]
                    race_slice = test_df.iloc[ridx].copy().reset_index(drop=True)
                    n = len(race_slice)
                    try:
                        preds = _pred.predict_race(race_slice)

                        # predict_race() returns rows sorted by predicted rank.
                        # Re-align to original race_slice order using the most
                        # specific shared identifiers available.
                        _key_cols = [
                            c for c in ["horse_name", "odds", "jockey", "trainer"]
                            if c in race_slice.columns and c in preds.columns
                        ]
                        _can_align = bool(_key_cols)
                        if _can_align:
                            _left = race_slice[_key_cols].copy()
                            _right = preds[
                                _key_cols + ["win_probability", "place_probability"]
                            ].copy()

                            # If keys are not unique, add occurrence index within
                            # identical key groups to make the join deterministic.
                            if _left.duplicated(_key_cols).any() or _right.duplicated(_key_cols).any():
                                _left["_dup_idx"] = _left.groupby(_key_cols, sort=False).cumcount()
                                _right["_dup_idx"] = _right.groupby(_key_cols, sort=False).cumcount()
                                _join_cols = _key_cols + ["_dup_idx"]
                            else:
                                _join_cols = _key_cols

                            _aligned = _left.merge(
                                _right[_join_cols + ["win_probability", "place_probability"]],
                                on=_join_cols,
                                how="left",
                            )
                            _wp = _aligned["win_probability"].fillna(1.0 / max(n, 1)).values
                            _pp = _aligned["place_probability"].fillna(3.0 / max(n, 1)).values
                        else:
                            # Last-resort fallback if we cannot build join keys.
                            _wp = preds.get(
                                "win_probability", pd.Series(np.full(n, 1.0 / max(n, 1))),
                            ).values
                            _pp = preds.get(
                                "place_probability", pd.Series(np.full(n, 3.0 / max(n, 1))),
                            ).values

                        all_probs[ridx] = _wp
                        all_place[ridx] = _pp
                    except Exception:
                        all_probs[ridx] = 1.0 / max(n, 1)
                        all_place[ridx] = 3.0 / max(n, 1)

                test_df["model_prob"] = all_probs
                test_df["place_prob"] = all_place
                if "won" not in test_df.columns:
                    test_df["won"] = (test_df["finish_position"] == 1).astype(int)

                st.session_state["cal_analysis_df"] = test_df

    analysis_df = st.session_state["cal_analysis_df"]

    _n_test_races = int(analysis_df["race_id"].nunique())
    _n_test_runners = len(analysis_df)
    _test_dates = f"{analysis_df['race_date'].min().date()} to {analysis_df['race_date'].max().date()}"
    _expected_races = None
    if isinstance(_active_run_meta, dict):
        _ta = _active_run_meta.get("test_analysis", {}) if isinstance(_active_run_meta.get("test_analysis", {}), dict) else {}
        _expected_races = _ta.get("test_races")
    if _cal_use_date_range is not None:
        st.info(
            f"Test set: **{_n_test_runners:,} runners** across "
            f"**{_n_test_races:,} races** ({_test_dates}) · "
            f"source: {_cal_split_source}"
        )
    else:
        st.info(
            f"Test set: **{_n_test_runners:,} runners** across "
            f"**{_n_test_races:,} races** ({_test_dates}) · "
            f"source: {_cal_split_source}"
        )
        st.caption(
            "ℹ️ No exact test date range was found for the active run — "
            f"using the last **{_cal_test_frac:.0%}** of the full dataset as a proxy."
        )
    if isinstance(_expected_races, (int, float)) and _expected_races > 0:
        _delta = int(_n_test_races - int(_expected_races))
        if _delta != 0:
            st.caption(
                f"Active run recorded **{int(_expected_races):,}** test races; current calibrator set has **{_n_test_races:,}** "
                f"({_delta:+,d})."
            )

    # Restore previous global after preparing calibrator context.
    config.TEST_SIZE = _prev_test_size

    # ── Parameter grid controls ──────────────────────────────────
    st.subheader("1️⃣ Parameter Grid")

    with st.expander("🔧 Customise search grid", expanded=True):
        gc1, gc2 = st.columns(2)

        with gc1:
            _vt_options = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
            _vt_defaults = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
            cal_vt = st.multiselect(
                "Value threshold",
                options=_vt_options,
                default=_vt_defaults,
                key="cal_vt",
                help="Base edge threshold for value bets (scaled by √(odds/3)).",
            )

            _mpe_options = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
            _mpe_defaults = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
            cal_mpe = st.multiselect(
                "Min place edge",
                options=_mpe_options,
                default=_mpe_defaults,
                key="cal_mpe",
                help="Minimum place_prob − implied_place_prob for EW bets.",
            )

        with gc2:
            _mino_options = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
            _mino_defaults = [2.0, 3.0, 4.0, 5.0, 6.0]
            cal_mino = st.multiselect(
                "Min odds",
                options=_mino_options,
                default=_mino_defaults,
                key="cal_mino",
                help="Minimum decimal odds for value & EW bets.",
            )

            _maxo_options = [11.0, 16.0, 21.0, 26.0, 31.0, 41.0, 51.0, 76.0, 101.0]
            _maxo_defaults = [21.0, 31.0, 51.0, 101.0]
            cal_maxo = st.multiselect(
                "Max odds",
                options=_maxo_options,
                default=_maxo_defaults,
                key="cal_maxo",
                help="Maximum decimal odds for value & EW bets.",
            )

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            cal_staking = st.selectbox(
                "Staking mode",
                ["flat", "kelly"],
                format_func=lambda x: "Flat 1-unit" if x == "flat" else "Fractional Kelly",
                key="cal_staking",
            )
        with sc2:
            cal_bankroll = st.number_input(
                "Bankroll (Kelly only)",
                min_value=10.0, max_value=10000.0, value=100.0, step=10.0,
                key="cal_bankroll",
            )
        with sc3:
            _ew_frac_map = {0.20: "1/5", 0.25: "1/4", 0.33: "1/3"}
            cal_ew_frac = st.selectbox(
                "EW odds fraction",
                options=[0.20, 0.25, 0.33],
                index=1,
                format_func=lambda x: _ew_frac_map.get(x, str(x)),
                key="cal_ew_frac",
            )

        if cal_staking == "kelly":
            _kf_options = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50]
            _kf_defaults = [0.05, 0.10, 0.15, 0.20, 0.25, 0.33]
            cal_kf = st.multiselect(
                "Kelly fractions",
                options=_kf_options,
                default=_kf_defaults,
                key="cal_kf",
            )
        else:
            cal_kf = [0.25]

    # Compute combo count — exclude invalid mino >= maxo pairs
    _valid_odds_pairs = sum(1 for a in cal_mino for b in cal_maxo if a < b)
    _n_combos = len(cal_vt) * len(cal_mpe) * _valid_odds_pairs * len(cal_kf)
    st.caption(f"**{_n_combos:,}** valid parameter combinations to evaluate")

    # ── Sort / rank control ──────────────────────────────────────
    st.subheader("2️⃣ Optimisation Target")
    _sort_options = {
        "combined_pnl": "Combined P&L",
        "combined_roi": "Combined ROI %",
        "combined_sharpe": "Combined Sharpe",
        "value_pnl": "Value P&L",
        "value_roi": "Value ROI %",
        "value_sharpe": "Value Sharpe",
        "ew_pnl": "EW P&L",
        "ew_roi": "EW ROI %",
        "ew_sharpe": "EW Sharpe",
    }
    oc1, oc2, oc3 = st.columns([2, 2, 3])
    with oc1:
        cal_sort_by = st.selectbox(
            "Rank results by",
            options=list(_sort_options.keys()),
            format_func=lambda k: _sort_options[k],
            key="cal_sort_by",
        )
    with oc2:
        cal_min_bets = st.slider(
            "Min bets (filter out low-volume combos)",
            min_value=0, max_value=200, value=10, step=5,
            key="cal_min_bets",
            help="Exclude parameter sets that generate fewer bets than this.",
        )
    with oc3:
        cal_max_decay = st.slider(
            "Max IS→OOS ROI decay (anti-overfit filter)",
            min_value=0, max_value=100, value=100, step=5,
            key="cal_max_decay",
            help=(
                "Hide combos where in-sample ROI exceeds out-of-sample ROI by more "
                "than this many percentage points. Lower = stricter anti-overfit. "
                "100 = no filter (show all)."
            ),
        )

    st.markdown("---")

    # ── Run button ───────────────────────────────────────────────
    if st.button("🚀 Run Calibration", type="primary", width="stretch"):
        grid = {
            "value_threshold": sorted(cal_vt),
            "min_place_edge": sorted(cal_mpe),
            "min_odds": sorted(cal_mino),
            "max_odds": sorted(cal_maxo),
            "kelly_fraction": sorted(cal_kf),
        }

        progress = st.progress(0, text="Running calibration …")

        def _cal_progress(cur, total):
            progress.progress(
                min(cur / total, 1.0),
                text=f"Evaluating {cur:,}/{total:,} combinations …",
            )

        vresult = run_validated_grid_search(
            analysis_df,
            grid=grid,
            staking_mode=cal_staking,
            bankroll=cal_bankroll,
            ew_fraction=cal_ew_frac,
            sort_by=cal_sort_by,
            min_bets=cal_min_bets,
            val_fraction=0.5,
            progress_fn=_cal_progress,
        )
        progress.empty()

        st.session_state["cal_vresult"] = vresult
        st.success(
            f"Evaluated **{vresult['combos_tried']:,}** combos on "
            f"**{vresult['cal_races']:,}** calibration races, "
            f"validated on **{vresult['val_races']:,}** held-out races."
        )

    # ── Display results ──────────────────────────────────────────
    if "cal_vresult" in st.session_state and st.session_state["cal_vresult"] is not None:
        vresult = st.session_state["cal_vresult"]
        cal_results = vresult["cal_results"]
        val_results = vresult["val_results"]
        merged_results = vresult.get("merged_results", pd.DataFrame()).copy()
        best_oos_params = vresult.get("best_oos_params", {})
        best_oos_is = vresult.get("best_oos_is", {})

        # ── Re-apply min_bets and sort live (slider works without re-running) ──
        _param_join = ["value_threshold", "min_place_edge", "min_odds", "max_odds"]

        def _make_min_bets_mask(df: pd.DataFrame, sort_key: str, min_b: int) -> pd.Series:
            """Return a boolean mask enforcing min_bets per relevant leg."""
            _vb = df["value_bets"] if "value_bets" in df.columns else pd.Series(0, index=df.index)
            _eb = df["ew_bets"] if "ew_bets" in df.columns else pd.Series(0, index=df.index)
            if sort_key.startswith("ew"):
                return _eb >= min_b
            elif sort_key.startswith("value"):
                return _vb >= min_b
            else:
                # combined: require EACH active leg to meet the minimum individually
                _ew_active = (_eb > 0).any()
                if _ew_active:
                    return (_vb >= min_b) & (_eb >= min_b)
                return _vb >= min_b

        if not cal_results.empty:
            filtered = cal_results[_make_min_bets_mask(cal_results, cal_sort_by, cal_min_bets)] \
                .sort_values(cal_sort_by, ascending=False).reset_index(drop=True)
        else:
            filtered = cal_results

        # Derive IS-best from top of live-filtered results
        best = filtered.iloc[0].to_dict() if not filtered.empty else {}

        # Look up OOS performance for the live IS-best params
        best_val = {}
        if not val_results.empty and best:
            _vr_match = val_results.copy()
            for _pc in _param_join:
                if _pc in _vr_match.columns and _pc in best:
                    _vr_match = _vr_match[_vr_match[_pc] == best[_pc]]
            if not _vr_match.empty:
                best_val = _vr_match.iloc[0].to_dict()

        # Also filter merged_results by live min_bets for the stability view
        if not merged_results.empty:
            merged_results = merged_results[
                _make_min_bets_mask(merged_results, cal_sort_by, cal_min_bets)
            ].reset_index(drop=True)

        st.subheader("🏆 Results")

        # Overfitting warning
        _n_oos_eval = len(val_results) if not val_results.empty else 0
        st.warning(
            f"**Overfitting risk**: {vresult['combos_tried']:,} parameter combos were "
            f"searched on {vresult['cal_races']:,} calibration races · "
            f"{_n_oos_eval:,} combos evaluated on {vresult['val_races']:,} held-out races. "
            f"Trust OOS numbers more than in-sample."
        )

        # Apply stability filter (IS→OOS ROI decay) to merged results
        _display_filtered = filtered
        if not merged_results.empty and cal_max_decay < 100 and "roi_decay" in merged_results.columns:
            _decay_mask = merged_results["roi_decay"] <= cal_max_decay
            _display_filtered = merged_results[_decay_mask].reset_index(drop=True)
            st.caption(
                f"Stability filter active (max decay {cal_max_decay}pp): "
                f"**{len(_display_filtered):,}** / {len(merged_results):,} combos pass"
            )
        else:
            st.caption(
                f"Showing **{len(filtered):,}** / {len(cal_results):,} combos "
                f"(min {cal_min_bets} bets) ranked by **{_sort_options[cal_sort_by]}**"
            )

        if _display_filtered.empty or not best:
            st.warning("No combinations met the minimum bets filter. Lower the threshold.")
        else:
            _has_oos = bool(best_val)
            _has_oos_best = bool(best_oos_params)

            # ── IS-Best vs OOS-Best tabs ──────────────────────────────
            _is_best_tab, _oos_best_tab = st.tabs([
                "🥇 IS-Best (in-sample optimised)",
                "🛡️ OOS-Best (out-of-sample optimised)",
            ])

            with _is_best_tab:
                st.caption(
                    "Parameters that scored highest **in-sample**. "
                    "More likely to be overfit — check OOS performance below."
                )
                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                bc1.metric("Value Threshold", f"{best['value_threshold']:.3f}")
                bc2.metric("Min Place Edge", f"{best['min_place_edge']:.3f}")
                bc3.metric("Min Odds", f"{best['min_odds']:.1f}")
                bc4.metric("Max Odds", f"{best['max_odds']:.1f}")
                bc5.metric(
                    "Kelly Frac" if cal_staking == "kelly" else "Staking",
                    f"{best['kelly_fraction']:.2f}" if cal_staking == "kelly" else "Flat",
                )

                is_tab, oos_tab = st.tabs([
                    f"📈 In-Sample ({vresult['cal_races']} races)",
                    f"🔍 Out-of-Sample ({vresult['val_races']} races)",
                ])
                for _tab, _b, _label in [
                    (is_tab, best, "In-Sample"),
                    (oos_tab, best_val, "Out-of-Sample"),
                ]:
                    with _tab:
                        if not _b:
                            st.info("No out-of-sample data available.")
                            continue
                        pc1, pc2, pc3 = st.columns(3)
                        with pc1:
                            st.markdown("**Win Value Bets**")
                            _vb = int(_b.get("value_bets", 0))
                            st.metric("Bets", _vb)
                            st.metric("Staked", f"{_b.get('value_staked', _vb):,.1f}u")
                            st.metric("P&L", f"{_b.get('value_pnl', 0):+.2f}u")
                            st.metric("ROI", f"{_b.get('value_roi', 0):+.1f}%")
                            st.metric("Strike", f"{_b.get('value_strike', 0):.1f}%")
                            st.metric("Avg Odds", f"{_b.get('value_avg_odds', 0):.1f}")
                            st.metric("Max DD", f"{_b.get('value_max_dd', 0):.1f}u")
                            st.metric("Sharpe", f"{_b.get('value_sharpe', 0):.3f}")
                        with pc2:
                            st.markdown("**Each-Way**")
                            _eb = int(_b.get("ew_bets", 0))
                            st.metric("Bets", _eb)
                            st.metric("Staked", f"{_b.get('ew_staked', _eb * 2):,.1f}u")
                            st.metric("P&L", f"{_b.get('ew_pnl', 0):+.2f}u")
                            st.metric("ROI", f"{_b.get('ew_roi', 0):+.1f}%")
                            st.metric("Place Rate", f"{_b.get('ew_place_rate', 0):.1f}%")
                            st.metric("Avg Odds", f"{_b.get('ew_avg_odds', 0):.1f}")
                            st.metric("Max DD", f"{_b.get('ew_max_dd', 0):.1f}u")
                            st.metric("Sharpe", f"{_b.get('ew_sharpe', 0):.3f}")
                        with pc3:
                            st.markdown("**Combined**")
                            st.metric("Staked", f"{_b.get('combined_staked', 0):,.1f}u")
                            st.metric("P&L", f"{_b.get('combined_pnl', 0):+.2f}u")
                            st.metric("ROI", f"{_b.get('combined_roi', 0):+.1f}%")
                            st.metric("Sharpe", f"{_b.get('combined_sharpe', 0):.3f}")

                if _has_oos:
                    _oos_roi = best_val.get("combined_roi", 0)
                    _is_roi = best.get("combined_roi", 0)
                    _decay = _is_roi - _oos_roi
                    _color = "green" if _oos_roi > 0 else "red"
                    st.markdown(
                        f"**OOS Reality Check**: IS ROI **{_is_roi:+.1f}%** → "
                        f"OOS ROI **:{_color}[{_oos_roi:+.1f}%]** "
                        f"(decay: **{_decay:+.1f}pp**) · "
                        f"{int(best_val.get('value_bets', 0))} value bets, "
                        f"{int(best_val.get('ew_bets', 0))} EW bets"
                    )

                if _has_oos and best_val.get("combined_roi", 0) < 0:
                    st.warning(
                        "⚠️ The IS-best combo has **negative OOS ROI**. "
                        "Use the **OOS-Best** tab instead, or tighten the anti-overfit filter."
                    )

                if st.button(
                    "✅ Apply IS-Best to Sidebar",
                    type="secondary",
                    key="cal_apply_is_best",
                    help="Copy in-sample-best parameters to the sidebar Value Strategy config.",
                ):
                    _pending = {
                        "value_threshold": float(best["value_threshold"]),
                        "value_min_odds": float(best["min_odds"]),
                        "value_max_odds": float(best["max_odds"]),
                        "ew_min_place_edge": float(best["min_place_edge"]),
                        "ew_min_odds": float(best["min_odds"]),
                        "ew_max_odds": float(best["max_odds"]),
                    }
                    if cal_staking == "kelly":
                        _pending["kelly_fraction"] = float(best["kelly_fraction"])
                        _pending["staking_mode"] = "kelly"
                    st.session_state["_pending_value_config"] = _pending
                    st.success("IS-Best parameters applied to sidebar config!")
                    st.rerun()

            with _oos_best_tab:
                st.caption(
                    "Parameters that scored highest **out-of-sample** across evaluated combos. "
                    "These were not directly optimised on the validation set — "
                    "they are genuinely more likely to generalise."
                )
                if not _has_oos_best:
                    st.info("No OOS results available yet. Run calibration first.")
                else:
                    ob1, ob2, ob3, ob4, ob5 = st.columns(5)
                    ob1.metric("Value Threshold", f"{best_oos_params.get('value_threshold', 0):.3f}")
                    ob2.metric("Min Place Edge", f"{best_oos_params.get('min_place_edge', 0):.3f}")
                    ob3.metric("Min Odds", f"{best_oos_params.get('min_odds', 0):.1f}")
                    ob4.metric("Max Odds", f"{best_oos_params.get('max_odds', 0):.1f}")
                    ob5.metric(
                        "Kelly Frac" if cal_staking == "kelly" else "Staking",
                        f"{best_oos_params.get('kelly_fraction', 0):.2f}" if cal_staking == "kelly" else "Flat",
                    )

                    oo1, oo2 = st.columns(2)
                    with oo1:
                        st.markdown("**Out-of-Sample Performance**")
                        pc1, pc2, pc3 = st.columns(3)
                        with pc1:
                            st.markdown("**Value Bets**")
                            _vb = int(best_oos_params.get("value_bets", 0))
                            st.metric("Bets", _vb)
                            st.metric("ROI", f"{best_oos_params.get('value_roi', 0):+.1f}%")
                            st.metric("Strike", f"{best_oos_params.get('value_strike', 0):.1f}%")
                            st.metric("Sharpe", f"{best_oos_params.get('value_sharpe', 0):.3f}")
                        with pc2:
                            st.markdown("**Each-Way**")
                            _eb = int(best_oos_params.get("ew_bets", 0))
                            st.metric("Bets", _eb)
                            st.metric("ROI", f"{best_oos_params.get('ew_roi', 0):+.1f}%")
                            st.metric("Place Rate", f"{best_oos_params.get('ew_place_rate', 0):.1f}%")
                            st.metric("Sharpe", f"{best_oos_params.get('ew_sharpe', 0):.3f}")
                        with pc3:
                            st.markdown("**Combined**")
                            st.metric("P&L", f"{best_oos_params.get('combined_pnl', 0):+.2f}u")
                            st.metric("ROI", f"{best_oos_params.get('combined_roi', 0):+.1f}%")
                            st.metric("Sharpe", f"{best_oos_params.get('combined_sharpe', 0):.3f}")
                    with oo2:
                        st.markdown("**In-Sample Performance (same combo)**")
                        _is_roi_oos = best_oos_is.get("combined_roi", best_oos_params.get("combined_roi", 0))
                        _oos_roi_oos = best_oos_params.get("combined_roi", 0)
                        _decay_oos = _is_roi_oos - _oos_roi_oos
                        _color_oos = "green" if _oos_roi_oos > 0 else "red"
                        st.markdown(
                            f"IS ROI: **{_is_roi_oos:+.1f}%**\n\n"
                            f"OOS ROI: **:{_color_oos}[{_oos_roi_oos:+.1f}%]**\n\n"
                            f"Decay: **{_decay_oos:+.1f}pp** "
                            f"{'✅ stable' if abs(_decay_oos) < 10 else '⚠️ moderate' if abs(_decay_oos) < 25 else '🔴 high'}"
                        )

                    if st.button(
                        "✅ Apply OOS-Best to Sidebar",
                        type="primary",
                        key="cal_apply_oos_best",
                        help="Copy out-of-sample-best parameters to the sidebar Value Strategy config.",
                    ):
                        _pending = {
                            "value_threshold": float(best_oos_params["value_threshold"]),
                            "value_min_odds": float(best_oos_params["min_odds"]),
                            "value_max_odds": float(best_oos_params["max_odds"]),
                            "ew_min_place_edge": float(best_oos_params["min_place_edge"]),
                            "ew_min_odds": float(best_oos_params["min_odds"]),
                            "ew_max_odds": float(best_oos_params["max_odds"]),
                        }
                        if cal_staking == "kelly":
                            _pending["kelly_fraction"] = float(best_oos_params["kelly_fraction"])
                            _pending["staking_mode"] = "kelly"
                        st.session_state["_pending_value_config"] = _pending
                        st.success("OOS-Best parameters applied to sidebar config!")
                        st.rerun()

            # Full results table — merged IS+OOS with stability columns
            st.markdown("---")
            st.markdown("#### 📋 All Results (IS + OOS with Stability)")
            st.caption(
                "**roi_decay** = IS ROI − OOS ROI. Low values = stable (less overfit). "
                "**stability_ratio** = OOS ROI / IS ROI (1.0 = perfect generalisation)."
            )

            _display_cols = [
                "value_threshold", "min_place_edge", "min_odds", "max_odds",
            ]
            if cal_staking == "kelly":
                _display_cols.append("kelly_fraction")
            _display_cols += [
                "value_bets", "value_roi", "value_sharpe",
                "ew_bets", "ew_roi", "ew_sharpe",
                "combined_roi", "combined_sharpe",
            ]
            # Add OOS and stability columns if available
            if not merged_results.empty:
                for _sc in ["oos_combined_roi", "oos_combined_sharpe", "roi_decay", "stability_ratio"]:
                    if _sc in merged_results.columns:
                        _display_cols.append(_sc)

            _tbl_src = merged_results if not merged_results.empty else filtered
            _available = [c for c in _display_cols if c in _tbl_src.columns]
            _fmt_tbl = _tbl_src[_available].copy()
            for c in _fmt_tbl.columns:
                if _fmt_tbl[c].dtype in (np.float64, np.float32):
                    _fmt_tbl[c] = _fmt_tbl[c].round(3)

            st.dataframe(
                _fmt_tbl.head(200),
                width="stretch",
                hide_index=True,
            )

            # Heatmap: value_threshold vs min_place_edge → combined P&L
            st.markdown("#### 🗺️ Parameter Heatmaps (Calibration Half)")
            _heat_metric = st.selectbox(
                "Heatmap metric",
                ["combined_pnl", "combined_roi", "combined_sharpe",
                 "value_pnl", "value_roi", "ew_pnl", "ew_roi"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="cal_heatmap_metric",
            )

            # Fix non-varied dims to best values so heatmap matches best result
            hm_t1, hm_t2 = st.tabs([
                "Value Thresh × Place Edge",
                "Min Odds × Max Odds",
            ])

            with hm_t1:
                # Hold min_odds & max_odds at best values
                _hm1_df = filtered[
                    (filtered["min_odds"] == best["min_odds"])
                    & (filtered["max_odds"] == best["max_odds"])
                ]
                if cal_staking == "kelly":
                    _hm1_df = _hm1_df[_hm1_df["kelly_fraction"] == best["kelly_fraction"]]
                hm1 = _hm1_df.groupby(
                    ["value_threshold", "min_place_edge"]
                )[_heat_metric].first().reset_index()
                hm1_pivot = hm1.pivot(
                    index="value_threshold",
                    columns="min_place_edge",
                    values=_heat_metric,
                )
                if not hm1_pivot.empty:
                    fig_hm1 = px.imshow(
                        hm1_pivot.values,
                        x=[str(c) for c in hm1_pivot.columns],
                        y=[str(r) for r in hm1_pivot.index],
                        labels=dict(x="Min Place Edge", y="Value Threshold", color=_heat_metric),
                        color_continuous_scale=["#ef4444", "#ffffff", "#22c55e"],
                        color_continuous_midpoint=0,
                        aspect="auto",
                        title=(
                            f"{_heat_metric.replace('_', ' ').title()}: Value Threshold × Place Edge"
                            f"  (odds {best['min_odds']:.0f}–{best['max_odds']:.0f})"
                        ),
                    )
                    fig_hm1.update_layout(height=400)
                    st.plotly_chart(fig_hm1, width="stretch")
                else:
                    st.info("Not enough data for this heatmap.")

            with hm_t2:
                # Hold value_threshold & min_place_edge at best values
                _hm2_df = filtered[
                    (filtered["value_threshold"] == best["value_threshold"])
                    & (filtered["min_place_edge"] == best["min_place_edge"])
                ]
                if cal_staking == "kelly":
                    _hm2_df = _hm2_df[_hm2_df["kelly_fraction"] == best["kelly_fraction"]]
                hm2 = _hm2_df.groupby(
                    ["min_odds", "max_odds"]
                )[_heat_metric].first().reset_index()
                hm2_pivot = hm2.pivot(
                    index="min_odds",
                    columns="max_odds",
                    values=_heat_metric,
                )
                if not hm2_pivot.empty:
                    fig_hm2 = px.imshow(
                        hm2_pivot.values,
                        x=[str(c) for c in hm2_pivot.columns],
                        y=[str(r) for r in hm2_pivot.index],
                        labels=dict(x="Max Odds", y="Min Odds", color=_heat_metric),
                        color_continuous_scale=["#ef4444", "#ffffff", "#22c55e"],
                        color_continuous_midpoint=0,
                        aspect="auto",
                        title=(
                            f"{_heat_metric.replace('_', ' ').title()}: Min Odds × Max Odds"
                            f"  (vt={best['value_threshold']:.2f}, mpe={best['min_place_edge']:.2f})"
                        ),
                    )
                    fig_hm2.update_layout(height=400)
                    st.plotly_chart(fig_hm2, width="stretch")
                else:
                    st.info("Not enough data for this heatmap.")

            # Download
            st.markdown("---")
            st.download_button(
                "📥 Download full results CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="strategy_calibration.csv",
                mime="text/csv",
            )
# =====================================================================
#  MODEL INSIGHTS
# =====================================================================
elif page == "📈 Model Insights":
    st.title("📈 Model Insights")

    if st.session_state.predictor is None:
        if os.path.exists(_ENSEMBLE_MODEL_PATH):
            load_existing_model()
            load_model_data(force=True)
        else:
            st.warning("No model available. Train a model first.")
            st.stop()

    predictor = st.session_state.predictor

    tab_fi, tab_overfit = st.tabs(
        [
            "🔑 Feature Importance",
            "🔬 Overfitting Diagnostics",
        ]
    )

    # ── Feature Importance ───────────────────────────────────────────
    with tab_fi:
        # Build a map of available sub-models on the predictor
        _fi_model_map = {
            "classifier":("Win Classifier",   getattr(predictor, "clf_model",   None)),
            "ranker":    ("Race Ranker",      getattr(predictor, "ranker_model", None)),
            "place":     ("Place Classifier",  getattr(predictor, "place_model", None)),
        }
        _fi_available = {k: (label, m) for k, (label, m) in _fi_model_map.items()
                         if m is not None and hasattr(m, "feature_importances_")}
        feat_cols = getattr(predictor, "feature_cols", None)

        if _fi_available and feat_cols is not None:
            _fi_sel = st.selectbox(
                "Sub-model",
                options=list(_fi_available.keys()),
                format_func=lambda k: _fi_available[k][0],
                key="fi_submodel",
            )
            model_obj = _fi_available[_fi_sel][1]
            top_n = st.slider(
                "Show top N features", 10, 50, 25, key="fi_top",
            )
            fi = get_feature_importance(
                model_obj, feat_cols, top_n=top_n,
            )

            fig = px.bar(
                fi.sort_values("importance"),
                x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Viridis",
                title=f"Top {top_n} Features",
            )
            fig.update_layout(
                height=max(400, top_n * 28),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, width="stretch")

            _full_importances = get_feature_importance(
                model_obj, feat_cols, top_n=len(feat_cols),
            )
            _conc = _build_concentration_charts(
                tuple(_full_importances["importance"].tolist()),
                tuple(_full_importances["feature"].tolist()),
            )

            st.markdown("### 📐 Feature Importance Concentration")
            st.caption(
                "A model that relies heavily on a few features is more brittle. "
                "The Lorenz curve shows how evenly importance is distributed; "
                "the closer to the diagonal, the more balanced."
            )
            _conc_stats = st.columns(4)
            _conc_stats[0].metric("Gini", f"{_conc['gini']:.3f}")
            _conc_stats[1].metric("Top 5 share", f"{_conc['top5']:.1%}")
            _conc_stats[2].metric("Top 10 share", f"{_conc['top10']:.1%}")
            _conc_stats[3].metric("Top 20 share", f"{_conc['top20']:.1%}")
            st.plotly_chart(_conc["lorenz_fig"], width="stretch")

            st.markdown("#### 📋 All Feature Importances")
            st.dataframe(
                _full_importances.style.format({"importance": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )
        else:
            if predictor is None:
                st.info(
                    "Feature importance is not available. "
                    "Train a model first on the Train & Tune page."
                )
            else:
                st.info(
                    "No sub-models with feature importances found on the active predictor. "
                    "The active model may not support this — try re-training."
                )

    # ── Overfitting Diagnostics ──────────────────────────────────────
    with tab_overfit:
        st.subheader("🔬 Model Health & Diagnostics")
        st.caption(
            "This view separates three different failure modes: generalization gap "
            "(OOF vs validation), calibration quality, and stability across runs or walk-forward folds."
        )

        _active_rid = st.session_state.get("active_run_id")
        _run_meta_diag = None
        _run_full_diag = None
        _test_metrics_diag = st.session_state.get("metrics") or {}
        _train_metrics_diag = {}
        _wf_summary_diag = pd.DataFrame()

        if _active_rid:
            try:
                _run_meta_diag = load_run_meta(_active_rid)
                _train_metrics_diag = _run_meta_diag.get("train_metrics") or {}
                if not _test_metrics_diag:
                    _test_metrics_diag = _run_meta_diag.get("metrics") or {}
            except Exception:
                _run_meta_diag = None

            try:
                _run_full_diag = load_run(_active_rid)
                _wf_summary_diag = _run_full_diag.get("wf_summary_df")
                if not isinstance(_wf_summary_diag, pd.DataFrame):
                    _wf_summary_diag = pd.DataFrame()
            except Exception:
                _run_full_diag = None
                _wf_summary_diag = pd.DataFrame()

        _diag_df = _build_generalization_frame(_train_metrics_diag, _test_metrics_diag)

        if _diag_df.empty and not _test_metrics_diag:
            st.info(
                "Run diagnostics are not available for the active model yet. "
                "Train or load a saved run with evaluation metrics to populate this page."
            )
        else:
            _flagged_df = _diag_df[_diag_df["status"] != "Healthy"].copy() if not _diag_df.empty else pd.DataFrame()
            _worst_gap = float(_diag_df["risk_gap"].max()) if not _diag_df.empty else np.nan
            _avg_gap = float(_diag_df["risk_gap"].mean()) if not _diag_df.empty else np.nan
            _validation_eces = []
            for _mk, _mv in (_test_metrics_diag or {}).items():
                if isinstance(_mv, dict) and isinstance(_mv.get("ece"), (int, float, np.integer, np.floating)):
                    _validation_eces.append(float(_mv.get("ece")))
            _worst_ece = max(_validation_eces) if _validation_eces else np.nan
            _healthy_share = (
                float((_diag_df["status"] == "Healthy").mean())
                if not _diag_df.empty else np.nan
            )

            _sum_cols = st.columns(4)
            _sum_cols[0].metric("Worst Risk Gap", "-" if pd.isna(_worst_gap) else f"{_worst_gap:.3f}")
            _sum_cols[1].metric("Average Gap", "-" if pd.isna(_avg_gap) else f"{_avg_gap:.3f}")
            _sum_cols[2].metric("Worst Validation ECE", "-" if pd.isna(_worst_ece) else f"{_worst_ece:.4f}")
            _sum_cols[3].metric("Healthy Metric Share", "-" if pd.isna(_healthy_share) else f"{_healthy_share:.0%}")

            _tradeoff_df, _trade_left_label, _trade_right_label = _build_model_tradeoff_frame(
                _test_metrics_diag,
                ("ranker",),
                ("win_classifier", "classifier"),
            )
            if not _tradeoff_df.empty:
                st.markdown("### ⚔️ Race Ranker vs Win Classifier")
                st.caption(
                    "Positive edge means the race ranker is better after adjusting for whether the metric is "
                    "higher-is-better or lower-is-better. This makes the forecasting vs value-betting tradeoff explicit."
                )

                _left_wins = int((_tradeoff_df["edge"] > 1e-12).sum())
                _right_wins = int((_tradeoff_df["edge"] < -1e-12).sum())
                _value_edge = _tradeoff_df.loc[_tradeoff_df["metric_key"] == "value_bet_roi", "edge"]
                _predictive = _tradeoff_df[_tradeoff_df["family"].isin(["Calibration", "Generalization"])]
                _predictive_edge = float(_predictive["edge"].mean()) if not _predictive.empty else np.nan
                _best_ranker_row = _tradeoff_df.iloc[0]
                _best_win_row = _tradeoff_df.sort_values("edge", ascending=True).iloc[0]

                _trade_cols = st.columns(4)
                _trade_cols[0].metric(f"{_trade_left_label} Better On", f"{_left_wins}/{len(_tradeoff_df)}")
                _trade_cols[1].metric(f"{_trade_right_label} Better On", f"{_right_wins}/{len(_tradeoff_df)}")
                _trade_cols[2].metric(
                    "Avg Predictive Edge",
                    "-" if pd.isna(_predictive_edge) else f"{_predictive_edge:+.4f}",
                )
                _trade_cols[3].metric(
                    "ROI Edge",
                    "-" if _value_edge.empty else f"{float(_value_edge.iloc[0]):+.2%}",
                )

                _trade_left, _trade_right = st.columns([1.1, 0.9])
                with _trade_left:
                    _plot_df = _tradeoff_df.sort_values("edge", ascending=True).copy()
                    _plot_df["color_group"] = np.where(_plot_df["edge"] >= 0, _trade_left_label, _trade_right_label)
                    _trade_fig = px.bar(
                        _plot_df,
                        x="edge",
                        y="metric",
                        color="color_group",
                        orientation="h",
                        hover_data={
                            "family": True,
                            "left_value": ":.4f",
                            "right_value": ":.4f",
                            "edge": ":.4f",
                            "color_group": False,
                        },
                        color_discrete_map={
                            _trade_left_label: "#2563eb",
                            _trade_right_label: "#f59e0b",
                        },
                        title="Direction-aware edge by metric",
                    )
                    _trade_fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
                    _trade_fig.update_layout(height=420, xaxis_title="Edge", yaxis_title="")
                    st.plotly_chart(_trade_fig, width="stretch")

                with _trade_right:
                    st.markdown(
                        f"**Biggest {_trade_left_label} edge:** {_best_ranker_row['metric']} ({_best_ranker_row['edge']:+.4f})"
                    )
                    st.markdown(
                        f"**Biggest {_trade_right_label} edge:** {_best_win_row['metric']} ({abs(float(_best_win_row['edge'])):+.4f})"
                    )
                    st.dataframe(
                        _tradeoff_df[["metric", "family", "left_value", "right_value", "edge", "leader"]]
                        .rename(columns={
                            "metric": "Metric",
                            "family": "Family",
                            "left_value": _trade_left_label,
                            "right_value": _trade_right_label,
                            "edge": "Edge",
                            "leader": "Leader",
                        })
                        .style.format({
                            _trade_left_label: "{:.4f}",
                            _trade_right_label: "{:.4f}",
                            "Edge": "{:+.4f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

                st.markdown("---")

            # ── Lift over the baselines ───────────────────────────
            _wc_payload = (
                _test_metrics_diag.get("win_classifier") or _test_metrics_diag.get("classifier")
                if isinstance(_test_metrics_diag, dict) else None
            )
            _baseline_rows = [
                ("baseline_win", "Linear baseline",
                 "Logistic regression on the same features — beats it or the tree complexity isn't paying for itself."),
                ("baseline_market", "Market baseline",
                 "Overround-normalised implied odds — beats it or the model knows nothing the market hasn't priced in."),
            ]
            if isinstance(_wc_payload, dict):
                _wc_ndcg = _first_metric_value(_wc_payload, "ndcg_at_1")
                _wc_brier = _first_metric_value(_wc_payload, "brier_score")
                _lift_shown = False
                for _bl_key, _bl_label, _bl_help in _baseline_rows:
                    _bl_payload = _test_metrics_diag.get(_bl_key)
                    if not isinstance(_bl_payload, dict):
                        continue
                    _bl_ndcg = _first_metric_value(_bl_payload, "ndcg_at_1")
                    _bl_brier = _first_metric_value(_bl_payload, "brier_score")
                    if None in (_bl_ndcg, _wc_ndcg, _bl_brier, _wc_brier):
                        continue
                    if not _lift_shown:
                        st.markdown("### 📏 Lift vs Baselines")
                        st.caption(
                            "How much the win classifier beats the reference models. "
                            "The market baseline is the bar that matters for betting."
                        )
                        _lift_shown = True
                    _bl_cols = st.columns([1, 1, 2])
                    _bl_cols[0].metric(
                        f"{_bl_label}: NDCG@1 lift",
                        _fmt_metric(_wc_ndcg - _bl_ndcg, "+.4f"),
                        help=f"{_bl_help} Win classifier {_wc_ndcg:.4f} vs {_bl_ndcg:.4f}.",
                    )
                    _bl_cols[1].metric(
                        f"{_bl_label}: Brier improvement",
                        _fmt_metric(_bl_brier - _wc_brier, "+.6f"),
                        help=f"Positive = win classifier better calibrated ({_wc_brier:.6f} vs {_bl_brier:.6f}).",
                    )
                    if _wc_ndcg <= _bl_ndcg:
                        _bl_cols[2].warning(f"⚠️ {_bl_label} ranks winners as well as the win classifier.")
                    else:
                        _bl_cols[2].success(f"✅ Win classifier beats the {_bl_label.lower()}.")
                if _lift_shown:
                    st.markdown("---")

            _full_snapshot_df, _value_snapshot_df = _build_metric_snapshot_frame(_test_metrics_diag)
            _selector_explainer = _build_ranker_selector_explainer(_test_metrics_diag)
            if not _full_snapshot_df.empty or not _value_snapshot_df.empty:
                st.markdown("### 🎯 Full Field vs Value-Bet Subset")
                st.caption(
                    "Full-field metrics score every runner in every race. Value-bet metrics score only the runners that "
                    "passed the betting threshold. A model can be worse globally but better on the smaller subset it actually bets."
                )

                _exp_cols = st.columns(4)
                _exp_cols[0].metric(
                    "Ranker Full-Field Brier Edge",
                    _fmt_metric(_selector_explainer.get("full_brier_edge"), "+.4f"),
                )
                _exp_cols[1].metric(
                    "Ranker VB Brier Edge",
                    _fmt_metric(_selector_explainer.get("selected_brier_edge"), "+.4f"),
                )
                _exp_cols[2].metric(
                    "Ranker ROI Edge",
                    _fmt_metric(_selector_explainer.get("roi_edge"), "+.2%"),
                )
                _exp_cols[3].metric(
                    "Bet Count Delta",
                    _fmt_metric(_selector_explainer.get("bet_count_delta"), "+.0f"),
                )

                _snap_left, _snap_right = st.columns(2)
                with _snap_left:
                    st.markdown("#### Forecasting All Runners")
                    if _full_snapshot_df.empty:
                        st.info("No full-field metrics available for the active run.")
                    else:
                        st.dataframe(
                            _full_snapshot_df.style.format({
                                "Brier": "{:.4f}",
                                "Log Loss": "{:.4f}",
                                "ECE": "{:.4f}",
                                "Top-1 Accuracy": "{:.2%}",
                                "Winner in Top 3": "{:.2%}",
                                "RPS": "{:.4f}",
                            }),
                            width="stretch",
                            hide_index=True,
                        )

                with _snap_right:
                    st.markdown("#### Quality of Selected Bets")
                    if _value_snapshot_df.empty:
                        st.info("No value-bet subset metrics available for the active run.")
                    else:
                        st.dataframe(
                            _value_snapshot_df.style.format({
                                "Value Bets": "{:.0f}",
                                "VB Strike Rate": "{:.2%}",
                                "VB Brier": "{:.4f}",
                                "VB Log Loss": "{:.4f}",
                                "Avg Edge": "{:+.4f}",
                                "Avg CLV": "{:.3f}",
                                "Exp ROI": "{:.1f}%",
                                "ROI": "{:.2%}",
                            }),
                            width="stretch",
                            hide_index=True,
                        )

                st.markdown("---")

            _top_left, _top_right = st.columns([1.25, 1.0])

            with _top_left:
                st.markdown("### 📊 Generalization Map")
                st.caption(
                    "Points above the diagonal are better on OOF than validation for higher-is-better metrics; "
                    "for lower-is-better metrics the risk score already flips that direction internally."
                )
                if _diag_df.empty:
                    st.info("No comparable OOF and validation metrics were found for this run.")
                else:
                    _scatter_df = _diag_df.copy()
                    _scatter_df["hover_label"] = (
                        _scatter_df["sub_model"] + " • " + _scatter_df["metric"]
                    )
                    _scatter_colors = {
                        "Healthy": "#22c55e",
                        "Watch": "#f59e0b",
                        "Moderate": "#ef4444",
                        "High": "#991b1b",
                    }
                    _scatter_fig = px.scatter(
                        _scatter_df,
                        x="Validation",
                        y="OOF",
                        color="status",
                        symbol="family",
                        hover_name="hover_label",
                        hover_data={
                            "sub_model": True,
                            "metric": True,
                            "family": True,
                            "risk_gap": ":.4f",
                            "Validation": ":.4f",
                            "OOF": ":.4f",
                            "status": True,
                            "hover_label": False,
                        },
                        color_discrete_map=_scatter_colors,
                    )
                    _vals = pd.concat([_scatter_df["Validation"], _scatter_df["OOF"]], ignore_index=True)
                    _axis_min = float(_vals.min()) if not _vals.empty else 0.0
                    _axis_max = float(_vals.max()) if not _vals.empty else 1.0
                    _scatter_fig.add_trace(go.Scatter(
                        x=[_axis_min, _axis_max],
                        y=[_axis_min, _axis_max],
                        mode="lines",
                        name="Ideal parity",
                        line=dict(color="#94a3b8", dash="dash"),
                    ))
                    _scatter_fig.update_layout(height=430, legend_title_text="")
                    st.plotly_chart(_scatter_fig, width="stretch")

            with _top_right:
                st.markdown("### 🌡️ Risk by Model / Metric")
                st.caption(
                    "Positive values mean the metric got worse on validation after adjusting for metric direction."
                )
                if _diag_df.empty:
                    st.info("Risk heatmap unavailable for this run.")
                else:
                    _heat_df = _diag_df.pivot(index="sub_model", columns="metric", values="risk_gap")
                    _zmax = float(np.nanmax(np.abs(_heat_df.values))) if _heat_df.size else 0.0
                    _zmax = max(_zmax, 0.05)
                    _heat_fig = px.imshow(
                        _heat_df.values,
                        x=_heat_df.columns.tolist(),
                        y=_heat_df.index.tolist(),
                        color_continuous_scale=["#22c55e", "#f8fafc", "#ef4444"],
                        zmin=-_zmax,
                        zmax=_zmax,
                        text_auto=".3f",
                        aspect="auto",
                    )
                    _heat_fig.update_layout(height=430, coloraxis_colorbar_title="Risk gap")
                    st.plotly_chart(_heat_fig, width="stretch")

            st.markdown("### 🚩 Flagged Issues")
            _issue_left, _issue_right = st.columns([1.15, 0.85])

            with _issue_left:
                if _flagged_df.empty:
                    st.success("No material generalization gaps are currently flagged for this run.")
                else:
                    st.dataframe(
                        _flagged_df[["sub_model", "metric", "family", "OOF", "Validation", "risk_gap", "status"]]
                        .rename(columns={
                            "sub_model": "Sub-Model",
                            "metric": "Metric",
                            "family": "Family",
                            "risk_gap": "Risk Gap",
                            "status": "Status",
                        })
                        .style.format({
                            "OOF": "{:.4f}",
                            "Validation": "{:.4f}",
                            "Risk Gap": "{:.4f}",
                        }),
                        width="stretch",
                        hide_index=True,
                    )

            with _issue_right:
                if _diag_df.empty:
                    st.info("No model summary available.")
                else:
                    _model_summary = (
                        _diag_df.groupby("sub_model", as_index=False)
                        .agg(
                            avg_risk_gap=("risk_gap", "mean"),
                            max_risk_gap=("risk_gap", "max"),
                            flagged=("status", lambda s: int((s != "Healthy").sum())),
                        )
                        .sort_values("max_risk_gap", ascending=False)
                    )
                    _model_fig = px.bar(
                        _model_summary,
                        x="sub_model",
                        y="max_risk_gap",
                        color="flagged",
                        color_continuous_scale="OrRd",
                        hover_data={
                            "avg_risk_gap": ":.4f",
                            "max_risk_gap": ":.4f",
                            "flagged": True,
                        },
                        title="Worst risk gap by sub-model",
                    )
                    _model_fig.update_layout(height=300, xaxis_title="", yaxis_title="Risk gap")
                    st.plotly_chart(_model_fig, width="stretch")

            st.markdown("---")
            st.markdown("### 🎯 Calibration Quality")
            _cal_models = {
                _mk: _mv for _mk, _mv in (_test_metrics_diag or {}).items()
                if isinstance(_mv, dict) and (
                    isinstance(_mv.get("reliability_bins"), list)
                    or isinstance(_mv.get("decile_calibration"), list)
                )
            }
            if not _cal_models:
                st.info("Calibration diagnostics are not available for this run.")
            else:
                _cal_sel = st.selectbox(
                    "Calibration sub-model",
                    options=list(_cal_models.keys()),
                    format_func=_model_display_name,
                    key="diag_cal_model",
                )
                _cal_payload = _cal_models[_cal_sel]
                _cal_top = st.columns(4)
                for _col, (_label, _value, _fmt) in zip(_cal_top, _calibration_metric_cards(_cal_payload)):
                    _col.metric(_label, _fmt_metric(_value, _fmt))

                _secondary_cards: list[tuple[str, float | None, str]] = []
                _place_precision = _first_metric_value(_cal_payload, "place_precision")
                if _place_precision is not None:
                    _secondary_cards.append(("Place Precision", _place_precision, ".3f"))
                _ndcg1 = _first_metric_value(_cal_payload, "ndcg_at_1")
                if _ndcg1 is not None:
                    _secondary_cards.append(("NDCG@1", _ndcg1, ".4f"))
                _top1 = _first_metric_value(_cal_payload, "top1_accuracy")
                if _top1 is not None:
                    _secondary_cards.append(("Top-1 Accuracy", _top1, ".4f"))

                if _secondary_cards:
                    _secondary_cols = st.columns(len(_secondary_cards))
                    for _col, (_label, _value, _fmt) in zip(_secondary_cols, _secondary_cards):
                        _col.metric(_label, _fmt_metric(_value, _fmt))

                _cal_left, _cal_right = st.columns(2)
                with _cal_left:
                    _rel_bins = _cal_payload.get("reliability_bins") or []
                    if _rel_bins:
                        _rel_df = pd.DataFrame(_rel_bins)
                        _rel_fig = go.Figure()
                        _rel_fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode="lines", name="Ideal",
                            line=dict(color="#94a3b8", dash="dash"),
                        ))
                        _rel_fig.add_trace(go.Scatter(
                            x=_rel_df["mean_pred"],
                            y=_rel_df["obs_rate"],
                            mode="lines+markers",
                            name="Observed",
                            marker=dict(
                                size=np.clip(np.sqrt(_rel_df["count"].clip(lower=1)) * 1.2, 6, 24),
                                color="#2563eb",
                            ),
                            line=dict(color="#2563eb"),
                            customdata=np.stack([_rel_df["count"]], axis=1),
                            hovertemplate="Predicted=%{x:.3f}<br>Observed=%{y:.3f}<br>Count=%{customdata[0]:,.0f}<extra></extra>",
                        ))
                        _rel_fig.update_layout(
                            height=340,
                            title="Reliability Curve",
                            xaxis_title="Mean predicted probability",
                            yaxis_title="Observed win/place rate",
                        )
                        st.plotly_chart(_rel_fig, width="stretch")
                    else:
                        st.info("Reliability bins are not available for this sub-model.")

                with _cal_right:
                    _decile = _cal_payload.get("decile_calibration") or []
                    if _decile:
                        _dec_df = pd.DataFrame(_decile)
                        _dec_df["calibration_error"] = _dec_df["obs_rate"] - _dec_df["mean_pred"]
                        _dec_fig = px.bar(
                            _dec_df,
                            x="decile",
                            y="calibration_error",
                            color="calibration_error",
                            color_continuous_scale=["#dc2626", "#f8fafc", "#16a34a"],
                            title="Observed minus predicted by decile",
                            hover_data={
                                "mean_pred": ":.4f",
                                "obs_rate": ":.4f",
                                "count": True,
                                "calibration_error": ":.4f",
                            },
                        )
                        _dec_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
                        _dec_fig.update_layout(height=340, xaxis_title="Decile", yaxis_title="Calibration error")
                        st.plotly_chart(_dec_fig, width="stretch")
                    else:
                        st.info("Per-decile calibration detail is not available for this sub-model.")

            st.markdown("---")
            _bottom_left, _bottom_right = st.columns(2)

            with _bottom_left:
                st.markdown("### 🕒 Cross-Run Trend")
                _saved_runs = list_runs()
                _trend_rows = []
                for _run in _saved_runs:
                    _metrics = _run.get("metrics") or {}
                    _train = _run.get("train_metrics") or {}
                    for _mk, _mv in _metrics.items():
                        _tr_mv = _train.get(_mk, {}) if isinstance(_train.get(_mk, {}), dict) else {}
                        if not isinstance(_mv, dict):
                            continue
                        for _metric_name, _val in _mv.items():
                            _oof = _tr_mv.get(_metric_name)
                            if not _is_diagnostic_metric(_metric_name):
                                continue
                            if isinstance(_val, (int, float, np.integer, np.floating)) and isinstance(_oof, (int, float, np.integer, np.floating)):
                                _trend_rows.append((
                                    _run.get("run_id", ""),
                                    _run.get("name", _run.get("run_id", "")),
                                    _run.get("timestamp", ""),
                                    _mk,
                                    _metric_name,
                                    float(_oof),
                                    float(_val),
                                ))

                _metric_options = []
                if not _diag_df.empty:
                    _metric_options = list(
                        _diag_df[["model_key", "metric_key"]]
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    )
                elif _trend_rows:
                    _metric_options = sorted({(_r[3], _r[4]) for _r in _trend_rows})

                if not _metric_options:
                    st.info("Not enough saved-run history for a trend view yet.")
                else:
                    _trend_sel = st.selectbox(
                        "Trend metric",
                        options=_metric_options,
                        format_func=lambda item: f"{_model_display_name(item[0])} • {_metric_label(item[1])}",
                        key="diag_trend_metric",
                    )
                    _trend_out = _build_generalization_trend_chart(tuple(_trend_rows), _trend_sel[0], _trend_sel[1])
                    if _trend_out["fig"] is None:
                        st.info("Need at least two comparable saved runs to plot a trend.")
                    else:
                        st.plotly_chart(_trend_out["fig"], width="stretch")

            with _bottom_right:
                st.markdown("### 🔁 Walk-Forward Stability")
                if isinstance(_wf_summary_diag, pd.DataFrame) and not _wf_summary_diag.empty:
                    _wf_metric_options = [
                        _c for _c in ["ndcg_at_1", "top1_accuracy", "brier_score", "place_precision"]
                        if _c in _wf_summary_diag.columns
                    ]
                    if not _wf_metric_options:
                        st.info("Walk-forward summary is present, but no recognised diagnostic metrics were found.")
                    else:
                        _wf_metric = st.selectbox(
                            "Walk-forward metric",
                            options=_wf_metric_options,
                            format_func=_metric_label,
                            key="diag_wf_metric",
                        )
                        _wf_series = pd.to_numeric(_wf_summary_diag[_wf_metric], errors="coerce").dropna()
                        _wf_cols = st.columns(4)
                        _wf_cols[0].metric("Folds", int(len(_wf_series)))
                        _wf_cols[1].metric("Mean", _fmt_metric(_wf_series.mean(), ".4f"))
                        _wf_cols[2].metric("Std Dev", _fmt_metric(_wf_series.std(ddof=0), ".4f"))
                        _wf_cols[3].metric("Range", _fmt_metric(_wf_series.max() - _wf_series.min(), ".4f"))

                        _wf_plot = pd.DataFrame({
                            "Fold": np.arange(1, len(_wf_series) + 1),
                            "Value": _wf_series.values,
                        })
                        _wf_fig = px.line(
                            _wf_plot,
                            x="Fold",
                            y="Value",
                            markers=True,
                            title=f"Walk-forward {_metric_label(_wf_metric)} by fold",
                        )
                        _wf_fig.update_layout(height=380, xaxis_dtick=1, yaxis_title=_metric_label(_wf_metric))
                        st.plotly_chart(_wf_fig, width="stretch")
                else:
                    st.info(
                        "Walk-forward stability is only available for runs that were trained with walk-forward validation enabled."
                    )

            st.caption(
                "Feature concentration still matters, but it now lives in the Feature Importance tab so this page can stay focused on model health."
            )