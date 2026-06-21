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
from src.featured_dataset import get_featured_dataset
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
from src.run_store import save_run, list_runs as _raw_list_runs, load_run as _raw_load_run, load_run_meta as _raw_load_run_meta, delete_run as _raw_delete_run, rename_run, get_latest_run_id, restore_run_model, run_has_model, get_run_processed_path, get_run_featured_path, prune_runs, run_disk_usage, get_production_run_id
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
    """Load the promoted **production** model via the registry pointer, falling
    back to the legacy MODELS_DIR artifact. Cached so joblib deserialization
    happens once per session. The serving app loads ONLY the lean model — never
    a featured dataset — so startup stays fast."""
    try:
        from src.run_store import load_production_predictor
        p = load_production_predictor()
        if p is not None:
            return p
    except Exception:
        pass
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
    """Path to the canonical featured dataset.

    The single persistent FeatureStore is now the one source of truth, so this
    resolves to its ``featured.parquet`` first and only falls back to the
    legacy global file for back-compat with old on-disk datasets.
    """
    store_path = os.path.join(config.PROCESSED_DATA_DIR, "feature_store", "featured.parquet")
    if os.path.exists(store_path):
        return store_path
    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.csv")
    if os.path.exists(pq_path):
        return pq_path
    if os.path.exists(csv_path):
        return csv_path
    return None


def _global_processed_dataset_path() -> str | None:
    """Path to the canonical processed dataset (FeatureStore copy first)."""
    store_path = os.path.join(config.PROCESSED_DATA_DIR, "feature_store", "processed.parquet")
    if os.path.exists(store_path):
        return store_path
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

    # No frozen snapshot for this run: fall through so the caller serves the
    # single canonical feature store (load_model_data's store fallback). The
    # store is read-only here — we never trigger a (slow) rebuild on a load.
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
        meta = _dataset_meta_from_frame(
            featured, data_source="store", featured_path=path, origin="feature_store"
        )
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


# ── Auto-restore production run metadata (lightweight; no dataset load) ──
# The lean picks app never loads the multi-GB featured dataset at startup —
# the predictor is loaded lazily by the page bodies (load_existing_model),
# and the featured set only when a tab that needs it asks for it.
if st.session_state.active_run_id is None and st.session_state.metrics is None:
    _prod_id = get_production_run_id()
    if _prod_id is not None:
        try:
            _run = load_run(_prod_id)
            st.session_state.active_run_id = _prod_id
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
        except Exception:
            pass  # no production run yet or corrupt — graceful fallback


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
    if st.session_state.predictor is None and not load_existing_model():
        raise MatchbookAPIError(
            "No trained model is available. Promote one in the training app first."
        )

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
        "💰 Today's Picks",
        "🔌 Matchbook API",
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

# ── Active (production) model ─────────────────────────────────────────
# The serving app loads the promoted model lazily via the registry pointer —
# no run-switcher and no featured-dataset load at startup, so it stays fast.
# Train, compare and promote models in the training app (training_app.py).
_prod_id = get_production_run_id()
st.session_state.setdefault("active_run_id", _prod_id)

if st.session_state.predictor is not None:
    st.sidebar.success("✅ **Win + Place model loaded**")
    if _prod_id:
        try:
            st.sidebar.caption(f"🚀 {load_run_meta(_prod_id).get('name', _prod_id)}  ({_prod_id})")
        except Exception:
            pass
elif _prod_id or os.path.exists(_ENSEMBLE_MODEL_PATH):
    if st.sidebar.button("Load model", key="sidebar_load_model"):
        if load_existing_model():
            st.session_state.active_run_id = _prod_id
            st.rerun()
        else:
            st.sidebar.error("Failed to load model.")
else:
    st.sidebar.warning("⚠️ No model — train & promote one in the training app.")

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
if page == "💰 Today's Picks":
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

    # ── ensure model is loaded (production pointer; NO dataset load) ──
    # Live picks feature-engineer scraped racecards with bounded history, so we
    # never load the full featured training set here — keeps the page fast.
    if st.session_state.predictor is None and not load_existing_model():
        st.warning(
            "⚠️ No model available. Train and **promote** one in the "
            "training app (`training_app.py`)."
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

    # ── Custom race entry (manual) ───────────────────────────────────
    # Dataset-driven prediction tools (predict on historical loaded races)
    # live in the training app; the picks app stays lean and never loads
    # the full feature store here.
    st.markdown("---")
    st.markdown("### 🔧 Custom race entry")
    _model_df = None  # manual entry uses default option lists (no store load)
    with st.container():
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