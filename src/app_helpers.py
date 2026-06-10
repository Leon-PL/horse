"""
App Helper Layer
================
Pure/diagnostic helpers extracted from app.py: cached run-store
wrappers, chart builders for the Experiments and Shortcomings pages,
pace/ranker diagnostic panels, SHAP rendering, metric formatting and
the experiment log.

These functions hold no page logic — app.py imports them and remains
the single Streamlit entry point.
"""

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
    TripleEnsemblePredictor,
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

def _ordinal(n: int) -> str:
    """Return ordinal string for an integer, e.g. 1→'1st', 11→'11th'."""
    if n <= 0:
        return "?"
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][min(n % 10, 3)] if n % 10 < 4 else 'th'}"


_PACE_DISPLAY_COLUMNS = [
    "pace_style_index",
    "pace_rank_pct",
    "pace_front_runner_flag",
    "pace_closer_flag",
    "lone_speed_flag",
    "hot_pace_flag",
    "closer_pace_setup",
    "field_pace_pressure",
]

_PACE_REQUIRED_COLUMNS = [
    "pace_style_index",
    "pace_rank_pct",
    "field_pace_pressure",
]


def _attach_pace_diagnostics(
    predictions: pd.DataFrame,
    feature_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach pace-style diagnostics from the engineered feature frame."""
    if feature_df is None or feature_df.empty or predictions is None or predictions.empty:
        return predictions

    join_cols = [c for c in ["race_id", "horse_name"] if c in predictions.columns and c in feature_df.columns]
    pace_cols = [c for c in _PACE_DISPLAY_COLUMNS if c in feature_df.columns]
    if not join_cols or not pace_cols:
        return predictions

    pace_frame = feature_df[join_cols + pace_cols].drop_duplicates(subset=join_cols, keep="last")
    merged = predictions.merge(pace_frame, on=join_cols, how="left")
    return merged


def _has_pace_diagnostics(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty:
        return False
    return all(col in df.columns for col in _PACE_REQUIRED_COLUMNS)


def _pace_runner_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if bool(row.get("lone_speed_flag", False)):
        tags.append("Lone speed")
    elif bool(row.get("pace_front_runner_flag", False)):
        tags.append("Front-runner")

    if bool(row.get("pace_closer_flag", False)):
        tags.append("Closer")

    cps = pd.to_numeric(pd.Series([row.get("closer_pace_setup")]), errors="coerce").iloc[0]
    if pd.notna(cps) and float(cps) >= 0.08:
        tags.append("Closer setup")

    return tags


def _pace_race_summary(predictions: pd.DataFrame) -> str | None:
    if predictions is None or predictions.empty or "field_pace_pressure" not in predictions.columns:
        return None

    front_count = int(pd.to_numeric(predictions.get("pace_front_runner_flag", 0), errors="coerce").fillna(0).sum())
    closer_count = int(pd.to_numeric(predictions.get("pace_closer_flag", 0), errors="coerce").fillna(0).sum())
    lone_speed = bool(pd.to_numeric(predictions.get("lone_speed_flag", 0), errors="coerce").fillna(0).any())
    hot_pace = bool(pd.to_numeric(predictions.get("hot_pace_flag", 0), errors="coerce").fillna(0).any())

    parts = [f"{front_count} likely pace pushers", f"{closer_count} closer candidates"]
    if lone_speed:
        parts.append("lone-speed candidate present")
    elif hot_pace:
        parts.append("pace looks hot")
    else:
        parts.append("pace looks balanced")
    return " | ".join(parts)


def _render_pace_panel(predictions: pd.DataFrame, *, key: str) -> None:
    """Render a compact pace diagnostics table for one race."""
    if predictions is None or predictions.empty or "pace_style_index" not in predictions.columns:
        return

    summary = _pace_race_summary(predictions)
    if summary:
        st.caption(f"Pace setup: {summary}")

    panel = predictions.copy()
    panel["pace_tags"] = panel.apply(lambda row: ", ".join(_pace_runner_tags(row)) or "-", axis=1)
    panel["pace_index"] = pd.to_numeric(panel.get("pace_style_index"), errors="coerce")
    panel["pace_rank_pct_display"] = pd.to_numeric(panel.get("pace_rank_pct"), errors="coerce")
    panel["closer_setup"] = pd.to_numeric(panel.get("closer_pace_setup"), errors="coerce")

    shown = panel[[
        "predicted_rank",
        "horse_name",
        "pace_index",
        "pace_rank_pct_display",
        "closer_setup",
        "pace_tags",
    ]].copy()
    shown = shown.rename(columns={
        "predicted_rank": "#",
        "horse_name": "Horse",
        "pace_index": "Pace",
        "pace_rank_pct_display": "Pace Rank",
        "closer_setup": "Closer Setup",
        "pace_tags": "Notes",
    })

    with st.expander("🏇 Pace Diagnostics", expanded=False):
        st.dataframe(
            shown.style.format({
                "Pace": "{:+.2f}",
                "Pace Rank": "{:.0%}",
                "Closer Setup": "{:.0%}",
            }),
            hide_index=True,
            width="stretch",
        )


def _attach_ranker_diagnostics(
    predictions: pd.DataFrame,
    predictor: TripleEnsemblePredictor,
    feature_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if predictions is None or predictions.empty or feature_df is None or feature_df.empty:
        return predictions
    ranker_model = getattr(predictor, "ranker_model", None)
    if ranker_model is None:
        return predictions
    if not hasattr(predictor, "_prepare_prediction_frame"):
        return predictions

    join_cols = [c for c in ["race_id", "horse_name"] if c in predictions.columns and c in feature_df.columns]
    if "horse_name" not in join_cols:
        return predictions

    try:
        predict_df, X_scaled = predictor._prepare_prediction_frame(feature_df)
        ranker_scores = np.asarray(ranker_model.predict(X_scaled), dtype=np.float64)
    except Exception:
        return predictions

    ranker_frame = predict_df[join_cols].copy() if join_cols else predict_df[["horse_name"]].copy()
    ranker_frame["ranker_score"] = ranker_scores
    group_cols = [c for c in ["race_id"] if c in ranker_frame.columns]
    if group_cols:
        ranker_frame["ranker_rank"] = ranker_frame.groupby(group_cols, sort=False)["ranker_score"].rank(
            ascending=False,
            method="min",
        ).astype(int)
    else:
        ranker_frame["ranker_rank"] = ranker_frame["ranker_score"].rank(ascending=False, method="min").astype(int)

    ranker_frame = ranker_frame.drop_duplicates(subset=join_cols, keep="last")
    merged = predictions.merge(ranker_frame, on=join_cols, how="left")
    if "predicted_rank" in merged.columns and "ranker_rank" in merged.columns:
        merged["rank_disagreement"] = (
            pd.to_numeric(merged["predicted_rank"], errors="coerce")
            - pd.to_numeric(merged["ranker_rank"], errors="coerce")
        ).abs()
        merged["ranker_disagrees_top_pick"] = (
            (pd.to_numeric(merged["predicted_rank"], errors="coerce") == 1)
            != (pd.to_numeric(merged["ranker_rank"], errors="coerce") == 1)
        )
    return merged


def _render_ranker_disagreement_panel(predictions: pd.DataFrame, *, key: str) -> None:
    if predictions is None or predictions.empty or "ranker_rank" not in predictions.columns:
        return

    top_pick = predictions.loc[pd.to_numeric(predictions.get("predicted_rank"), errors="coerce") == 1]
    ranker_top = predictions.loc[pd.to_numeric(predictions.get("ranker_rank"), errors="coerce") == 1]
    top_pick_name = top_pick["horse_name"].iloc[0] if not top_pick.empty else None
    ranker_top_name = ranker_top["horse_name"].iloc[0] if not ranker_top.empty else None

    if top_pick_name and ranker_top_name and top_pick_name == ranker_top_name:
        st.caption(f"Ranker agrees with the win model on the top pick: {top_pick_name}.")
    elif top_pick_name and ranker_top_name:
        st.caption(f"Ranker disagreement: win model prefers {top_pick_name}, ranker prefers {ranker_top_name}.")

    panel = predictions.copy()
    panel["rank_disagreement"] = pd.to_numeric(panel.get("rank_disagreement"), errors="coerce")
    shown = panel[[c for c in ["predicted_rank", "ranker_rank", "horse_name", "win_probability", "rank_disagreement"] if c in panel.columns]].copy()
    shown = shown.rename(columns={
        "predicted_rank": "Win Rank",
        "ranker_rank": "Ranker Rank",
        "horse_name": "Horse",
        "win_probability": "Win Prob",
        "rank_disagreement": "Δ Rank",
    }).sort_values(["Δ Rank", "Win Rank"], ascending=[False, True], kind="stable")

    with st.expander("⚔️ Ranker Disagreement", expanded=False):
        st.dataframe(
            shown.style.format({
                "Win Prob": "{:.1%}",
                "Δ Rank": "{:.0f}",
            }),
            hide_index=True,
            width="stretch",
        )


def _ranker_consensus_state(predictions: pd.DataFrame) -> tuple[str, str]:
    if predictions is None or predictions.empty or "ranker_rank" not in predictions.columns:
        return "", ""

    panel = predictions.copy()
    panel["predicted_rank"] = pd.to_numeric(panel.get("predicted_rank"), errors="coerce")
    panel["ranker_rank"] = pd.to_numeric(panel.get("ranker_rank"), errors="coerce")
    panel["rank_disagreement"] = pd.to_numeric(panel.get("rank_disagreement"), errors="coerce")

    top_pick = panel.loc[panel["predicted_rank"] == 1, "horse_name"]
    ranker_top = panel.loc[panel["ranker_rank"] == 1, "horse_name"]
    same_top = bool(not top_pick.empty and not ranker_top.empty and top_pick.iloc[0] == ranker_top.iloc[0])

    top_band = panel.loc[(panel["predicted_rank"] <= 3) | (panel["ranker_rank"] <= 3)]
    mean_gap = float(top_band["rank_disagreement"].dropna().mean()) if not top_band.empty else float("nan")

    if same_top and np.isfinite(mean_gap) and mean_gap <= 0.75:
        return "🟢 Strong consensus", "Win model and ranker agree tightly at the top of the race."
    if same_top and np.isfinite(mean_gap) and mean_gap <= 1.5:
        return "🟡 Moderate consensus", "Top pick agrees, but the rest of the ordering is less stable."
    if same_top:
        return "🟡 Narrow consensus", "They agree on the winner, but disagree materially underneath."
    return "🔴 Split view", "The ranker and win model disagree on the top of the race."


def _render_ranker_consensus_badge(predictions: pd.DataFrame) -> None:
    label, detail = _ranker_consensus_state(predictions)
    if not label:
        return
    st.caption(f"{label} — {detail}")


def _bet_confidence_state(row: pd.Series) -> tuple[str, str]:
    if pd.isna(row.get("ranker_rank")):
        return "", ""
    disagreement = pd.to_numeric(pd.Series([row.get("rank_disagreement")]), errors="coerce").iloc[0]
    if bool(row.get("ranker_disagrees_top_pick", False)) and int(pd.to_numeric(row.get("predicted_rank"), errors="coerce") or 0) == 1:
        return "⚠ Split", "Ranker does not back the model's top pick."
    if pd.notna(disagreement) and float(disagreement) >= 2:
        return "⚠ Caution", "Ranker is materially lower on this runner."
    if pd.notna(disagreement) and float(disagreement) <= 0:
        return "✅ Aligned", "Win model and ranker agree on this runner's rank."
    if pd.notna(disagreement) and float(disagreement) <= 1:
        return "✓ Supported", "Ranker broadly supports this runner."
    return "", ""


# ── Cached wrappers for disk I/O ─────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def list_runs() -> list[dict]:
    return _raw_list_runs()


@st.cache_data(ttl=60, show_spinner=False)
def load_run(run_id: str) -> dict:
    return _raw_load_run(run_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_run_meta(run_id: str) -> dict:
    return _raw_load_run_meta(run_id)


def delete_run(run_id: str) -> bool:
    result = _raw_delete_run(run_id)
    if result:
        list_runs.clear()
        load_run.clear()
        load_run_meta.clear()
    return result


@st.cache_data(ttl=30, show_spinner=False)
def _cached_db_stats() -> dict:
    try:
        return _raw_db_stats()
    except Exception:
        return {}


def _format_duration_compact(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _progress_timing_text(started_at: float, step_started_at: float, stage_progress: float) -> str:
    now = time.time()
    parts = [
        f"elapsed {_format_duration_compact(now - started_at)}",
        f"step {_format_duration_compact(now - step_started_at)}",
    ]
    if 0.02 <= stage_progress < 1.0:
        eta_seconds = max((now - step_started_at) * ((1.0 - stage_progress) / stage_progress), 0.0)
        parts.append(f"eta {_format_duration_compact(eta_seconds)}")
    return " · ".join(parts)



def _value_odds_range(value_config: dict | None) -> tuple[float, float]:
    cfg = value_config or {}
    min_odds = float(cfg.get("value_min_odds", 1.0))
    max_odds = float(cfg.get("value_max_odds", 101.0))
    if min_odds >= max_odds:
        return 1.0, 101.0
    return min_odds, max_odds


def _value_bet_mask(frame: pd.DataFrame, value_config: dict | None) -> pd.Series:
    if frame is None or frame.empty or "value_score" not in frame.columns or "odds" not in frame.columns:
        return pd.Series(False, index=getattr(frame, "index", pd.Index([])))

    cfg = value_config or {}
    base_thresh = float(cfg.get("value_threshold", 0.05))
    min_odds, max_odds = _value_odds_range(cfg)
    odds = pd.to_numeric(frame["odds"], errors="coerce")
    dyn_thresh = dynamic_value_threshold(base_thresh, odds)
    return (
        pd.to_numeric(frame["value_score"], errors="coerce").gt(dyn_thresh)
        & odds.ge(min_odds)
        & odds.le(max_odds)
    ).fillna(False)


def _invalidate_run_caches():
    """Call after saving a new run or deleting one."""
    list_runs.clear()
    load_run.clear()
    _build_overfit_section_charts.clear()
    _build_trend_chart.clear()
    # Clear cached predictions so a model switch takes effect immediately
    for _k in ("live_preds", "picks_preds", "picks_meta"):
        st.session_state.pop(_k, None)
    st.session_state.pop("picks_explanations", None)
    # Clear strategy calibrator caches tied to a specific model/data state
    for _k in list(st.session_state.keys()):
        if _k == "cal_analysis_df" or _k == "cal_results" or _k == "cal_analysis_sig" or _k.startswith("cal_precomputed_"):
            st.session_state.pop(_k, None)


def _calibration_signature(predictor, featured_df: pd.DataFrame) -> tuple:
    """Build a lightweight signature to detect stale calibrator state."""
    _min_dt = featured_df["race_date"].min() if "race_date" in featured_df.columns and not featured_df.empty else None
    _max_dt = featured_df["race_date"].max() if "race_date" in featured_df.columns and not featured_df.empty else None
    return (
        st.session_state.get("active_run_id"),
        type(predictor).__name__ if predictor is not None else None,
        tuple(sorted(getattr(predictor, "frameworks", {}).items())) if predictor is not None else (),
        tuple((k, round(float(v), 6)) for k, v in sorted(getattr(predictor, "weights", {}).items())) if predictor is not None else (),
        round(float(getattr(config, "TEST_SIZE", 0.2)), 6),
        len(featured_df) if featured_df is not None else 0,
        str(_min_dt),
        str(_max_dt),
    )


def _build_run_name(
    *,
    data_source: str,
    days_back: int | None,
    include_odds: bool,
    tune_mode_label: str,
    auto_trials: int,
) -> str:
    """Create a compact, informative run name for training."""
    _src = {
        "database": "db",
        "scrape": "scr",
        "sample": "smp",
    }.get(data_source, str(data_source)[:3])
    _days = f"{int(days_back)}d" if days_back is not None else "na"
    _odds = "oddsOn" if include_odds else "oddsOff"
    _tune = "saved" if tune_mode_label == "📦 Saved Autotune" else "manual"
    return f"ens_{_src}_{_days}_{_odds}_{_tune}_{datetime.now():%m%d_%H%M}"


# Columns derived from the raw 'odds' column. When odds are disabled these
# are dropped before training/autotuning so the model never sees market data.
_ODDS_DERIVED_COLS = [
    "implied_prob", "norm_implied_prob", "odds_rank",
    "is_favourite", "log_odds", "odds_vs_field", "overround",
    "odds_cv", "implied_prob_vs_base",
    "jockey_elo_x_fav",
    "mkt_x_win_rate", "logodds_x_elo",
    "odds_field_x_dropped", "mkt_x_speed",
    "odds_field_x_jock_elo",
    "odds_vs_elo_rank",
    "trainer_runner_rank_by_odds",
    "trainer_first_string_by_odds",
    "trainer_odds_vs_stablemate_best",
    "trainer_stable_market_share",
    "trainer_first_string_odds_or_agree",
    "trainer_first_string_odds_elo_agree",
    "trainer_string_disagreement_or_odds",
    "trainer_string_disagreement_elo_odds",
    "trainer_first_string_stable_jockey",
    "beaten_fav_last",
]


def _drop_market_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    to_drop = [col for col in _ODDS_DERIVED_COLS if col in df.columns]
    if not to_drop:
        return df, []
    return df.drop(columns=to_drop), to_drop


# ── Cached chart builders for Overfitting Diagnostics ─────────────────
# Plotly figure construction is expensive (~1.3 s for 7 charts).
# These functions cache the built figures so subsequent Streamlit
# reruns with the same data return instantly.

@st.cache_resource(show_spinner=False)
def _build_overfit_section_charts(run_id: str) -> dict:
    """Build bar charts, heatmap, and display DataFrame for sections 1-3.

    Keyed by run_id only — metrics are immutable once a run is saved, so
    there is no need to serialise the full metrics dicts on every rerun just
    to form a cache key.
    """
    try:
        _meta = _raw_load_run_meta(run_id)
    except Exception:
        return {"overfit_figs": [], "heatmap_fig": None, "display_df": None}
    test_metrics = _meta.get("metrics") or {}
    train_metrics = _meta.get("train_metrics") or {}

    key_metrics = ["ndcg_at_1", "ndcg_at_3", "top1_accuracy", "win_in_top3",
                    "brier_calibrated", "brier_raw", "place_precision"]
    rows: list[dict] = []
    for model_key in test_metrics:
        test_m = test_metrics.get(model_key, {})
        train_m = train_metrics.get(model_key, {})
        if not isinstance(test_m, dict) or not isinstance(train_m, dict):
            continue
        for mk in key_metrics:
            tv = test_m.get(mk)
            trv = train_m.get(mk)
            if tv is not None and trv is not None:
                rows.append({
                    "Sub-Model": model_key.replace("_", " ").title(),
                    "Metric": mk,
                    "OOF": round(float(trv), 4),
                    "Validation": round(float(tv), 4),
                    "Gap": round(float(trv) - float(tv), 4),
                })

    if not rows:
        return {"overfit_figs": [], "heatmap_fig": None, "display_df": None}

    of_df = pd.DataFrame(rows)

    # Bar charts — one per metric
    overfit_figs = []
    for mk in of_df["Metric"].unique():
        mdf = of_df[of_df["Metric"] == mk]
        melt = mdf.melt(
            id_vars=["Sub-Model"],
            value_vars=["OOF", "Validation"],
            var_name="Split",
            value_name="Score",
        )
        fig = px.bar(
            melt, x="Sub-Model", y="Score", color="Split",
            barmode="group",
            title=mk.replace("_", " ").upper(),
            color_discrete_map={"OOF": "#3b82f6", "Validation": "#22c55e"},
        )
        fig.update_layout(height=350, legend_title_text="")
        overfit_figs.append(fig)

    # Heatmap
    gap_pivot = of_df.pivot(index="Sub-Model", columns="Metric", values="Gap")
    heatmap_fig = px.imshow(
        gap_pivot.values,
        x=gap_pivot.columns.tolist(),
        y=gap_pivot.index.tolist(),
        color_continuous_scale=["#22c55e", "#fbbf24", "#ef4444"],
        zmin=0, zmax=0.3,
        text_auto=".3f",
        title="OOF − Validation Gap (lower is better)",
        aspect="auto",
    )
    heatmap_fig.update_layout(height=350)

    # Display table
    # For Brier score, lower = better, so overfitting = OOF < validation → gap is NEGATIVE.
    # For ranking metrics, higher = better, so overfitting = OOF > validation → gap is POSITIVE.
    _lower_better = {"brier_calibrated", "brier_raw"}
    display_df = of_df.copy()
    def _gap_status(row):
        g = row["Gap"]
        if row["Metric"] in _lower_better:
            g = -g  # flip: positive means overfit for lower-is-better metrics
        return "✅ OK" if g < 0.05 else ("⚠️ Moderate" if g < 0.15 else "🔴 High")
    display_df["Status"] = display_df.apply(_gap_status, axis=1)

    return {
        "overfit_figs": overfit_figs,
        "heatmap_fig": heatmap_fig,
        "display_df": display_df,
    }


def _metric_direction(metric_name: str) -> int:
    """Return +1 for higher-is-better metrics, -1 for lower-is-better."""
    lower_better = {
        "brier_calibrated",
        "brier_raw",
        "brier_score",
        "ece",
        "log_loss",
        "value_bet_brier",
        "value_bet_log_loss",
        "rps",
    }
    return -1 if metric_name in lower_better else 1


def _is_diagnostic_metric(metric_name: str) -> bool:
    """Return True only for scalar performance metrics suitable for diagnostics."""
    if not metric_name:
        return False

    metric_name = str(metric_name)
    allowed_metrics = {
        "accuracy",
        "avg_edge",
        "brier_calibrated",
        "brier_raw",
        "brier_score",
        "ece",
        "log_loss",
        "mae",
        "ndcg_at_1",
        "ndcg_at_3",
        "place_precision",
        "precision",
        "rank_ndcg_at_1",
        "rank_ndcg_at_3",
        "rank_top1_accuracy",
        "roc_auc",
        "rps",
        "top1_accuracy",
        "value_bet_brier",
        "value_bet_log_loss",
        "value_bet_roi",
        "value_bet_sr",
        "win_in_top3",
    }
    if metric_name in allowed_metrics:
        return True

    excluded_tokens = {
        "count",
        "decile",
        "fold",
        "races",
        "rows",
        "total",
    }
    metric_name_l = metric_name.lower()
    if any(token in metric_name_l for token in excluded_tokens):
        return False

    return False


def _metric_family(metric_name: str) -> str:
    """Group metrics into a small number of diagnostics families."""
    if metric_name in {"brier_calibrated", "brier_raw", "brier_score", "ece", "log_loss"}:
        return "Calibration"
    if metric_name in {"ndcg_at_1", "ndcg_at_3", "top1_accuracy", "top3_accuracy", "win_in_top3", "place_precision", "rps"}:
        return "Generalization"
    if metric_name.startswith("value_bet_") or metric_name in {"avg_edge", "value_bet_roi", "value_bet_sr"}:
        return "Value Betting"
    return "Other"


def _metric_label(metric_name: str) -> str:
    """Readable display labels for diagnostics metrics."""
    labels = {
        "ndcg_at_1": "NDCG@1",
        "ndcg_at_3": "NDCG@3",
        "top1_accuracy": "Top-1 Accuracy",
        "top3_accuracy": "Top-3 Accuracy",
        "win_in_top3": "Winner in Top 3",
        "place_precision": "Place Precision",
        "brier_calibrated": "Brier (Calibrated)",
        "brier_raw": "Brier (Raw)",
        "brier_score": "Brier",
        "ece": "ECE",
        "log_loss": "Log Loss",
        "rps": "RPS",
        "value_bet_brier": "Value Bet Brier",
        "value_bet_log_loss": "Value Bet Log Loss",
        "value_bet_roi": "Value Bet ROI",
        "value_bet_sr": "Value Bet Strike Rate",
        "avg_edge": "Average Edge",
    }
    return labels.get(metric_name, metric_name.replace("_", " ").title())


def _fmt_metric(value, fmt: str = ".4f", pct: bool = False) -> str:
    """Format scalar metrics safely for Streamlit metric cards."""
    if value is None:
        return "-"
    try:
        numeric_value = float(value)
    except Exception:
        return "-"
    if not np.isfinite(numeric_value):
        return "-"
    if pct:
        return f"{numeric_value:.1%}"
    return format(numeric_value, fmt)


def _first_metric_value(payload: dict | None, *keys: str):
    """Return the first finite scalar metric found in a payload."""
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            return float(value)
    return None


def _model_display_name(model_key: str) -> str:
    """Readable model labels for diagnostics views."""
    labels = {
        "win_classifier": "Win Classifier",
        "classifier": "Win Classifier",
        "place_classifier": "Place Classifier",
        "place": "Place Classifier",
        "ranker": "Race Ranker",
    }
    return labels.get(model_key, model_key.replace("_", " ").title())


def _calibration_metric_cards(payload: dict | None) -> list[tuple[str, float | None, str]]:
    """Return a consistent set of top-line calibration metrics for a sub-model."""
    return [
        ("ECE", _first_metric_value(payload, "ece"), ".4f"),
        ("Brier", _first_metric_value(payload, "brier_score", "brier_calibrated"), ".4f"),
        ("Raw Brier", _first_metric_value(payload, "brier_raw"), ".4f"),
        ("Log Loss", _first_metric_value(payload, "log_loss", "log_loss_raw"), ".4f"),
    ]


def _gap_status(risk_gap: float) -> str:
    """Bucket a direction-aware generalization gap into a compact status."""
    if risk_gap >= 0.15:
        return "High"
    if risk_gap >= 0.08:
        return "Moderate"
    if risk_gap >= 0.03:
        return "Watch"
    return "Healthy"


def _build_generalization_frame(train_metrics: dict | None, test_metrics: dict | None) -> pd.DataFrame:
    """Return one row per comparable metric with direction-aware gap scores."""
    rows: list[dict] = []
    train_metrics = train_metrics or {}
    test_metrics = test_metrics or {}
    model_keys = sorted(set(train_metrics) | set(test_metrics))

    for model_key in model_keys:
        train_model = train_metrics.get(model_key, {})
        test_model = test_metrics.get(model_key, {})
        if not isinstance(train_model, dict) or not isinstance(test_model, dict):
            continue

        for metric_name, test_value in test_model.items():
            train_value = train_model.get(metric_name)
            if not _is_diagnostic_metric(metric_name):
                continue
            if not isinstance(test_value, (int, float, np.integer, np.floating)):
                continue
            if not isinstance(train_value, (int, float, np.integer, np.floating)):
                continue
            if pd.isna(test_value) or pd.isna(train_value):
                continue

            direction = _metric_direction(metric_name)
            raw_gap = float(train_value) - float(test_value)
            risk_gap = raw_gap * direction

            rows.append({
                "model_key": model_key,
                "sub_model": _model_display_name(model_key),
                "metric_key": metric_name,
                "metric": _metric_label(metric_name),
                "family": _metric_family(metric_name),
                "OOF": float(train_value),
                "Validation": float(test_value),
                "raw_gap": raw_gap,
                "risk_gap": risk_gap,
                "abs_gap": abs(risk_gap),
                "status": _gap_status(risk_gap),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["risk_gap", "abs_gap"], ascending=[False, False]).reset_index(drop=True)
    return df


def _resolve_model_metric_payload(metrics_payload: dict | None, *model_keys: str) -> tuple[str | None, dict]:
    """Return the first matching model metrics payload from a saved metrics dict."""
    if not isinstance(metrics_payload, dict):
        return None, {}

    for model_key in model_keys:
        payload = metrics_payload.get(model_key)
        if isinstance(payload, dict):
            return model_key, payload

    return None, {}


def _build_model_tradeoff_frame(
    metrics_payload: dict | None,
    left_model_keys: tuple[str, ...],
    right_model_keys: tuple[str, ...],
) -> tuple[pd.DataFrame, str, str]:
    """Return a direction-aware comparison frame between two sub-models."""
    left_key, left_payload = _resolve_model_metric_payload(metrics_payload, *left_model_keys)
    right_key, right_payload = _resolve_model_metric_payload(metrics_payload, *right_model_keys)
    left_label = _model_display_name(left_key or left_model_keys[0])
    right_label = _model_display_name(right_key or right_model_keys[0])

    if not left_key or not right_key:
        return pd.DataFrame(), left_label, right_label

    rows: list[dict] = []
    for metric_name in sorted(set(left_payload) & set(right_payload)):
        if not _is_diagnostic_metric(metric_name):
            continue

        left_value = left_payload.get(metric_name)
        right_value = right_payload.get(metric_name)
        if not isinstance(left_value, (int, float, np.integer, np.floating)):
            continue
        if not isinstance(right_value, (int, float, np.integer, np.floating)):
            continue
        if pd.isna(left_value) or pd.isna(right_value):
            continue

        direction = _metric_direction(metric_name)
        edge = (float(left_value) - float(right_value)) * direction
        rows.append({
            "metric_key": metric_name,
            "metric": _metric_label(metric_name),
            "family": _metric_family(metric_name),
            "left_value": float(left_value),
            "right_value": float(right_value),
            "edge": float(edge),
            "abs_edge": abs(float(edge)),
            "leader": (
                left_label if edge > 1e-12 else
                right_label if edge < -1e-12 else
                "Tie"
            ),
        })

    if not rows:
        return pd.DataFrame(), left_label, right_label

    tradeoff_df = pd.DataFrame(rows)
    tradeoff_df = tradeoff_df.sort_values(["abs_edge", "edge"], ascending=[False, False]).reset_index(drop=True)
    return tradeoff_df, left_label, right_label


def _build_metric_snapshot_frame(metrics_payload: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact full-field and selected-bet metric tables by model."""
    if not isinstance(metrics_payload, dict):
        return pd.DataFrame(), pd.DataFrame()

    model_groups = [
        ("win_classifier", "classifier"),
        ("ranker",),
        ("place_classifier", "place"),
    ]

    full_rows: list[dict] = []
    value_rows: list[dict] = []

    for model_keys in model_groups:
        resolved_key, payload = _resolve_model_metric_payload(metrics_payload, *model_keys)
        if not resolved_key or not isinstance(payload, dict):
            continue

        label = _model_display_name(resolved_key)

        full_row = {
            "Model": label,
            "Brier": _first_metric_value(payload, "brier_score", "brier_calibrated"),
            "Log Loss": _first_metric_value(payload, "log_loss", "log_loss_raw"),
            "ECE": _first_metric_value(payload, "ece"),
            "Top-1 Accuracy": _first_metric_value(payload, "top1_accuracy", "rank_top1_accuracy", "accuracy"),
            "Winner in Top 3": _first_metric_value(payload, "win_in_top3"),
            "RPS": _first_metric_value(payload, "rps"),
        }
        if any(v is not None for k, v in full_row.items() if k != "Model"):
            full_rows.append(full_row)

        value_row = {
            "Model": label,
            "Value Bets": _first_metric_value(payload, "value_bets"),
            "VB Strike Rate": _first_metric_value(payload, "value_bet_sr"),
            "VB Brier": _first_metric_value(payload, "value_bet_brier"),
            "VB Log Loss": _first_metric_value(payload, "value_bet_log_loss"),
            "Avg Edge": _first_metric_value(payload, "avg_edge"),
            "Avg CLV": _first_metric_value(payload, "value_bet_avg_clv"),
            "Exp ROI": _first_metric_value(payload, "value_bet_exp_roi_pct"),
            "ROI": _first_metric_value(payload, "value_bet_roi"),
        }
        if any(v is not None for k, v in value_row.items() if k != "Model"):
            value_rows.append(value_row)

    return pd.DataFrame(full_rows), pd.DataFrame(value_rows)


def _build_ranker_selector_explainer(metrics_payload: dict | None) -> dict:
    """Return compact diagnostics explaining why ranker ROI can diverge."""
    _, ranker_payload = _resolve_model_metric_payload(metrics_payload, "ranker")
    _, win_payload = _resolve_model_metric_payload(metrics_payload, "win_classifier", "classifier")
    if not ranker_payload or not win_payload:
        return {}

    def _delta(metric_name: str, higher_is_better: bool) -> float | None:
        ranker_value = _first_metric_value(ranker_payload, metric_name)
        win_value = _first_metric_value(win_payload, metric_name)
        if ranker_value is None or win_value is None:
            return None
        raw = float(ranker_value) - float(win_value)
        return raw if higher_is_better else -raw

    return {
        "full_brier_edge": _delta("brier_score", higher_is_better=False),
        "full_log_loss_edge": _delta("log_loss", higher_is_better=False),
        "selected_brier_edge": _delta("value_bet_brier", higher_is_better=False),
        "selected_log_loss_edge": _delta("value_bet_log_loss", higher_is_better=False),
        "roi_edge": _delta("value_bet_roi", higher_is_better=True),
        "bet_count_delta": _delta("value_bets", higher_is_better=True),
    }


@st.cache_data(show_spinner=False)
def _build_generalization_trend_chart(trend_key: tuple, model_key: str, metric_key: str) -> dict:
    """Build a cross-run OOF vs validation trend chart for one metric."""
    rows: list[dict] = []
    for run_id, name, timestamp, mk, metric_name, oof_value, val_value in trend_key:
        if mk != model_key or metric_name != metric_key:
            continue
        if oof_value is None or val_value is None:
            continue
        direction = _metric_direction(metric_name)
        risk_gap = (float(oof_value) - float(val_value)) * direction
        rows.append({
            "Run": name or run_id,
            "Date": timestamp[:16].replace("T", " "),
            "OOF": float(oof_value),
            "Validation": float(val_value),
            "Risk Gap": float(risk_gap),
        })

    if len(rows) < 2:
        return {"fig": None, "df": pd.DataFrame(rows)}

    trend_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["Date"], y=trend_df["OOF"],
        mode="lines+markers", name="OOF",
        line=dict(color="#3b82f6"),
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Date"], y=trend_df["Validation"],
        mode="lines+markers", name="Validation",
        line=dict(color="#22c55e"),
    ))
    fig.add_trace(go.Bar(
        x=trend_df["Date"], y=trend_df["Risk Gap"],
        name="Risk Gap",
        marker_color="rgba(239,68,68,0.35)",
        yaxis="y2",
    ))
    fig.update_layout(
        height=380,
        title=f"{_model_display_name(model_key)} — {_metric_label(metric_key)} across runs",
        yaxis=dict(title=_metric_label(metric_key)),
        yaxis2=dict(title="Risk Gap", overlaying="y", side="right"),
        legend=dict(x=0.02, y=0.98),
    )
    return {"fig": fig, "df": trend_df}


@st.cache_resource(show_spinner=False)
def _build_concentration_charts(
    importances: tuple,
    feature_names: tuple,
) -> dict:
    """Build Lorenz curve figure and Gini stats."""
    fi_df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": list(importances),
    }).sort_values("importance", ascending=False)

    fi_sorted = fi_df.sort_values("importance", ascending=True)
    total_imp = fi_sorted["importance"].sum()
    cum_imp = fi_sorted["importance"].cumsum() / total_imp if total_imp > 0 else fi_sorted["importance"] * 0

    # Lorenz curve
    fig_lorenz = go.Figure()
    x_frac = np.linspace(0, 1, len(cum_imp))
    fig_lorenz.add_trace(go.Scatter(
        x=x_frac, y=cum_imp.values,
        mode="lines", name="Actual distribution",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
    ))
    fig_lorenz.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Perfect equality",
        line=dict(color="grey", dash="dash"),
    ))
    fig_lorenz.update_layout(
        title="Feature Importance Lorenz Curve",
        xaxis_title="Fraction of Features (sorted ascending)",
        yaxis_title="Cumulative Share of Importance",
        height=400, legend=dict(x=0.02, y=0.98),
    )

    # Gini coefficient
    vals = fi_sorted["importance"].values
    n = len(vals)
    gini = (
        (2 * np.sum((np.arange(1, n + 1)) * vals) / (n * np.sum(vals)))
        - (n + 1) / n
    ) if np.sum(vals) > 0 else 0.0

    return {
        "lorenz_fig": fig_lorenz,
        "n_feats": n,
        "top5": fi_df.nlargest(5, "importance")["importance"].sum() / total_imp if total_imp > 0 else 0,
        "top10": fi_df.nlargest(10, "importance")["importance"].sum() / total_imp if total_imp > 0 else 0,
        "top20": fi_df.nlargest(20, "importance")["importance"].sum() / total_imp if total_imp > 0 else 0,
        "gini": gini,
    }


@st.cache_data(show_spinner=False)
def _build_trend_chart(all_runs_key: tuple) -> dict:
    """Build the cross-run overfit trend chart.

    Each element of ``all_runs_key`` is a flat 7-tuple of primitives:
        (run_id, name, timestamp, val_ndcg1, val_top1, oof_ndcg1, oof_top1)
    Using primitives avoids @st.cache_data having to recursively hash nested
    dicts on every rerun (the old format with full metrics dicts was slow).
    """
    trend_rows: list[dict] = []
    for r in all_runs_key:
        run_id, name, timestamp, val_ndcg1, val_top1, oof_ndcg1, oof_top1 = r
        if val_ndcg1 == 0.0 or oof_ndcg1 == 0.0:
            continue
        trend_rows.append({
            "Run": name or run_id,
            "Date": timestamp[:16].replace("T", " "),
            "OOF NDCG@1": oof_ndcg1,
            "Validation NDCG@1": val_ndcg1,
            "Gap": round(oof_ndcg1 - val_ndcg1, 4),
            "OOF Top-1": oof_top1,
            "Validation Top-1": val_top1,
        })

    if len(trend_rows) < 2:
        return {"fig": None, "df": None, "n_rows": len(trend_rows)}

    trend_df = pd.DataFrame(trend_rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["Date"], y=trend_df["OOF NDCG@1"],
        mode="lines+markers", name="OOF NDCG@1",
        line=dict(color="#3b82f6"),
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Date"], y=trend_df["Validation NDCG@1"],
        mode="lines+markers", name="Validation NDCG@1",
        line=dict(color="#22c55e"),
    ))
    fig.add_trace(go.Bar(
        x=trend_df["Date"], y=trend_df["Gap"],
        name="Gap (OOF − Validation)",
        marker_color="rgba(239,68,68,0.5)",
        yaxis="y2",
    ))
    fig.update_layout(
        title="Win Classifier NDCG@1 — OOF vs Validation Across Runs",
        yaxis=dict(title="NDCG@1"),
        yaxis2=dict(
            title="Gap", overlaying="y", side="right",
            range=[0, 0.5],
        ),
        height=420,
        legend=dict(x=0.02, y=0.98),
    )

    return {"fig": fig, "df": trend_df, "n_rows": len(trend_rows)}


def _flatten_numeric_metrics(metrics: dict | None, prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dicts into a single numeric mapping."""
    out: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_numeric_metrics(value, full_key))
        elif isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value):
            out[full_key] = float(value)
    return out


def _build_shortcomings_run_frame(saved_runs: list[dict]) -> pd.DataFrame:
    """Assemble one row per saved run for metric-vs-ROI correlation."""
    rows: list[dict] = []
    for run in saved_runs:
        ta = run.get("test_analysis", {}) if isinstance(run.get("test_analysis", {}), dict) else {}
        stats = ta.get("stats", {}) if isinstance(ta.get("stats", {}), dict) else {}
        tp = stats.get("top_pick", {}) if isinstance(stats.get("top_pick", {}), dict) else {}
        vb = stats.get("value", {}) if isinstance(stats.get("value", {}), dict) else {}
        ew = stats.get("each_way", {}) if isinstance(stats.get("each_way", {}), dict) else {}
        row = {
            "run_id": run.get("run_id"),
            "name": run.get("name", run.get("run_id")),
            "timestamp": run.get("timestamp", ""),
            "top_pick_roi": tp.get("roi"),
            "value_roi": vb.get("roi"),
            "each_way_roi": ew.get("roi"),
            "combined_roi": None,
        }
        _staked = 0.0
        _pnl = 0.0
        for strat in (tp, vb, ew):
            _staked += float(strat.get("total_staked", 0) or 0)
            _pnl += float(strat.get("pnl", 0) or 0)
        if _staked > 0:
            row["combined_roi"] = _pnl / _staked * 100.0
        row.update(_flatten_numeric_metrics(run.get("metrics", {})))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_shortcomings_fold_frame(run_ids: tuple[str, ...]) -> pd.DataFrame:
    """Assemble one row per walk-forward fold across saved runs."""
    rows: list[pd.DataFrame] = []
    for run_id in run_ids:
        try:
            run_data = load_run(run_id)
        except Exception:
            continue
        wf = run_data.get("wf_summary_df")
        if not isinstance(wf, pd.DataFrame) or wf.empty:
            continue
        meta = run_data if isinstance(run_data, dict) else {}
        wf = wf.copy()
        wf["run_id"] = run_id
        wf["name"] = meta.get("name", run_id)
        wf["timestamp"] = meta.get("timestamp", "")
        if "top_pick_bets" in wf.columns and "top_pick_pnl" in wf.columns:
            wf["top_pick_roi"] = np.where(
                wf["top_pick_bets"] > 0,
                wf["top_pick_pnl"] / wf["top_pick_bets"] * 100.0,
                np.nan,
            )
        if "value_bets" in wf.columns and "value_pnl" in wf.columns:
            wf["value_roi"] = np.where(
                wf["value_bets"] > 0,
                wf["value_pnl"] / wf["value_bets"] * 100.0,
                np.nan,
            )
        if "ew_bets" in wf.columns and "ew_pnl" in wf.columns:
            wf["each_way_roi"] = np.where(
                wf["ew_bets"] > 0,
                wf["ew_pnl"] / (wf["ew_bets"] * 2.0) * 100.0,
                np.nan,
            )
        if {"top_pick_bets", "value_bets", "ew_bets", "top_pick_pnl", "value_pnl", "ew_pnl"}.issubset(wf.columns):
            _stakes = wf["top_pick_bets"] + wf["value_bets"] + (wf["ew_bets"] * 2.0)
            _pnl = wf["top_pick_pnl"] + wf["value_pnl"] + wf["ew_pnl"]
            wf["combined_roi"] = np.where(_stakes > 0, _pnl / _stakes * 100.0, np.nan)
        rows.append(wf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _add_shortcomings_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped buckets for race-condition slicing."""
    if df.empty:
        return df
    out = df.copy()
    if "num_runners" in out.columns:
        out["field_size_band"] = pd.cut(
            pd.to_numeric(out["num_runners"], errors="coerce"),
            bins=[0, 7, 11, 15, 99],
            labels=["1-7", "8-11", "12-15", "16+"],
            include_lowest=True,
        ).astype(str).replace("nan", "Unknown")
    if "distance_furlongs" in out.columns:
        out["distance_band"] = pd.cut(
            pd.to_numeric(out["distance_furlongs"], errors="coerce"),
            bins=[0, 6.5, 8.5, 11.5, 99],
            labels=["Sprint", "Mile", "Middle", "Staying"],
            include_lowest=True,
        ).astype(str).replace("nan", "Unknown")
    if "race_date" in out.columns:
        _dt = pd.to_datetime(out["race_date"], errors="coerce")
        out["month"] = _dt.dt.strftime("%b").fillna("Unknown")
    if "handicap" in out.columns:
        out["handicap_label"] = np.where(
            pd.to_numeric(out["handicap"], errors="coerce").fillna(0) > 0,
            "Handicap",
            "Non-Handicap",
        )
    return out


@st.cache_data(show_spinner=False)
def _prepare_shortcomings_run_frames(run_id: str) -> dict[str, pd.DataFrame]:
    """Prepare enriched per-runner and per-strategy frames for one saved run."""
    def _normalise_merge_key_pair(left: pd.DataFrame, right: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align merge-key dtypes across frames to avoid pandas int/object errors."""
        if key not in left.columns or key not in right.columns:
            return left, right

        if key == "race_id":
            left_num = pd.to_numeric(left[key], errors="coerce")
            right_num = pd.to_numeric(right[key], errors="coerce")
            if left_num.notna().all() and right_num.notna().all():
                left[key] = left_num.astype("int64")
                right[key] = right_num.astype("int64")
            else:
                left[key] = left[key].astype(str)
                right[key] = right[key].astype(str)
            return left, right

        left[key] = left[key].astype(str)
        right[key] = right[key].astype(str)
        return left, right

    run_data = _raw_load_run(run_id)
    preds = run_data.get("predictions_df")
    bets = run_data.get("bets_df")
    if not isinstance(preds, pd.DataFrame) or preds.empty:
        return {"base": pd.DataFrame(), "top_pick": pd.DataFrame(), "value": pd.DataFrame(), "each_way": pd.DataFrame()}

    featured_path = get_run_featured_path(run_id)
    if featured_path is None or not os.path.exists(featured_path):
        return {"base": pd.DataFrame(), "top_pick": pd.DataFrame(), "value": pd.DataFrame(), "each_way": pd.DataFrame()}

    featured = _cached_load_df(featured_path, os.path.getmtime(featured_path)).copy()
    merge_keys = [c for c in ["race_id", "horse_name"] if c in preds.columns and c in featured.columns]
    if len(merge_keys) < 2:
        return {"base": pd.DataFrame(), "top_pick": pd.DataFrame(), "value": pd.DataFrame(), "each_way": pd.DataFrame()}

    attr_cols = [
        "race_id", "horse_name", "race_date", "track", "race_type", "surface", "going",
        "num_runners", "distance_furlongs", "handicap", "won", "finish_position",
    ]
    attr_cols = [c for c in attr_cols if c in featured.columns]
    attr_df = featured[attr_cols].drop_duplicates(merge_keys).copy()
    base = preds.copy()
    for _mk in merge_keys:
        base, attr_df = _normalise_merge_key_pair(base, attr_df, _mk)
    base = base.merge(attr_df, on=merge_keys, how="left", suffixes=("", "_feat"))
    if "race_date" in base.columns:
        base["race_date"] = pd.to_datetime(base["race_date"], errors="coerce")
    if "won" not in base.columns and "finish_position" in base.columns:
        base["won"] = (pd.to_numeric(base["finish_position"], errors="coerce") == 1).astype(int)
    base = _add_shortcomings_bands(base)

    _sort_cols = [c for c in ["race_id", "model_prob", "horse_name"] if c in base.columns]
    top_pick = base.sort_values(_sort_cols, ascending=[True, False, True]).drop_duplicates("race_id").copy()
    top_pick["strategy"] = "top_pick"
    top_pick["stake"] = 1.0
    top_pick["pnl"] = np.where(
        pd.to_numeric(top_pick.get("won", 0), errors="coerce").fillna(0).astype(int) == 1,
        pd.to_numeric(top_pick.get("odds", 0), errors="coerce").fillna(0) - 1.0,
        -1.0,
    )
    top_pick["placed"] = (
        pd.to_numeric(top_pick.get("finish_position", 99), errors="coerce").fillna(99) <= 3
    ).astype(int)

    strategy_frames = {
        "base": base,
        "top_pick": top_pick,
        "value": pd.DataFrame(),
        "each_way": pd.DataFrame(),
    }

    if isinstance(bets, pd.DataFrame) and not bets.empty:
        bet_attr_cols = [
            c for c in [
                "race_id", "horse_name", "race_date", "track", "race_type", "surface", "going",
                "num_runners", "distance_furlongs", "handicap", "field_size_band",
                "distance_band", "month", "handicap_label",
            ] if c in base.columns
        ]
        bet_attrs = base[bet_attr_cols].drop_duplicates(["race_id", "horse_name"]).copy()
        bet_df = bets.copy()
        for _mk in ["race_id", "horse_name"]:
            bet_df, bet_attrs = _normalise_merge_key_pair(bet_df, bet_attrs, _mk)
        bet_df = bet_df.merge(
            bet_attrs,
            on=["race_id", "horse_name"],
            how="left",
            suffixes=("", "_base"),
        )
        if "race_date" in bet_df.columns:
            bet_df["race_date"] = pd.to_datetime(bet_df["race_date"], errors="coerce")
        bet_df = _add_shortcomings_bands(bet_df)
        strategy_frames["value"] = bet_df[bet_df["strategy"] == "value"].copy()
        strategy_frames["each_way"] = bet_df[bet_df["strategy"] == "each_way"].copy()

    return strategy_frames


def _build_shortcomings_correlation_table(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations of metrics vs target ROI."""
    if df.empty or target_col not in df.columns:
        return pd.DataFrame()
    target = pd.to_numeric(df[target_col], errors="coerce")
    rows: list[dict] = []
    exclude_terms = {
        "run_id", "name", "timestamp", "fold", "train_start", "train_end", "test_period",
        "top_pick_roi", "value_roi", "each_way_roi", "combined_roi",
        "top_pick_pnl", "value_pnl", "ew_pnl", "combined_pnl",
    }
    for col in df.columns:
        if col == target_col or col in exclude_terms:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mask = series.notna() & target.notna()
        if mask.sum() < 3:
            continue
        if series[mask].nunique() < 2 or target[mask].nunique() < 2:
            continue
        rows.append({
            "metric": col,
            "n": int(mask.sum()),
            "pearson": float(series[mask].corr(target[mask], method="pearson")),
            "spearman": float(series[mask].corr(target[mask], method="spearman")),
        })
    if not rows:
        return pd.DataFrame()
    corr_df = pd.DataFrame(rows)
    corr_df["abs_spearman"] = corr_df["spearman"].abs()
    return corr_df.sort_values(["abs_spearman", "pearson"], ascending=[False, False]).reset_index(drop=True)


def _summarise_shortcomings_slice(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate strategy outcomes by one race-condition dimension."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work[group_col] = work[group_col].fillna("Unknown").astype(str)
    if "stake" not in work.columns:
        work["stake"] = 1.0
    if "placed" not in work.columns:
        work["placed"] = work.get("won", 0)
    out = work.groupby(group_col, observed=True).agg(
        bets=("pnl", "count"),
        races=("race_id", "nunique"),
        winners=("won", "sum"),
        placed=("placed", "sum"),
        staked=("stake", "sum"),
        pnl=("pnl", "sum"),
        avg_odds=("odds", "mean"),
        avg_model_prob=("model_prob", "mean"),
    ).reset_index()
    out["strike_rate"] = np.where(out["bets"] > 0, out["winners"] / out["bets"] * 100.0, 0.0)
    out["place_rate"] = np.where(out["bets"] > 0, out["placed"] / out["bets"] * 100.0, 0.0)
    out["roi"] = np.where(out["staked"] > 0, out["pnl"] / out["staked"] * 100.0, 0.0)
    return out.sort_values(["roi", "bets"], ascending=[False, False]).reset_index(drop=True)


def _render_shap_explanation(
    explanations: dict[str, pd.DataFrame],
    predictions: pd.DataFrame,
    key_prefix: str = "shap",
    model_label: str | None = None,
):
    """
    Render a SHAP explainability section for a predicted race.

    Args:
        explanations: dict mapping horse_name → DataFrame with columns
                      ``feature``, ``shap_value``, ``feature_value``.
        predictions: the prediction DataFrame (must have ``horse_name``,
                     ``predicted_rank``).
        key_prefix: unique Streamlit widget key prefix.
    """
    st.markdown("---")
    st.markdown("### 🔍 Why This Ranking? (SHAP Explanations)")
    _label = model_label or "model score"
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows which features "
        f"pushed each horse's **{_label}** up (green) or down "
        "(red). Longer bars indicate larger influence."
    )

    sorted_preds = predictions.sort_values("predicted_rank")
    horse_names = sorted_preds["horse_name"].tolist()

    # Build rank lookup for display
    _rank_map = dict(zip(
        sorted_preds["horse_name"], sorted_preds["predicted_rank"],
    ))

    sel_horse = st.selectbox(
        "Select horse to explain",
        horse_names,
        format_func=lambda h: f"#{int(_rank_map[h])} — {h}",
        key=f"{key_prefix}_sel",
    )

    if sel_horse in explanations:
        expl = explanations[sel_horse].copy()

        # Waterfall-style horizontal bar chart
        expl = expl.sort_values("shap_value", ascending=True)
        colors = [
            "#22c55e" if v > 0 else "#ef4444"
            for v in expl["shap_value"]
        ]

        fig = go.Figure(go.Bar(
            x=expl["shap_value"],
            y=expl["feature"],
            orientation="h",
            marker_color=colors,
            text=expl["shap_value"].apply(lambda v: f"{v:+.3f}"),
            textposition="outside",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="grey")
        fig.update_layout(
            title=f"SHAP — {sel_horse}",
            xaxis_title=f"Impact on {_label.lower()}",
            yaxis_title="",
            height=max(300, len(expl) * 36),
            margin=dict(l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("📊 Raw SHAP values"):
            st.dataframe(
                expl.sort_values("shap_value", ascending=False)
                .style.format({
                    "shap_value": "{:+.4f}",
                    "feature_value": "{:.3f}",
                }),
                hide_index=True,
                width="stretch",
            )

    # ── All-horses comparison (summary bar) ──────────────────────
    with st.expander("📈 Compare top driver features across all runners"):
        rows = []
        for h_name, expl_df in explanations.items():
            top_feat = expl_df.iloc[0]
            rows.append({
                "horse": h_name,
                "top_feature": top_feat["feature"],
                "shap_value": top_feat["shap_value"],
            })
        comp_df = pd.DataFrame(rows)
        # sort by predicted rank
        rank_order = {
            h: r for h, r in zip(
                sorted_preds["horse_name"],
                sorted_preds["predicted_rank"],
            )
        }
        comp_df["rank"] = comp_df["horse"].map(rank_order)
        comp_df = comp_df.sort_values("rank")

        fig2 = px.bar(
            comp_df,
            x="horse", y="shap_value",
            color="top_feature",
            title="Strongest SHAP Feature per Horse",
            text="top_feature",
        )
        fig2.update_layout(height=380, xaxis_tickangle=-30)
        st.plotly_chart(fig2, width="stretch")


# ── Experiment log file ──────────────────────────────────────────────
EXPERIMENTS_FILE = os.path.join(config.DATA_DIR, "experiments.json")


@st.cache_data(ttl=5)
def _load_experiments() -> list[dict]:
    if os.path.exists(EXPERIMENTS_FILE):
        try:
            with open(EXPERIMENTS_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_experiments(experiments: list[dict]):
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(experiments, f, indent=2, default=str)
    _load_experiments.clear()


def _log_experiment(entry: dict):
    exps = _load_experiments()
    exps.append(entry)
    _save_experiments(exps)


@st.cache_data(show_spinner="Loading data …")
def _cached_load_df(path: str, _mtime: float) -> pd.DataFrame:
    """Read Parquet (or legacy CSV) — cached by path + modification time."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)
