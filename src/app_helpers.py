"""
App Helper Layer
================
Pure/diagnostic helpers shared by the Streamlit apps: cached run-store
wrappers, chart builders for the registry/insights views, pace diagnostic
panels, SHAP rendering, metric formatting and the experiment log.

These functions hold no page logic — the picks and training apps import
them.
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

import os
import sys
import logging

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
    build_config_snippet,
    create_autotune_session,
    delete_autotune_session,
    list_autotune_sessions,
    load_autotune_session,
    load_optuna_study,
    run_autotune_session,
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


# ── Cached wrappers for disk I/O ─────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def list_runs() -> list[dict]:
    return _raw_list_runs()


# Run snapshots are immutable once written (rename/delete clear these
# caches explicitly), so a long TTL avoids re-decompressing zstd
# snapshots every minute while browsing the Experiments pages.
@st.cache_data(ttl=3600, show_spinner=False)
def load_run(run_id: str) -> dict:
    return _raw_load_run(run_id)


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_resource(show_spinner="Loading data …")
def _cached_load_df(path: str, _mtime: float) -> pd.DataFrame:
    """Read Parquet (or legacy CSV) — cached by path + modification time.

    ``cache_resource`` (not ``cache_data``): featured datasets run to
    gigabytes, and cache_data deep-copies the value on EVERY call, which
    made each rerun pay seconds of copying and doubled memory. The
    returned frame is shared — treat it as read-only and ``.copy()``
    before mutating.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)
