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
from src.h2o_automl import h2o_is_available, run_h2o_automl, save_h2o_leader_model
from src.flaml_automl import flaml_is_available, run_flaml_automl
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
from src.run_store import save_run, list_runs as _raw_list_runs, load_run as _raw_load_run, load_run_meta as _raw_load_run_meta, delete_run as _raw_delete_run, rename_run, get_latest_run_id, restore_run_model, run_has_model, get_run_processed_path, get_run_featured_path
from src.utils import format_odds, kelly_criterion
from src.each_way import compute_ew_columns, ew_value_bets, get_ew_terms, kelly_ew, EachWayTerms

logger = logging.getLogger(__name__)


def _ordinal(n: int) -> str:
    """Return ordinal string for an integer, e.g. 1→'1st', 11→'11th'."""
    if n <= 0:
        return "?"
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][min(n % 10, 3)] if n % 10 < 4 else 'th'}"


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
    _tune = (
        f"auto{int(auto_trials)}"
        if tune_mode_label == "🔍 Auto (Optuna)"
        else ("saved" if tune_mode_label == "📦 Saved Autotune" else "manual")
    )
    return f"ens_{_src}_{_days}_{_odds}_{_tune}_{datetime.now():%m%d_%H%M}"


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
    "h2o_featured_data": None,
    "h2o_processed_data": None,
    "h2o_dataset_meta": None,
    "h2o_automl_result": None,
    "h2o_saved_model_path": None,
    "flaml_featured_data": None,
    "flaml_processed_data": None,
    "flaml_dataset_meta": None,
    "flaml_automl_result": None,
    "model_featured_data": None,
    "model_dataset_meta": None,
    "metrics": None,
    "bt_report": None,
    "test_analysis": None,
    "active_run_id": None,
    "value_config": {
        "staking_mode": "flat",
        "value_threshold": 0.05,
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


# ── Helpers ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def _cached_load_model():
    """Load model from disk — cached so joblib deserialization only happens once."""
    try:
        p = TripleEnsemblePredictor()
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


@st.cache_data(show_spinner="Loading data …")
def _cached_load_df(path: str, _mtime: float) -> pd.DataFrame:
    """Read Parquet (or legacy CSV) — cached by path + modification time."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


@st.cache_data(show_spinner="Rebuilding featured dataset for active run …")
def _cached_build_featured_from_processed(path: str, _mtime: float) -> pd.DataFrame:
    processed = _cached_load_df(path, _mtime)
    return engineer_features(processed.copy(), save=False)


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
    st.session_state.autotune_featured_data = featured_df
    st.session_state.autotune_processed_data = processed_df
    st.session_state.autotune_dataset_meta = dataset_meta or None


def _set_h2o_dataset(
    featured_df: pd.DataFrame | None,
    *,
    processed_df: pd.DataFrame | None = None,
    dataset_meta: dict[str, object] | None = None,
) -> None:
    st.session_state.h2o_featured_data = featured_df
    st.session_state.h2o_processed_data = processed_df
    st.session_state.h2o_dataset_meta = dataset_meta or None


def _set_flaml_dataset(
    featured_df: pd.DataFrame | None,
    *,
    processed_df: pd.DataFrame | None = None,
    dataset_meta: dict[str, object] | None = None,
) -> None:
    st.session_state.flaml_featured_data = featured_df
    st.session_state.flaml_processed_data = processed_df
    st.session_state.flaml_dataset_meta = dataset_meta or None


def _set_model_dataset(
    featured_df: pd.DataFrame | None,
    *,
    dataset_meta: dict[str, object] | None = None,
) -> None:
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
            _hist = _cached_load_df(_snap, _mtime)
            _last_hist = pd.to_datetime(_hist["race_date"], errors="coerce").max() if _hist is not None and not _hist.empty else None
            return _hist, {
                "path": _snap,
                "mtime": _mtime,
                "last_hist_date": _last_hist.date() if pd.notna(_last_hist) else None,
            }

    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.csv")
    path = pq_path if os.path.exists(pq_path) else csv_path
    if not os.path.exists(path):
        return None, {"path": None, "mtime": None, "last_hist_date": None}
    mtime = os.path.getmtime(path)
    hist = _cached_load_df(path, mtime)
    last_hist = pd.to_datetime(hist["race_date"], errors="coerce").max() if hist is not None and not hist.empty else None
    return hist, {
        "path": path,
        "mtime": mtime,
        "last_hist_date": last_hist.date() if pd.notna(last_hist) else None,
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


# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🏇 Horse Race Predictor")
st.sidebar.caption("v5.0 — 3-Model Pipeline")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🎓 Train & Tune",
        "🧭 Autotune",
        "🤖 H2O AutoML",
        "🤖 FLAML",
        "🧪 Experiments",
        "🔮 Predict",
        "💰 Today's Picks",
        "🔁 Backtest",
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
        format_func=lambda rid: _run_labels[rid],
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
        _label = "2-Model Pipeline" if _mtype == "TripleEnsemblePredictor" else _mtype
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
        _label = "2-Model Pipeline" if _mtype == "TripleEnsemblePredictor" else _mtype
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

    # ── Ensemble overview ────────────────────────────────────────────
    with st.expander("ℹ️ About the Models", expanded=False):
        st.markdown(
            "The system uses **2 task-specific classifiers**, each optimised for its betting strategy:\n\n"
            "| Model | Objective | Task |\n"
            "|-------|-----------|------|\n"
            "| **Win Classifier** | Log-loss / focal | Value Bets — calibrated P(win) for edge detection |\n"
            "| **Place Classifier** | Log-loss / focal | Each-Way — calibrated P(place) for EW value |\n\n"
            "Win classifier probabilities are calibrated via Platt scaling. "
            "Place classifier uses separate Platt calibration on P(place)."
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
        "Include odds / market features",
        value=False,
        help=(
            "When **enabled**, the model uses Starting Price (SP) odds "
            "to derive features like implied probability, favourite "
            "status, and market overround.  \n\n"
            "**⚠️ SP is only known at race-off** — backtest results "
            "will be optimistic vs. live betting where only early "
            "prices are available.  \n\n"
            "**Disable** this to train a purely form-based model "
            "that doesn't rely on market information at all, giving "
            "a more realistic view of predictive power."
        ),
    )

    # Columns derived from the raw 'odds' column.  When the user
    # disables odds these are dropped before training so the model
    # never sees any market information.
    _ODDS_DERIVED_COLS = [
        "implied_prob", "norm_implied_prob", "odds_rank",
        "is_favourite", "log_odds", "odds_vs_field", "overround",
        "odds_cv", "implied_prob_vs_base",
        # Interaction features that incorporate odds data
        "jockey_elo_x_fav",
        "mkt_x_win_rate", "logodds_x_elo",
        "odds_field_x_dropped", "mkt_x_speed",
        "odds_field_x_jock_elo",
        "odds_vs_elo_rank",
        # Historical odds signals (previous race favourite status)
        "beaten_fav_last",
    ]

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

        _cache_paths = _dataset_cache_paths(data_source, int(days_back) if data_source != "sample" else None)
        _cache_key = _cache_paths.get("cache_key")
        _cache_path = _cache_paths.get("featured")
        _processed_cache_path = _cache_paths.get("processed")
        _cache_hit = False

        if data_source != "sample" and _cache_path and os.path.exists(_cache_path):
            _prep_progress.progress(10, text="📦 Loading cached dataset …")
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
            _prep_progress.progress(10, text="📊 Collecting historical data …")
            with st.spinner("Collecting data …"):
                if data_source in ("database", "scrape"):
                    raw_data = collect_data(source=data_source, days_back=days_back)
                else:
                    raw_data = collect_data(source="sample", num_races=num_races)
            st.success(f"✅ Collected {len(raw_data):,} race entries")
            raw_data["_is_future"] = 0

            # ── Step 1b: Incremental RTV backfill for this window ───
            if data_source != "sample":
                _prep_progress.progress(15, text="🏇 Updating RTV cache for missing races …")
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

            # ── Step 2: Fetch racecards for next 7 days ──────────────
            _future_card_sigs: dict[str, str] = {}  # date_str → signature
            if data_source != "sample":
                _prep_progress.progress(20, text="🗓️ Fetching racecards for next 7 days …")
                _card_frames: list[pd.DataFrame] = []
                _today = datetime.now().date()
                for _offset in range(1, 8):
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
                if _card_frames:
                    _future_raw = pd.concat(_card_frames, ignore_index=True)
                    st.success(f"✅ Fetched {len(_future_raw):,} future racecard entries")
                    raw_data = pd.concat([raw_data, _future_raw], ignore_index=True, sort=False)

            # ── Step 3: Process combined dataset ─────────────────────
            _prep_progress.progress(35, text="🔧 Processing combined dataset …")
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
            _prep_progress.progress(55, text="⚙️ Engineering features …")
            with st.spinner("Feature engineering …"):
                combined_featured = engineer_features(combined_processed, save=False)

            # ── Step 5: Split historical vs future ───────────────────
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
            if not future_featured.empty:
                _prep_progress.progress(90, text="🔮 Saving lookahead cache …")
                clear_lookahead_cache()
                _future_dates = pd.to_datetime(
                    future_featured["race_date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d").unique()
                for _fd in sorted(_future_dates):
                    _fd_mask = (
                        pd.to_datetime(future_featured["race_date"], errors="coerce")
                        .dt.strftime("%Y-%m-%d") == _fd
                    )
                    _fd_rows = future_featured[_fd_mask]
                    _fd_sig = _future_card_sigs.get(_fd, cards_signature(_fd_rows))
                    save_lookahead_cache(_fd, _fd_rows, _fd_sig)
                st.caption(f"🔮 Lookahead cached: {', '.join(sorted(_future_dates))}")
            elif data_source != "sample":
                st.caption("🔮 No upcoming racecards found for lookahead.")

        _prep_progress.progress(100, text="✅ Data ready!")
        st.rerun()

    st.markdown("---")

    model_type = "triple_ensemble"

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

    st.markdown("---")

    # ── Hyperparameter Tuning ────────────────────────────────────────
    st.subheader("3️⃣ Hyperparameters")
    st.caption(
        "Set hyperparameters per sub-model manually, or let Optuna "
        "find the best settings for each enabled model automatically."
    )

    tune_mode = st.radio(
        "Tuning mode",
        ["⚙️ Manual", "📦 Saved Autotune", "🔍 Auto (Optuna)"],
        horizontal=True,
        help=(
            "**Manual** — choose hyperparameters per sub-model with sliders.\n\n"
            "**Saved Autotune** — reuse persisted Optuna results from the dedicated autotune page.\n\n"
            "**Auto** — Optuna searches automatically for each enabled "
            "sub-model, minimising RPS on a validation fold."
        ),
    )

    custom_hp: dict[str, dict] | None = None
    auto_n_trials: int = 30
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
    elif tune_mode == "🔍 Auto (Optuna)":
        ac1, ac2 = st.columns([2, 3])
        with ac1:
            auto_n_trials = st.slider(
                "Optuna trials per model", 5, 200, 30, 5,
                key="auto_trials",
                help=(
                    "More trials = better search but slower.  "
                    "5 is enough for a smoke test; 30 is a good default; 100+ for thorough search.  "
                    "Each enabled model gets its own Optuna study."
                ),
            )
        with ac2:
            st.caption(
                "Optuna will create a temporal train/validation split "
                "and search hyperparameter combinations per sub-model, "
                "optimising each model's own metric (LogLoss).  "
                "The best params for each model are then used to "
                "retrain on the full training set."
            )
        with st.expander("Search space", expanded=False):
            for _mk, _label in _TASK_MODELS.items():
                _fw = _frameworks.get(_mk, "lgbm")
                _specs = get_autotune_search_space(_mk, _fw, include_recency=True)
                _rows = []
                for _spec in _specs:
                    if _spec.get("kind") == "fixed":
                        _rows.append({
                            "Parameter": _spec["name"],
                            "Distribution": "fixed",
                            "Range / Value": str(_spec.get("value")),
                        })
                    else:
                        _range = f"{_spec.get('low')} → {_spec.get('high')}"
                        if _spec.get("step") is not None:
                            _range += f" (step {_spec.get('step')})"
                        _dist = "log-uniform" if _spec.get("log") else "uniform"
                        if _spec.get("kind") == "int":
                            _dist = f"int {_dist}"
                        _rows.append({
                            "Parameter": _spec["name"],
                            "Distribution": _dist,
                            "Range / Value": _range,
                        })
                st.markdown(f"**{_label}** via `{_fw}`")
                st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
    else:
        _autotune_sessions = list_autotune_sessions()
        if not _autotune_sessions:
            st.warning(
                "No saved autotune sessions found yet. Use the 🧭 Autotune page first, or switch to Manual/Auto."
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
        auto_trials=auto_n_trials,
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

        # Drop odds-derived features if the user opted out
        if not include_odds:
            _to_drop = [c for c in _ODDS_DERIVED_COLS if c in featured.columns]
            featured = featured.drop(columns=_to_drop)
            st.info(
                f"🚫 Odds features disabled — dropped {len(_to_drop)} "
                f"market columns (form-only model)"
            )

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

        if not skip_wf:
            # ── Walk-forward backtest ────────────────────────────────
            progress.progress(5, text="🔁 Starting walk-forward validation …")
            _vc = _value_config or {}

            def _wf_progress_cb(msg: str, pct: float) -> None:
                """Map WF progress (0-1) into the 5-55% range of overall progress."""
                overall = 0.05 + pct * 0.50
                progress.progress(min(overall, 0.55), text=f"🔁 {msg}")

            wf_report = walk_forward_validation(
                featured,
                model_type="triple_ensemble",
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
            progress.progress(60, text=f"✅ {n_folds} folds complete — retraining final model …")
        else:
            st.session_state.wf_report = None
            progress.progress(5, text="⏭️ Skipping walk-forward — training final model …")

        # ── Retrain final model on ALL data for live predictions ─────
        _auto_tune_cfg: dict | None = None
        if tune_mode == "🔍 Auto (Optuna)":
            _auto_tune_cfg = {"n_trials": auto_n_trials}
            st.info(
                f"🔍 Optuna will auto-tune each enabled sub-model "
                f"({auto_n_trials} trials per model) during final retraining."
            )
        elif tune_mode == "📦 Saved Autotune" and not custom_hp:
            st.error("The selected autotune session does not contain usable best parameters.")
            st.stop()

        def _training_cb(msg: str, pct: float) -> None:
            if skip_wf:
                overall = 0.05 + pct * 0.90
            else:
                overall = 0.60 + pct * 0.35
            progress.progress(min(overall, 0.95), text=f"🤖 {msg}")

        predictor = TripleEnsemblePredictor(frameworks=_frameworks)
        metrics = predictor.train(
            featured, params=custom_hp, progress_callback=_training_cb,
            value_config=_value_config,
            auto_tune=_auto_tune_cfg,
        )

        elapsed = time.time() - t0
        st.session_state.predictor = predictor
        st.session_state.metrics = metrics

        progress.progress(100, text="✅ Complete!")

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
                with eq1:
                    fig_pnl = px.line(
                        curves,
                        x="race_date", y="cum_pnl",
                        color="strategy",
                        markers=True,
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
                        markers=True,
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
            "tuning_mode": "auto" if tune_mode == "🔍 Auto (Optuna)" else ("saved" if tune_mode == "📦 Saved Autotune" else "manual"),
            "auto_tune_trials": int(auto_n_trials) if tune_mode == "🔍 Auto (Optuna)" else None,
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
            "tuning_mode": "auto" if tune_mode == "🔍 Auto (Optuna)" else ("saved" if tune_mode == "📦 Saved Autotune" else "manual"),
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
        if _auto_tune_cfg is not None:
            exp_entry["auto_tune"] = {
                "n_trials": _auto_tune_cfg.get("n_trials"),
            }
        elif _saved_autotune_session is not None:
            exp_entry["saved_autotune"] = {
                "session_id": _saved_autotune_session.get("session_id"),
                "name": _saved_autotune_session.get("name"),
            }
        _log_experiment(exp_entry)

        # ── Persist full run snapshot ────────────────────────────────
        _weights = getattr(predictor, "weights", None)
        _ens_weights = (
            {k: float(v) for k, v in _weights.items()}
            if isinstance(_weights, dict) else {}
        )

        _auto_info = None
        if _auto_tune_cfg is not None:
            _auto_info = {
                "n_trials": _auto_tune_cfg.get("n_trials"),
            }
        elif _saved_autotune_session is not None:
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
            ensemble_weights=_ens_weights,
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
        _at_trials = st.slider("Trials per model", 5, 200, 40, 5, key="_autotune_trials")
    with _setup_c2:
        _at_folds = st.slider("Purged walk-forward folds", 1, 5, 3, 1, key="_autotune_folds")
    if _at_folds == 1:
        st.caption("Single split: the training window is used as-is with a purged validation window at the end. Faster but less robust.")
    else:
        st.caption("Optuna scores each trial by averaging the objective over purged walk-forward folds built from the outer training split.")

    _at_frameworks: dict[str, str] = {}
    _fw_cols = st.columns(2)
    _fw_defaults = dict(getattr(config, "SUB_MODEL_FRAMEWORKS", {}))
    for _idx, (_mk, _label) in enumerate(_task_models.items()):
        with _fw_cols[_idx]:
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
            if _at_folds == 1:
                st.caption(f"Objective for this model: LogLoss on a single purged validation split.")
            else:
                st.caption(f"Objective for this model: mean LogLoss across {_at_folds} purged walk-forward folds.")

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
                f"{payload.get('message', 'Preparing autotune splits')} · target folds {payload.get('target_folds', _at_folds)}"
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
        _new_session = create_autotune_session(
            name=_at_name.strip() or f"autotune_{datetime.now():%Y%m%d_%H%M%S}",
            dataset_meta=_at_meta,
            frameworks=_at_frameworks,
            models=_at_models,
            n_trials=int(_at_trials),
            n_folds=int(_at_folds),
        )
        _manifest = run_autotune_session(
            session_id=_new_session["session_id"],
            featured_df=_at_featured.copy(),
            frameworks=_at_frameworks,
            models=_at_models,
            n_trials=int(_at_trials),
            n_folds=int(_at_folds),
            progress_callback=_render_autotune_progress,
        )
        _progress_bar.progress(1.0, text="Autotune complete")
        _progress_box.success(f"Saved autotune session {_manifest.get('name')} ({_manifest.get('session_id')}).")

    if _resume_clicked and _resume_target:
        if not isinstance(_at_featured, pd.DataFrame):
            st.error("Load the dataset that matches the session you want to resume.")
            st.stop()
        _resume_manifest = load_autotune_session(_resume_target)
        if _resume_manifest is None:
            st.error("Selected autotune session could not be loaded.")
            st.stop()
        _manifest = run_autotune_session(
            session_id=_resume_target,
            featured_df=_at_featured.copy(),
            frameworks=dict(_resume_manifest.get("frameworks") or {}),
            models=list(_resume_manifest.get("models") or []),
            n_trials=int(_at_trials),
            n_folds=int(_resume_manifest.get("target_folds") or _at_folds),
            progress_callback=_render_autotune_progress,
        )
        _progress_bar.progress(1.0, text="Autotune resume complete")
        _progress_box.success(f"Resumed autotune session {_manifest.get('name')} ({_manifest.get('session_id')}).")

    st.markdown("---")
    st.subheader("3️⃣ Saved Sessions")
    _at_sessions = list_autotune_sessions()
    if not _at_sessions:
        st.info("No autotune sessions saved yet.")
    else:
        _session_labels = []
        _session_lookup = {}
        for _session in _at_sessions:
            _meta = _session.get("dataset_meta") or {}
            _label = (
                f"{_session.get('name', _session.get('session_id'))} · {_session.get('status', 'unknown')} · "
                f"{_meta.get('data_source', '?')} {_meta.get('actual_days') or '—'}d · {_session.get('updated_at', '')[:16]}"
            )
            _session_labels.append(_label)
            _session_lookup[_label] = _session
        _selected_session = _session_lookup[st.selectbox("Saved sessions", _session_labels, key="_autotune_session_picker")]
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
                    columns=[c for c in _study.trials_dataframe().columns if c.startswith("user_attrs_") or c.startswith("system_attrs_")],
                    errors="ignore",
                )

                _vt1, _vt2 = st.tabs(["Charts", "Trials"])
                with _vt1:
                    st.plotly_chart(optuna.visualization.plot_optimization_history(_study), width="stretch")
                    _c1, _c2 = st.columns(2)
                    with _c1:
                        st.plotly_chart(optuna.visualization.plot_param_importances(_study), width="stretch")
                    with _c2:
                        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(_study), width="stretch")
                    if len(_trial_df) >= 2:
                        st.plotly_chart(optuna.visualization.plot_slice(_study), width="stretch")
                with _vt2:
                    st.dataframe(_trial_df, width="stretch", hide_index=True)
            except Exception as exc:
                st.warning(f"Could not render Optuna visualisations for this study: {exc}")


# =====================================================================
#  H2O AUTOML
# =====================================================================
elif page == "🤖 H2O AutoML":
    st.title("🤖 H2O AutoML")
    st.caption(
        "Run fast multi-model experiments with H2O AutoML using the same leak-safe dataset split as your main training pipeline."
    )

    _h2o_ok, _h2o_err = h2o_is_available()
    if not _h2o_ok:
        st.error("H2O is not installed in this environment.")
        st.code("pip install h2o", language="bash")
        st.caption(f"Import error: {_h2o_err}")
        st.stop()

    st.markdown("---")
    st.subheader("1️⃣ Dataset")
    _h2o_dataset_mode = st.radio(
        "Dataset source",
        ["Use current Train & Tune dataset", "Load latest featured dataset", "Build fresh dataset"],
        horizontal=True,
        key="_h2o_dataset_mode",
    )

    if _h2o_dataset_mode == "Use current Train & Tune dataset":
        if st.session_state.featured_data is not None:
            if st.button("Use loaded training dataset", key="_h2o_use_train_ds"):
                _set_h2o_dataset(
                    st.session_state.featured_data.copy(),
                    processed_df=st.session_state.get("train_processed_data"),
                    dataset_meta=dict(st.session_state.get("train_dataset_meta") or {}),
                )
                st.session_state.h2o_automl_result = None
                st.success("Current training dataset copied into the H2O workspace.")
        else:
            st.warning("No training dataset is loaded yet. Prepare one on Train & Tune, load latest featured data, or build fresh here.")
    elif _h2o_dataset_mode == "Load latest featured dataset":
        _global_featured = _global_featured_dataset_path()
        if _global_featured and os.path.exists(_global_featured):
            if st.button("Load latest featured dataset from disk", key="_h2o_load_global"):
                _featured = _cached_load_df(_global_featured, os.path.getmtime(_global_featured))
                _set_h2o_dataset(
                    _featured,
                    dataset_meta=_dataset_meta_from_frame(
                        _featured,
                        data_source="disk",
                        requested_days=None,
                        featured_path=_global_featured,
                        origin="h2o_global_featured",
                    ),
                )
                st.session_state.h2o_automl_result = None
                st.success("Loaded the latest featured dataset from disk.")
        else:
            st.warning("No global featured dataset file found yet.")
    else:
        _hd1, _hd2 = st.columns([2, 2])
        with _hd1:
            _h2o_source = st.selectbox("Data source", ["database", "scrape", "sample"], key="_h2o_source")
        with _hd2:
            if _h2o_source == "sample":
                _h2o_num_races = st.slider("Sample races", 500, 5000, 1500, 100, key="_h2o_num_races")
                _h2o_days_back = None
            else:
                _h2o_num_races = 1500
                _h2o_days_back = st.slider("Days of history", 1, 2000, 90, 7, key="_h2o_days_back")
        if st.button("📦 Prepare H2O dataset", type="secondary", key="_h2o_prepare"):
            _prep = st.progress(0, text="Preparing H2O dataset …")
            _prep.progress(10, text="Collecting and processing data …")
            _featured, _processed, _meta = build_autotune_dataset(
                data_source=_h2o_source,
                days_back=_h2o_days_back,
                num_races=_h2o_num_races,
            )
            _prep.progress(100, text="Dataset ready")
            _set_h2o_dataset(_featured, processed_df=_processed, dataset_meta=_meta)
            st.session_state.h2o_automl_result = None
            st.success(f"Prepared {_meta.get('rows', 0):,} rows for H2O AutoML.")

    _h2o_featured = st.session_state.get("h2o_featured_data")
    _h2o_meta = st.session_state.get("h2o_dataset_meta") or {}
    if isinstance(_h2o_featured, pd.DataFrame):
        st.success(
            f"H2O dataset ready: {_h2o_meta.get('rows', len(_h2o_featured)):,} rows · "
            f"{_h2o_meta.get('months') or '—'} months · {_h2o_meta.get('date_start') or '?'} → {_h2o_meta.get('date_end') or '?'}"
        )
    else:
        st.info("Load or prepare a dataset to run H2O AutoML.")

    st.markdown("---")
    st.subheader("2️⃣ AutoML Setup")
    _hs1, _hs2, _hs3 = st.columns(3)
    with _hs1:
        _h2o_target = st.selectbox(
            "Target",
            options=["won", "placed"],
            format_func=lambda x: "Winner (won)" if x == "won" else "Placed (placed)",
            key="_h2o_target",
        )
    with _hs2:
        _h2o_max_models = st.slider("Max models", 5, 100, 25, 5, key="_h2o_max_models")
    with _hs3:
        _h2o_max_runtime = st.slider(
            "Max runtime (seconds, 0 = unlimited)",
            0,
            7200,
            900,
            60,
            key="_h2o_max_runtime",
        )

    _hs4, _hs5 = st.columns(2)
    with _hs4:
        _h2o_sort_metric = st.selectbox(
            "Sort metric",
            options=["AUC", "logloss", "mean_per_class_error", "AUCPR"],
            key="_h2o_sort_metric",
        )
    with _hs5:
        _h2o_balance = st.checkbox("Balance classes", value=True, key="_h2o_balance")

    _h2o_exclude = st.multiselect(
        "Exclude algorithms (optional)",
        options=["DeepLearning", "DRF", "GBM", "GLM", "XGBoost", "StackedEnsemble", "XRT"],
        default=[],
        key="_h2o_exclude_algos",
    )

    st.markdown("#### Stacking & Blending")
    _sb1, _sb2, _sb3 = st.columns(3)
    with _sb1:
        _h2o_enable_stacking = st.checkbox(
            "Enable stacked ensembles",
            value=True,
            key="_h2o_enable_stacking",
            help="Allow AutoML to build StackedEnsemble models (BestOfFamily / AllModels).",
        )
    with _sb2:
        _h2o_use_blending = st.checkbox(
            "Use blending holdout",
            value=False,
            key="_h2o_use_blending",
            help="Use a dedicated holdout inside training for ensemble metalearner fitting.",
        )
    with _sb3:
        _h2o_nfolds = st.slider(
            "CV folds (nfolds)",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            key="_h2o_nfolds",
            help="Cross-validation folds for base learners. When blending is enabled, this is forced to 0.",
        )

    _h2o_blend_frac = 0.15
    if _h2o_use_blending:
        _h2o_blend_frac = st.slider(
            "Blending holdout fraction",
            min_value=0.05,
            max_value=0.40,
            value=0.15,
            step=0.05,
            key="_h2o_blend_frac",
            help="Fraction of the training split reserved as blending frame.",
        )
        if not _h2o_enable_stacking:
            st.info("Blending is mainly useful for stacked ensembles. Consider enabling stacked ensembles.")

    if st.button("🚀 Run H2O AutoML", type="primary", width="stretch", key="_h2o_run"):
        if not isinstance(_h2o_featured, pd.DataFrame):
            st.error("Prepare a dataset first.")
            st.stop()

        with st.spinner("Running H2O AutoML on the current pipeline split …"):
            try:
                _h2o_result = run_h2o_automl(
                    _h2o_featured.copy(),
                    target=_h2o_target,
                    max_models=int(_h2o_max_models),
                    max_runtime_secs=(None if int(_h2o_max_runtime) == 0 else int(_h2o_max_runtime)),
                    sort_metric=_h2o_sort_metric,
                    seed=getattr(config, "RANDOM_SEED", 42),
                    balance_classes=bool(_h2o_balance),
                    exclude_algos=list(_h2o_exclude),
                    nfolds=int(_h2o_nfolds),
                    use_blending=bool(_h2o_use_blending),
                    blending_fraction=float(_h2o_blend_frac),
                    include_stacked_ensembles=bool(_h2o_enable_stacking),
                )
                st.session_state.h2o_automl_result = _h2o_result
                st.success("H2O AutoML run complete.")
            except Exception as _h2o_exc:
                st.error(f"H2O AutoML failed: {_h2o_exc}")

    _h2o_result = st.session_state.get("h2o_automl_result")
    if isinstance(_h2o_result, dict):
        st.markdown("---")
        st.subheader("3️⃣ Results")
        _m = _h2o_result.get("metrics") or {}

        def _fmt_metric(v, pct: bool = False) -> str:
            if v is None:
                return "—"
            try:
                fv = float(v)
            except Exception:
                return "—"
            return f"{fv:.1%}" if pct else f"{fv:.4f}"

        _r1, _r2, _r3, _r4 = st.columns(4)
        _r1.metric("Leader Model", str(_h2o_result.get("leader_model_id", "—")))
        _r2.metric("Brier", _fmt_metric(_m.get("brier")))
        _r3.metric("LogLoss", _fmt_metric(_m.get("log_loss")))
        _r4.metric("ROC AUC", _fmt_metric(_m.get("roc_auc")))

        _r5, _r6, _r7, _r8 = st.columns(4)
        _r5.metric("Accuracy", _fmt_metric(_m.get("accuracy"), pct=True))
        _r6.metric("Top-1 Accuracy", _fmt_metric(_m.get("top1_accuracy"), pct=True))
        _r7.metric("NDCG@1", _fmt_metric(_m.get("ndcg_at_1")))
        _r8.metric("Precision", _fmt_metric(_m.get("precision"), pct=True))

        st.caption(
            f"Train rows: {_h2o_result.get('n_train_rows', 0):,} · "
            f"Test rows: {_h2o_result.get('n_test_rows', 0):,} · "
            f"Features: {_h2o_result.get('n_features', 0):,}"
        )
        _h2o_settings = _h2o_result.get("settings") or {}
        st.caption(
            f"Stacking: {'on' if _h2o_settings.get('include_stacked_ensembles', True) else 'off'} · "
            f"Blending: {'on' if _h2o_settings.get('use_blending') else 'off'} · "
            f"nfolds: {_h2o_settings.get('nfolds', '—')} · "
            f"Effective train rows: {_h2o_result.get('n_train_rows_effective', _h2o_result.get('n_train_rows', 0)):,}"
            + (
                f" · Blending rows: {_h2o_result.get('n_blending_rows', 0):,}"
                if int(_h2o_result.get('n_blending_rows', 0) or 0) > 0
                else ""
            )
        )

        _leader_id = str(_h2o_result.get("leader_model_id", ""))
        _leader_is_stack = _leader_id.lower().startswith("stackedensemble")
        _lt_col1, _lt_col2 = st.columns([1, 3])
        with _lt_col1:
            _lt_label = "Stacked Ensemble" if _leader_is_stack else "Base Model"
            st.metric("Leader Type", _lt_label)
        with _lt_col2:
            if _h2o_settings.get("include_stacked_ensembles", True):
                if _leader_is_stack:
                    st.success("Leader is an ensemble model (StackedEnsemble).")
                else:
                    st.info(
                        "Stacking was enabled, but the best model for this run is still an individual base model. "
                        "This can happen when ensembling does not improve the selected metric."
                    )

        _leaderboard = _h2o_result.get("leaderboard")
        if isinstance(_leaderboard, pd.DataFrame) and not _leaderboard.empty:
            _lb = _leaderboard.copy()
            _lb["model_family"] = _lb["model_id"].astype(str).str.split("_").str[0]

            _stack_rows = _lb[_lb["model_family"].str.lower() == "stackedensemble"]
            _base_rows = _lb[_lb["model_family"].str.lower() != "stackedensemble"]

            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Total Models", int(len(_lb)))
            _c2.metric("Stacked Ensembles", int(len(_stack_rows)))
            _c3.metric("Base Models", int(len(_base_rows)))

            if not _stack_rows.empty and not _base_rows.empty:
                _best_stack = _stack_rows.iloc[0]
                _best_base = _base_rows.iloc[0]
                _cmp1, _cmp2 = st.columns(2)
                with _cmp1:
                    st.caption("Best base model")
                    st.write(str(_best_base.get("model_id", "—")))
                with _cmp2:
                    st.caption("Best stacked ensemble")
                    st.write(str(_best_stack.get("model_id", "—")))

            _fam_counts = (
                _lb["model_family"]
                .value_counts()
                .rename_axis("Model Family")
                .reset_index(name="Count")
            )
            with st.expander("Model family breakdown", expanded=False):
                st.dataframe(_fam_counts, width="stretch", hide_index=True)

            _preferred_cols = [
                "model_id",
                "model_family",
                "auc",
                "logloss",
                "mean_per_class_error",
                "rmse",
                "mse",
            ]
            _show_cols = [c for c in _preferred_cols if c in _lb.columns]
            st.markdown("#### Leaderboard")
            st.dataframe(_lb[_show_cols] if _show_cols else _lb, width="stretch", hide_index=True)

        _save_col1, _save_col2 = st.columns([1, 3])
        with _save_col1:
            if st.button("💾 Save Leader", key="_h2o_save_leader"):
                try:
                    _save_dir = os.path.join(config.MODELS_DIR, "h2o")
                    os.makedirs(_save_dir, exist_ok=True)
                    _saved_path = save_h2o_leader_model(_h2o_result.get("leader"), _save_dir)
                    st.session_state.h2o_saved_model_path = _saved_path
                    st.success(f"Saved leader model to {_saved_path}")
                except Exception as _save_exc:
                    st.error(f"Could not save leader model: {_save_exc}")
        with _save_col2:
            if st.session_state.get("h2o_saved_model_path"):
                st.caption(f"Latest saved leader: {st.session_state.get('h2o_saved_model_path')}")


# =====================================================================
#  FLAML
# =====================================================================
elif page == "🤖 FLAML":
    st.title("🤖 FLAML AutoML")
    st.caption(
        "Run FLAML for classification or ranking on the same leak-safe dataset split used by your core pipeline."
    )

    _flaml_ok, _flaml_err = flaml_is_available()
    if not _flaml_ok:
        st.error("FLAML is not installed in this environment.")
        st.code("pip install flaml[automl]", language="bash")
        st.caption(f"Import error: {_flaml_err}")
        st.stop()

    st.markdown("---")
    st.subheader("1️⃣ Dataset")
    _fl_dataset_mode = st.radio(
        "Dataset source",
        ["Use current Train & Tune dataset", "Load latest featured dataset", "Build fresh dataset"],
        horizontal=True,
        key="_fl_dataset_mode",
    )

    if _fl_dataset_mode == "Use current Train & Tune dataset":
        if st.session_state.featured_data is not None:
            if st.button("Use loaded training dataset", key="_fl_use_train_ds"):
                _set_flaml_dataset(
                    st.session_state.featured_data.copy(),
                    processed_df=st.session_state.get("train_processed_data"),
                    dataset_meta=dict(st.session_state.get("train_dataset_meta") or {}),
                )
                st.session_state.flaml_automl_result = None
                st.success("Current training dataset copied into the FLAML workspace.")
        else:
            st.warning("No training dataset is loaded yet. Prepare one on Train & Tune, load latest featured data, or build fresh here.")
    elif _fl_dataset_mode == "Load latest featured dataset":
        _global_featured = _global_featured_dataset_path()
        if _global_featured and os.path.exists(_global_featured):
            if st.button("Load latest featured dataset from disk", key="_fl_load_global"):
                _featured = _cached_load_df(_global_featured, os.path.getmtime(_global_featured))
                _set_flaml_dataset(
                    _featured,
                    dataset_meta=_dataset_meta_from_frame(
                        _featured,
                        data_source="disk",
                        requested_days=None,
                        featured_path=_global_featured,
                        origin="flaml_global_featured",
                    ),
                )
                st.session_state.flaml_automl_result = None
                st.success("Loaded the latest featured dataset from disk.")
        else:
            st.warning("No global featured dataset file found yet.")
    else:
        _fd1, _fd2 = st.columns([2, 2])
        with _fd1:
            _fl_source = st.selectbox("Data source", ["database", "scrape", "sample"], key="_fl_source")
        with _fd2:
            if _fl_source == "sample":
                _fl_num_races = st.slider("Sample races", 500, 5000, 1500, 100, key="_fl_num_races")
                _fl_days_back = None
            else:
                _fl_num_races = 1500
                _fl_days_back = st.slider("Days of history", 1, 2000, 90, 7, key="_fl_days_back")
        if st.button("📦 Prepare FLAML dataset", type="secondary", key="_fl_prepare"):
            _prep = st.progress(0, text="Preparing FLAML dataset …")
            _prep.progress(10, text="Collecting and processing data …")
            _featured, _processed, _meta = build_autotune_dataset(
                data_source=_fl_source,
                days_back=_fl_days_back,
                num_races=_fl_num_races,
            )
            _prep.progress(100, text="Dataset ready")
            _set_flaml_dataset(_featured, processed_df=_processed, dataset_meta=_meta)
            st.session_state.flaml_automl_result = None
            st.success(f"Prepared {_meta.get('rows', 0):,} rows for FLAML AutoML.")

    _fl_featured = st.session_state.get("flaml_featured_data")
    _fl_meta = st.session_state.get("flaml_dataset_meta") or {}
    if isinstance(_fl_featured, pd.DataFrame):
        st.success(
            f"FLAML dataset ready: {_fl_meta.get('rows', len(_fl_featured)):,} rows · "
            f"{_fl_meta.get('months') or '—'} months · {_fl_meta.get('date_start') or '?'} → {_fl_meta.get('date_end') or '?'}"
        )
    else:
        st.info("Load or prepare a dataset to run FLAML AutoML.")

    st.markdown("---")
    st.subheader("2️⃣ FLAML Setup")
    _f1, _f2, _f3 = st.columns(3)
    with _f1:
        _fl_mode = st.selectbox(
            "Task mode",
            options=["classification", "ranking"],
            format_func=lambda x: "Classification" if x == "classification" else "Ranking (race ordering)",
            key="_fl_mode",
        )
    with _f2:
        _fl_target = st.selectbox(
            ("Target" if _fl_mode == "classification" else "Evaluation target"),
            options=["won", "placed"],
            format_func=lambda x: "Winner (won)" if x == "won" else "Placed (placed)",
            key="_fl_target",
        )
    with _f3:
        _fl_time_budget = st.slider("Time budget (seconds)", 30, 7200, 600, 30, key="_fl_time_budget")

    if _fl_mode == "ranking":
        st.info(
            "Ranking mode trains on race ordering relevance labels (1st=5, 2nd=2, 3rd=1, else=0). "
            "The evaluation target above is used for calibration/binary diagnostics only."
        )

    _f4, _f5 = st.columns(2)
    with _f4:
        if _fl_mode == "classification":
            _fl_metric = st.selectbox(
                "Metric",
                options=["log_loss", "roc_auc", "f1", "accuracy"],
                index=0,
                key="_fl_metric_cls",
            )
        else:
            _fl_metric = st.selectbox(
                "Metric",
                options=["ndcg", "map"],
                index=0,
                key="_fl_metric_rank",
                help="Ranking objective metric for FLAML task=rank.",
            )
    with _f5:
        _fl_estimators = st.text_input(
            "Estimators (comma-separated, optional)",
            value="",
            key="_fl_estimators",
            help="Leave blank for sensible defaults. Example: lgbm,xgboost,xgb_limitdepth",
        )

    st.markdown("#### Run Logging")
    _fl_log_c1, _fl_log_c2, _fl_log_c3 = st.columns(3)
    with _fl_log_c1:
        _fl_enable_logs = st.checkbox(
            "Write FLAML log file",
            value=True,
            key="_fl_enable_logs",
            help="Writes FLAML trial progress to a file under data/flaml_logs.",
        )
    with _fl_log_c2:
        _fl_verbose = st.selectbox(
            "Verbose level",
            options=[0, 1, 2, 3],
            index=2,
            key="_fl_verbose",
            help="Higher values print more FLAML progress to the Streamlit terminal logs.",
        )
    with _fl_log_c3:
        _fl_log_training_metric = st.checkbox(
            "Log training metric",
            value=True,
            key="_fl_log_training_metric",
        )

    if st.button("🚀 Run FLAML AutoML", type="primary", width="stretch", key="_fl_run"):
        if not isinstance(_fl_featured, pd.DataFrame):
            st.error("Prepare a dataset first.")
            st.stop()

        _est_list = [x.strip() for x in str(_fl_estimators).split(",") if x.strip()]
        if not _est_list:
            _est_list = None

        _fl_log_file = None
        if _fl_enable_logs:
            _fl_log_dir = os.path.join(config.DATA_DIR, "flaml_logs")
            os.makedirs(_fl_log_dir, exist_ok=True)
            _fl_log_file = os.path.join(_fl_log_dir, f"flaml_{datetime.now():%Y%m%d_%H%M%S}.log")

        with st.spinner("Running FLAML AutoML on the current pipeline split …"):
            try:
                _fl_result = run_flaml_automl(
                    _fl_featured.copy(),
                    mode=_fl_mode,
                    target=_fl_target,
                    time_budget=int(_fl_time_budget),
                    metric=str(_fl_metric or "auto"),
                    estimator_list=_est_list,
                    verbose=int(_fl_verbose),
                    log_file_name=_fl_log_file,
                    log_training_metric=bool(_fl_log_training_metric),
                )
                st.session_state.flaml_automl_result = _fl_result
                st.success("FLAML AutoML run complete.")
            except Exception as _fl_exc:
                st.error(f"FLAML AutoML failed: {_fl_exc}")

    _fl_result = st.session_state.get("flaml_automl_result")
    if isinstance(_fl_result, dict):
        st.markdown("---")
        st.subheader("3️⃣ Results")
        _m = _fl_result.get("metrics") or {}

        def _fmt_metric(v, pct: bool = False) -> str:
            if v is None:
                return "—"
            try:
                fv = float(v)
            except Exception:
                return "—"
            return f"{fv:.1%}" if pct else f"{fv:.4f}"

        _r1, _r2, _r3, _r4 = st.columns(4)
        _r1.metric("Best Estimator", str(_fl_result.get("best_estimator", "—")))
        _r2.metric("Best Loss", _fmt_metric(_fl_result.get("best_loss")))
        _r3.metric("Brier", _fmt_metric(_m.get("brier")))
        _r4.metric("ROC AUC", _fmt_metric(_m.get("roc_auc")))

        _r5, _r6, _r7, _r8 = st.columns(4)
        _r5.metric("Accuracy", _fmt_metric(_m.get("accuracy"), pct=True))
        _r6.metric("Top-1 Accuracy", _fmt_metric(_m.get("top1_accuracy"), pct=True))
        _r7.metric("NDCG@1", _fmt_metric(_m.get("ndcg_at_1")))
        _r8.metric("Precision", _fmt_metric(_m.get("precision"), pct=True))

        if _fl_result.get("mode") == "ranking":
            _rr1, _rr2, _rr3 = st.columns(3)
            _rr1.metric("Rank Top-1", _fmt_metric(_m.get("rank_top1_accuracy"), pct=True))
            _rr2.metric("Rank NDCG@1", _fmt_metric(_m.get("rank_ndcg_at_1")))
            _rr3.metric("Rank NDCG@3", _fmt_metric(_m.get("rank_ndcg_at_3")))

        _fl_settings = _fl_result.get("settings") or {}
        st.caption(
            f"Mode: {_fl_result.get('mode', '—')} · Target: {_fl_result.get('target', '—')} · "
            f"Task: {_fl_settings.get('task', '—')} · Metric: {_fl_settings.get('metric', '—')}"
        )
        if _fl_result.get("mode") == "ranking":
            st.caption("Training target: relevance labels grouped by race (ranking task).")
        st.caption(
            f"Train rows: {_fl_result.get('n_train_rows', 0):,} · "
            f"Test rows: {_fl_result.get('n_test_rows', 0):,} · "
            f"Features: {_fl_result.get('n_features', 0):,}"
        )

        _fl_log_file = _fl_result.get("log_file_name")
        if _fl_log_file:
            st.caption(f"FLAML log file: {_fl_log_file}")
            if os.path.exists(_fl_log_file):
                with st.expander("FLAML log tail", expanded=False):
                    try:
                        with open(_fl_log_file, "r", encoding="utf-8", errors="ignore") as _fh:
                            _lines = _fh.readlines()
                        _tail = "".join(_lines[-200:]) if _lines else "(log file exists but is currently empty)"
                        st.text(_tail)
                    except Exception as _log_exc:
                        st.warning(f"Could not read log file: {_log_exc}")

        _best_est_table = _fl_result.get("best_estimator_table")
        if isinstance(_best_est_table, pd.DataFrame) and not _best_est_table.empty:
            st.markdown("#### Estimator Summary")
            st.dataframe(_best_est_table, width="stretch", hide_index=True)

        _best_cfg = _fl_result.get("best_config")
        if isinstance(_best_cfg, dict) and _best_cfg:
            with st.expander("Best config", expanded=False):
                st.json(_best_cfg)


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
            "Odds Feats": "On" if tc.get("include_odds") is True else ("Off" if tc.get("include_odds") is False else "—"),
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
            "place_classifier/brier_calibrated",
            "place_classifier/brier_raw",
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
            eq1, eq2 = st.columns(2)
            with eq1:
                fig_pnl = px.line(
                    r_curves, x="race_date", y="cum_pnl",
                    color="strategy", markers=True,
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
                    color="strategy", markers=True,
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
                wf_eq1, wf_eq2 = st.columns(2)
                with wf_eq1:
                    _wf_hover = [c for c in ["bet_number", "horse_name", "odds", "pnl"] if c in r_wf_curves.columns]
                    fig_wf_pnl = px.line(
                        r_wf_curves, x="race_date", y="cum_pnl",
                        color="strategy", markers=True,
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
                        color="strategy", markers=True,
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
                        color="label", markers=True,
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
#  PREDICT
# =====================================================================
elif page == "🔮 Predict":
    st.title("🔮 Predictions")

    if st.session_state.predictor is None:
        if os.path.exists(_ENSEMBLE_MODEL_PATH):
            load_existing_model()
            load_model_data(force=True)
        else:
            st.warning(
                "⚠️ No model available. Train one on the "
                "**Train & Tune** page."
            )
            st.stop()

    _model_df = st.session_state.get("model_featured_data")
    if _model_df is None:
        load_model_data()
        _model_df = st.session_state.get("model_featured_data")
        if _model_df is None:
            st.warning("⚠️ No data available. Train a model first.")
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
            "This is the model used for all predictions below. "
            "To switch, go to **🧪 Experiments** and click "
            "**🔄 Activate This Model** on any saved run."
        )

    tabs = st.tabs(
        ["🏇 Race Day", "📋 From Dataset", "✏️ Custom Entry"]
    )

    # ── Today's Races ────────────────────────────────────────────────
    with tabs[0]:
        _dc1, _dc2 = st.columns([2, 3])
        with _dc1:
            _pred_date = st.date_input(
                "📅 Race date",
                value=datetime.now().date(),
                key="pred_race_date",
            )
        _pred_date_str = _pred_date.strftime("%Y-%m-%d")
        _is_today = _pred_date == datetime.now().date()

        with _dc2:
            st.subheader(
                "🏇 UK & Ireland Racecards"
                + ("" if _is_today else f" — {_pred_date_str}")
            )
        st.caption(
            "Racecards are automatically fetched and cached. "
            "Click **Refresh** to re-scrape from Sporting Life."
        )

        # Detect date change → clear stale cards
        if st.session_state.get("_pred_date_prev") != _pred_date_str:
            st.session_state["live_cards"] = None
            st.session_state["live_preds"] = None
            st.session_state["_pred_date_prev"] = _pred_date_str

        # --- auto-load from cache / scrape once ----------------
        force_refresh = st.button(
            "🔄 Refresh Racecards", type="secondary",
        )

        if (
            "live_cards" not in st.session_state
            or st.session_state["live_cards"] is None
            or force_refresh
        ):
            _scrape_prog = st.progress(
                0, text=f"Loading races for {_pred_date_str} …",
            )
            _scrape_status = st.empty()

            def _scrape_cb(current, total, track):
                _scrape_prog.progress(
                    current / total,
                    text=f"Scraping race {current}/{total} — {track} …",
                )

            cards_df = get_scraped_racecards(
                date_str=_pred_date_str,
                progress_callback=_scrape_cb,
                force_refresh=force_refresh,
            )
            _scrape_prog.empty()
            _scrape_status.empty()

            if cards_df is None or cards_df.empty:
                st.warning(
                    f"No racecards found for {_pred_date_str}. "
                    "Cards may not be published this far in advance."
                )
            else:
                source = "Refreshed" if force_refresh else "Loaded"
                st.success(
                    f"{source} {len(cards_df)} entries across "
                    f"{cards_df['race_id'].nunique()} races"
                )
                st.session_state["live_cards"] = cards_df

        if (
            "live_cards" in st.session_state
            and st.session_state["live_cards"] is not None
        ):
            cards = st.session_state["live_cards"]
            tracks_today = (
                sorted(cards["track"].unique())
                if "track" in cards.columns else []
            )
            if tracks_today:
                st.markdown(f"**Venues:** {', '.join(tracks_today)}")

            # --- batch predict (button-triggered) ---------------
            _run_preds = st.button(
                "▶️ Run Predictions",
                type="primary",
                key="btn_run_live_preds",
            )

            if (
                _run_preds
                or force_refresh
            ):
                _pred_prog = st.progress(0, text="Processing runners …")
                all_live_preds: dict[str, pd.DataFrame] = {}
                race_ids = cards["race_id"].unique()

                # ── Batch process + feature-engineer WITH history ──
                _all_cards = cards.copy()
                _all_cards["won"] = 0
                _all_cards["finish_position"] = 0
                _all_cards["finish_time_secs"] = 0.0
                _all_cards["lengths_behind"] = np.nan

                _pred_prog.progress(10, text="Processing data …")
                try:
                    _all_proc = process_data(df=_all_cards, save=False)
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.stop()

                # ── Gap-fill: scrape missing intermediate results ──
                _gap_extra = None
                if not _is_today:
                    _pred_prog.progress(20, text="Checking for date gaps …")
                    try:
                        def _gap_cb(cur, tot, ds):
                            _pred_prog.progress(
                                20 + int(10 * cur / tot),
                                text=f"Gap-fill: scraping results for {ds} ({cur}/{tot}) …",
                            )
                        _gap_extra = scrape_gap_fill(_pred_date_str, progress_fn=_gap_cb)
                    except Exception as e:
                        logger.warning(f"Gap-fill failed: {e}")

                _pred_prog.progress(30, text="Engineering features (with history) …")
                try:
                    _all_feat = feature_engineer_with_history(_all_proc, extra_history=_gap_extra)
                except Exception as e:
                    st.error(f"Feature engineering failed: {e}")
                    st.stop()

                for idx, rid in enumerate(race_ids):
                    _pred_prog.progress(
                        50 + int(50 * (idx + 1) / len(race_ids)),
                        text=f"Predicting race {idx+1}/{len(race_ids)} …",
                    )

                    try:
                        feat = _all_feat[_all_feat["race_id"] == rid].copy()
                        feat = feat.reset_index(drop=True)
                        if feat.empty:
                            continue
                        preds = st.session_state.predictor.predict_race(
                            feat,
                            ew_fraction=st.session_state.value_config.get("ew_fraction"),
                        )
                        all_live_preds[rid] = preds
                    except Exception as e:
                        all_live_preds[rid] = f"Error: {e}"

                _pred_prog.empty()
                st.session_state["live_preds"] = all_live_preds

            # --- display results --------------------------------
            if "live_preds" in st.session_state and st.session_state["live_preds"]:
                live_preds = st.session_state["live_preds"]

                for rid in cards["race_id"].unique():
                    race_slice = cards[cards["race_id"] == rid]
                    track = race_slice["track"].iloc[0] if "track" in race_slice.columns else "?"
                    off_time = race_slice["off_time"].iloc[0] if "off_time" in race_slice.columns else ""
                    race_name = race_slice["race_name"].iloc[0] if "race_name" in race_slice.columns else ""

                    header = (
                        f"🏟️ {track} — {off_time} — {race_name} "
                        f"({len(race_slice)} runners)"
                    )
                    with st.expander(header):
                        result = live_preds.get(rid)
                        if result is None:
                            st.info("Click **▶️ Run Predictions** above.")
                        elif isinstance(result, str):
                            st.error(result)
                        else:
                            preds = result
                            _ew_cfg = st.session_state.value_config
                            for _, row in preds.iterrows():
                                rank = int(row["predicted_rank"])
                                emoji = (
                                    "🥇" if rank == 1 else
                                    "🥈" if rank == 2 else
                                    "🥉" if rank == 3 else f"#{rank}"
                                )
                                c1, c2, c3, c4, c5 = st.columns(
                                    [1, 3, 2, 2, 2],
                                )
                                c1.markdown(f"**{emoji}**")
                                _tp_tag = " 🎯 TOP PICK" if rank == 1 else ""
                                c2.markdown(f"**{row['horse_name']}**{_tp_tag}")
                                c3.metric(
                                    "Win Prob",
                                    f"{row['win_probability']:.1%}",
                                )
                                if (
                                    "odds" in row
                                    and pd.notna(row["odds"])
                                ):
                                    c4.metric("Odds", f"{row['odds']:.1f}")

                                # Win value badge
                                _badges = []
                                _vt_base = _ew_cfg["value_threshold"]
                                _row_odds = float(row.get("odds", 3.0)) if pd.notna(row.get("odds")) else 3.0
                                _dyn_vt = _vt_base * np.sqrt(_row_odds / 3.0)
                                if (
                                    "value_score" in row
                                    and pd.notna(row["value_score"])
                                    and row["value_score"] > _dyn_vt
                                ):
                                    _badges.append(f"⭐ **+{row['value_score']:.1%}** win")
                                # EW value badge
                                if (
                                    _ew_cfg.get("ew_enabled", True)
                                    and row.get("place_value", False)
                                ):
                                    _ew_edge_base = _ew_cfg.get("ew_min_place_edge", 0.05)
                                    _ew_dyn = _ew_edge_base * np.sqrt(_row_odds / 3.0)
                                    if row.get("place_edge", 0) > _ew_dyn:
                                        _badges.append(
                                            f"🔀 **+{row['place_edge']:.1%}** EW"
                                        )
                                if _badges:
                                    c5.markdown(" · ".join(_badges))

                            # Summaries
                            if "value_score" in preds.columns and "odds" in preds.columns:
                                _vt_dyn = _ew_cfg["value_threshold"] * np.sqrt(preds["odds"].clip(lower=1.0) / 3.0)
                                vb = preds[preds["value_score"] > _vt_dyn]
                                if not vb.empty:
                                    st.success(
                                        "💰 Value picks: "
                                        f"{', '.join(vb['horse_name'])}"
                                    )
                            if (
                                _ew_cfg.get("ew_enabled", True)
                                and "ew_value" in preds.columns
                            ):
                                ew_bets = ew_value_bets(
                                    preds,
                                    min_place_edge=_ew_cfg.get("ew_min_place_edge", 0.05),
                                    min_odds=_ew_cfg.get("ew_min_odds", 4.0),
                                    max_odds=_ew_cfg.get("ew_max_odds", 51.0),
                                )
                                if not ew_bets.empty:
                                    _ew_names = ", ".join(ew_bets["horse_name"])
                                    st.info(f"🔀 EW value: {_ew_names}")
            else:
                st.info("Press **▶️ Run Predictions** to analyse all races.")

    # ── From Dataset ─────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("📋 Predict from Loaded Data")
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
            predictions = st.session_state.predictor.predict_race(
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
                _p_thresh = _pvc["value_threshold"]
                _p_dyn_thresh = _p_thresh * np.sqrt(predictions["odds"].clip(lower=1.0) / 3.0)
                vb = predictions[predictions["value_score"] > _p_dyn_thresh]
                if not vb.empty:
                    st.markdown("### 💰 Value Bets")
                    for _, row in vb.iterrows():
                        _kf = kelly_criterion(row['win_probability'], row['odds'], fraction=_pvc["kelly_fraction"])
                        _k_label = f"{_pvc['kelly_fraction']:.0%}"
                        _ks = f" · Kelly {_k_label} **{_kf*100:.1f}%**" if _kf > 0.001 else ""
                        _clv = row['win_probability'] * row['odds']
                        _clv_str = f" · CLV **{_clv:.3f}x**" if _clv > 1.0 else ""
                        st.info(
                            f"**{row['horse_name']}** — "
                            f"Model: {row['win_probability']:.1%} vs "
                            f"Market: {row['implied_prob']:.1%} — "
                            f"Value: +{row['value_score']:.1%}{_ks}{_clv_str}"
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
                        _terms_str = f"{int(row['ew_places'])} places at {row['ew_fraction_str']}"
                        st.success(
                            f"**{row['horse_name']}** @ {row['odds']:.1f} — "
                            f"{_place_str} — "
                            f"Edge: +{row['place_edge']:.1%} · "
                            f"EW EV: {row['ew_ev']:+.1%}{_ew_kelly_str}\n\n"
                            f"<small>Terms: {_terms_str} · "
                            f"Place odds: {row['place_odds']:.2f}</small>",
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
    with tabs[2]:
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
            predictions = st.session_state.predictor.predict_race(
                featured_custom,
                ew_fraction=st.session_state.value_config.get("ew_fraction"),
            )

            st.markdown("### 🏆 Custom Race Prediction")
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
            "Scrapes racecards for the selected date, runs the ensemble "
            "on every race, and surfaces horses where the model sees "
            "genuine value."
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
    _has_preds = "picks_preds" in st.session_state and st.session_state["picks_preds"] is not None
    _preds_stale = _has_preds and st.session_state.get("picks_model_fp") != _picks_model_fp

    # ── Determine if predictions need to run ─────────────────────
    _needs_preds = not _has_preds or _preds_stale or _odds_refresh

    if _needs_preds:
        cards_df = cards_df.reset_index(drop=True)

        # ── Load featured data from cache hierarchy ──────────────
        _feat_cache = st.session_state.get("picks_featured", {})

        _has_lookahead = lookahead_cache_valid(_picks_date_str, current_cards_sig=None)

        all_feat = None
        progress = st.progress(10, text="Loading features …")

        # 1. In-memory cache
        if _picks_date_str in _feat_cache and not _odds_refresh:
            all_feat = _feat_cache[_picks_date_str]
            progress.progress(50, text="⚡ Using cached features …")

        # 2. Lookahead cache
        if all_feat is None and _has_lookahead:
            _lookahead_feat = load_lookahead_cache(_picks_date_str, current_cards_sig=None)
            if _lookahead_feat is not None and not _lookahead_feat.empty:
                all_feat = _lookahead_feat
                progress.progress(50, text="⚡ Loaded lookahead cache …")

        # 3. Live feature cache on disk — scan for any cached file for this date
        if all_feat is None:
            _date_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "live_feature_cache", _picks_date_str)
            if os.path.isdir(_date_cache_dir):
                _pq_files = sorted(
                    (f for f in os.scandir(_date_cache_dir) if f.name.endswith(".parquet")),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True,
                )
                if _pq_files:
                    try:
                        _loaded = pd.read_parquet(_pq_files[0].path, engine="pyarrow")
                        if not _loaded.empty:
                            all_feat = _loaded
                            progress.progress(50, text="⚡ Loaded live feature cache …")
                    except Exception:
                        pass

        # 4. Global featured dataset (today/past)
        if all_feat is None and _picks_date <= datetime.now().date():
            _gfp = _global_featured_dataset_path()
            if _gfp:
                progress.progress(20, text="Loading featured dataset …")
                _full_feat = _cached_load_df(_gfp, os.path.getmtime(_gfp))
                if _full_feat is not None and "race_date" in _full_feat.columns:
                    _date_mask = (
                        pd.to_datetime(_full_feat["race_date"], errors="coerce")
                        .dt.strftime("%Y-%m-%d") == _picks_date_str
                    )
                    _date_slice = _full_feat.loc[_date_mask]
                    if not _date_slice.empty:
                        all_feat = _date_slice.copy()
                        progress.progress(50, text="⚡ Loaded from featured dataset …")

        if all_feat is None or all_feat.empty:
            progress.empty()
            st.warning(
                f"No featured data available for {_picks_date_str}. "
                "Please run **Prepare Data** or **Train** first to build "
                "the featured dataset for this date."
            )
            st.stop()

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
                full_preds = st.session_state.predictor.predict_races(
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
                    preds = st.session_state.predictor.predict_race(
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
        st.session_state["picks_model_label"] = "3-Model"

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
            _used_label = st.session_state.get("picks_model_label", "Ensemble")
            st.caption(f"ℹ️ Predictions computed with: **{_used_label}**")

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
            full_preds["dyn_threshold"] = (
                base_thresh * np.sqrt(full_preds["odds"] / 3.0)
            )
            full_preds["is_value"] = (
                full_preds["value_score"] > full_preds["dyn_threshold"]
            )
            value_df = full_preds[full_preds["is_value"]].copy()
        else:
            full_preds["is_value"] = False
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
                _val_pnl += (_vs["odds"] - 1.0) if _vs["result_won"] else -1.0
            _val_roi = (_val_pnl / _val_n * 100) if _val_n else 0.0

            # --- EW bet returns (exclude NR/void) ---
            _ew_settled = ew_df[ew_df["is_settled"] & ~ew_df["result_is_nr"]].copy() if not ew_df.empty else pd.DataFrame()
            _ew_n = len(_ew_settled)
            _ew_wins = 0
            _ew_placed = 0
            _ew_pnl = 0.0
            for _, _es in _ew_settled.iterrows():
                _fp = int(_es["result_fp"])
                _ew_pp = int(_es.get("ew_places", 3))
                _p_odds = float(_es.get("place_odds", 0))
                _w_odds = float(_es["odds"])
                _pnl_ew = -2.0  # cost 2 units
                if _es["result_won"]:
                    _pnl_ew += _w_odds + _p_odds
                    _ew_wins += 1
                    _ew_placed += 1
                elif 0 < _fp <= _ew_pp:
                    _pnl_ew += _p_odds
                    _ew_placed += 1
                _ew_pnl += _pnl_ew
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
                if _tp_best["result_won"]:
                    _tp_pnl += _tp_best["odds"] - 1.0
                    _tp_wins += 1
                else:
                    _tp_pnl -= 1.0

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
                if rank == 1:
                    flags.append("🎯")

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


# =====================================================================
#  BACKTEST
# =====================================================================
elif page == "🔁 Backtest":
    st.title("🔁 Walk-Forward Backtest")
    st.caption(
        "Run expanding-window validation on your featured dataset to compare "
        "Top Pick, Value, and Each-Way strategies across time."
    )

    _bt_df = st.session_state.get("model_featured_data")
    if _bt_df is None:
        load_model_data()
        _bt_df = st.session_state.get("model_featured_data")
    if _bt_df is None:
        st.warning("No featured data available. Train a model first.")
        st.stop()

    bt_df = _bt_df.copy()
    bt_df["race_date"] = pd.to_datetime(bt_df["race_date"], errors="coerce")
    bt_df = bt_df.dropna(subset=["race_date"]).copy()

    _n_races = int(bt_df["race_id"].nunique()) if "race_id" in bt_df.columns else 0
    _n_months = int(bt_df["race_date"].dt.to_period("M").nunique())

    st.info(
        f"Dataset loaded: **{len(bt_df):,} runners** across **{_n_races:,} races** "
        f"and **{_n_months} months**."
    )
    st.caption(
        "Backtest settles using final SP odds at race-off, which is typically "
        "more optimistic than pre-race/live execution prices."
    )

    _model_labels = {
        "triple_ensemble": "2-Model (Win Clf + Place Clf)",
    }
    _model_keys = list(_model_labels.keys())

    c1, c2, c3 = st.columns([1.4, 1.1, 1.1])
    with c1:
        bt_model = st.selectbox(
            "Model",
            options=_model_keys,
            index=_model_keys.index("triple_ensemble") if "triple_ensemble" in _model_keys else 0,
            format_func=lambda k: _model_labels.get(k, k),
            help="Model used for each fold's train/test step.",
        )
    with c2:
        bt_min_train = st.slider(
            "Min train months",
            min_value=1,
            max_value=max(1, min(24, _n_months - 1 if _n_months > 1 else 1)),
            value=min(3, max(1, _n_months - 1 if _n_months > 1 else 1)),
            help="Number of months required before first out-of-sample fold.",
        )
    with c3:
        bt_value_threshold = st.slider(
            "Value threshold",
            min_value=0.00,
            max_value=0.20,
            value=float(st.session_state.value_config.get("value_threshold", 0.05)),
            step=0.01,
            help="Minimum model edge for Value and EW selection rules.",
        )

    bt_fast_fold = st.checkbox(
        "Fast fold (halve tree count)",
        value=True,
        key="bt_fast_fold",
        help="Halve n_estimators per fold for speed. Disable for full HP match.",
    )

    run_bt = st.button("🚀 Run Backtest", type="primary")

    if run_bt:
        with st.spinner("Running walk-forward backtest..."):
            try:
                st.session_state.bt_report = walk_forward_validation(
                    bt_df,
                    model_type=bt_model,
                    min_train_months=bt_min_train,
                    value_threshold=bt_value_threshold,
                    fast_fold=bt_fast_fold,
                    ew_min_place_edge=float(
                        st.session_state.value_config.get("ew_min_place_edge", 0.15)
                    ),
                )
                st.success("Backtest complete.")
            except Exception as e:
                st.session_state.bt_report = None
                st.error(f"Backtest failed: {e}")

    report = st.session_state.bt_report
    if report:
        summary_df = report.get("summary", pd.DataFrame())
        bets_df = report.get("bets", pd.DataFrame())
        curves_df = report.get("curves", pd.DataFrame())

        if not summary_df.empty:
            st.markdown("### Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Folds", int(len(summary_df)))
            m2.metric("Avg RPS", f"{summary_df['rps'].mean():.4f}" if 'rps' in summary_df.columns else f"{summary_df['brier_score'].mean():.4f}")
            m3.metric("Avg NDCG@1", f"{summary_df['ndcg_at_1'].mean():.4f}")
            m4.metric("Avg Top-1", f"{summary_df['top1_accuracy'].mean():.1%}")

            p1, p2, p3, p4 = st.columns(4)
            _tp_pnl = float(summary_df["top_pick_pnl"].sum())
            _v_pnl = float(summary_df["value_pnl"].sum())
            _ew_pnl = float(summary_df["ew_pnl"].sum())
            _combined_pnl = _tp_pnl + _v_pnl + _ew_pnl
            p1.metric("Top Pick P&L", f"{_tp_pnl:+.2f}u")
            p2.metric("Value P&L", f"{_v_pnl:+.2f}u")
            p3.metric("EW P&L", f"{_ew_pnl:+.2f}u")
            p4.metric("Combined P&L", f"{_combined_pnl:+.2f}u")

            st.dataframe(
                summary_df,
                width="stretch",
                hide_index=True,
            )

        if not curves_df.empty:
            st.markdown("### Cumulative Curves")
            _strategy_labels = {
                "top_pick": "Top Pick",
                "value": "Value",
                "each_way": "Each-Way",
            }
            _curves_plot = curves_df.copy()
            _curves_plot["strategy"] = _curves_plot["strategy"].map(
                lambda s: _strategy_labels.get(s, s)
            )
            fig_curve = px.line(
                _curves_plot,
                x="race_date",
                y="cum_pnl",
                color="strategy",
                title="Cumulative P&L by Strategy",
                labels={"race_date": "Date", "cum_pnl": "Cumulative P&L"},
            )
            fig_curve.update_layout(height=460, legend_title_text="")
            st.plotly_chart(fig_curve, width="stretch")

        st.markdown("### Exports")
        d1, d2, d3 = st.columns(3)
        with d1:
            if not summary_df.empty:
                st.download_button(
                    "Download fold summary",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="fold_summary.csv",
                    mime="text/csv",
                )
        with d2:
            if not bets_df.empty:
                st.download_button(
                    "Download all bets",
                    data=bets_df.to_csv(index=False).encode("utf-8"),
                    file_name="all_bets.csv",
                    mime="text/csv",
                )
        with d3:
            if not curves_df.empty:
                st.download_button(
                    "Download curves",
                    data=curves_df.to_csv(index=False).encode("utf-8"),
                    file_name="cumulative_pnl.csv",
                    mime="text/csv",
                )

        if not bets_df.empty:
            with st.expander("View all settled bets"):
                st.dataframe(
                    bets_df,
                    width="stretch",
                    hide_index=True,
                )


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

    tab_fi, tab_metrics, tab_overfit = st.tabs(
        [
            "🔑 Feature Importance",
            "📊 Metrics & Features",
            "🔬 Overfitting Diagnostics",
        ]
    )

    # ── Feature Importance ───────────────────────────────────────────
    with tab_fi:
        # Build a map of available sub-models on the predictor
        _fi_model_map = {
            "classifier":("Win Classifier",   getattr(predictor, "clf_model",   None)),
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

            # Elo feature breakdown
            elo_feats = fi[
                fi["feature"].str.contains("elo", case=False)
            ]
            if not elo_feats.empty:
                st.markdown("#### ⚡ Elo Feature Contributions")
                st.dataframe(
                    elo_feats.style.format({"importance": "{:.4f}"}),
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

    # ── Metrics & Features ───────────────────────────────────────────
    with tab_metrics:
        metrics = st.session_state.metrics
        if not metrics:
            # Try loading from latest persisted run
            _lid = get_latest_run_id()
            if _lid:
                try:
                    _lr = load_run(_lid)
                    metrics = _lr.get("metrics")
                    st.session_state.metrics = metrics
                except Exception:
                    pass

        if metrics:
            st.subheader("📊 Last Training Metrics")
            if isinstance(metrics, dict) and any(
                isinstance(v, dict) for v in metrics.values()
            ):
                mdf = pd.DataFrame(metrics).T
                _num_cols = mdf.select_dtypes("number").columns.tolist()
                st.dataframe(
                    mdf.style.format("{:.4f}", subset=_num_cols),
                    width="stretch",
                )

                fig = px.bar(
                    mdf[_num_cols].reset_index().melt(id_vars="index"),
                    x="index", y="value", color="variable",
                    barmode="group", title="Model Comparison",
                )
                st.plotly_chart(fig, width="stretch")

        st.subheader("📖 Feature Reference")
        st.info(
            "Use this page for feature importance and model diagnostics.  \n"
            "For equity curves, calibration charts, bet logs, and saved run details, "
            "visit the **🧪 Experiments** page."
        )

    # ── Overfitting Diagnostics ──────────────────────────────────────
    with tab_overfit:
        st.subheader("🔬 Overfitting Diagnostics")
        st.caption(
            "Compare out-of-fold vs validation performance to spot overfitting. "
            "Large gaps between OOF and validation metrics indicate the model "
            "may still be too closely fitted to the development data."
        )

        # --- Collect train & test metrics from the active run ---------
        _active_rid = st.session_state.get("active_run_id")
        _run_data_overfit = None
        _test_metrics_of = st.session_state.get("metrics")
        _train_metrics_of = None

        if _active_rid:
            try:
                _run_data_overfit = load_run_meta(_active_rid)
                _train_metrics_of = _run_data_overfit.get("train_metrics")
                if not _test_metrics_of:
                    _test_metrics_of = _run_data_overfit.get("metrics")
            except Exception:
                pass

        if not _test_metrics_of or not _train_metrics_of:
            st.info(
                "⚠️ OOF vs validation metrics are not available for the current run. "
                "Re-train the model to generate overfitting diagnostics."
            )
        else:
            # Cached by run_id only — no json.dumps in the hot path.
            _of_result = _build_overfit_section_charts(_active_rid or "")

            if _of_result["overfit_figs"]:
                # --- 1) OOF vs validation grouped bar chart ----------
                st.markdown("### 📊 OOF vs Validation — Per Sub-Model")
                _of_cols = st.columns(min(len(_of_result["overfit_figs"]), 2))
                for i, fig_of in enumerate(_of_result["overfit_figs"]):
                    _of_cols[i % 2].plotly_chart(fig_of, width="stretch")

                # --- 2) Overfit gap heatmap ---------------------------
                st.markdown("### 🌡️ Overfit Gap Analysis")
                st.caption(
                    "**Gap = OOF − Validation.** A gap near 0 is ideal. "
                    "Gaps above ~0.10 suggest overfitting; "
                    "negative gaps indicate the model performs better on unseen data (rare but possible)."
                )
                st.plotly_chart(_of_result["heatmap_fig"], width="stretch")

                # --- 3) Summary table with colour-coded flags ---------
                st.markdown("### 📋 Detailed Comparison")
                st.dataframe(
                    _of_result["display_df"].style.format(
                        {"OOF": "{:.4f}", "Validation": "{:.4f}", "Gap": "{:+.4f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning("Could not parse train/test metrics for comparison.")

        # --- 4) Feature importance concentration ----------------------
        st.markdown("---")
        st.markdown("### 📐 Feature Importance Concentration")
        st.caption(
            "A model that relies heavily on a few features is more brittle. "
            "The Lorenz curve shows how evenly importance is distributed — "
            "the closer to the diagonal, the more balanced."
        )
        st.markdown(
            """
**Optuna** is a Bayesian hyperparameter optimisation framework.
Instead of grid-searching every combination, it uses past trial
results to *intelligently propose* the next set of parameters.

**The pipeline:**

1. The training data (excluding the final test set) is split into
   a **train fold** (80%) and a **validation fold** (20%).
2. Optuna proposes a set of hyperparameters and trains a model on
   the train fold.
3. The model is scored on the validation fold using **NDCG@1**.
4. Optuna repeats for *N* trials, guided by a Tree-structured
   Parzen Estimator (TPE) sampler.
5. The best parameters are used to **retrain on the full training
   set**, and the final test set produces the reported metrics.

**Parameters searched:**

| Parameter | Range |
|---|---|
| `n_estimators` | 100 – 1,500 |
| `max_depth` | 3 – 12 |
| `learning_rate` | 0.005 – 0.3 (log scale) |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.3 – 1.0 |
| `min_child_weight` / `min_child_samples` | 1 – 20 / 5 – 50 |

**How many trials?**

| Trials | Quality | Time |
|---|---|---|
| 10–20 | Quick scan | ~1 min |
| 30 (default) | Good balance | ~2 min |
| 100+ | Thorough | 5–15 min |

> 💡 Each sub-model is tuned via Optuna.
> The best params are used to retrain on the full training set.
            """
        )

    st.markdown("---")

    # -- Betting strategies --
    st.header("🎲 Betting Strategies")

    with st.expander("🎯 Top Pick Strategy", expanded=True):
        st.markdown(
            """
**Rule:** Bet 1 unit on the model's **#1 ranked horse** in every
race.

- Simple, always-on strategy — one bet per race.
- P&L = (odds − 1) if the pick wins, −1 if it loses.
- Best for models with high top-1 accuracy at reasonable odds.
            """
        )

    with st.expander("💎 Value Strategy"):
        st.markdown(
            """
**Rule:** Bet 1 unit on every horse where
`model_prob − implied_prob > threshold` (default 0.05).

- May produce **zero or multiple** bets per race.
- Targets situations where the market under-prices a horse.
- Often picks at longer odds → lower strike rate but bigger
  payoffs when they land.
- The threshold can be tuned on the backtest page.
            """
        )

    st.markdown("---")
    st.caption(
        "⚠️ This system is for **educational purposes only**. "
        "Past performance does not guarantee future results."
    )
