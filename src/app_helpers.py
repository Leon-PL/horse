"""
App Helper Layer
================
Pure/diagnostic helpers for the Streamlit picks app: cached run-metadata
and database-stat reads, pace diagnostic panels, SHAP rendering and the
value-bet mask.

These functions hold no page logic.
"""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.database import db_stats as _raw_db_stats
from src.run_store import load_run_meta as _raw_load_run_meta
from src.bet_settlement import DEFAULT_VALUE_THRESHOLD, dynamic_value_threshold

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

    from src.runner_identity import join_runner_columns
    pace_cols = [c for c in _PACE_DISPLAY_COLUMNS if c in feature_df.columns and c not in predictions.columns]
    if not all(c in predictions and c in feature_df for c in ("race_id", "horse_name")) or not pace_cols:
        return predictions
    return join_runner_columns(predictions, feature_df, pace_cols, require_all=False, skip_ambiguous=True)


def attach_result_outcomes(predictions, results):
    """Associate results by source identity; unknown namesakes stay unsettled."""
    from src.runner_identity import runner_positions
    import numpy as np
    positions = (runner_positions(predictions,results,require_all=False,skip_ambiguous=True)
                 if results is not None and not results.empty else np.full(len(predictions),-1,dtype=int))
    values = results.reset_index(drop=True).reindex(positions) if results is not None else pd.DataFrame(index=range(len(predictions)))
    out = predictions.copy()
    for source,target in (("finish_position","result_fp"),("won","result_won")):
        out[target] = pd.to_numeric(values.get(source,pd.Series(0,index=values.index)),errors="coerce").fillna(0).to_numpy(dtype=int)
    out["result_fp_label"] = values.get("finish_pos_label",pd.Series("",index=values.index)).fillna("").astype(str).to_numpy()
    out["is_settled"] = positions >= 0
    return out


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
# A long TTL avoids re-reading run metadata on every rerun; a rename made
# in the training app shows here within the hour.
@st.cache_data(ttl=3600, show_spinner=False)
def load_run_meta(run_id: str) -> dict:
    return _raw_load_run_meta(run_id)


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
    base_thresh = float(cfg.get("value_threshold", DEFAULT_VALUE_THRESHOLD))
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
