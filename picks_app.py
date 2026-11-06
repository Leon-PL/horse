"""
Picks App — lean serving UI
===========================
The day-to-day serving interface for the Horse Racing Prediction system.
It loads ONLY the production-promoted model via the registry pointer
(``src.run_store.load_production_predictor`` / ``production.json``) and never
loads the full feature store at startup, so it opens fast.

Pages:
- 💰 Today's Picks — live racecards, value picks, manual custom-race entry
- 🔌 Matchbook API — exchange connectivity harness

Training, tuning, backtesting, insights and promotion live in the separate
``training_app.py``.

Run with:
    streamlit run picks_app.py
"""

import hashlib
import os

# Limit numpy/BLAS threads to prevent ReleaseSemaphore crashes in Streamlit threadpool
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.ingestion.pipeline import get_scraped_racecards
from src.ingestion.sporting_life import scrape_todays_results
from src.live_prediction import (
    features_for_live_cards,
    feature_store_signature,
)
from src.model import RacePredictor
from src.run_store import (
    GENERATION_MATCH,
    GENERATION_MISMATCH,
    GENERATION_UNKNOWN,
    feature_generation_status,
    get_production_run_id,
    production_model_revision,
)
from src.utils import (
    kelly_criterion,
)
from src.bet_settlement import (
    DEFAULT_VALUE_CONFIG,
    dynamic_value_threshold,
    ew_placed_flag,
    settle_ew_bet,
    settle_win_bet,
)
from src.each_way import (
    ew_value_bets,
    get_ew_terms,
    kelly_ew,
)
from src.matchbook_client import MatchbookAPIError, MatchbookClient
from src.matchbook_signals import build_fake_prediction_frame, build_signal_frame
from src.paper_trade_store import append_paper_trades, build_paper_trades_from_signals, load_paper_trades, save_paper_trades, settle_paper_trades

logger = logging.getLogger(__name__)


from src.app_helpers import (
    _attach_pace_diagnostics,
    _cached_db_stats,
    _has_pace_diagnostics,
    _ordinal,
    _pace_race_summary,
    _render_pace_panel,
    _render_shap_explanation,
    _value_bet_mask,
    _value_odds_range,
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
    "active_run_id": None,
    "value_config": dict(DEFAULT_VALUE_CONFIG),
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
@st.cache_resource(show_spinner="Loading model …", max_entries=2)
def _cached_load_model(revision):
    """Cache exactly the promoted artifact, with promotion/replacement invalidation."""
    if revision is None:
        return None
    from src.run_store import load_run_predictor
    return load_run_predictor(revision[0])


def load_existing_model():
    revision = production_model_revision()
    try:
        p = _cached_load_model(revision)
    except Exception:
        logger.exception("Failed to load the production model")
        st.session_state.predictor = None
        return False
    st.session_state.predictor = p
    st.session_state.active_run_id = revision[0] if revision else None
    return p is not None


def _predictor_runtime_token(predictor: RacePredictor | None) -> str:
    if predictor is None:
        return "no-predictor"
    return f"{id(predictor)}:{st.session_state.get('_production_revision')}"


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


# Inputs the quality-control gates need carried onto the predictions frame.
_QC_INPUT_COLUMNS = ["horse_prev_races"]


def _attach_qc_inputs(preds: pd.DataFrame, feature_df: pd.DataFrame | None) -> pd.DataFrame:
    """Carry data-quality gate inputs (e.g. horse_prev_races) onto predictions."""
    if feature_df is None or feature_df.empty or preds is None or preds.empty:
        return preds
    from src.runner_identity import join_runner_columns
    qc_cols = [c for c in _QC_INPUT_COLUMNS if c in feature_df.columns and c not in preds.columns]
    if not all(col in preds and col in feature_df for col in ("race_id","horse_name")) or not qc_cols:
        return preds
    return join_runner_columns(preds,feature_df,qc_cols,require_all=False,skip_ambiguous=True)


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
        preds = _attach_qc_inputs(preds, feature_df)
        return _prediction_cache_set(cache_key, preds)

    if "race_id" not in feature_df.columns:
        preds = predictor.predict_race(feature_df, ew_fraction=ew_fraction)
        preds = _attach_pace_diagnostics(preds, feature_df)
        preds = _attach_qc_inputs(preds, feature_df)
        return _prediction_cache_set(cache_key, preds)

    all_preds: list[pd.DataFrame] = []
    for race_id in feature_df["race_id"].unique() if "race_id" in feature_df.columns else []:
        feat_slice = feature_df[feature_df["race_id"] == race_id].copy()
        if feat_slice.empty:
            continue
        preds = predictor.predict_race(feat_slice, ew_fraction=ew_fraction)
        preds = _attach_pace_diagnostics(preds, feat_slice)
        preds = _attach_qc_inputs(preds, feat_slice)
        preds["race_id"] = race_id
        all_preds.append(preds)

    combined = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return _prediction_cache_set(cache_key, combined)


# Refresh session state whenever another process promotes/replaces a model.
# The cache identity and displayed run must always describe the same artifact.
_production_revision = production_model_revision()
if st.session_state.get("_production_revision") != _production_revision:
    st.session_state["_production_revision"] = _production_revision
    st.session_state.predictor = None
    st.session_state.active_run_id = _production_revision[0] if _production_revision else None
    for _key in ("picks_preds", "picks_meta", "picks_model_fp", "picks_explanations",
                 "_prediction_result_cache", "_prediction_result_cache_order"):
        st.session_state.pop(_key, None)


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
    st.sidebar.info("🌐 Racecards via **Sporting Life**; **Racing TV** pace maps and telemetry are collected alongside for analysis.")

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
elif _prod_id:
    if st.sidebar.button("Load model", key="sidebar_load_model"):
        if load_existing_model():
            st.session_state.active_run_id = _prod_id
            st.rerun()
        else:
            st.sidebar.error("Failed to load model.")
else:
    st.sidebar.warning("⚠️ No model — train & promote one in the training app.")

# Always state whether the promoted model and the feature store agree. Silence
# here would read as "fine", and an unverifiable model is not fine.
if _prod_id:
    _gen_status = feature_generation_status(_prod_id)
    if _gen_status["state"] == GENERATION_MISMATCH:
        st.sidebar.error("🚫 **Stale model** — trained on different features.")
        st.sidebar.caption(_gen_status["message"])
    elif _gen_status["state"] == GENERATION_UNKNOWN:
        st.sidebar.warning("❓ Feature generation unverified.")
        st.sidebar.caption(_gen_status["message"])
    elif _gen_status["state"] == GENERATION_MATCH:
        st.sidebar.caption(f"🔒 Features verified ({_gen_status['run_fe_signature']})")

# Apply pending value-strategy updates before sidebar widgets are instantiated.
_pending_vc = st.session_state.pop("_pending_value_config", None)
if isinstance(_pending_vc, dict):
    st.session_state.value_config.update(_pending_vc)
    st.session_state["_g_staking_mode"] = st.session_state.value_config["staking_mode"]
    st.session_state["_g_value_threshold"] = float(st.session_state.value_config["value_threshold"])
    st.session_state["_g_value_odds_range"] = _value_odds_range(st.session_state.value_config)
    st.session_state["_g_kelly_fraction"] = float(st.session_state.value_config["kelly_fraction"])
    st.session_state["_g_bankroll"] = float(st.session_state.value_config["bankroll"])
    st.session_state["_g_ew_enabled"] = bool(st.session_state.value_config["ew_enabled"])
    st.session_state["_g_ew_fraction"] = float(st.session_state.value_config["ew_fraction"])
    st.session_state["_g_ew_min_edge"] = float(st.session_state.value_config["ew_min_place_edge"])
    st.session_state["_g_ew_odds_range"] = (
        float(st.session_state.value_config["ew_min_odds"]),
        float(st.session_state.value_config["ew_max_odds"]),
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
        value=st.session_state.value_config["ew_enabled"],
        key="_g_ew_enabled",
        help="Show each-way value bets alongside win value bets.",
    )
    _ew_frac_options = [0.20, 0.25, 0.33]
    _ew_frac_labels = {0.20: "1/5", 0.25: "1/4", 0.33: "1/3"}
    _g_ew_fraction = st.selectbox(
        "EW odds fraction",
        _ew_frac_options,
        index=_ew_frac_options.index(
            st.session_state.value_config["ew_fraction"]
        ) if st.session_state.value_config["ew_fraction"] in _ew_frac_options else 0,
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
        value=st.session_state.value_config["ew_min_place_edge"],
        step=0.01,
        key="_g_ew_min_edge",
        help="Minimum place_prob \u2212 implied_place_prob edge for EW bets.",
    )
    _ew_odds_range = st.slider(
        "EW odds range",
        min_value=2.0, max_value=101.0,
        value=(
            st.session_state.value_config["ew_min_odds"],
            st.session_state.value_config["ew_max_odds"],
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
    "<small>Built with Streamlit · LightGBM<br>"
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
            "To switch, promote a saved run on the training app’s **Registry** page."
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
        st.session_state.get("_production_revision"),
        feature_store_signature(),
        st.session_state.value_config.get("ew_fraction"),
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

        # The model must have been trained on the feature generation that will
        # now be fed to it. A mismatch yields confident, meaningless prices, so
        # refuse to price the card rather than publish numbers nobody can trust.
        _gen = feature_generation_status()
        if _gen["state"] == GENERATION_MISMATCH:
            st.error(
                f"🚫 **Model / feature mismatch — picks withheld.**\n\n{_gen['message']}"
            )
            st.stop()

        progress = st.progress(10, text="Checking live features …")
        try:
            # Exact card/history cache validation lives in the callable serving
            # core. A miss rebuilds only this card from persisted history state.
            all_feat = features_for_live_cards(cards_df, _picks_date_str)
        except (RuntimeError, ValueError) as exc:
            progress.empty()
            st.error(str(exc))
            st.stop()
        if all_feat is None or all_feat.empty:
            progress.empty()
            st.error("No usable runners were found in the current racecard.")
            st.stop()
        all_feat = all_feat.copy()
        all_feat["race_id"] = all_feat["race_id"].astype(str)
        cards_df["race_id"] = cards_df["race_id"].astype(str)
        st.session_state["picks_feat_source"] = "validated live card features"
        st.session_state.setdefault("picks_featured", {})[_picks_date_str] = all_feat

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
            from src.app_helpers import attach_result_outcomes
            full_preds = attach_result_outcomes(full_preds,_res_df)
            full_preds["result_is_nr"] = full_preds.result_fp_label.str.upper().isin(_NR_LABELS)
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
            _tp_ew_cfg["ew_enabled"]
            and "ew_value" in full_preds.columns
        ):
            ew_df = ew_value_bets(
                full_preds,
                min_place_edge=_tp_ew_cfg["ew_min_place_edge"],
                min_odds=_tp_ew_cfg["ew_min_odds"],
                max_odds=_tp_ew_cfg["ew_max_odds"],
            )
            full_preds["is_ew_value"] = full_preds.index.isin(ew_df.index)
        else:
            full_preds["is_ew_value"] = False
            ew_df = pd.DataFrame()

        # ── data-quality gate: suppress bets on unreliable inputs ──
        # Race-level block (e.g. a thin-history foreign raider mispricing the
        # whole race). Rules + thresholds live in src/quality_control.py and are
        # shown/tunable in the Quality Control panel below — never inlined here.
        from src.quality_control import apply_betting_gate
        _qc_cfg = st.session_state.get("qc_config", dict(config.QC_DEFAULTS))
        _qc = apply_betting_gate(full_preds, ["is_value", "is_ew_value"], _qc_cfg)
        full_preds["qc_blocked"] = _qc["race_blocked"].to_numpy()
        full_preds["qc_flagged"] = _qc["runner_flagged"].to_numpy()
        full_preds["qc_reason"] = _qc["reason"].to_numpy()
        # Re-derive the bet frames now that blocked races are suppressed.
        value_df = full_preds[full_preds["is_value"]].copy() if "is_value" in full_preds else value_df
        if not ew_df.empty:
            ew_df = ew_df[~ew_df.index.isin(full_preds.index[full_preds["qc_blocked"]])].copy()

        # ── headline banner ───────────────────────────────────────
        _any_settled = full_preds["is_settled"].any()
        _n_settled_races = full_preds.loc[full_preds["is_settled"], "race_id"].nunique() if _any_settled else 0
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("🏟️ Races Analysed", len(race_meta))
        h2.metric("🐴 Total Runners", len(full_preds))
        h3.metric("💰 Win Value Bets", len(value_df))
        h4.metric("🔀 EW Value Bets", len(ew_df))

        # ── Quality Control panel ─────────────────────────────────
        # Visible, tunable, auditable. The gate above already suppressed bets in
        # blocked races; this shows what was gated and why, and lets thresholds
        # be adjusted live. Keeps the data-quality rules out of the shadows.
        from src.quality_control import QC_CHECKS
        _qc_sum = _qc["summary"]
        _n_blocked = _qc_sum["n_races_blocked"]
        _n_flagged = _qc_sum["n_runners_flagged"]
        with st.expander(
            f"🛡️ Quality Control — {_n_blocked} race(s) blocked, {_n_flagged} runner(s) flagged",
            expanded=bool(_n_blocked),
        ):
            st.caption(
                "Data-quality gates suppress bets where the model's inputs are unreliable "
                "(distinct from value rules). Defined in `src/quality_control.py`; "
                "thresholds below are live-tunable."
            )
            _cfg_now = dict(st.session_state.get("qc_config", dict(config.QC_DEFAULTS)))
            qc1, qc2, qc3 = st.columns(3)
            _enabled = qc1.checkbox("Gates enabled", value=bool(_cfg_now.get("enabled", True)), key="qc_enabled")
            _max_odds = qc2.number_input(
                "Thin-history contender max SP", min_value=1.5, max_value=50.0, step=0.5,
                value=float(_cfg_now.get("thin_history_contender_max_odds", 8.0)), key="qc_max_odds",
                help="A runner at or below this SP counts as a market contender.",
            )
            _min_prev = qc3.number_input(
                "Thin-history max prior DB runs", min_value=0, max_value=10, step=1,
                value=int(_cfg_now.get("thin_history_min_prev_races", 1)), key="qc_min_prev",
                help="Contenders with this many or fewer prior runs in our DB are flagged.",
            )
            _new_cfg = {
                "enabled": bool(_enabled),
                "thin_history_contender_max_odds": float(_max_odds),
                "thin_history_min_prev_races": int(_min_prev),
            }
            if _new_cfg != _cfg_now:
                st.session_state["qc_config"] = _new_cfg
                st.rerun()
            for _chk in QC_CHECKS:
                _s = _qc_sum["checks"].get(_chk["id"], {})
                st.markdown(f"**{_chk['label']}** — {_s.get('n_runners', 0)} runner(s) flagged. {_chk['description']}")
            if _n_flagged:
                _fcols = [c for c in ["off_time", "track", "horse_name", "odds",
                                      "horse_prev_races", "qc_reason"] if c in full_preds.columns]
                _ftab = full_preds[full_preds["qc_flagged"]][_fcols].copy()
                _sortc = [c for c in ["track", "off_time"] if c in _ftab.columns]
                if _sortc:
                    _ftab = _ftab.sort_values(_sortc)
                st.dataframe(_ftab, hide_index=True, width="stretch")
            else:
                st.success("No data-quality issues detected in today's card.")

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
            custom_df["race_date"] = _picks_date_str
            custom_df["won"] = 0
            custom_df["finish_position"] = 0
            custom_df["finish_time_secs"] = 0
            custom_df["lengths_behind"] = np.nan

            try:
                featured_custom = features_for_live_cards(
                    custom_df, _picks_date_str, use_cache=False,
                )
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
                st.stop()
            predictions = _predict_featured_frame(
                st.session_state.predictor,
                featured_custom,
                ew_fraction=st.session_state.value_config.get("ew_fraction"),
            )

            st.markdown("### 🏆 Custom Race Prediction")
            _render_pace_panel(predictions, key="custom_pace")
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
