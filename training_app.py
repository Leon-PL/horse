"""Training app — a thin Streamlit shell over the callable training core.
=========================================================================

Separated from the serving app (``app.py``) so the live picks process never
imports the heavy training/tuning stack or loads full featured datasets. This
app is the *only* place models are built, tuned, backtested, inspected and
promoted. Every page is a thin wrapper over importable ``src/`` functions
(``src.training``, ``src.run_store``, ``src.feature_store`` …) — no ML logic
lives here.

Run it (own port, alongside the picks app):
    .venv/Scripts/streamlit run training_app.py --server.port 8502
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import config
from src.featured_dataset import fe_signature
from src.feature_store import FeatureStore
from src.run_store import (
    delete_run,
    get_production_run_id,
    list_runs,
    load_run,
    load_run_meta,
    promote,
    prune_runs,
    run_has_model,
)
from src.training import DEFAULT_VALUE_CONFIG, TrainingConfig, train_run

st.set_page_config(page_title="🏇 Training", page_icon="🎓", layout="wide")


# ── small cached readers (cheap, no model/featured loads) ─────────────
@st.cache_data(ttl=30, show_spinner=False)
def _db_stats() -> dict:
    try:
        from src.database import db_stats
        return db_stats()
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=30, show_spinner=False)
def _store_status() -> dict:
    s = FeatureStore()
    if not s.exists():
        return {"exists": False}
    m = s.meta()
    return {
        "exists": True,
        "rows": m.get("featured_rows"),
        "last_date": m.get("last_date"),
        "schema_version": m.get("schema_version"),
        "fe_match": m.get("fe_signature") == fe_signature(),
        "built_at": m.get("built_at") or m.get("appended_at"),
    }


@st.cache_data(ttl=30, show_spinner=False)
def _bsp_status() -> dict:
    path = os.path.join(config.PROCESSED_DATA_DIR, "betfair_bsp.parquet")
    if not os.path.exists(path):
        return {"exists": False}
    import pyarrow.parquet as pq
    md = pq.read_metadata(path)
    # date range is cheap from a single column read
    col = pd.read_parquet(path, columns=["race_date"])["race_date"]
    rd = pd.to_datetime(col, errors="coerce")
    return {"exists": True, "rows": md.num_rows,
            "min": str(rd.min().date()), "max": str(rd.max().date())}


def _evals_of(meta: dict) -> dict:
    return (meta.get("training_config") or {}).get("evals", {}) or {}


def _render_evals(evals: dict) -> None:
    """Render the three headline metric families (calibration / ROI / CLV)."""
    if not evals:
        st.info("No evals recorded for this run (likely a pre-Phase-1 run).")
        return
    cal, roi, clv = evals.get("calibration") or {}, evals.get("roi") or {}, evals.get("clv") or {}

    ccol, rcol, vcol = st.columns(3)
    with ccol:
        st.markdown("**Calibration**")
        crows = []
        for k, m in cal.items():
            if isinstance(m, dict) and "brier_score" in m:
                crows.append({"model": k, "brier": m.get("brier_score"),
                              "logloss": m.get("log_loss"), "ece": m.get("ece"),
                              "ndcg@1": m.get("ndcg_at_1")})
        st.dataframe(pd.DataFrame(crows), hide_index=True, use_container_width=True) \
            if crows else st.caption("—")
    with rcol:
        st.markdown("**Backtest ROI**")
        rrows = [{"strategy": s, "bets": m.get("bets"), "strike%": m.get("strike_rate"),
                  "roi%": m.get("roi"), "pnl": m.get("pnl")}
                 for s, m in roi.items() if isinstance(m, dict)]
        st.dataframe(pd.DataFrame(rrows), hide_index=True, use_container_width=True) \
            if rrows else st.caption("—")
    with vcol:
        st.markdown("**CLV vs BSP**")
        o = clv.get("overall") or {}
        if o:
            st.metric("Mean CLV", f"{o.get('mean_clv', float('nan')):.3f}")
            st.caption(f"beat close {100 * o.get('pct_beat_close', 0):.1f}% · "
                       f"coverage {100 * clv.get('coverage', 0):.0f}% · n={o.get('n')}")
        elif clv.get("bsp_available") is False:
            st.caption("BSP parquet not found.")
        else:
            st.caption("no matched bets")


# ── sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🎓 Training")
page = st.sidebar.radio(
    "Section",
    ["📦 Data", "🤖 Train", "🗂️ Registry"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

_prod_id = get_production_run_id()
if _prod_id:
    try:
        _pm = load_run_meta(_prod_id)
        st.sidebar.success(f"🚀 **Production:** {_pm.get('name', _prod_id)}")
        st.sidebar.caption(f"run {_prod_id}")
    except Exception:
        st.sidebar.info(f"🚀 Production: {_prod_id}")
else:
    st.sidebar.warning("No production model promoted")

_db = _db_stats()
if _db.get("total_runners"):
    st.sidebar.caption(
        f"📅 DB: {_db['total_runners']:,} runners · latest {_db.get('latest_date', '—')}"
    )


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════
if page == "📦 Data":
    st.title("📦 Data")
    st.caption("Feature store + source coverage. The store is built once and "
               "appended; training reads it via the signature-guarded seam.")

    ss = _store_status()
    c1, c2, c3 = st.columns(3)
    if ss.get("exists"):
        c1.metric("Featured rows", f"{ss['rows']:,}")
        c2.metric("Through", ss["last_date"])
        c3.metric("FE signature", "✅ current" if ss["fe_match"] else "⚠️ stale")
        if not ss["fe_match"]:
            st.warning("FE code/config changed since the store was built — the "
                       "next training run will trigger a full rebuild.")
        st.caption(f"schema v{ss['schema_version']} · built {ss.get('built_at', '?')}")
    else:
        st.error("Feature store not built yet. Run a training run (it builds on "
                 "first use) or `scripts/build_feature_store_batched.py`.")

    st.markdown("### Sources")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Race DB**")
        if _db.get("total_runners"):
            st.write(f"- {_db['total_runners']:,} runners / {_db.get('total_races', 0):,} races")
            st.write(f"- {_db.get('earliest_date')} → {_db.get('latest_date')}")
            st.write(f"- {_db.get('db_size_mb', 0):.0f} MB")
        else:
            st.write(_db.get("error", "no data"))
    with d2:
        st.markdown("**Betfair BSP**")
        bs = _bsp_status()
        if bs.get("exists"):
            st.write(f"- {bs['rows']:,} runner rows")
            st.write(f"- {bs['min']} → {bs['max']}")
        else:
            st.write("not built")

    st.markdown("### Catch-up scrape")
    cu = st.number_input("Days to scrape into the DB", 1, 90, 14)
    if st.button("Scrape recent days"):
        with st.spinner(f"Scraping last {cu}d …"):
            from src.database import sync_database
            df = sync_database(days_back=int(cu))
        st.success(f"DB synced — {len(df):,} rows in window.")
        _db_stats.clear()


# ══════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 Train":
    st.title("🤖 Train")
    st.caption("Configure a run → train via the shared `train_run` → review "
               "calibration · backtest ROI · CLV vs BSP → promote.")

    with st.form("train_cfg"):
        name = st.text_input("Run name", value="training run")
        col1, col2, col3 = st.columns(3)
        with col1:
            refresh = st.checkbox("Catch-up scrape first", value=False)
            catchup = st.number_input("Catch-up days", 1, 90, 14, disabled=not refresh)
        with col2:
            full_hist = st.checkbox("Full history", value=True)
            window = st.number_input("Train window (days)", 90, 3000, 600,
                                     disabled=full_hist,
                                     help="Only the last N days when 'Full history' is off.")
        with col3:
            clf = st.selectbox("Win framework", ["lgbm", "cat", "xgb"],
                               index=["lgbm", "cat", "xgb"].index(
                                   config.SUB_MODEL_FRAMEWORKS.get("classifier", "lgbm")))
            place = st.selectbox("Place framework", ["lgbm", "cat", "xgb"],
                                 index=["lgbm", "cat", "xgb"].index(
                                     config.SUB_MODEL_FRAMEWORKS.get("place", "lgbm")))
        colb1, colb2 = st.columns(2)
        with colb1:
            blend = st.checkbox("Blend market odds (Benter anchor)", value=False,
                                help="SP known only at off — anchored metrics look optimistic.")
        with colb2:
            wf = st.checkbox("Walk-forward validation", value=False,
                             help="Purged expanding-window backtest after training (slower).")
        submitted = st.form_submit_button("🚀 Train", type="primary")

    if submitted:
        cfg = TrainingConfig(
            name=name, refresh_data=refresh, catchup_days=int(catchup),
            train_window_days=None if full_hist else int(window),
            frameworks={"classifier": clf, "place": place},
            blend_market_odds=blend, run_walk_forward=wf,
            value_config=dict(DEFAULT_VALUE_CONFIG),
        )
        prog = st.progress(0.0, text="starting …")

        def _cb(msg: str, pct: float) -> None:
            prog.progress(min(max(pct, 0.0), 1.0), text=msg)

        try:
            run_id = train_run(cfg, progress_cb=_cb)
            st.session_state["last_run_id"] = run_id
            _db_stats.clear()
            st.success(f"✅ Trained — run **{run_id}**")
        except Exception as exc:
            st.exception(exc)

    # ── results of the last run this session ──
    last = st.session_state.get("last_run_id")
    if last:
        meta = load_run_meta(last)
        evals = _evals_of(meta)
        st.markdown(f"### Results — {meta.get('name')} ({last})")
        _render_evals(evals)
        if st.button("🚀 Promote to production", type="primary"):
            if promote(last):
                st.success("Promoted. The serving app will load this model.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════
elif page == "🗂️ Registry":
    st.title("🗂️ Registry")
    runs = list_runs()
    if not runs:
        st.info("No runs yet — train one on the Train page.")
        st.stop()

    rows = []
    for r in runs:
        ev = _evals_of(r)
        cal = (ev.get("calibration") or {}).get("win_classifier", {}) or {}
        roi = (ev.get("roi") or {}).get("value", {}) or {}
        clv = (ev.get("clv") or {}).get("overall", {}) or {}
        rows.append({
            "run_id": r["run_id"],
            "name": r.get("name", ""),
            "date": (r.get("timestamp", "") or "")[:10],
            "rows": r.get("data_rows"),
            "brier": cal.get("brier_score"),
            "value_roi%": roi.get("roi"),
            "mean_clv": clv.get("mean_clv"),
            "model": "✅" if run_has_model(r["run_id"]) else "",
            "prod": "🚀" if r["run_id"] == _prod_id else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Inspect / manage")
    ids = [r["run_id"] for r in runs]
    sel = st.selectbox("Run", ids,
                       format_func=lambda i: f"{next((x['name'] for x in runs if x['run_id']==i), i)} ({i})")
    meta = load_run_meta(sel)
    _render_evals(_evals_of(meta))
    with st.expander("Full config / metadata"):
        st.json(meta.get("training_config", {}))

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("🚀 Promote", disabled=not run_has_model(sel)):
            if promote(sel):
                st.success("Promoted."); st.rerun()
    with a2:
        if st.button("🗑️ Delete", disabled=(sel == _prod_id)):
            delete_run(sel); st.success("Deleted."); st.rerun()
    with a3:
        keep = st.number_input("Keep latest", 1, 50, 10)
        if st.button("Prune older"):
            deleted = prune_runs(keep_latest=int(keep))
            st.success(f"Pruned {len(deleted)} run(s)."); st.rerun()
