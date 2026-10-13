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
from src.featured_dataset import fe_signature, get_featured_dataset
from src.feature_store import FeatureStore
from src.run_store import (
    GENERATION_MATCH,
    GENERATION_MISMATCH,
    GENERATION_UNKNOWN,
    delete_run,
    feature_generation_status,
    get_production_run_id,
    list_runs,
    load_run,
    load_run_meta,
    load_run_predictor,
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


@st.cache_resource(show_spinner="Loading feature store …")
def _load_featured():
    """Load the full featured dataset from the store (heavy — cached for the
    session). Only the analysis pages call this; the picks app never does."""
    return get_featured_dataset()


def _window_days(featured: pd.DataFrame, days: int | None) -> pd.DataFrame:
    """Restrict to the last *days* by race_date (a split, not a rebuild)."""
    if not days:
        return featured
    rd = pd.to_datetime(featured["race_date"], errors="coerce")
    return featured[rd >= rd.max() - pd.Timedelta(days=int(days))].copy()


def _evals_of(meta: dict) -> dict:
    return (meta.get("training_config") or {}).get("evals", {}) or {}


def _promote_with_feedback(run_id: str, *, key: str) -> None:
    """Promote *run_id*, explaining a refusal instead of failing silently.

    ``promote`` blocks a model whose feature generation no longer matches the
    store — serving it would score features it was never trained on. Show the
    reason and require an explicit tick to override.
    """
    status = feature_generation_status(run_id)
    override = False
    if status["state"] == GENERATION_MISMATCH:
        st.error(status["message"])
        override = st.checkbox(
            "Promote anyway (I know these features do not match)", key=f"{key}_override",
        )
    elif status["state"] == GENERATION_UNKNOWN:
        st.warning(status["message"])
        override = st.checkbox(
            "Promote anyway (I accept these features cannot be verified)",
            key=f"{key}_override",
        )

    if st.button("🚀 Promote to production", type="primary", key=key,
                 disabled=not run_has_model(run_id)):
        if promote(run_id, allow_feature_mismatch=override):
            st.success("Promoted. The serving app will load this model.")
            st.rerun()
        else:
            st.error("Promotion refused — see the feature-generation warning above.")


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
        if crows:
            st.dataframe(pd.DataFrame(crows), hide_index=True, use_container_width=True)
        else:
            st.caption("—")
    with rcol:
        st.markdown("**Backtest ROI**")
        rrows = [{"strategy": s, "bets": m.get("bets"), "strike%": m.get("strike_rate"),
                  "roi%": m.get("roi"), "pnl": m.get("pnl")}
                 for s, m in roi.items() if isinstance(m, dict)]
        if rrows:
            st.dataframe(pd.DataFrame(rrows), hide_index=True, use_container_width=True)
        else:
            st.caption("—")
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
    ["📦 Data", "🤖 Train", "🧭 Tune", "🔬 Backtest", "📈 Insights",
     "🩺 Diagnostics", "🗂️ Registry"],
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
        st.success(f"DB synced — {len(df):,} rows total in the DB.")
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
            catchup = st.number_input("Catch-up days", 1, 90, 14,
                                      help="Only used when 'Catch-up scrape first' is ticked.")
        with col2:
            full_hist = st.checkbox("Full history", value=True)
            window = st.number_input("Train window (days)", 90, 3000, 600,
                                     help="Used only when 'Full history' is UNticked "
                                          "(ignored otherwise).")
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
        calibration = st.selectbox(
            "Win probability calibration", ["platt_isotonic", "race_temperature"],
            index=["platt_isotonic", "race_temperature"].index(config.WIN_CALIBRATION_METHOD),
            format_func=lambda value: {"platt_isotonic": "Platt + isotonic (current default)",
                                       "race_temperature": "Race temperature (experimental)"}[value],
            help="Temperature calibration improved the audit's chronological scores; "
                 "confirm on rebuilt features before promoting a model.",
        )
        submitted = st.form_submit_button("🚀 Train", type="primary")

    if submitted:
        cfg = TrainingConfig(
            name=name, refresh_data=refresh, catchup_days=int(catchup),
            train_window_days=None if full_hist else int(window),
            frameworks={"classifier": clf, "place": place},
            blend_market_odds=blend, run_walk_forward=wf,
            win_calibration_method=calibration,
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
        _promote_with_feedback(last, key="promote_after_train")


# ══════════════════════════════════════════════════════════════════════
# TUNE  (Optuna hyperparameter search)
# ══════════════════════════════════════════════════════════════════════
elif page == "🧭 Tune":
    st.title("🧭 Tune")
    st.caption("Optuna search over purged walk-forward folds. Best params are "
               "recorded per session — copy them into a Train run to use them.")

    from src.autotune import (
        create_autotune_session,
        delete_autotune_session,
        list_autotune_sessions,
        run_autotune_session,
    )

    with st.form("tune_cfg"):
        tname = st.text_input("Session name", value="tune")
        c1, c2, c3 = st.columns(3)
        with c1:
            models = st.multiselect("Models", ["classifier", "place"],
                                    default=["classifier"])
        with c2:
            n_trials = st.number_input("Trials per model", 5, 500, 30)
            n_folds = st.number_input("CV folds", 1, 6, 2)
        with c3:
            t_full = st.checkbox("Full history", value=False)
            t_window = st.number_input("Window (days)", 90, 3000, 600,
                                       help="Used only when 'Full history' is UNticked.")
        tf1, tf2 = st.columns(2)
        with tf1:
            t_clf = st.selectbox("Win framework", ["lgbm", "cat", "xgb"], key="t_clf")
        with tf2:
            t_place = st.selectbox("Place framework", ["lgbm", "cat", "xgb"], key="t_place")
        tsub = st.form_submit_button("🔎 Run search", type="primary")

    if tsub:
        if not models:
            st.warning("Pick at least one model to tune.")
        else:
            featured = _window_days(_load_featured(), None if t_full else int(t_window))
            frameworks = {"classifier": t_clf, "place": t_place}
            session = create_autotune_session(
                name=tname, dataset_meta={"rows": int(len(featured)),
                                          "window_days": None if t_full else int(t_window)},
                frameworks=frameworks, models=models,
                n_trials=int(n_trials), n_folds=int(n_folds),
            )
            prog = st.progress(0.0, text="building folds …")

            def _tcb(event: str, payload: dict) -> None:
                if event == "trial":
                    tot = max(payload.get("trial_total", 1), 1)
                    frac = payload.get("trial_num", 0) / tot
                    prog.progress(min(frac, 1.0),
                                  text=f"{payload.get('model_key')}: trial "
                                       f"{payload.get('trial_num')}/{tot} "
                                       f"· best {payload.get('score', float('nan')):.4f}")
                elif event == "model_start":
                    prog.progress(0.0, text=f"tuning {payload.get('model_key')} …")

            try:
                manifest = run_autotune_session(
                    session_id=session["session_id"], featured_df=featured,
                    frameworks=frameworks, models=models,
                    n_trials=int(n_trials), n_folds=int(n_folds),
                    progress_callback=_tcb,
                )
                st.success(f"✅ Search complete — session {manifest['session_id']}")
            except Exception as exc:
                st.exception(exc)

    st.markdown("### Sessions")
    sessions = list_autotune_sessions()
    if not sessions:
        st.info("No tuning sessions yet.")
    else:
        for s in sessions:
            sid = s.get("session_id")
            with st.expander(f"{s.get('name', sid)} · {s.get('status', '?')} "
                             f"· {len(s.get('best_params', {}))} model(s)"):
                bp = s.get("best_params") or {}
                if bp:
                    st.json(bp)
                else:
                    st.caption("No best params recorded.")
                if st.button("🗑️ Delete session", key=f"del_{sid}"):
                    delete_autotune_session(sid)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# BACKTEST  (walk-forward validation)
# ══════════════════════════════════════════════════════════════════════
elif page == "🔬 Backtest":
    st.title("🔬 Backtest")
    st.caption("Purged expanding-window walk-forward. View a saved run's report "
               "or run a fresh backtest over the feature store.")

    tab_run, tab_saved = st.tabs(["▶️ Fresh backtest", "🗂️ From a saved run"])

    with tab_run:
        with st.form("bt_cfg"):
            b1, b2, b3 = st.columns(3)
            with b1:
                min_train = st.number_input("Min train months", 1, 36, 6)
                test_win = st.number_input("Test window months", 1, 6, 1)
            with b2:
                vthr = st.number_input("Value threshold", 0.0, 0.5,
                                       float(DEFAULT_VALUE_CONFIG.get("value_threshold", 0.05)),
                                       step=0.01)
                fast = st.checkbox("Fast fold (cheaper)", value=True)
            with b3:
                b_full = st.checkbox("Full history", value=True, key="bt_full")
                b_window = st.number_input("Window (days)", 180, 3000, 900,
                                           key="bt_window",
                                           help="Used only when 'Full history' is UNticked.")
            bsub = st.form_submit_button("▶️ Run walk-forward", type="primary")

        if bsub:
            from src.backtester import walk_forward_validation
            featured = _window_days(_load_featured(), None if b_full else int(b_window))
            prog = st.progress(0.0, text="starting …")

            def _bcb(msg: str, pct: float = 0.0) -> None:
                prog.progress(min(max(pct, 0.0), 1.0), text=msg)

            try:
                report = walk_forward_validation(
                    featured, min_train_months=int(min_train),
                    test_window_months=int(test_win), value_threshold=float(vthr),
                    fast_fold=bool(fast), progress_callback=_bcb,
                )
                st.session_state["bt_report"] = report
            except Exception as exc:
                st.exception(exc)

        report = st.session_state.get("bt_report")
        if report is not None:
            summary = report.get("summary")
            if summary is not None and not summary.empty:
                st.markdown("### Per-fold summary")
                st.dataframe(summary, use_container_width=True, hide_index=True)
                _pnl_cols = [c for c in ["top_pick_pnl", "value_pnl", "ew_pnl"]
                             if c in summary.columns]
                if _pnl_cols:
                    st.markdown("### Cumulative P&L by strategy")
                    st.bar_chart(summary[_pnl_cols].sum())
            curves = report.get("curves")
            if curves is not None and not curves.empty and "cum_pnl" in curves.columns:
                st.markdown("### Cumulative P&L curve")
                st.line_chart(curves.pivot_table(index="bet_number",
                                                 columns="strategy", values="cum_pnl"))
            bets = report.get("bets")
            if bets is not None and not bets.empty:
                st.download_button("⬇️ Download bets CSV", bets.to_csv(index=False),
                                   file_name="walk_forward_bets.csv", mime="text/csv")

    with tab_saved:
        runs = list_runs()
        wf_runs = [r for r in runs
                   if os.path.isfile(os.path.join(config.DATA_DIR, "runs",
                                                   r["run_id"], "wf_summary.csv"))]
        if not wf_runs:
            st.info("No saved runs include a walk-forward report. Enable "
                    "'Walk-forward validation' on the Train page.")
        else:
            sel = st.selectbox("Run", [r["run_id"] for r in wf_runs],
                               format_func=lambda i: f"{next((x['name'] for x in wf_runs if x['run_id']==i), i)} ({i})")
            data = load_run(sel)
            wf = data.get("wf_summary_df")
            if wf is not None and not wf.empty:
                st.dataframe(wf, use_container_width=True, hide_index=True)
            wfc = data.get("wf_curves_df")
            if wfc is not None and not wfc.empty and "cum_pnl" in wfc.columns:
                st.line_chart(wfc.pivot_table(index="bet_number",
                                              columns="strategy", values="cum_pnl"))


# ══════════════════════════════════════════════════════════════════════
# INSIGHTS  (feature importance / calibration / overfit)
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Insights":
    st.title("📈 Insights")
    st.caption("Model internals for a saved run — feature importance, "
               "calibration and train-vs-test overfit gap.")

    runs = [r for r in list_runs() if run_has_model(r["run_id"])]
    if not runs:
        st.info("No runs with a model snapshot yet.")
        st.stop()

    default_idx = next((i for i, r in enumerate(runs) if r["run_id"] == _prod_id), 0)
    sel = st.selectbox("Run", [r["run_id"] for r in runs], index=default_idx,
                       format_func=lambda i: f"{next((x['name'] for x in runs if x['run_id']==i), i)} ({i})"
                       + (" 🚀" if i == _prod_id else ""))
    meta = load_run_meta(sel)

    st.markdown("### Feature importance (win classifier)")
    top_n = st.slider("Top N", 5, 50, 20)
    try:
        from src.model import get_feature_importance
        pred = load_run_predictor(sel)
        if pred is None or getattr(pred, "clf_model", None) is None:
            st.warning("Could not load a win model for this run.")
        else:
            fi = get_feature_importance(pred.clf_model, pred.feature_cols, top_n=int(top_n))
            st.bar_chart(fi.set_index("feature")["importance"])
            with st.expander("Importance table"):
                st.dataframe(fi, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.exception(exc)

    st.markdown("### Calibration · ROI · CLV")
    _render_evals(_evals_of(meta))

    st.markdown("### Overfit gap (train vs test)")
    train_m = meta.get("train_metrics") or {}
    test_m = meta.get("metrics") or {}
    grows = []
    for model_key in sorted(set(train_m) | set(test_m)):
        tr = train_m.get(model_key) or {}
        te = test_m.get(model_key) or {}
        if isinstance(tr, dict) and isinstance(te, dict) and ("brier_score" in tr or "brier_score" in te):
            tb, eb = tr.get("brier_score"), te.get("brier_score")
            grows.append({
                "model": model_key, "train_brier": tb, "test_brier": eb,
                "gap": (eb - tb) if (tb is not None and eb is not None) else None,
            })
    if grows:
        st.dataframe(pd.DataFrame(grows), use_container_width=True, hide_index=True)
        st.caption("A large positive gap (test ≫ train Brier) signals overfitting.")
    else:
        st.caption("No train/test metrics recorded for this run.")


# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS  (data quality / coverage)
# ══════════════════════════════════════════════════════════════════════
elif page == "🩺 Diagnostics":
    st.title("🩺 Diagnostics")
    st.caption("Feature-store health and source coverage. Low RTV coverage is "
               "expected — RTV only began in 2023 at select tracks.")

    ss = _store_status()
    if not ss.get("exists"):
        st.error("Feature store not built yet.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Featured rows", f"{ss['rows']:,}")
    c2.metric("Through", ss["last_date"])
    c3.metric("FE signature", "✅ current" if ss["fe_match"] else "⚠️ stale")
    st.caption(f"schema v{ss['schema_version']} · built {ss.get('built_at', '?')}")

    st.markdown("### Source coverage")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Race DB**")
        if _db.get("total_runners"):
            st.write(f"- {_db['total_runners']:,} runners / {_db.get('total_races', 0):,} races")
            st.write(f"- {_db.get('earliest_date')} → {_db.get('latest_date')}")
    with d2:
        st.markdown("**Betfair BSP**")
        bs = _bsp_status()
        if bs.get("exists"):
            st.write(f"- {bs['rows']:,} runner rows")
            st.write(f"- {bs['min']} → {bs['max']}")
        else:
            st.write("not built")

    st.markdown("### Column null rates (store sample)")
    n_sample = st.slider("Sample rows", 5_000, 100_000, 20_000, step=5_000)
    if st.button("Scan nulls"):
        with st.spinner("Sampling the store …"):
            df = _load_featured()
            samp = df.tail(int(n_sample))
            nulls = (samp.isna().mean() * 100).sort_values(ascending=False)
            nulls = nulls[nulls > 0].head(40)
        if nulls.empty:
            st.success("No nulls in the sampled columns. 🎉")
        else:
            st.dataframe(
                pd.DataFrame({"column": nulls.index, "null_%": nulls.round(2).values}),
                use_container_width=True, hide_index=True,
            )
            st.caption("RTV-derived columns dominating this list is expected "
                       "(2023+ select tracks).")


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
        _promote_with_feedback(sel, key="promote_registry")
    with a2:
        if st.button("🗑️ Delete", disabled=(sel == _prod_id)):
            delete_run(sel); st.success("Deleted."); st.rerun()
    with a3:
        keep = st.number_input("Keep latest", 1, 50, 10)
        if st.button("Prune older"):
            deleted = prune_runs(keep_latest=int(keep))
            st.success(f"Pruned {len(deleted)} run(s)."); st.rerun()
