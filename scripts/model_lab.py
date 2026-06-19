"""Fast train+eval harness for improving the (odds-free) win model.

Reuses the real split (prepare_multi_target_data) and per-race-normalised
calibration so log-loss / brier / rps are comparable to the app, but does a
single fit (no 3-fold OOF, no walk-forward) so each experiment runs in
~1 min instead of several. Use it to compare ideas by their delta vs the
market baseline before committing anything to the pipeline.

First run builds a slim parquet (model feature cols + meta) from a run's
featured_races.parquet so subsequent loads are fast.

Usage:
    python scripts/model_lab.py --build-slim <run_id>
    python scripts/model_lab.py --exp baseline
    python scripts/model_lab.py --exp baseline --frac 0.5     # subsample races
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import config  # noqa: E402
from src.model import (  # noqa: E402
    prepare_multi_target_data, get_feature_columns, rps_per_race,
    normalise_implied_prob_by_race, _grouped_normalize,
)
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

SLIM = os.path.join(config.PROCESSED_DATA_DIR, "model_lab_slim.parquet")
SLIM_EXT = os.path.join(config.PROCESSED_DATA_DIR, "model_lab_slim_ext.parquet")
_RAW_INPUTS = ["sire", "dam", "damsire", "going", "surface", "race_type",
               "race_class", "dist_category", "trainer", "jockey",
               "distance_furlongs", "age"]

BASE_PARAMS = dict(
    n_estimators=2000, num_leaves=35, max_depth=6, learning_rate=0.01,
    min_child_samples=50, subsample=0.6, colsample_bytree=0.6,
    reg_alpha=0.5, reg_lambda=3.0, objective="binary",
    subsample_freq=1, n_jobs=-1, verbose=-1,
)


def build_slim(run_id: str) -> None:
    import joblib
    fp = os.path.join(config.DATA_DIR, "runs", run_id, "featured_races.parquet")
    feat_cols = joblib.load(os.path.join(config.DATA_DIR, "runs", run_id, "model.joblib"))["feature_cols"]
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(fp).schema.names)
    meta = [c for c in ["race_id", "race_date", "horse_name", "finish_position",
                        "won", "num_runners", "odds", "handicap", "horse_prev_races",
                        "lengths_behind"] if c in avail]
    cols = sorted(set(feat_cols) | set(meta))
    cols = [c for c in cols if c in avail]
    df = pd.read_parquet(fp, columns=cols, engine="pyarrow")
    df.to_parquet(SLIM, index=False, engine="pyarrow")
    print(f"slim built: {len(df):,} rows × {len(cols)} cols → {SLIM}")


def build_slim_ext(run_id: str) -> None:
    import joblib, pyarrow.parquet as pq
    fp = os.path.join(config.DATA_DIR, "runs", run_id, "featured_races.parquet")
    feat_cols = joblib.load(os.path.join(config.DATA_DIR, "runs", run_id, "model.joblib"))["feature_cols"]
    avail = set(pq.ParquetFile(fp).schema.names)
    meta = ["race_id", "race_date", "horse_name", "finish_position", "won",
            "num_runners", "odds", "handicap", "horse_prev_races", "lengths_behind"]
    cols = sorted(set(feat_cols) | set(meta) | set(_RAW_INPUTS))
    cols = [c for c in cols if c in avail]
    pd.read_parquet(fp, columns=cols, engine="pyarrow").to_parquet(SLIM_EXT, index=False, engine="pyarrow")
    print(f"ext slim: {len(cols)} cols → {SLIM_EXT}")


def add_candidate_features(df: pd.DataFrame) -> list[str]:
    """Compute leak-safe candidate features in place; return new col names.

    Reuses the production race-safe helpers, so semantics match feature_engineer.
    df must be sorted chronologically (race_date, race_id) before calling.
    """
    from src.feature_engineer import _race_safe_cumsum, _race_safe_cumcount, _bayesian_shrink
    fp = pd.to_numeric(df["finish_position"], errors="coerce")
    df["_won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0.0)
    df["_placed"] = fp.isin([1, 2, 3]).astype(float)
    new = []

    # Sire / damsire aptitude by distance category and going (helps newcomers)
    if "dist_category" in df.columns:
        dcs = df["dist_category"].astype(str)
        for ent in ["sire", "damsire"]:
            if ent not in df.columns:
                continue
            key = f"_{ent}dc"
            df[key] = df[ent].astype(str) + "||" + dcs
            runs = _race_safe_cumcount(df, key)
            wins = _race_safe_cumsum(df, key, "_won")
            plc = _race_safe_cumsum(df, key, "_placed")
            df[f"{ent}_dc_wr_shrunk"] = _bayesian_shrink(
                np.where(runs > 0, wins / np.maximum(runs, 1), 0.0), runs, 0.10, 20.0)
            df[f"{ent}_dc_pr_shrunk"] = _bayesian_shrink(
                np.where(runs > 0, plc / np.maximum(runs, 1), 0.0), runs, 0.30, 20.0)
            df[f"{ent}_dc_runs"] = runs
            new += [f"{ent}_dc_wr_shrunk", f"{ent}_dc_pr_shrunk", f"{ent}_dc_runs"]
            df.drop(columns=[key], inplace=True)
    if "going" in df.columns:
        for ent in ["sire"]:
            key = f"_{ent}g"
            df[key] = df[ent].astype(str) + "||" + df["going"].astype(str)
            runs = _race_safe_cumcount(df, key)
            wins = _race_safe_cumsum(df, key, "_won")
            df[f"{ent}_going_wr_shrunk"] = _bayesian_shrink(
                np.where(runs > 0, wins / np.maximum(runs, 1), 0.0), runs, 0.10, 20.0)
            new += [f"{ent}_going_wr_shrunk"]
            df.drop(columns=[key], inplace=True)

    # Trainer first-time-out (debut) strike rate (helps debut runners)
    if "horse_prev_races" in df.columns and "trainer" in df.columns:
        deb = (pd.to_numeric(df["horse_prev_races"], errors="coerce").fillna(99) == 0).astype(float)
        df["_won_deb"] = df["_won"] * deb
        df["_run_deb"] = deb
        druns = _race_safe_cumsum(df, "trainer", "_run_deb")
        dwins = _race_safe_cumsum(df, "trainer", "_won_deb")
        df["trainer_fto_sr_shrunk"] = _bayesian_shrink(
            np.where(druns > 0, dwins / np.maximum(druns, 1), 0.0), druns, 0.08, 15.0)
        df["is_debut"] = deb
        new += ["trainer_fto_sr_shrunk", "is_debut"]
        df.drop(columns=["_won_deb", "_run_deb"], inplace=True)

    df.drop(columns=["_won", "_placed"], inplace=True)
    return new


def _logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def get_data(frac: float):
    df = pd.read_parquet(SLIM, engine="pyarrow")
    if frac < 1.0:
        rng = np.random.default_rng(0)
        rids = df["race_id"].unique()
        keep = set(rng.choice(rids, size=int(len(rids) * frac), replace=False))
        df = df[df["race_id"].isin(keep)].copy()
    return prepare_multi_target_data(df)


def fit_raw(data, params, calib_frac=0.15, framework="lgbm", recency_hl=None):
    """Fit once; return raw (per-race-normalised) probs on calib + test."""
    Xtr, ytr = data["X_train"], data["y_train_won"]
    gtr = data["groups_train"]
    cut = int(len(Xtr) * (1 - calib_frac))
    cum = np.cumsum(gtr)
    race_cut = int(np.searchsorted(cum, cut))
    row_cut = int(cum[race_cut - 1]) if race_cut > 0 else 0
    g_cal = gtr[race_cut:]
    Xf, yf = Xtr[:row_cut], ytr[:row_cut]
    Xc, yc = Xtr[row_cut:], ytr[row_cut:]
    sw = None
    if recency_hl:
        from src.model import compute_recency_sample_weights
        sw = compute_recency_sample_weights(data["train_race_dates"][:row_cut], half_life_days=recency_hl)
    t0 = time.time()
    if framework == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(
            iterations=params.get("n_estimators", 2000), depth=params.get("max_depth", 6),
            learning_rate=max(params.get("learning_rate", 0.03), 0.03),
            l2_leaf_reg=params.get("reg_lambda", 3.0), verbose=0, allow_writing_files=False)
        m.fit(Xf, yf, sample_weight=sw)
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params)
        m.fit(Xf, yf, sample_weight=sw)
    fit_s = time.time() - t0
    pc = _grouped_normalize(m.predict_proba(Xc)[:, 1], g_cal)
    pt = _grouped_normalize(m.predict_proba(data["X_test"])[:, 1], data["groups_test"])
    return dict(pc=pc, yc=yc.astype(float), g_cal=g_cal, pt=pt,
                yte=data["y_test_won"].astype(float), gte=data["groups_test"],
                test_df=data["test_df"], fit_s=round(fit_s, 1), n_feat=Xtr.shape[1])


def _temp(p, t):
    if t == 1.0:
        return p
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    return 1.0 / (1.0 + np.exp(-lp / t))


def eval_calib(raw, *, calib="platt_iso", temp=1.0, label=""):
    pc = _grouped_normalize(_temp(raw["pc"], temp), raw["g_cal"])
    pt = _grouped_normalize(_temp(raw["pt"], temp), raw["gte"])
    yc, yte, gte = raw["yc"], raw["yte"], raw["gte"]
    if calib in ("platt", "platt_iso"):
        lp = np.log(np.clip(pc, 1e-9, 1 - 1e-9) / np.clip(1 - pc, 1e-9, 1)).reshape(-1, 1)
        pl = LogisticRegression(C=1e6, solver="lbfgs").fit(lp, yc)
        def _p(p):
            x = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1)).reshape(-1, 1)
            return pl.predict_proba(x)[:, 1]
        pc = _grouped_normalize(_p(pc), raw["g_cal"])
        pt = _grouped_normalize(_p(pt), gte)
    if calib in ("iso", "platt_iso"):
        iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yc)
        pt = _grouped_normalize(np.clip(iso.predict(pt), 1e-9, 1.0), gte)
    res = {"label": label, "n_test": len(yte), "n_feat": raw["n_feat"], "fit_s": raw["fit_s"],
           "log_loss": _logloss(pt, yte), "brier": _brier(pt, yte),
           "ece": _ece(pt, yte), "rps": rps_per_race(pt, yte, gte)}
    if "odds" in raw["test_df"].columns:
        mkt = normalise_implied_prob_by_race(raw["test_df"])
        res.update({"mkt_log_loss": _logloss(mkt, yte), "mkt_brier": _brier(mkt, yte),
                    "gap_log_loss": _logloss(pt, yte) - _logloss(mkt, yte)})
    return res


def fit_eval(data, params, *, calib="platt_iso", temp=1.0, calib_frac=0.15, label=""):
    raw = fit_raw(data, params, calib_frac)
    return eval_calib(raw, calib=calib, temp=temp, label=label), raw


def _print(res):
    base = (f"{res['label']:22} n={res['n_test']:>6} feat={res['n_feat']:>4} fit={res['fit_s']:>5}s | "
            f"logloss {res['log_loss']:.4f}  brier {res['brier']:.4f}  ece {res['ece']:.4f}  rps {res['rps']:.4f}")
    if "gap_log_loss" in res:
        base += f"  | vs mkt {res['mkt_log_loss']:.4f} -> gap {res['gap_log_loss']:+.4f}"
    print(base, flush=True)


# ── experiment registry ───────────────────────────────────────────────
def exp_baseline(data):
    r, _ = fit_eval(data, BASE_PARAMS, calib="platt_iso", label="baseline")
    _print(r)


def exp_calib(data):
    """One fit; sweep calibration + temperature (overconfidence test)."""
    raw = fit_raw(data, BASE_PARAMS)
    print(f"(single fit {raw['fit_s']}s, sweeping calibration)")
    for calib in ["none", "platt", "iso", "platt_iso"]:
        _print(eval_calib(raw, calib=calib, label=f"calib={calib}"))
    for t in [1.25, 1.5, 2.0, 3.0]:
        _print(eval_calib(raw, calib="platt_iso", temp=t, label=f"platt_iso+temp{t}"))


def exp_reg(data):
    """Fewer leaves / more regularisation → less overconfident trees."""
    _print(fit_eval(data, BASE_PARAMS, label="baseline")[0])
    grids = {
        "leaves15": dict(num_leaves=15),
        "leaves15_mc100": dict(num_leaves=15, min_child_samples=100),
        "depth4": dict(max_depth=4, num_leaves=15),
        "lambda10": dict(reg_lambda=10.0, reg_alpha=1.0),
        "cs0.4": dict(colsample_bytree=0.4, subsample=0.5),
        "tight": dict(num_leaves=15, min_child_samples=150, reg_lambda=10.0, colsample_bytree=0.4),
    }
    for name, ov in grids.items():
        p = {**BASE_PARAMS, **ov}
        _print(fit_eval(data, p, label=name)[0])


def exp_capacity(data):
    """More trees at low lr (with the existing reg)."""
    for ne in [2000, 3500, 5000]:
        _print(fit_eval(data, {**BASE_PARAMS, "n_estimators": ne}, label=f"trees{ne}")[0])
    _print(fit_eval(data, {**BASE_PARAMS, "n_estimators": 4000, "learning_rate": 0.005}, label="lr0.005_4000")[0])


def exp_feat(frac):
    """Baseline vs +candidate features (uses platt calib = best from sweep)."""
    from src.feature_engineer import _event_sort_key
    df = pd.read_parquet(SLIM_EXT, engine="pyarrow")
    if frac < 1.0:
        rng = np.random.default_rng(0)
        rids = df["race_id"].unique()
        keep = set(rng.choice(rids, size=int(len(rids) * frac), replace=False))
        df = df[df["race_id"].isin(keep)].copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["_ev"] = _event_sort_key(df)
    df = df.sort_values(["_ev", "race_id"]).drop(columns=["_ev"]).reset_index(drop=True)
    new = add_candidate_features(df)
    print(f"added {len(new)} candidate features: {new}")
    base = df.drop(columns=new)
    print("data prepared; fitting baseline then +features (platt calib)…")
    r0, _ = fit_eval(prepare_multi_target_data(base), BASE_PARAMS, calib="platt", label="baseline(platt)")
    _print(r0)
    r1, _ = fit_eval(prepare_multi_target_data(df), BASE_PARAMS, calib="platt", label="+candidate_feats")
    _print(r1)
    print(f"\nΔ log-loss {r1['log_loss']-r0['log_loss']:+.4f}  Δ brier {r1['brier']-r0['brier']:+.4f}  Δ rps {r1['rps']-r0['rps']:+.4f}")


def exp_recency(data):
    _print(eval_calib(fit_raw(data, BASE_PARAMS), calib="platt", label="baseline(platt)"))
    for hl in [365, 180, 90]:
        raw = fit_raw(data, BASE_PARAMS, recency_hl=hl)
        _print(eval_calib(raw, calib="platt", label=f"recency_hl{hl}"))


def exp_ensemble(data):
    """Average K LGBM fits with different seeds (variance reduction)."""
    import lightgbm as lgb
    Xtr, ytr = data["X_train"], data["y_train_won"]
    gtr = data["groups_train"]
    cut = int(len(Xtr) * 0.85)
    cum = np.cumsum(gtr); rc = int(np.searchsorted(cum, cut))
    row = int(cum[rc - 1]) if rc > 0 else 0
    g_cal = gtr[rc:]
    Xf, yf, Xc = Xtr[:row], ytr[:row], Xtr[row:]
    Xte = data["X_test"]
    pc_acc = np.zeros(len(Xc)); pt_acc = np.zeros(len(Xte))
    K = 4
    for s in range(K):
        m = lgb.LGBMClassifier(**{**BASE_PARAMS, "random_state": s, "bagging_seed": s, "feature_fraction_seed": s})
        m.fit(Xf, yf)
        pc_acc += m.predict_proba(Xc)[:, 1]; pt_acc += m.predict_proba(Xte)[:, 1]
        raw = dict(pc=_grouped_normalize(pc_acc / (s + 1), g_cal), yc=ytr[row:].astype(float),
                   g_cal=g_cal, pt=_grouped_normalize(pt_acc / (s + 1), data["groups_test"]),
                   yte=data["y_test_won"].astype(float), gte=data["groups_test"],
                   test_df=data["test_df"], fit_s=0.0, n_feat=Xtr.shape[1])
        _print(eval_calib(raw, calib="platt", label=f"ensemble_{s+1}seed"))


def exp_distill(data):
    """Distillation: train on SP-implied-prob (soft labels) vs hard 0/1 `won`.

    SP is used only as a TRAINING label; inference is odds-free (features
    exclude odds). Tests whether the market's smooth probabilities are a
    better teacher than noisy binary outcomes. Eval is on actual `won`.
    """
    import lightgbm as lgb
    Xtr = data["X_train"]; gtr = data["groups_train"]
    cut = int(len(Xtr) * 0.85)
    cum = np.cumsum(gtr); rc = int(np.searchsorted(cum, cut))
    row = int(cum[rc - 1]) if rc > 0 else 0
    g_cal = gtr[rc:]
    Xf, Xc, Xte = Xtr[:row], Xtr[row:], data["X_test"]
    yhard = data["y_train_won"].astype(float)
    soft = normalise_implied_prob_by_race(data["train_df"]).astype(float)  # SP-implied, per-race
    soft = np.clip(soft, 1e-6, 1 - 1e-6)

    # Hard-label baseline (classifier) for reference
    _print(eval_calib(fit_raw(data, BASE_PARAMS), calib="platt", label="hard_label(baseline)"))

    p = {k: v for k, v in BASE_PARAMS.items() if k not in ("objective",)}
    p["objective"] = "cross_entropy"
    targets = {"distill_soft(SP)": soft, "distill_blend50": 0.5 * yhard + 0.5 * soft}
    for name, tgt in targets.items():
        m = lgb.train(p, lgb.Dataset(Xf, label=tgt[:row]), num_boost_round=BASE_PARAMS["n_estimators"])
        raw = dict(pc=_grouped_normalize(np.clip(m.predict(Xc), 1e-6, 1 - 1e-6), g_cal),
                   yc=yhard[row:], g_cal=g_cal,
                   pt=_grouped_normalize(np.clip(m.predict(Xte), 1e-6, 1 - 1e-6), data["groups_test"]),
                   yte=data["y_test_won"].astype(float), gte=data["groups_test"],
                   test_df=data["test_df"], fit_s=0.0, n_feat=Xtr.shape[1])
        _print(eval_calib(raw, calib="none", label=f"{name}[raw]"))
        _print(eval_calib(raw, calib="platt", label=f"{name}[platt]"))


def exp_dl(data):
    """Benchmark a deep-learning tabular model (pytabkit RealMLP / TabM) vs LightGBM.

    DL models need clean numeric input, so NaNs are median-imputed (fit on the
    train slice). Same per-race-normalise + Platt calibration + eval as the
    LGBM path, so numbers are directly comparable.
    """
    import time
    Xtr = data["X_train"]; gtr = data["groups_train"]
    cut = int(len(Xtr) * 0.85); cum = np.cumsum(gtr); rc = int(np.searchsorted(cum, cut))
    row = int(cum[rc - 1]) if rc > 0 else 0
    g_cal = gtr[rc:]
    yf = np.asarray(data["y_train_won"][:row]).astype(int)

    # median impute (fit on train-fit slice), then standardize-friendly float32
    med = np.nanmedian(Xtr[:row].astype(np.float64), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def _imp(X):
        X = X.astype(np.float32).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1]).astype(np.float32)
        return np.ascontiguousarray(X)
    Xf, Xc, Xte = _imp(Xtr[:row]), _imp(Xtr[row:]), _imp(data["X_test"])

    # LightGBM reference (same harness, full features)
    _print(eval_calib(fit_raw(data, BASE_PARAMS), calib="platt", label="lgbm(ref)"))

    # Optionally feed the DL models only the top-K features by LGBM importance
    # (485-feature MLPs are intractable on CPU); LightGBM keeps all features.
    topk = int(os.environ.get("MODEL_LAB_TOPK", "0"))
    if topk > 0 and topk < Xf.shape[1]:
        import lightgbm as lgb
        _imp_m = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05,
                                    n_jobs=-1, verbose=-1).fit(Xtr[:row], yf)
        keep = np.argsort(_imp_m.feature_importances_)[::-1][:topk]
        Xf, Xc, Xte = Xf[:, keep], Xc[:, keep], Xte[:, keep]
        print(f"DL feature-reduced to top-{topk} by LGBM importance", flush=True)

    models = []
    try:
        from pytabkit import RealMLP_TD_S_Classifier
        models.append(("RealMLP_TD_S", RealMLP_TD_S_Classifier))
    except Exception as e:
        print("RealMLP import failed:", e)
    try:
        from pytabkit import TabM_D_Classifier
        models.append(("TabM_D", TabM_D_Classifier))
    except Exception as e:
        print("TabM import failed:", e)

    for name, Cls in models:
        try:
            m = Cls(random_state=0)
        except TypeError:
            m = Cls()
        t0 = time.time()
        m.fit(Xf, yf)
        fit_s = round(time.time() - t0, 1)
        raw = dict(pc=_grouped_normalize(m.predict_proba(Xc)[:, 1], g_cal),
                   yc=np.asarray(data["y_train_won"][row:]).astype(float), g_cal=g_cal,
                   pt=_grouped_normalize(m.predict_proba(Xte)[:, 1], data["groups_test"]),
                   yte=data["y_test_won"].astype(float), gte=data["groups_test"],
                   test_df=data["test_df"], fit_s=fit_s, n_feat=Xtr.shape[1])
        _print(eval_calib(raw, calib="none", label=f"{name}[raw]"))
        _print(eval_calib(raw, calib="platt", label=f"{name}[platt]"))


def exp_mlp(data):
    """Fast, controllable DL baseline: sklearn MLP with early stopping.

    RealMLP/TabM (pytabkit) are intractable on CPU here; this gives an actual
    neural-net number vs LightGBM. Weaker than modern tabular DL, but bounded.
    """
    import time
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    Xtr = data["X_train"]; gtr = data["groups_train"]
    cut = int(len(Xtr) * 0.85); cum = np.cumsum(gtr); rc = int(np.searchsorted(cum, cut))
    row = int(cum[rc - 1]) if rc > 0 else 0
    g_cal = gtr[rc:]
    yf = np.asarray(data["y_train_won"][:row]).astype(int)
    med = np.nanmedian(Xtr[:row].astype(np.float64), axis=0); med = np.where(np.isfinite(med), med, 0.0)
    def _imp(X):
        X = X.astype(np.float64).copy(); bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1])
        return X
    sc = StandardScaler().fit(_imp(Xtr[:row]))
    Xf = sc.transform(_imp(Xtr[:row])); Xc = sc.transform(_imp(Xtr[row:])); Xte = sc.transform(_imp(data["X_test"]))

    _print(eval_calib(fit_raw(data, BASE_PARAMS), calib="platt", label="lgbm(ref)"))
    for hl in [(256, 128), (512, 256, 128)]:
        t0 = time.time()
        m = MLPClassifier(hidden_layer_sizes=hl, alpha=1e-3, batch_size=512,
                          early_stopping=True, n_iter_no_change=8, max_iter=200, random_state=0)
        m.fit(Xf, yf); fit_s = round(time.time() - t0, 1)
        raw = dict(pc=_grouped_normalize(m.predict_proba(Xc)[:, 1], g_cal),
                   yc=np.asarray(data["y_train_won"][row:]).astype(float), g_cal=g_cal,
                   pt=_grouped_normalize(m.predict_proba(Xte)[:, 1], data["groups_test"]),
                   yte=data["y_test_won"].astype(float), gte=data["groups_test"],
                   test_df=data["test_df"], fit_s=fit_s, n_feat=Xf.shape[1])
        _print(eval_calib(raw, calib="platt", label=f"MLP{hl}[platt]"))


def exp_framework(data):
    _print(eval_calib(fit_raw(data, BASE_PARAMS, framework="lgbm"), calib="platt", label="lgbm(platt)"))
    _print(eval_calib(fit_raw(data, BASE_PARAMS, framework="cat"), calib="platt", label="catboost(platt)"))


EXPERIMENTS = {
    "baseline": exp_baseline,
    "calib": exp_calib,
    "reg": exp_reg,
    "capacity": exp_capacity,
    "recency": exp_recency,
    "framework": exp_framework,
    "ensemble": exp_ensemble,
    "distill": exp_distill,
    "dl": exp_dl,
    "mlp": exp_mlp,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-slim", metavar="RUN_ID")
    ap.add_argument("--build-slim-ext", metavar="RUN_ID")
    ap.add_argument("--exp", default="baseline")
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()
    if args.build_slim:
        build_slim(args.build_slim)
        return
    if args.build_slim_ext:
        build_slim_ext(args.build_slim_ext)
        return
    if args.exp == "feat":
        exp_feat(args.frac)
        return
    if not os.path.exists(SLIM):
        raise SystemExit("No slim parquet. Run --build-slim <run_id> first.")
    data = get_data(args.frac)
    print(f"data: train={len(data['X_train']):,} test={len(data['X_test']):,} feat={data['X_train'].shape[1]}")
    EXPERIMENTS[args.exp](data)


if __name__ == "__main__":
    main()
