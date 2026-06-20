"""Self-contained GPU benchmark: modern tabular DL vs LightGBM (odds-free win model).

Upload TWO things to a GPU box (Colab / Kaggle / cloud) and run this file:
  1. gpu_bundle.parquet   (the data: 484 features + 5 meta cols)
  2. this script

It needs NOTHING from the horse repo. It reproduces the project's evaluation
(time split, per-race normalisation, Platt calibration, log-loss/brier/top-1/
rps vs the market baseline), computes its own LightGBM reference + the
market-probability distillation model, and benchmarks GPU DL models
(RealMLP, TabM, FT-Transformer, optionally TabPFN) that are intractable on CPU.

    pip install pytabkit lightgbm pyarrow scikit-learn pandas   # torch+CUDA preinstalled on Colab
    # optional: pip install tabpfn
    python gpu_tabular_benchmark.py --data gpu_bundle.parquet
    python gpu_tabular_benchmark.py --data gpu_bundle.parquet --frac 0.5 --models realmlp,tabm,lgbm

The bundle columns are exactly: <484 numeric features> + race_id, race_date,
won, odds, finish_position. Features = every column except those 5.

Context (from the CPU project): the form-only LightGBM is at its ceiling
(~0.319 log-loss, ~0.024 behind the market). Market-prob distillation gained
~-0.002 log-loss. The open question this script answers on GPU: do RealMLP /
TabM / FT-Transformer / TabPFN beat LightGBM here, or (per the literature)
merely tie at much higher compute?
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

META = ["race_id", "race_date", "won", "odds", "finish_position"]


# ── metrics ────────────────────────────────────────────────────────────
def _logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p, y):
    return float(np.mean((p - y) ** 2))


def _grouped_normalize(p, groups):
    """Make probabilities sum to 1 within each race (matches the project)."""
    out = np.asarray(p, dtype=float).copy()
    i = 0
    for g in groups:
        s = out[i:i + g].sum()
        if s > 0:
            out[i:i + g] /= s
        i += g
    return out


def _top1_rps(p, y, groups):
    top1 = tot = 0
    rps_acc = []
    i = 0
    for g in groups:
        pg, yg = p[i:i + g], y[i:i + g]
        i += g
        if g < 2 or yg.max() == yg.min():
            continue
        tot += 1
        if np.argmax(pg) == np.argmax(yg):
            top1 += 1
        # ranked-probability score (per race)
        order = np.argsort(-pg)
        cdf_p = np.cumsum(pg[order]); cdf_y = np.cumsum(yg[order])
        rps_acc.append(np.sum((cdf_p - cdf_y) ** 2) / max(g - 1, 1))
    return (top1 / max(tot, 1)), float(np.mean(rps_acc) if rps_acc else np.nan)


# ── data ───────────────────────────────────────────────────────────────
def load_split(path, frac, calib_frac=0.15, test_size=0.2):
    df = pd.read_csv(path) if path.endswith((".csv", ".csv.gz", ".gz")) else pd.read_parquet(path)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df[pd.to_numeric(df["finish_position"], errors="coerce").fillna(0) > 0].copy()
    if frac < 1.0:
        rng = np.random.default_rng(0)
        rids = df["race_id"].unique()
        keep = set(rng.choice(rids, int(len(rids) * frac), replace=False))
        df = df[df["race_id"].isin(rids[np.isin(rids, list(keep))])].copy()
    # chronological, race-contiguous
    df = df.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    feats = [c for c in df.columns if c not in META]

    # market-implied prob (overround-normalised per race) — baseline + soft target
    o = pd.to_numeric(df["odds"], errors="coerce")
    ip = (1.0 / o.where(o > 1)).fillna(0.0).clip(0, 1).to_numpy(float)
    df["_mkt"] = ip / df.assign(_i=ip).groupby("race_id")["_i"].transform("sum").clip(lower=1e-9).to_numpy()

    # race-aligned time split: last test_size as holdout
    sizes = df.groupby("race_id", sort=False).size()
    order = df["race_id"].drop_duplicates().tolist()
    csum = np.cumsum([sizes[r] for r in order])
    split_row = int(len(df) * (1 - test_size))
    split_race = int(np.searchsorted(csum, split_row))
    tr_rows = int(csum[split_race - 1]) if split_race > 0 else 0

    X = df[feats].to_numpy(np.float32)
    y = df["won"].to_numpy(float)
    mkt = df["_mkt"].to_numpy(float)
    g_all = sizes[order].to_numpy()
    n_tr_races = split_race
    Xtr, Xte = X[:tr_rows], X[tr_rows:]
    ytr, yte = y[:tr_rows], y[tr_rows:]
    mkt_tr, mkt_te = mkt[:tr_rows], mkt[tr_rows:]
    g_tr, g_te = g_all[:n_tr_races], g_all[n_tr_races:]

    # calibration slice = last calib_frac of train (race-aligned)
    ccum = np.cumsum(g_tr)
    crow = int(len(Xtr) * (1 - calib_frac))
    crace = int(np.searchsorted(ccum, crow))
    fit_rows = int(ccum[crace - 1]) if crace > 0 else 0
    return dict(
        feats=feats, Xtr=Xtr, ytr=ytr, mkt_tr=mkt_tr, g_tr=g_tr,
        Xte=Xte, yte=yte, mkt_te=mkt_te, g_te=g_te,
        fit_rows=fit_rows, g_cal=g_tr[crace:],
    )


def impute(Xref, *arrays):
    med = np.nanmedian(Xref.astype(np.float64), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    out = []
    for X in arrays:
        X = X.copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.where(bad)[1])
        out.append(np.ascontiguousarray(X))
    return out


# ── calibration + eval ─────────────────────────────────────────────────
def evaluate(name, pc, pt, d, fit_s):
    """pc/pt = raw P(win) on calib/test rows. Platt-calibrate, per-race-norm, score."""
    from sklearn.linear_model import LogisticRegression
    g_cal, g_te = d["g_cal"], d["g_te"]
    yc = d["ytr"][d["fit_rows"]:]
    yte = d["yte"]
    pc = _grouped_normalize(pc, g_cal); pt = _grouped_normalize(pt, g_te)
    lp = np.log(np.clip(pc, 1e-9, 1 - 1e-9) / np.clip(1 - pc, 1e-9, 1)).reshape(-1, 1)
    pl = LogisticRegression(C=1e6, solver="lbfgs").fit(lp, yc)
    lpt = np.log(np.clip(pt, 1e-9, 1 - 1e-9) / np.clip(1 - pt, 1e-9, 1)).reshape(-1, 1)
    pt = _grouped_normalize(pl.predict_proba(lpt)[:, 1], g_te)
    top1, rps = _top1_rps(pt, yte, g_te)
    return dict(name=name, fit_s=round(fit_s, 1), log_loss=_logloss(pt, yte),
                brier=_brier(pt, yte), top1=top1, rps=rps)


def run(args):
    d = load_split(args.data, args.frac)
    n_feat = len(d["feats"])
    print(f"data: train={len(d['Xtr']):,} ({len(d['g_tr'])} races)  "
          f"test={len(d['Xte']):,} ({len(d['g_te'])} races)  feat={n_feat}")
    # market baseline
    mb = dict(name="MARKET (baseline)", fit_s=0.0,
              log_loss=_logloss(d["mkt_te"], d["yte"]), brier=_brier(d["mkt_te"], d["yte"]),
              top1=_top1_rps(d["mkt_te"], d["yte"], d["g_te"])[0],
              rps=_top1_rps(d["mkt_te"], d["yte"], d["g_te"])[1])
    rows = [mb]

    Xf_raw = d["Xtr"][:d["fit_rows"]]
    yf = d["ytr"][:d["fit_rows"]]
    soft_f = d["mkt_tr"][:d["fit_rows"]]            # market-prob soft label (distillation)
    soft_f = np.where(soft_f > 0, soft_f, yf)
    Xc_raw = d["Xtr"][d["fit_rows"]:]
    Xte_raw = d["Xte"]
    Xf, Xc, Xte = impute(Xf_raw, Xf_raw, Xc_raw, Xte_raw)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    DEV = "cuda"
    try:
        import torch
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    print(f"device: {DEV}\n")

    LGB = dict(n_estimators=2000, num_leaves=35, max_depth=6, learning_rate=0.01,
               min_child_samples=50, subsample=0.6, colsample_bytree=0.6,
               reg_alpha=0.5, reg_lambda=3.0, subsample_freq=1, n_jobs=-1, verbose=-1)

    def add(r):
        rows.append(r)
        print(f"  {r['name']:24} fit={r['fit_s']:>6}s | logloss {r['log_loss']:.4f}  "
              f"brier {r['brier']:.4f}  top1 {r['top1']:.4f}  rps {r['rps']:.4f}", flush=True)

    for m in models:
        try:
            t0 = time.time()
            if m == "lgbm":
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(objective="binary", **LGB).fit(Xf, yf.astype(int))
                pc, pt = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xte)[:, 1]
                add(evaluate("LightGBM (hard)", pc, pt, d, time.time() - t0))
            elif m == "lgbm_distill":
                import lightgbm as lgb
                reg = lgb.LGBMRegressor(objective="cross_entropy", **LGB).fit(Xf, np.clip(soft_f, 0, 1))
                pc, pt = reg.predict(Xc), reg.predict(Xte)
                add(evaluate("LightGBM (distill)", pc, pt, d, time.time() - t0))
            elif m == "realmlp":
                from pytabkit import RealMLP_TD_Classifier
                clf = RealMLP_TD_Classifier(device=DEV, random_state=0).fit(Xf, yf.astype(int))
                pc, pt = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xte)[:, 1]
                add(evaluate("RealMLP_TD", pc, pt, d, time.time() - t0))
            elif m == "tabm":
                from pytabkit import TabM_D_Classifier
                clf = TabM_D_Classifier(device=DEV, random_state=0).fit(Xf, yf.astype(int))
                pc, pt = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xte)[:, 1]
                add(evaluate("TabM_D", pc, pt, d, time.time() - t0))
            elif m == "ftt":
                from pytabkit import FTT_D_Classifier
                clf = FTT_D_Classifier(device=DEV, random_state=0).fit(Xf, yf.astype(int))
                pc, pt = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xte)[:, 1]
                add(evaluate("FT-Transformer", pc, pt, d, time.time() - t0))
            elif m == "tabpfn":
                from tabpfn import TabPFNClassifier
                # TabPFN caps ~50k context; subsample the most recent fit rows
                n = min(len(Xf), 50000)
                clf = TabPFNClassifier(device=DEV).fit(Xf[-n:], yf[-n:].astype(int))
                pc, pt = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xte)[:, 1]
                add(evaluate("TabPFN-v2", pc, pt, d, time.time() - t0))
            else:
                print(f"  (unknown model '{m}', skipped)")
        except Exception as e:
            print(f"  {m}: FAILED — {type(e).__name__}: {e}", flush=True)

    print("\n=== ranked by log-loss (lower is better) ===")
    for r in sorted(rows, key=lambda x: x["log_loss"]):
        print(f"  {r['name']:24} logloss {r['log_loss']:.4f}  brier {r['brier']:.4f}  top1 {r['top1']:.4f}")
    print("\nNote: LightGBM (hard) is the form-model baseline; LightGBM (distill) is the "
          "market-prob soft-label model. A DL model is only interesting if it clearly "
          "beats both — per the literature, expect a tie at much higher compute.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gpu_bundle.parquet")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--models", default="lgbm,lgbm_distill,realmlp,tabm,ftt",
                    help="comma list: lgbm,lgbm_distill,realmlp,tabm,ftt,tabpfn")
    run(ap.parse_args())
