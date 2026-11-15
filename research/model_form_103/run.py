"""Elastic net and LightGBM linear trees against the screen LightGBM on the admission folds (research 103).

    python -m research.model_form_103.run elastic --threads 4
    python -m research.model_form_103.run linear-tree --threads 4
    python -m research.model_form_103.run compare

Same 87 inputs (feature_sets/win_selected_20261112.txt), same matrix, same expanding purged folds 2025Q3-2026Q2
(11,632 races) as ``src.modelling admit``. The LightGBM reference is the admission baseline already scored by
research 100 (screen capacity: 400 trees at 0.04), so every arm is compared race by race.

Elastic net: a race-wise conditional logit (softmax over the field, winner likelihood) on the inputs winsorised at
0.5/99.5%, median-filled with missing indicators and standardised on each fold's training rows. Penalty
lambda * (a * |w|_1 + (1 - a) / 2 * |w|^2), solved exactly with L-BFGS-B on split positive and negative weights.
lambda and a are chosen once, on the first fold's training rows (last 15% of its dates held out, purged), then fixed.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.modelling.selection import _evaluate, _split_folds
from src.modelling.spec import ModelSpec, read_features
from src.forecast_evaluation import paired_interval

MATRIX = Path("data/research/irish_class_100/matrix")
FEATURES = Path("feature_sets/win_selected_20261112.txt")
REFERENCE = Path("data/research/irish_class_100/admit/admission_baseline.parquet")
OUT = Path("data/research/model_form_103")
START = pd.Timestamp("2025-07-08")


def load():
    description = json.loads((MATRIX / "prepared.json").read_text())
    context = pd.read_parquet(MATRIX / "context.parquet")
    matrix = np.load(MATRIX / "matrix.npy", mmap_mode="r")
    names = description["features"]
    features = list(read_features(FEATURES))
    spec = ModelSpec("generalist", tuple(features))
    _, folds = _split_folds(context, spec, None, START, 8)
    return description, context, matrix, names, features, spec, folds


# ---------------------------------------------------------------- elastic net

class Prep:
    """Winsorise, median-fill with missing flags, standardise; fitted on training rows only."""

    def __init__(self, x):
        self.lo, self.hi = np.nanpercentile(x, .5, axis=0), np.nanpercentile(x, 99.5, axis=0)
        x = np.clip(x, self.lo, self.hi)
        self.med = np.nan_to_num(np.nanmedian(x, axis=0))
        self.flag = np.flatnonzero(np.isnan(x).mean(axis=0) > .01)
        z = self._fill(x)
        self.mu, self.sd = z.mean(axis=0), z.std(axis=0) + 1e-9

    def _fill(self, x):
        miss = np.isnan(x[:, self.flag]).astype(np.float64)
        return np.hstack([np.where(np.isnan(x), self.med, x), miss])

    def __call__(self, x):
        return (self._fill(np.clip(x, self.lo, self.hi)) - self.mu) / self.sd


def race_starts(race_id):
    r = np.asarray(race_id)
    return np.flatnonzero(np.r_[True, r[1:] != r[:-1]])


def fit_clogit(x, won, starts, lam, alpha):
    n, k = x.shape
    size = np.diff(np.r_[starts, n])
    total = np.add.reduceat(won, starts)
    has = total > 0
    y = won / np.repeat(np.maximum(total, 1e-12), size)
    keep = np.repeat(has, size).astype(np.float64)
    races = has.sum()

    def f(theta):
        w = theta[:k] - theta[k:]
        s = x @ w
        m = np.maximum.reduceat(s, starts)
        e = np.exp(s - np.repeat(m, size))
        z = np.add.reduceat(e, starts)
        p = e / np.repeat(z, size)
        ll = -(np.add.reduceat(y * s, starts) - m - np.log(z))[has].sum() / races
        g = x.T @ ((p - y) * keep) / races
        gw = g + lam * (1 - alpha) * w
        value = ll + lam * (alpha * theta.sum() + (1 - alpha) / 2 * (w @ w))
        return value, np.r_[gw + lam * alpha, -gw + lam * alpha]

    res = minimize(f, np.zeros(2 * k), jac=True, method="L-BFGS-B", bounds=[(0, None)] * (2 * k),
                   options={"maxiter": 3000})
    return res.x[:k] - res.x[k:]


def race_loss(x, w, won, starts):
    size = np.diff(np.r_[starts, len(x)])
    s = x @ w
    m = np.maximum.reduceat(s, starts)
    z = np.add.reduceat(np.exp(s - np.repeat(m, size)), starts)
    logp = s - np.repeat(m + np.log(z), size)
    return -np.add.reduceat(np.where(won > 0, logp, 0.), starts)


def stage_elastic():
    description, context, matrix, names, features, spec, folds = load()
    cols = [names.index(f) for f in features]
    first = pd.Timestamp(description["receipt"]["dataset_start"])
    won = context.won.to_numpy(np.float64)
    race_id = context.race_id.to_numpy()
    dates_all = context.race_date.to_numpy()
    choice, parts = None, []
    for fold in folds:
        tr = np.flatnonzero(fold.train & (context.race_date >= first).to_numpy())
        te = np.flatnonzero(fold.test)
        xtr = np.asarray(matrix[tr][:, cols], dtype=np.float64)
        if choice is None:
            dates = dates_all[tr]
            cut = np.datetime64(pd.Timestamp(np.quantile(dates.astype("int64"), .85)))
            a = dates < cut - np.timedelta64(spec.purge_days, "D")
            b = dates >= cut
            prep = Prep(xtr[a])
            xa, xb = prep(xtr[a]), prep(xtr[b])
            sa, sb = race_starts(race_id[tr][a]), race_starts(race_id[tr][b])
            grid = {}
            for lam in (1e-4, 1e-3, 1e-2):
                for alpha in (.1, .9):
                    w = fit_clogit(xa, won[tr][a], sa, lam, alpha)
                    grid[(lam, alpha)] = float(race_loss(xb, w, won[tr][b], sb).mean())
                    print(f"  inner lambda {lam:g} alpha {alpha}: {grid[(lam, alpha)]:.5f}, "
                          f"nonzero {(np.abs(w) > 1e-8).sum()} of {len(w)}", flush=True)
            choice = min(grid, key=grid.get)
            print("chosen", choice, flush=True)
        prep = Prep(xtr)
        w = fit_clogit(prep(xtr), won[tr], race_starts(race_id[tr]), *choice)
        xte = prep(np.asarray(matrix[te][:, cols], dtype=np.float64))
        st = race_starts(race_id[te])
        loss = race_loss(xte, w, won[te], st)
        race = context.iloc[te[st]][["race_id", "race_date"]].reset_index(drop=True)
        race["fold"], race["race_log_loss"] = fold.label, loss
        parts.append(race)
        print(fold.label, len(race), round(float(loss.mean()), 5), "nonzero", int((np.abs(w) > 1e-8).sum()),
              "of", len(w), flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "elastic.parquet", index=False)
    (OUT / "elastic.json").write_text(json.dumps({"lambda": choice[0], "alpha": choice[1]}))


def stage_linear_tree(threads):
    description, context, matrix, names, features, spec, folds = load()
    lookup = {n: i for i, n in enumerate(names)}
    lt = replace(spec, screen_params=tuple({**dict(spec.screen_params), "linear_tree": True,
                                            "linear_lambda": 1.0}.items()))
    races, _ = _evaluate(context, lambda n: matrix[:, lookup[n]], lt, features, folds,
                         dataset_start=description["receipt"]["dataset_start"], capacity="screen", threads=threads)
    races.to_parquet(OUT / "linear_tree.parquet", index=False)
    print(races.groupby("fold").race_log_loss.mean(), races.race_log_loss.mean())


def stage_compare():
    ref = pd.read_parquet(REFERENCE)[["race_id", "fold", "race_date", "race_log_loss"]]
    report = {"lightgbm_screen": float(ref.race_log_loss.mean()), "races": len(ref)}
    for arm in ("elastic", "linear_tree"):
        path = OUT / f"{arm}.parquet"
        if not path.exists():
            continue
        a = pd.read_parquet(path)[["race_id", "race_log_loss"]].rename(columns={"race_log_loss": "arm"})
        m = ref.merge(a, on="race_id", validate="one_to_one")
        change = m.arm - m.race_log_loss
        by = change.groupby(m.fold).mean()
        report[arm] = {"race_log_loss": float(m.arm.mean()), "races": len(m),
                       "vs_lightgbm": paired_interval(change, blocks=m.race_date),
                       "per_fold": {k: round(float(v), 5) for k, v in by.items()},
                       "folds_better": int((by < 0).sum())}
    print(json.dumps(report, indent=1, default=float))
    (OUT / "compare.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["elastic", "linear-tree", "compare"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "elastic":
        stage_elastic()
    elif args.stage == "linear-tree":
        stage_linear_tree(args.threads)
    else:
        stage_compare()
