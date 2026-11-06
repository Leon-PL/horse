"""Walk-forward test of ACTION_PLAN 4c: learn only what the market misses.

Market probabilities have been tried here as features, as soft labels and as
a post-hoc blend (the Benter anchor), never as a **fixed offset**. This
harness fits a per-race conditional logit (softmax over the field) whose
starting score for each runner is the log of the market's overround-free
probability at the decision price; LightGBM trees then add a correction
f(x) from the fundamental features:

    P(i wins) = exp(log m_i + f(x_i)) / sum_j exp(log m_j + f(x_j))

With f = 0 this *is* the market, so the model can only earn its keep by
learning the market's residual - the one configuration that does not spend
its capacity relearning class and ability (ledger 2c: the model's
disagreements with the market carry no information).

Everything a result could be tuned on is fixed here, before the first run
(pre-registration): hyperparameters, early-stopping protocol, price band,
the four selection rules, purged quarterly folds, and the golden hold-out
(config.GOLDEN_HOLDOUT_START), which is excluded unless ``--holdout``.

Protocols (decision price -> settlement):

* ``t5``      - Betfair LTP 5 minutes before the off (BASIC data), settled at
                BSP. CLV = ltp_300s / bsp. Ends 2026-06-10.
* ``morning`` - Betfair morning VWAP (public BSP files) as the market the
                model corrects, known when the morning closes (~11:00 UK);
                execution at BSP. **ROI at BSP is the valid metric; the CLV
                printed (morningwap / bsp) is not** - the VWAP averages
                prices from across the morning, so no bet placed on morning
                information could take it (ledger 14). Covers the hold-out.

Usage::

    python scripts/wf_market_offset.py --protocol t5
    python scripts/wf_market_offset.py --protocol t5 --max-folds 2     # smoke
    python scripts/wf_market_offset.py --protocol morning --holdout    # ONCE
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src.bet_settlement import settle_win_exchange_bets
from src.betfair import attach_bsp
from src.feature_store import FeatureStore
from src.model import get_feature_columns
from src.wf_eval import (
    bootstrap_median_ci,
    bootstrap_roi_ci,
    clv_summary,
    development_mask,
    expanding_period_folds,
    fold_summary,
    holdout_mask,
)

DIAG = ["race_id", "race_date", "won", "track", "off_time", "horse_name",
        "num_runners", "race_type"]
PUBLIC_BSP_PATH = os.path.join("data", "research", "betfair_public_legacy", "prices.parquet")

#: Pre-registered. Strong regularisation: the residual is small (ledger 2:
#: beta ~0.97 means the market carries ~20x the model's information).
PARAMS = dict(
    learning_rate=0.03, num_leaves=15, min_data_in_leaf=400,
    feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
    lambda_l2=10.0, max_bin=127, verbose=-1, num_threads=0,
    seed=config.RANDOM_SEED,
)
MAX_ROUNDS = 1500
EARLY_STOP = 100          # rounds without validation improvement
EVAL_EVERY = 10
INNER_VALID_FRAC = 0.15   # latest share of training dates used to stop early
BAND = (1.5, 50.0)        # decision-price band for striking a bet
RULES = {                 # edge = P_model / m_market - 1
    "edge>0": lambda e, top: e > 0.0,
    "edge>0.05": lambda e, top: e > 0.05,
    "edge>0.10": lambda e, top: e > 0.10,
    "top-edge": lambda e, top: top & (e > 0.0),
}


# ── data ──────────────────────────────────────────────────────────────────

def _prices(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    if protocol == "t5":
        out = attach_bsp(df, cols=("bsp", "ltp_300s"))
        out["_entry"] = pd.to_numeric(out["ltp_300s"], errors="coerce")
    elif protocol == "morning":
        if not os.path.exists(PUBLIC_BSP_PATH):
            raise SystemExit(f"{PUBLIC_BSP_PATH} missing: restore the frozen public research dataset")
        public = pd.read_parquet(PUBLIC_BSP_PATH)
        out = attach_bsp(df, public, cols=("bsp", "morningwap"))
        out["_entry"] = pd.to_numeric(out["morningwap"], errors="coerce")
    else:
        raise ValueError(protocol)
    out["_close"] = pd.to_numeric(out["bsp"], errors="coerce")
    return out


def load_frame(protocol: str, extra: tuple = ()) -> pd.DataFrame:
    store = FeatureStore()
    available = set(pq.read_schema(store.featured_path).names)
    allow = [line.strip() for line in open(config.FEATURE_ALLOWLIST, encoding="utf-8")
             if line.strip() and not line.lstrip().startswith("#")]
    cols = [c for c in sorted(set(allow) | set(DIAG) | set(extra)) if c in available]
    df = pd.read_parquet(store.featured_path, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["won"] = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int)
    df = _prices(df, protocol)

    # Complete races only: a race with an unpriced runner has no coherent
    # overround-free market, and a partial field biases the normalisation.
    ok = (df["_entry"] > 1.0) & (df["_close"] > 1.0)
    race = df.groupby("race_id")
    complete = ok.groupby(df["race_id"]).transform("all")
    one_winner = race["won"].transform("sum") == 1
    field = race["won"].transform("size") >= 2
    df = df[complete & one_winner & field].copy()
    df = df.sort_values(["race_date", "race_id"], kind="stable").reset_index(drop=True)
    implied = 1.0 / df["_entry"]
    df["mkt"] = implied / implied.groupby(df["race_id"]).transform("sum")
    df["offset"] = np.log(df["mkt"])
    return df


# ── model ─────────────────────────────────────────────────────────────────

def _starts(race_ids: np.ndarray) -> np.ndarray:
    change = np.r_[True, race_ids[1:] != race_ids[:-1]]
    return np.flatnonzero(change)


def race_softmax(scores: np.ndarray, starts: np.ndarray) -> np.ndarray:
    sizes = np.diff(np.r_[starts, len(scores)])
    shifted = scores - np.repeat(np.maximum.reduceat(scores, starts), sizes)
    e = np.exp(shifted)
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


def _objective(starts):
    def fobj(preds, data):   # preds include init_score (verified on lightgbm 4.6)
        p = race_softmax(preds, starts)
        return p - data.get_label(), np.maximum(p * (1.0 - p), 1e-9)
    return fobj


def race_logloss_vector(probs: np.ndarray, won: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Per-race negative log-probability of the winner."""
    ends = np.r_[starts[1:], len(probs)]
    out = np.empty(len(starts))
    for k, (a, b) in enumerate(zip(starts, ends)):
        w = np.flatnonzero(won[a:b])
        out[k] = -np.log(max(probs[a + w[0]], 1e-12))
    return out


class _EarlyStop:
    """Stop on inner-validation race log loss, computed explicitly with offsets."""

    def __init__(self, X, offset, won, starts):
        self.X, self.offset, self.won, self.starts = X, offset, won, starts
        self.best, self.best_iter = np.inf, 0

    def __call__(self, env):
        it = env.iteration + 1
        if it % EVAL_EVERY:
            return
        raw = env.model.predict(self.X, raw_score=True, num_iteration=it)
        loss = race_logloss_vector(race_softmax(self.offset + raw, self.starts),
                                   self.won, self.starts).mean()
        if loss < self.best - 1e-7:
            self.best, self.best_iter = loss, it
        elif it - self.best_iter >= EARLY_STOP:
            raise lgb.callback.EarlyStopException(self.best_iter, [])


def fit_offset_model(train: pd.DataFrame, features: list[str]):
    """Fit on the earlier dates of *train*; stop early on the latest dates."""
    dates = train["race_date"].to_numpy()
    cut = np.quantile(dates.astype("datetime64[ns]").astype(np.int64), 1 - INNER_VALID_FRAC)
    is_val = dates.astype("datetime64[ns]").astype(np.int64) > cut
    inner, val = train[~is_val], train[is_val]
    s_in, s_val = _starts(inner["race_id"].to_numpy()), _starts(val["race_id"].to_numpy())
    ds = lgb.Dataset(inner[features], label=inner["won"].to_numpy(),
                     init_score=inner["offset"].to_numpy(), free_raw_data=False)
    stopper = _EarlyStop(val[features], val["offset"].to_numpy(),
                         val["won"].to_numpy(), s_val)
    booster = lgb.train({**PARAMS, "objective": _objective(s_in)}, ds,
                        num_boost_round=MAX_ROUNDS, callbacks=[stopper])
    return booster, max(stopper.best_iter, 1), stopper.best


def predict(booster, best_iter: int, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = booster.predict(frame[features], raw_score=True, num_iteration=best_iter)
    return race_softmax(frame["offset"].to_numpy() + raw, _starts(frame["race_id"].to_numpy()))


# ── evaluation ────────────────────────────────────────────────────────────

def score_block(test: pd.DataFrame, probs: np.ndarray) -> dict:
    starts = _starts(test["race_id"].to_numpy())
    won = test["won"].to_numpy()
    ll_model = race_logloss_vector(probs, won, starts)
    ll_mkt = race_logloss_vector(test["mkt"].to_numpy(), won, starts)
    edge = probs / test["mkt"].to_numpy() - 1.0
    top = pd.Series(edge).groupby(test["race_id"].to_numpy()).transform("max").to_numpy() == edge
    band = test["_entry"].between(*BAND).to_numpy()
    pnl = settle_win_exchange_bets(test["_close"].to_numpy(), won.astype(bool))
    return {"delta_ll": ll_model - ll_mkt, "race_of_delta": test["race_id"].to_numpy()[starts],
            "edge": edge, "top": top, "band": band, "pnl": pnl}


def summarise(blocks: list[dict], tests: list[pd.DataFrame], labels: list[str], n_boot: int) -> dict:
    race_ids = np.concatenate([t["race_id"].to_numpy() for t in tests])
    entry = np.concatenate([t["_entry"].to_numpy() for t in tests])
    close = np.concatenate([t["_close"].to_numpy() for t in tests])
    pnl = np.concatenate([b["pnl"] for b in blocks])
    edge = np.concatenate([b["edge"] for b in blocks])
    top = np.concatenate([b["top"] for b in blocks])
    band = np.concatenate([b["band"] for b in blocks])
    delta = np.concatenate([b["delta_ll"] for b in blocks])
    delta_races = np.concatenate([b["race_of_delta"] for b in blocks])
    fold_of = np.concatenate([[lab] * len(t) for lab, t in zip(labels, tests)])

    report = {
        "races": int(len(delta)), "runners": int(len(pnl)),
        # Negative = the offset model scores the winner better than the market.
        "delta_race_logloss": {
            **bootstrap_roi_ci(delta, clusters=delta_races, n_boot=n_boot, scale=1.0),
            "folds_better": int(sum(b["delta_ll"].mean() < 0 for b in blocks)),
            "n_folds": len(blocks),
        },
    }

    def rule_report(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"n": 0}
        ratio = entry[mask] / close[mask]
        fold_rois = [float(pnl[mask & (fold_of == lab)].mean() * 100)
                     for lab in labels if (mask & (fold_of == lab)).any()]
        return {
            "roi_bsp": bootstrap_roi_ci(pnl[mask], clusters=race_ids[mask], n_boot=n_boot),
            "clv": {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in clv_summary(entry[mask], close[mask]).items()},
            "median_clv_ci": bootstrap_median_ci(ratio, clusters=race_ids[mask], n_boot=n_boot),
            "geo_clv_ci": bootstrap_median_ci(ratio, clusters=race_ids[mask], n_boot=n_boot,
                                              log_mean=True),
            "folds": fold_summary(fold_rois),
        }

    report["bet_all_in_band"] = rule_report(band)
    report["rules"] = {name: rule_report(rule(edge, top) & band) for name, rule in RULES.items()}
    return report


def run(protocol: str, *, holdout: bool = False, max_folds: int | None = None,
        n_boot: int = 2000, dump: str | None = None, drop_prefixes: tuple = (),
        only_features: tuple = ()) -> dict:
    df = load_frame(protocol, extra=tuple(only_features))
    if only_features:
        # Test one candidate signal on its own: can the residual model use it?
        features = [c for c in only_features if c in df.columns]
        if not features:
            raise SystemExit(f"none of {only_features} are in the store")
        # A coverage-gated signal (e.g. sectionals, which start 2023-08) is
        # judged on races that carry it; elsewhere a fold has nothing to learn.
        signal = [c for c in features if not c.startswith("has_")] or features
        covered = df[signal].notna().any(axis=1).groupby(df["race_id"]).transform("any")
        df = df[covered.to_numpy()].reset_index(drop=True)
    else:
        features = [c for c in get_feature_columns(df)
                    if c not in ("bsp", "ltp_300s", "morningwap", "_entry", "_close", "mkt", "offset")
                    and not c.startswith(tuple(drop_prefixes))]
    dev = development_mask(df["race_date"])
    blocks, tests, labels, per_fold, dumps = [], [], [], [], []

    if holdout:
        train, test = df[dev], df[holdout_mask(df["race_date"])]
        if test.empty:
            return {"error": f"no {protocol} prices in the golden hold-out"}
        plan = [("HOLDOUT", train, test)]
    else:
        work = df[dev].reset_index(drop=True)
        folds = expanding_period_folds(work["race_date"], freq="Q", min_train_periods=8)
        plan = [(f.label, work[f.train], work[f.test]) for f in folds][: max_folds or None]
    print(f"protocol {protocol}: rows {len(df):,}, features {len(features)}, "
          f"blocks {len(plan)}{' (GOLDEN HOLD-OUT)' if holdout else ''}", flush=True)

    for label, train, test in plan:
        booster, best_iter, val_loss = fit_offset_model(train, features)
        probs = predict(booster, best_iter, test, features)
        block = score_block(test, probs)
        blocks.append(block); tests.append(test); labels.append(label)
        gain = booster.feature_importance("gain", iteration=best_iter)
        top_feats = [features[i] for i in np.argsort(gain)[::-1][:5]]
        per_fold.append({"block": label, "train_rows": int(len(train)), "test_races": int(len(block["delta_ll"])),
                         "trees": best_iter, "delta_ll": round(float(block["delta_ll"].mean()), 5),
                         "top_features": top_feats})
        print(f"  {label}: trees={best_iter:4d} delta_ll={block['delta_ll'].mean():+.5f} "
              f"races={len(block['delta_ll']):,} top={top_feats[:3]}", flush=True)
        if dump:
            dumps.append(test[["race_id", "race_date", "horse_name", "won", "_entry", "_close", "mkt"]]
                         .assign(model_prob=probs, edge=block["edge"], block=label))

    report = summarise(blocks, tests, labels, n_boot)
    report["protocol"], report["holdout"], report["per_fold"] = protocol, holdout, per_fold
    if dump and dumps:
        pd.concat(dumps, ignore_index=True).to_parquet(dump, index=False)
    return report


def render(r: dict) -> str:
    if "error" in r:
        return r["error"]
    d = r["delta_race_logloss"]
    lines = [f"\n=== market-offset model, protocol {r['protocol']}"
             f"{' - GOLDEN HOLD-OUT' if r['holdout'] else ''} ===",
             f"races {r['races']:,}  runners {r['runners']:,}",
             *(["NOTE: morning protocol - read ROI@BSP; the CLV column is not a valid measure here"]
               if r["protocol"] == "morning" else []),
             f"delta race log loss (model - market): {d['mean']:+.5f} "
             f"CI[{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]  folds better {d['folds_better']}/{d['n_folds']}",
             ""]
    for name, rep in [("bet-all (band)", r["bet_all_in_band"]), *r["rules"].items()]:
        if not rep.get("n", 1):
            lines.append(f"  {name:15s} no bets"); continue
        roi, mc, gc = rep["roi_bsp"], rep["median_clv_ci"], rep["geo_clv_ci"]
        lines.append(
            f"  {name:15s} n={roi['n']:6,}  ROI@BSP {roi['mean']:+6.2f}% CI[{roi['ci_low']:+.2f},{roi['ci_high']:+.2f}]"
            f"  medCLV {mc['estimate']:.4f} CI[{mc['ci_low']:.4f},{mc['ci_high']:.4f}]"
            f"  geoCLV {gc['estimate']:.4f}  beat {rep['clv'].get('pct_beat_close')}"
            f"  folds+ {rep['folds']['folds_positive']}/{rep['folds']['n_folds']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", choices=("t5", "morning"), default="t5")
    ap.add_argument("--holdout", action="store_true", help="score the golden hold-out (once)")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--drop", action="append", default=[], metavar="PREFIX",
                    help="exclude features starting with PREFIX (repeatable)")
    ap.add_argument("--only", action="append", default=[], metavar="COLUMN",
                    help="use only these store columns as features (repeatable), e.g. a "
                         "shortlisted triage signal")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    report = run(args.protocol, holdout=args.holdout, max_folds=args.max_folds,
                 n_boot=args.n_boot, dump=args.dump, drop_prefixes=tuple(args.drop),
                 only_features=tuple(args.only))
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
