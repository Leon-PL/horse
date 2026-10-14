"""Ledger 20: execution timing - when to strike, not what to back.

Pre-registered in docs/NEGATIVE_RESULTS.md section 20 (commit 6f32c98) before
this was run. EXPLORATION_PLAN idea B.

For a bet already worth making, predict ``y_t = log(P_t / BSP)`` from what is
known at decision time *t* (11:00 UK, T-60, T-5) and strike early when the
prediction clears ``tau``; otherwise take BSP. ``P_t`` is the LTP one tick
down Betfair's ladder, cut by the reduction factor of every runner withdrawn
after the strike (and of any withdrawn between the last trade and the cut):
Betfair reduces matched odds for non-runners and BSP is struck after them.

Primary metric per universe: ``G`` = mean over the universe of ``y`` at the
strike time if struck, 0 if BSP was taken, in CLV points. Pass: ``G >= 0.5``
with a race-clustered 98.33% interval (Bonferroni over three universes)
above zero. Development period only; purged quarterly expanding folds.

Usage::

    python scripts/execution_timing.py
    python scripts/execution_timing.py --json out.json --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.microstructure_screen import load
from src.bet_settlement import (EXCHANGE_COMMISSION, apply_reduction_factor, betfair_tick_down,
                                settle_win_exchange_bets)
from src.betfair import attach_bsp
from src.wf_eval import (bootstrap_median_ci, bootstrap_roi_ci, clv_summary,
                         expanding_period_folds, fold_summary)

TIMING = os.path.join("data", "processed", "betfair_timing.parquet")
CUTS = ("1100", "3600s", "300s")
BAND = (1.5, 50.0)
TAU = 0.01
PRIMARY_ALPHA = 0.05 / 3
MIN_OFF_UK_MINUTES = 12 * 60 + 5
MIN_TRAIN_QUARTERS = 4
LGBM_R2_UPLIFT = 1.20

_TIMING_COLS = tuple(
    [f"ltp_{c}" for c in CUTS] + ["ltp_eve1800", "ltp_0800", "off_ms"]
    + [f"{p}_{c}" for c in CUTS for p in ("age", "nchg", "rf_after", "rf_stale",
                                           "rf25_after", "rf25_stale",
                                           "n_removed_after", "book_removed_after",
                                           "min_removed_ltp")]
)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

def _race_info() -> pd.DataFrame:
    con = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    info = pd.read_sql("select race_id, race_type, handicap, surface from results", con)
    info = info.drop_duplicates("race_id")
    info["jumps"] = info["race_type"].isin(["Hurdle", "Chase"]).astype(float)
    info["hcap"] = pd.to_numeric(info["handicap"], errors="coerce").fillna(0).astype(float)
    info["aw"] = (info["surface"].fillna("TURF").str.upper() != "TURF").astype(float)
    return info[["race_id", "jumps", "hcap", "aw"]]


def frame() -> pd.DataFrame:
    """Ledger 14's development frame + the timing columns, races off >= 12:05 UK,
    every runner traded at each cut."""
    df = load().drop(columns=["ltp_300s", "p_t5", "move_m2t5", "move_t52c"], errors="ignore")
    df = attach_bsp(df, pd.read_parquet(TIMING), cols=_TIMING_COLS)
    for c in _TIMING_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    off = pd.to_datetime(df["race_date"].astype(str) + " " + df["off_time"].astype(str), errors="coerce")
    local = off.dt.tz_localize("UTC").dt.tz_convert("Europe/London")
    df["off_local_min"] = (local.dt.hour * 60 + local.dt.minute).to_numpy()
    race = df["race_id"]
    priced = pd.Series(True, index=df.index)
    for c in CUTS:
        priced &= (df[f"ltp_{c}"] > 1)
    complete = priced.groupby(race).transform("all")
    df = df[complete & (df["off_local_min"] >= MIN_OFF_UK_MINUTES)].copy()
    df["race_id"] = df["race_id"].astype(str)
    info = _race_info()
    info["race_id"] = info["race_id"].astype(str)
    df = df.merge(info, on="race_id", how="left")
    for c in ("jumps", "hcap", "aw"):
        df[c] = df[c].fillna(0.0)
    df["q"] = pd.to_datetime(df["race_date"]).dt.to_period("Q").astype(str)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strike prices and features
# ---------------------------------------------------------------------------

def strike_price(d: pd.DataFrame, cut: str, *, ticks: int = 1, pct: float = 0.0,
                 reductions: bool = True, waived: bool = False) -> np.ndarray:
    """Price a back bet struck at *cut* would settle at. ``waived`` applies only
    reduction factors of 2.5% or more (if Betfair waives smaller ones)."""
    p = d[f"ltp_{cut}"].to_numpy(float)
    if ticks:
        p = betfair_tick_down(p, ticks)
    if pct:
        p = np.maximum(p * (1.0 - pct), 1.01)
    if reductions:
        pre = "rf25" if waived else "rf"
        factor = d[f"{pre}_stale_{cut}"].fillna(1.0).to_numpy(float) * d[f"{pre}_after_{cut}"].fillna(1.0).to_numpy(float)
        p = apply_reduction_factor(p, factor)
    return p


def features(d: pd.DataFrame, cut: str) -> pd.DataFrame:
    """Everything known at the cut; never the result, BSP or ``rf_after``."""
    race = d["race_id"]
    p = d[f"ltp_{cut}"].to_numpy(float)
    ip = 1.0 / p
    removed_book = d[f"book_removed_after_{cut}"].fillna(0.0).groupby(race).transform("first").to_numpy()
    over = pd.Series(ip).groupby(race.to_numpy()).transform("sum").to_numpy() + removed_book
    race_min = pd.Series(p).groupby(race.to_numpy()).transform("min").to_numpy()
    removed_min = d[f"min_removed_ltp_{cut}"].groupby(race).transform("first").fillna(np.inf).to_numpy()
    field = d.groupby("race_id")["race_id"].transform("size").to_numpy() + \
        d[f"n_removed_after_{cut}"].fillna(0).groupby(race).transform("first").to_numpy()
    race_vol = d["morningtradedvol"].groupby(race).transform("sum").to_numpy()
    if cut == "1100":
        mins = (d["off_local_min"].to_numpy(float) - 11 * 60)
    else:
        mins = np.full(len(d), 60.0 if cut == "3600s" else 5.0)
    f = {
        "lp": np.log(p), "lp2": np.log(p) ** 2, "ip": ip,
        "log_over": np.log(over), "p_norm": ip / over,
        "fav": (p <= np.minimum(race_min, removed_min)).astype(float),
        "log_field": np.log(field),
        "jumps": d["jumps"].to_numpy(float), "hcap": d["hcap"].to_numpy(float),
        "aw": d["aw"].to_numpy(float),
        "off_hour": d["off_local_min"].to_numpy(float) / 60.0,
        "log_mins": np.log(np.maximum(mins, 1.0)),
        "log_age": np.log1p(d[f"age_{cut}"].fillna(0).to_numpy(float)),
        "log_nchg": np.log1p(d[f"nchg_{cut}"].fillna(0).to_numpy(float)),
        "log_rf_stale": np.log(d[f"rf_stale_{cut}"].fillna(1.0).to_numpy(float)),
        "excess_vol": d["excess_vol"].to_numpy(float),
        "log_vol": np.log1p(d["morningtradedvol"].to_numpy(float)),
        "log_race_vol": np.log1p(race_vol),
        "move_mwap": np.log(d["morningwap"].to_numpy(float) / p),
    }
    for early in ("eve1800", "0800"):
        e = d[f"ltp_{early}"].to_numpy(float)
        ok = np.isfinite(e) & (e > 1)
        f[f"move_{early}"] = np.where(ok, np.log(np.where(ok, e, 1.0) / p), 0.0)
        f[f"miss_{early}"] = (~ok).astype(float)
    if cut in ("3600s", "300s"):
        f["move_1100"] = np.log(d["ltp_1100"].to_numpy(float) / p)
    if cut == "300s":
        f["move_3600s"] = np.log(d["ltp_3600s"].to_numpy(float) / p)
    return pd.DataFrame(f, index=d.index)


# ---------------------------------------------------------------------------
# Walk-forward predictions
# ---------------------------------------------------------------------------

def _ols_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-12
    Ztr = np.c_[np.ones(len(Xtr)), (Xtr[:, keep] - mu[keep]) / sd[keep]]
    Zte = np.c_[np.ones(len(Xte)), (Xte[:, keep] - mu[keep]) / sd[keep]]
    beta, *_ = np.linalg.lstsq(Ztr, ytr, rcond=None)
    return Zte @ beta


def _lgbm_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=500, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=0,
                              verbose=-1)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def walk_forward(d: pd.DataFrame, train_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Out-of-fold predictions of y_t (one-tick, reductions) per cut and model."""
    folds = expanding_period_folds(d["race_date"], freq="Q", min_train_periods=MIN_TRAIN_QUARTERS)
    preds = pd.DataFrame(index=d.index)
    r2 = {}
    for cut in CUTS:
        X = features(d, cut).to_numpy(float)
        y = np.log(strike_price(d, cut) / d["bsp"].to_numpy(float))
        for name, fn in (("ols", _ols_fit_predict), ("lgbm", _lgbm_fit_predict)):
            col = np.full(len(d), np.nan)
            for f in folds:
                tr = f.train & train_mask
                col[f.test] = fn(X[tr], y[tr], X[f.test])
            preds[f"yhat_{name}_{cut}"] = col
        preds[f"y_{cut}"] = y
        ok = preds[f"yhat_ols_{cut}"].notna() & train_mask
        resid_var = {m: float(np.mean((y[ok] - preds.loc[ok, f"yhat_{m}_{cut}"]) ** 2)) for m in ("ols", "lgbm")}
        tot = float(np.var(y[ok]))
        r2[cut] = {m: round(1.0 - v / tot, 5) for m, v in resid_var.items()}
        r2[cut]["chosen"] = "lgbm" if r2[cut]["lgbm"] >= LGBM_R2_UPLIFT * max(r2[cut]["ols"], 1e-9) else "ols"
        print(f"  {cut}: OOF R2 ols {r2[cut]['ols']:.4f} lgbm {r2[cut]['lgbm']:.4f} -> {r2[cut]['chosen']}",
              flush=True)
    for cut in CUTS:
        preds[f"yhat_{cut}"] = preds[f"yhat_{r2[cut]['chosen']}_{cut}"]
    preds["fold"] = np.nan
    for i, f in enumerate(folds):
        preds.loc[f.test, "fold"] = i
    return preds, r2


# ---------------------------------------------------------------------------
# Policy and scoring
# ---------------------------------------------------------------------------

def policy(preds: pd.DataFrame, tau: float) -> np.ndarray:
    """Index into CUTS of the first cut whose prediction clears tau; -1 = BSP."""
    strike = np.full(len(preds), -1)
    for i, cut in reversed(list(enumerate(CUTS))):
        strike = np.where(preds[f"yhat_{cut}"].to_numpy() > tau, i, strike)
    return strike


def taken_price(d: pd.DataFrame, strike: np.ndarray, **price_kw) -> np.ndarray:
    out = d["bsp"].to_numpy(float).copy()
    for i, cut in enumerate(CUTS):
        m = strike == i
        if m.any():
            out[m] = strike_price(d.loc[m], cut, **price_kw)
    return out


def score(d: pd.DataFrame, mask: np.ndarray, taken: np.ndarray, strike: np.ndarray,
          n_boot: int, alpha: float) -> dict:
    s = d[mask]
    t = taken[mask]
    bsp = s["bsp"].to_numpy(float)
    clusters = s["race_id"].to_numpy()
    g = np.log(t / bsp)
    won = s["won"].to_numpy().astype(bool)
    gain = np.where(won, (t - bsp) * (1.0 - EXCHANGE_COMMISSION), 0.0)
    st = strike[mask]
    per_q = pd.Series(g).groupby(s["q"].to_numpy()).mean().mul(100).tolist()
    return {
        "n": int(mask.sum()),
        "G": bootstrap_roi_ci(g, clusters=clusters, n_boot=n_boot, alpha=alpha),
        "pnl_gain": bootstrap_roi_ci(gain, clusters=clusters, n_boot=n_boot, alpha=alpha),
        "share_struck": {c: round(float((st == i).mean()), 4) for i, c in enumerate(CUTS)},
        "quarters": fold_summary(per_q),
        "alpha": alpha,
    }


def standalone(d: pd.DataFrame, mask: np.ndarray, taken: np.ndarray, n_boot: int) -> dict:
    """The struck bets as a strategy of their own (ACTION_PLAN 7)."""
    s = d[mask]
    if s.empty:
        return {"n": 0}
    t = taken[mask]
    clusters = s["race_id"].to_numpy()
    won = s["won"].to_numpy().astype(bool)
    clv = clv_summary(t, s["bsp"])
    return {
        "n": int(len(s)),
        "median_clv_ci": bootstrap_median_ci(t / s["bsp"].to_numpy(float), clusters=clusters, n_boot=n_boot),
        "geo_clv": clv.get("geometric_mean_clv"),
        "beat_close": clv.get("pct_beat_close"),
        "roi_taken": bootstrap_roi_ci(settle_win_exchange_bets(t, won), clusters=clusters, n_boot=n_boot),
        "roi_bsp": bootstrap_roi_ci(settle_win_exchange_bets(s["bsp"].to_numpy(float), won),
                                    clusters=clusters, n_boot=n_boot),
        "median_morning_vol": float(s["morningtradedvol"].median()),
    }


def universes(d: pd.DataFrame) -> dict[str, np.ndarray]:
    p = d["ltp_1100"]
    band = p.between(*BAND).to_numpy()
    fav = features(d, "1100")["fav"].to_numpy() > 0
    top = d["excess_vol"].where(band).groupby(d["race_id"]).transform("max").to_numpy()
    pick = band & (d["excess_vol"].to_numpy() == top)
    return {"U1 all in band": band, "U2 11:00 favourite": band & fav, "U3 morning-money pick": pick}


def run(n_boot: int) -> dict:
    d = frame()
    print(f"frame: {d['race_id'].nunique():,} races, {len(d):,} runners "
          f"({d['race_date'].min()}..{d['race_date'].max()})", flush=True)
    unis = universes(d)
    preds, r2 = walk_forward(d, train_mask=unis["U1 all in band"])
    tested = preds["fold"].notna().to_numpy()
    print(f"tested rows {int(tested.sum()):,} in {int(preds['fold'].nunique())} folds", flush=True)
    report: dict = {"r2": r2, "primary": {}, "variants": {}, "always": {}, "standalone": {}}
    strike = policy(preds, TAU)
    variants = {
        "one tick (primary)": dict(ticks=1),
        "no haircut": dict(ticks=0),
        "1% haircut": dict(ticks=0, pct=0.01),
        "two ticks": dict(ticks=2),
        "one tick, no reductions": dict(ticks=1, reductions=False),
        "one tick, reductions >= 2.5% only": dict(ticks=1, waived=True),
    }
    for uname, umask in unis.items():
        m = umask & tested
        taken = taken_price(d, strike, ticks=1)
        res = score(d, m, taken, strike, n_boot, PRIMARY_ALPHA)
        lo = res["G"]["ci_low"]
        res["pass"] = bool(res["G"]["mean"] >= 0.5 and lo is not None and lo > 0)
        report["primary"][uname] = res
        report["variants"][uname] = {}
        for vname, kw in variants.items():
            if vname == "one tick (primary)":
                continue
            report["variants"][uname][vname] = score(d, m, taken_price(d, strike, **kw), strike, n_boot, 0.05)["G"]
        for t in (0.0, 0.02):
            st = policy(preds, t)
            report["variants"][uname][f"tau {t}"] = score(d, m, taken_price(d, st, ticks=1), st, n_boot, 0.05)["G"]
        report["always"][uname] = {}
        for i, cut in enumerate(CUTS):
            st = np.full(len(d), i)
            report["always"][uname][cut] = score(d, m, taken_price(d, st, ticks=1), st, n_boot, 0.05)["G"]
        report["standalone"][uname] = standalone(d, m & (strike >= 0), taken, n_boot)
    return report


# ---------------------------------------------------------------------------
# Ledgers 14 and 19 re-scored with the reductions
# ---------------------------------------------------------------------------

def rescore_earlier(n_boot: int) -> dict:
    """Ledger 14's m11 rule and ledger 19's H2, with and without reductions."""
    from scripts.morning_money_segments import frame as mm_frame
    d = mm_frame()
    t = pd.read_parquet(TIMING)
    d = attach_bsp(d, t, cols=("rf_after_1100", "rf_stale_1100", "rf25_after_1100", "rf25_stale_1100"))
    factor = d["rf_after_1100"].fillna(1.0).to_numpy(float) * d["rf_stale_1100"].fillna(1.0).to_numpy(float)
    factor25 = d["rf25_after_1100"].fillna(1.0).to_numpy(float) * d["rf25_stale_1100"].fillna(1.0).to_numpy(float)
    out = {"share_reduced": round(float((factor < 1).mean()), 4),
           "share_reduced_ge25": round(float((factor25 < 1).mean()), 4)}
    for name, mask in (("ledger 14 m11: all picks", d["pick"]), ("ledger 19 H2: pick & favourite", d["pick"] & d["fav"])):
        s = d[mask]
        f = factor[mask.to_numpy()]
        f25 = factor25[mask.to_numpy()]
        res = {}
        for label, price in (("as recorded", s["ltp_1100"].to_numpy(float)),
                             ("with reductions", apply_reduction_factor(s["ltp_1100"].to_numpy(float), f)),
                             ("reductions >= 2.5% only", apply_reduction_factor(s["ltp_1100"].to_numpy(float), f25)),
                             ("reductions + one tick", apply_reduction_factor(
                                 betfair_tick_down(s["ltp_1100"].to_numpy(float)), f))):
            won = s["won"].to_numpy().astype(bool)
            clusters = s["race_id"].to_numpy()
            clv = clv_summary(price, s["bsp"])
            res[label] = {
                "n": int(len(s)),
                "median_clv_ci": bootstrap_median_ci(price / s["bsp"].to_numpy(float), clusters=clusters, n_boot=n_boot),
                "geo_clv": clv.get("geometric_mean_clv"),
                "beat_close": clv.get("pct_beat_close"),
                "roi_taken": bootstrap_roi_ci(settle_win_exchange_bets(price, won), clusters=clusters, n_boot=n_boot),
            }
        out[name] = res
    return out


def render(report: dict, earlier: dict) -> str:
    L = ["", "=== ledger 20: execution timing (development, walk-forward) ==="]
    for cut, r in report["r2"].items():
        L.append(f"  OOF R2 {cut}: ols {r['ols']:.4f}  lgbm {r['lgbm']:.4f}  -> {r['chosen']}")
    for u, res in report["primary"].items():
        g, pg = res["G"], res["pnl_gain"]
        L.append(f"\n  {u}: n={res['n']:,}  G {g['mean']:+.3f} [{g['ci_low']:+.3f},{g['ci_high']:+.3f}] "
                 f"(98.33%)  PASS={res['pass']}  PnL gain {pg['mean']:+.3f} [{pg['ci_low']:+.3f},{pg['ci_high']:+.3f}]"
                 f"  struck {res['share_struck']}  q+ {res['quarters']['folds_positive']}/{res['quarters']['n_folds']}")
        for v, gv in report["variants"][u].items():
            L.append(f"      {v:26s} G {gv['mean']:+.3f} [{gv['ci_low']:+.3f},{gv['ci_high']:+.3f}]")
        for cut, gv in report["always"][u].items():
            L.append(f"      always strike at {cut:8s} G {gv['mean']:+.3f} [{gv['ci_low']:+.3f},{gv['ci_high']:+.3f}]")
        sa = report["standalone"][u]
        if sa.get("n"):
            m = sa["median_clv_ci"]
            L.append(f"      struck bets alone: n={sa['n']:,} medCLV {m['estimate']:.4f} [{m['ci_low']:.4f},{m['ci_high']:.4f}]"
                     f" geo {sa['geo_clv']:.4f} beat {sa['beat_close']:.3f} ROI@taken {sa['roi_taken']['mean']:+.2f}%"
                     f" [{sa['roi_taken']['ci_low']:+.2f},{sa['roi_taken']['ci_high']:+.2f}] ROI@BSP {sa['roi_bsp']['mean']:+.2f}%"
                     f" morning vol median {sa['median_morning_vol']:.0f}")
    L.append(f"\n=== ledgers 14/19 re-scored (share of m11 runners with a later reduction: {earlier['share_reduced']}) ===")
    L.append(f"    (with a reduction of 2.5% or more: {earlier['share_reduced_ge25']})")
    for name, res in earlier.items():
        if name.startswith("share_reduced"):
            continue
        for label, r in res.items():
            m = r["median_clv_ci"]
            L.append(f"  {name:32s} {label:22s} n={r['n']:,} medCLV {m['estimate']:.4f} [{m['ci_low']:.4f},{m['ci_high']:.4f}]"
                     f" geo {r['geo_clv']:.4f} beat {r['beat_close']:.3f} ROI {r['roi_taken']['mean']:+.2f}%"
                     f" [{r['roi_taken']['ci_low']:+.2f},{r['roi_taken']['ci_high']:+.2f}]")
    return "\n".join(L) + "\n"


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    report = run(args.n_boot)
    earlier = rescore_earlier(args.n_boot)
    print(render(report, earlier))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "earlier": earlier}, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
