"""Screen market microstructure for information the price has not absorbed.

ACTION_PLAN 5a: the one live data hypothesis after gate 3 is that *where and
when money enters* the exchange carries the private information that beats
the model (ledger 2c/8). Betfair's frozen public BSP research files
add morning and pre-play matched volume, which the BASIC tars lack, so a
first, free version of 5a can be screened.

The trap, learned on jockey booking (ledger 6): a signal built from the
market can look predictive simply because it restates the price move, which
is not tradeable. So every test here is either

* **continuation** - does a signal known at the decision point predict the
  price move *after* it? (tradeable timing), or
* **orthogonality to the close** - does it predict the result beyond BSP?

Pre-registered tests (development period only; the golden hold-out is never
read). x is within-race demeaned and partialled on a price basis
(probability, log-odds, square) at the decision price, as in
triage_features:

  T1  morning excess volume  -> move morning VWAP -> BSP        (continuation)
  T2  morning excess volume  -> won - p_BSP                      (vs the close)
  T3  morning excess volume  -> won - p_morning                  (vs morning)
  T4  move morning -> T-5    -> move T-5 -> BSP                  (continuation)
  T5  morning excess volume  -> move T-5 -> BSP                  (continuation)
  T6  move morning -> T-5    -> won - p_BSP                      (vs the close)

"Excess volume" is log(runner's share of the race's morning volume) minus
log(its morning market share): money beyond what its price would attract.
T1 and T3 compare against a VWAP that is noisy and short for thinly traded
runners, which can fake a relationship mechanically; each is repeated on
runners with at least 100 matched in the morning. T4-T6 use the T-5 LTP and
cover only the BASIC period.

Bar (as triage): |partial corr| >= 0.01 and BH q <= 0.05, with q taken from
race-clustered bootstrap standard errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from math import erfc, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.betfair import attach_bsp, load_bsp
from src.wf_eval import development_mask
from scripts.triage_features import _bh_qvalues, _residualise, _within_race_demean, market_controls

PUBLIC = os.path.join("data", "research", "betfair_public_legacy", "prices.parquet")
LIQUID_MIN = 100.0


def load() -> pd.DataFrame:
    con = sqlite3.connect("file:data/races.db?mode=ro", uri=True)
    db = pd.read_sql("select race_id, race_date, track, off_time, horse_name, won from results", con)
    db = db[development_mask(db["race_date"])].copy()
    pub = pd.read_parquet(PUBLIC)
    db = attach_bsp(db, pub, cols=("bsp", "morningwap", "morningtradedvol"))
    db = attach_bsp(db, load_bsp(), cols=("ltp_300s",))
    for c in ("bsp", "morningwap", "morningtradedvol", "ltp_300s", "won"):
        db[c] = pd.to_numeric(db[c], errors="coerce")
    ok = (db["bsp"] > 1) & (db["morningwap"] > 1) & (db["morningtradedvol"] > 0)
    g = db.groupby("race_id")
    complete = ok.groupby(db["race_id"]).transform("all") & (g["won"].transform("sum") == 1)
    db = db[complete & (g["won"].transform("size") >= 2)].copy()
    race = db["race_id"]

    def norm(price):
        inv = 1.0 / price
        return inv / inv.groupby(race).transform("sum")

    db["p_bsp"] = norm(db["bsp"])
    db["p_morning"] = norm(db["morningwap"])
    vol_share = db["morningtradedvol"] / db["morningtradedvol"].groupby(race).transform("sum")
    db["excess_vol"] = np.log(vol_share) - np.log(db["p_morning"])
    db["move_m2c"] = np.log(db["morningwap"]) - np.log(db["bsp"])
    has_t5 = (db["ltp_300s"] > 1).groupby(race).transform("all")
    db.loc[has_t5, "p_t5"] = norm(db.loc[has_t5, "ltp_300s"])
    db["move_m2t5"] = np.log(db["morningwap"]) - np.log(db["ltp_300s"])
    db["move_t52c"] = np.log(db["ltp_300s"]) - np.log(db["bsp"])
    db["resid_close"] = db["won"] - db["p_bsp"]
    db["resid_morning"] = db["won"] - db["p_morning"]
    return db.reset_index(drop=True)


def partial_corr_clustered(df: pd.DataFrame, x: str, y: str, price: str,
                           n_boot: int, seed: int = 0) -> dict:
    sub = df[[x, y, price, "race_id"]].dropna()
    codes = pd.factorize(sub["race_id"])[0]
    xv = _within_race_demean(sub[x].to_numpy(float), codes)
    yv = _within_race_demean(sub[y].to_numpy(float), codes)
    z = market_controls(sub[price].to_numpy(float), codes)
    xr, yr = _residualise(xv, z), _residualise(yv, z)
    r = float(np.corrcoef(xr, yr)[0, 1])
    # Race-clustered bootstrap of the correlation of the residuals.
    rng = np.random.default_rng(seed)
    n_c = codes.max() + 1
    order = np.argsort(codes, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    sxx = np.add.reduceat(xr[order] * xr[order], starts)
    syy = np.add.reduceat(yr[order] * yr[order], starts)
    sxy = np.add.reduceat(xr[order] * yr[order], starts)
    sx = np.add.reduceat(xr[order], starts)
    sy = np.add.reduceat(yr[order], starts)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c)).astype(float)
        n = (w * sizes).sum()
        mx, my = (w * sx).sum() / n, (w * sy).sum() / n
        cov = (w * sxy).sum() / n - mx * my
        vx = (w * sxx).sum() / n - mx * mx
        vy = (w * syy).sum() / n - my * my
        boots[b] = cov / np.sqrt(vx * vy)
    se = float(boots.std(ddof=1))
    t = r / se if se > 0 else float("nan")
    return {"partial_corr": round(r, 5), "se_clustered": round(se, 5), "t": round(t, 2),
            "p": erfc(abs(t) / sqrt(2)) if np.isfinite(t) else float("nan"),
            "n": int(len(sub)), "races": int(n_c)}


TESTS = [
    ("T1", "excess_vol", "move_m2c", "p_morning", False),
    ("T1 liquid", "excess_vol", "move_m2c", "p_morning", True),
    ("T2", "excess_vol", "resid_close", "p_bsp", False),
    ("T3", "excess_vol", "resid_morning", "p_morning", False),
    ("T3 liquid", "excess_vol", "resid_morning", "p_morning", True),
    ("T4", "move_m2t5", "move_t52c", "p_t5", False),
    ("T5", "excess_vol", "move_t52c", "p_t5", False),
    ("T6", "move_m2t5", "resid_close", "p_bsp", False),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    df = load()
    liquid = df[df["morningtradedvol"] >= LIQUID_MIN]
    print(f"development rows {len(df):,} races {df['race_id'].nunique():,} "
          f"({df['race_date'].min()}..{df['race_date'].max()}); liquid rows {len(liquid):,}; "
          f"rows with T-5 {int(df['p_t5'].notna().sum()):,}", flush=True)
    rows = []
    for name, x, y, price, liq in TESTS:
        res = partial_corr_clustered(liquid if liq else df, x, y, price, args.n_boot)
        rows.append({"test": name, "x": x, "y": y, **res})
    q = _bh_qvalues(np.array([r["p"] for r in rows]))
    for r, qq in zip(rows, q):
        r["q"] = round(float(qq), 4)
        r["passes"] = bool(abs(r["partial_corr"]) >= 0.01 and qq <= 0.05)
        print(f"  {r['test']:10s} {r['x']:11s} -> {r['y']:13s} partial {r['partial_corr']:+.4f} "
              f"(se {r['se_clustered']:.4f}, t {r['t']:+.1f}, q {r['q']:.3g}) n={r['n']:,}"
              f"{'  PASSES' if r['passes'] else ''}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
