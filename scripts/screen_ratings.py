"""Screen the rating systems as standalone win predictors.

Recomputes the Elo / Glicko / TrueSkill sweeps over the development period of
the stored processed history (optionally with overridden settings) and scores
each rating with a within-race conditional logit: fitted on the first 60% of
races by date, scored on the last 40%. Reports race log loss for the rating on
its own and on top of the starting-price market, and the correlation of the
race-centred rating with ``won - market``.

This judges ratings as *skill measures*. A rating that improves here can still
add nothing to a model that already sees the market: read the "+market" column
and docs/NEGATIVE_RESULTS.md before treating a gain as an edge. The market is
SP, not BSP, so this is a screen, not a verdict.

    .venv/Scripts/python.exe scripts/screen_ratings.py
    .venv/Scripts/python.exe scripts/screen_ratings.py --set K_BASE=512 --set K_MIN=256
    .venv/Scripts/python.exe scripts/screen_ratings.py --stored   # the store's columns

``--set NAME=VALUE`` patches a ``src.ratings`` module binding (K_BASE,
GLICKO_EVIDENCE_POWER, ...) or, failing that, a ``config`` flag
(TRUESKILL_ENABLED, ...).
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from src import ratings
from src.feature_store import FeatureStore
from src.wf_eval import development_mask

STORE = ROOT / "data/processed/feature_store"
INPUTS = ["race_id", "race_date", "off_time", "horse_id", "horse_name", "jockey", "trainer",
          "finish_position", "won", "odds", "lengths_behind", "beaten_lengths",
          "distance_furlongs", "surface", "race_type"]

# name -> columns of one conditional-logit model
ELO_MODELS = {
    "horse_elo": ["horse_elo"],
    "horse_margin_elo": ["horse_margin_elo"],
    "combined_elo": ["combined_elo"],
    "elo + momentum": ["horse_elo", "horse_elo_momentum"],
    "elo + jockey + trainer": ["horse_elo", "jockey_elo", "trainer_elo"],
    "elo + dim edges": ["horse_elo", "horse_elo_surface_edge", "horse_elo_rt_edge",
                        "horse_elo_dist_cat_edge"],
}
GLICKO_MODELS = {
    "horse_glicko": ["horse_glicko"],
    "horse_glicko_lcb": ["horse_glicko_lcb"],
    "horse_ts_mu": ["horse_ts_mu"],
    "horse_ts_lcb": ["horse_ts_lcb"],
    "horse_glicko_margin": ["horse_glicko_margin"],
    "horse_glicko2": ["horse_glicko2"],
}


def apply_overrides(pairs: list[str]) -> None:
    for pair in pairs:
        name, _, raw = pair.partition("=")
        value = ast.literal_eval(raw)
        if hasattr(ratings, name):
            setattr(ratings, name, value)
        else:
            setattr(config, name, value)


def load_history() -> pd.DataFrame:
    """Development-period processed history, in event order."""
    import pyarrow.parquet as pq
    present = set(pq.ParquetFile(STORE / "processed.parquet").schema_arrow.names)
    frame = pd.read_parquet(STORE / "processed.parquet", columns=[c for c in INPUTS if c in present])
    frame["race_date"] = pd.to_datetime(frame["race_date"])
    frame = frame[development_mask(frame["race_date"])].reset_index(drop=True)
    return FeatureStore._with_dist_category(frame)


def compute(frame: pd.DataFrame, systems: tuple[str, ...] = ("elo", "glicko")) -> pd.DataFrame:
    out = frame.copy()
    if "elo" in systems:
        out = ratings.compute_elo_features(out)
    if "glicko" in systems:
        out = ratings.compute_glicko_features(out)
    return out


class Scorer:
    """Within-race conditional logit on races with one winner and full SP."""

    def __init__(self, frame: pd.DataFrame):
        odds = pd.to_numeric(frame["odds"], errors="coerce")
        won = pd.to_numeric(frame["won"], errors="coerce").fillna(0)
        race = frame["race_id"]
        ok = (odds.gt(1).groupby(race).transform("all")
              & won.groupby(race).transform("sum").eq(1))
        sub = frame.loc[ok]
        order = np.argsort(sub["race_id"].to_numpy(), kind="stable")
        self.rows = np.flatnonzero(ok.to_numpy())[order]
        rid = sub["race_id"].to_numpy()[order]
        self.starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1]
        self.sizes = np.diff(np.r_[self.starts, len(rid)])
        self.race_of = np.repeat(np.arange(len(self.starts)), self.sizes)
        self.won = won.to_numpy()[ok.to_numpy()][order]
        imp = 1.0 / odds.to_numpy()[ok.to_numpy()][order]
        self.mkt = imp / np.add.reduceat(imp, self.starts)[self.race_of]
        days = pd.to_datetime(sub["race_date"]).to_numpy()[order][self.starts].astype("int64")
        self.fit = days < np.quantile(days, 0.6)
        self.n_test = int((~self.fit).sum())

    def _standardise(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)[self.rows]
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)

    def _nll(self, beta, X, mask):
        z = X @ beta
        zmax = np.maximum.reduceat(z, self.starts)[self.race_of]
        ez = np.exp(z - zmax)
        denom = np.add.reduceat(ez, self.starts)
        p = ez / denom[self.race_of]
        ll = np.add.reduceat(z * self.won, self.starts) - (np.log(denom) + zmax[self.starts])
        grad_rows = (self.won - p)[:, None] * X
        grad = np.add.reduceat(grad_rows, self.starts, axis=0)
        w = mask.astype(float)
        return -(ll * w).sum() / w.sum(), -(grad * w[:, None]).sum(0) / w.sum()

    def fit_score(self, X: np.ndarray) -> tuple[float, float, np.ndarray]:
        beta = minimize(self._nll, np.zeros(X.shape[1]), args=(X, self.fit), jac=True,
                        method="L-BFGS-B").x
        return self._nll(beta, X, self.fit)[0], self._nll(beta, X, ~self.fit)[0], beta

    def evaluate(self, frame: pd.DataFrame, models: dict[str, list[str]]) -> pd.DataFrame:
        logm = np.log(self.mkt)[:, None]
        mkt_fit, mkt_test, _ = self.fit_score(logm)
        resid = self.won - self.mkt
        rows = []
        for name, cols in models.items():
            if not all(c in frame.columns for c in cols):
                continue
            X = np.column_stack([self._standardise(frame[c].to_numpy()) for c in cols])
            fit_ll, test_ll, _ = self.fit_score(X)
            _, joint_test, beta = self.fit_score(np.column_stack([logm, X]))
            centred = X[:, 0] - (np.add.reduceat(X[:, 0], self.starts) / self.sizes)[self.race_of]
            rows.append({
                "model": name, "fit_ll": fit_ll, "test_ll": test_ll,
                "market_gain": mkt_test - joint_test,
                "corr_resid": float(np.corrcoef(centred, resid)[0, 1]),
            })
        out = pd.DataFrame(rows)
        out.attrs.update(market_test=mkt_test, uniform_test=float(np.log(self.sizes[~self.fit]).mean()),
                         n_test=self.n_test)
        return out


def stored_frame(history: pd.DataFrame) -> pd.DataFrame:
    """The current store's rating columns aligned to ``history``."""
    import pyarrow.parquet as pq
    present = set(pq.ParquetFile(STORE / "featured.parquet").schema_arrow.names)
    wanted = {c for cols in [*ELO_MODELS.values(), *GLICKO_MODELS.values()] for c in cols}
    cols = ["race_id", "horse_name"] + sorted(wanted & present)
    stored = pd.read_parquet(STORE / "featured.parquet", columns=cols)
    return history.merge(stored, on=["race_id", "horse_name"], how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--systems", default="elo,glicko")
    parser.add_argument("--stored", action="store_true", help="score the store's columns instead")
    args = parser.parse_args()
    apply_overrides(args.set)

    history = load_history()
    rated = stored_frame(history) if args.stored else compute(history, tuple(args.systems.split(",")))
    scorer = Scorer(rated)
    table = scorer.evaluate(rated, {**ELO_MODELS, **GLICKO_MODELS})
    print(f"{scorer.n_test} test races; uniform {table.attrs['uniform_test']:.5f}, "
          f"market (SP) {table.attrs['market_test']:.5f}")
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 120):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
