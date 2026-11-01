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
    .venv/Scripts/python.exe scripts/screen_ratings.py --systems pl --set PL_TOP_K=4

``--set NAME=VALUE`` patches a ``src.ratings`` module binding (K_BASE,
GLICKO_EVIDENCE_POWER, ...) or, failing that, a ``config`` flag
(TRUESKILL_ENABLED, ...). ``PL_*`` names set the research-only Plackett–Luce
sweep defined here.
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

# ── Plackett–Luce (research only; not in the pipeline) ─────────────────
# A finishing order is a sequence of choices of who finishes next among the
# runners still unplaced. Two online fits of that likelihood:
#   "bayes": Weng & Lin (2011), a Gaussian skill (mu, sigma²) per horse;
#   "sgd":   one log-strength per horse, moved along the race's gradient
#            with an experience-decayed step (Elo's K schedule).
PL_MODE = "sgd"     # best in the 045 screen; "bayes" peaked at 2.055 fit LL
PL_MU = 25.0
PL_SIGMA = 25.0 / 3.0
PL_BETA = 25.0 / 6.0
PL_C = 2.0          # sigma growth per idle month, capped at PL_SIGMA (as TrueSkill)
PL_SCALE = "sum"    # field scale c: "sum" of variances (Weng & Lin) or their "mean"
PL_KAPPA = 1e-4     # floor on the per-race variance shrink factor
PL_K_BASE = 2.0     # sgd step for a debutant, in log-strength per unit gradient
PL_K_MIN = 0.5      # sgd step floor
PL_K_DECAY = 0.1    # sgd step decay per run
PL_TOP_K = 0        # places scored as choices; 0 = every finisher
PL_DISCOUNT = 0.85  # sgd: strength scaled by DISCOUNT^(place-1) at each later choice (Benter)
PL_MODELS = {
    "horse_pl_mu": ["horse_pl_mu"],
    "horse_pl_lcb": ["horse_pl_lcb"],
    "ts_mu + pl_mu": ["horse_ts_mu", "horse_pl_mu"],
}


def _pl_choices(score: np.ndarray, pos: np.ndarray, chosen: np.ndarray, discount: float = 1.0):
    """Choice probabilities ``P[s, i]`` of each runner at each scored place,
    who was placed there, and each choice's weight (dead heats share a choice;
    with ``discount`` < 1 later places use a damped strength and carry
    that factor into the gradient). Runners not ``chosen`` (non-finishers,
    places beyond top-k) only sit in the risk sets of the choices ahead."""
    q = np.flatnonzero(chosen)
    lam = discount ** (pos[q] - 1.0)
    e = np.exp(lam[:, None] * (score - score.max())[None, :])
    risk = pos[None, :] >= pos[q][:, None]           # [choice, runner] still unplaced
    P = risk * e / (risk * e).sum(axis=1, keepdims=True)
    picked = np.zeros_like(P)
    picked[np.arange(len(q)), q] = 1.0
    w = lam / (pos[q][:, None] == pos[None, :]).sum(axis=1)
    return P, picked, w


def _pl_bayes_update(mu, sig2, pos, chosen):
    total = sig2 + PL_BETA ** 2
    c = np.sqrt(total.sum() if PL_SCALE == "sum" else total.mean())
    P, picked, w = _pl_choices(mu / c, pos, chosen)
    omega = (sig2 / c) * (w[:, None] * (picked - P)).sum(axis=0)
    info = (w[:, None] * P * (1.0 - P)).sum(axis=0)
    shrink = (np.sqrt(sig2) / c) * (sig2 / c ** 2) * info
    return mu + omega, sig2 * np.maximum(1.0 - shrink, PL_KAPPA)


def _pl_sgd_update(theta, n_runs, pos, chosen):
    P, picked, w = _pl_choices(theta, pos, chosen, PL_DISCOUNT)
    k = PL_K_MIN + (PL_K_BASE - PL_K_MIN) * np.exp(-PL_K_DECAY * n_runs)
    return theta + k * (w[:, None] * (picked - P)).sum(axis=0), n_runs + 1


def add_plackett_luce(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-race Plackett–Luce strength (and, for "bayes", sigma and lcb),
    swept in event order."""
    bayes = PL_MODE == "bayes"
    keys = ratings._entity_keys(frame, id_col="horse_id", name_col="horse_name", prefix="horse")
    fp = pd.to_numeric(frame["finish_position"], errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(frame["race_date"]).to_numpy()
    order = np.argsort(ratings._event_sort_key(frame).to_numpy(), kind="stable")
    races = pd.unique(frame["race_id"].to_numpy()[order])
    groups = frame.groupby("race_id", sort=False).indices

    # key -> (mu, sigma², last race date) for "bayes"; (theta, runs, date) for "sgd"
    state: dict = {}
    a_col = np.full(len(frame), np.nan)
    b_col = np.full(len(frame), np.nan)
    one_day = np.timedelta64(1, "D")
    for race_id in races:
        idx = groups[race_id]
        idx = idx[[pd.notna(keys[i]) for i in idx]]
        if not len(idx):
            continue
        day = dates[idx[0]]
        a = np.full(len(idx), PL_MU if bayes else 0.0)
        b = np.full(len(idx), PL_SIGMA ** 2 if bayes else 0.0)
        for j, i in enumerate(idx):
            prior = state.get(keys[i])
            if prior is None:
                continue
            a[j], b[j] = prior[0], prior[1]
            if bayes:
                idle = max((day - prior[2]) / one_day, 0.0) / 30.44
                b[j] = min(b[j] + PL_C ** 2 * idle, PL_SIGMA ** 2)
        a_col[idx], b_col[idx] = a, b

        part, pos = ratings._race_outcome(fp[idx], ratings.DNF_AS_LOSS)
        chosen = (fp[idx] > 0) & part
        if PL_TOP_K:
            chosen &= fp[idx] <= PL_TOP_K
        if part.sum() < 2 or not chosen.any():
            continue
        update = _pl_bayes_update if bayes else _pl_sgd_update
        new_a, new_b = update(a[part], b[part], pos[part], chosen[part])
        for i, x, y in zip(idx[part], new_a, new_b):
            state[keys[i]] = (float(x), float(y), day)

    out = frame.copy()
    out["horse_pl_mu"] = a_col
    if bayes:
        out["horse_pl_sigma"] = np.sqrt(b_col)
        out["horse_pl_lcb"] = a_col - 3.0 * np.sqrt(b_col)
    return out


def apply_overrides(pairs: list[str]) -> None:
    for pair in pairs:
        name, _, raw = pair.partition("=")
        value = ast.literal_eval(raw)
        if name.startswith("PL_"):
            globals()[name] = value
        elif hasattr(ratings, name):
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
    if "pl" in systems:
        out = add_plackett_luce(out)
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
    parser.add_argument("--systems", default="elo,glicko,pl")
    parser.add_argument("--stored", action="store_true", help="score the store's columns instead")
    args = parser.parse_args()
    apply_overrides(args.set)

    history = load_history()
    rated = stored_frame(history) if args.stored else compute(history, tuple(args.systems.split(",")))
    scorer = Scorer(rated)
    table = scorer.evaluate(rated, {**ELO_MODELS, **GLICKO_MODELS, **PL_MODELS})
    print(f"{scorer.n_test} test races; uniform {table.attrs['uniform_test']:.5f}, "
          f"market (SP) {table.attrs['market_test']:.5f}")
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 120):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
