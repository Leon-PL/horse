"""Screen the controllable 055 misses against out-of-fold predictions (experiment 055).

055 (docs/research/MODEL_ERROR_REVIEW_055.md) found three model-side patterns among the
054 model's worst misses that public data could, in principle, correct:

- debutants are over-rated, most of all from big yards;
- the last run is missing from our DB (abroad or point-to-point), although its
  finishing position is in the pre-race form string;
- ratings carry across disciplines (a jumper running on the Flat, and the
  reverse).

Before building any of them into the store, this sizes each one as a
race-wise conditional-logit adjustment to the 049 ``general`` arm's
out-of-fold predictions. That arm is the 126 inputs with the production recipe
on 2025Q3–2026Q2. Each adjustment is fitted on earlier quarters only and
scored on the next one (expanding, never in-sample).

Every candidate is built from ``races.db`` using races on strictly earlier
dates. The trainer and sire debut record is a win rate, actual/expected
against prior SPs, and the market's usual respect for that yard's or sire's
newcomers.

    python -m scripts.screen_controllable_055
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_features import CONFIRM_QUARTERS, OUT as CACHE_DIR  # noqa: E402
from scripts.specialist_models import OUT as SPECIALIST_DIR  # noqa: E402
from src.utils import normalise_horse_key  # noqa: E402

OUT = ROOT / "data/research/controllable_055"
SHRINK = 20.0       # pseudo-runs behind each trainer/sire debut rate
RIDGE = 1e-4


def history() -> pd.DataFrame:
    con = sqlite3.connect(ROOT / "data/races.db")
    r = pd.read_sql("select race_id, race_date, off_time, race_type, race_name, horse_name, horse_id, trainer,"
                    " odds, won, days_since_last_run, form from results", con)
    ped = pd.read_sql("select horse_id, sire from horse_pedigree", con)
    r = r.merge(ped, on="horse_id", how="left")
    r["race_date"] = pd.to_datetime(r.race_date)
    r["race_id"] = r.race_id.astype(str)
    r["horse_name"] = r.horse_name.map(normalise_horse_key)
    r = r.sort_values(["race_date", "off_time", "race_id"]).reset_index(drop=True)
    bumper = r.race_name.str.contains(r"NH Flat|Bumper|Flat Race", case=False, regex=True) & r.race_type.eq("Flat")
    r["flat"] = r.race_type.eq("Flat") & ~bumper
    r["jumps"] = r.race_type.isin(["Hurdle", "Chase"])
    r["code"] = np.where(r.flat, "F", "N")
    inv = 1 / r.odds.where(r.odds > 1)
    r["p_sp"] = inv / inv.groupby(r.race_id).transform("sum")
    r["field"] = r.groupby("race_id").race_id.transform("size")
    horse = r.horse_id.fillna(r.horse_name)
    by_horse = r.groupby(horse)
    r["db_prev"] = by_horse.cumcount()
    r["db_gap"] = (r.race_date - by_horse.race_date.shift(1)).dt.days
    r["prev_jumps"] = by_horse.jumps.cumsum() - r.jumps
    r["prev_flat"] = by_horse.flat.cumsum() - r.flat
    r["last_flat"] = by_horse.flat.shift(1)
    sl = pd.to_numeric(r.days_since_last_run, errors="coerce")
    r["debut"] = (r.db_prev == 0) & sl.fillna(0).eq(0)
    # an SL-dated last run inside the DB's coverage that the DB does not hold
    implied = r.race_date - pd.to_timedelta(sl, unit="D")
    r["last_offdb"] = (sl > 0) & (implied >= r.race_date.min()) & (r.db_gap.isna() | (sl < r.db_gap - 3))
    last = r.form.fillna("").astype(str).str.replace(r"[-/ ]", "", regex=True).str[-1:]
    r["last_pos"] = pd.to_numeric(last.where(last.str.isdigit()), errors="coerce").replace(0, 10)
    return r


def debut_record(r: pd.DataFrame, keys: list[str], prefix: str, prior: float) -> pd.DataFrame:
    """Shrunk record of earlier debutants per *keys*, using strictly earlier dates only."""
    flag = r.debut & r[keys].notna().all(axis=1)
    sub = r.loc[flag, keys + ["race_date", "won", "p_sp", "field"]].assign(mk=lambda d: d.p_sp * d.field)
    daily = sub.groupby(keys + ["race_date"]).agg(n=("won", "size"), w=("won", "sum"), e=("p_sp", "sum"),
                                                  mk=("mk", "sum")).reset_index().sort_values(keys + ["race_date"])
    for c in ["n", "w", "e", "mk"]:
        daily[c] = daily.groupby(keys)[c].cumsum() - daily[c]
    got = r.loc[flag, keys + ["race_date"]].merge(daily, on=keys + ["race_date"], how="left")
    got.index = r.index[flag]
    out = pd.DataFrame(index=r.index)
    out[f"{prefix}_wr"] = (got.w + SHRINK * prior) / (got.n + SHRINK)
    out[f"{prefix}_ae"] = (got.w + SHRINK * prior) / (got.e + SHRINK * prior)
    out[f"{prefix}_mkt"] = (got.mk + SHRINK) / (got.n + SHRINK)
    return out


def candidates(r: pd.DataFrame) -> pd.DataFrame:
    prior = float(r.loc[r.debut, "won"].mean())
    deb = pd.concat([debut_record(r, ["trainer", "code"], "tr", prior),
                     debut_record(r, ["sire", "code"], "sire", prior)], axis=1)
    x = pd.DataFrame(index=r.index)
    d = r.debut.astype(float)
    x["debut"] = d
    for col in deb.columns:
        v = deb[col]
        x[f"debut_x_{col}"] = (d * (v - v[r.debut].mean()) / v[r.debut].std()).fillna(0)
    off = r.last_offdb.astype(float)
    x["offdb_won"] = off * r.last_pos.eq(1)
    x["offdb_2nd"] = off * r.last_pos.eq(2)
    x["offdb_4plus"] = off * r.last_pos.ge(4)
    x["flat_mainly_jumps"] = (r.flat & (r.prev_jumps >= 2) & (r.prev_jumps > r.prev_flat)).astype(float)
    x["jumps_last_flat"] = (r.jumps & r.last_flat.eq(True)).astype(float)
    return pd.concat([r[["race_id", "horse_name"]], x], axis=1)


def oof_frame(cands: pd.DataFrame) -> pd.DataFrame:
    targets = pd.read_parquet(CACHE_DIR / "targets.parquet", columns=["race_id", "horse_name"])
    targets["row"] = np.arange(len(targets))
    targets["horse_name"] = targets.horse_name.map(normalise_horse_key)
    arm = pd.concat([pd.read_parquet(SPECIALIST_DIR / f"general_{q}.parquet").assign(fold=q)
                     for q in CONFIRM_QUARTERS], ignore_index=True)
    arm = arm.merge(targets[["row", "horse_name"]], on="row", validate="one_to_one")
    arm["race_id"] = arm.race_id.astype(str)
    frame = arm.merge(cands, on=["race_id", "horse_name"], how="left", validate="one_to_one")
    if frame.debut.isna().any():
        raise ValueError(f"{int(frame.debut.isna().sum())} out-of-fold rows have no candidate row")
    return frame.sort_values(["race_id", "row"]).reset_index(drop=True)


class RaceLogit:
    """log p' = log p_model + X beta, renormalised within each race."""

    def __init__(self, frame: pd.DataFrame):
        p = frame.model / frame.groupby("race_id").model.transform("sum")
        self.log_p = np.log(p.clip(1e-12)).to_numpy()
        self.won = frame.won.to_numpy(np.float64)
        _, self.race = np.unique(frame.race_id.to_numpy(), return_inverse=True)

    def loss(self, x: np.ndarray, beta: np.ndarray, mask: np.ndarray) -> float:
        s = self.log_p[mask] + x[mask] @ beta
        e = np.exp(s - s.max())
        race = self.race[mask]
        den = np.bincount(race, weights=e)
        num = np.bincount(race, weights=e * self.won[mask])
        keep = den > 0
        return float(-np.log(num[keep] / den[keep]).mean())

    def fit(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        objective = lambda b: self.loss(x, b, mask) + RIDGE * float(b @ b)
        return minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B").x


SETS = {
    "debutant shrink (1 parameter)": ["debut"],
    "debutant trainer/sire record": ["debut", "debut_x_tr_wr", "debut_x_tr_ae", "debut_x_tr_mkt",
                                     "debut_x_sire_wr", "debut_x_sire_ae", "debut_x_sire_mkt"],
    "off-DB last run and its position": ["offdb_won", "offdb_2nd", "offdb_4plus"],
    "discipline carry-over": ["flat_mainly_jumps", "jumps_last_flat"],
}


def main() -> None:
    frame = oof_frame(candidates(history()))
    model = RaceLogit(frame)
    folds = frame.fold.to_numpy()
    p = frame.model / frame.groupby("race_id").model.transform("sum")
    segments = {name: frame[col].ne(0) for name, col in [("debutants", "debut"), ("off-DB last run", "offdb_won")]}
    segments["off-DB last run"] = frame[["offdb_won", "offdb_2nd", "offdb_4plus"]].ne(0).any(axis=1)
    summary = {"races": int(frame.race_id.nunique()), "runners": int(len(frame)), "segments": {}, "screens": {}}
    for name, mask in segments.items():
        summary["segments"][name] = {"runners": int(mask.sum()), "wins": int(frame.won[mask].sum()),
                                     "model_expected": round(float(p[mask].sum()), 1)}
    for name, cols in SETS.items():
        x = frame[cols].to_numpy(np.float64)
        per_quarter = {}
        for i in range(1, len(CONFIRM_QUARTERS)):
            train, test = np.isin(folds, CONFIRM_QUARTERS[:i]), folds == CONFIRM_QUARTERS[i]
            beta = model.fit(x, train)
            per_quarter[CONFIRM_QUARTERS[i]] = model.loss(x, beta, test) - model.loss(x, np.zeros(len(cols)), test)
        summary["screens"][name] = {"per_quarter": {q: round(v, 5) for q, v in per_quarter.items()},
                                    "mean": round(float(np.mean(list(per_quarter.values()))), 5),
                                    "quarters_better": f"{sum(v < 0 for v in per_quarter.values())}/{len(per_quarter)}"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{summary['races']:,} races, {summary['runners']:,} runners (049 general arm, out of fold)")
    for name, s in summary["segments"].items():
        print(f"  {name}: {s['runners']} runners, {s['wins']} wins, model expected {s['model_expected']}")
    print("Race log-loss change when fitted on earlier quarters only (negative = better):")
    for name, s in summary["screens"].items():
        print(f"  {name:36s} mean {s['mean']:+.5f}  better {s['quarters_better']}  {s['per_quarter']}")


if __name__ == "__main__":
    main()
