"""Do race-type or course specialists get closer to the market? (experiment 049)

Fits the production win recipe on the 048 inputs (``feature_sets/win_selected_048.txt``)
three ways on each confirmation quarter (2025Q3-2026Q2, purged expanding folds):

* ``general``: trained on every earlier development race, as production is;
* ``hurdle``: trained only on earlier hurdle races;
* ``wolverhampton``: trained only on earlier Wolverhampton races.

Each specialist is scored against the general model on the same test races (its
own subset), and both against the market (BSP-implied, else SP, normalised per
race). A second number asks whether the model adds anything *to* the market: a
two-weight conditional logit ``p ∝ market^a · model^b`` fitted on earlier
quarters' out-of-sample predictions and scored on the next quarter.

Reads the 048 cache, so the store is not rebuilt::

    python -m scripts.specialist_models          # fit (resumable), then report
    python -m scripts.specialist_models --report # report from saved predictions
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_features import CONFIRM_QUARTERS, OUT as CACHE_DIR, SELECTED, Cache, read_names  # noqa: E402
from src import feature_selection as fs  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.forecast_evaluation import paired_interval  # noqa: E402

OUT = ROOT / "data/research/specialist_049"
SUBSETS = {
    "hurdle": ("race_type", "Hurdle"),
    "wolverhampton": ("track", "Wolverhampton"),
}

logger = logging.getLogger("specialist_models")


def _context(cache: Cache) -> pd.DataFrame:
    """Race type and course for every cached row, read by store row."""
    rows = pd.read_parquet(CACHE_DIR / "targets.parquet", columns=["store_row"]).store_row.to_numpy()
    table = pq.read_table(FeatureStore().featured_path, columns=["race_type", "track"]).to_pandas()
    context = table.iloc[rows].reset_index(drop=True)
    if len(context) != len(cache.targets.won):
        raise ValueError("Context rows do not line up with the cache")
    return context


def _predict(cache: Cache, features: list[str], train: np.ndarray, test: np.ndarray) -> np.ndarray:
    t = cache.targets
    model = fs.fit_win_model(cache.take(train, features), t.won[train], t.soft_y[train],
                             fs.recency_weights(t.race_dates[train]), t.race_ids[train],
                             t.race_dates[train], capacity="native")
    raw = model.predict_proba(cache.take(test, features))[:, 1]
    return fs.race_normalise(raw, fs.race_groups(t.race_ids[test]))


def fit(cache: Cache, context: pd.DataFrame) -> None:
    features = read_names(SELECTED)
    t = cache.targets
    OUT.mkdir(parents=True, exist_ok=True)
    masks = {name: context[col].eq(value).to_numpy() for name, (col, value) in SUBSETS.items()}
    for fold in cache.fold_data(CONFIRM_QUARTERS):
        for arm in [*SUBSETS, "general"]:
            path = OUT / f"{arm}_{fold.label}.parquet"
            if path.exists():
                continue
            started = time.monotonic()
            train, test = fold.train_rows, fold.test_rows
            if arm != "general":
                train, test = train[masks[arm][train]], test[masks[arm][test]]
            p = _predict(cache, features, train, test)
            pd.DataFrame({
                "row": test, "race_id": t.race_ids[test], "race_date": t.race_dates[test],
                "won": t.won[test], "market": t.market[test], "model": p,
                "race_type": context.race_type.to_numpy()[test], "track": context.track.to_numpy()[test],
            }).to_parquet(path, index=False)
            logger.info("%s %s: %d train / %d test rows (%.0fs)", arm, fold.label, len(train), len(test),
                        time.monotonic() - started)


# ------------------------------------------------------------------ scoring --

def _race_frame(runners: pd.DataFrame, column: str) -> pd.Series:
    """Per-race log loss of the race-normalised *column* (runners sorted by race)."""
    groups = fs.race_groups(runners.race_id.to_numpy())
    p = fs.race_normalise(runners[column].to_numpy(np.float64), groups)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    loss = -np.add.reduceat(runners.won.to_numpy() * np.log(np.clip(p, 1e-12, None)), starts)
    return pd.Series(loss, index=runners.race_id.to_numpy()[starts])


def _blend_loss(weights, log_m, log_q, won, starts):
    score = weights[0] * log_m + weights[1] * log_q
    top = np.maximum.reduceat(score, starts)
    norm = np.log(np.add.reduceat(np.exp(score - np.repeat(top, np.diff(np.r_[starts, len(score)]))), starts)) + top
    return -(np.add.reduceat(won * score, starts) - norm)


def _blend_inputs(runners: pd.DataFrame):
    groups = fs.race_groups(runners.race_id.to_numpy())
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    log_m = np.log(np.clip(fs.race_normalise(runners.market.to_numpy(np.float64), groups), 1e-12, None))
    log_q = np.log(np.clip(runners.model.to_numpy(np.float64), 1e-12, None))
    return log_m, log_q, runners.won.to_numpy(np.float64), starts


def blended(runners: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Per-race loss of the market+model blend, weights fitted on earlier quarters only."""
    losses, fitted = [], []
    for k, quarter in enumerate(CONFIRM_QUARTERS[1:], start=1):
        history = runners[runners.fold.isin(CONFIRM_QUARTERS[:k])]
        current = runners[runners.fold.eq(quarter)]
        args = _blend_inputs(history)
        result = minimize(lambda w: _blend_loss(w, *args).mean(), x0=[1.0, 0.0], method="BFGS")
        log_m, log_q, won, starts = _blend_inputs(current)
        race_ids = current.race_id.to_numpy()[starts]
        losses.append(pd.Series(_blend_loss(result.x, log_m, log_q, won, starts), index=race_ids))
        fitted.append({"quarter": quarter, "market_weight": float(result.x[0]), "model_weight": float(result.x[1])})
    return pd.concat(losses), fitted


def _load(arm: str) -> pd.DataFrame:
    frames = [pd.read_parquet(OUT / f"{arm}_{q}.parquet").assign(fold=q) for q in CONFIRM_QUARTERS]
    return pd.concat(frames, ignore_index=True)


def _interval(delta: pd.Series, dates: pd.Series) -> list[float]:
    return paired_interval(delta.to_numpy(), blocks=dates.astype(str).to_numpy())["ci95"]


def report() -> dict:
    general = _load("general")
    rows, fitted = [], {}
    universes = {"all races": (general, None)}
    for name, (col, value) in SUBSETS.items():
        universes[name] = (general[general[col].eq(value)].reset_index(drop=True), _load(name))
    for name, (base, specialist) in universes.items():
        races = base.drop_duplicates("race_id").set_index("race_id")[["race_date", "fold"]]
        market = _race_frame(base, "market")
        arms = {"general": base} if specialist is None else {"general": base, "specialist": specialist}
        scored = {}
        for arm, runners in arms.items():
            if not np.array_equal(runners.race_id.unique(), base.race_id.unique()):
                raise ValueError(f"{name}/{arm} was not scored on the same races")
            model = _race_frame(runners, "model")
            blend, weights = blended(runners)
            scored[arm] = model
            fitted[f"{name}/{arm}"] = weights
            gap = model - market
            later = blend.index
            blend_gain = blend - market.loc[later]
            rows.append({
                "universe": name, "arm": arm, "races": len(races), "market_ll": market.mean(),
                "model_ll": model.mean(), "gap_to_market": gap.mean(),
                "gap_ci": _interval(gap, races.race_date),
                "quarters_closer": None,
                "blend_gain_vs_market": blend_gain.mean(),
                "blend_ci": _interval(blend_gain, races.race_date.loc[later]),
                "blend_races": len(later),
            })
        if "specialist" in scored:
            delta = scored["specialist"] - scored["general"]
            per_quarter = delta.groupby(races.fold).mean()
            rows[-1]["specialist_minus_general"] = delta.mean()
            rows[-1]["specialist_ci"] = _interval(delta, races.race_date)
            rows[-1]["quarters_closer"] = f"{int((per_quarter < 0).sum())}/{len(per_quarter)}"
            rows[-1]["per_quarter"] = {q: round(float(v), 5) for q, v in per_quarter.items()}
    summary = {"features": str(SELECTED.relative_to(ROOT)), "quarters": CONFIRM_QUARTERS,
               "rows": rows, "blend_weights": fitted}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _print(rows, fitted)
    return summary


def _print(rows: list[dict], fitted: dict) -> None:
    print(f"\nRace log loss on {', '.join(CONFIRM_QUARTERS)} (lower is better; 95% date-clustered intervals)")
    for r in rows:
        line = (f"{r['universe']:>14} {r['arm']:>10}  races {r['races']:>5}  market {r['market_ll']:.4f}  "
                f"model {r['model_ll']:.4f}  gap {r['gap_to_market']:+.4f} "
                f"[{r['gap_ci'][0]:+.4f}, {r['gap_ci'][1]:+.4f}]  blend-vs-market {r['blend_gain_vs_market']:+.4f} "
                f"[{r['blend_ci'][0]:+.4f}, {r['blend_ci'][1]:+.4f}]")
        print(line)
        if "specialist_minus_general" in r:
            print(f"{'':>26}specialist - general {r['specialist_minus_general']:+.4f} "
                  f"[{r['specialist_ci'][0]:+.4f}, {r['specialist_ci'][1]:+.4f}], closer in {r['quarters_closer']} "
                  f"quarters {r['per_quarter']}")
    print("\nBlend weights (market, model) fitted on earlier quarters:")
    for key, weights in fitted.items():
        print(f"  {key}: " + ", ".join(f"{w['quarter']} ({w['market_weight']:.3f}, {w['model_weight']:.3f})"
                                       for w in weights))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="report from saved predictions only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.report:
        cache = Cache()
        fit(cache, _context(cache))
    report()


if __name__ == "__main__":
    main()
