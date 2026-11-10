"""Pool stored component forecasts with the three-seed full 063 average.

    python -m research.meta_learner.pool_average [performance text leeway]

The first ensemble benchmark pooled performance/spread only with a single
full 063 fit, and judged that pool against the three-seed average. The Tier 0
screen suggested performance/spread also improves the average itself. This
check uses the benchmark's own log-pool combiner on verified cached forecasts
(no model fits): the pool for each later quarter is fitted on earlier out-of-fold
quarters only, starting from 2025Q1 and 2025Q2. Text and handicap leeway are
repeated as known nulls.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.meta_learner import first_ensemble as fe
from research.meta_learner import screening
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.evaluation import nested_pool_evaluation

ROOT = fe.ARTIFACTS / "first_ensemble_20261109"
OUT = Path("data/research/meta_learner/screens_20261109")


def stored(name):
    part = pd.read_parquet(ROOT / name)
    part.attrs = {}
    return part


def cached(component, name=None):
    part = screening.cached_forecasts(ROOT, component)
    return part.assign(component=name or component, available=True)


def seed_average():
    """Mean of the three cached seeds from 2025Q1, checked against the benchmark's stored later average."""
    seeds = [cached(c) for c in ("full-063", "full-063-seed-43", "full-063-seed-44")]
    first = seeds[0].loc[seeds[0].fold.isin(seeds[1].fold.unique())].reset_index(drop=True)
    parts = [first] + [p.set_index(screening.KEYS).loc[pd.MultiIndex.from_frame(first[screening.KEYS])].reset_index()
                       for p in seeds[1:]]
    average = fe.average_predictions(parts)
    saved = stored("screen_seed_average.parquet")
    check = average.merge(saved[[*screening.KEYS, "probability"]], on=screening.KEYS, suffixes=("", "_saved"))
    if len(check) != len(saved) or not np.allclose(check.probability, check.probability_saved, rtol=0, atol=1e-12):
        raise ValueError("Recomputed seed average differs from the benchmark's stored average")
    return average


def pooled_runners(frame, parts, purge, shrinkage):
    """Runner probabilities of the pool scored by ``fe.score_pool``, for use as a later base."""
    dates = parts[0].race_date.unique()
    target = frame.loc[frame.race_date.isin(dates)].reset_index(drop=True)
    folds = fe.folds_for(target, fe.LATER, purge, min_train_periods=len(fe.POOL_HISTORY))
    names = tuple(p.component.iloc[0] for p in parts)
    _, runners, _ = nested_pool_evaluation(target, pd.concat(parts, ignore_index=True), names, folds=folds,
                                           purge_days=purge, shrinkage=shrinkage)
    out = runners[[*screening.KEYS, "race_date", "pool_probability"]].rename(columns={"pool_probability": "probability"})
    out.attrs = {}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("components", nargs="*", default=["performance", "text", "leeway"])
    args = parser.parse_args()
    design = fe.read_json(ROOT / "registration.json")
    selected = fe.get_selected(ROOT, design)
    purge, shrinkage = selected.purge_days, design["pool_shrinkage"]
    average = seed_average()
    sources = {"performance": lambda: cached("performance"),
               "text": lambda: stored("text_oof.parquet"),
               "leeway": lambda: stored("leeway-screen_oof.parquet")}
    frame = fe.read_input(ROOT, design, [])
    base_summary, before = fe.score_pool(frame, [average], fe.LATER, purge, shrinkage)
    report = {"reference": "three-seed full 063 average", "average": base_summary,
              "evaluation_status": "Exploratory retrospective comparison on reused development quarters."}
    for name in args.components:
        component = sources[name]()
        summary, after = fe.score_pool(frame, [average, component], fe.LATER, purge, shrinkage)
        change = fe.difference(after, before)
        weights = [f.get("weights") for f in summary.get("pool_fits", [])]
        report[name] = {"summary": summary, "change_vs_average": change}
        runners = pooled_runners(frame, [average, component], purge, shrinkage)
        check = fe.metrics(frame, runners, fe.LATER)[1]
        if not np.allclose(check.loss.to_numpy(), after.loss.to_numpy(), rtol=0, atol=1e-12):
            raise ValueError("Saved pool forecasts do not reproduce the scored pool")
        atomic_parquet(runners, OUT / f"pool_average_{name}.parquet")
        print(f"{name:<12} {json.dumps(change)}  weights {weights}", flush=True)
    atomic_json(OUT / "pool_average.json", report)


if __name__ == "__main__":
    main()
