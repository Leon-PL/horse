"""Use the finishing order beyond third place (experiment 061, follow-up to 057).

057 confirmed the exploded softmax over the first three places, with stage
weights ``(1, 0.25, 0.0625)``. Its development screen showed that weighting
lower places more heavily hurts. This screen asks whether more of the order
helps once the lower places are treated as noisier rather than less
important: each stage's softmax is flattened by a temperature below 1
(``ExplodedSoftmax(stage_temps=...)``).

Stage *k* (0-based) gets weight ``r**k`` and temperature ``t**k``. Depth
``None`` means every placed finisher; non-finishers stay in every stage's
field and are never chosen, so they rank below every finisher.

The grid and the rule are fixed here, before any fit: the arm with the lowest
pooled race log loss on the selection quarters 2024Q1-2025Q2 at screen
capacity is written to ``registration_061.json``. If that is 057's own arm,
nothing goes forward. ``exploded_r0.25`` reuses 057's saved development
predictions.

    python -m scripts.exploded_depth_061 dev
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.exploded_softmax_057 import finish_positions  # noqa: E402
from scripts.model_ideas import (OUT, THREADS, Cache, baseline, compare, print_summary, production_inputs,  # noqa: E402
                                 run_arm, save_json, setup_logging)
from src import feature_selection as fs  # noqa: E402
from src.race_objectives import ExplodedSoftmax, fit_boosted, race_softmax, raw_scores  # noqa: E402

REFERENCE = "exploded_r0.25"
#: name -> (depth, weight ratio r, temperature ratio t)
ARMS = {
    REFERENCE: (3, 0.25, 1.0),
    "deep_r0.25": (None, 0.25, 1.0),
    "top3_t0.8": (3, 1.0, 0.8),
    "deep_t0.8": (None, 1.0, 0.8),
    "deep_t0.65": (None, 1.0, 0.65),
    "deep_t0.5": (None, 1.0, 0.5),
}
REGISTRATION = OUT / "registration_061.json"


def stages(depth: int, r: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(depth, dtype=np.float64)
    return r ** k, t ** k


def make_predict(cache: Cache, names: list[str], finish: np.ndarray, spec: tuple):
    t = cache.targets
    depth, r, temp = spec
    depth = int(fs.race_groups(t.race_ids).max()) if depth is None else depth
    weights_stage, temps_stage = stages(depth, r, temp)

    def predict(train, test, capacity):
        objective = ExplodedSoftmax(t.race_ids[train], finish[train], t.soft_y[train],
                                    fs.recency_weights(t.race_dates[train]), weights_stage, temps_stage)
        model = fit_boosted(cache.take(train, names), t.won[train].astype(np.float64), t.race_ids[train],
                            t.race_dates[train], objective=objective.objective, metric=objective.metric,
                            capacity=capacity, n_jobs=THREADS)
        bounds = np.r_[0, np.cumsum(fs.race_groups(t.race_ids[test]))]
        return race_softmax(raw_scores(model, cache.take(test, names)), bounds)
    return predict


def dev(_args) -> None:
    cache = Cache()
    names = production_inputs(cache)
    finish = finish_positions()
    baseline(cache, "dev")
    for name, spec in ARMS.items():
        run_arm(cache, "dev", name, make_predict(cache, names, finish, spec))
    arms = ["production", *ARMS]
    summary = compare("dev", arms, [(a, REFERENCE) for a in ARMS if a != REFERENCE] +
                      [(a, "production") for a in ARMS])
    print_summary(summary, "061 development screen: depth and flattening")
    chosen = min(ARMS, key=lambda a: summary["arms"][a]["race_ll"])
    if REGISTRATION.exists():
        registered = json.loads(REGISTRATION.read_text(encoding="utf-8"))
        if registered["arm"] != chosen:
            raise SystemExit(f"Registration already fixed {registered['arm']}; refusing to change it to {chosen}")
        return
    save_json(REGISTRATION, {
        "experiment": "061", "rule": "lowest pooled selection-quarter race LL over ARMS, screen capacity",
        "arms": {k: list(v) for k, v in ARMS.items()}, "arm": chosen, "goes_forward": chosen != REFERENCE,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dev": summary})
    print(f"\nRegistered {chosen} -> {REGISTRATION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["dev"])
    args = parser.parse_args()
    setup_logging()
    {"dev": dev}[args.stage](args)


if __name__ == "__main__":
    main()
