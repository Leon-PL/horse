"""Learn from the whole finishing order, not just the winner (experiment 057).

Three arms on identical inputs (``win_selected_056.txt``) and folds:

1. ``production``: the production win learner, cross-entropy on soft labels;
2. ``softmax_winner``: a race softmax with the same soft target, winner stage only;
3. ``exploded``: the rank-ordered softmax over winner, second and third, later
   stages weighted ``(1, r, r^2)``.

Arm 2 against arm 1 isolates the change of objective; arm 3 against arm 2
isolates the extra finishing-order information (``src/race_objectives.py``).

``r`` is chosen on the selection quarters 2024Q1-2025Q2 at screen capacity,
as the grid value with the lowest pooled race log loss, and written to
``registration_057.json`` before any confirmation fit. ``confirm`` refuses to
run without it.

    python -m scripts.exploded_softmax_057 dev
    python -m scripts.exploded_softmax_057 confirm
    python -m scripts.exploded_softmax_057 report
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
from scripts.model_ideas import (OUT, THREADS, Cache, baseline, compare, print_summary, production_inputs,  # noqa: E402
                                 run_arm, save_json, setup_logging, store_columns)
from src import feature_selection as fs  # noqa: E402
from src.race_objectives import ExplodedSoftmax, fit_boosted, race_softmax, raw_scores  # noqa: E402

R_GRID = [0.25, 0.5, 1.0]
REGISTRATION = OUT / "registration_057.json"


def stage_weights(r: float) -> tuple[float, ...]:
    return (1.0,) if r == 0 else (1.0, r, r * r)


def arm_name(r: float) -> str:
    return "softmax_winner" if r == 0 else f"exploded_r{r:g}"


def make_predict(cache: Cache, names: list[str], finish: np.ndarray, r: float):
    t = cache.targets
    weights_stage = stage_weights(r)

    def predict(train, test, capacity):
        objective = ExplodedSoftmax(t.race_ids[train], finish[train], t.soft_y[train],
                                    fs.recency_weights(t.race_dates[train]), weights_stage)
        model = fit_boosted(cache.take(train, names), t.won[train].astype(np.float64), t.race_ids[train],
                            t.race_dates[train], objective=objective.objective, metric=objective.metric,
                            capacity=capacity, n_jobs=THREADS)
        bounds = np.r_[0, np.cumsum(fs.race_groups(t.race_ids[test]))]
        return race_softmax(raw_scores(model, cache.take(test, names)), bounds)
    return predict


def finish_positions() -> np.ndarray:
    return store_columns(["finish_position"]).finish_position.to_numpy(np.float64)


def dev(_args) -> None:
    cache = Cache()
    names = production_inputs(cache)
    finish = finish_positions()
    baseline(cache, "dev")
    for r in [0.0, *R_GRID]:
        run_arm(cache, "dev", arm_name(r), make_predict(cache, names, finish, r))
    arms = ["production", arm_name(0.0), *(arm_name(r) for r in R_GRID)]
    summary = compare("dev", arms, [(a, "production") for a in arms[1:]] +
                      [(arm_name(r), arm_name(0.0)) for r in R_GRID])
    print_summary(summary, "057 development screen")
    chosen = min(R_GRID, key=lambda r: summary["arms"][arm_name(r)]["race_ll"])
    if REGISTRATION.exists():
        registered = json.loads(REGISTRATION.read_text(encoding="utf-8"))
        if registered["r"] != chosen:
            raise SystemExit(f"Registration already fixed r={registered['r']}; refusing to change it to {chosen}")
        return
    save_json(REGISTRATION, {
        "experiment": "057", "rule": "lowest pooled selection-quarter race LL over R_GRID, screen capacity",
        "r_grid": R_GRID, "r": chosen, "stage_weights": stage_weights(chosen),
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dev": summary})
    print(f"\nRegistered r = {chosen}: stage weights {stage_weights(chosen)} -> {REGISTRATION}")


def confirm(_args) -> None:
    if not REGISTRATION.exists():
        raise SystemExit("Run the development screen first: it fixes the stage weights.")
    r = json.loads(REGISTRATION.read_text(encoding="utf-8"))["r"]
    cache = Cache()
    names = production_inputs(cache)
    finish = finish_positions()
    baseline(cache, "confirm")
    run_arm(cache, "confirm", "softmax_winner", make_predict(cache, names, finish, 0.0))
    run_arm(cache, "confirm", "exploded", make_predict(cache, names, finish, r))
    report(_args)


def report(_args) -> dict:
    summary = compare("confirm", ["production", "softmax_winner", "exploded"],
                      [("softmax_winner", "production"), ("exploded", "softmax_winner"), ("exploded", "production")])
    summary["registration"] = json.loads(REGISTRATION.read_text(encoding="utf-8"))["stage_weights"]
    save_json(OUT / "summary_057.json", summary)
    print_summary(summary, f"057 confirmation, exploded stage weights {summary['registration']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["dev", "confirm", "report"])
    args = parser.parse_args()
    setup_logging()
    {"dev": dev, "confirm": confirm, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
