"""New-information groups as separate corrections on the current stack.

    python -m research.meta_learner.new_information corrections --groups NAME ... [--threads 6]
    python -m research.meta_learner.new_information compare --groups NAME ...

``corrections`` saves out-of-fold Tier 1 corrections (the cheap-screen recipe,
fitted on the seed-42 forecasts of earlier quarters) for each named group and
for three permuted twins of it, for 2025Q1-2026Q2.

``compare`` scores on 2025Q3-2026Q2, starting from the stack: the three-seed
average log-pooled with performance/spread, plus the tracking and same-day
corrections (``specialists.py corrections``). Each group's correction is added
on top, unweighted, and compared with the stack and with the same correction
fitted on its twins.
"""
import argparse

import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from research.meta_learner.specialists import CORRECTIONS as SPECIALISTS, QUARTERS, race_losses
from src.ingestion.storage import atomic_json, atomic_parquet

OUT = sc.OUT
KEYS = s.KEYS
LATER = s.SCORED
TWINS = sc.FOLLOWUP_TWINS


def corrections_file(name):
    return OUT / f"new_information_{name}.parquet"


def corrections(names, threads):
    base = sc.base_frame()
    for name in names:
        values, columns = sc.CANDIDATES[name]()(base)
        frame = s.attach(base, values)
        keep = frame.quarter.isin(QUARTERS).to_numpy()
        out = frame.loc[keep, [*KEYS, "race_date", "quarter"]].reset_index(drop=True)
        variants = {"": frame, **{f"_twin{seed}": s.permute_within(frame, columns, seed=seed) for seed in TWINS}}
        for suffix, data in variants.items():
            fit = s.residual_boost(columns, base="p_single", threads=threads, params=sc.TIER1["params"],
                                   **sc.TIER1["boost"])
            correction, _ = s.walk_forward(data, fit, scored=QUARTERS)
            if not np.isfinite(correction[keep]).all():
                raise ValueError(f"Missing {name}{suffix} correction")
            out[f"c{suffix}"] = correction[keep]
            print(f"{name}{suffix}: {len(columns)} inputs, {[f['seconds'] for f in fit.info]} s per quarter",
                  flush=True)
        atomic_parquet(out, corrections_file(name))


def compare(names):
    frame = sc.pool_frame()
    frame = s.attach(frame, pd.read_parquet(SPECIALISTS, columns=[*KEYS, "c_tracking", "c_same-day-bias"]))
    for name in names:
        stored = pd.read_parquet(corrections_file(name)).drop(columns=["race_date", "quarter"])
        frame = s.attach(frame, stored.rename(columns={c: f"{name}:{c}" for c in stored if c not in KEYS}))
    scored = frame.loc[frame.quarter.isin(LATER)].reset_index(drop=True)
    stack = np.log(scored.p_pool.to_numpy()) + scored.c_tracking.to_numpy() + scored["c_same-day-bias"].to_numpy()
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    loss = {"average": race_losses(scored, np.log(scored.p_average.to_numpy())), "stack": race_losses(scored, stack)}
    report = {"races": len(races), "groups": list(names), "twin_seeds": list(TWINS),
              "evaluation_status": "retrospective screen on reused development quarters",
              "stack_vs_average": s.paired(loss["stack"], loss["average"], races)}
    total = stack.copy()
    for name in names:
        added = loss[f"stack+{name}"] = race_losses(scored, stack + scored[f"{name}:c"].to_numpy())
        twins = [race_losses(scored, stack + scored[f"{name}:c_twin{seed}"].to_numpy()) for seed in TWINS]
        report[name] = {"vs_stack": s.paired(added, loss["stack"], races),
                        "vs_mean_twin": s.paired(added, np.mean(twins, axis=0), races),
                        "vs_each_twin": [s.paired(added, t, races)["mean"] for t in twins]}
        total = total + scored[f"{name}:c"].to_numpy()
    if len(names) > 1:
        report["all"] = {"vs_stack": s.paired(race_losses(scored, total), loss["stack"], races)}
    target = OUT / f"new_information_{'_'.join(names)}.json"
    atomic_json(target, report)
    r = report["stack_vs_average"]
    print(f"stack vs average {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    for name in [*names, "all"] if len(names) > 1 else names:
        for key in ("vs_stack", "vs_mean_twin"):
            if key in report[name]:
                r = report[name][key]
                print(f"{name:<20} {key:<13} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
                      f"{r['quarters_better']}/4", flush=True)
        if "vs_each_twin" in report[name]:
            print(f"{name:<20} each twin     {np.round(report[name]['vs_each_twin'], 5).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["corrections", "compare"])
    parser.add_argument("--groups", nargs="+", required=True, choices=sorted(sc.CANDIDATES))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "corrections":
        corrections(args.groups, args.threads)
    else:
        compare(args.groups)


if __name__ == "__main__":
    main()
