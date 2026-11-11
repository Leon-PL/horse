"""One correction model over every error input, against the weighted sum of one model per group.

    python -m research.error_anatomy.joint_corrector [--threads 6]

The stack fits one small correction per input group and then one weight per
correction. Beside horse state the old own-horse correction takes a weight of
-0.3, a sign that the sum is fighting itself, and the second hold-out read
found the stack's development increments shrink out of sample. A single
correction that sees all the inputs can use one group to condition another and
has no differences of near-duplicates to estimate.

Same teacher (75% past closing price), same rows (2022Q3 on), same base
(seed-42 offset), walk-forward. Arms, declared before scoring:

- ``joint-rich``: the per-group recipe (7 leaves, 1,500 per leaf, 300 rounds).
- ``joint-wide``: 15 leaves, 800 per leaf, 600 rounds, half the inputs per tree.

Compared with the linear stack of the same eight groups on the same base (the
three-seed average, no performance pool), and on the native 2025Q3 forecast.
"""
import argparse

import numpy as np
import pandas as pd

from research.error_anatomy import soft_corrections as soft
from research.error_anatomy.frame import OUT
from research.error_anatomy.native_check import NATIVE
from research.meta_learner import screening as s
from research.meta_learner.specialists import QUARTERS, linear_combiner, race_losses
from src.ingestion.storage import atomic_json, atomic_parquet

KEYS = s.KEYS
GROUPS = ("horse", "market-gap", "residual", "wave2", "same-day-bias", "tracking", "horse-state", "entity-recent")
ARMS = {
    "joint-rich": {"params": soft.RICH, "rounds": 300},
    "joint-wide": {"params": {**soft.RICH, "num_leaves": 15, "min_data_in_leaf": 800, "feature_fraction": .5},
                   "rounds": 600},
}


def inputs(frame):
    columns = []
    for name in GROUPS:
        values, names = soft.group_values(name, frame)
        names = [c for c in names if c not in frame]
        frame = s.attach(frame, values[[*KEYS, *names]])
        columns += names
    return frame, columns


def linear_stack():
    """The eight per-group corrections at weights fitted on earlier quarters, on the three-seed average."""
    saved = pd.read_parquet(OUT / "soft_stack_soft-long-rich.parquet")
    frame = soft.frames()["long"]
    frame = s.attach(frame.loc[frame.quarter.isin(QUARTERS)].reset_index(drop=True), saved)
    combiner = linear_combiner([f"{g}:c" for g in GROUPS], base="p_average")
    fitted, _ = s.walk_forward(frame, combiner, scored=s.SCORED)
    return frame.loc[frame.quarter.isin(s.SCORED), KEYS].assign(c=fitted[frame.quarter.isin(s.SCORED).to_numpy()])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    frame, columns = inputs(soft.frames()["long"])
    print(f"{len(columns)} inputs", flush=True)
    rows = frame.quarter.isin(s.SCORED).to_numpy()
    scored = frame.loc[rows].reset_index(drop=True)
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    average = np.log(scored.p_average.to_numpy())
    corrections = {"linear-stack": scored[KEYS].merge(linear_stack(), on=KEYS, validate="one_to_one").c.to_numpy()}
    report = {"inputs": len(columns), "groups": list(GROUPS), "arms": {}}
    for arm, recipe in ARMS.items():
        fit = soft.soft_boost(columns, target="y_soft", rounds=recipe["rounds"], params=recipe["params"],
                              threads=args.threads)
        correction, _ = s.walk_forward(frame, fit)
        corrections[arm] = correction[rows]
        report["arms"][arm] = {"top_inputs": fit.info[-1]["top_inputs"], "seconds": [i["seconds"] for i in fit.info]}
    loss = {name: race_losses(scored, average + c) for name, c in corrections.items()}
    loss["average"] = race_losses(scored, average)
    report["race_log_loss"] = {k: float(v.mean()) for k, v in loss.items()}
    report["vs_average"] = {k: s.paired(loss[k], loss["average"], races) for k in corrections}
    report["joint_vs_linear"] = {k: s.paired(loss[k], loss["linear-stack"], races) for k in ARMS}
    # Native-capacity check on 2025Q3: the same corrections added to the native forecast.
    native = pd.read_parquet(NATIVE, columns=[*KEYS, "probability"]).rename(columns={"probability": "p_native"})
    native["race_id"] = native.race_id.astype(str)
    quarter = scored.quarter.eq("2025Q3").to_numpy()
    part = s.attach(scored.loc[quarter, [*KEYS, "race_date", "quarter", "won", "p_single"]].reset_index(drop=True), native)
    part_bounds = s.bounds_of(part.race_id.to_numpy())
    part_races = part.iloc[part_bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    part_races["covered"] = True
    report["native_2025Q3"] = {}
    for base in ("p_single", "p_native"):
        reference = race_losses(part, np.log(part[base].to_numpy()))
        report["native_2025Q3"][base] = {
            k: s.paired(race_losses(part, np.log(part[base].to_numpy()) + c[quarter]), reference, part_races)
            for k, c in corrections.items()}
    atomic_json(OUT / "joint_corrector.json", report)
    atomic_parquet(scored[KEYS].assign(**{k: c for k, c in corrections.items()}), OUT / "joint_corrector.parquet")
    print({k: round(v, 5) for k, v in report["race_log_loss"].items()})
    for k, r in report["vs_average"].items():
        print(f"{k:<13} vs average {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    for k, r in report["joint_vs_linear"].items():
        print(f"{k:<13} vs linear  {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['quarters_better']}/4")
    for base, rows_ in report["native_2025Q3"].items():
        print(base, {k: round(r["mean"], 5) for k, r in rows_.items()})


if __name__ == "__main__":
    main()
