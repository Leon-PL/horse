"""Who the generalist persistently misjudges: running residuals by trainer, jockey, sire and horse.

    python -m research.error_anatomy.entity_residuals screen [--threads 4]
    python -m research.error_anatomy.entity_residuals long [--threads 4]

The store describes connections by raw strike rates, which mostly restate how
good their horses are. This module measures each entity against the model's
own forecast instead: on every earlier date with an out-of-fold forecast, how
did its runners finish relative to what the model expected?

Per-run signals, each zero in expectation if the forecast is right:

- win: ``won - p``;
- rivals beaten: the share of rivals the runner beat, minus the share the
  forecast expected it to beat (``mean_j p_i / (p_i + p_j)``, the pairwise
  chance under a Luce model). It uses the whole finishing order, so it carries
  several times the information of the win residual. Non-finishers rank behind
  every finisher and level with each other;
- market gap (optional): ``log(market p) - log(model p)`` at the starting
  price of that earlier race. It has no outcome noise: it records directly
  that the market rated the runner above or below the model.

For each entity key the inputs are shrunk means of those signals over runs on
strictly earlier dates, the number of such runs, and the means relative to the
field. A race never sees its own result or price, or any from the same day.

``screen`` is the first definition (forecasts from 2024Q3, no decay, no market
gap). ``long`` uses forecasts from 2022Q3 (``extend_oof.py``) and screens the
residual, market-gap and own-horse families separately.
"""
import argparse

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT, race_normalised, scored_frame
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json, atomic_parquet

KEYS = s.KEYS
from src.correction.inputs import (ENTITIES, HORSE, HORSE_SHRINK, SHRINK, WAVE2,  # noqa: F401  (one owner: src)
                                   entity_inputs, run_residuals)
from src.correction.inputs import entity_keys as _keys  # noqa: F401


def columns_of(values):
    return [c for c in values if c not in KEYS]


def screen(threads):
    """First definition: forecasts from 2024Q3, no decay."""
    frame = scored_frame(extra=["sire"], scored_only=False)
    values = entity_inputs(frame)
    base = s.attach(sc.base_frame(), values)
    columns = columns_of(values)
    sc.OUT = OUT  # keep this study's screens out of the 9 November folder
    report = {"all": sc.assess(base, "entity-residuals", columns, bases=("p_single", "p_average"), threads=threads)}
    for name in ENTITIES:
        group = [c for c in columns if c.startswith(f"er_{name}_")]
        report[name] = sc.assess(base, f"entity-residuals-{name}", group, bases=("p_single", "p_average"),
                                 threads=threads)
    atomic_json(OUT / "entity_residuals_screen.json", report)


def long_frame():
    """Every runner from 2022Q3 with a seed-42 out-of-fold forecast and a market probability."""
    late = sc.base_frame()
    early = pd.read_parquet(OUT / "early_oof_seed42.parquet")
    early.attrs = {}
    labels = pd.read_parquet(s.STUDY / "input.parquet", columns=[*s.LABELS, "bsp", "odds"],
                             filters=[("race_date", ">=", early.race_date.min())])
    labels.attrs = {}
    forecasts = pd.concat([early[[*KEYS, "probability"]].rename(columns={"probability": "p_single"}),
                           late[[*KEYS, "p_single"]]], ignore_index=True)
    frame = labels.merge(forecasts, on=KEYS, how="inner", validate="one_to_one")
    frame, _ = s.ordered_frame(frame)
    store = sc.read_store(["trainer", "jockey", "sire", "race_type", "days_since_last_run", "track", "going",
                           "damsire", "handicap", "surface", "distance_furlongs"],
                          start=frame.race_date.min()).drop(columns="race_date")
    frame = s.attach(frame, store)
    # Market probability at the off: BSP where the whole field has one, else SP.
    for column in ("bsp", "odds"):
        price = pd.to_numeric(frame[column], errors="coerce")
        whole = (price > 1).groupby(frame.race_id).transform("all")
        frame[f"_p_{column}"] = np.where(whole, race_normalised(1 / price.where(price > 1), frame.race_id), np.nan)
    frame["p_market"] = frame._p_bsp.fillna(frame._p_odds)
    return frame.drop(columns=["_p_bsp", "_p_odds"])


LONG_HALF_LIFE = 365.


def long_inputs(frame):
    """The three long-memory families: residuals, market gap and the horse's own record."""
    entity = entity_inputs(frame, market="p_market", half_life=LONG_HALF_LIFE)
    horse = entity_inputs(frame, market="p_market", half_life=LONG_HALF_LIFE, entities=HORSE, shrink=HORSE_SHRINK)
    wave2 = entity_inputs(frame, half_life=LONG_HALF_LIFE, entities=WAVE2)
    values = entity.merge(horse, on=KEYS, validate="one_to_one").merge(wave2, on=KEYS, validate="one_to_one")
    groups = {"residual": [c for c in columns_of(entity) if c.startswith("er_")],
              "market-gap": [c for c in columns_of(entity) if c.startswith("mg_")],
              "horse": columns_of(horse), "wave2": columns_of(wave2)}
    return values, groups


def long(threads):
    frame = long_frame()
    values, groups = long_inputs(frame)
    atomic_parquet(values, OUT / "long_inputs.parquet")
    base = s.attach(sc.base_frame(), values)
    sc.OUT = OUT
    report = {}
    for name, columns in groups.items():
        report[name] = sc.assess(base, f"long-{name}", columns, bases=("p_single", "p_average"), threads=threads)
        atomic_json(OUT / "long_screen.json", report)


def stack(threads, names):
    """Each named long-memory group as its own correction on the 9 November stack, against three twins.

    The stack is the three-seed average log-pooled with performance/spread plus the tracking and
    same-day corrections. Corrections here follow the same recipe (Tier 1, fitted on the seed-42
    forecasts of earlier quarters, out of fold for 2025Q1-2026Q2) and are summed onto it.
    """
    from research.meta_learner.specialists import CORRECTIONS, QUARTERS, race_losses
    values = pd.read_parquet(OUT / "long_inputs.parquet")
    _, groups = long_inputs_groups(values)
    base = s.attach(sc.base_frame(), values)
    keep = base.quarter.isin(QUARTERS).to_numpy()
    saved = base.loc[keep, [*KEYS, "race_date", "quarter"]].reset_index(drop=True)
    twins = sc.FOLLOWUP_TWINS
    for name in names:
        columns = groups[name]
        variants = {"": base, **{f"_twin{seed}": s.permute_within(base, columns, seed=seed) for seed in twins}}
        for suffix, data in variants.items():
            fit = s.residual_boost(columns, base="p_single", threads=threads, params=sc.TIER1["params"],
                                   **sc.TIER1["boost"])
            correction, _ = s.walk_forward(data, fit, scored=QUARTERS)
            saved[f"{name}:c{suffix}"] = correction[keep]
            print(f"{name}{suffix}: fitted", flush=True)
    atomic_parquet(saved, OUT / "stack_corrections.parquet")
    frame = s.attach(sc.pool_frame(), pd.read_parquet(CORRECTIONS, columns=[*KEYS, "c_tracking", "c_same-day-bias"]))
    frame = s.attach(frame, saved.drop(columns=["race_date", "quarter"]))
    scored = frame.loc[frame.quarter.isin(s.SCORED)].reset_index(drop=True)
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    average = np.log(scored.p_average.to_numpy())
    base_stack = np.log(scored.p_pool.to_numpy()) + scored.c_tracking.to_numpy() + scored["c_same-day-bias"].to_numpy()
    loss = {"average": race_losses(scored, average), "stack": race_losses(scored, base_stack)}
    report = {"races": len(races), "stack_vs_average": s.paired(loss["stack"], loss["average"], races)}
    total = base_stack.copy()
    for name in names:
        c = scored[f"{name}:c"].to_numpy()
        added = race_losses(scored, base_stack + c)
        twin_losses = [race_losses(scored, base_stack + scored[f"{name}:c_twin{seed}"].to_numpy()) for seed in twins]
        report[name] = {"on_average": s.paired(race_losses(scored, average + c), loss["average"], races),
                        "on_stack": s.paired(added, loss["stack"], races),
                        "on_stack_vs_mean_twin": s.paired(added, np.mean(twin_losses, axis=0), races),
                        "on_stack_vs_each_twin": [s.paired(added, t, races)["mean"] for t in twin_losses]}
        total = total + c
    everything = race_losses(scored, total)
    report["all_named"] = {"on_stack": s.paired(everything, loss["stack"], races),
                           "vs_average": s.paired(everything, loss["average"], races),
                           "race_log_loss": float(everything.mean())}
    atomic_json(OUT / f"stack_{'_'.join(names)}.json", report)
    for name in [*names, "all_named"]:
        for key, r in report[name].items():
            if isinstance(r, dict):
                print(f"{name:<12} {key:<22} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
                      f"{r['quarters_better']}/4", flush=True)


def combine(names):
    """Overlapping corrections: plain sums against weights fitted on earlier quarters.

    The four groups share signal, so summing them at full weight over-corrects. The fitted arm
    learns one coefficient per correction by a race logit on the pool, on earlier quarters only
    (2025Q1-Q2 before the first scored quarter), as ``specialists.linear_combiner`` does.
    """
    from research.meta_learner.specialists import CORRECTIONS, QUARTERS, linear_combiner, race_losses
    frame = s.attach(sc.pool_frame(), pd.read_parquet(CORRECTIONS, columns=[*KEYS, "c_tracking", "c_same-day-bias"]))
    saved = pd.read_parquet(OUT / "stack_corrections.parquet").drop(columns=["race_date", "quarter"])
    new = [f"{name}:c" for name in names]
    frame = s.attach(frame, saved[[*KEYS, *new]])
    frame = frame.loc[frame.quarter.isin(QUARTERS)].reset_index(drop=True)
    old = ["c_tracking", "c_same-day-bias"]
    fitted = {}
    for label, columns in (("fitted_all", [*old, *new]), ("fitted_old", old)):
        # The pool exists only for the scored quarters, so weights are fitted on the three-seed average.
        combiner = linear_combiner(columns, base="p_average")
        fitted[label] = (s.walk_forward(frame, combiner, scored=s.SCORED)[0], combiner.info)
    later = frame.quarter.isin(s.SCORED).to_numpy()
    scored = frame.loc[later].reset_index(drop=True)
    pool, average = np.log(scored.p_pool.to_numpy()), np.log(scored.p_average.to_numpy())
    stack_ = pool + scored[old].to_numpy().sum(1)
    arms = {"stack": stack_, "stack_plus_all_summed": stack_ + scored[new].to_numpy().sum(1),
            "stack_plus_all_fitted": pool + fitted["fitted_all"][0][later],
            "stack_fitted_old_only": pool + fitted["fitted_old"][0][later]}
    for column in new:
        arms[f"stack_plus_{column[:-2]}"] = stack_ + scored[column].to_numpy()
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    loss = {name: race_losses(scored, logp) for name, logp in arms.items()}
    loss["average"], loss["single"] = race_losses(scored, average), race_losses(scored, np.log(scored.p_single.to_numpy()))
    report = {"races": len(races), "coefficients": fitted["fitted_all"][1],
              "race_log_loss": {name: float(v.mean()) for name, v in loss.items()},
              "vs_stack": {n: s.paired(loss[n], loss["stack"], races) for n in arms if n != "stack"},
              "vs_average": {n: s.paired(loss[n], loss["average"], races) for n in arms},
              "vs_single": {n: s.paired(loss[n], loss["single"], races) for n in arms},
              "evaluation_status": "retrospective screen on reused development quarters"}
    atomic_json(OUT / "combine.json", report)
    for name in arms:
        if name != "stack":
            r, a = report["vs_stack"][name], report["vs_average"][name]
            print(f"{name:<28} vs stack {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
                  f"{r['quarters_better']}/4   vs average {a['mean']:+.5f} {a['quarters_better']}/4   "
                  f"loss {report['race_log_loss'][name]:.5f}", flush=True)
    for row in report["coefficients"]:
        print({k: v for k, v in row.items() if k == "fold" or k.startswith("c_") or k.endswith(":c")})


def stress(threads):
    """Timing stress test: the own-horse and market-gap groups built only from runs at least a week old.

    A timing leak (a same-day or next-day result or price reaching the inputs) would show as a
    collapse here. Real persistence should lose only what the newest runs carry.
    """
    frame = long_frame()
    sc.OUT = OUT
    report = {}
    for lag in (7, 30):
        horse = entity_inputs(frame, market="p_market", half_life=LONG_HALF_LIFE, entities=HORSE,
                              shrink=HORSE_SHRINK, lag_days=lag)
        entity = entity_inputs(frame, market="p_market", half_life=LONG_HALF_LIFE, lag_days=lag)
        gap = entity[[*KEYS, *[c for c in columns_of(entity) if c.startswith("mg_")]]]
        for name, values in (("horse", horse), ("market-gap", gap)):
            base = s.attach(sc.base_frame(), values)
            report[f"{name}-lag{lag}"] = sc.assess(base, f"stress-{name}-lag{lag}", columns_of(values),
                                                   bases=("p_single", "p_average"), threads=threads)
            atomic_json(OUT / "stress_screen.json", report)


def long_inputs_groups(values):
    columns = columns_of(values)
    wave2 = [c for c in columns if any(c.startswith(f"er_{name}_") for name in WAVE2)]
    horse = [c for c in columns if "_horse_" in c]
    rest = [c for c in columns if c not in wave2 and c not in horse]
    return values, {"residual": [c for c in rest if c.startswith("er_")],
                    "market-gap": [c for c in rest if c.startswith("mg_")], "horse": horse, "wave2": wave2}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["screen", "long", "stack", "combine", "stress"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--groups", nargs="+", default=["residual"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "stack":
        stack(args.threads, args.groups)
    elif args.stage == "combine":
        combine(args.groups)
    else:
        {"screen": screen, "long": long, "stress": stress}[args.stage](args.threads)


if __name__ == "__main__":
    main()
