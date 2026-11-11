"""Tier 2: candidate input groups inside the full 063 generalist, three seeds per arm.

    python -m research.meta_learner.tier2_inputs fit --arm ability-sameday [--threads 4]
    python -m research.meta_learner.tier2_inputs report

The cheap screens measure a group as a correction on top of the stored
forecasts. They understate inputs that change how the generalist uses what it
already has, so the plan's step 2 refits full 063 with the groups among its
inputs. Each arm is the registered full 063 recipe plus one or more groups from
``screen_candidates.CANDIDATES``, fitted at screen capacity with seeds 42, 43
and 44 for 2025Q3-2026Q2. The reference arm is the first ensemble benchmark's
cached three seeds on the same rows, folds and code.

Gate, fixed before any arm was fitted: the arm's three-seed average must beat
the cached three-seed average with a paired date-clustered 95% interval below
zero and at least three of four quarters better. Single-seed pairs are reported
but are inside refit noise (about 0.0015) and decide nothing.

Results go to ``data/research/meta_learner/tier2_20261110``.
"""
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.meta_learner import first_ensemble as fe
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.modelling.data import CONTEXT
from src.modelling.evaluation import code_signature, component_oof
from src.modelling.spec import ModelSpec

OUT = Path("data/research/meta_learner/tier2_20261110")
STUDY = s.STUDY
SEEDS = (42, 43, 44)
LATER = fe.LATER
KEYS = fe.KEYS
ARMS = {
    "ability-sameday": ("ability-lb", "same-day-bias"),
    "ability": ("ability-lb",),
    "sameday": ("same-day-bias",),
    "tracking": ("tracking",),
    "all-three": ("ability-lb", "same-day-bias", "tracking"),
}
GATE = ("Three-seed average of the arm against the cached three-seed average of full 063: paired date-clustered "
        "95% interval below zero and at least three of four quarters better.")


def design():
    out = fe.read_json(STUDY / "registration.json")
    if code_signature() != out["model_code"] or fe.sha(STUDY / "input.parquet") != out["input_sha256"]:
        raise ValueError("Modelling code or the prepared input changed since the reference arm was fitted")
    return out


def full_spec(registered):
    return next(ModelSpec.from_dict(x) for x in registered["specs"] if x["id"] == "full-063")


def arm_frame(registered, groups):
    """The benchmark's prepared rows plus the arm's extra inputs, joined by runner key."""
    full = full_spec(registered)
    frame = fe.read_input(STUDY, registered, list(full.features))
    extra = []
    for group in groups:
        values, columns = sc.CANDIDATES[group]()(frame)
        values = values[[*KEYS, *columns]].copy()
        values[columns] = values[columns].astype("float32")
        frame = s.attach(frame, values)
        extra += columns
    return frame, extra


def seed_spec(spec, seed):
    return spec if seed == spec.seed else replace(spec, id=f"{spec.id}-seed-{seed}", seed=seed)


def fit(arm, threads, seeds):
    registered = design()
    groups = ARMS[arm]
    frame, extra = arm_frame(registered, groups)
    spec = replace(full_spec(registered), id=f"full-063-{arm}", features=(*full_spec(registered).features, *extra))
    fe.immutable_json(OUT / f"{arm}_registration.json", {
        "arm": arm, "groups": list(groups), "extra_inputs": extra, "spec": spec.to_dict(), "seeds": list(SEEDS),
        "folds": list(LATER), "capacity": "screen", "gate": GATE, "reference_study": str(STUDY),
        "model_code": registered["model_code"], "input_sha256": registered["input_sha256"],
        "coverage": {c: float(frame[c].notna().mean()) for c in extra},
        "evaluation_status": "retrospective refit on reused development quarters"})
    folds = fe.folds_for(frame, LATER, spec.purge_days)
    for seed in seeds:
        result, _ = component_oof(frame, [seed_spec(spec, seed)], folds=folds, capacity="screen", threads=threads,
                                  dataset_start=registered["receipt"]["dataset_start"], cache_dir=OUT / "cache")
        result.attrs = {}
        atomic_parquet(result, OUT / f"{arm}_seed{seed}.parquet")
        print(f"{arm} seed {seed}: {len(result)} forecasts", flush=True)


def reference(seed):
    name = "full-063" if seed == 42 else f"full-063-seed-{seed}"
    out = s.cached_forecasts(STUDY, name)
    return out.loc[out.fold.isin(LATER)].reset_index(drop=True)


def aligned(parts):
    """Sort every seed's forecasts the same way so they can be averaged."""
    return [p.sort_values(KEYS).reset_index(drop=True) for p in parts]


def report():
    registered = design()
    frame = pd.read_parquet(STUDY / "input.parquet", columns=list(CONTEXT))
    base = aligned([reference(seed) for seed in SEEDS])
    losses = {f"reference_seed{seed}": fe.metrics(frame, part, LATER) for seed, part in zip(SEEDS, base)}
    losses["reference_average"] = fe.metrics(frame, fe.average_predictions(base), LATER)
    out = {"gate": GATE, "reference": {k: v[0]["race_log_loss"] for k, v in losses.items()}, "arms": {}}
    for arm in ARMS:
        files = {seed: OUT / f"{arm}_seed{seed}.parquet" for seed in SEEDS}
        done = [seed for seed in SEEDS if files[seed].exists()]
        if not done:
            continue
        parts = dict(zip(done, aligned([pd.read_parquet(files[seed]) for seed in done])))
        entry = {"seeds": done, "inputs_added": len(fe.read_json(OUT / f"{arm}_registration.json")["extra_inputs"]),
                 "single_seed_pairs": {}}
        for seed, part in parts.items():
            summary, races = fe.metrics(frame, part, LATER)
            entry["single_seed_pairs"][str(seed)] = fe.difference(races, losses[f"reference_seed{seed}"][1])
        if len(done) == len(SEEDS):
            summary, races = fe.metrics(frame, fe.average_predictions(list(parts.values())), LATER)
            entry["average"] = summary
            entry["average_vs_reference_average"] = change = fe.difference(races, losses["reference_average"][1])
            entry["average_vs_reference_single"] = fe.difference(races, losses["reference_seed42"][1])
            entry["clears_gate"] = fe.clears_gate(change)
        out["arms"][arm] = entry
        lines = {f"seed {seed} vs seed {seed}": r for seed, r in entry["single_seed_pairs"].items()}
        if "average_vs_reference_average" in entry:
            lines = {"average vs reference average": entry["average_vs_reference_average"], **lines}
        for label, r in lines.items():
            print(f"{arm:<16} {label:<30} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
                  f"{r['quarters_better']}/4", flush=True)
    atomic_json(OUT / "report.json", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["fit", "report"])
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        report()
        return
    if not args.arm:
        parser.error("fit requires --arm")
    with writer_lock(OUT / f".{args.arm}.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another process is fitting this arm")
        fit(args.arm, args.threads, args.seeds)


if __name__ == "__main__":
    main()
