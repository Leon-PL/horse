"""Variants of the single joint correction model, judged on native-capacity forecasts.

    python -m research.error_anatomy.joint_variants --variants NAME ... [--threads 6]

Section 10 of the report gave native-capacity forecasts for all four scored
quarters, so a larger correction can now be told apart from capacity repair:
it must gain on the native forecast, not only on the screen one. Variants,
declared before any was scored; the reference is ``joint-wide`` of
``joint_corrector.py`` (eight groups, 15 leaves, 600 rounds):

- ``more-groups``: adds franking, hidden form and past late moves.
- ``with-context``: ``more-groups`` plus the generalist's own inputs, so a
  correction can depend on the kind of race and runner. This is where capacity
  repair would show: on its own that group gained on screen fits and lost half
  on native.
- ``deeper``: the eight groups with 31 leaves, 400 per leaf, 1,000 rounds.

Same teacher, rows, base and walk-forward folds as every other correction.
"""
import argparse
import json

import numpy as np
import pandas as pd

from research.error_anatomy import soft_corrections as soft
from research.error_anatomy.frame import OUT
from research.error_anatomy.joint_corrector import ARMS, GROUPS
from research.error_anatomy.native_check import NATIVE
from research.meta_learner import screening as s
from research.meta_learner.specialists import race_losses
from src.ingestion.storage import atomic_json, atomic_parquet

KEYS = s.KEYS
WIDE = ARMS["joint-wide"]
MORE = (*GROUPS, "franking", "hidden-form", "market-moves")
VARIANTS = {
    "more-groups": {"groups": MORE, **WIDE},
    "with-context": {"groups": (*MORE, "own"), **WIDE},
    "deeper": {"groups": GROUPS, "rounds": 1000,
               "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Added 12 November after the three above were read: the two steps that each helped, together.
    "more-groups-deeper": {"groups": MORE, "rounds": 1000,
                           "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 12 November before scoring: the owner inputs (owner.py) inside the best variant's recipe.
    # Judged against more-groups-deeper on native.
    "with-owner": {"groups": (*MORE, "owner"), "rounds": 1000,
                   "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 12 November before scoring, after with-owner was read as a null: the newcomer inputs
    # (newcomer.py) inside the best variant's recipe. Judged against more-groups-deeper on native.
    "with-newcomer": {"groups": (*MORE, "newcomer"), "rounds": 1000,
                      "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 12 November before scoring: exact age from the foaling date (foal_age.py), -0.0015 alone
    # on the screen forecast. Judged against more-groups-deeper on native.
    # REJECTED AS A LEAK after scoring (-0.0016 on native): 22% of horses carry a placeholder foaling
    # date of 1 January in profiles all fetched in 2026, the placeholder is commoner the shorter the
    # career, and those horses win 7-23% less often than BSP implies. Do not use this variant.
    "with-foal-age": {"groups": (*MORE, "foal-age"), "rounds": 1000,
                      "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 13 November before scoring: the plot-race inputs (plot_races.py, the user's suggestion),
    # -0.0025 alone on the screen forecast. Judged against more-groups-deeper on native.
    "with-plot": {"groups": (*MORE, "plot"), "rounds": 1000,
                  "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 13 November before scoring: the filtered horse state (horse_filter.py). Judged against
    # more-groups-deeper on native.
    "with-filter": {"groups": (*MORE, "filter"), "rounds": 1000,
                    "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 13 November before scoring: ratings from past closing prices linked through rivals
    # (market_rating.py). Judged against more-groups-deeper on native.
    "with-market-rating": {"groups": (*MORE, "market-rating"), "rounds": 1000,
                           "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 13 November before scoring: the race chosen for a newcomer against the trainer's habit
    # (debut_choice.py). Judged against more-groups-deeper on native.
    "with-debut-choice": {"groups": (*MORE, "debut-choice"), "rounds": 1000,
                          "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 13 November before scoring: the three new groups together, in case each is too small alone.
    "with-three": {"groups": (*MORE, "filter", "market-rating", "debut-choice"), "rounds": 1000,
                   "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 13 November before scoring: the 11:00 exchange price of the race itself (morning_price.py).
    # The first variant that cannot forecast before 11:00. Judged against more-groups-deeper on native.
    "with-1100": {"groups": (*MORE, "price-1100"), "rounds": 1000,
                  "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 14 November before scoring: the yard-change inputs (yard_change.py, -0.0012 alone, 3/4) and the
    # weight-and-trip inputs (burden.py) together. Judged against more-groups-deeper on native.
    "with-yard-burden": {"groups": (*MORE, "yard", "burden"), "rounds": 1000,
                         "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 14 November before scoring: the trainer-move inputs (research/trainer_moves_098: size of the
    # move in trainer Elo, runs in the yard, the trainer's arrivals against SP) and their twin, shuffled
    # within quarter. Judged against more-groups-deeper and against each other on native.
    "with-trainer-move": {"groups": (*MORE, "trainer-move"), "rounds": 1000,
                          "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    "with-trainer-move-twin": {"groups": (*MORE, "trainer-move-twin"), "rounds": 1000,
                               "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 14 November before scoring: the horse's record on earlier reappearances (research/comebacks_099,
    # -0.00082 alone vs twin, 3/4) and its twin shuffled within quarter. Judged against each other on native.
    "with-comeback": {"groups": (*MORE, "comeback"), "rounds": 1000,
                      "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    "with-comeback-twin": {"groups": (*MORE, "comeback-twin"), "rounds": 1000,
                           "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 14 November before scoring (price timing, ERROR_ANATOMY next step 5): the 11:00 inputs plus hours
    # from 11:00 to the off, the same inputs one hour before the off, and five minutes before (upper reference).
    # Each judged against with-1100 on native; the fixed-offset arms read the move from the first recorded price.
    "with-1100-hours": {"groups": (*MORE, "price-1100-hours"), "rounds": 1000,
                        "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    "with-3600": {"groups": (*MORE, "price-3600"), "rounds": 1000,
                  "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    "with-300": {"groups": (*MORE, "price-300"), "rounds": 1000,
                 "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    # Declared 15 November before scoring: the correction's teacher at other market weights (0.75 everywhere else),
    # on the 11:00 recipe. Judged against with-1100 on native log loss and on 11:00 CLV (research 105).
    "with-1100-blend50": {"groups": (*MORE, "price-1100"), "rounds": 1000, "target": "y_soft50",
                          "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
    "with-1100-blend90": {"groups": (*MORE, "price-1100"), "rounds": 1000, "target": "y_soft90",
                          "params": {**WIDE["params"], "num_leaves": 31, "min_data_in_leaf": 400}},
}
SAVED = OUT / "joint_variants.parquet"
REPORT = OUT / "joint_variants.json"


def inputs(frame, groups):
    columns = []
    for name in groups:
        values, names = soft.group_values(name, frame)
        names = [c for c in names if c not in frame]
        values = values[[*KEYS, *names]].copy()
        values["horse_id"] = values.horse_id.astype(frame.horse_id.dtype)
        values["race_id"] = values.race_id.astype(frame.race_id.dtype)
        frame = frame.merge(values, on=KEYS, how="left", validate="one_to_one")
        columns += names
    return frame, columns


def native_frame(scored):
    parts = [pd.read_parquet(NATIVE, columns=[*KEYS, "probability"])]
    parts += [pd.read_parquet(p, columns=[*KEYS, "probability"]) for p in sorted((OUT / "native").glob("native_*.parquet"))]
    native = pd.concat(parts, ignore_index=True).rename(columns={"probability": "p_native"})
    native["race_id"] = native.race_id.astype(str)
    return scored.merge(native, on=KEYS, how="left", validate="one_to_one")


def assess(scored, corrections):
    """Each correction on the screen and native forecasts, and against the reference joint model on native."""
    frame = native_frame(scored)
    if frame.p_native.isna().any():
        raise ValueError("A scored runner has no native forecast")
    bounds = s.bounds_of(frame.race_id.to_numpy())
    races = frame.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    bases = {"screen": np.log(frame.p_single.to_numpy()), "native": np.log(frame.p_native.to_numpy())}
    loss = {k: race_losses(frame, v) for k, v in bases.items()}
    after = {name: {k: race_losses(frame, base + c) for k, base in bases.items()} for name, c in corrections.items()}
    out = {}
    for name in corrections:
        out[name] = {"on_screen": s.paired(after[name]["screen"], loss["screen"], races),
                     "on_native": s.paired(after[name]["native"], loss["native"], races),
                     "native_log_loss": float(after[name]["native"].mean())}
        if name != "joint-wide":
            out[name]["vs_joint_wide_on_native"] = s.paired(after[name]["native"], after["joint-wide"]["native"], races)
    return out


def show(report):
    for name, r in report.items():
        n = r["on_native"]
        line = (f"{name:<13} screen {r['on_screen']['mean']:+.5f}  native {n['mean']:+.5f} "
                f"[{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}] {n['quarters_better']}/4  loss {r['native_log_loss']:.5f}")
        if "vs_joint_wide_on_native" in r:
            v = r["vs_joint_wide_on_native"]
            line += f"   vs joint-wide {v['mean']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] {v['quarters_better']}/4"
        print(line, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=[], choices=list(VARIANTS))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    base = soft.frames()["long"]
    rows = base.quarter.isin(s.SCORED).to_numpy()
    scored = base.loc[rows, [*KEYS, "race_date", "quarter", "won", "p_single"]].reset_index(drop=True)
    saved = pd.read_parquet(SAVED) if SAVED.exists() else pd.read_parquet(OUT / "joint_corrector.parquet")[[*KEYS, "joint-wide"]]
    saved = scored[KEYS].merge(saved, on=KEYS, how="left", validate="one_to_one")
    info = json.loads(REPORT.read_text()).get("fits", {}) if REPORT.exists() else {}
    for name in args.variants:
        if name in saved:
            continue
        recipe = VARIANTS[name]
        frame, columns = inputs(base, recipe["groups"])
        print(f"{name}: {len(columns)} inputs", flush=True)
        fit = soft.soft_boost(columns, target=recipe.get("target", "y_soft"), rounds=recipe["rounds"], params=recipe["params"],
                              threads=args.threads)
        correction, _ = s.walk_forward(frame, fit)
        saved[name] = correction[frame.quarter.isin(s.SCORED).to_numpy()]
        info[name] = {"inputs": len(columns), "top_inputs": fit.info[-1]["top_inputs"],
                      "seconds": [i["seconds"] for i in fit.info]}
        atomic_parquet(saved, SAVED)
        del frame
        report = assess(scored, {c: saved[c].to_numpy() for c in saved if c not in KEYS})
        atomic_json(REPORT, {"fits": info, "results": report})
        show(report)
    report = assess(scored, {c: saved[c].to_numpy() for c in saved if c not in KEYS})
    atomic_json(REPORT, {"fits": info, "results": report})
    show(report)


if __name__ == "__main__":
    main()
