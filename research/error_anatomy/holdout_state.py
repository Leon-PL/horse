"""Second registered read of the hold-out: what horse-state and recent stable form add.

    python -m research.error_anatomy.holdout_state rehearse [--threads 6]
    python -m research.error_anatomy.holdout_state register
    python -m research.error_anatomy.holdout_state score [--threads 6]

The first read (``holdout_score.py``) confirmed seven corrections on races from
11 June 2026. The third pass then found, on development quarters only, that a
horse's disagreement with the market still repeated from run to run, and
``horse_state.py`` added two groups. This reads once what those two add to the
seven on the same races.

This is a second look at the same races, and the first read's results were
known when the new groups were built (they showed the own-horse group was the
strongest there). The new groups were designed and screened on development
quarters only. The registration states this, and the pass mark is a one-sided
p below 0.005, half the first read's, for the two looks.

``rehearse`` runs the whole path on development races and keeps its
corrections for the quarters before 2026Q2. ``score`` reuses those, fits only
2026Q2-2026Q4, checks that the development part of 2026Q2 matches the
rehearsal, reproduces the first read's seven-group figure, and refuses a
second run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.error_anatomy import holdout_score as first
from research.error_anatomy import horse_state as hs
from research.error_anatomy import soft_corrections as soft
from research.meta_learner import screening as s
from research.meta_learner.specialists import linear_combiner, race_losses
from src.ingestion.storage import atomic_json, atomic_parquet

ROOT = first.ROOT
KEYS = s.KEYS
ARM = first.ARM
SEVEN = first.SECONDARY
NEW = ("horse-state", "entity-recent")
NINE = (*SEVEN, *NEW)
CODE = ("entity_residuals.py", "soft_corrections.py", "franking.py", "horse_state.py", "holdout_score.py",
        "holdout_state.py")
CACHE = ROOT / "state_rehearsal_corrections.parquet"
RECEIPT = ROOT / "receipt_state.json"
REGISTRATION = ROOT / "registration_state.json"
PASS_P = .005


def code_digests():
    folder = Path(__file__).parent
    return {name: hashlib.sha256((folder / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in CODE}


def protocol():
    rehearsal = json.loads((ROOT / "rehearsal_state.json").read_text())
    return {
        "hypothesis": "Run-by-run horse state and short-memory stable form improve race log loss beyond the seven "
                      "corrections of the first read, on the same hold-out races.",
        "test_races": "As the first read: every race from the hold-out start with a complete settled field and a "
                      "seed-42 forecast for every runner.",
        "base": "Full 063, seed 42, screen capacity, purged quarterly refits.",
        "new_inputs": "horse_state.horse_state and horse_state.entity_recent on out-of-fold forecasts from 2022Q3.",
        "corrections": {"arm": ARM, "recipe": dict(soft.ARMS[ARM]), "target_blend": soft.BLEND,
                        "seeds": list(soft.SEEDS), "fitted": "walk-forward, earlier quarters only, 7-day purge"},
        "weights": f"One coefficient per correction, race logit on the seed-42 forecast, fitted on corrections from "
                   f"{first.COMBINER_FROM} up to the quarter before the scored one.",
        "primary": {"statistic": "mean over test races of race log loss(base + weighted nine) minus race log "
                                 "loss(base + weighted seven)",
                    "pass": f"one-sided p < {PASS_P} for a mean below zero, standard error clustered by race date"},
        "descriptive": ["nine against base", "each new group alone at unit weight", "by quarter", "by race type",
                        "gap to Betfair SP", "the seven against base, which must reproduce the first read"],
        "second_look": "These races were read once on 11 November for the seven corrections, and that receipt was "
                       "known when the new groups were built. The new groups were chosen on development quarters. "
                       "The pass mark is halved for the two looks.",
        "development_estimates": {"rehearsal_2026-01-01_to_2026-06-03": rehearsal["primary_nine_vs_seven"]["mean"],
                                  "horse_state_alone_on_native_2025Q3": -0.00470,
                                  "note": "chosen on these quarters, so the hold-out effect should be smaller"},
        "one_read": "score() refuses to run if receipt_state.json exists.",
        "code_sha256": code_digests(),
    }


def register():
    if RECEIPT.exists():
        raise RuntimeError("The second read has already been scored")
    value = protocol()
    if REGISTRATION.exists() and json.loads(REGISTRATION.read_text()) != value:
        raise RuntimeError("A different registration already exists; remove nothing, investigate")
    atomic_json(REGISTRATION, value)
    print(json.dumps(value, indent=1))


def inputs(frame):
    values, groups = first.group_inputs(frame)
    for name in NEW:
        part, columns = hs.load(name, frame)
        values = values.merge(part, on=KEYS, how="left", validate="one_to_one")
        groups[name] = columns
    return values, groups


def score(threads=6, rehearsal=False):
    test_start, test_quarters, target = first.holdout_start(), first.TEST_QUARTERS, RECEIPT
    if rehearsal:
        test_start, test_quarters, target = first.REHEARSAL_START, first.REHEARSAL_QUARTERS, ROOT / "rehearsal_state.json"
    else:
        if json.loads(REGISTRATION.read_text())["code_sha256"] != code_digests():
            raise RuntimeError("Code changed since registration; register again before scoring")
        if target.exists():
            raise RuntimeError("The second read has already been scored")
    frame, agreement = first.full_frame(development_only=rehearsal)
    values, groups = inputs(frame)
    frame = s.attach(frame, values)
    quarters = sorted(q for q in frame.quarter.unique() if q >= first.COMBINER_FROM)
    recipe = soft.ARMS[ARM]
    fit_quarters, cached, parity = quarters, None, None
    if not rehearsal:
        # Quarters before 2026Q2 hold development races only; their corrections were made in the rehearsal.
        cached = frame[KEYS].merge(pd.read_parquet(CACHE), on=KEYS, how="left", validate="one_to_one")
        fit_quarters = [q for q in quarters if q >= "2026Q2"]
        parity = {}
    for name in NINE:
        fit = soft.soft_boost(groups[name], target=recipe["target"], rounds=recipe["rounds"],
                              params=recipe["params"], threads=threads)
        correction, _ = s.walk_forward(frame, fit, scored=fit_quarters)
        if cached is not None:
            old = cached[f"{name}:c"].to_numpy()
            early = (frame.quarter < "2026Q2").to_numpy() & frame.quarter.isin(quarters).to_numpy()
            both = np.isfinite(old) & frame.quarter.eq("2026Q2").to_numpy()
            parity[name] = {"runners": int(both.sum()), "max_abs_difference": float(np.abs(old[both] - correction[both]).max())}
            correction = np.where(early, old, correction)
        frame[f"{name}:c"] = correction
        print(f"{name}: corrections fitted", flush=True)
    work = frame.loc[frame.quarter.isin(quarters)].reset_index(drop=True)
    if rehearsal:
        atomic_parquet(work[[*KEYS, *[f"{n}:c" for n in NINE]]], CACHE)
    fitted, weights = {}, {}
    for label, names in (("seven", SEVEN), ("nine", NINE)):
        combiner = linear_combiner([f"{n}:c" for n in names], base="p_single")
        fitted[label] = s.walk_forward(work, combiner, scored=test_quarters)[0]
        weights[label] = combiner.info
    work["fitted_seven"], work["fitted_nine"] = fitted["seven"], fitted["nine"]
    test = work.loc[work.race_date.ge(test_start)].reset_index(drop=True)
    if not np.isfinite(test[["fitted_seven", "fitted_nine"]].to_numpy()).all():
        raise ValueError("A test runner has no fitted correction")
    atomic_parquet(test[[*KEYS, "race_date", "quarter", "p_single", "p_bsp", "won", "fitted_seven", "fitted_nine",
                         *[f"{n}:c" for n in NINE]]],
                   ROOT / ("rehearsal_state_runners.parquet" if rehearsal else "scored_state_runners.parquet"))
    bounds = s.bounds_of(test.race_id.to_numpy())
    races = test.iloc[bounds[:-1]][["race_id", "race_date", "quarter", "race_type"]].reset_index(drop=True)
    base = np.log(test.p_single.to_numpy())
    loss = {"base": race_losses(test, base), "seven": race_losses(test, base + test.fitted_seven.to_numpy()),
            "nine": race_losses(test, base + test.fitted_nine.to_numpy())}
    test_of = lambda a, b: first.clustered_test(loss[a] - loss[b], races.race_date)
    receipt = {"registration_sha256": None if rehearsal else hashlib.sha256(REGISTRATION.read_bytes()).hexdigest(),
               "rehearsal": rehearsal, "first_date": str(test.race_date.min().date()),
               "last_date": str(test.race_date.max().date()), "refit_agreement_max_log_ratio": agreement,
               "rehearsal_parity_2026Q2": parity,
               "race_log_loss": {k: float(v.mean()) for k, v in loss.items()},
               "primary_nine_vs_seven": test_of("nine", "seven"), "nine_vs_base": test_of("nine", "base"),
               "seven_vs_base": test_of("seven", "base"), "weights": weights,
               "each_new_group_unit_weight": {}, "by_quarter": {}, "by_race_type": {}}
    receipt["primary_nine_vs_seven"]["passes"] = bool(receipt["primary_nine_vs_seven"]["p_one_sided"] < PASS_P)
    if not rehearsal:
        earlier = json.loads((ROOT / "receipt.json").read_text())["primary_all_seven"]["mean"]
        receipt["first_read_seven_vs_base"] = earlier
        receipt["seven_reproduced_within"] = float(abs(receipt["seven_vs_base"]["mean"] - earlier))
    for name in NEW:
        alone = race_losses(test, base + test[f"{name}:c"].to_numpy())
        receipt["each_new_group_unit_weight"][name] = first.clustered_test(alone - loss["base"], races.race_date)
    for key, column in (("by_quarter", "quarter"), ("by_race_type", "race_type")):
        for value, rows in races.groupby(column).indices.items():
            receipt[key][str(value)] = {"races": int(len(rows)),
                                        "nine_vs_seven": float((loss["nine"] - loss["seven"])[rows].mean()),
                                        "nine_vs_base": float((loss["nine"] - loss["base"])[rows].mean())}
    priced = pd.Series(test.p_bsp.notna().to_numpy()).groupby(test.race_id.to_numpy(), sort=False).all().to_numpy()
    market = race_losses(test.assign(p_bsp=test.p_bsp.fillna(1.)), np.log(test.p_bsp.fillna(1.).to_numpy()))
    receipt["gap_to_bsp"] = {"races": int(priced.sum()), "bsp": float(market[priced].mean()),
                             **{k: float(v[priced].mean() - market[priced].mean()) for k, v in loss.items()}}
    atomic_json(target, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "weights"}, indent=1))
    for row in weights["nine"]:
        print({k: v for k, v in row.items() if k == "fold" or k.endswith(":c")})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["rehearse", "register", "score"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "register":
        register()
    else:
        score(args.threads, rehearsal=args.stage == "rehearse")


if __name__ == "__main__":
    main()
