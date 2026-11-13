"""Second-round corrections: what do the corrections themselves still get wrong, entity by entity?

    python -m research.error_anatomy.second_round [--threads 6]

Every trainer, sire and horse record in the corrections is measured against the
uncorrected forecast. This measures them again against the corrected one.

Declared before scoring:

1. Round-one history. The best joint correction (``more-groups-deeper`` of
   ``joint_variants.py``) is walked forward over 2023Q3-2025Q2 as well, one
   seed, so every race from 2023Q3 has a corrected forecast fitted on earlier
   quarters only. The four scored quarters reuse the saved three-seed values.
2. Round-two inputs, all on strictly earlier dates against the corrected
   forecast: the entity residual and market-gap means (trainer, jockey, sire
   and their keyed versions, 365-day half-life), the horse's own record, and
   the run-by-run horse state.
3. Round-two correction: recipe ``soft-long-rich`` (7 leaves, 300 rounds, three
   seeds), offset the corrected forecast, target 0.75 market + 0.25 result,
   rows from 2023Q3, walked forward over the four scored quarters. A twin with
   the inputs shuffled within quarter is fitted the same way.

Judged on native forecasts: native + round one + round two against native +
round one.
"""
import argparse

import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.error_anatomy import horse_state as hs
from research.error_anatomy import soft_corrections as soft
from research.error_anatomy.frame import OUT
from research.error_anatomy.joint_variants import SAVED as ROUND_ONE_SCORED
from research.error_anatomy.joint_variants import VARIANTS, inputs, native_frame
from research.meta_learner import screening as s
from research.meta_learner.specialists import race_losses
from src.ingestion.storage import atomic_json, atomic_parquet

KEYS = s.KEYS
RECIPE = "more-groups-deeper"
HISTORY = ("2023Q3", "2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2")
HISTORY_FILE = OUT / "second_round_history.parquet"
REPORT = OUT / "second_round.json"
TWIN_SEED = 11


def round_one(base, threads):
    """Out-of-fold round-one correction for every runner from 2023Q3."""
    if HISTORY_FILE.exists():
        early = pd.read_parquet(HISTORY_FILE)
    else:
        recipe = VARIANTS[RECIPE]
        frame, columns = inputs(base, recipe["groups"])
        fit = soft.soft_boost(columns, target="y_soft", rounds=recipe["rounds"], params=recipe["params"],
                              threads=threads, seeds=(7,))
        correction, _ = s.walk_forward(frame, fit, scored=HISTORY)
        rows = frame.quarter.isin(HISTORY).to_numpy()
        early = frame.loc[rows, KEYS].assign(c1=correction[rows]).reset_index(drop=True)
        atomic_parquet(early, HISTORY_FILE)
    late = pd.read_parquet(ROUND_ONE_SCORED, columns=[*KEYS, RECIPE]).rename(columns={RECIPE: "c1"})
    both = pd.concat([early, late], ignore_index=True)
    both["race_id"] = both.race_id.astype(base.race_id.dtype)
    both["horse_id"] = both.horse_id.astype(base.horse_id.dtype)
    frame = base.merge(both, on=KEYS, how="inner", validate="one_to_one").reset_index(drop=True)
    if frame.c1.isna().any():
        raise ValueError("A runner from 2023Q3 has no round-one correction")
    bounds = s.bounds_of(frame.race_id.to_numpy())
    frame["p_r1"] = np.exp(s.race_log_softmax(np.log(frame.p_single.to_numpy()) + frame.c1.to_numpy(), bounds))
    return frame


def round_two_inputs(frame):
    entity = er.entity_inputs(frame, "p_r1", market="p_market", half_life=er.LONG_HALF_LIFE, prefix="r2er")
    horse = er.entity_inputs(frame, "p_r1", market="p_market", half_life=er.LONG_HALF_LIFE, entities=er.HORSE,
                             shrink=er.HORSE_SHRINK, prefix="r2er")
    state = hs.horse_state(frame, "p_r1")
    values = entity.merge(horse, on=KEYS, validate="one_to_one").merge(state, on=KEYS, validate="one_to_one")
    values = values.rename(columns={c: f"r2{c}" for c in values if c.startswith(("mg_", "hs_"))})
    return values, [c for c in values if c not in KEYS]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    base = soft.frames()["long"]
    frame = round_one(base, args.threads)
    print("round-one history:", len(frame), "runners from", frame.race_date.min().date(), flush=True)
    values, columns = round_two_inputs(frame)
    frame = s.attach(frame, values)
    print("round-two inputs:", len(columns), flush=True)
    made, top = {}, None
    for label, data in (("real", frame), ("twin", s.permute_within(frame, columns, seed=TWIN_SEED))):
        fit = soft.soft_boost(columns, target="y_soft", base="p_r1", rounds=soft.ARMS["soft-long-rich"]["rounds"],
                              params=soft.RICH, threads=args.threads)
        correction, _ = s.walk_forward(data, fit)
        made[label] = correction[data.quarter.isin(s.SCORED).to_numpy()]
        top = top or fit.info[-1]["top_inputs"]
    scored = frame.loc[frame.quarter.isin(s.SCORED), [*KEYS, "race_date", "quarter", "won", "p_single", "c1"]]
    scored = native_frame(scored.reset_index(drop=True))
    bounds = s.bounds_of(scored.race_id.to_numpy())
    races = scored.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    report = {"inputs": len(columns), "top_inputs": top}
    for name, column in (("screen", "p_single"), ("native", "p_native")):
        first = np.log(scored[column].to_numpy()) + scored.c1.to_numpy()
        loss = {k: race_losses(scored, first + v) for k, v in made.items()}
        loss["round_one"] = race_losses(scored, first)
        report[name] = {"round_one_log_loss": float(loss["round_one"].mean()),
                        "round_two_log_loss": float(loss["real"].mean()),
                        "round_two_vs_round_one": s.paired(loss["real"], loss["round_one"], races),
                        "twin_vs_round_one": s.paired(loss["twin"], loss["round_one"], races),
                        "round_two_vs_twin": s.paired(loss["real"], loss["twin"], races)}
        for key in ("round_two_vs_round_one", "twin_vs_round_one", "round_two_vs_twin"):
            r = report[name][key]
            print(f"{name:<7} {key:<24} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] "
                  f"{r['quarters_better']}/4", flush=True)
    atomic_parquet(scored[KEYS].assign(c2=made["real"], c2_twin=made["twin"]), OUT / "second_round.parquet")
    atomic_json(REPORT, report)
    print("top inputs:", top)


if __name__ == "__main__":
    main()
