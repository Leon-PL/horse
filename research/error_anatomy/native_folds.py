"""Native-capacity full 063 forecasts for more quarters, and the corrections applied to them.

    python -m research.error_anatomy.native_folds fit --quarters 2025Q4 2026Q1 2026Q2 [--threads 6]
    python -m research.error_anatomy.native_folds check

Every correction in this study was learned and scored against 400-tree screen
fits; one native-capacity forecast existed (2025Q3). ``fit`` makes seed-42
native forecasts for the other scored quarters by the same purged refits.
``check`` adds the saved corrections (each group alone, the weighted stack and
the single joint model) to the screen and to the native forecast on the same
races, so the size of the gain at serving capacity is measured, not inferred.

The corrections themselves are unchanged: they were fitted from the screen
forecast's offset. A correction refitted from native forecasts could do better
or worse; that is not tested here.
"""
import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from research.error_anatomy.native_check import NATIVE
from research.meta_learner import first_ensemble as fe
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s
from research.meta_learner import tier2_inputs as t2
from research.meta_learner.specialists import race_losses
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.data import CONTEXT
from src.modelling.evaluation import component_oof

KEYS = s.KEYS
ROOT = OUT / "native"
STACK = "soft_stack_soft-long-rich"
GROUPS = ("horse", "market-gap", "residual", "wave2", "same-day-bias", "tracking", "horse-state", "entity-recent")


def fit(quarters, threads):
    registered = t2.design()
    spec = replace(t2.full_spec(registered), id="full-063-native")
    frame = pd.read_parquet(s.STUDY / "input.parquet", columns=list(dict.fromkeys([*CONTEXT, *spec.features])))
    frame.attrs["modelling_receipt"] = registered["receipt"]
    ROOT.mkdir(parents=True, exist_ok=True)
    for quarter in quarters:
        target = ROOT / f"native_{quarter}.parquet"
        if target.exists():
            continue
        folds = fe.folds_for(frame, (quarter,), spec.purge_days)
        result, _ = component_oof(frame, [spec], folds=folds, capacity="native", threads=threads,
                                  dataset_start=registered["receipt"]["dataset_start"], cache_dir=ROOT / "cache")
        result.attrs = {}
        atomic_parquet(result[[*KEYS, "race_date", "fold", "probability"]], target)
        print(quarter, len(result), flush=True)


def check():
    parts = [pd.read_parquet(NATIVE, columns=[*KEYS, "probability"])]
    parts += [pd.read_parquet(p, columns=[*KEYS, "probability"]) for p in sorted(ROOT.glob("native_*.parquet"))]
    native = pd.concat(parts, ignore_index=True).rename(columns={"probability": "p_native"})
    native["race_id"] = native.race_id.astype(str)
    frame = sc.base_frame()
    frame = frame.loc[frame.quarter.isin(s.SCORED)].reset_index(drop=True)
    frame = frame.merge(native, on=KEYS, how="inner", validate="one_to_one")
    whole = frame.groupby("race_id").race_id.transform("size").eq(frame.num_runners)
    frame = frame.loc[whole].reset_index(drop=True)
    frame = s.attach(frame, pd.read_parquet(OUT / f"{STACK}.parquet"))
    frame = s.attach(frame, pd.read_parquet(OUT / "joint_corrector.parquet")[[*KEYS, "joint-wide"]])
    weights = {row["fold"]: row for row in json.loads((OUT / f"{STACK}_state.json").read_text())["coefficients"]}
    stack = np.zeros(len(frame))
    for quarter, row in weights.items():
        rows = frame.quarter.eq(quarter).to_numpy()
        stack[rows] = sum(row[f"{g}:c"] * frame.loc[rows, f"{g}:c"].to_numpy() for g in GROUPS)
    corrections = {"weighted stack of eight": stack, "one joint model": frame["joint-wide"].to_numpy(),
                   **{f"{g} alone": frame[f"{g}:c"].to_numpy() for g in GROUPS}}
    bounds = s.bounds_of(frame.race_id.to_numpy())
    races = frame.iloc[bounds[:-1]][["race_id", "race_date", "quarter"]].reset_index(drop=True)
    races["covered"] = True
    bases = {"screen": np.log(frame.p_single.to_numpy()), "native": np.log(frame.p_native.to_numpy())}
    loss = {k: race_losses(frame, v) for k, v in bases.items()}
    report = {"quarters": sorted(frame.quarter.unique()), "races": len(races),
              "race_log_loss": {k: float(v.mean()) for k, v in loss.items()},
              "native_vs_screen": s.paired(loss["native"], loss["screen"], races), "corrections": {}}
    print(report["quarters"], report["races"], {k: round(v, 5) for k, v in report["race_log_loss"].items()})
    print(f"{'correction':<26}{'on screen':>11}{'on native':>11}   native 95% interval      quarters  by quarter")
    for name, c in corrections.items():
        on = {k: s.paired(race_losses(frame, base + c), loss[k], races) for k, base in bases.items()}
        report["corrections"][name] = on
        n = on["native"]
        print(f"{name:<26}{on['screen']['mean']:>+11.5f}{n['mean']:>+11.5f}   [{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}]   "
              f"{n['quarters_better']}/{len(report['quarters'])}   {n['per_quarter']}")
    atomic_json(OUT / "native_folds_check.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["fit", "check"])
    parser.add_argument("--quarters", nargs="+", default=["2025Q4", "2026Q1", "2026Q2"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    fit(args.quarters, args.threads) if args.stage == "fit" else check()


if __name__ == "__main__":
    main()
