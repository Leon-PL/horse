"""Are the kept inputs working as well as they can? Repairs measured inside the 87-input model.

    python -m research.error_anatomy.input_repairs --arms baseline weight-rebuilt ...

``kept_inputs_audit.py`` profiled the 87 inputs of the 12 November selection.
Each arm below changes or adds a few columns and refits the same recipe on the
confirmation folds (races from 8 July 2025, reduced capacity), on the matrix
the selection used. Arms, declared before scoring:

- ``weight-rebuilt``: ``weight_vs_field`` and ``weight_change_vs_field`` as the
  store rebuilt on 13 November has them. They are now claim-adjusted and
  missing wherever the rider's claim is unverified (31% of 2020-2023 runners);
  the selection's store had them for every runner. Measures what that cost.
- ``weight-fallback``: the claim-adjusted value where it exists, the plain
  allocated-weight value otherwise.
- ``signed-margin``: the margin-based figures give every winner 0 lengths, so
  36% of ``speed_fig_best_5`` is exactly zero and a ten-length winner equals a
  nose winner. Adds the same figures with winners credited the distance back
  to the second (negative), as last run, best and mean of five, and against
  the field.
- ``hidden-form-masked``: the card form figures of runs the database lacks are
  filled for 36% of 2021 runners and 6% of 2025 runners, because early on they
  are mostly ordinary runs from before the database starts. Blanks them for
  horses first seen in the database's first six months.
"""
import argparse
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.spec import ModelSpec
from src.modelling.selection import _evaluate, _split_folds
from src.forecast_evaluation import paired_interval
from src.modelling.spec import read_features

PREPARED = Path("data/research/selection_20261111/matrix")
STORE = "data/processed/feature_store/featured.parquet"
LIST = "feature_sets/win_selected_20261112.txt"
OUT = Path("data/research/selection_20261111/repairs")
START = "2025-07-08"
KEYS = ["race_id", "horse_id"]


def store_columns(context, columns):
    store = pd.read_parquet(STORE, columns=[*KEYS, *columns])
    store["race_id"] = store.race_id.astype(context.race_id.dtype)
    return context[KEYS].merge(store, on=KEYS, how="left", validate="one_to_one")


@nb.njit(cache=True)
def _last_five(horse, day, value):
    n = len(value)
    out = np.full((n, 3), np.nan)
    start = 0
    for i in range(n):
        if i == 0 or horse[i] != horse[i - 1]:
            start = i
        total = 0.
        count = 0
        best = np.inf
        last = np.nan
        for j in range(i - 1, start - 1, -1):
            if day[j] == day[i] or np.isnan(value[j]):
                continue
            if count == 0:
                last = value[j]
            total += value[j]
            best = min(best, value[j])
            count += 1
            if count == 5:
                break
        if count:
            out[i, 0], out[i, 1], out[i, 2] = last, best, total / count
    return out


def signed_margin(context):
    """Lengths behind per furlong^0.6 with winners at minus the runner-up's deficit; history on earlier dates."""
    store = pd.read_parquet(STORE, columns=[*KEYS, "race_date", "finish_position", "lengths_behind", "distance_furlongs"])
    position = pd.to_numeric(store.finish_position, errors="coerce")
    behind = pd.to_numeric(store.lengths_behind, errors="coerce").where(position >= 1)
    second = behind.where(position == 2).groupby(store.race_id).transform("min")
    signed = np.where(position == 1, -second, behind)
    scale = np.power(store.distance_furlongs.replace(0, np.nan).fillna(8.).astype(float), .6)
    store["signed"] = np.clip(signed / scale, -6, 20)
    store = store.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index(drop=True)
    day = (store.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    made = _last_five(store.horse_id.to_numpy(np.int64), day, store.signed.to_numpy(float))
    out = store[KEYS].copy()
    out["sm_last"], out["sm_best_5"], out["sm_avg_5"] = made.T
    for column in ("sm_best_5", "sm_avg_5"):
        out[f"{column}_vs_field"] = out[column] - out[column].groupby(store.race_id.to_numpy()).transform("mean")
    out["race_id"] = out.race_id.astype(context.race_id.dtype)
    return context[KEYS].merge(out, on=KEYS, how="left", validate="one_to_one").drop(columns=KEYS)


def arms(context, base):
    """name -> (replaced columns, added columns), built lazily."""
    def weight_rebuilt():
        new = store_columns(context, ["weight_vs_field", "weight_change_vs_field"])
        return {c: new[c].to_numpy(float) for c in ("weight_vs_field", "weight_change_vs_field")}, {}

    def weight_fallback():
        new = store_columns(context, ["weight_vs_field", "weight_change_vs_field"])
        return {c: np.where(new[c].notna(), new[c].to_numpy(float), base(c)) for c in
                ("weight_vs_field", "weight_change_vs_field")}, {}

    def signed():
        made = signed_margin(context)
        return {}, {c: made[c].to_numpy(float) for c in made}

    def hidden_masked():
        first = context.groupby("horse_id").race_date.transform("min")
        early = (first < pd.Timestamp("2021-03-15")).to_numpy()
        return {"hidden_form_mean": np.where(early, np.nan, base("hidden_form_mean")),
                "hidden_form_best_vs_field": np.where(early, 0., base("hidden_form_best_vs_field"))}, {}

    return {"baseline": lambda: ({}, {}), "weight-rebuilt": weight_rebuilt, "weight-fallback": weight_fallback,
            "signed-margin": signed, "hidden-form-masked": hidden_masked}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    description = json.loads((PREPARED / "prepared.json").read_text())
    names = description["features"]
    context = pd.read_parquet(PREPARED / "context.parquet")
    matrix = np.load(PREPARED / "matrix.npy", mmap_mode="r")
    lookup = {name: i for i, name in enumerate(names)}
    spec = ModelSpec("generalist", read_features(LIST))
    _, folds = _split_folds(context, spec, None, pd.Timestamp(START), 14)
    base = lambda name: np.asarray(matrix[:, lookup[name]], dtype=np.float64)  # noqa: E731
    builders = arms(context, base)
    for name in args.arms:
        path = OUT / f"{name}.parquet"
        if path.exists():
            continue
        replaced, added = builders[name]()
        features = [*spec.features, *added]
        column = lambda c: replaced[c] if c in replaced else added[c] if c in added else base(c)  # noqa: E731
        races, _ = _evaluate(context, column, spec, features, folds, dataset_start=description["receipt"]["dataset_start"],
                             capacity="screen", threads=args.threads)
        atomic_parquet(races, path)
        print(f"{name}: fitted", flush=True)
    report = {}
    reference = pd.read_parquet(OUT / "baseline.parquet")
    for path in sorted(OUT.glob("*.parquet")):
        races = pd.read_parquet(path)
        if path.stem == "baseline":
            report["baseline"] = {"race_log_loss": float(races.race_log_loss.mean()), "races": int(len(races))}
            continue
        if not races.race_id.equals(reference.race_id):
            raise ValueError(f"{path.stem} scored different races")
        change = races.race_log_loss - reference.race_log_loss
        by_fold = change.groupby(races.fold.to_numpy()).mean()
        report[path.stem] = {"race_log_loss": float(races.race_log_loss.mean()),
                             **paired_interval(change, blocks=races.race_date),
                             "per_fold": {k: float(v) for k, v in by_fold.items()},
                             "folds_better": int((by_fold < 0).sum())}
        r = report[path.stem]
        print(f"{path.stem:<20} {r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}] {r['folds_better']}/4")
    atomic_json(OUT / "repairs.json", report)


if __name__ == "__main__":
    main()
