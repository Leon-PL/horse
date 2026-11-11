"""One registered read of the error corrections on races the choices have not seen.

    python -m research.error_anatomy.holdout prepare
    python -m research.error_anatomy.holdout fit [--threads 6]
    python -m research.error_anatomy.holdout register
    python -m research.error_anatomy.holdout score

The corrections in ``entity_residuals.py`` and ``soft_corrections.py`` were
chosen on development quarters 2025Q3-2026Q2. This reads them once on the
races from ``config.GOLDEN_HOLDOUT_START`` (11 June 2026) to the store's last
date, with the recipe, groups and statistic written to ``registration.json``
before any of those races is scored.

- ``prepare`` rebuilds the benchmark's prepared inputs for every date and
  checks that the development rows equal the benchmark's own file.
- ``fit`` makes seed-42 full 063 forecasts for 2026Q2-2026Q4 by the same purged
  quarterly refits as development (screen capacity).
- ``register`` writes the frozen protocol. It refuses if a score exists.
- ``score`` computes the inputs and corrections walk-forward, scores once and
  writes a receipt. It refuses a second run.
"""
import argparse
from dataclasses import replace
import json
import logging

import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from research.meta_learner import features_063
from research.meta_learner import first_ensemble as fe
from research.meta_learner import screening as s
from research.meta_learner import tier2_inputs as t2
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.data import CONTEXT, load_dataset
from src.modelling.evaluation import component_oof
from src.source_dataset import load_manifest, source_path, use_manifest

ROOT = OUT / "holdout"
KEYS = fe.KEYS
FOLDS = ("2026Q2", "2026Q3", "2026Q4")


def frozen_read(registered, columns):
    """The benchmark's store, every date, by the steps ``load_dataset`` applies after its checks.

    ``load_dataset`` refuses this store because feature code in the working tree has moved on since
    9 November (its signature guard, which protects training and serving from stale features). The
    store itself is frozen and is what the benchmark was fitted on, so it is read directly here and
    the parity check in ``prepare`` proves the result equals the benchmark's own rows.
    """
    from pathlib import Path
    from src.modelling.data import ordered_frame, valid_race_mask
    from src.source_prices import attach_bsp
    receipt = dict(registered["receipt"])
    manifest = load_manifest(receipt["dataset_manifest"]["id"], root=receipt["dataset_manifest"].get("root"))
    names = list(dict.fromkeys([c for c in CONTEXT if c != "bsp"] + list(columns) + list(features_063.HISTORY)))
    frame = pd.read_parquet(Path(receipt["store_root"]) / "featured.parquet", columns=names)
    frame.attrs = {}
    with use_manifest(manifest):
        frame = attach_bsp(frame, cols=("bsp",), overwrite=True)
    eligible = valid_race_mask(frame)
    bad = frame.loc[pd.to_numeric(frame.odds, errors="coerce").lt(1.01), "race_id"]
    eligible &= ~frame.race_id.isin(bad)
    frame, _ = ordered_frame(frame.loc[eligible])
    receipt["development_only"] = False
    receipt["read"] = "direct read of the frozen study store; see holdout.frozen_read"
    return frame, receipt


def prepare():
    registered = t2.design()
    full = t2.full_spec(registered)
    baseline = next(x for x in registered["specs"] if x["id"] == "generalist")
    columns = list(dict.fromkeys((*baseline["features"], *features_063.NUMERIC_INPUTS)))
    frame, receipt = frozen_read(registered, columns)
    frame["race_id"] = frame.race_id.astype(str)
    frame["horse_id"] = pd.to_numeric(frame.horse_id, errors="raise").astype("int64")
    with use_manifest(load_manifest(receipt["dataset_manifest"]["id"], root=receipt["dataset_manifest"].get("root"))):
        raw = pd.read_parquet(source_path("sl", "results"), columns=[*KEYS, "going"])
    raw["race_id"] = raw.race_id.astype(str)
    raw["horse_id"] = pd.to_numeric(raw.horse_id, errors="raise").astype("int64")
    raw = frame[KEYS].merge(raw, on=KEYS, how="left", validate="one_to_one")
    extra = pd.concat([features_063.rating_gaps(frame), features_063.relative_gaps(frame),
                       features_063.context_form(frame, raw.going),
                       frame[list(features_063.MODERATORS)].astype("float32").add_prefix("x063_context_")], axis=1)
    keep = list(dict.fromkeys([*CONTEXT, *baseline["features"]]))
    prepared = pd.concat([frame[keep], extra], axis=1)
    prepared.attrs = {}
    missing = set(full.features) - set(prepared)
    if missing:
        raise ValueError(f"Prepared inputs miss recipe features: {sorted(missing)[:5]}")
    # Parity: the development rows must equal the benchmark's prepared file, input by input.
    reference = pd.read_parquet(s.STUDY / "input.parquet", columns=[*KEYS, *full.features])
    reference.attrs = {}
    both = reference[KEYS].merge(prepared[[*KEYS, *full.features]], on=KEYS, how="left", validate="one_to_one")
    worst = {}
    for column in full.features:
        a = reference[column].to_numpy(float)
        b = both[column].to_numpy(float)
        differs = ~(np.isclose(a, b, rtol=1e-5, atol=1e-6) | (np.isnan(a) & np.isnan(b)))
        if differs.any():
            worst[column] = float(differs.mean())
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_parquet(prepared, ROOT / "input.parquet")
    report = {"rows": len(prepared), "last_date": str(prepared.race_date.max().date()),
              "development_rows_checked": len(reference), "inputs_checked": len(full.features),
              "inputs_differing": worst, "receipt": receipt}
    atomic_json(ROOT / "prepare.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "receipt"}, indent=1)[:3000])


def fit(threads):
    registered = t2.design()
    spec = replace(t2.full_spec(registered), id="full-063-holdout")
    frame = pd.read_parquet(ROOT / "input.parquet", columns=list(dict.fromkeys([*CONTEXT, *spec.features])))
    frame.attrs["modelling_receipt"] = json.loads((ROOT / "prepare.json").read_text())["receipt"]
    folds = fe.folds_for(frame, FOLDS, spec.purge_days)
    result, _ = component_oof(frame, [spec], folds=folds, capacity="screen", threads=threads,
                              dataset_start=registered["receipt"]["dataset_start"], cache_dir=ROOT / "cache")
    result.attrs = {}
    atomic_parquet(result[[*KEYS, "race_date", "fold", "probability"]], ROOT / "forecasts_seed42.parquet")
    print(result.groupby("fold").size().to_dict(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["prepare", "fit", "rehearse", "register", "score"])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "fit":
        fit(args.threads)
    else:
        from research.error_anatomy import holdout_score
        if args.stage == "register":
            holdout_score.register()
        else:
            getattr(holdout_score, args.stage)(args.threads)


if __name__ == "__main__":
    main()
