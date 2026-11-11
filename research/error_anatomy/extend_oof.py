"""Out-of-fold full 063 forecasts for the quarters before the benchmark's first fold.

    python -m research.error_anatomy.extend_oof [--threads 5]

The benchmark cached seed-42 forecasts from 2024Q3. Residual inputs that
measure entities against the model need a longer memory, so this fits the same
registered recipe (seed 42, screen capacity, purged expanding quarters) for
2022Q3-2024Q2 and saves the forecasts beside this study. Nothing in the
benchmark folder is written.
"""
import argparse
import logging

from research.error_anatomy.frame import OUT
from research.meta_learner import first_ensemble as fe
from research.meta_learner import tier2_inputs as t2
from src.ingestion.storage import atomic_parquet
from src.modelling.evaluation import component_oof

EARLY = ("2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4", "2024Q1", "2024Q2")
TARGET = OUT / "early_oof_seed42.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    OUT.mkdir(parents=True, exist_ok=True)
    registered = t2.design()
    spec = t2.full_spec(registered)
    frame = fe.read_input(t2.STUDY, registered, list(spec.features))
    folds = fe.folds_for(frame, EARLY, spec.purge_days)
    result, _ = component_oof(frame, [spec], folds=folds, capacity="screen", threads=args.threads,
                              dataset_start=registered["receipt"]["dataset_start"], cache_dir=OUT / "cache_oof")
    result.attrs = {}
    atomic_parquet(result[["race_id", "horse_id", "race_date", "fold", "probability"]], TARGET)
    print(f"{len(result)} forecasts for {result.fold.nunique()} quarters", flush=True)


if __name__ == "__main__":
    main()
