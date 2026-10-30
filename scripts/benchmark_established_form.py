"""One resumable matched comparison of the fixed established-horse family."""
from pathlib import Path
import argparse
import importlib.metadata
import gc
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config

from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import score_pair
from scripts.canonical_training_cache import CheckpointPredictor
from src.deployment_validation import file_sha256 as sha
from src.forecast_evaluation import valid_race_mask
from src.featured_dataset import fe_signature

OUTPUT = ROOT / "data/audit/established_form_029"
KEY = ["race_id", "horse_name"]


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def verify(manifest):
    if fe_signature() != manifest["fe_signature"] or sha(OUTPUT / "inputs.parquet") != manifest["inputs_sha256"]:
        raise ValueError("The audited feature inputs changed")
    if sha(ROOT / "data/processed/betfair_bsp.parquet") != manifest["teacher_sha256"]:
        raise ValueError("Keep teacher observations identical in both arms")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in manifest.get("reference_sha256", {}).items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Frozen evaluation reference changed: {name}")
    for name, expected in manifest.get("runtime", {}).items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Training runtime changed: {name}")


def run():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    verify(manifest)
    data = pd.read_parquet(OUTPUT / "inputs.parquet")
    dataset_start = data.race_date.min()
    data = data.loc[valid_race_mask(data)].copy()
    test = data.loc[data.race_date.between(manifest["evaluation_start"], manifest["evaluation_end"])].copy()
    reference = pd.read_parquet(ROOT / "data/audit/canonical_repairs_026/historical_predictions.parquet")
    if set(test.race_id.unique()) != set(reference.race_id.unique()):
        raise ValueError("Do not change the previously complete evaluation fields")
    frame = test[[c for c in ("race_id", "horse_name", "race_date", "won", "finish_position", "num_runners", "handicap", "race_type") if c in test]].copy()
    train = data.loc[data.race_date.le(manifest["history_cutoff"])].copy()
    del data
    gc.collect()
    original_allowlist = config.FEATURE_ALLOWLIST
    started = time.monotonic()
    try:
        for label in ("baseline", "candidate"):
            columns = manifest[f"{label}_features"]
            config.FEATURE_ALLOWLIST = columns
            path = OUTPUT / f"{label}.joblib"
            predictor = CheckpointPredictor(random_seed=manifest["seed"], frameworks={"classifier": "lgbm", "place": "lgbm"},
                checkpoint_directory=OUTPUT / "components" / label, checkpoint_manifest=sha(OUTPUT / "manifest.json"))
            predictor._autotune_njobs = manifest["threads"]
            write(OUTPUT / "status.json", {"stage": f"{label}_fit", "features": len(columns), "cutoff": manifest["history_cutoff"]})
            if path.exists():
                predictor.load(str(path))
                if predictor.eval_split_info.get("training_checkpoint_manifest") != sha(OUTPUT / "manifest.json"):
                    raise ValueError("Cached learner belongs to another comparison")
            else:
                predictor.fit_as_of(train, manifest["history_cutoff"], dataset_start=dataset_start, win_only=True)
                verify(manifest)
                predictor.save(str(path))
            if predictor.feature_cols != columns:
                raise ValueError("Training changed the audited input order")
            frame[f"{label}_probability"] = align_predictions(predictor, test)
            write(OUTPUT / f"{label}_fit.json", predictor.eval_split_info)
            print(json.dumps({"stage": f"{label}_complete", "races": test.race_id.nunique(), "runners": len(test),
                "selected_trees": predictor.eval_split_info.get("refit_rounds")}), flush=True)
            del predictor
            gc.collect()
    finally:
        config.FEATURE_ALLOWLIST = original_allowlist
    verify(manifest)
    frame.to_parquet(OUTPUT / "predictions.parquet", index=False)
    comparison = score_pair(frame)
    market = pd.read_parquet(ROOT / "data/audit/canonical_repairs_026/market_matched_predictions.parquet",
                             columns=KEY+["race_date", "won", "market_probability"])
    paired = frame.merge(market, on=KEY+["race_date", "won"], how="inner", validate="one_to_one")
    if paired.race_id.nunique() != 2469 or not valid_race_mask(paired).all():
        raise ValueError("Retain the frozen2469 complete market fields")
    old_context = pd.read_parquet(ROOT / "data/audit/established_horse_review_028/paired_context.parquet", columns=KEY+["horse_prev_races"])
    paired = paired.merge(old_context, on=KEY, validate="one_to_one")
    market_scores = score_forecasts(paired, {name: paired[f"{name}_probability"].to_numpy() for name in ("baseline", "candidate", "market")})
    cohorts = {}
    for n in (3, 5):
        ids = paired.groupby("race_id").horse_prev_races.min().loc[lambda v:v.ge(n)].index
        cohort = paired.loc[paired.race_id.isin(ids)]
        cohorts[f"every_runner_{n}_local_starts"] = score_pair(cohort)
    for label, condition in (("handicaps", paired.handicap.eq(1)), ("conditions", paired.handicap.eq(0))):
        cohorts[label] = score_pair(paired.loc[condition])
    paired.to_parquet(OUTPUT / "market_paired_predictions.parquet", index=False)
    report = {"scope": manifest["scope"], "comparison": comparison, "market": market_scores,
        "cohorts": cohorts, "elapsed_seconds": time.monotonic()-started,
        "manifest_sha256": sha(OUTPUT / "manifest.json"), "decision_rule": manifest["decision_rule"],
        "model_promoted": False}
    write(OUTPUT / "report.json", report)
    write(OUTPUT / "status.json", {"stage": "comparison_complete", "report": str(OUTPUT / "report.json")})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()
    OUTPUT = Path(args.output_dir).resolve()
    run()
