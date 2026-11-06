"""014: compare corrected result histories with the frozen native149 reference.

Resumable, complete-quarter fits. Each process owns its configuration; at
most two processes with two learner threads each run on this 16GB machine.
The baseline is replayed on OLD features, never on corrected histories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config

from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_history_features import align_reference
from scripts.benchmark_independent_features import META, advancement, canonical_hash, score_pair
from scripts.replay_market_snapshots import digest
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.wf_eval import development_mask, expanding_period_folds

OUTPUT = ROOT / "data/audit/model_improvement/014_margin_repair"
REFERENCE = ROOT / "data/audit/model_improvement/009_pace_family"
STORE = OUTPUT / "candidate_store/featured.parquet"
PERIODS = ("2025Q3", "2025Q4", "2026Q1")
SOURCES = list(dict.fromkeys([*_FE_SOURCE_FILES, "src/model.py", "src/betfair.py", "config.py",
    "src/ingestion/sporting_life.py",
    "src/forecast_evaluation.py", "src/probability_calibration.py", "src/wf_eval.py",
    "scripts/benchmark_margin_repair.py", "scripts/audit_prediction_inputs.py",
    "scripts/audit_result_integrity.py", "scripts/benchmark_independent_features.py",
    "scripts/build_feature_store_batched.py",
    "scripts/benchmark_forecasts.py", "scripts/benchmark_history_features.py"]))


def write(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True)
    if path.name == "manifest.json" and path.exists() and path.read_text() != encoded:
        raise ValueError("Manifest differs; do not overwrite a registered experiment")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def sources():
    return {p: digest(ROOT / p) for p in SOURCES}


def prepare():
    previous = json.loads((REFERENCE / "manifest.json").read_text())
    if digest(REFERENCE / "inputs.parquet") != previous["projection_sha256"]:
        raise ValueError("Frozen reference projection changed")
    names = previous["variants"]["baseline"]
    params = {"classifier": {**config.CLASSIFIER_PARAMS, "n_estimators": 2000}}
    if (params != previous["params"] or config.WIN_SOFT_LABEL_DISTILL
            or config.WIN_CALIBRATION_METHOD != previous["calibration"]
            or config.RECENCY_HALF_LIFE_DAYS != previous["recency_half_life_days"]):
        raise ValueError("Keep the native hard-label reference training recipe")
    audit = json.loads((OUTPUT / "input_invariance.json").read_text())
    if (not audit["passed"] or set(audit["selected_inputs"]) != set(names)
            or audit.get("fe_signature") != fe_signature()):
        raise ValueError("All selected inputs must pass the comprehensive audit")
    meta = json.loads(STORE.with_name("meta.json").read_text())
    if meta["fe_signature"] != fe_signature():
        raise ValueError("Candidate store and feature code differ")
    cols = list(dict.fromkeys(names + META + ["beaten_lengths"]))
    candidate = pd.read_parquet(STORE, columns=cols, filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    oldcols = [c for c in cols if c != "beaten_lengths"]
    old = pd.read_parquet(REFERENCE / "inputs.parquet", columns=oldcols)
    keys = ["race_date", "race_id", "horse_name"]
    candidate = candidate.sort_values(keys, kind="stable").reset_index(drop=True)
    old = old.sort_values(keys, kind="stable").reset_index(drop=True)
    if not candidate[keys].equals(old[keys]):
        raise ValueError("Corrected and original histories contain different rows")
    for col in ("won", "finish_position", "num_runners", "odds", "lengths_behind"):
        if not np.array_equal(candidate[col], old[col], equal_nan=True):
            raise ValueError(f"Original source/labels changed: {col}")
    profile = {}
    for col in names:
        a, b = old[col].to_numpy(np.float32), candidate[col].to_numpy(np.float32)
        changed = ~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)
        if changed.any():
            finite = np.isfinite(a) & np.isfinite(b)
            profile[col] = {"changed_rows": int(changed.sum()),
                            "max_finite_difference": float(np.abs(a[finite] - b[finite]).max()) if finite.any() else None}
    unexpected = [c for c in profile if not c.startswith(("horse_margin_", "speed_"))]
    write(OUTPUT / "feature_change_profile.json", {"changes": profile, "unexpected_columns": unexpected})
    if unexpected:
        raise ValueError(f"Unexpected feature changes require investigation: {unexpected}")
    if not profile:
        raise ValueError("Correction did not reach the selected model inputs")
    del old
    candidate[names] = candidate[names].astype(np.float32)
    candidate.to_parquet(OUTPUT / "inputs.parquet", index=False, compression="zstd")
    valid = valid_race_mask(candidate)
    eligible = candidate.loc[valid & development_mask(candidate.race_date)].reset_index(drop=True)
    folds = {f.label: f for f in expanding_period_folds(eligible.race_date)}
    cutoffs = [{"period": p, "cutoff": str(eligible.loc[folds[p].train, "race_date"].max())[:10],
                "test_start": str(eligible.loc[folds[p].test, "race_date"].min())[:10],
                "test_end": str(eligible.loc[folds[p].test, "race_date"].max())[:10]} for p in PERIODS]
    if cutoffs != previous["folds"]:
        raise ValueError("Original and corrected evaluation cutoffs differ")
    manifest = {"experiment": "014 correct adjacent-gap margins and row-dependent class scale",
        "features": names, "params": params, "seed": 42, "learner_threads": 2,
        "max_processes": 2, "folds": cutoffs, "source_sha256": sources(),
        "inputs_sha256": digest(OUTPUT / "inputs.parquet"), "candidate_fe_signature": fe_signature(),
        "reference_manifest_sha256": digest(REFERENCE / "manifest.json"),
        "reference_predictions_sha256": digest(REFERENCE / "paired_predictions.parquet"),
        "reference_models_sha256": {p: digest(REFERENCE / f"{p}_baseline.joblib") for p in PERIODS},
        "dataset_start": str(candidate.race_date.min())[:10], "calibration": config.WIN_CALIBRATION_METHOD,
        "recency_half_life_days": config.RECENCY_HALF_LIFE_DAYS,
        "invariance_sha256": digest(OUTPUT / "input_invariance.json"),
        "data_integrity_sha256": digest(OUTPUT / "data_integrity.json"),
        "rule": "All3quarters; pooled race log-loss CI upper<0, Brier and binary log loss no worse, >=2quarters improve",
        "scope": "Retrospective research on previously examined quarters; no claim of fresh confirmation",
        "reference_compatibility": "Core source changed deliberately; exact native replay on original inputs is required. Hard-label learner recipe unchanged."}
    write(OUTPUT / "manifest.json", manifest)
    if not (OUTPUT / "source_snapshot.zip").exists():
        with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in SOURCES:
                z.write(ROOT / p, p)
    print(json.dumps({"stage": "registered", "manifest_sha256": canonical_hash(manifest),
                      "changed_selected_inputs": list(profile), "folds": cutoffs}), flush=True)
    del eligible, candidate
    gc.collect()
    return manifest


def run_fold(period):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    manifest_hash = canonical_hash(manifest)
    if sources() != manifest["source_sha256"]:
        raise ValueError("Registered sources changed")
    if digest(OUTPUT / "inputs.parquet") != manifest["inputs_sha256"]:
        raise ValueError("Registered projection changed")
    names = manifest["features"]
    config.FEATURE_ALLOWLIST = names
    frame = pd.read_parquet(OUTPUT / "inputs.parquet")
    frame = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    fold = {f.label: f for f in expanding_period_folds(frame.race_date)}[period]
    test = frame.loc[fold.test].copy()
    old = pd.read_parquet(REFERENCE / "inputs.parquet", columns=list(dict.fromkeys(names + META)))
    old = old.loc[old.race_id.isin(test.race_id)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    if not old[["race_id", "horse_name"]].reset_index(drop=True).equals(test[["race_id", "horse_name"]].reset_index(drop=True)):
        raise ValueError("Native baseline replay uses different fields")
    baseline = RacePredictor()
    if digest(REFERENCE / f"{period}_baseline.joblib") != manifest["reference_models_sha256"][period]:
        raise ValueError("Frozen baseline model changed")
    baseline.load(str(REFERENCE / f"{period}_baseline.joblib"))
    native = align_predictions(baseline, old)
    frozen = pd.read_parquet(REFERENCE / "paired_predictions.parquet")
    base_p, replay_error = align_reference(test, frozen, native)
    del baseline, old, frozen
    gc.collect()
    predictor = RacePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 2
    model_path = OUTPUT / f"{period}_candidate.joblib"
    cutoff = frame.loc[fold.train, "race_date"].max()
    if model_path.exists():
        predictor.load(str(model_path))
        if predictor.eval_split_info.get("experiment_manifest_sha256") != manifest_hash:
            raise ValueError("Cached model belongs to a different experiment")
    else:
        train = frame.loc[fold.train].copy()
        if get_feature_columns(train) != names:
            raise ValueError("Native feature selection changed")
        print(json.dumps({"stage": "fitting", "period": period, "cutoff": str(cutoff.date()), "rows": len(train)}), flush=True)
        predictor.fit_as_of(train, cutoff, params=manifest["params"], dataset_start=pd.Timestamp(manifest["dataset_start"]), win_only=True)
        predictor.eval_split_info.update(experiment_manifest_sha256=manifest_hash, variant="candidate", period=period)
        if sources() != manifest["source_sha256"]:
            raise ValueError("Sources changed during fitting")
        predictor.save(str(model_path))
        del train
    scored = test[[c for c in META if c in test]].copy()
    scored["baseline_probability"] = base_p
    scored["candidate_probability"] = align_predictions(predictor, test)
    scored.to_parquet(OUTPUT / f"{period}_predictions.parquet", index=False)
    report = {"period": period, "manifest_sha256": manifest_hash,
              "baseline_native_replay_max_error": replay_error, **score_pair(scored)}
    write(OUTPUT / f"{period}_fit.json", predictor.eval_split_info)
    write(OUTPUT / f"{period}_report.json", report)
    print(json.dumps({"stage": "fold_complete", "period": period, "scores": report}), flush=True)
    return report


def summarise():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    if sources() != manifest["source_sha256"]:
        raise ValueError("Sources changed during experiment")
    reports = [json.loads((OUTPUT / f"{p}_report.json").read_text()) for p in PERIODS]
    if any(r["manifest_sha256"] != canonical_hash(manifest) for r in reports):
        raise ValueError("Completed fold does not match registration")
    paired = pd.concat([pd.read_parquet(OUTPUT / f"{p}_predictions.parquet") for p in PERIODS]).sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable")
    pooled = score_pair(paired)
    market = pd.read_parquet(REFERENCE / "market_context_predictions.parquet").drop(columns="candidate_probability")
    market = market.merge(paired[["race_id", "horse_name", "candidate_probability"]], on=["race_id", "horse_name"], how="left", validate="one_to_one")
    if market.candidate_probability.isna().any():
        raise ValueError("Market overlap lost a runner")
    market_scores = score_forecasts(market, {n: market[f"{n}_probability"].to_numpy(float) for n in ("baseline", "candidate", "market")})
    report = {"manifest_sha256": canonical_hash(manifest), "folds": reports, "pooled": pooled,
              "candidate_passes_numeric_advancement": advancement(pooled, reports),
              "market": market_scores, "scope": manifest["scope"], "production_changed": False}
    winners = paired.loc[paired.won.eq(1)]
    report["extreme_probability_diagnostics_not_selection"] = {
        name: {"winner_probability_below_0_001": int(winners[f"{name}_probability"].lt(.001).sum()),
               "winner_probability_at_or_below_1e_8": int(winners[f"{name}_probability"].le(1e-8).sum()),
               "minimum_winner_probability": float(winners[f"{name}_probability"].min())}
        for name in ("baseline", "candidate")}
    paired.to_parquet(OUTPUT / "paired_predictions.parquet", index=False)
    market.to_parquet(OUTPUT / "market_context_predictions.parquet", index=False)
    write(OUTPUT / "report.json", report)
    write(OUTPUT / "completion.json", {"manifest_sha256": canonical_hash(manifest), "completed_periods": list(PERIODS),
          "evidence_sha256": {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name not in {"completion.json"}}})
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--period", choices=PERIODS)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.summarise:
        summarise()
    elif args.period:
        run_fold(args.period)
    else:
        if not (OUTPUT / "manifest.json").exists():
            prepare()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run_fold, PERIODS))
        summarise()


if __name__ == "__main__":
    main()
