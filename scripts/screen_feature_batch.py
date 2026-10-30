"""One-window, matched, win-only screening after feature audits finish.

The cheap baseline is fitted at the same budget as the candidates. Historical
features keep their full prior history; only fitting rows are limited to two
years. A screen cannot qualify a production switch or prove market parity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.benchmark_forecasts import align_predictions
from scripts.benchmark_independent_features import META, canonical_hash, score_pair
from src.betfair import DEFAULT_BSP_PATH
from src.deployment_validation import RECIPE_SOURCES
from src.feature_semantics import history_coverage_features
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.public_history_features import FEATURES as PUBLIC_FEATURES
from src.wf_eval import development_mask, purge_days

OUTPUT = ROOT / "data/audit/model_improvement/018_feature_repairs"
BUNDLE = ROOT / "_research_snapshots/018_feature_repairs"
STORE = ROOT / "data/audit/feature_review_018/candidate_store_exact"
REFERENCE = ROOT / "data/audit/model_improvement/016_market_teacher"
TEST_START, TEST_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")
STAT_SOURCES = tuple(dict.fromkeys(["config.py", *RECIPE_SOURCES, "src/betfair.py", "src/bet_settlement.py", "src/each_way.py"]))


def runtime_versions():
    return {**{n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib")},
            "python": list(sys.version_info[:3])}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, default=str)
    if path.exists() and path.name == "manifest.json" and path.read_text() != encoded:
        raise ValueError("Do not amend an already registered screen")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def training_bounds():
    cutoff = TEST_START - pd.Timedelta(days=purge_days())
    return cutoff - pd.DateOffset(years=2), cutoff


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise ValueError("Screen already registered; resume without --prepare")
    audit = json.loads((OUTPUT / "input_invariance.json").read_text())
    if not audit["passed"]:
        raise ValueError("Finish the feature causality checks before any fitting")
    generation = json.loads((STORE / "meta.json").read_text())
    reviewed = json.loads((ROOT / "data/audit/feature_review_018/repaired_store_audit.json").read_text())
    if not reviewed["passed"] or reviewed["candidate_meta_sha256"] != sha(STORE / "meta.json"):
        raise ValueError("Review the rebuilt selected inputs before fitting")
    signature = subprocess.check_output([sys.executable, "-c",
        "from src.featured_dataset import fe_signature; print(fe_signature())"], cwd=BUNDLE, text=True).strip()
    if generation["fe_signature"] != signature or audit["fe_signature"] != signature:
        raise ValueError("Candidate source, built history and feature audit do not describe the same generation")
    reference = json.loads((REFERENCE / "manifest.json").read_text())
    if sha(REFERENCE / "inputs.parquet") != reference["inputs_sha256"]:
        raise ValueError("Reference features changed")
    features = reference["features"]
    from src.featured_dataset import _FE_SOURCE_FILES
    live_sources = tuple(dict.fromkeys([*STAT_SOURCES, "src/model.py", "src/betfair.py",
        "src/probability_calibration.py", "src/forecast_evaluation.py", "src/wf_eval.py",
        "src/feature_semantics.py", "src/public_history_features.py", "src/missing_history_snapshot.py",
        "scripts/build_missing_history_projection.py", "scripts/screen_feature_batch.py",
        "scripts/audit_repaired_store.py",
        "scripts/benchmark_forecasts.py", "scripts/benchmark_independent_features.py", "scripts/assess_forecast_parity.py"]))
    sources = {"live/"+n: sha(ROOT/n) for n in live_sources}
    sources.update({"candidate/"+n: sha(BUNDLE/n) for n in [*_FE_SOURCE_FILES, "src/feature_semantics.py",
                   "src/audited_surface_features.py", "src/live_prediction.py"]})
    low, cutoff = training_bounds()
    keys = ["race_date", "race_id", "horse_name"]
    extra = ["horse_prev_races", "days_since_last_race", "days_since_last_run", "prior_form_runs"]
    columns = list(dict.fromkeys(features + META + extra))
    baseline = pd.read_parquet(REFERENCE / "inputs.parquet")
    repaired = pd.read_parquet(STORE / "featured.parquet", columns=columns,
        filters=[("race_date", "<=", TEST_END)])
    baseline = baseline.loc[baseline.race_date.le(TEST_END)].sort_values(keys, kind="stable").reset_index(drop=True)
    repaired = repaired.sort_values(keys, kind="stable").reset_index(drop=True)
    if not baseline[keys].equals(repaired[keys]):
        raise ValueError("Original and repaired histories have different runners")
    for col in ["won", "finish_position", "num_runners", "odds"]:
        if not np.array_equal(baseline[col], repaired[col], equal_nan=True):
            raise ValueError(f"Original label or price changed: {col}")
    valid = valid_race_mask(baseline) & valid_race_mask(repaired) & development_mask(baseline.race_date)
    dates = baseline.race_date
    selected = valid & (((dates >= low) & (dates <= cutoff)) | dates.between(TEST_START, TEST_END))
    baseline, repaired = baseline.loc[selected].reset_index(drop=True), repaired.loc[selected].reset_index(drop=True)
    coverage = history_coverage_features(repaired)
    variants = {"baseline": baseline, "repairs": repaired,
                "coverage": pd.concat([repaired, coverage], axis=1)}
    external_root = ROOT / "data/audit/feature_review_018"
    external_receipt = json.loads((external_root / "combined_external_receipt_v2.json").read_text())
    external_path = external_root / "combined_external_features_v2.parquet"
    if sha(external_path) != external_receipt["projection_sha256"]:
        raise ValueError("Audited external form projection changed")
    for relative, expected in external_receipt["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("External form source changed")
    external = pd.read_parquet(external_path, columns=[*keys, *PUBLIC_FEATURES])
    variants["public"] = variants["coverage"].merge(external, on=keys, how="left", sort=False, validate="one_to_one")
    if not variants["public"][keys].equals(variants["coverage"][keys]) or variants["public"].external_prior_runs.isna().any():
        raise ValueError("External projection lost a registered runner")
    projection_hashes = {}
    changes = {}
    for name in features:
        a, b = baseline[name].to_numpy(np.float32), repaired[name].to_numpy(np.float32)
        count = int((~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).sum())
        if count:
            changes[name] = count
    feature_sets = {"baseline": features, "repairs": features, "coverage": [*features, *coverage.columns],
                    "public": [*features, *coverage.columns, *PUBLIC_FEATURES]}
    for name, frame in variants.items():
        columns = list(dict.fromkeys(feature_sets[name] + [c for c in META if c in frame]))
        frame = frame[columns].copy()
        frame[feature_sets[name]] = frame[feature_sets[name]].astype(np.float32)
        path = OUTPUT / f"{name}_inputs.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        projection_hashes[name] = sha(path)
    params = {"classifier": {**config.CLASSIFIER_PARAMS, "n_estimators": 500, "learning_rate": .025}}
    manifest = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "features": feature_sets,
        "test_start": str(TEST_START.date()), "test_end": str(TEST_END.date()),
        "training_start": str(low.date()), "training_cutoff": str(cutoff.date()),
        "dataset_start": reference["dataset_start"], "params": params,
        "win_only": True, "seed": 42, "threads": 2, "market_teacher": True, "market_teacher_blend": .5,
        "calibration": config.WIN_CALIBRATION_METHOD, "purge_days": purge_days(),
        "runtime_versions": runtime_versions(),
        "projection_sha256": projection_hashes, "source_sha256": sources,
        "market_teacher_sha256": sha(DEFAULT_BSP_PATH),
        "candidate_meta_sha256": sha(STORE / "meta.json"), "invariance_sha256": sha(OUTPUT / "input_invariance.json"),
        "rebuilt_input_review_sha256": sha(ROOT / "data/audit/feature_review_018/repaired_store_audit.json"),
        "external_projection_sha256": sha(external_path),
        "external_receipt_sha256": sha(external_root / "combined_external_receipt_v2.json"),
        "full_reference_predictions_sha256": sha(REFERENCE / "paired_predictions.parquet"),
        "changes_to_existing_features": changes,
        "comparisons": {"repairs": "repairs versus baseline: objective corrections only",
                        "coverage": "coverage versus repairs: separate missing-experience/recency design",
                        "public": "public versus coverage: separately test the actual dated external performances"},
        "selection_rule": "Shortlist only if paired race log loss and Brier improve; confidence intervals describe screening uncertainty. No automatic deployment or additional quarters.",
        "scope": "Retrospective development on an already used quarter; cheap settings are not a production estimate"}
    manifest["baseline_cache_key"] = canonical_hash({
        "inputs": projection_hashes["baseline"], "features": features, "params": params,
        "statistical_sources": {"live/"+name: sources["live/"+name] for name in STAT_SOURCES},
        "training_start": manifest["training_start"], "cutoff": manifest["training_cutoff"],
        "calibration": manifest["calibration"], "teacher": manifest["market_teacher_sha256"],
        "seed": 42, "threads": 2, "win_only": True,
        "runtime_versions": manifest["runtime_versions"],
    })
    write(OUTPUT / "manifest.json", manifest)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for identifier in sources:
            prefix, name = identifier.split("/", 1)
            archive.write((ROOT if prefix == "live" else BUNDLE) / name, identifier)
    print(json.dumps({"stage": "registered", "features": {k: len(v) for k, v in feature_sets.items()},
        "manifest_sha256": canonical_hash(manifest), "training_rows": int(baseline.race_date.le(cutoff).sum()),
        "test_races": int(baseline.loc[baseline.race_date.ge(TEST_START), "race_id"].nunique())}), flush=True)


def verify(manifest):
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("Statistical runtime versions changed")
    for identifier, expected in manifest["source_sha256"].items():
        prefix, name = identifier.split("/", 1)
        if sha((ROOT if prefix == "live" else BUNDLE) / name) != expected:
            raise ValueError(f"Screen source changed: {identifier}")
    if sha(DEFAULT_BSP_PATH) != manifest["market_teacher_sha256"]:
        raise ValueError("Market-teacher inputs changed")
    if sha(OUTPUT / "input_invariance.json") != manifest["invariance_sha256"]:
        raise ValueError("Feature audit changed")
    if sha(ROOT / "data/audit/feature_review_018/repaired_store_audit.json") != manifest["rebuilt_input_review_sha256"]:
        raise ValueError("Rebuilt input review changed")
    if sha(ROOT / "data/audit/feature_review_018/combined_external_receipt_v2.json") != manifest["external_receipt_sha256"]:
        raise ValueError("External source receipt changed")


def fit_one(variant):
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    verify(manifest)
    path = OUTPUT / f"{variant}_inputs.parquet"
    if sha(path) != manifest["projection_sha256"][variant]:
        raise ValueError("Registered projection changed")
    frame = pd.read_parquet(path)
    names = manifest["features"][variant]
    config.FEATURE_ALLOWLIST = names
    config.WIN_SOFT_LABEL_DISTILL, config.WIN_SOFT_LABEL_BLEND = True, .5
    if config.WIN_CALIBRATION_METHOD != manifest["calibration"] or purge_days() != manifest["purge_days"]:
        raise ValueError("Calibration or purge settings changed")
    predictor = RacePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 2
    cutoff = pd.Timestamp(manifest["training_cutoff"])
    test = frame.loc[frame.race_date.ge(manifest["test_start"])].copy()
    model_path = OUTPUT / f"{variant}.joblib"
    cache = ROOT / "data/audit/feature_screen_cache" / manifest["baseline_cache_key"]
    cached = False
    if (variant == "baseline" and not model_path.exists() and (cache / "model.joblib").exists()
            and (cache / "receipt.json").exists()):
        receipt = json.loads((cache / "receipt.json").read_text())
        if receipt["cache_key"] != manifest["baseline_cache_key"] or sha(cache / "model.joblib") != receipt["model_sha256"]:
            raise ValueError("Matched baseline cache changed")
        shutil.copy2(cache / "model.joblib", model_path)
        cached = True
    stamp = canonical_hash(manifest)
    begin = time.perf_counter()
    if model_path.exists():
        predictor.load(str(model_path))
        if (predictor.eval_split_info.get("screen_manifest_sha256") != stamp
                and not (variant == "baseline" and predictor.eval_split_info.get("screen_baseline_cache_key") == manifest["baseline_cache_key"])):
            raise ValueError("Cached model belongs to a different screen")
    else:
        train = frame.loc[frame.race_date.le(cutoff)].copy()
        if get_feature_columns(train) != names:
            raise ValueError("Feature order differs from registration")
        print(json.dumps({"stage": "fitting", "variant": variant, "training_rows": len(train), "features": len(names)}), flush=True)
        predictor.fit_as_of(train, cutoff, params=manifest["params"],
            dataset_start=pd.Timestamp(manifest["dataset_start"]), win_only=True)
        predictor.eval_split_info.update(screen_manifest_sha256=stamp, variant=variant)
        if variant == "baseline":
            predictor.eval_split_info["screen_baseline_cache_key"] = manifest["baseline_cache_key"]
        verify(manifest)
        temporary = model_path.with_suffix(".incomplete.joblib")
        predictor.save(str(temporary)); temporary.replace(model_path)
        if variant == "baseline":
            cache.mkdir(parents=True, exist_ok=True)
            cache_temp = cache / "model.incomplete.joblib"
            shutil.copy2(model_path, cache_temp); cache_temp.replace(cache / "model.joblib")
            write(cache / "receipt.json", {"cache_key": manifest["baseline_cache_key"], "model_sha256": sha(cache / "model.joblib"),
                "first_manifest_sha256": stamp, "scope": "Exact matched win-only screen budget and baseline projection"})
        del train
    probabilities = align_predictions(predictor, test)
    reloaded = RacePredictor(); reloaded.load(str(model_path))
    replay = align_predictions(reloaded, test)
    if not np.array_equal(probabilities, replay):
        raise ValueError("Saved win model does not reproduce its predictions")
    scored = test[[c for c in META if c in test]].copy()
    scored["probability"] = probabilities
    scored.to_parquet(OUTPUT / f"{variant}_predictions.parquet", index=False)
    write(OUTPUT / f"{variant}_fit.json", {"manifest_sha256": stamp, "model_sha256": sha(model_path),
        "prediction_sha256": sha(OUTPUT / f"{variant}_predictions.parquet"),
        "elapsed_seconds": time.perf_counter()-begin, "saved_model_replay_max_error": 0.,
        "matched_baseline_reused": cached,
        "fit": predictor.eval_split_info})
    print(json.dumps({"stage": "fit_complete", "variant": variant, "elapsed_seconds": time.perf_counter()-begin}), flush=True)


def summarise():
    manifest = json.loads((OUTPUT / "manifest.json").read_text()); verify(manifest)
    comparisons = {}
    for base, candidate, label in [("baseline", "repairs", "repairs"), ("repairs", "coverage", "coverage"),
                                  ("coverage", "public", "public"), ("repairs", "public", "public_vs_repairs")]:
        a = pd.read_parquet(OUTPUT / f"{base}_predictions.parquet")
        b = pd.read_parquet(OUTPUT / f"{candidate}_predictions.parquet")
        keys = ["race_id", "horse_name", "race_date", "won"]
        if not a[keys].equals(b[keys]):
            raise ValueError("Comparison contains different fields or labels")
        pair = a.drop(columns="probability").assign(baseline_probability=a.probability.to_numpy(),
                                                    candidate_probability=b.probability.to_numpy())
        comparisons[label] = score_pair(pair)
    if sha(REFERENCE / "paired_predictions.parquet") != manifest["full_reference_predictions_sha256"]:
        raise ValueError("Full-budget reference predictions changed")
    full_reference = pd.read_parquet(REFERENCE / "paired_predictions.parquet")
    keys = ["race_id", "horse_name", "race_date", "won"]
    full_reference = full_reference[keys + ["candidate_probability"]].rename(columns={"candidate_probability": "baseline_probability"})
    budget_context = {}
    for name in manifest["features"]:
        current = pd.read_parquet(OUTPUT / f"{name}_predictions.parquet").rename(columns={"probability": "candidate_probability"})
        aligned = current.merge(full_reference, on=keys, how="left", sort=False, validate="one_to_one")
        if aligned.baseline_probability.isna().any():
            raise ValueError("Full-budget reference is missing a test runner")
        budget_context[name] = score_pair(aligned)
    report = {"manifest_sha256": canonical_hash(manifest), "comparisons": comparisons,
        "full_budget_reference_context_not_feature_effect": budget_context,
        "scope": manifest["scope"], "production_changed": False, "remaining_quarters_queued": False}
    write(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--variant", choices=["baseline", "repairs", "coverage", "public"])
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.summarise:
        summarise()
    elif args.variant:
        fit_one(args.variant)
    else:
        for name in ["baseline", "repairs", "coverage", "public"]:
            fit_one(name)
            gc.collect()
        summarise()


if __name__ == "__main__":
    main()
