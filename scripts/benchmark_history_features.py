"""012: one existing history/uncertainty family, native probability training.

Reuse the sealed009 baseline models after verifying source, dataset, recipe,
fields and exact native predictions. Train only the three new165-input models.
All three declared quarters must complete before an advancement decision.
"""
from __future__ import annotations

import gc
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
import pyarrow.parquet as pq
import config

from scripts.audit_history_invariance import ADDED
from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import META, advancement, canonical_hash, score_pair
from scripts.benchmark_tail_calibration import publish
from scripts.replay_market_snapshots import digest
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.wf_eval import development_mask, expanding_period_folds, purge_days

PREVIOUS = ROOT / "data/audit/model_improvement/009_pace_family"
OUTPUT = ROOT / "data/audit/model_improvement/012_history_family_compact"
REGISTRATION = ROOT / "data/audit/model_improvement/012_history_family_registration_compact.json"
AUDIT = ROOT / "data/audit/model_improvement/012_history_invariance_compact.json"
FEATURES = ROOT / "data/processed/feature_store/featured.parquet"
SOURCES = ["scripts/benchmark_history_features.py", "scripts/audit_history_invariance.py",
           "tests/test_history_feature_benchmark.py", "scripts/benchmark_independent_features.py",
           "scripts/benchmark_forecasts.py", "scripts/benchmark_tail_calibration.py",
           "scripts/assess_forecast_parity.py", "scripts/replay_market_snapshots.py",
           "src/model.py", "src/forecast_evaluation.py", "src/wf_eval.py",
           "src/probability_calibration.py", "config.py"]


def verify_reference(manifest, source_hashes, feature_hash, params):
    if (manifest["source_file_sha256"] != feature_hash or manifest["params"] != params
            or manifest["seed"] != 42 or manifest["threads"] != 2
            or manifest["feature_dtype"] != "float32 for both variants"
            or manifest["recency_half_life_days"] != config.RECENCY_HALF_LIFE_DAYS
            or manifest["calibration"] != config.WIN_CALIBRATION_METHOD
            or manifest["distillation"] != bool(config.WIN_SOFT_LABEL_DISTILL)):
        raise ValueError("Reference dataset, dtype or training recipe differs")
    for source, expected in manifest["code_sha256"].items():
        if source in source_hashes and source_hashes[source] != expected:
            raise ValueError(f"Reference native/scoring code changed: {source}")


def align_reference(test, frozen, native):
    """Require the same whole fields, labels and forecasts; never silently drop."""
    keys = ["race_id", "horse_name"]
    subset = frozen.loc[pd.to_datetime(frozen.race_date).between(test.race_date.min(), test.race_date.max())]
    if len(test) != len(subset):
        raise ValueError("Reference has a different runner universe")
    aligned = test[keys + ["race_date", "won"]].merge(
        subset[keys + ["race_date", "won", "baseline_probability"]],
        on=keys, how="left", sort=False, validate="one_to_one", suffixes=("", "_reference"))
    if (aligned.baseline_probability.isna().any()
            or not pd.to_datetime(aligned.race_date).equals(pd.to_datetime(aligned.race_date_reference))
            or not np.array_equal(aligned.won, aligned.won_reference)):
        raise ValueError("Reference identities, dates or labels differ")
    error = float(np.max(np.abs(aligned.baseline_probability.to_numpy() - native)))
    if not np.isfinite(native).all() or error > 1e-12:
        raise ValueError(f"Saved baseline native replay changed: {error}")
    return aligned.baseline_probability.to_numpy(float), error


def qualify(pooled, folds):
    return bool(advancement(pooled, folds)
                and pooled["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"][1] < 0)


def output_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing artifact differs: {path}")
    else:
        publish(path, value)


def run():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "completion.json").exists():
        raise ValueError("Experiment is already sealed; do not overwrite it")
    registration, audit = (json.loads(p.read_text()) for p in (REGISTRATION, AUDIT))
    if (not audit["passed"] or audit["registration_content_sha256"] != canonical_hash(registration)
            or set(audit["tested_columns"]) != set(ADDED)
            or set(registration["candidate_features"]) - set(registration["baseline_features"]) != set(ADDED)):
        raise ValueError("Exact registered family must pass temporal/live-card invariance")
    for source, expected in audit["source_sha256"].items():
        if digest(ROOT / source) != expected:
            raise ValueError(f"Feature audit source changed: {source}")
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    meta = json.loads(FEATURES.with_name("meta.json").read_text())
    if (meta["fe_signature"] != fe_signature() or meta["fe_signature"] != registration["fe_signature"]
            or meta["data_signature"] != registration["data_signature"]
            or registration["baseline_features"] != previous["variants"]["baseline"]):
        raise ValueError("Registered feature/data/reference generation changed")
    source_hashes = {p: digest(ROOT / p) for p in set(SOURCES) | set(audit["source_sha256"])}
    feature_hash = digest(FEATURES)
    params = {"classifier": {**config.CLASSIFIER_PARAMS, "n_estimators": 2000}}
    verify_reference(previous, source_hashes, feature_hash, params)
    periods = registration["candidate_periods"]
    if periods != ["2025Q3", "2025Q4", "2026Q1"]:
        raise ValueError("Complete the three registered quarters")
    paths = [REGISTRATION, AUDIT, PREVIOUS / "manifest.json", PREVIOUS / "completion.json",
             PREVIOUS / "paired_predictions.parquet", PREVIOUS / "market_context_predictions.parquet",
             PREVIOUS / "market_context_report.json", FEATURES.with_name("meta.json")]
    paths += [PREVIOUS / f"{p}_baseline{ext}" for p in periods for ext in (".joblib", "_fit.json")]
    amendment = ROOT / "data/audit/model_improvement"
    paths += [amendment / n for n in ("012_history_profile.json", "012_history_family_registration.json",
              "012_history_invariance.json", "012_history_invariance_cards.json",
              "012_history_invariance_initial_nan_cards.py", "012_history_family/aborted_before_scoring.json",
              "012_history_family/manifest.json", "012_history_family/source_snapshot.zip")]
    preserved = [ROOT / "data/runs/20261026_140950_205296/model.joblib",
                 ROOT / "data/runs/20261026_140950_205296/meta.json", ROOT / "models/production.json"]
    study = ROOT / "data/forecasts/forward/20261026_baseline_v3"
    preserved += [study / n for n in ("protocol.json", "protocol.sha256.json", "source_snapshot.zip", "model.joblib", "model_meta.json")]
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    preservation = {str(p.relative_to(ROOT)): digest(p) for p in preserved}
    features = registration["candidate_features"]
    schema = pq.read_schema(FEATURES).names
    columns = [c for c in schema if c in set(features)] + [c for c in META if c in schema and c not in features]
    projection = OUTPUT / "inputs.parquet"
    if projection.exists():
        cached = json.loads((OUTPUT / "inputs.json").read_text())
        if (cached["source_sha256"] != feature_hash or cached["projection_sha256"] != digest(projection)
                or cached["columns"] != columns):
            raise ValueError("Cached projection changed")
        frame = pd.read_parquet(projection)
    else:
        frame = pd.read_parquet(FEATURES, columns=columns,
                                filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
        frame[features] = frame[features].astype(np.float32)
        frame.to_parquet(projection, index=False, compression="zstd")
        publish(OUTPUT / "inputs.json", {"source_sha256": feature_hash, "projection_sha256": digest(projection),
                                         "columns": columns, "dtype": "float32 features"})
    frame["race_date"] = pd.to_datetime(frame.race_date).dt.normalize()
    dataset_start = frame.race_date.min()
    valid = valid_race_mask(frame)
    excluded = int((~valid).sum())
    frame = frame.loc[valid & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    folds = {f.label: f for f in expanding_period_folds(frame.race_date)}
    cutoffs = [{"period": p, "cutoff": str(frame.loc[folds[p].train, "race_date"].max())[:10],
                "test_start": str(frame.loc[folds[p].test, "race_date"].min())[:10],
                "test_end": str(frame.loc[folds[p].test, "race_date"].max())[:10]} for p in periods]
    if cutoffs != previous["folds"]:
        raise ValueError("Reference and candidate folds differ")
    unsafe = set(registration["unsafe_composites_requiring_exclusion_or_rebuild"])
    if unsafe & set(features):
        raise ValueError("Quarantined features cannot enter the candidate")
    manifest = {"experiment": registration["experiment"], "registration": registration,
                "audit": audit, "source_sha256": source_hashes, "input_sha256": input_hashes,
                "source_feature_sha256": feature_hash, "projection_sha256": digest(projection),
                "preserved_sha256": preservation, "source_meta": meta, "params": params,
                "candidate_features": features, "folds": cutoffs, "seed": 42, "threads": 2,
                "invalid_rows_excluded": excluded, "purge_days": purge_days(),
                "baseline": "Reuse exact saved009 baseline149 native artifacts; verify every later probability within1e-12",
                "scope": "Previously inspected retrospective development, full valid fields; no fresh confirmation or automatic promotion",
                "market": "Reuse009's frozen identical complete T5 fields; preserve exclusions and outcomes",
                "selection": "Single fixed16-column addition after outcome-free deduplication, all3quarters required, no subsets or parameter sweep"}
    output_json(OUTPUT / "manifest.json", manifest)
    manifest_sha = digest(OUTPUT / "manifest.json")
    if not (OUTPUT / "source_snapshot.zip").exists():
        with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in source_hashes:
                archive.write(ROOT / source, source)
    print(json.dumps({"stage": "registered", "manifest_sha256": manifest_sha,
                      "features": len(features), "candidate_fits": 3, "baseline_refits": 0}), flush=True)
    old = pd.read_parquet(PREVIOUS / "paired_predictions.parquet")
    reports, predictions, replays = [], [], {}
    for period in periods:
        fold = folds[period]
        test = frame.loc[fold.test].copy()
        baseline = RacePredictor()
        baseline.load(str(PREVIOUS / f"{period}_baseline.joblib"))
        if baseline.feature_cols != registration["baseline_features"] or baseline.eval_split_info["as_of"] != cutoffs[periods.index(period)]["cutoff"]:
            raise ValueError("Wrong historical reference model")
        native = align_predictions(baseline, test)
        base_p, error = align_reference(test, old, native)
        replays[period] = error
        del baseline
        gc.collect()
        cutoff = frame.loc[fold.train, "race_date"].max()
        predictor = RacePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
        predictor._autotune_njobs = 2
        model_path = OUTPUT / f"{period}_candidate.joblib"
        if model_path.exists():
            predictor.load(str(model_path))
            if predictor.eval_split_info.get("experiment_manifest_sha256") != manifest_sha:
                raise ValueError("Cached candidate belongs to another manifest")
        else:
            print(json.dumps({"stage": "fitting", "period": period, "cutoff": str(cutoff.date()),
                              "features": len(features), "training_runners": int(fold.train.sum())}), flush=True)
            previous_allowlist = config.FEATURE_ALLOWLIST
            try:
                config.FEATURE_ALLOWLIST = features
                train = frame.loc[fold.train, columns].copy()
                if get_feature_columns(train) != features:
                    raise ValueError("Requested feature order/eligibility differs from native fitting")
                predictor.fit_as_of(train, cutoff, params=params, dataset_start=dataset_start, win_only=True)
                del train
            finally:
                config.FEATURE_ALLOWLIST = previous_allowlist
            if predictor.feature_cols != features:
                raise ValueError("Native fitting changed the registered family")
            predictor.eval_split_info.update(experiment_manifest_sha256=manifest_sha, variant="candidate", period=period)
            predictor.save(str(model_path))
        output_json(OUTPUT / f"{period}_candidate_fit.json", predictor.eval_split_info)
        scored = test[[c for c in META if c in test] + ADDED].copy()
        scored["baseline_probability"] = base_p
        scored["candidate_probability"] = align_predictions(predictor, test)
        scored.to_parquet(OUTPUT / f"{period}_predictions.parquet", index=False)
        report = {"period": period, **score_pair(scored)}
        reports.append(report)
        predictions.append(scored)
        print(json.dumps({"stage": "fold_complete", **report}), flush=True)
        (OUTPUT / "progress.json").write_text(json.dumps({"completed_folds": reports, "selection_permitted": False}, indent=2))
        del predictor, test
        gc.collect()
    paired = pd.concat(predictions, ignore_index=True).sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    pooled = score_pair(paired)
    market = pd.read_parquet(PREVIOUS / "market_context_predictions.parquet").drop(columns="candidate_probability")
    market = market.merge(paired[["race_id", "horse_name", "candidate_probability"]],
                          on=["race_id", "horse_name"], how="left", validate="one_to_one")
    if market.candidate_probability.isna().any():
        raise ValueError("Frozen market overlap no longer matches whole test fields")
    market_report = score_forecasts(market, {n: market[f"{n}_probability"].to_numpy(float) for n in ("baseline", "candidate", "market")})
    coverage = json.loads((PREVIOUS / "market_context_report.json").read_text())["coverage"]
    diagnostics = {}
    for label, ids in {
        "contains_DB_debutant": paired.groupby("race_id", sort=False).horse_prev_races.min().eq(0),
        "all_recorded_prior_starts": paired.groupby("race_id", sort=False).horse_prev_races.min().gt(0),
        "contains_external_form": paired.groupby("race_id", sort=False).has_hidden_form.max().eq(1),
    }.items():
        subset = paired.loc[paired.race_id.isin(ids.index[ids])]
        if subset.race_id.nunique() >= 30:
            diagnostics[label] = score_pair(subset)
    paired.to_parquet(OUTPUT / "paired_predictions.parquet", index=False)
    market.to_parquet(OUTPUT / "market_context_predictions.parquet", index=False)
    report = {"manifest_sha256": manifest_sha, "folds": reports, "pooled": pooled,
              "candidate_passes_numeric_advancement": qualify(pooled, reports), "production_changed": False,
              "baseline_native_replay_max_error": replays, "market": {"coverage": coverage, "comparison": market_report},
              "diagnostic_subgroups_not_selection": diagnostics, "scope": manifest["scope"],
              "elapsed_seconds": time.monotonic() - started}
    for path, expected in {**source_hashes, **input_hashes, **preservation}.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen code/input changed: {path}")
    if digest(FEATURES) != feature_hash:
        raise RuntimeError("Canonical features changed during fitting")
    publish(OUTPUT / "report.json", report)
    publish(OUTPUT / "completion.json", {"manifest_sha256": manifest_sha, "evidence_sha256": {
        str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.iterdir() if p.is_file()}, "production_changed": False})
    print(json.dumps({"stage": "complete", "passes_numeric_advancement": report["candidate_passes_numeric_advancement"],
                      "pooled": pooled, "market": market_report["metrics"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
