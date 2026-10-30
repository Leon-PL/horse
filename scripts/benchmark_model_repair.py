"""Configurable, resumable native comparisons for corrected prediction inputs.

The registration defines inputs and targets; each worker owns its settings.
Frozen references always replay on their original data. Model and prediction
files publish atomically so a shutdown cannot expose a half-written cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gc
import json
import logging
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import config

from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_history_features import align_reference
from scripts.benchmark_independent_features import META, advancement, canonical_hash, score_pair
from scripts.replay_market_snapshots import digest
from src.featured_dataset import _FE_SOURCE_FILES, fe_signature
from src.forecast_evaluation import valid_race_mask
from src.model import RacePredictor, get_feature_columns
from src.wf_eval import development_mask, expanding_period_folds
from src.deployment_validation import RECIPE_SOURCES, training_recipe
from src.training import TrainingConfig
from src.betfair import DEFAULT_BSP_PATH

SOURCES = list(dict.fromkeys([*_FE_SOURCE_FILES, *RECIPE_SOURCES, "src/model.py", "src/betfair.py", "config.py",
    "src/forecast_evaluation.py", "src/probability_calibration.py", "src/wf_eval.py", "src/each_way.py",
    "src/live_prediction.py", "src/database.py", "scripts/benchmark_model_repair.py",
    "scripts/benchmark_forecasts.py", "scripts/benchmark_history_features.py",
    "scripts/benchmark_independent_features.py", "scripts/assess_forecast_parity.py",
    "scripts/replay_market_snapshots.py", "scripts/audit_prediction_inputs.py",
    "tests/test_result_status.py", "tests/test_rating_reuse.py", "tests/test_native_head_independence.py"]))


def write(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True)
    if path.name == "manifest.json" and path.exists() and path.read_text() != encoded:
        raise ValueError("Do not overwrite an experiment manifest")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def source_hashes():
    return {p: digest(ROOT / p) for p in SOURCES}


def specification(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    output = path.parent
    reference = ROOT / spec.get("reference_directory", "data/audit/model_improvement/009_pace_family")
    return spec, output, reference


def configured_recipe(spec, params):
    """Record the effective native recipe, including experiment overrides."""
    names = ("FEATURE_ALLOWLIST", "WIN_SOFT_LABEL_DISTILL", "WIN_SOFT_LABEL_BLEND")
    previous = {name: getattr(config, name) for name in names}
    try:
        config.FEATURE_ALLOWLIST = spec["candidate_features"]
        config.WIN_SOFT_LABEL_DISTILL = bool(spec.get("market_teacher", False))
        config.WIN_SOFT_LABEL_BLEND = float(spec.get("market_teacher_blend", .5))
        cfg = TrainingConfig(refresh_data=False, frameworks={"classifier": "lgbm", "place": "lgbm"},
                             seed=42, hyperparams=params)
        return training_recipe(cfg), cfg.to_dict()
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def verify_teacher_inputs(manifest):
    for relative, expected in manifest.get("market_teacher_inputs_sha256", {}).items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Frozen market-teacher prices changed")


def prepare(path):
    spec, output, reference = specification(path)
    if (output / "manifest.json").exists():
        raise ValueError("Already registered; omit --prepare to resume")
    reference_variant = spec.get("reference_variant", "baseline")
    old_manifest = json.loads((reference / "manifest.json").read_text())
    old_input_hash = old_manifest.get("projection_sha256", old_manifest.get("inputs_sha256"))
    if digest(reference / "inputs.parquet") != old_input_hash:
        raise ValueError("Frozen reference inputs changed")
    baseline = spec["baseline_features"]
    candidate = spec["candidate_features"]
    if len(candidate) != len(set(candidate)) or len(baseline) != len(set(baseline)):
        raise ValueError("Feature names must be unique")
    audit = json.loads((output / "input_invariance.json").read_text())
    if (not audit["passed"] or set(audit["selected_inputs"]) != set(candidate)
            or audit["fe_signature"] != spec["fe_signature"]):
        raise ValueError("The complete candidate must pass its registered input audit")
    store = ROOT / spec["feature_store_directory"] / "featured.parquet" if spec.get("feature_store_directory") else output / "candidate_store/featured.parquet"
    meta = json.loads(store.with_name("meta.json").read_text())
    if meta["fe_signature"] != spec["fe_signature"] or fe_signature() != spec["fe_signature"]:
        raise ValueError("Candidate code and stored generation differ")
    end = max(pd.Period(p, freq="Q").end_time.normalize() for p in spec["candidate_periods"])
    schema = pq.read_schema(store).names
    extra = [c for c in ("beaten_lengths", "horse_prev_races", "has_hidden_form", "finish_pos_label") if c in schema]
    columns = list(dict.fromkeys(candidate + META + extra))
    frame = pd.read_parquet(store, columns=columns, filters=[("race_date", "<=", end)])
    extra_path = spec.get("extra_features_file")
    if extra_path:
        raise ValueError("Integrate and audit extra feature generation before registering its candidate store")
    old_schema = pq.read_schema(reference / "inputs.parquet").names
    original = pd.read_parquet(reference / "inputs.parquet", columns=list(dict.fromkeys(baseline + [c for c in META if c in old_schema])))
    keys = ["race_date", "race_id", "horse_name"]
    frame = frame.sort_values(keys, kind="stable").reset_index(drop=True)
    original = original.sort_values(keys, kind="stable").reset_index(drop=True)
    if not frame[keys].equals(original[keys]):
        raise ValueError("Reference and corrected inputs contain different runners")
    for name in ("won", "finish_position", "num_runners", "odds", "lengths_behind"):
        if not np.array_equal(frame[name], original[name], equal_nan=True):
            raise ValueError(f"Original outcomes or raw data changed: {name}")
    profile = {}
    for name in baseline:
        a, b = original[name].to_numpy(np.float32), frame[name].to_numpy(np.float32)
        changed = ~np.isclose(a, b, atol=0, rtol=0, equal_nan=True)
        if changed.any():
            finite = np.isfinite(a) & np.isfinite(b)
            profile[name] = {"changed_rows": int(changed.sum()),
                "max_finite_difference": float(np.abs(a[finite] - b[finite]).max()) if finite.any() else None}
    write(output / "feature_change_profile.json", {"changes_to_existing_inputs": profile,
        "new_inputs": {c: {"known_rows": int(frame[c].notna().sum()), "distinct_values": int(frame[c].nunique())}
                       for c in candidate if c not in baseline}})
    del original
    frame[candidate] = frame[candidate].astype(np.float32)
    projection = output / "inputs.parquet"
    temporary = projection.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(projection)
    eligible = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].reset_index(drop=True)
    folds = {f.label: f for f in expanding_period_folds(eligible.race_date)}
    cutoffs = [{"period": p, "cutoff": str(eligible.loc[folds[p].train, "race_date"].max())[:10],
                "test_start": str(eligible.loc[folds[p].test, "race_date"].min())[:10],
                "test_end": str(eligible.loc[folds[p].test, "race_date"].max())[:10]} for p in spec["candidate_periods"]]
    if cutoffs != old_manifest["folds"]:
        raise ValueError("Keep the reference's chronological splits")
    params = {"classifier": dict(config.CLASSIFIER_PARAMS), "place": dict(config.PLACE_CLASSIFIER_PARAMS)}
    if params["classifier"] != old_manifest["params"]["classifier"]:
        raise ValueError("Win hyperparameters differ from the fixed reference")
    recipe, native_config = configured_recipe(spec, params)
    teacher_inputs = {}
    if spec.get("market_teacher", False):
        if not (ROOT / DEFAULT_BSP_PATH).is_file():
            raise ValueError("Register the expected BSP cache before a market-teacher experiment")
        teacher_inputs[DEFAULT_BSP_PATH.replace("\\", "/")] = digest(ROOT / DEFAULT_BSP_PATH)
    secondary = {name: {"directory": directory,
        "predictions_sha256": digest(ROOT / directory / "paired_predictions.parquet"),
        "completion_sha256": digest(ROOT / directory / "completion.json")}
        for name, directory in spec.get("secondary_references", {}).items()}
    manifest = {"registration": spec, "registration_sha256": digest(path), "source_sha256": source_hashes(),
        "params": params, "folds": cutoffs, "features": candidate, "baseline_features": baseline,
        "inputs_sha256": digest(projection), "candidate_fe_signature": spec["fe_signature"],
        "dataset_start": str(frame.race_date.min())[:10], "invariance_sha256": digest(output / "input_invariance.json"),
        "reference_manifest_sha256": digest(reference / "manifest.json"),
        "reference_predictions_sha256": digest(reference / "paired_predictions.parquet"),
        "reference_variant": reference_variant,
        "reference_models_sha256": {p: digest(reference / f"{p}_{reference_variant}.joblib") for p in spec["candidate_periods"]},
        "calibration": config.WIN_CALIBRATION_METHOD, "recency_half_life_days": config.RECENCY_HALF_LIFE_DAYS,
        "seed": 42, "learner_threads": 2, "place_validation_periods": spec.get("place_validation_periods", []),
        "native_training_recipe": recipe, "native_training_config": native_config,
        "market_teacher_inputs_sha256": teacher_inputs, "secondary_references": secondary,
        "feature_store_meta_sha256": digest(store.with_name("meta.json")),
        "scope": spec["scope"]}
    write(output / "manifest.json", manifest)
    if not (output / "source_snapshot.zip").exists():
        with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
            for relative in SOURCES:
                z.write(ROOT / relative, relative)
    print(json.dumps({"stage": "registered", "features": len(candidate), "manifest_sha256": canonical_hash(manifest)}), flush=True)
    del frame, eligible
    gc.collect()
    return manifest


def fit_fold(period, *, registration):
    logging.basicConfig(level=logging.INFO)
    spec, output, reference = specification(registration)
    manifest = json.loads((output / "manifest.json").read_text())
    stamp = canonical_hash(manifest)
    if source_hashes() != manifest["source_sha256"] or digest(registration) != manifest["registration_sha256"]:
        raise ValueError("Registered code or recipe changed")
    if digest(output / "inputs.parquet") != manifest["inputs_sha256"]:
        raise ValueError("Registered input projection changed")
    verify_teacher_inputs(manifest)
    config.FEATURE_ALLOWLIST = manifest["features"]
    config.WIN_SOFT_LABEL_DISTILL = bool(spec.get("market_teacher", False))
    config.WIN_SOFT_LABEL_BLEND = float(spec.get("market_teacher_blend", .5))
    frame = pd.read_parquet(output / "inputs.parquet")
    frame = frame.loc[valid_race_mask(frame) & development_mask(frame.race_date)].sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    fold = {f.label: f for f in expanding_period_folds(frame.race_date)}[period]
    test = frame.loc[fold.test].copy()
    old = pd.read_parquet(reference / "inputs.parquet", columns=list(dict.fromkeys(manifest["baseline_features"] + META)))
    old = old.loc[old.race_id.isin(test.race_id)].sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    if not old[["race_id", "horse_name"]].reset_index(drop=True).equals(test[["race_id", "horse_name"]].reset_index(drop=True)):
        raise ValueError("Baseline replay has different fields")
    baseline = RacePredictor()
    reference_variant = manifest["reference_variant"]
    if digest(reference / f"{period}_{reference_variant}.joblib") != manifest["reference_models_sha256"][period]:
        raise ValueError("Reference model changed")
    baseline.load(str(reference / f"{period}_{reference_variant}.joblib"))
    if baseline.feature_cols != manifest["baseline_features"]:
        raise ValueError("Reference model uses another feature order")
    frozen = pd.read_parquet(reference / "paired_predictions.parquet")
    if reference_variant != "baseline":
        frozen = frozen[["race_id", "horse_name", "race_date", "won", f"{reference_variant}_probability"]].rename(
            columns={f"{reference_variant}_probability": "baseline_probability"})
    base_p, replay_error = align_reference(test, frozen, align_predictions(baseline, old))
    del baseline, frozen, old
    gc.collect()
    predictor = RacePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 2
    path = output / f"{period}_candidate.joblib"
    full = period in manifest["place_validation_periods"]
    if path.exists():
        predictor.load(str(path))
        if predictor.eval_split_info.get("experiment_manifest_sha256") != stamp:
            raise ValueError("Cached model belongs to another experiment")
    else:
        train = frame.loc[fold.train].copy()
        if get_feature_columns(train) != manifest["features"]:
            raise ValueError("Native feature order differs from the registration")
        cutoff = train.race_date.max()
        print(json.dumps({"stage": "fitting", "period": period, "features": len(predictor.feature_cols or manifest["features"]),
                          "rows": len(train), "fit_place": full, "cutoff": str(cutoff)[:10]}), flush=True)
        predictor.fit_as_of(train, cutoff, params=manifest["params"], dataset_start=pd.Timestamp(manifest["dataset_start"]), win_only=not full)
        predictor.eval_split_info.update(experiment_manifest_sha256=stamp, period=period, variant="candidate")
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Sources changed during fitting")
        verify_teacher_inputs(manifest)
        temporary = path.with_name(f"{period}_candidate.incomplete.joblib")
        predictor.save(str(temporary)); temporary.replace(path)
        del train
    scored = test[[c for c in META + ["horse_prev_races", "has_hidden_form"] if c in test]].copy()
    scored = scored.loc[:, ~scored.columns.duplicated()]
    scored["baseline_probability"] = base_p
    scored["candidate_probability"] = align_predictions(predictor, test)
    place_metrics = None
    if full:
        predicted = predictor.predict_races(test)
        aligned = test[["race_id", "horse_name"]].merge(predicted[["race_id", "horse_name", "place_probability"]],
            on=["race_id", "horse_name"], how="left", sort=False, validate="one_to_one")
        place = aligned.place_probability.to_numpy()
        _, matrix = predictor._prepare_prediction_frame(test)
        raw = predictor.place_model.predict_proba(matrix)[:, 1]
        place_metrics = predictor._evaluate_place_model(place, raw, test, test.groupby("race_id", sort=False).size().to_numpy())
        scored["place_probability"] = place
    temporary = output / f"{period}_predictions.parquet.tmp"
    scored.to_parquet(temporary, index=False); temporary.replace(output / f"{period}_predictions.parquet")
    report = {"period": period, "manifest_sha256": stamp, "baseline_native_replay_max_error": replay_error,
              "place_metrics": place_metrics, **score_pair(scored)}
    write(output / f"{period}_fit.json", predictor.eval_split_info)
    write(output / f"{period}_report.json", report)
    print(json.dumps({"stage": "fold_complete", **report}), flush=True)
    return report


def summarise(registration):
    spec, output, reference = specification(registration)
    manifest = json.loads((output / "manifest.json").read_text())
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Registered source changed")
    verify_teacher_inputs(manifest)
    periods = spec["candidate_periods"]
    reports = [json.loads((output / f"{p}_report.json").read_text()) for p in periods]
    if any(r["manifest_sha256"] != canonical_hash(manifest) for r in reports):
        raise ValueError("A fold does not match the registration")
    paired = pd.concat([pd.read_parquet(output / f"{p}_predictions.parquet") for p in periods]).sort_values(
        ["race_date", "race_id", "horse_name"], kind="stable")
    pooled = score_pair(paired)
    market = pd.read_parquet(reference / "market_context_predictions.parquet")
    reference_variant = manifest["reference_variant"]
    reference_probability = market[f"{reference_variant}_probability"].to_numpy(copy=True)
    market = market.drop(columns=["baseline_probability", "candidate_probability"])
    market["baseline_probability"] = reference_probability
    market = market.merge(paired[["race_id", "horse_name", "candidate_probability"]], on=["race_id", "horse_name"], how="left", validate="one_to_one")
    if market.candidate_probability.isna().any():
        raise ValueError("Market overlap lost a runner")
    comparison = score_forecasts(market, {n: market[f"{n}_probability"].to_numpy(float) for n in ("baseline", "candidate", "market")})
    metrics = comparison["metrics"]
    gap = metrics["baseline"]["race_log_loss"] - metrics["market"]["race_log_loss"]
    report = {"manifest_sha256": canonical_hash(manifest), "folds": reports, "pooled": pooled,
        "candidate_passes_numeric_advancement": advancement(pooled, reports), "market": comparison,
        "market_gap_closed_point_estimate": ((metrics["baseline"]["race_log_loss"] - metrics["candidate"]["race_log_loss"]) / gap if gap > 0 else None),
        "scope": manifest["scope"], "production_changed": False}
    report["diagnostic_race_types_not_selection"] = {str(kind): score_pair(group) for kind, group in paired.groupby("race_type", sort=False) if group.race_id.nunique() >= 30}
    report["secondary_comparisons"] = {}
    for name, evidence in manifest["secondary_references"].items():
        directory = ROOT / evidence["directory"]
        if (digest(directory / "paired_predictions.parquet") != evidence["predictions_sha256"]
                or digest(directory / "completion.json") != evidence["completion_sha256"]):
            raise ValueError("A frozen secondary reference changed")
        previous = pd.read_parquet(directory / "paired_predictions.parquet")
        aligned = paired.drop(columns="baseline_probability").merge(
            previous[["race_id", "horse_name", "race_date", "won", "candidate_probability"]].rename(
                columns={"candidate_probability": "baseline_probability"}),
            on=["race_id", "horse_name", "race_date", "won"], how="left", validate="one_to_one")
        if aligned.baseline_probability.isna().any():
            raise ValueError("Secondary reference lost an outcome or runner")
        report["secondary_comparisons"][name] = score_pair(aligned)
    winners = paired.loc[paired.won.eq(1)]
    report["extreme_probability_diagnostics_not_selection"] = {name: {
        "winner_probability_below_0_001": int(winners[f"{name}_probability"].lt(.001).sum()),
        "minimum_winner_probability": float(winners[f"{name}_probability"].min())} for name in ("baseline", "candidate")}
    paired.to_parquet(output / "paired_predictions.parquet", index=False)
    market.to_parquet(output / "market_context_predictions.parquet", index=False)
    write(output / "report.json", report)
    write(output / "completion.json", {"manifest_sha256": canonical_hash(manifest), "completed_periods": periods,
        "evidence_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "completion.json"}})
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--also-registration", action="append", default=[],
                        help="Run another frozen recipe in the same bounded worker pool")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--period")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    spec, output, _ = specification(args.registration)
    if args.prepare:
        prepare(args.registration)
    elif args.summarise:
        summarise(args.registration)
    elif args.period:
        if args.period not in spec["candidate_periods"]:
            raise ValueError("Period was not registered")
        fit_fold(args.period, registration=str(Path(args.registration).resolve()))
    else:
        registrations = [args.registration, *args.also_registration]
        specifications = []
        for registration in registrations:
            item, directory, _ = specification(registration)
            if not (directory / "manifest.json").exists():
                prepare(registration)
            specifications.append((registration, item))
        jobs = [(registration, period) for period in dict.fromkeys(
            period for _, item in specifications for period in item["candidate_periods"])
            for registration, item in specifications if period in item["candidate_periods"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_fold, period, registration=str(Path(registration).resolve()))
                       for registration, period in jobs]
            for future in futures:
                future.result()
        for registration in registrations:
            summarise(registration)


if __name__ == "__main__":
    main()
