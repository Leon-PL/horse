"""Fixed-tree chronological calibration experiment010. No production changes.

Fit on 2025Q3 OOS forecasts, validate on Q4, freeze the coefficient, then
score Q1 and April-May transfer with the saved December trees. All dates are
retrospective development data. The 149-feature comparison is primary; the
pace-model comparison is descriptive and cannot reverse experiment009.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import META, score_pair
from scripts.replay_market_snapshots import digest
from src.forecast_benchmark import match_market_fields
from src.forecast_evaluation import probability_metrics, valid_race_mask, validate_probabilities
from src.model import RacePredictor
from src.tail_calibration import blend, fit_weight, validate_weight
from src.wf_eval import development_mask, purge_days

PREVIOUS = ROOT / "data/audit/model_improvement/009_pace_family"
OUTPUT = ROOT / "data/audit/model_improvement/010_tail_calibration"
FEATURES = ROOT / "data/processed/feature_store/featured.parquet"
PERIODS = ("2025Q3", "2025Q4", "2026Q1")
VARIANTS = ("baseline", "candidate")
SOURCES = ["src/tail_calibration.py", "scripts/benchmark_tail_calibration.py",
           "src/model.py", "src/forecast_evaluation.py", "src/forecast_benchmark.py",
           "src/wf_eval.py", "src/probability_calibration.py", "config.py",
           "scripts/benchmark_forecasts.py", "scripts/benchmark_independent_features.py",
           "scripts/assess_forecast_parity.py", "scripts/replay_market_snapshots.py",
           "tests/test_tail_calibration.py"]
STAGES = {"fit": ("2025Q3", "2025-07-01", "2025-09-24"),
          "validation": ("2025Q4", "2025-10-01", "2025-12-22"),
          "test": ("2026Q1", "2026-01-01", "2026-03-31"),
          "transfer": ("2026Q1", "2026-04-01", "2026-05-31")}
MARKETS = [ROOT / "data/audit/market_performance/t5_snapshots_20250701_20260331.parquet",
           ROOT / "data/audit/market_performance/t5_snapshots_20260501_20261012.parquet"]


def publish(path, value):
    """Exclusive output: never silently overwrite a previous experiment."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sorted_fields(frame):
    return frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)


def groups_for(frame):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id, np.repeat(frame.race_id.drop_duplicates(), groups)):
        raise ValueError("Contiguous complete fields are required")
    return groups


def stage_frame(frame, stage, model_cutoff):
    _, start, end = STAGES[stage]
    dates = pd.to_datetime(frame.race_date).dt.normalize()
    result = sorted_fields(frame.loc[dates.between(start, end)].copy())
    if result.empty or not valid_race_mask(result).all() or not development_mask(result.race_date).all():
        raise ValueError("Expected nonempty valid whole fields inside the development boundary")
    if result.race_date.min() <= pd.Timestamp(model_cutoff) + pd.Timedelta(days=purge_days() - 1):
        raise ValueError("Model cutoff violates the chronological embargo")
    return result


def probability_stages(predictor, frame):
    """Reuse exact native tree/Platt/isotonic operations; never refit any map."""
    if predictor.market_anchor or predictor.win_calibration_method != "platt_isotonic":
        raise ValueError("This experiment requires an independent Platt-isotonic model")
    missing = set(predictor.feature_cols) - set(frame)
    if missing:
        raise ValueError(f"Missing model inputs: {sorted(missing)}")
    groups = groups_for(frame)
    prepared, x = predictor._prepare_prediction_frame(frame)
    raw = predictor.clf_model.predict_proba(x)[:, 1]
    existing = predictor._calibrate_win_probs(raw, groups, prepared)
    iso = predictor.win_iso_cal
    try:
        predictor.win_iso_cal = None
        platt = predictor._calibrate_win_probs(raw, groups, prepared)
    finally:
        predictor.win_iso_cal = iso
    for probabilities in (existing, platt):
        validate_probabilities(probabilities, frame.won, groups)
    # Every legacy probability is independently replayed through the public API.
    replay = align_predictions(predictor, frame)
    error = float(np.max(np.abs(replay - existing)))
    if error > 1e-12:
        raise ValueError(f"Native public inference replay differs: {error}")
    return existing, platt, error


def describe(frame, weight):
    scored = frame.copy()
    scored["baseline_probability"] = scored.existing_probability
    scored["candidate_probability"] = blend(scored.existing_probability, scored.platt_probability, weight)
    result = score_pair(scored)
    groups = groups_for(scored)
    result["platt_only_diagnostic"] = probability_metrics(scored.platt_probability, scored.won, groups)[0]
    result["weight"] = weight
    winner = scored.loc[scored.won == 1].copy()
    delta = -np.log(winner.candidate_probability) + np.log(winner.baseline_probability)
    winner["log_loss_delta"] = delta
    result["tail_diagnostic"] = {
        "legacy_winners_below_1e_7": int(winner.baseline_probability.lt(1e-7).sum()),
        "adjusted_winners_below_1e_7": int(winner.candidate_probability.lt(1e-7).sum()),
        "legacy_winners_below_0_001": int(winner.baseline_probability.lt(.001).sum()),
        "adjusted_winners_below_0_001": int(winner.candidate_probability.lt(.001).sum()),
        "legacy_max_winner_log_loss": float(-np.log(winner.baseline_probability.min())),
        "adjusted_max_winner_log_loss": float(-np.log(winner.candidate_probability.min())),
        "races_improved": int((delta < -1e-12).sum()),
        "races_worsened": int((delta > 1e-12).sum()),
        "total_log_loss_delta": float(delta.sum()),
        "largest_improvements": winner.nsmallest(5, "log_loss_delta")[[
            "race_id", "race_date", "horse_name", "baseline_probability",
            "candidate_probability", "log_loss_delta"]].assign(race_date=lambda x: x.race_date.astype(str)).to_dict("records")}
    return scored, result


def advances(result):
    d = result["paired_delta_candidate_minus_baseline"]
    return bool(result["weight"] > 0 and d["race_log_loss"]["ci95"][1] < 0
                and d["date_cluster_race_log_loss"]["ci95"][1] < 0
                and result["candidate"]["brier"] <= result["baseline"]["brier"]
                and result["candidate"]["binary_log_loss"] <= result["baseline"]["binary_log_loss"])


def run():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    previous_manifest = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))
    frozen_paths = [PREVIOUS / "inputs.parquet", PREVIOUS / "paired_predictions.parquet",
                    PREVIOUS / "manifest.json", FEATURES, FEATURES.with_name("meta.json")]
    frozen_paths += [PREVIOUS / f"{p}_{v}{ext}" for p in PERIODS for v in VARIANTS
                     for ext in (".joblib", "_fit.json")]
    frozen_paths += MARKETS + [p.with_suffix(".json") for p in MARKETS]
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    if input_hashes[str(FEATURES.relative_to(ROOT))] != previous_manifest["source_file_sha256"]:
        raise ValueError("Canonical features differ from the frozen experiment009 dataset")
    if input_hashes[str((PREVIOUS / "inputs.parquet").relative_to(ROOT))] != previous_manifest["projection_sha256"]:
        raise ValueError("Experiment009 feature projection changed")
    source_hashes = {p: digest(ROOT / p) for p in SOURCES}
    preserved = [ROOT / "data/runs/20261026_140950_205296/model.joblib",
                 ROOT / "data/runs/20261026_140950_205296/meta.json",
                 ROOT / "models/production.json"]
    study = ROOT / "data/forecasts/forward/20261026_baseline_v3"
    preserved += [study / name for name in ("protocol.json", "protocol.sha256.json",
                  "source_snapshot.zip", "model.joblib", "model_meta.json")]
    preservation_hashes = {str(p.relative_to(ROOT)): digest(p) for p in preserved}
    manifest = {"experiment": "010 fixed-tree isotonic tail shrinkage", "stages": STAGES,
                "source_sha256": source_hashes, "input_sha256": input_hashes,
                "preserved_sha256": preservation_hashes, "primary_variant": "baseline149",
                "secondary_variant": "pace176, descriptive only; cannot reverse009",
                "method": "p=(1-w)*existing+w*pre_isotonic_Platt; fit one w in [0,1]",
                "fit_rule": "Minimise race log loss subject to no runner Brier increase on fit forecasts",
                "validation_rule": "Retain learned w only with lower raceLL, Brier/binaryLL no worse; otherwise w=0; no refit",
                "advancement_rule": "Q1 race and date-cluster paired95% upper<0; runner Brier/binaryLL no worse; no automatic promotion",
                "transfer": "April-May uses December trees unchanged; model-aging stress, not a deployment-cadence estimate",
                "market": "Strict identical complete fields; observed T-5 snapshots; April unavailable; no SP substitute",
                "scope": "Retrospective chronological research on previously inspected development dates; not fresh confirmation",
                "source_experiment": "009 remains sealed; all valid races including extreme errors retained",
                "seed": 42, "threads": 2}
    publish(OUTPUT / "manifest.json", manifest)
    manifest_sha = digest(OUTPUT / "manifest.json")
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print(json.dumps({"stage": "registered", "manifest_sha256": manifest_sha}), flush=True)

    needed = list(dict.fromkeys(sum(previous_manifest["variants"].values(), []) + META))
    # Read just the six months needed, not the 2.5GB full feature matrix.
    historical = pd.read_parquet(PREVIOUS / "inputs.parquet", columns=needed,
                                 filters=[("race_date", ">=", pd.Timestamp("2025-07-01"))])
    historical["race_date"] = pd.to_datetime(historical.race_date).dt.normalize()
    historical = sorted_fields(historical.loc[valid_race_mask(historical) & development_mask(historical.race_date)].copy())
    previous_predictions = sorted_fields(pd.read_parquet(PREVIOUS / "paired_predictions.parquet"))
    selections, extracted, replay_evidence = {}, {}, {}

    # Fit and validation are processed before accessing new test alternatives.
    for variant in VARIANTS:
        for stage in ("fit", "validation"):
            period = STAGES[stage][0]
            predictor = RacePredictor()
            predictor.load(str(PREVIOUS / f"{period}_{variant}.joblib"))
            predictor.clf_model.set_params(n_jobs=2)
            frame = stage_frame(historical, stage, predictor.eval_split_info["as_of"])
            p, q, error = probability_stages(predictor, frame)
            old = frame[["race_id", "horse_name"]].merge(previous_predictions,
                   on=["race_id", "horse_name"], validate="one_to_one", how="left")[f"{variant}_probability"]
            legacy_error = float(np.max(np.abs(old.to_numpy() - p)))
            if not np.isfinite(old).all() or legacy_error > 1e-12:
                raise ValueError("Legacy prediction reconstruction differs from009")
            small = frame[[c for c in META if c in frame]].copy()
            small["existing_probability"], small["platt_probability"] = p, q
            small["model_cutoff"] = predictor.eval_split_info["as_of"]
            extracted[variant, stage] = small
            replay_evidence[f"{variant}_{stage}"] = {"native_error": error, "009_error": legacy_error}
            del predictor
            gc.collect()
        fit, validation = extracted[variant, "fit"], extracted[variant, "validation"]
        learned = fit_weight(fit.existing_probability, fit.platt_probability, fit.won, groups_for(fit))
        accepted = validate_weight(learned["weight"], validation.existing_probability,
                                   validation.platt_probability, validation.won, groups_for(validation))
        selections[variant] = {"fit": learned, "validation": accepted,
                               "selected_weight": accepted["selected_weight"],
                               "fit_dates": list(STAGES["fit"][1:]), "validation_dates": list(STAGES["validation"][1:])}
    publish(OUTPUT / "frozen_selection.json", {"manifest_sha256": manifest_sha, "variants": selections,
                                               "test_predictions_scored": False})
    selection_sha = digest(OUTPUT / "frozen_selection.json")
    print(json.dumps({"stage": "selection_frozen", "variants": selections}), flush=True)

    transfer = pd.read_parquet(FEATURES, columns=needed,
                              filters=[("race_date", ">=", pd.Timestamp("2026-04-01")),
                                       ("race_date", "<=", pd.Timestamp("2026-05-31"))])
    feature_cols = list(dict.fromkeys(sum(previous_manifest["variants"].values(), [])))
    transfer[feature_cols] = transfer[feature_cols].astype(np.float32)
    transfer["race_date"] = pd.to_datetime(transfer.race_date).dt.normalize()
    valid = valid_race_mask(transfer)
    exclusions = {"transfer_invalid_races": int(transfer.loc[~valid].race_id.nunique()),
                  "transfer_invalid_rows": int((~valid).sum())}
    transfer = sorted_fields(transfer.loc[valid].copy())
    snapshots = pd.concat([pd.read_parquet(p) for p in MARKETS], ignore_index=True)
    reports, scored_frames = {}, []
    for variant in VARIANTS:
        weight = selections[variant]["selected_weight"]
        predictor = RacePredictor()
        path = PREVIOUS / f"2026Q1_{variant}.joblib"
        predictor.load(str(path))
        predictor.clf_model.set_params(n_jobs=2)
        overlay = {"manifest_sha256": manifest_sha, "selection_sha256": selection_sha,
                   "model": str(path.relative_to(ROOT)), "model_sha256": digest(path),
                   "weight": weight, "variant": variant, "research_only": True}
        publish(OUTPUT / f"{variant}_overlay.json", overlay)
        reloaded = json.loads((OUTPUT / f"{variant}_overlay.json").read_text(encoding="utf-8"))
        reports[variant] = {}
        for stage, features in (("test", historical), ("transfer", transfer)):
            frame = stage_frame(features, stage, predictor.eval_split_info["as_of"])
            p, q, error = probability_stages(predictor, frame)
            if stage == "test":
                old = frame[["race_id", "horse_name"]].merge(previous_predictions,
                           on=["race_id", "horse_name"], how="left", validate="one_to_one")[f"{variant}_probability"]
                legacy_error = float(np.max(np.abs(old.to_numpy() - p)))
                if not np.isfinite(old).all() or legacy_error > 1e-12:
                    raise ValueError("Test legacy reconstruction differs from009")
            else:
                legacy_error = None
            small = frame[[c for c in META if c in frame]].copy()
            small["existing_probability"], small["platt_probability"] = p, q
            small["model_cutoff"] = predictor.eval_split_info["as_of"]
            extracted[variant, stage] = small
            scored, result = describe(small, weight)
            overlay_error = float(np.max(np.abs(blend(p, q, reloaded["weight"]) - scored.candidate_probability)))
            if overlay_error > 1e-12:
                raise ValueError("Saved overlay replay differs")
            replay_evidence[f"{variant}_{stage}"] = {"native_error": error,
                        "009_error": legacy_error, "overlay_error": overlay_error}
            market, coverage = match_market_fields(scored, snapshots)
            market.to_parquet(OUTPUT / f"{variant}_{stage}_market_predictions.parquet", index=False)
            market_report = {"coverage": coverage}
            if not market.empty:
                market_report["comparison"] = score_forecasts(market, {
                    "existing": market.baseline_probability.to_numpy(float),
                    "adjusted": market.candidate_probability.to_numpy(float),
                    "market": market.market_probability.to_numpy(float)})
            result["market_context"] = market_report
            reports[variant][stage] = result
            scored["variant"], scored["stage"] = variant, stage
            scored_frames.append(scored)
            print(json.dumps({"stage": stage, "variant": variant, "weight": weight,
                               "races": result["races"], "original": result["baseline"],
                               "adjusted": result["candidate"], "delta": result["paired_delta_candidate_minus_baseline"]}), flush=True)
        del predictor
        gc.collect()
    for (variant, stage), frame in extracted.items():
        frame.to_parquet(OUTPUT / f"{variant}_{stage}_inputs.parquet", index=False)
    pd.concat(scored_frames, ignore_index=True).to_parquet(OUTPUT / "test_predictions.parquet", index=False)
    if digest(OUTPUT / "frozen_selection.json") != selection_sha:
        raise RuntimeError("Selection changed after test scoring")
    for path, expected in {**input_hashes, **source_hashes, **preservation_hashes}.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen input/source changed during experiment: {path}")
    report = {"manifest_sha256": manifest_sha, "selection_sha256": selection_sha,
              "selection": selections, "results": reports, "replay": replay_evidence,
              "exclusions": exclusions, "primary_candidate_advances": advances(reports["baseline"]["test"]),
              "pace_diagnostic_passes_same_gate": advances(reports["candidate"]["test"]),
              "production_changed": False, "scope": manifest["scope"],
              "elapsed_seconds": time.monotonic() - started}
    publish(OUTPUT / "report.json", report)
    publish(OUTPUT / "completion.json", {"manifest_sha256": manifest_sha,
             "evidence_sha256": {str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.iterdir() if p.is_file()},
             "production_changed": False})
    print(json.dumps({"stage": "complete", "primary_candidate_advances": report["primary_candidate_advances"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    run()
