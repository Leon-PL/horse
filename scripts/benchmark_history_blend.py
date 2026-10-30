"""013: exploratory context ensemble after inspecting012's history-form lead.

All earlier results remain sealed. Fit two coefficients on Q3 OOS forecasts,
validate Q4, freeze before Q1 scoring. April-May transfer uses the same saved
December trees. Neither window is fresh confirmation of this post-result lead.
"""
from __future__ import annotations

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
from scripts.benchmark_history_features import align_reference
from scripts.benchmark_independent_features import META, score_pair
from scripts.benchmark_tail_calibration import groups_for, publish, stage_frame, STAGES
from scripts.replay_market_snapshots import digest
from src.forecast_evaluation import probability_metrics, valid_race_mask
from src.history_blend import CONTEXTS, MIN_RACES, apply_context_blend, contexts_from_flags, fit_context_blend
from src.model import RacePredictor

PREVIOUS = ROOT / "data/audit/model_improvement/012_history_family_compact"
BASELINE = ROOT / "data/audit/model_improvement/009_pace_family"
OLDER = ROOT / "data/audit/model_improvement/010_tail_calibration"
OUTPUT = ROOT / "data/audit/model_improvement/013_history_context_blend"
FEATURES = ROOT / "data/processed/feature_store/featured.parquet"
SOURCES = ["src/history_blend.py", "tests/test_history_blend.py", "scripts/benchmark_history_blend.py",
           "src/tail_calibration.py", "src/model.py", "src/forecast_evaluation.py", "src/wf_eval.py",
           "scripts/benchmark_forecasts.py", "scripts/benchmark_history_features.py",
           "scripts/benchmark_independent_features.py", "scripts/benchmark_tail_calibration.py",
           "scripts/assess_forecast_parity.py", "scripts/replay_market_snapshots.py", "config.py"]


def arrays(frame):
    g = groups_for(frame)
    context = contexts_from_flags(frame.has_hidden_form, g)
    return (frame.baseline_probability.to_numpy(float), frame.candidate_probability.to_numpy(float),
            frame.won.to_numpy(float), g, context)


def score(frame, weights):
    p, q, y, g, context = arrays(frame)
    scored = frame.copy()
    scored["global_history_probability"] = q
    scored["candidate_probability"] = apply_context_blend(p, q, g, context, weights)
    result = score_pair(scored)
    result["global_history_model"] = probability_metrics(q, y, g)[0]
    versus_global = scored.copy()
    versus_global["baseline_probability"] = q
    result["comparison_vs_global_history"] = score_pair(versus_global)
    result["context_races"] = {name: int((context == i).sum()) for i, name in enumerate(CONTEXTS)}
    return scored, result


def numeric_gate(result, weights):
    delta = result["paired_delta_candidate_minus_baseline"]
    return bool(any(w > 0 for w in weights.values())
                and delta["race_log_loss"]["ci95"][1] < 0
                and delta["date_cluster_race_log_loss"]["ci95"][1] < 0
                and result["candidate"]["brier"] <= result["baseline"]["brier"]
                and result["candidate"]["binary_log_loss"] <= result["baseline"]["binary_log_loss"])


def run():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((PREVIOUS / "completion.json").read_text())
    for name, expected in receipt["evidence_sha256"].items():
        if digest(PREVIOUS / name) != expected:
            raise ValueError(f"Sealed012 evidence changed: {name}")
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    feature_hash = digest(FEATURES)
    if feature_hash != previous["source_feature_sha256"]:
        raise ValueError("Transfer feature snapshot differs from012")
    paths = [PREVIOUS / n for n in receipt["evidence_sha256"]] + [PREVIOUS / "completion.json"]
    paths += [OLDER / "baseline_transfer_inputs.parquet", OLDER / "baseline_test_market_predictions.parquet",
              OLDER / "baseline_transfer_market_predictions.parquet", OLDER / "completion.json"]
    paths += [BASELINE / f"{p}_baseline.joblib" for p in ("2025Q3", "2025Q4", "2026Q1")]
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    preserved = previous["preserved_sha256"]
    for p, h in preserved.items():
        if digest(ROOT / p) != h:
            raise ValueError(f"Production/study artifact changed: {p}")
    sources = {s: digest(ROOT / s) for s in SOURCES}
    specification = {"experiment": "013 two-context whole-race probability ensemble",
        "hypothesis_origin": "012 inspected subgroup lead, including Q1 outcomes; exploratory retrospective only",
        "contexts": "Any pre-race has_hidden_form=1 in field versus none; same weight for every runner",
        "minimum_fit_and_validation_races_per_context": MIN_RACES, "stages": STAGES,
        "fit_rule": "One convex coefficient/context on earlier Q3 OOS predictions minimizing winnerLL under no worse runnerBrier",
        "validation_rule": "Accept learned coefficient on later Q4 only with lower LL and Brier/binary no worse; else0, never refit",
        "primary_test": "Q1 only; fit/validation periods are not pooled as test evidence",
        "transfer": "Apr1–May31 with same Dec22 trees/coefficients, aging stress; May-only market coverage",
        "numeric_gate": "Q1 race/date-cluster95%upper<0 versus149, no worse Brier/binary; no automatic promotion",
        "scope": "Post-result exploratory development; conditional weight learning is chronological, but context was chosen after inspecting later012 results; fresh confirmation required",
        "source_sha256": sources, "input_sha256": inputs, "preserved_sha256": preserved,
        "source_feature_sha256": feature_hash, "no_new_tree_fits": True, "seed": 42}
    publish(OUTPUT / "manifest.json", specification)
    manifest_sha = digest(OUTPUT / "manifest.json")
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print(json.dumps({"stage": "registered", "manifest_sha256": manifest_sha}), flush=True)
    cutoffs = {f["period"]: f["cutoff"] for f in previous["folds"]}
    earlier = {}
    for stage in ("fit", "validation"):
        period = STAGES[stage][0]
        raw = pd.read_parquet(PREVIOUS / f"{period}_predictions.parquet")
        earlier[stage] = stage_frame(raw, stage, cutoffs[period])
    selection = fit_context_blend(arrays(earlier["fit"]), arrays(earlier["validation"]))
    weights = {name: s["selected_weight"] for name, s in selection.items()}
    publish(OUTPUT / "frozen_selection.json", {"manifest_sha256": manifest_sha,
               "selection": selection, "weights": weights, "test_scored": False})
    selection_sha = digest(OUTPUT / "frozen_selection.json")
    publish(OUTPUT / "overlay.json", {"manifest_sha256": manifest_sha, "selection_sha256": selection_sha,
        "weights": weights, "context_rule": "Any has_hidden_form=1 in a complete field",
        "baseline_model": str((BASELINE / "2026Q1_baseline.joblib").relative_to(ROOT)),
        "history_model": str((PREVIOUS / "2026Q1_candidate.joblib").relative_to(ROOT)), "research_only": True})
    print(json.dumps({"stage": "selection_frozen", "selection": selection, "weights": weights}), flush=True)
    restored = json.loads((OUTPUT / "overlay.json").read_text())["weights"]
    raw = pd.read_parquet(PREVIOUS / "2026Q1_predictions.parquet")
    test = stage_frame(raw, "test", cutoffs["2026Q1"])
    features = previous["candidate_features"]
    columns = list(dict.fromkeys(features + META))
    transfer = pd.read_parquet(FEATURES, columns=columns,
               filters=[("race_date", ">=", pd.Timestamp("2026-04-01")),
                        ("race_date", "<=", pd.Timestamp("2026-05-31"))])
    transfer[features] = transfer[features].astype(np.float32)
    transfer["race_date"] = pd.to_datetime(transfer.race_date).dt.normalize()
    valid = valid_race_mask(transfer)
    exclusions = {"invalid_transfer_fields": int(transfer.loc[~valid].race_id.nunique()),
                  "invalid_transfer_runners": int((~valid).sum())}
    transfer = stage_frame(transfer.loc[valid].copy(), "transfer", cutoffs["2026Q1"])
    transfer.to_parquet(OUTPUT / "transfer_features.parquet", index=False, compression="zstd")
    model = RacePredictor()
    model.load(str(BASELINE / "2026Q1_baseline.joblib"))
    model.clf_model.set_params(n_jobs=2)
    native = align_predictions(model, transfer)
    saved = pd.read_parquet(OLDER / "baseline_transfer_inputs.parquet").rename(columns={"existing_probability": "baseline_probability"})
    transfer["baseline_probability"], base_error = align_reference(transfer, saved, native)
    model.load(str(PREVIOUS / "2026Q1_candidate.joblib"))
    model.clf_model.set_params(n_jobs=2)
    transfer["candidate_probability"] = align_predictions(model, transfer)
    results, forecasts, replay = {}, [], {}
    for stage, frame in (("test", test), ("transfer", transfer)):
        scored, result = score(frame, weights)
        p, q, y, g, context = arrays(frame)
        replay[stage] = float(np.max(np.abs(apply_context_blend(p, q, g, context, restored) - scored.candidate_probability)))
        if replay[stage] > 1e-12:
            raise ValueError("Context overlay replay differs")
        market = pd.read_parquet(OLDER / f"baseline_{stage}_market_predictions.parquet")
        market = market.drop(columns=["baseline_probability", "candidate_probability"])
        market = market.merge(scored[["race_id", "horse_name", "baseline_probability", "candidate_probability", "global_history_probability"]],
                              on=["race_id", "horse_name"], how="left", validate="one_to_one")
        if market[["baseline_probability", "candidate_probability", "global_history_probability"]].isna().any().any():
            raise ValueError("Frozen market field does not match")
        result["market"] = score_forecasts(market, {"baseline": market.baseline_probability.to_numpy(float),
            "context_blend": market.candidate_probability.to_numpy(float), "history": market.global_history_probability.to_numpy(float),
            "market": market.market_probability.to_numpy(float)})
        market.to_parquet(OUTPUT / f"{stage}_market_predictions.parquet", index=False)
        results[stage] = result
        scored["stage"] = stage
        forecasts.append(scored)
        print(json.dumps({"stage": stage, "result": result}), flush=True)
    pd.concat(forecasts, ignore_index=True).to_parquet(OUTPUT / "predictions.parquet", index=False)
    for path, expected in {**sources, **inputs, **preserved}.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen source/input changed: {path}")
    if digest(FEATURES) != feature_hash or digest(OUTPUT / "frozen_selection.json") != selection_sha:
        raise RuntimeError("Feature snapshot or selected weights changed after scoring")
    report = {"manifest_sha256": manifest_sha, "selection_sha256": selection_sha,
              "selection": selection, "weights": weights, "results": results, "overlay_replay_error": replay,
              "baseline_transfer_native_replay_error": base_error, "exclusions": exclusions,
              "primary_passes_numeric_gate": numeric_gate(results["test"], weights),
              "scope": specification["scope"], "fresh_confirmation": False,
              "production_changed": False, "elapsed_seconds": time.monotonic() - started}
    publish(OUTPUT / "report.json", report)
    publish(OUTPUT / "completion.json", {"manifest_sha256": manifest_sha, "evidence_sha256": {
        str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.iterdir() if p.is_file()}, "production_changed": False})
    print(json.dumps({"stage": "complete", "weights": weights,
                      "primary_passes_numeric_gate": report["primary_passes_numeric_gate"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    run()
