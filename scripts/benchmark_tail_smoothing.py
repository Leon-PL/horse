"""Experiment011: a narrow relative floor on saved OOS forecasts from010.

No tree refits, no change to any sealed experiment or serving path. Method
developed after010 was inspected, so all results are exploratory retrospective
research, including the later chronological comparison. No fresh-data claim.
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

from scripts.benchmark_forecasts import score_forecasts
from scripts.benchmark_independent_features import score_pair
from scripts.benchmark_tail_calibration import groups_for, publish, STAGES, stage_frame
from scripts.replay_market_snapshots import digest
from src.forecast_evaluation import probability_metrics, valid_race_mask
from src.tail_smoothing import (STRENGTHS, TAIL_THRESHOLD, activation_summary,
                                fit_strength, smooth_tail, validate_strength)

PREVIOUS = ROOT / "data/audit/model_improvement/010_tail_calibration"
OUTPUT = ROOT / "data/audit/model_improvement/011_tail_smoothing"
VARIANTS = ("baseline", "candidate")
SOURCES = ["src/tail_smoothing.py", "scripts/benchmark_tail_smoothing.py", "tests/test_tail_smoothing.py",
           "src/forecast_evaluation.py", "src/wf_eval.py", "config.py",
           "scripts/benchmark_independent_features.py", "scripts/benchmark_forecasts.py",
           "scripts/benchmark_tail_calibration.py", "scripts/assess_forecast_parity.py",
           "scripts/replay_market_snapshots.py"]


def source_frame(variant, stage):
    frame = pd.read_parquet(PREVIOUS / f"{variant}_{stage}_inputs.parquet")
    if frame.model_cutoff.nunique() != 1:
        raise ValueError("Each stage requires one historical model cutoff")
    checked = stage_frame(frame, stage, frame.model_cutoff.iloc[0])
    if len(checked) != len(frame) or not valid_race_mask(frame).all():
        raise ValueError("Do not drop or reorder invalid/partial input fields silently")
    return checked


def score(frame, strength):
    result = frame.copy()
    groups = groups_for(frame)
    result["baseline_probability"] = frame.existing_probability
    result["candidate_probability"] = smooth_tail(frame.existing_probability, frame.platt_probability, groups, strength)
    report = score_pair(result)
    winner = result.loc[result.won == 1].copy()
    winner["log_loss_delta"] = -np.log(winner.candidate_probability) + np.log(winner.baseline_probability)
    cols = ["race_id", "race_date", "horse_name", "baseline_probability", "candidate_probability", "log_loss_delta"]
    report.update(strength=strength,
        activation=activation_summary(frame.existing_probability, frame.platt_probability, groups, strength),
        tail={"legacy_winners_below_1e_7": int(winner.baseline_probability.lt(1e-7).sum()),
              "adjusted_winners_below_1e_7": int(winner.candidate_probability.lt(1e-7).sum()),
              "legacy_max_winner_log_loss": float(-np.log(winner.baseline_probability.min())),
              "adjusted_max_winner_log_loss": float(-np.log(winner.candidate_probability.min())),
              "races_improved": int(winner.log_loss_delta.lt(-1e-12).sum()),
              "races_worsened": int(winner.log_loss_delta.gt(1e-12).sum()),
              "total_log_loss_delta": float(winner.log_loss_delta.sum()),
              "largest_improvements": winner.nsmallest(5, "log_loss_delta")[cols].assign(
                  race_date=lambda x: x.race_date.astype(str)).to_dict("records")})
    return result, report


def passes_gate(report):
    delta = report["paired_delta_candidate_minus_baseline"]
    return bool(report["strength"] > 0
                and delta["race_log_loss"]["ci95"][1] < 0
                and delta["date_cluster_race_log_loss"]["ci95"][1] < 0
                and report["candidate"]["brier"] <= report["baseline"]["brier"]
                and report["candidate"]["binary_log_loss"] <= report["baseline"]["binary_log_loss"])


def run():
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    completion = json.loads((PREVIOUS / "completion.json").read_text())
    for name, expected in completion["evidence_sha256"].items():
        if digest(PREVIOUS / name) != expected:
            raise ValueError(f"Sealed010 evidence changed: {name}")
    paths = [PREVIOUS / name for name in completion["evidence_sha256"]] + [PREVIOUS / "completion.json"]
    paths += [ROOT / "data/audit/model_improvement/009_followup_calibration/zero_probability_trace.json"]
    original_manifest = json.loads((PREVIOUS / "manifest.json").read_text())
    for source in set(SOURCES) & set(original_manifest["source_sha256"]):
        if digest(ROOT / source) != original_manifest["source_sha256"][source]:
            raise ValueError(f"Shared010 scoring/inference source changed: {source}")
    # We consume only010's verified saved probability vectors. Preserve production,
    # its study and the six source models; do not reread/rebuild the feature store.
    preserved = {**original_manifest["preserved_sha256"], **{
        p: h for p, h in original_manifest["input_sha256"].items() if p.endswith(".joblib")}}
    for path, expected in preserved.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Preserved model/study changed: {path}")
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    source_hashes = {p: digest(ROOT / p) for p in SOURCES}
    manifest = {"experiment": "011 narrow relative probability floor", "stages": STAGES,
                "strengths": STRENGTHS, "tail_threshold": TAIL_THRESHOLD,
                "method": "Normalize max(p, strength*pre_isotonic_Platt) only where p<.01; strength<=.25",
                "fit_rule": "Q3 min winnerLL among fixed strengths with runnerBrier/binaryLL no worse; ties identity",
                "validation_rule": "Q4 accepts learned strength only with lower winnerLL and Brier/binaryLL no worse; else identity; no refit",
                "test_rule": "Primary149 Q1 race/date-cluster95%upper<0 and runnerBrier/binaryLL no worse; no automatic promotion",
                "secondary": "176pace descriptive only, cannot reverse009; April-May same December trees is aging stress",
                "scope": "Exploratory retrospective development; hypothesis follows inspected010 test outcomes; no fresh confirmation",
                "market": "Same strict complete-field overlap already matched in010; T5; April unavailable; no SP substitute",
                "tail_efficacy": "Q1/transfer previously had no near-zero winning forecast; known December error is explanatory validation only",
                "input_sha256": input_hashes, "source_sha256": source_hashes,
                "preserved_sha256": preserved, "seed": 42}
    publish(OUTPUT / "manifest.json", manifest)
    manifest_sha = digest(OUTPUT / "manifest.json")
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    print(json.dumps({"stage": "registered", "manifest_sha256": manifest_sha}), flush=True)

    selection = {}
    for variant in VARIANTS:
        fit, validation = (source_frame(variant, s) for s in ("fit", "validation"))
        learned = fit_strength(fit.existing_probability, fit.platt_probability, fit.won, groups_for(fit))
        accepted = validate_strength(learned["learned_strength"], validation.existing_probability,
                                    validation.platt_probability, validation.won, groups_for(validation))
        selection[variant] = {"fit": learned, "validation": accepted,
                              "selected_strength": accepted["selected_strength"]}
    publish(OUTPUT / "frozen_selection.json", {"manifest_sha256": manifest_sha, "selection": selection,
                                               "test_probabilities_scored": False})
    selection_sha = digest(OUTPUT / "frozen_selection.json")
    print(json.dumps({"stage": "selection_frozen", "selection": selection}), flush=True)

    reports, predictions, replay = {}, [], {}
    for variant in VARIANTS:
        strength = selection[variant]["selected_strength"]
        previous_overlay = json.loads((PREVIOUS / f"{variant}_overlay.json").read_text())
        overlay = {"manifest_sha256": manifest_sha, "selection_sha256": selection_sha,
                   "strength": strength, "tail_threshold": TAIL_THRESHOLD, "variant": variant,
                   "model": previous_overlay["model"], "model_sha256": previous_overlay["model_sha256"],
                   "method": "relative_floor_v1", "research_only": True}
        publish(OUTPUT / f"{variant}_overlay.json", overlay)
        restored = json.loads((OUTPUT / f"{variant}_overlay.json").read_text())
        reports[variant] = {}
        for stage in ("test", "transfer"):
            frame = source_frame(variant, stage)
            scored, report = score(frame, strength)
            regenerated = smooth_tail(frame.existing_probability, frame.platt_probability,
                                      groups_for(frame), restored["strength"])
            error = float(np.max(np.abs(regenerated - scored.candidate_probability)))
            if error > 1e-12:
                raise ValueError("Saved tail overlay differs on replay")
            replay[f"{variant}_{stage}"] = error
            market = pd.read_parquet(PREVIOUS / f"{variant}_{stage}_market_predictions.parquet")
            market = market.drop(columns=["candidate_probability", "baseline_probability"])
            market = market.merge(scored[["race_id", "horse_name", "baseline_probability", "candidate_probability"]],
                                  on=["race_id", "horse_name"], how="left", validate="one_to_one")
            if market[["baseline_probability", "candidate_probability"]].isna().any().any():
                raise ValueError("Complete frozen market fields no longer match")
            report["market"] = score_forecasts(market, {
                "existing": market.baseline_probability.to_numpy(float),
                "adjusted": market.candidate_probability.to_numpy(float),
                "market": market.market_probability.to_numpy(float)})
            market.to_parquet(OUTPUT / f"{variant}_{stage}_market_predictions.parquet", index=False)
            report["market_coverage"] = json.loads((PREVIOUS / "report.json").read_text())["results"][variant][stage]["market_context"]["coverage"]
            reports[variant][stage] = report
            scored["variant"], scored["stage"] = variant, stage
            predictions.append(scored)
            print(json.dumps({"stage": stage, "variant": variant, "strength": strength,
                               "races": report["races"], "original": report["baseline"],
                               "adjusted": report["candidate"], "delta": report["paired_delta_candidate_minus_baseline"],
                               "activation": report["activation"]}), flush=True)
    # Explain the already-inspected December failure with the actual complete field.
    # It is validation, never a new test or a reason to override rejected selection.
    trace = json.loads(paths[-1].read_text())
    diagnosis = {}
    for variant in VARIANTS:
        validation = source_frame(variant, "validation")
        field = validation.loc[validation.race_id == trace["race_id"]].copy()
        if len(field) != trace["runners"]:
            raise ValueError("Known-error diagnosis must retain its complete field")
        winner = field.won.eq(1).to_numpy()
        p, q = field.existing_probability.to_numpy(), field.platt_probability.to_numpy()
        if not np.isclose(p[winner][0], trace["stages"][variant]["final_probability"], atol=1e-12, rtol=0):
            raise ValueError("Known-error probability differs from the original native trace")
        learned = selection[variant]["fit"]["learned_strength"]
        selected = selection[variant]["selected_strength"]
        diagnosis[variant] = {"race_id": trace["race_id"], "scope": "Already inspected validation case, no test claim",
             "original_probability": float(p[winner][0]), "platt_probability": float(q[winner][0]),
             "learned_strength": learned, "selected_strength": selected,
             "learned_probability": float(smooth_tail(p, q, [len(field)], learned)[winner][0]),
             "selected_probability": float(smooth_tail(p, q, [len(field)], selected)[winner][0])}
    pd.concat(predictions, ignore_index=True).to_parquet(OUTPUT / "test_predictions.parquet", index=False)
    report = {"manifest_sha256": manifest_sha, "selection_sha256": selection_sha,
              "selection": selection, "results": reports, "known_error_diagnostic": diagnosis,
              "overlay_replay_max_error": replay, "primary_passes_numeric_gate": passes_gate(reports["baseline"]["test"]),
              "pace_passes_numeric_gate_diagnostic": passes_gate(reports["candidate"]["test"]),
              "production_changed": False, "scope": manifest["scope"],
              "elapsed_seconds": time.monotonic() - started}
    for path, expected in {**input_hashes, **source_hashes, **preserved}.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen source/input changed during experiment: {path}")
    if digest(OUTPUT / "frozen_selection.json") != selection_sha:
        raise RuntimeError("Selection changed after scoring")
    publish(OUTPUT / "report.json", report)
    publish(OUTPUT / "completion.json", {"manifest_sha256": manifest_sha,
             "evidence_sha256": {str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.iterdir() if p.is_file()},
             "production_changed": False})
    print(json.dumps({"stage": "complete", "primary_passes_numeric_gate": report["primary_passes_numeric_gate"],
                      "known_error_diagnostic": diagnosis, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    run()
