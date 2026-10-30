"""Post-result sensitivity: earlier best log-loss floor without the Brier veto.

This diagnostic cannot alter011's rejected selection or justify promotion.
It quantifies the trade-off exposed by that result. Choose solely from the
already recorded fitting scores, then show every later period and both models.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from scripts.benchmark_tail_calibration import groups_for, publish
from scripts.benchmark_tail_smoothing import score, source_frame, VARIANTS
from scripts.replay_market_snapshots import digest
from src.forecast_evaluation import probability_metrics
from src.tail_smoothing import smooth_tail

PRIMARY = ROOT / "data/audit/model_improvement/011_tail_smoothing"
OUTPUT = ROOT / "data/audit/model_improvement/011_tail_tradeoff_diagnostic"
SOURCES = ["scripts/diagnose_tail_tradeoff.py", "scripts/benchmark_tail_smoothing.py",
           "src/tail_smoothing.py", "scripts/benchmark_tail_calibration.py",
           "src/forecast_evaluation.py", "scripts/benchmark_independent_features.py",
           "scripts/assess_forecast_parity.py"]


def run():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    primary = json.loads((PRIMARY / "report.json").read_text())
    receipt = json.loads((PRIMARY / "completion.json").read_text())
    for name, expected in receipt["evidence_sha256"].items():
        if digest(PRIMARY / name) != expected:
            raise ValueError(f"Sealed011 changed: {name}")
    paths = list(PRIMARY.glob("*.json"))
    previous = PRIMARY.parent / "010_tail_calibration"
    paths += [previous / f"{v}_{s}_inputs.parquet" for v in VARIANTS
              for s in ("fit", "validation", "test", "transfer")]
    expected010 = json.loads((previous / "completion.json").read_text())["evidence_sha256"]
    for path in paths:
        if path.parent == previous and digest(path) != expected010[path.name]:
            raise ValueError(f"Sealed010 input changed: {path.name}")
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    sources = {s: digest(ROOT / s) for s in SOURCES}
    selected = {v: min(primary["selection"][v]["fit"]["candidates"],
                       key=lambda c: (c["metrics"]["race_log_loss"], c["strength"]))["strength"]
                for v in VARIANTS}
    specification = {"scope": "Post-result exploratory trade-off; no confirmation, promotion or reversal of011",
                     "reason": "Strict point-estimate Brier veto rejected all corrections in011",
                     "choice": "Earlier minimum fitting winner log loss among011's same seven strengths; no validation/test selection",
                     "strengths": selected, "input_sha256": inputs, "source_sha256": sources,
                     "promotion_eligible": False, "primary011_result_unchanged": True}
    publish(OUTPUT / "specification.json", specification)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    results, forecasts = {}, []
    for variant, strength in selected.items():
        results[variant] = {}
        for stage in ("fit", "validation", "test", "transfer"):
            frame = source_frame(variant, stage)
            if stage in ("fit", "validation"):
                sizes = groups_for(frame)
                probabilities = smooth_tail(frame.existing_probability, frame.platt_probability, sizes, strength)
                results[variant][stage] = {
                    "races": len(sizes), "runners": len(frame),
                    "original": probability_metrics(frame.existing_probability, frame.won, sizes)[0],
                    "adjusted": probability_metrics(probabilities, frame.won, sizes)[0]}
            else:
                scored, result = score(frame, strength)
                results[variant][stage] = result
                scored["variant"], scored["stage"] = variant, stage
                forecasts.append(scored)
        field = source_frame(variant, "validation")
        field = field.loc[field.race_id == 894033].copy()
        probabilities = smooth_tail(field.existing_probability, field.platt_probability, [len(field)], strength)
        winner = field.won.eq(1).to_numpy()
        results[variant]["known_validation_error"] = {
            "original": float(field.existing_probability.to_numpy()[winner][0]),
            "adjusted": float(probabilities[winner][0]), "scope": "Previously inspected validation example only"}
    pd.concat(forecasts, ignore_index=True).to_parquet(OUTPUT / "predictions.parquet", index=False)
    for path, expected in {**inputs, **sources}.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"Frozen diagnostic source/input changed: {path}")
    report = {"specification_sha256": digest(OUTPUT / "specification.json"), "results": results,
              "promotion_eligible": False, "production_changed": False, "primary011_result_unchanged": True}
    publish(OUTPUT / "report.json", report)
    publish(OUTPUT / "completion.json", {"evidence_sha256": {
        str(p.relative_to(OUTPUT)): digest(p) for p in OUTPUT.iterdir() if p.is_file()}})
    for v, stages in results.items():
        for s in ("fit", "validation", "test", "transfer"):
            item = stages[s]
            base = item.get("original", item.get("baseline"))
            adjusted = item.get("adjusted", item.get("candidate"))
            print(json.dumps({"variant": v, "stage": s, "strength": selected[v],
                "delta": {k: adjusted[k] - base[k] for k in ("race_log_loss", "brier", "binary_log_loss")},
                "paired": item.get("paired_delta_candidate_minus_baseline"),
                "activation": item.get("activation")}), flush=True)
    return report


if __name__ == "__main__":
    run()
