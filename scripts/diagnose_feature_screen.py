"""Describe the frozen 018 screen's market gap, coverage and probability tails.

This does not fit, tune, select subgroups for deployment or alter the screen.
All subgroup scores retain complete fields; overlapping groups are diagnostic.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.benchmark_forecasts import score_forecasts
from scripts.benchmark_tail_calibration import probability_stages
from scripts.screen_feature_batch import OUTPUT, REFERENCE, sha, verify
from src.forecast_evaluation import probability_metrics, valid_race_mask
from src.model import RacePredictor

KEYS = ["race_id", "horse_name", "race_date", "won"]


def metrics(frame, columns):
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    if not np.array_equal(frame.race_id.to_numpy(),
                         np.repeat(frame.race_id.drop_duplicates().to_numpy(), groups)):
        raise ValueError("Diagnostic requires contiguous complete fields")
    return {name: probability_metrics(frame[name], frame.won, groups)[0] for name in columns}


def main():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    verify(manifest)
    if not (OUTPUT / "report.json").exists():
        raise ValueError("Finish all registered fits and the primary report first")
    names = list(manifest["features"])
    fits = {name: json.loads((OUTPUT / f"{name}_fit.json").read_text()) for name in names}
    sources = {}
    frames = {}
    for name in names:
        path = OUTPUT / f"{name}_predictions.parquet"
        if sha(path) != fits[name]["prediction_sha256"]:
            raise ValueError("Screen predictions changed")
        sources[str(path.relative_to(ROOT))] = sha(path)
        frames[name] = pd.read_parquet(path)
    frame = frames[names[0]].drop(columns="probability")
    for name in names:
        if not frame[KEYS].equals(frames[name][KEYS]):
            raise ValueError("Diagnostic must use identical fields and labels")
        frame[name] = frames[name].probability.to_numpy()
    if not valid_race_mask(frame).all():
        raise ValueError("Diagnostic encountered an incomplete or invalid field")
    feature_names = ["local_history_available", "source_shows_unseen_experience",
                     "source_shows_intervening_run", "source_recency_conflict", "external_prior_runs"]
    inputs = pd.read_parquet(OUTPUT / "public_inputs.parquet", columns=[*KEYS, *feature_names])
    frame = frame.merge(inputs, on=KEYS, how="left", sort=False, validate="one_to_one")
    if frame[feature_names].isna().any().any():
        raise ValueError("Diagnostic lost a registered feature row")
    group = frame.groupby("race_id", sort=False)
    field_masks = {
        "contains_no_local_history": group.local_history_available.transform("min").eq(0),
        "contains_unseen_source_experience": group.source_shows_unseen_experience.transform("max").eq(1),
        "contains_intervening_run": group.source_shows_intervening_run.transform("max").eq(1),
        "contains_public_prior_form": group.external_prior_runs.transform("max").gt(0),
        "Flat": frame.race_type.eq("Flat"), "Hurdle": frame.race_type.eq("Hurdle"),
        "Chase": frame.race_type.eq("Chase"), "Cheltenham": frame.track.eq("Cheltenham"),
    }
    subgroups = {}
    for label, mask in field_masks.items():
        subset = frame.loc[mask]
        if subset.empty:
            subgroups[label] = {"races": 0, "runners": 0}
            continue
        if not valid_race_mask(subset).all():
            raise ValueError(f"Subgroup contains partial fields: {label}")
        subgroups[label] = {"races": int(subset.race_id.nunique()), "runners": len(subset),
                            "metrics": metrics(subset, names)}
    tail = {}
    bins = np.array([0, .001, .005, .01, .02, .05, .1, .2, .4, .6, .8, 1.0000001])
    winners = frame.loc[frame.won.eq(1), [*KEYS, "race_type", "track", *feature_names, *names]].copy()
    for name in names:
        p = frame[name].to_numpy()
        buckets = np.clip(np.searchsorted(bins, p, side="right")-1, 0, len(bins)-2)
        tail[name] = {
            "winner_probabilities_below": {str(t): int((frame.won.eq(1) & frame[name].lt(t)).sum())
                                           for t in (.001, .005, .01, .02)},
            "minimum_winner_probability": float(frame.loc[frame.won.eq(1), name].min()),
            "calibration_bins": [{"lower": float(bins[b]), "upper": min(float(bins[b+1]), 1.),
                "runners": int((buckets == b).sum()), "expected_wins": float(p[buckets == b].sum()),
                "observed_wins": int(frame.loc[buckets == b, "won"].sum())} for b in range(len(bins)-1)],
        }
        # Locate any near-zero error in the native probability chain without
        # fitting or selecting an alternative calibrator on these outcomes.
        model_path = OUTPUT / f"{name}.joblib"
        if sha(model_path) != fits[name]["model_sha256"]:
            raise ValueError("Saved model changed")
        test_inputs = pd.read_parquet(OUTPUT / f"{name}_inputs.parquet",
                                     filters=[("race_date", ">=", pd.Timestamp(manifest["test_start"]))])
        if not frame[KEYS].equals(test_inputs[KEYS]):
            raise ValueError("Native stage diagnostic lost runner alignment")
        predictor = RacePredictor()
        predictor.load(str(model_path))
        existing, platt, replay_error = probability_stages(predictor, test_inputs)
        if not np.array_equal(existing, frame[name].to_numpy()):
            raise ValueError("Stage replay differs from registered screen")
        sizes = frame.groupby("race_id", sort=False).size().to_numpy()
        tail[name]["native_stage_replay_max_error"] = replay_error
        tail[name]["pre_isotonic_metrics_diagnostic_only"] = probability_metrics(platt, frame.won, sizes)[0]
        tail[name]["winner_tail_loss_below_001"] = float(-np.log(np.clip(
            frame.loc[frame.won.eq(1) & frame[name].lt(.001), name], 1e-12, 1)).sum())
        winners[f"{name}_pre_isotonic"] = platt[frame.won.eq(1)]
        del test_inputs, predictor
    winners["repair_minus_baseline_winner_log_loss"] = (
        np.log(winners.baseline.clip(lower=1e-12))-np.log(winners.repairs.clip(lower=1e-12)))
    winners.sort_values("repairs").head(30).to_csv(OUTPUT / "low_probability_winners.csv", index=False)
    winners.sort_values("repair_minus_baseline_winner_log_loss", ascending=False).head(30).to_csv(
        OUTPUT / "largest_repair_log_loss_increases.csv", index=False)
    market_path = REFERENCE / "market_context_predictions.parquet"
    completed = json.loads((REFERENCE / "completion.json").read_text())
    if sha(market_path) != completed["evidence_sha256"][market_path.name]:
        raise ValueError("Frozen market evidence changed")
    sources[str(market_path.relative_to(ROOT))] = sha(market_path)
    market = pd.read_parquet(market_path)
    market = market.loc[pd.to_datetime(market.race_date).between(manifest["test_start"], manifest["test_end"])]
    paired = market.drop(columns=["baseline_probability", "candidate_probability"]).merge(
        frame[KEYS + names], on=KEYS, how="left", sort=False, validate="one_to_one")
    if paired[names].isna().any().any() or len(paired) != len(market) or not valid_race_mask(paired).all():
        raise ValueError("Every runner in each frozen market field must align")
    market_scores = score_forecasts(paired, {"market": paired.market_probability.to_numpy(),
                                           **{name: paired[name].to_numpy() for name in names}})
    scored = market_scores["metrics"]
    gap = scored["baseline"]["race_log_loss"]-scored["market"]["race_log_loss"]
    report = {"scope": "Descriptive development diagnostics on the fixed 018 screen, not confirmation or subgroup selection",
        "screen_manifest_sha256": sha(OUTPUT / "manifest.json"), "source_sha256": sources,
        "diagnostic_source_sha256": sha(Path(__file__)), "subgroups": subgroups, "probability_tails": tail,
        "market": market_scores,
        "fraction_of_matched_screen_market_gap_closed_point_estimate": {
            name: (scored["baseline"]["race_log_loss"]-scored[name]["race_log_loss"])/gap if gap > 0 else None
            for name in names if name != "baseline"}}
    (OUTPUT / "diagnostics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"market_races": market_scores["races"], "metrics": scored,
        "fraction_of_screen_gap_closed": report["fraction_of_matched_screen_market_gap_closed_point_estimate"],
        "low_probability_winners": {n: tail[n]["winner_probabilities_below"] for n in names}}, indent=2))


if __name__ == "__main__":
    main()
