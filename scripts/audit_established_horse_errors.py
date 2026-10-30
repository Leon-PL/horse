"""Trace established-horse errors using frozen, genuinely out-of-sample forecasts.

This is descriptive analysis of previously examined development races. It does
not train, alter the installed model, or select a winning recipe on those races.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.deployment_validation import file_sha256
from src.forecast_evaluation import paired_interval, probability_metrics, valid_race_mask

SOURCE = ROOT / "data/audit/canonical_repairs_026"
STORE = ROOT / "data/processed/feature_store"
OUTPUT = ROOT / "data/audit/established_horse_review_028"
KEY = ["race_id", "horse_name"]
CONTEXT = [
    "horse_id", "race_name", "race_class", "race_type", "track", "surface",
    "distance_furlongs", "going", "age", "sex", "jockey", "trainer", "form", "beaten_lengths",
    "days_since_last_run", "official_rating", "has_official_rating", "weight_lbs",
    "horse_prev_races", "horse_rt_runs", "horse_surface_runs", "horse_dist_cat_runs",
    "horse_place_rate", "horse_avg_pos_3", "horse_best_pos_5", "days_since_last_race",
    "horse_last_run_non_finish", "horse_non_finish_rate_shrunk", "horse_elo",
    "horse_elo_rt", "horse_elo_vs_field", "horse_margin_elo", "horse_glicko_rd",
    "horse_glicko_lcb", "horse_ts_mu", "horse_ts_sigma", "class_change", "or_change",
    "distance_change", "horse_track_runs", "horse_cd_runs", "horse_going_runs",
    "speed_fig_best_5", "speed_fig_adj_best_5", "speed_z_class_best_5",
    "horse_avg_prize_contested", "prize_money", "prize_money_original", "prize_currency",
]


def metrics(frame):
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable")
    if not len(frame):
        return {"races": 0, "runners": 0}
    groups = frame.groupby("race_id", sort=False).size().to_numpy()
    result = {"races": len(groups), "runners": len(frame)}
    losses = {}
    for name in ("candidate", "market"):
        scored = probability_metrics(
            frame[f"{name}_probability"].to_numpy(), frame.won.to_numpy(float), groups,
        )
        result[name], losses[name] = scored[:2]
    result["log_loss_gap"] = result["candidate"]["race_log_loss"] - result["market"]["race_log_loss"]
    result["date_cluster_log_loss_gap"] = paired_interval(losses["candidate"]-losses["market"],
        blocks=frame.groupby("race_id", sort=False).race_date.first().astype(str).to_numpy())
    return result


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = SOURCE / "market_matched_predictions.parquet"
    frame = pd.read_parquet(source)
    schema = pq.read_schema(STORE / "featured.parquet")
    columns = list(dict.fromkeys(KEY + ["race_date", "won", "finish_position"] + CONTEXT))
    columns = [c for c in columns if c in schema.names]
    context = pd.read_parquet(
        STORE / "featured.parquet", columns=columns,
        filters=[("race_date", ">=", frame.race_date.min()), ("race_date", "<=", frame.race_date.max())],
    )
    overlapping = [c for c in context if c in frame and c not in KEY]
    joined = frame.merge(context, on=KEY, how="left", validate="one_to_one", suffixes=("", "_context"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Missing feature context; never drop a runner to make the join work")
    metadata_repairs = []
    for c in overlapping:
        if c == "race_type":
            # The market join retained016's old descriptive race-type label.
            # Use the repaired type, validated against026's actual inputs below.
            changed = joined[c].ne(joined[c + "_context"])
            metadata_repairs += joined.loc[changed, KEY + [c, c + "_context"]].to_dict("records")
            joined["benchmark_race_type"] = joined[c]
            joined[c] = joined[c + "_context"]
        else:
            pd.testing.assert_series_equal(joined[c], joined[c + "_context"], check_names=False, check_dtype=False)
    frame = joined.drop(columns=[c + "_context" for c in overlapping] + ["_merge"])
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    if not valid_race_mask(frame).all():
        raise ValueError("Only complete valid paired fields are allowed")
    for c in ("candidate_probability", "market_probability"):
        np.testing.assert_allclose(frame.groupby("race_id")[c].sum(), 1, rtol=0, atol=1e-6)
    # Confirm the context comes from the same repaired feature generation.
    frozen_cols = [c for c in CONTEXT if c in pq.read_schema(SOURCE / "corrected_inputs.parquet").names]
    frozen = pd.read_parquet(SOURCE / "corrected_inputs.parquet", columns=KEY + frozen_cols,
                             filters=[("race_date", ">=", frame.race_date.min()), ("race_date", "<=", frame.race_date.max())])
    frozen = frame[KEY].merge(frozen, on=KEY, how="left", validate="one_to_one")
    for c in frozen_cols:
        pd.testing.assert_series_equal(frame[c].reset_index(drop=True), frozen[c], check_names=False, check_dtype=False)

    field = frame.groupby("race_id", sort=False)
    races = field[["race_date", "race_type", "handicap"]].first()
    races["minimum_local_starts"] = field.horse_prev_races.min()
    races["median_local_starts"] = field.horse_prev_races.median()
    cohorts = {"all_paired_fields": metrics(frame)}
    for n in (1, 3, 5, 10):
        ids = races.index[races.minimum_local_starts.ge(n)]
        cohorts[f"every_runner_at_least_{n}_local_starts"] = metrics(frame.loc[frame.race_id.isin(ids)])
    established = frame.horse_prev_races.ge(5)
    frame["established_local_5"] = established
    frame["probability_gap"] = frame.candidate_probability - frame.market_probability
    frame["log_market_over_model"] = np.log(frame.market_probability.clip(1e-15) / frame.candidate_probability.clip(1e-15))
    frame["runner_brier_gap"] = (frame.candidate_probability-frame.won)**2 - (frame.market_probability-frame.won)**2
    # Attribute whole-race log loss to its winner. The strict validity check
    # above accepts one labelled winner per complete field.
    shares = frame.won / frame.groupby("race_id").won.transform("sum")
    frame["race_log_loss_gap_contribution"] = shares * frame.log_market_over_model
    context_masks = {
        "handicap": frame.handicap.eq(1), "conditions": frame.handicap.eq(0),
        "at_least_3_starts_in_current_code": frame.horse_rt_runs.ge(3),
        "less_than_3_starts_in_current_code": frame.horse_rt_runs.lt(3),
        "source_reports_more_recent_run": frame.days_since_last_run.gt(0) & frame.days_since_last_run.lt(frame.days_since_last_race-2),
        "last_run_nonfinish": frame.horse_last_run_non_finish.eq(1),
        "layoff_180_plus": frame.days_since_last_race.ge(180),
        "compatible_mark_drop_at_least_3": frame.or_change.le(-3),
    }
    context_summary = []
    for label, mask in context_masks.items():
        sub = frame.loc[established & mask]
        context_summary.append({"context": label, "runners": len(sub), "winners": int(sub.won.sum()),
            "log_loss_gap_sum": float(sub.race_log_loss_gap_contribution.sum()),
            "brier_gap_sum": float(sub.runner_brier_gap.sum()),
            "mean_candidate": float(sub.candidate_probability.mean()), "mean_market": float(sub.market_probability.mean())})
    pd.DataFrame(context_summary).to_csv(OUTPUT / "established_context_breakdown.csv", index=False)
    probability_bins = pd.cut(frame.candidate_probability, [0, .05, .15, .3, .5, 1], include_lowest=True)
    reliability = frame.loc[established].groupby(probability_bins.loc[established], observed=True).agg(
        runners=("won", "size"), wins=("won", "sum"), win_rate=("won", "mean"),
        mean_candidate=("candidate_probability", "mean"), mean_market=("market_probability", "mean"))
    reliability.to_csv(OUTPUT / "established_probability_bins.csv")
    runner_summary = []
    for lo, hi, label in [(0, 2, "0-2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, np.inf, "20+")]:
        sub = frame.loc[frame.horse_prev_races.between(lo, hi)]
        runner_summary.append({"prior_local_starts": label, "runners": len(sub), "winners": int(sub.won.sum()),
            "race_log_loss_gap_sum": float(sub.race_log_loss_gap_contribution.sum()),
            "runner_brier_gap_sum": float(sub.runner_brier_gap.sum()),
            "candidate_mean_probability": float(sub.candidate_probability.mean()),
            "market_mean_probability": float(sub.market_probability.mean()), "win_rate": float(sub.won.mean())})
    # Inspect both directions, including occasions when the model was right.
    groups = {
        "missed_winners": frame.loc[established & frame.won.eq(1)].sort_values("log_market_over_model", ascending=False),
        "overrated_losers": frame.loc[established & frame.won.eq(0)].sort_values("runner_brier_gap", ascending=False),
        "model_advantage_winners": frame.loc[established & frame.won.eq(1)].sort_values("log_market_over_model"),
        "market_overrated_losers": frame.loc[established & frame.won.eq(0)].sort_values("runner_brier_gap"),
    }
    picked = []
    for label, rows in groups.items():
        rows.to_csv(OUTPUT / f"{label}.csv", index=False)
        selected = rows.head(15).copy()
        selected.insert(0, "case_group", label)
        picked.append(selected)
    cases = pd.concat(picked, ignore_index=True)
    cases.to_csv(OUTPUT / "selected_cases.csv", index=False)
    frame.loc[frame.race_id.isin(cases.race_id)].to_parquet(OUTPUT / "case_fields.parquet", index=False)
    history_cols = list(dict.fromkeys(KEY + ["race_date", "won", "finish_position", "lengths_behind", "num_runners"] + CONTEXT))
    history = pd.read_parquet(STORE / "featured.parquet", columns=[c for c in history_cols if c in schema.names],
                             filters=[("horse_id", "in", cases.horse_id.unique().tolist()), ("race_date", "<=", frame.race_date.max())])
    histories = []
    for row in cases.itertuples():
        sub = history.loc[history.horse_id.eq(row.horse_id) & history.race_date.le(row.race_date)].sort_values(["race_date", "race_id"]).tail(11).copy()
        sub.insert(0, "case_group", row.case_group)
        sub.insert(1, "case_race_id", row.race_id)
        histories.append(sub)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT / "case_histories.csv", index=False)
    frame.to_parquet(OUTPUT / "paired_context.parquet", index=False)
    total_gap = float(frame.race_log_loss_gap_contribution.sum())
    report = {
        "scope": "Descriptive Q1 2026 development analysis using saved026 forecasts trained through 2025-12-22; no new fitting or holdout claim",
        "established_definition": "At least five observed local starts before this race; all-field thresholds also shown",
        "market_scope": "Unchanged historical market benchmark016; not a newly observed live quote",
        "repaired_benchmark_descriptors": metadata_repairs,
        "cohorts": cohorts, "runner_attribution": runner_summary,
        "overlapping_established_contexts": context_summary,
        "established_winner_share_of_net_log_loss_gap": float(frame.loc[established, "race_log_loss_gap_contribution"].sum()/total_gap),
        "context_meta": json.loads((STORE / "meta.json").read_text()),
        "input_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in [source, SOURCE / "manifest.json", STORE / "meta.json"]},
        "source_sha256": file_sha256(Path(__file__)), "training_performed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    display = ["case_group", "race_date", "race_id", "horse_name", "horse_prev_races", "candidate_probability", "market_probability", "race_type", "race_class", "track", "horse_last_run_non_finish", "days_since_last_race"]
    print(cases.groupby("case_group", sort=False).head(5)[display].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
