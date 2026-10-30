"""Descriptive scoring of immutable forward forecasts against validated results."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

from src.forecast_evaluation import probability_metrics, paired_interval, valid_race_mask
from src.forward_store import ROOT, timestamp, utcnow, publish_json
from src.deployment_validation import file_sha256
from src.utils import normalise_horse_key


def assess_records(study, results, *, now=None, withdrawals=None):
    """Keep complete identical fields; pending/changed fields cannot alter p."""
    for name in ("src/utils.py", "src/forecast_evaluation.py"):
        expected = study.protocol.get("source_sha256", {}).get(name)
        if expected and file_sha256(ROOT / name) != expected:
            raise ValueError("Frozen matching/scoring implementation changed")
    coverage, paired, exclusions = Counter(), [], []
    labels = results.copy()
    labels["race_id"] = labels.race_id.astype(str)
    labels["horse_key"] = labels.horse_name.map(normalise_horse_key)
    valid = valid_race_mask(labels)
    labels["valid_field"] = valid
    selected = list(study.selected_records())
    for record in selected:
        frame = pd.read_parquet(study.path / "records" / record["capture_id"] / "forecasts.parquet")
        if frame.race_id.nunique() != 1 or frame.race_id.isna().any():
            raise ValueError("Recorded paired race identity is incomplete")
        rid = str(frame.race_id.iloc[0])
        outcome = labels.loc[labels.race_id.eq(rid)]
        reason = None
        if outcome.empty:
            reason = "results_pending"
        elif not outcome.valid_field.all():
            reason = "results_incomplete_or_ambiguous"
        elif (set(outcome.horse_key) != set(frame.horse_key)
                or len(outcome) != len(frame)
                or outcome.race_date.astype(str).str[:10].nunique() != 1
                or not outcome.race_date.astype(str).str[:10].eq(timestamp(record["first_start"]).date().isoformat()).all()):
            reason = "changed_field_or_identity"
        if reason:
            coverage[reason] += 1
            exclusions.append({"event_id": record["event_id"], "capture_id": record["capture_id"],
                "reason": reason, "recorded_horses": sorted(frame.horse_key),
                "result_horses": sorted(outcome.horse_key)})
            continue
        joined = frame.merge(outcome[["horse_key", "won", "finish_position", "race_date"]],
                             on="horse_key", how="outer", validate="one_to_one")
        joined["event_id"] = record["event_id"]
        joined["capture_id"] = record["capture_id"]
        paired.append(joined)
        coverage["scored_races"] += 1
    all_records = list((study.path / "records").glob("*/capture.json"))
    candidates = []
    moment = timestamp(now or utcnow())
    for path in (study.path / "events").glob("*.json"):
        value = json.loads(path.read_text())
        if timestamp(value["first_start"]).date().isoformat() >= study.protocol["first_eligible_date"]:
            candidates.append(value)
    due = [v for v in candidates if timestamp(v["first_start"]) <= moment]
    report = {"study_protocol_sha256": study.protocol_sha256, "assessed_at": moment.isoformat(),
        "scope": "Descriptive forward baseline monitoring; no registered confirmation sample size or automatic parity decision",
        "confirmation_assessed": False, "reference": study.protocol["reference"],
        "coverage": {"attempts": len(all_records), "selected_complete_paired_forecasts": len(selected),
                     "discovered_events_after_registered_start": len(candidates),
                     "started_discovered_events": len(due),
                     "started_events_without_eligible_forecast": sum(study.selected(v["event_id"]) is None for v in due),
                     **coverage}, "exclusions": exclusions}
    rejected = Counter()
    for path in all_records:
        rejected.update(json.loads(path.read_text()).get("reasons", []))
    report["coverage"]["attempt_rejection_reasons"] = dict(rejected)
    from src.issued_evaluation import assess_issued, tail_diagnostics, place_comparison
    report["as_issued"] = assess_issued(study, results, withdrawals)
    if not paired:
        return report, pd.DataFrame()
    frame = pd.concat(paired, ignore_index=True).sort_values(["race_date", "event_id", "horse_key"], kind="stable")
    groups = frame.groupby("event_id", sort=False).size().to_numpy()
    metrics = {}
    losses = {}
    for role in ("model", "market"):
        scores, race_ll, _ = probability_metrics(frame[f"{role}_probability"], frame.won, groups)
        metrics[role], losses[role] = scores, race_ll
    delta = losses["model"]-losses["market"]
    days = frame.groupby("event_id", sort=False).race_date.first().to_numpy()
    report.update({"races": len(groups), "runners": len(frame), "metrics": metrics,
        "tail_risk":{role:tail_diagnostics(frame[f"{role}_probability"],frame.won) for role in ("model","market")},
        "place_comparison":place_comparison(frame),
        "race_log_loss_difference_model_minus_market": {
            "mean": float(delta.mean()),
            "race_interval": paired_interval(delta) if len(groups) >= 2 else None,
            "date_interval": paired_interval(delta, blocks=days) if len(set(days)) >= 2 else None},
        "label_source": "Validated Sporting Life results in the local results database; no market-price substitution"})
    return report, frame


def load_results_readonly(path, *, race_ids=None):
    required = "race_id,horse_name,race_date,finish_position,won,num_runners".split(",")
    with sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True) as connection:
        available = {row[1] for row in connection.execute("PRAGMA table_info(results)")}
        columns = required+[c for c in ("going", "jockey", "weight_lbs", "draw", "official_rating") if c in available]
        base = f"SELECT {','.join(columns)} FROM results"
        if race_ids is None:
            return pd.read_sql_query(base,connection)
        ids = sorted({str(value) for value in race_ids})
        if not ids:
            return pd.read_sql_query(base+" WHERE 0=1",connection)
        parts = []
        for start in range(0,len(ids),900):
            batch = ids[start:start+900]
            parts.append(pd.read_sql_query(base+" WHERE race_id IN ("+",".join("?" for _ in batch)+")",
                                            connection,params=batch))
        return pd.concat(parts,ignore_index=True)


def save_assessment(study, results_db):
    from uuid import uuid4
    # Hourly assessment must not duplicate the complete750k-row historical
    # database on every tick. Preserve all outcomes for issued race IDs only.
    race_ids = set()
    for record in study.selected_records():
        frame = pd.read_parquet(study.path/"records"/record["capture_id"]/"forecasts.parquet",columns=["race_id"])
        race_ids.update(frame.race_id.astype(str))
    results = load_results_readonly(results_db,race_ids=race_ids)
    withdrawal_path = study.path/"outcomes"/"withdrawals.parquet"
    withdrawals = pd.read_parquet(withdrawal_path) if withdrawal_path.exists() else None
    report, paired = assess_records(study, results, withdrawals=withdrawals)
    output = study.path / "assessments" / (utcnow().strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    results.to_parquet(output / "results_snapshot.parquet", index=False)
    if not paired.empty:
        paired.to_parquet(output / "paired_forecasts.parquet", index=False)
    report["results_snapshot_sha256"] = file_sha256(output / "results_snapshot.parquet")
    if withdrawals is not None:
        withdrawals.to_parquet(output/"withdrawals_snapshot.parquet", index=False)
        report["withdrawals_snapshot_sha256"] = file_sha256(output/"withdrawals_snapshot.parquet")
    report["assessment_source_sha256"] = {name: file_sha256(ROOT / name) for name in (
        "src/forward_assessment.py", "src/issued_evaluation.py", "src/forecast_evaluation.py", "src/utils.py")}
    publish_json(output / "report.json", report)
    return report, output
