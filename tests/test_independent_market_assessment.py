import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.assess_independent_market import (
    fresh_field_mask, run, verify_complete_experiment, verify_snapshot_manifest)
from scripts.benchmark_independent_features import score_pair
from scripts.replay_market_snapshots import digest
from src.betfair import bf_horse_key, bf_track_key


def frozen_pair():
    return pd.DataFrame({
        "race_id": np.repeat([1, 2, 3], 2), "won": [1, 0] * 3,
        "race_date": np.repeat(["2025-07-01", "2025-10-01", "2026-01-01"], 2),
        "baseline_probability": [.5, .5] * 3, "candidate_probability": [.6, .4] * 3})


def test_market_context_requires_finished_periods_and_unchanged_primary_forecasts():
    frame = frozen_pair()
    periods = ["2025Q3", "2025Q4", "2026Q1"]
    primary = {"folds": [{"period": period} for period in periods], "pooled": score_pair(frame)}
    specification = {"periods": periods}
    verify_complete_experiment(frame, primary, specification)
    with pytest.raises(ValueError, match="Complete all"):
        verify_complete_experiment(frame, {**primary, "folds": primary["folds"][:2]}, specification)
    changed = frame.copy()
    changed.loc[:1, "candidate_probability"] = [.7, .3]
    with pytest.raises(ValueError, match="reproduce"):
        verify_complete_experiment(changed, primary, specification)


def test_freshness_excludes_whole_fields_and_rejects_future_observations():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3], "decision_ms": [1_000_000] * 6,
        "price_changed_ms": [700_000, 700_001, 699_999, 700_000, 900_000, 900_000],
        "observed_ms": [910_000, 910_000, 910_000, 910_000, 1_000_001, 910_000]})
    mask = fresh_field_mask(frame, {"every_runner_update_max_age_seconds": 300,
                                    "market_observation_max_age_seconds": 90})
    assert mask.tolist() == [True, True, False, False, False, False]


def test_snapshot_horizon_and_fingerprint_must_match_the_registration():
    manifest = {"seconds_before": 300, "output_sha256": "frozen"}
    verify_snapshot_manifest(manifest, {"horizon_seconds": 300}, "frozen")
    with pytest.raises(ValueError, match="horizon"):
        verify_snapshot_manifest(manifest, {"horizon_seconds": 600}, "frozen")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_snapshot_manifest(manifest, {"horizon_seconds": 300}, "changed")


def test_completed_assessment_preserves_primary_decision_and_paired_probabilities(tmp_path):
    frame = frozen_pair().assign(horse_name=["Horse A", "Horse B"] * 3,
                                finish_position=[1, 2] * 3, num_runners=2,
                                track="Ascot", off_time="14:00")
    periods = ["2025Q3", "2025Q4", "2026Q1"]
    specification = {"periods": periods, "horizon_seconds": 300,
                     "scope": "Retrospective test reference", "snapshot_start": "2025-07-01",
                     "snapshot_end": "2026-03-31", "freshness_diagnostic": {
                         "every_runner_update_max_age_seconds": 300,
                         "market_observation_max_age_seconds": 90}}
    registration = tmp_path / "market_context_registration.json"
    registration.write_text(json.dumps(specification))
    os.utime(registration, (1, 1))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b'{\r\n  "experiment": "native Windows manifest"\r\n}')
    primary = {"folds": [{"period": period} for period in periods], "pooled": score_pair(frame),
               "manifest_sha256": hashlib.sha256(manifest_path.read_text().encode()).hexdigest(),
               "candidate_advances": False}
    primary_path = tmp_path / "report.json"
    primary_path.write_text(json.dumps(primary))
    primary_hash = digest(primary_path)
    frame.to_parquet(tmp_path / "paired_predictions.parquet", index=False)
    snapshot = pd.DataFrame({
        "race_date": frame.race_date, "track_key": frame.track.map(bf_track_key),
        "off_key": "14:00", "horse_key": frame.horse_name.map(bf_horse_key),
        "market_id": frame.race_id.astype(str), "selection_id": range(6),
        "market_probability": [.7, .3] * 3, "active_runners": 2,
        "decision_ms": 1_000_000, "observed_ms": 999_000, "price_changed_ms": 999_000,
        "won": frame.won, "withdrawn_after_decision": False})
    snapshot_path = tmp_path / "snapshots.parquet"
    snapshot.to_parquet(snapshot_path, index=False)
    snapshot_path.with_suffix(".json").write_text(json.dumps({
        "seconds_before": 300, "output_sha256": digest(snapshot_path)}))
    result = run(SimpleNamespace(experiment=str(tmp_path), snapshots=str(snapshot_path), seed=42))
    assert result["pooled"]["races"] == 3
    assert result["freshness"]["races"] == 3
    assert result["primary_candidate_advances"] is False
    assert digest(primary_path) == primary_hash
    paired = pd.read_parquet(tmp_path / "market_context_predictions.parquet")
    np.testing.assert_array_equal(paired.candidate_probability, frame.candidate_probability)
