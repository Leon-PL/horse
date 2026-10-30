"""Chronological boundaries and numerical invariants of the audit benchmark."""
import numpy as np
import pandas as pd
import pytest

import config

from scripts.benchmark_calibration import (
    build_parser, fit_temperature, paired_interval, probability_metrics,
    race_temperature, research_boundary, valid_race_mask, window_masks,
)


def _races():
    return pd.DataFrame({
        "race_id": ["good"] * 3 + ["suspect"] * 3,
        "race_date": ["2026-01-01"] * 6,
        "horse_name": ["A", "B", "C", "D", "E", "F"],
        "won": [1, 0, 0, 1, 0, 0],
        "finish_position": [1, 0, 99, 1, 2, 3],
        "num_runners": [3] * 6,
    })


@pytest.mark.parametrize("column,values", [
    ("won", [1, np.nan, 0]),
    ("won", [-1, 2, 0]),
    ("won", [0.5, 0.5, 0]),
    ("won", [1, 1, 0]),
    ("won", [0, 0, 0]),
    ("race_id", [None, None, None]),
    ("race_id", [" ", " ", " "]),
    ("horse_name", ["D", None, "F"]),
    ("horse_name", ["D", "", "F"]),
    ("horse_name", ["D", " d ", "F"]),
    ("race_date", ["2026-01-01", "2026-01-02", "2026-01-01"]),
    ("race_date", ["2026-01-01", None, "2026-01-01"]),
    ("num_runners", [4, 4, 4]),
    ("num_runners", [3, np.nan, 3]),
    ("finish_position", [2, 1, 3]),
    ("finish_position", [1, -1, 3]),
    ("finish_position", [1, 2.5, 3]),
    ("finish_position", [1, np.inf, 3]),
])
def test_malformed_runner_drops_the_whole_race(column, values):
    frame = _races()
    frame[column] = frame[column].astype(object)
    frame.loc[3:, column] = values
    assert valid_race_mask(frame).tolist() == [True] * 3 + [False] * 3


def test_dnf_losers_remain_and_runner_counts_are_optional():
    frame = _races().drop(columns="num_runners")
    assert valid_race_mask(frame).all()


def test_research_boundary_excludes_reserved_dates_and_purge_by_default(monkeypatch):
    monkeypatch.setattr(config, "GOLDEN_HOLDOUT_START", "2026-02-01")
    monkeypatch.setattr(config, "PURGE_DAYS", 7)
    frame = pd.DataFrame({"race_date": pd.to_datetime([
        "2026-01-24", "2026-01-25", "2026-01-31", "2026-02-01", "2026-02-02",
    ])})
    selected, metadata = research_boundary(frame)
    assert selected["race_date"].tolist() == [pd.Timestamp("2026-01-24")]
    assert metadata["excluded_by_research_boundary_rows"] == 4
    assert metadata["reserved_holdout_rows_in_benchmark"] == 0
    assert metadata["evaluation_scope"] == "development_dates_only"
    assert metadata["include_reserved_holdout"] is False

    retrospective, metadata = research_boundary(frame, include_reserved_holdout=True)
    pd.testing.assert_frame_equal(retrospective, frame)
    assert metadata["excluded_by_research_boundary_rows"] == 0
    assert metadata["reserved_holdout_rows_in_benchmark"] == 2
    assert metadata["evaluation_scope"] == "retrospective_including_reserved_dates"
    assert "prior research" in metadata["confirmation_limitation"]


def test_reserved_holdout_requires_explicit_cli_opt_in():
    parser = build_parser()
    assert parser.parse_args(["--features", "toy.parquet"]).include_reserved_holdout is False
    assert parser.parse_args(["--features", "toy.parquet", "--include-reserved-holdout"]).include_reserved_holdout is True


def test_windows_are_disjoint_date_strict_and_purged():
    dates = pd.date_range("2020-01-01", "2026-06-01")
    dates = pd.Series(np.repeat(dates.to_numpy(), 3))
    masks = window_masks(dates, pd.Timestamp("2026-01-01"))
    assert np.max(np.sum(list(masks.values()), axis=0)) == 1
    previous = None
    for mask in masks.values():
        block = dates[mask]
        assert mask.groupby(dates).nunique().max() == 1
        if previous is not None:
            assert block.min() - previous > pd.Timedelta(days=7)
        previous = block.max()


def test_temperature_improves_deliberately_overconfident_predictions():
    groups = np.full(100, 2)
    raw = np.tile([0.99, 0.01], 100)
    y = np.tile([1.0, 0.0], 100)
    y[-60:] = np.tile([0.0, 1.0], 30)
    temp = fit_temperature(raw, groups, y)
    calibrated = race_temperature(raw, groups, temp)
    before, _, _ = probability_metrics(raw, y, groups)
    after, _, _ = probability_metrics(calibrated, y, groups)
    assert temp > 1
    assert after["brier"] < before["brier"]
    assert after["race_log_loss"] < before["race_log_loss"]
    np.testing.assert_allclose(calibrated.reshape(-1, 2).sum(axis=1), 1)
    assert np.all(calibrated[::2] > calibrated[1::2])


def test_metrics_perfect_probabilities_and_zero_paired_difference():
    metrics, losses, brier = probability_metrics(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 1.0, 0.0]), np.array([2, 3]),
    )
    assert metrics["brier"] < 1e-20
    assert metrics["top1_accuracy"] == 1
    assert len(losses) == len(brier) == 2
    assert paired_interval(losses - losses) == {"mean": 0.0, "ci95": [0.0, 0.0]}
