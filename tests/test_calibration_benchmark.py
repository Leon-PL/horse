"""Chronological boundaries and numerical invariants of the audit benchmark."""
import numpy as np
import pandas as pd

from scripts.benchmark_calibration import (
    fit_temperature, paired_interval, probability_metrics, race_temperature, window_masks,
)


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
