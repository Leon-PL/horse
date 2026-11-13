"""Regression checks for source interpretation and order-invariant histories."""
import numpy as np
import pandas as pd
import pytest

from src.data_processor import clean_data
from src.feature_engineer import add_speed_figure_features
from src.model import get_feature_columns
from src.result_margins import cumulative_beaten_lengths


def example():
    return pd.DataFrame({"race_id": [1, 1, 1, 2, 2, 2],
        "race_date": pd.to_datetime(["2025-01-01"]*3 + ["2025-01-02"]*3),
        "horse_name": ["A", "B", "C"]*2, "horse_id": [1, 2, 3]*2,
        "finish_position": [1, 2, 3]*2, "lengths_behind": [0., .3, .2, 0., .5, 1.],
        "distance_furlongs": 8., "race_class": "Class 4"})


def test_processor_preserves_raw_gaps_and_exposes_correct_totals():
    raw = example()
    processed = clean_data(raw)
    np.testing.assert_allclose(processed.beaten_lengths, [0, .3, .5, 0, .5, 1.5])
    np.testing.assert_array_equal(processed.lengths_behind, raw.lengths_behind)
    pd.testing.assert_series_equal(clean_data(processed).beaten_lengths, processed.beaten_lengths)


def test_current_cumulative_margin_is_never_an_automatic_model_input(monkeypatch):
    import config
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    f = clean_data(example()).assign(safe_lagged_feature=1.)
    assert "beaten_lengths" not in get_feature_columns(f)
    assert "safe_lagged_feature" in get_feature_columns(f)


def test_class_scale_is_identical_for_all_finishers_and_row_permutations():
    f = example()
    extra = f.iloc[3:].copy().assign(race_id=3, race_date=pd.Timestamp("2025-01-03"))
    f = pd.concat([f, extra], ignore_index=True)
    f["beaten_lengths"] = cumulative_beaten_lengths(f)
    first = add_speed_figure_features(f.copy()).set_index(["race_id", "horse_id"])
    shuffled = pd.concat([g.sample(frac=1, random_state=7) for _, g in f.groupby("race_id")])
    second = add_speed_figure_features(shuffled).set_index(["race_id", "horse_id"]).reindex(first.index)
    cols = [c for c in first if c.startswith("speed_")]
    for c in cols:
        np.testing.assert_allclose(first[c], second[c], equal_nan=True, atol=1e-10)


def test_winners_are_credited_their_winning_distance():
    import numpy as np
    import pandas as pd
    from src.feature_engineer import WINNER_MARGIN_CAP
    rows = []
    # Horse 1 wins by 2 lengths, then by 50 (capped); horse 2 is second both times.
    for day, (gap, race) in enumerate([(2., "a"), (50., "b"), (1., "c")]):
        for horse, position, beaten in [(1, 1, 0.), (2, 2, gap), (3, 3, gap + 1.)]:
            rows.append({"race_id": race, "horse_id": horse, "race_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * day),
                         "finish_position": position, "lengths_behind": beaten, "beaten_lengths": beaten,
                         "distance_furlongs": 8., "race_class": 4, "region": "UK"})
    out = add_speed_figure_features(pd.DataFrame(rows))
    scale = 8. ** .6
    winner = out[out.horse_id.eq(1)].sort_values("race_date")
    assert np.isnan(winner.speed_fig_best_5.iloc[0])
    assert winner.speed_fig_best_5.iloc[1] == pytest.approx(-2. / scale)
    assert winner.speed_fig_best_5.iloc[2] == pytest.approx(-WINNER_MARGIN_CAP / scale)
    second = out[out.horse_id.eq(2)].sort_values("race_date")
    assert second.speed_fig_best_5.iloc[2] == pytest.approx(2. / scale)
