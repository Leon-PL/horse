import numpy as np
import pandas as pd

from research.meta_learner.features_063 import context_form


def test_context_form_cannot_use_same_day_or_future_results():
    frame = pd.DataFrame([
        {"race_id": f"r{day}", "horse_id": horse, "race_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=day),
         "horse_elo": 1510. - 20 * horse + day * horse, "finish_position": horse + 1,
         "beaten_lengths": 4. * horse, "distance_furlongs": 8., "race_type": "Flat", "surface": "Turf",
         "weight_lbs": 125., "has_observed_weight": 1}
        for day in range(5) for horse in range(2)
    ])
    going = pd.Series("Good", index=frame.index)
    original = context_form(frame, going)
    changed = frame.copy()
    changed.loc[5, "beaten_lengths"] = 12.
    revised = context_form(changed, going)
    pd.testing.assert_frame_equal(original.iloc[:6], revised.iloc[:6])
    assert np.isfinite(original.x063_fc_ctx_vs_field.iloc[2:]).all()
    assert not np.allclose(original.x063_fc_ctx_vs_field.iloc[6:], revised.x063_fc_ctx_vs_field.iloc[6:])
