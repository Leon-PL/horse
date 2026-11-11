import numpy as np
import pandas as pd

from research.error_anatomy.sale_inputs import sale_inputs
from research.error_anatomy.soft_corrections import soft_boost
from research.meta_learner import screening as s
from research.meta_learner.test_screening import synthetic


def test_a_sale_is_used_only_after_it_happened_and_never_from_training_sales():
    sales = pd.DataFrame({
        "sale": ["OC1", "GBU", "JUL", "OC1"], "sale_year": [2023, 2024, 2024, 2021],
        "sire": ["Alpha", "Beta", "Gamma", "Alpha"], "dam": ["One", "Two", "Three", "Old"],
        "price_guineas": [100000., 50000., 80000., 20000.]})
    runs = pd.DataFrame({
        "race_id": ["a", "a", "a", "b", "b"], "horse_id": [1, 2, 3, 2, 3],
        "race_date": pd.to_datetime(["2024-04-10"] * 3 + ["2024-07-01"] * 2),
        "sire": ["Alpha", "Beta", "Gamma", "Beta", "Gamma"], "dam": ["One", "Two", "Three", "Two", "Three"],
        "foaled": ["2022-03-01", "2022-02-01", "2021-02-01", "2022-02-01", "2021-02-01"]})
    out = sale_inputs(runs, sales).set_index(["race_id", "horse_id"])
    assert np.isfinite(out.loc[("a", 1), "sale_log_price"])        # yearling sale the autumn before
    assert np.isnan(out.loc[("a", 2), "sale_log_price"])           # breeze-up sale later that spring
    assert np.isfinite(out.loc[("b", 2), "sale_log_price"])        # the same horse after the sale
    assert out.sale_log_price.xs(3, level="horse_id").isna().all()  # horses-in-training sale: never used
    # The sire's record uses earlier sale years only: Alpha's 2021 lot counts in 2024, its 2023 lot too,
    # and Beta's 2024 sale does not count for a 2024 race.
    assert out.loc[("a", 1), "sire_sale_lots"] == 2
    assert out.loc[("b", 2), "sire_sale_lots"] == 0


def test_soft_boost_learns_from_a_soft_target_without_seeing_results():
    frame = synthetic(effect=0.6)
    bounds = s.bounds_of(frame.race_id.to_numpy())
    truth = np.exp(s.race_log_softmax(np.log(frame.p_single.to_numpy()) + 0.6 * frame.x.to_numpy(), bounds))
    frame["y_soft"] = truth
    frame["won"] = 0  # results hidden: only the soft target can teach the correction
    fit = soft_boost(["x"], target="y_soft", rounds=60, threads=1, seeds=(7,),
                     params={"learning_rate": .1, "num_leaves": 3, "min_data_in_leaf": 50, "lambda_l2": 1.})
    correction, _ = s.walk_forward(frame, fit)
    scored = frame.quarter.isin(s.SCORED).to_numpy()
    assert np.corrcoef(correction[scored], frame.x.to_numpy()[scored])[0, 1] > 0.9
