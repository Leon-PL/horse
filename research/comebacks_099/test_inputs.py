import numpy as np
import pandas as pd

from research.comebacks_099 import inputs


def test_reappearance_record_uses_earlier_runs_only(monkeypatch):
    store = pd.DataFrame({
        "race_id": ["1", "2", "3", "4", "5"], "horse_id": [7] * 5,
        "race_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2023-06-20", "2024-01-01", "2024-01-20"]),
        "finish_position": [1, 8, 1, 9, 2], "num_runners": [9] * 5})
    monkeypatch.setattr(inputs.sc, "read_store", lambda columns: store.copy())
    frame = store[["race_id", "horse_id"]].assign(
        won=[1, 0, 1, 0, 0], p_single=.2, p_market=.1, quarter="2023Q1")
    out, _ = inputs.comeback_inputs(frame)
    out = out.set_index("race_id")
    # Run 2 is the first comeback: no earlier comeback record.
    assert out.loc["2", "cb_today"] == 1 and out.loc["2", "cb_n"] == 0 and np.isnan(out.loc["2", "cb_pct"])
    # Run 4 is the second comeback: it sees run 2 (8th of 9, pct 0.125) and run 3 (next run, won).
    assert out.loc["4", "cb_n"] == 1
    assert np.isclose(out.loc["4", "cb_pct"], (.125 + 2 * .5) / 3)
    assert np.isclose(out.loc["4", "cb_pct_next"], (1 + 2 * .5) / 3)
    assert np.isclose(out.loc["4", "cb_resid"], -.2 / 3)
    # Run 5 is not a comeback; its own today-interactions are zero.
    assert out.loc["5", "cb_today"] == 0 and out.loc["5", "cb_resid_today"] == 0
