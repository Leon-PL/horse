"""wf_feature_family_lift: the family is selected by prefix only."""

import os
import sys

import numpy as np
import pandas as pd

import config

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import wf_feature_family_lift as lift  # noqa: E402
import wf_ranking_compare as wrc  # noqa: E402


def test_extra_file_labels_never_become_features(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_ALLOWLIST", None)
    base = pd.DataFrame({
        "race_id": ["1", "1", "2", "2"], "horse_name": ["A", "B", "A", "C"],
        "race_date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
        "won": [1, 0, 0, 1], "horse_elo": [1500.0, 1490.0, 1510.0, 1480.0],
        "ltp_300s": [2.0, 3.0, 2.5, 4.0], "mkt": [0.6, 0.4, 0.6, 0.4], "scorable": [True] * 4,
    })
    monkeypatch.setattr(wrc, "load_frame", lambda: base.copy())
    extra = pd.DataFrame({
        "race_id": ["1", "1", "2", "2"], "horse_name": ["A", "B", "A", "C"],
        "cp_signal": [0.1, -0.1, 0.2, -0.2], "won": [1, 0, 0, 1], "num_runners": [2, 2, 2, 2],
    })
    path = tmp_path / "extra.parquet"
    extra.to_parquet(path, index=False)
    df, base_feats, fam = lift.load("cp_", str(path))
    assert fam == ["cp_signal"]
    assert "won" not in base_feats and "won" not in fam
    assert len(df) == 4
    np.testing.assert_allclose(df.sort_values(["race_id", "horse_name"])["cp_signal"], [0.1, -0.1, 0.2, -0.2])
