"""The store comparison has to fail on what it claims to check."""
import pickle
import sys

import pandas as pd
import pytest

from scripts import compare_feature_stores as compare


def _store(root, value=1.0, carried=1.0):
    root.mkdir()
    frame = pd.DataFrame({"race_id": [1, 2], "horse_id": [1, 2], "horse_name": ["a", "b"],
                          "race_date": ["2024-01-01", "2024-01-02"], "value": [value, 2.0]})
    frame.to_parquet(root / "featured.parquet", index=False)
    frame.to_parquet(root / "processed.parquet", index=False)
    with open(root / "ratings_state.pkl", "wb") as fh:
        pickle.dump({"elo": None}, fh)
    with open(root / "counts_state.pkl", "wb") as fh:
        pickle.dump({"tw::trainer::_np_cnt::365": {"a": [(1, carried)]}}, fh)
    return str(root)


def _exit_code(monkeypatch, left, right, report):
    monkeypatch.setattr(sys, "argv", ["compare", left, right, "--report", str(report)])
    with pytest.raises(SystemExit) as stopped:
        compare.main()
    return stopped.value.code


def test_stores_that_match_pass(tmp_path, monkeypatch):
    left, right = _store(tmp_path / "left"), _store(tmp_path / "right")
    assert _exit_code(monkeypatch, left, right, tmp_path / "report.json") == 0
    assert (tmp_path / "report.json").exists()


def test_a_changed_value_fails(tmp_path, monkeypatch):
    left, right = _store(tmp_path / "left"), _store(tmp_path / "right", value=1.5)
    assert _exit_code(monkeypatch, left, right, tmp_path / "report.json") == 1


def test_a_changed_carried_value_fails_with_no_tolerance(tmp_path, monkeypatch, capsys):
    left, right = _store(tmp_path / "left"), _store(tmp_path / "right", carried=1.0 + 2e-16)
    assert _exit_code(monkeypatch, left, right, tmp_path / "report.json") == 1
    assert "_np_cnt" in capsys.readouterr().out
