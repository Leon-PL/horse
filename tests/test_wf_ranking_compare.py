"""Ranking metrics must not reward row order: rows may sit in finish order."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "wf_ranking_compare", ROOT / "scripts" / "wf_ranking_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tied_favourites_share_top1_credit_whatever_the_row_order(mod):
    # Winner listed first (finish order) and tied with another runner on top.
    test = pd.DataFrame({"race_id": ["r"] * 4, "won": [1, 0, 0, 0]})
    probs = np.array([0.3, 0.3, 0.2, 0.2])
    out = mod._per_race(test, probs)
    assert out["top1"].iloc[0] == pytest.approx(0.5)
    flipped = mod._per_race(test.iloc[::-1].reset_index(drop=True), probs[::-1])
    assert flipped["top1"].iloc[0] == pytest.approx(0.5)


def test_clear_favourite_and_log_loss(mod):
    test = pd.DataFrame({"race_id": ["a", "a", "b", "b"], "won": [0, 1, 1, 0]})
    out = mod._per_race(test, np.array([2.0, 6.0, 1.0, 3.0]))  # unnormalised
    assert list(out["top1"]) == [1.0, 0.0]
    assert out["ll"].tolist() == pytest.approx([-np.log(0.75), -np.log(0.25)])
