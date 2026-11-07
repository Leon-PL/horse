"""Tests for add_hidden_form_features (src/features/horse.py).

Covers the leak-safe 'prior form not in our DB' signal (foreign raiders /
point-to-point graduates) built from pre-race fields only. ``has_hidden_form``
and ``experience_with_hidden`` were removed on 2026-11-07 (research 065): no
feature set used them.
"""

import pandas as pd

from src.features.horse import add_hidden_form_features


def _frame():
    return pd.DataFrame({
        # form is the PRE-RACE record (does not include today's result).
        "form": ["1", "", "1234", "1-2/3", "123456", "123456"],
        "days_since_last_run": [0, 0, 150, 40, 0, 200],
        "horse_prev_races": [0, 0, 0, 0, 10, 1],
        # a retroactive lifetime total — present but must be IGNORED (would leak).
        "horse_runs": [1, 0, 4, 3, 54, 30],
    })


def test_true_debutant_has_no_hidden_runs():
    out = add_hidden_form_features(_frame())
    # row 0: single-char placeholder, no gap -> true debut, not hidden.
    assert out.loc[0, "hidden_form_runs"] == 0
    # row 1: empty form, no gap -> true debut.
    assert out.loc[1, "hidden_form_runs"] == 0


def test_foreign_raider_counts_unseen_runs():
    out = add_hidden_form_features(_frame())
    # row 2: no DB runs but 4 form chars + 150-day gap -> 4 hidden runs.
    assert out.loc[2, "prior_form_runs"] == 4
    assert out.loc[2, "hidden_form_runs"] == 4
    # row 3: separators stripped -> 3 runs; 40-day gap -> all hidden.
    assert out.loc[3, "prior_form_runs"] == 3
    assert out.loc[3, "hidden_form_runs"] == 3


def test_well_tracked_horse_not_hidden():
    out = add_hidden_form_features(_frame())
    # row 4: 10 DB runs already -> no excess form.
    assert out.loc[4, "hidden_form_runs"] == 0


def test_lightly_raced_but_experienced():
    out = add_hidden_form_features(_frame())
    # row 5: 1 DB run but 6 form chars -> 5 of them are unseen prior runs.
    assert out.loc[5, "hidden_form_runs"] == 5


def test_lifetime_horse_runs_is_ignored():
    # Mutating the leaky retroactive total must NOT change any output.
    base = add_hidden_form_features(_frame())
    alt = _frame()
    alt["horse_runs"] = [99, 99, 99, 99, 99, 99]
    alt = add_hidden_form_features(alt)
    for c in ["hidden_form_runs", "prior_form_runs"]:
        assert (base[c].to_numpy() == alt[c].to_numpy()).all()
