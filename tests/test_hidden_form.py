"""Tests for add_hidden_form_features (src/feature_engineer.py).

Covers the leak-safe 'prior form not in our DB' signal (foreign raiders /
point-to-point graduates) built from pre-race fields only.
"""

import numpy as np
import pandas as pd

from src.feature_engineer import add_hidden_form_features


def _frame():
    return pd.DataFrame({
        # form is the PRE-RACE record (does not include today's result).
        "form": ["1", "", "1234", "1-2/3", "123456", "123456"],
        "days_since_last_run": [0, 0, 150, 40, 0, 200],
        "horse_prev_races": [0, 0, 0, 0, 10, 1],
        # a retroactive lifetime total — present but must be IGNORED (would leak).
        "horse_runs": [1, 0, 4, 3, 54, 30],
    })


def test_true_debutant_not_flagged():
    out = add_hidden_form_features(_frame())
    # row 0: single-char placeholder, no gap -> true debut, not hidden.
    assert out.loc[0, "has_hidden_form"] == 0
    assert out.loc[0, "hidden_form_runs"] == 0
    # row 1: empty form, no gap -> true debut.
    assert out.loc[1, "has_hidden_form"] == 0


def test_foreign_raider_flagged():
    out = add_hidden_form_features(_frame())
    # row 2: no DB runs but 4 form chars + 150-day gap -> hidden form.
    assert out.loc[2, "has_hidden_form"] == 1
    assert out.loc[2, "prior_form_runs"] == 4
    assert out.loc[2, "hidden_form_runs"] == 4
    assert out.loc[2, "experience_with_hidden"] == 4
    # row 3: separators stripped -> 3 runs; 40-day gap -> hidden form.
    assert out.loc[3, "prior_form_runs"] == 3
    assert out.loc[3, "has_hidden_form"] == 1


def test_well_tracked_horse_not_hidden():
    out = add_hidden_form_features(_frame())
    # row 4: 10 DB runs already -> not a hidden debutant; no excess form.
    assert out.loc[4, "has_hidden_form"] == 0
    assert out.loc[4, "hidden_form_runs"] == 0


def test_lightly_raced_but_experienced():
    out = add_hidden_form_features(_frame())
    # row 5: 1 DB run but 6 form chars -> 5 of them are unseen prior runs.
    assert out.loc[5, "has_hidden_form"] == 0          # not a *debutant*
    assert out.loc[5, "hidden_form_runs"] == 5
    assert out.loc[5, "experience_with_hidden"] == 6


def test_lifetime_horse_runs_is_ignored():
    # Mutating the leaky retroactive total must NOT change any output.
    base = add_hidden_form_features(_frame())
    alt = _frame(); alt["horse_runs"] = [99, 99, 99, 99, 99, 99]
    alt = add_hidden_form_features(alt)
    for c in ["has_hidden_form", "hidden_form_runs", "prior_form_runs", "experience_with_hidden"]:
        assert (base[c].to_numpy() == alt[c].to_numpy()).all()


def test_hidden_form_figures_read_only_the_missing_runs():
    frame = _frame()
    frame["race_id"] = ["a", "a", "a", "b", "b", "b"]
    out = add_hidden_form_features(frame)
    # Row 2: four runs, none in the database: figures 1, 2, 3, 4.
    row = out.loc[2]
    assert (row.hidden_form_wins, row.hidden_form_places, row.hidden_form_fails) == (1, 3, 0)
    assert (row.hidden_form_best, row.hidden_form_mean, row.hidden_form_last) == (1, 2.5, 4)
    # Row 5: six figures, one run in the database, so the latest figure (6) is not a missing run.
    assert out.loc[5, "hidden_form_runs"] == 5 and out.loc[5, "hidden_form_last"] == 5
    # No missing runs: every figure is absent, not zero.
    for name in ("wins", "places", "fails", "best", "mean", "last"):
        assert np.isnan(out.loc[0, f"hidden_form_{name}"]) and np.isnan(out.loc[4, f"hidden_form_{name}"])
    # Against the field: rows 0-2 share a race and only row 2 has a win among missing runs.
    assert np.isclose(out.loc[2, "hidden_form_wins_vs_field"], 1 - 1 / 3)
    assert np.isclose(out.loc[0, "hidden_form_wins_vs_field"], -1 / 3)


def test_hidden_form_letters_and_zero():
    from src.feature_engineer import hidden_form_figures
    out = hidden_form_figures(np.array(["0P", "F1-2", None], dtype=object), np.array([2, 2, 3]))
    assert out[0].tolist()[:4] == [0.0, 0.0, 1.0, 10.0] and out[0, 5] == 12.0
    assert out[1, 0] == 1.0 and out[1, 2] == 1.0 and out[1, 5] == 1.0
    assert np.isnan(out[2]).all()
