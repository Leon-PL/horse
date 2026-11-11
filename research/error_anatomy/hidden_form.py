"""What the racecard's form figures say about runs the database does not hold.

6.4% of runners (16.6% of hurdlers) show more runs in their card form string
than the database has for them: point-to-points, runs abroad, runs before
September 2020. The store gives the model a count (``hidden_form_runs``) and
nothing else, though the card shows how the horse finished in each. On the
scored quarters, runners whose missing runs include a win won 499 races
against the stack's 443 expected.

``hidden-form``, declared before scoring: for the oldest ``hidden_form_runs``
figures of the form string (the ones the database's own runs do not account
for), the number of wins, places and failures to finish, the best, mean and
most recent finishing figure, and the wins and best figure against the field.
Missing where the horse has no such runs. The form string is the card's
pre-race form (its last figure matches the previous run 96.9% of the time and
the race itself at chance).
"""
import numpy as np
import pandas as pd

from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
FAIL = set("PFURBCSDO")


def figures(form, hidden):
    """Per runner: wins, places, fails, best, mean, last and count over the oldest ``hidden`` form figures."""
    out = np.full((len(form), 7), np.nan)
    for i, (text, count) in enumerate(zip(form, hidden)):
        if not count or count <= 0 or not isinstance(text, str):
            continue
        marks = [c for c in text.upper() if c.isalnum()][:int(count)]
        places = [10 if c == "0" else int(c) for c in marks if c.isdigit()]
        fails = sum(c in FAIL for c in marks)
        if not marks:
            continue
        out[i, 0] = sum(p == 1 for p in places)
        out[i, 1] = sum(p <= 3 for p in places)
        out[i, 2] = fails
        if places:
            out[i, 3], out[i, 4] = min(places), float(np.mean(places))
        last = marks[-1]
        out[i, 5] = (10 if last == "0" else int(last)) if last.isdigit() else 12.
        out[i, 6] = len(marks)
    return out


def hidden_form(frame, form, hidden):
    values = figures(form, hidden)
    out = frame[KEYS].copy()
    for j, name in enumerate(("wins", "places", "fails", "best", "mean", "last", "runs")):
        out[f"hf_{name}"] = values[:, j].astype("float32")
    out["hf_win_share"] = (out.hf_wins / out.hf_runs).astype("float32")
    race = frame.race_id.to_numpy()
    for name in ("hf_wins", "hf_best"):
        filled = out[name].fillna(0. if name == "hf_wins" else 6.)
        out[f"{name}_vs_field"] = (filled - filled.groupby(race).transform("mean")).astype("float32")
    return out


def load(frame):
    store = sc.read_store(["form", "hidden_form_runs"], start=frame.race_date.min()).drop(columns="race_date")
    store["race_id"] = store.race_id.astype(frame.race_id.dtype)
    work = frame[KEYS].merge(store, on=KEYS, how="left", validate="one_to_one")
    values = hidden_form(frame, work.form.to_numpy(object), work.hidden_form_runs.fillna(0).to_numpy())
    return values, [c for c in values if c not in KEYS]
