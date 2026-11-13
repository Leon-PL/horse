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
from src.correction.inputs import form_figures as figures  # noqa: F401  (one owner: src)
from src.correction.inputs import hidden_form  # noqa: F401


def load(frame):
    store = sc.read_store(["form", "hidden_form_runs"], start=frame.race_date.min()).drop(columns="race_date")
    store["race_id"] = store.race_id.astype(frame.race_id.dtype)
    work = frame[KEYS].merge(store, on=KEYS, how="left", validate="one_to_one")
    values = hidden_form(frame, work.form.to_numpy(object), work.hidden_form_runs.fillna(0).to_numpy())
    return values, [c for c in values if c not in KEYS]
