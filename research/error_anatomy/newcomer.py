"""Newcomers: the model's past errors on unraced and lightly raced horses, by family and yard.

The fourth pass puts the remaining loss to the 08:00 exchange price in races
of unraced and lightly raced horses. The existing error inputs key a trainer
by the horse's state (``trainer_fresh``) but nothing else, and never use the
dam.

``newcomer``, declared before scoring: the shrunk, 365-day-decayed residual
and market-gap means of ``entity_residuals``, keyed by

- dam (her earlier runners, whichever horse they were);
- sire, dam's sire and jockey, each with the horse's state (debut, back from a
  break, fit);
- trainer with state and age band (two-year-old, three-year-old, older).

Strictly earlier dates only.
"""
import numpy as np
import pandas as pd

from research.error_anatomy import entity_residuals as er
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
ENTITIES = {"dam": ["dam"], "sire_fresh": ["sire", "_fresh"], "damsire_fresh": ["damsire", "_fresh"],
            "jockey_fresh": ["jockey", "_fresh"], "trainer_fresh_age": ["trainer", "_fresh", "_age"]}


def with_keys(frame, extra, entities, prefix):
    """``entity_inputs`` with extra key columns taken from ``extra`` (aligned to ``frame``)."""
    return er.entity_inputs(frame, market="p_market", half_life=er.LONG_HALF_LIFE, entities=entities, prefix=prefix,
                            extra_keys=extra)


def newcomer_inputs(frame):
    store = sc.read_store(["dam", "age"], start=frame.race_date.min()).drop(columns="race_date")
    store["race_id"] = store.race_id.astype(frame.race_id.dtype)
    work = frame[KEYS].merge(store, on=KEYS, how="left", validate="one_to_one")
    age = pd.to_numeric(work.age, errors="coerce")
    band = np.select([age <= 2, age == 3], ["2", "3"], "4+")
    return with_keys(frame, {"dam": work.dam.to_numpy(object), "_age": band}, ENTITIES, "nc")


def load(frame):
    values = newcomer_inputs(frame)
    return values, [c for c in values if c not in KEYS]
