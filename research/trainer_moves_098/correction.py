"""Trainer-move inputs as a group for the error-anatomy correction harness (research 098).

``trainer-move`` is the five inputs of ``inputs.py``; ``trainer-move-twin`` is
the same values shuffled within quarter, so the pair separates the inputs'
information from the cost of adding five columns. Both are run inside the
best joint recipe (``more-groups-deeper``) by
``research.error_anatomy.joint_variants``.
"""
import numpy as np
import pandas as pd

from research.trainer_moves_098.inputs import COLUMNS

INPUTS = "data/research/trainer_moves_098/inputs.parquet"
KEYS = ["race_id", "horse_id"]
TWIN_SEED = 11


def load(frame, *, twin=False):
    values = pd.read_parquet(INPUTS, columns=[*KEYS, *COLUMNS])
    rows = frame[[*KEYS, "quarter"]].copy()
    values["race_id"] = values.race_id.astype(str) if rows.race_id.dtype == object else values.race_id
    values["horse_id"] = values.horse_id.astype(rows.horse_id.dtype)
    out = rows.merge(values, on=KEYS, how="left", validate="one_to_one")
    if twin:
        rng = np.random.default_rng(TWIN_SEED)
        quarter = out.quarter.to_numpy()
        for label in pd.unique(quarter):
            idx = np.flatnonzero(quarter == label)
            perm = idx[rng.permutation(len(idx))]
            out.loc[idx, COLUMNS] = out.loc[perm, COLUMNS].to_numpy()
    names = [f"{c}_twin" for c in COLUMNS] if twin else list(COLUMNS)
    out = out.rename(columns=dict(zip(COLUMNS, names)))
    return out[[*KEYS, *names]], names
