"""Canonical Sporting Life runner identities; other sources need a crosswalk."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import normalise_horse_key

MISSING_IDS = ("", "0", "nan", "none", "<na>", "nat")


def identity_values(values):
    """Make inferred integer/float IDs comparable without rounding opaque IDs."""
    out = values.astype("string").str.strip().str.replace(r"^([0-9]+)\.0+$", r"\1", regex=True)
    return out.mask(out.str.lower().isin(MISSING_IDS))


def horse_ids(frame):
    return identity_values(frame.horse_id) if "horse_id" in frame else pd.Series(pd.NA, index=frame.index, dtype="string")


def runner_keys(frame):
    ids = horse_ids(frame)
    names = frame.horse_name.map(normalise_horse_key).astype("string")
    return ("sl:"+ids).where(ids.notna(), "name:"+names)


def runner_index(frame):
    return pd.MultiIndex.from_arrays([identity_values(frame.race_id), runner_keys(frame)], names=["race_id", "runner_key"])


def runner_positions(left, right, *, require_all=True, skip_ambiguous=False):
    """Map each left runner to one right row, preserving left order.

    A missing ID may use a name unique on both sides of that race. Different
    known IDs never match through a shared display name. Ambiguous identities
    fail before any join can expand the population.
    """
    li, ri = runner_index(left), runner_index(right)
    safe_left, safe_right = ~li.duplicated(keep=False), ~ri.duplicated(keep=False)
    if not (safe_left.all() and safe_right.all()) and not skip_ambiguous:
        raise ValueError("Duplicate runner identities")
    candidates = ri[safe_right].get_indexer(li)
    positions = np.full(len(left), -1, dtype=int)
    exact = safe_left & (candidates >= 0)
    positions[exact] = np.flatnonzero(safe_right)[candidates[exact]]
    missing = positions < 0
    if missing.any() and not right.empty:
        ln = pd.MultiIndex.from_arrays([identity_values(left.race_id), left.horse_name.map(normalise_horse_key)])
        rn = pd.MultiIndex.from_arrays([identity_values(right.race_id), right.horse_name.map(normalise_horse_key)])
        unique_right = ~rn.duplicated(keep=False) & safe_right
        candidates = rn[unique_right].get_indexer(ln)
        valid = missing & safe_left & ~ln.duplicated(keep=False) & (candidates >= 0)
        candidate_rows = np.flatnonzero(unique_right)[np.maximum(candidates, 0)] if unique_right.any() else np.zeros(len(left), dtype=int)
        left_missing = horse_ids(left).isna().to_numpy()
        right_missing = horse_ids(right).isna().to_numpy()
        valid &= left_missing | right_missing[candidate_rows]
        positions[valid] = candidate_rows[valid]
    if require_all and (positions < 0).any():
        raise ValueError("Missing or ambiguous runner identity match")
    return positions


def join_runner_columns(left, right, columns, *, require_all=True, skip_ambiguous=False):
    """Attach only requested values, with no change in runner count or order."""
    columns = list(columns)
    clashes = set(columns) & set(left.columns)
    if clashes:
        raise ValueError(f"Runner join would replace columns: {sorted(clashes)}")
    positions = runner_positions(left, right, require_all=require_all, skip_ambiguous=skip_ambiguous)
    values = right.reset_index(drop=True).reindex(positions)
    out = left.copy()
    for column in columns:
        out[column] = values[column].array
    return out


def identity_columns(frame):
    return [name for name in ("race_id", "horse_id", "horse_name") if name in frame]
