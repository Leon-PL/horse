"""Combine independently frozen missing-history cohorts without double counting."""
from __future__ import annotations

import pandas as pd


def combine_histories(frames):
    """Earliest pre-race eligibility wins; conflicting performances are rejected.

    A horse already selected in one earlier cohort need not wait for its later
    selection elsewhere. Later selection must never backdate a previously
    ineligible horse. Membership dates are retained even on duplicate records.
    """
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    from src.public_history_features import _ids
    combined = pd.concat(frames, ignore_index=True)
    combined["horse_id"] = _ids(combined.horse_id)
    combined["race_id"] = _ids(combined.race_id)
    eligibility = pd.to_datetime(combined.eligible_from_date, errors="coerce")
    if eligibility.isna().any():
        raise ValueError("Every cohort membership needs a known eligibility date")
    combined["eligible_from_date"] = eligibility.groupby(combined.horse_id).transform("min")
    keys = ["horse_id", "race_id"]
    relevant = [c for c in ("date", "position", "runner_count", "distance", "casualty") if c in combined]
    if combined[keys + relevant].astype(str).drop_duplicates().duplicated(keys).any():
        raise ValueError("Frozen cohorts disagree about an external performance")
    return combined.drop_duplicates(keys).reset_index(drop=True)
