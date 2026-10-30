"""Restore the distinction between settled non-finishers and pending cards."""
from __future__ import annotations

import pandas as pd
import numpy as np

NONSTARTER_LABELS = frozenset({"NR", "N/R", "WD", "W/D", "WITHDRAWN", "NON RUNNER", "NON-RUNNER", "VOID"})


def pending_card(df):
    """Clear every result signal before either live feature-generation path."""
    out = df.copy()
    out["won"] = 0
    out["finish_position"] = 0
    out["finish_pos_label"] = ""
    out["lengths_behind"] = np.nan
    out["beaten_lengths"] = np.nan
    for col in ("finish_time_secs", "winning_time_secs"):
        if col in out:
            out[col] = np.nan
    return out


def result_labels(df):
    """Preserve reported labels; reconstruct a generic DNF for legacy results.

    The scraper excludes non-runners from result fields. Older DB/CSV schemas
    lost finish_pos_label, so position0 in a complete field containing a winner
    is a non-completion. All-zero pending cards and visibly incomplete fields
    are left unknown. Do not guess a specific fall/pulled-up reason.
    """
    labels = (df["finish_pos_label"].fillna("").astype(str).str.strip().str.upper()
              if "finish_pos_label" in df else pd.Series("", index=df.index))
    labels = labels.mask(labels.isin({"0", "NAN", "NONE", "<NA>"}), "")
    if not {"race_id", "finish_position"}.issubset(df):
        return labels
    position = pd.to_numeric(df.finish_position, errors="coerce")
    settled = position.eq(1).groupby(df.race_id, sort=False).transform("any")
    if "num_runners" in df:
        size = df.groupby("race_id", sort=False).race_id.transform("size")
        declared = pd.to_numeric(df.num_runners, errors="coerce")
        complete = declared.eq(size).groupby(df.race_id, sort=False).transform("all")
        settled &= complete
    inferred = position.eq(0) & labels.eq("") & settled
    return labels.mask(inferred, "DNF")
