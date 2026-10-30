"""Convert the source's adjacent finishing gaps to distances behind the winner.

The historical raw ``lengths_behind`` column is misnamed: Sporting Life's
``finish_distance`` is the gap to the preceding finisher. Keep that source
column intact and expose the corrected meaning separately as ``beaten_lengths``.
Only complete, unambiguous chains of finishing positions have known totals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_beaten_lengths(df: pd.DataFrame) -> pd.Series:
    """Return cumulative gaps in input order, preserving the input index.

    Unknown/negative gaps invalidate that position and every following one;
    a missing finisher cannot be skipped. Dead heats share a position and
    distance: their source gaps must be zero except for at most one gap from
    the previous position. The next position advances by the number tied.
    Pending cards and non-finishers have no measured beaten distance.
    """
    required = {"race_id", "finish_position", "lengths_behind"}
    if not required.issubset(df):
        return pd.Series(np.nan, index=df.index, name="beaten_lengths")
    work = pd.DataFrame({
        "race_id": df["race_id"].to_numpy(),
        "position": pd.to_numeric(df["finish_position"], errors="coerce").to_numpy(),
        "gap": pd.to_numeric(df["lengths_behind"], errors="coerce").to_numpy(),
        "row": np.arange(len(df)),
    })
    work = work.loc[work.position.ge(1) & work.position.mod(1).eq(0)]
    work["valid"] = np.isfinite(work.gap) & work.gap.ge(0)
    work["positive"] = work.gap.gt(0).astype(int)
    out = np.full(len(df), np.nan)
    # Group tied positions first; summing their gaps is only valid when no
    # more than one is nonzero. Reject conflicting ties rather than choose
    # a value according to input row order.
    grouped = work.groupby(["race_id", "position"], sort=True).agg(
        gap=("gap", "sum"), count=("gap", "size"),
        valid=("valid", "all"), positive=("positive", "sum"),
    ).reset_index()
    if grouped.empty:
        return pd.Series(out, index=df.index, name="beaten_lengths")
    grouped["expected_position"] = grouped.groupby("race_id", sort=False)["count"].cumsum() - grouped["count"] + 1
    # Winners establish the origin, even when their source gap is blank.
    grouped.loc[grouped.position.eq(1), ["gap", "valid", "positive"]] = [0.0, True, 0]
    grouped["valid"] &= grouped.position.eq(grouped.expected_position) & grouped.positive.le(1)
    grouped["chain_valid"] = grouped.groupby("race_id", sort=False).valid.cummin()
    grouped["total"] = grouped.groupby("race_id", sort=False).gap.cumsum().where(grouped.chain_valid)
    measured = work.merge(grouped[["race_id", "position", "total"]], on=["race_id", "position"], how="left", validate="many_to_one")
    out[measured.row.to_numpy()] = measured.total.to_numpy()
    return pd.Series(out, index=df.index, name="beaten_lengths")


def margin_values(df: pd.DataFrame) -> pd.Series:
    """Prefer explicit corrected margins; legacy featured snapshots stay readable."""
    return df["beaten_lengths"] if "beaten_lengths" in df else df["lengths_behind"]
