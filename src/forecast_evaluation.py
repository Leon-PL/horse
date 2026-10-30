"""Shared, strict probability scoring for complete, single-winner races.

Inputs must contain contiguous whole races. Never repair a partial field by
dropping runners or silently normalising a malformed forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import normalise_horse_key


def valid_race_mask(df):
    """Reject the whole field on ambiguous labels/identities; retain DNFs."""
    ids = df["race_id"]
    groups = df.groupby("race_id", sort=False, dropna=False)
    counts = groups["race_id"].transform("size")
    winners = pd.to_numeric(df["won"], errors="coerce")
    finishes = pd.to_numeric(df["finish_position"], errors="coerce")
    dates = pd.to_datetime(df["race_date"], errors="coerce").dt.normalize()
    names = df["horse_name"]
    runner_keys = pd.DataFrame({"race_id": ids, "horse": names.map(normalise_horse_key)})
    valid = (
        ids.notna() & ids.astype("string").str.strip().ne("").fillna(False)
        & dates.notna() & (dates.groupby(ids, dropna=False).transform("nunique") == 1)
        & names.notna() & names.astype("string").str.strip().ne("").fillna(False)
        & runner_keys.horse.ne("") & ~runner_keys.duplicated(keep=False)
        & winners.isin([0, 1]) & (winners.groupby(ids, dropna=False).transform("sum") == 1)
        & finishes.ge(0) & finishes.mod(1).eq(0)
        & ((finishes == 1) == (winners == 1)) & (counts >= 2)
    )
    if "num_runners" in df:
        valid &= counts == pd.to_numeric(df["num_runners"], errors="coerce")
    return valid.groupby(ids, sort=False, dropna=False).transform("all")


def validate_probabilities(probabilities, winners, groups):
    p, y, g = (np.asarray(v, dtype=float) for v in (probabilities, winners, groups))
    if any(v.ndim != 1 for v in (p, y, g)) or not len(g):
        raise ValueError("Expected nonempty one-dimensional whole-race arrays")
    if not np.isfinite(g).all() or (g < 2).any() or (g != np.floor(g)).any():
        raise ValueError("Group sizes must be integers of at least two")
    g = g.astype(np.int64)
    if len(p) != len(y) or g.sum() != len(p):
        raise ValueError("Probability, outcome and race sizes disagree")
    starts = np.r_[0, np.cumsum(g)[:-1]]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    if not np.allclose(np.add.reduceat(p, starts), 1, atol=1e-8, rtol=0):
        raise ValueError("Probabilities must sum to one within every complete race")
    if not np.isin(y, [0, 1]).all() or (np.add.reduceat(y, starts) != 1).any():
        raise ValueError("Every race must have exactly one binary winner")
    return p, y, g, starts


def probability_metrics(probabilities, winners, groups):
    p, y, groups, starts = validate_probabilities(probabilities, winners, groups)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    brier = np.square(p - y)
    binary_loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    race_loss = -np.add.reduceat(y * np.log(clipped), starts)
    per_race_brier = np.add.reduceat(brier, starts) / groups
    bucket = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bucket == b].sum() - y[bucket == b].sum())
              for b in range(10)) / len(p)
    # Expected accuracy under random tie-breaking, independent of row order.
    maxima = np.repeat(np.maximum.reduceat(p, starts), groups)
    ties = np.isclose(p, maxima, atol=1e-12, rtol=0)
    accuracy = np.add.reduceat(y * ties, starts) / np.add.reduceat(ties, starts)
    return {
        "brier": float(brier.mean()), "race_mean_brier": float(per_race_brier.mean()),
        "binary_log_loss": float(binary_loss.mean()),
        "race_log_loss": float(race_loss.mean()), "ece": float(ece),
        "top1_accuracy": float(accuracy.mean()),
    }, race_loss, per_race_brier


def paired_interval(delta, seed=42, draws=1000, blocks=None):
    """Paired whole-race bootstrap; optional date/meeting cluster sensitivity."""
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Expected finite nonempty paired race deltas")
    if isinstance(draws, bool) or int(draws) != draws or draws < 1:
        raise ValueError("draws must be a positive integer")
    if blocks is None:
        totals, counts = delta, np.ones(len(delta))
    else:
        labels = np.asarray(blocks)
        if labels.shape != delta.shape or pd.isna(labels).any():
            raise ValueError("One nonmissing bootstrap block is required per race")
        codes, unique = pd.factorize(labels, sort=False)
        totals = np.bincount(codes, weights=delta, minlength=len(unique))
        counts = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(seed)
    means = np.empty(int(draws))
    for i in range(int(draws)):
        chosen = rng.integers(len(totals), size=len(totals))
        means[i] = totals[chosen].sum() / counts[chosen].sum()
    return {"mean": float(delta.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist()}
