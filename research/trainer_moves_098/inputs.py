"""Trainer-move inputs (research 098).

A horse that changes yard is judged by the old yard's form while it runs for
the new one. These inputs describe the move itself, using only runs before
the current race:

- ``tm_elo_change``: the new trainer's pre-race Elo minus the old trainer's
  pre-race Elo at the horse's last run for the old yard. Carried for every run
  in the new yard; missing for a horse that has never moved.
- ``tm_elo_change_fresh``: the same, only on the first three runs for the yard.
- ``tm_runs_in_yard``: earlier runs for the current trainer.
- ``tm_arrival_edge``: how the trainer's earlier new arrivals (their first
  three runs for the yard) did against their starting prices, shrunk toward
  zero. Only races on earlier dates count.
- ``tm_arrivals_n``: how many such arrival runs that edge rests on.

A change between two names that share a surname (C & M Johnston to C Johnston,
Lucinda Russell to L Russell & M Scudamore) is a licence change at the same
yard, not a move.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COLUMNS = ["tm_elo_change", "tm_elo_change_fresh", "tm_runs_in_yard", "tm_arrival_edge", "tm_arrivals_n"]
STORE_COLUMNS = ["race_id", "horse_id", "race_date", "trainer", "trainer_elo", "won", "odds"]
FRESH_RUNS = 3
SHRINK = 30.0
_TITLES = {"mrs", "miss", "mr", "ms", "dr", "the", "and", "sir", "lady", "capt", "major"}


def _surnames(name) -> frozenset:
    return frozenset(t for t in re.split(r"[^a-z']+", str(name).lower()) if len(t) > 2 and t not in _TITLES)


def same_yard(previous: pd.Series, current: pd.Series) -> np.ndarray:
    """True where two trainer names share a surname token (a licence change, not a move)."""
    pairs = pd.DataFrame({"a": previous.to_numpy(), "b": current.to_numpy()})
    known = pairs.a.notna() & pairs.b.notna() & (pairs.a != pairs.b)
    unique = pairs[known].drop_duplicates()
    shared = {(a, b): bool(_surnames(a) & _surnames(b)) for a, b in zip(unique.a, unique.b)}
    out = np.zeros(len(pairs), dtype=bool)
    idx = np.flatnonzero(known.to_numpy())
    out[idx] = [shared[(a, b)] for a, b in zip(pairs.a.to_numpy()[idx], pairs.b.to_numpy()[idx])]
    return out


def sp_probability(odds: pd.Series, race_id: pd.Series) -> np.ndarray:
    """Starting-price probability normalised within complete fields; missing otherwise."""
    raw = np.where(odds.to_numpy(dtype=float) > 1.0, 1.0 / odds.to_numpy(dtype=float), np.nan)
    s = pd.Series(raw)
    total = s.groupby(race_id.to_numpy()).transform("sum")
    complete = s.notna().groupby(race_id.to_numpy()).transform("all")
    return np.where(complete & (total > 0), s / total, np.nan)


def build(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``race_id``, ``horse_id`` and the trainer-move inputs for every row of *frame*."""
    missing = set(STORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    f = frame[STORE_COLUMNS].copy()
    f["race_date"] = pd.to_datetime(f.race_date)
    if f.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate runner rows")
    f = f.sort_values(["horse_id", "race_date", "race_id"], kind="mergesort").reset_index(drop=True)
    by_horse = f.groupby("horse_id", sort=False)

    previous_trainer = by_horse.trainer.shift()
    previous_elo = by_horse.trainer_elo.shift()
    changed = previous_trainer.notna() & (f.trainer != previous_trainer)
    move = changed & ~same_yard(previous_trainer, f.trainer)
    yard = move.astype(np.int64).groupby(f.horse_id).cumsum()       # 0 until the first move
    runs_in_yard = f.groupby([f.horse_id, yard]).cumcount()           # earlier runs for this yard
    old_elo = pd.Series(np.where(move, previous_elo, np.nan)).groupby([f.horse_id, yard]).transform("first")
    elo_change = (f.trainer_elo - old_elo).where(yard > 0)

    f["tm_elo_change"] = elo_change
    f["tm_elo_change_fresh"] = elo_change.where(runs_in_yard < FRESH_RUNS)
    f["tm_runs_in_yard"] = runs_in_yard.astype(float)

    # Trainer's earlier arrivals against SP, on strictly earlier dates.
    mkt = sp_probability(f.odds, f.race_id)
    arrival = (yard > 0) & (runs_in_yard < FRESH_RUNS) & np.isfinite(mkt)
    contrib = pd.DataFrame({"trainer": f.trainer, "race_date": f.race_date,
                            "res": np.where(arrival, f.won - mkt, 0.0), "n": arrival.astype(float)})
    daily = contrib.groupby(["trainer", "race_date"], sort=True)[["res", "n"]].sum()
    earlier = daily.groupby(level="trainer").cumsum() - daily          # excludes the same date
    earlier = earlier.reindex(pd.MultiIndex.from_frame(f[["trainer", "race_date"]]))
    res, n = earlier.res.to_numpy(), earlier.n.to_numpy()
    f["tm_arrival_edge"] = np.where(f.trainer.notna(), res / (n + SHRINK), np.nan)
    f["tm_arrivals_n"] = np.where(f.trainer.notna(), n, np.nan)
    return f[["race_id", "horse_id", *COLUMNS]]
