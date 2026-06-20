"""Incremental state primitives for build-once / append-forever feature
engineering (taxonomy class **C**; see docs/FE_TAXONOMY.md).

`CountsAccumulator` generalises the Elo/Glicko `return_state` pattern to the
~15 cumulative stages. For each row it produces the prior-only, **race-safe**
cumulative sum of a value per entity — the sum over that entity's rows in
*strictly earlier races* (the whole current race excluded). This matches
`feature_engineer._race_safe_cumsum` exactly, and equals the plain
`groupby(entity)[v].cumsum() - v` idiom when an entity has one row per race
(horses), so one primitive covers every class-C stage.

Why it is parity-correct by construction:

    full(hist+new)[row]                                  # a from-scratch build
      = sum of entity's value over races strictly before this row's race
      = sum over history rows of entity            (a constant per entity)
        + sum over new rows of entity before this race
      = seed_total[entity] + race_safe_cumsum(new_frame)[row]

So the carried state is just `seed_total` (per-entity grand total), and a seeded
append is `in_frame_race_safe_cumsum + seed_total` — the *same* formula used for
the build. This holds because an append only adds rows strictly after history
and no race spans the boundary (a race lives on one date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CountsAccumulator:
    """Resumable prior-only race-safe per-entity cumulative sum (one metric)."""

    def __init__(self, state: dict | None = None):
        # entity value -> grand total of the metric over all rows seen so far.
        self.state: dict = dict(state or {})

    def to_state(self) -> dict:
        return dict(self.state)

    @staticmethod
    def _prior_to_race(keys: np.ndarray, race_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
        """In-frame race-safe cumulative sum (total before each row's race).

        Identical maths to ``feature_engineer._race_safe_cumsum``:
        ``cumsum(entity) - cumsum(entity, race)``.
        """
        frame = pd.DataFrame({"k": keys, "r": race_ids, "v": values})
        naive = frame.groupby("k", sort=False)["v"].cumsum()
        within = frame.groupby(["k", "r"], sort=False)["v"].cumsum()
        return (naive - within).to_numpy()

    def transform(self, keys, race_ids, values) -> np.ndarray:
        """Prior-only race-safe totals for *values*, then advance state.

        Rows must be in chronological event order (as in the FE pipeline). When
        called on appended rows with a seeded state, the result equals a full
        recompute over history+new for those rows.
        """
        keys = np.asarray(pd.Series(keys).to_numpy())
        race_ids = np.asarray(pd.Series(race_ids).to_numpy())
        values = np.asarray(values, dtype="float64")

        base = self._prior_to_race(keys, race_ids, values)
        if self.state:
            seed = pd.Series(keys).map(self.state).fillna(0.0).to_numpy()
        else:
            seed = 0.0
        out = base + seed

        # advance: add each entity's grand total over this frame
        totals = pd.Series(values).groupby(keys, sort=False).sum()
        for k, t in totals.items():
            self.state[k] = self.state.get(k, 0.0) + float(t)
        return out


def cumulative_counts(
    keys,
    race_ids,
    metrics: dict[str, object],
    *,
    state: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute several prior-only race-safe cumulative metrics for one entity.

    The reusable API the class-C stages call to become seed-aware: pass the
    entity key, race ids, and a ``{out_column: value_array}`` mapping; get back
    ``({out_column: array}, new_state)``. With ``state=None`` (a full build) the
    output is identical to the inline ``groupby(entity)[v].cumsum() - v`` /
    ``_race_safe_cumsum`` the stages use today; with a seeded ``state`` (an
    append) it equals a full recompute for the new rows.

    ``state`` / ``new_state`` are ``{out_column: accumulator_state}`` and are
    JSON-serialisable for persistence in the FeatureStore.
    """
    state = state or {}
    out: dict[str, np.ndarray] = {}
    new_state: dict = {}
    for col, values in metrics.items():
        acc = CountsAccumulator(state.get(col))
        out[col] = acc.transform(keys, race_ids, values)
        new_state[col] = acc.to_state()
    return out, new_state
