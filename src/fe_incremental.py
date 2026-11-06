"""Incremental state primitives for build-once / append-forever feature
engineering (taxonomy class **C**; see docs/architecture/FE_TAXONOMY.md).

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

from itertools import repeat

import numba as nb
import numpy as np
import pandas as pd


@nb.njit(cache=True)
def _nb_resume_group_cumsum(values, labels, totals, compensation):
    """``groupby.cumsum`` for one float column, bit for bit, from any start.

    pandas compensates with Kahan summation, so plain running totals would
    differ from it in the last bits. A NaN row and a row without a group stay
    NaN and leave the totals alone. ``totals`` and ``compensation`` hold where
    each group has reached and are advanced in place, so resuming from them
    gives the same bits as a single pass over everything.
    """
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        group = labels[i]
        if group < 0:
            continue
        value = values[i]
        if value != value:
            continue
        adjusted = value - compensation[group]
        total = totals[group] + adjusted
        compensation[group] = total - totals[group] - adjusted
        totals[group] = total
        out[i] = total
    return out


def _nb_group_cumsum(values, labels, n_groups):
    """One pass from nothing. Returns the running and final totals."""
    totals = np.zeros(n_groups)
    out = _nb_resume_group_cumsum(values, labels, totals, np.zeros(n_groups))
    return out, totals


def _entity_labels(keys: np.ndarray) -> tuple[np.ndarray, list]:
    """Integer group labels in first-seen order; a missing key has no group."""
    labels, uniques = pd.factorize(keys)
    return labels.astype(np.int64), uniques.tolist()


def _entity_race_labels(entity: np.ndarray, race_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Labels of ``groupby([entity, race])`` built from the entity labels."""
    races, unique_races = pd.factorize(race_ids)
    pairs = entity * (len(unique_races) + 1) + races
    labels, uniques = pd.factorize(pairs)
    labels = labels.astype(np.int64)
    labels[(entity < 0) | (races < 0)] = -1
    return labels, len(uniques)


def _group_totals(values: np.ndarray, labels: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Group totals; pandas guards ``sum`` against infinity but not ``cumsum``."""
    if not np.isinf(values).any():
        return totals
    known = labels >= 0
    summed = pd.Series(values[known]).groupby(labels[known]).sum()
    return summed.reindex(range(len(totals)), fill_value=0.0).to_numpy()


def _carried(state: dict, entities: list) -> np.ndarray:
    """Carried totals for this frame's entities: never O(all history)."""
    if not state:
        return np.zeros(len(entities))
    carried = np.fromiter(map(state.get, entities, repeat(0.0)), np.float64, len(entities))
    carried[np.isnan(carried)] = 0.0
    return carried


class _RunningTotals:
    """The per-entity position of a Kahan sum, kept between chunks.

    Each total needs its compensation to carry on exactly where it stopped: a
    chunk that restarts from the total alone rounds differently, so two equal
    histories can stop being equal and a tied rank comes apart. Whole-number
    metrics have no compensation, so only non-zero ones are kept.
    """

    def __init__(self, state: dict | None = None, compensation: dict | None = None):
        # entity value -> grand total of the metric over all rows seen so far.
        self.state: dict = dict(state or {})
        self.compensation: dict = dict(compensation or {})

    def to_state(self) -> dict:
        return dict(self.state)

    def _running(self, entity, entities, values) -> np.ndarray:
        """Totals after each row, which also moves the carried position on."""
        started = _carried(self.state, entities)
        totals = started.copy()
        compensation = _carried(self.compensation, entities)
        running = _nb_resume_group_cumsum(values, entity, totals, compensation)
        stuck = ~np.isfinite(totals)
        if stuck.any():
            in_frame = _group_totals(values, entity, np.zeros(len(totals)))
            totals[stuck] = started[stuck] + in_frame[stuck]
            compensation[stuck] = 0.0
        self.state.update(zip(entities, totals.tolist()))
        if compensation.any() or self.compensation:
            for key, value in zip(entities, compensation.tolist()):
                if value:
                    self.compensation[key] = value
                else:
                    self.compensation.pop(key, None)
        return running


class CountsAccumulator(_RunningTotals):
    """Resumable prior-only race-safe per-entity cumulative sum (one metric)."""

    def _advance(self, entity, entities, pair, n_pairs, values) -> np.ndarray:
        running = self._running(entity, entities, values)
        within, _ = _nb_group_cumsum(values, pair, n_pairs)
        return running - within

    def transform(self, keys, race_ids, values) -> np.ndarray:
        """Prior-only race-safe totals for *values*, then advance state.

        Rows must be in chronological event order (as in the FE pipeline). When
        called on appended rows with a seeded state, the result equals a full
        recompute over history+new for those rows.
        """
        keys = np.asarray(pd.Series(keys).to_numpy())
        race_ids = np.asarray(pd.Series(race_ids).to_numpy())
        values = np.asarray(values, dtype="float64")
        entity, entities = _entity_labels(keys)
        pair, n_pairs = _entity_race_labels(entity, race_ids)
        return self._advance(entity, entities, pair, n_pairs, values)


class PlainCountsAccumulator(_RunningTotals):
    """Resumable prior-only **plain** per-entity cumulative sum (one metric).

    Unlike :class:`CountsAccumulator` this does *not* exclude same-race peers —
    it reproduces the inline ``groupby(key)[v].cumsum() - v`` idiom (row-order
    prior sum). Used by the few stages whose historical formula is plain
    cumsum-minus-current rather than race-safe (e.g. track×draw counts). Still
    exact across an append: new rows are strictly after history and within-frame
    row order is preserved, so seed_total + in-frame(cumsum - current) equals a
    full recompute. No ``race_ids`` needed.
    """

    def _advance(self, entity, entities, values) -> np.ndarray:
        return self._running(entity, entities, values) - values

    def transform(self, keys, values) -> np.ndarray:
        keys = np.asarray(pd.Series(keys).to_numpy())
        values = np.asarray(values, dtype="float64")
        entity, entities = _entity_labels(keys)
        return self._advance(entity, entities, values)


def _compensation_key(column: str) -> str:
    """Where a column's compensation sits in the carried state."""
    return f"{column}::compensation"


def cumulative_counts_plain(
    keys,
    metrics: dict[str, object],
    *,
    state: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Plain (non-race-safe) sibling of :func:`cumulative_counts`.

    Same ``{out_column: value_array} -> ({out_column: array}, new_state)`` API,
    but each column is a plain prior-only cumsum-minus-current (the seed-aware
    equivalent of ``groupby(key)[v].cumsum() - v``). For one-row-per-race
    entities this equals the race-safe version; use it only where the original
    stage was deliberately plain.
    """
    state = state or {}
    out: dict[str, np.ndarray] = {}
    new_state: dict = {}
    entity, entities = _entity_labels(np.asarray(pd.Series(keys).to_numpy()))
    for col, values in metrics.items():
        kept = _compensation_key(col)
        acc = PlainCountsAccumulator(state.get(col), state.get(kept))
        out[col] = acc._advance(entity, entities, np.asarray(values, dtype="float64"))
        new_state[col], new_state[kept] = acc.state, acc.compensation
    return out, new_state


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

    ``state`` / ``new_state`` are ``{out_column: accumulator_state}``, plus each
    column's ``{out_column}::compensation``, and are JSON-serialisable for
    persistence in the FeatureStore.
    """
    state = state or {}
    out: dict[str, np.ndarray] = {}
    new_state: dict = {}
    entity, entities = _entity_labels(np.asarray(pd.Series(keys).to_numpy()))
    pair, n_pairs = _entity_race_labels(entity, np.asarray(pd.Series(race_ids).to_numpy()))
    for col, values in metrics.items():
        kept = _compensation_key(col)
        acc = CountsAccumulator(state.get(col), state.get(kept))
        out[col] = acc._advance(entity, entities, pair, n_pairs, np.asarray(values, dtype="float64"))
        new_state[col], new_state[kept] = acc.state, acc.compensation
    return out, new_state
