"""Sibling (dam-produce) form — the classic public read on an unraced horse.

A debutant has no form, and the model's handicap there is specific and
measured: it spreads debutant win probabilities over a 13.9x range where the
market uses 53.6x, so it cannot separate the 1.4 shot from the 200/1 shot
(``docs/NEGATIVE_RESULTS.md`` 2d). The most-used *public* substitute for form
on an unraced horse is what its relatives have done — "full brother to a Group
winner" — and the store contains nothing of the kind. ``dam`` is already in the
results table, so it is computable.

This module is deliberately **standalone and unwired**: it computes the
features from a results frame so they can be screened for orthogonality to the
market before any FE stage imports it. Screening first costs minutes;
integrating first costs a multi-hour store rebuild. If a screen clears the bar,
wire it into ``feature_engineer`` and register this file in
``_FE_SOURCE_FILES`` (CLAUDE.md hard rule 4) — ``tests/test_fe_signature_cove
rage.py`` will insist.

Leakage rules this obeys, both of which matter here (hard rule 1):

* **Self-exclusion.** A horse is not its own sibling. Its own earlier runs are
  removed from the dam's totals, so the feature is strictly about *other*
  offspring.
* **Date-strictness.** Siblings often run on the same day, and at prediction
  time that day's earlier results do not exist. State is advanced only at a day
  boundary, so every run on date D sees sibling form through D-1 — the same
  rule ``_entity_day_start_view`` applies to jockey and trainer history.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Shrinkage strength for the sibling win rate: a dam with two runs should not
#: read as a 50% producer. Matches the convention of the ``*_shrunk`` pedigree
#: features already in the store.
SHRINK_PRIOR_RUNS = 20.0

#: The rate shrunk toward — a **fixed constant**, deliberately not the mean of
#: whatever frame was passed in. Roughly 1/9, the typical field size.
#:
#: Using the sample mean breaks store-append parity: the same settled row would
#: get one value in a full build and a different one after an append, because
#: the population mean moves. A settled row's features must never change as
#: later races arrive (CLAUDE.md hard rule 1, and the invariant
#: ``tests/test_feature_store.py`` proves for the whole pipeline). The tests
#: here caught exactly that.
PRIOR_WIN_RATE = 0.11

#: Columns this module produces.
SIBLING_FEATURES = (
    "sibling_runs",
    "sibling_wins",
    "sibling_win_rate_shrunk",
    "sibling_count",
    "sibling_any_winner",
    "sibling_best_or",
    "sibling_best_or_vs_field",
    "has_sibling_history",
)


def _dam_key(series: pd.Series) -> pd.Series:
    """Normalise a dam name for grouping. Blank/unknown dams group to NaN."""
    key = series.astype("string").str.strip().str.upper()
    key = key.str.replace(r"\s+", " ", regex=True)
    return key.where(key.notna() & (key != "") & (key != "UNKNOWN"))


def compute_sibling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sibling-form features for every row of *df*, indexed like *df*.

    Requires ``dam``, ``race_date``, ``horse_name`` (or ``horse_id``), ``won``
    and optionally ``official_rating`` and ``race_id``.

    Single date-ordered pass, carrying per-dam state: totals for the dam, plus
    per-horse totals so a horse's own record can be subtracted. The running
    best official rating is held as a top-two by horse, which is what makes
    "best rating by a *sibling*" exact rather than contaminated by the horse's
    own best.
    """
    if "dam" not in df.columns:
        logger.info("sibling features: no dam column; emitting empties")
        return _empty(df)

    work = pd.DataFrame({
        "dam": _dam_key(df["dam"]),
        "horse": (df["horse_id"] if "horse_id" in df.columns else df["horse_name"]).astype(str),
        "date": pd.to_datetime(df["race_date"], errors="coerce"),
        "won": pd.to_numeric(df.get("won"), errors="coerce").fillna(0).astype(np.int8),
        "or_": pd.to_numeric(df.get("official_rating"), errors="coerce"),
    })
    order = np.lexsort((work["horse"].to_numpy(), work["date"].to_numpy()))

    n = len(work)
    out_runs = np.zeros(n); out_wins = np.zeros(n)
    out_count = np.zeros(n); out_best = np.full(n, np.nan)

    # Per-dam carried state, advanced only at day boundaries.
    dam_runs: dict = defaultdict(float)
    dam_wins: dict = defaultdict(float)
    horse_runs: dict = defaultdict(float)
    horse_wins: dict = defaultdict(float)
    dam_horses: dict = defaultdict(set)
    # top two (rating, horse) per dam, so self can be excluded exactly
    dam_top: dict = defaultdict(list)

    pending: list[tuple] = []          # rows of the current day, applied at its end
    current_day = None

    def flush():
        for dam, horse, won, rating in pending:
            dam_runs[dam] += 1.0
            dam_wins[dam] += won
            horse_runs[(dam, horse)] += 1.0
            horse_wins[(dam, horse)] += won
            dam_horses[dam].add(horse)
            if np.isfinite(rating):
                top = dam_top[dam]
                top.append((float(rating), horse))
                # keep the best entry per horse, then the best two horses
                best_by_horse: dict = {}
                for value, who in top:
                    if who not in best_by_horse or value > best_by_horse[who]:
                        best_by_horse[who] = value
                dam_top[dam] = sorted(
                    ((v, h) for h, v in best_by_horse.items()), reverse=True,
                )[:2]
        pending.clear()

    dates = work["date"].to_numpy()
    dams = work["dam"].to_numpy()
    horses = work["horse"].to_numpy()
    wons = work["won"].to_numpy()
    ors = work["or_"].to_numpy()

    for i in order:
        day = dates[i]
        if current_day is None or day != current_day:
            flush()
            current_day = day
        dam, horse = dams[i], horses[i]
        if dam is pd.NA or dam is None or (isinstance(dam, float) and np.isnan(dam)):
            continue
        # Subtract the horse's own record: a horse is not its own sibling.
        runs = dam_runs[dam] - horse_runs[(dam, horse)]
        wins = dam_wins[dam] - horse_wins[(dam, horse)]
        out_runs[i] = max(runs, 0.0)
        out_wins[i] = max(wins, 0.0)
        out_count[i] = len(dam_horses[dam] - {horse})
        top = dam_top[dam]
        for value, who in top:
            if who != horse:
                out_best[i] = value
                break
        pending.append((dam, horse, wons[i], ors[i]))
    flush()

    rate = (out_wins + SHRINK_PRIOR_RUNS * PRIOR_WIN_RATE) / (out_runs + SHRINK_PRIOR_RUNS)

    result = pd.DataFrame({
        "sibling_runs": out_runs,
        "sibling_wins": out_wins,
        "sibling_win_rate_shrunk": rate,
        "sibling_count": out_count,
        "sibling_any_winner": (out_wins > 0).astype(np.int8),
        "sibling_best_or": out_best,
        "has_sibling_history": (out_runs > 0).astype(np.int8),
    }, index=df.index)

    # Only the relative standing of a runner in its own field can pay, so
    # offer the race-relative form too (a race-level constant cancels in any
    # per-race-normalised model).
    if "race_id" in df.columns:
        best = result["sibling_best_or"]
        result["sibling_best_or_vs_field"] = (
            best - best.groupby(df["race_id"]).transform("mean")
        )
    else:
        result["sibling_best_or_vs_field"] = np.nan

    # Rates are meaningless with no sibling record; leave them missing rather
    # than implying an average producer (the RTV median-fill lesson).
    blank = result["has_sibling_history"] == 0
    result.loc[blank, "sibling_win_rate_shrunk"] = np.nan
    return result[list(SIBLING_FEATURES)]


def _empty(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in SIBLING_FEATURES:
        out[col] = 0 if col in ("has_sibling_history", "sibling_any_winner") else np.nan
    return out


def add_sibling_features(df: pd.DataFrame) -> pd.DataFrame:
    """FE-stage shape: return *df* with the sibling columns attached."""
    logger.info("Engineering sibling (dam-produce) features...")
    return pd.concat([df, compute_sibling_features(df)], axis=1)
