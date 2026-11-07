"""Canonical surface classification, plus a reference pedigree-surface calculation.

The feature pipeline uses synthetic_surface. pedigree_surface_history is not
a pipeline stage: tests/test_surface_features.py uses it as an independent
reference that the native sire all-weather counters must match. Natural beach
sand is distinct from synthetic surfaces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC = frozenset({"aw", "allweather", "polytrack", "tapeta", "fibresand", "fibersand"})
NATURAL = frozenset({"turf", "grass", "sand", "dirt"})
MODEL_FEATURES = ("sire_aw_edge", "race_is_synthetic", "sire_aw_prior_runs_log1p")


def synthetic_surface(values):
    """Known synthetic=1, known natural=0, unavailable/unrecognised=NaN."""
    source = pd.Series(values, copy=False).astype("string").str.lower().str.replace(r"[^a-z]", "", regex=True)
    result = pd.Series(np.nan, index=source.index, dtype=float)
    result.loc[source.isin(SYNTHETIC)] = 1.
    result.loc[source.isin(NATURAL)] = 0.
    return result


def pedigree_surface_history(cards, history, *, entities=("sire", "damsire")):
    """Compute corrected rates and support without using a card's outcomes.

    The full history may contain future events. As-of lookups exclude the
    current calendar date entirely. NR/withdrawals and pending cards are not
    losses; recognised non-finishers are starts and losses.
    """
    from src.feature_engineer import _add_result_history_flags

    result = pd.DataFrame(index=cards.index)
    result["race_is_synthetic"] = synthetic_surface(cards.surface).to_numpy()
    columns = list(dict.fromkeys(["race_id", "race_date", "finish_position", "won", "surface",
        *[c for c in ("finish_pos_label", "num_runners") if c in history], *[c for c in entities if c in history]]))
    past = _add_result_history_flags(history[columns].copy())
    past["_day"] = pd.to_datetime(past.race_date, errors="raise").dt.normalize()
    card_dates = pd.to_datetime(cards.race_date, errors="raise").dt.normalize()
    if past._day.isna().any() or card_dates.isna().any():
        raise ValueError("Surface histories require known race dates")
    aw = synthetic_surface(past.surface).eq(1)
    starts = past._result_start.astype(float)
    wins = past._won_for_history.fillna(0).astype(float)
    places = past._placed_for_history.fillna(0).astype(float)
    metrics = pd.DataFrame({"starts": starts, "wins": wins, "places": places,
        "aw_starts": starts.where(aw, 0.), "aw_wins": wins.where(aw, 0.),
        "aw_places": places.where(aw, 0.)}, index=past.index)
    for entity in entities:
        if entity not in cards or entity not in past:
            continue
        names = past[entity].astype("string")
        known = names.notna() & names.str.strip().ne("")
        daily = metrics.loc[known].copy()
        daily["_entity"] = names.loc[known].to_numpy()
        daily["_day"] = past.loc[known, "_day"].to_numpy()
        daily = daily.groupby(["_entity", "_day"], sort=True)[list(metrics)].sum().reset_index()
        daily[list(metrics)] = daily.groupby("_entity", sort=False)[list(metrics)].cumsum()
        left = pd.DataFrame({"_entity": cards[entity].astype("string").to_numpy(),
            "_day": card_dates.to_numpy(), "_row": np.arange(len(cards))})
        known_card = left._entity.notna() & left._entity.str.strip().ne("")
        counts = np.full((len(cards), len(metrics.columns)), np.nan)
        if known_card.any():
            if daily.empty:
                counts[known_card.to_numpy()] = 0.
            else:
                # Global date order is required by merge_asof even with `by`.
                aligned = pd.merge_asof(left.loc[known_card].sort_values("_day", kind="stable"),
                    daily.sort_values("_day", kind="stable"), by="_entity", on="_day",
                    direction="backward", allow_exact_matches=False)
                counts[aligned._row.to_numpy()] = aligned[list(metrics)].fillna(0).to_numpy(float)
        total_n, total_w, _, aw_n, aw_w, aw_p = counts.T
        overall = (total_w + 2.) / (total_n + 20.)
        rate = (aw_w + 1.) / (aw_n + 10.)
        result[f"{entity}_aw_win_rate_shrunk"] = rate
        result[f"{entity}_aw_place_rate_shrunk"] = (aw_p + 3.) / (aw_n + 10.)
        result[f"{entity}_aw_edge"] = rate - overall
        result[f"{entity}_aw_prior_runs"] = aw_n
        result[f"{entity}_aw_prior_runs_log1p"] = np.log1p(aw_n)
    return result
