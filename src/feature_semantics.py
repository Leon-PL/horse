"""Canonical parsing/validity contracts used by normal training and live cards.

The separately named history_coverage_features function remains a research
projection; importing this module does not add those predictors to training.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def normalise_going(value, detailed_map=None):
    """Retain known mappings; classify unlisted variants by their main going.

    Whitespace and case cannot turn yielding/heavy ground into Good. Genuine
    missing/unrecognised descriptions are an explicit Unknown category. Already
    normalised values remain unchanged on a second cleaning pass.
    """
    if value is None or pd.isna(value):
        return "Unknown"
    text = re.sub(r"\s+", " ", str(value).strip().casefold())
    text = re.sub(r"\s+\)", ")", text)
    if detailed_map and text in detailed_map:
        return detailed_map[text]
    canonical = {s.casefold(): s for s in ("Fast", "Good", "Soft", "Heavy", "Yielding", "Unknown")}
    if text in canonical:
        return canonical[text]
    main = text.split("(", 1)[0].strip()
    main = re.sub(r"\s*/\s*", "/", main)
    groups = {"standard": "Fast", "standard/fast": "Fast", "standard/slow": "Good",
        "firm": "Fast", "good to firm": "Fast", "good": "Good",
        "good to soft": "Soft", "soft": "Soft", "soft to heavy": "Heavy", "heavy": "Heavy",
        "good to yielding": "Yielding", "yielding": "Yielding", "yielding to soft": "Yielding"}
    return groups.get(main, "Unknown")


def class_number(values):
    """An unavailable class is not Class0 or Class5; Grade1 is not UK Class1."""
    text = pd.Series(values, copy=False).astype("string").str.strip()
    parsed = text.str.extract(r"(?i)^(?:class\s*)?([1-7])(?:\.0+)?$", expand=False)
    return pd.to_numeric(parsed, errors="coerce").astype(float)


def observed_rating(frame):
    """Mask field-median substitutes before computing a historical mark move."""
    values = pd.to_numeric(frame["official_rating"], errors="coerce")
    known = values.gt(0) & np.isfinite(values)
    if "has_official_rating" in frame:
        known &= pd.to_numeric(frame.has_official_rating, errors="coerce").eq(1)
    return values.where(known)


def jumps_mask(frame):
    """Use the actual category rather than the presence of another race's dummy."""
    if "race_type" in frame:
        return frame.race_type.astype("string").str.casefold().isin(["hurdle", "chase"])
    result = pd.Series(False, index=frame.index)
    for name in ("race_type_Hurdle", "race_type_Chase"):
        if name in frame:
            result |= frame[name].fillna(0).eq(1)
    return result


def history_coverage_features(frame):
    """Research design: expose missing local experience and observed recency.

    Positive pre-race source recency proves a previous start, not a complete
    number of career runs. A source date older than a verified local start is
    contradictory, so it cannot overwrite that local start.
    """
    local_runs = pd.to_numeric(frame.horse_prev_races, errors="coerce")
    local_gap = pd.to_numeric(frame.days_since_last_race, errors="coerce").where(local_runs.gt(0))
    source_gap = pd.to_numeric(frame.days_since_last_run, errors="coerce")
    source_gap = source_gap.where(np.isfinite(source_gap) & source_gap.gt(0) & source_gap.mod(1).eq(0))
    form_runs = pd.to_numeric(frame.prior_form_runs, errors="coerce").fillna(0)
    source_experience = source_gap.notna() | form_runs.ge(2)
    consistent = local_gap.isna() | source_gap.le(local_gap)
    usable_source = source_gap.where(consistent)
    gap = local_gap.where(usable_source.isna(), usable_source)
    return pd.DataFrame({
        "local_history_available": local_runs.gt(0).astype(np.int8),
        "source_shows_unseen_experience": (local_runs.eq(0) & source_experience).astype(np.int8),
        "source_shows_intervening_run": (local_gap.notna() & source_gap.lt(local_gap)).astype(np.int8),
        "source_recency_conflict": (source_gap.notna() & ~consistent).astype(np.int8),
        "observed_days_since_run": gap,
        "observed_days_since_run_log": np.log1p(gap),
    }, index=frame.index)
