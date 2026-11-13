"""Canonical parsing/validity contracts used by normal training and live cards.

The separately named history_coverage_features function remains a research
projection; importing this module does not add those predictors to training.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


# Historical titles include the abbreviation and a misspelling in mixed races.
_CONDITIONAL = r"\b(?:conditionals?|conditonals?|cjs?['’]?)"
_AMATEUR = r"\bamateurs?"
_RIDER_GROUP_SEPARATOR = r"\s*(?:[/&]|\band\b)\s*"
_CONDITIONAL_AND_AMATEUR = (
    _CONDITIONAL + _RIDER_GROUP_SEPARATOR + _AMATEUR + r"\b|"
    + _AMATEUR + _RIDER_GROUP_SEPARATOR + _CONDITIONAL + r"\b"
)
_RIDER_RESTRICTIONS = {
    "race_conditional_jockeys": r"\bconditionals?\s+(?:jockeys?|riders?)\b|" + _CONDITIONAL_AND_AMATEUR,
    "race_apprentice_jockeys": (r"\bapprentices?['’]?\s+(?:jockeys?|riders?|handicap|classified|maiden|"
                                r"claiming|optional|rated|restricted|derby|nursery|training series|fillies|"
                                r"selling|race|stakes)\b"),
    "race_amateur_riders": (r"\bamateur\s+(?:(?:male|female|lady|gentleman)\s+)?(?:jockeys?|riders?)\b|"
                            r"\b(?:lady|ladies|gentlemen|gentleman|male|female|military)\s+amateurs?\b|"
                            r"\bqualified riders?\b|\(\s*q\.?\s*r\.?\s*\)|\(\s*amateurs?\s*\)|"
                            + _CONDITIONAL_AND_AMATEUR),
}


# BHA race-type rules and dated fixture changes. Ireland follows the IHRB.
_HUNTER_CHASE = r"\b(?:foxhunters?|hunters?)['’]?\s+(?:open\s+)?(?:steeple\s*)?chase\b"
_CHELTENHAM_KIM_MUIR = r"\bfulke walwyn kim muir\b"
_CHELTENHAM_NH_CHASE = r"\bnational hunt (?:challenge cup|chase)\b"


def claim_pounds(values):
    """Only reported whole pounds, including an explicit zero, are usable."""
    values = pd.to_numeric(values.astype("string"), errors="coerce")
    return values.where(values.between(0, 30) & values.mod(1).eq(0)).astype(float)


CLAIM_ADJUSTED_WEIGHT_INPUTS = ("weight_lbs", "has_observed_weight", "reported_jockey_claim_lbs")


def claim_adjusted_weight(frame):
    """Estimate pre-race burden from observed allocated weight and allowance.

    Sporting Life supplies the allocation in pounds. A verified zero claim
    preserves it. An unknown claim or imputed allocation leaves the estimate
    missing. Weigh-out adjustments such as overweight require separate evidence.
    """
    if not {"weight_lbs", "reported_jockey_claim_lbs"}.issubset(frame):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    allocated = pd.to_numeric(frame.weight_lbs, errors="coerce").astype(float)
    observed = allocated.between(50, 220)
    if "has_observed_weight" in frame:
        observed &= pd.to_numeric(frame.has_observed_weight, errors="coerce").eq(1)
    return allocated.where(observed) - claim_pounds(frame.reported_jockey_claim_lbs)


def burden_weight(frame):
    """Weight for comparisons: claim-adjusted where the claim is verified, allocated weight otherwise.

    The allocated weight is known for every runner, while a verified claim is missing for about
    30% of 2020-2023 runners. Comparing only verified runners left ``weight_vs_field`` and the
    weight-change inputs blank for those rows and biased the field mean towards verified riders.
    Most riders claim nothing, so the allocation is the best estimate when the claim is unknown.
    ``claim_adjusted_weight_lbs`` itself stays missing in that case, so the model can tell.
    """
    adjusted = claim_adjusted_weight(frame)
    if "weight_lbs" not in frame:
        return adjusted
    allocated = pd.to_numeric(frame.weight_lbs, errors="coerce").astype(float)
    observed = allocated.between(50, 220)
    if "has_observed_weight" in frame:
        observed &= pd.to_numeric(frame.has_observed_weight, errors="coerce").eq(1)
    return adjusted.fillna(allocated.where(observed))


def rider_name_key(values):
    """Compare the declared person's name without guessing expansions of initials."""
    return (values.astype("string").str.casefold().str.replace(".", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True).str.strip().replace(["", "unknown", "none", "nan"], pd.NA))


def rider_restriction_features(frame):
    """Combine advertised rider eligibility with verified race-type and fixture rules."""
    titles = frame.get("race_name", pd.Series(index=frame.index, dtype="string")).astype("string").str.strip()
    titles = titles.mask(titles.eq(""))
    flags = pd.DataFrame({name: titles.str.contains(pattern, case=False).astype(float)
                          for name, pattern in _RIDER_RESTRICTIONS.items()}, index=frame.index)
    if {"race_date", "track"}.issubset(frame):
        from src.prize_currency_features import course_currency
        from src.utils import canonical_track_key

        day = pd.to_datetime(frame.race_date, errors="coerce")
        track = frame.track.map(canonical_track_key)
        currency = course_currency(frame.track)
        britain = currency.eq("GBP")
        ireland = currency.eq("EUR")
        # Amateurs were suspended in Britain from 16 January until 29 March 2021.
        suspended = britain & day.between("2021-01-16", "2021-03-28")
        hunter = titles.str.contains(_HUNTER_CHASE, case=False, na=False)
        festival = track.eq("cheltenham") & (
            titles.str.contains(_CHELTENHAM_KIM_MUIR, case=False, na=False)
            | (day.lt("2025-01-01") & titles.str.contains(_CHELTENHAM_NH_CHASE, case=False, na=False)))
        amateur = flags["race_amateur_riders"]
        amateur = amateur.mask(day.notna() & ((hunter & (britain | ireland)) | festival), 1.)
        flags["race_amateur_riders"] = amateur.mask(suspended & amateur.eq(1), 0.)
    if "race_id" in frame:
        flags = flags.groupby(frame["race_id"], sort=False).transform("max").fillna(flags)
    return flags


def distance_category(values):
    """Shared historical distance bins, including the existing unknown fallback."""
    return pd.cut(values, bins=[0, 6.5, 8.5, 11.5, 50], labels=[0, 1, 2, 3],
                  include_lowest=True).astype(float).fillna(1)


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
    canonical = {s.casefold(): s for s in ("Fast", "Good", "Soft", "Heavy", "Yielding", "Standard", "Slow", "Unknown")}
    if text in canonical:
        return canonical[text]
    main = text.split("(", 1)[0].strip()
    main = re.sub(r"\s*/\s*", "/", main)
    # All-weather goings stay apart from turf; British Good to Soft shares
    # Irish Yielding's group.
    groups = {"standard": "Standard", "standard/fast": "Standard", "standard/slow": "Slow",
        "firm": "Fast", "good to firm": "Fast", "good": "Good",
        "good to soft": "Yielding", "soft": "Soft", "soft to heavy": "Heavy", "heavy": "Heavy",
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
