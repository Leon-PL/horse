"""
Data Processor Module
=====================
Cleans and preprocesses raw horse racing data for feature engineering.

Handles:
- Data type conversions and validation
- Missing value imputation
- Outlier detection and handling
- Data normalization and encoding
"""

import logging

import numpy as np
import pandas as pd
from src.source_dataset import processing_inputs
from src.feature_schema import CATEGORICAL_FIELDS, FIXED_DOMAINS
from src.utils import normalise_off_time_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Title-cased sex codes (results use letters; full words kept for safety) to
#: the sex at birth. Anything else becomes missing rather than a new category.
_BIRTH_SEX = {
    "F": "Female", "M": "Female", "Filly": "Female", "Mare": "Female",
    "C": "Male", "G": "Male", "H": "Male", "R": "Male",
    "Colt": "Male", "Gelding": "Male", "Horse": "Male", "Rig": "Male",
}


def _off_time_to_seconds(off_time: pd.Series) -> pd.Series:
    """Parse off-time strings to seconds after midnight with safe fallbacks."""
    s = off_time.astype(str).str.strip()
    parts = s.str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?")
    h = pd.to_numeric(parts["h"], errors="coerce")
    m = pd.to_numeric(parts["m"], errors="coerce")
    sec = pd.to_numeric(parts["s"], errors="coerce").fillna(0)

    bad = h.isna() | m.isna()
    if bad.any():
        try:
            dt = pd.to_datetime(s[bad], errors="coerce", format="mixed")
        except TypeError:
            dt = pd.to_datetime(s[bad], errors="coerce")
        h.loc[bad] = dt.dt.hour
        m.loc[bad] = dt.dt.minute
        sec.loc[bad] = dt.dt.second

    h = h.fillna(12).clip(0, 23)
    m = m.fillna(0).clip(0, 59)
    sec = sec.fillna(0).clip(0, 59)
    return h * 3600 + m * 60 + sec


def load_raw_data() -> pd.DataFrame:
    """Load the selected dataset's accepted results."""
    from src.database import read_results_snapshot
    from src.source_dataset import source_path
    path = source_path("sl", "results")
    if not path.exists():
        raise FileNotFoundError(
            f"The selected source dataset has no results at {path}. "
            "Collect and publish first: python -m src.ingestion --sync 14"
        )

    df = read_results_snapshot(path)
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def load_database_history() -> pd.DataFrame:
    """Current database rows, typed exactly as the published snapshot is.

    The store's history is keyed on those types, so the raw SQLite frame would
    hash differently even when the data is identical.
    """
    from src.database import load_from_database, results_snapshot_frame
    return results_snapshot_frame(load_from_database())


_PROFILE_FEATURE_INPUTS = {
    "profile_exact_age_days": "foaled",
    "profile_foaling_day_of_year": "foaled",
    "rtv_profile_foaled_abroad": "rtv_country_of_foaling",
    "rtv_profile_known_overseas_starts": "rtv_profile_overseas_starts",
    "rtv_profile_days_since_overseas_start": "rtv_profile_last_overseas_start",
}


def _profile_features(df: pd.DataFrame, pedigree: pd.DataFrame) -> pd.DataFrame:
    """Materialise only facts from a profile captured before the race cutoff."""
    result = pd.DataFrame(np.nan, index=df.index, columns=_PROFILE_FEATURE_INPUTS, dtype="float32")
    if not {"rtv_foaled_observed_at", "sl_foaled_observed_at"}.intersection(pedigree) or not {"horse_id", "race_date"}.issubset(df):
        return result
    ids = df.horse_id.astype(str)
    day = pd.to_datetime(df.race_date, utc=True, errors="coerce").dt.normalize()
    off = df.off_time.map(normalise_off_time_key) if "off_time" in df else pd.Series("", index=df.index)
    scheduled = pd.to_datetime(day.dt.strftime("%Y-%m-%d") + " " + off.where(off.ne(""), "00:00"),
                               format="%Y-%m-%d %H:%M", utc=True, errors="coerce")
    cutoff = scheduled.fillna(day) - pd.Timedelta(minutes=5)
    if "prediction_as_of" in df:
        as_of = pd.to_datetime(df.prediction_as_of, utc=True, errors="raise")
        cutoff = cutoff.mask(as_of.lt(cutoff), as_of)
    fields = pedigree.reindex(columns=dict.fromkeys([
        *_PROFILE_FEATURE_INPUTS.values(), "foaling_date_source", "foaled_sporting_life",
        "rtv_foaled_observed_at", "sl_foaled_observed_at",
    ]))
    observed = pd.to_datetime(ids.map(fields.rtv_foaled_observed_at), utc=True, errors="coerce")
    eligible = observed.le(cutoff)
    # Sporting Life's retained capture timestamps are local and have no offset.
    # Use an earlier recorded day so same-day evidence cannot pass on a guessed time.
    sl_capture_day = pd.to_datetime(ids.map(fields.sl_foaled_observed_at).astype("string").str[:10],
                                    format="%Y-%m-%d", utc=True, errors="coerce")
    sl_birth = pd.to_datetime(ids.map(fields.foaled_sporting_life), utc=True, errors="coerce")
    birth = pd.to_datetime(ids.map(fields.foaled), utc=True, errors="coerce")
    age = (day - birth).dt.days
    rtv_exact = eligible & ids.map(fields.foaling_date_source).eq("racing_tv")
    sl_exact = sl_capture_day.lt(cutoff.dt.normalize()) & sl_birth.eq(birth)
    exact = (rtv_exact | sl_exact) & ~(birth.dt.month.eq(1) & birth.dt.day.eq(1)) & age.gt(0)
    result["profile_exact_age_days"] = age.where(exact).astype("float32")
    result["profile_foaling_day_of_year"] = birth.dt.dayofyear.where(exact).astype("float32")

    country = ids.map(fields.rtv_country_of_foaling).astype("string").str.upper()
    known_country = eligible & country.str.fullmatch(r"[A-Z]{3}", na=False) & ~country.isin(["UNK", "XXX"])
    result["rtv_profile_foaled_abroad"] = (~country.isin(["GBR", "IRE"])).where(known_country).astype("float32")

    starts = pd.to_numeric(ids.map(fields.rtv_profile_overseas_starts), errors="coerce")
    last = pd.to_datetime(ids.map(fields.rtv_profile_last_overseas_start), utc=True, errors="coerce")
    known_form = eligible & starts.gt(0) & last.lt(observed.dt.normalize()) & last.lt(day)
    result["rtv_profile_known_overseas_starts"] = starts.where(known_form).astype("float32")
    result["rtv_profile_days_since_overseas_start"] = (day-last).dt.days.where(known_form).astype("float32")
    return result


def apply_pedigree(df: pd.DataFrame) -> pd.DataFrame:
    """Fill sire/dam/damsire (and foaled) from the selected dataset's pedigree."""
    if "horse_id" not in df.columns:
        return df
    from src.source_dataset import read_table
    ped = read_table("pedigree", "horses")
    if ped.empty:
        return df.assign(**_profile_features(df, ped))
    ped["horse_id"] = ped["horse_id"].astype(str)
    # Plain mapping avoids merge suffix headaches with existing null columns.
    idx = ped.set_index("horse_id")
    hid = df["horse_id"].astype(str)
    for col in ("sire", "dam", "damsire"):
        mapped = hid.map(idx[col])
        if col in df.columns:
            df[col] = df[col].where(df[col].notna() & (df[col] != ""), mapped)
        else:
            df[col] = mapped
    if "foaled" not in df.columns:
        df["foaled"] = hid.map(idx["foaled"])
    coverage = df["sire"].notna().mean()
    logger.info("Pedigree applied: %.1f%% of rows have a sire", 100 * coverage)
    return df.assign(**_profile_features(df, idx))


# ── Going normalisation map ──────────────────────────────────────────────
# Sporting Life uses 20+ going descriptions.  We collapse them into five turf
# groups and two all-weather groups so one-hot encoding creates dense,
# meaningful columns. All-weather "Standard" is not turf Good to Firm, and a
# horse's record on one says nothing about the other. Irish Yielding is the
# official counterpart of British Good to Soft, so both share "Yielding".
_GOING_MAP: dict[str, str] = {}
for _g in [
    "Good To Firm", "Good (Good To Firm In Places)",
    "Good To Firm (Good In Places)", "Good To Firm (Firm In Places)",
    "Firm", "Firm (Good To Firm In Places)",
]:
    _GOING_MAP[_g.lower()] = "Fast"
for _g in [
    "Good",
]:
    _GOING_MAP[_g.lower()] = "Good"
for _g in [
    "Standard", "Standard / Fast",
]:
    _GOING_MAP[_g.lower()] = "Standard"
for _g in [
    "Standard / Slow", "Slow",
]:
    _GOING_MAP[_g.lower()] = "Slow"
for _g in [
    "Good To Soft", "Good (Good To Soft In Places)",
    "Good To Soft (Good In Places)", "Good To Soft (Soft In Places)",
]:
    _GOING_MAP[_g.lower()] = "Yielding"
for _g in [
    "Soft", "Soft (Good To Soft In Places)", "Soft (Yielding To Soft In Places)",
]:
    _GOING_MAP[_g.lower()] = "Soft"
for _g in [
    "Heavy", "Soft (Heavy In Places)", "Soft To Heavy",
    "Heavy (Soft In Places)", "Heavy (Heavy To Soft In Places)",
]:
    _GOING_MAP[_g.lower()] = "Heavy"
for _g in [
    "Good To Yielding", "Yielding", "Good (Good To Yielding In Places)",
    "Yielding (Soft In Places)", "Yielding To Soft",
]:
    _GOING_MAP[_g.lower()] = "Yielding"


def _on_distinct(values: pd.Series, compute) -> pd.Series:
    """Run a Series transform once per distinct value and spread the result.

    All missing values count as one, so ``compute`` must treat them alike.
    """
    codes, distinct = pd.factorize(values, use_na_sentinel=False)
    result = compute(pd.Series(distinct, dtype=values.dtype))
    return pd.Series(result.to_numpy()[codes], index=values.index, name=values.name)


def _normalise_going(going: str) -> str:
    """Map a detailed going description to one of the going groups."""
    from src.feature_semantics import normalise_going
    return normalise_going(going, _GOING_MAP)


# ── Form string parsers ──────────────────────────────────────────────────

def _parse_form_last(form_str: str) -> float:
    """Extract last finishing position from form string like '213-41'."""
    try:
        digits = [int(c) for c in str(form_str) if c.isdigit()]
        return float(digits[-1]) if digits else 5.0
    except (ValueError, IndexError):
        return 5.0


def _parse_form_avg(form_str: str) -> float:
    """Calculate average position from form string."""
    try:
        digits = [int(c) for c in str(form_str) if c.isdigit() and int(c) > 0]
        return sum(digits) / len(digits) if digits else 5.0
    except (ValueError, ZeroDivisionError):
        return 5.0


def _parse_form_wins(form_str: str) -> int:
    """Count number of 1s (wins) in form string."""
    try:
        return str(form_str).count("1")
    except Exception:
        return 0


def _parse_form_dnf(form_str: str) -> int:
    """Count non-completion indicators: P(ulled up), F(ell), U(nseated), R(efused)."""
    try:
        s = str(form_str).upper()
        return sum(s.count(c) for c in "PFUR")
    except Exception:
        return 0


def _parse_form_has_break(form_str: str) -> int:
    """1 if form contains '/' indicating a season/time break."""
    try:
        return int("/" in str(form_str))
    except Exception:
        return 0


def _parse_form_trend(form_str: str) -> float:
    """Slope of finishing positions — negative = improving, positive = declining.

    A simple OLS-style slope over the digits in the form string.
    """
    try:
        digits = [int(c) for c in str(form_str) if c.isdigit() and int(c) > 0]
        n = len(digits)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(digits) / n
        num = sum((i - x_mean) * (d - y_mean) for i, d in enumerate(digits))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0
    except Exception:
        return 0.0


def _normalise_person_names(names: pd.Series) -> pd.Series:
    """Canonical form of jockey/trainer names ("j.p. spencer" → "J P Spencer").

    Initials-only names are kept as they are; full first names are never
    manufactured. Idempotent, so cleaned frames can be cleaned again.
    """
    return _on_distinct(names.astype(str), lambda distinct: (
        distinct.str.strip().str.title()
        .str.replace(r'\.(?=[A-Z])', ' ', regex=True)   # "J.P" → "J P"
        .str.replace(r'\.\s*', ' ', regex=True)          # trailing dots
        .str.replace(r'\s{2,}', ' ', regex=True)         # collapse spaces
        .str.strip()
        .str.title()
    ))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate raw race data.

    Steps:
    1. Parse dates
    2. Handle missing values
    3. Standardize string fields
    4. Remove invalid records
    5. Convert data types
    """
    logger.info("Cleaning data...")
    df = df.copy()

    # --- Parse dates ---
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    if df["race_date"].isna().any():
        df = df[df["race_date"].notna()]

    # --- Standardize string fields ---
    string_cols = [
        "horse_name", "jockey", "trainer", "track", "going",
        "race_class", "race_type", "sex",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = _on_distinct(df[col].astype(str), lambda distinct: distinct.str.strip().str.title())

    from src.verified_race_metadata import apply_verified_race_types, apply_verified_runner_ratings
    df = apply_verified_race_types(df)
    df = apply_verified_runner_ratings(df)

    # --- Sex: birth sex only (look-ahead leak, ledger 15) ---
    # Results pages report a horse's sex as of the day they were SCRAPED, not
    # race day (2,842 two-year-olds are coded as mares). The code therefore
    # encodes the horse's future: kept entire (stud value), gelded later, a
    # filly raced on past four. Runners with an impossible race-day code
    # returned +21.5% at BSP. Only the sex at birth survives that, so the
    # gelding/entire distinction is dropped for history and live alike.
    if "sex" in df.columns:
        df["sex"] = df["sex"].map(_BIRTH_SEX)

    # --- Normalise entity names (jockey / trainer) ---
    # Racing Post data contains multiple representations of the same person:
    #   "J P Spencer", "J.P. Spencer", "Jamie Spencer" → "Jamie Spencer"
    # Inconsistent names split rolling histories, halving effective sample
    # size for cumulative features.  We apply a consistent canonical form:
    #   1. Strip punctuation (dots, hyphens) internal to initials
    #   2. Collapse multiple spaces
    #   3. Normalise title-cased initials that appear as a first name
    #      (single letter followed by space) to keep the longer form when
    #      both exist — but since we don't have a lookup table this is a
    #      structural pass only; the scraper provides .title() already.
    for _nc in ("jockey", "trainer"):
        if _nc in df.columns:
            df[_nc] = _normalise_person_names(df[_nc])
    # Sporting Life also renames people outright ("M P Sheehy" → "Mikey
    # Sheehy"); the Racing TV-derived alias table gives each one name.
    from src.verified_race_metadata import apply_connection_aliases
    df = apply_connection_aliases(df, _normalise_person_names)

    # --- Ensure numeric types ---
    numeric_cols = {
        "finish_position": "int",
        "age": "int",
        "weight_lbs": "int",
        "draw": "int",
        "num_runners": "int",
        "distance_furlongs": "float",
        "odds": "float",
        "finish_time_secs": "float",
        "lengths_behind": "float",
        "prize_money": "float",
        "won": "int",
    }
    for col, dtype in numeric_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Historical results and live cards both report the allowance separately
    # from allocated weight. Legacy or unreported claims stay missing.
    from src.feature_semantics import claim_pounds, rider_restriction_features
    df["reported_jockey_claim_lbs"] = claim_pounds(
        df.get("reported_jockey_claim_lbs", pd.Series(np.nan, index=df.index)))
    restrictions = rider_restriction_features(df)
    df[restrictions.columns] = restrictions

    # Sporting Life's raw finish_distance is the gap to the PREVIOUS horse.
    # The legacy raw column is unfortunately named lengths_behind. Preserve
    # it for source replay; all new history calculations use this explicit
    # distance behind the winner. Reprocessing is idempotent.
    if "lengths_behind" in df.columns and "finish_position" in df.columns:
        from src.result_margins import cumulative_beaten_lengths
        df["beaten_lengths"] = cumulative_beaten_lengths(df)
    if "finish_position" in df:
        from src.result_status import result_labels
        df["finish_pos_label"] = result_labels(df)

    # --- Parse form string into numeric features ---
    if "form" in df.columns:
        df["form_str"] = df["form"].astype(str).str.strip()
        parsers = {
            "form_last_pos": _parse_form_last,
            "form_avg": _parse_form_avg,
            "form_wins_count": _parse_form_wins,
            "form_length": lambda x: len([c for c in str(x) if c.isdigit()]),
            "form_dnf_count": _parse_form_dnf,
            "form_has_break": _parse_form_has_break,
            "form_trend": _parse_form_trend,
        }
        codes, distinct = pd.factorize(df["form_str"], use_na_sentinel=False)
        distinct = pd.Series(distinct, dtype=df["form_str"].dtype)
        for name, parse in parsers.items():
            df[name] = distinct.apply(parse).to_numpy()[codes]

    # --- Normalise going descriptions (20+ -> 5 groups) ---
    if "going" in df.columns:
        df["going"] = _on_distinct(df["going"], lambda distinct: distinct.apply(_normalise_going))

    # --- Ensure surface column exists ---
    if "surface" in df.columns:
        df["surface"] = df["surface"].astype(str).str.strip().str.title()
    else:
        df["surface"] = "Turf"

    # --- Ensure handicap column exists ---
    if "handicap" in df.columns:
        df["handicap"] = pd.to_numeric(df["handicap"], errors="coerce").fillna(0).astype(int)
    else:
        df["handicap"] = 0

    # Unrated runners may use their own field's available ratings. An entirely
    # unrated race stays unknown; borrowing the dataset median uses future races
    # and differs when a live card is processed by itself.
    if "official_rating" in df.columns:
        ratings = pd.to_numeric(df["official_rating"], errors="coerce")
        observed = ratings.gt(0)
        if "has_official_rating" in df:
            observed &= pd.to_numeric(df.has_official_rating, errors="coerce").eq(1)
        df["has_official_rating"] = observed.astype(int)
        ratings = ratings.where(observed)
        race_median_or = ratings.groupby(df["race_id"]).transform("median")
        df["official_rating"] = ratings.fillna(race_median_or)

    # --- Handle days_since_last_run ---
    if "days_since_last_run" in df.columns:
        df["days_since_last_run"] = pd.to_numeric(df["days_since_last_run"], errors="coerce").fillna(60)

    # --- Remove invalid records ---
    initial_len = len(df)

    # Must have key fields
    keep = df["horse_name"].notna() & df["race_id"].notna()
    if not keep.all():
        df = df[keep]

    # For results data: finish_position must be positive
    # For racecard data (predictions): finish_position may be 0
    if "finish_position" in df.columns:
        # Keep entries with position 0 (racecards) or positive (results)
        df = df[df["finish_position"] >= 0]

    # Preserve unavailable prices. An invented median price contaminates
    # market normalisation, the market anchor, and simulated settlement.
    if "odds" in df.columns:
        df["odds"] = df["odds"].where(
            np.isfinite(df["odds"]) & (df["odds"] >= 1.01)
        )

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} invalid records")

    # --- Handle missing numeric values ---
    if "weight_lbs" in df.columns:
        # Keep impossible source values out of both today's field and future
        # weight-change histories. No guessed replacement of the raw record.
        df["weight_lbs"] = df["weight_lbs"].where(df["weight_lbs"].between(50, 220))
        observed = df["weight_lbs"].notna()
        if "has_observed_weight" in df:
            observed &= pd.to_numeric(df.has_observed_weight, errors="coerce").eq(1)
        df["has_observed_weight"] = observed.astype(np.int8)
        df["weight_lbs"] = df["weight_lbs"].where(observed)
        field_weight = df.groupby("race_id")["weight_lbs"].transform("median")
        df["weight_lbs"] = df["weight_lbs"].fillna(field_weight)

    # Missing stalls are unknown, not an average draw from unrelated races.

    if "prize_money" in df.columns:
        df["prize_money"] = df["prize_money"].fillna(0)
        from src.prize_currency_features import normalise_prize_money
        df = normalise_prize_money(df)

    logger.info(f"Clean data: {len(df)} records remaining")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for model consumption.

    Uses a combination of:
    - Label encoding for high-cardinality fields (horse, jockey, trainer)
    - One-hot encoding for low-cardinality fields (going, race_class, etc.)
    """
    logger.info("Encoding categorical variables...")
    df = df.copy()

    # --- One-hot encode low-cardinality categoricals ---
    low_cardinality = CATEGORICAL_FIELDS
    for col in low_cardinality:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            # A card containing only Flat races or only males still needs the
            # model's Chase/Female indicators, correctly set to zero. Their
            # existence must not depend on unrelated races in the same batch.
            required = FIXED_DOMAINS.get(col, ())
            for category in required:
                name = f"{col}_{category}"
                if name not in dummies:
                    dummies[name] = 0
            df = pd.concat([df, dummies], axis=1)

    # --- Expanding frequency encoding for high-cardinality fields ---
    # Uses cumulative count up to each row so that future data is never
    # used.  The DataFrame must already be sorted by race_date.
    #
    # IMPORTANT: when an entity appears on MORE THAN ONE row of the same race,
    # naive cumcount() produces sequential numbers within that race that
    # correlate with finish-position order (a result leak). This bites two cases:
    #   * race-level vars (e.g. track): identical for every runner -> the whole
    #     race is one cumcount run; collapse per race.
    #   * trainer: a trainer running several horses in one race gets consecutive
    #     cumcounts in finishing order across its stablemates -> collapse per
    #     (race, trainer) so all stablemates share one pre-race count.
    # horse_name/jockey appear once per race, so the per-(race, entity) collapse
    # is a no-op for them. We collapse ALL high-cardinality freqs per (race,
    # entity) min (== first/pre-race count) to be uniformly leak-safe.
    race_level_vars = {"track"}
    high_cardinality = ["horse_name", "jockey", "trainer", "track"]
    for col in high_cardinality:
        if col in df.columns:
            df[f"{col}_freq"] = df.groupby(col).cumcount() + 1
            if "race_id" in df.columns:
                grp = "race_id" if col in race_level_vars else ["race_id", col]
                df[f"{col}_freq"] = df.groupby(grp)[f"{col}_freq"].transform("min")

    logger.info("Categorical encoding complete")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from race date."""
    df = df.copy()

    if "race_date" in df.columns:
        df["month"] = df["race_date"].dt.month
        df["day_of_week"] = df["race_date"].dt.dayofweek
        df["day_of_year"] = df["race_date"].dt.dayofyear
        df["year"] = df["race_date"].dt.year

        # Season encoding
        df["season"] = df["month"].map(
            {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Autumn", 10: "Autumn", 11: "Autumn",
            }
        )
        season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
        df = pd.concat([df, season_dummies], axis=1)
        df = df.drop(columns=["season"])

    return df


@processing_inputs
def process_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full data processing pipeline.

    Args:
        df: Raw DataFrame (if None, loads from file)

    Returns:
        Processed DataFrame ready for feature engineering
    """
    if df is None:
        df = load_raw_data()

    # Results/racecard APIs never carry breeding; fill sire/dam/damsire from
    # the selected pedigree (no-op when the dataset has none). A selected
    # accepted input must never silently disappear, so errors propagate.
    df = apply_pedigree(df)
    from src.rtv_features import add_rider_claims
    df = clean_data(add_rider_claims(df))

    # Chronological ordering is required before any cumulative encodings.
    if "race_date" in df.columns:
        race_dt = pd.to_datetime(df["race_date"], errors="coerce")
        if "off_time" in df.columns:
            off_secs = _off_time_to_seconds(df["off_time"]).astype(float)
            df["_event_dt"] = race_dt + pd.to_timedelta(off_secs, unit="s")
        else:
            df["_event_dt"] = race_dt
        sort_cols = ["_event_dt"] + (["race_id"] if "race_id" in df.columns else [])
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df = add_time_features(df)
    df = encode_categorical(df)
    df = df.drop(columns=["_event_dt"], errors="ignore")

    from src.source_dataset import require_manifest
    manifest = require_manifest()
    if manifest.get("id"):
        df.attrs["dataset_manifest_id"] = manifest["id"]
    return df
