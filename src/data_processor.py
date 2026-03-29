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

import os
import logging

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """Load raw race data from CSV."""
    if filepath is None:
        filepath = os.path.join(config.RAW_DATA_DIR, "race_results.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No data file found at {filepath}. "
            "Run data collection first: python -m src.data_collector"
        )

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df


# ── Going normalisation map ──────────────────────────────────────────────
# Sporting Life uses 20+ going descriptions.  We collapse them into five
# broad groups so one-hot encoding creates dense, meaningful columns.
_GOING_MAP: dict[str, str] = {}
for _g in [
    "Standard", "Good To Firm", "Good (Good To Firm In Places)",
    "Good To Firm (Good In Places)", "Good To Firm (Firm In Places)",
    "Firm", "Firm (Good To Firm In Places)",
]:
    _GOING_MAP[_g.lower()] = "Fast"
for _g in [
    "Good", "Standard / Slow",
]:
    _GOING_MAP[_g.lower()] = "Good"
for _g in [
    "Good To Soft", "Soft", "Good (Good To Soft In Places)",
    "Good To Soft (Good In Places)", "Good To Soft (Soft In Places)",
    "Soft (Good To Soft In Places)",
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
    "Soft (Yielding To Soft In Places)",
]:
    _GOING_MAP[_g.lower()] = "Yielding"


def _normalise_going(going: str) -> str:
    """Map a detailed going description to one of five groups."""
    return _GOING_MAP.get(str(going).strip().lower(), "Good")


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
    df = df.dropna(subset=["race_date"])

    # --- Standardize string fields ---
    string_cols = [
        "horse_name", "jockey", "trainer", "track", "going",
        "race_class", "race_type", "sex",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

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
    import re as _re
    _INITIAL_PREFIX = _re.compile(r'^([A-Z](?:\s[A-Z])*\s)(.+)$')
    for _nc in ("jockey", "trainer"):
        if _nc not in df.columns:
            continue
        # Remove stray internal dots/hyphens used in initials ("J.P." → "J P")
        df[_nc] = (
            df[_nc]
            .str.replace(r'\.(?=[A-Z])', ' ', regex=True)   # "J.P" → "J P"
            .str.replace(r'\.\s*', ' ', regex=True)          # trailing dots
            .str.replace(r'\s{2,}', ' ', regex=True)         # collapse spaces
            .str.strip()
        )
        # Where a name is just initials + surname ("J P Spencer") keep it as-is
        # — we don't manufacture full first names.  But ensure consistent
        # title-casing so "j p spencer" == "J P Spencer".
        df[_nc] = df[_nc].str.title()

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

    # --- Parse form string into numeric features ---
    if "form" in df.columns:
        df["form_str"] = df["form"].astype(str).str.strip()
        df["form_last_pos"] = df["form_str"].apply(_parse_form_last)
        df["form_avg"] = df["form_str"].apply(_parse_form_avg)
        df["form_wins_count"] = df["form_str"].apply(_parse_form_wins)
        df["form_length"] = df["form_str"].apply(lambda x: len([c for c in str(x) if c.isdigit()]))
        df["form_dnf_count"] = df["form_str"].apply(_parse_form_dnf)
        df["form_has_break"] = df["form_str"].apply(_parse_form_has_break)
        df["form_trend"] = df["form_str"].apply(_parse_form_trend)

    # --- Normalise going descriptions (20+ -> 5 groups) ---
    if "going" in df.columns:
        df["going"] = df["going"].apply(_normalise_going)

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

    # --- Handle official_rating zeros ---
    # Many non-handicap runners have OR = 0 (unrated).  Replace with the
    # race-field median so the model doesn't confuse 'unrated' with 'low'.
    if "official_rating" in df.columns:
        df["official_rating"] = pd.to_numeric(
            df["official_rating"], errors="coerce"
        ).astype(float).fillna(0.0)
        df["has_official_rating"] = (df["official_rating"] > 0).astype(int)
        # Fill 0s with per-race median (or global median as fallback)
        global_median_or = df.loc[
            df["official_rating"] > 0, "official_rating"
        ].median()
        if pd.isna(global_median_or):
            global_median_or = 0
        race_median_or = df.groupby("race_id")["official_rating"].transform(
            lambda x: x[x > 0].median() if (x > 0).any() else global_median_or
        )
        mask_zero = df["official_rating"] == 0
        df.loc[mask_zero, "official_rating"] = race_median_or[mask_zero]
        df["official_rating"] = df["official_rating"].fillna(global_median_or)

    # --- Handle days_since_last_run ---
    if "days_since_last_run" in df.columns:
        df["days_since_last_run"] = pd.to_numeric(df["days_since_last_run"], errors="coerce").fillna(60)

    # --- Remove invalid records ---
    initial_len = len(df)

    # Must have key fields
    df = df.dropna(subset=["horse_name", "race_id"])

    # For results data: finish_position must be positive
    # For racecard data (predictions): finish_position may be 0
    if "finish_position" in df.columns:
        # Keep entries with position 0 (racecards) or positive (results)
        df = df[df["finish_position"] >= 0]

    # Odds: replace 0 or negative with NaN, then fill with median
    if "odds" in df.columns:
        df.loc[df["odds"] <= 0, "odds"] = np.nan
        median_odds = df["odds"].median()
        if pd.isna(median_odds):
            median_odds = 5.0
        df["odds"] = df["odds"].fillna(median_odds)

    # Implausibly short odds (< 1.01) indicate a data error — an SP of 1.01
    # or below means the horse was a near-certainty, which almost never occurs
    # legitimately and typically reflects scraping artefacts.  Clamp rather
    # than drop so we don't remove the whole race.
    if "odds" in df.columns:
        implausible_odds_mask = df["odds"] < 1.01
        if implausible_odds_mask.any():
            df.loc[implausible_odds_mask, "odds"] = np.nan
            # Re-fill with race median to avoid NaN propagation
            race_med = df.groupby("race_id")["odds"].transform(
                lambda x: x.median() if x.notna().any() else 5.0
            )
            df["odds"] = df["odds"].fillna(race_med).fillna(5.0)
            logger.info(
                f"Clamped {implausible_odds_mask.sum()} implausible odds values (< 1.01)"
            )

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} invalid records")

    # --- Handle missing numeric values ---
    if "weight_lbs" in df.columns:
        df["weight_lbs"] = df["weight_lbs"].fillna(df["weight_lbs"].median())

    if "draw" in df.columns:
        df["draw"] = df["draw"].fillna(df["draw"].median())

    if "prize_money" in df.columns:
        df["prize_money"] = df["prize_money"].fillna(0)

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
    low_cardinality = ["going", "race_class", "race_type", "region", "sex", "surface"]
    for col in low_cardinality:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)

    # --- Expanding frequency encoding for high-cardinality fields ---
    # Uses cumulative count up to each row so that future data is never
    # used.  The DataFrame must already be sorted by race_date.
    #
    # IMPORTANT: race-level variables (e.g. track) are the same for
    # every runner in a race, so naive cumcount() produces sequential
    # numbers within a race that correlate with finish-position order.
    # We collapse to one value per race to prevent leakage.
    race_level_vars = {"track"}
    high_cardinality = ["horse_name", "jockey", "trainer", "track"]
    for col in high_cardinality:
        if col in df.columns:
            df[f"{col}_freq"] = df.groupby(col).cumcount() + 1
            if col in race_level_vars:
                # All runners in the same race get the SAME value
                # (the first/minimum cumcount in that group).
                df[f"{col}_freq"] = df.groupby("race_id")[f"{col}_freq"].transform("min")

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


def process_data(
    df: pd.DataFrame = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Full data processing pipeline.

    Args:
        df: Raw DataFrame (if None, loads from file)
        save: Whether to save the processed data

    Returns:
        Processed DataFrame ready for feature engineering
    """
    if df is None:
        df = load_raw_data()

    df = clean_data(df)

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

    if save:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "processed_races.parquet")
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info(f"Saved processed data to {output_path}")

    return df


if __name__ == "__main__":
    df = process_data()
    print(f"\nProcessed dataset shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nSample:\n{df.head()}")
