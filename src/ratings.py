"""
Horse Rating System (Elo-style)
================================
Computes dynamic Elo ratings for horses, jockeys, and trainers based on
head-to-head race results.

The key insight: if Horse A beats Horse B, and Horse B beats Horse C,
then Horse A should be rated above Horse C — *even if they never met*.
Traditional per-horse win-rate features cannot capture this transitive
strength.  Elo ratings propagate strength through the entire population.

Algorithm
---------
After each race, every pair of finishers is compared:

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))   # expected score for A vs B
    S_A = 1 if A beat B, 0.5 if tied, 0 if lost

    R_A_new = R_A + K * (S_A - E_A)

With N runners in a race, each horse is updated against all N-1 opponents.
The K-factor is divided by (N-1) to normalise the update magnitude so a
big-field race doesn't cause wild swings.

Usage::

    from src.ratings import compute_elo_features

    df = compute_elo_features(df)
    # Adds: horse_elo, jockey_elo, trainer_elo (ratings at the *start*
    # of each race, before that race's update).
"""

import logging
from collections import defaultdict

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


def _event_sort_key(df: pd.DataFrame) -> pd.Series:
    """Build event-time key using race_date + off_time when available."""
    race_dt = pd.to_datetime(df["race_date"], errors="coerce")
    if "off_time" not in df.columns:
        return race_dt
    off_secs = _off_time_to_seconds(df["off_time"]).astype(float)
    return race_dt + pd.to_timedelta(off_secs, unit="s")

# ── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_RATING = 1500.0
K_FACTOR = 16.0          # legacy fallback (per-opponent K)
MIN_RACES_FOR_RATING = 0  # include from the first race

# Adaptive K-factor settings (from config, with fallbacks)
K_BASE = getattr(config, "ELO_K_BASE", 32.0)
K_MIN = getattr(config, "ELO_K_MIN", 8.0)
K_DECAY = getattr(config, "ELO_K_DECAY", 0.05)


def _adaptive_k(n_races: int) -> float:
    """
    Compute an adaptive K-factor based on experience.

    Young / unexposed horses should have a higher K (their rating changes
    more with each result), while established horses need a lower K.

    Uses exponential decay:  K = K_MIN + (K_BASE - K_MIN) * exp(-K_DECAY * n)
    """
    return K_MIN + (K_BASE - K_MIN) * np.exp(-K_DECAY * n_races)


# ── Core Elo helpers ──────────────────────────────────────────────────────

def _expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B given their ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_ratings_for_race(
    finish_order: list[tuple[str, int]],
    ratings: dict[str, float],
    k: float = K_FACTOR,
    race_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Update Elo ratings for all runners in a single race.

    Args:
        finish_order: list of (entity_id, finish_position) — position 1 is
                      best.  Entities with the same position are treated as
                      a draw.
        ratings: mutable dict of entity_id → current rating.
        k: K-factor per opponent pair (divided by field-1 internally).
            Used as fallback when race_counts is None.
        race_counts: Optional dict of entity_id → number of prior races.
            When provided, an adaptive K is computed per entity.

    Returns:
        dict mapping entity_id → rating change (delta) for this race.
    """
    n = len(finish_order)
    if n < 2:
        return {}

    deltas: dict[str, float] = defaultdict(float)

    for i, (id_a, pos_a) in enumerate(finish_order):
        r_a = ratings.get(id_a, DEFAULT_RATING)
        # Adaptive K: use entity's experience if available
        if race_counts is not None:
            k_entity = _adaptive_k(race_counts.get(id_a, 0))
        else:
            k_entity = k
        # Normalise K by number of opponents
        k_adj = k_entity / (n - 1)

        for j, (id_b, pos_b) in enumerate(finish_order):
            if i == j:
                continue
            r_b = ratings.get(id_b, DEFAULT_RATING)

            expected = _expected_score(r_a, r_b)

            if pos_a < pos_b:
                actual = 1.0
            elif pos_a == pos_b:
                actual = 0.5
            else:
                actual = 0.0

            deltas[id_a] += k_adj * (actual - expected)

    # Apply deltas
    for entity_id, delta in deltas.items():
        ratings[entity_id] = ratings.get(entity_id, DEFAULT_RATING) + delta

    return dict(deltas)


# ── Public API ────────────────────────────────────────────────────────────

def compute_elo_features(
    df: pd.DataFrame,
    k: float = K_FACTOR,
) -> pd.DataFrame:
    """
    Add Elo rating features to the DataFrame.

    The function processes races **chronologically** and records each
    entity's rating *before* the race takes place (i.e. the rating the
    entity carried into the race).  This avoids any look-ahead leakage.

    New columns added:
        - ``horse_elo``   — horse's Elo rating entering this race
        - ``jockey_elo``  — jockey's rating entering this race
        - ``trainer_elo`` — trainer's rating entering this race
        - ``horse_elo_delta``  — rating change after this race
        - ``jockey_elo_delta`` — jockey rating change after this race

    Args:
        df: DataFrame with ``race_id``, ``race_date``, ``finish_position``,
            ``horse_name``, and optionally ``jockey`` and ``trainer``.
        k: K-factor (higher = more reactive).

    Returns:
        DataFrame with new Elo columns.
    """
    logger.info("Computing Elo ratings (horse, jockey, trainer)...")
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])

    # Sort chronologically using event-time when available.
    df["_event_dt"] = _event_sort_key(df)
    df = df.sort_values(["_event_dt", "race_id"]).reset_index(drop=True)

    # Rating dictionaries — persist across all races
    horse_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    jockey_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)
    trainer_ratings: dict[str, float] = defaultdict(lambda: DEFAULT_RATING)

    # Race-count dictionaries for adaptive K-factor
    horse_race_counts: dict[str, int] = defaultdict(int)
    jockey_race_counts: dict[str, int] = defaultdict(int)
    trainer_race_counts: dict[str, int] = defaultdict(int)

    # Columns to populate
    horse_elo_col = np.full(len(df), DEFAULT_RATING)
    jockey_elo_col = np.full(len(df), DEFAULT_RATING)
    trainer_elo_col = np.full(len(df), DEFAULT_RATING)
    horse_delta_col = np.zeros(len(df))
    jockey_delta_col = np.zeros(len(df))

    has_jockey = "jockey" in df.columns
    has_trainer = "trainer" in df.columns

    # Group by race (preserving chronological order)
    # Pre-group indices: O(N) once instead of O(N) per-race boolean scan.
    race_ids_ordered = df.drop_duplicates("race_id", keep="first")["race_id"].values
    _race_groups = df.groupby("race_id", sort=False)

    # Pre-extract columns as numpy arrays for fast inner-loop access
    _horse_names = df["horse_name"].values
    _finish_pos = df["finish_position"].values if "finish_position" in df.columns else None
    _jockey_names = df["jockey"].values if has_jockey else None
    _trainer_names = df["trainer"].values if has_trainer else None

    for race_id in race_ids_ordered:
        idx = _race_groups.indices[race_id]

        # --- Record *pre-race* ratings (no look-ahead) ---
        # Must happen for ALL races including future/unfinished ones
        # so live racecards get their accumulated Elo ratings.
        for i in idx:
            horse_elo_col[i] = horse_ratings[_horse_names[i]]
            if has_jockey and pd.notna(_jockey_names[i]):
                jockey_elo_col[i] = jockey_ratings[_jockey_names[i]]
            if has_trainer and pd.notna(_trainer_names[i]):
                trainer_elo_col[i] = trainer_ratings[_trainer_names[i]]

        # Skip races without finish positions (e.g. racecards / future)
        if _finish_pos is None:
            continue
        fp_race = _finish_pos[idx]
        valid_mask = np.isfinite(fp_race) & (fp_race > 0)
        if valid_mask.sum() < 2:
            continue

        valid_idx = idx[valid_mask]

        # --- Build finish order for Elo update (array access, no iterrows) ---
        horse_finish = [
            (_horse_names[i], int(fp_race[valid_mask][j]))
            for j, i in enumerate(valid_idx)
        ]

        # --- Update horse ratings ---
        h_deltas = _update_ratings_for_race(
            horse_finish, horse_ratings, k,
            race_counts=horse_race_counts,
        )
        for i in valid_idx:
            horse_name = _horse_names[i]
            horse_delta_col[i] = h_deltas.get(horse_name, 0.0)
            horse_race_counts[horse_name] += 1

        # --- Update jockey ratings ---
        if has_jockey:
            jockey_finish = [
                (_jockey_names[i], int(_finish_pos[i]))
                for i in valid_idx
                if pd.notna(_jockey_names[i])
            ]
            if len(jockey_finish) >= 2:
                j_deltas = _update_ratings_for_race(
                    jockey_finish, jockey_ratings, k,
                    race_counts=jockey_race_counts,
                )
                for i in valid_idx:
                    jname = _jockey_names[i]
                    if pd.notna(jname):
                        jockey_delta_col[i] = j_deltas.get(jname, 0.0)
                        jockey_race_counts[jname] += 1

        # --- Update trainer ratings ---
        if has_trainer:
            trainer_finish = [
                (_trainer_names[i], int(_finish_pos[i]))
                for i in valid_idx
                if pd.notna(_trainer_names[i])
            ]
            if len(trainer_finish) >= 2:
                _update_ratings_for_race(
                    trainer_finish, trainer_ratings, k,
                    race_counts=trainer_race_counts,
                )
                for i in valid_idx:
                    tname = _trainer_names[i]
                    if pd.notna(tname):
                        trainer_race_counts[tname] += 1

    df["horse_elo"] = horse_elo_col
    df["jockey_elo"] = jockey_elo_col
    df["trainer_elo"] = trainer_elo_col
    df["horse_elo_delta"] = horse_delta_col
    df["jockey_elo_delta"] = jockey_delta_col

    # Relative-to-field features (race-aware)
    race_avg_elo = df.groupby("race_id")["horse_elo"].transform("mean")
    race_max_elo = df.groupby("race_id")["horse_elo"].transform("max")
    df["horse_elo_vs_field"] = df["horse_elo"] - race_avg_elo
    df["horse_elo_rank"] = df.groupby("race_id")["horse_elo"].rank(
        ascending=False, method="min"
    )
    df["horse_elo_top"] = (df["horse_elo"] == race_max_elo).astype(int)

    if has_jockey:
        j_avg = df.groupby("race_id")["jockey_elo"].transform("mean")
        df["jockey_elo_vs_field"] = df["jockey_elo"] - j_avg

    if has_trainer:
        t_avg = df.groupby("race_id")["trainer_elo"].transform("mean")
        df["trainer_elo_vs_field"] = df["trainer_elo"] - t_avg

    # Combined strength
    df["combined_elo"] = df["horse_elo"] + 0.3 * df.get("jockey_elo", DEFAULT_RATING)
    combo_avg = df.groupby("race_id")["combined_elo"].transform("mean")
    df["combined_elo_vs_field"] = df["combined_elo"] - combo_avg
    df = df.drop(columns=["_event_dt"], errors="ignore")

    n_horses = len(horse_ratings)
    n_jockeys = len(jockey_ratings) if has_jockey else 0
    n_trainers = len(trainer_ratings) if has_trainer else 0
    logger.info(
        f"Elo complete: {n_horses} horses, {n_jockeys} jockeys, "
        f"{n_trainers} trainers rated"
    )

    return df


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, config

    pq_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.parquet")
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, "featured_races.csv")
    featured_path = pq_path if os.path.exists(pq_path) else csv_path
    if not os.path.exists(featured_path):
        print("❌ No featured data found. Run training first.")
        raise SystemExit(1)

    df = pd.read_parquet(featured_path, engine="pyarrow") if featured_path.endswith(".parquet") else pd.read_csv(featured_path)
    df = compute_elo_features(df)

    print(f"\nRated {df['horse_elo'].nunique()} unique horse Elo values")
    print(f"Elo range: {df['horse_elo'].min():.0f} – {df['horse_elo'].max():.0f}")
    print(f"\nTop 10 horses by final Elo:")
    latest = df.sort_values("race_date").drop_duplicates("horse_name", keep="last")
    top = latest.nlargest(10, "horse_elo")[["horse_name", "horse_elo"]]
    print(top.to_string(index=False))
