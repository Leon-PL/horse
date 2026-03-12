"""
Each-Way Value Calculator
=========================
Identifies value in each-way betting by comparing the model's predicted
place probability to the implied place probability from standard EW terms.

Key insight
-----------
Bookmaker each-way terms apply a **fixed fraction** (1/4 or 1/5) of the
win odds to every runner, regardless of ability.  But the true ratio of
place probability to win probability varies hugely across runners:

    - Favourites:  P(place) ≈ 2× P(win)  → EW terms assume ≈ 4×
    - Outsiders:   P(place) ≈ 4–6× P(win) → EW terms assume ≈ 4×

This creates a structural edge on outsiders' place legs, especially in
large, competitive handicap fields.

Standard UK Each-Way Rules
--------------------------
| Runners | Places Paid | Fraction |
|---------|-------------|----------|
| ≤4      | —           | —        |  (win only)
| 5–7     | 1st, 2nd    | 1/4      |
| 8–11    | 1st, 2nd, 3rd | 1/4   |
| 12–15   | 1st, 2nd, 3rd | 1/4   |
| 16+ (handicap) | 1st–4th | 1/4  |
| 12–15 (handicap) | 1st–3rd | 1/4 |

Some races (large-field handicaps at festivals) offer 1/5 terms with
extra places — these are treated as promotions and can be configured.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Place-Probability Adjustment ──────────────────────────────────────

def adjust_place_probs_for_race(
    place_probs: np.ndarray,
    win_probs: np.ndarray,
    places_paid: int,
    model_k: int = 3,
) -> np.ndarray:
    """Adjust model P(top-3) → P(top-*places_paid*) for a single race.

    Two-step process:

    1. **Power scaling** – shifts probability mass towards favourites
       when fewer places are paid (``exponent > 1``) or spreads it when
       more places are paid (``exponent < 1``).  Uses the ratio
       ``model_k / places_paid`` as the exponent, which is a Harville-
       inspired approximation.

    2. **Per-race normalisation** – rescales so the adjusted
       probabilities sum to ``places_paid`` within the race.  This
       corrects any systematic over/under-prediction from the model.

    Finally, each horse's place probability is floored at its win
    probability (you can't win without placing).

    Args:
        place_probs: Raw model P(top-3) values for each runner.
        win_probs: Per-race softmax win probabilities (lower bound).
        places_paid: Actual number of places paid by EW terms.
        model_k: The *k* the model was trained on (default 3).

    Returns:
        Adjusted and normalised place probabilities.
    """
    if places_paid <= 0:
        return np.zeros_like(place_probs)

    n = len(place_probs)
    adjusted = place_probs.astype(np.float64).copy()

    # Per-race normalise → sum ≈ places_paid
    # (Model is trained on variable EW places_paid targets, so no
    #  power-scaling is needed — just normalise and floor.)
    total = adjusted.sum()
    if total > 1e-8:
        adjusted *= places_paid / total
    else:
        adjusted = np.full(n, places_paid / n, dtype=np.float64)

    # Floor: P(place) >= P(win) — you can't win without placing
    adjusted = np.maximum(adjusted, win_probs.astype(np.float64))

    return np.clip(adjusted, 0.0, 1.0)


# ── EW Terms Lookup ───────────────────────────────────────────────────

@dataclass
class EachWayTerms:
    """Represents each-way betting terms for a race."""
    places_paid: int          # e.g. 2, 3, or 4
    fraction: float           # e.g. 0.25 (1/4) or 0.20 (1/5)
    eligible: bool = True     # False if EW not available (≤4 runners)

    @property
    def fraction_str(self) -> str:
        if abs(self.fraction - 0.25) < 0.01:
            return "1/4"
        if abs(self.fraction - 0.20) < 0.01:
            return "1/5"
        return f"{self.fraction:.2f}"


def get_ew_terms(
    num_runners: int,
    is_handicap: bool = False,
    extra_places: int = 0,
    fraction_override: float | None = None,
) -> EachWayTerms:
    """
    Determine standard UK each-way terms for a race.

    Args:
        num_runners: Number of runners in the race.
        is_handicap: Whether the race is a handicap.
        extra_places: Promotional extra places (e.g. +1 for "paying 4 places").
        fraction_override: If set, use this fraction instead of the standard
                           1/4.  Useful for promotions offering 1/5 terms etc.

    Returns:
        EachWayTerms with places_paid, fraction, and eligibility.
    """
    if num_runners <= 4:
        return EachWayTerms(places_paid=0, fraction=0.0, eligible=False)

    if num_runners <= 7:
        places = 2
    elif num_runners <= 11:
        places = 3
    elif num_runners <= 15:
        places = 3
    else:
        # 16+ runners
        if is_handicap:
            places = 4
        else:
            places = 3

    places += extra_places
    fraction = fraction_override if fraction_override is not None else 0.25

    return EachWayTerms(places_paid=places, fraction=fraction, eligible=True)


def place_odds_decimal(win_odds_decimal: float, fraction: float) -> float:
    """
    Calculate decimal place odds from win odds and EW fraction.

    E.g. win odds 11.0 (10/1) at 1/4 terms → place odds = 1 + (11-1)*0.25 = 3.50
    """
    return 1.0 + (win_odds_decimal - 1.0) * fraction


def implied_place_prob(win_odds_decimal: float, fraction: float) -> float:
    """Implied probability of placing from the EW terms."""
    p_odds = place_odds_decimal(win_odds_decimal, fraction)
    return 1.0 / p_odds


# ── EW Value Calculation ─────────────────────────────────────────────

def ew_value(
    win_prob: float,
    place_prob: float,
    win_odds: float,
    ew_terms: EachWayTerms,
) -> dict:
    """
    Calculate the expected value of an each-way bet.

    An each-way bet is TWO bets of equal stake:
      1) Win bet at full win odds
      2) Place bet at (win_odds - 1) × fraction + 1

    Returns a dict with:
      - win_ev:      Expected value of the win leg per unit staked
      - place_ev:    Expected value of the place leg per unit staked
      - ew_ev:       Combined EW expected value per unit total staked (2 units)
      - place_odds:  Decimal place odds
      - place_edge:  place_prob − implied_place_prob
      - place_value: Boolean — is the place leg alone +EV?
      - ew_value:    Boolean — is the combined EW bet +EV?
    """
    if not ew_terms.eligible or win_odds <= 1.0:
        return {
            "win_ev": 0.0,
            "place_ev": 0.0,
            "ew_ev": 0.0,
            "place_odds": 0.0,
            "place_edge": 0.0,
            "place_value": False,
            "ew_value": False,
        }

    p_odds = place_odds_decimal(win_odds, ew_terms.fraction)

    # EV per unit staked on each leg
    win_ev = win_prob * win_odds - 1.0          # per £1 on win
    place_ev = place_prob * p_odds - 1.0        # per £1 on place

    # Combined: you stake 2 units (1 win + 1 place)
    # Total return = win_prob * win_odds + place_prob * p_odds
    # Total cost = 2
    ew_ev = (win_ev + place_ev) / 2.0           # per £1 total outlay

    imp_place_prob = 1.0 / p_odds
    place_edge = place_prob - imp_place_prob

    return {
        "win_ev": win_ev,
        "place_ev": place_ev,
        "ew_ev": ew_ev,
        "place_odds": p_odds,
        "place_edge": place_edge,
        "place_value": place_ev > 0,
        "ew_value": ew_ev > 0,
    }


def kelly_ew(
    win_prob: float,
    place_prob: float,
    win_odds: float,
    ew_terms: EachWayTerms,
    fraction: float = 0.25,
) -> dict:
    """
    Kelly sizing for each-way bets.

    Returns separate Kelly fractions for:
      - win_only: Kelly stake if you only bet win
      - place_only: Kelly stake if you could bet place only
      - ew_kelly: Simplified EW Kelly — average of win & place Kelly,
        scaled down because EW requires equal stakes on both legs.
    """
    if not ew_terms.eligible or win_odds <= 1.0:
        return {"win_kelly": 0.0, "place_kelly": 0.0, "ew_kelly": 0.0}

    p_odds = place_odds_decimal(win_odds, ew_terms.fraction)

    # Win leg Kelly
    b_win = win_odds - 1.0
    k_win = max(0.0, (b_win * win_prob - (1 - win_prob)) / b_win)

    # Place leg Kelly
    b_place = p_odds - 1.0
    k_place = max(0.0, (b_place * place_prob - (1 - place_prob)) / b_place)

    # EW Kelly: Since EW forces equal stakes on both legs, we use the
    # more conservative of the two individual Kellys, weighted by
    # how much each leg contributes to EV.
    if k_win + k_place > 0:
        # Weight by each leg's positive contribution
        w_win = max(0.0, k_win)
        w_place = max(0.0, k_place)
        ew_kelly = min(k_win, k_place) + 0.5 * abs(k_win - k_place)
        # If one leg is -EV, reduce substantially
        if k_win <= 0 or k_place <= 0:
            ew_kelly = max(k_win, k_place) * 0.5
    else:
        ew_kelly = 0.0

    return {
        "win_kelly": k_win * fraction,
        "place_kelly": k_place * fraction,
        "ew_kelly": ew_kelly * fraction,
    }


# ── Batch (DataFrame) Operations ─────────────────────────────────────

def compute_ew_columns(
    df: pd.DataFrame,
    win_prob_col: str = "win_probability",
    place_prob_col: str = "place_probability",
    odds_col: str = "odds",
    num_runners_col: str = "num_runners",
    handicap_col: str = "handicap",
    fraction_override: float | None = None,
) -> pd.DataFrame:
    """
    Add each-way value columns to a predictions DataFrame.

    Args:
        fraction_override: If set, use this EW fraction instead of the
                           standard 1/4 (e.g. 0.20 for 1/5 terms).

    Adds:
      - ew_eligible:    Whether EW betting is available
      - ew_places:      Number of places paid
      - ew_fraction:    Odds fraction (0.25 or 0.20)
      - place_odds:     Decimal place odds
      - place_edge:     place_prob − implied_place_prob
      - place_ev:       Expected value of place leg (per £1)
      - ew_ev:          Combined EW expected value (per £1 total)
      - ew_value:       Boolean — is the EW bet +EV?
      - place_value:    Boolean — is place leg alone +EV?
    """
    df = df.copy()

    # Determine if handicap
    if handicap_col in df.columns:
        is_hcap = df[handicap_col].fillna(0).astype(bool)
    else:
        is_hcap = pd.Series(False, index=df.index)

    # Get num_runners
    if num_runners_col in df.columns:
        n_runners = df[num_runners_col].fillna(0).astype(int)
    else:
        # Fall back: count runners per race_id
        if "race_id" in df.columns:
            n_runners = df.groupby("race_id")["race_id"].transform("count")
        else:
            n_runners = pd.Series(8, index=df.index)  # safe default

    # Ensure required columns exist
    has_odds = odds_col in df.columns
    has_place_prob = place_prob_col in df.columns

    # Initialise output columns
    df["ew_eligible"] = False
    df["ew_places"] = 0
    df["ew_fraction"] = 0.0
    df["ew_fraction_str"] = ""
    df["place_odds"] = 0.0
    df["place_edge"] = 0.0
    df["place_ev"] = 0.0
    df["ew_ev"] = 0.0
    df["ew_value"] = False
    df["place_value"] = False

    if not (has_odds and has_place_prob):
        logger.warning(
            "Missing %s — cannot compute EW values",
            "odds" if not has_odds else "place_probability",
        )
        return df

    for idx in df.index:
        nr = int(n_runners.loc[idx])
        hcap = bool(is_hcap.loc[idx])
        terms = get_ew_terms(nr, is_handicap=hcap, fraction_override=fraction_override)

        df.loc[idx, "ew_eligible"] = terms.eligible
        df.loc[idx, "ew_places"] = terms.places_paid
        df.loc[idx, "ew_fraction"] = terms.fraction
        df.loc[idx, "ew_fraction_str"] = terms.fraction_str if terms.eligible else ""

        if not terms.eligible:
            continue

        win_p = float(df.loc[idx, win_prob_col])
        place_p = float(df.loc[idx, place_prob_col])
        odds = float(df.loc[idx, odds_col])

        if odds <= 1.0 or np.isnan(odds):
            continue

        result = ew_value(win_p, place_p, odds, terms)
        df.loc[idx, "place_odds"] = result["place_odds"]
        df.loc[idx, "place_edge"] = result["place_edge"]
        df.loc[idx, "place_ev"] = result["place_ev"]
        df.loc[idx, "ew_ev"] = result["ew_ev"]
        df.loc[idx, "ew_value"] = result["ew_value"]
        df.loc[idx, "place_value"] = result["place_value"]

    return df


def ew_value_bets(
    df: pd.DataFrame,
    min_place_edge: float = 0.05,
    min_odds: float = 4.0,
    max_odds: float = 51.0,
) -> pd.DataFrame:
    """
    Filter for the best each-way value bets.

    Criteria:
      - EW eligible (5+ runners)
      - Place edge > dynamic threshold (min_place_edge scaled by odds)
      - Odds between min_odds and max_odds
      - Place leg alone is +EV

    Returns a filtered, sorted DataFrame.
    """
    if "ew_eligible" not in df.columns:
        return pd.DataFrame()

    # Dynamic threshold — tighter at short odds, looser at long odds
    _odds = df["odds"].clip(lower=1.0) if "odds" in df.columns else pd.Series(3.0, index=df.index)
    _dyn_thresh = min_place_edge * np.sqrt(_odds / 3.0)

    mask = (
        df["ew_eligible"]
        & (df["place_edge"] > _dyn_thresh)
        & (df["odds"] >= min_odds)
        & (df["odds"] <= max_odds)
        & df["place_value"]
    )

    result = df[mask].copy()

    if result.empty:
        return result

    return result.sort_values("place_edge", ascending=False)
