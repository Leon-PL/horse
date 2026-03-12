"""
Utility Functions
=================
Helper functions used across the horse racing prediction application.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def format_odds(odds: float) -> str:
    """Convert decimal odds to fractional display."""
    if odds <= 1:
        return "EVS"
    numerator = odds - 1
    # Common fractional odds
    common_fracs = [
        (0.5, "1/2"), (0.67, "4/6"), (0.8, "4/5"),
        (1.0, "EVS"), (1.5, "6/4"), (2.0, "2/1"),
        (2.5, "5/2"), (3.0, "3/1"), (4.0, "4/1"),
        (5.0, "5/1"), (6.0, "6/1"), (8.0, "8/1"),
        (10.0, "10/1"), (12.0, "12/1"), (14.0, "14/1"),
        (16.0, "16/1"), (20.0, "20/1"), (25.0, "25/1"),
        (33.0, "33/1"), (50.0, "50/1"), (100.0, "100/1"),
    ]
    closest = min(common_fracs, key=lambda x: abs(x[0] - numerator))
    return closest[1]


def implied_probability(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1.0 / odds if odds > 0 else 0.0


def kelly_criterion(
    prob: float,
    odds: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly Criterion bet size.

    Args:
        prob: Estimated win probability
        odds: Decimal odds offered
        fraction: Fraction of full Kelly to use (default 1/4 Kelly)

    Returns:
        Recommended stake as fraction of bankroll
    """
    b = odds - 1  # Net odds
    q = 1 - prob

    kelly = (b * prob - q) / b
    kelly = max(0, kelly)  # Never recommend negative bets

    return kelly * fraction


def calculate_roi(
    predictions: pd.DataFrame,
    stake: float = 1.0,
    strategy: str = "top_pick",
) -> dict:
    """
    Calculate theoretical ROI based on predictions.

    Args:
        predictions: DataFrame with predictions and actual results
        stake: Stake per bet
        strategy: Betting strategy to evaluate

    Returns:
        Dict with ROI statistics
    """
    total_bets = 0
    total_staked = 0
    total_return = 0

    if strategy == "top_pick":
        # Bet on highest predicted probability in each race
        for race_id in predictions["race_id"].unique():
            race = predictions[predictions["race_id"] == race_id]
            top_pick = race.loc[race["win_probability"].idxmax()]

            total_bets += 1
            total_staked += stake
            if top_pick.get("won", 0) == 1:
                total_return += stake * top_pick.get("odds", 2.0)

    elif strategy == "value":
        # Bet when predicted probability > implied probability
        for _, row in predictions.iterrows():
            if row.get("value_score", 0) > 0.05:
                total_bets += 1
                total_staked += stake
                if row.get("won", 0) == 1:
                    total_return += stake * row.get("odds", 2.0)

    profit = total_return - total_staked
    roi = (profit / total_staked * 100) if total_staked > 0 else 0

    return {
        "total_bets": total_bets,
        "total_staked": total_staked,
        "total_return": round(total_return, 2),
        "profit": round(profit, 2),
        "roi_pct": round(roi, 2),
        "win_rate": round(
            (total_return > 0).sum() / total_bets * 100
            if total_bets > 0
            else 0,
            2,
        ) if isinstance(total_return, pd.Series) else 0,
    }


def get_form_string(positions: list[int], n: int = 5) -> str:
    """
    Generate a form string from recent finishing positions.
    E.g., [1, 3, 2, 5, 1] -> "1-3-2-5-1"
    """
    recent = positions[-n:] if len(positions) > n else positions
    return "-".join(str(p) for p in recent)


def load_latest_data() -> Optional[pd.DataFrame]:
    """Load the most recently processed/featured data."""
    for name in ("featured_races", "processed_races"):
        for ext in (".parquet", ".csv"):
            path = os.path.join(config.PROCESSED_DATA_DIR, name + ext)
            if os.path.exists(path):
                if ext == ".parquet":
                    return pd.read_parquet(path, engine="pyarrow")
                return pd.read_csv(path)
    return None


def print_race_prediction(results: pd.DataFrame):
    """Pretty-print race prediction results."""
    print("\n" + "=" * 70)
    print("  RACE PREDICTION")
    print("=" * 70)
    print(
        f"{'Rank':<6}{'Horse':<25}{'Win Prob':<12}"
        f"{'Odds':<10}{'Value':<10}"
    )
    print("-" * 70)

    for _, row in results.iterrows():
        rank = int(row["predicted_rank"])
        name = row["horse_name"][:24]
        prob = f"{row['win_probability']:.1%}"
        odds = f"{row.get('odds', 'N/A')}"
        value = f"{row.get('value_score', 0):+.3f}" if "value_score" in row else "N/A"

        marker = " ⭐" if rank == 1 else ""
        print(f"{rank:<6}{name:<25}{prob:<12}{odds:<10}{value:<10}{marker}")

    print("=" * 70)
