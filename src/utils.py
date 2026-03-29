"""
Utility Functions
=================
Helper functions used across the horse racing prediction application.
"""

import logging

import pandas as pd

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
