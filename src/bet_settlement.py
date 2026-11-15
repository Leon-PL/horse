"""
Bet Selection & Settlement Rules
================================
Single source of truth for the betting-strategy rules shared by:

- the test-set betting simulation     (src/bet_analysis.py : analyse_test_set)
- each-way pick filtering             (src/each_way.py : ew_value_bets)
- the Matchbook paper record          (src/matchbook/paper.py, value_bet_selection)

Strategies
----------
1. **Top pick** — 1 unit on the runner with the highest win probability.
2. **Win value** — stake when the model's edge over the implied
   probability beats an odds-scaled dynamic threshold.
3. **Each-way value** — 2 units (1 win + 1 place) when the place edge
   beats the dynamic threshold, the place leg is +EV, and the win odds
   sit inside the EW band.

Keep every magic number here — never inline them at call sites, or the
research results stop matching live settlement.
"""

import numpy as np

# Each-way bets are only struck inside this win-odds band (inclusive).
EW_MIN_ODDS = 4.0
EW_MAX_ODDS = 51.0
# An each-way bet costs 2 units: 1 on the win leg + 1 on the place leg.
EW_STAKE_UNITS = 2.0
# Base edge for the odds-scaled value threshold (dynamic_value_threshold).
DEFAULT_VALUE_THRESHOLD = 0.05

# Default staking and selection settings shared by the apps and training runs.
DEFAULT_VALUE_CONFIG: dict = {
    "staking_mode": "flat",
    "value_threshold": DEFAULT_VALUE_THRESHOLD,
    "value_min_odds": 1.0,
    "value_max_odds": 101.0,
    "kelly_fraction": 0.25,
    "bankroll": 100.0,
    "ew_enabled": True,
    "ew_fraction": 0.20,
    "ew_min_place_edge": 0.15,
    "ew_min_odds": EW_MIN_ODDS,
    "ew_max_odds": EW_MAX_ODDS,
}


def valid_decimal_odds(odds):
    """A bet needs an observed finite decimal price strictly greater than one."""
    values = np.asarray(odds, dtype=np.float64)
    return np.isfinite(values) & (values > 1.0)


def dynamic_value_threshold(base_threshold, odds):
    """Odds-scaled edge threshold: ``base * sqrt(odds / 3)``.

    Tighter at short odds (where calibration is better), looser at long
    odds (where small edges are masked by variance).  Accepts a scalar
    or an array/Series of decimal odds; returns the matching shape.
    """
    odds_arr = np.clip(np.asarray(odds, dtype=np.float64), 1.0, None)
    out = float(base_threshold) * np.sqrt(odds_arr / 3.0)
    return float(out) if out.ndim == 0 else out


def value_bet_selection(
    probs,
    odds,
    value_threshold: float,
    implied_prob=None,
) -> dict:
    """Return the value-bet mask plus edge/CLV diagnostics.

    ``implied_prob`` defaults to raw ``1/odds`` (consistent with the
    strategy calibrator).  The Matchbook paper record passes the
    race-normalised 11:00 share instead; supply that argument explicitly to keep
    an alternative convention.
    """
    probs_arr = np.asarray(probs, dtype=np.float64)
    odds_arr = np.asarray(odds, dtype=np.float64)
    if implied_prob is None:
        implied = np.divide(
            1.0, odds_arr, out=np.zeros_like(odds_arr), where=odds_arr > 0,
        )
    else:
        implied = np.asarray(implied_prob, dtype=np.float64)
    edge = probs_arr - implied
    dyn_threshold = dynamic_value_threshold(value_threshold, odds_arr)
    clv = probs_arr * odds_arr
    expected_roi = clv - 1.0
    mask = (
        valid_decimal_odds(odds_arr)
        & np.isfinite(probs_arr)
        & (edge > dyn_threshold)
    )
    return {
        "mask": mask,
        "implied_prob": implied,
        "edge": edge,
        "dyn_threshold": dyn_threshold,
        "clv": clv,
        "expected_roi": expected_roi,
    }


def settle_win_bet(odds: float, won, stake: float = 1.0) -> float:
    """PnL of a single win bet at decimal *odds* for *stake* units."""
    return stake * (float(odds) - 1.0) if won else -float(stake)


def settle_win_bets(odds, won, stake: float = 1.0) -> np.ndarray:
    """Vectorised :func:`settle_win_bet` over arrays of odds / won flags."""
    odds_arr = np.asarray(odds, dtype=np.float64)
    won_arr = np.asarray(won).astype(bool)
    return np.where(won_arr, stake * (odds_arr - 1.0), -float(stake))


def ew_odds_in_band(
    odds,
    min_odds: float = EW_MIN_ODDS,
    max_odds: float = EW_MAX_ODDS,
) -> bool:
    """True when win odds sit inside the (inclusive) each-way band."""
    try:
        odds_f = float(odds)
    except (TypeError, ValueError):
        return False
    return min_odds <= odds_f <= max_odds


def ew_bet_selected(
    place_edge: float,
    place_ev: float,
    odds: float,
    base_threshold: float,
) -> bool:
    """Each-way selection rule: place edge beats the dynamic threshold
    and the place leg alone is +EV."""
    return (
        float(place_edge) > dynamic_value_threshold(base_threshold, odds)
        and float(place_ev) > 0
    )


def ew_placed_flag(finish_position, places_paid) -> int:
    """1 when *finish_position* earns the place leg.

    Non-finishers (position 0, NaN, or missing) never place.
    """
    try:
        fp = int(finish_position)
    except (TypeError, ValueError):
        return 0
    return int(0 < fp <= int(places_paid))


def placed_mask(finish_positions, places_paid) -> np.ndarray:
    """Vectorised :func:`ew_placed_flag`: 1 where a runner earns the place leg.

    Non-finishers are stored as position 0 and never place. ``fp <= places``
    alone counts them as placed, which is how leaving them out of training and
    evaluation once looked harmless.
    """
    fp = np.asarray(finish_positions, dtype=np.float64)
    pp = np.broadcast_to(np.asarray(places_paid, dtype=np.float64), fp.shape)
    return ((fp >= 1) & (fp <= pp)).astype(int)  # NaN compares False: never placed


def settle_ew_bet(win_odds, place_odds, won, placed) -> float:
    """PnL of a 2-unit each-way bet (1 win + 1 place).

    Both legs pay when the horse wins; only the place leg pays when it
    places without winning.
    """
    pnl = -EW_STAKE_UNITS
    if won:
        pnl += float(win_odds) + float(place_odds)
    elif placed:
        pnl += float(place_odds)
    return pnl


# ── Exchange settlement (win and place markets) ───────────────────────
# Exchange bets are settled on Betfair prices (BSP or LTP), NOT bookmaker
# each-way terms. Winnings are netted by commission (Betfair charges commission
# on net market winnings); losers simply lose the stake. These are the single
# formulas shared by the walk-forward backtests and the research harnesses in
# scripts/ — keep them here so all of them agree (hard rule #2). Four research
# scripts
# previously carried their own `COMM = 0.02` and open-coded this expression,
# which is exactly the drift that makes two ROI figures incomparable.
EXCHANGE_COMMISSION = 0.02  # operator-confirmed Betfair commission

#: Backwards-compatible alias; the rate is the same on both markets.
PLACE_EXCHANGE_COMMISSION = EXCHANGE_COMMISSION


def settle_win_exchange_bet(odds, won, stake: float = 1.0,
                            commission: float = EXCHANGE_COMMISSION) -> float:
    """PnL of a single win bet struck at exchange decimal *odds* for *stake*."""
    if won:
        return float(stake) * (float(odds) - 1.0) * (1.0 - float(commission))
    return -float(stake)


def settle_win_exchange_bets(odds, won, stake: float = 1.0,
                             commission: float = EXCHANGE_COMMISSION) -> np.ndarray:
    """Vectorised :func:`settle_win_exchange_bet` (stake may be scalar or array)."""
    odds_arr = np.asarray(odds, dtype=np.float64)
    won_arr = np.asarray(won).astype(bool)
    stake_arr = np.asarray(stake, dtype=np.float64)
    return np.where(won_arr, stake_arr * (odds_arr - 1.0) * (1.0 - commission), -stake_arr)


def settle_place_exchange_bet(place_odds, placed, stake: float = 1.0,
                              commission: float = PLACE_EXCHANGE_COMMISSION) -> float:
    """PnL of a single place bet at exchange decimal *place_odds* for *stake*."""
    if placed:
        return float(stake) * (float(place_odds) - 1.0) * (1.0 - float(commission))
    return -float(stake)


def settle_place_exchange_bets(place_odds, placed, stake: float = 1.0,
                               commission: float = PLACE_EXCHANGE_COMMISSION) -> np.ndarray:
    """Vectorised :func:`settle_place_exchange_bet` over arrays (stake may be
    scalar or array-like)."""
    odds_arr = np.asarray(place_odds, dtype=np.float64)
    placed_arr = np.asarray(placed).astype(bool)
    stake_arr = np.asarray(stake, dtype=np.float64)
    return np.where(placed_arr, stake_arr * (odds_arr - 1.0) * (1.0 - commission), -stake_arr)


# ---------------------------------------------------------------------------
# Striking before the off: the exchange price grid and non-runner reductions
# ---------------------------------------------------------------------------
# An early last-traded price (LTP) is not a guaranteed back price: the last
# trade may have been a layer taking the back side, a tick above what a backer
# can get. And a back bet matched before a runner is withdrawn has its odds
# cut by that runner's reduction factor, while BSP (struck after every
# withdrawal) is not - so comparing an early price with BSP without the
# reduction flatters the early price. Both matter to any "take the early
# price" rule (EXPLORATION_PLAN B; ledgers 14 and 19 applied neither).

#: Betfair's odds ladder: (upper bound of band, increment within the band).
BETFAIR_TICK_BANDS = (
    (2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2),
    (20.0, 0.5), (30.0, 1.0), (50.0, 2.0), (100.0, 5.0), (1000.0, 10.0),
)
BETFAIR_MIN_ODDS = 1.01


def betfair_tick_down(odds, ticks: int = 1) -> np.ndarray:
    """*odds* moved *ticks* steps down Betfair's ladder (floored at 1.01).

    The increment is the one just below the price, so 2.0 -> 1.99 and
    10.0 -> 9.8. Off-ladder prices (a BSP) step down by the increment of the
    band they fall in.
    """
    out = np.asarray(odds, dtype=np.float64).copy()
    uppers = np.array([b for b, _ in BETFAIR_TICK_BANDS])
    incs = np.array([i for _, i in BETFAIR_TICK_BANDS])
    for _ in range(int(ticks)):
        band = np.searchsorted(uppers, out - 1e-9, side="left")
        inc = incs[np.clip(band, 0, len(incs) - 1)]
        out = np.where(np.isfinite(out), np.round(out - inc, 2), out)
        out = np.maximum(out, BETFAIR_MIN_ODDS)
    return out


def apply_reduction_factor(odds, factor) -> np.ndarray:
    """Matched back odds after non-runner reductions.

    *factor* is the product of ``(1 - reduction_factor/100)`` over runners
    withdrawn after the bet was matched (1.0 = no withdrawal). Betfair cuts
    the decimal odds by the factor, floored at 1.01.
    """
    reduced = np.asarray(odds, dtype=np.float64) * np.asarray(factor, dtype=np.float64)
    return np.maximum(reduced, BETFAIR_MIN_ODDS)
