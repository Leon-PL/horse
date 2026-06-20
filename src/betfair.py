"""
Betfair Exchange BSP integration
================================

Loader + join helpers for the historical Betfair Starting Price (BSP) and
pre-off traded prices extracted by ``scripts/extract_betfair_bsp.py`` into
``data/processed/betfair_bsp.parquet``.

BSP is the sharp consensus closing price and a genuinely bettable price
(unlike scraped bookmaker SP). It is used for:

* the CLV harness (``scripts/clv_report.py``) — does the model beat the close?
* a BSP-preferred market-probability *teacher* for distillation / the market
  anchor, with bookmaker SP as the fallback where BSP is absent (pre-2025 and
  any unmatched race). BSP is post-race only, so it is never a live-inference
  feature — only a training target / settlement price.

Join keys mirror ``src/utils`` (track/off-time) plus a punctuation-insensitive
horse key, because Betfair strips the apostrophes/accents the RTV-sourced DB
keeps ("Georges Lad" vs "George's Lad"). The shared ``normalise_horse_key`` is
left untouched (it is correct for RTV<->Matchbook); we wrap it here.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from src.utils import normalise_horse_key, normalise_off_time_key, normalise_track_key

DEFAULT_BSP_PATH = os.path.join("data", "processed", "betfair_bsp.parquet")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")

# Betfair uses shorter course names than the RTV-sourced DB for a few tracks.
# Map DB-side normalised keys -> the Betfair-side normalised key so both
# canonicalise to the same value for the join.
_TRACK_ALIASES = {
    "epsom downs": "epsom",
    "royal ascot": "ascot",
    "ballinarobe": "ballinrobe",      # Betfair misspells Ballinrobe
    "bangor on dee": "bangor-on-dee",  # Betfair drops the hyphens
    "bangor": "bangor-on-dee",         # ...and sometimes uses the bare name
}


def bf_track_key(track) -> str:
    """Track key for the Betfair join, with course-name aliases applied."""
    k = normalise_track_key(track)
    return _TRACK_ALIASES.get(k, k)


def bf_horse_key(name) -> str:
    """Punctuation-insensitive horse key for the Betfair join.

    Wraps the shared ``normalise_horse_key`` and additionally drops punctuation
    so Betfair's apostrophe-stripped names match the DB. Both sides of any
    Betfair join must use this.
    """
    return _PUNCT_RE.sub("", normalise_horse_key(name).lower()).strip()


def load_bsp(path: str = DEFAULT_BSP_PATH) -> pd.DataFrame | None:
    """Load the BSP parquet, or ``None`` if it has not been built yet."""
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def attach_bsp(
    df: pd.DataFrame,
    bsp: pd.DataFrame | None = None,
    *,
    date_col: str = "race_date",
    track_col: str = "track",
    off_col: str = "off_time",
    horse_col: str = "horse_name",
    cols: tuple[str, ...] = ("bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_last"),
) -> pd.DataFrame:
    """Left-join Betfair price columns onto *df* by date/track/off/horse.

    *df* must carry race_date, track, off_time and horse_name (any naming via
    the ``*_col`` args). Returns a copy of *df* with *cols* added (NaN where no
    Betfair match). Does not mutate the input.
    """
    out = df.copy()
    if bsp is None:
        bsp = load_bsp()
    if bsp is None or bsp.empty:
        for c in cols:
            out[c] = np.nan
        return out

    def _date10(s):
        return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")

    left_keys = pd.DataFrame(
        {
            "race_date": _date10(out[date_col]),
            "track_key": out[track_col].map(bf_track_key),
            "off_key": out[off_col].map(normalise_off_time_key),
            "horse_key": out[horse_col].map(bf_horse_key),
        }
    )
    # Recompute the right-side track key from the raw course name so aliases
    # apply on both sides regardless of what was stored in the parquet.
    right = bsp[["race_date", "track", "off_key", "horse_key", *cols]].copy()
    right["race_date"] = _date10(right["race_date"])
    right["track_key"] = right["track"].map(bf_track_key)
    right = right[["race_date", "track_key", "off_key", "horse_key", *cols]].drop_duplicates(
        subset=["race_date", "track_key", "off_key", "horse_key"]
    )
    merged = left_keys.merge(
        right, on=["race_date", "track_key", "off_key", "horse_key"], how="left"
    )
    for c in cols:
        out[c] = merged[c].to_numpy()
    return out


def _grouped_overround_normalise(prob: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalise raw implied probabilities to sum to 1 within each race."""
    s = pd.Series(prob, dtype="float64")
    grp = s.groupby(pd.Series(race_ids))
    totals = grp.transform("sum")
    out = np.where((totals > 0) & np.isfinite(totals), s / totals, np.nan)
    return out


def bsp_implied_prob(bsp, race_ids) -> np.ndarray:
    """Overround-normalised BSP-implied win probability, per race.

    Exchange overround is tiny but non-zero; normalising per race makes the
    probabilities a proper distribution (and comparable to the SP-implied
    convention used elsewhere). NaN BSP -> NaN out.
    """
    bsp_arr = np.asarray(bsp, dtype="float64")
    raw = np.divide(1.0, bsp_arr, out=np.full_like(bsp_arr, np.nan), where=bsp_arr > 1.0)
    return _grouped_overround_normalise(raw, np.asarray(race_ids))


def market_prob_bsp_preferred(
    df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    bsp_col: str = "bsp",
    sp_col: str = "odds",
) -> np.ndarray:
    """Unified market probability: BSP-implied where present, SP-implied else.

    Both legs are overround-normalised within their race so the fallback is on
    the same scale. This is the distillation/market-anchor *teacher*; pre-2025
    rows (no BSP) fall back to the bookmaker SP that was always used there.
    """
    race_ids = df[race_id_col].to_numpy()
    bsp_p = bsp_implied_prob(df[bsp_col].to_numpy(), race_ids) if bsp_col in df else None
    sp_raw = np.asarray(df[sp_col].to_numpy(), dtype="float64")
    sp_p = _grouped_overround_normalise(
        np.divide(1.0, sp_raw, out=np.full_like(sp_raw, np.nan), where=sp_raw > 1.0),
        race_ids,
    )
    if bsp_p is None:
        return sp_p
    return np.where(np.isfinite(bsp_p), bsp_p, sp_p)
