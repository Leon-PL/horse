"""
Betfair Exchange BSP integration
================================

Loader + join helpers for the historical Betfair Starting Price (BSP) and
pre-off traded prices collected by ``src.ingestion.betfair_history`` into
``data/processed/betfair_bsp.parquet``.

BSP is the sharp consensus closing price and a genuinely bettable price
(unlike scraped bookmaker SP). It is used for:

* the CLV harness (``scripts/clv_report.py``) — does the model beat the close?
* a BSP-preferred market-probability *teacher* for distillation / the market
  anchor, with bookmaker SP as the fallback where BSP is absent or unmatched.
  BSP is post-race only, so it is never a live-inference
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

import config
from src.runner_identity import runner_positions
from src.utils import normalise_horse_key, normalise_off_time_key, normalise_track_key

DEFAULT_BSP_PATH = os.path.join(config.PROCESSED_DATA_DIR, "betfair_bsp.parquet")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
PRICE_KEYS = ["race_date", "track_key", "off_key", "horse_key"]

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

# The 2020-2021 Betfair "BASIC" streams abbreviate the course in eventName
# ("Bev 5th Jun" -> "Bev"); later years use full names. Map each abbreviation's
# normalised key to the full DB track key so both sides of the join agree.
# Without this ~6.5k pre-2022 Betfair races fail to join to data/races.db.
_TRACK_ABBREVIATIONS = {
    # aint/list/nott were missing until 2026-10-13: every 2020-21 Aintree,
    # Listowel and Nottingham race failed to join (124 races in the public
    # BSP files alone), found by betfair_db_audit against the public census.
    "aint": "aintree", "list": "listowel", "nott": "nottingham",
    "ballin": "ballinrobe", "bang": "bangor-on-dee", "belle": "bellewstown",
    "bev": "beverley", "carl": "carlisle", "cart": "cartmel",
    "catt": "catterick", "chelmc": "chelmsford city", "chelt": "cheltenham",
    "chep": "chepstow", "chest": "chester", "clon": "clonmel",
    "curr": "curragh", "donc": "doncaster", "downp": "downpatrick",
    "downr": "down royal", "dund": "dundalk", "epsm": "epsom",
    "extr": "exeter", "fairy": "fairyhouse", "fake": "fakenham",
    "ffosl": "ffos las", "font": "fontwell", "gal": "galway",
    "good": "goodwood", "gowp": "gowran park", "ham": "hamilton",
    "hayd": "haydock", "here": "hereford", "hex": "hexham",
    "hunt": "huntingdon", "kemp": "kempton", "kilb": "kilbeggan",
    "killar": "killarney", "leic": "leicester", "leop": "leopardstown",
    "lim": "limerick", "ling": "lingfield", "ludl": "ludlow",
    "mrktr": "market rasen", "muss": "musselburgh", "newb": "newbury",
    "newc": "newcastle", "newm": "newmarket", "newt": "newton abbot",
    "plump": "plumpton", "ponte": "pontefract", "punch": "punchestown",
    "redc": "redcar", "rosc": "roscommon", "salis": "salisbury",
    "sand": "sandown", "sedge": "sedgefield", "sthl": "southwell",
    "strat": "stratford", "taun": "taunton", "thurl": "thurles",
    "tipp": "tipperary", "tram": "tramore", "uttox": "uttoxeter",
    "warw": "warwick", "weth": "wetherby", "wex": "wexford",
    "winc": "wincanton", "wind": "windsor", "wolv": "wolverhampton",
    "yarm": "yarmouth",
}

_TRACK_KEY_MAP = {**_TRACK_ALIASES, **_TRACK_ABBREVIATIONS}


def bf_track_key(track) -> str:
    """Track key for the Betfair join, with course-name aliases applied."""
    k = normalise_track_key(track)
    return _TRACK_KEY_MAP.get(k, k)


def bf_horse_key(name) -> str:
    """Punctuation-insensitive horse key for the Betfair join.

    Wraps the shared ``normalise_horse_key`` and additionally drops punctuation
    so Betfair's apostrophe-stripped names match the DB. Both sides of any
    Betfair join must use this.
    """
    return _PUNCT_RE.sub("", normalise_horse_key(name).lower()).strip()


def load_bsp(path: str | None = None) -> pd.DataFrame | None:
    """Load the project-rooted cache, or ``None`` if it has not been built yet."""
    if path is None:
        path = DEFAULT_BSP_PATH
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def price_runner_positions(left, right, *, skip_ambiguous=True):
    """Match Betfair selections within a race; names are a unique fallback only.

    The shared identity matcher receives Betfair IDs on both sides here. Sporting
    Life horse IDs never act as exchange selection IDs or supply a crosswalk.
    """
    races = pd.MultiIndex.from_frame(pd.concat(
        [left[PRICE_KEYS[:3]], right[PRICE_KEYS[:3]]], ignore_index=True))
    codes, _ = pd.factorize(races)
    def identity(frame, race_codes):
        return pd.DataFrame({"race_id": race_codes,
            "horse_id": frame.get("bf_runner_id", pd.Series(pd.NA, index=frame.index)).to_numpy(),
            "horse_name": frame.horse_key.to_numpy()})
    return runner_positions(identity(left, codes[:len(left)]), identity(right, codes[len(left):]),
                            require_all=False, skip_ambiguous=skip_ambiguous)


def attach_bsp(
    df: pd.DataFrame,
    bsp: pd.DataFrame | None = None,
    *,
    date_col: str = "race_date",
    track_col: str = "track",
    off_col: str = "off_time",
    horse_col: str = "horse_name",
    cols: tuple[str, ...] = ("bsp", "ltp_preoff", "ltp_60s", "ltp_300s", "ltp_last"),
    overwrite: bool = False,
) -> pd.DataFrame:
    """Left-join Betfair price columns onto *df* by date/track/off/horse.

    *df* must carry race_date, track, off_time and horse_name (any naming via
    the ``*_col`` args). Returns a copy of *df* with *cols* added (NaN where no
    Betfair match). Does not mutate the input.

    Refuses to replace a column *df* already has unless ``overwrite=True``:
    asking for ``cols=("won",)`` used to swap the caller's own label for
    Betfair's without a word, so an audit compared the exchange with itself.
    """
    clash = [c for c in cols if c in df.columns]
    if clash and not overwrite:
        raise ValueError(
            f"attach_bsp would replace existing column(s) {clash} with Betfair's; "
            "rename yours first, or pass overwrite=True if that is the intent"
        )
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
    identity_columns = ["bf_runner_id"] if "bf_runner_id" in bsp else []
    right = bsp[list(dict.fromkeys(["race_date", "track", "off_key", "horse_key", *identity_columns, *cols]))].copy()
    right["race_date"] = _date10(right["race_date"])
    right["track_key"] = right["track"].map(bf_track_key)
    right = right[list(dict.fromkeys([*PRICE_KEYS, *identity_columns, *cols]))].drop_duplicates().reset_index(drop=True)
    if "bf_runner_id" in out:
        left_keys["bf_runner_id"] = out.bf_runner_id.to_numpy()
    positions = price_runner_positions(left_keys, right)
    merged = right.reindex(positions)
    for c in cols:
        out[c] = merged[c].to_numpy()
    return out


def _grouped_overround_normalise(prob: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalise complete positive fields; an incomplete field stays unknown."""
    s = pd.Series(prob, dtype="float64")
    ids = pd.Series(race_ids)
    grp = s.groupby(ids)
    totals = grp.transform("sum")
    complete = pd.Series(np.isfinite(s) & s.gt(0)).groupby(ids).transform("all")
    out = np.where(complete & (totals > 0) & np.isfinite(totals), s / totals, np.nan)
    return out


def bsp_implied_prob(bsp, race_ids) -> np.ndarray:
    """Overround-normalised BSP-implied win probability, per race.

    Exchange overround is tiny but non-zero; normalising per race makes the
    probabilities a proper distribution (and comparable to the SP-implied
    convention used elsewhere). Any unavailable/invalid BSP makes the whole
    field unavailable; a covered subset is not a probability distribution.
    """
    bsp_arr = np.asarray(bsp, dtype="float64")
    raw = np.divide(1.0, bsp_arr, out=np.full_like(bsp_arr, np.nan), where=np.isfinite(bsp_arr) & (bsp_arr > 1.0))
    return _grouped_overround_normalise(raw, np.asarray(race_ids))


def market_prob_bsp_preferred(
    df: pd.DataFrame,
    *,
    race_id_col: str = "race_id",
    bsp_col: str = "bsp",
    sp_col: str = "odds",
) -> np.ndarray:
    """One complete market per race: BSP, else SP, else unavailable.

    Never mix separately normalised sources runner by runner. If num_runners
    is supplied, require that the input contains the declared complete field.
    These are training targets, not prices available before the historical off.
    """
    race_ids = df[race_id_col].to_numpy()
    bsp_p = bsp_implied_prob(df[bsp_col].to_numpy(), race_ids) if bsp_col in df else None
    sp_raw = (np.asarray(df[sp_col].to_numpy(), dtype="float64") if sp_col in df
              else np.full(len(df), np.nan))
    sp_p = _grouped_overround_normalise(
        np.divide(1.0, sp_raw, out=np.full_like(sp_raw, np.nan), where=np.isfinite(sp_raw) & (sp_raw > 1.0)),
        race_ids,
    )
    result = sp_p if bsp_p is None else np.where(np.isfinite(bsp_p), bsp_p, sp_p)
    if "num_runners" in df:
        ids = pd.Series(race_ids)
        declared = pd.Series(pd.to_numeric(df["num_runners"], errors="coerce").to_numpy())
        size = ids.groupby(ids).transform("size")
        complete = declared.eq(size).groupby(ids).transform("all").to_numpy()
        result = np.where(complete, result, np.nan)
    return result
