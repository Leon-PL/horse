"""Canonical GBP purse inputs plus a research oracle for the native histories.

Venue units are explicitly source-verified. Rates must precede the race date;
current-day afternoon publications cannot enter that morning's features.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("horse_max_prize_contested", "horse_avg_prize_contested", "prize_money_change_vs_avg")


def _paths():
    """The selected dataset's currency units and ECB rates."""
    from src.source_dataset import source_path
    return source_path("fx", "units"), source_path("fx", "rates")


@lru_cache(maxsize=8)
def _read_units(path, modified, size):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["currency_by_track"]
    return {name.strip().casefold(): unit for name, unit in raw.items()}


@lru_cache(maxsize=8)
def _read_rates(path, modified, size):
    return pd.read_csv(path, parse_dates=["observation_date"], float_precision="round_trip")


def _load_rates():
    _, path = _paths()
    stat = path.stat()
    return _read_rates(str(path), stat.st_mtime_ns, stat.st_size)


def reference_signature():
    """Also invalidate live vectors when reference values change without FE edits."""
    digest = hashlib.sha256()
    for path in _paths():
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def course_currency(tracks):
    unit_path, _ = _paths()
    stat = unit_path.stat()
    units = _read_units(str(unit_path), stat.st_mtime_ns, stat.st_size)
    return tracks.astype("string").str.strip().str.casefold().map(units)


def prepare_prize_references(frame):
    """Read the references before the cache lookup. Refreshing is `ingestion.fx`'s job."""
    if not {"track", "race_date"}.issubset(frame):
        return
    euro = course_currency(frame.track).eq("EUR")
    dates = pd.to_datetime(frame.loc[euro, "race_date"], errors="coerce").dropna()
    if len(dates):
        _load_rates()


def normalise_prize_money(frame):
    """Retain raw evidence and express every supported purse in the same unit.

    Reprocessing uses the original amount, so EUR is never converted twice.
    Unknown venues stay unknown rather than silently becoming GBP. Zero is zero
    in either currency, preserving the established zero-prize convention.
    """
    result = frame.copy()
    unit_path, _ = _paths(); stat = unit_path.stat()
    units = _read_units(str(unit_path), stat.st_mtime_ns, stat.st_size)
    amount = pd.to_numeric(result.get("prize_money_original", result.prize_money), errors="coerce").astype(float)
    tracks = result.get("track", pd.Series(pd.NA, index=result.index)).astype("string").str.strip().str.casefold()
    currency = tracks.map(units).fillna("Unknown")
    result["prize_money_original"] = amount
    result["prize_currency"] = currency
    result["prize_money"] = amount.where(currency.eq("GBP") | amount.eq(0))
    fx_dates = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    euro = currency.eq("EUR")
    if euro.any():
        rates = _load_rates()
        rows = result.loc[euro].assign(track=tracks.loc[euro], prize_money=amount.loc[euro])
        result.loc[euro, "prize_money"] = convert_prizes(rows, units, rates)
        positions = np.searchsorted(rates.observation_date.to_numpy(), pd.to_datetime(rows.race_date).dt.normalize().to_numpy(), side="left")-1
        fx_dates.loc[euro] = rates.observation_date.to_numpy()[positions]
    result["prize_fx_date"] = fx_dates
    return result


def convert_prizes(frame, currency_by_track, rates):
    dates = pd.to_datetime(frame.race_date, errors="coerce")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Expected valid, timezone-free race dates")
    currency = frame.track.map(currency_by_track)
    if currency.isna().any() or not currency.isin(["GBP", "EUR"]).all():
        raise ValueError("Every venue needs an explicitly verified GBP/EUR currency")
    observed = pd.to_datetime(rates.observation_date, errors="coerce")
    values = pd.to_numeric(rates.gbp_per_eur, errors="coerce").to_numpy(float)
    if (observed.isna().any() or observed.duplicated().any() or not observed.is_monotonic_increasing
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Exchange-rate history must be unique, ordered and positive")
    positions = np.searchsorted(observed.to_numpy(), dates.dt.normalize().to_numpy(), side="left")-1
    euro = currency.eq("EUR").to_numpy()
    if (positions[euro] < 0).any():
        raise ValueError("No earlier published exchange rate for an EUR race")
    factor = np.ones(len(frame), dtype=float)
    factor[euro] = values[positions[euro]]
    result = pd.to_numeric(frame.prize_money, errors="coerce").astype(float) * factor
    return result


def native_prize_history(frame, prizes):
    """Replay the existing exact history maths; replace only its monetary units.

    This narrow pass retains original prior-start denominators, zero prizes,
    cold-start values and missing-value treatment to isolate the currency fix.
    It neither reads odds nor updates the installed incremental history state.
    """
    from src.feature_engineer import _event_sort_key, add_prize_money_features, counts_state_scope
    if not frame.index.is_unique or len(prizes) != len(frame):
        raise ValueError("Expected uniquely aligned runner inputs")
    columns = ["race_id", "race_date", "horse_id", "horse_name", "horse_prev_races"]
    if "off_time" in frame:
        columns.append("off_time")
    work = frame[columns].copy()
    work["prize_money"] = np.asarray(prizes, dtype=float)
    work["_original_position"] = np.arange(len(work))
    work["_event_dt"] = _event_sort_key(work)
    work = work.sort_values(["_event_dt", "race_id", "horse_name"], kind="stable")
    with counts_state_scope(None):
        result = add_prize_money_features(work)
    result = result.sort_values("_original_position")[list(FEATURES)]
    result.index = frame.index
    return result
