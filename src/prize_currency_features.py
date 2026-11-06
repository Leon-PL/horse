"""Canonical GBP purse inputs plus a research oracle for the native histories.

Venue units are explicitly source-verified. Rates must precede the race date;
current-day afternoon publications cannot enter that morning's features.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import config

logger = logging.getLogger(__name__)
ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml"

FEATURES = ("horse_max_prize_contested", "horse_avg_prize_contested", "prize_money_change_vs_avg")


def _paths():
    base = Path(config.BASE_DIR) / "reference"
    return base / "track_prize_currency.json", base / "ecb_gbp_per_eur.csv"


@lru_cache(maxsize=8)
def _read_units(path, modified, size):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))["currency_by_track"]
    return {name.strip().casefold(): unit for name, unit in raw.items()}


@lru_cache(maxsize=8)
def _read_rates(path, modified, size):
    return pd.read_csv(path, parse_dates=["observation_date"], float_precision="round_trip")


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reference-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _rates_for(dates, *, refresh=False):
    _, path = _paths()
    stat = path.stat()
    rates = _read_rates(str(path), stat.st_mtime_ns, stat.st_size)
    # Historic processing never needs a live request. For newer cards/results,
    # refresh the public reference once per day; weekends may add no observation.
    needed = min(pd.to_datetime(dates).max().normalize()-pd.Timedelta(days=1),
                 pd.Timestamp.now().normalize()-pd.Timedelta(days=1))
    receipt_path = Path(config.DATA_DIR) / "reference/ecb_checked.json"
    today = pd.Timestamp.now().date().isoformat()
    try:
        checked = json.loads(receipt_path.read_text(encoding="utf-8")).get("date") == today
    except (OSError, ValueError):
        checked = False
    if refresh and rates.observation_date.max() < needed and not checked:
        try:
            with urlopen(ECB_URL, timeout=20) as response:
                document = ET.fromstring(response.read())
            rows = []
            for node in document.iter():
                if "time" in node.attrib:
                    for quote in node:
                        if quote.attrib.get("currency") == "GBP":
                            rows.append((node.attrib["time"], float(quote.attrib["rate"])))
            fresh = pd.DataFrame(rows, columns=["observation_date", "gbp_per_eur"])
            fresh["observation_date"] = pd.to_datetime(fresh.observation_date)
            fresh = fresh.sort_values("observation_date").reset_index(drop=True)
            if fresh.empty or fresh.observation_date.duplicated().any() or not np.isfinite(fresh.gbp_per_eur).all() or fresh.gbp_per_eur.le(0).any():
                raise ValueError("Invalid ECB reference response")
            if fresh.observation_date.max() < rates.observation_date.max():
                raise ValueError("ECB response would truncate existing reference history")
            _atomic_text(path, fresh.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g"))
            _atomic_text(receipt_path, json.dumps({"date": today, "source_url": ECB_URL,
                "checked_at_utc": datetime.now(timezone.utc).isoformat()}))
            stat = path.stat(); rates = _read_rates(str(path), stat.st_mtime_ns, stat.st_size)
        except Exception as error:
            logger.warning("ECB update unavailable; retaining dated known reference rates: %s", error)
    return rates


def refresh_prize_references():
    """Source-stage ECB update; ordinary processing only reads dated rates."""
    dates = pd.Series([pd.Timestamp.now().normalize()])
    rates = _rates_for(dates, refresh=True)
    receipt = Path(config.DATA_DIR)/"reference/ecb_checked.json"
    try:
        checked = json.loads(receipt.read_text(encoding="utf-8")).get("date") == pd.Timestamp.now().date().isoformat()
    except (OSError, ValueError):
        checked = False
    current = rates.observation_date.max() >= dates.max()-pd.Timedelta(days=1)
    return {"state": "ok" if checked or current else "partial", "latest_observation": str(rates.observation_date.max().date()),
            "checked_today": checked}


def reference_signature():
    """Also invalidate live vectors when reference values change without FE edits."""
    digest = hashlib.sha256()
    for path in _paths():
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def prepare_prize_references(frame):
    """Refresh before cache lookup, so a new rate never mixes cache generations."""
    if not {"track", "race_date"}.issubset(frame):
        return
    unit_path, _ = _paths(); stat = unit_path.stat()
    units = _read_units(str(unit_path), stat.st_mtime_ns, stat.st_size)
    euro = frame.track.astype("string").str.strip().str.casefold().map(units).eq("EUR")
    dates = pd.to_datetime(frame.loc[euro, "race_date"], errors="coerce").dropna()
    if len(dates):
        _rates_for(dates)


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
        rates = _rates_for(result.loc[euro, "race_date"])
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
