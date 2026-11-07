"""European Central Bank reference rates for prize money.

Collection only. ``src.prize_currency_features`` reads the published CSV and never
makes a request, so rebuilding features cannot depend on the network.
"""
from __future__ import annotations

from datetime import datetime, timezone
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


def reference_paths():
    """The collected reference files. Features read their published copies."""
    base = Path(config.BASE_DIR) / "reference"
    return base / "track_prize_currency.json", base / "ecb_gbp_per_eur.csv"


def _read_rates(path):
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


def _update_rates(today):
    """Fetch the public history at most once a day, when a newer rate can exist."""
    _, path = reference_paths()
    rates = _read_rates(path)
    # Weekends may add no observation.
    needed = today-pd.Timedelta(days=1)
    receipt_path = Path(config.DATA_DIR) / "reference/ecb_checked.json"
    checked_day = today.date().isoformat()
    try:
        checked = json.loads(receipt_path.read_text(encoding="utf-8")).get("date") == checked_day
    except (OSError, ValueError):
        checked = False
    if rates.observation_date.max() < needed and not checked:
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
            _atomic_text(receipt_path, json.dumps({"date": checked_day, "source_url": ECB_URL,
                "checked_at_utc": datetime.now(timezone.utc).isoformat()}))
            rates = _read_rates(path)
        except Exception as error:
            logger.warning("ECB update unavailable; retaining dated known reference rates: %s", error)
    return rates


def refresh_prize_references():
    """Source-stage ECB update; ordinary processing only reads dated rates."""
    today = pd.Timestamp.now().normalize()
    rates = _update_rates(today)
    receipt = Path(config.DATA_DIR)/"reference/ecb_checked.json"
    try:
        checked = json.loads(receipt.read_text(encoding="utf-8")).get("date") == pd.Timestamp.now().date().isoformat()
    except (OSError, ValueError):
        checked = False
    current = rates.observation_date.max() >= today-pd.Timedelta(days=1)
    return {"state": "ok" if checked or current else "partial", "latest_observation": str(rates.observation_date.max().date()),
            "checked_today": checked}
