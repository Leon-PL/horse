"""Date-strict predictors from public results absent from the local race DB.

Research family: not part of the default selected inputs. Individual horse
records must never be inserted into a full-field rating sweep. Profile
ratings, current age, lifetime totals, prices and adjacent margins are not
used. Counts mean KNOWN external results, not complete career histories.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

FEATURES = ("external_prior_runs", "external_prior_quality_last",
            "external_prior_quality_mean_3", "external_prior_quality_best_3",
            "external_prior_win_rate_shrunk", "external_prior_days_since_last",
            "external_prior_distance_change")


def _ids(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or numeric.mod(1).ne(0).any():
        raise ValueError("External history requires unambiguous numeric source IDs")
    return numeric.astype("int64").astype(str)


def _distance(value):
    """Parse explicit source distances; unknown is not a fabricated10f trip."""
    text = str(value or "").lower().strip()
    match = re.fullmatch(r"(?:(\d+)\s*m\s*)?(?:(\d+(?:\.\d+)?)\s*f\s*)?(?:(\d+)\s*y)?", text)
    if not match or not any(match.groups()):
        return np.nan
    miles, furlongs, yards = (float(x or 0) for x in match.groups())
    total = miles * 8 + furlongs + yards / 220
    return total if total > 0 else np.nan


def load_cached_results(directory):
    """Load the sanitised, versioned per-horse cache; no network or DB writes."""
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        cached = json.loads(path.read_text())
        identity = str(cached["horse_id"])
        for row in cached["results"]:
            if str(row.get("horse_id")) != identity:
                raise ValueError(f"Cached result identity differs: {path.name}")
            # The pilot cohort was selected using pre-race metadata on this
            # date. Its availability must not appear on any earlier local
            # row, even if that horse already had an uncompleted local run.
            rows.append({**row, "eligible_from_date": cached["first_local_date"]})
    return pd.DataFrame(rows)


def external_history_features(cards, prior_results, local_results):
    """Return seven predictors using only results on strictly earlier dates.

    ``local_results`` supplies horse/race IDs to prevent counting the same
    performance twice. It can include future local dates: matching future
    performances cannot affect earlier features. NR/withdrawals and unknown
    positions are excluded. Recognised non-completions are poor performances;
    an unavailable position is not silently a loser.
    """
    out = pd.DataFrame(np.nan, index=cards.index, columns=FEATURES)
    out["external_prior_runs"] = 0.
    if prior_results.empty or cards.empty:
        return out
    past = prior_results.copy()
    past["_horse"] = _ids(past.horse_id)
    past["_race"] = _ids(past.race_id)
    local = pd.DataFrame({"_horse": _ids(local_results.horse_id), "_race": _ids(local_results.race_id)})
    keys = ["_horse", "_race"]
    if past.duplicated(keys).any():
        # Exact duplicates may be collapsed; contradictory outcomes cannot.
        relevant = [c for c in ("date", "position", "runner_count", "distance", "casualty") if c in past]
        flattened = past[keys + relevant].astype(str).drop_duplicates()
        if flattened.duplicated(keys).any():
            raise ValueError("Conflicting external history records")
        past = past.drop_duplicates(keys)
    local_index = pd.MultiIndex.from_frame(local.drop_duplicates())
    past = past.loc[~pd.MultiIndex.from_frame(past[keys]).isin(local_index)].copy()
    if past.empty:
        return out
    past["_date"] = pd.to_datetime(past.date, errors="coerce").dt.normalize()
    position = pd.to_numeric(past.position, errors="coerce")
    size = pd.to_numeric(past.runner_count, errors="coerce")
    text = past.get("casualty", pd.Series("", index=past.index)).map(
        lambda c: " ".join(str(v).upper() for v in c.values()) if isinstance(c, dict) else str(c).upper())
    withdrawn = text.str.contains(r"\b(?:NR|WD|VOID|NON[ -]?RUNNER|WITHDRAWN)\b", regex=True)
    # Full source/profile payloads use CamelCase reason enums as well as
    # abbreviated codes. Uppercasing alone leaves PULLEDUP/UNSEATEDRIDER.
    dnf = text.str.contains(r"\b(?:PU|F|UR|BD|RR|RO|FELL|PULLED|PULLEDUP|UNSEATED|UNSEATEDRIDER|REFUSED|BROUGHT)\b", regex=True)
    finished = position.between(1, size) & position.mod(1).eq(0)
    eligible = past._date.notna() & size.ge(2) & size.mod(1).eq(0) & ~withdrawn & (finished | dnf)
    past = past.loc[eligible].copy()
    past["_quality"] = np.where(finished.loc[eligible],
        (size.loc[eligible] - position.loc[eligible]) / (size.loc[eligible] - 1), 0.)
    past["_won"] = position.loc[eligible].eq(1).astype(float)
    past["_distance"] = past.distance.map(_distance)
    past = past.sort_values(["_horse", "_date", "_race"], kind="stable")
    card_keys = _ids(cards.horse_id)
    dates = pd.to_datetime(cards.race_date, errors="coerce").dt.normalize().to_numpy()
    if pd.isna(dates).any():
        raise ValueError("Target dates must be known")
    distances = (pd.to_numeric(cards.distance_furlongs, errors="coerce").to_numpy()
                 if "distance_furlongs" in cards else np.full(len(cards), np.nan))
    groups = card_keys.groupby(card_keys, sort=False).indices
    result = out.to_numpy(copy=True)
    for horse, history in past.groupby("_horse", sort=False):
        indices = groups.get(horse)
        if indices is None:
            continue
        when = history._date.to_numpy()
        count = np.searchsorted(when, dates[indices], side="left")
        known = count > 0
        if "eligible_from_date" in history:
            eligibility = pd.to_datetime(history.eligible_from_date, errors="coerce").dt.normalize()
            if eligibility.isna().any() or eligibility.nunique() != 1:
                raise ValueError("A cached horse needs one valid pilot eligibility date")
            known &= dates[indices] >= eligibility.iloc[0].to_datetime64()
        if not known.any():
            continue
        target, prefix = indices[known], count[known] - 1
        quality = history._quality.reset_index(drop=True)
        wins = history._won.cumsum().to_numpy()
        result[target, 0] = count[known]
        result[target, 1] = quality.to_numpy()[prefix]
        result[target, 2] = quality.rolling(3, min_periods=1).mean().to_numpy()[prefix]
        result[target, 3] = quality.rolling(3, min_periods=1).max().to_numpy()[prefix]
        result[target, 4] = (wins[prefix] + 5. / 8.) / (count[known] + 5.)
        result[target, 5] = (dates[target] - when[prefix]) / np.timedelta64(1, "D")
        result[target, 6] = distances[target] - history._distance.to_numpy()[prefix]
    return pd.DataFrame(result, index=cards.index, columns=FEATURES)
