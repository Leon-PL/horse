"""Shared identity, timing and quality rules for accepted source observations.

Provider IDs remain namespaced. A name is evidence for a unique crosswalk,
never a replacement for a known, conflicting identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils import normalise_horse_key, normalise_off_time_key, normalise_track_key


@dataclass(frozen=True)
class TableContract:
    grain: str
    required: tuple[str, ...]
    keys: tuple[str, ...] = ()
    role: str = "history"
    delay_hours: int = 0
    types: tuple[tuple[str, str], ...] = ()
    event_column: str = "race_date"

    def validate(self, frame):
        missing = set(self.required) - set(frame)
        if missing:
            raise ValueError(f"{self.grain}: missing columns {sorted(missing)}")
        if self.keys and not frame.empty:
            if frame[list(self.keys)].isna().any(axis=None):
                raise ValueError(f"{self.grain}: null native identity")
            if frame.duplicated(list(self.keys)).any():
                raise ValueError(f"{self.grain}: duplicate native identity")
            if frame[list(self.keys)].astype("string").apply(lambda values: values.str.strip().eq("")).any(axis=None):
                raise ValueError(f"{self.grain}: blank native identity")
        for column, kind in self.types:
            if column not in frame:
                raise ValueError(f"{self.grain}: missing typed column {column}")
            values = frame[column].dropna()
            if kind == "number":
                pd.to_numeric(values, errors="raise")
            elif kind == "timestamp":
                pd.to_datetime(values, utc=True, errors="raise")
            elif kind == "text" and not values.map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"{self.grain}: {column} must hold text")
        if "race_date" in frame and pd.to_datetime(frame.race_date, errors="coerce").isna().any():
            raise ValueError(f"{self.grain}: invalid event date")


def _ids(series):
    return series.astype("string").str.replace(r"\.0$", "", regex=True).replace(
        ["", "0", "None", "nan", "<NA>"], pd.NA)


def resolve_runners(frame, canonical, *, provider, track_key=None, horse_key=None):
    """One receipt per source row, including rejected and unresolved links.

    Existing canonical IDs are checked against the field, not trusted blindly.
    Known native IDs are retained separately and never compared across providers.
    """
    out = pd.DataFrame({"source_row": range(len(frame))})
    out["provider"] = provider
    out["sl_race_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["sl_horse_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["match_status"] = "unmatched"
    out["match_method"] = "none"
    native = {"atr": ("atr_raceid", "atr_horse_id"), "betfair": ("market_id", "bf_runner_id")}
    for target, candidates in (("source_race_id", (native.get(provider, ("", ""))[0], "rtv_race_id")),
                               ("source_runner_id", (native.get(provider, ("", ""))[1], "rtv_runner_id"))):
        out[target] = pd.NA
        for column in candidates:
            if column in frame:
                out[target] = _ids(frame[column]).to_numpy()
                break
    if frame.empty or canonical.empty:
        return out
    track_key = track_key or normalise_track_key
    horse_key = horse_key or normalise_horse_key

    def keys(data):
        result = pd.DataFrame(index=range(len(data)))
        result["day"] = pd.to_datetime(data.race_date, errors="coerce").dt.strftime("%Y-%m-%d").to_numpy()
        result["track"] = data.track.map(track_key).to_numpy()
        result["off"] = data["off_time"].map(normalise_off_time_key).to_numpy() if "off_time" in data else data.off_key.to_numpy()
        result["horse"] = data.horse_name.map(horse_key).to_numpy() if "horse_name" in data else data.horse_key.to_numpy()
        return result

    frame = frame.copy(deep=False)
    if "track" not in frame and "track_key" in frame:
        frame = frame.assign(track=frame.track_key)
    needed = {"race_date", "track"}
    if not needed.issubset(frame) or not ({"horse_name", "horse_key"} & set(frame)) or not ({"off_time", "off_key"} & set(frame)):
        # Materialised RTV rows already carry SL IDs, verified below.
        left = None
    else:
        left = keys(frame)
    right = keys(canonical)
    right["sl_race_id"] = _ids(canonical.race_id).to_numpy()
    right["sl_horse_id"] = _ids(canonical.horse_id).to_numpy()
    columns = ["day", "track", "off", "horse"]
    if left is not None:
        left["source_row"] = out.source_row
        bad = left[columns].isna().any(axis=1) | left[columns].isin(["", "None", "Nan"]).any(axis=1)
        duplicate_keys = columns
        if "observed_at" in frame:
            left["observed_at"] = pd.to_datetime(frame.observed_at, utc=True, errors="raise").to_numpy()
            duplicate_keys = [*columns, "observed_at"]
        ambiguous = left.duplicated(duplicate_keys, keep=False)
        right_ambiguous = right.duplicated(columns, keep=False)
        ambiguous |= pd.MultiIndex.from_frame(left[columns]).isin(pd.MultiIndex.from_frame(right.loc[right_ambiguous, columns]))
        out.loc[ambiguous, "match_status"] = "ambiguous"
        out.loc[bad, "match_status"] = "invalid_key"
        joined = left.loc[~bad & ~ambiguous].merge(right.loc[~right_ambiguous], on=columns, validate="many_to_one")
        indices = joined.source_row.to_numpy()
        out.loc[indices, ["sl_race_id", "sl_horse_id"]] = joined[["sl_race_id", "sl_horse_id"]].to_numpy()
        out.loc[indices, ["match_status", "match_method"]] = ["matched", "unique_name_time"]
    # A verified crosswalk or already canonical materialisation takes precedence.
    race_col, horse_col = ("sl_race_id", "sl_horse_id") if {"sl_race_id", "sl_horse_id"}.issubset(frame) else (
        ("race_id", "horse_id") if provider == "rtv" and {"race_id", "horse_id"}.issubset(frame) else (None, None))
    if race_col:
        known = pd.DataFrame({"source_row": out.source_row, "sl_race_id": _ids(frame[race_col]).to_numpy(),
                              "sl_horse_id": _ids(frame[horse_col]).to_numpy()})
        valid = known.dropna().merge(right, on=["sl_race_id", "sl_horse_id"], validate="many_to_one")
        if left is not None:
            source_keys = left.set_index("source_row").reindex(valid.source_row)
            equal = source_keys[columns].to_numpy() == valid[columns].to_numpy()
            valid = valid.loc[equal.all(axis=1)]
        indices = valid.source_row.to_numpy()
        supplied = known[["sl_race_id", "sl_horse_id"]].notna().all(axis=1)
        out.loc[supplied, ["sl_race_id", "sl_horse_id"]] = pd.NA
        out.loc[supplied, ["match_status", "match_method"]] = ["identity_conflict", "none"]
        out.loc[indices, ["sl_race_id", "sl_horse_id"]] = valid[["sl_race_id", "sl_horse_id"]].to_numpy()
        out.loc[indices, ["match_status", "match_method"]] = ["matched", "verified_crosswalk"]
    # Repeated snapshots are legitimate; duplicate native observations at the
    # same instant are handled by the table contract, before linking.
    return out


def observation_metadata(frame, contract, links=None):
    """Keep absent, unavailable and unmatched distinct; never fabricate fetch times."""
    out = pd.DataFrame({"source_row": range(len(frame))})
    day = pd.to_datetime(frame.get(contract.event_column, pd.Series(pd.NaT, index=frame.index)), utc=True, errors="coerce").reset_index(drop=True)
    off_column = next((c for c in ("off_time", "off_key") if c in frame), None)
    if off_column:
        off = frame[off_column].map(normalise_off_time_key).reset_index(drop=True)
        day = pd.to_datetime(day.dt.strftime("%Y-%m-%d") + " " + off, utc=True, errors="coerce")
    out["event_at"] = day
    observed = next((c for c in ("observed_at", "fetched_at", "collected_at") if c in frame), None)
    out["observed_at"] = pd.to_datetime(frame[observed], utc=True, errors="raise").to_numpy() if observed else pd.NaT
    available = next((c for c in ("available_at", "available_date") if c in frame), None)
    out["available_at"] = (pd.to_datetime(frame[available], utc=True, errors="raise").to_numpy() if available
                           else day + pd.Timedelta(hours=contract.delay_hours))
    out["availability_basis"] = ("recorded" if available == "available_at" else "source_policy" if available
                                 else f"assumed_event_plus_{contract.delay_hours}h")
    if contract.role == "snapshot":
        out["available_at"] = pd.to_datetime(out.observed_at, utc=True)
        out["availability_basis"] = "recorded_capture"
    out["missing_reason"] = "none"
    context = {"race_id", "horse_id", "sl_race_id", "sl_horse_id", "horse_name", "horse_key", "track", "track_key",
               "race_date", "off_time", "off_key", "observed_at", "fetched_at", "collected_at", "available_at", "available_date"}
    measured = [c for c in frame if c not in context and not c.endswith("_id")]
    out["missing_field_count"] = frame[measured].isna().sum(axis=1).to_numpy() if measured else 0
    if links is not None:
        out.loc[links.match_status.ne("matched").to_numpy(), "missing_reason"] = links.loc[links.match_status.ne("matched"), "match_status"].to_numpy()
    return out


def coverage_report(frame, links=None, *, eligible=None, total=None, metadata=None, now=None):
    matched = int(links.match_status.eq("matched").sum()) if links is not None else len(frame)
    linked_runners = (int(links.loc[links.match_status.eq("matched"), ["sl_race_id", "sl_horse_id"]].drop_duplicates().shape[0])
                      if links is not None else len(frame))
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    usable = matched
    latest = None
    if metadata is not None:
        available = pd.to_datetime(metadata.available_at, utc=True)
        mask = available.le(now)
        if links is not None:
            mask &= links.match_status.eq("matched")
        usable = int(mask.sum())
        latest_event = pd.to_datetime(metadata.loc[mask, "event_at"], utc=True).max()
        latest = None if pd.isna(latest_event) else latest_event.isoformat()
    return {"collected_rows": len(frame), "matched_rows": matched, "linked_runners": linked_runners,
            "usable_rows": usable, "eligible_runners": eligible, "all_runners": total,
            "eligible_coverage": linked_runners / eligible if eligible else None,
            "all_runner_coverage": linked_runners / total if total else None,
            "latest_usable_event": latest,
            "missing_values_by_column": {name: int(value) for name, value in frame.isna().sum().items() if value},
            "age_days": (now - pd.Timestamp(latest)).total_seconds() / 86400 if latest else None,
            "link_reasons": links.match_status.value_counts().to_dict() if links is not None else {}}
