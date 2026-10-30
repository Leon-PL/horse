"""Auditable dated ride-context descriptors; unselected research inputs.

Mention flags describe source wording, not a diagnosis or a probability bonus.
No current profile prose or retrospectively revised rating enters this family.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from src.utils import normalise_horse_key

PATTERNS = {
    "impeded": r"\b(?:hampered|impeded|short of room|not clear run|denied (?:a )?clear run|checked|blocked)\b",
    "slow_start": r"\b(?:slowly away|slow (?:to (?:start|begin)|away)|dwelt|reared (?:at|as) (?:the )?start)\b",
    "jump_error": r"\b(?:mistake|mistakes|blundered|bad jump|jumped (?:left|right|poorly))\b",
    "eased": r"\b(?:eased|not persevered with|not knocked about)\b",
    "stayed_on": r"\b(?:stayed on|ran on|strongly (?:at|near) (?:the )?finish)\b",
}
FEATURES = tuple(["context_"+k+"_mention_rate_3" for k in PATTERNS]+
                 ["context_description_support_3", "context_accidental_nonfinish_rate_5",
                  "context_pulled_up_rate_5", "context_reason_support_5"])


def description_mentions(value):
    if not isinstance(value, str) or not value.strip():
        return {name: np.nan for name in PATTERNS}
    text = " ".join(value.casefold().split())
    result = {}
    for name, pattern in PATTERNS.items():
        matches = list(re.finditer(pattern, text))
        # Explicit negation cannot count as affirmative factual evidence.
        positive = any(not re.search(r"\b(?:not|never|without|no)\s+(?:\w+\s+){0,2}$", text[max(0,m.start()-30):m.start()])
                       for m in matches)
        result[name] = float(positive)
    return result


def context_history(frame, *, state=None, return_state=False):
    """Emit start-of-day context from the preceding three/five actual starts.

    Every actual start advances the bounded window, including an observation
    with unknown commentary. Pending rows and nonstarters never advance it.
    A seed ends strictly before every input date and is never mutated.
    """
    state = state or {}
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    if dates.isna().any() or (state.get("last_date") and len(frame) and dates.min() <= pd.Timestamp(state["last_date"])):
        raise ValueError("Context seed must precede every dated input")
    keys = frame.horse_name.map(normalise_horse_key)
    ids = pd.to_numeric(frame.get("horse_id", pd.Series(np.nan, index=frame.index)), errors="coerce")
    valid_ids = ids.gt(0)&ids.mod(1).eq(0)
    safe_ids = ids.where(valid_ids).astype("Int64").astype(str)
    keys = pd.Series(np.where(valid_ids, "id:"+safe_ids, "name:"+keys), index=frame.index)
    history = {k: list(v) for k,v in state.get("horses", {}).items()}
    out = {c: np.full(len(frame), np.nan) for c in FEATURES}
    finish = pd.to_numeric(frame.get("finish_position", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy()
    labels = frame.get("finish_pos_label", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().to_numpy()
    descriptions = frame.get("ride_description", pd.Series(None, index=frame.index)).to_numpy()
    reasons = frame.get("nonfinish_reason", pd.Series(None, index=frame.index)).to_numpy()
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), dates.to_numpy()))
    day_values = dates.to_numpy()[order]
    boundaries = np.r_[0, np.flatnonzero(day_values[1:] != day_values[:-1])+1, len(frame)]
    actual_dates = []
    for a,b in zip(boundaries[:-1], boundaries[1:]):
        rows = order[a:b]
        for i in rows:
            prior = history.get(keys.iloc[i], [])
            tail = prior[-3:]
            support = sum(np.isfinite(r["impeded"]) for r in tail)
            out["context_description_support_3"][i] = support
            for name in PATTERNS:
                values = [r[name] for r in tail if np.isfinite(r[name])]
                out["context_"+name+"_mention_rate_3"][i] = np.mean(values) if values else np.nan
            known = [r for r in prior if r["reason"] is not None]
            out["context_reason_support_5"][i] = len(known)
            if known:
                out["context_accidental_nonfinish_rate_5"][i] = np.mean([r["reason"] in {"F", "UR", "BD", "Fell", "UnseatedRider", "BroughtDown"} for r in known])
                out["context_pulled_up_rate_5"][i] = np.mean([r["reason"] in {"PU", "PulledUp"} for r in known])
        for i in rows:
            label = labels[i]
            started = finish[i] > 0 or label not in {"", "NR", "NONRUNNER", "NON-RUNNER", "WITHDRAWN"}
            if not started:
                continue
            key = keys.iloc[i]
            reason = reasons[i].strip() if isinstance(reasons[i], str) and reasons[i].strip() else None
            if reason not in {"F","UR","BD","PU","R","Fell","UnseatedRider","BroughtDown","PulledUp","Refused"}:
                reason = None
            if finish[i] > 0:
                reason = "FINISHED"
            elif reason is None and label in {"F", "UR", "BD", "PU", "R"}:
                reason = label
            history[key] = (history.get(key, [])+[{**description_mentions(descriptions[i]), "reason": reason}])[-5:]
            actual_dates.append(dates.iloc[i])
    result = pd.DataFrame(out, index=frame.index)
    updated = {"horses": history, "last_date": str(max(actual_dates).date()) if actual_dates else state.get("last_date")}
    return (result, updated) if return_state else result
