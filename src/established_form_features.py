"""Prior-day discipline experience and comparable handicap-mark histories.

The same bounded state serves full builds, appends and read-only live cards.
Handicap context uses observed marks and declared weights, never field-median
substitutes. These are historical descriptors, not estimated performance ratings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.audited_surface_features import synthetic_surface
from src.feature_semantics import observed_rating
from src.result_status import NONSTARTER_LABELS, result_labels
from src.utils import normalise_horse_key

FEATURES = (
    "race_is_bumper", "race_is_novice", "race_is_maiden",
    "horse_discipline_runs_log", "horse_discipline_exposure_share",
    "horse_discipline_win_rate_shrunk", "horse_discipline_place_rate_shrunk",
    "horse_discipline_form_3", "horse_discipline_dnf_rate_shrunk",
    "days_since_discipline_race_log", "or_vs_last_win_mark",
    "or_vs_last_place_mark", "or_vs_career_high_prev",
    "handicap_mark_vs_high_5", "handicap_mark_support_5",
    "handicap_weight_vs_last_win", "handicap_discipline_runs_log",
)
LEGACY_FLAGS = ("below_last_win_mark", "above_career_high_or")
STATE_KEY = "established_form::v1"


def race_discipline(frame):
    """Split NH Flat/bumper titles from the legacy coarse Flat code.

    This provider also calls Irish bumpers 'Flat Race': verified against dated
    public performance records, without using their finishing positions.
    Hurdle/Chase codes take precedence over advertising words in their titles.
    """
    kind = frame.get("race_type", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip().str.casefold()
    mapped = kind.map({"flat": "Flat", "hurdle": "Hurdle", "chase": "Chase",
                       "nh flat": "Bumper", "nhflat": "Bumper", "bumper": "Bumper"})
    title = frame.get("race_name", pd.Series("", index=frame.index)).fillna("").astype(str)
    kinds = title.str.lower().str.findall(r"\b(?:[i.]?n[.]?h[.]?\s+flat|national\s+hunt\s+flat|flat\s+race|bumpers?|handicap|stakes|hurdle|chase)\b")
    # Advertising a bumper before the actual 'Handicap'/'Stakes' title does
    # not make an ordinary Flat race a bumper.
    final_kind = kinds.map(lambda values: values[-1] if values else "")
    explicit = final_kind.str.contains(r"flat|bumper", regex=True)
    return mapped.mask(mapped.eq("Flat") & explicit, "Bumper").astype("string")


def mark_scale(frame):
    """Separate Flat turf/AW, hurdle, chase and bumper rating histories."""
    code = race_discipline(frame)
    surface = synthetic_surface(frame.get("surface", pd.Series(None, index=frame.index)))
    scale = code.copy()
    flat = code.eq("Flat").fillna(False)
    scale.loc[flat] = surface.loc[flat].map({0.: "Flat|0", 1.: "Flat|1"}).astype("string")
    return scale


def _horse_keys(frame):
    # IDs read back from parquet/cards may be integers, floats or text.
    ids = pd.to_numeric(frame.get("horse_id", pd.Series(np.nan, index=frame.index)), errors="coerce")
    ids = ids.where(ids.gt(0) & np.isfinite(ids) & ids.mod(1).eq(0)).astype("Int64").astype("string")
    names = frame["horse_name"].map(normalise_horse_key).replace("", pd.NA).astype("string")
    return ("id:"+ids).fillna("name:"+names).to_numpy(dtype=object, na_value=None)


def established_form_features(frame, *, state=None, return_state=False):
    """Return inputs aligned to frame; exclude all outcomes on the current day.

    A seed ends on a complete earlier date. Pending/nonstarter rows never
    advance history. At most three finish-quality values and five handicap
    marks per horse/scale are carried. Input state is never mutated.
    """
    n = len(frame)
    outputs = {c: np.full(n, np.nan) for c in (*FEATURES, *LEGACY_FLAGS)}
    state = state or {}
    dates = pd.to_datetime(frame.race_date, errors="raise").dt.normalize()
    if dates.isna().any():
        raise ValueError("Discipline history requires a date for every runner")
    if n and state.get("last_date") and dates.min() <= pd.Timestamp(state["last_date"]):
        raise ValueError("Append discipline history only after the last complete seeded date")
    code = race_discipline(frame).to_numpy(dtype=object, na_value=None)
    scales = mark_scale(frame).to_numpy(dtype=object, na_value=None)
    keys = _horse_keys(frame)
    title = frame.get("race_name", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().str.replace("[’']", "", regex=True)
    modifiers = r"(?:(?:limited|handicap|selling|claiming|fillies|mares|juvenile|auction)\s+)*"
    outputs["race_is_bumper"] = (code == "Bumper").astype(float)
    outputs["race_is_novice"] = title.str.contains(r"\bnovices?\s+"+modifiers+r"(?:hurdle|chase|stakes|race)\b", regex=True).to_numpy(float)
    outputs["race_is_maiden"] = (
        title.str.contains(r"\bmaiden\s+"+modifiers+r"(?:hurdle|chase|stakes|race)\b", regex=True)
        | title.str.contains(r"\bbeginners?\s+chase\b", regex=True)
    ).to_numpy(float)
    marks = observed_rating(frame).to_numpy(float) if "official_rating" in frame else np.full(n, np.nan)
    weight = pd.to_numeric(frame.get("weight_lbs", pd.Series(np.nan, index=frame.index)), errors="coerce")
    known_weight = weight.between(50, 220)
    if "has_observed_weight" in frame:
        known_weight &= frame.has_observed_weight.eq(1)
    weights = weight.where(known_weight).to_numpy(float)
    handicap = pd.to_numeric(frame.get("handicap", pd.Series(0, index=frame.index)), errors="coerce").eq(1).to_numpy()
    finish = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    fields = pd.to_numeric(frame.num_runners, errors="coerce").to_numpy(float)
    labels = result_labels(frame)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    days = dates.to_numpy(dtype="datetime64[D]").astype(np.int64)
    # Deterministic within-day updates; outcomes are read only after emitting
    # every runner's start-of-day features. Event order matters for last marks.
    off = frame.get("off_time", pd.Series("12:00", index=frame.index)).astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    minutes = pd.to_numeric(off[0], errors="coerce").fillna(12)*60 + pd.to_numeric(off[1], errors="coerce").fillna(0)
    order = np.lexsort((frame.race_id.astype(str).to_numpy(), minutes.to_numpy(), days))
    discipline = dict(state.get("discipline", {}))
    mark_history = dict(state.get("marks", {}))
    totals = dict(state.get("totals", {}))
    copied_d, copied_m = set(), set()
    boundaries = np.r_[0, np.flatnonzero(np.diff(days[order]))+1, n]
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        rows = order[lo:hi]
        if not len(rows):
            continue
        day = days[rows[0]]
        for i in rows:
            horse, kind, scale = keys[i], code[i], scales[i]
            if horse is None or kind is None:
                continue
            d = discipline.get((horse, kind), {})
            runs = d.get("runs", 0)
            outputs["horse_discipline_runs_log"][i] = np.log1p(runs)
            outputs["horse_discipline_exposure_share"][i] = runs/totals[horse] if totals.get(horse, 0) else np.nan
            outputs["horse_discipline_win_rate_shrunk"][i] = (d.get("wins", 0)+.6)/(runs+6.)
            outputs["horse_discipline_place_rate_shrunk"][i] = (d.get("places", 0)+1.8)/(runs+6.)
            outputs["horse_discipline_dnf_rate_shrunk"][i] = (d.get("dnfs", 0)+.5)/(runs+10.)
            quality = [v for v in d.get("quality", []) if np.isfinite(v)]
            outputs["horse_discipline_form_3"][i] = np.mean(quality) if quality else np.nan
            if "day" in d:
                outputs["days_since_discipline_race_log"][i] = np.log1p(day-d["day"])
            if not handicap[i] or scale is None:
                continue
            m = mark_history.get((horse, scale), {})
            outputs["handicap_discipline_runs_log"][i] = np.log1p(m.get("runs", 0))
            tail = [v for v in m.get("tail", []) if np.isfinite(v)]
            outputs["handicap_mark_support_5"][i] = len(tail)
            for column, prior in (("or_vs_last_win_mark", "win"), ("or_vs_last_place_mark", "place"), ("or_vs_career_high_prev", "high")):
                outputs[column][i] = marks[i]-m.get(prior, np.nan)
            outputs["handicap_mark_vs_high_5"][i] = marks[i]-max(tail) if tail else np.nan
            outputs["handicap_weight_vs_last_win"][i] = weights[i]-m.get("win_weight", np.nan)
            if np.isfinite(outputs["or_vs_last_win_mark"][i]):
                outputs["below_last_win_mark"][i] = float(outputs["or_vs_last_win_mark"][i] < 0)
            if np.isfinite(outputs["or_vs_career_high_prev"][i]):
                outputs["above_career_high_or"][i] = float(outputs["or_vs_career_high_prev"][i] > 0)
        for i in rows:
            horse, kind, scale = keys[i], code[i], scales[i]
            if not starts[i] or horse is None or kind is None:
                continue
            key = (horse, kind)
            if key not in copied_d:
                discipline[key] = {**discipline.get(key, {})}
                discipline[key]["quality"] = list(discipline[key].get("quality", []))
                copied_d.add(key)
            d = discipline[key]
            d["runs"] = d.get("runs", 0)+1
            d["wins"] = d.get("wins", 0)+int(finish[i] == 1)
            d["places"] = d.get("places", 0)+int(1 <= finish[i] <= 3)
            d["dnfs"] = d.get("dnfs", 0)+int(not finish[i] > 0)
            quality = 1-(finish[i]-1)/(fields[i]-1) if fields[i] >= 2 and 1 <= finish[i] <= fields[i] else np.nan
            d["quality"] = (d["quality"]+[quality])[-3:]
            d["day"] = int(day)
            totals[horse] = totals.get(horse, 0)+1
            if scale is None:
                continue
            key = (horse, scale)
            if key not in copied_m:
                mark_history[key] = {**mark_history.get(key, {})}
                mark_history[key]["tail"] = list(mark_history[key].get("tail", []))
                copied_m.add(key)
            m = mark_history[key]
            if np.isfinite(marks[i]):
                m["high"] = max(m.get("high", -np.inf), marks[i])
            if handicap[i]:
                m["runs"] = m.get("runs", 0)+1
                m["tail"] = (m["tail"]+[marks[i]])[-5:]
                if finish[i] == 1:
                    m["win"], m["win_weight"] = marks[i], weights[i]
                if 1 <= finish[i] <= 3:
                    m["place"] = marks[i]
    out = pd.DataFrame(outputs, index=frame.index)
    advanced = {"discipline": discipline, "marks": mark_history, "totals": totals,
                "last_date": str(dates.loc[starts].max().date()) if starts.any() else state.get("last_date")}
    return (out, advanced) if return_state else out
