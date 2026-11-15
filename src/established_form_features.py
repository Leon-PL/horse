"""Prior-day discipline experience and comparable handicap-mark histories.

The same bounded state serves full builds, appends and read-only live cards.
Handicap context uses observed marks and declared weights, never field-median
substitutes. These are historical descriptors, not estimated performance ratings.
"""
from __future__ import annotations

from math import inf, isfinite, nan

import numpy as np
import pandas as pd

from src.surface_features import synthetic_surface
from src.feature_semantics import claim_adjusted_weight, observed_rating
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


# np.log1p of the whole numbers this loop meets, looked up per cell.
_LOG1P = np.log1p(np.arange(1 << 15)).tolist()


def _log1p(value):
    return _LOG1P[value] if 0 <= value < len(_LOG1P) else float(np.log1p(value))


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
    weights = claim_adjusted_weight(frame).to_numpy(float)
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
    # This loop reads and writes one cell at a time, which is far cheaper on
    # Python lists than on arrays.
    keys, code, scales = keys.tolist(), code.tolist(), scales.tolist()
    marks, weights, handicap = marks.tolist(), weights.tolist(), handicap.tolist()
    finish, fields, starts_at = finish.tolist(), fields.tolist(), starts.tolist()
    day_of, order_of = days.tolist(), order.tolist()
    missing = [nan]*n
    runs_log, exposure, win_rate, place_rate, dnf_rate, form = (list(missing) for _ in range(6))
    days_since, mark_runs, mark_support, vs_win, vs_place, vs_high = (list(missing) for _ in range(6))
    vs_recent_high, vs_win_weight, below_win, above_high = (list(missing) for _ in range(4))
    for lo, hi in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
        rows = order_of[lo:hi]
        if not rows:
            continue
        day = day_of[rows[0]]
        for i in rows:
            horse, kind, scale = keys[i], code[i], scales[i]
            if horse is None or kind is None:
                continue
            d = discipline.get((horse, kind), {})
            runs = d.get("runs", 0)
            runs_log[i] = _log1p(runs)
            exposure[i] = runs/totals[horse] if totals.get(horse, 0) else nan
            win_rate[i] = (d.get("wins", 0)+.6)/(runs+6.)
            place_rate[i] = (d.get("places", 0)+1.8)/(runs+6.)
            dnf_rate[i] = (d.get("dnfs", 0)+.5)/(runs+10.)
            quality = 0.
            seen = 0
            for value in d.get("quality", ()):
                if isfinite(value):
                    quality += value
                    seen += 1
            if seen:
                form[i] = quality/seen
            if "day" in d:
                days_since[i] = _log1p(day-d["day"])
            if not handicap[i] or scale is None:
                continue
            m = mark_history.get((horse, scale), {})
            mark = marks[i]
            mark_runs[i] = _log1p(m.get("runs", 0))
            tail = [v for v in m.get("tail", ()) if isfinite(v)]
            mark_support[i] = len(tail)
            margin_win = vs_win[i] = mark-m.get("win", nan)
            vs_place[i] = mark-m.get("place", nan)
            margin_high = vs_high[i] = mark-m.get("high", nan)
            if tail:
                vs_recent_high[i] = mark-max(tail)
            vs_win_weight[i] = weights[i]-m.get("win_weight", nan)
            if isfinite(margin_win):
                below_win[i] = float(margin_win < 0)
            if isfinite(margin_high):
                above_high[i] = float(margin_high > 0)
        for i in rows:
            horse, kind, scale = keys[i], code[i], scales[i]
            if not starts_at[i] or horse is None or kind is None:
                continue
            key = (horse, kind)
            if key not in copied_d:
                discipline[key] = {**discipline.get(key, {})}
                discipline[key]["quality"] = list(discipline[key].get("quality", []))
                copied_d.add(key)
            d = discipline[key]
            place, field = finish[i], fields[i]
            d["runs"] = d.get("runs", 0)+1
            d["wins"] = d.get("wins", 0)+int(place == 1)
            d["places"] = d.get("places", 0)+int(1 <= place <= 3)
            d["dnfs"] = d.get("dnfs", 0)+int(not place > 0)
            quality = 1-(place-1)/(field-1) if field >= 2 and 1 <= place <= field else nan
            d["quality"] = (d["quality"]+[quality])[-3:]
            d["day"] = day
            totals[horse] = totals.get(horse, 0)+1
            if scale is None:
                continue
            key = (horse, scale)
            if key not in copied_m:
                mark_history[key] = {**mark_history.get(key, {})}
                mark_history[key]["tail"] = list(mark_history[key].get("tail", []))
                copied_m.add(key)
            m = mark_history[key]
            mark = marks[i]
            if isfinite(mark):
                m["high"] = max(m.get("high", -inf), mark)
            if handicap[i]:
                m["runs"] = m.get("runs", 0)+1
                m["tail"] = (m["tail"]+[mark])[-5:]
                if place == 1:
                    m["win"], m["win_weight"] = mark, weights[i]
                if 1 <= place <= 3:
                    m["place"] = mark
    filled = {
        "horse_discipline_runs_log": runs_log, "horse_discipline_exposure_share": exposure,
        "horse_discipline_win_rate_shrunk": win_rate, "horse_discipline_place_rate_shrunk": place_rate,
        "horse_discipline_dnf_rate_shrunk": dnf_rate, "horse_discipline_form_3": form,
        "days_since_discipline_race_log": days_since, "handicap_discipline_runs_log": mark_runs,
        "handicap_mark_support_5": mark_support, "or_vs_last_win_mark": vs_win,
        "or_vs_last_place_mark": vs_place, "or_vs_career_high_prev": vs_high,
        "handicap_mark_vs_high_5": vs_recent_high, "handicap_weight_vs_last_win": vs_win_weight,
        "below_last_win_mark": below_win, "above_career_high_or": above_high,
    }
    outputs.update({name: np.array(values, dtype="float64") for name, values in filled.items()})
    out = pd.DataFrame(outputs, index=frame.index)
    advanced = {"discipline": discipline, "marks": mark_history, "totals": totals,
                "last_date": str(dates.loc[starts].max().date()) if starts.any() else state.get("last_date")}
    return (out, advanced) if return_state else out
