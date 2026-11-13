"""Running style and pace set-up from in-running comments, with full history.

The selection of 12 November kept none of the store's 37 pace inputs. They are
built from At The Races data that starts in August 2023 (0% of rows before
2023, 17-25% in 2023, about two thirds after), so most training rows have no
value. Racing TV's close-up comment ("led, headed 2f out...", "held up in
rear...") exists for every result since September 2020 and was never used for
running style.

``pace-comments``, declared before scoring. Each past run gets an early
position score from the first matching phrase of its comment (1 led, 0.75
prominent, 0.5 midfield, 0.25 held up, 0 rear). Inputs, all from runs on
strictly earlier dates:

- the horse's style: run-weighted average, last run, share of runs it led,
  number of scored runs;
- the race: how many likely leaders, the field's summed lead rates without
  this horse (pace pressure), the horse's style against the field and as a
  rank, whether it is the only likely leader;
- the course: how much better leaders have done than hold-up horses over this
  track, surface and distance band on earlier dates (shrunk), alone and times
  the horse's style;
- style times pace pressure.
"""
import re

import numba as nb
import numpy as np
import pandas as pd

from research.error_anatomy.frame import OUT
from research.error_anatomy.owner import linked_runner_columns
from research.meta_learner import screening as s
from src.ingestion.storage import atomic_parquet

KEYS = s.KEYS
SAVED = OUT / "run_style.parquet"
STORE = "data/processed/feature_store/featured.parquet"
# Order matters only for ties at the same character position; the earliest phrase in the comment wins.
PHRASES = [
    (1.0, r"made all|made most|\bled\b|disputed lead|disputed the lead|with leader|pressed leader|soon led|in front"),
    (0.75, r"prominent|chased leader|tracked leader|close up|pressed leaders|chased winner|chased clear leader|handy"),
    (0.5, r"mid-division|midfield|in touch|chased leaders|tracked leaders|mid-field"),
    (0.25, r"held up|towards rear|waited with|steadied start|patiently ridden"),
    (0.0, r"in rear|always behind|behind|outpaced|slowly away|slowly into stride|detached|last pair|dwelt|missed break"),
]
COMPILED = [(score, re.compile(pattern)) for score, pattern in PHRASES]
ALPHA = .6


def early_score(comment):
    """Score of the earliest style phrase in the first part of the comment; NaN if none."""
    if not isinstance(comment, str):
        return np.nan
    text = comment.lower()[:90]
    best, where = np.nan, 10 ** 6
    for score, pattern in COMPILED:
        found = pattern.search(text)
        if found and found.start() < where:
            best, where = score, found.start()
    return best


def build():
    table = linked_runner_columns(["comment"])
    table["early"] = table.comment.map(early_score).astype("float32")
    out = table[[*KEYS, "early"]]
    atomic_parquet(out, SAVED)
    return out


@nb.njit(cache=True)
def _history(group, same_day, value, alpha):
    """Per row: weighted mean, last value, lead share and count over the group's earlier-date rows."""
    n = len(value)
    out = np.full((n, 4), np.nan)
    start = 0
    for i in range(n):
        if i == 0 or group[i] != group[i - 1]:
            start = i
        total = weight = leads = count = 0.
        last = np.nan
        w = 1.
        for j in range(i - 1, max(start - 1, i - 13), -1):
            if same_day[i] == same_day[j] or np.isnan(value[j]):
                continue
            if count == 0.:
                last = value[j]
            total += w * value[j]
            weight += w
            leads += value[j] >= 1.
            count += 1.
            w *= alpha
        if count > 0.:
            out[i, 0] = total / weight
            out[i, 1] = last
            out[i, 2] = leads / count
        out[i, 3] = count
    return out


def course_bias(runs):
    """Leaders' beaten-rivals share minus hold-up horses', by course, surface and distance band, earlier dates."""
    work = runs.loc[runs.early.notna() & runs.beat.notna(), ["cell", "race_date", "early", "beat"]].copy()
    work["front"] = (work.early >= .75).astype(float)
    work["back"] = (work.early <= .25).astype(float)
    work["front_beat"] = work.front * work.beat
    work["back_beat"] = work.back * work.beat
    daily = work.groupby(["cell", "race_date"], sort=True)[["front", "back", "front_beat", "back_beat"]].sum()
    totals = daily.groupby(level="cell").cumsum().reset_index().sort_values("race_date")
    lookup = runs[["cell", "race_date"]].reset_index().sort_values("race_date", kind="stable")
    joined = pd.merge_asof(lookup, totals, on="race_date", by="cell", allow_exact_matches=False).set_index("index")
    joined = joined.reindex(runs.index).fillna(0.)
    shrink = 30.
    front = (joined.front_beat + .5 * shrink) / (joined.front + shrink)
    back = (joined.back_beat + .5 * shrink) / (joined.back + shrink)
    return (front - back).astype("float32")


def pace_inputs():
    store = pd.read_parquet(STORE, columns=[*KEYS, "race_date", "finish_position", "num_runners", "track", "surface",
                                            "distance_furlongs"])
    store["race_id"] = store.race_id.astype(str)
    style = pd.read_parquet(SAVED) if SAVED.exists() else build()
    style["horse_id"] = style.horse_id.astype(store.horse_id.dtype)
    runs = store.merge(style, on=KEYS, how="left", validate="one_to_one")
    runs = runs.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index(drop=True)
    position = pd.to_numeric(runs.finish_position, errors="coerce")
    field = runs.num_runners.astype(float)
    runs["beat"] = np.where(position.ge(1), (field - position) / (field - 1).clip(lower=1), 0.).clip(0, 1)
    band = pd.cut(runs.distance_furlongs.astype(float), [0, 5.5, 6.5, 8.5, 10.5, 13, 17, 21, 99], labels=False)
    runs["cell"] = runs.track.astype(str) + "|" + runs.surface.astype(str) + "|" + band.astype(str)
    day = (runs.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    made = _history(runs.horse_id.to_numpy(np.int64), day, runs.early.to_numpy(float), ALPHA)
    out = runs[KEYS].copy()
    out["pc_style"], out["pc_last"], out["pc_lead_rate"], out["pc_runs"] = made.T
    race = runs.race_id.to_numpy()
    grouped = out.groupby(race)
    lead = out.pc_lead_rate.fillna(0.)
    likely = (out.pc_lead_rate >= .5).astype(float)
    out["pc_leaders"] = likely.groupby(race).transform("sum")
    out["pc_pressure"] = lead.groupby(race).transform("sum") - lead
    out["pc_style_vs_field"] = out.pc_style - grouped.pc_style.transform("mean")
    out["pc_style_rank"] = grouped.pc_style.rank(pct=True, ascending=False)
    out["pc_lone_leader"] = ((likely == 1) & (out.pc_leaders == 1)).astype(float).where(out.pc_style.notna())
    out["pc_known_share"] = out.pc_style.notna().groupby(race).transform("mean")
    out["pc_course_bias"] = course_bias(runs)
    out["pc_style_x_bias"] = (out.pc_style - .5) * out.pc_course_bias
    out["pc_style_x_pressure"] = (out.pc_style - .5) * out.pc_pressure
    columns = [c for c in out if c not in KEYS]
    out[columns] = out[columns].astype("float32")
    return out


def load(frame):
    values = pace_inputs()
    values["horse_id"] = values.horse_id.astype(frame.horse_id.dtype)
    return values, [c for c in values if c not in KEYS]


if __name__ == "__main__":
    table = build()
    print(len(table), "runs,", f"{table.early.notna().mean():.3f} scored")
    print(table.early.value_counts(normalize=True).sort_index().round(3).to_string())
