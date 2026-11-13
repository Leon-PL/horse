"""Where a trainer chooses to start a horse, against that trainer's own habit.

The loss to the morning price is in races full of unraced horses, where the
stable knows something the form book cannot. One public trace of that
knowledge is the race chosen for a debut: a yard that usually starts its
newcomers in small races and sends this one to a valuable maiden is saying
something. The newcomer group (``newcomer.py``) keyed the model's errors by
family and by trainer with age; nothing keys them by the race chosen.

``debut-choice``, declared before scoring.

- The trainer's habit: the mean log prize of the trainer's earlier first-time
  starters under the same code (flat or jumps), from the start of the store,
  strictly earlier dates, shrunk towards the all-trainer mean by five runs.
  Inputs: today's log prize minus that habit, the habit itself, and the number
  of earlier first-time starters; given for horses with at most two runs.
- The shrunk, 365-day-decayed residual and market-gap means of
  ``entity_residuals`` keyed by: trainer, horse's state and whether today's
  race is above or below the trainer's habit by 0.25 in log prize; trainer,
  state and track; track and state; trainer, jockey and state; trainer, state
  and season (to June, July-August, later) for two-year-olds.
"""
import numpy as np
import pandas as pd

from research.error_anatomy.newcomer import with_keys
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
SHRINK = 5.
STEP = .25
ENTITIES = {"trainer_choice": ["trainer", "_fresh", "_choice"], "trainer_fresh_track": ["trainer", "_fresh", "track"],
            "track_fresh": ["track", "_fresh"], "trainer_jockey_fresh": ["trainer", "jockey", "_fresh"],
            "trainer_season": ["trainer", "_fresh", "_season"]}


def habits():
    """Per runner in the store: the trainer's earlier first-time starters' mean log prize under today's code."""
    store = sc.read_store(["trainer", "race_type", "prize_log", "horse_prev_races", "age", "month"])
    store["_code"] = np.where(store.race_type.isin(["Hurdle", "Chase"]), "J", "F")
    prize = pd.to_numeric(store.prize_log, errors="coerce")
    debut = pd.to_numeric(store.horse_prev_races, errors="coerce").eq(0) & prize.notna() & store.trainer.notna()
    first = store.loc[debut, ["trainer", "_code", "race_date"]].assign(total=prize[debut], runs=1.)
    daily = first.groupby(["trainer", "_code", "race_date"], sort=True)[["total", "runs"]].sum()
    totals = daily.groupby(level=["trainer", "_code"]).cumsum().reset_index().sort_values("race_date")
    lookup = store[["trainer", "_code", "race_date"]].reset_index().sort_values("race_date", kind="stable")
    known = lookup.trainer.notna()
    joined = pd.merge_asof(lookup[known], totals, on="race_date", by=["trainer", "_code"], allow_exact_matches=False)
    joined = joined.set_index("index").reindex(store.index)
    runs = joined.runs.fillna(0.)
    # The all-trainer mean of earlier debuts, by code, as the shrinkage target (expanding, earlier dates only).
    overall = first.groupby(["_code", "race_date"], sort=True)[["total", "runs"]].sum()
    overall = overall.groupby(level="_code").cumsum().reset_index().sort_values("race_date")
    centre = pd.merge_asof(store[["_code", "race_date"]].reset_index().sort_values("race_date", kind="stable"), overall,
                           on="race_date", by="_code", allow_exact_matches=False).set_index("index").reindex(store.index)
    centre = (centre.total / centre.runs).fillna(prize.mean())
    habit = (joined.total.fillna(0.) + SHRINK * centre) / (runs + SHRINK)
    out = store[KEYS].copy()
    lightly = pd.to_numeric(store.horse_prev_races, errors="coerce").le(2)
    out["dc_prize_vs_habit"] = (prize - habit).where(lightly).astype("float32")
    out["dc_habit"] = habit.where(lightly).astype("float32")
    out["dc_debuts"] = runs.where(lightly).astype("float32")
    out["_choice"] = np.select([(prize - habit) > STEP, (prize - habit) < -STEP], ["above", "below"], "usual")
    month = pd.to_numeric(store.month, errors="coerce")
    two = pd.to_numeric(store.age, errors="coerce").eq(2)
    out["_season"] = np.where(two, np.select([month <= 6, month <= 8], ["early", "mid"], "late"), "not2")
    return out


def choice_inputs(frame):
    extra = habits()
    extra["horse_id"] = extra.horse_id.astype(frame.horse_id.dtype)
    work = frame[KEYS].merge(extra, on=KEYS, how="left", validate="one_to_one")
    values = with_keys(frame, {"_choice": work._choice.to_numpy(object), "_season": work._season.to_numpy(object)},
                       ENTITIES, "dc")
    values = values.rename(columns={c: c.replace("mg_", "dcg_") for c in values if c.startswith("mg_")})
    direct = ["dc_prize_vs_habit", "dc_habit", "dc_debuts"]
    values = values.merge(work[[*KEYS, *direct]], on=KEYS, validate="one_to_one")
    values["dc_prize_vs_habit_vs_field"] = (values.dc_prize_vs_habit.fillna(0.)
                                            - values.dc_prize_vs_habit.fillna(0.).groupby(frame.race_id.to_numpy()).transform("mean")
                                            ).astype("float32")
    return values


def load(frame):
    values = choice_inputs(frame)
    return values, [c for c in values if c not in KEYS]
