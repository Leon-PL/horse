"""A horse's record for its present yard, in today's discipline, and Irish yards sending runners to Britain.

    python -m research.error_anatomy.yard_change        # describe the model's errors by situation

Started from one race (Cheltenham 12:35, 14 November 2026): Dinoland ran 20
times for two yards with two wins, then won two chases and was second in
another in six runs for a third yard, and was sent from Ireland to Cheltenham.
The model rated him an average runner. Three things about him are in no input:

- his form divides at a change of yard (the store has no stable-change input;
  ``plot_races`` tried a bare flag only);
- his chase form for the present yard is far better than his career record
  (ratings mix disciplines and yards);
- an Irish yard chose to send him to Britain (``region`` in the store is one
  constant; trainer travel in miles was tested, the trainer's base was not).

``yard``, declared before scoring. For each runner, from the horse's runs on
strictly earlier dates:

- the spell: unbroken earlier runs for today's trainer. Runs in the spell,
  whether the horse ran for another yard before, runs before the spell;
- finishing percentile (1 = won, 0 = last or did not finish), shrunk towards
  0.5 by two runs: mean in the spell, mean before it, and the difference;
- the same in the spell under today's race type, with runs and wins;
- official rating today minus the rating at the first run of the spell, and
  minus the rating six runs back;
- the trainer's share of earlier-date runners at Irish tracks, and whether
  today is away from that base;
- the field-relative versions of the four means and rating gains;
- shrunk, 365-day-decayed residual and market-gap means of
  ``entity_residuals`` keyed by trainer and stage of the spell (first run for
  the yard, runs two to four, later, or never moved), by trainer, stage and
  code, by trainer and home/away, by home/away and code, and by home/away and
  track.
"""
import numpy as np
import pandas as pd

from research.error_anatomy.newcomer import with_keys
from research.meta_learner import screen_candidates as sc
from research.meta_learner import screening as s

KEYS = s.KEYS
SHRINK = 2.
IRISH = {"ballinrobe", "bellewstown", "clonmel", "cork", "curragh", "down royal", "downpatrick", "dundalk",
         "fairyhouse", "galway", "gowran park", "kilbeggan", "killarney", "laytown", "leopardstown", "limerick",
         "listowel", "naas", "navan", "punchestown", "roscommon", "sligo", "thurles", "tipperary", "tramore",
         "wexford"}
BASE_RUNS = 20          # earlier runners before a trainer's base is called
ENTITIES = {"trainer_stage": ["trainer", "_stage"], "trainer_stage_code": ["trainer", "_stage", "_code"],
            "trainer_away": ["trainer", "_away"], "away_code": ["_away", "_code"], "away_track": ["_away", "track"]}


def _prior_mean(total, count, centre=.5):
    return (total + SHRINK * centre) / (count + SHRINK)


def situations():
    """One row per runner in the store: spell, form by spell and discipline, rating gains, trainer base."""
    store = sc.read_store(["trainer", "race_type", "track", "finish_position", "num_runners", "official_rating"])
    store["_row"] = np.arange(len(store))
    h = store.sort_values(["horse_id", "race_date", "_row"], kind="stable").reset_index(drop=True)
    horse = h.groupby("horse_id", sort=False)
    position = pd.to_numeric(h.finish_position, errors="coerce")
    runners = pd.to_numeric(h.num_runners, errors="coerce")
    pct = (1 - (position - 1) / (runners - 1).clip(lower=1)).where(position >= 1, 0.).clip(0, 1)
    first = horse.cumcount().eq(0)
    moved = (h.trainer != horse.trainer.shift(1)) & ~first
    h["_spell"] = moved.groupby(h.horse_id, sort=False).cumsum()
    spell = [h.horse_id, h._spell]
    code = [h.horse_id, h._spell, h.race_type]
    one = pd.Series(1., index=h.index)

    def before(values, by):          # running total over earlier rows of the group
        return values.groupby(by, sort=False).cumsum() - values

    career_n, career_sum = before(one, h.horse_id), before(pct, h.horse_id)
    spell_n, spell_sum = before(one, spell), before(pct, spell)
    code_n, code_sum = before(one, code), before(pct, code)
    wins = position.eq(1).astype(float)
    out = h[KEYS + ["_row"]].copy()
    out["yd_runs"] = spell_n
    out["yd_changed"] = h._spell.gt(0).astype(float)
    out["yd_runs_before"] = career_n - spell_n
    out["yd_pct"] = _prior_mean(spell_sum, spell_n).where(spell_n > 0)
    out["yd_pct_before"] = _prior_mean(career_sum - spell_sum, career_n - spell_n).where(career_n > spell_n)
    out["yd_gain"] = out.yd_pct - out.yd_pct_before
    out["yd_code_runs"] = code_n
    out["yd_code_pct"] = _prior_mean(code_sum, code_n).where(code_n > 0)
    out["yd_code_wins"] = before(wins, code)
    out["yd_code_vs_all"] = out.yd_code_pct - _prior_mean(career_sum, career_n).where(career_n > 0)
    rating = pd.to_numeric(h.official_rating, errors="coerce")
    start = rating.groupby(spell, sort=False).transform("first")
    out["yd_or_gain"] = (rating - start).where(spell_n > 0)
    out["yd_or_gain6"] = rating - rating.groupby(h.horse_id, sort=False).shift(6)
    # A second run on the same date is not an earlier date.
    same_day = (horse.race_date.shift(1) == h.race_date).to_numpy()
    columns = [c for c in out if c.startswith("yd_")]
    out.loc[same_day, columns] = np.nan
    stage = np.select([h._spell.eq(0), spell_n.eq(0), spell_n.le(3)], ["never", "first", "early"], "settled")
    out["_stage"] = stage
    out = out.sort_values("_row").reset_index(drop=True)
    # The trainer's base: share of its runners on strictly earlier dates that ran at Irish tracks.
    irish = store.track.astype(str).str.lower().str.replace("-", " ").isin(IRISH)
    daily = pd.DataFrame({"trainer": store.trainer, "race_date": store.race_date, "n": 1., "ire": irish.astype(float)})
    daily = daily.groupby(["trainer", "race_date"], sort=True)[["n", "ire"]].sum()
    totals = daily.groupby(level="trainer").cumsum().reset_index().sort_values("race_date")
    lookup = store[["trainer", "race_date"]].reset_index().sort_values("race_date", kind="stable")
    joined = pd.merge_asof(lookup, totals, on="race_date", by="trainer", allow_exact_matches=False)
    joined = joined.set_index("index").reindex(store.index)
    share = (joined.ire / joined.n).where(joined.n >= BASE_RUNS)
    out["yd_trainer_irish"] = share.to_numpy()
    away = np.select([share.isna(), (share > .8) & ~irish, (share < .2) & irish], ["unknown", "ire_in_gb", "gb_in_ire"],
                     "home")
    out["_away"] = away
    out["yd_away"] = np.select([away == "ire_in_gb", away == "gb_in_ire"], [1., -1.], 0.)
    out[[c for c in out if c.startswith("yd_")]] = out[[c for c in out if c.startswith("yd_")]].astype("float32")
    return out.drop(columns="_row")


def yard_inputs(frame):
    extra = situations()
    extra["horse_id"] = extra.horse_id.astype(frame.horse_id.dtype)
    extra["race_id"] = extra.race_id.astype(frame.race_id.dtype)
    work = frame[KEYS].merge(extra, on=KEYS, how="left", validate="one_to_one")
    values = with_keys(frame, {"_stage": work._stage.to_numpy(object), "_away": work._away.to_numpy(object)},
                       ENTITIES, "yd")
    values = values.rename(columns={c: c.replace("mg_", "ydg_") for c in values if c.startswith("mg_")})
    direct = [c for c in work if c.startswith("yd_")]
    values = values.merge(work[[*KEYS, *direct]], on=KEYS, validate="one_to_one")
    race = frame.race_id.to_numpy()
    for column in ("yd_pct", "yd_gain", "yd_code_pct", "yd_or_gain"):
        filled = values[column].fillna(values[column].groupby(race).transform("mean"))
        values[f"{column}_vs_field"] = (filled - filled.groupby(race).transform("mean")).astype("float32")
    return values, work[["_stage", "_away"]]


def load(frame):
    values, _ = yard_inputs(frame)
    return values, [c for c in values if c not in KEYS]


def describe():
    """Winners against the forecast and the closing price, by situation, on the scored quarters."""
    import research.error_anatomy.frame  # noqa: F401  (store fallback)
    from research.error_anatomy import soft_corrections as soft
    frame = soft.frames()["long"]
    values, keys = yard_inputs(frame)
    scored = frame.quarter.isin(s.SCORED).to_numpy()
    f = pd.concat([frame.loc[scored, ["won", "p_single", "p_market", "race_type"]].reset_index(drop=True),
                   values.loc[scored].reset_index(drop=True), keys.loc[scored].reset_index(drop=True)], axis=1)
    f = f.loc[f.p_market.notna()]
    jump = f.race_type.isin(["Hurdle", "Chase"])
    groups = {"all": pd.Series(True, index=f.index),
              "first run for a new yard": f._stage.eq("first"),
              "runs 2-4 for a new yard": f._stage.eq("early"),
              "runs 2-4, form up 0.15+ on before": f._stage.eq("early") & f.yd_gain.gt(.15),
              "settled after a move, form up 0.15+": f._stage.eq("settled") & f.yd_gain.gt(.15),
              "today's code beats career by 0.15+ (2+ runs)": f.yd_code_vs_all.gt(.15) & f.yd_code_runs.ge(2),
              "chase, 2+ chase wins for this yard": f.race_type.eq("Chase") & f.yd_code_wins.ge(2),
              "rating up 10+ in this spell": f.yd_or_gain.ge(10),
              "rating up 15+ in six runs": f.yd_or_gain6.ge(15),
              "Irish yard in Britain": f._away.eq("ire_in_gb"),
              "Irish yard in Britain, jumps": f._away.eq("ire_in_gb") & jump,
              "British yard in Ireland": f._away.eq("gb_in_ire")}
    print(f"{'situation':<46}{'runners':>8}{'winners':>9}{'model':>9}{'market':>9}{'won/model':>11}{'won/market':>12}")
    for name, rows in groups.items():
        g = f.loc[rows]
        won, model, market = g.won.sum(), g.p_single.sum(), g.p_market.sum()
        print(f"{name:<46}{len(g):>8}{won:>9.0f}{model:>9.1f}{market:>9.1f}{won / model:>11.3f}{won / market:>12.3f}")


if __name__ == "__main__":
    describe()
