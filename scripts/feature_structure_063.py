"""Experiment 063: how the win model uses its inputs, and what sub-models could add.

A development screen on the selection quarters 2024Q1-2025Q2 at screen capacity.
It answers two questions before anything is built properly:

1. Does the model lack interactions, race context or domain-shaped inputs?
   ``depth``, ``context``, ``ratings`` and ``form`` change one thing each on the
   production objective (exploded softmax on the 129 production inputs).
2. Which intermediate targets carry something a meta learner could use?
   ``seeds`` measures plain averaging, ``views`` splits the inputs by subject,
   and ``targets`` fits one LightGBM per candidate target. ``stack`` scores all of
   them walk-forward on top of the baseline.

The screen reads the 2 November cache of ``scripts/select_features.py`` (store
FE 69babeb9c4262ee2), which is what every saved 057-062 prediction was made
from, so it stays comparable with them. Its store check is bypassed on purpose:
the cache is a frozen snapshot. Labels come from ``labels.parquet``, keyed on
race and horse. Confirmation quarters and the golden hold-out are never read.

``race_kinds`` tests a finer race description from race titles, and ``teacher``
changes the market share of the first stage's soft target. ``additive`` checks
whether the input, target and sub-model gains add up. Results are written up in
docs/research/FEATURES_AND_STACKING_063.md.

    python -m scripts.feature_structure_063 run depth seeds context ratings ratings_parts form relative_gap \
        depth8 race_kinds teacher combined combined_teacher views targets transfer
    python -m scripts.feature_structure_063 stack
    python -m scripts.feature_structure_063 additive

The two candidates are then confirmed against saved 057 at native capacity on
2025Q3-2026Q2. ``register`` fixes them and the pass rule first, and ``confirm``
refuses to run without that. It also reproduces one saved 057 fold before
fitting anything.

    python -m scripts.feature_structure_063 register
    python -m scripts.feature_structure_063 confirm
    python -m scripts.feature_structure_063 confirm_report
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import model_ideas as mi  # noqa: E402
from scripts.select_features import SELECTION_QUARTERS, Cache, read_names  # noqa: E402
from src import feature_selection as fs  # noqa: E402
from src.forecast_evaluation import paired_interval  # noqa: E402
from src.race_objectives import (CentredRegression, ExplodedSoftmax, fit_boosted, learner_params,  # noqa: E402
                                 race_bounds, race_softmax, raw_scores)

OUT = ROOT / "data/research/feature_structure_063"
BASELINE_DIR = ROOT / "data/research/model_ideas_057/dev"
BASELINE_ARM = "exploded_r0.25"
CONFIRM_BASE_DIR = ROOT / "data/research/model_ideas_057/confirm"
REGISTRATION = OUT / "registration_063.json"
FEATURES = ROOT / "feature_sets/win_selected_056.txt"
STAGE_WEIGHTS = (1.0, 0.25, 0.0625)
THREADS = int(os.environ.get("N_THREADS", "4"))
PURGE_DAYS = 7
QUARTERS = SELECTION_QUARTERS
LAST4 = QUARTERS[2:]

mi.OUT = OUT
mi.THREADS = THREADS
logger = logging.getLogger("feature_structure_063")

#: Race-level variables a tree can use only by interacting with runner inputs.
MODERATORS = [
    "race_type_Chase", "race_type_Flat", "race_type_Hurdle", "handicap", "race_is_novice", "race_is_maiden",
    "race_is_bumper", "race_class_1.0", "race_class_2.0", "race_class_3.0", "race_class_4.0", "race_class_5.0",
    "race_class_6.0", "distance_furlongs", "going_Good", "going_Soft", "going_Heavy", "going_Standard",
    "surface_Turf", "surface_Allweather", "month_sin", "month_cos", "field_avg_or", "field_std_or",
    "field_top2_or_gap", "field_std_elo", "field_top2_elo_gap", "field_experience_gap",
]

VIEWS = {
    "ability": r"(elo|glicko|horse_ts_|^or_|speed_fig|time_fig|handicap_discipline)",
    "connections": r"^(trainer|jockey|jt_|connections|same_trainer)",
    "expert": r"^(rtv_card|rtv_text|rtv_verdict)",
    "telemetry": r"^(rtv_|atr_)",
    "conditions": r"(going|surface|track|course|draw|distance_change|horse_cd_|dist_cat|horse_rt_)",
    "form": r"(horse_avg|horse_ewma|horse_lb|form_slope|horse_prev|prior_form|hidden_form|horse_non_finish|"
            r"days_since|class_change|weight_change|or_change|horse_avg_prize|prize_money|prior_market)",
    # Everything left: age, sex, weight, pedigree, field size and headgear.
    "profile": r".",
}


class Screen:
    """The frozen cache plus row-aligned labels."""

    def __init__(self):
        manifest = json.loads((mi.CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))
        with patch("scripts.select_features.store_meta", lambda _store: manifest):
            self.cache = Cache()
        self.labels = pd.read_parquet(OUT / "labels.parquet")
        self.t = self.cache.targets
        if not np.array_equal(self.labels.race_id.astype(str).to_numpy(), self.t.race_ids.astype(str)):
            raise SystemExit("labels.parquet is not aligned with the cache")
        self.names = read_names(FEATURES)
        self.finish = self.labels.finish_position.to_numpy(np.float64)

    def take(self, rows, extra=None):
        x = self.cache.take(rows, self.names)
        return x if extra is None else np.hstack([x, extra[rows].astype(np.float32)])


# ----------------------------------------------------------------- arms ----

def exploded_predict(s: Screen, *, names=None, extra=None, seed=42, params=None, teacher=None):
    """Production objective at the stage's capacity, or with *params* for the depth ladder.

    *teacher* replaces the 0.5 market share of the first stage's soft target.
    """
    t = s.t
    names = names or s.names
    soft = t.soft_y if teacher is None else np.where(t.market > 0, teacher * t.market + (1 - teacher) * t.won, t.won)

    def take(rows):
        x = s.cache.take(rows, names)
        return x if extra is None else np.hstack([x, extra[rows].astype(np.float32)])

    def predict(train, test, capacity):
        objective = ExplodedSoftmax(t.race_ids[train], s.finish[train], soft[train],
                                    fs.recency_weights(t.race_dates[train]), STAGE_WEIGHTS)
        x, y = take(train), t.won[train].astype(np.float64)
        if params is None:
            model = fit_boosted(x, y, t.race_ids[train], t.race_dates[train], objective=objective.objective,
                                metric=objective.metric, capacity=capacity, seed=seed, n_jobs=THREADS)
        else:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(objective=objective.objective(0, len(train)), metric="None",
                                  **learner_params(params), subsample_freq=1, random_state=seed,
                                  n_jobs=THREADS, verbose=-1)
            model.fit(x, y)
        return race_softmax(raw_scores(model, take(test)), race_bounds(t.race_ids[test]))
    return predict


def labelled_rows(s: Screen, train, y, available=None, cutoff=None, minimum=2):
    rows = train[np.isfinite(y[train])]
    if available is not None:
        rows = rows[available[rows] < np.datetime64(cutoff)]
    if minimum > 1:
        ids = pd.Series(s.t.race_ids[rows])
        rows = rows[ids.groupby(ids).transform("size").to_numpy() >= minimum]
    return rows


def target_predict(s: Screen, y, *, binary=False, available=None):
    """One LightGBM on the production inputs for an intermediate target."""
    t = s.t

    def predict(train, test, _capacity):
        cutoff = pd.Timestamp(t.race_dates[test].min()) - pd.Timedelta(days=PURGE_DAYS)
        rows = labelled_rows(s, train, y, available, cutoff, minimum=1 if binary else 2)
        weights = fs.recency_weights(t.race_dates[rows])
        x = s.take(rows)
        if binary:
            model = fit_boosted(x, y[rows], t.race_ids[rows], t.race_dates[rows],
                                objective=lambda _lo, _hi: "cross_entropy", sample_weight=weights,
                                capacity="screen", n_jobs=THREADS)
        else:
            objective = CentredRegression(t.race_ids[rows], y[rows], weights)
            model = fit_boosted(x, y[rows], t.race_ids[rows], t.race_dates[rows], objective=objective.objective,
                                metric=objective.metric, capacity="screen", n_jobs=THREADS)
        score = raw_scores(model, s.take(test))
        # "model" keeps run_arm's bookkeeping valid; "score" is what the stack reads.
        return pd.DataFrame({"model": race_softmax(score, race_bounds(t.race_ids[test])), "score": score})
    return predict


def run(s: Screen, arm: str, predict, quarters=QUARTERS) -> None:
    mi.STAGES["dev"]["quarters"] = quarters
    started = time.monotonic()
    mi.run_arm(s.cache, "dev", arm, predict)
    logger.info("arm %s finished in %.0fs", arm, time.monotonic() - started)


def views() -> dict[str, list[str]]:
    remaining = read_names(FEATURES)
    out = {}
    for view, pattern in VIEWS.items():
        out[view] = [n for n in remaining if re.search(pattern, n)]
        remaining = [n for n in remaining if n not in out[view]]
    if remaining:
        raise SystemExit(f"Inputs without a view: {remaining}")
    return out


# ---------------------------------------------------------- derived inputs --

def within(frame: pd.DataFrame, column: str) -> pd.core.groupby.SeriesGroupBy:
    return frame.groupby("race_id", sort=False)[column]


def rating_inputs(s: Screen) -> np.ndarray:
    """The Timeform rating in race-relative form, and marks read at the weights.

    The card rating is already at the weights: within a handicap it correlates
    0.12 with weight carried, against 0.95 for the official mark.
    """
    frame = s.labels[["race_id", "official_rating", "weight_lbs", "rtv_card_tf_rating"]].copy()
    tf = frame.rtv_card_tf_rating
    frame["mark"] = frame.official_rating.where(frame.official_rating > 0)
    frame["at_weights"] = frame.mark - frame.weight_lbs
    frame["tf_edge"] = tf - frame.mark
    out = pd.DataFrame({
        "tf_vs_field": tf - within(frame, "rtv_card_tf_rating").transform("mean"),
        "tf_vs_best": tf - within(frame, "rtv_card_tf_rating").transform("max"),
        "tf_rank_pct": within(frame, "rtv_card_tf_rating").rank(pct=True),
        "tf_std": within(frame, "rtv_card_tf_rating").transform("std"),
        "mark_at_weights_vs_field": frame.at_weights - within(frame, "at_weights").transform("mean"),
        "mark_at_weights_vs_best": frame.at_weights - within(frame, "at_weights").transform("max"),
        "tf_over_mark_vs_field": frame.tf_edge - within(frame, "tf_edge").transform("mean"),
        "tf_over_mark_vs_best": frame.tf_edge - within(frame, "tf_edge").transform("max"),
    })
    return out.to_numpy(np.float32)


#: Strong base signals the store holds only in raw form. A tree cannot centre a
#: column within a race, so raw values mix runner quality with race quality.
RELATIVE_GAP = ["rtv_text_logit", "horse_glicko_lcb", "horse_ts_lcb", "horse_elo_surface", "horse_elo_dist_cat",
                "horse_elo_rt", "horse_avg_pos_20", "rtv_time_fig_mean3", "rtv_time_fig_best3", "rtv_time_fig_last"]


def relative_gap_inputs(s: Screen) -> np.ndarray:
    """Field-relative and best-other forms of RELATIVE_GAP, as the store would build them."""
    frame = pd.DataFrame(s.cache.take(np.arange(len(s.t.won)), RELATIVE_GAP), columns=RELATIVE_GAP)
    frame["race_id"] = s.t.race_ids
    out = {}
    for name in RELATIVE_GAP:
        values = -frame[name] if name == "horse_avg_pos_20" else frame[name]
        out[f"{name}_vs_field"] = values - values.groupby(frame.race_id).transform("mean")
        out[f"{name}_vs_best"] = values - values.groupby(frame.race_id).transform("max")
    return pd.DataFrame(out).to_numpy(np.float32)


KINDS = ["Group/Grade", "Listed", "Nursery", "Selling", "Claiming", "Bumper", "Beginners", "Novice handicap",
         "Novice", "Maiden handicap", "Maiden", "Hunter", "Handicap", "Other conditions"]


def race_kind(name, handicap) -> str:
    """The kind of race from its title. Race titles are published before the off."""
    t = name.lower() if isinstance(name, str) else ""
    if re.search(r"\bgroup [123]\b|\bgrade [123]\b|\(group|\(grade|\bg[123]\b", t):
        return "Group/Grade"
    for word, kind in (("listed", "Listed"), ("nursery", "Nursery"), ("selling", "Selling"), ("claim", "Claiming")):
        if word in t:
            return kind
    if "bumper" in t or "national hunt flat" in t or "nh flat" in t or "inh flat" in t:
        return "Bumper"
    if "beginners" in t:
        return "Beginners"
    if "novice" in t:
        return "Novice handicap" if handicap else "Novice"
    if "maiden" in t:
        return "Maiden handicap" if handicap else "Maiden"
    if "hunter" in t:
        return "Hunter"
    return "Handicap" if handicap else "Other conditions"


def kind_table() -> pd.DataFrame:
    """Every run with its race kind, rider and sex restrictions, and age band."""
    import sqlite3

    from src.utils import normalise_horse_key

    con = sqlite3.connect(f"file:{ROOT / 'data/races.db'}?mode=ro", uri=True)
    runs = pd.read_sql_query("""select race_id, race_date, race_name, race_type, handicap, horse_name, horse_id,
        trainer, jockey, age, num_runners, finish_position from results""", con)
    runs["race_id"] = runs.race_id.astype(str)
    runs["hk"] = runs.horse_name.map(normalise_horse_key)
    runs = runs.drop_duplicates(["race_id", "hk"]).reset_index(drop=True)
    races = runs.drop_duplicates("race_id")
    kinds = pd.Series([race_kind(n, bool(h)) for n, h in zip(races.race_name, races.handicap.fillna(0))],
                      index=races.race_id)
    runs["kind"] = runs.race_id.map(kinds)
    title = runs.race_name.fillna("").str.lower()
    runs["apprentice"] = title.str.contains("apprentice|conditional").astype(float)
    runs["amateur"] = title.str.contains("amateur|gentlemen|lady riders").astype(float)
    runs["fillies_mares"] = title.str.contains("fillies|mares").astype(float)
    age = pd.to_numeric(runs.age, errors="coerce")
    runs["youngest"] = age.groupby(runs.race_id).transform("min")
    runs["oldest"] = age.groupby(runs.race_id).transform("max")
    return runs


def kind_record(runs: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Wins against a fair share in this race kind, from strictly earlier days, shrunk."""
    work = runs[[entity, "kind", "race_date"]].copy()
    work["w"] = (pd.to_numeric(runs.finish_position, errors="coerce") == 1).astype(float)
    work["x"] = 1.0 / pd.to_numeric(runs.num_runners, errors="coerce").clip(lower=2)
    work["n"] = 1.0
    daily = work.groupby([entity, "kind", "race_date"], sort=True)[["w", "x", "n"]].sum().reset_index()
    totals = daily.groupby([entity, "kind"], sort=False)[["w", "x", "n"]].cumsum()
    for c in "wxn":
        daily[f"prior_{c}"] = totals[c] - daily[c]
    merged = work[[entity, "kind", "race_date"]].merge(daily, on=[entity, "kind", "race_date"], how="left",
                                                       validate="m:1")
    return pd.DataFrame({"ratio": np.log((merged.prior_w + 2.0) / (merged.prior_x + 2.0)),
                         "support": np.log1p(merged.prior_n)})


def race_kind_inputs(s: Screen) -> tuple[np.ndarray, np.ndarray]:
    """(race description columns, connections' and horse's records in this kind of race)."""
    from src.utils import normalise_horse_key

    runs = kind_table()
    described = pd.DataFrame({f"kind_{k}": runs.kind.eq(k).astype(float) for k in KINDS})
    described[["apprentice", "amateur", "fillies_mares", "youngest", "oldest"]] = runs[
        ["apprentice", "amateur", "fillies_mares", "youngest", "oldest"]]
    records = {}
    for entity in ("trainer", "jockey", "horse_id"):
        table = kind_record(runs, entity)
        relative = table.ratio - table.ratio.groupby(runs.race_id).transform("mean")
        records[f"{entity}_kind_edge_vs_field"] = relative
        records[f"{entity}_kind_support"] = table.support
    records = pd.DataFrame(records)
    cached = pd.DataFrame({"race_id": s.t.race_ids.astype(str)})
    cached["hk"] = pd.read_parquet(mi.CACHE_DIR / "targets.parquet", columns=["horse_name"]).horse_name.map(
        normalise_horse_key)
    row = cached.merge(runs[["race_id", "hk"]].assign(run_row=np.arange(len(runs))), on=["race_id", "hk"],
                       how="left", validate="1:1").run_row
    logger.info("race kind rows matched: %d of %d", int(row.notna().sum()), len(row))
    described = described.reindex(row.to_numpy()).to_numpy(np.float32)
    records = records.reindex(row.to_numpy()).to_numpy(np.float32)
    return described, records


def parse_going(text) -> float:
    """Going on one scale, firm to heavy, moved a quarter towards any 'in places' note."""
    scale = {"hard": 1.0, "firm": 1.5, "good to firm": 2.0, "good": 3.0, "good to yielding": 3.5, "yielding": 4.0,
             "good to soft": 4.0, "yielding to soft": 4.5, "soft": 5.0, "soft to heavy": 5.5, "heavy": 6.0,
             "fast": 2.0, "standard to fast": 2.5, "standard": 3.0, "standard to slow": 3.5,
             "standard / slow": 3.5, "slow": 4.0}
    if not isinstance(text, str):
        return np.nan
    text = text.lower().strip()
    main, _, rest = text.partition("(")
    value = scale.get(main.strip(), np.nan)
    note = scale.get(rest.replace("in places)", "").strip(), np.nan)
    return value if np.isnan(note) else value + 0.25 * (note - value)


@njit(cache=False)
def _context_form(starts, ends, day, furlongs, going, code, figure, max_runs, ctx, allruns, neff):
    code_similarity = np.array([[1.0, 0.5, 0.2, 0.2], [0.5, 1.0, 0.2, 0.2],
                                [0.2, 0.2, 1.0, 0.5], [0.2, 0.2, 0.5, 1.0]])
    for h in range(len(starts)):
        a, b = starts[h], ends[h]
        for i in range(a, b):
            sum_w = sum_wf = sum_w2 = sum_r = sum_rf = 0.0
            seen = 0
            for j in range(i - 1, a - 1, -1):
                if day[j] >= day[i] or not np.isfinite(figure[j]):
                    continue
                age = day[i] - day[j]
                if age > 1095 or seen >= max_runs:
                    break
                seen += 1
                recency = np.exp(-age / 365.0)
                trip = np.log(furlongs[j] / furlongs[i]) / 0.2
                ground = 0.0
                if code[i] == code[j] and code[i] != 1 and np.isfinite(going[i]) and np.isfinite(going[j]):
                    ground = (going[j] - going[i]) / 1.0
                weight = recency * np.exp(-0.5 * (trip * trip + ground * ground)) * code_similarity[code[i], code[j]]
                sum_r += recency
                sum_rf += recency * figure[j]
                sum_w += weight
                sum_wf += weight * figure[j]
                sum_w2 += weight * weight
            if sum_r > 0:
                overall = sum_rf / sum_r
                allruns[i] = overall
                # One pseudo-run at the horse's overall level shrinks thin evidence.
                ctx[i] = (sum_wf + overall) / (sum_w + 1.0)
                neff[i] = sum_w * sum_w / sum_w2 if sum_w2 > 0 else 0.0


def run_table() -> pd.DataFrame:
    """Every run in the database in pounds, by horse and date, cached once.

    A run's figure is the field's mean pre-race Elo, in pounds, less the pounds
    the horse finished behind the winner. So it can be compared across races.
    """
    path = OUT / "runs.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import sqlite3

    import pyarrow.parquet as pq

    from src.feature_store import FeatureStore
    from src.result_margins import cumulative_beaten_lengths
    from src.thurstone import pounds_behind
    from src.utils import normalise_horse_key

    con = sqlite3.connect(f"file:{ROOT / 'data/races.db'}?mode=ro", uri=True)
    runs = pd.read_sql_query("""select race_id, race_date, off_time, track, horse_name, horse_id, race_type, surface,
        race_class, handicap, num_runners, age, weight_lbs, official_rating, odds, distance_furlongs, going,
        finish_position, lengths_behind from results""", con)
    runs["race_id"] = runs.race_id.astype(str)
    runs["hk"] = runs.horse_name.map(normalise_horse_key)
    runs = runs.drop_duplicates(["race_id", "hk"]).reset_index(drop=True)
    runs["beaten"] = cumulative_beaten_lengths(runs).to_numpy()
    runs["lb_behind"] = pounds_behind(runs.beaten, runs.distance_furlongs)

    store = pq.read_table(FeatureStore().featured_path, columns=["race_id", "horse_name", "horse_elo"]).to_pandas()
    store["race_id"] = store.race_id.astype(str)
    store["hk"] = store.horse_name.map(normalise_horse_key)
    runs = runs.merge(store[["race_id", "hk", "horse_elo"]].drop_duplicates(["race_id", "hk"]),
                      on=["race_id", "hk"], how="left", validate="1:1")
    race_elo = runs.groupby("race_id").horse_elo.transform("mean")

    # Pounds per Elo point from the first window only, before any scored quarter.
    early = pd.to_datetime(runs.race_date) < pd.Timestamp("2023-01-01")
    rel = runs.horse_elo - race_elo
    lb = runs.lb_behind - runs.groupby("race_id").lb_behind.transform("mean")
    ok = early & rel.notna() & lb.notna()
    slope = float(-(rel[ok] * lb[ok]).sum() / (rel[ok] ** 2).sum())
    logger.info("pounds per Elo point %.4f on %d runs", slope, int(ok.sum()))
    runs["figure"] = slope * race_elo - runs.lb_behind

    odds = pd.to_numeric(runs.odds, errors="coerce")
    implied = 1.0 / odds.where(odds > 1.0)
    runs["sp_logp"] = np.log(implied / implied.groupby(runs.race_id).transform("sum"))
    runs["going_scale"] = runs.going.map(parse_going)
    surface = runs.surface.fillna("").str.upper()
    code = np.where(runs.race_type.eq("Chase"), 3, np.where(runs.race_type.eq("Hurdle"), 2,
                    np.where(surface.eq("TURF"), 0, 1)))
    runs["code"] = code
    runs["day"] = pd.to_datetime(runs.race_date).astype("int64") // 86_400_000_000_000
    runs = runs.sort_values(["horse_id", "day", "off_time"]).reset_index(drop=True)
    runs.drop(columns=["going", "surface", "beaten", "lengths_behind", "off_time"]).to_parquet(path, index=False)
    return pd.read_parquet(path)


def context_form_inputs(s: Screen) -> np.ndarray:
    """Past runs in pounds, weighted by how closely they resemble today's race.

    The kernel widths are fixed in advance: 20% of trip, one step of going, a
    year of recency, and a discount across codes. Nothing is tuned on outcomes.
    Win rates in conditions (which the store has) use one bit per run; this uses
    a figure per run. Only strictly earlier days enter a runner's value.
    """
    from src.utils import normalise_horse_key

    runs = run_table()
    horse = runs.horse_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, horse[1:] != horse[:-1]])
    ends = np.r_[starts[1:], len(runs)]
    n = len(runs)
    ctx, allruns, neff = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n)
    _context_form(starts, ends, runs.day.to_numpy(np.float64), runs.distance_furlongs.to_numpy(np.float64),
                  runs.going_scale.to_numpy(np.float64), runs.code.to_numpy(np.int64),
                  runs.figure.to_numpy(np.float64), 30, ctx, allruns, neff)
    runs["fc_ctx"], runs["fc_all"], runs["fc_neff"] = ctx, allruns, neff
    keys = pd.DataFrame({"race_id": s.t.race_ids.astype(str),
                         "hk": pd.Series(s.t.race_ids).map(lambda _: None)})
    keys["hk"] = pd.read_parquet(mi.CACHE_DIR / "targets.parquet", columns=["horse_name"]).horse_name.map(
        normalise_horse_key)
    frame = keys.merge(runs[["race_id", "hk", "fc_ctx", "fc_all", "fc_neff"]], on=["race_id", "hk"], how="left",
                       validate="1:1")
    frame["weight_lbs"] = s.labels.weight_lbs.to_numpy()
    frame["pref"] = frame.fc_ctx - frame.fc_all
    frame["adj"] = frame.fc_ctx - frame.weight_lbs
    out = pd.DataFrame({
        "fc_ctx_vs_field": frame.fc_ctx - within(frame, "fc_ctx").transform("mean"),
        "fc_ctx_vs_best": frame.fc_ctx - within(frame, "fc_ctx").transform("max"),
        "fc_ctx_rank_pct": within(frame, "fc_ctx").rank(pct=True),
        "fc_all_vs_field": frame.fc_all - within(frame, "fc_all").transform("mean"),
        "fc_pref": frame.pref,
        "fc_pref_vs_field": frame.pref - within(frame, "pref").transform("mean"),
        "fc_adj_vs_field": frame.adj - within(frame, "adj").transform("mean"),
        "fc_neff": frame.fc_neff,
    })
    logger.info("context form coverage %.3f", float(out.fc_ctx_vs_field.notna().mean()))
    return out.to_numpy(np.float32)


# ---------------------------------------------------- run-by-run transfer ---

TRANSFER_RUNS = 4


def transfer_pairs(s: Screen) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per (today's run, one earlier run), as a form reader takes them.

    Each row asks what one past run says about today, given how the two races
    differ. The target is today's finish in pounds against this field. Every
    pair uses strictly earlier days, so a horse's own later runs never enter.
    """
    from src.utils import normalise_horse_key

    runs = run_table()
    cached = pd.DataFrame({"race_id": s.t.race_ids.astype(str)})
    cached["hk"] = pd.read_parquet(mi.CACHE_DIR / "targets.parquet", columns=["horse_name"]).horse_name.map(
        normalise_horse_key)
    cached["row"] = np.arange(len(cached))
    row = runs[["race_id", "hk"]].merge(cached, on=["race_id", "hk"], how="left", validate="1:1").row.to_numpy()

    n = len(runs)
    horse = pd.factorize(runs.horse_id)[0]
    race = runs.race_id.to_numpy()
    day = runs.day.to_numpy(np.float64)
    a = {c: pd.to_numeric(runs[c], errors="coerce").to_numpy(np.float64) for c in
         ("distance_furlongs", "going_scale", "code", "num_runners", "age", "weight_lbs", "figure", "lb_behind",
          "finish_position", "sp_logp", "race_class")}
    mark = pd.to_numeric(runs.official_rating, errors="coerce")
    mark = mark.where(mark > 0).to_numpy(np.float64)
    hcap = pd.to_numeric(runs.handicap, errors="coerce").fillna(0).to_numpy(np.float64)
    track = pd.factorize(runs.track)[0]
    # The horse's own form level before each run: an average of earlier figures only.
    figures = pd.Series(a["figure"])
    own = figures.groupby(horse).transform(lambda v: v.shift(1).ewm(halflife=4, ignore_na=True).mean())
    field = own.groupby(race).transform("mean").to_numpy()
    own = own.to_numpy()
    lb_mean = pd.Series(a["lb_behind"]).groupby(race).transform("mean").to_numpy()
    target = lb_mean - a["lb_behind"]

    frames, today, order = [], [], []
    for k in range(1, TRANSFER_RUNS + 1):
        i = np.arange(k, n)
        j = i - k
        keep = (horse[i] == horse[j]) & (day[j] < day[i]) & (day[i] - day[j] <= 730) & np.isfinite(row[i])
        i, j = i[keep], j[keep]
        finished = a["finish_position"][j] >= 1
        frames.append(pd.DataFrame({
            "run_back": k, "days_ago": day[i] - day[j],
            "past_lb": a["lb_behind"][j], "past_won": a["finish_position"][j] == 1, "past_completed": finished,
            "past_share": np.where(finished, (a["finish_position"][j] - 1) / np.maximum(a["num_runners"][j] - 1, 1), 1.0),
            "past_field": a["num_runners"][j], "past_class": a["race_class"][j], "past_handicap": hcap[j],
            "past_code": a["code"][j], "past_going": a["going_scale"][j], "past_furlongs": a["distance_furlongs"][j],
            "past_market": a["sp_logp"][j],
            "form_vs_today_field": a["figure"][j] - field[i], "form_vs_own": a["figure"][j] - own[i],
            "own_vs_today_field": own[i] - field[i],
            "trip_change": np.log(a["distance_furlongs"][i] / a["distance_furlongs"][j]),
            "going_change": a["going_scale"][i] - a["going_scale"][j],
            "same_code": a["code"][i] == a["code"][j], "same_track": track[i] == track[j],
            "class_change": a["race_class"][i] - a["race_class"][j],
            "weight_change": a["weight_lbs"][i] - a["weight_lbs"][j], "mark_change": mark[i] - mark[j],
            "today_furlongs": a["distance_furlongs"][i], "today_going": a["going_scale"][i],
            "today_code": a["code"][i], "today_class": a["race_class"][i], "today_handicap": hcap[i],
            "today_field": a["num_runners"][i], "today_age": a["age"][i],
        }).astype(np.float32))
        today.append(row[i].astype(np.int64))
        order.append(target[i].astype(np.float32))
    return pd.concat(frames, ignore_index=True), np.concatenate(today), np.concatenate(order)


def transfer_predict(s: Screen, pairs: pd.DataFrame, today: np.ndarray, target: np.ndarray):
    """Fit the run-by-run model on earlier races; a runner's score is the mean over its past runs."""
    from lightgbm import LGBMRegressor

    t = s.t
    x = pairs.to_numpy(np.float32)
    per_runner = np.bincount(today, minlength=len(t.won)).astype(np.float64)

    def predict(train, test, _capacity):
        in_train = np.zeros(len(t.won), dtype=bool)
        in_train[train] = True
        in_test = np.zeros(len(t.won), dtype=bool)
        in_test[test] = True
        fit = np.flatnonzero(in_train[today] & np.isfinite(target))
        recency = pd.Series(fs.recency_weights(t.race_dates[train]), index=train)
        weights = recency.reindex(today[fit]).to_numpy() / per_runner[today[fit]]
        model = LGBMRegressor(objective="regression", **learner_params({"n_estimators": 400, "learning_rate": 0.04}),
                              subsample_freq=1, random_state=42, n_jobs=THREADS, verbose=-1)
        model.fit(x[fit], target[fit], sample_weight=weights)
        scored = np.flatnonzero(in_test[today])
        mean = pd.Series(model.predict(x[scored])).groupby(today[scored]).mean()
        score = mean.reindex(test).to_numpy()
        ids = pd.Series(t.race_ids[test])
        filled = pd.Series(score).fillna(pd.Series(score).groupby(ids).transform("mean")).fillna(0.0)
        return pd.DataFrame({"model": race_softmax(filled.to_numpy(), race_bounds(t.race_ids[test])), "score": score})
    return predict


# --------------------------------------------------------------- targets ---

def target_specs(s: Screen) -> dict:
    lab = s.labels
    ids = lab.race_id
    finish = lab.finish_position.to_numpy(np.float64)

    def log_prob(price):
        """Log chance from a complete book, normalised within the race; NaN if any price is missing."""
        p = 1.0 / pd.Series(np.where(price > 1.0, price, np.nan), index=lab.index)
        complete = p.notna().groupby(ids).transform("all")
        return np.log((p / p.groupby(ids).transform("sum")).where(complete)).to_numpy()

    market = np.log(np.clip(s.t.market, 1e-6, None))
    morning = log_prob(lab.ltp_1100.to_numpy(np.float64))
    mark = lab.official_rating.where(lab.official_rating > 0)
    next_mark = lab.or_next.where(lab.or_next > 0)
    return {
        "figure": {"y": -lab.lb_behind.to_numpy(np.float64)},
        "completed": {"y": (finish >= 1).astype(np.float64), "binary": True},
        "placed": {"y": ((finish >= 1) & (finish <= 3)).astype(np.float64), "binary": True},
        "early_position": {"y": lab.early_position.to_numpy(np.float64)},
        "late_gain": {"y": lab.late_gain.to_numpy(np.float64)},
        "finish_speed": {"y": lab.fsp_relative.to_numpy(np.float64)},
        "time_figure": {"y": lab.time_fig.to_numpy(np.float64)},
        "market_close": {"y": market},
        "market_morning": {"y": morning},
        "market_move": {"y": market - morning},
        "handicapper": {"y": (next_mark - mark).to_numpy(np.float64),
                        "available": pd.to_datetime(lab.next_date).to_numpy("datetime64[ns]")},
    }


# ----------------------------------------------------------------- stack ---

def clogit(x, bounds, won, offset, l2) -> np.ndarray:
    """Conditional logit, score = offset + x @ b, with a ridge on b."""
    sizes = np.diff(bounds)
    starts = bounds[:-1]

    def loss(b):
        s = offset + x @ b
        top = np.repeat(np.maximum.reduceat(s, starts), sizes)
        e = np.exp(s - top)
        total = np.repeat(np.add.reduceat(e, starts), sizes)
        p = e / total
        nll = -(won * np.log(np.clip(p, 1e-300, None))).sum() / len(starts) + l2 * (b @ b)
        grad = x.T @ (p * np.repeat(np.add.reduceat(won.astype(np.float64), starts), sizes) - won) / len(starts)
        return nll, grad + 2 * l2 * b

    return minimize(loss, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 200}).x


def walk_forward(frame: pd.DataFrame, columns: list[str], offset: str | None, l2: float = 1e-5,
                 groups: str | None = None) -> pd.Series:
    """Per-race loss of a stack fitted on earlier quarters and scored on the next one.

    *groups* lets every coefficient differ by a race-context code, on top of a
    pooled coefficient: this is the context-gated meta learner.
    """
    losses = []
    x = frame[columns].to_numpy(np.float64)
    if groups is not None:
        codes = frame[groups].to_numpy()
        blocks = [x] + [x * (codes == c)[:, None] for c in np.unique(codes)]
        x = np.hstack(blocks)
    base = frame[offset].to_numpy(np.float64) if offset else np.zeros(len(frame))
    won = frame.won.to_numpy(np.float64)
    fold = frame.fold.to_numpy()
    for i, quarter in enumerate(QUARTERS[1:], start=1):
        fit = np.isin(fold, QUARTERS[:i])
        test = fold == quarter
        b = clogit(x[fit], race_bounds(frame.race_id.to_numpy()[fit]), won[fit], base[fit], l2)
        scored = frame.loc[test, ["race_id", "won"]].copy()
        scored["p"] = race_softmax(base[test] + x[test] @ b, race_bounds(scored.race_id.to_numpy()))
        losses.append(mi.race_losses(scored, "p"))
    return pd.concat(losses)


def meta_gbm(frame: pd.DataFrame, columns: list[str], l2: float = 1e-5, seed: int = 63) -> pd.Series:
    """A small boosted race softmax on the second-stage inputs, offset by the linear stack.

    It starts from the conditional logit on *columns*' first entry and may then use
    every column, so it can learn context-dependent weights the linear stack cannot.
    Fitted on earlier quarters and scored on the next, like :func:`walk_forward`.
    """
    from lightgbm import LGBMRegressor

    losses = []
    fold = frame.fold.to_numpy()
    won = frame.won.to_numpy(np.float64)
    race = frame.race_id.to_numpy()
    first = frame[columns[:1]].to_numpy(np.float64)
    x = frame[columns].to_numpy(np.float64)
    for i, quarter in enumerate(QUARTERS[1:], start=1):
        fit = np.isin(fold, QUARTERS[:i])
        test = fold == quarter
        bounds = race_bounds(race[fit])
        b = clogit(first[fit], bounds, won[fit], np.zeros(int(fit.sum())), l2)
        offset = first[fit] @ b
        target = won[fit]

        def objective(_y, pred):
            p = race_softmax(offset + pred, bounds)
            return p - target, np.maximum(p * (1 - p), 1e-6)

        model = LGBMRegressor(objective=objective, n_estimators=150, learning_rate=0.02, max_depth=3, num_leaves=7,
                              min_child_samples=300, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=20.0, random_state=seed, n_jobs=2, verbose=-1)
        model.fit(x[fit], won[fit])
        scored = frame.loc[test, ["race_id", "won"]].copy()
        scored["p"] = race_softmax(first[test] @ b + raw_scores(model, x[test]), race_bounds(race[test]))
        losses.append(mi.race_losses(scored, "p"))
    return pd.concat(losses)


def paired(delta: pd.Series, dates: pd.Series, fold: pd.Series) -> dict:
    per_quarter = delta.groupby(fold.reindex(delta.index)).mean()
    lo, hi = paired_interval(delta.to_numpy(), blocks=dates.reindex(delta.index).astype(str).to_numpy())["ci95"]
    return {"delta": round(float(delta.mean()), 5), "ci95": [round(float(lo), 5), round(float(hi), 5)],
            "better": f"{int((per_quarter < 0).sum())}/{len(per_quarter)}"}


def load_saved(arm: str, quarters=QUARTERS, directory: Path | None = None) -> pd.DataFrame:
    directory = directory or OUT / "dev"
    return pd.concat([pd.read_parquet(directory / f"{arm}_{q}.parquet").assign(fold=q) for q in quarters],
                     ignore_index=True)


def centred_z(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Race-centred and scaled to unit spread; a missing score sits at the race mean."""
    v = pd.Series(frame[column].to_numpy(np.float64))
    v = (v - v.groupby(frame.race_id.to_numpy()).transform("mean")).fillna(0.0)
    return (v / v.std()).to_numpy()


def direct_report(arms: list[str], quarters: list[str], reference: str = BASELINE_ARM) -> dict:
    """Paired race log loss of arms that replace the baseline on the same races."""
    base = load_saved(reference, quarters, BASELINE_DIR)
    races = base.drop_duplicates("race_id").set_index("race_id")
    base_loss = mi.race_losses(base, "model")
    market = mi.race_losses(base, "market")
    out = {"races": len(races), "baseline_ll": round(float(base_loss.mean()), 5),
           "market_ll": round(float(market.mean()), 5), "arms": {}}
    for arm in arms:
        loss = mi.race_losses(load_saved(arm, quarters), "model")
        row = paired(loss - base_loss, races.race_date, races.fold)
        row["race_ll"] = round(float(loss.mean()), 5)
        out["arms"][arm] = row
    return out


def stack(_args) -> None:
    base = load_saved(BASELINE_ARM, directory=BASELINE_DIR)[["row", "race_id", "race_date", "won", "market", "model", "fold"]]
    base["logp"] = np.log(np.clip(base.model, 1e-9, None))
    report: dict = {"baseline_arm": BASELINE_ARM}

    report["direct"] = {
        "depth": direct_report(["depth1", "depth2", "depth3", "depth4", "depth8"], LAST4),
        "inputs": direct_report(["noise8", "context_shuffled", "context", "context_form", "ratings", "ratings_tf",
                                 "ratings_marks", "ratings_over_mark", "relative_gap", "kinds", "kinds_records", "kinds_shuffled",
                                 "combined"], LAST4),
        "teacher": direct_report(["teacher_0.25", "teacher_0.75", "teacher_1", "combined_teacher_0.75"], LAST4),
    }

    frame = base.copy()
    rng = np.random.default_rng(63)
    frame["noise"] = rng.normal(size=len(frame))
    frame["noise"] = centred_z(frame, "noise")
    for arm in ("seed7", "seed8"):
        saved = load_saved(arm)
        frame[arm] = np.log(np.clip(saved.model.to_numpy(), 1e-9, None))
    for view in views():
        saved = load_saved(f"view_{view}")
        frame[f"view_{view}"] = np.log(np.clip(saved.model.to_numpy(), 1e-9, None))
    candidates = [*target_specs(Screen()), "transfer"]
    targets = [t for t in candidates if (OUT / "dev" / f"target_{t}_{QUARTERS[-1]}.parquet").exists()]
    for target in targets:
        frame[f"z_{target}"] = centred_z(frame.assign(s=load_saved(f"target_{target}").score.to_numpy()), "s")

    def reference():
        return walk_forward(frame, ["logp"], None)

    temperature = reference()
    races = frame.drop_duplicates("race_id").set_index("race_id")
    scored = races.loc[temperature.index]
    report["stack_races"] = len(scored)
    report["baseline_ll_with_temperature"] = round(float(temperature.mean()), 5)

    def score(label, columns, **kwargs):
        loss = walk_forward(frame, columns, None, **kwargs)
        report.setdefault("stack", {})[label] = paired(loss - temperature, scored.race_date, scored.fold)
        logger.info("%s %s", label, report["stack"][label])
        return loss

    score("noise control (1 column)", ["logp", "noise"])
    score("seed average, 2 fits", ["logp", "seed7"])
    score("seed average, 3 fits", ["logp", "seed7", "seed8"])
    view_columns = [c for c in frame if c.startswith("view_")]
    loss = walk_forward(frame, view_columns, None)
    report["views_alone"] = {
        "race_ll": round(float(loss.mean()), 5),
        "vs_full_model": paired(loss - temperature, scored.race_date, scored.fold),
        "each": {c: round(float(mi.race_losses(frame.rename(columns={"model": "_m"}).assign(
            model=np.exp(frame[c])), "model").loc[scored.index].mean()), 5) for c in view_columns},
    }
    score("full model + views", ["logp", *view_columns])
    for target in targets:
        score(f"+ {target}", ["logp", f"z_{target}"])
    z_columns = [f"z_{t}" for t in targets]
    if z_columns:
        score("+ all targets", ["logp", *z_columns], l2=1e-4)
        score("+ all targets + views + seeds", ["logp", "seed7", "seed8", *view_columns, *z_columns], l2=1e-4)

    # The second models that survived on their own, alone and with plain averaging.
    useful = ["z_market_close", "z_figure"]
    score("+ market head + figure", ["logp", *useful])
    score("+ market head + figure + seeds", ["logp", "seed7", "seed8", *useful])
    score("+ market head + figure + seeds + views", ["logp", "seed7", "seed8", *view_columns, *useful], l2=1e-4)

    screen = Screen()
    labels = screen.labels.iloc[frame.row.to_numpy()].reset_index(drop=True)
    frame["ctx"] = (labels.race_type.fillna("?") + labels.handicap.fillna(0).astype(bool).astype(str)).to_numpy()
    score("views, weights by race type x handicap", ["logp", *view_columns], groups="ctx", l2=1e-4)
    score("market head + figure, weights by race type x handicap", ["logp", *useful], groups="ctx", l2=1e-4)
    # The finer description of the race: do the models deserve different weights by kind?
    kinds = kind_table().drop_duplicates("race_id").set_index("race_id")
    code = frame.race_id.astype(str).map(kinds.race_type + ": " + kinds.kind)
    counts = code.value_counts()
    frame["kind_group"] = code.where(code.map(counts) >= 5000, "other").to_numpy()
    score("temperature by race kind", ["logp"], groups="kind_group", l2=1e-4)
    score("views, weights by race kind", ["logp", *view_columns], groups="kind_group", l2=1e-4)
    score("market head + figure, weights by race kind", ["logp", *useful], groups="kind_group", l2=1e-4)

    # A boosted meta learner, with the race context it needs to vary the weights.
    frame["c_field"] = labels.num_runners.to_numpy(np.float64)
    frame["c_furlongs"] = labels.distance_furlongs.to_numpy(np.float64)
    frame["c_handicap"] = labels.handicap.fillna(0).astype(float).to_numpy()
    frame["c_flat"] = labels.race_type.eq("Flat").astype(float).to_numpy()
    frame["c_chase"] = labels.race_type.eq("Chase").astype(float).to_numpy()
    frame["c_class"] = pd.to_numeric(labels.race_class, errors="coerce").to_numpy(np.float64)
    frame["c_runs"] = np.asarray(screen.cache.take(frame.row.to_numpy(), ["horse_prev_races"]), dtype=np.float64)[:, 0]
    context_columns = [c for c in frame if c.startswith("c_")]
    for label, columns in (("boosted meta: market head + figure + context", ["logp", *useful, *context_columns]),
                           ("boosted meta: + seeds + views + context",
                            ["logp", "seed7", "seed8", *view_columns, *useful, *context_columns])):
        loss = meta_gbm(frame, columns)
        report.setdefault("stack", {})[label] = paired(loss - temperature, scored.race_date, scored.fold)
        logger.info("%s %s", label, report["stack"][label])

    mi.save_json(OUT / "summary_063.json", report)
    print(json.dumps(report, indent=2, default=float))


def additive(_args) -> None:
    """Do the input, target and sub-model gains add up? Same races, walk-forward over LAST4."""
    global QUARTERS
    QUARTERS = LAST4
    frame = load_saved(BASELINE_ARM, LAST4, BASELINE_DIR)[["row", "race_id", "race_date", "won", "market", "model", "fold"]]
    frame["logp"] = np.log(np.clip(frame.model, 1e-9, None))
    frame["logm"] = np.log(np.clip(frame.market, 1e-9, None))
    arms = ["teacher_0.75", "ratings", "combined", "combined_teacher_0.75"]
    for arm in arms:
        frame[f"p_{arm}"] = np.log(np.clip(load_saved(arm, LAST4).model.to_numpy(), 1e-9, None))
    for target in ("figure", "market_close"):
        frame[f"z_{target}"] = centred_z(frame.assign(s=load_saved(f"target_{target}", LAST4).score.to_numpy()), "s")
    for seed in ("seed7", "seed8"):
        frame[seed] = np.log(np.clip(load_saved(seed, LAST4).model.to_numpy(), 1e-9, None))

    reference = walk_forward(frame, ["logp"], None)
    market = walk_forward(frame, ["logm"], None)
    races = frame.drop_duplicates("race_id").set_index("race_id").loc[reference.index]
    report: dict = {"races": len(races), "baseline_ll": round(float(reference.mean()), 5),
                    "market_ll": round(float(market.mean()), 5), "arms": {}, "market": {}}
    heads = ["z_market_close", "z_figure"]

    def score(label, columns, l2=1e-5):
        loss = walk_forward(frame, columns, None, l2=l2)
        report["arms"][label] = {"race_ll": round(float(loss.mean()), 5),
                                 **paired(loss - reference, races.race_date, races.fold)}
        logger.info("%s %s", label, report["arms"][label])

    score("baseline + market head", ["logp", "z_market_close"])
    score("baseline + figure", ["logp", "z_figure"])
    score("baseline + both heads", ["logp", *heads])
    for arm in arms:
        column = f"p_{arm}"
        score(arm, [column])
        score(f"{arm} + market head", [column, "z_market_close"])
        score(f"{arm} + figure", [column, "z_figure"])
        score(f"{arm} + both heads", [column, *heads])
    centred = frame.logm - frame.groupby("race_id").logm.transform("mean")
    for label, column in (("baseline", "logp"), *((arm, f"p_{arm}") for arm in arms)):
        values = frame[column] - frame.groupby("race_id")[column].transform("mean")
        blend = walk_forward(frame, ["logm", column], None)
        report["market"][label] = {"corr_with_market": round(float(np.corrcoef(values, centred)[0, 1]), 4),
                                   "blend_vs_market": paired(blend - market, races.race_date, races.fold)}
    mi.save_json(OUT / "additive_063.json", report)
    print(json.dumps(report, indent=2, default=float))


# ---------------------------------------------------------- confirmation ---

def combined_block(s: Screen) -> np.ndarray:
    """The 64 extra inputs: ratings, field gaps, weighted form and race context."""
    moderators = s.cache.take(np.arange(len(s.t.won)), MODERATORS)
    return np.hstack([rating_inputs(s), relative_gap_inputs(s), np.load(OUT / "context_form.npy"), moderators])


def register(_args) -> None:
    """Fix the candidates and the pass rule before any confirmation fit."""
    if REGISTRATION.exists():
        raise SystemExit(f"Already registered: {REGISTRATION}")
    direct = json.loads((OUT / "summary_063.json").read_text(encoding="utf-8"))["direct"]
    mi.save_json(REGISTRATION, {
        "experiment": "063",
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quarters": mi.STAGES["confirm"]["quarters"],
        "capacity": "native",
        "seed": 42,
        "reference": "057 'exploded': 129 inputs, market share 0.5, saved in model_ideas_057/confirm",
        "primary": {
            "ratings_t75": "129 inputs + 8 rating-gap columns, market share 0.75",
            "combined_t75": "129 inputs + 64 extra columns, market share 0.75",
        },
        "pass_rule": "race LL below the reference, 95% date-clustered interval entirely below zero, "
                     "and better in at least 3 of 4 quarters",
        "choice_rule": "if both pass, keep ratings_t75 unless combined_t75 beats it with its own "
                       "interval entirely below zero",
        "secondary": "not part of the pass rule, run only if asked: market share 0.75 alone; 8 columns alone",
        "not_read": "golden hold-out from 2026-06-11",
        "development_results": {
            "combined_t75": direct["teacher"]["arms"]["combined_teacher_0.75"],
            "combined": direct["inputs"]["arms"]["combined"],
            "ratings": direct["inputs"]["arms"]["ratings"],
            "teacher_0.75": direct["teacher"]["arms"]["teacher_0.75"],
        },
    })
    print(f"Registered -> {REGISTRATION}")


def confirm(args) -> None:
    """Reproduce one saved 057 fold, then fit both candidates at native capacity."""
    if not REGISTRATION.exists():
        raise SystemExit("Register first: python -m scripts.feature_structure_063 register")
    s = Screen()
    quarters = mi.STAGES["confirm"]["quarters"]
    mi.STAGES["confirm"]["quarters"] = quarters[:1]
    mi.run_arm(s.cache, "confirm", "check_native", exploded_predict(s))
    mi.STAGES["confirm"]["quarters"] = quarters
    mine = pd.read_parquet(OUT / "confirm" / f"check_native_{quarters[0]}.parquet")
    theirs = pd.read_parquet(CONFIRM_BASE_DIR / f"exploded_{quarters[0]}.parquet")
    gap = float(np.abs(mine.model.to_numpy() - theirs.model.to_numpy()).max())
    logger.info("native harness check: largest difference from saved 057 fold %.3g", gap)
    if gap > 1e-6:
        raise SystemExit(f"The harness does not reproduce 057's native fold (largest difference {gap:.3g})")
    mi.run_arm(s.cache, "confirm", "ratings_t75", exploded_predict(s, extra=rating_inputs(s), teacher=0.75))
    mi.run_arm(s.cache, "confirm", "combined_t75", exploded_predict(s, extra=combined_block(s), teacher=0.75))
    confirm_report(args)


def confirm_report(_args) -> None:
    """Score the candidates against saved 057 by the registered rule."""
    global QUARTERS
    registered = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    quarters = registered["quarters"]
    QUARTERS = quarters
    base = load_saved("exploded", quarters, CONFIRM_BASE_DIR)
    races = base.drop_duplicates("race_id").set_index("race_id")
    base_loss = mi.race_losses(base, "model")
    market = mi.race_losses(base, "market")
    per_quarter = lambda loss: {q: round(float(v), 5) for q, v in loss.groupby(races.fold.reindex(loss.index)).mean().items()}  # noqa: E501
    report: dict = {"races": len(races), "reference_ll": round(float(base_loss.mean()), 5),
                    "reference_per_quarter": per_quarter(base_loss),
                    "market_ll": round(float(market.mean()), 5), "arms": {}}
    frame = base[["race_id", "won", "fold", "market", "model"]].copy()
    frame["logm"] = np.log(np.clip(frame.market, 1e-9, None))
    centred_market = frame.logm - frame.groupby("race_id").logm.transform("mean")

    def closeness(values: pd.Series) -> dict:
        return {"corr_with_market": round(float(np.corrcoef(values - values.groupby(frame.race_id).transform("mean"),
                                                            centred_market)[0, 1]), 4)}

    frame["logp"] = np.log(np.clip(frame.model, 1e-9, None))
    report["reference_market"] = closeness(frame.logp)
    losses = {}
    for arm in registered["primary"]:
        saved = load_saved(arm, quarters, OUT / "confirm")
        if not np.array_equal(saved.row.to_numpy(), base.row.to_numpy()):
            raise SystemExit(f"{arm} was not scored on the same runners as 057")
        losses[arm] = mi.race_losses(saved, "model")
        row = paired(losses[arm] - base_loss, races.race_date, races.fold)
        row.update(race_ll=round(float(losses[arm].mean()), 5), per_quarter=per_quarter(losses[arm]),
                   gap_to_market=round(float((losses[arm] - market).mean()), 5))
        row["passed"] = row["ci95"][1] < 0 and row["delta"] < 0 and int(row["better"].split("/")[0]) >= 3
        frame[arm] = np.log(np.clip(saved.model.to_numpy(), 1e-9, None))
        row.update(closeness(frame[arm]))
        blend = walk_forward(frame, ["logm", arm], None)
        alone = walk_forward(frame, ["logm"], None)
        row["blend_vs_market"] = paired(blend - alone, races.race_date, races.fold)
        report["arms"][arm] = row
    first, second = registered["primary"]
    report["combined_vs_ratings"] = paired(losses[second] - losses[first], races.race_date, races.fold)
    passed = [arm for arm in registered["primary"] if report["arms"][arm]["passed"]]
    if len(passed) == 2:
        passed = [second] if report["combined_vs_ratings"]["ci95"][1] < 0 else [first]
    report["chosen"] = passed[0] if passed else None
    mi.save_json(OUT / "summary_063_confirm.json", report)
    print(json.dumps(report, indent=2, default=float))


# ------------------------------------------------------------------ main ---

def run_groups(args) -> None:
    s = Screen()
    for group in args.groups:
        if group == "check":
            # Reproduces the saved baseline fold exactly when the harness matches 057.
            run(s, "check_seed42", exploded_predict(s), QUARTERS[-1:])
            mine = load_saved("check_seed42", QUARTERS[-1:])
            theirs = load_saved(BASELINE_ARM, QUARTERS[-1:], BASELINE_DIR)
            print("max abs difference from the saved fold:", float(np.abs(mine.model - theirs.model).max()))
        elif group == "depth":
            for depth, trees in ((1, 1600), (2, 1000), (3, 700), (4, 500)):
                params = {"max_depth": depth, "num_leaves": 2 ** depth, "n_estimators": trees, "learning_rate": 0.04}
                run(s, f"depth{depth}", exploded_predict(s, params=params), LAST4)
        elif group == "seeds":
            for seed in (7, 8):
                run(s, f"seed{seed}", exploded_predict(s, seed=seed))
        elif group == "context":
            names = s.names + [m for m in MODERATORS if m not in s.names]
            missing = [m for m in names if m not in s.cache.index]
            if missing:
                raise SystemExit(f"Not in the cache: {missing}")
            run(s, "context", exploded_predict(s, names=names), LAST4)
            # The size-matched control: real context columns, attached to the wrong races.
            block = s.cache.take(np.arange(len(s.t.won)), MODERATORS)
            race_index = np.unique(s.t.race_ids, return_inverse=True)[1]
            first_row = pd.Series(np.arange(len(race_index))).groupby(race_index).first().to_numpy()
            order = np.random.default_rng(63).permutation(len(first_row))
            run(s, "context_shuffled", exploded_predict(s, extra=block[first_row[order[race_index]]]), LAST4)
        elif group == "ratings":
            run(s, "ratings", exploded_predict(s, extra=rating_inputs(s)), LAST4)
            noise = np.random.default_rng(64).normal(size=(len(s.t.won), 8)).astype(np.float32)
            run(s, "noise8", exploded_predict(s, extra=noise), LAST4)
        elif group == "teacher":
            for share in (0.25, 0.75, 1.0):
                run(s, f"teacher_{share:g}", exploded_predict(s, teacher=share), LAST4)
        elif group == "race_kinds":
            described, records = race_kind_inputs(s)
            logger.info("kind columns %s, record columns %s", described.shape, records.shape)
            run(s, "kinds", exploded_predict(s, extra=described), LAST4)
            run(s, "kinds_records", exploded_predict(s, extra=np.hstack([described, records])), LAST4)
            # The size-matched control: the same description attached to the wrong races.
            race_index = np.unique(s.t.race_ids, return_inverse=True)[1]
            first_row = pd.Series(np.arange(len(race_index))).groupby(race_index).first().to_numpy()
            order = np.random.default_rng(65).permutation(len(first_row))
            run(s, "kinds_shuffled", exploded_predict(s, extra=described[first_row[order[race_index]]]), LAST4)
        elif group == "ratings_parts":
            block = rating_inputs(s)
            run(s, "ratings_tf", exploded_predict(s, extra=block[:, :4]), LAST4)
            run(s, "ratings_marks", exploded_predict(s, extra=block[:, 4:6]), LAST4)
            run(s, "ratings_over_mark", exploded_predict(s, extra=block[:, 6:8]), LAST4)
        elif group == "relative_gap":
            run(s, "relative_gap", exploded_predict(s, extra=relative_gap_inputs(s)), LAST4)
        elif group == "depth8":
            params = {"max_depth": 8, "num_leaves": 63, "n_estimators": 400, "learning_rate": 0.04}
            run(s, "depth8", exploded_predict(s, params=params), LAST4)
        elif group == "combined":
            run(s, "combined", exploded_predict(s, extra=combined_block(s)), LAST4)
        elif group == "combined_teacher":
            run(s, "combined_teacher_0.75", exploded_predict(s, extra=combined_block(s), teacher=0.75), LAST4)
        elif group == "form":
            path = OUT / "context_form.npy"
            if not path.exists():
                np.save(path, context_form_inputs(s))
            run(s, "context_form", exploded_predict(s, extra=np.load(path)), LAST4)
        elif group == "views":
            for view, names in views().items():
                logger.info("view %s: %d inputs", view, len(names))
                run(s, f"view_{view}", exploded_predict(s, names=names))
        elif group == "transfer":
            pairs, today, target = transfer_pairs(s)
            logger.info("%d run pairs for %d runners", len(pairs), len(np.unique(today)))
            run(s, "target_transfer", transfer_predict(s, pairs, today, target))
        elif group == "targets":
            for name, spec in target_specs(s).items():
                run(s, f"target_{name}", target_predict(s, spec["y"], binary=spec.get("binary", False),
                                                        available=spec.get("available")))
        else:
            raise SystemExit(f"Unknown group {group}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("groups", nargs="+")
    runner.set_defaults(func=run_groups)
    sub.add_parser("stack").set_defaults(func=stack)
    sub.add_parser("additive").set_defaults(func=additive)
    sub.add_parser("register").set_defaults(func=register)
    sub.add_parser("confirm").set_defaults(func=confirm)
    sub.add_parser("confirm_report").set_defaults(func=confirm_report)
    args = parser.parse_args()
    mi.setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
