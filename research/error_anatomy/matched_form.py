"""Form matched to today's conditions, on continuous scales.

The store presents suitability as win rates per category: the horse's record
on this going bucket, at this distance band, at this course. Each is a ratio
over a handful of runs, and going is one of seven buckets although the results
carry 193 descriptions ("Good to Firm (Good in places)" and "Firm" are both
"Fast"). The selection kept 2 of 53 going inputs and 3 of 58 distance inputs.

``matched-form``, declared before scoring: one weighted average of the horse's
past performances in which each earlier run counts by how closely its
conditions match today's. Up to twelve runs on strictly earlier dates, recent
runs weighted more, then multiplied by a similarity:

- distance: exp(-|log ratio of distances| / 0.15);
- going: exp(-|difference on a continuous going scale|), 0.3 across turf and
  all-weather;
- course: 1 at the same track, 0.25 elsewhere;
- class: exp(-|class difference| / 1.5);
- everything: distance x going x same code (flat, hurdle, chase).

Two performance measures per run: the share of rivals beaten, and that share
minus what the starting prices implied. For each of the six weightings (none,
distance, going, course, class, everything): the weighted mean of each
measure and the total weight; each mean against the field; and the
everything-matched mean minus the unmatched one (suited by today's conditions).
"""
import glob
import os
import re

import numba as nb
import numpy as np
import pandas as pd

from research.meta_learner import screening as s

KEYS = s.KEYS
STORE = "data/processed/feature_store/featured.parquet"
SCALE = {"hard": 0., "firm": 1., "good to firm": 2., "good": 3., "good to yielding": 3.5, "good to soft": 4.,
         "yielding": 4.5, "yielding to soft": 5., "soft": 5.5, "soft to heavy": 6.25, "heavy": 7.,
         "fast": 2., "standard to fast": 2.5, "standard": 3., "standard to slow": 3.5, "standard / slow": 3.5,
         "slow": 4.}
KERNELS = ("any", "dist", "going", "course", "class", "all")
RECENCY = .85


def going_index(text):
    """Continuous going: 'X (Y in places)' is three parts X to one part Y. NaN when unreadable."""
    if not isinstance(text, str):
        return np.nan
    text = text.lower().strip()
    found = re.match(r"^([a-z /]+?)\s*(?:\((.+?) in places\))?\s*$", text)
    if not found:  # an unusual bracket ("Good (Watered)"): keep the main description
        return SCALE.get(text.split("(")[0].strip(), np.nan)
    main = SCALE.get(found.group(1).strip(), np.nan)
    places = SCALE.get(found.group(2).strip(), np.nan) if found.group(2) else np.nan
    return main if np.isnan(places) else .75 * main + .25 * places


def expected_beat(p, bounds):
    """Share of rivals a runner is expected to beat, from within-race probabilities."""
    out = np.full(len(p), np.nan)
    for a, b in zip(bounds[:-1], bounds[1:]):
        q = p[a:b]
        if b - a < 2 or not np.isfinite(q).all():
            continue
        pair = q[:, None] / (q[:, None] + q[None, :])
        np.fill_diagonal(pair, 0.)
        out[a:b] = pair.sum(1) / (b - a - 1)
    return out


@nb.njit(cache=True)
def _matched(horse, day, dist, going, turf, track, klass, code, beat, rel, recency):
    n = len(horse)
    means = np.full((n, 12), np.nan)
    weights = np.zeros((n, 6))
    start = 0
    for i in range(n):
        if i == 0 or horse[i] != horse[i - 1]:
            start = i
        total = np.zeros(12)
        mass = np.zeros(12)
        w = 1.
        used = 0
        for j in range(i - 1, start - 1, -1):
            if day[j] == day[i]:
                continue
            if used == 12:
                break
            used += 1
            k = np.ones(6)
            if dist[i] > 0 and dist[j] > 0:
                k[1] = np.exp(-abs(np.log(dist[j] / dist[i])) / .15)
            if np.isnan(going[i]) or np.isnan(going[j]):
                k[2] = .5
            elif turf[i] != turf[j]:
                k[2] = .3
            else:
                k[2] = np.exp(-abs(going[j] - going[i]))
            k[3] = 1. if track[i] == track[j] else .25
            if np.isnan(klass[i]) or np.isnan(klass[j]):
                k[4] = .5
            else:
                k[4] = np.exp(-abs(klass[j] - klass[i]) / 1.5)
            k[5] = k[1] * k[2] * (1. if code[i] == code[j] else .2)
            for m in range(6):
                weight = w * k[m]
                weights[i, m] += weight
                if not np.isnan(beat[j]):
                    total[m] += weight * beat[j]
                    mass[m] += weight
                if not np.isnan(rel[j]):
                    total[6 + m] += weight * rel[j]
                    mass[6 + m] += weight
            w *= recency
        for m in range(12):
            if mass[m] > 0:
                means[i, m] = total[m] / mass[m]
    return means, weights


def matched_inputs():
    store = pd.read_parquet(STORE, columns=[*KEYS, "race_date", "finish_position", "num_runners", "track", "surface",
                                            "distance_furlongs", "race_type", "race_class", "odds"])
    store["race_id"] = store.race_id.astype(str)
    source = sorted(glob.glob("data/sources/sl/*/results.parquet"), key=os.path.getmtime)[-1]
    raw = pd.read_parquet(source, columns=["race_id", "going"]).drop_duplicates("race_id")
    raw["race_id"] = raw.race_id.astype(str)
    raw["going_index"] = raw.going.map(going_index)
    runs = store.merge(raw[["race_id", "going_index"]], on="race_id", how="left")
    runs = runs.sort_values(["race_id", "horse_id"], kind="stable").reset_index(drop=True)
    position = pd.to_numeric(runs.finish_position, errors="coerce")
    field = runs.num_runners.astype(float)
    runs["beat"] = np.where(position.ge(1), (field - position) / (field - 1).clip(lower=1), 0.).clip(0, 1)
    price = pd.to_numeric(runs.odds, errors="coerce")
    p = (1 / price.where(price > 1)).to_numpy(float)
    runs["rel"] = runs.beat - expected_beat(p, s.bounds_of(runs.race_id.to_numpy()))
    runs = runs.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index(drop=True)
    means, weights = _matched(
        runs.horse_id.to_numpy(np.int64), (runs.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64),
        runs.distance_furlongs.to_numpy(float), runs.going_index.to_numpy(float),
        runs.surface.eq("Turf").to_numpy(np.int64), pd.factorize(runs.track)[0].astype(np.int64),
        pd.to_numeric(runs.race_class, errors="coerce").to_numpy(float),
        pd.factorize(runs.race_type)[0].astype(np.int64), runs.beat.to_numpy(float), runs.rel.to_numpy(float), RECENCY)
    out = runs[KEYS].copy()
    race = runs.race_id.to_numpy()
    for m, kernel in enumerate(KERNELS):
        out[f"mf_beat_{kernel}"] = means[:, m]
        out[f"mf_rel_{kernel}"] = means[:, 6 + m]
        out[f"mf_weight_{kernel}"] = weights[:, m]
    for column in [c for c in out if c.startswith(("mf_beat_", "mf_rel_"))]:
        out[f"{column}_vs_field"] = out[column] - out[column].groupby(race).transform("mean")
    out["mf_beat_suited"] = out.mf_beat_all - out.mf_beat_any
    out["mf_rel_suited"] = out.mf_rel_all - out.mf_rel_any
    out["mf_going_index"] = runs.going_index.to_numpy()
    columns = [c for c in out if c not in KEYS]
    out[columns] = out[columns].astype("float32")
    return out


def load(frame):
    values = matched_inputs()
    values["horse_id"] = values.horse_id.astype(frame.horse_id.dtype)
    return values, [c for c in values if c not in KEYS]
