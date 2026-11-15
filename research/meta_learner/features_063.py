"""Reconstruct the 063 research inputs from the selected canonical history."""
import numpy as np
import pandas as pd
from numba import njit

from src.feature_semantics import CLAIM_ADJUSTED_WEIGHT_INPUTS, claim_adjusted_weight
from src.modelling.performance import PerformanceSpec, performance_targets


RATING_INPUTS = ("rtv_card_tf_rating", "official_rating")
GAP_INPUTS = ("rtv_text_logit", "horse_glicko_lcb", "horse_ts_lcb", "horse_elo_surface",
              "horse_elo_dist_cat", "horse_elo_rt", "horse_avg_pos_20", "rtv_time_fig_mean3",
              "rtv_time_fig_best3", "rtv_time_fig_last")
MODERATORS = (
    "race_type_Chase", "race_type_Flat", "race_type_Hurdle", "handicap", "race_is_novice",
    "race_is_maiden", "race_is_bumper", *(f"race_class_{i}.0" for i in range(1, 7)),
    "distance_furlongs", "going_Good", "going_Soft", "going_Heavy", "going_Standard",
    "surface_Turf", "surface_Allweather", "month_sin", "month_cos", "field_avg_or",
    "field_std_or", "field_top2_or_gap", "field_std_elo", "field_top2_elo_gap", "field_experience_gap",
)
HISTORY = ("race_type", "surface", "beaten_lengths", "has_official_rating", *CLAIM_ADJUSTED_WEIGHT_INPUTS)
NUMERIC_INPUTS = tuple(dict.fromkeys((*RATING_INPUTS, *GAP_INPUTS, *MODERATORS, "horse_elo")))


def rating_gaps(frame):
    race = frame.race_id
    rating = frame.rtv_card_tf_rating
    mark = frame.official_rating.where(frame.has_official_rating.eq(1) & frame.official_rating.gt(0))
    weight = claim_adjusted_weight(frame)
    at_weights, edge = mark - weight, rating - mark
    group = rating.groupby(race, sort=False)
    result = {"tf_vs_field": rating - group.transform("mean"), "tf_vs_best": rating - group.transform("max"),
              "tf_rank_pct": group.rank(pct=True), "tf_std": group.transform("std")}
    for name, values in (("mark_at_weights", at_weights), ("tf_over_mark", edge)):
        group = values.groupby(race, sort=False)
        result[f"{name}_vs_field"] = values - group.transform("mean")
        result[f"{name}_vs_best"] = values - group.transform("max")
    return pd.DataFrame(result, index=frame.index, dtype=np.float32).add_prefix("x063_")


def relative_gaps(frame):
    out = {}
    for name in GAP_INPUTS:
        values = -frame[name] if name == "horse_avg_pos_20" else frame[name]
        group = values.groupby(frame.race_id, sort=False)
        out[f"{name}_vs_field"] = values - group.transform("mean")
        out[f"{name}_vs_best"] = values - group.transform("max")
    return pd.DataFrame(out, index=frame.index, dtype=np.float32).add_prefix("x063_")


GOING_SCALE = {"hard": 1., "firm": 1.5, "good to firm": 2., "good": 3., "good to yielding": 3.5,
               "good to soft": 4., "yielding": 4.,
               "yielding to soft": 4.5, "soft": 5., "yielding to heavy": 5., "soft to heavy": 5.5,
               "heavy": 6., "standard to fast": 2.5, "standard": 3., "standard to slow": 3.5,
               "standard / slow": 3.5}


def parse_going(value):
    if not isinstance(value, str):
        return np.nan
    main, _, note = value.lower().strip().partition("(")
    main = GOING_SCALE.get(main.strip(), np.nan)
    note = GOING_SCALE.get(note.replace("in places)", "").strip(), np.nan)
    return main if np.isnan(note) else main + .25 * (note - main)


def prior_elo_slope(frame, margins):
    """Fit on completed earlier dates, freezing the reference at the 2023 boundary."""
    day = frame.race_date
    elo = frame.horse_elo
    relative = elo - elo.groupby(frame.race_id, sort=False).transform("mean")
    centred = margins - margins.groupby(frame.race_id, sort=False).transform("mean")
    usable = day.lt("2023-01-01") & relative.notna() & centred.notna()
    daily = pd.DataFrame({"cross": (-relative * centred).where(usable, 0.),
                          "square": relative.pow(2).where(usable, 0.)}).groupby(day).sum().sort_index()
    prior = daily.cumsum().shift(1)
    slope = prior.cross / prior.square.where(prior.square.gt(0))
    return day.map(slope).to_numpy(dtype=float)


@njit
def _historical_form(starts, ends, days, distance, going, code, field_elo, margins, slope):
    similarity = np.array([[1., .5, .2, .2], [.5, 1., .2, .2], [.2, .2, 1., .5], [.2, .2, .5, 1.]])
    context = np.full(len(days), np.nan)
    overall = np.full(len(days), np.nan)
    support = np.zeros(len(days))
    for start, end in zip(starts, ends):
        for current in range(start, end):
            if not np.isfinite(slope[current]) or not (distance[current] > 0):
                continue
            total, weighted, total_squared, recent, recent_weighted = 0., 0., 0., 0., 0.
            seen = 0
            for earlier in range(current - 1, start - 1, -1):
                age = days[current] - days[earlier]
                if age > 1095 or seen == 30:
                    break
                if age <= 0 or not (distance[earlier] > 0):
                    continue
                figure = slope[current] * field_elo[earlier] - margins[earlier]
                if not np.isfinite(figure):
                    continue
                seen += 1
                recency = np.exp(-age / 365.)
                trip = np.log(distance[earlier] / distance[current]) / .2
                ground = 0.
                if code[current] == code[earlier] and code[current] != 1:
                    if np.isfinite(going[current]) and np.isfinite(going[earlier]):
                        ground = going[earlier] - going[current]
                weight = recency * np.exp(-.5 * (trip * trip + ground * ground)) * similarity[code[current], code[earlier]]
                recent += recency
                recent_weighted += recency * figure
                total += weight
                weighted += weight * figure
                total_squared += weight * weight
            if recent > 0:
                overall[current] = recent_weighted / recent
                context[current] = (weighted + overall[current]) / (total + 1.)
                support[current] = total * total / total_squared if total_squared > 0 else 0.
    return context, overall, support


def context_form(frame, raw_going):
    margins = pd.Series(-performance_targets(frame, PerformanceSpec("figure-063", ("horse_elo",))), index=frame.index)
    slope = prior_elo_slope(frame, margins)
    field_elo = frame.horse_elo.groupby(frame.race_id, sort=False).transform("mean")
    order = frame.sort_values(["horse_id", "race_date", "race_id"], kind="stable").index.to_numpy()
    horses = frame.horse_id.to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, horses[1:] != horses[:-1]])
    ends = np.r_[starts[1:], len(horses)]
    days = frame.race_date.to_numpy(dtype="datetime64[D]").astype(np.int64)
    race_type = frame.race_type.astype(str).str.casefold()
    surface = frame.surface.astype(str).str.casefold()
    code = np.select([race_type.eq("chase"), race_type.eq("hurdle"), surface.eq("turf")], [3, 2, 0], default=1)
    going = raw_going.map({value: parse_going(value) for value in raw_going.dropna().unique()}).to_numpy(dtype=float)
    distance = frame.distance_furlongs.to_numpy(dtype=float)
    values = _historical_form(starts, ends, days[order], distance[order], going[order], code[order],
                              field_elo.to_numpy(dtype=float)[order], margins.to_numpy()[order], slope[order])
    ctx, overall, support = (pd.Series(v, index=order).reindex(frame.index) for v in values)
    race = frame.race_id
    field = ctx.groupby(race, sort=False)
    pref = ctx - overall
    adj = ctx - claim_adjusted_weight(frame)
    out = {"fc_ctx_vs_field": ctx - field.transform("mean"), "fc_ctx_vs_best": ctx - field.transform("max"),
           "fc_ctx_rank_pct": field.rank(pct=True),
           "fc_all_vs_field": overall - overall.groupby(race, sort=False).transform("mean"),
           "fc_pref": pref, "fc_pref_vs_field": pref - pref.groupby(race, sort=False).transform("mean"),
           "fc_adj_vs_field": adj - adj.groupby(race, sort=False).transform("mean"), "fc_neff": support}
    return pd.DataFrame(out, index=frame.index, dtype=np.float32).add_prefix("x063_")
