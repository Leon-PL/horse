"""New-information candidates: ability in pounds, relatives' ratings and ground preference.

All three use only information available before the race:

- ``ability_in_pounds`` runs a Kalman filter per horse and code (Flat or jumps)
  over races in time order. A run's performance is pounds behind the winner (the
  performance model's 15/furlongs scale, capped at 30 lb) plus claim-adjusted weight.
  Each race's level is estimated from the runners' prior abilities, so the
  scale follows official ratings. Young or lightly raced horses get more
  process noise and so move faster. Features read the state before the race.
- ``relatives_ratings`` summarises the peak abilities of the dam's other
  offspring on earlier dates, raw and relative to their trainers' average.
- ``ground_preference`` compares today's going with the going of races the
  horse was withdrawn from for the going, using withdrawals published before
  the race.

Model settings below were fixed before any screening and are not tuned.
"""
import numpy as np
import pandas as pd
from numba import njit

from research.meta_learner.candidate_features import _name, _off_minutes
from src.feature_semantics import claim_adjusted_weight

KEYS = ["race_id", "horse_id"]
LB_PER_LENGTH_FURLONGS = 15.  # src.modelling.performance scale
MARGIN_CAP_LB = 30.
JUMPS = ("Hurdle", "Chase")
TURF_GOING = {"Fast": 1., "Good": 2., "Yielding": 3., "Soft": 4., "Heavy": 5.}
SETTINGS = {
    "observation_sd": 8.,         # lb, a run's noise around ability
    "margin_noise": .3,           # extra lb of noise per lb beaten
    "young_step_sd": 4.,          # lb per run for young or lightly raced horses
    "exposed_step_sd": 1.5,       # lb per run otherwise
    "monthly_variance": .5,       # lb^2 per 30 days off
    "rating_prior_sd": 8.,        # lb, a newcomer's prior when it has an official rating
    "unanchored_sd": 15.,         # lb, race level when no runner has a prior
    "lightly_raced": 4,           # runs in the code below which a horse counts as unexposed
}


def ordered_runs(history):
    """Rows in race order (date, scheduled time, race); returns the frame and its race boundaries."""
    h = history.copy()
    h["_off"] = _off_minutes(h.off_time.astype(str)).fillna(0).to_numpy()
    h = h.sort_values(["race_date", "_off", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    race = h.race_id.astype(str).to_numpy()
    first = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    # A race must be one block: rows with conflicting dates or times would split it.
    if len(first) != len(np.unique(race)):
        raise ValueError("Races are not contiguous in time order")
    return h, np.r_[first, len(h)]


def performance_lb(h):
    position = pd.to_numeric(h.finish_position, errors="coerce").to_numpy(float)
    lengths = pd.to_numeric(h.beaten_lengths, errors="coerce").to_numpy(float)
    distance = pd.to_numeric(h.distance_furlongs, errors="coerce").to_numpy(float)
    weight = claim_adjusted_weight(h).to_numpy(float)
    lengths = np.where(position == 1, 0., lengths)
    behind = np.minimum(lengths * LB_PER_LENGTH_FURLONGS / np.maximum(distance, 5.), MARGIN_CAP_LB)
    valid = (np.isfinite(position) & (position >= 1) & np.isfinite(lengths) & (lengths >= 0)
             & np.isfinite(distance) & (distance > 0) & np.isfinite(weight))
    return np.where(valid, weight - behind, np.nan), np.where(valid, behind, np.nan), valid


@njit(cache=True)
def _filter(starts, state, perf, behind, valid, rating, young, day, level, n_states,
            obs_sd, margin_noise, young_q, exposed_q, monthly_q, rating_sd, unanchored_sd, lightly_raced):
    n = len(perf)
    mu = np.full(n_states, np.nan)
    var = np.full(n_states, np.nan)
    runs = np.zeros(n_states)
    peak = np.full(n_states, np.nan)
    last_day = np.zeros(n_states)
    change = np.full(n_states, np.nan)
    f_mu = np.full(n, np.nan)
    f_sd = np.full(n, np.nan)
    f_runs = np.zeros(n)
    f_peak = np.full(n, np.nan)
    f_change = np.full(n, np.nan)
    f_days = np.full(n, np.nan)
    post = np.full(n, np.nan)
    resid = np.full(n, np.nan)  # performance against the horse's own prior ability
    jump = np.full(n, np.nan)   # how far the run raised the horse's peak
    for r in range(len(starts) - 1):
        a, b = starts[r], starts[r + 1]
        prior_mu = np.full(b - a, np.nan)
        prior_var = np.full(b - a, np.nan)
        for k in range(a, b):
            s = state[k]
            f_runs[k] = runs[s]
            if runs[s] > 0:
                gap = day[k] - last_day[s]
                q = young_q if (young[k] or runs[s] < lightly_raced) else exposed_q
                prior_mu[k - a] = mu[s]
                prior_var[k - a] = var[s] + q + monthly_q * gap / 30.
                f_mu[k] = mu[s]
                f_sd[k] = np.sqrt(prior_var[k - a])
                f_peak[k] = peak[s]
                f_change[k] = change[s]
                f_days[k] = gap
            elif np.isfinite(rating[k]):
                prior_mu[k - a] = rating[k]
                prior_var[k - a] = rating_sd * rating_sd
        # Race level: precision-weighted agreement between priors and performances.
        weight_sum, level_sum, perf_sum, count = 0., 0., 0., 0
        for k in range(a, b):
            if valid[k]:
                perf_sum += perf[k]
                count += 1
                if np.isfinite(prior_mu[k - a]):
                    noise = obs_sd * obs_sd + (margin_noise * behind[k]) ** 2
                    w = 1. / (prior_var[k - a] + noise)
                    weight_sum += w
                    level_sum += w * (prior_mu[k - a] - perf[k])
        if count == 0:
            continue
        if weight_sum > 0:
            offset = level_sum / weight_sum
            offset_var = 1. / weight_sum
        else:
            offset = level[a] - perf_sum / count
            offset_var = unanchored_sd * unanchored_sd
        for k in range(a, b):
            if not valid[k]:
                continue
            s = state[k]
            z = perf[k] + offset
            noise = obs_sd * obs_sd + (margin_noise * behind[k]) ** 2 + offset_var
            if np.isfinite(prior_mu[k - a]):
                gain = prior_var[k - a] / (prior_var[k - a] + noise)
                new_mu = prior_mu[k - a] + gain * (z - prior_mu[k - a])
                new_var = (1. - gain) * prior_var[k - a]
            else:
                new_mu, new_var = z, noise
            if runs[s] > 0:
                resid[k] = z - prior_mu[k - a]
                jump[k] = max(new_mu - peak[s], 0.)
            change[s] = new_mu - mu[s] if runs[s] > 0 else np.nan
            mu[s], var[s] = new_mu, new_var
            peak[s] = new_mu if runs[s] == 0 else max(peak[s], new_mu)
            runs[s] += 1
            last_day[s] = day[k]
            post[k] = new_mu
    return f_mu, f_sd, f_runs, f_peak, f_change, f_days, post, resid, jump


def _code(race_type):
    return np.where(pd.Series(race_type).isin(JUMPS).to_numpy(), "J", "F")


def _field(frame, column):
    values = frame[column]
    by = values.groupby(frame.race_id)
    return values - by.transform("mean"), values - by.transform("max"), by.rank(ascending=False, pct=True)


def ability_in_pounds(history, settings=SETTINGS, return_post=False):
    """Pre-race ability features; with ``return_post`` also each run's post-race ability (for relatives)."""
    h, starts = ordered_runs(history)
    perf, behind, valid = performance_lb(h)
    code = _code(h.race_type)
    state, _ = pd.factorize(pd.Series(h.horse_id.astype(str).to_numpy()) + "|" + code)
    age = pd.to_numeric(h.age, errors="coerce").to_numpy(float)
    young = np.where(code == "F", age <= 3, age <= 5)
    rating = pd.to_numeric(h.official_rating, errors="coerce").to_numpy(float)
    rating = np.where(rating > 0, rating, np.nan)
    # Unanchored races (every runner new and unrated) start from the code's median rating in the
    # first year of the store; that constant comes from data before every scored period.
    early = h.race_date.lt(h.race_date.min() + pd.DateOffset(years=1)).to_numpy()
    medians = {c: rating[early & (code == c) & np.isfinite(rating)] for c in ("F", "J")}
    medians = {c: float(np.median(v)) if len(v) else 70. for c, v in medians.items()}
    level = np.where(code == "F", medians["F"], medians["J"])
    day = (h.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    s = settings
    out = _filter(starts.astype(np.int64), state.astype(np.int64), perf, behind, valid, rating,
                  young.astype(np.bool_), day, level, int(state.max()) + 1,
                  s["observation_sd"], s["margin_noise"], s["young_step_sd"] ** 2, s["exposed_step_sd"] ** 2,
                  s["monthly_variance"], s["rating_prior_sd"], s["unanchored_sd"], s["lightly_raced"])
    f_mu, f_sd, f_runs, f_peak, f_change, f_days, post, resid, jump = out
    f = h[KEYS].copy()
    f["ab_mu"], f["ab_sd"], f["ab_runs"] = f_mu, f_sd, f_runs
    f["ab_peak"], f["ab_last_change"], f["ab_days_since"] = f_peak, f_change, f_days
    f["ab_below_peak"] = f_mu - f_peak
    weight = claim_adjusted_weight(h).to_numpy(float)
    f["_at_weights"] = f_mu - weight
    f["_vs_mark"] = f_mu - rating
    f["ab_mu_vs_field"], f["ab_mu_vs_best"], f["ab_mu_rank_pct"] = _field(f, "ab_mu")
    f["ab_at_weights_vs_field"], f["ab_at_weights_vs_best"], _ = _field(f, "_at_weights")
    f["ab_vs_mark_vs_field"], _, _ = _field(f, "_vs_mark")
    f["ab_peak_vs_field"], _, _ = _field(f, "ab_peak")
    f = f.drop(columns=["_at_weights", "_vs_mark"])
    if return_post:
        keep = [c for c in ("off_time", "trainer", "dam", "damsire", "sire", "finish_position", "beaten_lengths",
                            "distance_furlongs") if c in h]
        runs = h[[*KEYS, "race_date", *keep]].copy()
        runs["code"], runs["post_mu"], runs["prior_runs"] = code, post, f_runs
        runs["resid"], runs["peak_jump"] = resid, jump
        return f, runs
    return f


# --------------------------------------------------------------------------
# Relatives


def _families(runs):
    dam = _name(runs.dam)
    key = (dam + "|" + _name(runs.damsire)).where(dam.ne(""))
    pairs = pd.DataFrame({"horse_id": runs.horse_id, "family": key}).dropna()
    counts = pairs.value_counts().rename("n").reset_index()
    modal = counts.sort_values(["horse_id", "n", "family"], ascending=[True, False, True]).drop_duplicates("horse_id")
    return runs.horse_id.map(modal.set_index("horse_id").family)


class _Peaks:
    """Running per-horse maxima in one family, with their sum and the top two."""
    __slots__ = ("peak", "total", "top")

    def __init__(self):
        self.peak, self.total, self.top = {}, 0., []

    def update(self, horse, value):
        old = self.peak.get(horse)
        if old is not None and value <= old:
            return
        self.peak[horse] = value
        self.total += value - (old or 0.)
        self.top = sorted([t for t in self.top if t[1] != horse] + [(value, horse)], reverse=True)[:2]

    def excluding(self, horse):
        own = self.peak.get(horse)
        n = len(self.peak) - (own is not None)
        if n == 0:
            return 0, np.nan, np.nan
        best = next((v for v, h in self.top if h != horse), np.nan)
        return n, (self.total - (own or 0.)) / n, best


def relatives_ratings(history):
    """Peak abilities of the dam's other offspring on earlier dates, raw and relative to their trainers."""
    _, runs = ability_in_pounds(history, return_post=True)
    runs["family"] = _families(runs)
    # Each run's ability relative to its trainer's average ability on earlier dates (same code).
    r = runs.dropna(subset=["post_mu"]).copy()
    day = r.groupby(["trainer", "code", "race_date"], sort=True).post_mu.agg(["sum", "count"])
    cum = day.groupby(level=[0, 1]).cumsum() - day
    mean_before = (cum["sum"] / cum["count"].where(cum["count"] > 0)).rename("trainer_mean")
    r = r.merge(mean_before.reset_index(), on=["trainer", "code", "race_date"], how="left")
    r["vs_trainer"] = r.post_mu - r.trainer_mean
    events = r.dropna(subset=["family"]).sort_values("race_date", kind="stable")
    targets = runs.dropna(subset=["family"])[[*KEYS, "race_date", "family", "code"]]
    out = {k: np.full(len(targets), np.nan) for k in ("relr_siblings", "relr_mean_peak", "relr_best_peak",
                                                       "relr_mean_vs_trainer", "relr_best_vs_trainer")}
    raw, adjusted = {}, {}
    event_days = events.race_date.to_numpy()
    event_cols = (events.family.to_numpy(), events.code.to_numpy(), events.horse_id.to_numpy(),
                  events.post_mu.to_numpy(), events.vs_trainer.to_numpy())
    t_days = targets.race_date.to_numpy()
    t_order = np.argsort(t_days, kind="stable")
    t_cols = (targets.family.to_numpy(), targets.code.to_numpy(), targets.horse_id.to_numpy())
    e = 0
    for i in t_order:
        while e < len(events) and event_days[e] < t_days[i]:  # strictly earlier dates only
            fam, code, horse, value, rel = (c[e] for c in event_cols)
            raw.setdefault((fam, code), _Peaks()).update(horse, value)
            if np.isfinite(rel):
                adjusted.setdefault((fam, code), _Peaks()).update(horse, rel)
            e += 1
        key = (t_cols[0][i], t_cols[1][i])
        horse = t_cols[2][i]
        if key in raw:
            n, mean, best = raw[key].excluding(horse)
            out["relr_siblings"][i], out["relr_mean_peak"][i], out["relr_best_peak"][i] = n, mean, best
        else:
            out["relr_siblings"][i] = 0
        if key in adjusted:
            _, mean, best = adjusted[key].excluding(horse)
            out["relr_mean_vs_trainer"][i], out["relr_best_vs_trainer"][i] = mean, best
    result = targets[KEYS].copy()
    for name, values in out.items():
        result[name] = values
    result["relr_mean_peak_vs_field"] = result.relr_mean_peak - result.groupby("race_id").relr_mean_peak.transform("mean")
    return result


# --------------------------------------------------------------------------
# Ground preference from withdrawals


def ground_preference(history, withdrawals):
    """Today's going against the going of races the horse was withdrawn from for the going.

    ``withdrawals`` needs horse_id, available_at (UTC) and ``withdrawn_going``: the going of the
    race the horse was withdrawn from, on the turf scale. A withdrawal counts once it is available
    before the race's scheduled start.
    """
    h = history[[*KEYS, "race_date", "off_time", "going"]].copy()
    h["horse_id"] = h.horse_id.astype(str)
    h["start"] = pd.to_datetime(h.race_date.dt.strftime("%Y-%m-%d") + " " + h.off_time.astype(str).str[:5],
                                utc=True, errors="coerce")
    h["g"] = h.going.map(TURF_GOING)
    # The horse's usual turf going on earlier dates, to say which way its dislike points.
    turf = h.dropna(subset=["g"])
    day = turf.groupby(["horse_id", "race_date"]).g.agg(["sum", "count"])
    cum = day.groupby(level=0).cumsum() - day
    usual = (cum["sum"] / cum["count"].where(cum["count"] > 0)).rename("usual_going").reset_index()
    h = h.merge(usual, on=["horse_id", "race_date"], how="left")
    h["row"] = np.arange(len(h))
    w = withdrawals.dropna(subset=["withdrawn_going"])[["horse_id", "available_at", "withdrawn_going"]].copy()
    w["horse_id"] = w.horse_id.astype(str)
    w["available_at"] = pd.to_datetime(w.available_at, utc=True)
    pairs = h[["row", "horse_id", "start", "g", "usual_going"]].merge(w, on="horse_id")
    pairs = pairs.loc[pairs.available_at < pairs.start]
    pairs["same"] = (pairs.withdrawn_going == pairs.g).astype(float).where(pairs.g.notna())
    pairs["gap"] = (pairs.withdrawn_going - pairs.g).abs()
    by = pairs.groupby("row")
    summary = pd.DataFrame({
        "gw_count": by.size(),
        "gw_softest": by.withdrawn_going.max(),
        "gw_firmest": by.withdrawn_going.min(),
        "gw_same_going": by.same.sum(min_count=1),
        "gw_distance": by.gap.min(),
        "_mean": by.withdrawn_going.mean(),
        "_today": by.g.first(),
        "_usual": by.usual_going.first(),
    })
    # Positive when today's going is off the horse's usual going in the direction it was pulled out of.
    summary["gw_direction"] = (summary._mean - summary._usual) * (summary._today - summary._usual)
    out = h[KEYS].copy()
    out["horse_id"] = history.horse_id.to_numpy()
    summary = summary.drop(columns=["_mean", "_today", "_usual"]).reindex(np.arange(len(h)))
    for name in summary:
        out[name] = summary[name].to_numpy()
    out["gw_count"] = out.gw_count.fillna(0.)
    return out


def going_withdrawals(history):
    """BHA going withdrawals with the going of the meeting each was made at.

    Availability follows the store's policy (src.bha_features): 48 hours after the race.
    """
    from src.bha_features import load_bundle, source_paths
    from src.source_dataset import committed_dataset, use_manifest
    from src.utils import canonical_track_key
    with use_manifest(committed_dataset()):
        paths = source_paths()
        _, _, withdrawals, _ = load_bundle(paths)
    reports = pd.read_parquet(paths["bha/reports"], columns=["report_key", "course", "race_date"]).drop_duplicates("report_key")
    w = withdrawals.loc[withdrawals.going.gt(0)].merge(reports, on="report_key", how="left")
    w["track_key"] = w.course.map(canonical_track_key)
    w["race_date"] = pd.to_datetime(w.race_date)
    meetings = history[["track", "race_date", "going"]].copy()
    meetings["track_key"] = meetings.track.map(canonical_track_key)
    meetings["g"] = meetings.going.map(TURF_GOING)
    # The meeting's most common turf going; changes during a meeting are rare.
    modal = (meetings.dropna(subset=["g"]).groupby(["track_key", "race_date"]).g
             .agg(lambda x: x.value_counts().index[0]).rename("withdrawn_going").reset_index())
    w = w.merge(modal, on=["track_key", "race_date"], how="left")
    w["available_at"] = pd.to_datetime(w.event_at, utc=True) + pd.Timedelta(hours=48)
    return w


def ground_from_withdrawals(history):
    return ground_preference(history, going_withdrawals(history))


def ability_in_pounds_jumps(history):
    """Exploratory, chosen after seeing the all-races screen: ability inputs on jumps races only.

    Flat rows are blank, so every Flat runner takes the same branch and the correction is constant
    within a Flat race, which leaves its forecast unchanged.
    """
    features = ability_in_pounds(history)
    kind = features[KEYS].merge(history[[*KEYS, "race_type"]], on=KEYS, how="left", validate="one_to_one").race_type
    flat = ~kind.isin(JUMPS).to_numpy()
    columns = [c for c in features if c not in KEYS]
    features.loc[flat, columns] = np.nan
    return features


# --------------------------------------------------------------------------
# Improvement (S14) and bounce (S15), from each run's residual against the horse's prior ability

UNEXPOSED_RUNS = 6
SIRE_SHRINK, TRAINER_SHRINK = 20., 30.


def _earlier_sum(runs, by, value, mask, window_days=None):
    """Per row: sum and count of ``value`` over ``mask`` rows in the same group on strictly earlier dates."""
    day = (runs.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    group = runs[by].astype(str).agg("|".join, axis=1) if len(by) > 1 else runs[by[0]].astype(str)
    missing = runs[by].isna().any(axis=1).to_numpy()
    codes, labels = pd.factorize(group)
    source = mask & ~missing & np.isfinite(value)
    daily = (pd.DataFrame({"g": codes[source], "d": day[source], "v": value[source]})
             .groupby(["g", "d"]).v.agg(["sum", "count"]).reset_index())
    key = daily.g.to_numpy(np.int64) * 1_000_000 + daily.d.to_numpy(np.int64)
    cs = np.r_[0., np.cumsum(daily["sum"].to_numpy())]
    cn = np.r_[0., np.cumsum(daily["count"].to_numpy())]
    target = codes.astype(np.int64) * 1_000_000 + day
    upper = np.searchsorted(key, target, side="left")  # strictly earlier dates
    lower = np.searchsorted(key, codes.astype(np.int64) * 1_000_000, side="left")
    if window_days is not None:
        lower = np.maximum(lower, np.searchsorted(key, target - window_days, side="left"))
    total, count = cs[upper] - cs[lower], cn[upper] - cn[lower]
    total[missing], count[missing] = np.nan, np.nan
    return total, count


def _previous(runs, column, by=("horse_id", "code"), lag=1):
    return runs.groupby(list(by), sort=False)[column].shift(lag)


def _with_field(frame, columns):
    for c in columns:
        frame[c + "_vs_field"] = frame[c] - frame.groupby("race_id")[c].transform("mean")
    return frame


def improvement_inputs(history):
    """S14: how much horses like this one have run above their prior ability on earlier dates."""
    _, runs = ability_in_pounds(history, return_post=True)
    resid = runs.resid.to_numpy(float)
    unexposed = (runs.prior_runs < UNEXPOSED_RUNS).to_numpy()
    out = runs[KEYS].copy()
    total, count = _earlier_sum(runs, ["sire", "code"], resid, unexposed)
    out["imp_sire_unexposed"] = total / (count + SIRE_SHRINK)
    total, count = _earlier_sum(runs, ["trainer", "code"], resid, np.ones(len(runs), bool), window_days=365)
    out["imp_trainer_365"] = total / (count + TRAINER_SHRINK)
    total, count = _earlier_sum(runs, ["trainer", "code"], resid, unexposed, window_days=365)
    out["imp_trainer_unexposed_365"] = total / (count + TRAINER_SHRINK)
    last3 = np.column_stack([_previous(runs, "resid", lag=k).to_numpy(float) for k in (1, 2, 3)])
    seen = np.isfinite(last3).sum(axis=1)
    out["imp_horse_last3"] = np.where(seen > 0, np.nansum(last3, axis=1) / np.maximum(seen, 1), np.nan)
    out["imp_prior_runs"] = runs.prior_runs.to_numpy(float)
    return _with_field(out, ["imp_sire_unexposed", "imp_trainer_365"])


def bounce_inputs(history):
    """S15: the last run's effort and how soon the horse is back."""
    _, runs = ability_in_pounds(history, return_post=True)
    out = runs[KEYS].copy()
    position = pd.to_numeric(runs.finish_position, errors="coerce")
    lengths = pd.to_numeric(runs.beaten_lengths, errors="coerce")
    runs["_close"] = ((position == 1) | ((position > 0) & (lengths < 1))).astype(float).where(position.notna())
    out["bn_last_resid"] = _previous(runs, "resid").to_numpy(float)
    out["bn_last_jump"] = _previous(runs, "peak_jump").to_numpy(float)
    prior_at_last = _previous(runs, "prior_runs").to_numpy(float)
    out["bn_last_best"] = np.where(np.isfinite(out.bn_last_jump), ((out.bn_last_jump > 0) & (prior_at_last >= 2)).astype(float), np.nan)
    out["bn_last_close"] = _previous(runs, "_close").to_numpy(float)
    out["bn_last_distance"] = _previous(runs, "distance_furlongs").to_numpy(float)
    out["bn_days_since"] = (runs.race_date - _previous(runs, "race_date")).dt.days.to_numpy(float)
    _, count = _earlier_sum(runs, ["horse_id"], np.ones(len(runs)), np.ones(len(runs), bool), window_days=60)
    out["bn_runs_60d"] = count
    out["bn_quick_return_x_resid"] = out.bn_last_resid * (out.bn_days_since < 21)
    return out


def jumps_margin_inputs(history):
    """Ability in pounds and bounce inputs on jumps races only (the jumps margin specialist)."""
    ability = ability_in_pounds_jumps(history)
    bounce = bounce_inputs(history)
    kind = bounce[KEYS].merge(history[[*KEYS, "race_type"]], on=KEYS, how="left", validate="one_to_one").race_type
    columns = [c for c in bounce if c not in KEYS]
    bounce.loc[~kind.isin(JUMPS).to_numpy(), columns] = np.nan
    return ability.merge(bounce, on=KEYS, how="inner", validate="one_to_one")
