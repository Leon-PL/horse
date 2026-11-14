"""Inputs of the forecast correction: how the model's earlier forecasts turned out.

The win model's features describe a horse and its connections by raw records.
These inputs measure the same entities against the model's own out-of-sample
forecasts: on every earlier date, how did a trainer's, sire's or horse's
runners finish, and how were they priced at the off, relative to what the
model said? ``src.modelling.correction`` fits a second model on them.

Every function takes a *history frame*: one row per runner, races contiguous
and in date order, with a forecast column (NaN where the runner has none) and
optionally the closing market probability of that race. Rows of races not yet
run (no result, no price) contribute nothing and receive inputs like any
other row. Everything reads strictly earlier dates only: a race never sees its
own result or price, nor any from the same day.

Measured in ``docs/research/ERROR_ANATOMY_2026-11-10.md``; the research
modules in ``research/error_anatomy`` import these definitions.
"""
import numba
import numpy as np
import pandas as pd

KEYS = ["race_id", "horse_id"]
# (win shrinkage in units of forecast variance, order and gap shrinkage in runs)
SHRINK = (20., 30.)
HORSE_SHRINK = (2., 3.)
ENTITIES = {
    "trainer": ["trainer"],
    "jockey": ["jockey"],
    "sire": ["sire"],
    "trainer_jockey": ["trainer", "jockey"],
    "trainer_code": ["trainer", "_code"],
    "trainer_fresh": ["trainer", "_fresh"],
}
HORSE = {"horse": ["horse_id"]}
WAVE2 = {
    "trainer_track": ["trainer", "track"],
    "trainer_handicap": ["trainer", "_handicap"],
    "sire_surface": ["sire", "_surface"],
    "sire_going": ["sire", "going"],
    "sire_trip": ["sire", "_trip"],
    "damsire": ["damsire"],
}
LONG_HALF_LIFE = 365.
# Columns a history frame needs beyond KEYS, race_date, won, finish_position and the forecast.
ENTITY_COLUMNS = ("trainer", "jockey", "sire", "race_type", "days_since_last_run", "track", "going", "damsire",
                  "handicap", "surface", "distance_furlongs")


def bounds_of(race_ids):
    ids = np.asarray(race_ids)
    change = np.r_[True, ids[1:] != ids[:-1]]
    if len(pd.unique(ids)) != int(change.sum()):
        raise ValueError("Races must be contiguous")
    return np.r_[np.flatnonzero(change), len(ids)].astype(np.int64)


def race_normalised(values, race_id):
    values = pd.Series(np.asarray(values, dtype=float))
    total = values.groupby(np.asarray(race_id)).transform("sum")
    return (values / total).to_numpy()


def closing_probability(frame):
    """Market probability at the off: BSP where the whole field has one, else SP, else NaN."""
    made = {}
    for column in ("bsp", "odds"):
        price = pd.to_numeric(frame[column], errors="coerce")
        whole = (price > 1).groupby(frame.race_id).transform("all")
        made[column] = pd.Series(np.where(whole, race_normalised(1 / price.where(price > 1), frame.race_id), np.nan),
                                 index=frame.index)
    return made["bsp"].fillna(made["odds"])


def run_residuals(frame, probability="p_single"):
    """Per-run win and rivals-beaten residuals against the forecast; NaN where there is no forecast.

    Rivals beaten is the share of rivals the runner beat minus the share the forecast expected
    (``mean_j p_i / (p_i + p_j)``). Non-finishers rank behind every finisher and level with each other.
    """
    bounds = bounds_of(frame.race_id.to_numpy())
    p = frame[probability].to_numpy(float)
    won = frame.won.to_numpy(float)
    position = pd.to_numeric(frame.finish_position, errors="coerce").to_numpy(float)
    position = np.where(np.isfinite(position) & (position >= 1), position, np.inf)  # non-finishers last
    order = np.full(len(frame), np.nan)
    for a, b in zip(bounds[:-1], bounds[1:]):
        q, pos = p[a:b], position[a:b]
        n = b - a
        if n < 2 or not np.isfinite(q).all():
            continue
        beat = (pos[:, None] < pos[None, :]) + .5 * (pos[:, None] == pos[None, :])
        expect = q[:, None] / (q[:, None] + q[None, :])
        np.fill_diagonal(beat, 0.)
        np.fill_diagonal(expect, 0.)
        order[a:b] = (beat.sum(1) - expect.sum(1)) / (n - 1)
    return pd.DataFrame({"win": won - p, "variance": p * (1 - p), "order": order}, index=frame.index)


def entity_keys(frame):
    work = pd.DataFrame(index=frame.index)
    for column in ("trainer", "jockey", "sire", "horse_id", "track", "going", "damsire"):
        if column in frame:
            work[column] = frame[column]
    if "handicap" in frame:
        work["_handicap"] = pd.to_numeric(frame.handicap, errors="coerce").fillna(0).astype(int)
    if "surface" in frame:
        work["_surface"] = np.where(frame.surface.eq("Turf"), "turf", "aw")
    if "distance_furlongs" in frame:
        trip = pd.to_numeric(frame.distance_furlongs, errors="coerce")
        work["_trip"] = pd.cut(trip, [0, 6.5, 9.5, 13, 18, 24, 99], labels=False).astype("float").fillna(-1).astype(int)
    work["_code"] = np.where(frame.race_type.isin(["Hurdle", "Chase"]), "J", "F")
    days = pd.to_numeric(frame.days_since_last_run, errors="coerce")
    work["_fresh"] = np.where(days.isna() | (days <= 0), "debut", np.where(days > 60, "break", "fit"))
    return work


def entity_inputs(frame, probability="p_single", *, entities=ENTITIES, shrink=SHRINK, half_life=None,
                  market=None, prefix="er", lag_days=0, residual=None, extra_keys=None):
    """Shrunk prior-date signal means per entity key, plus field-relative versions.

    ``half_life`` (days) down-weights older runs. ``market`` names a column of race-normalised
    market probabilities; when given, the market-gap family ``mg_*`` is added. ``lag_days``
    ignores runs in the last that-many days before the race (a stress test for timing leaks).
    ``extra_keys`` maps further key column names to values aligned with ``frame``.
    """
    work = entity_keys(frame)
    for column, values in (extra_keys or {}).items():
        work[column] = np.asarray(values)
    residual = run_residuals(frame, probability) if residual is None else residual
    signals = {"win": residual.win, "variance": residual.variance, "order": residual.order}
    known = residual.win.notna() & residual.order.notna()
    if market is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.log(frame[market].to_numpy(float)) - np.log(frame[probability].to_numpy(float))
        gap = pd.Series(np.where(np.isfinite(gap), gap, np.nan), index=frame.index).clip(-3, 3)
        signals["gap"] = gap.fillna(0.)
        signals["gap_runs"] = gap.notna().astype(float)
    day = (frame.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    weight = np.ones(len(frame)) if half_life is None else np.exp2((day - day.min()) / half_life)
    out = frame[KEYS].copy()
    for name, by in entities.items():
        text = [work[column].astype(str) for column in by]
        key = text[0]
        for part in text[1:]:
            key = key + "|" + part
        blank = np.logical_or.reduce([part.isin(["", "nan", "None"]).to_numpy() for part in text])
        valid = known & work[by].notna().all(axis=1) & ~blank
        table = pd.DataFrame({"_key": key, "race_date": frame.race_date, "runs": weight})
        for signal, values in signals.items():
            table[signal] = values.to_numpy(float) * weight
        daily = table[valid.to_numpy()].groupby(["_key", "race_date"], sort=True).sum()
        totals = daily.groupby(level="_key").cumsum().reset_index()
        # Totals up to each date the entity ran, read back for strictly earlier dates only.
        lookup = pd.DataFrame({"_key": key, "race_date": frame.race_date - pd.Timedelta(days=lag_days)}).reset_index()
        lookup = lookup.sort_values("race_date", kind="stable")
        joined = pd.merge_asof(lookup, totals.sort_values("race_date"), on="race_date", by="_key",
                               allow_exact_matches=False).set_index("index").reindex(frame.index).fillna(0.)
        joined = joined[[c for c in joined if c not in ("_key", "race_date")]].div(weight, axis=0)
        made = {f"{prefix}_{name}_win": joined.win / (joined.variance + shrink[0]),
                f"{prefix}_{name}_order": joined.order / (joined.runs + shrink[1])}
        if market is not None:
            made[f"mg_{name}_gap"] = joined.gap / (joined.gap_runs + shrink[1])
        for column, values in made.items():
            out[column] = values.astype("float32")
            out[f"{column}_vs_field"] = (out[column] - out[column].groupby(frame.race_id).transform("mean")
                                         ).astype("float32")
        out[f"{prefix}_{name}_runs"] = joined.runs.astype("float32")
    return out


def long_inputs(frame, probability="p_single", market="p_market", residual=None):
    """Residuals and market gap by connection, the horse's own record, and the finer second-wave keys."""
    residual = run_residuals(frame, probability) if residual is None else residual
    entity = entity_inputs(frame, probability, market=market, half_life=LONG_HALF_LIFE, residual=residual)
    horse = entity_inputs(frame, probability, market=market, half_life=LONG_HALF_LIFE, entities=HORSE,
                          shrink=HORSE_SHRINK, residual=residual)
    wave2 = entity_inputs(frame, probability, half_life=LONG_HALF_LIFE, entities=WAVE2, residual=residual)
    return entity.merge(horse, on=KEYS, validate="one_to_one").merge(wave2, on=KEYS, validate="one_to_one")


# --------------------------------------------------------------------------
# Run-by-run horse state and short-memory stable form

LAGS = {"gap": 5, "order": 3, "win": 3}
ALPHAS = (.5, .25)
CONDITIONS = ("_code", "_surface", "_trip")
RECENT = {"trainer": ["trainer"], "jockey": ["jockey"], "trainer_jockey": ["trainer", "jockey"], "sire": ["sire"]}
HALF_LIVES = (30., 90.)


@numba.njit(cache=True)
def moving_mean(group, values, alpha):
    """Run-weighted mean of earlier finite values within each group; rows are sorted by group then time."""
    out = np.full(len(values), np.nan)
    total = weight = 0.
    for i in range(len(values)):
        if i == 0 or group[i] != group[i - 1]:
            total = weight = 0.
        if weight > 0:
            out[i] = total / weight
        if np.isfinite(values[i]):
            total = (1 - alpha) * total + values[i]
            weight = (1 - alpha) * weight + 1.
        else:
            total *= 1 - alpha
            weight *= 1 - alpha
    return out


def run_signals(frame, probability="p_single", market="p_market", residual=None):
    residual = run_residuals(frame, probability) if residual is None else residual
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.log(frame[market].to_numpy(float)) - np.log(frame[probability].to_numpy(float))
    gap = pd.Series(np.where(np.isfinite(gap), gap, np.nan), index=frame.index).clip(-3, 3)
    gap = gap - gap.groupby(frame.race_id).transform("mean")
    return pd.DataFrame({"gap": gap, "order": residual.order, "win": residual.win}, index=frame.index)


def horse_state(frame, probability="p_single", market="p_market", residual=None):
    """One row per runner of ``frame`` (race order kept) with the horse's run-by-run state."""
    work = pd.concat([frame[[*KEYS, "race_date"]], run_signals(frame, probability, market, residual),
                      entity_keys(frame)[list(CONDITIONS)]], axis=1)
    work["_row"] = np.arange(len(work))
    work = work.sort_values(["horse_id", "race_date", "_row"], kind="stable").reset_index(drop=True)
    horse = work.groupby("horse_id", sort=False)
    code = horse.ngroup().to_numpy()
    day = (work.race_date - pd.Timestamp("2000-01-01")).dt.days.astype(float)
    made = {}
    # A second run on the same date is not an earlier date: such rows get no state at all.
    same_day = (horse.race_date.shift(1) == work.race_date).to_numpy()
    for name, lags in LAGS.items():
        for k in range(1, lags + 1):
            made[f"hs_{name}_l{k}"] = horse[name].shift(k)
        for alpha in ALPHAS:
            made[f"hs_{name}_ma{int(alpha * 100)}"] = moving_mean(code, work[name].to_numpy(float), alpha)
    for k in (1, 2, 3):
        made[f"hs_days_l{k}"] = day - day.groupby(work.horse_id, sort=False).shift(k)
    made["hs_gap_change"] = made["hs_gap_l1"] - made["hs_gap_l2"]
    for condition in CONDITIONS:
        by = work.groupby(["horse_id", condition], sort=False)
        for name in ("gap", "order"):
            value = work[name]
            runs = value.notna().astype(float)
            total = value.fillna(0.)
            # Running totals minus the row itself: earlier rows only.
            prior_sum = total.groupby([work.horse_id, work[condition]], sort=False).cumsum() - total
            prior_runs = runs.groupby([work.horse_id, work[condition]], sort=False).cumsum() - runs
            made[f"hs_{name}_same{condition}"] = prior_sum / (prior_runs + 1.)
        made[f"hs_runs_same{condition}"] = by.cumcount().astype(float)
    out = pd.DataFrame(made, index=work.index).astype("float32")
    out[same_day] = np.nan
    out = pd.concat([work[[*KEYS, "_row"]], out], axis=1).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    relative = [c for c in out if c.startswith(("hs_gap_l1", "hs_gap_ma", "hs_order_ma"))]
    for column in relative:
        out[f"{column}_vs_field"] = (out[column] - out[column].groupby(frame.race_id.to_numpy()).transform("mean")
                                     ).astype("float32")
    return out


def entity_recent(frame, probability="p_single", market="p_market", residual=None):
    out = frame[KEYS].copy()
    residual = run_residuals(frame, probability) if residual is None else residual
    for half_life in HALF_LIVES:
        values = entity_inputs(frame, probability, entities=RECENT, half_life=half_life, market=market,
                               prefix=f"er{int(half_life)}", residual=residual)
        values = values.rename(columns={c: c.replace("mg_", f"mg{int(half_life)}_") for c in values if c.startswith("mg_")})
        out = out.merge(values, on=KEYS, validate="one_to_one")
    return out


# --------------------------------------------------------------------------
# Has the form worked out? Later runs of a horse's past rivals, against the model

FRANK_SHRINK = 4.  # rival runs
FRANK_LOOKBACK = 3


@numba.njit(cache=True)
def _frank(h_start, h_dates, cum, cum_n, race_of, pos_of_run, race_start, race_members, position, lookback):
    """For each run (in horse order), sums of rivals' later residuals over the last 1 and ``lookback`` races.

    ``cum``/``cum_n`` are per-horse running totals including the run itself, shape (runs, signals).
    Output: [last race, all lookback races] x [rivals ahead, rivals behind] x signals, then counts.
    """
    n, k = cum.shape
    sums = np.zeros((n, 2, 2, k))
    counts = np.zeros((n, 2, 2))
    for horse in range(len(h_start) - 1):
        a, b = h_start[horse], h_start[horse + 1]
        for t in range(a, b):
            today = h_dates[t]
            for back in range(1, lookback + 1):
                r = t - back
                if r < a:
                    break
                if h_dates[r] >= today:
                    continue
                race = race_of[r]
                for m in range(race_start[race], race_start[race + 1]):
                    j = race_members[m]            # the rival's run in that race, in horse order
                    if j == r:
                        continue
                    rival = pos_of_run[j]          # which horse block
                    jb = h_start[rival + 1]
                    # last run of the rival strictly before today
                    lo, hi = j, jb
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        if h_dates[mid] < today:
                            lo = mid
                        else:
                            hi = mid
                    if lo == j:
                        continue                   # the rival has not run since
                    side = 0 if position[j] < position[r] else 1
                    added = cum_n[lo] - cum_n[j]
                    if added <= 0:
                        continue
                    for s in range(k):
                        value = cum[lo, s] - cum[j, s]
                        sums[t, 1, side, s] += value
                        if back == 1:
                            sums[t, 0, side, s] += value
                    counts[t, 1, side] += added
                    if back == 1:
                        counts[t, 0, side] += added
    return sums, counts


def franking_inputs(history, forecasts, market=None, probability="p_single", residual=None):
    """Shrunk means of past rivals' later residuals, split by whether the rival finished ahead or behind.

    ``history``: every run (race_id, horse_id, race_date, finish_position), all dates. ``forecasts``:
    a history frame of the runs that have a forecast. One output row per run of ``history``.
    """
    residual = run_residuals(forecasts, probability) if residual is None else residual
    signals = pd.DataFrame({"order": residual.order, "win": residual.win}, index=forecasts.index)
    if market is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.log(forecasts[market].to_numpy(float)) - np.log(forecasts[probability].to_numpy(float))
        signals["gap"] = np.clip(np.where(np.isfinite(gap), gap, np.nan), -3, 3)
    signals[KEYS[0]], signals[KEYS[1]] = forecasts.race_id.astype(str).to_numpy(), forecasts.horse_id.to_numpy()
    h = history[[*KEYS, "race_date", "finish_position"]].copy()
    h["race_id"] = h.race_id.astype(str)
    h = h.merge(signals, on=KEYS, how="left", validate="one_to_one")
    names = [c for c in ("order", "win", "gap") if c in h]
    h["_valid"] = h[names].notna().all(axis=1).astype(float)
    h = h.sort_values(["horse_id", "race_date", "race_id"], kind="stable").reset_index(drop=True)
    position = pd.to_numeric(h.finish_position, errors="coerce").to_numpy(float)
    position = np.where(np.isfinite(position) & (position >= 1), position, 999.)
    horse_code, _ = pd.factorize(h.horse_id)
    h_start = np.r_[np.flatnonzero(np.r_[True, horse_code[1:] != horse_code[:-1]]), len(h)].astype(np.int64)
    values = h[names].fillna(0.).to_numpy(float) * h._valid.to_numpy()[:, None]
    cum = pd.DataFrame(values).groupby(horse_code).cumsum().to_numpy(float)
    cum_n = h._valid.groupby(horse_code).cumsum().to_numpy(float)
    race_code, _ = pd.factorize(h.race_id)
    by_race = np.argsort(race_code, kind="stable").astype(np.int64)
    race_start = np.r_[0, np.cumsum(np.bincount(race_code))].astype(np.int64)
    dates = (h.race_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(np.int64)
    sums, counts = _frank(h_start, dates, np.ascontiguousarray(cum), cum_n, race_code.astype(np.int64),
                          horse_code.astype(np.int64), race_start, by_race, position, FRANK_LOOKBACK)
    out = h[KEYS].copy()
    for w, window in enumerate(("last", "three")):
        for side, label in enumerate(("ahead", "behind")):
            n = counts[:, w, side]
            out[f"fr_{window}_{label}_runs"] = n.astype("float32")
            for k, name in enumerate(names):
                out[f"fr_{window}_{label}_{name}"] = (sums[:, w, side, k] / (n + FRANK_SHRINK)).astype("float32")
        total = counts[:, w].sum(1)
        for k, name in enumerate(names):
            out[f"fr_{window}_all_{name}"] = (sums[:, w, :, k].sum(1) / (total + FRANK_SHRINK)).astype("float32")
    relative = [c for c in out if c.startswith("fr_three_") and not c.endswith("_runs")]
    for column in relative:
        out[f"{column}_vs_field"] = (out[column] - out[column].groupby(h.race_id).transform("mean")).astype("float32")
    return out


# --------------------------------------------------------------------------
# What the card's form figures say about runs the database does not hold

FORM_FAIL = set("PFURBCSDO")


def form_figures(form, hidden):
    """Per runner: wins, places, fails, best, mean, last and count over the oldest ``hidden`` form figures."""
    out = np.full((len(form), 7), np.nan)
    for i, (text, count) in enumerate(zip(form, hidden)):
        if not count or count <= 0 or not isinstance(text, str):
            continue
        marks = [c for c in text.upper() if c.isalnum()][:int(count)]
        places = [10 if c == "0" else int(c) for c in marks if c.isdigit()]
        fails = sum(c in FORM_FAIL for c in marks)
        if not marks:
            continue
        out[i, 0] = sum(p == 1 for p in places)
        out[i, 1] = sum(p <= 3 for p in places)
        out[i, 2] = fails
        if places:
            out[i, 3], out[i, 4] = min(places), float(np.mean(places))
        last = marks[-1]
        out[i, 5] = (10 if last == "0" else int(last)) if last.isdigit() else 12.
        out[i, 6] = len(marks)
    return out


def hidden_form(frame, form, hidden):
    values = form_figures(form, hidden)
    out = frame[KEYS].copy()
    for j, name in enumerate(("wins", "places", "fails", "best", "mean", "last", "runs")):
        out[f"hf_{name}"] = values[:, j].astype("float32")
    out["hf_win_share"] = (out.hf_wins / out.hf_runs).astype("float32")
    race = frame.race_id.to_numpy()
    for name in ("hf_wins", "hf_best"):
        filled = out[name].fillna(0. if name == "hf_wins" else 6.)
        out[f"{name}_vs_field"] = (filled - filled.groupby(race).transform("mean")).astype("float32")
    return out


# --------------------------------------------------------------------------
# Did the money come late on the horse's earlier runs?

SNAPSHOTS = ("ltp_0800", "ltp_1100", "ltp_3600s")
MOVE_ALPHA = .5


def lagged(frame, signals, lags, prefix, relative=()):
    """Lags and a run-weighted average of each signal over the horse's runs on strictly earlier dates."""
    work = pd.concat([frame[[*KEYS, "race_date"]].reset_index(drop=True), signals.reset_index(drop=True)], axis=1)
    work["_row"] = np.arange(len(work))
    work = work.sort_values(["horse_id", "race_date", "_row"], kind="stable").reset_index(drop=True)
    horse = work.groupby("horse_id", sort=False)
    code = horse.ngroup().to_numpy()
    same_day = (horse.race_date.shift(1) == work.race_date).to_numpy()
    made = {}
    for name in signals:
        for k in range(1, lags + 1):
            made[f"{prefix}_{name}_l{k}"] = horse[name].shift(k)
        made[f"{prefix}_{name}_ma50"] = moving_mean(code, work[name].to_numpy(float), MOVE_ALPHA)
    out = pd.DataFrame(made, index=work.index).astype("float32")
    out[same_day] = np.nan
    out = pd.concat([work[[*KEYS, "_row"]], out], axis=1).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    race = frame.race_id.to_numpy()
    for name in relative:
        for column in (f"{prefix}_{name}_l1", f"{prefix}_{name}_ma50"):
            out[f"{column}_vs_field"] = (out[column] - out[column].groupby(race).transform("mean")).astype("float32")
    return out


def price_moves(frame, prices):
    """Race-centred log move from each exchange snapshot to BSP; positive means the horse shortened.

    ``prices`` is row-aligned with ``frame`` and holds ``bf_bsp`` and the snapshot columns.
    """
    out = {}
    bsp = prices.bf_bsp.where(prices.bf_bsp > 1)
    for column in SNAPSHOTS:
        early = prices[column].where(prices[column] > 1)
        move = np.log(early / bsp).clip(-3, 3)
        out[column.replace("ltp_", "move_")] = (move - move.groupby(frame.race_id.to_numpy()).transform("mean")).to_numpy()
    return pd.DataFrame(out)


def market_moves(frame, prices):
    """Past late moves only: no price of the race being forecast is read."""
    return lagged(frame, price_moves(frame, prices), 3, "mh", relative=("move_1100", "move_3600s"))
