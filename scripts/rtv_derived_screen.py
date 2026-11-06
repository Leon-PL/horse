"""Research screen: features derivable from collected RTV data that the pipeline does not build.

Per-run measures are derived from the RTV tables (sectional traces, metrics,
cards), carried into history as the mean of the last three runs that had
matured two days before the race (as ``src.rtv_features`` does), and scored
on the race_profile evaluation frame in two harnesses:

* over the T-5 market (does it carry information the price lacks?), and
* over a market-free form proxy: the existing RTV history family, the store
  speed figures, official rating and Elo (does it add to what the model has?).

Each family is reported next to the same number of pure-noise columns.

Candidate families
  time_fig       per-runner race time from the sectional trace vs a conditions par
                 plus a leave-one-race-out meeting variant (speed figure in seconds)
  pace_adjusted  finish percentile minus the advantage its early position had in that
                 race's realised shape; and that advantage itself (form flattered by pace)
  sect_gaps      time behind the leader at 25% (early speed) and time made up from
                 75% to the line, both as a fraction of race time
  metric_rel     stride/frequency/acceleration measures normalised within race
                 (the stored family averages raw values, which carry trip and going)
  jump_rel       jumping measures normalised within race, incl. unused exit speed
                 and time in the jump envelope
  ground_action  within-race stride length and frequency x today's going
  timeform_card  Timeform rating, stars, tipped place, symbols, focus flag (current card)
  hint_x_style   Timeform pace forecast x the runner's prior early position
                 (the forecast is revised on race day; timing unverified)

Requires ``scripts/race_profile_screen.py`` and ``scripts/race_profile_fit.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from race_profile_screen import OUT, PACE_HINT, _read, going_number, load_races, load_runners, race_labels  # noqa: E402
from race_profile_fit import STORE, _cond_ll, _race_blocks, fit_conditional_logit  # noqa: E402
from triage_features import _partial_corr, market_controls  # noqa: E402
from src.rtv_feature_collection import observation_tables  # noqa: E402
from src.rtv_features import HISTORY_FEATURES, SECTIONAL_BAD  # noqa: E402
from src.source_pipeline import match_runners  # noqa: E402
from src.wf_eval import expanding_period_folds  # noqa: E402

PAR_END = "2024-06-30"
REL_METRICS = ("acceleration", "avg_length", "avg_frequency", "max_frequency", "stride_length",
               "exit_speed", "time_in_envelope", "speed_lost", "recovery_time", "exit_entry")


def _within_race_z(values: pd.Series, race_ids: pd.Series, min_n: int = 4) -> pd.Series:
    g = values.groupby(race_ids)
    z = (values - g.transform("mean")) / g.transform("std").clip(lower=1e-9)
    return z.where(g.transform("count").ge(min_n)).clip(-4, 4)


def sectional_measures(races: pd.DataFrame) -> pd.DataFrame:
    sect = _read("sectionals", ["race_id", "runner_id", "sector_number", "cumulative_sector_time", "time_behind_leader"])
    flags = _read("quality_flags", ["race_id", "runner_id", "check"])
    bad = flags.loc[flags.check.isin(SECTIONAL_BAD)]
    sect = sect.loc[~sect.race_id.isin(bad.loc[bad.runner_id.isna(), "race_id"])]
    bad_pairs = pd.MultiIndex.from_frame(bad.dropna(subset=["runner_id"])[["race_id", "runner_id"]].astype("int64"))
    sect = sect.loc[~pd.MultiIndex.from_frame(sect[["race_id", "runner_id"]]).isin(bad_pairs)]
    n = sect.groupby("race_id").sector_number.transform("max") + 1
    sect = sect.assign(leader_elapsed=sect.cumulative_sector_time - sect.time_behind_leader)

    def at(frac):
        idx = np.ceil(frac * n).astype(int) - 1
        cut = sect.loc[sect.sector_number.eq(idx)].set_index(["race_id", "runner_id"])
        return cut[~cut.index.duplicated()]

    s25, s75, end = at(0.25), at(0.75), at(1.0)
    out = pd.DataFrame(index=end.index)
    out["total_time"] = end.cumulative_sector_time
    out["early_gap"] = (s25.time_behind_leader / s25.leader_elapsed).reindex(out.index)
    out["closing_gain"] = ((s75.time_behind_leader.reindex(out.index) - end.time_behind_leader)
                           / end.leader_elapsed)
    return out.reset_index()


def time_figures(races: pd.DataFrame, runs: pd.DataFrame) -> pd.Series:
    """100 x (log par time + meeting variant - log own time): positive = faster."""
    x = races[["distance_yards", "going_num", "aw", "race_type", "track_name"]].copy()
    for c in ("race_type", "track_name"):
        x[c] = x[c].astype("category")
    fit = races.log_duration.notna() & races.race_date.le(PAR_END)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                              verbose=-1).fit(x[fit], races.log_duration[fit])
    par = pd.Series(model.predict(x), index=races.index)
    dev = races.log_duration - par
    meeting = [races.race_date, races.track_name, races.aw]
    total, count = dev.groupby(meeting).transform("sum"), dev.groupby(meeting).transform("count")
    loo = ((total - dev.fillna(0)) / (count - dev.notna().astype(int))).where(count - dev.notna() >= 2, 0.)
    expected = (par + loo).set_axis(races.race_id)
    return 100 * (runs.race_id.map(expected) - np.log(runs.total_time.where(runs.total_time > 0)))


def pace_adjustment(runs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Advantage of the runner's early position given the race's realised shape."""
    q = runs.quality
    e_bin = pd.cut(runs.early_pct, [-.01, .2, .4, .6, .8, 1.01], labels=False)
    fit = runs.race_date.le(PAR_END) & q.notna() & e_bin.notna() & runs.finish_shape.notna()
    s_edges = np.quantile(runs.loc[fit, "finish_shape"], [0, .2, .4, .6, .8, 1])
    s_edges[0], s_edges[-1] = -np.inf, np.inf
    s_bin = pd.cut(runs.finish_shape, s_edges, labels=False)
    cell = q[fit].groupby([e_bin[fit], s_bin[fit]]).mean()
    row = cell.groupby(level=0).mean()
    adv = cell - row.reindex(cell.index.get_level_values(0)).to_numpy()
    keys = pd.MultiIndex.from_arrays([e_bin, s_bin])
    advantage = pd.Series(adv.reindex(keys).to_numpy(), index=runs.index)
    return q - advantage, advantage


def per_run_table() -> pd.DataFrame:
    races = load_races()
    runners = load_runners(races)
    races = races.join(race_labels(runners), on="race_id")
    runners = runners.merge(races[["race_id", "finish_shape"]], on="race_id", how="left")
    field = runners.groupby("race_id").runner_id.transform("count")
    fin = runners.is_finisher.fillna(False).astype(bool)
    runners["quality"] = ((field - runners.pos) / np.maximum(field - 1, 1)).where(fin)
    runs = runners.merge(sectional_measures(races), on=["race_id", "runner_id"], how="left")
    runs["time_fig"] = time_figures(races, runs).where(fin)
    runs["pace_adj_q"], runs["pace_luck"] = pace_adjustment(runs)
    metrics = _read("metrics", ["race_id", "runner_id", "metric", "value"])
    metrics = metrics.loc[metrics.metric.isin({*REL_METRICS, "entry_speed"})] \
        .drop_duplicates(["race_id", "runner_id", "metric"])
    wide = metrics.pivot(index=["race_id", "runner_id"], columns="metric", values="value").reset_index()
    wide["exit_entry"] = wide.exit_speed / wide.entry_speed.where(wide.entry_speed > 0) \
        if "entry_speed" in wide else np.nan
    runs = runs.merge(wide, on=["race_id", "runner_id"], how="left")
    for m in REL_METRICS:
        if m in runs:
            runs[f"{m}_rel"] = _within_race_z(pd.to_numeric(runs[m], errors="coerce"), runs.race_id)
    return runs


def history(runs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean (and best for time_fig) of the last three runs matured before each race."""
    r = runs.dropna(subset=["horse_id"]).copy()
    r["race_date"] = pd.to_datetime(r.race_date)
    r = r.sort_values(["horse_id", "race_date", "race_id"])
    g = r.groupby("horse_id", sort=False)
    rolled = pd.DataFrame({"horse_id": r.horse_id, "available_date": r.race_date + pd.Timedelta(days=2)})
    means = g[cols].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    for c in cols:
        rolled[f"{c}_mean3"] = means[c]
    rolled["time_fig_best3"] = g.time_fig.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    rolled["time_fig_last"] = g.time_fig.ffill()
    left = r[["race_id", "runner_id", "horse_id", "race_date"]].sort_values("race_date")
    out = pd.merge_asof(left, rolled.sort_values("available_date"), left_on="race_date",
                        right_on="available_date", by="horse_id", direction="backward")
    return out.drop(columns=["available_date"])


def card_fields() -> pd.DataFrame:
    c = _read("runners", ["race_id", "runner_id", "timeform_rating", "timeform_rating_symbol",
                          "card_timeform_star_rating", "card_timeform_tipped_place", "card_focus_horse"])
    sym = c.timeform_rating_symbol.astype("string").fillna("")
    focus = c.card_focus_horse.astype("string").str.lower().eq("true")
    return pd.DataFrame({"race_id": c.race_id, "runner_id": c.runner_id,
        "tf_rating": pd.to_numeric(c.timeform_rating, errors="coerce").where(lambda s: s > 0),
        "tf_stars": pd.to_numeric(c.card_timeform_star_rating, errors="coerce"),
        "tf_tipped": pd.to_numeric(c.card_timeform_tipped_place, errors="coerce").map({1: 3., 2: 2., 3: 1.}).fillna(0.),
        "tf_improver": sym.str.contains("p", case=False).astype(float),
        "tf_doubtful": sym.str.contains(r"\?").astype(float), "tf_focus": focus.astype(float)})


def attach(ev: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    start, end = str(ev.race_date.min().date()), str(ev.race_date.max().date())
    tables = observation_tables(start, end)
    links = match_runners(tables["races"], tables["runners"],
                          ev[["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]])
    links = links.rename(columns={"sl_race_id": "race_id", "sl_horse_id": "horse_id"})
    rel = [f"{m}_rel" for m in REL_METRICS if f"{m}_rel" in runs]
    hist = history(runs, ["time_fig", "pace_adj_q", "pace_luck", "early_gap", "closing_gain", *rel])
    hist = hist.rename(columns={"race_id": "rtv_race_id", "runner_id": "rtv_runner_id"}).drop(columns="horse_id")
    cards = card_fields().rename(columns={"race_id": "rtv_race_id", "runner_id": "rtv_runner_id"})
    hint = load_races().set_index("race_id").pace_hint.rename("pace_hint")
    links = links.merge(hist.drop(columns="race_date"), on=["rtv_race_id", "rtv_runner_id"], how="left") \
                 .merge(cards, on=["rtv_race_id", "rtv_runner_id"], how="left")
    links["pace_hint"] = links.rtv_race_id.map(hint)
    keep = [c for c in links if c not in ("horse_name", "race_date", "rtv_race_id", "rtv_runner_id")]
    return ev.merge(links[keep].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")


def _design(ev, cols, market):
    parts = [np.log(ev.mkt_imp.to_numpy())] if market else []
    for c in cols:
        v = pd.to_numeric(ev[c], errors="coerce").astype(float)
        v = (v - v.groupby(ev.race_id).transform("mean")).fillna(0.).to_numpy()
        sd = v.std()
        parts.append(v / sd if sd > 0 else v)
    return np.column_stack(parts)


def gain(ev, base, extra, folds, market) -> np.ndarray:
    won, codes = ev.won.to_numpy(float), ev.race_code.to_numpy()
    xb, xc = _design(ev, base, market), _design(ev, base + extra, market)
    out = []
    for fold in folds:
        tr, te = fold.train, fold.test
        st_tr, st_te = _race_blocks(codes[tr]), _race_blocks(codes[te])
        tb, tc = fit_conditional_logit(xb[tr], won[tr], st_tr), fit_conditional_logit(xc[tr], won[tr], st_tr)
        out.append(_cond_ll(tc, xc[te], won[te], st_te)[0] - _cond_ll(tb, xb[te], won[te], st_te)[0])
    return np.concatenate(out)


def summarise(d: np.ndarray, noise: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    diff = d - noise
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)])
    return {"gain": d.mean(), "noise": noise.mean(), "excess": diff.mean(),
            "excess_lo": np.quantile(boots, .025), "excess_hi": np.quantile(boots, .975)}


def main() -> None:
    runs = per_run_table()
    runs.to_parquet(OUT / "rtv_runs.parquet", index=False)
    ev = pd.read_parquet(OUT / "evaluation_frame.parquet")
    store_cols = ["race_id", "horse_name", *HISTORY_FEATURES, "speed_fig_avg_5", "speed_fig_best_5",
                  "speed_fig_adj_avg_5", "speed_fig_adj_best_5", "official_rating", "horse_elo"]
    have = pq.read_schema(STORE / "featured.parquet").names
    store = pd.read_parquet(STORE / "featured.parquet", columns=[c for c in store_cols if c in have],
                            filters=[("race_date", ">=", ev.race_date.min())])
    ev = ev.drop(columns=[c for c in store if c in ev and c not in ("race_id", "horse_name")])
    ev = ev.merge(store.drop_duplicates(["race_id", "horse_name"]), on=["race_id", "horse_name"], how="left")
    ev = attach(ev, runs).sort_values(["race_date", "race_id"]).reset_index(drop=True)
    ev["race_code"] = pd.factorize(ev.race_id)[0]
    gdev = ev.going.map(going_number) - 3.0
    ev["ground_length"] = ev.avg_length_rel_mean3 * gdev
    ev["ground_frequency"] = ev.avg_frequency_rel_mean3 * gdev
    early = ev.rtv_early_position_mean3 - ev.groupby("race_id").rtv_early_position_mean3.transform("mean")
    ev["hint_x_style"] = early * (ev.pace_hint - 2.5)

    families = {
        "time_fig": ["time_fig_mean3", "time_fig_best3", "time_fig_last"],
        "pace_adjusted": ["pace_adj_q_mean3", "pace_luck_mean3"],
        "sect_gaps": ["early_gap_mean3", "closing_gain_mean3"],
        "metric_rel": [f"{m}_rel_mean3" for m in ("acceleration", "avg_length", "avg_frequency",
                                                   "max_frequency", "stride_length")],
        "jump_rel": [f"{m}_rel_mean3" for m in ("exit_speed", "time_in_envelope", "speed_lost",
                                                 "recovery_time", "exit_entry")],
        "ground_action": ["ground_length", "ground_frequency"],
        "timeform_card": ["tf_rating", "tf_stars", "tf_tipped", "tf_improver", "tf_doubtful", "tf_focus"],
        "hint_x_style": ["hint_x_style"],
    }
    families = {k: [c for c in v if c in ev] for k, v in families.items()}
    form_base = [c for c in [*HISTORY_FEATURES[:-4], "speed_fig_avg_5", "speed_fig_best_5",
                             "speed_fig_adj_avg_5", "speed_fig_adj_best_5", "official_rating", "horse_elo"] if c in ev]
    print(f"{ev.race_id.nunique():,} races, {len(ev):,} runners; form proxy has {len(form_base)} inputs")
    print("coverage:", {c: round(float(ev[c].notna().mean()), 3) for v in families.values() for c in v})

    codes = ev.race_code.to_numpy()
    resid = ev.won.to_numpy(float) - ev.mkt_imp.to_numpy()
    resid = resid - pd.Series(resid).groupby(codes).transform("mean").to_numpy()
    controls = market_controls(ev.mkt_imp.to_numpy(), codes)
    rows = []
    for fam, cols in families.items():
        for c in cols:
            v = pd.to_numeric(ev[c], errors="coerce").to_numpy(float)
            v = v - pd.Series(v).groupby(codes).transform("mean").to_numpy()
            _, partial, n = _partial_corr(v, resid, controls)
            rows.append({"family": fam, "feature": c, "partial": partial})
    print("\nMarket-residual partial correlation:\n" + pd.DataFrame(rows).round(4).to_string(index=False))

    folds = expanding_period_folds(ev.race_date, freq="Q", min_train_periods=2, embargo_days=0)
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(size=(len(ev), 8)), columns=[f"_n{i}" for i in range(8)])
    ev = pd.concat([ev, noise], axis=1)
    results = []
    for label, base, market in (("over market", [], True), ("over form proxy", form_base, False)):
        noise_gain = {k: gain(ev, base, list(noise)[:k], folds, market) for k in {len(v) for v in families.values()}}
        for fam, cols in families.items():
            results.append({"harness": label, "family": fam, "k": len(cols),
                            **summarise(gain(ev, base, cols, folds, market), noise_gain[len(cols)])})
    res = pd.DataFrame(results)
    res.to_csv(OUT / "rtv_derived_scores.csv", index=False)
    print("\nOut-of-sample race log-likelihood gain per race; excess = gain minus k noise columns "
          "(95% race-bootstrap interval):")
    print(res.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))


if __name__ == "__main__":
    main()
