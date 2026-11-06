"""Research test: does a runner's fit to the *projected* race profile add information?

Builds runner x race-profile features from strictly earlier runs and scores
them, alongside the existing families they would extend, in one harness:

* market residual partial correlation (``scripts/triage_features.py``), and
* out-of-sample race log loss of a conditional logit on the T-5 market price
  plus the family, against the market alone (quarterly expanding folds,
  race-resampled bootstrap interval).

Families
  trip_furlong   the existing trip-suitability kernel on nominal furlongs (control)
  trip_par       the same kernel on effective duration: history = realised winning
                 time (RTV) or a conditions par, query = out-of-fold par projection
  trip_proj      as trip_par, query = full out-of-fold duration projection
  stamina_reach  projected duration vs the longest duration the horse has run well over
  shape_fit      the kernel on finishing shape: history = realised race FSP,
                 query = out-of-fold projected FSP
  style_x_shape  prior early position (RTV sectionals) x projected finishing shape
  pace_store     the 009 pace family as it sits in the feature store (re-measured)

Requires ``scripts/race_profile_screen.py`` to have been run first.
Development period only; evaluation from 2024-07-01 so the par never sees it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
from race_profile_screen import OUT, going_number, load_races, load_runners  # noqa: E402
from triage_features import _partial_corr, market_controls  # noqa: E402
from src.betfair import attach_bsp  # noqa: E402
from src.established_form_features import _horse_keys, mark_scale  # noqa: E402
from src.forecast_evaluation import valid_race_mask  # noqa: E402
from src.result_status import NONSTARTER_LABELS, result_labels  # noqa: E402
from src.ingestion.rtv_observations import observation_tables  # noqa: E402
from src.ingestion.pipeline import match_runners  # noqa: E402
from src.trip_suitability_research import trip_suitability_features  # noqa: E402
from src.wf_eval import development_mask, expanding_period_folds  # noqa: E402

STORE = Path(config.DATA_DIR) / "processed" / "feature_store"
EVAL_START, PAR_END = "2024-07-01", "2024-06-30"
SEC_PER_UNIT = 12.0  # durations are carried in ~furlong-sized units so 4..50 bounds still apply
PACE_STORE = ["pace_style_index", "pace_finish_bias", "pace_style_balance", "pace_style_trend",
    "pace_style_vs_field", "pace_finish_bias_vs_field", "pace_front_runner_flag", "pace_closer_flag",
    "pace_rank_pct", "pace_gap_to_leader", "pace_gap_to_slowest", "pace_style_x_sprint",
    "pace_x_course_leaders", "field_pace_pressure", "field_pace_pressure_share", "field_closer_share",
    "front_runner_vulnerability", "front_runner_pressure", "frontrunner_at_bias_track",
    "closer_setup_intensity", "closer_pace_setup", "lone_speed_flag", "rtv_early_position_mean3"]


def load_frame() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "horse_name", "race_date", "off_time", "track", "race_type", "race_name",
            "surface", "distance_furlongs", "going", "finish_position", "finish_pos_label", "num_runners", "won"]
    f = pd.read_parquet(STORE / "processed.parquet", columns=cols)
    f["race_date"] = pd.to_datetime(f.race_date)
    f = f.loc[development_mask(f.race_date)].reset_index(drop=True)
    return f


def attach_rtv(f: pd.DataFrame) -> pd.DataFrame:
    """RTV race profile (realised + out-of-fold projections) and runner early position."""
    start, end = "2023-06-01", str(f.race_date.max().date())
    tables = observation_tables(start, end)
    canon = f.loc[f.race_date.ge(start), ["race_id", "horse_id", "horse_name", "race_date", "track", "off_time"]]
    links = match_runners(tables["races"], tables["runners"], canon)
    races = load_races()
    runners = load_runners(races)
    links = links.merge(runners[["race_id", "runner_id", "early_pct"]],
                        left_on=["rtv_race_id", "rtv_runner_id"], right_on=["race_id", "runner_id"], how="left")
    race_map = links.groupby("sl_race_id").rtv_race_id.first()
    profiles = pd.read_parquet(OUT / "race_profiles.parquet")
    keep = ["race_id", "win_time_s", "finish_shape", "proj_log_duration_L1", "proj_log_duration_L3",
            "proj_finish_shape_L3"]
    prof = profiles[keep].rename(columns={"race_id": "rtv_race_id"})
    out = f.copy()
    out["rtv_race_id"] = out.race_id.map(race_map)
    out = out.merge(prof, on="rtv_race_id", how="left")
    # Races outside the screen (e.g. before its first fold) still carry realised values.
    realised = races.set_index("race_id").win_time_s
    out["win_time_s"] = out.win_time_s.fillna(out.rtv_race_id.map(realised))
    early = links.set_index(["sl_race_id", "sl_horse_id"]).early_pct
    early = early[~early.index.duplicated()]
    idx = pd.MultiIndex.from_arrays([out.race_id, out.horse_id])
    out["early_pct"] = early.reindex(idx).to_numpy()
    return out


def add_par(f: pd.DataFrame) -> pd.Series:
    """Conditions par (log seconds) fitted only on RTV races up to PAR_END."""
    x = pd.DataFrame({"distance_furlongs": f.distance_furlongs, "going_num": f.going.map(going_number),
                      "race_type": f.race_type.astype("category"), "track": f.track.astype("category"),
                      "surface": f.surface.astype("category")})
    y = np.log(f.win_time_s)
    speed = f.distance_furlongs * 220 / f.win_time_s
    fit = y.notna() & speed.between(10, 20) & f.race_date.le(PAR_END)
    first = ~f.race_id.duplicated()
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                              verbose=-1).fit(x[fit & first], y[fit & first])
    return pd.Series(model.predict(x), index=f.index)


def kernel_features(f: pd.DataFrame, axes: dict) -> pd.DataFrame:
    """The trip-suitability mechanics, generalised to any history axis.

    axes: name -> (query, stored, log_axis, bandwidth). History is per horse and
    discipline, 20 starts / 730 days, 5 pseudo-starts at 0.5, outcomes of the
    current date excluded (rows are appended only after their whole date).
    Also returns stamina reach and prior early position.
    """
    days = f.race_date.to_numpy(dtype="datetime64[D]").astype(np.int64)
    keys, scales = _horse_keys(f), mark_scale(f).to_numpy(dtype=object, na_value=None)
    finish = pd.to_numeric(f.finish_position, errors="coerce").to_numpy(float)
    size = pd.to_numeric(f.num_runners, errors="coerce").to_numpy(float)
    labels = result_labels(f)
    starts = (finish > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS)).to_numpy()
    quality = np.where(finish > 0, (size - finish) / np.maximum(size - 1, 1), 0.)
    quality = np.where(valid_race_mask(f).to_numpy() & starts, quality, np.nan)
    quality = np.where((quality >= 0) & (quality <= 1), quality, np.nan)
    early = f.early_pct.to_numpy(float)
    names = list(axes)
    q = np.column_stack([np.asarray(axes[n][0], float) for n in names])
    s = np.column_stack([np.asarray(axes[n][1], float) for n in names])
    dur_q, dur_s = np.asarray(axes["trip_proj"][0], float), np.asarray(axes["trip_proj"][1], float)
    out = {f"{n}_{k}": np.full(len(f), np.nan) for n in names for k in ("quality", "edge", "support")}
    out.update({"stamina_reach_good": np.full(len(f), np.nan), "stamina_reach_any": np.full(len(f), np.nan),
                "style_early_prior": np.full(len(f), np.nan)})
    history: dict = {}
    order = np.lexsort((f.race_id.astype(str).to_numpy(), days))
    bounds = np.r_[0, np.flatnonzero(np.diff(days[order])) + 1, len(f)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        rows = order[lo:hi]
        for i in rows:
            if keys[i] is None or scales[i] is None:
                continue
            for j, n in enumerate(names):
                if np.isfinite(q[i, j]):
                    out[f"{n}_support"][i] = 0.
            past = [h for h in history.get((keys[i], scales[i]), []) if days[i] - h[0] <= 730]
            if not past:
                continue
            arr = np.asarray([h[1:] for h in past], float)  # quality, early, stored axes...
            qs, es, stored = arr[:, 0], arr[:, 1], arr[:, 2:]
            for j, n in enumerate(names):
                _, _, log_axis, bw = axes[n]
                ok = np.isfinite(stored[:, j]) & np.isfinite(qs)
                if not np.isfinite(q[i, j]) or not ok.any():
                    continue
                d = np.log(stored[ok, j] / q[i, j]) / np.log(bw) if log_axis else (stored[ok, j] - q[i, j]) / bw
                w, qq = np.exp(-.5 * d ** 2), qs[ok]
                out[f"{n}_quality"][i] = (np.dot(w, qq) + 2.5) / (w.sum() + 5.)
                out[f"{n}_edge"][i] = np.dot(w, qq - qq.mean()) / (w.sum() + 5.)
                out[f"{n}_support"][i] = np.log1p(w.sum())
            run = np.where(np.isfinite(qs), stored[:, names.index("trip_proj")], np.nan)
            if np.isfinite(dur_q[i]) and np.isfinite(run).any():
                good = np.isfinite(run) & (qs >= 0.5)
                out["stamina_reach_any"][i] = np.log(dur_q[i] / np.nanmax(run))
                if good.any():
                    out["stamina_reach_good"][i] = np.log(dur_q[i] / run[good].max())
            known = es[np.isfinite(es)]
            if known.size:
                out["style_early_prior"][i] = known[-5:].mean()
        for i in rows:
            # Mirrors trip_suitability_features: every start is stored; NaN quality is filtered at use.
            if starts[i] and keys[i] is not None and scales[i] is not None:
                key = (keys[i], scales[i])
                history[key] = (history.get(key, []) + [(int(days[i]), float(quality[i]), float(early[i]),
                                                          *map(float, s[i]))])[-20:]
    res = pd.DataFrame(out, index=f.index)
    for c in [c for c in res if c.endswith(("_quality", "_edge"))]:
        res[c] -= res[c].groupby(f.race_id, sort=False).transform("mean")
    return res


# ---------------------------------------------------------------- evaluation

def _race_blocks(race_codes: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(np.diff(race_codes)) + 1]


def _cond_ll(theta, x, won, starts):
    s = x @ theta
    m = np.maximum.reduceat(s, starts)
    e = np.exp(s - np.repeat(m, np.diff(np.r_[starts, len(s)])))
    z = np.add.reduceat(e, starts)
    lse = m + np.log(z)
    p = e / np.repeat(z, np.diff(np.r_[starts, len(s)]))
    ll = np.add.reduceat(s * won, starts) - lse
    grad = x.T @ (won - p)
    return ll, grad


def fit_conditional_logit(x, won, starts):
    def fun(theta):
        ll, grad = _cond_ll(theta, x, won, starts)
        return -ll.sum(), -grad
    theta0 = np.zeros(x.shape[1])
    theta0[0] = 1.0
    return minimize(fun, theta0, jac=True, method="L-BFGS-B").x


def design(ev: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """log market + within-race demeaned, standardised candidates (NaN -> race mean)."""
    parts = [np.log(ev.mkt_imp.to_numpy())]
    for c in cols:
        v = pd.to_numeric(ev[c], errors="coerce").astype(float)
        v = (v - v.groupby(ev.race_id).transform("mean")).fillna(0.).to_numpy()
        sd = v.std()
        parts.append(v / sd if sd > 0 else v)
    return np.column_stack(parts)


def score_family(ev: pd.DataFrame, cols: list[str], folds) -> dict:
    won = ev.won.to_numpy(float)
    x_base, x_cand = design(ev, []), design(ev, cols)
    codes = ev.race_code.to_numpy()
    deltas, n_test = [], 0
    for fold in folds:
        tr, te = fold.train, fold.test
        if te.sum() == 0:
            continue
        st_tr, st_te = _race_blocks(codes[tr]), _race_blocks(codes[te])
        tb = fit_conditional_logit(x_base[tr], won[tr], st_tr)
        tc = fit_conditional_logit(x_cand[tr], won[tr], st_tr)
        ll_b, _ = _cond_ll(tb, x_base[te], won[te], st_te)
        ll_c, _ = _cond_ll(tc, x_cand[te], won[te], st_te)
        deltas.append(ll_c - ll_b)  # log-likelihood gain: positive = candidate lowers log loss
        n_test += len(st_te)
    d = np.concatenate(deltas)
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    return {"races": n_test, "ll_gain_per_race": d.mean(), "ci_lo": np.quantile(boots, .025),
            "ci_hi": np.quantile(boots, .975)}


def noise_baseline(ev: pd.DataFrame, k: int, folds, seed: int = 1) -> dict:
    """The same score for k pure-noise columns: what overfitting alone costs."""
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(rng.normal(size=(len(ev), k)), columns=[f"_noise{i}" for i in range(k)], index=ev.index)
    return score_family(pd.concat([ev, noise], axis=1), list(noise), folds)


def main() -> None:
    t0 = time.monotonic()
    f = load_frame()
    f = attach_rtv(f)
    f["par_log_duration"] = add_par(f)
    unit = lambda logs: np.exp(logs) / SEC_PER_UNIT  # noqa: E731
    realised = f.win_time_s.where((f.distance_furlongs * 220 / f.win_time_s).between(10, 20))
    stored_dur = (realised / SEC_PER_UNIT).fillna(unit(f.par_log_duration))
    q_par = unit(f.proj_log_duration_L1)
    q_proj = unit(f.proj_log_duration_L3)
    furl = f.distance_furlongs.where(f.distance_furlongs.between(4, 50))
    axes = {"trip_furlong": (furl, furl, True, 1.15),
            "trip_par": (q_par, stored_dur, True, 1.15),
            "trip_proj": (q_proj, stored_dur, True, 1.15),
            "shape_fit": (f.proj_finish_shape_L3, f.finish_shape, False, 2.5)}
    feats = kernel_features(f, axes)
    # The furlong axis must reproduce the existing research function exactly.
    ref = trip_suitability_features(f)
    for a, b in zip(("quality", "edge", "support"),
                    ("horse_trip_quality_vs_field", "horse_trip_specialist_edge_vs_field", "horse_trip_support_log")):
        gap = (feats[f"trip_furlong_{a}"] - ref[b]).abs().max()
        print(f"control check trip_furlong_{a} vs {b}: max abs diff {gap:.2e}")
    f = pd.concat([f, feats], axis=1)
    shape_dev = f.proj_finish_shape_L3 - f.proj_finish_shape_L3.mean()
    early_vs_field = f.style_early_prior - f.style_early_prior.groupby(f.race_id).transform("mean")
    f["style_x_shape"] = early_vs_field * shape_dev
    f.to_parquet(OUT / "runner_features.parquet", index=False)
    ev, store_cols = evaluation_frame(f)
    ev.to_parquet(OUT / "evaluation_frame.parquet", index=False)
    print(f"\nevaluation: {ev.race_id.nunique():,} races, {len(ev):,} runners, "
          f"{ev.race_date.min().date()}..{ev.race_date.max().date()}")
    score_all(ev, store_cols)
    print(f"\nelapsed {time.monotonic() - t0:.0f}s")


def evaluation_frame(f: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Complete-field starters from EVAL_START with a T-5 market and store pace columns."""
    ev = f.loc[f.race_date.ge(EVAL_START) & f.proj_finish_shape_L3.notna()].copy()
    ev = ev.loc[valid_race_mask(ev).to_numpy()]
    labels = result_labels(ev)
    starter = (pd.to_numeric(ev.finish_position, errors="coerce") > 0) | (labels.ne("") & ~labels.isin(NONSTARTER_LABELS))
    ev = ev.loc[starter.to_numpy()]
    ev = attach_bsp(ev, cols=("ltp_300s",))
    price = pd.to_numeric(ev.ltp_300s, errors="coerce")
    complete = (price > 1).groupby(ev.race_id).transform("all")
    ev = ev.loc[complete & ev.groupby("race_id").won.transform("sum").eq(1)].copy()
    implied = 1 / ev.ltp_300s
    ev["mkt_imp"] = implied / implied.groupby(ev.race_id).transform("sum")
    store_cols = [c for c in PACE_STORE if c in pq.read_schema(STORE / "featured.parquet").names]
    store = pd.read_parquet(STORE / "featured.parquet", columns=["race_id", "horse_name", *store_cols],
                            filters=[("race_date", ">=", pd.Timestamp(EVAL_START))])
    ev = ev.merge(store.drop_duplicates(["race_id", "horse_name"]), on=["race_id", "horse_name"], how="left")
    ev = ev.sort_values(["race_date", "race_id"]).reset_index(drop=True)
    ev["race_code"] = pd.factorize(ev.race_id)[0]
    return ev, store_cols


def score_all(ev: pd.DataFrame, store_cols: list[str]) -> None:
    families = {
        "trip_furlong": [f"trip_furlong_{k}" for k in ("quality", "edge", "support")],
        "trip_par": [f"trip_par_{k}" for k in ("quality", "edge", "support")],
        "trip_proj": [f"trip_proj_{k}" for k in ("quality", "edge", "support")],
        "stamina_reach": ["stamina_reach_good", "stamina_reach_any"],
        "shape_fit": [f"shape_fit_{k}" for k in ("quality", "edge", "support")],
        "style_x_shape": ["style_early_prior", "style_x_shape"],
        "pace_store": store_cols,
    }
    families["new_all"] = [c for n in ("trip_proj", "stamina_reach", "shape_fit", "style_x_shape")
                           for c in families[n]]

    codes = ev.race_code.to_numpy()
    resid = ev.won.to_numpy(float) - ev.mkt_imp.to_numpy()
    controls = market_controls(ev.mkt_imp.to_numpy(), codes)
    rows = []
    for fam, cols in families.items():
        for c in cols:
            v = pd.to_numeric(ev[c], errors="coerce").to_numpy(float)
            vd = v - pd.Series(v).groupby(codes).transform("mean").to_numpy()
            raw, partial, n = _partial_corr(vd, resid - pd.Series(resid).groupby(codes).transform("mean").to_numpy(), controls)
            rows.append({"family": fam, "feature": c, "coverage": np.isfinite(v).mean(), "partial": partial,
                         "t": partial * np.sqrt((n - 2) / max(1e-12, 1 - partial ** 2)) if np.isfinite(partial) else np.nan})
    triage = pd.DataFrame(rows)
    print("\nMarket-residual partial correlation (within race, price controlled):")
    print(triage.round(4).to_string(index=False))

    folds = expanding_period_folds(ev.race_date, freq="Q", min_train_periods=2, embargo_days=0)
    scores = pd.DataFrame({fam: score_family(ev, cols, folds) for fam, cols in families.items()}).T
    scores["k"] = [len(families[f]) for f in scores.index]
    scores["noise_same_k"] = [noise_baseline(ev, k, folds)["ll_gain_per_race"] for k in scores.k]
    print("\nConditional logit on market + family, out-of-sample race log-likelihood gain per race "
          "(positive = better than market alone; noise_same_k = k pure-noise columns):")
    print(scores.to_string(float_format=lambda v: f"{v:.5f}"))
    triage.to_csv(OUT / "fit_triage.csv", index=False)
    scores.to_csv(OUT / "fit_scores.csv")


if __name__ == "__main__":
    main()
