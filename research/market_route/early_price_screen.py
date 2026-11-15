"""Does the 11:00 exchange price plus the fundamentals forecast predict the close?

    python -m research.market_route.early_price_screen [--threads 6]

The market-anchored route: take the 11:00 Betfair price as given and learn only
where the price will close. Target, per runner, ``log(P / BSP)`` where ``P`` is
the price a back bet at 11:00 would settle at (one tick below the last traded
price, cut by every non-runner reduction). Positive means the bet beats the close.

Three ways to rank runners at 11:00 are compared on the same universe:

- ``market``: LightGBM on the 11:00 book alone (ledger 20's frame, no morning volume).
- ``market+model``: the same plus the stored out-of-fold fundamentals forecast.
- ``gap``: no fit, the forecast's raw disagreement with the 11:00 price.

Exploratory, development data only (stored forecasts 2022Q3-2026Q2). The
forecast's inputs know the final field, so the primary universe is races with
no withdrawal after 11:00; the all-races universe is reported as a flagged
secondary. Going and jockey inputs are final values, not 11:00 values.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.error_anatomy.frame import OUT as ANATOMY
from research.meta_learner import screen_candidates as sc
from research.meta_learner import tier2_inputs as t2
from src import wf_eval
from src.bet_settlement import (EXCHANGE_COMMISSION, apply_reduction_factor, betfair_tick_down,
                                settle_win_exchange_bets)
from src.clv import summarise_clv
from src.ingestion.storage import atomic_json, atomic_parquet
from src.source_dataset import load_manifest, read_table, use_manifest
from src.source_prices import attach_bsp

OUT = Path("data/research/market_route_20261111")
BAND = (1.5, 50.0)
TOP = 0.10
N_BOOT = 1000
CUTS = {"1100": 660, "0800": 480}  # decision time, minutes after midnight UK
PER_CUT = {"ltp": "ltp_cut", "rf_after": "rf_after", "rf_stale": "rf_stale", "age": "age", "nchg": "nchg",
           "prev_ltp": "prev_ltp", "book_removed_after": "book_removed", "n_removed_after": "n_removed"}
SNAPSHOTS = ("ltp_eve1800", "ltp_0800", "ltp_1100")
FIRST_OFF = 725  # races off at or after 12:05 UK, whichever decision time is scored
MARKET = ["l", "l2", "inv", "log_p_mkt", "over", "rank", "fav", "field", "log_age", "log_nchg", "log_rf_stale",
          "last_move", "move_0800", "move_eve", "mins_to_off", "race_kind", "handicap_flag", "surface_kind"]
MODEL = ["log_p_model", "gap", "gap_centred", "model_rank", "gap_rank"]
PARAMS = {"objective": "regression", "learning_rate": .03, "num_leaves": 31, "min_data_in_leaf": 200,
          "feature_fraction": .8, "bagging_fraction": .8, "bagging_freq": 1, "lambda_l2": 10., "verbose": -1}
ROUNDS = 400


def load(cut="1100"):
    """Runners with a stored out-of-fold forecast, race context and the exchange fields at *cut*."""
    early = pd.read_parquet(ANATOMY / "early_oof_seed42.parquet").rename(columns={"probability": "p_model"})
    late = pd.read_parquet(ANATOMY / "oof_all.parquet", columns=["race_id", "horse_id", "race_date", "p_single"])
    frame = pd.concat([early[["race_id", "horse_id", "race_date", "p_model"]],
                       late.rename(columns={"p_single": "p_model"})], ignore_index=True)
    frame["race_id"] = frame.race_id.astype(str)
    start = frame.race_date.min()
    labels = sc.read_input(["num_runners", "won"], start)
    labels["race_id"] = labels.race_id.astype(str)
    context = sc.read_store(["off_time", "race_type", "handicap", "surface"], start=start).drop(columns="race_date")
    frame = frame.merge(labels, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    frame = frame.merge(context, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    receipt = t2.design()["receipt"]["dataset_manifest"]
    with use_manifest(load_manifest(receipt["id"], root=receipt.get("root"))):
        timing = read_table("betfair", "timing", linked=True).rename(columns={"bsp": "bf_bsp"})
    frame = attach_bsp(frame, timing, cols=("bf_bsp", "off_ms", *SNAPSHOTS,
                                            *[f"{k}_{cut}" for k in PER_CUT if k != "ltp"]))
    frame = frame.rename(columns={f"{k}_{cut}": v for k, v in PER_CUT.items() if k != "ltp"})
    frame["ltp_cut"] = frame[f"ltp_{cut}"]
    # Only prices struck before the decision time may be inputs.
    for column in SNAPSHOTS[SNAPSHOTS.index(f"ltp_{cut}"):]:
        frame[column] = np.nan
    frame["cut_minute"] = CUTS[cut]
    frame = frame.loc[wf_eval.development_mask(frame.race_date)]
    return frame.sort_values(["race_date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)


def prepare(frame):
    """Inputs known at 11:00, the settled strike price and the target. Rows are whole races."""
    race = frame.race_id
    ltp = frame.ltp_cut.where(frame.ltp_cut > 1)
    bsp = frame.bf_bsp.where((frame.bf_bsp > 1) & np.isfinite(frame.bf_bsp))
    complete = (ltp.notna() & bsp.notna() & frame.p_model.gt(0)).groupby(race).transform("all")
    off = pd.to_datetime(frame.off_ms, unit="ms", utc=True).dt.tz_convert("Europe/London")
    minute = off.dt.hour * 60 + off.dt.minute
    frame = frame.assign(mins_to_off=minute - frame.cut_minute)
    frame = frame.loc[complete & minute.ge(FIRST_OFF)].reset_index(drop=True)
    race, ltp = frame.race_id, frame.ltp_cut
    inv = 1 / ltp
    over = inv.groupby(race).transform("sum") + frame.book_removed.groupby(race).transform("max")
    p_model = frame.p_model / frame.p_model.groupby(race).transform("sum")
    made = {
        "l": np.log(ltp), "l2": np.log(ltp) ** 2, "inv": inv, "log_p_mkt": np.log(inv / over), "over": over,
        "rank": ltp.groupby(race).rank(method="min"),
        "field": frame.num_runners + frame.n_removed.groupby(race).transform("max"),
        "log_age": np.log1p(frame.age), "log_nchg": np.log1p(frame.nchg),
        "log_rf_stale": np.log(frame.rf_stale),
        "last_move": np.log(frame.prev_ltp.where(frame.prev_ltp > 1) / ltp),
        "move_0800": np.log(frame.ltp_0800.where(frame.ltp_0800 > 1) / ltp),
        "move_eve": np.log(frame.ltp_eve1800.where(frame.ltp_eve1800 > 1) / ltp),
        "race_kind": frame.race_type.astype("category").cat.codes,
        "handicap_flag": pd.to_numeric(frame.handicap, errors="coerce"),
        "surface_kind": frame.surface.astype("category").cat.codes,
        "log_p_model": np.log(p_model),
    }
    made["fav"] = (made["rank"] == 1).astype(float)
    made["gap"] = made["log_p_model"] - made["log_p_mkt"]
    made["gap_centred"] = made["gap"] - made["gap"].groupby(race).transform("mean")
    made["model_rank"] = p_model.groupby(race).rank(ascending=False, method="min")
    made["gap_rank"] = made["gap"].groupby(race).rank(ascending=False, method="min")
    frame = frame.assign(**made)
    # The price to rank on is known at 11:00; the settled price also takes later reductions.
    quoted = betfair_tick_down(ltp.to_numpy(), 1) * frame.rf_stale.to_numpy()
    for ticks in (1, 2, 3):
        taken = apply_reduction_factor(betfair_tick_down(ltp.to_numpy(), ticks) * frame.rf_stale.to_numpy(),
                                       frame.rf_after.to_numpy())
        frame[f"taken_{ticks}"] = taken
    frame["quoted"] = quoted
    frame["y"] = np.log(frame.taken_1 / frame.bf_bsp).clip(-1.5, 1.5)
    frame["in_band"] = ltp.between(*BAND)
    frame["no_withdrawal"] = frame.n_removed.groupby(race).transform("max").eq(0)
    frame["quarter"] = frame.race_date.dt.to_period("Q").astype(str)
    return frame


def fit_predict(columns, threads, seed=7):
    def run(train, test):
        train = train.loc[train.in_band]
        model = lgb.train({**PARAMS, "num_threads": threads, "seed": seed},
                          lgb.Dataset(train[columns], train.y), ROUNDS)
        return model.predict(test[columns])
    return run


def bets(sub):
    """How a set of 11:00 strikes fared against the close, and as bets."""
    if not len(sub):
        return {"n": 0}
    clv = (sub.taken_1 / sub.bf_bsp).to_numpy()
    won = sub.won.to_numpy()
    summary = summarise_clv(clv)
    median = wf_eval.bootstrap_median_ci(clv, clusters=sub.race_id, n_boot=N_BOOT)
    roi = wf_eval.bootstrap_roi_ci(settle_win_exchange_bets(sub.taken_1, won), clusters=sub.race_id, n_boot=N_BOOT)
    at_bsp = wf_eval.bootstrap_roi_ci(settle_win_exchange_bets(sub.bf_bsp, won), clusters=sub.race_id, n_boot=N_BOOT)
    fall = sub.loc[sub.last_move > 0]
    return {"n": len(sub), "races": int(sub.race_id.nunique()),
            "median_clv": summary["median_clv"], "median_ci": [median["ci_low"], median["ci_high"]],
            "geometric_clv": summary["geometric_mean_clv"], "beat_close": summary["pct_beat_close"],
            "roi_taken": roi["mean"], "roi_taken_ci": [roi["ci_low"], roi["ci_high"]],
            "roi_bsp": at_bsp["mean"], "roi_bsp_ci": [at_bsp["ci_low"], at_bsp["ci_high"]],
            "win_rate": float(won.mean()), "bsp_implied": float((1 / sub.bf_bsp).mean()),
            "median_clv_2_ticks": float(np.median(sub.taken_2 / sub.bf_bsp)),
            "median_clv_3_ticks": float(np.median(sub.taken_3 / sub.bf_bsp)),
            "median_clv_after_fall": float(np.median(fall.taken_1 / fall.bf_bsp)) if len(fall) else None,
            "n_after_fall": len(fall), "median_price": float(sub.quoted.median())}


def top_share(scored, column):
    """The top share of in-band runners by *column* within each test quarter."""
    cut = scored.groupby("quarter")[column].transform(lambda x: x.quantile(1 - TOP))
    return scored.loc[scored[column] >= cut]


def r_squared(scored, column):
    resid = ((scored.y - scored[column]) ** 2).sum()
    total = ((scored.y - scored.groupby("quarter").y.transform("mean")) ** 2).sum()
    return float(1 - resid / total)


def outcome_blend(frame, folds):
    """Winner log loss of the 11:00 price, a walk-forward blend with the forecast, and BSP."""
    race = frame.race_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    one_winner = pd.Series(frame.won.to_numpy()).groupby(race).transform("sum").eq(1).to_numpy()
    logs = np.column_stack([frame.log_p_mkt, frame.log_p_model])
    won = frame.won.to_numpy(float)

    def log_softmax(score, rows):
        r = race[rows]
        first = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        peak = np.maximum.reduceat(score, first)
        size = np.diff(np.r_[first, len(score)])
        total = np.log(np.add.reduceat(np.exp(score - np.repeat(peak, size)), first)) + peak
        return score - np.repeat(total, size), len(first)

    blend = np.full(len(frame), np.nan)
    weights = {}
    for fold in folds:
        train, test = np.flatnonzero(fold.train & one_winner), np.flatnonzero(fold.test)

        def loss(w):
            logp, n = log_softmax(logs[train] @ w, train)
            return -(logp * won[train]).sum() / n
        best = minimize(loss, [1., 0.], method="BFGS").x
        weights[fold.label] = [float(v) for v in best]
        blend[test] = log_softmax(logs[test] @ best, test)[0]
    scored = np.isfinite(blend) & one_winner
    bsp = (1 / frame.bf_bsp) / (1 / frame.bf_bsp).groupby(frame.race_id).transform("sum")
    ids = frame.race_id[scored]
    out = {"races": int(ids.nunique()),
           "price_1100": wf_eval.race_logloss(np.exp(log_softmax(logs[:, 0], np.arange(len(frame)))[0])[scored],
                                              won[scored], ids),
           "blend_1100_model": wf_eval.race_logloss(np.exp(blend[scored]), won[scored], ids),
           "model": wf_eval.race_logloss(np.exp(frame.log_p_model.to_numpy()[scored]), won[scored], ids),
           "bsp": wf_eval.race_logloss(bsp.to_numpy()[scored], won[scored], ids), "weights": weights}
    assert len(starts) == frame.race_id.nunique(), "runners of a race must be contiguous"
    return out


def universe(frame, name, threads, cut):
    folds = wf_eval.expanding_period_folds(frame.race_date, freq="Q", min_train_periods=4)
    frame = frame.copy()
    for arm, columns in (("market", MARKET), ("market+model", MARKET + MODEL)):
        frame[f"pred_{arm}"] = wf_eval.oof_predict_expanding(frame, fit_predict=fit_predict(columns, threads),
                                                             folds=folds)
    frame["pred_gap"] = frame.gap_centred
    scored = frame.loc[frame["pred_market"].notna() & frame.in_band]
    report = {"runners_scored": len(scored), "races_scored": int(scored.race_id.nunique()),
              "quarters": [f.label for f in folds], "baseline_all_in_band": bets(scored),
              "share_later_withdrawal": float(1 - frame.no_withdrawal.mean())}
    print(f"\n== {name}: {report['races_scored']} races, {len(scored)} in-band runners, "
          f"{report['quarters'][0]}..{report['quarters'][-1]}", flush=True)
    print(line("all in band", report["baseline_all_in_band"]), flush=True)
    for arm in ("market", "market+model", "gap"):
        column = f"pred_{arm}"
        picks = top_share(scored, column)
        entry = {"top": bets(picks), "by_year": {str(y): bets(g) for y, g in picks.groupby(picks.race_date.dt.year)},
                 "quarters_median_above_1": int(sum(np.median(g.taken_1 / g.bf_bsp) > 1
                                                    for _, g in picks.groupby("quarter"))),
                 "threshold_by_quarter": scored.groupby("quarter")[column].quantile(1 - TOP).round(4).to_dict()}
        if arm != "gap":
            entry["r_squared"] = r_squared(scored, column)
            deciles = pd.qcut(scored[column], 10, labels=False, duplicates="drop")
            entry["mean_y_by_decile"] = scored.groupby(deciles).y.mean().round(4).tolist()
        report[arm] = entry
        print(line(f"top {TOP:.0%} {arm}", entry["top"]) + f"  quarters>1 {entry['quarters_median_above_1']}/"
              f"{len(folds)}" + (f"  R2 {entry['r_squared']:.3f}" if arm != "gap" else ""), flush=True)
        for year, row in entry["by_year"].items():
            print(line(f"    {year}", row), flush=True)
    both = top_share(scored, "pred_market+model")
    only = both.loc[~both.index.isin(top_share(scored, "pred_market").index)]
    report["model_only_picks"] = bets(only)
    print(line("picked only with model", report["model_only_picks"]), flush=True)
    report["outcome"] = outcome_blend(frame, folds)
    print("winner log loss", {k: round(v, 4) for k, v in report["outcome"].items() if k != "weights" and k != "races"},
          "races", report["outcome"]["races"], flush=True)
    atomic_parquet(frame[["race_id", "horse_id", "race_date", "quarter", "in_band", "no_withdrawal", "quoted",
                          "taken_1", "bf_bsp", "won", "y", "pred_market", "pred_market+model", "pred_gap"]],
                   OUT / f"predictions_{name}_{cut}.parquet")
    return report


def line(label, row):
    if not row.get("n"):
        return f"{label:28s} n 0"
    return (f"{label:28s} n {row['n']:6d}  median CLV {row['median_clv']:.4f} [{row['median_ci'][0]:.4f}, "
            f"{row['median_ci'][1]:.4f}]  geo {row['geometric_clv']:.4f}  beat {row['beat_close']:.1%}  "
            f"ROI {row['roi_taken']:+.1f}% [{row['roi_taken_ci'][0]:+.1f}, {row['roi_taken_ci'][1]:+.1f}]  "
            f"@BSP {row['roi_bsp']:+.1f}%  2t {row['median_clv_2_ticks']:.4f}  3t {row['median_clv_3_ticks']:.4f}  "
            f"fall {row['median_clv_after_fall'] or float('nan'):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cut", choices=list(CUTS), default="1100")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = prepare(load(args.cut))
    print(f"decision time {args.cut}: {frame.race_id.nunique()} races, {len(frame)} runners, {frame.race_date.min().date()}.."
          f"{frame.race_date.max().date()}; commission {EXCHANGE_COMMISSION:.0%}", flush=True)
    report = {"note": "exploratory, development data, one tick down plus reductions, commission on winnings",
              "no_withdrawal": universe(frame.loc[frame.no_withdrawal].reset_index(drop=True), "no_withdrawal",
                                        args.threads, args.cut),
              "all_races_final_field_leak": universe(frame, "all_races", args.threads, args.cut)}
    atomic_json(OUT / f"early_price_screen_{args.cut}.json", report)


if __name__ == "__main__":
    main()
