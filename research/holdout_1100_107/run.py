"""One registered look at the 11:00 betting rule on the golden hold-out (research 107).

    python -m research.holdout_1100_107.run register
    python -m research.holdout_1100_107.run rehearse [--threads 7]    # development races only
    python -m research.holdout_1100_107.run score [--threads 7]       # once; refuses a second run

The rule (research 105, chosen on development quarters 2025Q3-2026Q2) is fixed in ``RULE`` and in
``docs/research/HOLDOUT_1100_107.md`` before any hold-out race is scored:

- forecast: the stored 87-input win forecast (``src.correction`` history; hold-out forecasts come from a model
  trained to 2026-06-03) with the production correction (``src.correction``, groups ``GROUPS``) plus the eight
  11:00 price inputs (``research.error_anatomy.morning_price``, races off at or before 11:00 blanked);
- fit: one correction fitted on every race with a forecast before 2026-06-11, applied unchanged to every hold-out
  race (no refit inside the hold-out);
- bets: ``src.bet_settlement.value_bet_selection`` at base threshold 0.02 against the race-normalised 11:00
  share of the field; whole field priced at 11:00 and at BSP; off after 11:00; taken price = 11:00 last traded
  price cut by later reduction factors, then one Betfair tick worse; taken price above 1 and at most 51;
- primary: geometric-mean CLV (taken / BSP). Pass if the one-sided 95% lower bound, from a bootstrap over race
  days, is above 1.0;
- reported, not decisive: ROI at the taken price after 2% commission (races resampled), the same at the last
  traded price, threshold 0, every priced runner in the band (same universe), race log loss of the corrected
  forecast against the stored forecast.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from research.error_anatomy.morning_price import load_timed
from research.t60_clv_105.run import prices as bet_prices
from src.bet_settlement import (apply_reduction_factor, betfair_tick_down, settle_win_exchange_bets,
                                value_bet_selection)
from src.clv import summarise_clv
from src.correction import data as cd
from src.correction import model as cm
from src.ingestion.storage import atomic_json, atomic_parquet
from src.wf_eval import bootstrap_roi_ci

OUT = Path("data/research/holdout_1100_107")
KEYS = cm.KEYS
HOLDOUT = pd.Timestamp(config.GOLDEN_HOLDOUT_START)
RULE = {"threshold": 0.02, "ticks_worse": 1, "max_odds": 51.0, "boot": 4000, "seed": 0, "alpha_one_sided": 0.05}
CODE = ("research/holdout_1100_107/run.py", "research/error_anatomy/morning_price.py",
        "research/t60_clv_105/run.py", "src/bet_settlement.py", "src/correction/model.py",
        "src/correction/inputs.py", "src/correction/data.py")


def code_sha256():
    digest = hashlib.sha256()
    for path in CODE:
        digest.update(path.encode())
        digest.update(Path(path).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def build(development):
    """Frame (races with a forecast), row-aligned inputs and their names, with the 11:00 price group."""
    frame, prices, tracking, receipt = cd.assemble(development=development)
    values, columns = cm.build_inputs(frame, prices=prices, tracking=tracking)
    rows = frame.p_base.notna().to_numpy()
    frame, values = frame.loc[rows].reset_index(drop=True), values.loc[rows].reset_index(drop=True)
    work = frame[KEYS].assign(race_id=frame.race_id.astype(str), p_single=frame.p_base.to_numpy())
    price, names = load_timed(work, when="1100-hours")
    names = [c for c in names if c != "pm_hours_to_off"]
    if not price.race_id.equals(work.race_id) or not price.horse_id.equals(work.horse_id):
        raise ValueError("Price inputs changed the runner order")
    values = pd.concat([values, price[names].reset_index(drop=True)], axis=1)
    return frame, values, [*columns, *names], receipt


def geometric_clv(log_clv, days, rng, boot):
    table = pd.DataFrame({"l": log_clv, "d": days}).groupby("d").l.agg(["sum", "size"])
    s, n = table["sum"].to_numpy(), table["size"].to_numpy()
    draws = np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, len(s), len(s))
        draws[b] = s[i].sum() / n[i].sum()
    return float(np.exp(np.mean(log_clv))), np.exp(draws)


def bets(frame, probability):
    """The rule's selections and the same-universe runners, with taken prices."""
    p = bet_prices()
    p["horse_id"] = p.horse_id.astype(frame.horse_id.dtype)
    f = frame[[*KEYS, "race_date", "won"]].assign(race_id=frame.race_id.astype(str), p=probability)
    f = f.merge(p[[*KEYS, "bsp", "ltp_1100", "rf_after_1100"]], on=KEYS, how="left", validate="one_to_one")
    hours = load_timed(f[KEYS].assign(p_single=f.p), when="1100-hours")[0].pm_hours_to_off
    race = f.race_id.to_numpy()
    price = f.ltp_1100.where(f.ltp_1100 > 1)
    whole = (price.notna() & f.bsp.gt(1)).groupby(race).transform("all") & hours.gt(0).to_numpy()
    f = f.loc[whole].reset_index(drop=True)
    race = f.race_id.to_numpy()
    f["share"] = (1 / f.ltp_1100) / (1 / f.ltp_1100).groupby(race).transform("sum")
    f["p"] = f.p / f.p.groupby(race).transform("sum")
    ltp = apply_reduction_factor(f.ltp_1100, f.rf_after_1100)
    f["taken_ltp"] = ltp
    f["taken"] = betfair_tick_down(ltp, RULE["ticks_worse"])
    return f


def summary(f, mask, taken, rng):
    pnl = settle_win_exchange_bets(taken[mask], f.won.to_numpy()[mask])
    roi = bootstrap_roi_ci(pnl, clusters=f.race_id.to_numpy()[mask])
    log_clv = np.log(taken[mask] / f.bsp.to_numpy()[mask])
    geo, draws = geometric_clv(log_clv, f.race_date.to_numpy()[mask], rng, RULE["boot"])
    clv = summarise_clv(taken[mask] / f.bsp.to_numpy()[mask])
    return {"bets": int(mask.sum()), "races": int(pd.unique(f.race_id.to_numpy()[mask]).size),
            "days": int(pd.unique(f.race_date.to_numpy()[mask]).size),
            "geometric_mean_clv": geo, "geo_clv_lower_one_sided_95": float(np.quantile(draws, RULE["alpha_one_sided"])),
            "geo_clv_ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "median_clv": clv["median_clv"], "pct_beat_close": clv["pct_beat_close"], "roi": roi}


def evaluate(frame, probability):
    f = bets(frame, probability)
    rng = np.random.default_rng(RULE["seed"])
    out = {"first_date": str(f.race_date.min().date()), "last_date": str(f.race_date.max().date()),
           "races": int(f.race_id.nunique())}
    for label, taken in (("one_tick_worse", f.taken.to_numpy()), ("last_traded", f.taken_ltp.to_numpy())):
        band = (taken > 1) & (taken <= RULE["max_odds"])
        for threshold in (RULE["threshold"], 0.0):
            mask = value_bet_selection(f.p.to_numpy(), taken, threshold, implied_prob=f.share.to_numpy())["mask"] & band
            out[f"{label} threshold {threshold}"] = summary(f, mask, taken, rng)
        out[f"{label} universe"] = summary(f, band, taken, rng)
    primary = out[f"one_tick_worse threshold {RULE['threshold']}"]
    out["primary"] = {"statistic": "geometric_mean_clv at one tick worse, threshold 0.02",
                      "value": primary["geometric_mean_clv"], "lower_one_sided_95": primary["geo_clv_lower_one_sided_95"],
                      "pass": bool(primary["geo_clv_lower_one_sided_95"] > 1.0)}
    return out


def race_log_loss(frame, probability):
    lp = np.log(np.maximum(np.asarray(probability, float), 1e-300))
    won = frame.won.to_numpy(float) > 0
    return pd.Series(np.where(won, -lp, 0.)).groupby(frame.race_id.to_numpy()).sum()


def register():
    target = OUT / "registration.json"
    if (OUT / "receipt.json").exists():
        raise SystemExit("The hold-out has been scored; registration is closed")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(target, {"study": "research 107", "holdout_start": str(HOLDOUT.date()), "rule": RULE,
                         "groups": list(cm.GROUPS) + ["price-1100 (eight inputs)"], "code_sha256": code_sha256(),
                         "registered_at": pd.Timestamp.now().isoformat(timespec="seconds")})
    print(target.read_text())


def rehearse(threads):
    """The scoring path on development races only: walk-forward quarters 2025Q3-2026Q2."""
    frame, values, columns, receipt = build(development=True)
    adjustment, labels = cm.walk_forward(frame, values, columns, first_scored="2025-07-01", threads=threads)
    kept = np.isfinite(adjustment)
    frame = frame.loc[kept].reset_index(drop=True)
    p = cm.corrected(frame.p_base.to_numpy(float), adjustment[kept], frame.race_id.to_numpy())
    report = {"quarters": labels, "inputs": len(columns), **evaluate(frame, p),
              "log_loss_corrected": float(race_log_loss(frame, p).mean()),
              "log_loss_base": float(race_log_loss(frame, frame.p_base).mean())}
    atomic_json(OUT / "rehearsal.json", report)
    print(json.dumps({k: v for k, v in report.items() if k in ("primary", "log_loss_corrected", "log_loss_base")}, indent=1))


def score(threads):
    registration = json.loads((OUT / "registration.json").read_text())
    if registration["code_sha256"] != code_sha256():
        raise SystemExit("Code changed since registration; register again before scoring")
    if (OUT / "receipt.json").exists():
        raise SystemExit("The hold-out has already been scored")
    frame, values, columns, receipt = build(development=False)
    train = frame.race_date.lt(HOLDOUT).to_numpy()
    model = cm.CorrectionModel(columns).fit(frame.loc[train].reset_index(drop=True),
                                            values.loc[train].reset_index(drop=True), threads=threads)
    test = ~train
    held, held_values = frame.loc[test].reset_index(drop=True), values.loc[test].reset_index(drop=True)
    p = model.predict(held, held_values)
    atomic_parquet(held[[*KEYS, "race_date", "won", "p_base"]].assign(p_corrected=p), OUT / "scored_runners.parquet")
    result = evaluate(held, p)
    receipt = {"registration_sha256": hashlib.sha256((OUT / "registration.json").read_bytes()).hexdigest(),
               "correction_training": model.lineage, "inputs": len(columns), **result,
               "log_loss_corrected": float(race_log_loss(held, p).mean()),
               "log_loss_base": float(race_log_loss(held, held.p_base).mean()),
               "scored_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    atomic_json(OUT / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("primary", "first_date", "last_date", "log_loss_corrected",
                                              "log_loss_base")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["register", "rehearse", "score"])
    parser.add_argument("--threads", type=int, default=7)
    args = parser.parse_args()
    {"register": register, "rehearse": lambda: rehearse(args.threads), "score": lambda: score(args.threads)}[args.stage]()
