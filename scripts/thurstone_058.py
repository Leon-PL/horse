"""Predict a performance figure and its spread, then simulate the race (experiment 058).

Per confirmation fold (2025Q3-2026Q2, purged, native capacity), on the
production inputs: a mean-figure regression (pounds behind the winner, centred
within the race), a spread regression on the out-of-fold log squared residual,
and one spread scale fitted on the purged early-stopping slab
(``src/thurstone.py``). Win chances come from normal performance draws by
quadrature. The arm is scored against the production arm on the same races.

Secondary readings (``report``):

* a two-weight blend ``p ~ production^a * thurstone^b``, fitted on earlier
  confirmation quarters only and scored on the next;
* whether the spread is wider for debutants and first runs for a new yard;
* whether the 055 drifter/steamer bias shrinks, with 055's labels: the move
  from the 11:00 Betfair price to BSP (:func:`price_move`).

    python -m scripts.thurstone_058 confirm
    python -m scripts.thurstone_058 report
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.model_ideas import (OUT, THREADS, Cache, baseline, compare, load_arm, print_summary,  # noqa: E402
                                 production_inputs, race_losses, run_arm, save_json, setup_logging, stage_dir,
                                 store_columns)
from scripts.select_features import CONFIRM_QUARTERS, OUT as CACHE_DIR  # noqa: E402
from src import feature_selection as fs  # noqa: E402
from src.connection_profiles import new_yard  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.forecast_evaluation import paired_interval  # noqa: E402
from src.thurstone import fit_figure_model, pounds_behind  # noqa: E402

logger = logging.getLogger("thurstone_058")


def figures() -> np.ndarray:
    """Performance figure per cached row: minus pounds behind the winner (NaN for non-finishers)."""
    store = store_columns(["beaten_lengths", "distance_furlongs", "finish_position"])
    lengths = store.beaten_lengths.to_numpy(np.float64)
    lengths = np.where(store.finish_position.eq(1), 0.0, lengths)
    return -pounds_behind(lengths, store.distance_furlongs.to_numpy(np.float64))


def confirm(args) -> None:
    cache = Cache()
    names = production_inputs(cache)
    figure = figures()
    t = cache.targets
    logger.info("figures for %.1f%% of development runners", 100 * np.isfinite(figure).mean())
    baseline(cache, "confirm")

    def predict(train, test, capacity):
        model = fit_figure_model(cache.take(train, names), figure[train], t.won[train], t.race_ids[train],
                                 t.race_dates[train], fs.recency_weights(t.race_dates[train]),
                                 capacity=capacity, n_jobs=THREADS, log=logger.info)
        label = pd.Period(pd.Timestamp(t.race_dates[test][0]), "Q")
        save_json(stage_dir("confirm") / f"thurstone_info_{label}.json", model.info)
        return model.predict(cache.take(test, names), t.race_ids[test]).rename(columns={"p": "model"})

    run_arm(cache, "confirm", "thurstone", predict)
    report(args)


# ------------------------------------------------------------------ report --

def _blend_loss(w, log_a, log_b, won, starts):
    score = w[0] * log_a + w[1] * log_b
    sizes = np.diff(np.r_[starts, len(score)])
    top = np.maximum.reduceat(score, starts)
    norm = np.log(np.add.reduceat(np.exp(score - np.repeat(top, sizes)), starts)) + top
    return -(np.add.reduceat(won * score, starts) - norm)


def _blend_inputs(a: pd.DataFrame, b: pd.DataFrame):
    groups = fs.race_groups(a.race_id.to_numpy())
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    log_a = np.log(np.clip(fs.race_normalise(a.model.to_numpy(np.float64), groups), 1e-12, None))
    log_b = np.log(np.clip(fs.race_normalise(b.model.to_numpy(np.float64), groups), 1e-12, None))
    return log_a, log_b, a.won.to_numpy(np.float64), starts


def blended(production: pd.DataFrame, thurstone: pd.DataFrame) -> dict:
    """Production x Thurstone blend, weights fitted on earlier quarters only."""
    rows, deltas, dates = [], [], []
    for k in range(1, len(CONFIRM_QUARTERS)):
        history = production.fold.isin(CONFIRM_QUARTERS[:k]).to_numpy()
        current = production.fold.eq(CONFIRM_QUARTERS[k]).to_numpy()
        args = _blend_inputs(production[history], thurstone[history])
        fitted = minimize(lambda w: _blend_loss(w, *args).mean(), x0=[1.0, 0.0], method="BFGS").x
        log_a, log_b, won, starts = _blend_inputs(production[current], thurstone[current])
        blend = _blend_loss(fitted, log_a, log_b, won, starts)
        base = _blend_loss([1.0, 0.0], log_a, log_b, won, starts)
        deltas.append(blend - base)
        dates.append(production[current].race_date.to_numpy()[starts])
        rows.append({"quarter": CONFIRM_QUARTERS[k], "production_weight": float(fitted[0]),
                     "thurstone_weight": float(fitted[1]), "delta": float((blend - base).mean())})
    delta, date = np.concatenate(deltas), np.concatenate(dates)
    ci = paired_interval(delta, blocks=pd.Series(date).astype(str).to_numpy())["ci95"]
    return {"races": int(len(delta)), "delta_vs_production": float(delta.mean()), "ci95": ci,
            "quarters_better": f"{sum(r['delta'] < 0 for r in rows)}/{len(rows)}", "per_quarter": rows}


def price_move(context: pd.DataFrame, race_ids: np.ndarray) -> np.ndarray:
    """055's labels: BSP-implied over 11:00-implied chance, each normalised within the race.

    As in 055 (and ledger 19) a race counts only when it is off at 11:30 UK
    or later and every runner traded by 11:00.
    """
    from scripts.microstructure_clv import _with_early_price
    keyed = context[["race_date", "off_time", "track", "horse_name"]].assign(race_id=race_ids,
                                                                            position=np.arange(len(context)))
    early = _with_early_price(keyed, str(ROOT / "data/processed/betfair_bsp_early.parquet"), "ltp_1100", 11 * 60 + 30)
    from src.betfair import attach_bsp
    early = attach_bsp(early, pd.read_parquet(ROOT / "data/processed/betfair_bsp_early.parquet"), cols=("bsp",))
    early = early[early.bsp.gt(1).groupby(early.race_id).transform("all")]
    late = 1 / early.bsp
    morning = 1 / early.ltp_1100
    ratio = (late / late.groupby(early.race_id).transform("sum")) / \
        (morning / morning.groupby(early.race_id).transform("sum"))
    labels = np.full(len(context), "no 11:00 market", dtype=object)
    labels[early.position.to_numpy()] = np.select([ratio.ge(1.5), ratio.le(1 / 1.5)], ["steamer", "drifter"],
                                                  default="stable")
    return labels


def segments(frame: pd.DataFrame) -> dict:
    """Spread and calibration by debutant / new-yard / other, and by price move."""
    rows = pd.read_parquet(CACHE_DIR / "targets.parquet", columns=["store_row"]).store_row.to_numpy()
    history = pq.read_table(FeatureStore().featured_path,
                            columns=["horse_id", "race_date", "trainer", "horse_prev_races", "off_time", "track",
                                     "horse_name"]).to_pandas()
    history["new_yard"] = new_yard(history)
    context = history.iloc[rows].reset_index(drop=True).iloc[frame.row.to_numpy()].reset_index(drop=True)
    kind = np.select([context.horse_prev_races.eq(0).to_numpy(), context.new_yard.to_numpy()],
                     ["debutant", "new yard"], default="other")
    move = price_move(context, frame.race_id.to_numpy())
    groups = fs.race_groups(frame.race_id.to_numpy())
    p = {name: fs.race_normalise(frame[col].to_numpy(np.float64), groups)
         for name, col in [("production", "production"), ("thurstone", "model"), ("market", "market")]}

    def table(labels):
        out = {}
        for label in pd.unique(labels):
            m = labels == label
            out[str(label)] = {"runners": int(m.sum()), "wins": int(frame.won[m].sum()),
                               **{f"{k}_expected": round(float(v[m].sum()), 1) for k, v in p.items()},
                               "mean_sigma": round(float(frame.sigma[m].mean()), 2)}
        return out
    return {"runner_kind": table(kind), "price_move": table(move)}


def report(_args) -> dict:
    summary = compare("confirm", ["production", "thurstone"], [("thurstone", "production")])
    production, thurstone = load_arm("confirm", "production"), load_arm("confirm", "thurstone")
    summary["blend"] = blended(production, thurstone)
    joined = thurstone.assign(production=production.model.to_numpy())
    summary["segments"] = segments(joined)
    summary["fold_info"] = {q: pd.read_json(stage_dir("confirm") / f"thurstone_info_{q}.json", typ="series").to_dict()
                            for q in CONFIRM_QUARTERS if (stage_dir("confirm") / f"thurstone_info_{q}.json").exists()}
    save_json(OUT / "summary_058.json", summary)
    print_summary(summary, "058 Thurstone figure model")
    b = summary["blend"]
    print(f"  blend (fitted on earlier quarters) - production: {b['delta_vs_production']:+.5f} "
          f"[{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}] on {b['races']} races, better {b['quarters_better']}")
    for name, table in summary["segments"].items():
        print(f"\n  {name}:")
        print(pd.DataFrame(table).T.to_string())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["confirm", "report"])
    args = parser.parse_args()
    setup_logging()
    {"confirm": confirm, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
