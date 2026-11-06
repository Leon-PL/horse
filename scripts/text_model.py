"""Does Timeform's pre-race comment make the win model stronger? (experiment 052)

Racing TV cards carry Timeform's analyst comment for each runner and a race
verdict, back to September 2020 (``src.ingestion.pipeline.load_matched_text``). The
same text is on Sporting Life racecards.

1. ``build``: one row per development starter (before the golden hold-out)
   with its comment and the verdict sentences that name it.
2. ``score``: a race-wise softmax regression on word and word-pair counts
   (TF-IDF, vocabulary and IDF taken from the training rows only), refitted
   for every quarter on earlier quarters only (``expanding_period_folds``,
   purged). The out-of-fold score is the text model's view of each runner. The
   penalty is chosen on quarters before 2025Q3.
3. ``evaluate``: the production win recipe on the 048 inputs, with and
   without the text score, on the 048/049 confirmation quarters 2025Q3–2026Q2.
   The without arm is experiment 049's ``general`` arm, read from disk.

Leak checks: the text-only model must not beat the market; its strongest words
must read as pre-race language; and its gain must hold in every year.

    python -m scripts.text_model build
    python -m scripts.text_model score
    python -m scripts.text_model evaluate
    python -m scripts.text_model report
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from scripts.select_features import (CONFIRM_QUARTERS, OUT as CACHE_DIR, SELECTED, SELECTION_QUARTERS,  # noqa: E402
                                     Cache, read_names)
from scripts.specialist_models import OUT as SPECIALIST_DIR, _interval, _predict, _race_frame, blended  # noqa: E402
from src import feature_selection as fs  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.rtv_text import counts, fit_softmax, idf, prepare, race_index, race_probs as _race_probs, tfidf  # noqa: E402
from src.ingestion.pipeline import load_matched_text  # noqa: E402
from src.wf_eval import expanding_period_folds, golden_holdout_start  # noqa: E402

OUT = ROOT / "data/research/text_052"
START = "2020-09-01"
MIN_DF = 20                       # training rows an n-gram must appear in
PENALTIES = [1e-6, 3e-6, 1e-5, 3e-5]  # L2 per weight, chosen on the 048 selection quarters
TEXT_FEATURES = ["text_logit", "text_prob", "verdict_rank"]
COMMENT_FIELDS = {"tf_comment": "c", "verdict_text": "v"}   # text column -> n-gram prefix

logger = logging.getLogger("text_model")


# ------------------------------------------------------------------ build --

def build(_args) -> None:
    end = (golden_holdout_start() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with sqlite3.connect(Path(config.DATA_DIR) / "races.db") as con:
        field = pd.read_sql("SELECT race_id, horse_id, horse_name, race_date, finish_position FROM results "
                            "WHERE race_date BETWEEN ? AND ?", con, params=(START, end))
    field["race_id"], field["horse_id"] = field.race_id.astype(str), pd.to_numeric(field.horse_id)
    chunks = []
    for year in range(int(START[:4]), int(end[:4]) + 1):
        lo, hi = max(START, f"{year}-01-01"), min(end, f"{year}-12-31")
        text = load_matched_text(lo, hi)
        logger.info("%s: %d linked runners", year, len(text))
        chunks.append(text[["race_id", "horse_id", "tf_comment", "tf_verdict"]])
    text = pd.concat(chunks, ignore_index=True)
    text["race_id"], text["horse_id"] = text.race_id.astype(str), pd.to_numeric(text.horse_id)
    frame = field.merge(text, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    frame["won"] = pd.to_numeric(frame.finish_position, errors="coerce").eq(1).astype(np.int8)
    frame = frame[frame.groupby("race_id").won.transform("sum").gt(0)]
    frame = prepare(frame)
    frame = frame.sort_values(["race_date", "race_id", "horse_id"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["race_id", "horse_id", "horse_name", "race_date", "won", "tf_comment", "verdict_text", "verdict_rank"]
    frame[keep].to_parquet(OUT / "text.parquet", index=False)
    print(f"{len(frame):,} starters in {frame.race_id.nunique():,} races; comment present "
          f"{frame.tf_comment.notna().mean():.3f}; named in verdict {frame.verdict_rank.gt(0).mean():.3f}")


# ------------------------------------------------------------------ score --

def _race_ll(p, won, race_ids):
    starts, sizes = race_index(race_ids)
    return -np.add.reduceat(won * np.log(np.clip(p, 1e-12, None)), starts) / np.add.reduceat(won, starts)


def _design(matrix, fold):
    """TF-IDF train/test matrices on the n-grams frequent in this fold's training rows."""
    train, test = np.flatnonzero(fold.train), np.flatnonzero(fold.test)
    columns = np.flatnonzero(np.asarray((matrix[train] > 0).sum(axis=0)).ravel() >= MIN_DF)
    weights = idf(matrix[train][:, columns])
    return train, test, columns, tfidf(matrix[train][:, columns], weights), tfidf(matrix[test][:, columns], weights)


def score(_args) -> None:
    fit_scores(pd.read_parquet(OUT / "text.parquet"), COMMENT_FIELDS, OUT)


def fit_scores(frame: pd.DataFrame, fields: dict[str, str], out: Path) -> None:
    """Out-of-fold text scores for every quarter after the first two, saved to ``out``."""
    matrix, names = counts(frame, fields=fields)
    won = frame.won.to_numpy(np.float64)
    race_ids = frame.race_id.to_numpy()
    folds = expanding_period_folds(frame.race_date, min_train_periods=2)
    logger.info("%d starters, %d n-grams, %d folds (%s..%s)", len(frame), len(names), len(folds),
                folds[0].label, folds[-1].label)
    logits, weights = {}, {}

    # 1. The penalty, by pooled race LL on the 048 selection quarters.
    losses = {p: [] for p in PENALTIES}
    for fold in [f for f in folds if f.label in SELECTION_QUARTERS]:
        started = time.monotonic()
        train, test, columns, x_train, x_test = _design(matrix, fold)
        for penalty in PENALTIES:
            w = fit_softmax(x_train, won[train], race_ids[train], penalty)
            logits[(fold.label, penalty)], weights[(fold.label, penalty)] = x_test @ w, pd.Series(w, index=columns)
            losses[penalty].append(_race_ll(_race_probs(logits[(fold.label, penalty)], race_ids[test]),
                                            won[test], race_ids[test]))
        logger.info("select %s: %d n-grams (%.0fs)", fold.label, len(columns), time.monotonic() - started)
    losses = {p: float(np.concatenate(v).mean()) for p, v in losses.items()}
    penalty = min(losses, key=losses.get)
    logger.info("Race LL by penalty: %s -> %g", losses, penalty)

    # 2. Every quarter with that penalty, warm-started from the previous quarter.
    rows = pd.DataFrame({"race_id": race_ids, "horse_id": frame.horse_id, "race_date": frame.race_date,
                        "won": won, "verdict_rank": frame.verdict_rank, "fold": None, "text_logit": np.nan})
    previous = None
    for fold in folds:
        test = np.flatnonzero(fold.test)
        if (fold.label, penalty) not in logits:
            started = time.monotonic()
            train, test, columns, x_train, x_test = _design(matrix, fold)
            w0 = None if previous is None else previous.reindex(columns).fillna(0).to_numpy()
            w = fit_softmax(x_train, won[train], race_ids[train], penalty, w0=w0)
            logits[(fold.label, penalty)], weights[(fold.label, penalty)] = x_test @ w, pd.Series(w, index=columns)
            logger.info("%s: %d train / %d test rows, %d n-grams (%.0fs)", fold.label, len(train), len(test),
                        len(columns), time.monotonic() - started)
        previous = weights[(fold.label, penalty)]
        rows.loc[test, "fold"] = fold.label
        rows.loc[test, "text_logit"] = logits[(fold.label, penalty)]
    final = rows[rows.fold.notna()].reset_index(drop=True)
    final["text_prob"] = _race_probs(final.text_logit.to_numpy(), final.race_id.to_numpy())
    final.to_parquet(out / "scores.parquet", index=False)
    last = max(f.label for f in folds if f.label < CONFIRM_QUARTERS[0])
    w = weights[(last, penalty)]
    top = pd.Series(w.to_numpy(), index=np.asarray(names)[w.index]).sort_values()
    (out / "score.json").write_text(json.dumps({
        "penalty_losses": losses, "penalty": penalty, "folds": [f.label for f in folds],
        "weights_from_fold": last, "most_negative": top.head(40).round(3).to_dict(),
        "most_positive": top.tail(40)[::-1].round(3).to_dict()}, indent=2), encoding="utf-8")


# --------------------------------------------------------------- evaluate --

def _cache_keys(cache: Cache) -> pd.DataFrame:
    rows = pd.read_parquet(CACHE_DIR / "targets.parquet", columns=["store_row"]).store_row.to_numpy()
    keys = pq.read_table(FeatureStore().featured_path, columns=["race_id", "horse_id"]).to_pandas().iloc[rows]
    keys = keys.reset_index(drop=True)
    keys["race_id"], keys["horse_id"] = keys.race_id.astype(str), pd.to_numeric(keys.horse_id)
    if len(keys) != len(cache.targets.won) or not np.array_equal(keys.race_id.to_numpy(), cache.targets.race_ids.astype(str)):
        raise ValueError("Store keys do not line up with the cache")
    return keys


class ExtraCache:
    """The 048 cache with extra per-runner columns appended (NaN where a runner has none)."""

    def __init__(self, cache: Cache, extra: pd.DataFrame):
        self.cache, self.targets, self.folds = cache, cache.targets, cache.folds
        self.names = [c for c in extra.columns if c not in ("race_id", "horse_id")]
        extra = extra.assign(race_id=extra.race_id.astype(str))
        joined = _cache_keys(cache).merge(extra, on=["race_id", "horse_id"], how="left", validate="one_to_one")
        self.values = joined[self.names].to_numpy(np.float64)
        logger.info("Extra columns present on %.3f of cached rows", np.isfinite(self.values).any(axis=1).mean())

    def take(self, rows: np.ndarray, names: list[str]) -> np.ndarray:
        base = [n for n in names if n not in self.names]
        extra = [self.names.index(n) for n in names if n in self.names]
        return np.hstack([self.cache.take(rows, base), self.values[np.ix_(rows, extra)]])

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def fold_data(self, labels):
        return self.cache.fold_data(labels)


def fit_arm(cache: ExtraCache, features: list[str], out: Path, arm: str) -> None:
    """Production win recipe on ``features`` for each confirmation quarter (resumable)."""
    t = cache.targets
    for fold in cache.fold_data(CONFIRM_QUARTERS):
        path = out / f"{arm}_{fold.label}.parquet"
        if path.exists():
            continue
        started = time.monotonic()
        p = _predict(cache, features, fold.train_rows, fold.test_rows)
        frame = pd.DataFrame({"row": fold.test_rows, "race_id": t.race_ids[fold.test_rows],
                              "race_date": t.race_dates[fold.test_rows], "won": t.won[fold.test_rows],
                              "market": t.market[fold.test_rows], "model": p})
        if "text_prob" in cache.names:
            frame["text_prob"] = cache.column("text_prob")[fold.test_rows]
        frame.to_parquet(path, index=False)
        logger.info("%s %s: %d train / %d test rows (%.0fs)", arm, fold.label, len(fold.train_rows),
                    len(fold.test_rows), time.monotonic() - started)


def evaluate(_args) -> None:
    scores = pd.read_parquet(OUT / "scores.parquet", columns=["race_id", "horse_id", *TEXT_FEATURES])
    fit_arm(ExtraCache(Cache(), scores), read_names(SELECTED) + TEXT_FEATURES, OUT, "with_text")


# ----------------------------------------------------------------- report --

def report(_args) -> dict:
    load = lambda d, arm: pd.concat([pd.read_parquet(d / f"{arm}_{q}.parquet").assign(fold=q)
                                     for q in CONFIRM_QUARTERS], ignore_index=True)
    base, text = load(SPECIALIST_DIR, "general"), load(OUT, "with_text")
    if not np.array_equal(base.row.to_numpy(), text.row.to_numpy()):
        raise ValueError("The two arms were not scored on the same rows")
    races = base.drop_duplicates("race_id").set_index("race_id")[["race_date", "fold"]]
    market, without, with_text = _race_frame(base, "market"), _race_frame(base, "model"), _race_frame(text, "model")
    text_only = _race_frame(text.assign(model=text.text_prob.fillna(1.0)), "model")
    gain = with_text - without
    per_quarter = gain.groupby(races.fold).mean()
    blends = {}
    for name, runners in (("without", base), ("with_text", text)):
        loss, weights = blended(runners)
        blends[name] = {"gain_vs_market": float((loss - market.loc[loss.index]).mean()),
                        "ci": _interval(loss - market.loc[loss.index], races.race_date.loc[loss.index]),
                        "weights": weights}
    scores = pd.read_parquet(OUT / "scores.parquet")
    years = scores.assign(year=pd.to_datetime(scores.race_date).dt.year)
    by_year = {}
    for year, g in years.groupby("year"):
        ll = _race_ll(g.text_prob.to_numpy(), g.won.to_numpy(np.float64), g.race_id.to_numpy())
        sizes = fs.race_groups(g.race_id.to_numpy())
        by_year[int(year)] = {"races": int(len(sizes)), "text_ll": float(ll.mean()),
                              "uniform_ll": float(np.log(sizes).mean())}
    summary = {
        "races": int(len(races)), "market_ll": float(market.mean()), "without_ll": float(without.mean()),
        "with_text_ll": float(with_text.mean()), "text_only_ll": float(text_only.mean()),
        "with_minus_without": float(gain.mean()), "ci": _interval(gain, races.race_date),
        "quarters_better": f"{int((per_quarter < 0).sum())}/{len(per_quarter)}",
        "per_quarter": {q: round(float(v), 5) for q, v in per_quarter.items()},
        "gap_to_market_without": float((without - market).mean()),
        "gap_to_market_with_text": float((with_text - market).mean()),
        "text_only_minus_market": float((text_only - market).mean()),
        "blends": blends, "text_by_year": by_year,
        "score": json.loads((OUT / "score.json").read_text(encoding="utf-8")),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"Race log loss, {', '.join(CONFIRM_QUARTERS)}, {len(races):,} races (lower is better)")
    print(f"  market {summary['market_ll']:.4f}  text only {summary['text_only_ll']:.4f}  "
          f"model {summary['without_ll']:.4f}  model+text {summary['with_text_ll']:.4f}")
    print(f"  model+text - model {gain.mean():+.4f} [{summary['ci'][0]:+.4f}, {summary['ci'][1]:+.4f}], "
          f"better in {summary['quarters_better']} quarters {summary['per_quarter']}")
    print(f"  gap to market {summary['gap_to_market_without']:+.4f} -> {summary['gap_to_market_with_text']:+.4f}")
    for name, b in blends.items():
        print(f"  blend with market ({name}): {b['gain_vs_market']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
    print("  text-only by year:", {y: round(v["text_ll"] - v["uniform_ll"], 4) for y, v in by_year.items()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "score", "evaluate", "report"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    {"build": build, "score": score, "evaluate": evaluate, "report": report}[args.step](args)


if __name__ == "__main__":
    main()
