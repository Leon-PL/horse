"""Walk-forward selection of the win model's inputs (experiment 048).

Chooses the allowlist by the model's own out-of-sample race log loss over
purged quarterly folds, then confirms the choice on later quarters that took
no part in it. The protocol and its rationale are in ``src/feature_selection.py``
and ``docs/research/FEATURE_SELECTION_048.md``. The golden hold-out is never loaded.

Stages, each resumable::

    python -m scripts.select_features prepare     # cache the development matrix (~3 GB)
    python -m scripts.select_features eliminate   # recursive elimination on 2024Q1-2025Q2
    python -m scripts.select_features confirm     # selected vs win.txt vs all, 2025Q3-2026Q2
    python -m scripts.select_features confirm --capacity native --arms selected current

``prepare`` refuses a stale store and every later stage refuses a cache built
from a different store; rerun ``prepare`` after a rebuild.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from src import feature_selection as fs  # noqa: E402
from src.feature_store import FeatureStore  # noqa: E402
from src.featured_dataset import _REQUIRED_KEEP, _atr_signature, _rtv_signature, fe_signature  # noqa: E402
from src.forecast_evaluation import paired_interval, valid_race_mask  # noqa: E402
from src.model import get_feature_columns, prepare_multi_target_data  # noqa: E402
from src.wf_eval import development_mask, expanding_period_folds  # noqa: E402

OUT = ROOT / "data/research/feature_selection_048"
CURRENT = ROOT / "feature_sets/win.txt"
SELECTED = ROOT / "feature_sets/win_selected_048.txt"
SELECTION_QUARTERS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2"]
CONFIRM_QUARTERS = ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]
READ_CHUNK = 64

logger = logging.getLogger("select_features")


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def store_meta(store: FeatureStore) -> dict:
    return json.loads((Path(store.featured_path).parent / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- prepare --

def prepare(_args, only: list[str] | None = None) -> None:
    """Cache the development matrix; *only* restricts the columns (for smoke tests)."""
    store = FeatureStore()
    meta = store_meta(store)
    stale = []
    if meta["fe_signature"] != fe_signature():
        stale.append("FE code")
    # Compare against the dataset the next refresh would select.
    from src.source_dataset import committed_dataset, use_manifest
    dataset = committed_dataset() or meta.get("dataset_manifest")
    if not meta.get("dataset_manifest") or dataset is None:
        stale.append("store predates source publication")
    with use_manifest(dataset or {"sources": {}}):
        atr = meta.get("atr_signature")
        if isinstance(atr, dict) and atr.get("cutoff") and _atr_signature(atr["cutoff"])["sig"] != atr.get("sig"):
            stale.append("ATR sectional table")
        rtv, current_rtv = meta.get("rtv_signature"), _rtv_signature(meta["last_date"])
        if (rtv is not None or any(current_rtv["sig"].values())) and rtv != current_rtv:
            stale.append("RTV observation/card table")
    if stale:
        raise SystemExit(f"The feature store is stale ({', '.join(stale)} changed beneath it); rebuild "
                         "it first (python -m scripts.build_feature_store_batched).")
    OUT.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(store.featured_path)
    empty = parquet.schema_arrow.empty_table().to_pandas()
    with patch.object(config, "FEATURE_ALLOWLIST", None):
        eligible = get_feature_columns(empty)
    context = [n for n in parquet.schema_arrow.names if n not in eligible or n in _REQUIRED_KEEP]
    if only is not None:
        eligible = [n for n in eligible if n in set(only)]
    logger.info("%d eligible features, %d context columns", len(eligible), len(context))

    frame = parquet.read(columns=context).to_pandas()
    frame["store_row"] = np.arange(len(frame), dtype=np.int64)
    dataset_start = pd.to_datetime(frame.race_date).min()
    frame = frame.loc[development_mask(frame.race_date)]
    frame = frame.loc[valid_race_mask(frame)]
    with patch.object(config, "FEATURE_ALLOWLIST", None):
        data = prepare_multi_target_data(frame, dataset_start=dataset_start, training_only=True)
    del frame
    rows = data["train_df"]
    won = data["y_train_won"].astype(np.int8)
    teacher = np.asarray(data["win_teacher_train"], dtype=np.float64)
    soft = None
    if bool(getattr(config, "WIN_SOFT_LABEL_DISTILL", False)):
        blend = float(config.WIN_SOFT_LABEL_BLEND)
        soft = np.where(teacher > 0, blend * teacher + (1 - blend) * won, won)
    targets = pd.DataFrame({
        "store_row": rows.store_row.to_numpy(), "race_id": rows.race_id.to_numpy(),
        "race_date": pd.to_datetime(rows.race_date).to_numpy(), "horse_name": rows.horse_name.to_numpy(),
        "won": won, "soft_y": soft if soft is not None else np.nan, "market": teacher,
    })
    targets.to_parquet(OUT / "targets.parquet", index=False)
    logger.info("%d development runners in %d races, %s to %s", len(targets), targets.race_id.nunique(),
                targets.race_date.min().date(), targets.race_date.max().date())

    positions = targets.store_row.to_numpy()
    matrix = np.lib.format.open_memmap(OUT / "matrix.npy", mode="w+", dtype=np.float32,
                                       shape=(len(positions), len(eligible)), fortran_order=True)
    for start in range(0, len(eligible), READ_CHUNK):
        names = eligible[start:start + READ_CHUNK]
        table = parquet.read(columns=names)
        for j, name in enumerate(names):
            column = table.column(name).to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)
            matrix[:, start + j] = column[positions]
        logger.info("  matrix columns %d/%d", min(start + READ_CHUNK, len(eligible)), len(eligible))
        del table
    matrix.flush()

    kept, dropped = fs.hygiene_filter(matrix, eligible, prefer=read_names(CURRENT))
    (OUT / "columns.json").write_text(json.dumps(eligible), encoding="utf-8")
    (OUT / "hygiene.json").write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=2),
                                      encoding="utf-8")
    manifest = {"fe_signature": meta["fe_signature"], "data_signature": meta["data_signature"],
                "store_rows": meta["featured_rows"], "development_runners": int(len(targets)),
                "eligible": len(eligible), "kept_after_hygiene": len(kept),
                "soft_labels": soft is not None, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("hygiene kept %d of %d (dropped %d)", len(kept), len(eligible), len(dropped))


# ------------------------------------------------------------- the cache ---

class Cache:
    """The prepared matrix, row targets and folds, checked against the store."""

    def __init__(self):
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        current = store_meta(FeatureStore())
        if (manifest["fe_signature"], manifest["data_signature"]) != (current["fe_signature"],
                                                                     current["data_signature"]):
            raise SystemExit("The selection cache was built from a different store; rerun prepare.")
        self.manifest = manifest
        self.matrix = np.load(OUT / "matrix.npy", mmap_mode="r")
        self.columns = json.loads((OUT / "columns.json").read_text(encoding="utf-8"))
        self.index = {name: i for i, name in enumerate(self.columns)}
        self.kept = json.loads((OUT / "hygiene.json").read_text(encoding="utf-8"))["kept"]
        frame = pd.read_parquet(OUT / "targets.parquet")
        self.targets = fs.Targets(
            won=frame.won.to_numpy(np.int64),
            soft_y=frame.soft_y.to_numpy(np.float64) if manifest["soft_labels"] else None,
            race_ids=frame.race_id.to_numpy(), race_dates=pd.DatetimeIndex(frame.race_date),
            market=frame.market.to_numpy(np.float64),
        )
        self.folds = {f.label: f for f in expanding_period_folds(frame.race_date)}

    def take(self, rows: np.ndarray, names: list[str]) -> np.ndarray:
        missing = [n for n in names if n not in self.index]
        if missing:
            raise ValueError(f"Not in the prepared matrix: {missing[:10]}")
        return np.ascontiguousarray(self.matrix[np.ix_(rows, [self.index[n] for n in names])])

    def fold_data(self, labels: list[str]) -> list[fs.FoldData]:
        absent = [q for q in labels if q not in self.folds]
        if absent:
            raise SystemExit(f"No walk-forward fold for {absent}; available {sorted(self.folds)}")
        return [fs.FoldData(q, np.flatnonzero(self.folds[q].train), np.flatnonzero(self.folds[q].test))
                for q in labels]


def _save(directory: Path, name: str, result: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result["races"].to_parquet(directory / f"{name}_races.parquet", index=False)
    payload = {k: v for k, v in result.items() if k != "races"}
    if "importance" in payload:
        payload["importance"] = [float(v) for v in payload["importance"]]
    (directory / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load(directory: Path, name: str) -> dict | None:
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    result["races"] = pd.read_parquet(directory / f"{name}_races.parquet")
    if "importance" in result:
        result["importance"] = np.asarray(result["importance"], dtype=np.float64)
    return result


def _evaluate_cached(cache: Cache, directory: Path, name: str, features: list[str],
                     folds: list[fs.FoldData], args, **kwargs) -> dict:
    stored = _load(directory, name)
    if stored is not None and stored["features"] == features:
        return stored
    logger.info("Arm %s: %d features", name, len(features))
    result = fs.evaluate_subset(cache.take, features, folds, cache.targets,
                                seed=args.seed, n_jobs=args.threads, **kwargs)
    _save(directory, name, result)
    return result


# -------------------------------------------------------------- eliminate --

def eliminate(args) -> None:
    cache = Cache()
    folds = cache.fold_data(SELECTION_QUARTERS)
    directory = OUT / "selection"
    current = _current_inputs(cache)
    reference = _evaluate_cached(cache, directory, "reference_win_txt", current, folds, args,
                                 importance=False)
    _print_pooled("win.txt on selection quarters", reference)

    steps = fs.run_elimination(
        cache.take, list(cache.kept), folds, cache.targets, n_min=args.n_min,
        seed=args.seed, n_jobs=args.threads,
        on_step=lambda i, r: _save(directory, f"step_{i:02d}", r),
        resume=lambda i: _load(directory, f"step_{i:02d}"),
    )
    races = [s["races"] for s in steps]
    for frame in races[1:]:
        if not np.array_equal(frame.race_id.to_numpy(), races[0].race_id.to_numpy()):
            raise ValueError("Steps were not scored on the same races")
    losses = np.vstack([frame.race_log_loss.to_numpy() for frame in races])
    clusters = races[0].race_date.astype(str).to_numpy()
    choice = fs.one_se_choice([len(s["features"]) for s in steps], losses, clusters)

    ref = reference["races"]
    if not np.array_equal(ref.race_id.to_numpy(), races[0].race_id.to_numpy()):
        raise ValueError("Reference arm was not scored on the selection races")
    for row, step in zip(choice["steps"], steps):
        delta = step["races"].race_log_loss.to_numpy() - ref.race_log_loss.to_numpy()
        row["delta_vs_win_txt"] = float(delta.mean())
        row["se_vs_win_txt"] = fs.clustered_se(delta, clusters)
        row["race_mean_brier"] = step["pooled"]["race_mean_brier"]
        row["gap_to_market"] = step["pooled"].get("gap_to_market")
    chosen = steps[choice["chosen_step"]]
    best = steps[choice["best_step"]]
    summary = {"selection_quarters": SELECTION_QUARTERS, "win_txt": reference["pooled"],
               **choice, "chosen_features": chosen["features"], "best_features": best["features"]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ordered = fs.keep_strongest(chosen["features"], chosen["importance"], len(chosen["features"]))
    header = [f"# 048 walk-forward selection: {len(ordered)} of {len(cache.kept)} hygiene-kept inputs,",
              f"# one-SE step of out-of-sample race log loss on {SELECTION_QUARTERS[0]}-{SELECTION_QUARTERS[-1]}",
              f"# (scripts/select_features.py; docs/research/FEATURE_SELECTION_048.md). Strongest first.",
              f"# store fe_signature {cache.manifest['fe_signature']} data {cache.manifest['data_signature']}"]
    SELECTED.write_text("\n".join(header + ordered) + "\n", encoding="utf-8")
    (directory / "best.txt").write_text("\n".join(fs.keep_strongest(
        best["features"], best["importance"], len(best["features"]))) + "\n", encoding="utf-8")

    print("\nRecursive elimination, pooled over", ", ".join(SELECTION_QUARTERS))
    print(f"win.txt ({len(current)} inputs): race LL {reference['pooled']['race_log_loss']:.5f}")
    table = pd.DataFrame(choice["steps"])
    print(table[["step", "n_features", "race_log_loss", "delta_vs_best", "se", "within_one_se",
                 "delta_vs_win_txt", "se_vs_win_txt", "race_mean_brier", "gap_to_market"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.5f}" if abs(v) < 0.1 else f"{v:.5f}"))
    print(f"\nbest step {choice['best_step']} ({len(best['features'])} inputs); "
          f"chosen step {choice['chosen_step']} ({len(chosen['features'])} inputs) -> {SELECTED}")
    _tripwire(chosen["pooled"])


# ---------------------------------------------------------------- confirm --

def confirm(args) -> None:
    cache = Cache()
    folds = cache.fold_data(CONFIRM_QUARTERS)
    directory = OUT / f"confirm_{args.capacity}"
    summary = json.loads((OUT / "selection" / "summary.json").read_text(encoding="utf-8"))
    arms = {
        "current": _current_inputs(cache),
        "selected": read_names(SELECTED),
        "best": summary["best_features"],
        "all": list(cache.kept),
    }
    if set(arms["best"]) == set(arms["selected"]):
        arms.pop("best")
    wanted = [a for a in (args.arms or list(arms)) if a in arms]
    if "current" not in wanted:
        wanted.insert(0, "current")
    results = {name: _evaluate_cached(cache, directory, name, arms[name], folds, args,
                                      capacity=args.capacity, importance=False)
               for name in wanted}
    base = results["current"]["races"]
    rows = []
    for name, result in results.items():
        races = result["races"]
        if not np.array_equal(races.race_id.to_numpy(), base.race_id.to_numpy()):
            raise ValueError(f"{name} was not scored on the same races")
        ll = races.race_log_loss.to_numpy() - base.race_log_loss.to_numpy()
        brier = races.race_brier.to_numpy() - base.race_brier.to_numpy()
        dates = races.race_date.astype(str).to_numpy()
        ll_ci = paired_interval(ll, blocks=dates) if name != "current" else {"ci95": [0.0, 0.0]}
        br_ci = paired_interval(brier, blocks=dates) if name != "current" else {"ci95": [0.0, 0.0]}
        per_fold = races.assign(delta=ll).groupby("fold").delta.mean()
        rows.append({"arm": name, "n_features": len(result["features"]),
                     "race_log_loss": result["pooled"]["race_log_loss"],
                     "delta_ll": float(ll.mean()), "ll_ci_lo": ll_ci["ci95"][0], "ll_ci_hi": ll_ci["ci95"][1],
                     "race_brier": result["pooled"]["race_mean_brier"], "delta_brier": float(brier.mean()),
                     "brier_ci_lo": br_ci["ci95"][0], "brier_ci_hi": br_ci["ci95"][1],
                     "folds_better": int((per_fold < 0).sum()), "folds": int(len(per_fold)),
                     "gap_to_market": result["pooled"].get("gap_to_market")})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "comparison.csv", index=False)
    print(f"\nConfirmation ({args.capacity} capacity) on {', '.join(CONFIRM_QUARTERS)}; "
          "deltas are arm minus win.txt, negative is better, date-clustered 95% intervals:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.5f}" if abs(v) < 0.1 else f"{v:.5f}"))
    for name, result in results.items():
        _tripwire(result["pooled"], name)


# ----------------------------------------------------------------- shared --

def _current_inputs(cache: Cache) -> list[str]:
    names = read_names(CURRENT)
    missing = [n for n in names if n not in cache.index]
    if missing:
        logger.warning("win.txt inputs absent from the store, left out of its arm: %s", missing)
    return [n for n in names if n in cache.index]


def _print_pooled(label: str, result: dict) -> None:
    pooled = result["pooled"]
    print(f"{label}: {len(result['features'])} inputs, race LL {pooled['race_log_loss']:.5f}, "
          f"race Brier {pooled['race_mean_brier']:.6f}, gap to market {pooled.get('gap_to_market')}")


def _tripwire(pooled: dict, label: str = "chosen") -> None:
    gap = pooled.get("gap_to_market")
    if gap is not None and gap < 0:
        print(f"!! TRIPWIRE ({label}): the model beats the market's race log loss by {-gap:.4f}. "
              "A public-form model should not; look for a leaked input before trusting this.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "eliminate", "confirm"])
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-min", type=int, default=30, help="smallest set elimination visits")
    parser.add_argument("--capacity", choices=["screen", "native"], default="screen")
    parser.add_argument("--arms", nargs="*", help="confirm only these arms (win.txt is always included)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("src.model").setLevel(logging.WARNING)
    {"prepare": prepare, "eliminate": eliminate, "confirm": confirm}[args.stage](args)


if __name__ == "__main__":
    main()
