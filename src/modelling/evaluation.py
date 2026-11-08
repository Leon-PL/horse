"""Shared chronological component predictions and nested pool evaluation."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_selection import race_groups
from src.forecast_evaluation import probability_metrics, paired_interval
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.data import ordered_frame
from src.modelling.pool import LogPool
from src.modelling.components import component_model, training_columns
from src.wf_eval import expanding_period_folds, oof_predict_expanding, validate_forward_folds

logger = logging.getLogger(__name__)


def code_signature():
    """Only modelling consumers/math, not a new source-pinned study system."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/modelling").glob("*.py")) + [root / "src" / name for name in (
        "race_objectives.py", "probability_calibration.py", "wf_eval.py", "model.py", "betfair.py", "feature_selection.py", "rtv_text.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def frame_signature(frame, columns):
    digest = hashlib.sha256()
    names = list(dict.fromkeys(columns))
    digest.update(json.dumps([(c, str(frame[c].dtype)) for c in names]).encode())
    for name in names:
        digest.update(pd.util.hash_pandas_object(frame[name], index=False).to_numpy().tobytes())
    return digest.hexdigest()


def component_oof(frame, specs, *, folds=None, capacity="native", threads=1, dataset_start=None,
                  cache_dir=None, min_train_periods=8):
    """Produce keyed OOF records. A failed/corrupt cache cannot become a fit input."""
    frame, order = ordered_frame(frame)
    if folds is not None:
        from dataclasses import replace
        folds = [replace(f, train=f.train[order], test=f.test[order]) for f in folds]
    if not specs or len({s.id for s in specs}) != len(specs):
        raise ValueError("Expected unique component specifications")
    gaps = {s.purge_days for s in specs}
    if len(gaps) != 1:
        raise ValueError("Components must share the experiment's chronological purge")
    gap = next(iter(gaps))
    folds = expanding_period_folds(frame.race_date, min_train_periods=min_train_periods,
                                   embargo_days=gap) if folds is None else list(folds)
    validate_forward_folds(frame, folds, embargo_days=gap)
    if not folds:
        raise ValueError("No usable component folds")
    source = code_signature()
    result, models = [], {}
    first_to_fold = {int(np.flatnonzero(f.test)[0]): f for f in folds}
    for spec in specs:
        def predict(train, test):
            fold = first_to_fold[int(test.index[0])]
            logger.info("Component %s, %s: %d train / %d predict runners", spec.id, fold.label, len(train), len(test))
            columns = list(spec.features) + [c for c in ("race_id", "horse_id", "horse_name", "race_date", "num_runners", "won", "finish_position", "bsp", "odds") if c in frame] + list(training_columns([spec]))
            identity = {"recipe": spec.signature, "code": source, "capacity": capacity,
                        "dataset_start": str(dataset_start), "train": frame_signature(train, columns),
                        "test": frame_signature(test, columns), "fold": fold.label,
                        "dataset": (frame.attrs.get("modelling_receipt") or {}).get("dataset_manifest", {}).get("id")}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            path = Path(cache_dir) / key if cache_dir else None
            if path and (path / "receipt.json").exists():
                receipt = json.loads((path / "receipt.json").read_text())
                stored = path / "predictions.parquet"
                if receipt["identity"] != identity or hashlib.sha256(stored.read_bytes()).hexdigest() != receipt["sha256"]:
                    raise ValueError("OOF cache receipt/content mismatch")
                records = pd.read_parquet(stored)
                if records[["race_id", "horse_id"]].astype(str).values.tolist() != test[["race_id", "horse_id"]].astype(str).values.tolist():
                    raise ValueError("OOF cache runner alignment mismatch")
            else:
                model = component_model(spec).fit(train, dataset_start=dataset_start, capacity=capacity, threads=threads)
                targets = [c for c in training_columns([spec]) if c not in spec.features]
                probabilities = model.predict(test.drop(columns=["won", "finish_position", *targets], errors="ignore"))
                records = test[["race_id", "horse_id", "race_date"]].copy()
                records["component"], records["fold"] = spec.id, fold.label
                records["probability"], records["available"] = probabilities, True
                records["training_last_date"] = model.lineage["training_last_date"]
                records["calibration_last_date"] = model.lineage["calibration_last_date"]
                records["recipe_signature"] = spec.signature
                records["cache_key"] = key
                if path:
                    atomic_parquet(records, path / "predictions.parquet")
                    atomic_json(path / "receipt.json", {"identity": identity, "lineage": model.lineage,
                        "sha256": hashlib.sha256((path / "predictions.parquet").read_bytes()).hexdigest()})
                models[spec.id] = model
            probability_metrics(records.probability.to_numpy(), test.won.to_numpy(), race_groups(test.race_id))
            boundary = pd.to_datetime(records.race_date).min() - pd.Timedelta(days=gap)
            if (pd.to_datetime(records.training_last_date) > boundary).any() or (pd.to_datetime(records.calibration_last_date) > boundary).any():
                raise ValueError("Cached predictions violate chronology")
            result.append(records.reset_index(drop=True))
            return records.probability.to_numpy()
        oof_predict_expanding(frame, fit_predict=predict, folds=folds, embargo_days=gap, min_train_rows=1)
    if code_signature() != source:
        raise RuntimeError("Modelling code changed during fitting; rerun with the final implementation")
    return pd.concat(result, ignore_index=True), models


def prediction_matrix(frame, records, components):
    """Align by canonical identities; never drop a runner to accommodate an expert."""
    if records.duplicated(["race_id", "horse_id", "component"]).any():
        raise ValueError("Duplicate component runner prediction")
    if set(records.component) - set(components):
        raise ValueError("Prediction table contains an unknown component")
    keys = ["race_id", "horse_id"]
    matrix = []
    target = frame[keys].copy()
    for component in components:
        part = records.loc[records.component.eq(component)].copy()
        if "available" in part:
            part.loc[~part.available, "probability"] = np.nan
        joined = target.merge(part[keys + ["probability"]], on=keys, how="left", validate="one_to_one")
        matrix.append(joined.probability.to_numpy())
    return np.column_stack(matrix)


def nested_pool_evaluation(frame, records, components, *, folds=None, min_train_periods=4, purge_days=7, shrinkage=.01):
    """Fit pool weights on earlier OOF predictions, score later OOF races only."""
    frame, _ = ordered_frame(frame)
    matrix = prediction_matrix(frame, records, components)
    # Burn-in rows with no generalist OOF forecast stay outside meta fitting.
    complete = pd.Series(np.isfinite(matrix[:, 0])).groupby(frame.race_id).transform("all").to_numpy()
    if not complete.any():
        raise ValueError("No complete generalist OOF fields")
    if folds is not None and not complete.all():
        raise ValueError("Explicit meta folds must refer to the OOF-covered frame")
    frame, matrix = frame.loc[complete].reset_index(drop=True), matrix[complete]
    columns = [f"_component_{i}" for i in range(len(components))]
    joined = pd.concat([frame, pd.DataFrame(matrix, columns=columns)], axis=1)
    folds = expanding_period_folds(joined.race_date, min_train_periods=min_train_periods,
                                   min_train_rows=100, embargo_days=purge_days) if folds is None else folds
    if not folds:
        raise ValueError("Not enough earlier OOF periods to evaluate a fitted pool")
    fits = []
    def fit_predict(train, test):
        pool = LogPool(tuple(components), shrinkage=shrinkage).fit(train[columns], train.won, train.race_id)
        fits.append({"training_last_date": str(train.race_date.max().date()),
                     "test_first_date": str(test.race_date.min().date()), "weights": dict(zip(components, pool.weights))})
        return pool.predict(test[columns], test.race_id)
    probabilities = oof_predict_expanding(joined, fit_predict=fit_predict, folds=folds,
        embargo_days=purge_days, min_train_rows=1)
    scored = np.isfinite(probabilities)
    target = joined.loc[scored].reset_index(drop=True)
    groups = race_groups(target.race_id)
    pooled, losses, _ = probability_metrics(probabilities[scored], target.won, groups)
    baseline, baseline_losses, _ = probability_metrics(target[columns[0]], target.won, groups)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    races = target.iloc[starts][["race_id", "race_date"]].copy()
    races["race_log_loss"], races["baseline_log_loss"] = losses, baseline_losses
    target["pool_probability"] = probabilities[scored]
    return {"pool": pooled, "baseline": baseline, "fits": fits,
            "paired_difference": paired_interval(losses - baseline_losses, blocks=races.race_date),
            "evaluation_status": "retrospective nested walk-forward; not untouched confirmation"}, target, races
