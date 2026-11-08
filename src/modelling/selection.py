"""Supported recursive selection using the actual win adapter and later confirmation."""
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import feature_selection as fs
from src.forecast_evaluation import paired_interval, probability_metrics
from src.ingestion.storage import atomic_json, atomic_parquet
from src.modelling.data import CONTEXT, ordered_frame
from src.modelling.evaluation import code_signature, frame_signature
from src.modelling.win import WinModel
from src.wf_eval import development_mask, expanding_period_folds, validate_forward_folds

logger = logging.getLogger(__name__)


def prepare_matrix(store_root, directory, *, start=None, end=None, batch_columns=32):
    """Create a column-batched float64 memmap, avoiding a wide frame in RAM."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.modelling.data import load_dataset
    from src.feature_store import FeatureStore
    from src.model import EXCLUDE_COLUMNS
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "matrix.npy").exists():
        raise ValueError("Selection matrix already exists; choose a new preparation directory")
    store = FeatureStore(str(store_root) if store_root else None)
    with store._access():
        schema = pq.read_schema(store.featured_path)
    names = [f.name for f in schema if f.name not in EXCLUDE_COLUMNS and
             (pa.types.is_integer(f.type) or pa.types.is_floating(f.type) or pa.types.is_boolean(f.type))]
    if not names:
        raise ValueError("No eligible numeric candidates")
    frame, receipt = load_dataset(store_root, features=[names[0]], start=start, end=end)
    context = frame.loc[:, [c for c in CONTEXT if c in frame]].copy()
    keys = pd.MultiIndex.from_frame(context[["race_id", "horse_id"]])
    matrix = np.lib.format.open_memmap(root / "matrix.npy", mode="w+", dtype=np.float64,
        shape=(len(frame), len(names)), fortran_order=True)
    filters = [("race_date", ">=", str(context.race_date.min().date())),
               ("race_date", "<=", str(context.race_date.max().date()))]
    date_type = schema.field("race_date").type
    if not (pa.types.is_string(date_type) or pa.types.is_large_string(date_type)):
        filters = [(c, op, pd.Timestamp(v)) for c, op, v in filters]
    with store._access():
        if store.meta().get("data_signature") != receipt["data_signature"]:
            raise RuntimeError("Feature generation changed during preparation")
        for first in range(0, len(names), batch_columns):
            batch = names[first:first + batch_columns]
            logger.info("Selection matrix: columns %d-%d of %d", first + 1, first + len(batch), len(names))
            part = pd.read_parquet(store.featured_path, columns=list(dict.fromkeys(["race_id", "horse_id", *batch])), filters=filters)
            source = pd.MultiIndex.from_frame(part[["race_id", "horse_id"]])
            if not source.is_unique:
                raise ValueError("Ambiguous runner identities in feature store")
            positions = source.get_indexer(keys)
            if (positions < 0).any():
                raise ValueError("Selection matrix lost a canonical runner")
            for j, name in enumerate(batch, first):
                values = part[name].to_numpy(dtype=np.float64, na_value=np.nan)[positions]
                if np.isinf(values).any():
                    raise ValueError(f"Infinite input in {name}")
                matrix[:, j] = values
        matrix.flush()
    atomic_parquet(context, root / "context.parquet")
    from src.modelling.catalog import feature_metadata
    atomic_json(root / "catalog.json", [feature_metadata(name) for name in names])
    atomic_json(root / "prepared.json", {"receipt": receipt, "features": names,
        "context_signature": frame_signature(context, context.columns), "rows": len(context),
        "matrix_signature": matrix_signature(matrix, names)})
    return context, matrix, names, receipt


def matrix_signature(matrix, names):
    """Hash one contiguous column at a time, including precision and column order."""
    digest = hashlib.sha256(json.dumps({"shape": matrix.shape, "dtype": str(matrix.dtype), "features": list(names)}).encode())
    for column in range(matrix.shape[1]):
        digest.update(np.ascontiguousarray(matrix[:, column]).tobytes())
    return digest.hexdigest()


def select_features(context, matrix, names, spec, *, selection_end, confirmation_start,
                    dataset_start=None, n_min=30, min_train_periods=8, capacity="screen", threads=1,
                    output=None):
    """Choose only on early folds, then compare frozen lists on later folds.

    Confirmation never changes the chosen list. No write to a configured feature
    list or global config occurs; the output is a new candidate specification.
    """
    context, order = ordered_frame(context)
    if not np.array_equal(order, np.arange(len(context))):
        raise ValueError("Selection matrix/context must already be in canonical order")
    if matrix.shape != (len(context), len(names)) or len(set(names)) != len(names):
        raise ValueError("Selection matrix shape/names disagree")
    if set(spec.features) - set(names):
        raise ValueError("Baseline inputs are absent from the selection matrix")
    if not development_mask(context.race_date).all():
        raise ValueError("Feature selection cannot use the reserved holdout")
    boundary, confirm = pd.Timestamp(selection_end), pd.Timestamp(confirmation_start)
    if confirm <= boundary + pd.Timedelta(days=spec.purge_days):
        raise ValueError("Selection and confirmation require a chronological purge gap")
    folds = expanding_period_folds(context.race_date, min_train_periods=min_train_periods,
        min_train_rows=100, embargo_days=spec.purge_days)
    validate_forward_folds(context, folds, embargo_days=spec.purge_days)
    selection = [replace(f, test=f.test & context.race_date.le(boundary).to_numpy()) for f in folds
                 if (f.test & context.race_date.le(boundary).to_numpy()).any()]
    confirmation = [replace(f, test=f.test & context.race_date.ge(confirm).to_numpy()) for f in folds
                    if (f.test & context.race_date.ge(confirm).to_numpy()).any()]
    if not selection or not confirmation:
        raise ValueError("Need earlier selection folds and separate later confirmation folds")
    # Label-free hygiene also stays before the later confirmation period.
    hygiene_rows = context.race_date.le(boundary).to_numpy()
    kept, rejected = fs.hygiene_filter(matrix, names, prefer=spec.features, rows=hygiene_rows, dtype=np.float64)
    if not kept:
        raise ValueError("No candidates survive feature hygiene")
    lookup = {name: i for i, name in enumerate(names)}
    source = code_signature()
    identity = {"code_signature": source, "recipe": spec.to_dict(), "matrix_signature": matrix_signature(matrix, names),
        "context_signature": frame_signature(context, context.columns), "selection_end": str(boundary),
        "confirmation_start": str(confirm), "n_min": n_min, "capacity": capacity,
        "dataset_start": str(dataset_start), "min_train_periods": min_train_periods}
    if output:
        manifest = Path(output) / "registration.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Selection output belongs to a different data/recipe registration")
        atomic_json(manifest, identity)
    def evaluate(features, chosen_folds, importance):
        recipe = spec.subset(features)
        scores, contributions = [], []
        for fold in chosen_folds:
            logger.info("Selection %s: %d features", fold.label, len(features))
            def part(mask):
                rows = np.flatnonzero(mask)
                base = context.iloc[rows].reset_index(drop=True)
                values = np.asarray(matrix[np.ix_(rows, [lookup[n] for n in features])])
                return base, values
            train, train_matrix = part(fold.train)
            model = WinModel(recipe).fit(train, dataset_start=dataset_start, capacity=capacity, threads=threads,
                matrix=train_matrix, matrix_columns=tuple(features))
            del train_matrix
            test, test_matrix = part(fold.test)
            p = model.predict(test, matrix=test_matrix, matrix_columns=tuple(features))
            groups = fs.race_groups(test.race_id)
            _, loss, brier = probability_metrics(p, test.won, groups)
            starts = np.r_[0, np.cumsum(groups)[:-1]]
            race = test.iloc[starts][["race_id", "race_date"]].copy()
            race["fold"], race["race_log_loss"], race["race_brier"] = fold.label, loss, brier
            scores.append(race)
            if importance:
                contributions.append(fs.mean_abs_shap(model.estimator, test_matrix))
            del test_matrix
        return pd.concat(scores, ignore_index=True), fs.aggregate_importance(contributions) if importance else None
    steps, current = [], kept
    for step, size in enumerate(fs.elimination_schedule(len(kept), n_min=min(n_min, len(kept)))):
        if step:
            current = fs.keep_strongest(current, steps[-1]["importance"], size)
        saved = Path(output) / f"selection_{step:02d}.json" if output else None
        if saved and saved.exists():
            prior = json.loads(saved.read_text())
            path = Path(output) / f"selection_{step:02d}.parquet"
            if prior["features"] != current or prior["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Selection cache identity/content mismatch")
            races, importance = pd.read_parquet(path), np.asarray(prior["importance"])
        else:
            races, importance = evaluate(current, selection, True)
        steps.append({"features": list(current), "importance": importance, "races": races})
        if output:
            atomic_parquet(races, Path(output) / f"selection_{step:02d}.parquet")
            atomic_json(Path(output) / f"selection_{step:02d}.json", {"features": current, "importance": importance.tolist(), "code_signature": source,
                "sha256": hashlib.sha256((Path(output) / f"selection_{step:02d}.parquet").read_bytes()).hexdigest()})
    loss = np.vstack([s["races"].race_log_loss.to_numpy() for s in steps])
    choice = fs.one_se_choice([len(s["features"]) for s in steps], loss, steps[0]["races"].race_date)
    selected = steps[choice["chosen_step"]]["features"]
    # Preserve baseline input order and selection's importance order explicitly.
    arms = {"baseline": list(spec.features), "selected": selected, "broad": kept}
    confirmation_rows, summary = {}, {}
    for arm, features in arms.items():
        if set(features) - lookup.keys():
            raise ValueError(f"Baseline references inputs absent from the matrix: {arm}")
        path = Path(output) / f"confirmation_{arm}.parquet" if output else None
        stamp = Path(output) / f"confirmation_{arm}.json" if output else None
        if stamp and stamp.exists():
            prior = json.loads(stamp.read_text())
            if prior["features"] != features or prior["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Confirmation cache identity/content mismatch")
            races = pd.read_parquet(path)
        else:
            races, _ = evaluate(features, confirmation, False)
        confirmation_rows[arm] = races
        summary[arm] = {"features": len(features), "races": len(races), "race_log_loss": float(races.race_log_loss.mean())}
        if output:
            atomic_parquet(races, Path(output) / f"confirmation_{arm}.parquet")
            atomic_json(stamp, {"features": features, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    baseline = confirmation_rows["baseline"]
    for arm in ("selected", "broad"):
        races = confirmation_rows[arm]
        if races.race_id.tolist() != baseline.race_id.tolist():
            raise ValueError("Confirmation arms scored different race universes")
        summary[arm]["paired_vs_baseline"] = paired_interval(races.race_log_loss - baseline.race_log_loss, blocks=races.race_date)
    result = {"choice": choice, "hygiene_rejections": rejected, "confirmation": summary,
        "selected_spec": replace(spec, features=tuple(selected)).to_dict(),
        "selection_through": str(boundary.date()), "confirmation_from": str(confirm.date()),
        "capacity": capacity, "code_signature": source,
        "evaluation_status": "retrospective confirmation on previously studied history; not a fresh holdout"}
    if code_signature() != source:
        raise RuntimeError("Modelling code changed during selection; results cannot qualify the current recipe")
    if output:
        atomic_json(Path(output) / "selection.json", result)
        atomic_json(Path(output) / "selected_spec.json", result["selected_spec"])
        path = Path(output) / "selected_features.txt"
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return result
