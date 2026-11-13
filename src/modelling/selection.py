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


IMPORTANCE = ("auto", "shap", "permutation")


def _evaluate(context, column, spec, features, folds, *, dataset_start, capacity, threads,
              importance=None, permutation_races=1500):
    """Fit *spec* on *features* for every fold and score its test races.

    *column(name)* returns that input for every row. *importance* is None, "shap" (how much the
    model leans on each input) or "permutation" (the race log loss each input prevents), both
    measured on the fold's unseen races.
    """
    recipe = spec.subset(features)
    scores, contributions = [], []
    for fold in folds:
        logger.info("Selection %s: %d features", fold.label, len(features))

        def part(mask):
            rows = np.flatnonzero(mask)
            values = np.empty((len(rows), len(features)), dtype=np.float64)
            for j, name in enumerate(features):
                values[:, j] = np.asarray(column(name))[rows]
            return context.iloc[rows].reset_index(drop=True), values
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
        if importance == "shap":
            contributions.append(fs.mean_abs_shap(model.estimator, test_matrix))
        elif importance == "permutation":
            # The fold's latest races, whole: enough to rank inputs at a fraction of the cost.
            skipped = max(0, len(groups) - permutation_races)
            first = int(starts[skipped])
            part_frame = test.iloc[first:].reset_index(drop=True)
            contributions.append(fs.permutation_importance(
                lambda m: model.predict(part_frame, matrix=m, matrix_columns=tuple(features)),
                test_matrix[first:], part_frame.won.to_numpy(), groups[skipped:], seed=spec.seed))
        elif importance is not None:
            raise ValueError(f"Unknown importance {importance!r}")
        del test_matrix
    races = pd.concat(scores, ignore_index=True)
    if importance == "shap":
        return races, fs.aggregate_importance(contributions)
    if importance == "permutation":
        return races, np.mean(contributions, axis=0)
    return races, None


def _split_folds(context, spec, boundary, confirm, min_train_periods):
    folds = expanding_period_folds(context.race_date, min_train_periods=min_train_periods,
        min_train_rows=100, embargo_days=spec.purge_days)
    validate_forward_folds(context, folds, embargo_days=spec.purge_days)
    early = [replace(f, test=f.test & context.race_date.le(boundary).to_numpy()) for f in folds
             if boundary is not None and (f.test & context.race_date.le(boundary).to_numpy()).any()]
    late = [replace(f, test=f.test & context.race_date.ge(confirm).to_numpy()) for f in folds
            if (f.test & context.race_date.ge(confirm).to_numpy()).any()]
    return early, late


class _Columns:
    """A view of chosen columns of a (memory-mapped) matrix, read one column at a time."""

    def __init__(self, matrix, positions):
        self.matrix, self.positions = matrix, list(positions)
        self.shape = (matrix.shape[0], len(self.positions))

    def __getitem__(self, key):
        rows, j = key
        return self.matrix[rows, self.positions[j]]


def select_features(context, matrix, names, spec, *, selection_end, confirmation_start,
                    dataset_start=None, n_min=30, min_train_periods=8, capacity="screen", threads=1,
                    output=None, near_copy=0.95, importance="auto", permutation_below=150,
                    confirm_capacity=None):
    """Choose only on early folds, then compare frozen lists on later folds.

    Confirmation never changes the chosen list. No write to a configured feature
    list or global config occurs; the output is a new candidate specification.

    *near_copy* drops all but one of each cluster of near-identical columns before any fit
    (None keeps them). *importance* ranks inputs for elimination: "shap", "permutation", or
    "auto", which uses the cheap SHAP share while more than *permutation_below* inputs remain
    and the loss each input prevents after that, where the choice is made. *confirm_capacity*
    repeats the baseline and selected arms at another capacity, normally "native" after a
    selection at "screen": a reduced-capacity fit can favour inputs that only make up for the
    capacity it lacks.
    """
    if importance not in IMPORTANCE:
        raise ValueError(f"importance must be one of {IMPORTANCE}")
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
    selection, confirmation = _split_folds(context, spec, boundary, confirm, min_train_periods)
    if not selection or not confirmation:
        raise ValueError("Need earlier selection folds and separate later confirmation folds")
    # Label-free hygiene also stays before the later confirmation period.
    hygiene_rows = context.race_date.le(boundary).to_numpy()
    kept, rejected = fs.hygiene_filter(matrix, names, prefer=spec.features, rows=hygiene_rows, dtype=np.float64)
    if not kept:
        raise ValueError("No candidates survive feature hygiene")
    lookup = {name: i for i, name in enumerate(names)}
    column = lambda name: matrix[:, lookup[name]]  # noqa: E731
    near_copies = {}
    if near_copy is not None:
        kept, near_copies = fs.near_copy_filter(_Columns(matrix, [lookup[n] for n in kept]), kept,
            threshold=near_copy, prefer=spec.features, rows=hygiene_rows, seed=spec.seed)
    source = code_signature()
    identity = {"code_signature": source, "recipe": spec.to_dict(), "matrix_signature": matrix_signature(matrix, names),
        "context_signature": frame_signature(context, context.columns), "selection_end": str(boundary),
        "confirmation_start": str(confirm), "n_min": n_min, "capacity": capacity,
        "dataset_start": str(dataset_start), "min_train_periods": min_train_periods,
        "near_copy": near_copy, "importance": importance, "permutation_below": permutation_below,
        "confirm_capacity": confirm_capacity}
    if output:
        manifest = Path(output) / "registration.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Selection output belongs to a different data/recipe registration")
        atomic_json(manifest, identity)
    settings = {"dataset_start": dataset_start, "threads": threads}

    def ranking(count):
        if importance != "auto":
            return importance
        return "permutation" if count <= permutation_below else "shap"
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
            races, ranked = pd.read_parquet(path), np.asarray(prior["importance"])
        else:
            races, ranked = _evaluate(context, column, spec, current, selection, capacity=capacity,
                                      importance=ranking(len(current)), **settings)
        steps.append({"features": list(current), "importance": ranked, "races": races})
        if output:
            atomic_parquet(races, Path(output) / f"selection_{step:02d}.parquet")
            atomic_json(Path(output) / f"selection_{step:02d}.json", {"features": current, "importance": ranked.tolist(),
                "ranking": ranking(len(current)), "code_signature": source,
                "sha256": hashlib.sha256((Path(output) / f"selection_{step:02d}.parquet").read_bytes()).hexdigest()})
    loss = np.vstack([s["races"].race_log_loss.to_numpy() for s in steps])
    choice = fs.one_se_choice([len(s["features"]) for s in steps], loss, steps[0]["races"].race_date)
    selected = steps[choice["chosen_step"]]["features"]
    # Preserve baseline input order and selection's importance order explicitly.
    arms = {"baseline": list(spec.features), "selected": selected, "broad": kept}

    def confirm_arms(chosen, level, prefix):
        rows, summary = {}, {}
        for arm in chosen:
            features = arms[arm]
            if set(features) - lookup.keys():
                raise ValueError(f"Baseline references inputs absent from the matrix: {arm}")
            path = Path(output) / f"{prefix}_{arm}.parquet" if output else None
            stamp = Path(output) / f"{prefix}_{arm}.json" if output else None
            if stamp and stamp.exists():
                prior = json.loads(stamp.read_text())
                if prior["features"] != features or prior["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("Confirmation cache identity/content mismatch")
                races = pd.read_parquet(path)
            else:
                races, _ = _evaluate(context, column, spec, features, confirmation, capacity=level, **settings)
            rows[arm] = races
            summary[arm] = {"features": len(features), "races": len(races), "race_log_loss": float(races.race_log_loss.mean())}
            if output:
                atomic_parquet(races, path)
                atomic_json(stamp, {"features": features, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        baseline = rows["baseline"]
        for arm in chosen:
            if arm == "baseline":
                continue
            races = rows[arm]
            if races.race_id.tolist() != baseline.race_id.tolist():
                raise ValueError("Confirmation arms scored different race universes")
            summary[arm]["paired_vs_baseline"] = paired_interval(races.race_log_loss - baseline.race_log_loss, blocks=races.race_date)
        return summary
    summary = confirm_arms(("baseline", "selected", "broad"), capacity, "confirmation")
    result = {"choice": choice, "hygiene_rejections": rejected, "near_copy_rejections": near_copies,
        "confirmation": summary,
        "selected_spec": replace(spec, features=tuple(selected)).to_dict(),
        "selection_through": str(boundary.date()), "confirmation_from": str(confirm.date()),
        "capacity": capacity, "code_signature": source,
        "evaluation_status": "retrospective confirmation on previously studied history; not a fresh holdout"}
    if confirm_capacity and confirm_capacity != capacity:
        # The broad arm is left out: hundreds of inputs at full capacity cost hours and decide nothing.
        result["confirmation_" + confirm_capacity] = confirm_arms(("baseline", "selected"), confirm_capacity,
                                                                  "confirmation_" + confirm_capacity)
    if code_signature() != source:
        raise RuntimeError("Modelling code changed during selection; results cannot qualify the current recipe")
    if output:
        atomic_json(Path(output) / "selection.json", result)
        atomic_json(Path(output) / "selected_spec.json", result["selected_spec"])
        path = Path(output) / "selected_features.txt"
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return result


def admit_features(context, matrix, names, spec, candidates, *, start, dataset_start=None,
                   min_train_periods=8, capacity="screen", threads=1, seed=11, output=None):
    """Would adding *candidates* to *spec* help? A forward test, without re-running elimination.

    Three arms on the folds from *start*: the recipe as it is, the recipe with the candidates,
    and a twin in which the candidates are shuffled within each quarter. The twin has the same
    number of inputs and the same marginal distributions, so the difference between the real
    arm and the twin is what the candidates' information is worth; the difference between the
    twin and the baseline is what extra inputs cost or gain by themselves. Nothing is selected
    or written to a configured list.
    """
    context, order = ordered_frame(context)
    if not np.array_equal(order, np.arange(len(context))):
        raise ValueError("Selection matrix/context must already be in canonical order")
    candidates = [c for c in dict.fromkeys(candidates) if c not in spec.features]
    lookup = {name: i for i, name in enumerate(names)}
    if not candidates or set(candidates) - lookup.keys() or set(spec.features) - lookup.keys():
        raise ValueError("Candidates must be new inputs present in the prepared matrix")
    if not development_mask(context.race_date).all():
        raise ValueError("Feature admission cannot use the reserved holdout")
    fixed = [c for c in candidates if c in context]
    if fixed:
        # The model checks such inputs against the context frame, so a shuffled twin cannot be built.
        raise ValueError(f"Candidates that are also context columns cannot be twinned: {fixed}")
    _, folds = _split_folds(context, spec, None, pd.Timestamp(start), min_train_periods)
    if not folds:
        raise ValueError("No folds from the requested start")
    quarter = context.race_date.dt.to_period("Q").astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = {}
    for name in candidates:
        values = np.asarray(matrix[:, lookup[name]], dtype=np.float64).copy()
        for label in np.unique(quarter):
            rows = np.flatnonzero(quarter == label)
            values[rows] = values[rows[rng.permutation(len(rows))]]
        shuffled[name] = values
    real = lambda name: matrix[:, lookup[name]]  # noqa: E731
    twin = lambda name: shuffled[name] if name in shuffled else matrix[:, lookup[name]]  # noqa: E731
    features = [*spec.features, *candidates]
    settings = {"dataset_start": dataset_start, "capacity": capacity, "threads": threads}
    races = {"baseline": _evaluate(context, real, spec, list(spec.features), folds, **settings)[0],
             "with_candidates": _evaluate(context, real, spec, features, folds, **settings)[0],
             "twin": _evaluate(context, twin, spec, features, folds, **settings)[0]}
    if len({tuple(r.race_id) for r in races.values()}) != 1:
        raise ValueError("Admission arms scored different race universes")

    def paired(a, b):
        change = races[a].race_log_loss - races[b].race_log_loss
        by_fold = change.groupby(races[a].fold.to_numpy()).mean()
        return {**paired_interval(change, blocks=races[a].race_date),
                "per_fold": {k: float(v) for k, v in by_fold.items()}, "folds_better": int((by_fold < 0).sum())}
    result = {"candidates": candidates, "capacity": capacity, "from": str(pd.Timestamp(start).date()),
        "folds": [f.label for f in folds], "races": int(len(races["baseline"])),
        "race_log_loss": {k: float(v.race_log_loss.mean()) for k, v in races.items()},
        "with_candidates_vs_baseline": paired("with_candidates", "baseline"),
        "with_candidates_vs_twin": paired("with_candidates", "twin"),
        "twin_vs_baseline": paired("twin", "baseline"),
        "evaluation_status": "retrospective comparison on previously studied history; not a fresh holdout"}
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        for arm, frame in races.items():
            atomic_parquet(frame, Path(output) / f"admission_{arm}.parquet")
        atomic_json(Path(output) / "admission.json", result)
    return result
