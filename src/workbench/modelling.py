"""Read saved modelling evidence and submit jobs to the existing modelling CLI."""
from contextlib import redirect_stdout
from datetime import date, timedelta
import hashlib
import io
import json
from math import isfinite
import os
from pathlib import Path
import re

from src.workbench.data import read_json
from src.workbench.features import Features


MODELLING_JOBS = {"model_benchmark", "model_fit"}
RUN_ID = re.compile(r"\d{8}_\d{6}(?:_\d{6})?")


def artifact_json(path):
    """Treat incomplete or unreadable local artifacts as unavailable."""
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def fitted_feature_importance(predictor):
    """Report the fitted LightGBM trees separately for each supported model head."""
    from src.modelling.candidate import WinCandidate
    from src.modelling.performance import PerformanceModel
    from src.modelling.win import WinModel

    heads = []
    if isinstance(predictor, WinCandidate):
        for name, model in predictor.models.items():
            if isinstance(model, WinModel):
                heads.append((name, model.estimator, model.spec.features))
            elif isinstance(model, PerformanceModel):
                heads.extend(((f"{name} · mean", model.mean, model.spec.features),
                              (f"{name} · spread", model.spread, model.spec.features)))
    else:
        heads.extend((("Win", predictor.clf_model, predictor.feature_cols),
                      ("Place", predictor.place_model, predictor.feature_cols)))

    components = []
    for label, estimator, features in heads:
        booster = getattr(estimator, "booster_", None)
        if booster is None:
            continue
        gains = booster.feature_importance("gain").tolist()
        splits = booster.feature_importance("split").tolist()
        if len(gains) != len(features) or len(splits) != len(features) or not all(isfinite(g) and g >= 0 for g in gains):
            raise ValueError("Fitted feature names and tree importance could not be aligned")
        total_gain = sum(gains)
        ranked = [{"feature": name, "gain_share": gain/total_gain if total_gain else 0.0, "splits": count}
                  for name, gain, count in zip(features, gains, splits, strict=True)]
        ranked.sort(key=lambda row: (-row["gain_share"], -row["splits"], row["feature"]))
        components.append({"label": label, "features": ranked})
    return {"components": components}


class Modelling:
    def __init__(self, catalog, features=None):
        self.catalog = catalog
        self.features = features or Features(catalog)
        self.runs_root = catalog.root/"runs"
        self.output = catalog.root/"modelling/workbench"

    def inputs(self, dataset):
        from src.wf_eval import golden_holdout_start, purge_days

        features = self.features.overview(dataset)
        store = features["store"]
        reasons = list(store["reasons"])
        if not features["selected_dataset"]:
            reasons.append("Select an accepted dataset before fitting.")
        if not store.get("data_signature") and store["state"] == "current":
            reasons.append("The store has no recorded training data signature.")
        last_date = store.get("last_date")
        if not last_date and store["state"] == "current":
            reasons.append("The store has no recorded settled cutoff.")
        latest_fit = min(last_date, (date.today()-timedelta(days=1)).isoformat()) if last_date else None
        end = (golden_holdout_start().date()-timedelta(days=purge_days()+1)).isoformat()
        return {"dataset": features["selected_dataset"], "store": store, "blockers": reasons,
                "latest_fit": latest_fit, "development_through": min(end, latest_fit) if latest_fit else None,
                "recipes": features["feature_sets"], "max_threads": os.cpu_count() or 1}

    def overview(self, dataset):
        inputs = self.inputs(dataset)
        sources = self.catalog.overview(inputs["dataset"])
        history = artifact_json(self.catalog.root/"processed/forecast_history/meta.json")
        latest = sources["database"]["latest_date"]
        through = history.get("last_date")
        lag = (date.fromisoformat(latest)-date.fromisoformat(through)).days if latest and through else None
        selected, runs = self.saved_runs(inputs["store"])
        return {**inputs, "latest_results": latest, "sources": sources["readiness"],
                "promoted": next((r for r in runs if r["id"] == selected), None),
                "promoted_id": selected,
                "latest_candidate": next((r for r in runs if r["type"] == "win_stack_candidate"), None),
                "history": {"through": through, "races": history.get("races"), "lag_days": lag,
                            "recipe_count": len(history.get("recipe_signatures", []))}}

    def benchmark_paths(self):
        for root in (self.catalog.root/"modelling", self.catalog.root/"research/meta_learner"):
            for pattern in ("registration.json", "*/registration.json", "*/*/registration.json"):
                for file in root.glob(pattern):
                    folder = file.parent
                    if folder.resolve() != folder or any(
                            (folder/name).resolve().parent != folder
                            for name in ("registration.json", "report.json", "oof.parquet")):
                        continue
                    relative = folder.relative_to(self.catalog.root).as_posix()
                    yield hashlib.sha256(relative.encode()).hexdigest()[:24], folder

    def benchmark_row(self, identifier, folder, inputs):
        from src.modelling.evaluation import code_signature

        registration = artifact_json(folder/"registration.json")
        receipt = registration.get("receipt") or {}
        specs = registration.get("specs") or []
        folds = registration.get("folds") or []
        if not specs or not folds or not receipt.get("dataset_manifest"):
            return None
        report = artifact_json(folder/"report.json")
        completed = bool(report.get("components") and report.get("oof_sha256") and (folder/"oof.parquet").is_file())
        last_scored = max(fold["test_last_date"] for fold in folds)
        dataset = receipt["dataset_manifest"]["id"]
        store = inputs["store"]
        reasons = list(inputs["blockers"])
        if not completed:
            reasons.append("The benchmark has no completed report and prediction file.")
        if dataset != inputs["dataset"]:
            reasons.append("Select the benchmark’s accepted dataset.")
        if (receipt.get("fe_signature"), receipt.get("data_signature")) != (store.get("signature"), store.get("data_signature")):
            reasons.append("The benchmark used a different feature-store generation.")
        if registration.get("code_signature") != self.catalog.cached("modelling_code", 5, code_signature):
            reasons.append("The benchmark used different modelling code. Run a new benchmark to fit a candidate.")
        if not inputs["latest_fit"] or inputs["latest_fit"] < last_scored:
            reasons.append("Settled feature history must cover the last scored race.")
        return {"id": identifier, "label": folder.relative_to(self.catalog.root).as_posix(),
                "selected_dataset": inputs["dataset"], "build_id": store.get("build_id"),
                "completed": completed, "fit_blockers": reasons, "latest_fit": inputs["latest_fit"],
                "capacity": registration.get("capacity"), "dataset": dataset,
                "feature_signature": receipt.get("fe_signature"), "data_signature": receipt.get("data_signature"),
                "excluded_races": receipt.get("excluded_races"),
                "last_scored": last_scored, "folds": folds, "specs": specs, "report": report,
                "evaluation_status": report.get("evaluation_status") or receipt.get("evaluation_status"),
                "max_threads": inputs["max_threads"]}

    def benchmarks(self, dataset):
        inputs = self.inputs(dataset)
        rows = []
        for identifier, folder in self.benchmark_paths():
            row = self.benchmark_row(identifier, folder, inputs)
            if row:
                rows.append({k: v for k, v in row.items() if k not in {"specs", "folds"}} | {
                    "components": [spec["id"] for spec in row["specs"]], "fold_count": len(row["folds"])})
        return {"rows": sorted(rows, key=lambda r: r["report"].get("created_at") or "", reverse=True),
                "inputs": inputs}

    def benchmark(self, identifier, dataset, inputs=None):
        if not re.fullmatch(r"[a-f0-9]{24}", identifier or ""):
            raise ValueError("Choose a saved benchmark")
        for key, folder in self.benchmark_paths():
            if key == identifier:
                row = self.benchmark_row(key, folder, inputs or self.inputs(dataset))
                if row:
                    return row, folder
        raise FileNotFoundError("Benchmark was not found")

    def saved_runs(self, store):
        promoted = artifact_json(self.catalog.base/"models/production.json").get("run_id")
        runs = []
        for file in sorted(self.runs_root.glob("*/meta.json"), reverse=True):
            identifier = file.parent.name
            if not RUN_ID.fullmatch(identifier) or file.resolve() != file:
                continue
            meta = artifact_json(file)
            if not meta or meta.get("run_id") != identifier:
                continue
            config = meta.get("training_config") or {}
            signatures = config.get("signatures") or {}
            split = config.get("split") or {}
            candidate = meta.get("model_type") == "win_stack_candidate"
            signature = signatures.get("fe_signature")
            matches = signature == store.get("signature") if signature and store.get("signature") else None
            has_model = (file.parent/"model.joblib").is_file()
            if candidate:
                promotion = "Win-only candidate. The promoted predictor requires win/place support."
            elif not has_model:
                promotion = "Saved model weights are unavailable."
            elif matches is False:
                promotion = "Feature definitions differ from the installed store."
            elif matches is None:
                promotion = "Feature compatibility could not be verified."
            else:
                promotion = "Feature definitions match. Promotion requires the serving compatibility check."
            runs.append({"id": identifier, "name": meta.get("name"), "type": meta.get("model_type"),
                         "created_at": meta.get("timestamp"), "promoted": identifier == promoted,
                         "trained_through": split.get("training_last_date"),
                         "features": meta.get("n_features"), "runners": meta.get("data_rows"),
                         "signature": signature, "matches": matches, "promotion": promotion,
                         "has_model": has_model, "dataset": (config.get("dataset_manifest") or {}).get("id"),
                         "fit_mode": split.get("mode"), "holdout_races": split.get("holdout_races")})
        return promoted, runs

    def runs(self, dataset):
        inputs = self.inputs(dataset)
        promoted, rows = self.saved_runs(inputs["store"])
        return {"rows": rows, "promoted_id": promoted}

    def run(self, identifier, dataset):
        row = next((r for r in self.runs(dataset)["rows"] if r["id"] == identifier), None)
        if row is None:
            raise FileNotFoundError("Saved run was not found")
        meta = artifact_json(self.runs_root/identifier/"meta.json")
        config = meta.get("training_config") or {}
        lineage = config.get("modelling") or {}
        reference = config.get("evaluation_reference")
        benchmark_id = next((key for key, folder in self.benchmark_paths()
                             if str((folder/"report.json").resolve()) == reference), None)
        return {**row, "benchmark_id": benchmark_id,
                "components": lineage.get("components", {}),
                "performance_claim": lineage.get("performance_claim"),
                "metrics": (config.get("evals") or {}).get("calibration", {}) if row["holdout_races"] else {}}

    def importance(self, identifier):
        from src.run_store import load_run_predictor, run_model_path

        if not RUN_ID.fullmatch(identifier):
            raise FileNotFoundError("Saved run was not found")
        folder = self.runs_root/identifier
        metadata, model = folder/"meta.json", folder/"model.joblib"
        if metadata.resolve() != metadata or model.resolve() != model:
            raise FileNotFoundError("Saved model weights were not found")
        meta = artifact_json(metadata)
        path = run_model_path(identifier)
        if meta.get("run_id") != identifier or not path or Path(path).resolve() != model:
            raise FileNotFoundError("Saved model weights were not found")
        stat = model.stat()
        key = ("model_importance", identifier, meta.get("model_type"), stat.st_mtime_ns, stat.st_size)

        def read_importance():
            predictor = load_run_predictor(identifier)
            if predictor is None:
                raise FileNotFoundError("Saved model weights were not found")
            return fitted_feature_importance(predictor)

        return self.catalog.cached(key, 60, read_importance)

    def validate_job(self, kind, params):
        inputs = self.inputs(params.get("dataset"))
        if inputs["blockers"]:
            raise ValueError(" ".join(inputs["blockers"]))
        store = inputs["store"]
        if params.get("build_id") != store["build_id"]:
            raise ValueError("The feature store changed. Refresh Modelling before queuing a job.")
        threads = params.get("threads", min(2, inputs["max_threads"]))
        if type(threads) is not int or not 1 <= threads <= inputs["max_threads"]:
            raise ValueError(f"Choose between 1 and {inputs['max_threads']} training threads")
        pinned = {"dataset": inputs["dataset"], "build_id": store["build_id"], "threads": threads}
        if kind == "model_benchmark":
            from src.modelling.spec import ModelSpec

            recipe = next((s for s in inputs["recipes"] if s["id"] == params.get("feature_set")), None)
            if not recipe:
                raise ValueError("Choose a registered feature set")
            if params.get("feature_set_signature") != recipe["signature"]:
                raise ValueError("The feature set changed. Refresh the benchmark form.")
            if recipe["missing_columns"]:
                raise ValueError("The selected feature set has inputs missing from this store.")
            capacity = params.get("capacity", "native")
            if capacity not in ("native", "screen"):
                raise ValueError("Choose native or screen capacity")
            fold_start = date.fromisoformat(params["fold_start"]).isoformat() if params.get("fold_start") else None
            if fold_start and fold_start > inputs["development_through"]:
                raise ValueError("The first scored date must be within the development period.")
            return {**pinned, "capacity": capacity, "fold_start": fold_start, "feature_set": recipe["id"],
                    "spec": ModelSpec("generalist", tuple(recipe["features"])).to_dict()}
        benchmark, folder = self.benchmark(params.get("benchmark"), inputs["dataset"], inputs)
        if benchmark["fit_blockers"]:
            raise ValueError(" ".join(benchmark["fit_blockers"]))
        cutoff = date.fromisoformat(params.get("as_of") or "").isoformat()
        if not benchmark["last_scored"] <= cutoff <= inputs["latest_fit"]:
            raise ValueError("The fit cutoff must cover the last scored race and be within settled feature history.")
        name = params.get("name", "")
        if not isinstance(name, str) or len(name) > 120:
            raise ValueError("Use a run name of at most 120 characters")
        return {**pinned, "benchmark": benchmark["id"], "as_of": cutoff,
                "benchmark_digest": benchmark_digest(folder), "name": name.strip() or f"Win candidate through {cutoff}"}


def benchmark_digest(folder):
    digest = hashlib.sha256()
    for name in ("registration.json", "report.json"):
        digest.update((folder/name).read_bytes())
    return digest.hexdigest()


def run_modelling_job(service, job, progress):
    from src.ingestion.storage import atomic_json
    from src.modelling.__main__ import main
    from src.source_dataset import use_manifest

    params = job["params"]
    inputs = service.inputs(params["dataset"])
    if inputs["blockers"] or inputs["store"].get("build_id") != params["build_id"]:
        raise ValueError("The pinned feature store is unavailable or changed. Refresh Modelling and queue again.")
    shared = ["--store-root", str(service.features.root), "--threads", str(params["threads"])]
    with use_manifest(service.catalog.manifest(params["dataset"])):
        if job["kind"] == "model_benchmark":
            output = service.output/job["id"]
            recipe = output/"recipe.json"
            atomic_json(recipe, params["spec"])
            args = ["benchmark", *shared, "--spec", str(recipe),
                    "--capacity", params["capacity"], "--output", str(output)]
            if params["fold_start"]:
                args.extend(["--fold-start", params["fold_start"]])
            progress(phase="benchmark", message="Fitting on earlier quarters and scoring later races. See the log for fold progress.")
            main(args)
            key = hashlib.sha256(output.relative_to(service.catalog.root).as_posix().encode()).hexdigest()[:24]
            return {"benchmark_id": key}
        benchmark, folder = service.benchmark(params["benchmark"], params["dataset"], inputs)
        if benchmark["fit_blockers"] or benchmark_digest(folder) != params["benchmark_digest"]:
            raise ValueError("The pinned benchmark became unavailable or changed. Review it and queue again.")
        progress(phase="fitting", message="Verifying benchmark predictions and fitting through the selected cutoff.")
        result = io.StringIO()
        try:
            with redirect_stdout(result):
                main(["fit", *shared, "--benchmark", str(folder), "--as-of", params["as_of"], f"--name={params['name']}"])
        finally:
            print(result.getvalue(), end="", flush=True)
        return {"run_id": json.loads(result.getvalue().strip().splitlines()[-1])["run_id"]}
