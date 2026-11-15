"""Compare the refreshed generalist, 063 recipes and the first win ensemble."""
import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from research.meta_learner import features_063
from src.feature_selection import race_groups
from src.feature_semantics import CLAIM_ADJUSTED_WEIGHT_INPUTS
from src.forecast_evaluation import paired_interval, probability_metrics
from src.ingestion.storage import atomic_json, atomic_parquet, writer_lock
from src.modelling.data import CONTEXT, load_dataset
from src.modelling.evaluation import code_signature, component_oof, nested_pool_evaluation
from src.modelling.performance import PerformanceSpec, TRAINING_COLUMNS
from src.modelling.spec import ModelSpec
from src.source_dataset import load_manifest, model_table, source_path, use_manifest
from src.wf_eval import expanding_period_folds, oof_predict_expanding


logger = logging.getLogger(__name__)
EARLIER = ("2024Q3", "2024Q4", "2025Q1", "2025Q2")
LATER = ("2025Q3", "2025Q4", "2026Q1", "2026Q2")
POOL_HISTORY = ("2025Q1", "2025Q2")
KEYS = ["race_id", "horse_id"]
TEXT_INPUTS = {"rtv_text_logit", "rtv_text_prob", "rtv_verdict_rank",
               "x063_rtv_text_logit_vs_field", "x063_rtv_text_logit_vs_best"}
RAW_TEXT = ["tf_comment", "tf_verdict"]
ARTIFACTS = Path("data/research/meta_learner")
BASELINE = ARTIFACTS / "baseline_native_2026q2/registration.json"
PERFORMANCE = ARTIFACTS / "performance_spread_20261108/performance_spec.json"
TEXT = Path("reference/modelling/timeform_text_market_v1.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Registration differs at {path}")
    else:
        atomic_json(path, value)


def prepare(root, store):
    if (root / "registration.json").exists():
        raise ValueError("Prepared experiment already exists")
    baseline = ModelSpec.from_dict(read_json(BASELINE)["specs"][0])
    columns = list(dict.fromkeys((*baseline.features, *features_063.NUMERIC_INPUTS)))
    frame, receipt = load_dataset(store, features=columns, label_columns=features_063.HISTORY)
    frame["race_id"] = frame.race_id.astype(str)
    frame["horse_id"] = pd.to_numeric(frame.horse_id, errors="raise").astype("int64")
    with use_manifest(load_manifest(receipt["dataset_manifest"]["id"], root=receipt["dataset_manifest"].get("root"))):
        raw = pd.read_parquet(source_path("sl", "results"), columns=[*KEYS, "going"])
    raw["race_id"] = raw.race_id.astype(str)
    raw["horse_id"] = pd.to_numeric(raw.horse_id, errors="raise").astype("int64")
    raw = frame[KEYS].merge(raw, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not raw._merge.eq("both").all():
        raise ValueError("A modelling runner is missing from the accepted raw results")
    rating = features_063.rating_gaps(frame)
    gaps = features_063.relative_gaps(frame)
    form = features_063.context_form(frame, raw.going)
    # Preserve the appended column order and aliases used by the archived recipe.
    moderators = frame[list(features_063.MODERATORS)].astype("float32").add_prefix("x063_context_")
    extra = pd.concat([rating, gaps, form, moderators], axis=1)
    if extra.shape != (len(frame), 64) or not extra.index.equals(frame.index):
        raise ValueError("The reconstructed 063 inputs are not aligned")
    compact = replace(baseline, id="rating-063", features=(*baseline.features, *rating.columns), soft_label_blend=.75)
    full = replace(compact, id="full-063", features=(*baseline.features, *extra.columns))
    keep = list(dict.fromkeys([*CONTEXT, *baseline.features, *TRAINING_COLUMNS, "handicap",
                               "official_rating", "has_official_rating", *CLAIM_ADJUSTED_WEIGHT_INPUTS]))
    prepared = pd.concat([frame[keep], extra], axis=1)
    prepared.attrs = {}
    atomic_parquet(prepared, root / "input.parquet")
    design = {"receipt": receipt, "input_sha256": sha(root / "input.parquet"), "model_code": code_signature(),
              "feature_reconstruction_sha256": sha(Path(features_063.__file__)), "rows": len(frame),
              "races": int(frame.race_id.nunique()), "specs": [s.to_dict() for s in (baseline, compact, full)],
              "earlier": list(EARLIER), "later": list(LATER), "pool_history": list(POOL_HISTORY),
              "repeat_seeds": [43, 44], "pool_shrinkage": .01, "text_recipe": read_json(TEXT),
              "performance_template": read_json(PERFORMANCE),
              "choice_rule": "Paired date-cluster 95% interval below zero and wins on at least three quarters. Prefer the smaller recipe unless full clears the same rule against it.",
              "scope": "Retrospective benchmark on an accepted store with repaired 063 chronology. No serving claim."}
    atomic_json(root / "registration.json", design)
    print(json.dumps({"rows": design["rows"], "races": design["races"], "recipes": [len(s.features) for s in (baseline, compact, full)]}))


def read_input(root, design, columns):
    names = list(dict.fromkeys([*CONTEXT, *columns]))
    frame = pd.read_parquet(root / "input.parquet", columns=names)
    frame.attrs["modelling_receipt"] = design["receipt"]
    return frame


def folds_for(frame, labels, purge, min_train_periods=8):
    folds = [f for f in expanding_period_folds(frame.race_date, embargo_days=purge,
             min_train_periods=min_train_periods) if f.label in labels]
    if set(f.label for f in folds) != set(labels):
        raise ValueError("A registered fold is missing from the accepted history")
    return folds


def forecasts(root, design, spec, labels, capacity, threads):
    targets = TRAINING_COLUMNS if isinstance(spec, PerformanceSpec) else ()
    frame = read_input(root, design, [*spec.features, *targets])
    folds = folds_for(frame, labels, spec.purge_days)
    result, _ = component_oof(frame, [spec], folds=folds, capacity=capacity, threads=threads,
        dataset_start=design["receipt"]["dataset_start"], cache_dir=root / "cache")
    result.attrs = {}
    return result


def metrics(frame, records, labels):
    period = frame.race_date.dt.to_period("Q").astype(str)
    target = frame.loc[period.isin(labels), [*KEYS, "race_date", "won"]].merge(
        records[[*KEYS, "probability"]], on=KEYS, how="left", validate="one_to_one")
    groups = race_groups(target.race_id)
    summary, loss, _ = probability_metrics(target.probability.to_numpy(), target.won.to_numpy(), groups)
    starts = np.r_[0, np.cumsum(groups)[:-1]]
    races = target.iloc[starts][["race_id", "race_date"]].assign(loss=loss).set_index("race_id")
    summary.update(races=len(races), runners=len(target),
                   per_quarter=races.groupby(races.race_date.dt.to_period("Q").astype(str)).loss.mean().to_dict())
    return summary, races


def difference(challenger, reference):
    if not challenger.index.equals(reference.index):
        raise ValueError("The compared forecasts did not score the same fields in order")
    delta = challenger.loss - reference.loss
    quarters = delta.groupby(reference.race_date.dt.to_period("Q").astype(str)).mean()
    return {**paired_interval(delta.to_numpy(), blocks=reference.race_date),
            "per_quarter": quarters.to_dict(), "quarters_better": int(quarters.lt(0).sum())}


def clears_gate(change):
    return change["ci95"][1] < 0 and change["quarters_better"] >= 3


def screen(root, design, threads):
    frame = read_input(root, design, [])
    specs = [ModelSpec.from_dict(s) for s in design["specs"]]
    summaries, losses = {}, {}
    for spec in specs:
        result = forecasts(root, design, spec, EARLIER, "screen", threads)
        summaries[spec.id], losses[spec.id] = metrics(frame, result, EARLIER)
    reference, compact, full = [s.id for s in specs]
    changes = {s: difference(losses[s], losses[reference]) for s in (compact, full)}
    changes["full_minus_compact"] = difference(losses[full], losses[compact])
    chosen = compact if clears_gate(changes[compact]) else reference
    if clears_gate(changes[full]) and (chosen == reference or clears_gate(changes["full_minus_compact"])):
        chosen = full
    report = {"chosen": chosen, "summaries": summaries, "changes": changes, "capacity": "screen"}
    atomic_json(root / "screen.json", report)
    print(json.dumps(report, indent=2))


def get_selected(root, design):
    chosen = read_json(root / "screen.json")["chosen"]
    return next(ModelSpec.from_dict(s) for s in design["specs"] if s["id"] == chosen)


def average_predictions(parts):
    first = parts[0].copy()
    for part in parts[1:]:
        if not first[KEYS].equals(part[KEYS]):
            raise ValueError("Repeat fits did not forecast identical runners")
    first["probability"] = np.mean([p.probability.to_numpy() for p in parts], axis=0)
    first["component"] = "seed-average"
    return first


def average_with_pool_history(root, design, selected, comparison, capacity, threads):
    seed_path = root / f"{capacity}_seed_average.parquet"
    if sha(seed_path) != comparison["forecast_sha256"]["seed_average"]:
        raise ValueError("Repeat-fit forecasts differ from the completed comparison")
    earlier = [forecasts(root, design, selected, POOL_HISTORY, capacity, threads)]
    for seed in design["repeat_seeds"]:
        spec = replace(selected, id=f"{selected.id}-seed-{seed}", seed=seed)
        earlier.append(forecasts(root, design, spec, POOL_HISTORY, capacity, threads))
    return pd.concat([average_predictions(earlier), pd.read_parquet(seed_path)], ignore_index=True)


def score_pool(frame, parts, labels, purge, shrinkage):
    # Restrict the stack to the exact race population covered by its primary model.
    dates = parts[0].race_date.unique()
    target = frame.loc[frame.race_date.isin(dates)].reset_index(drop=True)
    folds = folds_for(target, labels, purge, min_train_periods=len(POOL_HISTORY))
    names = tuple(p.component.iloc[0] for p in parts)
    report, runners, _ = nested_pool_evaluation(target, pd.concat(parts, ignore_index=True), names,
        folds=folds, purge_days=purge, shrinkage=shrinkage)
    summary, races = metrics(frame, runners.rename(columns={"pool_probability": "probability"}), labels)
    return {**summary, "pool_fits": report["fits"]}, races


def performance_spec(design, spec, name="performance"):
    return PerformanceSpec.from_dict({**design["performance_template"], "features": list(spec.features), "id": name})


def compare(root, design, capacity, threads):
    frame = read_input(root, design, [])
    selected = get_selected(root, design)
    reference = ModelSpec.from_dict(design["specs"][0])
    performance = performance_spec(design, selected)
    labels = (*POOL_HISTORY, *LATER)
    immutable_json(root / f"compare_{capacity}_registration.json", {"capacity": capacity,
        "specs": [selected.to_dict(), reference.to_dict(), performance.to_dict()],
        "repeat_seeds": design["repeat_seeds"], "folds": list(labels)})
    generalist = forecasts(root, design, selected, labels, capacity, threads)
    repeat = [forecasts(root, design, replace(selected, id=f"{selected.id}-seed-{seed}", seed=seed),
                        LATER, capacity, threads) for seed in design["repeat_seeds"]]
    averaged = average_predictions([generalist.loc[generalist.fold.isin(LATER)].reset_index(drop=True), *repeat])
    spread = forecasts(root, design, performance, labels, capacity, threads)
    summary, losses = {}, {}
    summary["generalist"], losses["generalist"] = metrics(frame, generalist, LATER)
    summary["seed_average"], losses["seed_average"] = metrics(frame, averaged, LATER)
    summary["generalist_plus_performance"], losses["generalist_plus_performance"] = score_pool(
        frame, [generalist, spread], LATER, selected.purge_days, design["pool_shrinkage"])
    if selected.id != reference.id:
        control = forecasts(root, design, reference, LATER, capacity, threads)
        summary["original_129"], losses["original_129"] = metrics(frame, control, LATER)
    changes = {f"{arm}_minus_generalist": difference(loss, losses["generalist"])
               for arm, loss in losses.items() if arm != "generalist"}
    changes["performance_pool_minus_seed_average"] = difference(losses["generalist_plus_performance"], losses["seed_average"])
    if "original_129" in losses:
        changes["generalist_minus_original_129"] = difference(losses["generalist"], losses["original_129"])
    output_hashes = {}
    for name, predictions in (("generalist", generalist), ("performance", spread), ("seed_average", averaged)):
        path = root / f"{capacity}_{name}.parquet"
        atomic_parquet(predictions, path)
        output_hashes[name] = sha(path)
    report = {"selected_recipe": selected.id, "capacity": capacity, "summaries": summary, "changes": changes,
              "forecast_sha256": output_hashes,
              "performance_clears_gate": clears_gate(changes["generalist_plus_performance_minus_generalist"])
                  and clears_gate(changes["performance_pool_minus_seed_average"])}
    atomic_json(root / f"compare_{capacity}.json", report)
    print(json.dumps(report, indent=2))


def expert_forecasts(root, design, frame, name, recipe, labels, purge, fit_predict):
    fits = []
    for fold in folds_for(frame, labels, purge):
        fit_path = root / f"{name}_{fold.label}.parquet"
        fit_receipt = fit_path.with_suffix(".json")
        expected = {"dataset": design["receipt"]["dataset_manifest"]["id"], "input": design["input_sha256"], "recipe": recipe,
                    "code": design["model_code"], "fold": fold.label, "purge": purge}
        target = frame.loc[fold.test, [*KEYS, "race_date", "won"]].reset_index(drop=True)
        if fit_receipt.exists():
            saved = read_json(fit_receipt)
            if saved["identity"] != expected or saved["sha256"] != sha(fit_path):
                raise ValueError(f"{name} forecast receipt mismatch")
            part = pd.read_parquet(fit_path)
        else:
            lineage = {}

            def predict(train, test):
                p, details = fit_predict(train, test)
                lineage.update(details)
                return p

            logger.info("%s %s: %d earlier / %d later runners", name, fold.label, fold.train.sum(), fold.test.sum())
            p = oof_predict_expanding(frame, fit_predict=predict, folds=[fold], embargo_days=purge)[fold.test]
            boundary = target.race_date.min() - pd.Timedelta(days=purge)
            if pd.Timestamp(lineage["training_last_date"]) > boundary:
                raise ValueError(f"{name} fit violates the purge")
            part = target[[*KEYS, "race_date"]].assign(component=name, fold=fold.label,
                probability=p, available=np.isfinite(p))
            part.attrs = {}
            atomic_parquet(part, fit_path)
            atomic_json(fit_receipt, {"identity": expected, "sha256": sha(fit_path), "lineage": lineage})
        if not part[KEYS].equals(target[KEYS]) or np.isinf(part.probability).any():
            raise ValueError(f"{name} forecast alignment or probability error")
        if part.available.groupby(part.race_id).nunique().gt(1).any():
            raise ValueError(f"{name} returned a partially available field")
        support = part.available.to_numpy()
        if support.any():
            probability_metrics(part.probability.to_numpy()[support], target.won.to_numpy()[support],
                                race_groups(target.race_id.to_numpy()[support]))
        fits.append(part)
    result = pd.concat(fits, ignore_index=True)
    atomic_parquet(result, root / f"{name}_oof.parquet")
    return result


def native_comparison(root, design, threads):
    from src.modelling.win import WinModel

    reference = ModelSpec.from_dict(design["specs"][0])
    selected = get_selected(root, design)
    if selected.id == reference.id:
        raise ValueError("Native comparison needs a distinct screening finalist")
    repeat = [replace(selected, id=f"{selected.id}-seed-{seed}", seed=seed) for seed in design["repeat_seeds"]]
    specs = [reference, selected, *repeat]
    registration = {"specs": [s.to_dict() for s in specs], "capacity": "native", "threads": threads,
                    "folds": list(LATER), "input_sha256": design["input_sha256"], "model_code": design["model_code"],
                    "driver_sha256": sha(Path(__file__)), "screen_sha256": sha(root / "screen.json"),
                    "selection_rule": "Advance a challenger only if the paired date-bootstrap interval is below zero and it wins at least three quarters. Require the average to clear both single-fit comparisons."}
    immutable_json(root / "native_registration.json", registration)
    forecasts_by_id, timing = {}, {}
    for spec in specs:
        frame = read_input(root, design, spec.features)
        name = f"native-{spec.id}"
        recipe = {"spec": spec.to_dict(), "capacity": "native", "threads": threads,
                  "driver_sha256": registration["driver_sha256"]}

        def fit_predict(train, test, spec=spec):
            start = perf_counter()
            model = WinModel(spec).fit(train, dataset_start=design["receipt"]["dataset_start"], capacity="native", threads=threads)
            fit_seconds = perf_counter() - start
            inputs = test.drop(columns=["won", "finish_position", "odds", "bsp"])
            start = perf_counter()
            probability = model.predict(inputs)
            predict_seconds = perf_counter() - start
            logger.info("Native %s: %d trees, %.1fs fit, %.3fs forecast", spec.id,
                        model.lineage["selected_rounds"], fit_seconds, predict_seconds)
            return probability, {**model.lineage, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds}

        forecasts_by_id[spec.id] = expert_forecasts(root, design, frame, name, recipe, LATER, spec.purge_days, fit_predict)
        timing[spec.id] = {}
        for label in LATER:
            lineage = read_json(root / f"{name}_{label}.json")["lineage"]
            timing[spec.id][label] = {k: lineage[k] for k in ("fit_seconds", "predict_seconds", "selected_rounds", "temperature")}
        del frame

    seed_average = average_predictions([forecasts_by_id[s.id] for s in [selected, *repeat]])
    average_path = root / "native_seed_average.parquet"
    atomic_parquet(seed_average, average_path)
    arms = {"original_129": forecasts_by_id[reference.id], "full_063": forecasts_by_id[selected.id], "seed_average": seed_average}
    frame = read_input(root, design, [])
    summary, losses = {}, {}
    for name, prediction in arms.items():
        summary[name], losses[name] = metrics(frame, prediction, LATER)
    changes = {"full_minus_original": difference(losses["full_063"], losses["original_129"]),
               "average_minus_original": difference(losses["seed_average"], losses["original_129"]),
               "average_minus_full": difference(losses["seed_average"], losses["full_063"])}
    chosen = "original_129"
    if clears_gate(changes["full_minus_original"]):
        chosen = "full_063"
    if clears_gate(changes["average_minus_original"]) and clears_gate(changes["average_minus_full"]):
        chosen = "seed_average"
    output_hashes = {"original_129": sha(root / f"native-{reference.id}_oof.parquet"),
                     "full_063": sha(root / f"native-{selected.id}_oof.parquet"), "seed_average": sha(average_path)}
    if code_signature() != design["model_code"] or sha(Path(__file__)) != registration["driver_sha256"]:
        raise ValueError("Modelling or benchmark code changed during the native fits")
    report = {"summaries": summary, "changes": changes, "chosen": chosen, "timing_by_fit": timing,
              "forecast_sha256": output_hashes, "capacity": "native", "threads": threads}
    atomic_json(root / "native_comparison.json", report)
    print(json.dumps(report, indent=2))


def text_forecasts(root, design, labels, purge):
    from src.modelling.text import TextExpert

    frame = read_input(root, design, [])
    manifest = design["receipt"]["dataset_manifest"]
    with use_manifest(load_manifest(manifest["id"], root=manifest.get("root"))):
        raw = model_table("rtv", "text")[["sl_race_id", "sl_horse_id", *RAW_TEXT]].rename(
            columns={"sl_race_id": "race_id", "sl_horse_id": "horse_id"})
    raw["race_id"], raw["horse_id"] = raw.race_id.astype(str), pd.to_numeric(raw.horse_id, errors="raise")
    frame = frame.merge(raw, on=KEYS, how="left", validate="one_to_one")

    def fit_predict(train, test):
        expert = TextExpert(**design["text_recipe"]).fit(train)
        return expert.predict(test.drop(columns=["won", "finish_position", "bsp", "odds"])), expert.lineage

    return expert_forecasts(root, design, frame, "text", design["text_recipe"], labels, purge, fit_predict)


def text_comparison(root, design, capacity, threads):
    selected = get_selected(root, design)
    no_text = replace(selected, id=f"{selected.id}-no-text", features=tuple(c for c in selected.features if c not in TEXT_INPUTS))
    frame = read_input(root, design, [])
    labels = (*POOL_HISTORY, *LATER)
    compare_report = read_json(root / f"compare_{capacity}.json")
    include_spread = compare_report["performance_clears_gate"]
    plain = forecasts(root, design, no_text, labels, capacity, threads)
    existing = forecasts(root, design, selected, labels, capacity, threads)
    text = text_forecasts(root, design, labels, selected.purge_days)
    parts = {"legacy_text_inputs": [existing], "without_legacy_text": [plain]}
    if include_spread:
        parts["legacy_text_inputs"].append(forecasts(root, design, performance_spec(design, selected), labels, capacity, threads))
        parts["without_legacy_text"].append(forecasts(root, design, performance_spec(design, no_text, "performance-no-text"), labels, capacity, threads))
    summary, losses = {}, {}
    for name, base in parts.items():
        for arm, components in ((name, base), (name + "_plus_text_expert", [*base, text])):
            summary[arm], losses[arm] = score_pool(frame, components, LATER, selected.purge_days, design["pool_shrinkage"])
    changes = {f"{arm}_minus_legacy_text_inputs": difference(loss, losses["legacy_text_inputs"])
               for arm, loss in losses.items() if arm != "legacy_text_inputs"}
    changes["text_expert_on_no_text_base"] = difference(losses["without_legacy_text_plus_text_expert"], losses["without_legacy_text"])
    seed_path = root / f"{capacity}_seed_average.parquet"
    if sha(seed_path) != compare_report["forecast_sha256"]["seed_average"]:
        raise ValueError("Repeat-fit forecasts differ from the completed comparison")
    _, seed_loss = metrics(frame, pd.read_parquet(seed_path), LATER)
    vs_seed = {arm: difference(loss, seed_loss) for arm, loss in losses.items()}
    report = {"capacity": capacity, "with_performance": include_spread, "summaries": summary, "changes": changes,
              "differences_vs_seed_average": vs_seed,
              "supported_text_fields": int(text.loc[text.available & text.fold.isin(LATER), "race_id"].nunique()),
              "removed_columns": sorted(set(selected.features) & TEXT_INPUTS)}
    atomic_json(root / f"text_comparison_{capacity}.json", report)
    print(json.dumps(report, indent=2))


def leeway_comparison(root, design, capacity, threads):
    from research.meta_learner import handicap_leeway

    selected = get_selected(root, design)
    labels = (*POOL_HISTORY, *LATER)
    comparison = read_json(root / f"compare_{capacity}.json")
    text_test = read_json(root / f"text_comparison_{capacity}.json")
    chosen_text_arm = "legacy_text_inputs"
    qualified = [(delta["mean"], name.removesuffix("_minus_legacy_text_inputs"))
                 for name, delta in text_test["changes"].items()
                 if name.endswith("_minus_legacy_text_inputs") and clears_gate(delta)
                 and clears_gate(text_test["differences_vs_seed_average"][name.removesuffix("_minus_legacy_text_inputs")])]
    if qualified:
        chosen_text_arm = min(qualified)[1]
    use_average = not qualified and not comparison["performance_clears_gate"]
    if chosen_text_arm.startswith("without_legacy_text"):
        selected = replace(selected, id=f"{selected.id}-no-text", features=tuple(c for c in selected.features if c not in TEXT_INPUTS))
    if use_average:
        base = [average_with_pool_history(root, design, selected, comparison, capacity, threads)]
    else:
        base = [forecasts(root, design, selected, labels, capacity, threads)]
    if comparison["performance_clears_gate"]:
        name = "performance-no-text" if chosen_text_arm.startswith("without_legacy_text") else "performance"
        base.append(forecasts(root, design, performance_spec(design, selected, name), labels, capacity, threads))
    if chosen_text_arm.endswith("plus_text_expert"):
        base.append(text_forecasts(root, design, labels, selected.purge_days))
    frame = read_input(root, design, [*selected.features, *TRAINING_COLUMNS, *handicap_leeway.INPUTS])
    summary, before = score_pool(frame, base, LATER, selected.purge_days, design["pool_shrinkage"])
    repeat_text = None
    if use_average:
        text = text_forecasts(root, design, labels, selected.purge_days)
        with_text, text_losses = score_pool(frame, [*base, text], LATER, selected.purge_days, design["pool_shrinkage"])
        change = difference(text_losses, before)
        repeat_text = {"summary": with_text, "change": change}
        if clears_gate(change):
            base.append(text)
            chosen_text_arm = "legacy_text_inputs_plus_text_expert"
            summary, before = with_text, text_losses
    recipe = {"spec": selected.to_dict(), "capacity": capacity, "leeway_code": sha(Path(handicap_leeway.__file__)),
              "base": [p.component.iloc[0] for p in base], "chosen_text_arm": chosen_text_arm}
    immutable_json(root / f"leeway_{capacity}_registration.json", recipe)

    def fit_predict(train, test):
        return handicap_leeway.fit_predict(train, test.drop(columns=["won", "finish_position", "beaten_lengths", "bsp", "odds"]),
            selected, design["receipt"]["dataset_start"], capacity, threads)

    leeway = expert_forecasts(root, design, frame, f"leeway-{capacity}", recipe, labels, selected.purge_days, fit_predict)
    added, after = score_pool(frame, [*base, leeway], LATER, selected.purge_days, design["pool_shrinkage"])
    support = leeway.loc[leeway.fold.isin(LATER) & leeway.available, "race_id"].unique()
    report = {"capacity": capacity, "base_components": recipe["base"], "base": summary, "repeat_average_plus_text": repeat_text,
              "plus_leeway": added, "change": difference(after, before),
              "supported_fields": len(support), "supported_field_change": difference(after.loc[support], before.loc[support]),
              "evaluation_status": "Exploratory retrospective comparison. Baseline choice used the same historical periods."}
    atomic_json(root / f"leeway_{capacity}.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "screen", "compare", "text", "leeway", "native"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--store-root")
    parser.add_argument("--capacity", choices=("screen", "native"), default="screen")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root / ".benchmark.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another phase owns this benchmark directory")
        if args.stage == "prepare":
            if not args.store_root:
                parser.error("prepare requires --store-root")
            prepare(root, args.store_root)
            return
        design = read_json(root / "registration.json")
        if (code_signature() != design["model_code"] or sha(root / "input.parquet") != design["input_sha256"]
                or sha(Path(features_063.__file__)) != design["feature_reconstruction_sha256"]):
            raise ValueError("Registered code or prepared input changed")
        if args.stage == "screen":
            screen(root, design, args.threads)
        elif args.stage == "compare":
            compare(root, design, args.capacity, args.threads)
        elif args.stage == "text":
            text_comparison(root, design, args.capacity, args.threads)
        elif args.stage == "native":
            native_comparison(root, design, args.threads)
        else:
            leeway_comparison(root, design, args.capacity, args.threads)


if __name__ == "__main__":
    main()
