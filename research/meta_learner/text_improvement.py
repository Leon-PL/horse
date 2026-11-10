"""Registered chronological text-model development on accepted raw observations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.storage import atomic_json, atomic_parquet


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidates():
    base = {"penalty": 3e-5, "min_df": 20, "ngram_max": 2, "roles": False, "separate_fields": False,
            "half_life_days": None, "stage_weights": [1.], "soft_label_blend": 0., "characters": False}
    recipes = {"corrected": base}
    roles = {**base, "roles": True}
    blocks = {**base, "separate_fields": True}
    combined = {**roles, "separate_fields": True}
    tri = {**combined, "ngram_max": 3}
    recipes.update(roles=roles, blocks=blocks, roles_blocks=combined,
        blocks_weak_l2={**combined, "penalty": 1e-5}, blocks_strong_l2={**combined, "penalty": 1e-4},
        trigrams=tri, trigrams_weak_l2={**tri, "penalty": 1e-5}, trigrams_strong_l2={**tri, "penalty": 1e-4},
        rare_terms={**combined, "min_df": 5}, recent_2y={**tri, "half_life_days": 730.},
        recent_4y={**tri, "half_life_days": 1460.}, finishing_order={**tri, "stage_weights": [1., .25, .0625]},
        soft_quarter={**tri, "soft_label_blend": .25}, soft_half={**tri, "soft_label_blend": .5},
        characters={**combined, "characters": True},
        order_soft={**tri, "stage_weights": [1., .25, .0625], "soft_label_blend": .25})
    return recipes


def prepare(root):
    from src.source_dataset import committed_dataset, use_manifest, model_table
    from src import rtv_text
    from src.modelling.data import ordered_frame
    from src.wf_eval import golden_holdout_start, purge_days
    prior = Path("data/research/meta_learner/performance_spread_20261108")
    receipt = json.loads((prior / "input_receipt.json").read_text())
    if sha(prior / "input.parquet") != receipt["input_sha256"]:
        raise ValueError("Previously verified canonical inputs changed")
    manifest = committed_dataset()
    if manifest["id"] != receipt["receipt"]["dataset_manifest"]["id"]:
        raise ValueError("Text source and canonical labels use different accepted datasets")
    with use_manifest(manifest):
        text = model_table("rtv", "text")
    text = text[["sl_race_id", "sl_horse_id", "tf_comment", "tf_verdict"]].rename(
        columns={"sl_race_id": "race_id", "sl_horse_id": "horse_id"})
    text["race_id"] = text.race_id.astype(str)
    text["horse_id"] = pd.to_numeric(text.horse_id)
    columns = ["race_id", "horse_id", "horse_name", "race_date", "num_runners", "won", "finish_position", "bsp", "odds"]
    frame = pd.read_parquet(prior / "input.parquet", columns=columns)
    frame["race_id"] = frame.race_id.astype(str)
    frame = frame.merge(text, on=["race_id", "horse_id"], how="left", validate="one_to_one")
    frame, _ = ordered_frame(frame)
    if frame.race_date.max() >= golden_holdout_start() - pd.Timedelta(days=purge_days()):
        raise ValueError("Development corpus must exclude the reserved holdout and purge")
    frame = rtv_text.prepare(frame)
    frame["complete_text"] = pd.Series(rtv_text.has_text(frame)).groupby(frame.race_id).transform("all").to_numpy()
    frame.attrs = {}
    atomic_parquet(frame, root / "corpus.parquet")
    atomic_json(root / "corpus_receipt.json", {"dataset_id": manifest["id"], "canonical_input_sha256": receipt["input_sha256"],
        "corpus_sha256": sha(root / "corpus.parquet"), "rows": len(frame), "races": int(frame.race_id.nunique()),
        "complete_text_races": int(frame.loc[frame.complete_text, "race_id"].nunique()),
        "last_date": str(frame.race_date.max().date()), "text_source_version": manifest["sources"]["rtv"]["version"]})
    print(json.dumps(json.loads((root / "corpus_receipt.json").read_text()), indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "refine", "confirm"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        design = {"reference_commits": ["d0320cf", "fdcf7c6"], "candidates": candidates(),
            "screen_quarters": ["2024Q1", "2025Q2"], "refine_quarters": ["2024Q3", "2024Q4", "2025Q1"],
            "confirmation_quarters": ["2025Q3", "2025Q4", "2026Q1", "2026Q2"],
            "purge_days": 7, "calibration_days": 91, "calibration_share": .25,
            "advance": "Top three recipes including the corrected reference; select the best over all five development quarters",
            "primary_metric": "paired race log loss, 95% date-cluster intervals; report period stability and calibration",
            "confirmation": "Freeze recipe before scoring later quarters; require improvement and stable calibration before replacing defaults",
            "evaluation_status": "Retrospective; historical periods already informed prior text and feature research"}
        path = root / "pre_registration.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if {k: v for k, v in previous.items() if k != "created_at"} != design:
                raise ValueError("Output belongs to a different text experiment")
        else:
            atomic_json(path, {**design, "created_at": datetime.now(timezone.utc).isoformat()})
        prepare(root)
        return
    evaluate(root, args.command)


def evaluate(root, stage):
    import time
    import joblib
    from src.modelling.text import TextExpert, prepare_roles
    from src.modelling.evaluation import code_signature, frame_signature
    from collections import OrderedDict
    from src.feature_selection import race_groups
    from src.forecast_evaluation import probability_metrics, paired_interval
    from src.wf_eval import expanding_period_folds, oof_predict_expanding
    from src.ingestion.storage import writer_lock
    design = json.loads((root / "pre_registration.json").read_text())
    receipt = json.loads((root / "corpus_receipt.json").read_text())
    if sha(root / "corpus.parquet") != receipt["corpus_sha256"]:
        raise ValueError("Text corpus changed")
    source = code_signature()
    script = sha(__file__)
    identity = {"code": source, "script": script, "corpus": receipt["corpus_sha256"]}
    frame = pd.read_parquet(root / "corpus.parquet")
    role_path = root / ("roles_" + source[:16] + ".parquet")
    if not role_path.exists():
        atomic_parquet(prepare_roles(frame), role_path)
    roles = pd.read_parquet(role_path)
    frame = pd.concat([frame, roles], axis=1).loc[frame.complete_text].reset_index(drop=True)
    folds = {f.label: f for f in expanding_period_folds(frame.race_date, min_train_periods=8, embargo_days=7)}
    recipes = design["candidates"]
    if stage == "screen":
        arms, labels = list(recipes), design["screen_quarters"]
    else:
        previous = json.loads((root / ("screen.json" if stage == "refine" else "selection.json")).read_text())
        if previous["identity"] != identity:
            raise ValueError("Earlier experiment used different code or inputs")
        arms = previous["advance"] if stage == "refine" else list(dict.fromkeys(["corrected", previous["selected"]]))
        labels = design["refine_quarters"] if stage == "refine" else design["confirmation_quarters"]
    predictions, summaries = {}, {}
    encodings = OrderedDict()
    encoding_bytes = 0
    cache_hits = 0
    def encode(rows, encoder):
        nonlocal encoding_bytes, cache_hits
        settings = {k: getattr(encoder, k) for k in ("min_df", "ngram_max", "roles", "separate_fields", "characters")}
        key = (json.dumps(settings, sort_keys=True), frame_signature(rows, ["race_id", "horse_id"]))
        if key in encodings:
            cache_hits += 1
            value = encodings.pop(key)
            encodings[key] = value
            return value[0], value[1]
        while len(encodings) >= 4:
            _, old = encodings.popitem(last=False)
            encoding_bytes -= old[2]
        x = encoder.fit_transform(rows)
        size = x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
        encodings[key] = (x, encoder, size)
        encoding_bytes += size
        while encoding_bytes > 2 * 1024 ** 3 and encodings:
            _, old = encodings.popitem(last=False)
            encoding_bytes -= old[2]
        return x, encoder
    with writer_lock(root / ".experiment.lock") as acquired:
        if not acquired:
            raise RuntimeError("Another text experiment owns this directory")
        for arm in arms:
            records = []
            settings = {**recipes[arm], "stage_weights": tuple(recipes[arm]["stage_weights"])}
            recipe_key = hashlib.sha256(json.dumps({**identity, "settings": settings}, sort_keys=True).encode()).hexdigest()
            for label in labels:
                folder = root / "cache" / recipe_key / label
                target = frame.loc[folds[label].test].reset_index(drop=True)
                if (folder / "receipt.json").exists():
                    cached = json.loads((folder / "receipt.json").read_text())
                    if cached["sha256"] != sha(folder / "predictions.parquet") or cached["identity"] != identity:
                        raise ValueError("Text cache content/identity mismatch")
                    part = pd.read_parquet(folder / "predictions.parquet")
                else:
                    started = time.monotonic()
                    models = []
                    def predict(train, test):
                        logging.info("%s %s: %d earlier / %d later runners", arm, label, len(train), len(test))
                        model = TextExpert(id=arm, **settings).fit(train, prepared=True, encoding_provider=encode)
                        models.append(model)
                        return model.predict(test.drop(columns=["won", "finish_position", "odds", "bsp"]), prepared=True)
                    p = oof_predict_expanding(frame, folds=[folds[label]], fit_predict=predict, min_train_rows=100, embargo_days=7)
                    part = target[["race_id", "horse_id", "race_date", "won"]].copy()
                    part["probability"], part["quarter"] = p[folds[label].test], label
                    part.attrs = {}
                    atomic_parquet(part, folder / "predictions.parquet")
                    atomic_json(folder / "receipt.json", {"identity": identity, "settings": settings,
                        "sha256": sha(folder / "predictions.parquet"), "lineage": models[0].lineage,
                        "elapsed_seconds": time.monotonic() - started})
                    if stage == "confirm":
                        joblib.dump(models[0], folder / "model.joblib")
                if not np.array_equal(part[["race_id", "horse_id"]].to_numpy(), target[["race_id", "horse_id"]].to_numpy()):
                    raise ValueError("Text cache field alignment changed")
                metrics, loss, _ = probability_metrics(part.probability, part.won, race_groups(part.race_id))
                logging.info("Finished %s %s: race LL %.8f", arm, label, metrics["race_log_loss"])
                part["race_loss"] = np.repeat(loss, race_groups(part.race_id))
                records.append(part)
            predictions[arm] = pd.concat(records, ignore_index=True)
        if stage == "refine":
            for arm in arms:
                old = pd.read_parquet(root / f"screen_{arm}.parquet")
                predictions[arm] = pd.concat([old, predictions[arm]], ignore_index=True).sort_values(["race_date", "race_id", "horse_id"]).reset_index(drop=True)
        for arm, part in predictions.items():
            races = part.drop_duplicates("race_id")
            summaries[arm] = {"race_log_loss": float(races.race_loss.mean()), "races": len(races),
                "per_quarter": races.groupby("quarter").race_loss.mean().to_dict()}
            atomic_parquet(part, root / f"{stage}_{arm}.parquet")
        if code_signature() != source or sha(__file__) != script:
            raise RuntimeError("Text experiment code changed during fitting")
        order = sorted(summaries, key=lambda name: summaries[name]["race_log_loss"])
        result = {"identity": identity, "summaries": summaries, "ranking": order, "evaluation_status": design["evaluation_status"],
                  "encoding_cache_hits": cache_hits, "encoding_cache_bytes": encoding_bytes}
        if stage == "screen":
            result["advance"] = ["corrected"] + [name for name in order if name != "corrected"][:2]
            atomic_json(root / "screen.json", result)
        elif stage == "refine":
            result["selected"] = order[0]
            result["selected_settings"] = recipes[order[0]]
            atomic_json(root / "selection.json", result)
            atomic_json(root / "selected_text_spec.json", recipes[order[0]])
        else:
            baseline = predictions["corrected"].drop_duplicates("race_id").set_index("race_id")
            differences = {}
            for arm, part in predictions.items():
                if arm == "corrected":
                    continue
                races = part.drop_duplicates("race_id").set_index("race_id").loc[baseline.index]
                differences[arm] = paired_interval(races.race_loss.to_numpy() - baseline.race_loss.to_numpy(), blocks=baseline.race_date)
            result["paired_differences"] = differences
            atomic_json(root / "confirmation.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
