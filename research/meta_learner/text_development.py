"""Development-grid amendment: vary targets independently of field normalisation.

No confirmation outcomes were scored when this amendment was made. Completed
identical forecasts retain their original receipts and bytes when reused.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from research.meta_learner import text_improvement as study
from src.ingestion.storage import atomic_json
from src.modelling.evaluation import code_signature


def recipes():
    base = study.candidates()["corrected"]
    roles = {**base, "roles": True}
    block = {**base, "separate_fields": True}
    return {
        "corrected": base,
        "joint_weak_l2": {**base, "penalty": 1e-5},
        "joint_strong_l2": {**base, "penalty": 1e-4},
        "joint_recent_2y": {**base, "half_life_days": 730.},
        "joint_recent_4y": {**base, "half_life_days": 1460.},
        "joint_finishing_order": {**base, "stage_weights": [1., .25, .0625]},
        "joint_soft_quarter": {**base, "soft_label_blend": .25},
        "joint_soft_half": {**base, "soft_label_blend": .5},
        "joint_order_soft": {**base, "stage_weights": [1., .25, .0625], "soft_label_blend": .25},
        "roles": roles,
        "roles_soft_half": {**roles, "soft_label_blend": .5},
        "joint_trigrams": {**base, "ngram_max": 3},
        "roles_trigrams": {**roles, "ngram_max": 3},
        "blocks": block,
        "blocks_weak_l2": {**block, "penalty": 1e-5},
        "blocks_strong_l2": {**block, "penalty": 1e-4},
        "roles_blocks": {**block, "roles": True},
        "joint_characters": {**base, "characters": True},
        "joint_rare_terms": {**base, "min_df": 5},
    }


def prepare(root, initial):
    original = json.loads((initial / "pre_registration.json").read_text())
    revised = {**original, "candidates": recipes(), "amendment_created_at": datetime.now(timezone.utc).isoformat(),
        "initial_plan_sha256": study.sha(initial / "pre_registration.json"),
        "reason": "Initial block-normalisation ablation worsened 2024Q1. Test phrase/target/recency axes on the original layout, with separate-layout controls and L2 tuning.",
        "confirmation_status_at_amendment": "No confirmation text forecasts scored", "protocol_script_sha256": study.sha(__file__)}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pre_registration.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("amendment_created_at",):
            revised[key] = existing[key]
        if revised != existing:
            raise ValueError("Development output belongs to a different amended protocol")
    else:
        atomic_json(path, revised)
    for name in ("corpus.parquet", "corpus_receipt.json", "ensemble_pre_registration.json"):
        if not (root / name).exists():
            shutil.copyfile(initial / name, root / name)
    source = code_signature()
    role = "roles_" + source[:16] + ".parquet"
    if (initial / role).exists() and not (root / role).exists():
        shutil.copyfile(initial / role, root / role)
    identity = {"code": source, "script": study.sha(study.__file__),
                "corpus": json.loads((root / "corpus_receipt.json").read_text())["corpus_sha256"]}
    reused = []
    for receipt_path in (initial / "cache").glob("*/*/receipt.json"):
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["settings"] not in list(recipes().values()):
            continue
        if receipt["sha256"] != study.sha(receipt_path.parent / "predictions.parquet"):
            raise ValueError("Earlier completed prediction bytes changed")
        destination = root / "cache" / receipt_path.parent.relative_to(initial / "cache")
        if not destination.exists():
            shutil.copytree(receipt_path.parent, destination)
        reused.append(str(receipt_path.parent))
    atomic_json(root / "amendment_provenance.json", {"initial_experiment": str(initial.resolve()),
        "reused_original_prediction_receipts": reused, "identity": identity})
    print(json.dumps({"candidates": len(recipes()), "reused_completed_folds": len(reused)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "refine", "confirm"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial", default="data/research/meta_learner/text_improvement_20261108")
    args = parser.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "prepare":
        prepare(Path(args.output), Path(args.initial))
    else:
        study.evaluate(Path(args.output), args.command)


if __name__ == "__main__":
    main()
