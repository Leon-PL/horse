"""Earlier-only target/regularisation refinement after the broad text screen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from research.meta_learner import text_improvement as study
from src.ingestion.storage import atomic_json, writer_lock
from src.modelling.evaluation import code_signature


def prepare(root, parent):
    screened = json.loads((parent / "screen.json").read_text())
    original = json.loads((parent / "pre_registration.json").read_text())
    best_name = screened["ranking"][0]
    best = original["candidates"][best_name]
    candidates = {"corrected": original["candidates"]["corrected"]}
    layouts = [("best", best)]
    if not best["characters"] and not best["separate_fields"] and best["ngram_max"] == 2:
        layouts.append(("alternate_roles", {**best, "roles": not best["roles"]}))
    for layout, template in layouts:
        for share in (.25, .5, .75, 1.):
            for penalty in (1e-5, 3e-5, 1e-4):
                candidates[f"{layout}_market{round(share * 100)}_penalty{penalty:.0e}"] = {
                    **template, "soft_label_blend": share, "penalty": penalty}
    # On the broad screen, finishing-order targets improved the same .25
    # teacher recipe. Check that interaction at larger shares before locking.
    for share in (.5, .75, 1.):
        for penalty in (3e-5, 1e-4):
            candidates[f"order_market{round(share * 100)}_penalty{penalty:.0e}"] = {
                **best, "stage_weights": [1., .25, .0625],
                "soft_label_blend": share, "penalty": penalty}
    for name in screened["ranking"][:2]:
        if original["candidates"][name] not in list(candidates.values()):
            candidates["initial_" + name] = original["candidates"][name]
    design = {**original, "candidates": candidates, "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_screen_sha256": study.sha(parent / "screen.json"), "best_initial_recipe": best_name,
        "tuning_reason": "Resolve target-blend/L2 and the observed finishing-order interaction on earlier quarters; include 0.75 and full teacher targets before later confirmation.",
        "confirmation_status_at_tuning": "No confirmation text forecasts scored", "protocol_script_sha256": study.sha(__file__)}
    root.mkdir(parents=True, exist_ok=True)
    path = root / "pre_registration.json"
    if path.exists():
        previous = json.loads(path.read_text())
        design["created_at"] = previous["created_at"]
        if design != previous:
            raise ValueError("Tuning output belongs to another protocol")
    else:
        atomic_json(path, design)
    source = code_signature()
    names = ["corpus.parquet", "corpus_receipt.json", "ensemble_pre_registration.json", "roles_" + source[:16] + ".parquet"]
    for name in names:
        if not (root / name).exists():
            try:
                (root / name).hardlink_to(parent / name)
            except OSError:
                shutil.copyfile(parent / name, root / name)
    identity = {"code": source, "script": study.sha(study.__file__),
                "corpus": json.loads((root / "corpus_receipt.json").read_text())["corpus_sha256"]}
    if identity != screened["identity"]:
        raise ValueError("Tuning requires the same model mathematics and accepted corpus")
    reused = []
    for receipt_path in (parent / "cache").glob("*/*/receipt.json"):
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["settings"] not in list(candidates.values()):
            continue
        if receipt["sha256"] != study.sha(receipt_path.parent / "predictions.parquet"):
            raise ValueError("Earlier completed forecast bytes changed")
        destination = root / "cache" / receipt_path.parent.relative_to(parent / "cache")
        if not destination.exists():
            shutil.copytree(receipt_path.parent, destination)
        reused.append(str(receipt_path.parent))
    atomic_json(root / "tuning_provenance.json", {"parent_experiment": str(parent.resolve()),
        "reused_original_prediction_receipts": reused, "identity": identity})
    print(json.dumps({"initial_best": best_name, "candidates": len(candidates), "reused_folds": len(reused)}), flush=True)


def extend_penalty_boundary(root):
    """Check two weaker penalties if the full teacher wins at the lower edge."""
    design_path, screen_path = root / "pre_registration.json", root / "screen.json"
    design, screened = (json.loads(p.read_text()) for p in (design_path, screen_path))
    if (root / "confirmation.json").exists() or (root / "selection.json").exists():
        return
    if "low_penalty_extension" not in design:
        best = design["candidates"][screened["ranking"][0]]
        if best["penalty"] != 1e-5 or best["soft_label_blend"] != 1.:
            return
        extra = {}
        for name, recipe in design["candidates"].items():
            if recipe["penalty"] == 1e-5 and recipe["soft_label_blend"] == 1.:
                for penalty in (3e-6, 1e-6):
                    extra[name.replace("penalty1e-05", f"penalty{penalty:.0e}")] = {**recipe, "penalty": penalty}
        shutil.copyfile(design_path, root / "pre_registration_primary.json")
        shutil.copyfile(screen_path, root / "screen_primary.json")
        design["low_penalty_extension"] = {
            "created_at": datetime.now(timezone.utc).isoformat(), "candidates": list(extra),
            "reason": "A full-market teacher won the earlier screen at the weakest tested penalty. Check the lower boundary before selection.",
            "primary_screen_sha256": study.sha(screen_path), "primary_registration_sha256": study.sha(design_path),
            "refinement_driver_sha256": study.sha(__file__), "confirmation_status": "No later text forecasts scored"}
        design["candidates"].update(extra)
        atomic_json(design_path, design)
    if set(screened["summaries"]) != set(design["candidates"]):
        study.evaluate(root, "screen")


def evaluate_later(root, stage):
    """Precompute independent quarter caches in at most two fitting processes."""
    import logging
    import psutil
    design_path = root / "pre_registration.json"
    design = json.loads(design_path.read_text())
    label_key = "refine_quarters" if stage == "refine" else "confirmation_quarters"
    labels = design[label_key]
    if len(labels) < 2 or psutil.virtual_memory().available < 7 * 1024 ** 3:
        study.evaluate(root, stage)
        return
    earlier_name = "screen.json" if stage == "refine" else "selection.json"
    earlier = json.loads((root / earlier_name).read_text())
    immutable = ["corpus.parquet", "corpus_receipt.json", earlier_name,
                 "roles_" + earlier["identity"]["code"][:16] + ".parquet"]
    if stage == "refine":
        immutable += [f"screen_{arm}.parquet" for arm in earlier["advance"]]
    workers = []
    with writer_lock(root / f".{stage}_workers.lock") as acquired:
        if not acquired:
            raise RuntimeError(f"Another process owns the {stage} quarter workers")
        for index in range(2):
            child = root / f"{stage}_workers" / f"batch_{index + 1}"
            child.mkdir(parents=True, exist_ok=True)
            partition = {**design, label_key: labels[index::2],
                         "parent_registration_sha256": study.sha(design_path)}
            registration = child / "pre_registration.json"
            if registration.exists() and json.loads(registration.read_text()) != partition:
                raise ValueError("Quarter worker registration changed")
            if not registration.exists():
                atomic_json(registration, partition)
            for name in immutable:
                if not (child / name).exists():
                    try:
                        (child / name).hardlink_to(root / name)
                    except OSError:
                        shutil.copyfile(root / name, child / name)
            logging.info("%s batch %d: %s", stage, index + 1, ", ".join(partition[label_key]))
            with (child / "worker.log").open("w", encoding="utf-8") as log:
                worker = subprocess.Popen([sys.executable, "-u", "-m", "research.meta_learner.text_improvement",
                    stage, "--output", str(child)], stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workers.append((child, worker))
        failures = [str(child) for child, worker in workers if worker.wait() != 0]
        if failures:
            raise RuntimeError(f"Text quarter workers failed: {failures}; see their worker.log files")
        reused = []
        for child, _ in workers:
            for receipt in (child / "cache").glob("*/*/receipt.json"):
                fit = receipt.parent
                record = json.loads(receipt.read_text())
                if record["identity"] != earlier["identity"] or record["sha256"] != study.sha(fit / "predictions.parquet"):
                    raise ValueError("Quarter worker returned mismatched forecast bytes or identity")
                if stage == "confirm" and not (fit / "model.joblib").is_file():
                    raise ValueError("Confirmation worker did not save a fitted model")
                destination = root / "cache" / fit.relative_to(child / "cache")
                if not destination.exists():
                    shutil.copytree(fit, destination)
                reused.append(str(fit))
        atomic_json(root / f"{stage}_worker_provenance.json", {"identity": earlier["identity"],
                    "driver_sha256": study.sha(__file__), "original_worker_caches": reused})
        study.evaluate(root, stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "screen", "refine", "confirm"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent", default="data/research/meta_learner/text_development_20261108")
    args = parser.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "prepare":
        prepare(Path(args.output), Path(args.parent))
    else:
        if args.command == "refine":
            extend_penalty_boundary(Path(args.output))
        if args.command in ("refine", "confirm"):
            evaluate_later(Path(args.output), args.command)
        else:
            study.evaluate(Path(args.output), args.command)


if __name__ == "__main__":
    main()
