"""Finish the audited repair generation and all four registered cheap fits.

Runs locally to completion, checkpoints each step, and stops on any failed
guard. This controller never changes the installed model or canonical store.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "_research_snapshots/018_feature_repairs"
OUTPUT = ROOT / "data/audit/feature_review_018"


def status(stage, **extra):
    value = {"stage": stage, "at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    (OUTPUT / "workflow_status.json").write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def run(stage, script, arguments=(), *, candidate=False):
    directory = BUNDLE if candidate else ROOT
    log = ROOT / f"data/audit/{stage}-018.log"
    status(stage, log=str(log))
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-u", script, *arguments], cwd=directory,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        status("failed", failed_stage=stage, exit_code=result.returncode, log=str(log))
        raise SystemExit(result.returncode)


def main():
    run("feature-invariance-exact", "scripts/audit_prediction_inputs.py",
        ["--output-dir", "data/audit/model_improvement/018_feature_repairs"], candidate=True)
    if not (OUTPUT / "candidate_source_exact.zip").exists():
        run("feature-source-exact", "scripts/export_feature_repair_batch.py", ["--exact-ewma"])
    store = OUTPUT / "candidate_store_exact"
    if not (store / "meta.json").exists():
        run("feature-build-exact", "scripts/build_feature_store_batched.py", [
            "--chunk-rows", "80000", "--store-root", str(store),
            "--reuse-ratings-from", str(ROOT / "data/processed/feature_store"),
            "--ratings-receipt", str(OUTPUT / "rating_source_receipt.json"),
            "--allow-nonrating-repairs"], candidate=True)
    run("feature-review-exact", "scripts/audit_repaired_store.py")
    manifest = ROOT / "data/audit/model_improvement/018_feature_repairs/manifest.json"
    if not manifest.exists():
        run("feature-screen-registration", "scripts/screen_feature_batch.py", ["--prepare"])
    run("feature-screen-fits", "scripts/screen_feature_batch.py")
    status("screen_complete", production_changed=False,
        report=str(manifest.with_name("report.json")), extra_quarters_queued=False)


if __name__ == "__main__":
    main()
