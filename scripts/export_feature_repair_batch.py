"""Save reviewable candidate/native diffs without replacing installed sources."""
from __future__ import annotations

import difflib
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "_research_snapshots/018_feature_repairs"
OUTPUT = ROOT / "data/audit/feature_review_018"
FILES = ["src/data_processor.py", "src/feature_engineer.py", "src/live_prediction.py",
    "src/featured_dataset.py", "src/rating_reuse.py", "scripts/build_feature_store_batched.py",
    "src/feature_semantics.py", "tests/test_feature_semantics.py", "tests/test_feature_repair_batch.py",
    "tests/test_rating_reuse.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-ewma", action="store_true")
    args = parser.parse_args()
    suffix = "_exact" if args.exact_ewma else ""
    names = [*FILES, "tests/test_exact_ewma_history.py"] if args.exact_ewma else FILES
    target = OUTPUT / f"candidate_source{suffix}.zip"
    if target.exists():
        raise ValueError("Preserve the earlier sealed source package")
    differences, sources = [], {}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            original, revised = ROOT / name, BUNDLE / name
            before = original.read_text(encoding="utf-8") if original.exists() else ""
            after = revised.read_text(encoding="utf-8")
            differences.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile="a/"+name, tofile="b/"+name))
            archive.write(revised, "candidate/"+name)
            if original.exists():
                archive.write(original, "installed_before/"+name)
            sources[name] = {"candidate_sha256": sha(revised),
                             "installed_sha256": sha(original) if original.exists() else None}
    patch = OUTPUT / f"candidate_changes{suffix}.patch"
    patch.write_text("".join(differences), encoding="utf-8")
    receipt = {"sources": sources, "patch_sha256": sha(patch), "zip_sha256": sha(target),
        "candidate_invariance_sha256": sha(ROOT / "data/audit/model_improvement/018_feature_repairs/input_invariance.json"),
        "candidate_test_log_sha256": sha(ROOT / ("data/audit/exact-ewma-tests-018.log" if args.exact_ewma else "data/audit/feature-repair-candidate-tests-018-sealed.log")),
        "scope": "Reviewable source package only; installed feature generation remains unchanged"}
    (OUTPUT / f"candidate_source_receipt{suffix}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"files": len(sources), "patch_lines": len(differences), "zip_sha256": sha(target)}), flush=True)


if __name__ == "__main__":
    main()
