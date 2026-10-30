"""Close the fixed041 auxiliary-place check without any further fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deployment_validation import file_sha256 as sha

CASE = ROOT / "data/audit/auxiliary_win_041"


def main():
    if (CASE / "final_status.json").exists():
        raise ValueError("Preserve the completed041 record")
    read = lambda name: json.loads((CASE / name).read_text())
    preparation, recipe, report = map(read, ("preparation.json", "recipe.json", "report.json"))
    if (recipe["selected"] != "identity" or recipe["parameters"]["auxiliary"]["alpha"] != 0.
            or report["advance_to_native_calibration_transfer_check"] or report["new_tree_fits"] != 0):
        raise ValueError("This closure only describes the registered zero-weight, failed-gate result")
    if preparation["native_win_replay_max_error"] or preparation["current_price_and_label_poison_max_error"]:
        raise ValueError("Native replay/causality qualification failed")
    if "19 passed" not in (CASE / "tests.log").read_text():
        raise ValueError("Finish the helper regressions")
    status = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "complete_not_selected", "selected": "identity", "learned_place_weight": 0.,
        "decision": "Stop this fixed bounded positive auxiliary-place correction; it collapses to the win temperature control and loses on purged validation. No further native/current-data/tree fits.",
        "new_tree_fits": 0, "coefficient_fits": 2, "helper_regressions_passed": 19,
        "production_writes": False, "overall": report["overall"]["auxiliary"],
        "established_handicaps": report["established_handicaps"]["auxiliary"],
        "evidence_sha256": {name: sha(CASE / name) for name in
                            ("plan.json", "preparation.json", "recipe.json", "report.json", "test_predictions.parquet", "source_snapshot.zip")},
        "label_version_note": "Completed before042's result-source audit. Preserve original labels/scores; do not mix later corrected labels into these archived comparisons.",
        "scope": report["scope"]}
    with (CASE / "final_status.json").open("x", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2, allow_nan=False)
    print(json.dumps({k: status[k] for k in ("stage", "selected", "learned_place_weight", "new_tree_fits")}, indent=2))


if __name__ == "__main__":
    main()
