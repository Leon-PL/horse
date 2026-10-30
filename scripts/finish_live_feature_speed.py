"""Publish operational notes only after the installed speed rollout is verified."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.run_store import get_production_run_id
from src.forward_store import ForwardStudy

OUTPUT = ROOT / "data/audit/live_feature_speed_20261029/installation"
read = lambda name: json.loads((OUTPUT / name).read_text())
plan, qualified, activation, live = map(read, ("plan.json", "qualification.json", "activation.json", "live_parity.json"))
status, scheduler = map(read, ("readiness.json", "scheduler.json"))
if not status["servable"] or not qualified["passed"] or not live["passed"] or not scheduler["verified"]:
    raise ValueError("Finish only a verified installed rollout")
if not re.search(r"\b88 passed\b", (OUTPUT / "tests.log").read_text(encoding="utf-8")):
    raise ValueError("The installed regression suite has not passed")
forward = status["sections"].get("forward_recording") or {}
if (forward.get("state") != "OK" or forward.get("study") != activation["study"]
        or (forward.get("heartbeat") or {}).get("mode") != "recording"):
    raise ValueError("The new recorder has not restarted with a fresh recording heartbeat")
if get_production_run_id() != activation["run_id"]:
    raise ValueError("Production changed before publication")
study = ForwardStudy(activation["study"])
study.verify()
checked = status["checked_at"]
run_id, parent = activation["run_id"], plan["parent_run"]
study_name = Path(activation["study"]).name
old_settings = json.loads((OUTPUT / "previous_files/data/forecasts/recorder.json").read_text())
old_study_name = Path(old_settings["study"]).name
summary = f"""# Installed live-feature speed improvements

Installed and verified in `C:\\Users\\Leonp\\Documents\\horse` at {checked}.
Normal CLI training, the picks app and future scheduled recorder sessions now
use the optimized implementation.

The live path loads its counts-state file once, removes its redundant deep copy,
and copies only changed expanding/distinct histories. The complete store was
rebuilt and qualified against the preserved generation: **{qualified['rows_compared']:,}
rows × {qualified['columns_compared']:,} columns**, with exact values and dtypes.
Processed inputs and every carried-state family also matched. The main project's
focused suite passed **88 tests**.

The raw replay exposed the original last-build/one-day-append boundary in nine
unused ATR columns, plus approximately 2e-12 prize-history rounding on 20 runners.
The migration preserved those eleven historical columns and retained original
carried-state bytes, then reran the complete strict comparison. The preserved
raw replay and its diagnostic remain in the installation audit. No ATR formula
or selected feature definition was changed by this speed release.

The original measured seven-runner saved-card comparison took **110.304 s before
and 42.162 s after**, or **2.62 times faster**. Peak process working set fell
from 5.175 GB to 4.487 GB. This was one fresh-process comparison, baseline first;
it is not a repeated throughput estimate. Features and win/place probabilities
matched exactly. The installed public serving path also replayed the saved
features/probabilities exactly and passed constraints on {live['recent_races']}
recent complete fields / {live['recent_runners']:,} runners.

Current promoted run: `{run_id}`. It retains **byte-identical** model artifact
`{activation['model_sha256']}` from `{parent}`: 154 inputs, the same win/place
trees and calibration, and **zero new model fits**. The new run records the
verified feature-generation migration; historical scores remain historical,
with no new predictive-accuracy claim.

Feature generation is `{activation['fe_signature']}`. Windows task
`HorseForecastRecorder_038` now launches `{study_name}` from the actual project,
using its existing daily 09:00 and user-logon schedule. The earlier study,
store, sources and recorder configuration are preserved. The task and study
pins were verified after restart; live feed errors, if any, are reported by
`project_status.py` and are separate from source/store readiness.

Receipts: `data/audit/live_feature_speed_20261029/installation/`, including
`qualification.json`, `live_parity.json`, `tests.log`, `readiness.json`,
`scheduler.json`, `activation.json` and `final_status.json`. The original
benchmark remains in `data/audit/live_feature_speed_20261029/native_card/`.
"""
(ROOT / "docs/LIVE_FEATURE_SPEED_20261029.md").write_text(summary, encoding="utf-8")
readme_path = ROOT / "README.md"
readme = readme_path.read_text(encoding="utf-8")
readme = readme.replace(old_study_name, study_name).replace(f"`{parent}`", f"`{run_id}`")
readiness_start = readme.index("**Full readiness check,")
readiness_end = readme.index("**Current installation:**", readiness_start)
readiness_text = (
    f"**Full readiness check, {checked}: ready to serve.**\n"
    "Database contents, feature history, current source and the installed model agree.\n"
    f"**Installed serving-path check, {live['checked_at_utc']}: passed.**\n"
    f"The saved seven-runner card's features and probabilities replay exactly. Checks on "
    f"{live['recent_races']} recent complete fields / {live['recent_runners']:,} runners pass, "
    "with exact single-race/batch agreement. These are operational checks, not a new "
    "predictive-accuracy estimate. Rerun status for the latest state.\n\n"
)
readme = readme[:readiness_start] + readiness_text + readme[readiness_end:]
readme = re.sub(r"\*\*Verified running at 14:51 GMT on 29 October\*\*,\s*with fresh heartbeats and no recorder errors\.",
                f"**Recorder restart verified at {checked}.** Check its current heartbeat for feed errors.", readme)
note = (f"**Speed update installed {checked}:** normal runs use the optimized history-state path. "
        "The full rebuilt store is exactly equivalent and the original model bytes are retained. "
        "[Timing, rollout and checks](docs/LIVE_FEATURE_SPEED_20261029.md).\n\n")
if "**Speed update installed " not in readme:
    split = readme.find("\n\n") + 2
    readme = readme[:split] + note + readme[split:]
previous_docs = OUTPUT / "previous_docs"
previous_docs.mkdir(exist_ok=True)
if not (previous_docs / "README.md").exists():
    (previous_docs / "README.md").write_bytes(readme_path.read_bytes())
readme_path.write_text(readme, encoding="utf-8")
handoff_path = ROOT / "CONTINUATION.md"
handoff = handoff_path.read_text(encoding="utf-8")
heading = "# Installed feature-speed rollout (29 October 2026)"
if not handoff.startswith(heading):
    if not (previous_docs / "CONTINUATION.md").exists():
        (previous_docs / "CONTINUATION.md").write_bytes(handoff_path.read_bytes())
    prefix = f"""{heading}

The user requested the tested optimization in the actual project. It is installed
in `C:\\Users\\Leonp\\Documents\\horse`, not only in the temporary worktree.
Current run **{run_id}**, FE **{activation['fe_signature']}**. Artifact bytes
are identical to parent **{parent}**; zero fits and unchanged probabilities.
Full {qualified['rows_compared']:,}-row/{qualified['columns_compared']:,}-column rebuild,
processed inputs and carried state match exactly. 88 main-project tests pass.
Installed saved-card replay and recent-field constraints pass. Full readiness
passes at {checked}; recorder task/source pins were updated and restarted.
New study: **{study_name}**. Preserve all earlier studies and receipts.
See `docs/LIVE_FEATURE_SPEED_20261029.md` and the speed audit's installation
`final_status.json`. The previous 038 operational identifiers below are historical.

## Preserved 038 handoff

"""
    handoff_path.write_text(prefix + handoff, encoding="utf-8")
final = {"completed_at": datetime.now(timezone.utc).isoformat(), "installed_in_actual_project": True,
         "root": str(ROOT), "run_id": run_id, "parent_run": parent, "model_bytes_unchanged": True,
         "new_model_fits": 0, "fe_signature": activation["fe_signature"], "store_exact": True,
         "main_project_tests_passed": 88, "installed_public_serving_parity": True,
         "deep_readiness_passed": True, "study": str(study.path), "scheduler_verified": True,
         "forward_recording": status["sections"].get("forward_recording")}
(OUTPUT / "final_status.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
activation["recorder_task_update_pending"] = False
(OUTPUT / "activation.json").write_text(json.dumps(activation, indent=2), encoding="utf-8")
print(json.dumps(final, indent=2))
