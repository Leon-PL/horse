# Installed live-feature speed improvements

Installed and verified in `C:\Users\Leonp\Documents\horse` at 2026-10-29T22:45:22.
Normal CLI training, the picks app and future scheduled recorder sessions now
use the optimized implementation.

The live path loads its counts-state file once, removes its redundant deep copy,
and copies only changed expanding/distinct histories. The complete store was
rebuilt and qualified against the preserved generation: **754,842
rows × 1,187 columns**, with exact values and dtypes.
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
features/probabilities exactly and passed constraints on 115
recent complete fields / 1,108 runners.

Current promoted run: `20261029_222758_104473`. It retains **byte-identical** model artifact
`73747ec2166fb6598687c9dedbffbaa37f20694256933cda4d03e864b4e156b0` from `20261029_141132_815571`: 154 inputs, the same win/place
trees and calibration, and **zero new model fits**. The new run records the
verified feature-generation migration; historical scores remain historical,
with no new predictive-accuracy claim.

Feature generation is `3697230d16bbd0d8`. Windows task
`HorseForecastRecorder_038` now launches `20261029_feature_speed_20261029_222758_104473` from the actual project,
using its existing daily 09:00 and user-logon schedule. The earlier study,
store, sources and recorder configuration are preserved. The task and study
pins were verified after restart; live feed errors, if any, are reported by
`project_status.py` and are separate from source/store readiness.

Receipts: `data/audit/live_feature_speed_20261029/installation/`, including
`qualification.json`, `live_parity.json`, `tests.log`, `readiness.json`,
`scheduler.json`, `activation.json` and `final_status.json`. The original
benchmark remains in `data/audit/live_feature_speed_20261029/native_card/`.
