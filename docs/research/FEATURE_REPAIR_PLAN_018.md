# Feature audit and repair batch 018

**Historical repair-batch plan, 27 October 2026.** This records the requested audit
sequence. Later mandatory repairs are documented in [026](CANONICAL_FEATURE_REPAIRS_026.md);
use the [README](../../README.md) for the normal workflow.

Requested 27 October 2026: inspect features before training; collect objective
errors into a single repair batch; test proposed redesigns separately. Include
overseas and point-to-point experience, particularly Cheltenham visitors.

1. Inspect all selected feature families and their dependencies. Reconcile
   definitions with source values, labels, missingness, chronology and live cards.
2. Reproduce each objective failure with a small example and measure its reach
   across the stored history. Check surface classification, class/mark changes,
   missing values, field context and the meaning of experience.
3. Implement and test one coherent correction layer. Preserve the installed
   model and matching histories during research; never silently give it new
   definitions. Validate full versus incremental and historical versus card use.
4. Audit missing-history runners by pre-race metadata, including jumps racing.
   Examine dated public results. A French birthplace or name is not proof of
   French racing experience. Partial foreign fields must not enter Elo sweeps.
5. Only after these checks, run one cheaper, matched win-model comparison for
   the repair batch. Keep new feature designs in a separate comparison against
   that corrected baseline. Use the same chronological window, training budget,
   calibration and data; do not queue every quarter for every idea.
6. Confirm and integrate a useful candidate with matching live features before
   any production change. Record uncertainty and retain recoverable artifacts.

Existing historical test periods have already informed development. A result
on them is screening evidence, not a fresh demonstration of market parity.
Do not inspect or optimise against the reserved holdout.

Production at start: 20261027_165656_616818,151 inputs, history to25 October.
The 017 synthetic-surface screen mixed a predicate repair with two new inputs;
its uncertain result does not establish whether the predicate repair alone helps.

Work and evidence will be recorded in `data/audit/feature_review_018/`.
