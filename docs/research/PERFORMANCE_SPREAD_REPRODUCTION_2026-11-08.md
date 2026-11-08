# Performance/spread reproduction, 8 November 2026

This is an initial retrospective reproduction of the 058 modelling idea on
the accepted data used for the new generalist benchmark. It tests whether a
separate prediction of performance and uncertainty adds useful information.
It is a single-quarter pilot, not independent confirmation or serving approval.

## Fixed method

- Inputs: the generalist's ordered 129-feature recipe and accepted dataset
  `cdcc01a5d698ad70fe4f7b6f4c0cd9398f6eb15bcb944eb10100a5c3a9c0b636`.
- Mean target: negative cumulative beaten lengths × 15/furlongs, with distance
  floored at 5f and margins capped at 30 lb. Winners have zero margin. Missing
  margins, invalid distances and non-finishers have no performance target.
  Races need at least two measured finishers for the centred regression.
- Spread target: log(squared centred residual + 0.25). Predictions generating
  these residuals use the shared expanding chronological evaluator over four
  date blocks, with a seven-day purge. The first block has no residual target.
- Learners: LightGBM with the baseline's explicit numerical controls, recency
  weights and burn-in. Native fits use purged early stopping and full refits.
- Calibration: the final 20% of training races form a later scale-calibration
  slab. Both calibration models are fitted before that slab, with a purge;
  their spread targets also exclude later labels. Final mean/spread models are
  then refitted on all permitted training rows. Spreads are floored at 1 lb.
- Forecast: 400-point numerical integration of independent normal performance
  distributions. Every runner, including non-finishers, remains in the scored
  field. Prediction reads pre-race features and saved trees/scale only.
- Outer period: 1 April–3 June 2026, retaining the existing holdout exclusion.
  Pool weights learn through 30 April and score from 8 May, with a seven-day
  purge and fixed shrinkage 0.01 towards the generalist.

The constants follow 058; none is searched on this scored period. The new
chronological residual construction intentionally replaces 058's random race
halves. Its explicit scale-calibration boundary applies at screen capacity too.
Those changes and the repaired data prevent claiming an exact numerical replay
of the old experiment.

The current combiner uses nonnegative log-pooling weights summing to one,
with shrinkage towards the generalist. Original 058 used two freely fitted
coefficients. This pilot measures the component under the new combination
contract and keeps extra global temperature fitting out of that comparison.

The four residual blocks are a fixed pilot choice. A broader qualification
should also check the intended quarterly update cadence. Incremental evidence
needs to survive the stronger generalist and a simple repeat-fit averaging
control before attributing a small gain to the uncertainty model itself.

## Reference and evidence boundary

The reference is the completed native measurement at
`data/research/meta_learner/baseline_native_2026q2/`. Its forecasts and archived
source manifest are checked by SHA-256. The new head records its own source
identity separately; old benchmark fingerprints are never rewritten. This
research comparison does not qualify a new current-code generalist fit.

The measured modelling sources, configuration and dependency lock are also
archived under this experiment's `source_snapshot/`, with 23 file digests and
the measured code fingerprint. This is a review archive for one result; it adds
no runtime source-pinning mechanism or exception to candidate fitting guards.

The design was written before the performance fit under
`data/research/meta_learner/performance_spread_20261108/pre_registration.json`.
The experiment owns its checked input, recipe, predictions and report there.
Reusable model/target mechanics live in `src/modelling/performance.py`; the
registered comparison lives in `research/meta_learner/performance_probe.py`.

058's earlier complementary gain is a hypothesis to reproduce. Later 063
research found that improved inputs and a stronger generalist target largely
reduced the figure model's contribution. Any promising pilot therefore needs a
broader period comparison and that stronger-reference control before adoption.
The already examined historical periods remain labelled retrospective.

Preparation verified 700,639 development runners. Of these, 655,435 have a
known performance figure; 45,204 do not, including 43,362 non-finishers. Known
margin figures have a median of 10.768 lb, and the 90th percentile reaches the
fixed 30 lb cap. These are label-coverage diagnostics, not model scores.

## Verification and result

Eight focused tests pass, covering analytic two-runner integration, numerical
symmetries, valid margin targets, complete-field and single/batch prediction,
chronological lineage, later-label isolation, cache invalidation, mixed-component
candidates, exact fallback and reload agreement. The affected regression group
also passes: **78 passed** in 166.79 seconds, including the eight performance
checks. Ruff and whitespace checks pass.

The native fit completed in **4,141.2 seconds (69.0 minutes)** after preparation,
with sampled peak resident memory **4.814 GB**. It uses 589,871 warmed training
runners in 63,112 races, through 25 March 2026. The final mean uses 553,055
eligible measured finishers and 3,801 trees; the spread uses 406,126 earlier-only
residual targets and 1,448 trees. Its calibrated spread multiplier is 1.5900.
The saved two-head artifact is 20.9 MB.

All three models below score the same **1,142 races / 10,610 runners across
27 dates**, from 8 May through 3 June:

| Forecast | Race log loss |
|---|---:|
| Generalist | 1.8508472 |
| Performance/spread alone | 1.8528890 |
| Generalist plus performance/spread | **1.8488525** |

The paired pool change is **−0.0019947**, with a 95% date-cluster interval of
**[−0.0032247, −0.0007044]**. Earlier April forecasts fitted weights of
**87.8125% generalist / 12.1875% performance-spread**. Brier score and calibration
also improve slightly on this scored period. The component is weaker alone on
race log loss and useful in this combination, supporting the incremental-value
criterion for submodels.

For the full April–3 June period, both components forecast **2,544 races /
23,725 runners**. Generalist race log loss is 1.8458393 and performance/spread
alone is 1.8513115. The full-period and later pool scores have different
denominators and should be kept separate.

Independent verification reloads the actual saved artifact and reproduces every
exported probability **exactly**. Doubling integration resolution from 400 to
800 points changes probabilities by at most **3.9651 × 10⁻¹²**. Input, reference,
model and result identities are recorded in `verification.json`.

**Decision: retain for broader evaluation; no adoption yet.** This pilot
reproduces the direction of the earlier complementary result under the corrected
chronological method. Confirmation across more periods, the stronger 063
generalist and repeat-fit averaging controls remains necessary. Prospective
confirmation and serving/place qualification are subsequent gates.

The selected model and source collectors are unchanged.

## Re-running the registered comparison

```powershell
.venv\Scripts\python.exe -m research.meta_learner.performance_probe prepare --benchmark data/research/meta_learner/baseline_native_2026q2 --store-root data/processed/modelling_feature_store_20261108_serial --output data/research/meta_learner/performance_spread_20261108
.venv\Scripts\python.exe -m research.meta_learner.performance_probe run --benchmark data/research/meta_learner/baseline_native_2026q2 --store-root data/processed/modelling_feature_store_20261108_serial --output data/research/meta_learner/performance_spread_20261108 --threads 2
.venv\Scripts\python.exe -m research.meta_learner.verify_performance --experiment data/research/meta_learner/performance_spread_20261108
```

The first step performs the deep accepted-source checks and writes a projected
input with a content digest. The second checks that input and the archived
reference, locks its own recipe/source identity and runs the fixed comparison.
The final command verifies the saved artifact and numerical convergence without
fitting or selecting anything.
Resolved `performance_spec.json` and `generalist_spec.json` can also be supplied
as separate recipes to the supported benchmark command for a new comparison.
The standard nested pool needs enough earlier OOF quarters to fit its weights;
the within-quarter split above is explicitly a small initial probe.
