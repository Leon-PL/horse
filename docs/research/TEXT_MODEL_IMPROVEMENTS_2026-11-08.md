# Text-model development, 8 November 2026

This evaluates improvements beyond comment-model fixes `d0320cf` and
`fdcf7c6`. The accepted text observations are unchanged. The corrected baseline
is refitted from raw text with held-out scale calibration; old stored scores
are not presented as the repaired baseline.

## Audit and corrections

The earlier repairs add chronological calibration, distinguish missing text from
a constant score and keep feature-stage missingness outside `TextModel.logits`.
Their pre-race availability check matched 68 captured races. Published scores,
the feature store and the selected win model still require their normal refresh
before those repairs become model inputs.

Further issues addressed here:

- A previous-quarter warm start can include labels from the next model's scale
  calibration slab. Calibration fits now start cold; full-history refits can
  still use legitimate earlier weights. Finite optimisation need not erase an
  inappropriate starting state exactly.
- The explicit `TextExpert` previously omitted scale calibration and treated
  a shared nonempty verdict as evidence that every runner had text. It now
  calibrates on a purged later training slab and masks an incomplete expert for
  the whole field, using comments or sentences actually naming each runner.
- Shared verdict sentences gave the same words to horses with opposite roles.
  An optional representation marks the target as `selfhorse` and active rivals
  as `rivalhorse`, normalises typographic punctuation and matches longest names
  first. This makes preference and comparison phrases runner-specific.
- Comment words, verdict words and verdict-rank dummies can be normalised
  separately, so rank strength is not implicitly divided by comment length.

Reusable encoders and learners live in `src/modelling/text.py`. Model prediction
uses saved vocabulary, IDF, weights and scale; it never collects data or fits.
Experiments remain under `research/meta_learner/`.

## Registered development process

The initial study is `data/research/meta_learner/text_improvement_20261108/`. Its corpus
joins accepted RTV text links to the same verified canonical results used in
the modelling baseline. It contains 700,639 runners / 74,427 valid races through
3 June 2026. Of these, 74,081 races have complete text support. Training and
standalone comparisons retain complete fields, including non-finishers.

The first 17-variant grid bundled several richer targets with separate field
normalisation. That layout worsened the first development quarter, so the grid
was amended before any confirmation outcomes were scored. The primary amended
study is `data/research/meta_learner/text_development_20261108/`, run through
`research/meta_learner/text_development.py`. Its 19 variants test target,
regularisation and recency changes on the original layout, with role/trigram,
separate-layout, character and rare-term controls. This avoids allowing one poor
layout to obscure a useful target. The amendment and already observed
development evidence are recorded explicitly; completed identical predictions
are copied with their original receipts and bytes, never restamped.

Prices are training targets only and never inference inputs.

All choices use earlier data:

1. Screen on 2024Q1 and 2025Q2.
2. Refine the promising target/L2 interaction on those same earlier quarters,
   through `text_tuning.py`: teacher shares 0.25, 0.5, 0.75 and 1.0 crossed with
   penalties 1e-5, 3e-5 and 1e-4. For a winning joint word/bigram layout, include
   both original text and self/rival attribution. Keep the original top two
   recipes if absent from this grid. Also cross shares 0.5, 0.75 and 1.0 with
   penalties 3e-5 and 1e-4 using finishing-order weights (1, 0.25, 0.0625) on
   the best initial layout: that objective improved the same 0.25-teacher
   recipe in both screening quarters. This amendment precedes later text scoring;
   it responds to the observed training-target gains rather than claiming an
   independent second screen.
   The first full-teacher fits improved both quarters at the lower penalty
   boundary. If a full-teacher recipe still leads at 1e-5 after the full grid,
   test 3e-6 and 1e-6 on both word layouts using the same two earlier quarters.
   Preserve the initial tuning registration and results before adding these
   candidates. This is one boundary check before selecting, not an open-ended
   search on the later comparison.
3. Retain the corrected baseline and the best two alternatives; add 2024Q3,
   2024Q4 and 2025Q1, then select on all five development quarters.
4. Lock that recipe before comparing it with the corrected baseline on
   2025Q3–2026Q2. Report paired race log loss, date-cluster intervals, quarter
   stability, coverage and calibration. A current candidate refit requires
   the upper end of the paired interval below zero, improvement in at least
   three quarters, and successful replay of the saved artifacts.
5. Run the separately registered incremental-value probe against the saved
   generalist/performance forecasts: fit pool weights through April, score from
   8 May, and compare corrected versus selected text on identical fields.

Every stage has a seven-day purge. Vocabulary and IDF fit only on the relevant
earlier rows. Optimiser outcomes and calibration boundaries are recorded.
Common encodings are reused in a bounded four-entry / 2 GiB in-memory cache;
the fitted artifact carries no training matrix. Prediction caches verify actual
inputs, recipes, source code, content bytes and runner alignment before reuse.
Later quarter fits can run in two separate batches when at least 7 GiB of
memory is available. Each batch runs the same evaluator and writes private
receipts. The final evaluation assembles their verified forecasts on the
original dates, without changing the choice rule or scoring population.

These comparisons remain retrospective: these historical periods have informed
previous research. They do not create a fresh untouched holdout or qualify a
new live ensemble automatically.

## Verification and measured results

The final modelling, text, feature-signature, layout, leakage-pattern and
chronological-fold regression group passes **113 tests**. Its 20 warnings are
existing pandas/LightGBM warnings. Ruff passes the touched Python files. Logs
are under `data/audit/text_improvement_20261108/`.

An accepted-source check for the eventual current refit contains 757,098
runners / 80,480 valid races through 6 November, excluding 137 invalid or
invalid-price fields. Winner labels agree with recorded first-place finishes.
This checks the input contract, without scoring those later outcomes.

The projected raw-source path reproduces all 700,639 historical canonical
runner keys with identical winner, finish, field-size and price values. Its
name spelling also matches after the canonical strip/title operation. The
candidate refit enforces exact reconciliation of this historical prefix,
including text and BSP, before it fits; it cannot quietly use different
preparation for the current model.

`verify_text_inputs.py` independently recomputed all legacy and self/rival text
preparation from the raw corpus and matched the cached values. There are no
duplicate runner keys and no differences in whole-field availability between
the two attribution modes. One runner differs individually, in a race already
unavailable under both modes. Its receipt is `input_verification.json` in the
amended development directory.

The 19-recipe screen covers **6,178 identical races / 55,776 runners** in
2024Q1 and 2025Q2. These development results select further tests; they are not
the later-period estimate.

| Screen recipe | Race log loss |
|---|---:|
| Corrected calibrated reference | 1.912134 |
| Self/rival attribution + 50% teacher | **1.895838** |
| Original layout + 50% teacher | 1.896247 |
| Original layout + 25% teacher + finishing order | 1.900057 |
| Original layout + finishing order | 1.907452 |
| Original layout + stronger L2 (1e-4) | 1.909853 |
| Character patterns | 1.911583 |
| Self/rival attribution alone | 1.911622 |
| Rare terms (min_df=5) | 1.911801 |
| Separate field normalisation | 1.921801 |

Separate field normalisation loses against joint normalisation at all three
tested penalties in both quarters. Recency and longer phrases also fail to
improve the reference in this screen. Character results are mixed, and rare
terms yield only a very small gain. Those representation controls use hard winner targets;
their result does not refute every possible representation/target interaction.

The primary focused grid contains 31 recipes (including the corrected reference),
with eight completed, byte-verified fold forecasts reused unchanged. The
boundary check added four recipes, for **50 distinct recipes** across the
broad and focused screens. The strongest matched comparisons were:

| Full-teacher penalty | Self/rival layout | Original word layout |
|---|---:|---:|
| 1e-5 | 1.883424 | 1.883549 |
| 3e-6 | **1.882043** | **1.882090** |
| 1e-6 | 1.885005 | 1.884969 |

These are the same 6,178 development races. The two 3e-6 recipes and the
corrected reference advance. The four-quarter comparison remains outside
these choices. The original tuning registration and scores are preserved in
`pre_registration_primary.json` and `screen_primary.json`.

The choice was locked on **15,430 races / 142,637 runners** across five earlier
quarters. The corrected reference scored 1.936723, self/rival attribution
scored 1.909792, and the original word layout scored **1.909457**. Both new
recipes beat the reference in every development quarter. The small difference
between the layouts favours retaining the original words and verdict ranks.

`selected_text_spec.json` fixes the original jointly normalised word
unigrams/bigrams with minimum document frequency 20, penalty 3e-6, uniform
race weights and a single win stage. The full teacher uses a complete BSP
field where available, then complete SP, then the actual winner. A separate
purged later slice of training calibrates the score against actual winners.
The recipe was written before either later-quarter batch started.

## Later-quarter evaluation

The locked recipe improves every later quarter on **11,893 matched races /
109,495 runners**. Race log loss falls from **1.928073 to 1.908891** (about 1%).
The paired change is **−0.019183**, with a 95% interval of
**[−0.022945, −0.015032]** from resampling race dates. This measures a
chronological retrospective comparison, not prospective results.

| Quarter | Corrected reference | Selected text | Change |
|---|---:|---:|---:|
| 2025Q3 | 1.962132 | 1.943996 | −0.018136 |
| 2025Q4 | 1.918629 | 1.900785 | −0.017845 |
| 2026Q1 | 1.875158 | 1.849183 | −0.025975 |
| 2026Q2 through 3 June | 1.946651 | 1.931223 | −0.015427 |

`confirmation.json` and the keyed `confirm_*.parquet` files contain the
original results. Each later quarter uses its own earlier training rows and
purged scale calibration. The 2026Q2 evaluation stops before the reserved
holdout and purge.

The complete text fields cover 11,893 of 11,905 eligible later races (99.90%).
The other 12 fields are unavailable for this expert. On the matched fields,
runner Brier error improves from 0.088545 to 0.088049, and top-pick accuracy
rises from 27.91% to 28.48%. The ten-band calibration error rises slightly,
from 0.00173 to 0.00214. The log-loss and Brier gains therefore do not imply
improvement in every calibration statistic.

`verification.json` records an exact replay of all eight saved models from
raw text, without outcome or price columns. It verifies purge dates, shuffled
runner order, independence from label/price values, whole-field masking when
text is missing, and the saved model digests.

## Incremental ensemble value

The registered probe fits pool weights on April forecasts and scores **1,142
races / 10,610 runners** from 8 May through 3 June 2026, with a seven-day purge.
It keeps the full reference population and falls back to the remaining
components when text is unavailable.

| Pool | Race log loss | Text weight | Change vs generalist + performance [95% date interval] |
|---|---:|---:|---:|
| Generalist + performance | 1.848852 | — | — |
| Plus corrected text | 1.848693 | 3.43% | −0.000159 [−0.000858, +0.000567] |
| Plus selected text | 1.848756 | 6.66% | −0.000097 [−0.001492, +0.001462] |

Replacing corrected text with selected text in the pool changes log loss by
+0.000063 [−0.000850, +0.001056]. Neither comparison establishes incremental
value. The saved generalist already carries older text inputs; this is a
small retrospective probe against that frozen reference. The next admission
test should use a refreshed generalist, with a matched comparison that removes
its old learned text inputs before adding the separate text expert.

## Current candidate and adoption

The standalone recipe clears the refit criteria. The fitted candidate is at
`data/research/meta_learner/text_tuning_20261108/current_candidate_20261106/model.joblib`.
It uses **753,425 runners** with complete text through **6 November 2026** and
111,292 terms. Its calibration fit uses data through 31 July, with the scale
estimated on 7 August–6 November and the final coefficients refitted through
6 November. Its saved scale is 1.068089.

`metadata.json` records the accepted dataset, exact historical-prefix check,
recipe and artifact digests. `candidate_verification.json` records the
independent reload, recipe match, valid race sums and unchanged predictions
under shuffled runner order on 102 complete sample races without outcomes or
prices. The artifact is 3.2 MB. Those replay checks are mechanical; the full
refit has no separate held-out performance estimate.

Keep the fitted text expert as a separate candidate. The one-quarter pool
probe did not justify adding it to the existing reference pool. The legacy
published text scores, installed store and selected win run
`20261104_235300_444862` were not replaced. `status_after_code.json` records
the existing store/model compatibility problem; their normal refresh and
refit are separate from this text-model result.

## Reproduction and use

The final tuning experiment is
`data/research/meta_learner/text_tuning_20261108/`. Its prerequisites are the
completed amended screen and the accepted corpus described above. These are
deliberately fixed historical references; a source refresh is a new experiment,
not a reason to overwrite their receipts. Use a new output directory for a
replay of the experiment. The source review copy in `source_snapshot/` records
the original screening entry point separately from the later refinement and
quarter-batching entry point.

```powershell
$textRun = 'data/research/meta_learner/text_tuning_replay'
.venv\Scripts\python.exe -m research.meta_learner.text_tuning prepare --output $textRun
if ($LASTEXITCODE -ne 0) { throw 'Text preparation failed' }
foreach ($phase in @('screen', 'refine', 'confirm')) {
    .venv\Scripts\python.exe -m research.meta_learner.text_tuning $phase --output $textRun
    if ($LASTEXITCODE -ne 0) { throw "Text $phase failed" }
}
.venv\Scripts\python.exe -m research.meta_learner.verify_text --experiment $textRun
if ($LASTEXITCODE -ne 0) { throw 'Text artifact replay failed' }
.venv\Scripts\python.exe -m research.meta_learner.text_ensemble_probe --experiment $textRun
.venv\Scripts\python.exe -m research.meta_learner.fit_text_candidate --experiment $textRun --as-of 2026-11-06 --output "$textRun/current_candidate_20261106"
```

The last command fits once into a new directory. It reads the exact accepted
dataset from the experiment, validates complete settled fields, uses raw RTV
text and saved canonical BSP links, and verifies joblib reload before saving
metadata. The final refit can include more recent settled results; its
evaluation claim belongs to the earlier recipe forecasts, not those refit rows.

The saved object is a pipeline `src.modelling.text.TextExpert`. Supply complete
fields with `race_id`, `horse_id`, `horse_name`, `race_date`, `num_runners`,
`tf_comment` and `tf_verdict` to its `predict` method. Missing measurements use
blank/None values, not fabricated text. Probabilities preserve the input row
order and sum to one within supported fields; unsupported fields return NaN.
Its default prediction path rebuilds attribution from those raw columns using
the saved encoder. Research caches are not part of inference.

The reusable [candidate recipe](../../reference/modelling/timeform_text_market_v1.json)
contains the full `TextExpert` constructor settings and matches the locked
development recipe. It does not alter either the legacy score materialiser or
the selected numeric predictor.
