# Established-handicap investigation030

Completed 28 October 2026. The installed 029 win/place model remains unchanged and
its feature generation and 42 pinned source files still match. This investigation
used saved forecasts and source checks, followed by one small screen: **two
400-tree win fits, 107.6 seconds**. No calibration learners, place models, feature
rebuild or production refit were run.

## What the weight audit establishes

The 7,910 runners in 951 complete Q1 handicap fields all had at least five prior
local starts. Every stored previous-run weight change reproduces correctly:
there is no arithmetic/data-lag error to patch in this input.

Most of the net model-versus-market log-loss deficit, 87.8%, comes from winners
whose preceding race was also a handicap in the same discipline. Merely fixing
cross-code transfers does not address the bulk of these errors.

A declared-weight move is not the same quantity as a handicap-rating move.
Among 6,583 runners with observed marks in comparable current/previous handicap
scales, median absolute weight movement is 4 lb versus 1 rating point. Of 2,047
weight movements of at least 7 lb, 1,447 have mark movements of no more than 2 points.
Rubber Ball's allocated weight fell 11 lb before his 28 February race, while his
mark fell only 1 point. The remaining movement reflects a different race
allocation and potentially different allowances/conditions; it cannot simply
be labelled an11lb improvement in his handicap position.

The [BHA's explanation](https://www.britishhorseracing.com/races-can-horse-run-handicap-rating/)
sets out how field ratings, age allowances, rider claims and penalties affect
allocated weights. This is an interpretation issue and feature-design lead,
not evidence that the recorded declared weight should be replaced arbitrarily.

## Missing source information

The scraper discards the source's `jockey_claim`. In a fixed simple-random
sample of 32 of the 951 fields, 80 of 269 runners have explicit 3 lb, 5 lb or 7 lb claims;
27 fields contain at least one. The full 48-field sample, including 16 targeted
error cases, contains 391 runners and 116 explicit claims. All 391 declared
weights match the local data exactly. Four runners also have an explicit 5 lb
weight-penalty field, which is retained as separate metadata.

The [BHA weighing-out guidance](https://www.britishhorseracing.com/faqs/what-are-the-weighing-out-procedures/)
explains the claim deduction. Independent Racing Post result checks confirm:

| Runner | Source declared lb | Reported claim lb | Racing Post carried lb |
| --- | ---: | ---: | ---: |
| Marche D'Aligre, Newbury 28 February |149|5|144|
| Brentford Hope,same race |166|7|159|
| Applicationofcash, Taunton 9 March |156|7|149|

Sources: [Newbury result](https://www.racingpost.com/results/36/newbury/2026-02-28/912399/),
[Taunton result](https://www.racingpost.com/results/73/taunton/2026-03-09/913314/).
The collector keeps declared weight, claim and penalty separate; it does not
add an already-incorporated penalty again or silently convert absent claims
to observed zeros. Claim-adjusted weight is not asserted to include overweight.
An existing archived pre-race card also contains six explicit claims among nine
runners, confirming this information is available before the result.

This gap is common but is not yet shown to explain most prediction errors.
In the 32 random fields, claimants won 12 races; the model assigned them 10.09
expected wins and the market 9.67. The model was not collectively below the
market on these riders. No large historical backfill or probability bonus was
launched on the strength of this small sample.

## Saved-model explanations and redundant inputs

The existing historical 029 model exactly reproduces all 122 runner probabilities
in 16 selected handicap error fields. Only the 16 requested runners receive tree
explanations, avoiding unnecessary work on the other runners. These explain
raw learner scores, not causal effects or calibrated probability increments.

Margin momentum, Glicko rank and connections remain prominent in the errors.
Two selected input pairs are algebraically redundant:

- centring an already centred momentum signal;
- subtracting the maximum from centred momentum instead of raw momentum.

Across all 566,442 actual historical training runners, the largest pairwise
discrepancies are below 1.8e-15. One pair is identical at float32 precision on
every row; the other differs on just two rows. Removing these copies changes
column sampling during learning, so redundancy alone does not guarantee better
forecasts.

## Cheap screening result

The one fixed candidate removes just those two copies: 149 inputs versus 151.
Both small learners use 400 trees, learning rate 0.04 and otherwise the native
win learner, trained on 239,828 runners from 24 September 2023 to 23 September 2025.
Evaluation is on the same 3,233 complete October-December 2025 fields and 30,081
runners. Both use the same simple race normalisation, without fitted calibration.
This is development triage, not independent confirmation or an estimate of the
installed full model's performance.

| Cohort | Baseline log loss | 149 inputs | Baseline Brier | 149 inputs |
| --- | ---: | ---: | ---: | ---: |
| All 3,233 fields |1.847941|1.850848|0.085482|0.085569|
| 1,071 established-handicap fields |1.987342|1.990500|0.091748|0.091940|

Both primary scores worsen. Log-loss uncertainty includes zero, so this does
not prove harm in every setting; it provides no reason to pay for a full
retraining cycle now. The 149-input list remains an unselected experiment.
There was no follow-up retuning on this window.

The reusable `scripts/quick_win_feature_screen.py` caches each native learner by
its actual numeric inputs, targets, weights, feature names, learner settings,
seed and runtime. The baseline key excludes the candidate and report directory,
so an unchanged baseline can be reused for later subset screens. Separate
receipt directories are selected with `--output`. The script as executed is
preserved in `quick_screen/source_at_execution.py`; the later CLI-only addition
of `--output` does not change learner mathematics.

## Implemented tools and next priority

- `scripts/audit_handicap_weights.py`: complete-field source collector,
  separate claim/penalty evidence, weight/mark decomposition and cached resume.
- `scripts/explain_established_horse_errors.py --handicap-review`: shared
  saved-model explanation tool; no fitting.
- `feature_sets/handicap_compact_030.txt`: tested 149-input proposal, not selected.
- 15 targeted tests pass; the existing production weights, default 151 inputs
  and feature-store generation are unchanged.

The next performance candidate should represent recent form in the context of
burden and opposition, keeping official-mark movement distinct from changes in
race allocation. Claims and penalties should be captured as dated source
metadata before they enter features. Current revised official performance
figures must not be inserted into old pre-race rows as if they were known then.
Use one capped win-only screen first; reserve full validation and deployment
work for a sufficiently promising, coherent batch.

Evidence is in `data/audit/handicap_weight_review_030/`, starting with
`final_status.json`, `weight_mark_decomposition.json`, `source_summary.json`,
`explanation_report.json`, and `quick_screen/report.json`. Earlier404 responses
were a collector URL-format error, repaired before successful source collection;
sample membership was unchanged and no model training was involved.
