# Established-horse error audit, 28 October 2026

This audit follows the user's request to focus on established horses. It uses
the saved026 forecasts, made by a model trained through 22 December 2025, on
January–March 2026 races. It does not score the current installed model on its
training races. These are previously examined development dates, not a new
untouched test. The market comparator is the existing016 historical benchmark.

**A definite Glicko calculation error is fixed in the normal source code.
The historical store and installed027 model have not yet been rebuilt.**
The new code deliberately makes the feature generation stale; normal live
prediction refuses that mismatch. Rebuild and train once after the audit batch;
do not bypass the guard or mix repaired inputs with old trained weights.
No prediction improvement from this fix is claimed before that evaluation.

## Where the gap remains

“Established” means at least five observed local starts before the race,
including known non-finishes. This does not guarantee complete overseas or
point-to-point history, or five starts in today's discipline.

| Complete fields | Races | Model race log loss | Market race log loss | Model winner accuracy | Market winner accuracy |
|---|---:|---:|---:|---:|---:|
| All market-paired fields | 2,469 | 1.78863 | 1.68661 | 31.25% | 35.93% |
| Every horse has at least 3 local starts | 1,756 | 1.87937 | 1.79098 | 28.28% | 32.63% |
| Every horse has at least 5 local starts | 1,129 | 1.84122 | 1.75756 | 28.54% | 32.55% |
| Every horse has at least 10 local starts | 373 | 1.78352 | 1.70560 | 28.69% | 33.24% |

Lower log loss is better. For fields where every horse has five starts, the
model-minus-market loss gap is +0.08366; a date-cluster bootstrap interval is
[+0.06214, +0.10616]. There is a clear remaining gap in familiar fields.

Runner-level attribution gives established winners 62.64% of the total net
log-loss gap. Among established runners, 92.37% of that gap comes from horses
with at least three starts in today's code. Sparse experience in the current
discipline therefore does not explain most of the deficit either. These
attributions are descriptive and overlap with race-type/handicap breakdowns.

Broad probability calibration is reasonably close: the 100 established horses
assigned more than 50% averaged 63.54% predicted probability and won 62 races.
This small group does not prove perfect calibration, but it argues against
flattening every strong forecast as the main repair. Which horse receives the
probability is a larger remaining problem.

## Verified calculation error and normal-source repair

Glicko records a rating and its uncertainty after an informative result. At the
next race, uncertainty increases with the time since that result. The old code
overwrote the stored uncertainty when computing a pre-race value, even if the
horse subsequently did not finish and received no rating update. Its stored
date still pointed to the earlier informative result. The next appearance
added the same elapsed time again.

For example, start at uncertainty100 on 1 January. A non-finish on 1 February
followed by a race on 1 March should give pre-race uncertainty139.63 under the
existing70-per-square-root-month setting. The old code produced156.48, because
it charged both the January–February and January–March gaps.

The same mutation let two runners belonging to one trainer receive different
pre-race uncertainties in the same field. Pending cards could also advance
uncertainty without advancing the date. Trainer Glicko and Glicko-2 are disabled
in the active recipe, but their shared time-accounting defect is also repaired.
The policy of excluding non-finishers from the rating outcome update is
unchanged; deciding how to learn from falls and pulled-up runs is separate work.

The normal implementations in `src/ratings.py` now calculate pre-race
uncertainty separately and persist it only with the corresponding result
update and date. This follows the uncertainty accounting in
[Glickman's original specification](https://www.glicko.net/glicko/glicko.pdf).
There is no feature flag or alternate training track for the correction.

The first regression run failed six of eight checks under the old code. After
the repair, all eight pass, including pending-card state preservation,
duplicate-entity consistency and batch/incremental agreement. Together with
the existing rating suites, 29 rating checks pass. The store/live/signature
suites add 16 distinct passing checks after fixing a separate missing-date
comparison: `src/rating_reuse.py` incorrectly treated matching unavailable FX
dates as changed inputs. That repair changes validation of reuse, not the
feature values or permission to reuse old, incompatible rating states.

A replay over 678,085 development-history rows first reproduces the old stored
ratings exactly at storage precision. Applying the fixed calculation changes
the within-race Glicko rank of 789 of 15,521 established runners in the paired Q1
fields (5.08%). Across all 21,374 paired runners, the mean absolute uncertainty
change is2.46 rating points and the maximum127.97. Small effects spread through
opponents and subsequent updates; the high number of changed values is not a
count of severe direct errors.

Examples among the missed winners:

| Horse | Old uncertainty | Repaired uncertainty | Old rank | Repaired rank |
|---|---:|---:|---:|---:|
| James The Brave |179.16|156.76|5|5|
| Rubber Ball |155.40|144.54|5|5|
| Dj Pete |160.34|149.22|4|4|
| Constitution Hill |350.00|311.03|5|4|

These are input corrections, not forecasts from a retrained model.

## What the case review suggests next

Sixty cases cover four directions: established winners the model underrated,
losers it overrated, winners where it beat the market and losers where it was
more cautious correctly. The saved historical learner exactly reproduces all
477 runner forecasts in these60 complete fields. Its additive explanations
reconstruct its raw scores; they are descriptive contributions, not causal
effects or calibrated probability increments.

1. **Handicap recovery and marks.** Twelve of the15 largest missed-winner cases
   are handicaps. Palamon received9.85% versus the market's41.89%; recent poor
   form/margin momentum and Glicko rank were major negative inputs. James The
   Brave received5.18% versus35.94%; the model recognised his lower weight/mark
   but strong negative trainer and recent-margin signals outweighed that.
   Investigate comparable marks, weight-adjusted performance and opposition
   strength together. Do not treat a lower official mark alone as proof of
   improvement: all established runners with a comparable drop of at least
   three points averaged7.00% model probability versus5.56% actual wins.

2. **Novice/conditions context and transferable form.** Thirteen of the15
   largest overrated-loser cases are non-handicaps; many have plenty of local
   starts but only a few in today's discipline. Douglas Hyde received61.61%
   versus31.08%, Frank Stamper60.03% versus33.13%, and Shoda53.81% versus29.15%.
   Glicko rank was the largest positive learner contribution in these cases.
   Distinguish bumper, Flat, hurdle and chase experience, and inspect whether
   global ranking overstates evidence from different conditions. This is a
   hypothesis requiring a matched feature comparison, not a diagnosed bug in
   every unsuccessful strong favourite.

3. **Known history gaps among established horses.** For116 of15,521 established
   runners, the source reports a positive gap at least three days shorter than
   the local-history gap. Bellaney Lord's source says28 days while local results
   imply363; the model gave2.24% against16.03%. A fresh, identity-checked public
   profile still lists no additional prior regulated result after10 March2025.
   It does not establish what the intervening activity was. Keep local-result
   recency and source-reported recency distinct; do not invent missing results
   or overwrite the historical date from a current profile.

The next batch should finish these input checks before one combined rebuild
and refit. Use the same repaired features and teacher inputs in both arms of
any experimental feature comparison. The current027 teacher adds older price
coverage as well as recent dates, so a new fit using it cannot attribute all
changes against026 specifically to the Glicko repair. Never transplant repaired
ratings into the old model and report that as a valid retrained improvement.

## Reproducible evidence

All new outputs are in `data/audit/established_horse_review_028/`:

- `report.json`, `paired_context.parquet`, cohort/reliability summaries.
- `selected_cases.csv`, complete `case_fields.parquet`, dated `case_histories.csv`.
- `case_explanations.csv`, historical importance and exact-reproduction receipt.
- `ratings_before_fix.py`, `glicko_report.json`, corrected development-only
  rating projection and `case_glicko_corrections.csv`.
- `source_cases/1007503.json`: sanitised dated public-source results and receipt.

Read-only audit scripts live in `scripts/audit_established_horse_errors.py`,
`scripts/explain_established_horse_errors.py` and
`scripts/audit_glicko_inactivity.py`. Their original contextual audit must not
be blindly rerun against a later rebuilt feature generation: the saved026
probabilities describe the old Glicko inputs. The strict join checks fail if
the context no longer matches those frozen inputs. No sealed026/027 evidence
was rewritten, no reserved later-period performance evaluated and no new model
fit launched in this audit.
