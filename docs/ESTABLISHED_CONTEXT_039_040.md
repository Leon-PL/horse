# Established-horse context: two further checks,039–040

Completed29 October2026. Neither fixed proposal earns a production change.
These are new win-feature comparisons after the other agent's038 work, which
already preserves claims, penalties, factual ride descriptions and reasons,
builds a context prototype, and installs the recent place-probability layer.
That work was inspected and retained rather than reimplemented. A separate
live-speed generation migration was running during these experiments.

Both experiments use immutable historical inputs. They do not modify ordinary
training, the current model, the feature store, the recorder or its source pins.
The two prototypes remain outside the154 default selected inputs.

##039: connection form relative to market expectations

Existing connection rates describe results but can confound recent form with
the quality of assigned horses. The fixed proposal instead aggregates earlier
wins minus complete-book starting-price expectations for each trainer/jockey
over60 calendar days. Its score is:

`(observed wins - sum of normalized earlier market probabilities) / (priced starts +20)`

The20 neutral pseudo-starts shrink sparse evidence. Four inputs are appended:
trainer/jockey excess rates centred on today's field, and log1p priced-start
support for each. Unknown names are not pooled; aliases are not guessed.
A missing price invalidates the entire earlier book rather than inventing
neutral evidence. Nonfinishers are genuine zero-win starts. This is a wording-
free descriptive result history, not a causal skill or profitability estimate.

All same-day forecasts use the unchanged prior-day view. Same-day results and
prices cannot enter that day's inputs. Full versus seeded calculations agree
strictly on all678,085 source rows, including missing values. Canonical price
math matches within1e-15 and complete-field expected wins sum to one. A real
permutation check agrees within1e-14; future mutations leave earlier rows
unchanged. Nine helper tests pass.

In the actual30,081-runner Q4 score population, trainer evidence exists for
29,881 runners and jockey evidence for29,797. Median support is49 and63 priced
starts. In the9,777-runner established-handicap target, support exists for
9,724/9,678 runners. Sparse source coverage therefore does not explain away
the measured target result.

##040: a horse's own observed racing pattern

The selected freshness inputs describe elapsed days and population/field
comparisons. This proposal adds the difference between today's log1p local
gap and the horse's median log1p gap, that historical median, and its support.
It retains the last five observed intervals ending within730 days and requires
at least three earlier positive intervals before declaring a usual pattern.
These constants were fixed before scoring.

Only actual starts advance the calendar, including nonfinishers. Pending cards
and nonstarters do not. A horse with two actual starts on one date is rejected
as ambiguous instead of arbitrarily ordered. Source IDs preserve identity
across name spellings. These are observed local intervals; unrecorded overseas
activity may exist. They do not diagnose fitness or claim complete exercise
history. No automatic probability bonus is assigned for a shorter/longer gap.

Eight helper tests pass. Full/prefix/seed agreement is exact on678,085 rows;
real permutation and future-outcome controls also agree exactly. Usable routine
exists for21,896 of30,081 scored runners, including9,697 of9,777 target runners.

## One small comparison per idea

The corrected154-input win baseline is the same as033/038. Both final production
win heads remain outside these experiments. The existing400-tree triage fit
is reused; only one400-tree candidate is fitted for each proposal, with158
inputs for039 and157 for040. Training uses24 September2023–23 September2025;
the fixed score period is October–December2025. These dates were already
inspected development, not fresh confirmation.

The original win training module is loaded from its sealed034 source archive.
Runtime, classifier settings, feature order and every numeric baseline training
argument must match its cached fit. Both screens reproduce all saved baseline
probabilities exactly before fitting a candidate. New place calibration and
the unrelated speed migration cannot force another unchanged win fit.
The generic entry point is `scripts/screen_frozen_win_candidate.py`.

Lower race log loss and runner Brier are better:

| Population | Baseline LL |039 LL |040 LL | Baseline Brier |039 Brier |040 Brier |
|---|---:|---:|---:|---:|---:|---:|
|All3,233 fields /30,081 runners|1.845981|1.845247|1.845848|.08542366|.08540733|.08537541|
|Established handicaps:1,071 fields /9,777 runners|1.981946|1.983893|1.982299|.09162586|.09172128|.09164178|

Both candidates show tiny overall point gains and worsen both target scores.
The overall date-cluster95% LL-delta intervals are039[-.003708,+.002430] and
040[-.003247,+.003270]. Neither excludes zero. Overall top-pick accuracy also
falls from31.09% to30.37%/30.78%; target accuracy from25.30% to24.37%/25.02%.
Binned calibration improves in some slices, but does not replace the worse
target proper scores.

**Stop both fixed proposals here.** The advance rule required lower overall
and target LL, neither runner Brier worse, and an overall date-cluster upper
bound below zero before another costly native check. There is no subsequent
window, shrinkage, feature-subset, seed or joint search on these outcomes.
This does not prove every connection-form or individual-cadence design useless;
it does mean these particular recipes have not earned installation.

The two small checks took106.8 and82.6 seconds end to end. Baseline fits:zero;
new candidate fits:two; native/place/calibration/current-data fits:zero.
Feature projection audits took148.4/146.0 seconds. A related27-test run passes,
including the17 new helper cases and10 existing cache/prior-market cases.
Those counts overlap the separate9/8 helper runs.

## Evidence and next boundary

Evidence directories are `data/audit/connection_form_039/` and
`data/audit/recovery_pattern_040/`. Each contains the fixed plan, source capsule,
preparation, historical projection, exact baseline replay, paired scores and
final_status.json.039 has a supplemental strict-array/source-coverage receipt.
Completed controllers and reports must not be overwritten.

No new source error was established by these two audits. The next work should
seek stronger incremental information rather than repeat these same patterns.
For the dated context already implemented in038, first expand a representative
coverage sample selected without outcomes and preserve observation times.
The original error-selected pilot is not sufficient evidence to train and
adopt commentary or claim-based predictors. Any current-market trajectory or
liquidity investigation is a distinct market-assisted proposal with its own
decision-time source contract, not a rescue of the failed simple036 blend.
