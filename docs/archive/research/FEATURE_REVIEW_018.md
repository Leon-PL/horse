# Feature review 018

**Current policy:** the user's28 October clarification supersedes the earlier
decision to leave verified repairs isolated after a weaker score. The026
[canonical integration](CANONICAL_FEATURE_REPAIRS_026.md) makes those fixes
part of ordinary training and live preprocessing. Experimental additions
remain separate. The018 results below remain unchanged historical evidence.

The installed 151-input model remains paired with its existing histories while
the next feature generation is checked. No prediction models were fitted during
the source audit. Candidate code is in `_research_snapshots/018_feature_repairs`.

## Reproduced errors

| Finding | Reach in the existing 753,734-row store | Candidate correction |
|---|---:|---|
| Actual synthetic surfaces missed by the pedigree predicate | 179,758 runners in the preceding audit | Recognise Allweather, Polytrack, Fibresand and Tapeta; keep natural beach sand distinct. No new AW predictors in the repair-only comparison. |
| Unlisted going descriptions silently labelled Good | 42,532 rows across 99 descriptions | Normalise whitespace, preserve explicitly defined existing mappings, otherwise classify the main description; missing/unrecognised stays Unknown. This count includes variants already meaning Good, so it is not the number of incorrect final labels. |
| An imputed official mark used to calculate a real rating movement | 74,896 nonzero movements involving an imputed current or preceding mark | Compute movements only from observed marks; an unknown latest mark cannot be replaced by an older convenient known mark. |
| Different rating scales subtracted | 57,967 racing-code transitions, 38,250 with nonzero old mark changes | Do not interpret hurdle/chase/Flat scale switches as mark rises. Conservatively leave turf/AW switches unknown without both published ratings. |
| Missing race class used as numeric zero, or assumed Class5 for adjusted speed | 197,646 unavailable current classes; 22,251 known/unknown transitions with nonzero changes | Strict class1..7 parsing; unavailable transitions and class-adjusted performances remain unknown. |
| Four-runner baseline assumes every horse places | 12,756 rows | Match the model's top-three target in these fields: prior0.75. |
| Impossible historical weights remain in features despite the new scraper guard | 3 source weights of1–5lb | Preserve raw evidence, mask before cleaning, retain an observation flag, and exclude imputed weights from historical weight movements. |
| A card's columns depend on unrelated categories in its batch | Reproduced with one Flat, all-male card | Always emit the selected Chase and Female indicators; absent categories are zero. |
| Missing measurements replaced by zero in the older full-history live path | Reproduced with missing marks, draws and margins | Preserve measurement uncertainty; fill only actual binary category indicators. |
| No-stalls Flat/bumpers acquire fabricated draw interactions | Structural reproduction | Mask draw-derived measurements when a stall is unavailable, as well as in hurdle/chase races. |
| Several connection/condition histories count unknown result rows as starts |92 unresolved historical result rows exist; effect propagates to later histories | Use the existing observed-start flags in those counters. |
| Recent-form averages depend on history-build batch size | First rebuilt generation changes four selected encoding signals; maximum field-standardised difference0.0204 | Replace truncated-tail approximations with exact per-entity exponentially weighted accumulators. Missing-result streaks retain the correct earlier state. |

The BHA documents separate hurdle/chase ratings and possible separate turf/AW
ratings. That supports withholding unsupported mark differences; it does not
say every horse has a distinct rating on every surface.
[BHA handicap-rating guide](https://www.britishhorseracing.com/regulation/guide-to-handicapping/translating-performance-figures-initial-handicap-rating/).
The official going description represents the main condition, with variations
reported in brackets. The existing broad grouping is retained rather than tuned
against prediction results.
[BHA racecourse instructions](https://media.britishhorseracing.com/bha/Rules/BHAGI/Section3_The_Racecourse.pdf).

## Overseas and missing local experience

There are26,961 first-local rows whose pre-race source indicates experience,
including13,492 hurdle/chase rows.315 such rows occur at Cheltenham, although
this includes the beginning of the local database and is not315 proven French
visitors. The existing hidden-history flag is not selected by the current model;
only its simple form-character count is selected. The earlier public-results
pilot was restricted to Flat racing.

A new, separate cohort selected180 Cheltenham horses in2021–March2026 using
pre-race experience only. All180 profiles were retrieved without errors.
They contain2,331 dated records, of which1,022 precede the selection date.
After outcome-validity checks and removal of locally recorded performances,
97 horses have usable earlier external results, covering589 local rows.
Earlier venues include Auteuil, Compiegne, Pau and other French courses.
These numbers describe coverage, not improved predictive accuracy.

A second cohort extends this beyond Cheltenham to first-local visitors in
UK Class1 hurdles/chases:176 profiles, no collection errors,164 horses with
usable earlier results across1,543 local rows. Its membership uses race class,
discipline and pre-race experience, without selecting on results or guessing
nationality. Combined with the existing Flat pilot and after duplicate checks,
the three snapshots supply earlier form for1,759 horses across17,283 local
runner entries. The dated source union contains48,310 unique performances;
every prediction still filters those records to strictly earlier dates.

For example, Kaline Des Boullat has dated French performances before its first
local Cheltenham row on24 January2026. Its profile is archived with identity,
retrieval time and hashes. No current rating, current age, lifetime total or
later race outcome is used as an earlier feature.
[Sporting Life profile](https://www.sportinglife.com/racing/profiles/horse/1094983).

The dated-result features exclude the prediction date, respect cohort
eligibility and avoid counting local results twice. Partial foreign results
are never inserted into Elo/Glicko field calculations. Unseen local experience
continues to mean uncertainty, not a fabricated beginner's career or invented
ability rating.

## Designs kept separate from corrections

Six proposed predictors distinguish available local history, source evidence
of unseen experience, an intervening unrecorded run, contradictory source
recency, and missing-aware observed recency in days/log days. They do not turn
a source form string into an asserted complete career count. Contradictory
source gaps cannot override a more recent verified local start.

These six predictors receive their own comparison against the corrected
baseline. Dated public results remain an additional research projection until
their coverage and predictive value support integration. Duplicate feature
removal, alternative speed scaling, country/context terms and DNF-aware
connection encodings are also design questions, not automatic fixes.

## Validation and compute

Source/cohort/helper checks:29 passed. The sealed candidate checks:56 passed,
including state resumption, observation flags, rating scales and historical/live
parity. All151 selected inputs passed future, same-day outcome, permutation and
price invariance. No horse appears in two different races on one historical
date, so an additional same-day horse-history ambiguity was ruled out here.
Reviewing the first rebuilt generation exposed the averaging bug in the final
row above. Its correction passed38 additional targeted checks, including400
consecutive missing results and several different batch sizes. The first
generation and its reports are retained; it was never trained or installed.
The final candidate signature is `93e81f0cd8e06fe4`. Its rebuilt history passed
review at21:36UTC on27October2026. All151 selected inputs were checked across
753,734 rows, with no wholly missing or infinite input. The four selected EWMA
derivatives exactly match an independent single full pass over real history
(maximum absolute difference0 for every derivative). Labels, prices, runner
identity and14 unrelated control features were preserved. There are32 changed
selected inputs; their row counts and missingness changes are recorded in the
review receipt. No model was fitted on the earlier truncated-state generation.
Results and current signatures are saved
under `data/audit/feature_review_018` and `data/audit/model_improvement/018_feature_repairs`.

The planned screen uses one October–December2025 window, two years of fitting
rows,500-tree cap, learning rate0.025 and win-only fitting. Feature histories
retain all earlier observations. Four matched fits compare original features,
objective repairs, the six coverage predictors, then seven actual dated-public-
performance predictors. The final comparison combines the separately frozen
Flat, Cheltenham and Class1 jumps cohorts, rejects conflicting records and preserves the
earliest independently eligible date. Its source/causality checks passed9tests.
Calibration and purging
remain native. This is a development screen on an already used period, with
no automatic deployment or further quarterly jobs.

The four sequential fits completed at21:48UTC on27October2026, in about ten
minutes. The registered window contains240,225 training runners and3,233 test
races/30,081 runners. Fitting and calibration use the native training path.
The matched original baseline is cached by its complete input, recipe and
runtime fingerprint for reuse in subsequent compatible screens.

| Fixed screen variant | Race log loss | Runner Brier |
|---|---:|---:|
| Original151 inputs, matched cheap fit | 1.849008 | 0.08548189 |
| Objective repairs151 | 1.851337 | 0.08547510 |
| Repairs plus coverage157 | 1.850816 | 0.08554568 |
| Repairs, coverage and dated public form164 | 1.849820 | 0.08548911 |

None improves both primary scores against the original matched baseline.
The repair log-loss difference is+0.002330, race-bootstrap95% interval
[-0.001050,+0.005609]. Adding public form to coverage improves the point
estimate by0.000997, but its interval[-0.004396,+0.002447] includes either
direction and the combined candidate still trails the original baseline.
No candidate is promoted and no extra quarterly fits are queued.

The common market comparison retains3,125 complete fields/29,077 runners.
Race log loss is1.738314 for the same-time market,1.850606 for the cheap
original baseline,1.853321 for repairs,1.852437 for coverage and1.851518 for
public form. This batch has not demonstrated a smaller market gap. The full
016 historical Q4 model scores1.844302 on all3,233 races; that context uses
more training history and trees, so it is not an isolated feature comparison.

The dated public information reaches326 test runners in287 races:158 Flat,
105 hurdle and63 chase rows. On the287 whole fields containing public form,
the public candidate has log loss1.982370 versus1.982059 for the original
baseline. Cheltenham has only47 test races. These overlapping subgroup
diagnostics do not justify a venue-specific or overseas-specific promotion.

Independent replay of all four saved models and both calibration stages
passes. No screen winner receives a probability below0.001. The repairs'
pre-isotonic score is1.849236 versus1.849555 for the baseline at that same
stage; the final isotonic step reverses this small difference. This motivates
019's separate training-only calibration-family check, not a post-test switch
to whichever stage looks best. Primary results, calibration-bin counts and
largest winner-loss changes are preserved in `report.json`, `diagnostics.json`,
`low_probability_winners.csv` and `largest_repair_log_loss_increases.csv`.

## Live pedigree parity

The historical pedigree backfill queues horses after they appear in local
results. A new arrival could therefore receive breeding features in historical
training but have no corresponding pedigree on its first live card. The live
card path now fills missing sire, dam and damsire from the existing local
pedigree table, then from an identity-checked public horse profile if needed.
A separate card-only cache stores those immutable parent identities and source
metadata. It does not rewrite historical results or import current ratings,
age, trainer, form totals or later performances. Provided parents remain
unchanged. Missing or mismatched profiles retain unknown values.

The ingestion checks passed104 tests, including simultaneous look-ahead cards
for the same horse. The installed model and feature definitions remain the
016 generation (`8fd1c9c690c4a608`); the generation compatibility check still
passes. This is a serving correction, with no measured accuracy claim yet.

Because the prospective comparison pins ingestion code too, a fresh same-model
study was registered at
`data/forecasts/forward/20261027T213713Z_card_pedigree`, eligible from28October.
The superseded study had no observations or forecasts; its original protocol
is retained intact. No forecast collection was scheduled.

## Further information gaps

An original Galway source response quotes its first prize in EUR and exposes
per-ride `jockey_claim`, neither of which survives in the current raw feature
contract. Prize comparisons across jurisdictions can therefore mix nominal
currencies. The claim field's units also need verification before it can alter
carried weight. These require a source-backed historical reconstruction or a
separately tested missing-aware design; a modern exchange rate or an assumed
claim unit must not be applied retrospectively. Original response archived as
`data/audit/feature_review_018/source_schema_galway.json`.

An age/birthdate cross-check found109 disagreements among753,405 rows with a
known birthdate. Many involve southern-hemisphere imports, whose racing age
cannot be reconstructed by blindly subtracting calendar birth years. A few
northern-hemisphere records also contain contradictory source fields. These
cases are retained in `age_birth_date_cases.csv`; no unsupported blanket age
replacement was made. No recorded age falls outside2–35 years.
