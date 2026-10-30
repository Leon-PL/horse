# Findings during the prediction audit

The original raw margins have the wrong interpretation in modelling. This
is a measured defect affecting 548,270 known finishing margins; it is not a
speculative extra feature. Corrections preserve source gaps and derive a
separate `beaten_lengths` result column, excluded from current-race inputs.
Its influence reaches the model only through lagged histories.

The class scale used by a margin-based speed feature also depended on the
order of finishers in the source table. The corrected scale uses all earlier
dates and excludes every result from the performance's own date. All 149
selected inputs pass seven invariance checks after this repair.

Incomplete BSP markets caused 133 teacher distributions to exceed a total
probability of one, with a maximum of 1.6357. They now fall back to one
complete SP market for the entire field. Ambiguous normalised identity
matches cannot silently pick the first price. Fields missing declared
runners have no teacher. This repairs training targets; the current hard-label
model did not use those soft targets, so it is not the explanation for all
of the current market gap.

The scraper also substituted 5.0 decimal odds for unreadable prices and zero
for unrecognised margins. New ingestion preserves these as unknown. There
is no reliable way to distinguish an old fabricated 5.0 from a genuine 4/1
quote, so historical quotes have not been guessed away. The real-data audit's
'zero invalid prices' describes stored values, not proof that all original
source prices were present. No race has every runner priced at the old 5.0
default.

## Avoiding false diagnoses

- All 753,734 training rows have horse IDs. A possible mixed-ID fallback
  problem is not evidence of widespread broken stored horse histories.
- No duplicate race/horse-ID rows or negative prior-start counts were found.
- Twenty horses with at least three records have an inconsistent inferred
  birth year. This broader screen includes short spans; the earlier
  year-spanning screen flagged ten. Imports and source age conventions
  need checking before editing them.
- There are 128 fields without exactly one winner and 22 with a declared
  size mismatch. Existing whole-field evaluation excludes invalid fields;
  dead heats are not necessarily bad results.
- A very low sum of raw implied probabilities is not by itself a corrupt
  race. The most extreme two-runner example, Wolverhampton 8 April 2024,
  followed a false start and nine withdrawals, with a large Rule 4 deduction.
  Its unusual published starting prices are real. We therefore did not add
  an arbitrary overround filter. Source result:
  https://www.racingpost.com/results/513/wolverhampton-aw/2024-04-08/863476/
  and original provider meeting:
  https://www.sportinglife.com/racing/meeting/2024-04-08/wolverhampton/103785

## Information gaps to address after the repair comparison

The active speed inputs remain margin proxies, despite a separate winning-time
cache. The old removed time-figure implementation has been recovered from Git
for inspection. It used a full-history par estimate and the same misinterpreted
margin column. It must not be restored unchanged: an estimate of a structural
quantity can still use future information. A replacement needs date-strict
pars, corrected cumulative margins and a documented time conversion. The
BHA publishes its lengths-per-second scale:
https://www.britishhorseracing.com/wp-content/uploads/2014/04/Lengths-Per-Second-Scale-Tables-2019.pdf

Public official performance figures are potentially useful, but historical
figures are revised as later form emerges. Publication dates and historical
versions are therefore required before retrospective use. No current revised
figure has been inserted into past training rows.


## Lost non-finisher labels (separate015 generation)

The scraper supplied finish_pos_label, but SQLite schema/column lists dropped it.
Histories recognised completed losses from numeric finishing positions or that
label, so non-finishers with position0 disappeared from prior-start counts.
45,097 such runners existed; reconstruction only accepts declared complete
settled fields containing a winner, restoring45,005. The remaining92 belong
to incomplete fields and stay unknown. No specific fall/pull-up reason is invented.
Explicit withdrawn/non-runner labels remain excluded. Live cards clear outcome
labels before history calculation. The repaired column now persists in SQLite.
The rebuilt753,734-row store changes previous-start counts on141,607 later
runners; maximum17 restored starts. Last-run non-completion is now informative
on34,079 rows rather than constantly0. The smoothed rate already had its prior
on old rows; its historical numerator/denominator now include genuine DNFs.

These correctness findings precede015/016 fits. They do not themselves prove
that the151-feature candidate predicts better. Original discovery and separate
post-rebuild audit receipts are preserved in015_nonfinisher_history.


## Raw recency is a demonstrated information gap for a later experiment

Read-only audit `scripts/audit_source_recency.py`, receipt
`data/audit/source_recency_20261027.json`: among680,477 rows with positive raw
source and observed local gaps,674,704 (99.15%) agree exactly.5,709 source gaps
are shorter, consistent with extra races absent locally;64 are longer and need
case review.26,947 first-local rows already contain a positive source gap,
where the active DB-derived recency defaults to60days. On1,541 eligible cohort
rows with a known prior public finisher,1,373 gaps agree exactly;167 source gaps
are shorter and1longer. Public profiles need not contain every start.

The blanket `src/model.py` exclusion comment claiming scrape-time recency is
contradicted by the data; the feature engineer already describes this field as
pre-race. This does not justify blindly replacing a date-strict feature: retain
missing/nonpositive handling, review conflicts, establish live/result parity,
and test the candidate on the same frozen chronology. No015/016 input or recipe
was changed following this finding. No profile availability is backdated.


## Evaluated and live numeric precision now agree

Registered benchmarks write selected inputs as float32; the full feature store
retains139 doubles among151 selected inputs. The old identity scaler preserved
whatever dtype it received, so a fresh full-store refit could train on different
numeric inputs, and live cards could bypass the benchmark's precision contract.
The shared scaler now converts to float32 in both training and prediction. Its
pickle reconstruction uses the same implementation. New native tests fit32/64
input frames and assert identical forecasts before and after serialization.

Frozen015/016 execution sources remain untouched. Their inputs were already
float32, so this restricted current-source upgrade preserves their values. All
four completed Q3/Q4 models replay exactly, both from their original projections
and the original double-precision store values (6,765 races per recipe, maximum
probability error0). Before adoption, the script repeats this proof for allthree
quarters and the Q1 place head, checks the exact source patch, archives current
qualification sources and records a new qualification recipe/source hash. It
never restamps the old benchmark as if it used the new source. The feature-store
generation/signature is unchanged because this is a downstream model input step.
