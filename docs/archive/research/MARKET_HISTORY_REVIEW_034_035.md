# Market-history transfer and performance context:034–035

Two proposals follow the installed033 update. Current production remains
`20261029_085433_011356`, with 154 inputs; no normal feature definitions or
weights have changed during this investigation.

## 034: transfer and recency of earlier market assessments

The same-scale histories introduced in033 correctly preserve separate Turf,
AW, Hurdle, Chase and Bumper records, but lose useful price information when an
established horse changes racing type. Source inspection confirms that Arklow
Lad had six Turf starts before his first AW race and Brown Boots six Hurdles
before his first Chase. Both had zero same-scale price support.

Across the 993 established-handicap Q1 fields,300 of the 377 zero-support runners
have valid prices in their three most recent starts across racing types. They
appear in 219 fields. Only 41 of those 300 cross the broad Flat/jumps boundary;
most are within a family. This is availability evidence, not a performance score.
In the Q4 screen cohort,622 of 726 zero-support runners have transferable history.

The fixed five-input proposal adds mean cross-scale log(N*p), latest price
assessment minus the earlier mean, price support, and explicit scale/family
change flags. The two price measurements are centred on today's field. Only
three actual prior starts within 365 days are retained. A missing latest book
cannot be skipped to manufacture a current trend. Official-rating units stay
separate and are never pooled into these price-only inputs.

The 678,085-row source snapshot contains no duplicate horse identities within
a day. Eight checks cover causal timing, independent row order, pending cards,
state immutability, scale flags, missing/expired books and ambiguous latest
starts. Full-history versus prefix/seed calculations agree on all 678,085 rows.

## Correct current baseline and small comparisons

The old 151-input screen is no longer the installed recipe. The new baseline
uses the corrected033 inputs and all 154 selected features. Every old input
column is preserved exactly; only missing horse-experience metadata is appended
for cohort reporting. One new 400-tree baseline is fitted, then reused exactly
for both candidates. There is no calibration, place, capacity search or quarter
sweep in these screens.

Train 24 September 2023–23 September 2025; test Q4 2025. These are previously inspected
development dates. Lower log loss/Brier is better.

| Q4 population | 154-input baseline LL | 034 transfer LL | Baseline Brier | 034 Brier |
|---|---:|---:|---:|---:|
|All 3,233 fields / 30,081 runners|1.845981|1.843494|.08542366|.08535360|
|Established handicaps: 1,071 fields / 9,777 runners|1.981946|1.979996|.09162586|.09157262|

034 passes its point-score rule. Date-cluster 95% LL-delta intervals cross zero:
overall[-.006468,+.001477],target[-.007992,+.004287]. Overall top-pick accuracy
falls 31.09%→30.59%; target accuracy rises 25.30%→25.49%. It is a probability-score
candidate, not evidence of a universal improvement. The candidate fit took
72.73 seconds; its 154-input baseline was an exact cache hit.

Descriptive source-group checks also show improvements on both sides of the
identified coverage gap. In the394 established-handicap fields containing
recoverable history, log loss improves1.991669→1.989379; in the other677,
1.976287→1.974536. No subgroup is selected for deployment. The new broad price
mean ranks10th and its recent-change input17th by training gain; importance
shows usage and does not establish a causal contribution or an independent gain.

## 035: finishing performance relative to earlier market assessments

A second proposal measures how a horse finished relative to its price-implied
pair comparisons. It treats normalized old win prices as strengths in the
Bradley–Terry expression p_i/(p_i+p_j), then compares expected and actual
fraction of other finishers beaten, with half credit for a tie. The comparison
formula is established; using win prices as those strengths is this project's
approximation and needs empirical testing. [Hunter 2004, Eq. 1](https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.pdf).

Only actual finishers are compared, with a complete original price book
required. A nonfinish is unknown ability evidence, and surviving a rival's
fall earns no extra comparison credit. This defines a conditional descriptor,
not an unbiased market error or an official performance rating. The proposal
adds a same-scale three-start mean, latest descriptor and support, all prior-day
and 365-day bounded. Ratings and pound conversions do not enter it.

Eight checks cover its mathematical direction/bounds, ties, nonfinishers,
complete books, state, timing and scales. Full prefix/seed equality passes;
the descriptor's per-field sum is numerically zero among compared finishers.
Historical measurements exist for 7,908 of 8,385 target Q1 runners; their latest
observation is known for 7,417.

| Q4 population | 154-input baseline LL | 035 context LL | Baseline Brier | 035 Brier |
|---|---:|---:|---:|---:|
|All 3,233 fields / 30,081 runners|1.845981|1.845130|.08542366|.08540685|
|Established handicaps: 1,071 fields / 9,777 runners|1.981946|1.982161|.09162586|.09165907|

035 fails the target rule and stops after its single 74.05-second candidate fit.
The slight overall gain does not justify advancement. No joint fit or parameter
revision follows: the registered joint rule required both standalone families
to pass. This does not rule out every expectation-based performance descriptor.

## One native win check, 034 only

The fixed 159-input candidate was checked through 22 December 2025 against
saved 033 full-native 154-input Q1 forecasts. The baseline is 033's updated
probability column, not its older 151-input reference. There are three native
win components for capacity/refit/calibration, with no baseline/place/current
fit in this check. Exact component checkpoints support interruption recovery.

Before fitting, advancement was set to require lower overall and established-
handicap race log loss, neither runner Brier worse, and an overall date-cluster
95% LL-delta upper bound below zero. A small uncertain point gain therefore
does not automatically trigger another expensive installed-model update.
Q1 was already inspected development; these intervals do not account for the
wider research selection.

| Q1 population | 154-input baseline LL | 034 transfer LL | Baseline Brier | 034 Brier |
|---|---:|---:|---:|---:|
|All 2,591 fields / 22,824 runners|1.798224|1.795672|.08995795|.08987524|
|Established handicaps: 993 fields / 8,385 runners|1.905518|1.905004|.09756486|.09760743|

**034 does not advance.** Its target Brier worsens, and the overall date-cluster
95% LL-delta interval is [-.005913,+.000824]. The target interval also crosses
zero. Overall top-pick accuracy falls 30.77%→30.47%, target 26.44%→25.70%.
The modest overall probability-score gains remain recorded; they do not meet
the rule fixed before fitting. No further feature revision, combined proposal,
place fit, current-data fit or installation follows. The native check took
1,824.4 seconds and selected 3,712 win trees. Current-price removal on 100 fields
changes probabilities by exactly zero. Saved baseline metrics reproduce the
033 historical 154-input candidate exactly.

Total new training across034–035: one corrected154-input small baseline,
one small candidate per family, and three native win components for034 only.
The small baseline is now reusable. There was no quarterly or parameter sweep.

Evidence directories: `data/audit/market_transfer_034/` and
`data/audit/market_finish_surprise_035/`. Their source snapshots, preparations,
tests, decisions and screen outputs preserve the exact definitions. Normal
production033 weights remain intact. Subsequent parser repair037 requires a
new source-pinned forward study with these same weights; see
[the source-recovery report](NONFINISH_SOURCE_037.md).
