# Negative Results Ledger

**Read this before proposing a new feature, data source or edge.**

Every entry below is an idea that was built, measured and *refuted*. They are
recorded because the expensive failure mode in this project is not a bug — it
is re-testing something that was already disproved, or worse, re-discovering a
"+15% edge" that is really a stale cache. Several entries here cost weeks.

Dates are when the result was established. "Dead" means it was measured and
did not survive; it does not mean the code was deleted — most of the data
collection is still in place and worth keeping.

---

## 0. How an edge claim is judged here

Bitter experience produced this bar. Apply it to every candidate:

1. **Orthogonality first.** Correlate the candidate with the *market residual*
   (`won − market_implied_prob`), not with the outcome. If the correlation is
   ~0, the market already prices it — stop, whatever the raw predictiveness.
2. **CLV at BSP is the metric; ROI is secondary.** Betfair Starting Price is
   sharp. Beating it on closing-line value is the only durable claim. Read it
   from `median_clv` / `geometric_mean_clv` / `pct_beat_close`
   (`src.clv.summarise_clv`), **never the mean ratio alone** — the mean is
   upward-biased at long prices and has reversed a verdict here (§2).
3. **The ROI@BSP-vs-CLV consistency check is the leak tripwire.** A strategy
   with high ROI and CLV ≈ 1.0 is almost always leaking, not winning. This is
   what caught the `trainer_freq` leak (§3).
4. **Confirm the measuring instrument before the measurement.** Check
   `scripts/project_status.py` first. A stale feature store has produced a
   phantom edge at least once (§1).
5. **Never fit a second stage on in-sample first-stage predictions.** Use
   `wf_eval.oof_predict_expanding`. Getting this wrong buried a lead for
   three months (§2).
6. **Cluster confidence intervals by race.** Bets in one race are not
   independent; `wf_eval.bootstrap_roi_ci` resamples races. An i.i.d.
   bootstrap over bets is how a result gets called "significant".
7. **Give the method a baseline it must beat by construction.** If a model
   that nests the market scores *worse than market-only on race log loss*, the
   fit is broken (`wf_eval.check_combination_nesting`). Note this is about
   probability scoring, not ROI: losing to "bet everything in the band" is a
   staking comparison and says nothing about the fit.
8. **Do not overturn a result on a partial re-run.** A two-fold re-run of §2
   disagreed with the recorded verdict; eight folds reproduced it. Extend the
   sample before rewriting the conclusion.

---

## 1. Place-market edge — OVERTURNED (2026-06-24)

A walk-forward place-market strategy showed roughly **+15% ROI**. It was a
**stale feature-store artifact**: the store on disk had been built by older FE
code than the evaluation assumed. Rebuilt correctly, the same walk-forward
gives **+1.5% with a confidence interval spanning zero**, and it is negative on
jumps. There is no place-market edge.

**Lasting consequence:** this is why `fe_signature` exists, why
`tests/test_fe_signature_coverage.py` forces every FE dependency to be
registered, and why the serving app refuses to price a card when the promoted
model's feature generation does not match the store.

## 2. Market combination (Benter two-stage) — DEAD, re-confirmed (2026-10-12)

Combining the fundamental model with the market, settled at BSP with 2%
commission, expanding quarterly walk-forward. **Measured twice, with two
different harnesses, and the answer held both times.**

| | original (2026-06-29) | corrected harness (2026-10-12) |
|---|---|---|
| Combination ROI @ BSP | **−2.64%** CI[−4.45, −0.81] | **−3.55%** CI[−5.42, −1.55] |
| Model alone | −3.65% | −4.49% |
| Bet all (in band) | −1.83% | −2.68% |
| Mean CLV (biased; see below) | 1.0047 | 1.0076 |
| Folds positive | 2/13 | 1/8 |

The 2026-10-12 figures come from a rewritten harness: blend fitted
**out-of-fold**, the production per-race conditional logit
(`model.fit_market_anchor`) instead of a binary logistic regression, purged
folds, and race-clustered confidence intervals. The earlier harness had all
three of those wrong (details below) — and **fixing them did not change the
conclusion**. It is still significantly negative and still loses to simply
backing everything in the band.

### What the two runs agree on

- **The anchor helps the model**: the combination beats the model alone in both
  runs (−3.65 → −2.64; −4.49 → −3.55).
- **The selection rule is the weak link**: the combination still loses to
  bet-all, so picking runners where `combo_prob > mkt_imp` selects bets that
  are *worse than average* inside the band. The loss is in the selection, not
  in the blend.
- **α is positive and stable** — +0.037 to +0.057, positive in **8/8** folds
  (mean +0.046). The fundamental model genuinely carries information the price
  lacks, consistent with the production golden hold-out. It is simply far too
  small to overcome commission: β sits at ~0.97, i.e. the market is ~20× more
  informative.
- **CLV does not clear 1 on an unbiased estimator.** The mean ratio reads
  1.0076, but the median is 0.9865 and the geometric mean lower still; the
  picks edge out bet-all on CLV while still drifting against the close. See
  the measurement note below.

**Conclusion: no bankable edge in the market combination.** Do not re-run this
without a *new* selection rule or a materially better fundamental model; the
combination layer itself has now been measured twice and is not the lever.

### Harness defects found and fixed (worth knowing, did not change the verdict)

1. **The blend was fitted on in-sample model predictions** — the win model was
   trained on the fold's training rows, then the blend fitted on that same
   model's predictions over those same rows. In-sample probabilities are
   over-sharp, so the blend over-weights the model.
   `tests/test_benter_harness.py` demonstrates this on synthetic data: the
   in-sample fit yields a larger `alpha` and a *worse* out-of-sample race log
   loss than an out-of-fold fit on identical data. Fixed with
   `wf_eval.oof_predict_expanding`.
2. **A binary logistic regression stood in for the conditional logit.** Fixed
   by calling the production estimator, so research and production now measure
   the same object (hard rule #2).
3. **No purge gap**, while production purges `config.PURGE_DAYS`. Fixed.
4. The model-alone and bet-all baselines were only accumulated on folds where
   the combination happened to bet, and the win model was given the *place*
   model's hyperparameters. Both fixed.

### Where the loss actually comes from (2026-10-12)

The combination beats the model alone yet loses to backing the whole band, so
the damage is in the **selection rule**. `scripts/audit_selection_rule.py`
localised it on 196,664 runners across the eight folds:

| price band | share of stakes | ROI selected | ROI whole band | gap | median CLV |
|---|---|---|---|---|---|
| 1.5-3 | 4.6% | -1.12% | -2.55% | **+1.43** | 1.0000 |
| 3-6 | 18.4% | -2.15% | -3.03% | **+0.88** | 0.9962 |
| 6-12 | 30.4% | -0.79% | +0.42% | **-1.21** | 0.9804 |
| 12-25 | 28.7% | -6.41% | -2.87% | **-3.54** | 0.9843 |
| 25-50 | 17.9% | -3.49% | -4.99% | **+1.49** | 0.9691 |

**The longshot hypothesis was wrong.** The obvious explanation — that a value
filter is a longshot filter in disguise, since that is where a model least
agrees with the price — does not hold. The selection rate barely rises with
price (45% to 55%), and at the *longest* prices the filter is one of its best.
The loss is concentrated in the **middle of the book**, 6-25, which carries
59% of stakes.

Nor is it a calibration failure. Predicted win rate is within 0.0023 of actual
in every band. What separates the losing bands is that the **market** is
closer to the outcome than the model is; the filter only adds value where the
model is the better of the two.

Measured against a **matched** baseline (backing everything in the same
universe the rule bets in, not the whole band), the only sub-rule that beats
no-filter is `odds < 6`: **+0.98 points**, n=23,584 — still -1.95% absolute,
so useful as a diagnosis rather than a strategy. Note the trap: `odds < 12`
scores -1.29% against a whole-band baseline of -2.68% and looks like a
+1.39-point win, but backing *everything* at odds < 12 returns -1.07%, so
inside its own universe the filter is 0.22 points worse. Essentially all of
the apparent gain was avoiding long prices.

One structural finding for any future selection rule: an **absolute edge
threshold is vacuous here.** Because the fitted beta sits at ~0.97, the
anchored probability stays very close to the market, so `combo_prob/mkt_imp >
1.10` matches only **16 bets out of 104,146**. A rule has to work on rank or
relative edge within a race, not on a ratio threshold.

### Measurement note: never read CLV off the mean alone (2026-10-12)

CLV is this project's primary metric, and the research harnesses were
reporting it as the **mean of `entry_odds / close_odds`**. That estimator is
upward-biased, badly so at long prices, because the upside of a price ratio is
unbounded and the downside is not: a 25.0 shot closing at 12.5 scores 2.0,
closing at 50.0 scores 0.5, and averaging those gives 1.25 for two moves that
are symmetric in probability terms.

It changes verdicts. The same selections scored:

| price band | mean CLV | median CLV | geometric mean |
|---|---|---|---|
| 1.5-6 | 1.0006 | 0.9982 | 0.9873 |
| 6-12 | 0.9969 | 0.9804 | 0.9719 |
| 12-25 | **1.0147** | 0.9843 | 0.9739 |
| 25-50 | **1.0229** | 0.9691 | 0.9594 |

The mean says these picks beat the close; the median and geometric mean say
they drift. On the honest measures the combination's picks have **median CLV
0.9865 against 0.9786 for bet-all** — better than backing everything, but
still below 1, so there is no closing-line value to bank.

`src/clv.py` always reported median, geometric mean and %-beat-close
correctly; it was the harnesses that quoted only the mean.
`src/wf_eval.clv_summary` now routes every harness through the same function,
and any CLV figure quoted in this ledger from before this date that came from
a harness mean should be treated as optimistic.

### A caution about how this entry was nearly mis-corrected

A first re-run limited to **two folds** gave −1.59% with a CI spanning zero and
the combination beating bet-all, and that was briefly written up here as
"re-opened". Eight folds reversed it. Two folds of racing data is not enough to
overturn anything — and a partial re-run that happens to agree with the
hypothesis is exactly when to extend the sample rather than publish.

A related error: the combination losing to **bet-all** was read as a nesting
violation (a market-anchored model nests the market at α=0, β=1, so it cannot
be much worse than the market). That reasoning does not apply — "bet every
runner in the band" is a staking rule, not the market probability model, so
losing to it is not evidence of a broken fit. The genuine nesting check is
`wf_eval.check_combination_nesting`, which compares race log loss against
market-only; it **passes** in the corrected runs.

## 2b. ATR sectionals — 20 of 21 signals already priced (2026-10-12)

The At The Races / TPD sectional scrape is the most expensive dataset this
project has acquired: a Fastly TLS/JS challenge to clear, `curl_cffi` or a
system `curl` binary, ~2.5s per request, GB-only, a resumable multi-week
backfill. It produced **216 `atr_*` columns** in the store — velocity curves,
finishing-speed percentages, closing speed, turn of foot, energy efficiency,
stride length and frequency, sectional-adjusted upgrades, and nine
in-running comment tags.

**None of them were ever in the allowlist, and none had ever been triaged.**
That work was in flight when the session that built it ended.

Triaged with `scripts/triage_features.py` on 184,035 rows that carry the data
(coverage 2024-01-04 to 2026-06-10, 31% of priced rows), scoring each column's
partial correlation with the market residual — both demeaned within race, with
the price partialled out on probability, log-odds and quadratic terms:

| signal (strongest column of family) | coverage | partial corr | t | verdict |
|---|---|---|---|---|
| `atr_closing_in_hand` | 100% | **+0.0105** | +4.5 | shortlist |
| `atr_turn_of_foot` | 100% | +0.0073 | +3.1 | priced |
| `atr_efficiency_score` | 100% | +0.0070 | +3.0 | priced |
| `atr_stride_x_gallop` | 80% | −0.0060 | −2.3 | priced |
| `atr_max_speed` | 100% | +0.0057 | +2.4 | priced |
| `atr_closing_speed` | 100% | −0.0056 | −2.4 | priced |
| `atr_energy_eff` | 100% | −0.0054 | −2.3 | priced |
| `atr_finishing_speed_pct` | 100% | +0.0051 | +2.2 | priced |
| `atr_peak_stride_length` | 80% | +0.0035 | +1.3 | priced |
| `atr_speed_score` | 100% | +0.0032 | +1.4 | priced |
| `atr_cmt_trouble` | 100% | **−0.0014** | −0.6 | priced |

(11 of 21 shown; the remaining comment-tag families all score below ±0.005.)

**One signal clears the bar, marginally.** `atr_closing_in_hand` — a horse
finishing fast while apparently not fully extended — at +0.0105 with t=+4.5.
For scale, the dead sale-price feature scored 0.0000 and the best *rejected*
sectional signal here is 0.0073, so this is barely separated from the pack. A
partial correlation of 0.01 explains about 0.01% of residual variance.

**The one genuinely interesting structural finding:** the *derived* "had more
to give" measure is the strongest signal in the family, while the *explicit*
in-running trouble comment (`atr_cmt_trouble`) is the **weakest of all 21** at
−0.0014. The prediction that a data-native trouble signal would beat parsed
text was correct in direction — and both are too small to matter. That also
closes the idea, recorded in §7, that sectionals would succeed where stewards
reports failed: they fail the same way.

**Status:** the shortlisted signal still needs the CLV test at BSP under the
locked walk-forward; correlation is necessary, not sufficient. Expect it to
fail — the market-combination work established that β sits at ~0.97, so a
signal this small cannot clear commission.

**Caveat:** this ran against the store built 2026-07-03, whose `fe_signature`
differs from the current FE code. The result is uniform enough (20 of 21 under
0.0075) that code drift is very unlikely to reverse it, but re-confirm after
the store rebuild before treating it as final.

**This fires decision gate 3 in [ACTION_PLAN.md](ACTION_PLAN.md):** the
sectionals were the last unexploited asset already in hand. With them priced,
market microstructure (traded volume, ladder depth) is the only live
hypothesis left that attacks the informational gap rather than re-processing
public form.

## 2c. Why the model loses: its disagreements are uninformative (2026-10-12)

A head-to-head of the market-blind fundamental model against the market on
**25,388 walk-forward races**, each with exactly one recorded winner:

| | top-1 accuracy |
|---|---|
| model's top pick | 0.2921 |
| market's top pick | 0.3398 |
| they pick the same horse in | 55.1% of races |
| when they agree, that horse wins | 0.3878 |

**The entire deficit is in the 44.9% of races where they disagree**, and there
the market wins the head-to-head 0.2809 to 0.1746.

### The decisive test

In those disagreement races, compare the model's pick to what the market
already charged for it:

| | actual win rate | market-implied | edge |
|---|---|---|---|
| the horse the model **promoted** | 0.1746 | 0.1788 | **−0.43 pp** |
| the horse the model **demoted** | 0.2809 | 0.2814 | **−0.05 pp** |

**The model's disagreement carries no information.** Its preferred horse wins
exactly as often as the price says, and the horse it rejects wins exactly as
often as the price says. This holds in every disagreement-magnitude bucket
(−0.80 to +0.54 pp from 1.2× through 3.5×+), every field-size band, every
exposure band and every race type — nothing exceeds ±1.9 pp, and nothing
survives a race-clustered interval.

So the model is **not mis-ranking correctable things**. It is re-ordering
horses whose probabilities the market has already got right.

### The mechanism

Standardised mean difference between the model's pick and the market's pick,
across all 11,389 disagreement races:

| feature | model's pick minus market's pick (SD) |
|---|---|
| `horse_glicko_rank` | **−0.65** (lower = better) |
| `horse_glicko_vs_field` | **+0.56** |
| `horse_elo_momentum_vs_best_in_race` | **+0.54** |
| `horse_elo_momentum_vs_field` | +0.51 |
| `horse_elo_momentum` | +0.43 |
| `combined_elo_vs_field_z_in_race` | +0.37 |
| `horse_avg_pos_3` | −0.36 (better recent finishes) |

**When the model overrules the market it is keying on rating momentum and
relative rating strength** — recent trajectory and "rates better than this
field". These are the most public, most cheaply computed signals in racing,
which thousands of market participants also calculate.

And that basis is worse than useless. Conditioning on the quartile where each
driver most favoured the model's pick:

| driver | model pick's edge vs its price |
|---|---|
| `horse_elo_momentum` strongest | **−1.11 pp** |
| `horse_glicko_vs_field` strongest | **−1.16 pp** |
| `horse_glicko_rank` strongest | −0.38 pp |

### What is NOT wrong

- **Not calibration.** Model probability over actual win rate is 0.98–1.04× in
  *every* prior-starts bucket (0, 1, 2-3, 4-7, 8-19, 20+) and every momentum
  quintile — as tight as the market's own 0.98–1.02×.
- **Not a broken subgroup.** No exposure band, field size or race type shows
  an edge distinguishable from zero.
- **Not a bias toward unexposed horses.** The model's top pick has a median 9
  prior starts and is a debutant 3.0% of the time; the *market's* pick is a
  debutant 5.9% of the time. If anything the model under-backs newcomers.

### A fix was tested, and it failed

The model's most *confident* errors do cluster on lightly-raced horses with
spectacular ratings — a horse with one good run gets a huge
`glicko_vs_field` and momentum with no record to regress it toward the mean.
Bets on 1-start horses returned −11.57% against a −4.83% matched baseline.

But declining those bets does nothing:

| rule | n | ROI @ BSP | vs matched baseline |
|---|---|---|---|
| current (`combo_prob > mkt_imp`) | 104,146 | −3.55% | −0.87 |
| drop thin-form, hot-momentum bets | 96,837 | −3.31% | **−0.63** |
| the dropped bets alone | 7,309 | −6.70% | −4.02 |

0.24 points of improvement, and none of the segment effects survive a
race-clustered interval (1-start bets: CI[−20.10, −3.45] against a −4.83%
baseline that sits inside it). The tail is real-looking and too small and too
noisy to bank.

### Methodological note, recorded because it nearly produced a wrong answer

Three example races were first selected as "the model's most confident errors"
and all three showed the same vivid pattern: the model backing a 1-start horse
on a huge rating while a well-fancied debutant won. It was a **cherry-pick** —
that selection filter surfaces precisely those cases, and the aggregate
contradicts the story (see "What is NOT wrong"). Hand-picked extremes in this
project have repeatedly suggested structure that the population does not have.
Always put the aggregate next to the anecdote.

### Can it be resolved?

Not by calibration, not by excluding a segment, and not by better ranking —
no re-ordering of the same information can beat a price that already contains
it. The one fix this diagnosis points at directly is **§4c of the action
plan**: the model has no way to know *which of its features the market has
already priced*, so it spends its capacity, and its disagreements, on momentum.
Training with the market log-odds as a fixed offset (`init_score`) would force
it to learn only the residual. That is the untried item, and this is the
strongest argument yet for trying it.

Beyond that the ceiling is informational, not architectural (§1c).

## 2d. Debutants: the model can rank them, but only over a 14x range (2026-10-12)

Follow-up to §2c, asking whether the "well-fancied debutant" the market sees
is measurable from public data.

**The model is not blind to debutants.** On 18,130 debutant runners it reaches
**AUC 0.773** against the winner, and in races with two or more debutants it
picks the winning one 48.3% of the time against 19.6% for random.

**But debutants are its worst band relative to the market**, and the mechanism
is compression, not bias:

| debutant probabilities | sd | p10 | p90 | p90/p10 |
|---|---|---|---|---|
| model | 0.082 | 0.0132 | 0.1834 | **13.9x** |
| market | 0.108 | 0.0039 | 0.2102 | **53.6x** |

AUC by prior starts, model minus market: **0 starts −0.052** (the largest gap),
1 start −0.030, 2-3 −0.029, 4-7 −0.033, 8-19 −0.035, 20+ −0.043.

The market spreads debutants across a ~54x range of probability; the model
compresses them toward the 8% base rate. It is perfectly calibrated *on
average* over debutants (1.01x) while being unable to separate the 1.4 shot
from the 200/1 shot.

**Note it under-spreads at both ends.** Its p10 is 0.0132 against the market's
0.0039, so it also cannot identify the no-hopers — and that half of the problem
is a *public* one (weak yard, weak pedigree, weak jockey booking). It also
matters more than it looks: in a per-race softmax, pushing genuine 200/1 shots
down redistributes probability to the real contenders, so improving the bottom
end sharpens the top end for free.

### What public proxies exist, and what is missing

| signal | in store | allowlisted |
|---|---|---|
| `sire_*` | 14 | 5 |
| `damsire_*` | 14 | 4 |
| `dam_*` (the dam's own produce record) | 2 | **0** |
| sibling / progeny / produce form | **0** | **0** |
| first-time-out / debut features | 11 | **0** |
| `trainer_first_string_*` | 7 | **0** |
| `first_time_headgear` | 1 | **0** |

Two gaps worth acting on:

1. **Sibling form does not exist at all.** "Full brother to a Group winner" is
   the most-used public debutant signal in racing and the market leans on it
   heavily. `dam` is already in the results table, so it is computable: for a
   debutant, the career record of other horses sharing its dam — wins, best
   official rating, best class — strictly from races before today. Mind hard
   rule 1: this is a history feature and must exclude the current race, and
   siblings running on the same day must be date-strict.
2. **Everything debut-specific that already exists is invisible to the model** —
   the dam's produce record, all 11 debut features, all 7
   `trainer_first_string_*` columns. Free to test with
   `scripts/triage_features.py`.

**Expectation-setting:** sale price is the canonical public proxy for a fancied
debutant and it tested **dead on debutants specifically** (residual correlation
0.005, §5). That is real evidence against this whole family. Sibling form is
different in kind — racecourse evidence from relatives rather than a price —
but do not expect much.

## 2e. Public data audit: what is collected and never used (2026-10-12)

The store holds **1,178 columns**; the allowlist is **151**. Whole families sit
at zero, and the allowlist is three-quarters trainer/horse/jockey history.

**Collected and never shown to the model:**

| family | in store | allowlisted |
|---|---|---|
| sectionals (`atr_*`) | 216 | 0 |
| weather | 10 | 0 |
| going detail | 11 | 0 |
| headgear / first-time equipment | 4 | 0 |
| debut features | 11 | 0 |
| trainer-first-string | 7 | 0 |
| course / track detail | 22 | 3 |
| draw | 11 | 2 |

The important nuance: **the allowlist was selected for raw win prediction, not
for orthogonality to the market.** A family can be legitimately unimportant for
predicting winners and still carry residual signal, so "not allowlisted" is not
the same as "tested and dead". `scripts/triage_features.py --prefix <family>_`
settles each one in minutes.

**Not collected, though published:**

- **Stalls position — entirely absent.** There are 11 draw columns but no
  stalls side (high / low / centre) and no rail movement. **Draw bias is close
  to meaningless without it:** a low draw is decisive at one configuration and
  irrelevant at another, and rail movements change the effective trip. This is
  in every published going report and is the largest course-data omission.
- **GoingStick numeric reading — absent.** All 11 `going_*` columns are
  one-hots of the going *word*. The penetrometer value is published per GB
  meeting and is a finer-grained measure than the category.
- **Exact distance in yards** — only `distance_furlongs` is stored. Races are
  run over e.g. 1m 2f 7y, which matters for stamina at the margin.
- **Course configuration** — inner/outer, round/straight not captured.

Given §2b (sectionals priced) and the saturation finding, most of the
already-collected families will probably also be priced. The stalls-position
gap is different: it is a genuine collection omission that makes an existing
allowlisted feature (`draw`) weaker than it needs to be.

## 2f. Label audit: labels are clean, bar amended results (2026-10-13)

`plan.txt` Stage 1 asked for this and it was never done. Every accuracy figure
and every refutation in this ledger rests on `won` and `finish_position` being
right, so an unmeasured error rate means every reported gap carries an unknown
offset. Run with `scripts/audit_labels.py` over 749,296 runners / 79,121 races
(2020-09-15 to 2026-10-11), cross-checked against Betfair's own settled result.

**Internal consistency is perfect:**

| check | result |
|---|---|
| races with no winner stored | **0** |
| races with >1 winner | 128 (0.162%) — dead heats, legitimate |
| `won` disagreeing with `finish_position == 1` | **0** |
| races with fewer rows than the declared field | **0** |
| non-finishers (`fp <= 0`) | 6.79% of runners (falls, pulled up, unseated) |

**Against Betfair's settled result:** 586,970 runners matched (78.3%), **252
disagreements = 0.0429%**, near-symmetric (130 where we say won and Betfair
says lost, 122 the other way). Place labels against the exchange's settled
place market: 123 of 677,988 = **0.0181%**.

So the labels are not a source of error worth worrying about, and the
model-versus-market gap in §2c carries no meaningful label offset.

### The residual 0.04% is amended results, not scrape noise

The obvious explanation — dead heats — is wrong: only 9 of the 252 fall in a
race where our DB records multiple winners. The rest arrive in **pairs**, and
always the same shape:

```
2021-02-17 Dundalk    Barretstown      fp=2  ours=0  betfair=1
2021-02-17 Dundalk    Star Harbour     fp=1  ours=1  betfair=0
2021-04-24 Sandown    Potterman        fp=2  ours=0  betfair=1
2021-04-24 Sandown    Enrilo           fp=1  ours=1  betfair=0
```

Our DB stores the **first past the post**; Betfair settles on the **official
amended result** after a disqualification or placing reversal. ~243 runners is
~121 affected races, 0.15% of the history — consistent with the real-world rate
of amended results in GB/IRE, and spread evenly across years (36/52/51/30/64/10
for 2021-2026).

**Why it is worth knowing despite being tiny.** For *training* labels 0.15% of
races is immaterial. For *settlement* it is not neutral: the backtesters settle
on our `won`, so a bet on a horse later disqualified is credited as a win the
exchange never paid. The direction is **favourable to us**, which is the
direction that matters. The magnitude is well under 0.1 pp of ROI on a 104k-bet
run, so nothing in this ledger changes — but the cheap fix is to prefer
Betfair's settled `won` for settlement wherever it is matched, and that would
remove the bias entirely.

### A trap in `attach_bsp` worth knowing

`attach_bsp` assigns `out[c] = merged[c]`, so calling it with `cols=("won",)`
**silently overwrites the caller's own label** with Betfair's. An audit written
that way compares the exchange against itself and reports flawless agreement
forever. This was nearly shipped; `tests/test_audit_labels.py` now constructs a
deliberate one-race disagreement so the test fails if the label is ever
clobbered again.

## 2g. The never-allowlisted families: all priced (2026-10-13)

Acting on 2e. Every family that sits in the store and not in the allowlist,
triaged with `scripts/triage_features.py`. **43 signals across 12 families, and
not one clears the bar** (|partial| >= 0.01 with q <= 0.05):

| family | signals | strongest abs(partial) |
|---|---|---|
| `weather_*` | 7 | **0.0000** |
| `going_*` | 5 | 0.0019 |
| `draw*` | 7 | 0.0023 |
| `course_*` | 7 | 0.0028 |
| `trainer_first_string_*` | 7 | 0.0034 |
| `first_time_*` | 4 | 0.0035 |
| `dam_*` | 2 | 0.0011 |
| headgear / first-time equipment | 4 | 0.0049 |

So "in the store but not allowlisted" turned out, in every case, to mean the
same thing as "tested and dead". The allowlist was not leaving signal behind.

### Why the weather features are *exactly* 0.0000 - a structural lesson

All seven weather signals score 0.0000, not merely small. So do
`course_favours_leaders` and `course_quirk_score`. The reason is not that
weather is uninformative: **those features are race-level constants.** Every
runner in a race shares the same temperature, wind and course quirk, so they
have no within-race variance, and both this triage and any
per-race-normalised model cancel them exactly.

A race-level feature cannot differentiate runners by construction. It can only
pay through a **runner-specific interaction** - this horse's record in *these*
conditions. The two going features built that way, `going_wr_x_runs` (+0.0019)
and `going_specialist_edge` (+0.0011), are the only non-zero members of their
family, which confirms the mechanism while being far too small to matter.
`weather_wind_x_dist` scores 0.0000 because wind and distance are *both*
race-level, so the interaction is too.

**Consequence:** collecting the GoingStick number would add another race-level
column and cannot help on its own. Only worth it as an input to interactions
with each horse's own going record - and that family scores 0.0019, so expect
nothing.

## 2h. Sibling (dam-produce) form: priced overall; the debutant lead failed on the hold-out (2026-10-13)

Built because 2d found the model spreads debutant probabilities over a 13.9x
range against the market's 53.6x, and sibling form - "full brother to a Group
winner" - is the most-used public read on an unraced horse and **did not exist
in the store at all**.

`src/sibling_features.py` computes it from the `horse_pedigree` dam, strictly
self-excluding (a horse is not its own sibling) and date-strict (siblings
running the same day are invisible to each other, since their results are not
known pre-off). 64.7% of runners carry sibling history, averaging 17.5 prior
sibling runs.

**Overall: priced.** On 396,898 rows with sibling history, every signal is
under abs(0.005):

| signal | partial | t |
|---|---|---|
| `sibling_any_winner` | +0.0049 | +3.1 |
| `sibling_win_rate_shrunk` | +0.0035 | +2.2 |
| `sibling_best_or_vs_field` | +0.0031 | +2.0 |
| `sibling_runs` | +0.0013 | +0.8 |

### The one result that clears the bar - treat it with suspicion

Restricted to debutants and one-start horses (n=3,746):

| signal | partial | t |
|---|---|---|
| `sibling_best_or_vs_field` | **-0.0641** | -3.9 |
| `sibling_any_winner` | -0.0346 | -2.1 |
| `sibling_win_rate_shrunk` | -0.0160 | -1.0 |

Note the **sign**. Negative means that among unraced horses a *better* sibling
record goes with a *worse* result than the price implies - i.e. **the market
over-rates the well-bred newcomer**, and the implied trade is to fade them
rather than back them. That is the opposite of what the feature was built for.

**Why to be sceptical anyway:**

- It is a **subgroup examined after the headline test came back null**, which
  is the exact pattern that generates false discoveries and has already misled
  this project twice in one session (the cherry-picked examples in 2c, and the
  two-fold re-run in 2). With roughly 6 exposure bands x 7 signals available to
  look at, the multiple-testing bar is about abs(t) > 3.2, which -3.9 clears
  only narrowly.
- n = 3,746, so the standard error on the correlation is about 0.016.
- Correlation with the residual is **necessary, not sufficient** (see 0). Every
  refuted entry in this ledger cleared some correlation test first.

**The only thing that would settle it:** CLV at BSP under the locked
walk-forward, restricted to unraced runners, with the ROI-vs-CLV tripwire, on
the rebuilt store - this screen ran against the DB directly. Until then it is
a lead, not a finding.

### Hold-out verdict (2026-10-13): does not confirm

Tested once on the golden hold-out (races from 2026-06-11, never seen by any
residual analysis), with the population, statistic and bar written into
`docs/SESSION_LOG.md` before the script existed
(`scripts/holdout_sibling_test.py`): runners with at most one prior UK/IRE
start and sibling history, in races with a complete BSP market; partial
correlation of `sibling_best_or_vs_field` with won - p_BSP, within-race
demeaned, price basis partialled out, race-clustered bootstrap SE.

| | n runners (races) | partial corr | SE | t | one-sided p |
|---|---|---|---|---|---|
| discovery (vs T-5 price) | 3,746 | -0.0641 | ~0.016 | -3.9 | - |
| **golden hold-out (vs BSP)** | **3,711 (848)** | **-0.0160** | 0.0179 | -0.89 | **0.19** |

**Fails** (bar: p < 0.01). The sign agrees, but the effect is a quarter of
the discovery's and its 95% interval, [-0.051, +0.019], excludes the
discovered -0.064: the winner's curse of a subgroup found after a null
headline, as this entry predicted. **The sibling lead is closed.** One
discrepancy recorded rather than explained: the same population definition
yields 61,558 runners in the development period, against the 3,746 the
discovery reported, so the discovery applied restrictions it did not
document.

### A bug the tests caught, worth recording

The first implementation shrank the sibling win rate toward **the mean of
whatever frame was passed in**. That breaks store-append parity: the same
settled row gets one value in a full build and a different one after an
append, because the population mean moves. The "settled rows do not change
when later races are appended" invariant in
`tests/test_sibling_features.py` caught it. The shrinkage target is now a fixed
constant, and any future `*_shrunk` feature must do the same.

## 3. `trainer_freq` stablemate leak — RESOLVED (2026-06-26)

A **+56.8% ROI@BSP** was traced to `trainer_freq`, built with a `cumcount`
that encodes same-race finish order — the feature knew the result. Fixed in
`150506e`; the store was rebuilt and re-validated: **+56.8% → +0.7%**.

Note the first suspect (`trainer_wins_*`, fixed in `ab70426`) was *not* the
real leaker. Caught by the ROI-vs-CLV tripwire, not by inspection.

## 4. Par-time / going-corrected speed figures — redundant (2026-06-27)

`winning_time` was backfilled to 99.9% coverage and the measured going
validated, then used to derive going-corrected speed figures (`tspf`). This is
the **most discriminative new feature family** built, and it is still
**CLV-flat** — the market already prices it.

**Keep the data** (it is good data, and it makes other speed features
better). Do not add the features to the allowlist.

## 5. Sale price (Tattersalls) — fully priced (2026-06-29)

Built a working Tattersalls sale-price scraper (4DCGI API). The feature is
dead: residual correlation **0.0000** overall, **0.005** on debutants. The
market prices public sale prices completely.

This sharpened the picture in §8: the debutant edge the market holds is
*genuinely private* information (gallops, fitness, intent), not a public price
anyone could scrape. **Do not scrape Goffs for this.**

## 6. Intent signals — all three dead (2026-07-03)

| candidate | verdict |
|---|---|
| Declared wind surgery | fully priced (residual t = −0.33) |
| Trainer yard→track travel | noise |
| Jockey booking | CLV 1.033 is the SP-vs-LTP **move artifact** — the feature is built from odds, so it is not tradeable |

The jockey-booking result is the instructive one: a feature derived from odds
can show apparent CLV purely because it proxies the price move itself. Data
kept; features not allowlisted. Store rebuilt at 1178 columns.

## 7. Stewards reports — redundant with form (2026-06-26)

BHA (token gotcha: `Origin` header + last `Bearer`) and IHRB (`sitemap.xml`)
report scrapers were built and verified. The derived "trouble in running /
unlucky" features proved redundant with ordinary form.

Better replacement, if revisited: derive trouble from **sectionals** instead —
a horse closing fastest at the line but beaten is the complete, full-coverage
version of the same signal, with no text parsing. That is what
`atr_closing_in_hand` measures (see 2b). `src/stewards_features.py` and the two
report scrapers were deleted on 2026-10-13 — recover from git if revisited.

## 8. Why the model loses to the market — characterised (2026-06-27)

The market beats the model specifically on **favourites and debutants**, and
the gap vanishes once the private information is exposed. The **price move**
captures about half the accuracy gap, but:

- it is **not tradeable** (move continuation correlation ≈ 0.01), and
- it destroys orthogonality, since the feature is built from odds.

Gotcha worth remembering: `market_id` is a **meeting** key, not a race key —
normalise within `(date, off_time, track)`.

## 9. Feature-store saturation (2026-06-26)

The store is effectively exhaustive for public form. Adding the **173 unused**
stored features produces **no CLV improvement**. The one genuinely untapped
area identified is **market microstructure** (`ltp_*` columns in the BSP file:
traded volume, ladder depth, timing and shape of the move) — still unexplored.

## 10. Quality-control gates (2026-06-22)

The thin-history gate was built, then quantified as **unhelpful**; it defaults
**OFF**. Related finding: a **form-only model has no edge**. Burn-in is now
absolute rather than relative.

## 11. Course-dynamics features (2026-06-22)

A new course-dynamics FE stage produced 18 candidate features; **4 made the
allowlist** with marginal positive lift. Not a breakthrough, but not dead.

---

## 12. Consolidation: what was deleted and why (2026-10-13)

The repository had accumulated one live pipeline and a long tail of side
experiments, with no way to tell them apart by reading the tree. 32 files were
deleted and two live modules pruned — **≈6,100 lines in all**. **Nothing here was removed because it looked
unused — every item was either refuted with a number in this ledger, or
provably unreachable.** The code is in git history; the measurements are here.

### Refuted strategies and feature families

| Deleted | Evidence |
|---|---|
| `place_market_model`, `paper_trading`, 3 scripts, 2 tests | 1 — the +15% was a stale-store artifact; on correct features +1.5% with a CI spanning zero, going negative. Opt-in and unrun since June (one 290-byte paper-trade log). |
| `stewards_features`, `bha_stewards_scraper`'s report half, `ihrb_daily_enquiries_scraper` | 7 and 2b — "unlucky" features are redundant with form, and the sectional version (`atr_closing_in_hand`) beat the text version while both were tiny. |
| `par_time_features` | 4 — the most discriminative new family measured, and CLV-flat. |
| `tattersalls_sales_scraper`, `build_sale_price_join`, `test_sale_price_signal` | 5 — residual correlation 0.0000; the debutant edge is private information, not public sale price. |
| `backfill_wind_surgery`'s feature consumers, `test_intent_signal` | 6 — wind surgery fully priced, travel is noise, booking's CLV was an SP-vs-LTP artifact. (The backfill script is kept: the DB column stays.) |

### Provably unreachable

- **`rtv_scraper` + `add_rtv_features`** (≈890 lines). ATR replaced Racing TV as
  the sectional source and `add_rtv_features` was no longer in the stage list,
  so the function could not run. Removing it is a **provable no-op on feature
  values**: the four `rtv_acceleration_*` names sat in a list filtered by
  `if col in df.columns`, and the `has_rtv_history` block behind
  `if "rtv_top_speed_avg_3" in df.columns` — and those columns cannot exist when
  the stage that creates them never runs. `src/app_helpers.py` also imported
  eight RTV names and used none of them, three being a **duplicate set of merge-key
  normalisers** (`_normalise_horse_key` and friends) pulled from the retired
  scraper instead of `src/utils.py` — a quiet hard-rule-3 violation that only
  showed up when the scraper was removed.
- **`strategy_calibrator`** (717 lines). Zero importers, and it grid-searched
  betting parameters **on the test set** — the overfitting trap that produced
  two of this project's false edges. Not worth repairing.
- **`config.MODEL_FILE` / `SCALER_FILE` / `FEATURE_COLUMNS_FILE`** and seven
  March-2026 `models/` artifacts (49 MB) from the removed LambdaRank path:
  referenced nowhere, and actively misleading about which model serves.
- **17 one-off scripts** whose questions are answered here: `train_600d*` (two
  pointed at a run snapshot that no longer exists), `ab_market_add`,
  `ab_race_model`, `probe_stablemate_leak` (leak found and fixed in 150506e),
  `repair_feature_leakage_snapshot` (superseded by the rebuild), `audit_edge`,
  `gpu_tabular_benchmark` (needs a bundle that was never kept),
  `wf_each_way` (hardcoded snapshot **and** a hand-rolled fold loop, which hard
  rule 6 exists to forbid), `cleanup_non_runners` and `retrain_clean_store`
  (completed repairs).

### Dead code *inside* live modules

Module-level deletion missed the larger share. A reference scan over every
top-level function in `src/`, re-run after each pass so a helper used only by
something already deleted is caught on the next one, found:

- **`src/app_helpers.py` was 63% dead** — 882 of 1,404 lines. 17 orphaned
  chart/frame builders, then 14 more that only those called: the Insights and
  Diagnostics panels of the retired monolithic `app.py`. The current panels
  never referenced them, which is why nothing failed and nothing noticed.
- **`src/live_prediction.py` carried two caches, one superseded.** The
  `live_feature_cache_*` family (5 functions, 48 lines, plus two directories it
  created on import) was replaced by the `lookahead_cache_*` family that
  `app_helpers` actually imports. Two cache systems in the serving module, one
  dead — the single most confusing thing in the live path.

Six single orphaned functions were **kept** on purpose; they are listed in
`PROJECT_OUTLINE.md` §4.3. A reference count of zero is a prompt to look, not
a licence to delete: a settlement formula, an append-machinery helper and a
market-anchor teacher all have obvious future callers, and stripping a
module's API surface on a grep is how a codebase loses optionality.

### `_FE_UNWIRED_SOURCES` is gone, and that is the point

`featured_dataset` carried a documented exemption list so that editing
`par_time_features` or `stewards_features` would not force a multi-GB rebuild.
That mechanism existed **only** to work around dead code. Deleting the code
removed the need for it, along with its test. The real guard —
`test_every_fe_dependency_is_hashed_or_explicitly_exempt` — is untouched.

The general lesson: a workaround that protects dead code is a signal to delete
the code, not to keep the workaround.

### Cost of doing this mid-build

The deletions touched `feature_engineer.py` and `featured_dataset.py` while a
full store rebuild was running, so the finished store (749,296 rows, 1,178
features, through 2026-10-11) carries `fe_signature` `724ce1afe9ad7611` against
a current `3e3b9bbb5d79bacc`. The guard will correctly force a rebuild. It was
**not** hand-re-stamped, even though the reasoning above says the values are
unaffected: re-stamping a store on an argument rather than a measurement is
exactly the move that produced the phantom +15% place edge. Spend the rebuild.

## 13. Phase A audit: what the pipeline was silently getting wrong (2026-10-13)

A stage-by-stage audit (NEXT_AGENT_BRIEF §3). Nothing below changed a
refutation in this ledger; several items change what the data and the
instrument can be trusted for.

### The results database was missing races, systematically

The old date-level sync kept **only the first 50 listed races per day**
(`race_links[:50]`) and skipped any race page whose fetch raised, after which
the date counted as present and was never revisited. Neither path logged
anything louder than a warning.

| | days with exactly 50 races | days with more than 50 |
|---|---|---|
| 2020-21 (capped sync) | 50 | **0** |
| 2022-25 (an uncapped collection path) | 6-10 a year | 30-38 a year |
| 2026 (capped sync, incl. the June-October refresh) | 25 | 5 |

Histogram of races per day: 53 days at 49, **105 at 50**, 24 at 51. A dry-run
relist of 30 sample dates found 32 missing races; the listing puts UK meetings
first, so the cap cut **Irish** cards (Navan, Cork, Limerick, Sligo, Kilbeggan,
Tramore). The skip path is separate and hits any day: 2021-08-08 was missing a
whole 7-race Haydock card on a 28-race day, and **2022-12-17 (42 races) was
missing entirely**. Sampled 2022-25 dates came back complete.

Why it matters despite being under 1% of races: every horse that ran in a
missing race has a hole in its history (wrong days-since-run, a skipped rating
update, a missing form line), the market saw that run and the model did not,
and the 2026 holes fall on exactly the horses live serving prices.

Repaired by a full relist through the new resumable ledger
(`python -m src.database --audit-history --sync 2219`): **543 races / 6,410
runner rows added** (2020: 44, 2021: 229, 2022: 27, 2023: 27, 2026: 216, of
which 47 are the newly synced 2026-10-12), all at the truncated meetings
(Navan 53, Limerick 43, Killarney 40, Tramore 35, Punchestown 32, ...). Races
per day around 50 is now smooth (49: 41, 50: 36, 51: 42) and 2021 has 46 days
above 50 instead of none. Correction to the dry run: 2022-12-17 was *not*
missing - all six cards were abandoned in the freeze.

3,472 listed races still yield no result: 3,084 in meetings with nothing
stored, 388 isolated inside stored meetings. A random sample of 80 found
**none that ran and was missed**: ABANDONED 54, DORMANT 19, RACEVOID 3,
Arabian 2, HTTP 404 2 (95% upper bound on genuine misses ~3.7% of the set).
The sync now closes such races with their reason (`RaceNotRun`), so its
"pending" count means "failed, unexplained".

**Checked against an independent census.** Betfair's public files list every
GB/IE win market, so `scripts/betfair_db_audit.py` run against them is a
full-history completeness test of the scraper: 79,579 of 79,693 Betfair races
(99.9%) are in the DB. Of the 114 absent, 45 ran on 2020-09-14, the day before
the DB starts; 66 are Arabian races the DB excludes by design; **3 are genuine
omissions** (Chepstow 2023-04-10 13:35, Bellewstown 2024-07-05 18:35, Curragh
2025-06-27 19:30), races Sporting Life never listed. The same audit exposed a
join bug: `src/betfair.py` lacked the 2020 abbreviations `Aint`, `List`, `Nott`,
so 124 Aintree/Listowel/Nottingham races (Sep-Dec 2020, public files only - the
BASIC parquet never used them) could not join. Fixed.

### The flat draw backfill was finished, not pending

`draw_backfill_done` covers 46,791 flat races through 2026-06-13, and the
3,645 later flat races were scraped by the fixed scraper. Verified on the data
rather than the checkpoint: the per-race Spearman correlation between `draw`
and weight carried in flat handicaps is **~0 (|mean| < 0.06) in every month
from 2020-09 to 2026-10**. Saddle-cloth numbers, which the old bug stored,
are assigned by weight in handicaps and would give ~-1. `draw` is a real stall
draw everywhere.

### The feature build was not a pure function of the database

`add_weather_features` called the Open-Meteo API at build time for rows with
no stored weather, and filled whatever failed with constants (14.0 C, 1.5 mm,
...); `add_trainer_travel_features` read an unsigned CSV. Neither input is in
`data_signature`, so two builds of the same database could differ. Both
families were already measured dead (2g: every weather signal 0.0000; 6:
travel is noise). Both stages were removed after a parity check: engineering
106,467 rows / 10,460 races with and without them drops **exactly their 10
columns and leaves the other 1,058 bit-identical**. `weather.py`,
`trainer_locations.py` and `bha_stewards_scraper.py` left `_FE_SOURCE_FILES`.

### `fe_signature` changed with line endings

It hashed raw bytes; the tree holds a mix of LF and CRLF sources and git
autocrlf rewrites them on checkout, stash or worktree add. Converting one file
(`feature_store.py`) moved the signature `3e3b9bbb...` to `e7f6193d...`. That
fails safe but forced ~56-minute rebuilds for no change, and made signatures
incomparable between checkouts. Now normalised; identical in the main tree and
a fresh worktree.

### `promote` let unverifiable runs through

It refused a feature-generation **mismatch** but promoted a run with **no
recorded signature** after a log warning. All 8 saved runs predate the
`trainer_freq` leak fix (2026-06-26) and 6 carry no signature, so one click
would have served a leaky model. Unverifiable runs now need the same explicit
override as a mismatch.

### Plan 4b's class-weighting hypothesis describes code that does not run

ACTION_PLAN 1b led with "the win classifier trains with scale_pos_weight ~
(N-1)". It does not: every call site uses `_train_win_classifier`, which
passes no class weight to LightGBM, and the LightGBM place model is unweighted
too (only the XGBoost paths take `scale_pos_weight`; the focal-loss trainer is
never called). "Drop class weighting" is already the configuration. What is
left of 4b is regularisation and learning rate.

### Seven allowlisted features are race-level

`base_win_rate`, `base_place_rate`, `num_runners`, `race_type_Chase`,
`handicap`, `field_std_or` and `field_avg_elo` never vary within a race;
`weight_per_furlong` (0.7% of its variance within-race) and `or_x_handicap`
(1.6%) nearly never do. In the current binary GBM they are not dead by
construction: `num_runners` sets each runner's base rate before per-race
normalisation, and the rest can act through interactions. Under a per-race
softmax or a market offset (plan 4c) their main effects cancel exactly. The
allowlist holds 150 features, not the 151 documented.

### A free closing-price source, and the first golden hold-out

Betfair publishes daily BSP files for anyone
(`promo.betfair.com/betfairsp/prices`, `scripts/fetch_betfair_sp_public.py`):
per runner, BSP, morning and pre-play volume-weighted prices, price ranges,
and **matched volume** (morning, pre-play, in-play) - volume the BASIC tars do
not carry at all. All 4,440 files for 2020-09-14..2026-10-12 (GB and IE)
downloaded; validated:

| check | result |
|---|---|
| BSP vs the BASIC parquet, 589,123 shared runners | 99.99% within 1% (BASIC rounds to 2 dp) |
| winner flag vs BASIC / vs our DB | 99.99% / 99.94% |
| DB runners matched, by quarter | 95-100% in every quarter, incl. the BASIC gaps 2020Q4 (94.9%), 2021Q3 (97.9%), 2021Q4 (96.9%) |
| golden hold-out (races from 2026-06-11) | 4,812 races, 96.2% with a complete closing and morning market |

Three traps, each silent if missed and each tested: the file is dated the day
after its races; off times are UK local while the DB is UTC (every BST race
would fail to join); two header styles.

Two measured facts any morning-price protocol must respect:

- A runner with no morning trade carries `morningwap = 1.0`, a placeholder
  (2.5% of runners). Requiring a valid price drops any race containing one.
- **Pre-off prices sit well short of BSP on the typical runner**: median
  morning VWAP / BSP = 0.8925 (geometric mean 0.8236); pre-play VWAP / BSP =
  0.88. Most runners are outsiders, whose BSP routinely exceeds the traded
  price. This is why bet-all CLV at T-5 is already 0.9786 (2), and it makes
  ACTION_PLAN 7's "median CLV > 1.0" a much higher bar at morning prices.

**The golden hold-out** (`config.GOLDEN_HOLDOUT_START = 2026-06-11`,
committed in `3800c33` before any Phase B experiment) qualifies because every
CLV and market-residual harness drops rows without a Betfair price, and the
hand-downloaded data stopped on 2026-06-10: no such analysis has ever seen
those races. They have closing and morning prices now, but **no T-5 price**
(`ltp_300s` exists only in the BASIC tars). A T-5 result can therefore only be
confirmed on the hold-out once the operator downloads BASIC data for June to
October 2026; a morning-protocol result can be confirmed now.

### Stalls position and rail movements: collected, not yet tested (plan 5a-bis)

The BHA API behind britishhorseracing.com publishes each GB fixture's going
report: stalls position by distance, rail movements, watering, and (per
fixture, on a second endpoint) GoingStick with occasional inside / centre /
outside splits. `scripts/fetch_bha_going.py` collected all **9,311 fixtures
2020-09..2026-10**; stalls text is present on 96.8% of flat fixtures and the
fixtures join to **100% of the 8,682 GB meetings** in the DB. The API
enforces a long-window quota (HTTP 429 for every request for over an hour once
tripped); at 5 s per request it completed.

Not done, deliberately: turning the text into a draw signal. The stalls field
is free text in 1,253 forms ("7f 36y outside; remainder inside", "straight
course - stands side, round - inside"), and which stall numbers sit nearest
the rail depends on each course's handedness and straight-course layout - days
of careful work for a family whose prior is low (every racecard prints the
stalls position; draw features scored at most 0.0023 in 2g). GoingStick was
not fetched (~9,300 requests, ~13 hours at the tolerated rate) because a
single reading is race-level and cancels (2g). The data is in place for
whoever takes this on; see docs/ROADMAP.md.

### Checked and clean

- Arabian purge held: 0 `(A)` runners, 0 Arab race names, 0 "Extras" tracks.
  Sporting Life files Arabian races under a "TRP Extras" pseudo-course that
  passes the UK/IRE filter; they are now dropped at listing time as well.
- Pedigree: 100% sire/dam coverage for debutants in every quarter 2024Q3-2026Q4.
- `data_signature` excludes exactly 31 derived one-hot columns, each
  reproducible from a categorical that is hashed; no other column matches.
- `attach_bsp` now refuses to replace a column the caller owns (2f's trap)
  unless `overwrite=True`; no current caller overlaps.

## 14. Market microstructure, free version: the close absorbs it all (2026-10-13)

ACTION_PLAN 5a's hypothesis is that *where and when money enters* carries the
private information that beats the model. Betfair's public files add matched
volume (morning, pre-play, in-play), so a first version is testable without
the paid ADVANCED/PRO tiers. `scripts/microstructure_screen.py`, development
period only (574k runners / 64k races, 2020-09-15..2026-06-03), six tests
written down before the run, each either **continuation** (does a signal known
at the decision point predict the move after it?) or **orthogonality to the
close**. Within-race demeaned, price basis partialled, race-clustered
bootstrap SEs (calibrated on null data: t mean 0.01, sd 1.10).

"Excess volume" = log(runner's share of the race's morning matched volume) -
log(its morning market share): money beyond what its price would attract.

| test | partial corr | t | reading |
|---|---|---|---|
| T1 excess volume -> move morning VWAP -> BSP | +0.169 (liquid runners +0.103) | +69 | morning money predicts where the price goes |
| **T2 excess volume -> won - p_BSP** | **+0.0004** | +0.4 | **nothing beyond the close** |
| T3 excess volume -> won - p_morning | +0.015 (liquid +0.021) | +13.5 | real information, ahead of the morning price |
| T4 move morning -> T-5 -> move T-5 -> BSP | -0.033 | -9.2 | reversal: bid-ask bounce (the same noisy LTP enters both moves with opposite signs) |
| T5 excess volume -> move T-5 -> BSP | +0.014 | +8.1 | small continuation after T-5 |
| **T6 move morning -> T-5 -> won - p_BSP** | **+0.0007** | +0.5 | **nothing beyond the close** |

**The closing price contains all of it.** Morning money is informative - it
predicts the result better than the morning price (T3) and predicts the later
move (T1, T5) - but BSP has fully absorbed it by the off (T2, T6 ~ 0). For a
bettor settling at BSP there is nothing here. What remains is *timing*: taking
an earlier price on the horses the money is backing, before the close catches
up. T5 says that effect exists after T-5 but is small. Its economic size is
measured below.

### The timing edge, priced (`scripts/microstructure_clv.py`)

One rule per protocol, fixed before the run: in each race, back the runner
with the highest excess morning volume if its decision price is 1.5-50;
matched baseline = every runner in that band. Development period, 2%
commission, race-clustered CIs.

| protocol | rule | n | median CLV [95% CI] | ROI at taken price | quarters + |
|---|---|---|---|---|---|
| T-5 LTP -> BSP | bet all in band | 411,632 | 0.9809 [0.9801, 0.9816] | -2.29% | 2/21 |
| T-5 LTP -> BSP | **top excess volume** | 55,412 | **0.9953 [0.9932, 0.9978]** | -2.15% [-4.24, -0.12] | 6/21 |
| same, races off >= 14:00 UTC | top excess volume | 41,458 | 0.9954 [0.9929, 0.9985] | -2.13% | 6/21 |
| morning VWAP -> BSP | bet all in band | 498,798 | 0.9210 | -6.11% | 0/24 |
| morning VWAP -> BSP | top excess volume | 64,209 | *1.0187 [1.0157, 1.0217]* | *+1.59% [-0.30, +3.44]* | *15/24* |

**At T-5 the timing edge is real and too small to bank**: +1.4 CLV points over
the matched baseline, CIs well separated, robust to dropping every race near
the morning cutoff - and still below 1.0, with ROI no better than backing the
band. Fails ACTION_PLAN 7.

**The morning row is an artifact, and was labelled one in the script before it
ran.** The signal uses the whole morning's volume, which is only known when
the morning ends, while `morningwap` averages the whole morning's prices -
including the longer ones a steamer traded at before the money arrived. It
credits the bettor with prices that had already gone: the jockey-booking trap
(6) in a new form. **Do not read 1.0187 as a result.** The honest version
strikes the bet at the price available when the morning closes; Betfair does
not publish the cutoff, but the morning share of pre-play volume collapses for
races off after ~11:00 UK (median share 1.24 before 11:00, 0.20 at
11:00-11:30, 0.10 at 11:30-12:00), so `ltp_1100` is being extracted from the
BASIC stream for that test.

### The honest early-price tests: break-even at best

`scripts/extract_betfair_bsp.py` now also records the last trade at or before
11:00 UK (`ltp_1100`, present for 96.4% of runners) and at T-60. Same rule,
same matched baseline, development period, races off at least 30 minutes
after the decision:

| decision price | rule | median CLV [95% CI] | geo CLV | beat close | ROI at taken price | quarters + |
|---|---|---|---|---|---|---|
| 11:00 UK | bet all in band | 0.9524 | 0.9143 | 43.4% | -2.04% | 2/24 |
| 11:00 UK | **top excess volume** | **1.0000 [1.0000, 1.0011]** | 0.9711 | 49.7% | **+0.85% [-1.12, +2.85]** | 13/24 |
| T-60 | top excess volume | 0.9927 [0.9902, 0.9949] | 0.9708 | 47.6% | -1.89% | 8/24 |
| T-5 | top excess volume | 0.9958 [0.9939, 0.9980] | 0.9801 | 47.6% | -2.12% | 8/24 |

**The closest thing to a positive result this project has produced, and it is
not an edge.** Struck at 11:00, following the morning money lifts CLV ~5 points
over the matched band and reaches break-even - and the value is gone by T-60,
the shape information absorption predicts. But it fails ACTION_PLAN 7: the
median's interval *touches* 1.0 rather than excluding it, the geometric mean
is below 1, it beats the close less than half the time, and ROI's interval
spans zero. The median sits on 1.0000 partly because of ties: BASIC rounds BSP
to the two-decimal grid LTPs trade on, and ties concentrate in the short-priced,
heavily traded runners this rule selects - read the geometric mean and the
beat-close rate alongside it. Capacity would be thin anyway: the median
runner trades about GBP 345 in the whole morning.

Confirming even break-even on the golden hold-out needs BASIC data for June to
October 2026 (the hold-out has volume but no 11:00 price) and, at ~4,000
selections, could not resolve an effect of +0.85% (ROI CI roughly +-6 points).

### A data gap found on the way: the canonical BSP parquet was incomplete

`data/processed/betfair_bsp.parquet` was written at 17:09 on 2026-06-21; the
last two BASIC tars arrived at 17:38 the same day and were never extracted.
The re-extraction found **99,917 runners** missing - exactly 2020Q3-Q4 and
2021Q3-Q4, the "gaps" every T-5 analysis in this ledger worked around. It cost
those analyses power, not correctness (a missing early quarter thins the
training data; it cannot bias a later test quarter). The full extraction is
now canonical (the old file is in `data/backups/`); BSP and `ltp_300s` agree on
99.998% of the 646,895 runners the two share.

## 15. The `sex` look-ahead leak, and plan 4c without it (2026-10-13)

### The third phantom edge, caught by the tripwire

Plan 4c - a per-race conditional logit with the market's log-probability as a
fixed offset, so trees can only learn the market's residual
(`scripts/wf_market_offset.py`, T-5 protocol, 13 purged quarterly folds
2023Q2-2026Q2, 40,462 races, pre-registered rules) - first came back with:

| rule | n | ROI @ BSP [95% CI] | median CLV |
|---|---|---|---|
| edge > 0.10 | 13,681 | **+11.62% [+3.69, +19.77]** | 0.9930 |
| top-edge | 33,699 | +5.58% [+1.17, +9.97] | 0.9972 |

with a race-log-loss gain of only -0.00040 and median CLV *below* 1.0: the
ROI-vs-CLV tripwire (0.3) fired. The residual model's top features in almost
every fold were `sex_H` and `sex_F`.

### The leak

The `sex` column is the horse's sex **on the day its results page was
scraped** (669k rows in March 2026), not race day:

- 2,842 two-year-olds are coded as mares and 255 as entire horses (both mean
  five or older); 4.76% of rows carry an impossible age/sex pair, falling
  toward the scrape date: 6.95% for 2020 races, 6.44% for 2023, 1.85% for
  2025, 0.06% for 2026;
- only 2.3% of horses with 8+ runs ever show a second code, although every
  filly becomes a mare and every entire colt a horse at five.

Sporting Life freezes a retired horse's code and updates the active ones, so
the code records the horse's *future*: "c" = a colt never gelded (kept entire
for stud value), "g" includes colts gelded later, "m" on a young filly = she
raced on past four, "f" on an old race = she retired young. **Model-free, the
21,292 runners with an impossible race-day code returned +21.5% at BSP
[+15.5, +28.2].** By sex code in 2020-23, entire males ("h") +43.7%, fillies
("f") -11.9%; the gap narrows toward the scrape date.

Every other per-race field was checked the same way and is as-of-race: trainer
changes over 47% of careers, official rating 99.9%, headgear 84%, form 100%
(and the next race's form string ends with this race's position 96.6% of the
time).

### Plan 4c without the leak

Same harness, every `sex_*` feature excluded:

| | with `sex_*` | **without** |
|---|---|---|
| race log loss vs market | -0.00040 | **-0.00000 [-0.00030, +0.00020]** |
| edge > 0.10 | +11.62% (13,681 bets) | **-4.98% [-19.83, +10.59]** (2,958) |
| top-edge | +5.58% | +2.01% [-2.17, +6.31] |
| edge > 0 | +0.69% | -1.50% [-3.20, +0.21] |

**The market-offset model learns nothing the market does not already know.**
Early stopping keeps 10-100 trees per fold - the residual signal is so weak
the model stops almost at once. Plan 4c, the best-motivated untried idea in
the project, fails cleanly: not a configuration problem but an informational
one. (Re-confirmed on the rebuilt store with the legitimate `sex_Female`
below, once it exists.)

### Consequences

- **Fixed in `e689d98`**: sex is collapsed to the sex at birth
  (`sex_Female` / `sex_Male`) - the only part history supports. Live cards
  carry race-day sex, so this also removes a train/serve skew. The
  never-working `first_time_gelded` is removed rather than "fixed" into a
  leak.
- **Every model trained on the allowlist since `sex_F`/`sex_H` entered it has
  used future information**, including the production run promoted earlier
  today. It is retrained on the rebuilt store.
- **No refutation in this ledger is weakened.** The leak flattered the model,
  so every recorded negative (market combination -3.55%, the disagreement
  analysis, ...) was measured on a model better-informed than any real one
  could be. They stand, and would only get worse.
- **The tripwire is now three for three.** High ROI with CLV at or below 1.0
  has meant a leak every time it has fired here.

## Things that are true and worth not re-deriving

- **RTV sectional coverage starts in 2023 at selected tracks.** Low coverage is
  *expected* — do not flag it as a bug or attempt a backfill. RTV has since
  been superseded by the At The Races / TPD scraper
  (`src/attheraces_scraper.py`); `add_rtv_features` is retired and no longer in
  the pipeline.
- **At The Races scraping** needs `curl_cffi` or the system `curl` to clear the
  Fastly TLS/JS challenge. ~2.5s per request, GB-only, resumable.
- **Bodyweight / run-to-run weight change** — the single most valuable
  biometric — is **not published in GB/IRE**. Do not hunt for it here.
- **Pedigree** is noise for exposed horses. If revisited, gate it to
  debutants, first-time-soft and new-trip runners only.

## Still open (not refuted, not done)

- **Market microstructure** — traded volume and ladder depth from Betfair
  historic PRO files. The only untapped orthogonal signal identified (§9).
- **Race-aware modelling** — conditional logit / softmax over the field, or
  learning-to-rank, instead of a per-horse binary classifier plus
  normalisation. Note a Plackett-Luce LightGBM objective already lost to the
  plain binary classifier once (2026-06-12, `scripts/_ab_pl_objective.py`).
- **Each-way 1/5-odds re-run** and a locked CLV harness — parked.
- **Retrain on the leak-fixed store** — pending; see
  `scripts/project_status.py`, which currently reports no promoted model.
