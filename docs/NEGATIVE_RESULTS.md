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
version of the same signal, with no text parsing. `src/stewards_features.py`
exists but is **not wired into the pipeline**.

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
- **Flat draw backfill** following the draw-bug fix — pending.
- **Retrain on the leak-fixed store** — pending; see
  `scripts/project_status.py`, which currently reports no promoted model.
