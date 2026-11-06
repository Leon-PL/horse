# Action Plan

**Historical research plan, retained with later annotations.** The [README](../../README.md)
is the current operating guide. Use the [inventory](../PROJECT_INVENTORY.md) and
[implementation record](../MARKET_PERFORMANCE_PROGRESS.md) to place this plan's
work queues and decision gates in their historical context.

**Current work, 28 October 2026:** use [the canonical correction report](../research/CANONICAL_FEATURE_REPAIRS_026.md),
[the expanded teaching-data report](../research/BSP_TEACHER_REFRESH_027.md), and
[market-performance progress](../MARKET_PERFORMANCE_PROGRESS.md). Verified fixes
are part of normal training; their adoption depends on correctness, not a
retrospective performance veto. Experimental feature/model designs still
require measured comparisons against this corrected reference. The primary
objective is calibrated win probabilities, judged by race log loss and Brier;
winner accuracy is secondary. The older diagnoses below describe earlier
models and do not override that objective or the latest measured evidence.

**Written 2026-10-12.** Supersedes the old `plan.txt` (deleted 2026-10-13;
recoverable from git), whose Stage 2–5 assumptions predate the refutations in
[NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md).

Read in this order: [NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md) for what is
already dead, then §1 here for why the model loses, then pick up at §2.

Every item below states a **success criterion** that can fail. Items without
one are not work, they are hope.

---

## 1. Diagnosis: why is the model wrong?

From run `20260623_223148` (646 test races — indicative, not definitive; see
the caveat at the end of this section):

| metric | model | logistic baseline | **market** |
|---|---|---|---|
| top-1 accuracy | 0.268 | 0.292 | **0.348** |
| NDCG@1 | 0.374 | 0.399 | **0.446** |
| winner in top 3 | 0.605 | 0.629 | **0.691** |
| log loss | 0.295 | 0.297 | **0.276** |
| Brier | 0.0855 | 0.0860 | **0.0808** |
| RPS | 0.246 | 0.258 | **0.213** |
| ECE (calibration) | **0.0054** | 0.0075 | 0.0147 |

Read the last row against the first. **The model is the best-calibrated of the
three and the worst at discrimination.** It knows how often it is right; it
does not know *which* horse wins. The market picks the winner ~30% more often
(0.348 vs 0.268 — eight percentage points).

That splits the problem into five separate failures with five different fixes,
and they are not equally expensive to attack.

### 1a. Earlier aggregate calibration evidence is incomplete

The earlier ECE0.0054 and broad price-band agreement do not establish that
every probability range is reliable. Later audits reproduced isotonic
near-zero probabilities for eventual winners, with a material log-loss cost.
The first calibration alternatives did not pass their fixed validation gates;
see [010](../research/MODEL_CALIBRATION_010.md) and [011](../research/MODEL_TAIL_SMOOTHING_011.md).
Further changes require measured proper-score improvements. Aggregate ECE
alone is insufficient to accept or dismiss them.

### 1b-prime. The sharper diagnosis (2026-10-12): the disagreements are noise

A head-to-head on **25,388 walk-forward races** (ledger §2c) locates the
deficit exactly. The model and the market pick the same horse in 55% of races,
and that horse wins 38.8% of the time. The whole gap is in the other 45%,
where the market wins 0.2809 to 0.1746 — and there:

- the horse the model **promotes** wins 0.1746 against a market-implied
  **0.1788** (−0.43 pp), and
- the horse it **demotes** wins 0.2809 against **0.2814** (−0.05 pp).

**The disagreement carries no information.** The model is re-ordering runners
the market has already priced correctly. When it overrules, it keys on
`horse_elo_momentum` and `horse_glicko_vs_field` — the most public signals in
racing — and conditioning on those drivers makes the pick *worse* than its
price (−1.11 and −1.16 pp).

This narrows §1b below. Calibration is right at every exposure band and
momentum quintile (0.98–1.04×), no subgroup is broken, and a tested fix
(declining thin-form hot-momentum bets) moved the rule by 0.24 points, inside
noise. So the remaining §1b question is only whether the *configuration* is
costing discrimination — worth the day it takes (§4b), but it can no longer be
the main explanation, because a perfect re-rank of priced information still
cannot beat the price.

**What it promotes:** §4c (market log-odds as `init_score`) is now the
best-motivated item in this plan. The model's defect is that it does not know
which of its features the market has already priced; an offset is the one
design that forces it to learn only the residual.

### 1b. Is the *configuration* costing discrimination?

The model loses to **plain logistic regression** on every ranking metric
(top-1 0.268 vs 0.292, NDCG@1 0.374 vs 0.399). A 2,000-tree LightGBM with 151
engineered features should not rank worse than a linear model on the same
data. Candidate causes, in order of likelihood:

- ~~**Class weighting distorts ranking.**~~ **Premise wrong (ledger 13):** the
  LightGBM win and place models are already unweighted; only the XGBoost paths
  take `scale_pos_weight`. Original text kept for the record: the win classifier trains with
  `scale_pos_weight ≈ (N−1)` / `auto_class_weights="Balanced"`. For a
  one-winner-per-race problem that massively upweights winners and reshapes
  where the model spends capacity. It is the right choice for balanced
  accuracy and the wrong one for ranking within a race.
- **Over-regularisation.** `learning_rate 0.008` with `reg_alpha 0.5`,
  `reg_lambda 3.0`, `min_child_samples 50` is a heavily damped configuration;
  combined with early stopping it may be underfitting the discriminative
  signal while fitting the marginal rate well (which is what good ECE
  measures).
- **The binary-then-normalise architecture** ignores that exactly one horse
  wins. See §4a.

**Caveat on the numbers:** 646 test races puts the standard error on top-1 at
roughly 1.8 points, so model-vs-logistic (2.4 points) is suggestive, not
established — about 1.3σ. Model-vs-market (8.0 points) is ~4.4σ and is real.
**Re-measuring this properly on the rebuilt store is item 3a** and it is the
single cheapest high-information check available.

### 1c. It is under-informed relative to the market — partly irreducibly

Even fixed, the model would still trail. The market price is a weighted
consensus of thousands of opinions including professional syndicates running
their own models, and it contains information no public source carries:

- **Private stable information** — gallop reports, fitness, intent, undisclosed
  niggles. Previously characterised: the gap concentrates on **favourites and
  debutants** and vanishes once that information is exposed. The sale-price
  work confirmed the debutant gap is genuinely private, not a public price
  anyone can scrape (ledger §5).
- **Aggregation.** Beating a consensus requires being better than its best
  participants, not better than its average one.

This part is **not rectifiable from public data**. Any plan that implicitly
assumes it is will fail the way the last four months did.

### 1d. 216 sectional features existed unused — now triaged, and mostly priced

The feature store holds **1,178 columns**, of which **216 are `atr_*`** —
velocity curves, finishing-speed percentages, stride length and frequency,
energy distribution, sectional-adjusted placings, from the At The Races / TPD
scrape that cost a Fastly-challenge workaround and a resumable multi-week
backfill at ~2.5s per request.

**The allowlist is 151 features and contains zero of them.** Its composition:

| family | count |
|---|---|
| trainer | 29 |
| horse | 28 |
| jockey | 19 |
| jockey×trainer | 7 |
| field/context | 7 |
| days-since | 6 |
| sire/damsire | 9 |
| official rating | 5 |
| speed figures | 3 |
| Elo | 3 |
| other | ~35 |

Three-quarters of it is entity history — trainer, horse and jockey form. That
is precisely the public form the market prices most efficiently, and precisely
what the saturation finding said adds nothing. Meanwhile the one dataset that
is genuinely hard to obtain and hard to process was sitting unused.

**Closed 2026-10-12, and the answer was not the hoped-for one.** Triaging all
216 columns (ledger §2b) found **20 of 21 signals already priced**, all under
|0.0075| partial correlation with the market residual. Only
`atr_closing_in_hand` clears the bar, at +0.0105 — marginal, and still owing a
CLV test.

So this was the largest gap between what the project owns and what it uses,
and closing it did not produce an edge. It does, however, settle the question:
**public-form feature engineering is finished here.** The market prices even
velocity curves and stride biomechanics. What remains is §5a.

### 1e. The selection rule then gives back whatever edge survives

The value filter costs **0.87 points** against backing everything in the band,
concentrated in the middle of the book (6–25, 59% of stakes). Full breakdown in
ledger §2. Fixing the model without fixing the rule banks nothing.

### Summary

| failure | fixable? | cost | where |
|---|---|---|---|
| calibration | already fine | — | stop |
| model under-discriminates | probably | low | §3a, §4a |
| sectionals unused | closed — 20/21 priced | done | ledger §2b |
| selection rule loses value | probably | low | §4c |
| market holds private info | **no** | — | accept |
| market aggregates many opinions | **no** | — | accept |

---

## 2. Phase 0 — restore the instrument (blocking everything else)

`scripts/project_status.py` currently reports **not ready** on three counts.
Nothing measured before this is finished can be trusted, and two of this
project's largest "edges" were artifacts of exactly this.

| # | task | success criterion |
|---|---|---|
| 2a | Refresh results — **DONE 2026-10-13**, plus a full-history relist that restored 543 races the old sync had dropped (ledger 13). | met: database to 2026-10-12 |
| 2b | Rebuild the feature store — **DONE 2026-10-13** (52 min; signature now line-ending invariant and stamped from build start). | met: code and data signatures match |
| 2c | Retrain and promote — **DONE 2026-10-13**: run `20261013_162640` promoted; `promote` now refuses unverifiable runs. | met: `project_status.py` exits 0 |
| 2d | Flat **draw backfill** — **DONE** (verified 2026-10-13, ledger 13: draw-vs-weight correlation ~0 in every month, where saddle cloths would give ~-1). | met |

---

## 3. Phase 1 — data correctness, and use what you own

Highest information per unit of effort. Do all of this before touching
modelling architecture or buying data.

### 3a. Re-measure the model-vs-logistic gap properly — **DONE 2026-10-13: the GBM does not rank worse (ledger 16)**

On 24,405 races: top-1 0.2930 vs 0.2893 (+0.36 pt, CI spans 0); race log loss
GBM better by 0.021. The 646-race result was noise. Market: 0.3396.


Re-run the §1 comparison on the rebuilt store with the full test set, not 646
races. **Success criterion:** a definitive answer to "does a 2,000-tree GBM
rank worse than logistic regression on the same features?" If yes, §4b is the
priority and the whole modelling configuration is suspect. If no, §1b closes
and attention moves to §3b.

### 3b. Triage the 216 sectional features — **DONE 2026-10-12, mostly negative**

**Result: 20 of 21 signals are already priced** (ledger §2b). Only
`atr_closing_in_hand` clears the orthogonality bar, at partial correlation
+0.0105 (t=+4.5 on 184k rows) — marginal, and still owing a CLV test. The
derived "had more to give" measure beat the explicit in-running trouble
comment, which scored weakest of all 21, so the data-native-beats-text
prediction was directionally right and economically irrelevant.

This fires **decision gate 3** below. The remaining work on this item is the
CLV test of the one shortlisted signal, and re-confirming the triage after the
store rebuild (it ran against a store whose `fe_signature` is stale).

<details><summary>Original plan for this item, for reference</summary>

Run the established triage on each family:

1. correlation with the market residual (`won − mkt_imp`) — ~0 means priced, stop;
2. CLV at BSP under the locked walk-forward, with the ROI-vs-CLV tripwire.

Priority order, most-orthogonal first:

- **Sectional-derived trouble** — horses closing fastest at the line but
  beaten ("staying on", trip short, traffic). This is the complete,
  full-coverage, no-text-parsing version of the signal that stewards reports
  failed to provide (ledger §7), and it captures incidents the stewards never
  reported.
- **Stride biomechanics** — length × frequency, and how a horse *produces*
  speed. Much harder for the market to price than split times.
- **Energy distribution / closing speed in hand** — did the horse have more to
  give?
- **Race-level pace projection** (§3c) built on the sectional pace styles.

**Success criterion:** each family either enters the allowlist on a CLV
improvement, or gets an entry in the ledger. Either outcome is progress; the
current state — 216 columns with no verdict — is not.

**Caveat to respect:** ATR coverage is GB-only and starts partway through the
history. Gate on a coverage flag; do not median-fill (the RTV lesson).

</details>

### 3c. Build the race-shape model *(≈2 days)*

The store has per-horse `pace_style_index` but no **race-level** pace scenario:
count the likely front-runners, project whether the race sets up as a
lone-leader steal or a collapse, then score each horse's fit to that scenario
and to the draw. Pace fit is technical enough that the market plausibly prices
it imperfectly, and this is a derived feature — no new data.

**Success criterion:** CLV improvement under the walk-forward, or a ledger entry.

### 3c-bis. Triage the never-seen families - DONE 2026-10-13, all 43 priced (ledger 2g)

Before building anything new, point the triage tool at each family that is in
the store and absent from the allowlist (ledger §2e): weather (10 columns, 0
allowlisted), going detail (11/0), headgear and first-time equipment (4/0),
debut features (11/0), `trainer_first_string_*` (7/0), draw (11/2), course and
track detail (22/3).

The allowlist was selected for **raw win prediction**, not for orthogonality to
the market, so "not allowlisted" does not mean "tested and dead". Each family
is one command:

    python scripts/triage_features.py --prefix weather_

**Success criterion:** every family either enters the shortlist or gets a
ledger line. Expect mostly "priced" after §2b, but the cost is minutes.

### 3c-ter. Sibling / dam-produce form - BUILT 2026-10-13; priced overall; debutant lead FAILED on the golden hold-out (ledger 2h)

The model ranks debutants over a 13.9x probability range where the market uses
53.6x (ledger §2d) — it cannot separate the 1.4 shot from the 200/1 shot.
Sibling form is the most-used public debutant signal in racing and **does not
exist in the store at all**. `dam` is in the results table, so it is
computable: for each runner, the career record of other horses sharing its dam
(wins, best official rating, best class) from races strictly before today.

Mind hard rule 1 — it is a history feature, and siblings running on the same
day must be date-strict (`tests/test_live_parity.py` will catch a slip).

**Expectation-setting:** sale price, the canonical public proxy for a fancied
debutant, tested dead on debutants (residual 0.005). Sibling form is evidence
rather than a price, but do not expect much.

**Success criterion:** improves debutant AUC toward the market's 0.825, or a
ledger entry.

### 3d. Label correctness audit — **DONE 2026-10-13, labels are clean**

0 races without a winner, 0 `won`/`finish_position` disagreements, 0 short
fields, and **0.0429%** disagreement with Betfair's settled result across
586,970 matched runners (ledger §2f). The residual is amended results
(disqualifications and placing reversals), ~0.15% of races, which the scrape
stores as first-past-the-post.

Remaining action, small: the backtesters settle on our `won`, so a bet on a
horse later disqualified is credited with a win the exchange never paid — a
bias in our favour, well under 0.1 pp of ROI. Prefer Betfair's settled `won`
for settlement where it is matched.

<details><summary>Original item</summary>

`plan.txt` Stage 1 called for this and it was never carried out. Cross-check
against Betfair: `won` / `placed`, non-runners, **dead heats**, voids,
places-paid, and the horse-name/track/off-time merge keys. Labels that are
wrong even 0.5% of the time put a ceiling on achievable accuracy and quietly
flatter or punish every experiment.

**Success criterion:** a disagreement rate, and a fix or a documented
explanation for each category.

</details>

### 3e. Retire dead weight — LARGELY DONE 2026-10-13 (ledger 12)

Done: `par_time_features`, `stewards_features`, `add_rtv_features` +
`rtv_scraper`, the sale-price scraper, the refuted place-market strategy and
its paper trader, `strategy_calibrator`, and 17 one-off scripts are deleted
(≈5,200 lines). `_FE_UNWIRED_SOURCES` is gone with them — the mechanism only
existed to stop dead code forcing rebuilds. See `docs/archive/PROJECT_OUTLINE_before_cleanup_2026-11-05.md` §2.0 for
the map that replaced the guesswork.

Still open:

- **Prune the allowlist**: ablate under the locked walk-forward and drop what
  does not pay. 151 features, three-quarters entity history, and the
  saturation finding says the marginal ones add nothing. This is the one item
  here that can change a number, so it needs CLV, not a code reading.
- 2 of 8 runs carry no feature signature.

---

## 4. Phase 2 — modelling

Only after Phase 1. Ordered by expected value given §1.

### 4a. Race-aware modelling *(≈3–4 days)*

The current design trains a per-horse binary classifier and normalises within
race, which ignores that exactly one horse wins. Given §1 — well calibrated,
poorly discriminating — the gain from respecting race structure should land
exactly where the deficit is.

- **Conditional logit / softmax over the field** (Harville/Henery), or
  **learning-to-rank** (LambdaMART, native in LightGBM).
- **Place probabilities derived from win probabilities** (Harville) instead of
  a separate place classifier — more coherent, and one less thing to break.

**Honest prior:** a Plackett-Luce LightGBM objective already lost to the plain
binary classifier once (2026-06-12). But that test predates the leak fix and
the current store, and — per §1b — "beats the binary classifier" may be a low
bar if the binary classifier ranks worse than logistic regression. Any retest
must use race log loss and Brier, with ranking metrics as
secondary diagnostics. The historical assertion that probability quality was
already competitive is superseded by the matched market comparisons.

**Current success criterion:** improve race log loss and Brier against the
matched current model on the fixed chronological comparison. Report top-1,
NDCG and market context separately.

### 4b. Fix the objective before changing the architecture — **DONE 2026-10-13: ruled out (ledgers 13, 16)**

Class weighting was never on for LightGBM; loosening regularisation worsens
race log loss by 0.013 (significant). The configuration is not the problem.


Two configuration experiments that follow directly from §1b:

- **Drop class weighting** (`scale_pos_weight` / `auto_class_weights`) and
  compare ranking metrics. Balanced weighting serves calibration, which is
  already solved, and may be costing discrimination.
- **Loosen regularisation / raise the learning rate** and compare. The current
  configuration is heavily damped.

**Success criterion:** ranking metrics improve, or both are ruled out with
numbers.

### 4c. Train on the market's residual, not on the outcome — **TESTED 2026-10-13: FAILS (ledger 15)**

**Result:** a per-race conditional logit with the market as a fixed offset
learns nothing the market lacks: race log loss vs market **-0.00000
[-0.00030, +0.00020]** over 13 purged quarters, early stopping at 10-100
trees, every selection rule's CI spanning zero. Its first run showed +11.62%
at BSP with CLV < 1; the tripwire fired, and it was a look-ahead leak in the
`sex` column (scrape-time, encoding each horse's future career), now fixed.

<details><summary>Original item</summary>


This is the genuinely different idea, and it has not been tried. Three things
have: market probabilities as *features* (fund+market), as *soft labels*
(`WIN_SOFT_LABEL_DISTILL` — which teaches the model to **imitate** the market),
and as a *post-hoc blend* (the Benter anchor).

What has not been tried is the market log-odds as a **fixed offset**
(LightGBM `init_score`, XGBoost `base_margin`). A model trained on `won`
spends nearly all its capacity relearning what the market already knows — the
large, easy signal of class and ability — leaving a sliver for the residual.
With the market as an offset, the model can *only* learn what the market
misses. That is Benter's actual design, and it attacks §1c from the one angle
that is not structurally blocked.

**Success criterion:** positive CLV at BSP under the locked walk-forward. Watch
the ROI-vs-CLV tripwire — an odds-derived target makes leakage easy.

</details>

### 4d. Conditional edge: find *where* you beat the market *(≈1 day)*

Stop asking "do we beat the market" and ask "where". Fit the Benter α
separately by price band, field size, race type and runner exposure. The
selection-rule audit already hints at structure: the filter adds value at
odds < 6 and 25–50 and loses in the middle, and the model is better calibrated
than the market at the extremes.

Related: a **new selection rule**. The current `combo_prob > mkt_imp` is
vacuous at the margin — because β ≈ 0.97 pins the anchored probability to the
market, `combo_prob/mkt_imp > 1.10` matches 16 bets out of 104,146. Any
replacement must use **rank or relative edge within a race**, not an absolute
ratio threshold.

**Success criterion:** a segment with α materially above the pooled +0.046 that
survives out-of-sample, or the conclusion that no such segment exists.

### 4e. Specialised sub-models + stacking *(only if 4a–4d show signal)*

Flat vs jumps, distance bands, handicap vs non-handicap, with isotonic
calibration per race type and a stacked meta-model. Low prior — GBMs on
segmented data usually just lose sample size — so do not start here.

---

## 5. Phase 3 — new public data

Most public form is priced; the ledger records seven refutations. Only three
categories are plausibly orthogonal, and they are ranked by
(orthogonality × availability) ÷ effort.

### 5a. Betfair market microstructure — **free version TESTED 2026-10-13 (ledger 14)**

**Result (free public data: morning / pre-play matched volume):** morning
money predicts the later price move and the result *beyond the morning price*,
but **nothing beyond BSP** (partial 0.0004, t 0.4): the close absorbs it. The
timing edge it leaves at T-5 is real and too small (median CLV 0.9953 vs 0.9809
for the matched band; < 1.0). A morning-VWAP version that looked like a pass
(1.0187) is look-ahead by construction; the honest 11:00-price test is in
ledger 14. What remains untested is the **paid** version (ADVANCED/PRO ladder
depth and volume through time), an operator spending decision - see
docs/archive/ROADMAP.md.

Original item:

The project holds the *price trajectory* (`ltp_300s`, `ltp_60s`,
`ltp_preoff`, `bsp`) but **no traded volume and no ladder depth**. Volume-
weighted moves, which side of the book money enters, and how late it arrives
are the most direct available proxy for the private information in §1c — and
it is a latency/processing edge rather than a form edge, which is the category
that has not already failed here.

Source: Betfair historic data (`historicdata.betfair.com`); the richer files
carry matched volume per runner over time. Licence tiers differ — check what
the account allows before planning around it. Live equivalent: the Exchange
Stream API ladder.

**Caveat learned the hard way:** features built *from odds* can show apparent
CLV purely by proxying the price move itself, which is not tradeable (the
jockey-booking result, ledger §6). Validate on **continuation**, not
correlation: does the signal predict the move *after* the observation point?

### 5a-bis. Stalls position and rail movement — a collection gap, not a redundancy

**The store has 11 draw columns and no stalls side.** Draw bias is close to
meaningless without knowing whether the stalls are high, low or centre, and
rail movements change the effective trip. Both are published in every GB going
report. Unlike most candidates this does not add a new signal so much as make
an **existing allowlisted feature (`draw`) mean what it is supposed to mean** —
which is why it is worth doing even though the ledger says most public data is
priced. See ledger §2e.

**Success criterion:** `draw`-derived features show a materially different
(ideally stronger) partial correlation once conditioned on stalls position, or
the conditioning is shown to add nothing.

### 5b. GoingStick numeric readings - DOWNGRADED, see ledger 2g

A GoingStick reading is another **race-level constant**, and race-level
features cancel exactly in any per-race-normalised model - which is why all
seven weather signals score 0.0000 rather than merely small. It could only
pay through an interaction with each horse's own going record, and that
family already scores 0.0019. Low priority.

<details><summary>Original item</summary>

The official penetrometer value (e.g. 7.2) rather than the verbal "good to
soft", published per GB meeting and rarely used quantitatively. The going
*category* is priced; the numeric reading is finer-grained and readable pre-off
from the early races on a card. Pairs naturally with the par-time work, which
was CLV-flat on its own but whose underlying data is sound and kept.

</details>

### 5c. GPS / stride biomechanics beyond sectionals - highest ceiling, commercial

Total Performance Data (`totalperformancedata.com`) is the official tracking
provider for a large set of GB tracks: stride length and frequency,
position-in-running, speed by furlong. Commercial licence. Only worth pursuing
if §3b shows the ATR sectionals already carry signal — if the cheap version of
this data is priced, the expensive version will be too.

### 5d. Jurisdiction expansion — a strategic choice, not a task

The single most valuable biometric, **run-to-run bodyweight change**, is
published in Hong Kong, Japan and Australia and **not** in GB/IRE. Expanding
jurisdiction would also mean smaller fields and different market structure. It
is a months-long rebuild of scrapers, merge keys, track config and ratings, so
treat it as a decision about what the project is, not as a backlog item.

### Do not bother

Weather (held, priced), going category (held, priced), pedigree (held, priced —
and only conditionally useful for debutants and new trips), anything derived
purely from past finishing positions, and public sale prices (ledger §5). See
the ledger before adding to this list.

---

## 6. Phase 4 — strategy and staking

**Gate: do not start until CLV at BSP is positive on an unbiased estimator**
(`median_clv` / `geometric_mean_clv`, never the mean ratio — see ledger).

Then: Kelly fraction and bankroll under realistic liquidity, execution against
the ladder rather than at an assumed price, commission tiers, and the each-way
1/5-odds re-run that is still parked. All of it is worthless without a
positive closing-line edge, and all of it is cheap once there is one.

---

## 7. Decision gates

Checkpoints where the honest move may be to stop:

1. **After Phase 0 — PASSED 2026-10-13.** `project_status.py` is green on all
   four links (database to 2026-10-12; store rebuilt at `e522ca1b86174217`
   and verified on code *and* data; run `20261013_162640` promoted and
   matching), after repairing 543 missing races and closing the gaps that
   let a stale store pass (ledger 13).
2. **After 3a + 4b — FIRED 2026-10-13.** With weighting and regularisation
   addressed the GBM ties logistic regression on top-1 (better only on log
   loss) and both trail the market by ~5 points (ledger 16): the feature set
   is the binding constraint. §1d (sectionals) was already priced; §5a fails
   in its free form (gate 4).
3. **After 3b — FIRED 2026-10-12.** The sectionals were the last unexploited
   asset in hand, and 20 of 21 signals are priced (ledger §2b). That is strong
   evidence the market prices everything reachable from public form, and it
   promotes §5a (market microstructure) to the only live hypothesis. Treat
   further *public-form* feature engineering as closed.
4. **After 4c + 5a — FIRED 2026-10-13.** Residual-targeted training finds
   nothing beyond the market (race log loss -0.00010 [-0.00040, +0.00020],
   ledger 15) and the free microstructure data is fully absorbed by BSP, with
   a timing edge that does not reach break-even at 11:00 once non-runner
   reductions are counted (ledgers 14, 20). Only
   paid order-book data (Betfair ADVANCED/PRO) remains untested. There is no
   public-data edge to find; continuing is a hobby rather than an investment -
   a perfectly good reason to continue, but a conscious one. See
   docs/archive/ROADMAP.md.

**What success looks like:** median CLV > 1.0 with a race-clustered confidence
interval excluding 1.0, on a locked walk-forward, holding on a golden hold-out
never used for tuning. Nothing short of that is an edge, whatever the ROI says.

---

## Appendix: non-negotiables

Carried from `CLAUDE.md`, each corresponding to a bug that actually shipped:

1. History features must exclude the current race; never a grouped cumulative
   followed by a global `.shift()`.
2. Bet selection and settlement live in `src/bet_settlement.py`.
3. Merge keys live in `src/utils.py`.
4. Anything that can change a feature value is registered in
   `_FE_SOURCE_FILES`.
5. Check `scripts/project_status.py` before trusting any number.
6. Walk-forward evaluation uses `src/wf_eval.py` — purged folds, out-of-fold
   second stages, race-clustered intervals.
7. Read CLV from `src.clv.summarise_clv`; never the mean ratio alone.
8. Compare a selection rule against the **same universe it bets in**.
