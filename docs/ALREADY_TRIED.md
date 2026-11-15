# Already tried

A one-line index of ideas that were built and measured before the live 11:00
rule and the Matchbook paper record existed (November 2026). The full entries
are in [archive/NEGATIVE_RESULTS.md](archive/NEGATIVE_RESULTS.md) and the
progress log in [archive/MARKET_PERFORMANCE_PROGRESS.md](archive/MARKET_PERFORMANCE_PROGRESS.md).

**Read these as history, not current fact.** Each verdict was measured on the
pipeline of its day: older stores, older models, sometimes known defects that
were fixed later. A "dead" entry lowers the prior for an idea; it does not
forbid it. Retry only with a concrete reason the old test no longer applies
(new data, a new target such as the correction or the 11:00 price, or a defect
in the old harness), and record the new result here.

## How a claim is judged

Still current; each rule was learned from a real mistake:

1. Judge a candidate by orthogonality to the market (correlation with
   `won − market_implied_prob`) and by CLV at BSP, not raw predictiveness.
2. Read CLV from `src.clv.summarise_clv` (median, geometric mean,
   percentage beating the close), never the mean of `taken / bsp` alone.
3. High ROI with CLV ≈ 1.0 is a leak, not an edge.
4. Check `scripts/project_status.py` before trusting any number.
5. Never fit a second stage on in-sample first-stage predictions
   (`wf_eval.oof_predict_expanding`).
6. Cluster confidence intervals by race (`wf_eval.bootstrap_roi_ci`).
7. Compare a selection rule against the same universe it bets in.
8. Do not overturn a result on a partial re-run.

## Facts worth not re-deriving

- RTV RaceiQ coverage starts 2023-06-10; there is no earlier history.
- At The Races needs `curl_cffi` or system `curl` (Fastly challenge); GB only.
- Bodyweight is not published in GB/IRE.
- Name keys are the weakness of every join; use the keys in `src/utils.py`
  and runner ids where they exist.

## Index (112 entries, newest first as in the ledger)

- [107. Hold-out: the 11:00 betting rule PASSES its registered test (15 November 2026)](archive/NEGATIVE_RESULTS.md#107-hold-out-the-1100-betting-rule-passes-its-registered-test-15-november-2026)
- [106. The correction's teacher: market weight 0.5 or 0.9 instead of 0.75 — no change (15 November 2026)](archive/NEGATIVE_RESULTS.md#106-the-corrections-teacher-market-weight-05-or-09-instead-of-075--no-change-15-november-2026)
- [105. Betting the corrected forecast at a morning price: beats the close retrospectively; 11:00 beats T-60 (15 November 2026)](archive/NEGATIVE_RESULTS.md#105-betting-the-corrected-forecast-at-a-morning-price-beats-the-close-retrospectively-1100-beats-t-60-15-november-2026)
- [104. Price timing: a fixed hour before the off beats 11:00; hours-to-off adds nothing (15 November 2026)](archive/NEGATIVE_RESULTS.md#104-price-timing-a-fixed-hour-before-the-off-beats-1100-hours-to-off-adds-nothing-15-november-2026)
- [103. Model form: elastic net far behind, linear trees a tie (15 November 2026)](archive/NEGATIVE_RESULTS.md#103-model-form-elastic-net-far-behind-linear-trees-a-tie-15-november-2026)
- [101. Debutants: Timeform's pre-race comment adds nothing beyond the text score (14 November 2026)](archive/NEGATIVE_RESULTS.md#101-debutants-timeforms-pre-race-comment-adds-nothing-beyond-the-text-score-14-november-2026)
- [100. Irish races without a class: effective class from prize money — not admitted (14 November 2026)](archive/NEGATIVE_RESULTS.md#100-irish-races-without-a-class-effective-class-from-prize-money--not-admitted-14-november-2026)
- [099. Seasonal comebacks: the horse's own reappearance record — null inside the correction (14 November 2026)](archive/NEGATIVE_RESULTS.md#099-seasonal-comebacks-the-horses-own-reappearance-record--null-inside-the-correction-14-november-2026)
- [098. Trainer moves: size of the move — not admitted as inputs, null inside the correction (14 November 2026)](archive/NEGATIVE_RESULTS.md#098-trainer-moves-size-of-the-move--not-admitted-as-inputs-null-inside-the-correction-14-november-2026)
- [Cheap screens: relatives' form, going reasons, pairwise and completion models, improvement, store pedigree and first string — null (9 November 2026)](archive/NEGATIVE_RESULTS.md#cheap-screens-relatives-form-going-reasons-pairwise-and-completion-models-improvement-store-pedigree-and-first-string--null-9-november-2026)
- [Text representation controls — little gain from more tokens (8 November 2026)](archive/NEGATIVE_RESULTS.md#text-representation-controls--little-gain-from-more-tokens-8-november-2026)
- [063. Race context, finer race types, subject models and meta learners — no gain (4 November 2026)](archive/NEGATIVE_RESULTS.md#063-race-context-finer-race-types-subject-models-and-meta-learners--no-gain-4-november-2026)
- [062. CatBoost pairwise — ties the 057 model at best; not worth a second framework (4 November 2026)](archive/NEGATIVE_RESULTS.md#062-catboost-pairwise--ties-the-057-model-at-best-not-worth-a-second-framework-4-november-2026)
- [061. Finishing order below third place — nothing more (2 November 2026)](archive/NEGATIVE_RESULTS.md#061-finishing-order-below-third-place--nothing-more-2-november-2026)
- [058–060. Luck weights, connection profiles and the figure model on its own — not confirmed (2 November 2026)](archive/NEGATIVE_RESULTS.md#058060-luck-weights-connection-profiles-and-the-figure-model-on-its-own--not-confirmed-2-november-2026)
- [055. Fixes for the 054 model's worst misses — all within noise (2 November 2026)](archive/NEGATIVE_RESULTS.md#055-fixes-for-the-054-models-worst-misses--all-within-noise-2-november-2026)
- [053. Unused Racing TV card data — nothing beyond the Timeform comment (2 November 2026)](archive/NEGATIVE_RESULTS.md#053-unused-racing-tv-card-data--nothing-beyond-the-timeform-comment-2-november-2026)
- [049. Race-type and course specialist win models — worse than the general model (1 November 2026)](archive/NEGATIVE_RESULTS.md#049-race-type-and-course-specialist-win-models--worse-than-the-general-model-1-november-2026)
- [044. Fixed RTV history family — uncertain development gain (30 October 2026)](archive/NEGATIVE_RESULTS.md#044-fixed-rtv-history-family--uncertain-development-gain-30-october-2026)
- [038. Date-local smooth win calibration — not selected (29 October2026)](archive/NEGATIVE_RESULTS.md#038-date-local-smooth-win-calibration--not-selected-29-october2026)
- [32. Earlier-fitted history-context blend — small uncertain gain (2026-10-27)](archive/NEGATIVE_RESULTS.md#32-earlier-fitted-history-context-blend--small-uncertain-gain-2026-10-27)
- [31. History depth, pooled rates and availability family — not qualified (2026-10-27)](archive/NEGATIVE_RESULTS.md#31-history-depth-pooled-rates-and-availability-family--not-qualified-2026-10-27)
- [30. Narrow relative floors on extreme-tail forecasts — no qualified gain (2026-10-26)](archive/NEGATIVE_RESULTS.md#30-narrow-relative-floors-on-extreme-tail-forecasts--no-qualified-gain-2026-10-26)
- [29. Broad shrinkage of isotonic forecasts toward Platt — not confirmed (2026-10-26)](archive/NEGATIVE_RESULTS.md#29-broad-shrinkage-of-isotonic-forecasts-toward-platt--not-confirmed-2026-10-26)
- [28. Existing pace/setup family in independent forecasts — no reliable gain (2026-10-26)](archive/NEGATIVE_RESULTS.md#28-existing-pacesetup-family-in-independent-forecasts--no-reliable-gain-2026-10-26)
- [1. Place-market edge — OVERTURNED (2026-06-24)](archive/NEGATIVE_RESULTS.md#1-place-market-edge--overturned-2026-06-24)
- [2. Market combination (Benter two-stage) — DEAD, re-confirmed (2026-10-12)](archive/NEGATIVE_RESULTS.md#2-market-combination-benter-two-stage--dead-re-confirmed-2026-10-12)
- [# What the two runs agree on](archive/NEGATIVE_RESULTS.md#-what-the-two-runs-agree-on)
- [# Harness defects found and fixed (worth knowing, did not change the verdict)](archive/NEGATIVE_RESULTS.md#-harness-defects-found-and-fixed-worth-knowing-did-not-change-the-verdict)
- [# Where the loss actually comes from (2026-10-12)](archive/NEGATIVE_RESULTS.md#-where-the-loss-actually-comes-from-2026-10-12)
- [# Measurement note: never read CLV off the mean alone (2026-10-12)](archive/NEGATIVE_RESULTS.md#-measurement-note-never-read-clv-off-the-mean-alone-2026-10-12)
- [# A caution about how this entry was nearly mis-corrected](archive/NEGATIVE_RESULTS.md#-a-caution-about-how-this-entry-was-nearly-mis-corrected)
- [2b. ATR sectionals — 20 of 21 signals already priced (2026-10-12)](archive/NEGATIVE_RESULTS.md#2b-atr-sectionals--20-of-21-signals-already-priced-2026-10-12)
- [2c. Why the model loses: its disagreements are uninformative (2026-10-12)](archive/NEGATIVE_RESULTS.md#2c-why-the-model-loses-its-disagreements-are-uninformative-2026-10-12)
- [# The decisive test](archive/NEGATIVE_RESULTS.md#-the-decisive-test)
- [# The mechanism](archive/NEGATIVE_RESULTS.md#-the-mechanism)
- [# What is NOT wrong](archive/NEGATIVE_RESULTS.md#-what-is-not-wrong)
- [# A fix was tested, and it failed](archive/NEGATIVE_RESULTS.md#-a-fix-was-tested-and-it-failed)
- [# Methodological note, recorded because it nearly produced a wrong answer](archive/NEGATIVE_RESULTS.md#-methodological-note-recorded-because-it-nearly-produced-a-wrong-answer)
- [# Can it be resolved?](archive/NEGATIVE_RESULTS.md#-can-it-be-resolved)
- [2d. Debutants: the model can rank them, but only over a 14x range (2026-10-12)](archive/NEGATIVE_RESULTS.md#2d-debutants-the-model-can-rank-them-but-only-over-a-14x-range-2026-10-12)
- [# What public proxies exist, and what is missing](archive/NEGATIVE_RESULTS.md#-what-public-proxies-exist-and-what-is-missing)
- [2e. Public data audit: what is collected and never used (2026-10-12)](archive/NEGATIVE_RESULTS.md#2e-public-data-audit-what-is-collected-and-never-used-2026-10-12)
- [2f. Label audit: labels are clean, bar amended results (2026-10-13)](archive/NEGATIVE_RESULTS.md#2f-label-audit-labels-are-clean-bar-amended-results-2026-10-13)
- [# The residual 0.04% is amended results, not scrape noise](archive/NEGATIVE_RESULTS.md#-the-residual-004-is-amended-results-not-scrape-noise)
- [# A trap in `attach_bsp` worth knowing](archive/NEGATIVE_RESULTS.md#-a-trap-in-attach_bsp-worth-knowing)
- [2g. The never-allowlisted families: all priced (2026-10-13)](archive/NEGATIVE_RESULTS.md#2g-the-never-allowlisted-families-all-priced-2026-10-13)
- [# Why the weather features are *exactly* 0.0000 - a structural lesson](archive/NEGATIVE_RESULTS.md#-why-the-weather-features-are-exactly-00000---a-structural-lesson)
- [2h. Sibling (dam-produce) form: priced overall; the debutant lead failed on the hold-out (2026-10-13)](archive/NEGATIVE_RESULTS.md#2h-sibling-dam-produce-form-priced-overall-the-debutant-lead-failed-on-the-hold-out-2026-10-13)
- [# The one result that clears the bar - treat it with suspicion](archive/NEGATIVE_RESULTS.md#-the-one-result-that-clears-the-bar---treat-it-with-suspicion)
- [# Hold-out verdict (2026-10-13): does not confirm](archive/NEGATIVE_RESULTS.md#-hold-out-verdict-2026-10-13-does-not-confirm)
- [# A bug the tests caught, worth recording](archive/NEGATIVE_RESULTS.md#-a-bug-the-tests-caught-worth-recording)
- [3. `trainer_freq` stablemate leak — RESOLVED (2026-06-26)](archive/NEGATIVE_RESULTS.md#3-trainer_freq-stablemate-leak--resolved-2026-06-26)
- [4. Par-time / going-corrected speed figures — redundant (2026-06-27)](archive/NEGATIVE_RESULTS.md#4-par-time--going-corrected-speed-figures--redundant-2026-06-27)
- [5. Sale price (Tattersalls) — fully priced (2026-06-29)](archive/NEGATIVE_RESULTS.md#5-sale-price-tattersalls--fully-priced-2026-06-29)
- [6. Intent signals — all three dead (2026-07-03)](archive/NEGATIVE_RESULTS.md#6-intent-signals--all-three-dead-2026-07-03)
- [7. Stewards reports — redundant with form (2026-06-26)](archive/NEGATIVE_RESULTS.md#7-stewards-reports--redundant-with-form-2026-06-26)
- [8. Why the model loses to the market — characterised (2026-06-27)](archive/NEGATIVE_RESULTS.md#8-why-the-model-loses-to-the-market--characterised-2026-06-27)
- [9. Feature-store saturation (2026-06-26)](archive/NEGATIVE_RESULTS.md#9-feature-store-saturation-2026-06-26)
- [10. Quality-control gates (2026-06-22)](archive/NEGATIVE_RESULTS.md#10-quality-control-gates-2026-06-22)
- [11. Course-dynamics features (2026-06-22)](archive/NEGATIVE_RESULTS.md#11-course-dynamics-features-2026-06-22)
- [12. Consolidation: what was deleted and why (2026-10-13)](archive/NEGATIVE_RESULTS.md#12-consolidation-what-was-deleted-and-why-2026-10-13)
- [# Refuted strategies and feature families](archive/NEGATIVE_RESULTS.md#-refuted-strategies-and-feature-families)
- [# Provably unreachable](archive/NEGATIVE_RESULTS.md#-provably-unreachable)
- [# Dead code *inside* live modules](archive/NEGATIVE_RESULTS.md#-dead-code-inside-live-modules)
- [# `_FE_UNWIRED_SOURCES` is gone, and that is the point](archive/NEGATIVE_RESULTS.md#-_fe_unwired_sources-is-gone-and-that-is-the-point)
- [# Cost of doing this mid-build](archive/NEGATIVE_RESULTS.md#-cost-of-doing-this-mid-build)
- [13. Phase A audit: what the pipeline was silently getting wrong (2026-10-13)](archive/NEGATIVE_RESULTS.md#13-phase-a-audit-what-the-pipeline-was-silently-getting-wrong-2026-10-13)
- [# The results database was missing races, systematically](archive/NEGATIVE_RESULTS.md#-the-results-database-was-missing-races-systematically)
- [# The flat draw backfill was finished, not pending](archive/NEGATIVE_RESULTS.md#-the-flat-draw-backfill-was-finished-not-pending)
- [# The feature build was not a pure function of the database](archive/NEGATIVE_RESULTS.md#-the-feature-build-was-not-a-pure-function-of-the-database)
- [# `fe_signature` changed with line endings](archive/NEGATIVE_RESULTS.md#-fe_signature-changed-with-line-endings)
- [# Live picks silently dropped every horse's history on a fresh scrape](archive/NEGATIVE_RESULTS.md#-live-picks-silently-dropped-every-horses-history-on-a-fresh-scrape)
- [# `promote` let unverifiable runs through](archive/NEGATIVE_RESULTS.md#-promote-let-unverifiable-runs-through)
- [# Plan 4b's class-weighting hypothesis describes code that does not run](archive/NEGATIVE_RESULTS.md#-plan-4bs-class-weighting-hypothesis-describes-code-that-does-not-run)
- [# Seven allowlisted features are race-level](archive/NEGATIVE_RESULTS.md#-seven-allowlisted-features-are-race-level)
- [# A free closing-price source, and the first golden hold-out](archive/NEGATIVE_RESULTS.md#-a-free-closing-price-source-and-the-first-golden-hold-out)
- [# Stalls position and rail movements: collected, not yet tested (plan 5a-bis)](archive/NEGATIVE_RESULTS.md#-stalls-position-and-rail-movements-collected-not-yet-tested-plan-5a-bis)
- [# Checked and clean](archive/NEGATIVE_RESULTS.md#-checked-and-clean)
- [14. Market microstructure, free version: the close absorbs it all (2026-10-13)](archive/NEGATIVE_RESULTS.md#14-market-microstructure-free-version-the-close-absorbs-it-all-2026-10-13)
- [# The timing edge, priced (`scripts/microstructure_clv.py`)](archive/NEGATIVE_RESULTS.md#-the-timing-edge-priced-scriptsmicrostructure_clvpy)
- [# The honest early-price tests: break-even at best](archive/NEGATIVE_RESULTS.md#-the-honest-early-price-tests-break-even-at-best)
- [# A data gap found on the way: the canonical BSP parquet was incomplete](archive/NEGATIVE_RESULTS.md#-a-data-gap-found-on-the-way-the-canonical-bsp-parquet-was-incomplete)
- [15. The `sex` look-ahead leak, and plan 4c without it (2026-10-13)](archive/NEGATIVE_RESULTS.md#15-the-sex-look-ahead-leak-and-plan-4c-without-it-2026-10-13)
- [# The third phantom edge, caught by the tripwire](archive/NEGATIVE_RESULTS.md#-the-third-phantom-edge-caught-by-the-tripwire)
- [# The leak](archive/NEGATIVE_RESULTS.md#-the-leak)
- [# Plan 4c without the leak](archive/NEGATIVE_RESULTS.md#-plan-4c-without-the-leak)
- [# The last shortlisted sectional signal, given its economic test](archive/NEGATIVE_RESULTS.md#-the-last-shortlisted-sectional-signal-given-its-economic-test)
- [# Consequences](archive/NEGATIVE_RESULTS.md#-consequences)
- [16. GBM vs logistic regression vs the market (plans 3a, 4b) (2026-10-13)](archive/NEGATIVE_RESULTS.md#16-gbm-vs-logistic-regression-vs-the-market-plans-3a-4b-2026-10-13)
- [18. Non-finishers were dropped from training and evaluation - the fourth tripwire catch (2026-10-14)](archive/NEGATIVE_RESULTS.md#18-non-finishers-were-dropped-from-training-and-evaluation---the-fourth-tripwire-catch-2026-10-14)
- [19. Morning money on debutants and favourites (pre-registered 2026-10-14)](archive/NEGATIVE_RESULTS.md#19-morning-money-on-debutants-and-favourites-pre-registered-2026-10-14)
- [# Result (`scripts/morning_money_segments.py`): both fail](archive/NEGATIVE_RESULTS.md#-result-scriptsmorning_money_segmentspy-both-fail)
- [20. Execution timing: when to strike, not what to back (EXPLORATION_PLAN B; pre-registered 2026-10-14)](archive/NEGATIVE_RESULTS.md#20-execution-timing-when-to-strike-not-what-to-back-exploration_plan-b-pre-registered-2026-10-14)
- [# Result (`scripts/execution_timing.py`): fails in every universe; and the reductions overturn ledgers 14 and 19](archive/NEGATIVE_RESULTS.md#-result-scriptsexecution_timingpy-fails-in-every-universe-and-the-reductions-overturn-ledgers-14-and-19)
- [# Exploratory checks on the lead (`scripts/execution_timing_diagnostics.py`; not tests)](archive/NEGATIVE_RESULTS.md#-exploratory-checks-on-the-lead-scriptsexecution_timing_diagnosticspy-not-tests)
- [22. Morning market-making, phase 1: feasible on paper, unknowable on free data (EXPLORATION_PLAN E) (2026-10-14)](archive/NEGATIVE_RESULTS.md#22-morning-market-making-phase-1-feasible-on-paper-unknowable-on-free-data-exploration_plan-e-2026-10-14)
- [23. Forecast ground changes, bet early (EXPLORATION_PLAN C; pre-registered 2026-10-14)](archive/NEGATIVE_RESULTS.md#23-forecast-ground-changes-bet-early-exploration_plan-c-pre-registered-2026-10-14)
- [# Result (`scripts/ground_forecast.py`): fails](archive/NEGATIVE_RESULTS.md#-result-scriptsground_forecastpy-fails)
- [24. Tote exotics priced from BSP (EXPLORATION_PLAN A): step 0 blocked on the operator (2026-10-14)](archive/NEGATIVE_RESULTS.md#24-tote-exotics-priced-from-bsp-exploration_plan-a-step-0-blocked-on-the-operator-2026-10-14)
- [25. Full ATR family in the independent probability model (2026-10-26)](archive/NEGATIVE_RESULTS.md#25-full-atr-family-in-the-independent-probability-model-2026-10-26)
- [26. Race-objective comparison: linear loses; tree primary gain unresolved (2026-10-26)](archive/NEGATIVE_RESULTS.md#26-race-objective-comparison-linear-loses-tree-primary-gain-unresolved-2026-10-26)
- [27. Monthly refitting: small Brier gain, primary benefit unresolved (2026-10-26)](archive/NEGATIVE_RESULTS.md#27-monthly-refitting-small-brier-gain-primary-benefit-unresolved-2026-10-26)
- [28. Market-history proposals034–035 (2026-10-29)](archive/NEGATIVE_RESULTS.md#28-market-history-proposals034035-2026-10-29)
- [29. Simple market-assisted refresh036 (2026-10-29)](archive/NEGATIVE_RESULTS.md#29-simple-market-assisted-refresh036-2026-10-29)
- [039–040. Market-adjusted connection form and own cadence: not selected (29October2026)](archive/NEGATIVE_RESULTS.md#039040-market-adjusted-connection-form-and-own-cadence-not-selected-29october2026)
- [041. Raw place-head information for win: not selected (29October2026)](archive/NEGATIVE_RESULTS.md#041-raw-place-head-information-for-win-not-selected-29october2026)
- [045. Repaired skill ratings: much better skill measures, still no market gain (30 October 2026)](archive/NEGATIVE_RESULTS.md#045-repaired-skill-ratings-much-better-skill-measures-still-no-market-gain-30-october-2026)
- [046. Projected race profile and RTV-derived inputs over the market (31 October 2026)](archive/NEGATIVE_RESULTS.md#046-projected-race-profile-and-rtv-derived-inputs-over-the-market-31-october-2026)
- [047. Re-encoding the 154 model inputs: real faults, no accuracy gain (1 November 2026)](archive/NEGATIVE_RESULTS.md#047-re-encoding-the-154-model-inputs-real-faults-no-accuracy-gain-1-november-2026)
- [Tier 2. Ability and same-day inputs inside the generalist: no better than as corrections (10 November 2026)](archive/NEGATIVE_RESULTS.md#tier-2-ability-and-same-day-inputs-inside-the-generalist-no-better-than-as-corrections-10-november-2026)
- [Error anatomy, 10-11 November 2026: what held, what did not](archive/NEGATIVE_RESULTS.md#error-anatomy-10-11-november-2026-what-held-what-did-not)
