# History depth and estimate uncertainty012 — 26–27 October 2026

Complete: the 165-input candidate shows a small overall point improvement,
but **does not pass the advancement rule**. Winter log loss worsens and the
pooled confidence intervals cross zero. Production remains unchanged. The
strongest diagnostic lead is races containing form missing from our database;
that is a new hypothesis for a separate conditional blend, not a rescue of 012.

## What changes

The candidate has 165 inputs: the retained 149-input independent recipe plus
16 existing history-support columns. No new data is collected and no feature
generation or serving code changes. The original model already uses Glicko
rating uncertainty, relative experience and pre-race form counts. This test
adds absolute support, more explicit availability and pooled estimates.

| Group | Added inputs |
|---|---|
| Historical support | `horse_prev_races`, `jockey_prev_rides`, `trainer_prev_runs`, `jt_prev_runs` |
| Pooled rates | `horse_win_rate_shrunk`, `horse_place_rate_shrunk`, `jockey_win_rate_shrunk`, `trainer_win_rate_shrunk`, `jt_win_rate_shrunk` |
| Rating uncertainty | `horse_ts_sigma` |
| Rating availability | `has_horse_elo`, `has_jockey_elo`, `has_trainer_elo` |
| Form outside this database | `has_hidden_form`, `hidden_form_runs`, `experience_with_hidden` |

Counts and rates use earlier results only. The raw retroactive career total
`horse_runs` is not an input; an explicit poisoning test verifies that it
cannot alter the candidate's historical features. Pedigree support counts
are not stored, so none is invented or inferred from later outcomes.

The earlier betting/exposure diagnostics in ledger2c do not establish the
proper-score performance of this exact feature family. The002 sectional and
009 pace families are disjoint. This is one fixed family addition, without
feature subsets or a hyperparameter search.

## Input audit and amendment

The initial 17-column registration included both `has_horse_elo` and
`has_horse_glicko`. An outcome-free audit of every row in the first training
prefix found those flags exactly equal. The initial attempt was stopped
before any model was saved or any test fold was scored. Its manifest, source
archive, projection, log and abort receipt remain preserved. A new compact
registration removes only `has_horse_glicko`, giving 16 additions. This is
an input-quality amendment made without prediction-score selection.

The same audit found two exact float32 duplicate pairs already in the
baseline: `trainer_elo_vs_field` and its double-centered descendant, and
`horse_margin_elo_momentum_vs_field` and its double-centered descendant.
Those are separate cleanup findings. Both recipes retain them in this
comparison so the test measures only the history-family addition. No input
in the 165-column prefix is constant or entirely missing.

All seven actual-feature controls pass: prefix/full-history equality,
changed future outcomes, appended future races, removed current-day outcomes,
swapped current-day results, shuffled input order and poisoned retroactive
career totals. Cold-start, hidden-form and positive pooled-prior semantics
also pass. The test uses the native feature and rating functions, including
same-day shared jockey/trainer history.

The first pending-card fixture used NaN winner labels, which did not match
the serving contract and produced invalid pooled-rate values. Its failed
audit and source are retained. The corrected fixture uses the existing
`src/live_prediction.py` convention `won=0, finish_position=0` and adds the
separate swapped-result control. No feature-code change was used to pass
the audit, and robustness to NaN winner inputs is not claimed.

## Comparison design

Three new native LightGBM win models use the same 2000-tree cap, seed 42,
two threads, 180-day recency recipe, chronological capacity selection, full
training refit and training-only Platt/isotonic calibration as 009.

| Evaluation period | Last permitted training date |
|---|---|
| July–September2025 | June24,2025 |
| October–December2025 | September24,2025 |
| January–March2026 | December22,2025 |

The three saved 009 149-input models are reused after verifying the exact
native code, dataset, parameters, feature order, fitting cutoffs, labels and
full-field forecasts. Every baseline test probability must reproduce the
retained record within 1e-12. Their fitting metadata is not rewritten to
make them appear newly trained. Only the three new 165-input models are fitted.

Both recipes use the same float32 feature values and valid complete fields.
DNFs remain genuine losing starters. Dates are inspected retrospective
development; no fresh or prospective confirmation is claimed.

The primary advancement rule requires pooled winner log-loss improvement
with both race and date-cluster 95% upper bounds below zero, no increase in
runner Brier or binary loss, and improvement in at least two of three
quarters. All quarters must finish. No automatic production promotion.

Market context uses 009's frozen identical complete T-5 fields and their
retained field/label exclusions. Any fraction of the market gap closed must
use the same matched population for all three forecasts. Subgroups containing
DB debutants or external form are whole-race diagnostics, never selection
rules or a reason to discard an inconvenient field.

## Results

All three new models completed in 45.4 minutes. Saved artifacts confirm they
finished before the later PC shutdown; no training restart was required.

| Quarter | Races | Baseline winner loss | Candidate winner loss | Delta |
|---|---:|---:|---:|---:|
|2025Q3|3,532|1.92063259|1.92001199|−0.00062060|
|2025Q4|3,233|1.86228139|1.85986310|−0.00241829|
|2026Q1|2,591|1.81799380|1.81928807|+0.00129427|
|Pooled|9,356|1.87204489|1.87133338|−0.00071151|

The pooled race-log-loss 95% interval is [-0.00269877,+0.00126808]; date-cluster
sensitivity is[-0.00270503,+0.00114020]. The observed reduction is only about
0.038% of the baseline's own loss. Q1's interval also crosses zero.

On 85,863 runners, Brier improves 0.08754346→0.08750067 and binary loss improves
0.30088334→0.30077555. Runner Brier's one-sided95% upper delta is+0.00002082,
so its small pooled gain is also uncertain. Winner accuracy improves
29.0348%→29.2743%, while ten-bin ECE worsens0.00146349→0.00207259. Accuracy
and ECE do not override the primary proper-score rule.

The model improves primary loss in two of three quarters and the pooled
Brier/binary point guardrails hold, but **both primary uncertainty gates fail**.
Do not replace that result with the better autumn score or a favourable slice.

## Market and context diagnostics

On 8,942 identical complete fields/81,696 runners, baseline/candidate/market
winner losses are **1.86879634/1.86833831/1.74928256**. The independent model's
gap closes by only about **0.38%** in point estimate; the remaining gap is
substantial. Candidate minus market is+0.11905576,95%[+0.10877081,+0.12974341].
The frozen overlap excludes 394 unmatched/changed fields and 20 label conflicts;
none is removed for model performance. This is conditional retrospective
coverage, not live as-issued racecards or executable betting returns.

The predeclared whole-field diagnostics show a more specific lead:

- Races containing DB debutants:2,003fields, log loss1.65335984→1.65034556;
  the LL interval crosses zero, while Brier improves with a negative upper bound.
- Races containing external prior form:776fields, log loss1.70122800→1.69284715,
  delta−0.00838085, race95%[-0.01563035,−0.00120445] and date95%
  [-0.01482526,−0.00200918]. Runner Brier also improves.
- Races where every horse has a recorded prior start:7,353fields, virtually
  unchanged log loss1.93161597→1.93153175; the interval crosses zero.

The external-form group is a subset of the DB-debutant group, not an
additional disjoint population. These inspected diagnostics cannot select a
gate or retroactively qualify012. They motivate one separately registered
exploratory blend whose weights must use earlier OOS forecasts; any later
result still needs fresh confirmation because the context lead was found
using these already inspected periods.

## Evidence

Compact registration: `data/audit/model_improvement/012_history_family_registration_compact.json`.
Invariance: `012_history_invariance_compact.json`. Running package:
`data/audit/model_improvement/012_history_family_compact`. Frozen manifest:
`d20feecdb41a73238d0c8fa8509a8384ba713a5996befaa4445ec9240ce976fb`.
The original unscored attempt is preserved separately in `012_history_family`.

Outcome-free support/alias profiles are `012_history_profile.json` and
`012_baseline_feature_redundancy.json`. First-model gain/split usage is saved
as `012_feature_usage_Q3.json`; it is descriptive model usage, not an
incremental-value ablation and not a feature-selection rule.

All 48 focused checks pass, including the 5 new reference-reuse/qualification
checks and existing hidden-form, rating and native-probability checks. The
seven temporal/live-card invariance controls also pass. No full-suite rerun
is claimed for these isolated research additions.

Read-only qualification verifies all 17 sealed evidence files, source archive
CRC/contents, model/data/source fingerprints and unchanged production/study
artifacts. All baseline probabilities reproduce exactly. Each new model
replays 64 shuffled complete fields within 1e-12 (largest observed error 2.01e-14).
The saved tree counts are 1,986/1,647/1,977. The three candidate artifacts are
research models fitted through their historical cutoffs, not current deployed
models with new held-out claims.

Execution/test/verification logs: `data/audit/20261026/model-012-*.log`.
Read-only final qualification: `data/audit/model_improvement/verify_012.py`.
Production20261026_140950_205296, FE221fcd4db0fc5631/dataf3259ac47f8249b0 and
the activebaseline_v3 forward study remain unchanged. All001–011 sealed work
and the unscored initial166-input attempt are preserved.
