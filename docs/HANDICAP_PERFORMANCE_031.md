# Historical handicap performance proposal031

Implemented28 October2026 after the user requested continuous work with fewer
model fits. The installed029 model and its151 selected inputs remain unchanged.

## Proposal and audit

The existing speed inputs are margin proxies. This proposal puts a past
handicap result on an approximate ability scale using event-specific official
marks, declared weights and cumulative beaten lengths. It then compares the
last-three-start mean and best ability, less today's declared weight, with
the field. A third feature reports valid historical support.

For each of the first three finishers, compute `rating - weight + lb_per_length
* beaten_lengths`. Their median anchors the race; a runner's performance is
`anchor + weight - lb_per_length * beaten_lengths`. All three anchors require
observed marks, weights, valid cumulative margins and mature ages. This is a
mechanical approximation, not an official BHA performance figure.

The [BHA guide](https://www.britishhorseracing.com/regulation/performance-figures/)
explains the roles of weight, opposition and distance. Its typical Jumps graph
is saved with the audit. Flat uses a fixed17.05/f approximation to the given
5f/14f examples; Jumps interpolates the published typical scale with end slopes
extended to16f/36f. No coefficient was fitted. Neither approximation includes
ground, pace, sectionals, time or discretionary adjustments. The BHA adds rider
allowances back when assessing performance; these figures therefore use the
source's declared weights. They do not require a speculative claim backfill.

Weight-for-age is not guessed: assess Flat5+ and Jumps6+ only, and require the
top-three anchors to meet those restrictions. This costs coverage. Histories
are separate for Turf, AW, Hurdle and Chase; Bumpers are unsupported. An actual
start with no valid figure advances the three-start window as unknown. Nonstarters
and pending cards do not. Figures expire after365 days. Current-day outcomes
never enter pre-race features.

12 targeted tests pass. Full real-data prefix and seeded-append calculations
agree exactly. Preparation/audit took70.05s, without a model fit. Raw extremes
were inspected: very low figures arise from large recorded finishing margins;
they were not converted to invented ordinary performances. Coverage and
quantiles are in `data/audit/handicap_performance_031/audit.json`; the aligned
projection has SHA256 `d27e75122abe7c070dcf5bab79064d60326a4a77ad167b416f7c6a0a14c2022e`.

## Fixed inexpensive screen

Add the three inputs to the frozen151 baseline features. Reuse the exact030
baseline learner. One400-tree candidate win fit, common native labels, teacher,
weights and settings; no calibrator, place fit, capacity search or full rebuild.
Train24 September2023–23 September2025; evaluate October–December2025.
This is previously inspected development data, not a fresh holdout.

Completed with exactly one new fit in75.91s, including preparation/prediction
overhead. The baseline cache key remains `77f065e8ecd6...` and was reused.

| Population | Baseline log loss | Proposal log loss | Baseline Brier | Proposal Brier |
| --- | ---: | ---: | ---: | ---: |
| All3233 fields/30081 runners |1.847941|1.848717|0.085482|0.085473|
| Established handicaps1071 fields/9777 runners |1.987342|1.993360|0.091748|0.091995|

The all-field primary score worsens despite a tiny runner-Brier improvement.
Both scores worsen in the target established handicaps. Target log-loss delta
is+0.006019, date-cluster95% interval[-0.000093,+0.012629]. The screen provides
no reason to advance this proxy to full training. No coefficient/age/window
retuning or extra-period evaluation followed. The three proposal inputs remain
unselected, research-only. These results do not refute better sourced speed or
performance information; this crude approximation adds no demonstrated benefit.

Evidence: `data/audit/handicap_performance_031/quick_screen/report.json` and
paired predictions. The normal store, training path,151-input default and
installed029 model are unchanged. All objective repairs from earlier audits
remain in ordinary training.
