# Meta learner experiments

Experiments use `src.modelling` and the shared `src.wf_eval` fold owner.
The supported benchmark/selection commands are documented in
`docs/operations/MODELLING.md`; registrations and outputs go under
`data/research/meta_learner/`.

`text_probe.py` is a small retrospective experiment over a completed generalist
OOF benchmark. It compares the generalist with a pool that also receives the
stored text-only probability distribution. Its earlier fitting and later
scoring dates must be declared before scoring, with a purge gap. It does not
refit a base model or treat the text as new information absent from the
generalist. Missing text masks the entire text expert for that race.

`matrix_smoke.py` checks the explicit matrix fit/predict route on a prepared
real-data slice and records sampled process memory. It is a software/resource
check, not a feature-selection result or an accuracy comparison. The completed
baseline, text probe and wide-input check are recorded in
`docs/architecture/MODELLING_IMPLEMENTATION_2026-11-08.md`.

`performance_probe.py prepare` verifies the accepted history and prepares the
fixed reference recipe plus margin labels. `run` fits the explicit performance
head at native capacity and evaluates its pool on the registered later period.
It reuses the intact archived generalist measurement without restamping it as
current code. See `docs/research/PERFORMANCE_SPREAD_REPRODUCTION_2026-11-08.md`.

`verify_performance.py` reloads that saved component, checks forecast and field
alignment, compares finer numerical integration and writes artifact digests.
It performs no fitting or model selection.

`text_improvement.py` owns the accepted raw-text corpus, shared-fold evaluation
and verified sparse-encoding reuse. `text_development.py` records the amended
19-recipe screen; `text_tuning.py` refines target blend and regularisation using
earlier quarters, then locks a recipe before the four later quarters are
scored. `text_ensemble_probe.py` measures that recipe alongside the saved
generalist/performance forecasts. `verify_text.py` replays saved artifacts and
checks calibration, field coverage, order and inference without labels/prices.
`verify_text_inputs.py` independently checks cached attribution and coverage
against the accepted raw corpus.
`fit_text_candidate.py` refits the locked recipe through an explicit settled
date against the same accepted sources, saving a separate unpromoted artifact.
See `docs/research/TEXT_MODEL_IMPROVEMENTS_2026-11-08.md` for commands, results
and the limits of the historical comparison.

`screening.py`, `screen_candidates.py` and `pool_average.py` screen candidate
inputs and components on the stored full 063 forecasts of
`first_ensemble_20261109` without refitting the generalist: a Tier 0 offset
logit, a Tier 1 regularised LightGBM correction judged against twins whose
inputs are shuffled within quarter, and log-pools of cached components with the
three-seed average. `candidate_features.py` computes relatives' form and
same-day bias from store history. Stages: `validate` replays known results,
`candidates` screens each group, `followup` re-checks survivors against three
twins, `stack` scores the current stack. `specialists.py` saves separate
out-of-fold corrections, fits a standalone tracking model on 2023+ rows and
compares the ways of combining them. `ability.py` builds the new-information
candidates (ability in pounds by Kalman filter, relatives' ratings, going
withdrawal comparisons) and `new_information.py` scores a group's correction,
with three twin corrections, summed onto the stack. `forecast_components.py`
registers and runs the plan's untested targets: a head-to-head (Bradley-Terry)
component, a jumps completion model, and improvement and bounce inputs
(built in `ability.py`). See
`docs/research/CHEAP_SCREENS_2026-11-09.md`.

`tier2_inputs.py` refits full 063 with a screened group among its inputs, three
seeds per arm, against the benchmark's cached three seeds (`fit --arm NAME`,
then `report`). The ability and same-day arm failed its gate on 10 November;
see `docs/research/TIER2_INPUTS_2026-11-10.md`.

Do not fit combination weights on base-model training predictions. Existing
2024–2026 results and feature lists have informed previous choices, including
the formerly reserved holdout. Label their results retrospective. Keep the
existing holdout exclusion; fresh confirmation starts only after locking the
new recipe/update policy. This folder is never imported by the pipeline.
