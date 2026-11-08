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

Do not fit combination weights on base-model training predictions. Existing
2024–2026 results and feature lists have informed previous choices, including
the formerly reserved holdout. Label their results retrospective. Keep the
existing holdout exclusion; fresh confirmation starts only after locking the
new recipe/update policy. This folder is never imported by the pipeline.
