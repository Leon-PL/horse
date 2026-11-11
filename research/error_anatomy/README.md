# Error anatomy

Starts from where the stored full 063 forecasts are wrong, then builds inputs
from those errors. Report: `docs/research/ERROR_ANATOMY_2026-11-10.md`.
Outputs: `data/research/error_anatomy_20261110/`.

- `frame.py`: scored runners with the model's forecast, BSP, SP and race
  context.
- `extend_oof.py`: seed-42 out-of-fold forecasts for 2022Q3–2024Q2, so the
  inputs below have history before the first trained correction.
- `entity_residuals.py`: each horse's, trainer's, jockey's and sire's past
  results and closing prices against the model's own forecast, on strictly
  earlier dates. Stages: `screen` (first definition), `long` (long memory,
  four groups against twins), `stack` (each group on the 9 November stack
  with three twins), `combine` (summed against fitted weights), `stress`
  (inputs rebuilt with a 7 and 30 day lag).

It reuses the cheap-screen harness in `research/meta_learner` and the
benchmark's cached forecasts. Results are retrospective.

Second round (11 November):

- `market_disagreement.py`: which inputs predict the market's disagreement
  with the model.
- `soft_corrections.py`: corrections taught by past closing prices (`screen`,
  `stack`).
- `native_check.py`: every correction applied to the one native-capacity
  forecast, to separate information from capacity repair.
- `second_pass.py`: the error anatomy repeated on the corrected forecast.
- `franking.py`: how a horse's past rivals have run since.
- `sale_inputs.py`: pre-racing auction prices (null).
- `holdout.py`, `holdout_score.py`: the registered single read on races from
  11 June 2026 (`prepare`, `fit`, `rehearse`, `register`, `score`). It has
  been scored; `score` refuses to run again.

Third round (11 November):

- `third_pass.py`: what the corrected forecast still gets wrong, by runner
  type, and whether a horse's disagreement with the market still repeats.
- `horse_state.py`: the horse's errors run by run (`horse-state`) and stable
  form with 30 and 90 day memory (`entity-recent`), screened through
  `soft_corrections.py`.
- `market_history.py`: the same price history without the model
  (`market-level`) and past late price moves (`market-moves`).
- `joint_corrector.py`: one correction over all the error inputs against the
  weighted sum of one per group (a tie).
- `holdout_state.py`: the second registered read on the same hold-out races,
  for what the two `horse_state` groups add to the first read's seven
  (`rehearse`, `register`, `score`). Scored; the increment did not pass.

Input audit (11 November):

- `card_audit.py`: prints a race as a card, with each horse's earlier runs
  and the raw values behind the main inputs, for reading by eye.
- `card_vs_result.py`: cached pre-race racecards against stored results,
  field by field.
- `hidden_form.py`: the card's form figures for runs the database lacks
  (`hidden-form`), screened through `soft_corrections.py`.
