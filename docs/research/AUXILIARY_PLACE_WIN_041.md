# Can the place model improve win probabilities?041

Completed29 October2026. This fixed proposal does not earn installation. It
reuses both existing heads; **zero tree models are fitted**. The fitted positive
place weight is exactly zero, and the remaining win-temperature adjustment
worsens the purged validation scores. The original win forecast is selected
before opening the later test. No native transfer or current-data fit follows.

The place classifier could contain useful ranking information not captured by
the win classifier. To test that incremental information, turn its **raw**
probabilities into within-field logit scores, normalize them into an auxiliary
book, and use `softmax(beta * log(win) + alpha * log(auxiliary))`. The fixed
bounds are alpha0–1 and beta0.5–1.5, with ridge0.001 toward identity. A win-only
temperature control separates sharpening from information supplied by place.
The served place probabilities are excluded because they already contain win
floors and other reconciliation. This is not another win-derived place trial.

Both heads are from the same frozen154-input December model trained through
22 December2025, SHA256
`3b593700ad21d8885de7ab5845a5664930c9b7275d7e8ad167ec7edfdf1e5f63`.
Those exact heads generate the earlier fit, validation and later test inputs.
Coefficients use1 January–21 February2026 (1,448 complete fields). Validation
uses28 February–31 March (977 fields,355 established-handicap target fields).
The separated later test is1 May–10 June (1,727 fields /16,042 runners,577 target
fields). These dates were already inspected development; no fresh confirmation
or reserved11June-onward predictive evaluation is claimed.

The joint fit converges to alpha0, beta1.062803855. The temperature control is
beta1.062803847; their negligible numerical difference carries no extra place
information. Lower log loss and runner Brier are better:

|Population|Original LL|Joint LL|Original Brier|Joint Brier|
|---|---:|---:|---:|---:|
|Validation,977 fields|1.840156|1.842319|.08967973|.08982948|
|Validation target,355 fields|1.880653|1.881153|.09783748|.09787433|
|Later test,1,727 fields|1.878711|1.881416|.08625072|.08632474|
|Later test target,577 fields|1.936083|1.937554|.09199311|.09203804|

The overall later-test date-cluster95% LL-difference interval is
[+.00069427,+.00473063], favouring the original. Identity remains the selected
forecast and therefore reproduces every original test probability exactly.
The same1,351 fresh, identical-field BetfairT-5 last-traded books retained in036
provide a descriptive market comparison; they are not executable quotes.

The native win replay is exact on all22,824 calibration runners /2,591 fields.
Poisoning current labels and prices on20 fields changes neither raw head.
Nineteen helper/optimizer checks pass. The preserved source capsule, fixed plan,
calibration slab, frozen coefficients, paired test forecasts and final receipt
are in `data/audit/auxiliary_win_041/`. Do not retune bounds/windows/subsets on
these outcomes. This closes this bounded positive-place design, not every
possible multi-task or stacking model.

This check predates042's result-source audit. Its original labels and scores
remain immutable; any future comparison with repaired results must use the
same corrected truth for both arms and recompute their causal input histories.
