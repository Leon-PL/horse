# Calibration choice019: completed development screen

The018 repair model improved slightly before isotonic calibration and worsened
after it. This follow-up selected a calibration family on earlier training
history, retaining the same final trees and all original test predictions.

One disposable500-tree native calibration learner was replayed from the frozen
018 inputs. Its fitted Platt coefficients matched within1e-12 and its isotonic
arrays matched exactly. It supplied47,277 out-of-sample predictions in5,120
training-period races,16May–24September2025. Six tests passed for probability
distributions, date separation and the decision gate.

Calibration maps were fitted on3,362 races through8August. After a seven-day
gap,1,556 races from15August selected among the existing native chain,
Platt-only and race temperature. An alternative needed race- and date-cluster
log-loss95% upper bounds below zero, with no worse Brier or binary log loss.
Both alternatives passed; temperature had the lowest winner log loss. The
chosen map was refitted on the whole earlier calibration slab, then applied
to the saved repaired151-input final trees. No final tree refit or additional
quarter was run. Saved-artifact replay error is0.

| Q4 development test,3,233 races | Race log loss | Runner Brier |
|---|---:|---:|
| Original018 matched baseline | 1.849008 | 0.08548189 |
| Repaired018 native calibration | 1.851337 | 0.08547510 |
| Repairs plus training-selected temperature | 1.849218 | 0.08546720 |

Temperature reduces the repaired model's log loss by0.002120. Its race-bootstrap
95% interval is[-0.004391,-0.000037], while the date-cluster interval
[-0.004313,+0.000011] just spans zero. Against the original matched baseline,
the difference is+0.000210, interval[-0.002949,+0.003123]. This has not
demonstrated an overall improvement over the starting recipe.

Production remains016. The019 hypothesis was motivated by an already inspected
development window; these results are not fresh confirmation. No additional
quarters are queued. The code, exact saved artifact, training-only predictions,
selection report and source snapshots are sealed under
`data/audit/model_improvement/019_calibration_choice/`.
