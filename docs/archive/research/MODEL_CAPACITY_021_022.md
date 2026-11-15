# Model capacity021/022: improvement confirmed in two development periods

Every018 cheap fit selected essentially its whole500-tree allowance. One
fixed follow-up doubled that allowance to1,000 with every other setting
unchanged: original151 inputs, two-year fitting rows with full prior feature
histories,0.025 learning rate, seed42, two threads and native calibration.
The original500-tree baseline was reused without a new fit. No rate or
capacity grid was searched.

| Same3,233 test races | Race log loss | Runner Brier | Top-pick accuracy |
|---|---:|---:|---:|
| Original500-tree cap | 1.849008 | 0.08548189 | 30.45% |
| Selected1,000-tree cap | 1.843333 | 0.08531899 | 30.77% |

The log-loss difference is-0.005675, race-bootstrap95% interval
[-0.009027,-0.002014] and date-cluster interval[-0.009029,-0.002174].
Runner Brier improves0.00016290; its race-cluster one-sided95% upper bound
is-0.00006229. Both point scores improve, and the main uncertainty checks
support a gain. This is approximately0.31% lower log loss and0.19% lower
Brier on this development window. The saved artifact reproduces all
predictions exactly.

The main native early-stopping fit selected998 trees; its validation
cross-entropy improves from0.299422 at500 trees to a best0.298943. The
disposable calibration learner selected760 trees. Full curves and source
snapshots are preserved under `data/audit/model_improvement/021_model_capacity`.
This demonstrates an overly restrictive screening budget; it is not yet an
estimate of a change to the installed production recipe.

## Production-scale validation022

Only the shortlisted capacity change advances. Two periods are registered:
Q42025 and Q12026, totalling5,824 test races/52,905 runners. Their saved016
baselines exactly replay under the current native code. Training populations
also match:536,310 and566,442 runners after the original burn-in. Neither
baseline is refitted.

The production win cap changes from2,000 to4,000 at its original0.008
learning rate. Feature definitions, data, seed, regularisation, recency and
calibration remain fixed. Native early stopping still selects the actual
capacity, followed by a refit on all eligible historical rows. Both periods
must complete; each point log loss must improve, pooled race/date log-loss
upper95 bounds must be below zero, and pooled Brier/binary loss must not
worsen. Market comparisons keep exactly the original complete paired fields.
This is wider retrospective development validation, without a reserved
holdout or a claim of prospective market parity.

The first full-data early-stopping fit selected3,920 trees, supporting that
the original2,000-tree cap was binding. Its full Q42025 comparison completed
in1,680 seconds: log loss1.844302 ->1.839774; Brier0.08537513 ->0.08520327;
top-pick accuracy30.45% ->30.96%. The paired log-loss difference is-0.004528,
race95% interval[-0.007342,-0.001457] and date interval
[-0.007522,-0.001357]. Thus the screening gain carried over to this full-data
production-recipe comparison. The second period is running; the two-period
decision was then completed by the winter comparison below.
Source and immutable recipe are registered under
`data/audit/model_improvement/022_full_capacity/manifest.json`.

The winter comparison also improves: Q12026 log loss1.809235 ->1.803081,
Brier0.09026020 ->0.09009255 and top-pick accuracy30.14% ->30.30%.
Its selected capacity is3,984 trees. Both saved models replay their frozen
forecasts exactly. The registered two-period advancement conditions pass.

| All5,824 races /52,905 runners | Race log loss | Runner Brier | Top-pick accuracy |
|---|---:|---:|---:|
| Original production recipe | 1.828701 | 0.08748263 | 30.31% |
| Increased win capacity | 1.823450 | 0.08731258 | 30.67% |

The paired log-loss delta is-0.005251; race95% interval
[-0.007344,-0.003261], date-cluster interval[-0.007434,-0.002990].
The Brier upper95 delta is-0.00010917. On the5,594 complete fields with
the original market reference, model log loss falls1.824228 ->1.818985
versus market1.715495. This closes4.82% of the remaining gap on those
development fields. It does not establish prospective market parity.

Remaining weaknesses are retained in the record: binned ECE rises slightly
from0.003146 to0.003614 despite better proper scores; winners assigned
<=0.1% increase from2 to3, with9 winners<=1% under either model. The extra
extreme case is Powerful Glory at Ascot on2025-10-18:0.3523% ->0.0729%,
at source odds201. No probabilities are retrospectively floored to hide it.
Camino Vino, with no local history and source odds3.25, remains a much more
concerning information gap (model0.3262%). Tail review is saved separately
as `data/audit/model_improvement/022_winner_tail_review.csv`.

## Installation023 completed

`scripts/adopt_capacity_upgrade.py` requires the completed022 advancement
gate. Before a full deployment refit, it checks the new final-quarter win
component together with the saved place component trained through the same
cutoff. Shared input matrices must match, and native win predictions must
replay unchanged. Place-score non-inferiority uses race and date upper95
bounds no greater than0.0001, because the final place consistency adjustment
can depend on win probabilities.

The changed win component was refitted through2026-10-25 on667,110 runners
in71,400 races, selecting3,720 trees. The existing same-cutoff place component
and its calibration were retained. The combined Q1 place Brier check passes
both race and date non-inferiority bounds; its rounded score remains0.1746.

Run `20261028_084608_624611` was installed at2026-10-28 08:46 UTC. Both saved
probability outputs replay exactly; artifact SHA-256 is
`a5b0fc8b1f7deb7ec1af65cae0491f7f6fd439332ccbc0cea4c2c3a41db0f8b2`.
Only the default win tree cap changes2,000->4,000. The feature generation
remains `8fd1c9c690c4a608`, with the same151 inputs. The previous model and
configuration remain recoverable under023's `previous_generation` directory.

Historical validation is saved separately as run `20261028_080921_274533`.
The latest full refit has no held-out performance scores of its own; the
figures above evaluate the unchanged recipe in the two historical periods.
The model remains independent of current-race odds at inference, while
learning from the established50/50 historical market/outcome teacher.

The initial new-model forward study is preserved. The subsequent025 parser,
withdrawal and cache repairs are pinned in the new
`20261028T092600Z_source_repair_025_20261028_084608_624611` setup, first eligible
2026-10-29. No prospective comparisons or scheduled collection have been
recorded in this phase. Daily result histories advanced to2026-10-27 under
this fixed model without another fit. All158 final integration checks and
the deep readiness check pass, as does the public forecast path for a real
eight-runner Nottingham card. See [source audit025](RACE_TYPE_AUDIT_025.md).
