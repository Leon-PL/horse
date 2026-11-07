# Documentation

Start with the [operating guide](../README.md) and
[current project structure](architecture/PROJECT_STRUCTURE.md).

| Folder or index | Contents |
|---|---|
| [Project inventory](PROJECT_INVENTORY.md) | Apps, source responsibilities and individual tools |
| [architecture/](architecture/) | Current boundaries and feature-build contracts |
| [operations/](operations/) | Collection, backfills, source audits and maintenance instructions |
| [research/](research/) | Numbered experiments, feature reviews and measured results |
| [archive/](archive/) | Earlier plans, handoffs and superseded architecture |
| [Negative results](NEGATIVE_RESULTS.md) | Evidence ledger; consult before repeating experiments |
| [Research progress](MARKET_PERFORMANCE_PROGRESS.md) | Chronology and experiment references |

For current collection work, read the [consolidation report](operations/PRE_FEATURE_CONSOLIDATION_2026-11-06.md), the [cleanup report](operations/PRE_FEATURE_CLEANUP_2026-11-05.md),
the [automatic Betfair archive setup](operations/BETFAIR_HISTORY_AUTOMATION.md),
the [Betfair import speed measurements](operations/BETFAIR_IMPORT_SPEED_2026-11-06.md),
the [Betfair runner identity recovery](operations/BETFAIR_RUNNER_IDENTITY_2026-11-06.md),
the [ATR backfill guide](operations/ATR_REBUILD_2026-11-05.md), and the
[source implementation notes](operations/STAGE1_IMPLEMENTATION_2026-11-04.md).
Historical reports describe the state at their recorded date. Use the runtime
status command for current data and model compatibility.

New ATR candidate derivations and their measured source coverage are in
[the rich-data feature report](research/ATR_RICH_FEATURES_2026-11-06.md).
The [efficiency, consistency and field-usage follow-up](research/ATR_EFFICIENCY_AND_USAGE_2026-11-06.md)
records corrected score interpretation and the full collected/used inventory.
That inventory covers the existing sectional collector. The
[jumps-data and source-gap audit](research/ATR_JUMPS_AND_SOURCE_GAPS_2026-11-06.md)
corrects the missed Jumps Data tab, verifies its separate fluency/consistency
measurements and records the tested collector's current scope.
The [five-tab collection guide](operations/ATR_TAB_COVERAGE_2026-11-06.md) supersedes
that prototype's collection status: both Jumps Data and Distance Ran are now
integrated, with per-tab catch-up, empty-template checks and navigation monitoring.
The [new-tab feature pack](research/ATR_NEW_TAB_FEATURES_2026-11-06.md) documents
the derived jumping/path predictors, point-in-time guards and frozen export.
