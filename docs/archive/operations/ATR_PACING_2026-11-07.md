# ATR pacing trial — 7 November 2026

The continuous ATR backfill resumed at **17:01 UTC** with
`--request-delay 1.5`, following a successful 50-race trial. The previous worker
was stopped through its `STOP` signal and flushed all daily exports before the
trial began. Only the runtime request-delay argument changed.

## Measured result

| Measure | Result |
|---|---:|
| Previous hour, at a two-second delay | 284 successful races/hour |
| Trial, at a 1.5-second delay | 344.19 successful races/hour |
| Throughput improvement | 21.19% |
| Trial duration, including startup and final exports | 522.97 seconds |
| Successful races | 50 of 50 |
| Saved runner rows added | 428 |
| Live fragment requests | 250 |
| Retained responses reused | 0 |
| New retries, review failures or global backoff | 0 |

All five performance tabs were checked for every race. Each receipt was terminal
(`data` or confirmed `empty`). Forty-five races had complete runner coverage and
five had validated partial coverage. The trial logged no warnings or errors,
flushed its exports, and increased saved rows from 55,522 to 55,950. All five
pinned collector files retained their existing hashes. Parsing, identity checks,
publication delays and provider backoff were unchanged.

This compares different race samples, rather than replaying the same races at
both speeds. The 50-race result supports retaining the faster setting; it does
not establish sustained throughput across the whole remaining history.

At qualification, 21,283 known-ID races remained. The measured rate implies
about **62 more hours**, compared with 75 hours at the previous baseline. This
excludes 34,197 races without native IDs and assumes continued source access.

## Current operation

Resume with the qualified setting only after checking that no worker is running:

```powershell
.venv\Scripts\python.exe -u -m src.ingestion.atr_rebuild --request-delay 1.5
```

The actual launch and pacing setting are recorded in
`data/atr_cache/rebuild_v2/launch_options.json`. The live worker PID and progress
are in `status.json` in the same directory. The worker retains its existing
global rate-limit and discovery cooldown behaviour.

Post-launch verification confirmed continued successful collection with unchanged
collector hashes and no new rate-limit or challenge response. A previously
failing endpoint returned HTTP 500; the existing transient backoff paused
collection and it then resumed. This observation is recorded in `post_launch.json`.

Evidence is under `data/audit/atr_pacing_20261107/`: the previous launch and
stopped status, baseline, bounded trial logs, per-race qualification and
`continuous_launch.json`. `qualification.json` records the checks and hashes.
