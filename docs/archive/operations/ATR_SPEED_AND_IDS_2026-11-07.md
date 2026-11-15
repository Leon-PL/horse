# ATR speed and native ID repair — 7 November 2026

The historical worker now saves every exact ID match from a daily results page,
loads saved daily indices on restart, and recovers pending daily exports after a
shutdown. Source parsing, all five performance tabs and runner validation remain
unchanged. Initial qualification used two seconds plus jitter. The later
[50-race pacing trial](ATR_PACING_2026-11-07.md) qualified 1.5 seconds and resumed
continuous collection at that setting.

## Identity findings

The stopped queue contained 62,432 races, 43,857 saved runner records and 34,247
races without a native ID. Eleven native IDs appeared on two canonical races
each; each pair was exactly one hour apart. Legacy time fallback had allowed
the wrong summer-time race to supply an ID hint.

The offline audit compares complete canonical runner lists against the legacy
cache on the same date and course. It supports 21,471 hints and proposes clearing
13 conflicting assignments. These are lookup hints, not accepted observations.
Already accepted fields are protected. New observations must still pass source
date/local-time, runner identity and five-tab completion checks.

Two public daily pages were read in the interactive browser: 15 September 2020
and 4 November 2026. Their explicitly observed UK article/link pairs contain 65
IDs, of which 63 were missing. They are retained as partial rendered-link receipts
with their own capture hashes; they are not represented as full raw HTML captures.
IDs are never inferred from numeric sequences or matched to a neighbouring time.

The missing-ID breakdown depends on the date boundary: 29,651 races are before
1 August 2023, 3,939 are in June–November 2026, and 657 are elsewhere. Including
1–9 August in the first group gives the previously quoted 30,166 / 3,939 / 142.

## Access remains a separate limitation

The public pages load in the user's normal browser and the interactive browser.
Direct HTTP, a separate unattended browser, a separate normal browser window,
and one mobile-page check still received Client Challenge responses. No challenge
was solved automatically. The unsuccessful browser prototype and its added
dependencies were removed rather than made the default collector.

The latest probe's one-hour discovery cooldown is preserved. Known native IDs
can continue through the working public AJAX transport. Actual HTTP 429 limits
apply globally and Retry-After is honoured. An HTTP-200 identity-page challenge
pauses discovery without suppressing known-ID collection. Neither a challenge nor
an unrecognised page establishes permanent absence.

The remaining historical gaps therefore still need working unattended access or
additional source-backed saved indices. The main worker imports saved receipts
automatically and retries discovery after cooldown. Reading two days successfully
does not establish access to all missing days or availability of old telemetry.

## Performance changes

- A daily lookup saves all exact canonical ID matches immediately, even if the
  first sectional request fails. Saved indices are forwarded through the replay
  client; previously that wrapper hid the index directory.
- Daily discovery alternates newest and oldest missing dates, with at most one
  scheduled lookup per minute and provider backoff. Incomplete cached days are
  deferred so they cannot starve other dates.
- Separate partial queue indices remove repeated full ordering of active jobs.
  A native-ID index also avoids full scans while checking ownership.
- Status is written at most every ten seconds during normal work. Daily Parquet
  exports are grouped at 50 races or three minutes, with a durable SQLite dirty-day
  queue. Every accepted field still commits immediately.
- A later retained challenge cannot hide an older successful raw response for the
  same URL. Valid retained fragments are reparsed instead of downloaded again.

The read-only 62,432-job benchmark selected exactly the same races as the old
query. Median queue selection fell from 53–84 milliseconds to 15–19 microseconds.
This measures local queue work only; it is not an end-to-end scraper speedup.
Network pacing, five-tab coverage and source availability still dominate duration.

## Operation and evidence

Resume one worker, after checking its heartbeat and locks:

```powershell
.venv\Scripts\python.exe -u -m src.ingestion.atr_rebuild --request-delay 1.5
```

Read `data/atr_cache/rebuild_v2/status.json` for current progress. Do not launch a
second collector or delete SQLite sidecars or lock files. The fingerprint now pins
five collection modules: `attheraces.py`, `atr_rebuild.py`, `atr_tabs.py`,
`atr_jumps.py` and `atr_ids.py`. Stop and review before changing them. Feature
derivation remains separate from this raw collector.

Reusable read-only tools:

```powershell
.venv\Scripts\python.exe -B -m src.ingestion.atr_ids --out data/audit/atr_id_audit.json
.venv\Scripts\python.exe -B -m scripts.benchmark_atr_queue --out data/audit/atr_queue_benchmark.json
```

Evidence is under `data/audit/atr_speed_ids_20261107/`: consistent queue/raw/cache
backups, the complete ID plan, access probes, source equivalence, queue timings,
test XML and the reviewed adoption receipt. The 138-test collection suite passed;
29 queue/identity tests passed again after adding the native-ID index. The adoption
receipt records exact before/after runner-payload digests and source fingerprints.
Live qualification and launch receipts record the actual subsequent worker result.

The deployment preserved all 43,857 saved payloads with the same digest
`0d94375b8d165d0915df70d01effcd61202cc8800988f254b53813ba47c0faeb` and the
unchanged accepted cache. Duplicate IDs fell from eleven to zero. Clearing 13
conflicts and recovering 63 missing IDs leaves 34,197 missing IDs. One interrupted
daily export was queued for recovery.

The ten-race live check, 11:53–11:55 UTC, accepted all ten fields and 117 runner
records in 121.63 seconds: about 296 races/hour. It checked all 50 performance
tabs, retained 49 new live responses, and reused one saved fragment, including
races whose IDs had just been
recovered. It reached 5,504 accepted fields and 43,974 saved runner records, with
no new retries or reviews, and flushed all pending exports. This short, different
race sample is not a controlled throughput comparison or a full-backfill ETA.
The continuous worker was then relaunched; consult `launch.json` and live status.

## Importing saved results pages

The 9 November check found that ATR's daily results pages for 16 September 2020
and 3 November 2026 loaded in the browser while unattended discovery received a
challenge. The rendered race links matched 58 previously missing queue entries
by exact date, course and UK local time.

Save a dated ATR results page from the browser as HTML, then import it. You can
pass several saved files in the same command.

```powershell
.venv\Scripts\python.exe -m src.ingestion.atr_ids --import-results 'C:\path\to\saved-results.html'
```

The importer uses the page's canonical ATR date and the native ID linked within
each result article. It saves a dated receipt with a hash of the input file. The
running historical worker reads new receipts within about a minute, matches them
to canonical jobs, and fetches eligible fields at its configured pace. A saved
ID identifies a race. Availability of sectional telemetry is determined by its
performance tabs.

The worker now gives fresh known-ID fields priority over retries. An identical
HTTP 500 response repeated at the same race endpoint defers that field for a day
when ATR supplies no Retry-After. The Distance Ran parser also recognises empty
split placeholders and keeps measurements for other runners in the same tab.

The 9 November deployment passed 145 ATR checks. Replaying retained responses
accepted 27 new fields and completed five fields with earlier partial data,
adding 236 runner records. After restart at 1.5 seconds plus jitter, the worker
imported all 58 exact IDs. The first live check saved nine of the newly mapped
November fields with all five tab outcomes recorded. The September 2020 fields
entered the queue for collection. Verification and the remaining date list are
under `data/audit/atr_recovery_20261109/`.
