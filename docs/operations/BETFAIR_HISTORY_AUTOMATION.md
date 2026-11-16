# Automatic Betfair historical downloads

The project can download the **same full BASIC historical market files** used
by the manual archive workflow. This preserves timed exchange prices, market
definitions, runner changes and settlements. The full compressed source files
remain available for replay, including fields not yet used by the price parsers.
The public daily BSP downloader and manual win/place extraction commands are
retired. The default main collection plan includes this feed as `betfair`.
Recently settled markets wait for Betfair's publication window.

## Fees and access

Betfair's £499 activation fee applies to a Live App Key for live betting
applications. This downloader uses the separate Historical Data API and
certificate login; it does not require that activation. Certificate login uses
an application identifier which Betfair explicitly says is not validated as an
App Key. There are no purchase or betting operations in the downloader.

You still need the relevant Horse Racing BASIC months available in your
Betfair **My Data**. Paid data packages remain subject to their own prices.
Missing entitlements are reported; the application does not buy them.

Official references:

- [Exchange API activation fees](https://support.developer.betfair.com/hc/en-us/articles/115003864531-Are-there-any-costs-associated-with-API-access).
- [Historical download API and access requirements](https://support.developer.betfair.com/hc/en-us/articles/12859956891932-How-Can-I-Make-HTTP-Requests-to-the-Historical-Data-API).
- [Certificate login](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive%2Bbot%2Blogin).
- [Five-day publication delay](https://support.developer.betfair.com/hc/en-us/articles/360002427051-How-frequently-is-the-historical-data-updated).

## One-time setup

This installation is already connected (see the live verification below). The
setup and catch-up controls lived on the **Data** page of `training_app.py`,
which was removed with the other Streamlit code on 2026-11-07. There is no UI
for these steps now; they are Python calls.

1. Create a login certificate if one is not already present
   (`scripts/setup_betfair_history.ps1 -CreateCertificate`). On this installation
   it has already been prepared under `data/betfair/auth/`.
2. Upload the **public certificate** (`data/betfair/auth/client.crt`) in
   [Betfair account security](https://myaccount.betfair.com/accountdetails/mysecurity?showAPI=1),
   under **Automated Betting Program Access**. Keep the private `.key` file local.
3. Call `src.ingestion.betfair_history.connect_account(username, password, cert, key)`
   with the paths from `src.ingestion.betfair_credentials.certificate_paths()`.
   It checks certificate login and historical access before saving anything.
   Saved login settings use Windows account encryption and stay outside Git.

If connection fails, the result identifies the failed step: Betfair sign-in,
historical-data access, or encrypted settings storage. Recognised Betfair login
codes are paired with a specific action. Missing Horse Racing BASIC months are
reported as a successful sign-in with missing data access, with a link to My Data.
The last attempt is retained in `data/betfair/auth/connection_status.json` using
only the time, step and safe outcome message; it contains no login details or tokens.
The live service requires GET for `GetMyData` (POST returns HTTP 405). Filtered
file listings use POST; individual market downloads use GET. The initial live
connection confirmed certificate sign-in before identifying this method mismatch.

After connecting, ordinary source refreshes fetch available full historical
markets. Missing login, a catch-up backlog, and a publication delay are
reported explicitly by the main `betfair` source. For a longer catch-up run
`python -m src.ingestion.betfair_history --start YYYY-MM-DD --drain`.
It resumes saved work and skips already imported market files.

Betfair publishes historical files five days after event settlement, which can
be later than the race date. The automatic coordinator conservatively requests
through six days ago. Missing newly published events can be recovered by the
overlapping recent refresh window. A new month may need adding to My Data before
it is available to download.

## Storage and import

- `data/betfair/history/raw/BASIC/`: complete source BZ2 market files.
- `data/betfair/history/queue.sqlite`: durable download/import state, hashes,
  original request ranges and parsed observations.
- `data/betfair/history/status.json`: progress, missing months, failures and outputs.
- `data/betfair/history/backfill_status.json`: resumable catch-up worker, totals and code pins.
- `data/betfair/history/archive_payloads.sqlite`: verified, deduplicated members from existing manual archives.
- `data/betfair/history/archive_consolidation.json`: preservation and removal receipts for the original archives.
- `data/betfair/auth/client.crt`: public login certificate.
- `data/betfair/auth/client.key`: private login key; never upload or share this file.
- `data/betfair/auth/connection.json`: Windows-account encrypted login settings.

WIN prices update `betfair_bsp.parquet` and `betfair_bsp_early.parquet`; fuller
timing metrics update `betfair_timing.parquet`. PLACE prices update
`betfair_place_bsp.parquet`. Previously accepted non-null values and unrelated
columns remain; absent observations are filled and new runners appended.
Disagreeing known BSP prices are counted and preserved for review. Runners match
by exchange selection ID within the race, with a unique-name fallback when an ID
is absent. Distinct selections may share a name. Ambiguous legacy observations
and repeated accepted observations of one ID are retained in `identity_review/`
before the cache is replaced; duplicate incoming identities still stop publication.
Old PLACE parses gain IDs from their exact retained source revision during normal
publication, including records published before that field was collected. See the
[runner identity recovery](../archive/operations/BETFAIR_RUNNER_IDENTITY_2026-11-06.md).
New files publish atomically under the source
writer lock. A crash between outputs can be resumed idempotently.

The API requests British/Irish WIN and PLACE **market files only**. Betfair says
event and market files contain the same information at different grouping levels,
so fetching both would duplicate it. Four download workers share a five-requests
per-second limiter, below the documented 100 requests per ten seconds. Throttling
and temporary faults retry; authentication failure stops further account requests.
Corrupt local copies are quarantined and replaced.
Both the older `/data/xds/historic/BASIC/` and current
`/xds_nfs/edp_processed/BASIC/` remote layouts map to the same local market file;
the source path cannot choose a local folder or a different data tier.

## Existing archives and catch-up

`python -m src.ingestion.betfair_archives --remove-verified` consolidates the
existing top-level `data/betfair/*.tar` files. Every regular member is retained,
including event files and different raw revisions. Compressed payloads are stored
once by SHA-256, and original archive/member provenance is retained. Each original
is removed only after all member bodies pass a final read-back hash check, the
source archive still matches its initial hash, and a FULL SQLite commit and
checkpoint have completed. An interrupted migration resumes from its member manifest.
Incomplete tar files are preserved intact in `history/quarantine/`, recorded as
issues, and do not block consolidation of the other archives. Complete markets
already retained from them are replayable; missing markets are fetched via the API.

The main collector looks up retained payloads by market ID before making a network
download. Native contents must still match the requested race date, country, market
type and settlement checks. Archived payloads are not copied into another raw file.

Historical catch-up uses the same queue, parsers and publishing path as ordinary
refreshes. `--drain` continues bounded batches; four workers share the five-request
per-second limiter. Listing windows are checkpointed, with recent windows rechecked
after six hours. Repeated failures receive a cooldown and enter review after three
attempts. During consolidation it replays retained payloads without network
downloads; afterward it fetches the remaining gaps. It checks its code pins between
batches, and stops if `data/betfair/history/STOP` is present. It is separate from
the ATR worker and does not alter that worker's files or locks.

Import checkpoints group up to 128 parsed markets or two seconds of completed
work in one durable transaction. Complete downloaded source files are flushed
before their queue rows are accepted. Existing files and consolidated payloads
are reused without rewriting them. A small number of in-flight jobs bounds memory;
five-second progress updates use incremental counters instead of scanning the
whole queue for every file. The final report recounts committed state and includes
`import_seconds`, excluding price-cache publication. Interrupted uncommitted work
can be replayed from retained raw data. WIN/PLACE coverage, parsing, the request
limiter and SQLite durability settings are unchanged.

`scripts/benchmark_betfair_ingestion.py` (removed in the November 2026 cleanup;
at tag `pre-cleanup-2026-11`) compared retained real markets against
their accepted observations using read-only production connections and disposable
queues. `--baseline-source` can compare a trusted local source snapshot. Its timing
report distinguishes checkpoint writes, raw retention, lookup and parsing; storage
timings alone are not an end-to-end backfill estimate.

The former public data was frozen in `data/research/betfair_public_legacy/`; nothing
read it after the cleanup and it was deleted on 16 November 2026. Its volume and VWAP fields cannot be
reconstructed from BASIC, which has no traded volume; see the
[package comparison](https://historicdata.betfair.com/). No public downloads run.

## CLI and environment setup

`connect_account` (see One-time setup) is the usual setup route. An alternative is the local `.env`
configuration described in `.env.example`, using certificate paths and credentials.
A temporary `BETFAIR_SESSION_TOKEN` also works, but cannot provide lasting unattended login.

```powershell
.venv\Scripts\python.exe -m src.ingestion --sync 14
.venv\Scripts\python.exe -m src.ingestion.betfair_history --start 2020-09-15 --end 2026-10-31 --drain --max-files 5000
```

Repeat the command to drain saved work. An incomplete run returns a nonzero
exit code and retains its checkpoint. A downloaded file is not considered
accepted until BZ2 integrity, JSON, market identity, country, type, settlement
and original requested dates have been checked. The code reuses the existing
WIN/PLACE parsers. No feature build, model fit or promotion is performed.

Local archive compatibility and initial account-encryption checks are recorded in
`data/audit/betfair_history_20261105/`.

Live verification on 6 November is recorded in `data/audit/betfair_connection_20261106/`.
The certificate is uploaded and the account is connected, with BASIC access through
October 2026. The initial five-market check was followed by a successful main
refresh importing all 92 markets for 31 October with no failures. Existing
BSP price disagreements were preserved rather than overwritten; their source
observations remain in the raw files and history queue for review. Connection,
transport, UI and path checks passed (54 tests); the Windows encryption case also
passed separately under the Windows account running the app. Temporary test files
were removed after verification.

Initial implementation verification covered 101 passing integration/regression cases, the dated-file-path
case, and a separate passing encryption test under the normal Windows user.
The restricted test process lacked profile encryption keys and skipped that one
case; the normal-user run verified the actual encryption round trip. The setup
controls also rendered successfully using Streamlit's test runner. Five existing
archive markets (WIN and PLACE, 61 runner records) passed the strict source checks.
The prepared certificate/key pair loaded successfully from both Windows execution
contexts. Targeted lint, Python parsing and documentation link checks passed.
