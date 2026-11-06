# Automatic Betfair historical downloads

The project can download the **same full BASIC historical market files** used
by the manual archive workflow. This preserves timed exchange prices, market
definitions, runner changes and settlements. The full compressed source files
remain available for replay, including fields not yet used by the price parsers.
Daily public BSP files continue to cover the newest dates independently.

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

Open `training_app.py`, then **Data → Historical exchange data**.

1. Create a login certificate if one is not already present. On this installation
   it has already been prepared under `data/betfair/auth/`.
2. Save the **public certificate** using the button. Upload that file in
   [Betfair account security](https://myaccount.betfair.com/accountdetails/mysecurity?showAPI=1),
   under **Automated Betting Program Access**. Keep the private `.key` file local.
3. Enter your Betfair username and password in the application's connection form.
   The app checks certificate login and historical access before saving anything.
   Saved login settings use Windows account encryption and stay outside Git.

After connecting, ordinary **Scrape recent days** source refreshes also fetch
available historical markets. A successful public BSP update can coexist with
a historical backlog or publication delay; the two statuses are reported separately.
Use **Download and import missing markets** for a longer initial catch-up.
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
- `data/betfair/auth/client.crt`: public login certificate.
- `data/betfair/auth/client.key`: private login key; never upload or share this file.
- `data/betfair/auth/connection.json`: Windows-account encrypted login settings.

WIN prices update `betfair_bsp.parquet` and `betfair_bsp_early.parquet`; fuller
timing metrics update `betfair_timing.parquet`. PLACE prices update
`betfair_place_bsp.parquet`. Previously accepted non-null values and unrelated
columns remain; absent observations are filled and new runners appended.
Disagreeing known BSP prices are counted and preserved for review. Ambiguous
runner keys stop publication. New files publish atomically under the source
writer lock. A crash between outputs can be resumed idempotently.

The API requests British/Irish WIN and PLACE **market files only**. Betfair says
event and market files contain the same information at different grouping levels,
so fetching both would duplicate it. Four download workers share a five-requests
per-second limiter, below the documented 100 requests per ten seconds. Throttling
and temporary faults retry; authentication failure stops further account requests.
Corrupt local copies are quarantined and replaced.

## CLI and environment setup

The Data page is the usual setup route. An alternative is the local `.env`
configuration described in `.env.example`, using certificate paths and credentials.
A temporary `BETFAIR_SESSION_TOKEN` also works, but cannot provide lasting unattended login.

```powershell
.venv\Scripts\python.exe -m src.ingestion.betfair_history --start 2026-10-15 --end 2026-10-30 --max-files 500
```

Repeat the command to drain saved work. An incomplete run returns a nonzero
exit code and retains its checkpoint. A downloaded file is not considered
accepted until BZ2 integrity, JSON, market identity, country, type, settlement
and original requested dates have been checked. The code reuses the existing
WIN/PLACE parsers. No feature build, model fit or promotion is performed.

Local archive compatibility and account-encryption checks are recorded in
`data/audit/betfair_history_20261105/`. An authenticated download still requires
the account setup above; no Betfair credentials were present during implementation.

Verification covered 101 passing integration/regression cases, the dated-file-path
case, and a separate passing encryption test under the normal Windows user.
The restricted test process lacked profile encryption keys and skipped that one
case; the normal-user run verified the actual encryption round trip. The setup
controls also rendered successfully using Streamlit's test runner. Five existing
archive markets (WIN and PLACE, 61 runner records) passed the strict source checks.
The prepared certificate/key pair loaded successfully from both Windows execution
contexts. Targeted lint, Python parsing and documentation link checks passed.
