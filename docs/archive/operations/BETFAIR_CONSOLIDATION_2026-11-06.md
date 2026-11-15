# Betfair collection consolidation — 6 November 2026

Full Historical BASIC WIN/PLACE markets are the default `betfair` source in
`python -m src.ingestion --sync 14`. There is one queue, one set of stream parsers
and one publication path for routine refreshes and historical catch-up.
The shared importer now uses grouped durable saves and lighter progress counting;
see [speed measurements](BETFAIR_IMPORT_SPEED_2026-11-06.md). Both market types
and all existing fields remain included.

## Removed and retained

- Removed the public BSP downloader, its wrapper command, and the separate manual
  win/place archive extraction commands. Public-only merge/receipt logic is gone.
- Moved 4,472 public source files and their 45 MB parsed dataset into
  `data/research/betfair_public_legacy/`. Research and field-repair readers use that
  frozen location. BASIC does not contain its volume/VWAP fields.
- Preserved the canonical BSP, timed-price and place-price caches. Incoming data
  fills missing values; disagreements with accepted prices remain available for
  review in the retained raw streams and queue observations.
- The 17 manual tar files total 10,099,180,032 bytes. A resumable consolidation
  retains all regular members, including event files and raw revisions, once by
  SHA-256. Original files are removed individually after read-back verification,
  an unchanged source hash, and a completed durable checkpoint.

## Running work

The main-process live refresh for 31 October completed successfully: 92 markets
published, including 87 new downloads after the earlier five-market verification,
with zero failures. This exercised the actual coordinator and account connection.

The historical catch-up range is 15 September 2020–31 October 2026. It replays
retained archive payloads during consolidation and requests remaining gaps after
consolidation finishes. Four workers share the existing request limiter. It
rechecks recent listings, checkpoints older listings, applies retry cooldowns,
and routes persistent failures to review. It checks collector code pins between
batches and honours `data/betfair/history/STOP`.

Read current state from:

- `data/betfair/history/archive_import_status.json`
- `data/betfair/history/archive_consolidation.json`
- `data/betfair/history/backfill_status.json`
- `data/betfair/history/status.json`

Only verified originals are removed; an in-progress consolidation does not mean
all 17 tar files have already been deleted. The independent ATR worker is retained.

The old `2020.tar` contains an incomplete final member. Complete retained members
remain reusable; the original is moved intact to `history/quarantine/` with its
hash preserved, and consolidation continues with the other archives. Missing
markets are subsequently recovered through the authenticated API. Quarantined
originals are separate from verified originals removed after deduplication.

## Validation and evidence

The core integration run passed 103 tests, with the restricted Windows encryption
case skipped; that case and real saved-account authentication were verified under
the normal Windows account. Additional focused checks cover the default CLI source
plan, durable checkpoints, corrupted-copy retention, raw path safety, overlapping
archives, and local replay without duplicate downloads. Reports, moves, source
hashes and launch records are under `data/audit/betfair_consolidation_20261106/`.

The provider publishes historical markets five days after settlement; the source
plan conservatively waits six calendar days. Available account packages currently
extend through October 2026. See [setup and operation](BETFAIR_HISTORY_AUTOMATION.md)
for the canonical commands and preserved research data.
