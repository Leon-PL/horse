# Matchbook paper record

The 11:00 rule (research 107, passed on the hold-out) followed forward on live
Matchbook prices before any money is staked. Code: `src/matchbook/`. The
workbench shows it under **Paper record**.

## No account needed

Matchbook's public feed (`api.matchbook.com/edge/rest/events`) returns every
runner's best three back and lay prices and stakes without logging in. The
recorder sends no session token, so it cannot place, cancel or see bets.
Credentials are needed only when real betting starts. When that time comes,
keep them out of git: put `MATCHBOOK_USERNAME` and `MATCHBOOK_PASSWORD` in the
repo-root `.env`, which is gitignored.

## The day

| UK time | Step | Files |
|---|---|---|
| 07:50 | settle earlier days whose Betfair SP file is out; start the `--refresh` child: results sync (3 days), store append or rebuild, forecast-history extend | `paper/<date>/settled.parquet`, `logs/<date>_refresh.log` |
| 08:00 | whole-market snapshot (for the 08:00→11:00 move) | `snapshots/<date>/080000_0800.parquet` |
| refresh done, by 10:30 at the latest | `prepare`: card, features, win model, correction inputs except price | `paper/<date>/prepared.{parquet,json}` |
| 11:00 | snapshot, price inputs, correction, paper bets (about one second) | `paper/<date>/{forecast,bets,links}.parquet`, `run.json` |
| to last off | near-off quotes each minute, whole snapshot each 30 min | `snapshots/<date>/*_close.parquet` |

All under `data/matchbook/`. One command runs the day:

```powershell
.venv\Scripts\python.exe -m src.matchbook day --refresh
```

Timings measured on 15 November (after research 109):

| Step | Time |
|---|---|
| 14-day results sync | 35 min |
| Full store rebuild | 15–16.5 min |
| Card forecast before research 109, cold caches | up to 27 min |
| `prepare` (card, features, win model, correction inputs) | about 3 min |
| 11:00 step after `prepare` | 1.1 s |

Nothing slow waits for 11:00, so the bet prices are the 11:00 quotes. A runner
withdrawn between `prepare` and 11:00 is dropped, and its race is
renormalised. `run.json` counts these runners in `withdrawn_since_prepared`.

It must run outside the Claude app (a process started from the app stops when
the app closes): use a Windows scheduled task starting at 07:50.

## Forecast

The win model (`config.json` `win_run`) and the correction fitted by
`python -m src.matchbook fit --as-of DATE` (`data/matchbook/model`). That is the
production correction groups plus six 11:00 price inputs. Research 107's two
Betfair trade-activity inputs are dropped because Matchbook has no trades
(research 108 measures what that costs on development races). The model is
fitted on Betfair last traded prices. It is served on the geometric middle of
Matchbook's best back and lay. Where the lay is more than 25% above the back,
it uses the best back instead: a middle of back 6.0 and lay 220 would fake an
edge. Each race records `race_tight`, the share of its field quoted within
25%. The summary reports bets in races at 80% or more separately, because thin
books make the 11:00 share unreliable. The evening before the first run,
120 of 155 runners had wider spreads.

## Rule and settlement

- **Selection:** `value_bet_selection` at base threshold 0.02 against the
  race-normalised 11:00 Matchbook share.
- **Races:** every card runner linked and backable, off after 11:00.
- **Price:** best back at most 51.
- **Taken price:** the best back at 11:00. A £10 fill is walked down three levels.
- **Close and result:** Betfair's public SP file, published the next morning.
  CLV is taken / BSP, read as geometric mean and median (`src.clv`).
- **ROI:** after 2% commission, races resampled.
- **Non-runners:** void. Reduction factors are not applied to the paper price.

## Reading it

Judge it by geometric CLV against BSP first, then ROI. Expect little
volume: Matchbook stakes at 11:00 are often a few pounds. The fill rate shows
whether the edge is reachable at size.

Five bets a day take a month or more to say anything. Settlement therefore also
keeps Betfair SP and the result for every forecast runner (`field.parquet`,
settled apart from the bets, so a fault there never changes a bet). The
summary's `field` block and the workbench **Whole field** panel read it on
every eligible race:

- log loss per race of the win model, the corrected forecast, the 11:00
  Matchbook share and the Betfair SP book, on the same races;
- `forecast_minus_sp`: below zero means the forecast beat the close;
- `edge_result_corr`: correlation of (forecast − 11:00 share) with
  (won − SP share), the orthogonality test, which should stay above zero;
- CLV (taken / BSP) for every runner the forecast rated above and below the
  11:00 share, not just the bets.
