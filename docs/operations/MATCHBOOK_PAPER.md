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
| on start | settle earlier days whose Betfair SP file is out | `paper/<date>/settled.parquet` |
| 08:00 | whole-market snapshot (for the 08:00→11:00 move) | `snapshots/<date>/080000_0800.parquet` |
| 08:00–11:00 | with `--refresh`: results sync (3 days), store append, forecast-history extend | `logs/<date>.log` |
| 11:00 | snapshot, card forecast, paper bets | `paper/<date>/{forecast,bets,links}.parquet`, `run.json` |
| to last off | near-off quotes each minute, whole snapshot each 30 min | `snapshots/<date>/*_close.parquet` |

All under `data/matchbook/`. One command runs the day:

```powershell
.venv\Scripts\python.exe -m src.matchbook day --refresh
```

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
