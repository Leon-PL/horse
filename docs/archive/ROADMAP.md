# Roadmap after the October 2026 audit

**Historical decision document, 13 October 2026.** Its diagnosis and priorities
belong to that checkpoint. See the [operating guide](../../README.md) for the
current workflow and the [implementation record](../MARKET_PERFORMANCE_PROGRESS.md)
for later evidence.

*Written 2026-10-13 for the operator, to decide what this project should be
next. Every number here is recorded, with its method and caveats, in
[NEGATIVE_RESULTS.md](../NEGATIVE_RESULTS.md) (sections 13-16 are this
session); the decision gates are [ACTION_PLAN.md](ACTION_PLAN.md) section 7.*

## The short version

**There is no edge in this project's public data, and the two approaches the
plan held back for last have now been measured and found empty.** Training on
the market's residual learns nothing the price lacks, and the money-flow data
that is free to obtain is fully absorbed by the closing price. That is
**decision gate 4**, and gate 2 fired alongside it: the model already extracts
what its features hold (a tuned GBM ties logistic regression and both trail
the market by five points of top-1). Continuing is now a hobby, not an
investment, unless you choose to buy the one dataset that remains untested
(section 3).

What changed this session is that the answer can be trusted. Before measuring
anything, the instrument was repaired: 543 races the scraper had silently
dropped were restored; a look-ahead leak in the `sex` column, which had
flattered every model trained here, was found and fixed; the status check
now verifies the data underneath the feature store, not just the code that
built it; and live picks, which had been pricing horses with their history
silently erased whenever racecards were freshly scraped, now see the features
the model was trained on. A pre-registered golden hold-out (races from 2026-06-11) now exists
for any future claim.

## 1. What was measured

| idea | decisive measurement | verdict |
|---|---|---|
| Market as a fixed offset (plan 4c), the best-motivated untried idea | race log loss vs the market **-0.00010 [-0.00040, +0.00020]** over 16 purged quarters; every selection rule's ROI interval spans zero, every median CLV < 1 | **fails** |
| Morning money vs the close (plan 5a, free data) | excess morning volume vs the result beyond BSP: partial corr **+0.0004** (t 0.4, 574k runners) | **the close absorbs it** |
| Following the money, struck at T-5 | median CLV 0.9958 [0.9939, 0.9980] vs 0.9811 for the matched band; ROI unchanged | real, too small |
| Following the money, struck at 11:00 (honest morning) | recorded as median CLV 1.0000, ROI +0.85%; **with later non-runner reductions (ledger 20): 0.9849 [0.9822, 0.9877], ROI -1.84% [-3.75, +0.12]** | **loses to the close**, fails the bar |
| Sibling-form debutant lead, on the hold-out | partial corr -0.016, p 0.19 (discovery had -0.064) | **fails** |
| `atr_closing_in_hand`, the one sectional signal that survived triage | alone in the offset model: race log loss vs market -0.00000 [-0.00040, +0.00030] | **fails** |
| GBM vs logistic vs market (plans 3a/4b) | 24,405 races: top-1 GBM 0.293, logistic 0.289, market **0.340**; loosening the GBM makes it worse | the model is not the problem; the features are (gate 2) |
| Earlier sessions | market combination -3.55% [-5.42, -1.55]; model disagreements uninformative; 43 never-allowlisted families and 20 of 21 sectional signals already priced | dead |

The `sex` leak deserves its own line. It made one run of plan 4c look like
**+11.6% at BSP**, the third phantom edge in this project's history and the
third caught by the same tripwire: high ROI with closing-line value at or
below 1.0. Every refutation recorded before it still stands; the leak only
made the model look better than it was.

## 2. The realistic ceiling

The market at five minutes before the off, and certainly at BSP, contains
everything this project can assemble from public data. That includes form,
ratings, pedigree, sectionals, weather and going, and now the morning money
too. The fitted market combination put the price at about twenty times as
informative as the model, and the residual model cannot find a measurable
correction to it.

For a strategy settled at BSP on public data, the realistic ceiling is
therefore about **minus the commission, roughly -2%**. An edge, if one exists,
would have to come from somewhere this project has no access to:

- **Timing**: taking a price before information is absorbed. Measured: worth
  about 5 CLV points over the matched band at 11:00, which still loses to the
  close by 1.5 points once later non-runner reductions are counted (ledger
  20), and it is gone by an hour before the off. A learned timing policy
  saves 0.1-0.16 CLV points per bet (ledger 20). Morning liquidity is thin (median runner: about
  GBP 345 matched all morning).
- **Private information** (gallops, stable intent): not obtainable, and the
  part of the market the model loses to (favourites, debutants).
- **Structural value** that needs no model: bookmaker each-way terms and
  extra-place offers. Genuine, but a different activity, and capped by
  account restrictions.

## 3. The remaining levers, ranked by expected value per unit of cost

1. **Paid microstructure: Betfair ADVANCED or PRO historical data, then the
   live Stream API.** The only untested data hypothesis. ADVANCED gives
   1-second prices, the top three ladder levels and traded volume; PRO gives
   50 ms ticks and the full ladder. Prices are shown only in the account's
   purchase flow. Live use needs a live application key (GBP 299 one-off for
   personal use) plus engineering for streaming and fast execution.
   *Expected value: low.* The free version showed the information in money
   flow is already priced by the close; what remains is a short-horizon
   order-book game against professional bots, which is a different discipline
   from form modelling. *If you try it:* buy one or two months first, and fix
   the success test before looking - median CLV > 1.0 with a race-clustered
   interval excluding 1.0, on a walk-forward plus the hold-out.
2. **Stalls position and rail movements (plan 5a-bis).** The data is now
   collected for every GB fixture since 2020 (`data/processed/bha_going.parquet`).
   Turning it into a draw signal is a few days' work: parse the free text
   by distance and map stall numbers to the rail per course. *Expected value:
   low* - every racecard prints the stalls position, and draw features score
   at most 0.0023 against the market.
3. **Commercial GPS tracking (Total Performance Data, plan 5c).** Terms are
   negotiated, not published. *Expected value: very low* - the public sectional
   version of the same data is 20/21 priced.
4. **Staking and selection (plan 6).** Worth nothing without a positive
   closing-line edge; cheap to build once there is one.
5. **Jurisdiction expansion (plan 5d).** See section 5.

## 4. What stopping looks like

Gate 4 says that if residual training and microstructure both fail, there is
no public-data edge to find. Both have failed in the forms that are free to
test. The honest options:

- **A. Stop investing.** Keep the pipeline running as a forecasting hobby:
  a daily `python -m src.database --sync 14` and a store append cost minutes.
  The instrument, the guards and the ledger keep any future idea honest.
- **B. One bounded experiment with paid data.** A fixed budget, one or two
  months of ADVANCED data, and the success test above written down before
  the data arrives. Stop if it fails.
- **C. Change the activity.** Structural value (each-way terms, extra places)
  is real but needs no model and does not scale past account limits.

What would reopen the question: a result that clears ACTION_PLAN section 7,
meaning median CLV above 1.0 with a race-clustered interval excluding 1.0, on
a locked walk-forward, holding on the golden hold-out. Nothing in this project
has come close without a leak behind it.

## 5. Jurisdiction or market expansion (plan 5d)

The one public signal GB/IRE racing lacks is run-to-run bodyweight. Hong Kong
and Japan publish it, but both are pari-mutuel markets with far heavier costs:
Hong Kong's win takeout was historically 17.5% (12% duty plus 5.5%
commission, with a 10% losing rebate for large bettors), and Japan's is about
20% (background knowledge, not sourced here). Against Betfair's 2% on net
winnings, an edge would need to be roughly ten times larger before costs.
Both markets are also deep and syndicate-dominated.

Doing it means new scrapers, merge keys, track configuration and rating
warm-up, then this whole audit again: months of work, and a decision about
what the project is, not a backlog item.

## 6. Actions only you can take

1. ~~Download Betfair BASIC files for 2026-06-11 to today.~~ **Done
   2026-10-15** to 2026-10-10 (`data/betfair/2026-06.tar`, `2026-07to10.tar`;
   in `data/processed/betfair_timing.parquet`). Keep adding months: the
   frozen 11:00 rule (ledger 20) needs ~600 unseen strikes, about five
   quarters, before its one hold-out test is worth spending.
2. **Decide on paid ADVANCED/PRO data**, if you choose option B.
3. **Know where things live:** `docs/archive/SESSION_LOG.md` for this session's state;
   `scripts/project_status.py` before trusting any number (it now checks the
   data, the sectional cache and the store's freshness, not just code).

## 7. Addendum 2026-10-14: the five unconventional ideas (EXPLORATION_PLAN)

| idea | outcome | ledger |
|---|---|---|
| B. Execution timing | **fails** the pre-registered bar (saves 0.10-0.16 CLV points per bet vs 0.5). Found that ledgers 14 and 19 ignored non-runner reductions: the 11:00 "break-even" is median CLV **0.985**. A descriptive 11:00 lead (median CLV ~1.03-1.05 after a tick, survives staleness, morning-cut-off and bounce checks) is frozen for a future test | 20 |
| A. Tote exotics from BSP | **blocked at step 0**: Sporting Life has every dividend since 2020 but its terms prohibit scraping; ATR rate-limited | 24 |
| C. Forecast ground changes | **fails**: continuation partial +0.006 (p 0.04) vs the 0.01 bar; the economic rule loses | 23 |
| D. Breeze-up times | **stopped at step 0**: no official European breeze-up times exist | 21 |
| E. Morning market-making | **phase 1 does not stop it**: break-even half-spread 0.7 ticks (balanced flow) to 1.2-1.7 (informed flow); GBP 303k a day of morning volume; whether a newcomer gets filled is unknowable without an order book | 22 |

Decisions only the operator can make, added by this round:
4. **Sporting Life dividends** for idea A: permission to collect them (about
   80k pages, a day at one request a second), or a licensed source.
5. **A live Betfair app key and a few weeks of Stream API order-book
   recording** (no orders). One recording serves both E phase 2 and the 11:00
   timing lead (it shows how much was available at the price, which BASIC
   cannot).
