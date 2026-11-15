# Prize-currency correction020

The three selected prize predictors mix first-prize amounts in GBP and EUR.
The raw scraper strips the currency suffix, so a horse moving from a10,000EUR
race to a9,000GBP race appears to drop1,000 in purse. At a prior conversion
rate of0.8GBP/EUR, the comparable change would be an increase of1,000GBP.
This is a unit error in the current inputs, independent of race outcomes.

## Source audit before fitting

Early, middle and late positive-purse races were chosen at each of86 stored
venues, without selecting on results. All258 source responses retain matching
race identity/date and explicit first-prize units. Their numerical amounts
exactly match the stored amounts. Sixty venues quote GBP;26 quote EUR,
including Down Royal and Downpatrick. Country alone is therefore an unsafe
currency rule. The map is inferred from these consistent venue samples; this
is not a complete per-race currency census. Explicit response summaries,
retrieval times and hashes are retained under `data/audit/prize_currency_020`.

An initial case-sensitive cache check rejected the three Bangor-On-Dee samples
because the source spells its name with different capitalisation. Both source
and cache identity checks now use the same normalised course spelling, while
still requiring the exact numeric race ID and date. The initial audit remains
preserved. The final audit has no unresolved venues or amount disagreements.

Historical GBP-per-EUR observations were downloaded directly from the
[ECB reference-rate history](https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.xml).
There are7,123 observations from4January1999 to27October2026. Because the ECB
publishes reference rates in the afternoon, conversion uses only the latest
observation strictly before each race date, including across weekends and
holidays. Same-day and later rates cannot enter that race's predictors.
[ECB publication methodology](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html).
The original XML, narrow rate projection and receipt are frozen in
`data/research/ecb_fx_020`. These are statistical comparison units, not a claim
about historical transactions or settlement exchange rates.

## Narrow reconstruction and fixed screen

The existing prize-history formula was replayed on all753,734 runners before
conversion. Every original value in all three selected columns matches
exactly. After conversion, the479,919 rows belonging to horses with entirely
GBP-only histories remain exactly unchanged. The source contains212,512 EUR
rows. Currency conversion changes204,345 historical maximum-purse values,
230,899 historical means and257,409 purse-change values.

Twelve checks pass for the conversion direction, same-day exclusion, weekend
availability, invalid/unknown units, current-race exclusion, permutation,
and the cross-border purse-change example. Existing zero-prize, cold-start,
prior-start denominator and missing-value definitions remain unchanged to
isolate the unit correction. The12 zero-purse rows all belong to three
Newmarket Challenge Whip races; they were not blindly classified as bad data.

Only these three columns are replaced in018's frozen baseline inputs. All
other features, labels, prices, runners and column order remain identical.
The exact cached018 baseline is reused. One win-only151-input candidate uses
the same two-year training window, native calibration,500-tree cap and0.025
learning rate, then the same3,233-race Q42025 test. No complete feature-store
rebuild, baseline refit, additional quarters or automatic promotion is queued.
This remains retrospective development on an already inspected window.

Candidate source, input hashes and method are registered under
`data/audit/model_improvement/020_prize_currency/manifest.json`. Source code is
isolated from the active feature pipeline; the installed016 model remains
paired with its original histories.

## Completed prediction comparison

The single candidate completed in154seconds. On the same3,233 test races,
race log loss changes from1.849008 to1.850940 and Brier from0.08548189 to
0.08551467. The paired log-loss difference is+0.001932,95% interval
[-0.000509,+0.004234]; the date-cluster interval also spans zero. Both point
scores are worse, so the unit correction is not shortlisted for wider model
validation or installed as a new generation.

On3,125 identical market-covered fields, the original cheap baseline scores
1.850606, the currency candidate1.852421 and the same-time market1.738314.
This test has not reduced the market gap. Saving and reloading the candidate
reproduces every prediction exactly. The correct unit reconstruction and
source evidence remain available for later coherent training; the negative
screen does not make the original mixed units semantically correct.
