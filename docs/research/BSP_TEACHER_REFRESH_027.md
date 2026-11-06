# Expanded historical exchange-price teaching data

The normal win model learns from actual winners and a 50% blend of historical
market probabilities. Its default exchange-price cache stopped on 10 June
2026. Later races used the existing whole-field bookmaker starting-price
fallback, even though newer exchange observations were already stored in
other project files.

The default cache now incorporates the existing BASIC timing observations,
the public archive, and 30 additional daily GB/Irish files through 27 October.
The new files came from [Betfair's public directory](https://promo.betfair.com/betfairsp/prices).
Their file date is the day after racing; their local UK off-times are
converted to UTC by the existing tested parser.

The update keeps every original row and price. It appends 69,257 observations,
bringing the table from 746,812 to 816,069 rows. Within the actual current
training population, complete exchange-price fields rise from 63,977 to
71,302 of 71,477 races: **89.5% → 99.8%**, or 7,325 additional complete fields.
Donor fields with contradictory independent winner identities are
quarantined. No sporting result labels are overwritten.

Genuine timed prices are retained from BASIC. Public rows carry their BSP;
their unavailable timed-price columns remain missing. A volume-weighted price
is never substituted for a price at a particular time. Extra public volume
and in-play fields are excluded from the standard price table. These prices
serve as historical teaching targets. Forecasts use the same 151 pre-race
inputs.

The installed cache is the ordinary `data/processed/betfair_bsp.parquet`.
Future normal training reads it automatically. Its default path now resolves
from the project directory, so launching training elsewhere also finds the
cache. A regression check changes the working directory and verifies this.

The completed win-only refit uses the ordinary training core and the same
recipe, corrected feature store and settled cutoff as026. The freshly trained
026 place component can be reused because its hard labels, dates, feature
order and input transform are identical. Exact guards enforce that scope.
Native fixture tests show that both final win and place forecasts match a
complete fresh refit exactly after changing the win-teaching prices. Twelve
join, parsing, portability and refit tests passed.

The Q1 performance figures in026 belong to its feature-repair comparison.
This source-data extension has no additional held-out performance estimate.
It does not establish market parity. The previous exchange-price cache and
the026 model remain preserved with hashes for reproducibility.

Installed model: `20261028_142429_576384`, fitted through 27 October 2026 on
667,873 runners and 71,477 races. Its win component selected 3,712 trees;
its corrected place component retains the verified026 fit of 2,993 trees.
The model SHA-256 is
`3c4d7964ffb96c0dfb347377ef9b1d7d0d8377a35e10ee2d6ce905dce0615acd`.

Final deep readiness has no blockers. The pinned native artifact check
passed, and the actual public path passed on the 14-runner Dundalk 15:00 card
(race945745). Probability checks cover 100 recent races and 955 runners;
single/batch forecasts agree exactly on five checked fields. This is not a
prospective score. A new fixed-artifact forward setup is eligible from
29 October, with zero recorded pairs and no scheduled collector.

Refit status, immutable completion receipt and source archive:
`data/audit/027_market_teacher_refresh/`. Public-card receipt:
`data/audit/live_upgrade_027/report.json`.
