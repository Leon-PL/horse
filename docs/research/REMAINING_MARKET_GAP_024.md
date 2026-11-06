# Remaining market gap after the capacity improvement

This review reuses the frozen022 forecasts on5,594 complete market-matched
fields. It fits no models and changes no prediction or feature definitions.
The two development periods have already been inspected; these subgroup
descriptions are leads for further work, not independent significance tests.
Market prices retain the original016 historical benchmark's limitations.

| Complete-field cohort | Races | New model minus market log loss | Capacity change in log loss |
|---|---:|---:|---:|
| Chase | 1,091 | +0.084318 | -0.002924 |
| Hurdle | 1,771 | +0.113256 | -0.005171 |
| Flat | 2,732 | +0.104815 | -0.006217 |
| No horse without local history | 4,453 | +0.094234 | -0.004246 |
| At least one horse without local history | 1,141 | +0.139612 | -0.009138 |
| Most horses have at least three local starts | 4,488 | +0.092989 | -0.004162 |
| At least half have fewer than three local starts | 1,106 | +0.146100 | -0.009632 |

Lower log loss is better. The capacity change improves every listed cohort.
The native Flat label includes National Hunt Flat/bumpers; it does not mean
that all these races belong to the ordinary Flat programme (see025).
Zero local history does not establish that a horse is a genuine debutant or
French: identifying earlier overseas runs requires a separate source match.

Races without a locally unseen horse account for about72.5% of the remaining
aggregate gap. Sparse-history races have a higher loss gap per race, but
improving established-horse forecasts is still essential. Races without any
ATR-covered runner are not the worst cohort here; lack of sectionals alone
does not explain the deficit. These overlapping cohort families must never
have their contributions added together.

The next feature priorities are therefore:

1. Separate genuine debuts, missing local records and verified overseas
   experience. Preserve uncertainty when a public source has no match. The
   existing public-history union is a research input; the018 additions did
   not demonstrate a probability-score gain, so they remain unpromoted.
2. Inspect the worst disagreements with the market for established horses,
   tracing the underlying records and selected ratings, margins and class
   histories before proposing new interactions. The saved example table
   includes both obvious outsiders and short-priced winners that the model
   almost dismissed; these require different explanations.
3. Retest only a concrete information or representation hypothesis. Use one
   matched inexpensive screen, with full prior feature histories, followed
   by wider validation only for a clear shortlisted gain. Keep the new
   production-capacity recipe as the reference for future comparisons.

The native race-normalisation and calibration design can also be improved,
but the019 calibration experiment already failed to beat the original
recipe overall. Do not repeat a temperature sweep against these same test
outcomes. The market-informed reference remains M0 until a restrained
correction demonstrates out-of-sample lift.

Evidence: `data/audit/model_improvement/024_gap_diagnostics/report.json`,
`winner_gap_examples.csv` and `scripts/diagnose_remaining_market_gap.py`.
All source projections have fingerprints, the overall metrics replay022
exactly, and every scored subgroup contains whole race fields.
