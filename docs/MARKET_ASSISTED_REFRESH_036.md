# Market-assisted forecast refresh:036

Neither a simple market recalibration nor a combination with the current
independent recipe earned selection. Production remains the installed033
independent model. This investigation required **zero new tree fits**.

## Timing and source audit

The same frozen 154-input historical model, trained through22 December2025,
supplies all calibration and later test forecasts. The October production
weights are never applied retrospectively to these outcomes. Replay against
20 saved historical fields has maximum probability error zero.

Prices are Betfair BASIC last-traded prices at five minutes before the first
published start. Only matching complete final-starter fields with no late
withdrawal qualify. Every runner price is at most300 seconds old and the
market observation at most90 seconds old; schedule/definition timing is also
verified. There are2,080 eligible calibration fields and1,351 later test fields
/11,951 runners, including487 established-handicap fields /4,188 runners.

These are previously examined development dates, retrospective final-starter
features and last-traded prices, not executable quotes or fully issued historic
cards. The frozen model ages across the evaluation. Results cannot establish
profitability, fresh confirmation, or validity for Matchbook best-back prices.

## Fixed small comparison

The probability formula is softmax(alpha*log(model)+beta*log(market)). The
identity alpha=0,beta=1 reproduces the market exactly. Two bounded convex
optimizations fit a market-only temperature and a combination, with a small
penalty towards the market identity. Analytic gradients, whole-field sums,
parameter bounds and optimizer success are checked. A failed or worsening
optimization falls back to the identity; it is never clipped after an
unconstrained fit. Nine regression checks pass.

Fit dates are1 January–21 February2026,1,181 fields. After a seven-day gap,
validation is28 February–31 March,760 fields. Selection requires better race
log loss, no worse runner Brier, and no worsening of either target score when
there are at least50 established-handicap validation fields. The selected
recipe is saved **before** evaluating1 May–10 June2026. Reserved dates from
11 June onward are not evaluated in this investigation.

| Forecast | Fitted alpha | Fitted beta | Validation race LL | Validation Brier |
|---|---:|---:|---:|---:|
|Market reference|0|1|1.699829|.09409332|
|Market recalibration|0|.991287|1.699850|.09409018|
|Model/market combination|.082016|.934247|1.702137|.09427223|

**Validation selects the unchanged market reference.** Both optimizers converge,
but neither fitted correction meets the rule. Only two tiny coefficient fits
are needed; the entire fit/readout takes about6.2 seconds.

| Later test population | Independent LL | Market LL | Recalibrated LL | Combined LL |
|---|---:|---:|---:|---:|
|All1,351 fields|1.884735|1.778875|1.778783|1.780645|
|Established handicaps:487 fields|1.898041|1.789571|1.789953|1.792246|

Overall Brier is .08788918 for the market versus .08793078 for the combination.
The combination's date-cluster LL-delta interval versus market is
[+.000188,+.003582]; target [+.000181,+.005328]. Both favour the market.
The temperature's tiny overall test gain is uncertain and did not earn prior
validation selection. It is not chosen after seeing the test.

The selected forecast equals the market, so matching that reference here is
not independent predictive skill or model-added value. No correction is wired
into ordinary forecasts. This closes the fixed simple-correction proposal,
not every possible market-assisted model.

Evidence: `data/audit/market_assisted_036/plan.json`, `preparation.json`,
`recipe.json`, `report.json`, `test_predictions.parquet` and `final_status.json`.
Implementation: `src/market_correction_research.py` and
`scripts/check_market_assisted.py`, both research-only.
