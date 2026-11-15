# Debutants and Timeform's pre-race comment (research 101, 14 November 2026)

Question: debutants carry a large part of the gap to Betfair SP (0.091 race
log loss per race with a debutant against 0.060 overall). Earlier work closed
the public routes: sibling form (2h), sale price (ledger 5), trainer and sire
debut record (055), connection profiles (060), the newcomer and debut-choice
correction groups, and morning money (19). The one public signal left is the
Timeform analyst comment on the newcomer: pedigree, price, yard intent and
phrases such as "market check advised". Does the model use all of it?

Debutant = no prior form figures (`prior_form_runs` = 0). 15,477 debutant
runners from the stored 87-input forecasts; 99.6% have a comment.

## Phrase screen

`research/debutant_text_101/screen.py`. Winners against expected, by phrase
in the comment:

| Phrase | Runners | Won | Model | z | BSP | z |
|---|---:|---:|---:|---:|---:|---:|
| Market interest ("market", "money", "support") | 1,579 | 106 | 113.5 | −0.76 | 119.6 | −1.37 |
| Price or purchase ("€", "gns", "cost", "bought") | 7,400 | 602 | 608.8 | −0.30 | 636.2 | −1.54 |
| Family ("brother", "sister", "dam", "related") | 13,787 | 1,067 | 1,070.6 | −0.12 | 1,092.7 | −0.88 |
| Positive ("should", "interesting", "promising") | 3,047 | 339 | 362.1 | −1.36 | 363.0 | −1.45 |
| Negative ("modest", "needs time", "will need") | 727 | 45 | 38.3 | +1.18 | 36.7 | +1.53 |
| Yard's newcomers | 783 | 74 | 72.3 | +0.23 | 74.0 | 0.00 |

No phrase separates winners from the forecast by more than z 1.6, against
either the model or BSP.

## Debutant-only text model

`research/debutant_text_101/text_fit.py`. A ridge regression of
won − forecast on TF-IDF word and word-pair counts of debutant comments,
applied as a tilt exp(k × score) on debutants and renormalised within the
race. The ridge is fitted on earlier dates. k is fitted on the latest fifth
of those dates, held out of the ridge. Each period is scored on races with a
debutant (hold-out untouched):

| Scored | Races | k | Race log loss change |
|---|---:|---:|---:|
| 2025H1 | 1,177 | 0.00 | 0.00000 |
| 2025H2 | 1,398 | 0.00 | 0.00000 |
| 2026-01-01 to 06-10 | 921 | 6.85 | +0.00194 [−0.00210, +0.00608] |

Null. Fitted honestly, the tilt is zero in two periods and makes the third
worse. (A first version that fitted k on the ridge's own training rows ran
k to its bound and lost 0.07–0.11. That is overfitting, not information.)

## Reading

The 87-input text score (`rtv_text_logit`, 052/056) already takes what the
comment says about newcomers. What remains of the debutant gap is private
information that shows in the money, the price-timing study's subject.
Debutants are closed for public pre-race data.

Outputs: `data/research/debutant_text_101/` (`screen.csv`, `text_fit.csv`).
