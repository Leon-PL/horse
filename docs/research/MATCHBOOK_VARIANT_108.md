# The 11:00 rule without Betfair trade inputs (research 108)

Matchbook's public feed has no trades, so the live paper record
(`src/matchbook`, `docs/operations/MATCHBOOK_PAPER.md`) cannot feed
`pm_age` (time since last trade) or `pm_changes` (price changes so far). This
re-runs the research 107 rehearsal without them. The rehearsal used development
races, walk-forward 2025Q3–2026Q2, the registered rule, and Betfair prices
for the bets. The hold-out was not read.
Code: `research/matchbook_variant_108/run.py`.

| Rule, threshold 0.02, one tick worse | Bets | Geo CLV [95%] | ROI [95%] | Race log loss |
|---|---:|---:|---:|---:|
| Research 107 (eight price inputs) | 4,324 | 1.0372 | +5.5% [+0.5, +10.7] | 1.7806 |
| **Without trade inputs (six)** | **4,396** | **1.0379 [1.0307, 1.0449]** | **+6.7% [+1.6, +11.5]** | **1.7809** |

The two trade-activity inputs add nothing. The six-input model is what the
Matchbook paper record serves.
