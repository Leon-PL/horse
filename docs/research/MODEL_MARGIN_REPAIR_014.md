# Corrected result margins: measured prediction improvement (014)

Completed 27 October 2026. Evidence: `data/audit/model_improvement/014_margin_repair`.
This is retrospective development research. Production has not yet been changed.

## Result

Correct cumulative beaten distances improved the native 149-input model across all three registered quarters.
The rows, original outcomes, hard-label learner, calibration, recency settings and chronological cutoffs were unchanged.
All original saved models reproduced their frozen predictions exactly on the original input values.

| Period | Races | Original race log loss | Corrected race log loss | Change |
|---|---:|---:|---:|---:|
| 2025Q3 | 3,532 | 1.920633 | 1.915777 | -0.004855 |
| 2025Q4 | 3,233 | 1.862281 | 1.855808 | -0.006473 |
| 2026Q1 | 2,591 | 1.817994 | 1.817376 | -0.000617 |

Across 9,356 races / 85,863 runners, log loss fell from 1.872045 to 1.867804.
The paired change was -0.004241; race-bootstrap 95% interval [-0.0068703557034024455, -0.001687749763460736]; date-cluster interval [-0.006658097733013023, -0.001781302291564508].
Runner Brier fell from 0.08754346 to 0.08741422; binary log loss also improved. The predeclared numeric advancement checks passed.
Top-pick accuracy changed from 29.03% to 29.58% (a secondary point estimate).

On the same 8,942 market-covered races, original / corrected / market log loss was 1.868796 / 1.864647 / 1.749283.
The correction closed approximately **3.47% of the original measured market gap**. The market remains substantially better; these results do not establish a betting edge.

## What was repaired

Sporting Life reports the gap to the previous finisher. The project stored this as `lengths_behind`, and treated it as total distance behind the winner. We preserve that raw column and derive explicit `beaten_lengths`, handling ties and unknown chains.
548,270 recorded finisher margins changed. Margin Elo, margin form and margin-based speed histories were rebuilt before fitting.
An input-order audit also found a class-speed scaling defect. All runners of a class on a date now use the same prior-day class distribution.
All selected inputs passed seven causal and ordering checks.

## Limits and next experiment

The severe lower-tail errors remain: original and corrected models each assigned one eventual winner a probability at or below 1e-8. This repair did not solve that problem.
The independent non-finisher-label defect was discovered before reading the first 014 score but repaired in the separate 015 feature generation. Its effect must be measured separately.
015 restores non-finishers in histories and tests two completion predictors. 016 tests one fixed 50/50 winner/complete-field market teacher on those same corrected inputs. Both recipes were recorded before either was fitted.
Public missing-history profiles are cached and a seven-input causal prototype is tested, but these are not inputs to 014/015/016.

## Reproduction

The exact 014 execution source is `_research_snapshots/014_margin_repair`, with `source_snapshot.zip` and source digests in the experiment package. Current sources have advanced to the separate DNF repair, so do not run the old manifest with current FE code.
Manifest content SHA256: `84b3bf7c1b0cfc901f8baa11c6b2d8499be2ffa149b47ce8ffe1be7a144bdad5`. All 30 sealed file digests were verified after completion.
