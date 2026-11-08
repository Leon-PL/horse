# Meta-learner plan: further sub-model ideas

Appendix to [the meta-learner plan](META_LEARNER_PLAN.md). It is a separate
file because that plan was still being edited when this was added.

## Further sub-model ideas (brainstorm, 8 November 2026)

Added verbatim from a Claude session at the user's request. These are ideas
only; none has been measured.

**Your pedigree idea, in brief:** yes, it makes sense, but only as a starting estimate for horses that have barely run.
- **Limit:** racing ability is only partly inherited. Published estimates of how much of the variation in racing ratings is inherited mostly fall between about 0.1 and 0.35. A foal also gets a random half of each parent's genes, so even knowing both parents perfectly explains only about half of that, around 5–17% of the variation in ability. A few of the horse's own runs soon outweigh it.
- **What we can measure:** most sires and dams raced before 2020, so their own ability isn't in our data. What we do have is how their other offspring performed, and that's actually a better guide to what they pass on.
- **Improvements on the planned pedigree sub-model (S4):**
  - Measure relatives in rating points rather than win rates. Win rates mislead because they depend on what class of race the offspring ran in.
  - Weight each sire's figure by how many offspring it rests on.
  - Allow for trainer, since well-bred horses tend to go to better yards.
  - Identify dams by dam plus dam-sire, not dam name alone.
- **Expectation:** sibling form and sale price both tested as fully priced by the market. This would make the model stronger, not find an edge.

### Other sub-models worth considering

**New information (the strongest kind of difference, per the plan)**
1. **Same-day track bias.** Earlier races on the same card show draw, pace and rail bias, and how the ground is riding. It only applies to later races and close-to-the-off bets.
2. **Ground preference from non-runner reasons.** The BHA reports now cover everything, and they record why horses were withdrawn (for example "Self Cert (Going)"). That shows what ground the connections think suits a horse, which feeds the conditions sub-model.
3. **Health history.** The BHA vet findings after races (bled, lame, irregular heartbeat, lost a shoe) could predict a recurrence or explain a poor run. Caution: stewards' reports were measured as priced.
4. **Which runner the stable prefers.** When a trainer or owner has several runners, the retained jockey and the owner's first colours show which they prefer. The `trainer_first_string_*` columns exist but aren't used, and the results table has no owner column. Caution: the jockey-booking idea was dead.

**Different targets**
5. **Improvement trajectory.** Predict how far a horse will improve from its current level, from age, how many runs it has had, gaps between runs, and late-maturing sire lines. Pairs naturally with pedigree.
6. **Handicap leeway.** Predict the gap between the level a horse runs to and its official mark, or its next rating change. Handicaps are about half of all races.
7. **Completion risk (jumps).** Predict falls, unseats, pull-ups and refusals from jumping errors (RTV), chase experience, ground and previous falls. The win chance is then the chance of completing times the chance of winning if it completes. On the flat, the equivalent is a horse's tendency to start slowly.
8. **Bounce and recovery.** Predict how a horse runs after a career-best or a hard race, using the effort measured from sectionals and the days since that run.
9. **How predictable the race is.** A race-level model of how open the race is. The meta-learner can then sharpen or flatten its probabilities race by race.

**Structure and the market tier**
10. **Uncertainty.** Measure, for each runner, how much the sub-models disagree. The meta-learner uses it to shrink weights, and betting can avoid horses the model is unsure about.
11. **Market bias.** Model where the market is systematically wrong, such as overrating long shots, by field size, race type, meeting quality, time of day and liquidity. It uses only the price and context, and belongs in the market tier.
12. **A proper ability tracker.** The plan lists one measuring ability in pounds as missing. One option is a running estimate of each horse's ability that is allowed to change faster for improvers than for exposed horses.
13. **Likely trouble in running.** Predict the chance of a troubled run from running style, draw, field size and how tight the track is. It looks ahead, unlike the existing trip and luck sub-model, which looks back.

**Where I'd start:**
- **5 and 6:** new targets on data you already have.
- **7:** a clear structure for jumps.
- **1:** genuinely new information, if you'll bet near the off.
- **10:** cheap, and it improves every other sub-model.
