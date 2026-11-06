# Rating systems repair and tuning (045) — 30 October 2026

A review of the skill ratings in `src/ratings.py` (Elo, margin Elo, Glicko-1,
Glicko-2, TrueSkill, jockey/trainer and surface/race-type/distance ratings)
found scale and modelling faults that made several of them weak measures of
ability. All were repaired and the settings re-tuned. The ratings are now much
better standalone predictors of the winner. **On top of the starting price they
still add nothing**, which is what the ledger already says about Elo and Glicko
(see [model disagreements](../NEGATIVE_RESULTS.md)). This is a feature-quality
repair, not a new edge.

## Method

`scripts/screen_ratings.py` recomputes the ratings over the development period
of the stored processed history (golden hold-out excluded by
`development_mask`). It scores each rating, or a small set of rating columns,
with a within-race conditional logit: fitted on the first 60% of races by date
and scored on the last 40% (29,786 races). Reported per model:

- **fit / test log loss**, in nats per race (lower is better; uniform over the
  field is 2.157 and SP is 1.758 on the test races);
- **market gain**: test log loss of SP alone minus SP plus the rating, so
  positive means the rating improves on the market;
- the correlation of the race-centred rating with `won − market`.

Settings were chosen on **fit-period** log loss only. The market is SP, not BSP,
so this is a screen of skill measures, not an edge verdict. Sweeps ran in
parallel over the same frame, one setting changed at a time from the working
baseline.

## What changed

| Area | Before | After |
|---|---|---|
| Elo scale | K 16 → 8 per update on a 400-point scale. Ratings barely moved, so expected scores sat near 0.5 and the rating became a tally of fields beaten. | K 512 → 128 on the mean residual over opponents (a race moves a debutant up to about ±512 points). |
| Non-finishers | Fallers, pulled-up and unseated runners were left out of every update. | They rank tied last, when the race has a finisher (`RATING_DNF_AS_LOSS`). |
| Jockey / trainer Elo | Rated like horses from their runners' finishing order, i.e. mostly the quality of horses they were given. | Value-added offsets: residual of the runner against what horse rating + offsets expected; stablemates are not opponents; one update per entity per race; read at the start of the race day. `jockey_elo = 1500 + offset`. The jockey race-type edge is a shrunk offset like the horse dimensions below; its updates are held until the day ends, because its all-type mean moves with every ride. |
| Surface / race-type / distance Elo | Separate rating pools per dimension, each starting at 1500, so one all-weather run gave an all-weather rating near 1500 whatever the turf record. | Global rating plus a shrunk offset: mean residual in the dimension minus mean residual overall, shrunk with 3 prior runs. Exactly 0 for single-dimension horses. Column names unchanged (`horse_elo_surface` = elo + edge, `_edge` = the offset). |
| Momentum | Undecayed EMA of rating changes, in points that depend on K. | EMA of the race residual in rating points (`1600/ln 10` per unit residual), decayed to the race date with a 30-day half-life. |
| Margin Elo | Updated only when two or more runners finished. | Also updates with a single finisher. |
| Glicko-1 | Each field of n runners counted as n−1 independent games, so RD collapsed after a few runs. | Each race's evidence weighted by `n_opponents^−0.25`. |
| Glicko margin | One margin scale at every trip. | Distance-scaled, as margin Elo already was. |
| Glicko-2 | φ inflated twice per race (volatility step and elapsed-time step). | Volatility solved on the stored φ, inflated once. |
| TrueSkill | No growth in uncertainty during a layoff (only τ per race). | σ grows by `TRUESKILL_C`² per idle month, capped at the starting σ; C = 2. |

Incremental state stays plain dicts; batch and seeded incremental sweeps still
match exactly (`tests/test_incremental_ratings.py`).

## Results

Test-period log loss, stored (old) columns against the recomputed ratings with
the final settings. Market gain is in nats; ±0.0002 is noise at this sample.

| Model | Old test LL | New test LL | Change | Old market gain | New market gain |
|---|---|---|---|---|---|
| horse_elo | 2.12496 | 2.01030 | −0.1147 | −0.00007 | +0.00007 |
| horse_margin_elo | 2.12519 | 2.00106 | −0.1241 | −0.00007 | +0.00010 |
| combined_elo | 2.06714 | 1.96850 | −0.0986 | +0.00003 | +0.00010 |
| elo + momentum | 2.03597 | 2.00577 | −0.0302 | +0.00008 | +0.00011 |
| elo + jockey + trainer | 2.02967 | 1.95199 | −0.0777 | −0.00010 | +0.00006 |
| elo + dim edges | 2.11952 | 2.00932 | −0.1102 | −0.00013 | +0.00004 |
| horse_glicko | 2.01165 | 2.00422 | −0.0074 | +0.00009 | +0.00009 |
| horse_glicko_lcb | 2.00781 | 2.00498 | −0.0028 | +0.00005 | +0.00002 |
| horse_ts_mu | 2.04998 | 1.99605 | −0.0539 | +0.00000 | +0.00007 |
| horse_ts_lcb | 2.08885 | 2.03097 | −0.0579 | −0.00003 | −0.00014 |

The old Elo was barely better than knowing nothing about the horses (uniform
2.157). Its momentum column carried most of what it knew, because with K 16 the
rating itself hardly moved: elo + momentum was 2.036 against 2.125 for the
rating alone. TrueSkill μ is now the best single rating, ahead of margin Elo.
Value-added jockey and trainer offsets take Elo from 2.010 to 1.952.

Old columns are the current store's (built by FE `b20404f117a63897`); new
columns are recomputed over the same rows.

## Settings chosen

| Setting | Value | Sweep (fit log loss) |
|---|---|---|
| `ELO_K_BASE` / `ELO_K_MIN` | 512 / 128 | 128/64 2.0683, 256/128 2.0434, 512/128 2.0231, 768/128 2.0217, 1024/512 2.0341. 768 is slightly better alone but worse with jockey + trainer (1.9675 vs 1.9656), so 512. |
| `ELO_K_DECAY` | 0.05 | 0.02 and 0.1 worse (measured with the rejected newcomer start; same direction). |
| `ELO_ENTITY_K_BASE` / `_MIN` | 32 / 4 | elo + jockey + trainer: 16/2 1.9662, 32/4 1.9656, 64/8 1.9686, 128/16 1.9760. |
| `ELO_MOMENTUM_HALF_LIFE_DAYS` | 30 | elo + momentum: 365 2.0224, 180 2.0222, 60 2.0217, 30 2.0213, 15 2.0211. Flat below 30, so the less extreme value. |
| `RATING_DIM_PRIOR_RUNS` | 3 | 1, 3 and 10 within 0.0003. |
| `RATING_DNF_AS_LOSS` | True | Off: Elo 2.0466 vs 2.0434, Glicko 2.0295 vs 2.0258. |
| `MARGIN_ELO_SCALE` | 3 (unchanged) | 3 2.0128, 5 2.0133, 8 2.0162. |
| `GLICKO_EVIDENCE_POWER` | 0.25 | 0 2.0183, 0.25 2.0149, 0.5 2.0170. |
| `GLICKO_C` | 70 (unchanged) | 35 2.0207, 50 2.0163, 70 2.0149, 90 2.0165, 110 2.0194. |
| `TRUESKILL_BETA` / `TRUESKILL_C` | 25/6 (library default) / 2 | C at β 25/6: 0 2.0470, 0.5 2.0358, 1 2.0217, 2 2.0105, 3 2.0110, 4 2.0130, 6 2.0197. β at C 1: 25/12 2.0277, 25/6 2.0217, 25/3 2.0353, 12.5 2.0532. τ 0 vs default: no difference. |

## Negative results

**Field-relative newcomer starts ratchet.** Starting a debutant below the mean
of the rated runners it meets — by a fixed gap of 50, 100 or 200 points, or a
gap learned from how debutants actually finish — always lost to a fixed start,
and lost more on the later test period than in the fit period. Maiden fields
are mostly recent debutants, so each cohort starts below the last one's
start and the scale drifts. Elo fit log loss: fixed 2.0231, gap 50 2.0376,
gap 100 2.0590, gap 200 2.1049 (test 2.1977), learned 2.0511–2.0525. Glicko:
fixed 2.0149, gap 100 2.0457, gap 200 2.0782, learned 2.0433–2.0444. Removed.

**No market gain.** Every rating, old or new, moves SP log loss by at most
0.0002 nats. The improvements make the ratings better descriptions of form the
market already prices.

## Plackett–Luce follow-up (1 November 2026)

`scripts/screen_ratings.py --systems pl` adds a research-only online
Plackett–Luce rating. It reads a finishing order as successive choices of
who finishes next among the runners still unplaced, so it is the likelihood
the screen's own conditional logit uses for the winner. Non-finishers sit in
every risk set but are never chosen. Pre-race values are unchanged when the
same day's results are scrambled. Two fits were swept, about 50 settings in
all, on the same fit-period log loss:

- **Weng & Lin (2011) Bayesian update**, a Gaussian skill per horse. Their own
  field scale (c over the sum of variances) barely moves ratings: fit
  2.119–2.120 at every layoff C. A field-size-free scale reaches 2.076 (C 4),
  and scoring only the top five places 2.055. A larger σ cap or a smaller β
  is worse.
- **Gradient update**, one log-strength per horse stepped along the race's
  log-likelihood gradient with Elo's experience-decayed K. Undamped, the best
  is 2.054 (K 1 → 0.25, top five places). Benter's discounted rank-ordered
  logit, which scales a horse's strength by d^(place−1) at later choices,
  is the one change that helped: d 0.85 with K 2 → 0.5 and decay 0.1 gives
  **2.032** (test 2.022). d 0.7 and below, top-k cuts, and leaving
  non-finishers out are all worse.

Undamped, the back of the field dominates the Plackett–Luce gradient: a horse
is charged at every place it failed to take, so tail places and fallers move
ratings most, yet they say least about ability.

| Model | Fit LL | Test LL | Market gain |
|---|---|---|---|
| horse_ts_mu | 2.01048 | 1.99605 | +0.00007 |
| horse_pl_mu (best) | 2.03222 | 2.02189 | +0.00006 |
| ts_mu + pl_mu | 2.01018 | 1.99597 | +0.00006 |

The best Plackett–Luce trails TrueSkill by 0.022 nats and pairwise Elo by
0.009. It adds 0.0003 nats to TrueSkill in the fit period and nothing on the
test period or over SP. Not added to the pipeline.

## Not done

- **Value-added Glicko jockey/trainer.** `GLICKO_JOCKEY_TRAINER` still rates
  them like horses; it is off by default.
- The off-by-default Glicko flags (jockey/trainer, dimensional, margin,
  Glicko-2) were judged on a 2-fold NDCG@1 sweep on the pre-leak-fix harness.
  Re-deciding them needs model fits, not this screen.

## Next steps

`src/ratings.py` and `config.py` changed, so `fe_signature()` changed and
`scripts/project_status.py` reports the store as stale: the next build
rebuilds every rating column (about an hour). The promoted run
`20261030_135650_092286` uses 37 inputs derived from these ratings, so it
must be retrained on the rebuilt store; until then the pipeline is not ready
to serve. Judge the retrained model by its proper scores and CLV at BSP
against that run, on the development period only.
