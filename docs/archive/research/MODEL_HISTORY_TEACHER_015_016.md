# Corrected histories and market teaching: completed results

Retrospective chronological research on all three registered quarters (2025Q3, 2025Q4, 2026Q1), 9,356 races.

| Recipe | Race log loss | Runner Brier | Top-pick accuracy |
|---|---:|---:|---:|
| Original 149 inputs | 1.872045 | 0.08754346 | 29.03% |
| 014: corrected margins | 1.867804 | 0.08741422 | 29.58% |
| 015_nonfinisher_history | 1.868009 | 0.08739293 | 29.61% |
| 016_market_teacher | 1.857492 | 0.08714276 | 29.87% |

Lower log loss and Brier are better. Accuracy is a secondary diagnostic.

015_nonfinisher_history: log-loss change versus original -0.004036; race-bootstrap 95% interval [-0.007856318353372924, 0.00016631723933877924].
On the same 8,942 market-covered fields, candidate log loss 1.864918; market 1.749283.
Original market gap closed (point estimate): 3.24%.

016_market_teacher: log-loss change versus original -0.014553; race-bootstrap 95% interval [-0.01747661092027206, -0.01153510944351667].
On the same 8,942 market-covered fields, candidate log loss 1.854783; market 1.749283.
Original market gap closed (point estimate): 11.73%.

Selected for final qualification: **016_market_teacher**.
Selection requires the registered complete-period gate and improvement over the best completed 014 model; teaching must also improve over 015. No race-type subgroup selection was used.

The 50/50 teacher uses historical complete-field market probabilities only during training; its live forecasts do not require odds.
A separate final-quarter win/place artifact validates the recipe. Any later full-history deployment refit has no new held-out performance estimate.

The public-profile prototype and raw-source-recency finding are not inputs to these comparisons.
Original evidence is preserved under data/audit/model_improvement/015_nonfinisher_history and 016_market_teacher.
Qualification and deployment state: data/audit/model_improvement/015_016_adoption/status.json.

Production upgraded to `20261027_165656_616818` using settled history through 2026-10-25. Installation resumed from the saved completed fit; no retraining. The historical table describes separate evaluation fits. Previous model, store and settings are recoverable.
