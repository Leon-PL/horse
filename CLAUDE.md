# CLAUDE.md

Start with **[README.md](README.md)**, the current operating guide, then use
**[the project inventory](docs/PROJECT_INVENTORY.md)** to find operational,
maintenance and research tools. Run `scripts/project_status.py` for actual
readiness; dated prose is not a substitute for that check.

**[CONTINUATION.md](CONTINUATION.md)** holds session handoff detail.
**[Implementation progress](docs/MARKET_PERFORMANCE_PROGRESS.md)** and the
**[negative-results ledger](docs/NEGATIVE_RESULTS.md)** index measured evidence.
`PROJECT_OUTLINE.md`, older plans and `NEXT_AGENT_BRIEF.md` retain historical
architecture and research context; their old work queues and model IDs do not
define the current task or installation.

The rules below are the ones that cause real damage when violated.

## Hard rules

1. **Data-leakage:** any feature describing a horse's past must strictly
   exclude the current race. Never use
   `df.groupby(key)[col].cummax().shift(1)` — the `.shift(1)` is global and
   leaks across horses. Use `transform(lambda x: x.cummax().shift(1))`.
2. **Bet rules live in `src/bet_settlement.py`.** Value thresholds, the
   each-way odds band, and settlement PnL maths are shared by the test-set
   simulation, the walk-forward backtester, and live settlement. Never
   inline them — backtest and live results must use identical formulas.
3. **Merge keys live in `src/utils.py`** (`normalise_horse_key`,
   `normalise_track_key`, `normalise_off_time_key`). Use them for any
   horse/track/off-time join.
4. **A module that can change a feature value must be registered in
   `_FE_SOURCE_FILES`** (`src/featured_dataset.py`), or exempted in
   `FE_SIGNATURE_EXEMPT` with a reason. Otherwise `fe_signature()` stays
   unchanged, the store is not rebuilt, and stale features are served
   silently — this is what produced the phantom "+15% place edge".
   `tests/test_fe_signature_coverage.py` enforces it.
5. **Never trust a number without checking the instrument first:**
   `.venv/Scripts/python.exe scripts/project_status.py`. It verifies that
   results → store → model → promoted model all agree, and exits non-zero
   when they do not. Two of this project's biggest "edges" were artifacts of
   links that silently disagreed.
6. **Walk-forward evaluation primitives live in `src/wf_eval.py`.** Use
   `expanding_period_folds` (purged), `oof_predict_expanding` (never fit a
   second stage on in-sample predictions) and `bootstrap_roi_ci` (resamples
   races, not bets). Hand-rolled fold loops produced a biased result that
   closed a real lead for three months.

## Before proposing a new feature or data source

Read **[docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md)**. It records what
has already been built, measured and refuted, with numbers: sale price, wind
surgery, trainer travel, jockey booking, stewards reports and par-time speed
figures are all dead, as is the market combination (measured twice, on two
different harnesses).

Judge candidates by **orthogonality to the market** (correlation with
`won − market_implied_prob`) and by **CLV at BSP**, not raw predictiveness.
High ROI with CLV ≈ 1.0 means a leak, not an edge.

Read CLV from `src.clv.summarise_clv` — `median_clv`, `geometric_mean_clv`,
`pct_beat_close`. **Never the mean of `taken_odds / bsp` alone:** a price
ratio has unbounded upside and bounded downside, so its mean runs high at long
prices, and that has already reversed a verdict (mean 1.0229 vs median 0.9691
on the same bets).

When comparing a selection rule to "no filter", compare against the **same
universe the rule bets in**, not the whole band. A price-capped rule looks
like a big win against an unrestricted baseline purely for avoiding long
prices; `scripts/audit_selection_rule.py` reports both.

## Practical notes

- Live environment: existing `.venv` on Windows. Use `scripts/tooling.ps1 setup`
  for isolated development; do not synchronize a running recorder's `.venv`
  during setup. Dependencies are declared in `pyproject.toml`, resolved in
  `uv.lock`, and exported to `requirements.lock`.
- Run `scripts/tooling.ps1 test` for pilot checks or
  `scripts/tooling.ps1 test -TestPath tests` for the full suite. See
  [development tools](docs/DEVELOPMENT_TOOLS.md) for compatibility limits.
- Default model framework is **LightGBM** (see
  `config.SUB_MODEL_FRAMEWORKS`), not XGBoost/CatBoost.
- `data/` and `models/` are gitignored and large; run snapshots are
  pruned from the Experiments tab (Delete Runs → Disk Usage & Pruning).
- Performance: features are computed over hundreds of thousands of rows —
  prefer Numba/vectorised pandas over Python loops.
