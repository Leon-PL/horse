# CLAUDE.md

Start with **[README.md](README.md)**, the current operating guide. The live
system is the Matchbook paper record of the 11:00 rule
([operations guide](docs/operations/MATCHBOOK_PAPER.md)); the
**[module map](docs/PROJECT_INVENTORY.md)** lists the pipeline modules. Run `scripts/project_status.py` for actual
readiness; dated prose is not a substitute for that check.

**Layout:** `src/` is pipeline code, `scripts/` operating commands only, and new
experiments, screens and audits go in **[research/](research/README.md)**.
Research may import `src`; nothing in `src/`, `scripts/` or `tests/` may import
`research` (`tests/test_layout.py`). Everything removed in the November 2026
cleanup (old training/backtest path, retired research, feature sets, dated docs)
is in git at tag `pre-cleanup-2026-11`.

**[CONTINUATION.md](CONTINUATION.md)** holds the current session handoff.
**[docs/ALREADY_TRIED.md](docs/ALREADY_TRIED.md)** indexes earlier ideas and
their verdicts; the full ledger, progress log, plans and audits are in
`docs/archive/`. They were measured on older pipelines: history, not current
fact, and their work queues and model IDs do not define the current task.

The rules below are the ones that cause real damage when violated.

## Hard rules

1. **Data-leakage:** any feature describing a horse's past must strictly
   exclude the current race. Never use
   `df.groupby(key)[col].cummax().shift(1)` — the `.shift(1)` is global and
   leaks across horses. Use `transform(lambda x: x.cummax().shift(1))`.
2. **Bet rules live in `src/bet_settlement.py`.** Value thresholds, the
   each-way odds band, and settlement PnL maths are shared by the test-set
   simulation, research rehearsals and the Matchbook paper record. Never
   inline them — backtest and live results must use identical formulas.
3. **Merge keys live in `src/utils.py`** (`normalise_horse_key`,
   `normalise_track_key`, `normalise_off_time_key`). Use them for any
   horse/track/off-time join.
4. **A module that can change a feature value must be registered in
   `_FE_SOURCE_FILES`** (`src/featured_dataset.py`), or exempted in
   `FE_SIGNATURE_EXEMPT` with a reason. Otherwise `fe_signature()` stays
   unchanged, the store is not rebuilt, and stale features are served
   silently — this is what produced the phantom "+15% place edge".
   `tests/test_fe_signature_coverage.py` enforces it. The signature hashes
   each source's AST, so comments, docstrings and formatting do not force the
   ~15 minute rebuild; any code, constant or config change still does.
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

Read **[docs/ALREADY_TRIED.md](docs/ALREADY_TRIED.md)**. It indexes what has
already been built and measured: sale price, wind surgery, trainer travel,
jockey booking, stewards reports, par-time speed figures and the market
combination all came out dead on the pipelines of their day. Retry one only
with a concrete reason the old test no longer applies.

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
prices.

## Practical notes

- Environment: `.venv` on Windows, built from `uv.lock` by
  `scripts/tooling.ps1 venv` on the project-local Python in
  `data/tooling/python` (not AppData, which the Claude app virtualizes).
  `scripts/tooling.ps1 setup` builds the separate tooling env. Dependencies
  are declared in `pyproject.toml`, resolved in `uv.lock`, and exported to
  `requirements.lock`.
- The live model is the Matchbook paper record (win run
  `20261114_122027_967139` plus the correction in `data/matchbook/model`,
  scheduled task `HorseMatchbookPaper`). It is read-only: nothing places bets.
  Do not change `src/matchbook`, `src/correction` or the store on a racing
  morning (07:50 until the last race). Do not reintroduce source-pinned
  studies, which made every edit a new study.
- The feature store keeps only `src/store_columns.py`'s 338 numeric columns,
  generated by `research/feature_pruning_109/keep_list.py`. To use a dropped
  column, add it to a feature-set file, regenerate and rebuild (~15 min).
- Run `scripts/tooling.ps1 test` for pilot checks or
  `scripts/tooling.ps1 test -TestPath tests` for the full suite. See
  [development tools](docs/operations/DEVELOPMENT_TOOLS.md) for compatibility limits.
- **LightGBM** is the only model framework. XGBoost and CatBoost were
  removed on 2026-10-30; `RacePredictor` rejects any other framework.
  The explicit win modelling path keeps LightGBM for boosted estimators;
  its small linear text expert and conditional-logit pool use the installed
  scipy/sklearn stack. See `docs/operations/MODELLING.md`.
- `data/` and `models/` are gitignored and large; run snapshots are
  pruned with `src.run_store.prune_runs` / `delete_run` (the Streamlit
  apps were removed on 2026-11-07).
- Performance: features are computed over hundreds of thousands of rows —
  prefer Numba/vectorised pandas over Python loops.
