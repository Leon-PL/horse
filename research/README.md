# Research

Experiments, screens, audits and benchmarks live here, apart from the pipeline.

| Folder | Holds |
|---|---|
| `src/` | Pipeline code: collection, features, training, prediction. |
| `scripts/` | Operating commands (see [scripts/README.md](../scripts/README.md)). |
| `tests/` | Tests of the pipeline and commands. |
| `research/` | Everything exploratory. Nothing outside this folder imports it. |

## Rules

1. **One-way imports.** Research code may import `src` and `config`. Nothing in
   `src/`, `scripts/`, `tests/` or `config.py` imports `research`;
   `tests/test_layout.py` checks this.
2. **One folder per experiment**, named subject first and number last, such as
   `research/pace_shape_064/` (a name starting with a digit cannot be run
   with `python -m`). Keep its tests in the same folder (`test_*.py`)
   and run them with `scripts\tooling.ps1 test -TestPath research`, so the normal
   suite stays about the pipeline.
3. **Same correctness rules as the pipeline** (see [CLAUDE.md](../CLAUDE.md)):
   evaluation primitives from `src/wf_eval.py`, bet rules from
   `src/bet_settlement.py`, CLV from `src.clv.summarise_clv`, and no global
   `.shift` after a grouped cumulative. `tests/test_leakage_patterns.py`
   scans every file in this folder.
4. **Outputs** go to `data/research/<experiment>/`. Record the result in
   [docs/research/](../docs/research/) and, if refuted, in the
   [already-tried index](../docs/ALREADY_TRIED.md).
5. **Promotion into the pipeline** means moving the code into `src/` with tests.
   A module that can change a feature value must be registered in
   `_FE_SOURCE_FILES` (`src/featured_dataset.py`).

Run an experiment from the project root, for example
`.venv\Scripts\python.exe -m research.pace_shape_064.run`.

## What is kept

Only the studies the live 11:00 rule rests on, with the modules they import:

| Folder | Why |
|---|---|
| `holdout_1100_107/` | The registered one-look hold-out test of the 11:00 rule (passed). |
| `matchbook_variant_108/` | The rule without Betfair trade inputs, as the Matchbook paper record serves it. |
| `feature_pruning_109/` | Store keep list (`keep_list.py` generates `src/store_columns.py`), fingerprints and pinned rebuilds. |
| `error_anatomy/`, `meta_learner/`, `trainer_moves_098/`, `comebacks_099/`, `t60_clv_105/` | Only the files 107 imports (correction groups, 11:00 and T-60 prices). |

## Earlier research

Everything else was removed on 2026-11-16; it is in git at tag
`pre-cleanup-2026-11` (and the research before 2026-11-07 at commit `26d1b6f`).
Its write-ups are summarised in [docs/research/](../docs/research/). They were
measured on earlier pipelines: treat them as history, not current fact.
