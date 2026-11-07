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
   `src/`, `scripts/`, `tests/`, `train.py` or `config.py` imports `research`;
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
   [negative-results ledger](../docs/NEGATIVE_RESULTS.md).
5. **Promotion into the pipeline** means moving the code into `src/` with tests.
   A module that can change a feature value must be registered in
   `_FE_SOURCE_FILES` (`src/featured_dataset.py`).

Run an experiment from the project root, for example
`.venv\Scripts\python.exe -m research.pace_shape_064.run`.

## Earlier research

The previous research scripts, research-only modules and their tests were
removed on 2026-11-07. They are in git at commit `26d1b6f`
(`git show 26d1b6f:scripts/<name>.py`). One uncommitted script,
`benchmark_atr_queue.py`, is in `data/archive/removed_research_2026-11-07/`.
Their results remain in `docs/research/`, `docs/NEGATIVE_RESULTS.md`,
`data/research/` and `data/audit/`.
