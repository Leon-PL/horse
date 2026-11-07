# Development tools and library pilots

Implemented and measured on **29 October 2026**. The entry point is
`scripts/tooling.ps1`. It uses a dedicated Python and environment under
`data/tooling/`. Since 30 October 2026 the apps' and CLI's `.venv` is built
from the same lock on the same Python; see [the project .venv](#the-project-venv).

## Commands

Run from the project root in PowerShell:

```powershell
.\scripts\tooling.ps1 setup
.\scripts\tooling.ps1 venv
.\scripts\tooling.ps1 test
.\scripts\tooling.ps1 validate
.\scripts\tooling.ps1 lint
```

Setup bootstraps uv 0.13.6, installs a workspace-local Python 3.14.0 without
registering it globally, and synchronizes the locked development/tooling groups.
It is safe to rerun. The first setup needs a working `python` on PATH and
downloads dependencies. No model fitting, source scraping or promotion occurs.

`test` runs the three new test modules. For the full suite:

```powershell
.\scripts\tooling.ps1 test -TestPath tests
```

`lint` reports existing code issues and exits nonzero when it finds them. It
applies no fixes or formatting. Treat its baseline as a review queue rather
than a count of model defects. Formatting no longer changes the feature
signature (it hashes the AST), but it still changes the raw-hash recipe
certificates in `src/deployment_validation.py`.

`venv` rebuilds or updates the project `.venv` from `uv.lock` (runtime and dev
groups, no tooling group). Use it after a lock change. Prefer it to bare
`uv sync`, which would pick up whatever Python and install directory the shell
happens to provide.

## The project .venv

On 30 October 2026 the `.venv` was rebuilt from `uv.lock`. Before that it was a
hand-grown installation on a broken base (`C:\Python314` had no standard
library and loaded it through a registry `PythonPath` key, so `python -I`
failed). It now runs on the uv-managed Python 3.14.0 in `data/tooling/python`,
which is inside the project because the Claude desktop app virtualizes
`AppData` and a base there is invisible to other shells. Do not delete
`data/tooling/python` without rebuilding `.venv`.

The scientific core was already at the locked versions (pandas 2.3.3,
numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, LightGBM 4.6.0, numba 0.64.0,
pyarrow 23.0.1). llvmlite moved from 0.46.0b1 to 0.46.0, shap from 0.50.0 to
0.54.1, and transitive packages to their locked versions. Parity was checked
before and after the switch. The promoted model scored the 2026-10-30 card
through the full live feature path (478 runners, 45 races) and 5,022 stored
runners from 16 to 29 October. All 154 inputs and every probability column
were bit-identical between the old and new environments. The old environment
is kept at `data/tooling/venv-old_20261030` for rollback: move `.venv` aside and
move that folder back to `.venv`.

## Dependency ownership and compatibility

| File | Responsibility |
|---|---|
| [pyproject.toml](../../pyproject.toml) | Direct core dependencies, development/tooling groups and Ruff rules. |
| [uv.lock](../../uv.lock) | Resolved versions, dependency graph and artifact hashes. |
| [requirements.lock](../../requirements.lock) | Runtime-only pip export from uv.lock. |
| [requirements.txt](../../requirements.txt) | Compatibility pointer to that export. |

The lock targets **Windows/Python 3.14**. It is a maintained project environment,
not a byte-for-byte copy of the old installation. Direct scientific versions
are retained except that the old `llvmlite==0.46.0b1` was unavailable from the
package index; the isolated environment uses published `0.46.0`. Transitive
dependencies are resolved and locked separately.

SHAP (race explanations in `src/model.py`) is locked at `0.54.1`, the first
release that resolves with `numba==0.64.0`; the live installation's `0.50.0`
pins an unavailable numba prerelease on Python 3.14 and cannot be locked.
`psutil` is a runtime dependency: `src/store_build.py` reports the build's
memory with it when present, and `scripts/benchmark_data_queries.py` requires it.

On 2026-10-30 every direct dependency was checked against the project's
imports. XGBoost, CatBoost and lxml had no remaining use and were removed with
their code paths (the XGBoost/CatBoost training options, the CatBoost and
pytabkit experiments in `scripts/model_lab.py`, and an unused template scraper
in `src/data_collector_sample.py`). All saved runs had used LightGBM only.

H2O, FLAML, experimental neural-network packages and other unrelated extras
are absent from the new graph. On 2026-10-30 they were also uninstalled from
the live `.venv`: 43 packages (about 5 GB, mostly a CUDA build of torch) that
no project code imports. The pre-uninstall `pip freeze` and the removed list
are in `data/audit/library_consolidation_20261030/`. That environment was then
replaced by the locked rebuild described above. The 2026-10-29
requirements and installed versions are under
`data/audit/library_consolidation_20261029/`.

For a deliberate lock update, set `UV_PROJECT_ENVIRONMENT` to
`data/tooling/project` and `UV_PYTHON_INSTALL_DIR` to `data/tooling/python`
before using the local uv executable. Regenerate the pip export afterwards:

```powershell
data\tooling\bootstrap\Scripts\uv.exe export --frozen --no-dev --no-group tooling --no-hashes --no-emit-project --output-file requirements.lock
```

## Pandera: settled-field contracts

[data_contracts.py](../../src/data_contracts.py) defines an opt-in schema for
required columns, identities, dates, positions, winner flags and field sizes.
Its whole-field check delegates to the existing `valid_race_mask` in
`src/forecast_evaluation.py`, keeping one racing-specific rule owner.

The schema never coerces types, adds columns, drops rows or fills histories.
Genuine non-finishers remain starts. Extra feature columns and legitimate
missing form values are preserved. It is an audit pilot outside the pinned
training/serving imports.

`tooling.ps1 validate` reads metadata columns from processed history. Its
explicit `--eligible-only` mode first applies the existing whole-field rule:

| Item | Runners | Fields |
|---|---:|---:|
| Source | 754,842 | 80,250 |
| Existing exclusions | 1,549 | 162 |
| Successfully validated | 753,293 | 80,088 |

The source was unchanged. Invoke `scripts/validate_data_contracts.py` without
`--eligible-only` to audit all stored fields strictly; known invalid fields
will cause failure rather than being silently repaired.

## Hypothesis: generated racing cases

[Generated invariant tests](../../tests/test_generated_racing_invariants.py)
exercise existing behavior on deterministic, bounded random cases:

- Permutation and real non-finishers preserve field eligibility.
- Removing a runner rejects its whole race without rejecting another race.
- Pending cards cannot masquerade as settled observations.
- Place probabilities retain bounds, win floors and expected place mass.
- Full and chunked rolling/EWMA histories agree with missing observations and
  repeated entities; future outcomes cannot change the prefix.

Contract and query tests add malformed-field and read-only fixture checks.
Optional-library tests skip in the old environment if dependencies are absent;
the tooling setup installs and executes them.

## DuckDB: measured query pilot

```powershell
.\scripts\tooling.ps1 benchmark
```

The default report path is exclusive. For another run, invoke
`scripts/benchmark_data_queries.py` with a new `--output` using the isolated
Python. Each fresh worker reads the same dates and four numeric inputs from
the 2.71 GB feature file and aggregates daily counts and means. Pandas already
uses PyArrow column selection and date filters; the baseline does not load
the full feature matrix unnecessarily.

Both engines returned equivalent results for **32,523 runners / 90 dates**.
Three fresh-worker repetitions per engine, two threads, alternating order:

| Engine | Median query time | Median sampled peak process RSS |
|---|---:|---:|
| pandas/PyArrow | 0.305 seconds | 158.5 MiB |
| DuckDB | 0.401 seconds | 190.9 MiB |

DuckDB did not improve this query, so the normal read path is retained.
It remains optional for future join-heavy research queries with their own
benchmarks. This result does not measure full training or all SQL workloads.
OS file caches were not cleared. Whole-process RSS was sampled every 5 ms and
includes imported libraries. The live recorder was left running.
Both workers import the same comparison libraries before timing, so these are
benchmark-process memory figures rather than the normal app's memory use.

## Evidence

Local reports under `data/audit/library_consolidation_20261029/`:

- `baseline.json`: original live package versions and protected file hashes.
- `validation.json`: eligible-field audit and explicit exclusion counts.
- `duckdb_benchmark.json`: timings, sampled memory, source identity and parity.
- `ruff_all.json`: reporting-only baseline of existing findings.
- `full_tests.log`: regression run in the isolated locked environment.
- `final_status.json`: completion receipt and final preservation checks.
- `compatibility.json`: all 57 protected file hashes match, live package
  versions are unchanged, the pinned study verifies and its recorder is healthy.

The scoped Ruff check passes on all six new Python files. The initial existing
code baseline contains 1,042 findings in 162 files, mostly import order and
compact statements; no automatic fixes were applied to those files.
The complete regression suite passed: **1,308 tests**, 190 warnings, in
648.75 seconds. The focused pilot run passed all 15 tests before that full run.

The pilot changes no forecast calculations or selected model inputs. Model,
source/study pins and recorder health are checked separately afterwards;
operational checks do not establish prediction accuracy.
