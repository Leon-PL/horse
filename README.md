# Horse racing probabilities

A local system that collects UK and Irish racing data, builds a historical
feature store, fits a win model plus a forecast correction, and follows one
betting rule forward on live prices.

**The live system (from 16 November 2026)** is the Matchbook paper record of the
11:00 rule: every racing morning it forecasts the day's card and records the paper
bets the rule would make at 11:00 Matchbook prices, then settles them against
Betfair SP. Nothing places real bets. Operating guide:
[docs/operations/MATCHBOOK_PAPER.md](docs/operations/MATCHBOOK_PAPER.md).

Check the installed state before trusting any number:

```powershell
.venv\Scripts\python.exe scripts/project_status.py
```

## How it fits together

| Stage | Code | Command |
|---|---|---|
| Collect sources (Sporting Life results and cards, Racing TV, At The Races, Betfair history, BHA reports, FX, French/UAE form) and publish immutable versions | `src/ingestion/`, `src/source_dataset.py` | `python -m src.ingestion --sync 14 --results-only` |
| Build or append the feature store (338 numeric features kept) | `src/store_refresh.py`, `src/store_build.py`, `src/feature_engineer.py`, `src/store_columns.py` | `python -c "from src.store_refresh import get_training_dataset; get_training_dataset()"` |
| Win model: 87 inputs, run `20261114_122027_967139` | `src/modelling/`, `src/model.py`, `src/run_store.py` | `python -m src.modelling --help` |
| Forecast correction: how the win model's earlier forecasts turned out | `src/correction/` | `python -m src.correction status` |
| 11:00 rule on Matchbook prices (correction with the race's own 11:00 price, threshold 0.02, odds up to 51) | `src/matchbook/`, `src/bet_settlement.py` | `python -m src.matchbook day --refresh` (scheduled task `HorseMatchbookPaper`, 07:50) |
| Inspect races, sources, features, models and the paper record | `src/workbench/`, `ui/dist/` | `python scripts/workbench.py`, then http://127.0.0.1:8765 |

Run commands from the project root with `.venv\Scripts\python.exe`. The module
map is in [docs/PROJECT_INVENTORY.md](docs/PROJECT_INVENTORY.md).

## Evidence the rule rests on

- [Research 107](docs/research/HOLDOUT_1100_107.md): the 11:00 rule passed its
  registered one-look hold-out test (geometric CLV 1.0445, ROI +8.3% on 1,944
  bets). Fills and capacity are unproven; that is what the paper record is for.
- [Research 108](docs/research/MATCHBOOK_VARIANT_108.md): the same rule without
  Betfair trade inputs, as Matchbook serves it.
- [Research 109](docs/research/FEATURE_PRUNING_109.md): store pruned and sped up
  with every column the models read unchanged.

Earlier research was measured on older pipelines. Its one-line index is in
[docs/ALREADY_TRIED.md](docs/ALREADY_TRIED.md); the full records are in
[docs/archive/](docs/archive/). Read the index before proposing a new feature or
data source.

## Development

- Python 3.14 in `.venv`, built from `uv.lock` by `scripts/tooling.ps1 venv`.
  Dependencies are declared in `pyproject.toml`.
- Tests: `scripts/tooling.ps1 test` (quick) or
  `scripts/tooling.ps1 test -TestPath tests` (full, ~15 min).
- `src/` is pipeline code, `scripts/` operating commands, and new experiments go
  in [research/](research/README.md). Nothing outside `research/` imports it.
- `data/` and `models/` are local and gitignored.
- Read [CLAUDE.md](CLAUDE.md) before changing feature or settlement code.

Everything removed in the November 2026 cleanup is in git at tag
`pre-cleanup-2026-11`.
