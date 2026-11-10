# Feature-Engineering: target architecture & migration plan

**Historical design and migration record.** Several stages below have since
shipped. Training now enters through `src.store_refresh.get_training_dataset`,
the only refresh route; the `get_featured_dataset` seam described below and
`FeatureStore.append` were retired on 2026-11-07. See the
[inventory's consolidation boundaries](../PROJECT_INVENTORY.md#consolidation-boundaries)
and the [README](../../README.md) for operation.

For the implemented source, categorical-schema and private execution-state
contracts as of 8 November, use [FEATURE_CONTRACTS.md](FEATURE_CONTRACTS.md).
The older phase statuses and timings below are historical, not the current work
queue; rolling and cumulative state already resume across chunks.

> **Status (2026-10-13): phases 0–2 are built and shipped**, so the sections
> below describing them are a record, not a to-do. The single seam
> (then `featured_dataset.get_featured_dataset`, since 2026-11-07
> `store_refresh.get_training_dataset`), both freshness guards
> (`fe_signature` / `data_signature`), the new-rows-only append and the
> counts accumulator are all in the tree and under test. **Phases 3–4
> (bounded ring buffers for the rolling/lag stages, then retiring the
> recompute fallback) are still open** — they are a build-time optimisation,
> not a correctness fix, and a full rebuild currently takes ~56 minutes.
> Prioritise against [ACTION_PLAN.md](../archive/ACTION_PLAN.md) before picking them up.

Goal: **separate FE from the app**, build the dataset **once** then **append**,
and make the burn-in / "grab future races" behaviour explicit and correct.
Grounded in the stage classes in [FE_TAXONOMY.md](FE_TAXONOMY.md).

## Where we are

- `engineer_features` is one 3,945-line pass over the whole frame → 1,091 cols.
- It is called from **three** places, all full-frame:
  - `train.py` and the app's cached builders → rebuild everything from scratch;
  - the live path (`feature_engineer_with_history_core`) → prepends a *bounded*
    history to the new ("future-edge") races and features only that slice. This
    is the right instinct — it already avoids a full rebuild for prediction.
  - `FeatureStore` (append-only, parity-tested) → **adopted nowhere**. Only
    Elo/Glicko carry true incremental state; everything else recomputes.
- Burn-in is a *train-time* hack (`BURN_IN_MONTHS` strips early rows after FE),
  so a bounded build has no warm-up story and cold-start rows carry noisy
  near-zero cumulative stats.

The live path and the training path are the **same problem** — "feature new rows
using prior context without rebuilding history" — solved twice, inconsistently.
Unify them behind one store.

## Target architecture

One FE owner; everyone else is a consumer.

```
                    ┌─────────────────────────────┐
 races.db ──────▶   │  src/featured_dataset.py     │  ◀── single entry point
                    │  get_featured_dataset()      │
                    │   • build()  (cold)          │
                    │   • append() (new dates)     │
                    │   • staleness guard (rebuild │
                    │     when FE code/config hash  │
                    │     changes)                  │
                    └──────────────┬───────────────┘
                                   │ wraps
                          ┌────────▼────────┐
                          │  FeatureStore   │  featured.parquet + processed.parquet
                          │  + per-stage    │  + state blobs (ratings, counts,
                          │    incremental  │    buffers) + meta(fe_signature)
                          └────────┬────────┘
                                   │ uses
                          ┌────────▼────────┐
                          │ engineer_features (unchanged) — fallback &
                          │ source of truth for parity tests              │
                          └─────────────────┘

consumers:  train.py   app.py   fe_app.py (bounded ad-hoc builds)   live_prediction
```

Key idea: **append never rebuilds history.** The featured history is immutable
(R/W/L/B/C/F are all backward-or-guarded), so new dates are featured from
carried state + a bounded recent buffer and concatenated on.

## Interfaces

```python
# src/featured_dataset.py  — the single build path
def get_featured_dataset(
    *, processed: DataFrame | None = None,
    rebuild: bool = False,          # force a full rebuild
    fast_append: bool = False,      # rating-only fast path (parity-tested)
) -> DataFrame: ...
    # build if cold or stale (fe_signature changed); else append new dates; load.

# bounded, ad-hoc build (training experiments, the FE studio) — no store mutation
def build_window(
    *, start, end, burn_in_months: int = 0,
    options: dict | None = None,    # config overrides, restored after
) -> DataFrame: ...
    # window = [start - burn_in, end]; engineer_features; drop pre-start prefix.
```

Two staleness guards, both stored in `meta` and checked on every
`get_featured_dataset`:

- **`fe_signature`** = hash of `feature_engineer.py` + `ratings.py` + the
  FE-relevant `config` flags + `STORE_SCHEMA_VERSION`. Catches *code/config*
  changes.
- **`data_signature`** = cheap fingerprint of the processed history over the
  region at-or-before the store's last date (row count + date span + value
  sums of `draw`/`won`/`finish_position`/`num_runners`). Catches *in-place data
  edits* — backfills/cleanups that change historical rows (e.g. the draw
  backfill changes `draw`'s sum, the non-runner purge changes the row count).
  The append model assumes history is immutable, so any change there forces a
  rebuild instead of a (silently stale) append.

Either mismatch forces a full rebuild, so adopting the store never silently
serves stale features.

## Burn-in, resolved

Two regimes, made explicit:

1. **Incremental store** — carries *full* per-entity state, so there is **no
   cold start at the append edge**: once the store exists, appended (and live)
   rows are warmed by the carried state and are exactly correct, with no
   burn-in needed for them. NB this removes the burn-in worry only at the
   *append edge* — the **head of the full history is still cold** (the dataset
   still begins at the true start with unwarmed cumulative/rolling/Elo stats),
   so `BURN_IN_MONTHS` is still applied as a train-split trim (see below).
2. **Bounded rebuild** (`build_window`, the FE studio, walk-forward folds) —
   cannot see full history, so it uses **warm-up-then-drop**: load
   `[start − burn_in, end]`, engineer, drop the pre-start prefix. Rolling/365d
   features are warmed; deep career-cumulative features near `start` stay
   approximate. Pick burn-in ≥ the longest window that matters (≥1y covers
   rolling + TE; career stats need more or the full store).

`BURN_IN_MONTHS` (config.py, applied in `src/model.py`) stays as the
*train-split* trim that drops the cold head of history before the time split.
The store does **not** remove this — it only removes cold-start at the append
edge. What the store *does* change: burn-in is no longer entangled with *which
dataset/window* you built — there is one canonical dataset, and burn-in is just
a slice of it.

## "Grab future races", precisely

- Live prediction concatenates recent history with the new races and features
  the slice — *not* future leakage, just bounded context to avoid a rebuild.
- The only forward-looking feature (collateral form, `shift(-1)`) is
  date-guarded, so it is class **F**: append-safe. The append-safety test
  forbids any *un-guarded* forward reference or whole-dataset (class **G**)
  statistic from creeping in.

## State-threading mechanism (Phase 2, built)

How seed-state flows through the monolith without changing every stage
signature:

- `engineer_features(..., counts_state: dict | None = None)`. `None` = full
  build (byte-identical to before). A dict = a seeded run.
- `counts_state_scope(state)` sets a module-level `_COUNTS_STATE` for the stage
  loop; `_cumulative(keys, races, metrics)` (used by class-C stages) seeds its
  `cumulative_counts` from it and writes the advanced state back. State is a
  flat `{column: accumulator_state}` dict (column names are globally unique, so
  no per-stage keying).

Seeded-append algorithm (exact, no approximation):

1. Persist `counts_state` = the per-entity totals through the store's last date
   (the C stages write it back during the build).
2. On append, run `engineer_features(new_rows, counts_state=seed)` — class-C
   columns for the new rows = `seed[entity] + in-frame race-safe cumsum`, i.e.
   exactly a full recompute; the dict is advanced and re-persisted.
3. **For B/L (bounded-window/lag) stages to also be exact, they must be
   seed-aware too** (carry the last N races / 365d / last row per entity as
   state) — otherwise the append would need the full history (no win) or a
   recent buffer (approximate, breaking the append==rebuild parity guarantee).
   That is Phase 3.

So the flip is exact once **both** C (counts state) and B/L (bounded-buffer
state) stages are converted; ratings already carry state; W/R need only the new
rows. Until then `counts_state=None` everywhere and behaviour is unchanged.

## Migration plan (incremental, each step shippable & reversible)

**Phase 0 — single seam (this change).**
`src/featured_dataset.py` + route `train.py` through it + `test_append_safety.py`.
Behaviour identical on first run (store builds via `engineer_features`); from
then on training appends instead of rebuilding. Delete `data/processed/
feature_store/` (or `rebuild=True`) to force a clean rebuild.

**Phase 1 — free stages on new rows only.**
Tag R/W stages; in `append`, run them over just the new races (they need no
history). Removes them from the history-recompute. ~11 stages, zero risk
(stateless), parity-tested.

**Phase 2 — counts accumulator (class C).**
One shared per-entity accumulator (the Elo/Glicko `return_state` pattern,
generalised). Convert the cumulative stages group by group behind parity tests
(`tests/test_feature_store.py` style). ~15 stages — the bulk of the speed-up.

**Phase 3 — bounded buffers (classes B/L).**
One ring-buffer state (last *N* races / 365d / last row per entity). Convert the
rolling / time-window / lag stages. ~10 stages.

**Phase 4 — retire the recompute fallback** once all stages are incremental;
`engineer_features` stays as the parity oracle and the bounded-build engine.

## Risks & guards

- **Stale store after FE edits** → `fe_signature` rebuild guard (Phase 0).
- **Stale store after data backfills/cleanups** → `data_signature` rebuild
  guard over the historical region (Phase 0).
- **Silent divergence** → every converted stage gated by a parity test against a
  full `engineer_features` rebuild (the existing pattern).
- **Append-safety regressions** → `test_append_safety.py` fails CI on un-guarded
  `shift(-1)` / whole-dataset transforms.
- **Config/option drift** → options fold into `fe_signature`, so a build with
  different flags is treated as a different dataset.
