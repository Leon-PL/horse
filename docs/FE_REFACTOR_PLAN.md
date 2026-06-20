# Feature-Engineering: target architecture & migration plan

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
   cold start and no burn-in needed**. This is the real answer to the burn-in
   worry: once the store exists, appended rows are exactly correct.
2. **Bounded rebuild** (`build_window`, the FE studio, walk-forward folds) —
   cannot see full history, so it uses **warm-up-then-drop**: load
   `[start − burn_in, end]`, engineer, drop the pre-start prefix. Rolling/365d
   features are warmed; deep career-cumulative features near `start` stay
   approximate. Pick burn-in ≥ the longest window that matters (≥1y covers
   rolling + TE; career stats need more or the full store).

`BURN_IN_MONTHS` stays available as the *train-split* trim, but is no longer the
only lever.

## "Grab future races", precisely

- Live prediction concatenates recent history with the new races and features
  the slice — *not* future leakage, just bounded context to avoid a rebuild.
- The only forward-looking feature (collateral form, `shift(-1)`) is
  date-guarded, so it is class **F**: append-safe. The append-safety test
  forbids any *un-guarded* forward reference or whole-dataset (class **G**)
  statistic from creeping in.

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
