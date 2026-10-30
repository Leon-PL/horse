"""Memory-bounded history build, shared by normal training and explicit rebuilds.

A monolithic ``get_featured_dataset(rebuild=True)`` over the whole DB (~707k
rows → a ~6 GB featured frame, plus several intermediate copies during FE)
peaks past 16 GB of RAM and thrashes the machine. This script produces a
byte-equivalent store while never holding more than one chunk in memory:

  1. Load the full processed history once (cheap; processed is narrow).
  2. Walk it in date-ordered chunks. Each chunk is featured on its OWN rows
     only, with ratings + every pandas stage seeded from the *previous*
     chunk's carried state — exactly the incremental append path, which is
     parity-proven against a monolithic build (tests/test_feature_store.py).
     So N chunks == one full build.
  3. Write each featured chunk to a part file. When done, stream the parts
     into the single ``featured.parquet`` with pyarrow (one record-batch at a
     time, schemas unified) so the consolidation never materialises the whole
     frame in pandas either.
  4. Persist ratings_state + counts_state and stamp meta with the same
     fe_signature / data_signature get_featured_dataset() expects, so the very
     next train APPENDS instead of rebuilding.
  5. Commit all five outputs together under the feature-store lock. A failed
     build leaves the previous store intact; an interrupted commit is resumed
     before the next store operation.

Run:
  .venv/Scripts/python.exe -u scripts/build_feature_store_batched.py --chunk-rows 120000
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.feature_store import STORE_SCHEMA_VERSION, FeatureStore, _stringify_objects
from src.featured_dataset import data_signature, fe_signature
from src.feature_inputs import atr_cache_signature, rtv_cache_signature
from src.parquet_io import combine_parquet_parts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("batched_build")


def _rss_gb() -> float:
    try:
        import psutil  # optional

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return float("nan")


def _date_chunks(dates: np.ndarray, chunk_rows: int, counts: dict) -> list[tuple]:
    """Group sorted unique dates into spans of ~chunk_rows rows each.

    Splits only on date boundaries so a race is never cut across chunks.
    Returns a list of (lo_date, hi_date) inclusive ranges.
    """
    spans, lo, acc = [], None, 0
    for d in dates:
        if lo is None:
            lo = d
        acc += counts[d]
        if acc >= chunk_rows:
            spans.append((lo, d))
            lo, acc = None, 0
    if lo is not None:
        spans.append((lo, dates[-1]))
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-rows", type=int, default=120_000,
                    help="approx rows per chunk (split on date boundaries)")
    ap.add_argument("--store-root", default=None,
                    help="optional separate store for a versioned candidate build")
    ap.add_argument("--reuse-ratings-from", default=None,
                    help="unchanged rating snapshot, requiring a frozen receipt")
    ap.add_argument("--ratings-receipt", default=None)
    args = ap.parse_args()

    store = FeatureStore(args.store_root) if args.store_root is not None else FeatureStore()
    ratings_cache = None
    if bool(args.reuse_ratings_from) != bool(args.ratings_receipt):
        ap.error("--reuse-ratings-from and --ratings-receipt are required together")
    if args.reuse_ratings_from:
        from src.rating_reuse import ValidatedRatingCache
        ratings_cache = ValidatedRatingCache(args.reuse_ratings_from, args.ratings_receipt)
    with store._staged_update() as pending:
        meta = _build(pending, args.chunk_rows, ratings_cache=ratings_cache)
    logger.info("Feature store committed: %d featured rows", meta["featured_rows"])
    print(json.dumps(meta, indent=2))


def _build(store: FeatureStore, chunk_rows: int, *, ratings_cache=None, processed=None) -> dict:
    """Build into the supplied staging store without touching the live files."""
    t0 = time.perf_counter()
    # Fix the signature BEFORE any work. Reading it at the end described
    # whatever the source files said after an hour of building, not the code
    # that computed the values: the store in use on 2026-10-12 was stamped
    # mid-edit that way and matched neither generation (ledger 12).
    fe_sig_at_start = fe_signature()
    parts_dir = os.path.join(store.root, "_build_parts")
    os.makedirs(parts_dir, exist_ok=True)
    for old in glob.glob(os.path.join(parts_dir, "*.parquet")):
        os.remove(old)

    logger.info("Loading full processed history…")
    processed = process_data(save=False) if processed is None else processed.copy()
    processed["race_date"] = pd.to_datetime(processed["race_date"], errors="coerce")
    processed = processed.sort_values("race_date", kind="stable").reset_index(drop=True)
    reused_ratings = ratings_cache.prepare(processed) if ratings_cache is not None else None
    if reused_ratings:
        logger.info("Reusing %d rating columns after validating all %d processed rows",
                    len(reused_ratings["rating_columns"]), len(processed))
    rd = processed["race_date"]
    vc = rd.value_counts()
    dates = np.sort(rd.dropna().unique())
    counts = {d: int(vc[d]) for d in dates}
    spans = _date_chunks(dates, chunk_rows, counts)
    logger.info("Full history: %d rows, %d dates → %d chunks (RSS %.1f GB)",
                len(processed), len(dates), len(spans), _rss_gb())

    ratings_state: dict | None = None
    counts_state: dict = {}
    total_rows = 0
    part_paths: list[str] = []

    for i, (lo, hi) in enumerate(spans):
        chunk = processed[(rd >= lo) & (rd <= hi)].copy()
        # Seed ratings from the prior chunk's end-state (None == from scratch),
        # attach the rating columns, then feature the chunk's rows only with
        # every pandas stage resumed from carried counts_state.
        if ratings_cache is None:
            keyed, _cols, ratings_state = store._seeded_rating_frame(chunk, ratings_state)
        else:
            keyed = ratings_cache.for_chunk(chunk)
            ratings_state = ratings_cache.final_state
        rated = chunk.merge(keyed, on=["race_id", "horse_name"], how="left")
        featured = engineer_features(
            rated, save=False, skip_ratings=True, counts_state=counts_state,
        )

        part = os.path.join(parts_dir, f"part_{i:04d}.parquet")
        _stringify_objects(featured).to_parquet(part, index=False, engine="pyarrow")
        part_paths.append(part)
        total_rows += len(featured)
        logger.info(
            "chunk %d/%d [%s..%s]: %d rows featured (%d cols) → %s | cum %d | RSS %.1f GB | %.0fs",
            i + 1, len(spans), str(lo)[:10], str(hi)[:10], len(featured),
            featured.shape[1], os.path.basename(part), total_rows, _rss_gb(),
            time.perf_counter() - t0,
        )
        del chunk, keyed, rated, featured
        gc.collect()

    # ── unify schemas across chunks (later chunks gain RTV columns once RTV
    # data starts; all such extras are float, NaN for earlier rows in a
    # monolithic build, so adding them as typed nulls is exact). ───────────

    # Preserve missing speed histories; no future-dependent global fill.

    # Write only to staging. Consolidation must finish before any of these
    # states, rows or metadata can replace the previous store generation.
    last_date = rd.max()
    if fe_signature() != fe_sig_at_start:
        # Raising here aborts the staged update, so the previous store stays.
        raise RuntimeError(
            "FE code changed while the store was building; its values describe "
            "neither generation. Rebuild without editing FE sources meanwhile."
        )
    _stringify_objects(processed).to_parquet(
        store.processed_path, index=False, engine="pyarrow"
    )
    store._save_state(ratings_state or {"elo": None, "glicko": None})
    store._save_counts_state(counts_state)
    meta = {
        "schema_version": STORE_SCHEMA_VERSION,
        "last_date": str(last_date.date()),
        "featured_rows": int(total_rows),
        "processed_rows": int(len(processed)),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "fe_signature": fe_sig_at_start,
        "data_signature": data_signature(processed, cutoff=last_date),
        # The sectional cache is read inside FE, outside both signatures.
        "atr_signature": {"cutoff": str(last_date.date()),
                          "sig": atr_cache_signature(last_date)},
        "rtv_signature": {"cutoff": str(last_date.date()), "sig": rtv_cache_signature(last_date)},
    }
    from src.prize_currency_features import reference_signature
    meta["prize_reference_signature"] = reference_signature()
    if reused_ratings:
        meta["rating_reuse"] = reused_ratings

    # ── consolidate parts → single featured.parquet (streaming, pyarrow) ──
    # Conform each part to the unified schema (add missing columns as typed
    # nulls; never .cast a narrower table) without imputing unknown histories.
    logger.info("Consolidating %d parts → %s", len(part_paths), store.featured_path)
    written = combine_parquet_parts(part_paths, store.featured_path)
    if written != total_rows:
        raise RuntimeError("Batched feature consolidation changed the row count")
    logger.info("featured.parquet written (RSS %.1f GB)", _rss_gb())

    with open(store.meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)

    # Best-effort temp cleanup: a leftover part is harmless (the next build
    # clears _build_parts first), and on Windows the parquet handle can linger.
    for p in part_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(parts_dir)
    except OSError:
        pass

    logger.info(
        "Prepared: %d featured rows over %s..%s in %.0fs (RSS %.1f GB)",
        total_rows, str(dates[0])[:10], str(dates[-1])[:10],
        time.perf_counter() - t0, _rss_gb(),
    )
    return meta


if __name__ == "__main__":
    main()
