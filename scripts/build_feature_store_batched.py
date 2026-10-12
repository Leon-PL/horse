"""One-time, memory-bounded build of the feature store over the full history.

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
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src.data_processor import process_data
from src.feature_engineer import engineer_features
from src.feature_store import STORE_SCHEMA_VERSION, FeatureStore, _stringify_objects
from src.featured_dataset import data_signature, fe_signature

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
    args = ap.parse_args()

    t0 = time.perf_counter()
    store = FeatureStore()
    parts_dir = os.path.join(store.root, "_build_parts")
    os.makedirs(parts_dir, exist_ok=True)
    for old in glob.glob(os.path.join(parts_dir, "*.parquet")):
        os.remove(old)

    logger.info("Loading full processed history…")
    processed = process_data(save=False)
    processed["race_date"] = pd.to_datetime(processed["race_date"], errors="coerce")
    processed = processed.sort_values("race_date", kind="stable").reset_index(drop=True)
    rd = processed["race_date"]
    vc = rd.value_counts()
    dates = np.sort(rd.dropna().unique())
    counts = {d: int(vc[d]) for d in dates}
    spans = _date_chunks(dates, args.chunk_rows, counts)
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
        keyed, _cols, ratings_state = store._seeded_rating_frame(chunk, ratings_state)
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
    schemas = [pq.read_schema(p) for p in part_paths]
    unified = pa.unify_schemas(schemas)

    # Preserve missing speed histories; no future-dependent global fill.

    # ── save processed + carried state + meta BEFORE the streaming write, so a
    # consolidation hiccup never discards the expensive per-chunk work. ─────
    last_date = rd.max()
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
        "fe_signature": fe_signature(),
        "data_signature": data_signature(processed, cutoff=last_date),
    }

    # ── consolidate parts → single featured.parquet (streaming, pyarrow) ──
    # Conform each part to the unified schema (add missing columns as typed
    # nulls; never .cast a narrower table) without imputing unknown histories.
    def _conform(tbl: pa.Table) -> pa.Table:
        have = set(tbl.schema.names)
        arrays = [
            tbl.column(f.name).cast(f.type) if f.name in have
            else pa.nulls(tbl.num_rows, f.type)
            for f in unified
        ]
        return pa.Table.from_arrays(arrays, schema=unified)

    logger.info("Consolidating %d parts → %s", len(part_paths), store.featured_path)
    writer = pq.ParquetWriter(store.featured_path, unified)
    for p in part_paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=50_000):
            tbl = _conform(pa.Table.from_batches([batch]))
            writer.write_table(tbl)
        del pf  # release the file handle so the cleanup below can unlink it (Windows)
    writer.close()
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
        "DONE: %d featured rows over %s..%s in %.0fs (RSS %.1f GB)",
        total_rows, str(dates[0])[:10], str(dates[-1])[:10],
        time.perf_counter() - t0, _rss_gb(),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
