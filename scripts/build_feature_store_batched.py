"""Rebuild the feature store from the full history with bounded memory.

The build itself is ``src.store_build.build_store``, which training also runs
when the store must be rebuilt; see that module for how it bounds memory.
This command adds a separate store root for candidate builds, validated
rating reuse and replay of a recorded dataset.

Run:
  .venv/Scripts/python.exe -u -m scripts.build_feature_store_batched --chunk-rows 120000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_store import FeatureStore
from src.store_build import LOG_FORMAT, build_store

logger = logging.getLogger("build_feature_store_batched")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunk-rows", type=int, default=120_000,
                    help="approx rows per chunk (split on date boundaries)")
    ap.add_argument("--store-root", default=None,
                    help="optional separate store for a versioned candidate build")
    ap.add_argument("--no-helpers", action="store_true",
                    help="use one feature process to bound memory on a busy machine")
    ap.add_argument("--reuse-ratings-from", default=None,
                    help="unchanged rating snapshot, requiring a frozen receipt")
    ap.add_argument("--ratings-receipt", default=None)
    ap.add_argument("--dataset-manifest", default=None, help="Replay a verified immutable dataset ID; no collection")
    args = ap.parse_args(argv)

    store = FeatureStore(args.store_root) if args.store_root is not None else FeatureStore()
    ratings_cache = None
    if bool(args.reuse_ratings_from) != bool(args.ratings_receipt):
        ap.error("--reuse-ratings-from and --ratings-receipt are required together")
    if args.reuse_ratings_from:
        from src.rating_reuse import ValidatedRatingCache
        ratings_cache = ValidatedRatingCache(args.reuse_ratings_from, args.ratings_receipt)
    from src.source_dataset import load_manifest, use_manifest
    selected = load_manifest(args.dataset_manifest) if args.dataset_manifest else None
    with use_manifest(selected), store._staged_update() as pending:
        meta = build_store(pending, args.chunk_rows, ratings_cache=ratings_cache, use_helpers=not args.no_helpers)
    logger.info("Feature store committed: %d featured rows", meta["featured_rows"])
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
