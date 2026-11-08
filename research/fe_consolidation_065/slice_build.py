"""Build features for a recent slice of history, from raw results, without the store.

The FE consolidation (065) is checked by building the same slice before and
after each change and comparing the outputs with ``compare.py``. A slice
starts cold (its history features see only the slice), which is fine for a
before/after comparison: both sides see the same input. Nothing is written to
the feature store.

Sources come from a published dataset manifest. ``--data-root`` points at the
``data`` folder that holds it, so a git worktree without its own data can read
the main checkout's accepted sources.

Run:
  .venv/Scripts/python.exe -m research.fe_consolidation_065.slice_build \
      --data-root C:/Users/Leonp/Documents/horse/data --start 2025-06-01 \
      --output data/research/fe_consolidation_065/baseline.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd


def _manifest(data_root: Path, identifier: str | None) -> dict:
    datasets = data_root / "sources" / "datasets"
    if identifier is None:
        identifier = json.loads((datasets / "latest.json").read_text(encoding="utf-8"))["id"]
    manifest = json.loads((datasets / f"{identifier}.json").read_text(encoding="utf-8"))
    manifest["root"] = str(data_root / "sources")
    return manifest


def build(data_root: Path, start: str, end: str | None, manifest_id: str | None) -> pd.DataFrame:
    from src.data_processor import load_raw_data, process_data
    from src.feature_engineer import engineer_features
    from src.source_dataset import use_manifest

    manifest = _manifest(data_root, manifest_id)
    with use_manifest(manifest):
        raw = load_raw_data()
        dates = pd.to_datetime(raw["race_date"], errors="coerce")
        keep = dates >= pd.Timestamp(start)
        if end:
            keep &= dates <= pd.Timestamp(end)
        raw = raw.loc[keep].reset_index(drop=True)
        processed = process_data(raw)
        featured = engineer_features(processed, save=False)
    featured.attrs.clear()
    return featured


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    logging.getLogger().setLevel(logging.WARNING)
    began = time.perf_counter()
    featured = build(Path(args.data_root), args.start, args.end, args.manifest)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(out, index=False, engine="pyarrow")
    print(f"{len(featured)} rows x {featured.shape[1]} columns in "
          f"{time.perf_counter() - began:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
