"""Backfill the Timeform comment score (src/rtv_text.py).

Reads the card comment and race verdict for every linked starter since
September 2020, fits each missing quarter model in order and scores all
history into ``data/rtv_cache/feature_text_scores.parquet``. Daily refreshes
then keep the window current. ``--refit`` deletes the quarter models first.

    python -m scripts.backfill_rtv_text [--refit]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rtv_text  # noqa: E402
from src.ingestion.rtv_observations import refresh_text_history  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refit", action="store_true", help="delete and refit every quarter model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.refit:
        for path in rtv_text.models_dir().glob("*.npz"):
            path.unlink()
    print(json.dumps(refresh_text_history(), indent=2, default=str))


if __name__ == "__main__":
    main()
