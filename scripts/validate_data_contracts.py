"""Read-only Pandera audit of settled fields using existing eligibility rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data_contracts import RESULT_COLUMNS, validate_settled_races  # noqa: E402
from src.forecast_evaluation import valid_race_mask  # noqa: E402


def audit(path: Path, *, eligible_only: bool = False) -> dict:
    """Read metadata columns only; never modify the canonical input file."""
    frame = pd.read_parquet(path, columns=RESULT_COLUMNS)
    total_rows, total_races = len(frame), frame.race_id.nunique(dropna=False)
    if eligible_only:
        frame = frame.loc[valid_race_mask(frame)].copy()
    validate_settled_races(frame)
    return {
        "source": str(path.resolve()),
        "scope": "existing eligible fields" if eligible_only else "all stored fields",
        "source_rows": total_rows,
        "source_races": total_races,
        "validated_rows": len(frame),
        "validated_races": frame.race_id.nunique(dropna=False),
        "excluded_rows": total_rows - len(frame),
        "excluded_races": total_races - frame.race_id.nunique(dropna=False),
        "valid": True,
        "source_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/processed/feature_store/processed.parquet")
    parser.add_argument("--eligible-only", action="store_true",
                        help="Explicitly apply the normal whole-field exclusion rule first")
    args = parser.parse_args()
    print(json.dumps(audit(args.source, eligible_only=args.eligible_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
