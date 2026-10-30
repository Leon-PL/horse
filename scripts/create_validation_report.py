"""Export a recorded evaluation recipe for later deployment fitting."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deployment_validation import write_validation_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="completed evaluation run ID")
    parser.add_argument("--output", required=True, help="new certificate JSON path")
    args = parser.parse_args()
    write_validation_report(args.run, args.output)
    print(f"Recipe validation saved to {args.output}; no model fitted or promoted.")
