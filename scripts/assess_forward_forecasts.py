"""Assess recorded five-minute pairs; do not alter or regenerate forecasts."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forward_store import ForwardStudy
from src.forward_assessment import save_assessment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--results-db", type=Path, default=ROOT / "data/races.db")
    args = parser.parse_args()
    study = ForwardStudy(args.study)
    report, output = save_assessment(study, args.results_db)
    print(f"Assessment saved to {output}. {report.get('races', 0)} complete matched races; descriptive monitoring only.")
