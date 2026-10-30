"""Register/check a fixed baseline-pilot readout; no collection or scheduling."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forward_store import ForwardStudy, publish_json
from src.forward_readout import register_plan, load_plan, readout_readiness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--output", type=Path, help="new exclusive readiness JSON, optional")
    args = parser.parse_args()
    study = ForwardStudy(args.study)
    if args.register:
        register_plan(study, args.plan)
    plan = load_plan(args.plan)
    report = readout_readiness(study, plan)
    if args.output:
        publish_json(args.output, report)
    print(json.dumps(report, indent=2))
