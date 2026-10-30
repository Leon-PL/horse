"""Create a pinned study, discover races, or record one current public forecast."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forward_store import ForwardStudy, create_study
from src.forward_capture import capture_event, discover_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--event", help="exact Matchbook event ID; capture current feeds once")
    parser.add_argument("--model", help="saved run to pin at creation; defaults to production")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--at-window", action="store_true", help="one manual attempt: compute now, then wait up to ten minutes before fresh final reads")
    args = parser.parse_args()
    if args.at_window and not args.event:
        parser.error("--at-window requires --event")
    if args.create:
        study = create_study(args.study, model_id=args.model)
        print(json.dumps({"study": str(study.path), "model": study.protocol["model_id"],
            "first_eligible_date": study.protocol["first_eligible_date"],
            "scope": study.protocol["assessment_scope"]}, indent=2))
    elif args.list:
        result = discover_events(ForwardStudy(args.study), hours=args.hours)
        print(json.dumps(result, indent=2))
    else:
        result = capture_event(ForwardStudy(args.study), args.event, wait_for_issue_window=args.at_window)
        print(json.dumps(result, indent=2))
        return 0 if result["paired_available"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
