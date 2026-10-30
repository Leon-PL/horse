"""Run bounded continuous capture for a pinned forward study."""
import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.forward_store import ForwardStudy
from src.forward_recorder import run_recorder
from src.matchbook_client import MatchbookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--duration-minutes", type=float, default=1080)
    parser.add_argument("--refresh-history", action="store_true")
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    args.study.mkdir(parents=True, exist_ok=True)
    logs = args.study/"recorder_logs"
    logs.mkdir(exist_ok=True)
    # Durable logs also work when Task Scheduler starts a hidden process.
    with open(logs/(datetime.now().strftime("%Y%m%dT%H%M%S%f")+".log"), "a", encoding="utf-8", buffering=1) as log:
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(log), redirect_stderr(log):
            import logging
            logging.basicConfig(level=logging.INFO, stream=log, force=True)
            return run_recorder(ForwardStudy(args.study), client=MatchbookClient(),
                duration_minutes=args.duration_minutes, refresh=args.refresh_history, discover_only=args.discover_only)


if __name__ == "__main__":
    main()
