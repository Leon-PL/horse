"""Matchbook paper record of the 11:00 rule (read-only: nothing here can place a bet).

    python -m src.matchbook configure --win-run RUN_ID [--correction data/matchbook/model]
    python -m src.matchbook fit --as-of DATE [--output data/matchbook/model] [--threads 6]
    python -m src.matchbook snapshot [--label manual]
    python -m src.matchbook watch
    python -m src.matchbook eleven [--date DATE] [--no-snapshot]
    python -m src.matchbook settle [--date DATE]          # default: every unsettled day
    python -m src.matchbook summary
    python -m src.matchbook day [--refresh]               # the whole day; for the scheduled task

``day`` settles earlier days, takes the 08:00 snapshot, optionally refreshes results, the feature store and the
forecast history, makes the 11:00 forecast and paper bets, then records near-off prices until the last race.
"""
import argparse
import json
import subprocess
import sys
import time
import traceback

import pandas as pd

from src.ingestion.storage import atomic_json
from src.matchbook import model as mm
from src.matchbook import paper, record

LOGS = record.ROOT / "logs"


def log(message):
    stamp = record.uk_now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"{stamp[:10]}.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sleep_until(clock):
    now = record.uk_now()
    target = pd.Timestamp(f"{now.date()} {clock}").tz_localize(record.LONDON)
    if target > now:
        log(f"waiting until {clock}")
        time.sleep((target - now).total_seconds())


def settle_all():
    for day in paper.unsettled_days():
        try:
            rows = paper.settle(day)
            log(f"settle {day}: " + ("Betfair SP file not published yet" if rows is None else f"{len(rows)} bets"))
        except Exception as exc:
            log(f"settle {day} failed: {exc!r}")


REFRESH_DEADLINE = "12:00"


def refresh_steps(win_run):
    """Results, store and forecast history, one after another, in a child process."""
    code = ("import subprocess, sys\n"
            f"steps = {[['-m', 'src.ingestion', '--sync', '3', '--results-only'], ['-c', 'from src.store_refresh import get_training_dataset; get_training_dataset()'], ['-m', 'src.correction', 'extend', '--run', win_run]]!r}\n"
            "for step in steps:\n"
            "    print('step', ' '.join(step), flush=True)\n"
            "    print('exit', subprocess.call([sys.executable, *step]), flush=True)\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / f"{record.uk_now().date()}_refresh.log", "a", encoding="utf-8")
    log("refresh started: results sync, store append, forecast-history extend")
    return subprocess.Popen([sys.executable, "-c", code], stdout=out, stderr=subprocess.STDOUT)


def stay_awake(on=True):
    """Keep Windows from sleeping while the day runs (a timer wake otherwise sleeps again within minutes)."""
    try:
        import ctypes
        continuous, system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(continuous | (system_required if on else 0))
    except (AttributeError, OSError):
        pass


def day(args):
    stay_awake()
    log("paper day started")
    settle_all()
    if record.uk_now().strftime("%H:%M") < "08:00":
        sleep_until("08:00")
    try:
        path, frame = record.snapshot("0800")
        log(f"snapshot {path} ({len(frame)} runners)")
    except Exception as exc:
        log(f"08:00 snapshot failed: {exc!r}")
    worker = refresh_steps(paper.config()["win_run"]) if args.refresh else None
    sleep_until(paper.RULE["clock"])
    try:
        path, frame = record.snapshot("1100")
        log(f"snapshot {path} ({len(frame)} runners)")
    except Exception as exc:
        log(f"11:00 snapshot failed: {exc!r}")
    while worker is not None and worker.poll() is None and record.uk_now().strftime("%H:%M") < REFRESH_DEADLINE:
        time.sleep(30)
    if worker is not None:
        log("refresh " + (f"finished, exit {worker.returncode}" if worker.poll() is not None
                          else f"still running at {REFRESH_DEADLINE}; forecasting without it"))
    try:
        run = paper.eleven(snapshot=False)
        log("eleven: " + json.dumps({k: run.get(k) for k in ("status", "eligible_races", "bets")} if run else {}))
    except Exception:
        log("eleven failed: " + traceback.format_exc().strip().splitlines()[-1])
        atomic_json(paper.day_dir(record.uk_now().date()) / "error.json",
                    {"stage": "eleven", "error": traceback.format_exc(), "at": str(record.uk_now())})
    record.watch(log=log)
    log("paper day finished")
    stay_awake(False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("configure")
    c.add_argument("--win-run", required=True)
    c.add_argument("--correction", default=str(record.ROOT / "model"))
    f = commands.add_parser("fit")
    f.add_argument("--as-of", required=True)
    f.add_argument("--output", default=str(record.ROOT / "model"))
    f.add_argument("--threads", type=int, default=6)
    commands.add_parser("snapshot").add_argument("--label", default="manual")
    commands.add_parser("watch")
    e = commands.add_parser("eleven")
    e.add_argument("--date", default=None)
    e.add_argument("--no-snapshot", action="store_true")
    commands.add_parser("settle").add_argument("--date", default=None)
    commands.add_parser("summary")
    commands.add_parser("day").add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "configure":
        atomic_json(paper.CONFIG, {"win_run": args.win_run, "correction": args.correction})
        print(paper.CONFIG.read_text())
    elif args.command == "fit":
        model = mm.fit(args.as_of, args.output, threads=args.threads)
        print(json.dumps({k: v for k, v in model.lineage.items() if k != "params"}, indent=2, default=str))
    elif args.command == "snapshot":
        path, frame = record.snapshot(args.label)
        print(f"{path}: {len(frame)} runners")
    elif args.command == "watch":
        record.watch()
    elif args.command == "eleven":
        print(json.dumps(paper.eleven(args.date, snapshot=not args.no_snapshot), indent=2, default=str))
    elif args.command == "settle":
        if args.date:
            rows = paper.settle(args.date)
            print("not yet settleable" if rows is None else f"{len(rows)} bets settled")
        else:
            settle_all()
    elif args.command == "summary":
        print(json.dumps(paper.summary(), indent=2, default=float))
    elif args.command == "day":
        day(args)


if __name__ == "__main__":
    main()
