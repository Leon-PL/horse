"""Create immutable T-5 forecast research inputs from local Betfair archives.

Writes a new research snapshot, never the canonical BSP or feature store.
All prices and field membership are from messages at/before the decision.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.market_replay import ReplayRejected, replay_market, split_market_streams


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Choose a new output path; research snapshots are immutable")
    rows, counts, sources = [], Counter(), []
    started = time.monotonic()
    for path in args.tar:
        sources.append({"path": str(Path(path).resolve()), "sha256": digest(path)})
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".bz2"):
                    continue
                counts["streams"] += 1
                try:
                    raw = bz2.decompress(archive.extractfile(member).read())
                    candidates = 0
                    for md, market in split_market_streams(raw):
                        candidates += 1
                        counts["eligible_market_streams"] += 1
                        if args.start and md.get("marketTime", "")[:10] < args.start:
                            counts["before_requested_start"] += 1
                            continue
                        if args.end and md.get("marketTime", "")[:10] > args.end:
                            counts["after_requested_end"] += 1
                            continue
                        try:
                            replay = replay_market(market, seconds_before=args.seconds_before)
                        except ReplayRejected as exc:
                            counts[str(exc)] += 1
                            continue
                        for row in replay:
                            row["archive_member"] = member.name
                        rows.extend(replay)
                        counts["accepted_market_streams"] += 1
                    if not candidates:
                        counts["files_without_gb_ie_win"] += 1
                except ReplayRejected as exc:
                    counts[str(exc)] += 1
                except (OSError, ValueError, EOFError):
                    counts["unreadable_stream"] += 1
                if counts["streams"] % 500 == 0:
                    print(json.dumps(dict(counts)), flush=True)
    if not rows:
        raise ValueError(f"No eligible markets: {dict(counts)}")
    frame = pd.DataFrame(rows)
    keys = ["market_id", "selection_id"]
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        # Archive overlap is allowed only when every forecast/outcome agrees.
        values = [c for c in frame if c not in keys + ["archive_member"]]
        varying = frame.loc[duplicate].groupby(keys, dropna=False)[values].nunique(dropna=False)
        if varying.gt(1).any().any():
            raise ValueError("Conflicting overlapping markets; inspect sources rather than picking the last one")
    frame = frame.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    report = {
        "protocol": "First published schedule minus fixed horizon; active field and LTP known by decision; final definition for labels only",
        "scope": "As-issued market forecast; conditional no-late-withdrawal subset required for existing final-starter feature store",
        "price_limitation": "BASIC last traded price, not an executable quote",
        "seconds_before": args.seconds_before, "sources": sources, "counts": dict(counts),
        "races": int(frame.market_id.nunique()), "runners": len(frame),
        "late_withdrawal_races": int(frame.loc[frame.withdrawn_after_decision, "market_id"].nunique()),
        "output": str(output.resolve()), "output_sha256": digest(output),
        "code_sha256": {p: digest(ROOT / p) for p in (
            "src/market_replay.py", "scripts/replay_market_snapshots.py", "scripts/extract_betfair_bsp.py",
            "src/betfair.py", "src/utils.py")},
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds-before", type=int, default=300)
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
