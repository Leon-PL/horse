"""Measure real retained markets; production databases are opened read-only.

Only isolated temporary queues are written. The JSON report retains timings,
market types and output-equivalence results, never account settings or payloads.
"""
import argparse
from contextlib import closing
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import betfair_history as history  # noqa: E402
from src.ingestion.betfair_archives import payloads  # noqa: E402


def measure(action):
    started = time.perf_counter()
    result = action()
    return result, time.perf_counter()-started


def benchmark(root, output, limit=16, baseline_source=None):
    root, output = Path(root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect((root/"queue.sqlite").resolve().as_uri()+"?mode=ro", uri=True)) as con:
        sample = con.execute("SELECT remote,first_date,last_date,sha256,rows_json FROM files "
                             "WHERE state IN ('downloaded','published') ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        _, counts_seconds = measure(lambda: list(con.execute("SELECT state,COUNT(*) FROM files GROUP BY state")))
    if not sample:
        raise RuntimeError("No retained parsed markets are available for comparison")
    records, stages = [], []
    for remote, first, last, digest, expected in sample:
        market_id = PurePosixPath(remote).name.removesuffix(".bz2")
        target, path_seconds = measure(lambda: history.local_path(root, remote))
        if target.exists():
            body, lookup_seconds = measure(target.read_bytes)
        else:
            candidates, lookup_seconds = measure(lambda: list(payloads(root, market_id)))
            body = next(body for sha, body in candidates if sha == digest)
        (market, rows), parse_seconds = measure(lambda: history.parse_download(
            body, remote, date.fromisoformat(first), date.fromisoformat(last)))
        if rows != json.loads(expected):
            raise AssertionError("Replayed market output differs from its accepted queue observations")
        records.append((remote, body, market, rows))
        stages.append({"market_id": market_id, "market": market, "bytes": len(body), "runners": len(rows),
                       "path_seconds": path_seconds, "lookup_seconds": lookup_seconds,
                       "parse_seconds": parse_seconds})
    first = min(row[1] for row in sample)
    last = max(row[2] for row in sample)
    baseline = history
    if baseline_source:
        spec = importlib.util.spec_from_file_location("_baseline_betfair_history", baseline_source)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
    with tempfile.TemporaryDirectory(prefix="queues_", dir=output.parent) as temporary:
        one = baseline.HistoryStore(Path(temporary)/"per_file")
        bulk = history.HistoryStore(Path(temporary)/"bulk")
        for store in (one, bulk):
            store.discover([row[0] for row in records], first, last)
        def per_file():
            for remote, body, market, rows in records:
                one.save(remote, body, market, rows, retain_file=False)
        _, per_file_seconds = measure(per_file)
        encoded, encoding_seconds = measure(lambda: [
            (market, json.dumps(rows), hashlib.sha256(body).hexdigest(), len(body), history.now(), remote)
            for remote, body, market, rows in records])
        def batched():
            with bulk.connection() as con:
                con.executemany("UPDATE files SET state='downloaded',market=?,rows_json=?,sha256=?,bytes=?,"
                                "error=NULL,updated_at=? WHERE remote=?", encoded)
        _, batch_seconds = measure(batched)
        def observations(store):
            with store.connection() as con:
                return list(con.execute("SELECT remote,state,market,rows_json,sha256,bytes FROM files ORDER BY remote"))
        if observations(one) != observations(bulk):
            raise AssertionError("Batching changed retained observations or payload hashes")
        optimized_seconds = None
        raw_seconds = None
        if hasattr(history.HistoryStore, "save_many"):
            optimized = history.HistoryStore(Path(temporary)/"optimized")
            optimized.discover([row[0] for row in records], first, last)
            entries = [{"remote": remote, "payload": body, "market": market, "rows": rows, "retain_file": False}
                       for remote, body, market, rows in records]
            _, optimized_seconds = measure(lambda: optimized.save_many(entries))
            if observations(one) != observations(optimized):
                raise AssertionError("Optimized checkpoint output differs from the legacy writer")
            optimized_raw = history.HistoryStore(Path(temporary)/"optimized_raw")
            optimized_raw.discover([row[0] for row in records], first, last)
            retained = [{**entry, "retain_file": True} for entry in entries]
            _, raw_seconds = measure(lambda: optimized_raw.save_many(retained))
            if observations(one) != observations(optimized_raw):
                raise AssertionError("Raw retention changed checkpoint observations")
            if not all(history.local_path(optimized_raw.root, item["remote"]).read_bytes() == item["payload"]
                       for item in retained):
                raise AssertionError("Retained compressed source bytes differ")
    report = {"markets": len(records), "counts_seconds": counts_seconds, "stages": stages,
              "per_file_write_seconds": per_file_seconds, "encoding_seconds": encoding_seconds,
              "batch_write_seconds": batch_seconds, "observations_exact": True,
              "optimized_write_seconds": optimized_seconds, "baseline_source": str(baseline_source),
              "optimized_with_raw_seconds": raw_seconds,
              "production_data_modified": False}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/betfair/history")
    parser.add_argument("--markets", type=int, default=16)
    parser.add_argument("--out", default="data/audit/betfair_ingestion_benchmark.json")
    parser.add_argument("--baseline-source", help="Trusted local source snapshot for the legacy writer")
    args = parser.parse_args()
    if not 1 <= args.markets <= 1000:
        parser.error("Choose between 1 and 1000 sample markets")
    print(json.dumps(benchmark(args.root, args.out, args.markets, args.baseline_source), indent=2))


if __name__ == "__main__":
    main()
