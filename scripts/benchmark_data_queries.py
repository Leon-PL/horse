"""Compare equivalent filtered Parquet summaries in fresh, bounded workers.

Both engines select the same columns and date predicate. This compares with
the existing pandas/PyArrow filtered read, not an unnecessarily full read.
Workers alternate order to reduce warm-cache bias; no model is fitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import duckdb
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def summarize_pandas(path: Path, since: str, features: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["race_date", "race_id", *features],
                            filters=[("race_date", ">=", pd.Timestamp(since))])
    return frame.groupby("race_date", sort=True).agg(
        runners=("race_id", "size"),
        **{f"mean_{name}": (name, "mean") for name in features},
    ).reset_index()


def summarize_duckdb(path: Path, since: str, features: list[str], *, threads=2) -> pd.DataFrame:
    means = "".join(f", avg({_quoted(name)}) AS {_quoted('mean_' + name)}"
                    for name in features)
    with duckdb.connect(config={"threads": threads, "memory_limit": "512MB"}) as conn:
        return conn.execute(
            f"SELECT race_date, count(*) AS runners{means} "
            "FROM read_parquet(?) WHERE race_date >= ? "
            "GROUP BY race_date ORDER BY race_date",
            [str(path), pd.Timestamp(since).to_pydatetime()],
        ).fetchdf()


def assert_equivalent(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left, right = left.copy(), right.copy()
    for frame in (left, right):
        frame["race_date"] = pd.to_datetime(frame.race_date)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False,
                                  rtol=1e-12, atol=1e-12)


def _worker(args) -> dict:
    pa.set_cpu_count(args.threads)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    try:
        if args.worker == "pandas":
            frame = summarize_pandas(args.source, args.since, args.feature)
        else:
            frame = summarize_duckdb(args.source, args.since, args.feature,
                                      threads=args.threads)
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    return {"engine": args.worker, "seconds": elapsed,
            "sampled_peak_rss_mb": peak[0] / (1024 ** 2),
            "result": json.loads(frame.to_json(orient="split", date_format="iso",
                                                double_precision=15))}


def _since(path: Path, days: int) -> str:
    with pq.ParquetFile(path) as source:
        index = source.schema.names.index("race_date")
        maxima = [source.metadata.row_group(i).column(index).statistics
                  for i in range(source.metadata.num_row_groups)]
        dates = [s.max for s in maxima if s is not None and s.has_min_max]
    if not dates:
        raise ValueError("Date statistics are required; supply --since explicitly")
    return (pd.Timestamp(max(dates)) - pd.Timedelta(days=days - 1)).date().isoformat()


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def benchmark(args) -> dict:
    if args.output.exists():
        raise FileExistsError("Choose a new --output path; preserve earlier benchmark reports")
    before = _identity(args.source)
    since = args.since or _since(args.source, args.days)
    schema = pq.read_schema(args.source)
    if not args.feature:
        selected = (ROOT / "feature_sets/win.txt").read_text().splitlines()
        args.feature = [name.strip() for name in selected
                        if name.strip() in schema.names
                        and pa.types.is_floating(schema.field(name.strip()).type)][:4]
    if not args.feature or len(set(args.feature)) != len(args.feature):
        raise ValueError("Select at least one unique numeric --feature")
    for name in args.feature:
        field = schema.field(name)
        if not (pa.types.is_floating(field.type) or pa.types.is_integer(field.type)):
            raise ValueError(f"Not a numeric feature: {name}")
    observations, reference = [], None
    for repeat in range(args.repeats):
        engines = ["pandas", "duckdb"] if repeat % 2 == 0 else ["duckdb", "pandas"]
        for engine in engines:
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--worker", engine, "--source", str(args.source.resolve()),
                       "--since", since, "--threads", str(args.threads)]
            for name in args.feature:
                command.extend(["--feature", name])
            environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                               OPENBLAS_NUM_THREADS=str(args.threads))
            completed = subprocess.run(command, capture_output=True, text=True,
                                       check=True, env=environment)
            value = json.loads(completed.stdout)
            payload = value.pop("result")
            result = pd.DataFrame(payload["data"], columns=payload["columns"])
            if result.empty:
                raise ValueError("Benchmark date window has no rows")
            if reference is None:
                reference = result
            assert_equivalent(reference, result)
            observations.append({"repeat": repeat + 1, **value})
    if _identity(args.source) != before:
        raise RuntimeError("The source changed during the benchmark; do not compare these runs")
    summary = {}
    for engine in ("pandas", "duckdb"):
        values = [v for v in observations if v["engine"] == engine]
        summary[engine] = {
            "median_seconds": statistics.median(v["seconds"] for v in values),
            "median_sampled_peak_rss_mb": statistics.median(v["sampled_peak_rss_mb"] for v in values),
        }
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": before, "since_inclusive": since, "features": args.feature,
        "threads": args.threads, "repeats_per_engine": args.repeats,
        "results_equivalent": True, "output_date_groups": len(reference),
        "queried_runners": int(reference.runners.sum()),
        "summary": summary, "observations": observations,
        "duckdb_speed_ratio": summary["pandas"]["median_seconds"] / summary["duckdb"]["median_seconds"],
        "scope": "One read-only summary query; not training or full feature generation",
        "measurement": "Fresh workers, alternating order; 5ms sampled whole-process RSS including imports. OS file cache is not cleared.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/processed/feature_store/featured.parquet")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/audit/library_consolidation_20261029/duckdb_benchmark.json")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--since")
    parser.add_argument("--feature", action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--worker", choices=("pandas", "duckdb"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.days, args.repeats, args.threads) < 1:
        parser.error("days, repeats and threads must be positive")
    print(json.dumps(_worker(args) if args.worker else benchmark(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
