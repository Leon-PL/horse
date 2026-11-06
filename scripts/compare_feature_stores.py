"""Prove two feature stores hold the same rows, column by column.

Compares the schema, row count, row order and every column of two
``featured.parquet`` files, plus ``processed.parquet``. Columns are read in
small groups, so a full store fits in memory. Floats are compared bit for bit
first; a value that differs is then measured against ``--tolerance``, which is
absolute and relative to the larger value, so a difference in the last bits of
a double shows up as "within tolerance" and never as "equal".

The carried rating and counts state decides every later append, so it must
match exactly: the tolerance does not apply to it.

Run:
  .venv/Scripts/python.exe scripts/compare_feature_stores.py LEFT_ROOT RIGHT_ROOT \
      --report data/audit/compare.json

Exits non-zero when any column differs beyond the tolerance, or the state differs at all.
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEYS = ("race_id", "horse_id", "horse_name", "race_date")


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _values(column: pa.ChunkedArray) -> np.ndarray:
    """A numpy view that keeps nulls: NaN for numbers, None for everything else."""
    kind = column.type
    if pa.types.is_floating(kind):
        return column.to_numpy(zero_copy_only=False).astype("float64")
    if pa.types.is_integer(kind) or pa.types.is_boolean(kind):
        if column.null_count:
            return column.cast(pa.float64()).to_numpy(zero_copy_only=False)
        return column.to_numpy(zero_copy_only=False)
    if pa.types.is_timestamp(kind) or pa.types.is_date(kind):
        return column.cast(pa.timestamp("ns")).cast(pa.int64()).to_numpy(zero_copy_only=False).astype("float64")
    return np.asarray(column.to_pylist(), dtype=object)


def compare_arrays(left: np.ndarray, right: np.ndarray, tolerance: float) -> dict:
    """Counts of exact matches, tolerance-only matches and real differences."""
    n = len(left)
    numeric = left.dtype != object and right.dtype != object
    if not numeric:
        same = np.fromiter((a == b or (a is None and b is None) for a, b in zip(left, right)), dtype=bool, count=n)
        return {"rows": n, "equal": int(same.sum()), "within_tolerance": 0, "different": int(n - same.sum()),
                "first_different_row": int(np.argmin(same)) if not same.all() else None}
    a, b = left.astype("float64"), right.astype("float64")
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    both_nan = a_nan & b_nan
    with np.errstate(invalid="ignore"):
        same = (a == b) | both_nan
    pattern = a_nan != b_nan
    finite = ~(same | pattern)
    within = np.zeros(n, dtype=bool)
    max_abs = max_rel = 0.0
    if finite.any():
        with np.errstate(invalid="ignore", over="ignore"):
            gap = np.abs(a[finite] - b[finite])
            scale = np.maximum(np.maximum(np.abs(a[finite]), np.abs(b[finite])), 1.0)
            rel = gap / scale
        ok = np.nan_to_num(rel, nan=np.inf) <= tolerance
        within[np.flatnonzero(finite)[ok]] = True
        max_abs = float(np.nanmax(gap)) if len(gap) else 0.0
        max_rel = float(np.nanmax(rel)) if len(rel) else 0.0
    different = ~(same | within)
    return {"rows": n, "equal": int(same.sum()), "within_tolerance": int(within.sum()),
            "different": int(different.sum()), "null_pattern_mismatch": int(pattern.sum()),
            "max_abs_gap": max_abs, "max_rel_gap": max_rel,
            "first_different_row": int(np.argmax(different)) if different.any() else None}


def compare_parquet(left_path, right_path, *, tolerance=1e-9, group=48, row_keys=KEYS) -> dict:
    left, right = pq.ParquetFile(left_path), pq.ParquetFile(right_path)
    ls, rs = left.schema_arrow, right.schema_arrow
    only_left = [n for n in ls.names if n not in set(rs.names)]
    only_right = [n for n in rs.names if n not in set(ls.names)]
    shared = [n for n in ls.names if n in set(rs.names)]
    report: dict = {
        "left": str(left_path), "right": str(right_path),
        "left_rows": left.metadata.num_rows, "right_rows": right.metadata.num_rows,
        "left_columns": len(ls.names), "right_columns": len(rs.names),
        "same_column_order": ls.names == rs.names,
        "only_left": only_left, "only_right": only_right,
        "type_mismatches": {n: [str(ls.field(n).type), str(rs.field(n).type)]
                            for n in shared if ls.field(n).type != rs.field(n).type},
        "row_groups": [left.metadata.num_row_groups, right.metadata.num_row_groups],
    }
    if report["left_rows"] != report["right_rows"]:
        report["same_rows"] = False
        return report
    columns: dict = {}
    for names in _chunks(shared, group):
        lt, rt = left.read(columns=names), right.read(columns=names)
        for name in names:
            columns[name] = compare_arrays(_values(lt.column(name)), _values(rt.column(name)), tolerance)
    report["columns"] = columns
    report["row_keys_equal"] = {k: columns[k]["different"] == 0 for k in row_keys if k in columns}
    report["exact_columns"] = sum(1 for c in columns.values() if c["equal"] == c["rows"])
    report["tolerance_only_columns"] = sorted(n for n, c in columns.items()
                                              if c["different"] == 0 and c["within_tolerance"] > 0)
    report["different_columns"] = sorted(n for n, c in columns.items() if c["different"] > 0)
    report["same_rows"] = all(report["row_keys_equal"].values())
    return report


def identical(report: dict) -> bool:
    return bool(report.get("same_rows") and report.get("same_column_order") and not report.get("only_left")
                and not report.get("only_right") and not report.get("type_mismatches")
                and not report.get("different_columns"))


def state_differences(left, right, *, limit=20) -> list[str]:
    """Where two carried states differ. Python and numpy numbers may mix."""
    found: list[str] = []
    stack = [(left, right, "state")]
    while stack and len(found) < limit:
        a, b, at = stack.pop()
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                found.append(f"{at}: {len(a.keys() ^ b.keys())} keys in only one state")
            else:
                stack.extend((a[key], b[key], f"{at}[{key!r}]") for key in a)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                found.append(f"{at}: lengths {len(a)} and {len(b)}")
            else:
                stack.extend((x, y, f"{at}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            x, y = np.asarray(a), np.asarray(b)
            if not np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f"):
                found.append(f"{at}: arrays differ")
        elif a != b and not (a != a and b != b):
            found.append(f"{at}: {a!r} and {b!r}")
    return found


def _load_state(path: Path):
    """Read a carried state with collection off: at 14 million entries,
    collection took a load from 39 seconds to six minutes."""
    gc.disable()
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    finally:
        gc.enable()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    started = time.perf_counter()
    reports = {}
    for name in ("featured.parquet", "processed.parquet"):
        reports[name] = compare_parquet(Path(args.left) / name, Path(args.right) / name, tolerance=args.tolerance)
    states = {name: state_differences(_load_state(Path(args.left) / name), _load_state(Path(args.right) / name))
              for name in ("ratings_state.pkl", "counts_state.pkl")}
    ok = all(identical(r) for r in reports.values()) and not any(states.values())
    summary = {name: {k: v for k, v in r.items() if k != "columns"} for name, r in reports.items()}
    print(json.dumps({"identical": ok, "seconds": round(time.perf_counter() - started, 1), **summary,
                      "state_differences": states}, indent=1))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({**reports, "state_differences": states}, indent=1),
                                     encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
