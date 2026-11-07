"""Compare two featured frames or stores column by column.

Restored from the retired ``scripts/compare_feature_stores.py`` (commit
``26d1b6f``), which proved the 6 November store build byte-identical. Floats are
compared bit for bit first; a value that differs is then measured against
``--tolerance`` (absolute, and relative to the larger value), so a difference
in the last bits of a double shows as "within tolerance", never as "equal".

Run:
  .venv/Scripts/python.exe -m research.fe_consolidation_065.compare LEFT.parquet RIGHT.parquet \
      --report data/research/fe_consolidation_065/compare.json [--columns feature_sets/win_selected_056.txt]

Exits non-zero when a shared column differs beyond the tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEYS = ("race_id", "horse_id", "race_date")


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
        return column.to_numpy(zero_copy_only=False).astype("float64")
    if pa.types.is_timestamp(kind) or pa.types.is_date(kind):
        return column.cast(pa.timestamp("ns")).cast(pa.int64()).to_numpy(zero_copy_only=False).astype("float64")
    return np.asarray(column.to_pylist(), dtype=object)


def compare_arrays(left: np.ndarray, right: np.ndarray, tolerance: float) -> dict:
    """Counts of exact matches, tolerance-only matches and real differences."""
    n = len(left)
    if left.dtype == object or right.dtype == object:
        same = np.fromiter((a == b or (a is None and b is None) or (a != a and b != b)
                            for a, b in zip(left, right)), dtype=bool, count=n)
        return {"rows": n, "equal": int(same.sum()), "within_tolerance": 0, "different": int(n - same.sum()),
                "first_different_row": int(np.argmin(same)) if not same.all() else None}
    a, b = left.astype("float64"), right.astype("float64")
    a_nan, b_nan = np.isnan(a), np.isnan(b)
    with np.errstate(invalid="ignore"):
        same = (a == b) | (a_nan & b_nan)
    pattern = a_nan != b_nan
    finite = ~(same | pattern)
    within = np.zeros(n, dtype=bool)
    max_abs = max_rel = 0.0
    if finite.any():
        with np.errstate(invalid="ignore", over="ignore"):
            gap = np.abs(a[finite] - b[finite])
            rel = gap / np.maximum(np.maximum(np.abs(a[finite]), np.abs(b[finite])), 1.0)
        ok = np.nan_to_num(rel, nan=np.inf) <= tolerance
        within[np.flatnonzero(finite)[ok]] = True
        max_abs = float(np.nanmax(gap))
        max_rel = float(np.nanmax(rel))
    different = ~(same | within)
    return {"rows": n, "equal": int(same.sum()), "within_tolerance": int(within.sum()),
            "different": int(different.sum()), "null_pattern_mismatch": int(pattern.sum()),
            "max_abs_gap": max_abs, "max_rel_gap": max_rel,
            "first_different_row": int(np.argmax(different)) if different.any() else None}


def compare_parquet(left_path, right_path, *, tolerance=1e-9, columns=None, group=64) -> dict:
    left, right = pq.ParquetFile(left_path), pq.ParquetFile(right_path)
    ls, rs = left.schema_arrow, right.schema_arrow
    right_names, left_names = set(rs.names), set(ls.names)
    wanted = None if columns is None else list(columns)
    report: dict = {
        "left": str(left_path), "right": str(right_path),
        "left_rows": left.metadata.num_rows, "right_rows": right.metadata.num_rows,
        "left_columns": len(ls.names), "right_columns": len(rs.names),
        "same_column_order": ls.names == rs.names,
        "only_left": [n for n in ls.names if n not in right_names],
        "only_right": [n for n in rs.names if n not in left_names],
    }
    shared = [n for n in ls.names if n in right_names]
    if wanted is not None:
        report["requested_missing_left"] = [c for c in wanted if c not in left_names]
        report["requested_missing_right"] = [c for c in wanted if c not in right_names]
        shared = [c for c in wanted if c in left_names and c in right_names]
        shared = list(dict.fromkeys([k for k in KEYS if k in left_names and k in right_names] + shared))
    report["type_mismatches"] = {n: [str(ls.field(n).type), str(rs.field(n).type)]
                                 for n in shared if ls.field(n).type != rs.field(n).type}
    if report["left_rows"] != report["right_rows"]:
        report["same_rows"] = False
        return report
    found: dict = {}
    for names in _chunks(shared, group):
        lt, rt = left.read(columns=names), right.read(columns=names)
        for name in names:
            found[name] = compare_arrays(_values(lt.column(name)), _values(rt.column(name)), tolerance)
    report["columns"] = found
    report["row_keys_equal"] = {k: found[k]["different"] == 0 for k in KEYS if k in found}
    report["same_rows"] = all(report["row_keys_equal"].values())
    report["exact_columns"] = sum(1 for c in found.values() if c["equal"] == c["rows"])
    report["tolerance_only_columns"] = sorted(n for n, c in found.items()
                                              if c["different"] == 0 and c["within_tolerance"] > 0)
    report["different_columns"] = sorted(n for n, c in found.items() if c["different"] > 0)
    return report


def identical(report: dict, *, subset: bool = False) -> bool:
    if not report.get("same_rows") or report.get("type_mismatches") or report.get("different_columns"):
        return False
    if subset:
        return not report.get("requested_missing_left") and not report.get("requested_missing_right")
    return bool(report.get("same_column_order") and not report.get("only_left") and not report.get("only_right"))


def _names(path):
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--report")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--columns", help="feature-set file; compare only these (plus row keys)")
    args = parser.parse_args(argv)
    columns = _names(args.columns) if args.columns else None
    report = compare_parquet(args.left, args.right, tolerance=args.tolerance, columns=columns)
    same = identical(report, subset=columns is not None)
    summary = {k: report.get(k) for k in ("left_rows", "right_rows", "left_columns", "right_columns",
                                          "same_rows", "exact_columns", "tolerance_only_columns",
                                          "different_columns", "type_mismatches",
                                          "requested_missing_left", "requested_missing_right")}
    summary["only_left"] = len(report.get("only_left", []))
    summary["only_right"] = len(report.get("only_right", []))
    print(json.dumps(summary, indent=1, default=str))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("IDENTICAL" if same else "DIFFERENT")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
