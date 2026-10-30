"""Report strict rebuild differences without changing either store."""
from pathlib import Path
import json
import sys
import gc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.feature_store import FeatureStore


def exact_state_values(a, b):
    """Zero-tolerance value comparison, including NaNs and large integers.

    Compare numeric/list payloads directly instead of repeatedly pickling tens
    of millions of scalars for joblib hashes. Object arrays preserve integers;
    they never coerce nanosecond dates or large IDs to floating point.
    """
    if a is b:
        return True
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(exact_state_values(value, b[key]) for key, value in a.items()))
    if isinstance(a, (set, frozenset)):
        return type(a) is type(b) and a == b
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape:
            return False
        x, y = a, b
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False
        try:
            x, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        except (ValueError, TypeError):
            return all(exact_state_values(x, y) for x, y in zip(a, b))
    else:
        try:
            return bool(a == b) or bool(pd.isna(a) and pd.isna(b))
        except (ValueError, TypeError):
            return False
    try:
        if x.shape == y.shape and np.all((x == y) | (pd.isna(x) & pd.isna(y))):
            return True
    except (ValueError, TypeError):
        pass
    return (isinstance(a, (list, tuple))
            and all(exact_state_values(x, y) for x, y in zip(a, b)))


# Guard the comparison itself against exactly the losses this audit must catch.
assert exact_state_values({"v": [1.0, float("nan")]}, {"v": [1.0, float("nan")]})
assert not exact_state_values([1.0], [np.nextafter(1.0, np.inf)])
assert not exact_state_values([2**62+1], [2**62+2])
assert not exact_state_values([(2**62+1, np.nan)], [(2**62+2, np.nan)])
assert not exact_state_values(np.array([1]), np.array([1.0]))

OUT = ROOT / "data/audit/live_feature_speed_20261029/installation"
old = FeatureStore(str(OUT / "previous_store"))
new = FeatureStore(str(OUT / "rebuilt_store"))
meta = json.loads((OUT / "parent_meta.json").read_text())
selected = set(meta["training_config"]["recipe"]["feature_allowlist"])
columns = sorted(pq.read_schema(old.featured_path).names)
frames = [pd.read_parquet(store.featured_path, columns=columns,
                         filters=[("race_date", ">=", pd.Timestamp("2026-07-30"))]) for store in (old, new)]
a, b = frames
pd.testing.assert_frame_equal(a[["race_id", "horse_name"]], b[["race_id", "horse_name"]], check_exact=True)
differences = []
for col in columns:
    try:
        pd.testing.assert_series_equal(a[col], b[col], check_exact=True)
    except AssertionError:
        missing_equal = np.array_equal(a[col].isna(), b[col].isna())
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x, y = a[col].to_numpy(float), b[col].to_numpy(float)
            neq = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            maximum = float(np.nanmax(np.abs(x - y)))
            float32 = bool(np.array_equal(x.astype("float32"), y.astype("float32"), equal_nan=True))
        else:
            neq = ~(a[col].eq(b[col]) | (a[col].isna() & b[col].isna()))
            maximum, float32 = None, None
        differences.append({"column": col, "selected": col in selected, "different_rows": int(np.sum(neq)),
            "max_error": maximum, "float32_exact": float32, "missingness_equal": missing_equal,
            "before_dtype": str(a[col].dtype), "after_dtype": str(b[col].dtype),
            "first_date": str(a.loc[neq, "race_date"].min()), "last_date": str(a.loc[neq, "race_date"].max())})
result = {"scope": "Strict diagnostic only; neither store published or changed", "late_rows": len(a),
          "differences": differences, "selected_columns_exact": not any(v["selected"] for v in differences)}
print(json.dumps(result, indent=2), flush=True)
del frames, a, b
gc.collect()
state = {}
for name, loader in (("ratings", "load_state"), ("counts", "load_counts_state")):
    a, b = getattr(old, loader)(), getattr(new, loader)()
    changed = [str(key) for key in a if key not in b or not exact_state_values(a[key], b[key])]
    state[name] = {"before_families": len(a), "after_families": len(b), "different_families": changed}
    print(name + ": " + json.dumps(state[name]), flush=True)
    del a, b
    gc.collect()
result["state"] = state
(OUT / "raw_rebuild_differences.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
