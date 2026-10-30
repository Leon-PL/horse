"""Compare selected model inputs in two stores with identical runner order.

Streams Arrow batches; never loads two full feature stores or fits a model.
Different row-group boundaries are supported. Different runner order fails
explicitly rather than comparing unrelated horses.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from src.deployment_validation import file_sha256


def compare(before, after, features):
    keys = ["race_id", "horse_name"]
    left, right = (iter(pq.ParquetFile(p).iter_batches(batch_size=32768, columns=keys+features)) for p in (before,after))
    a = b = None
    counts = {c:0 for c in features}
    newly_known = {c:0 for c in features}
    newly_unknown = {c:0 for c in features}
    total = changed_rows = 0
    affected_fields = set()
    while True:
        if a is None or not a.num_rows:
            a = next(left, None)
        if b is None or not b.num_rows:
            b = next(right, None)
        if a is None and b is None:
            break
        if a is None or b is None:
            raise ValueError("Store populations differ")
        n = min(a.num_rows,b.num_rows)
        x, y = a.slice(0,n), b.slice(0,n)
        for c in keys:
            if not x.column(c).equals(y.column(c)):
                raise ValueError(f"Runner order/identity differs at offset{total}: {c}")
        any_change = np.zeros(n,dtype=bool)
        for c in features:
            old = np.asarray(x.column(c).to_numpy(zero_copy_only=False),dtype=np.float32)
            new = np.asarray(y.column(c).to_numpy(zero_copy_only=False),dtype=np.float32)
            unknown_old, unknown_new = np.isnan(old), np.isnan(new)
            difference = ~((old == new) | (unknown_old & unknown_new))
            counts[c] += int(difference.sum())
            newly_known[c] += int((unknown_old & ~unknown_new).sum())
            newly_unknown[c] += int((~unknown_old & unknown_new).sum())
            any_change |= difference
        changed_rows += int(any_change.sum())
        affected_fields.update(np.asarray(x.column("race_id").to_numpy(zero_copy_only=False))[any_change].tolist())
        total += n
        a,b = a.slice(n),b.slice(n)
    return {"scope":"Descriptive feature-store difference, not a prediction or performance comparison",
        "runners":total,"selected_features":len(features), "changed_runner_rows":changed_rows,
        "affected_fields":len(affected_fields),"precision":"float32, matching native training/prediction",
        "changed_features":{c:{"changed_rows":counts[c],"became_known":newly_known[c],"became_unknown":newly_unknown[c]}
                            for c in features if counts[c]},
        "unchanged_features":[c for c in features if not counts[c]]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,required=True)
    parser.add_argument("--after",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    features=[s.strip() for s in args.features.read_text().splitlines() if s.strip() and not s.lstrip().startswith("#")]
    report=compare(args.before,args.after,features)
    report["files_sha256"]={str(p.resolve()):file_sha256(p) for p in (args.before,args.after,args.features,Path(__file__))}
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("files_sha256","unchanged_features","changed_features")},indent=2),flush=True)
