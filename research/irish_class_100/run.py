"""Build the effective-class inputs and an admission matrix (research 100).

    python -m research.irish_class_100.run build
    python -m research.irish_class_100.run matrix
    python -m src.modelling admit --prepared data/research/irish_class_100/matrix \
        --features feature_sets/win_selected_20261112.txt --candidates class_eff class_change_eff_vs_field \
        --from 2025-07-08 --capacity screen --output data/research/irish_class_100/admit
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.irish_class_100.inputs import COLUMNS, STORE_COLUMNS, build
from src.feature_store import FeatureStore

OUT = Path("data/research/irish_class_100")
SOURCE_MATRIX = Path("data/research/selection_20261111/matrix")
FEATURES = Path("feature_sets/win_selected_20261112.txt")


def stage_build():
    store = FeatureStore()
    with store._access():
        frame = pd.read_parquet(store.featured_path, columns=STORE_COLUMNS + ["track", "class_change_vs_field"])
    out = build(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "inputs.parquet", index=False)
    joined = frame[["race_date", "track", "class_change_vs_field"]].reset_index(drop=True).join(out[COLUMNS])
    recent = joined[joined.race_date >= "2025-07-01"]
    from research.error_anatomy.yard_change import IRISH
    ire = recent.track.astype(str).str.lower().str.replace("-", " ").isin(IRISH)
    print(pd.DataFrame({"GB": recent[~ire][["class_change_vs_field", *COLUMNS]].isna().mean(),
                        "IRE": recent[ire][["class_change_vs_field", *COLUMNS]].isna().mean()}).round(3))


def stage_matrix():
    description = json.loads((SOURCE_MATRIX / "prepared.json").read_text())
    names = description["features"]
    features = [x.strip() for x in FEATURES.read_text().splitlines() if x.strip() and not x.startswith("#")]
    context = pd.read_parquet(SOURCE_MATRIX / "context.parquet")
    source = np.load(SOURCE_MATRIX / "matrix.npy", mmap_mode="r")
    inputs = pd.read_parquet(OUT / "inputs.parquet")
    keys = pd.MultiIndex.from_frame(context[["race_id", "horse_id"]].astype("int64"))
    positions = pd.MultiIndex.from_frame(inputs[["race_id", "horse_id"]].astype("int64")).get_indexer(keys)
    if (positions < 0).any():
        raise ValueError(f"{(positions < 0).sum()} matrix runners have no inputs")
    target = OUT / "matrix"
    target.mkdir(parents=True, exist_ok=True)
    out_names = [*features, *COLUMNS]
    matrix = np.lib.format.open_memmap(target / "matrix.npy", mode="w+", dtype=np.float64,
                                       shape=(len(context), len(out_names)), fortran_order=True)
    for j, name in enumerate(features):
        matrix[:, j] = source[:, names.index(name)]
    for j, name in enumerate(COLUMNS, len(features)):
        matrix[:, j] = inputs[name].to_numpy(dtype=np.float64)[positions]
    matrix.flush()
    from src.modelling.evaluation import frame_signature
    from src.modelling.selection import matrix_signature
    context.to_parquet(target / "context.parquet", index=False)
    receipt = {**description["receipt"], "features": out_names, "note": "research 100: 87 inputs plus effective class"}
    (target / "prepared.json").write_text(json.dumps({"receipt": receipt, "features": out_names,
        "context_signature": frame_signature(context, context.columns), "rows": len(context),
        "matrix_signature": matrix_signature(matrix, out_names)}, default=str))
    print(f"matrix {matrix.shape}")


if __name__ == "__main__":
    {"build": stage_build, "matrix": stage_matrix}[sys.argv[1]]()
