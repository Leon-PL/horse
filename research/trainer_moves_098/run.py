"""Build and screen the trainer-move inputs (research 098).

    python -m research.trainer_moves_098.run build
    python -m research.trainer_moves_098.run screen
    python -m research.trainer_moves_098.run matrix

``screen`` compares runners grouped by the move against the 87-input model's
out-of-sample forecasts (data/processed/forecast_history) and Betfair SP, on
development races only. ``matrix`` writes a prepared directory for
``python -m src.modelling admit``: the 87 inputs from the 2026-11-11 selection
matrix plus these inputs, on the same runners.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.trainer_moves_098.inputs import COLUMNS, STORE_COLUMNS, build
from src.feature_store import FeatureStore

OUT = Path("data/research/trainer_moves_098")
HOLDOUT_START = pd.Timestamp("2026-06-11")
SOURCE_MATRIX = Path("data/research/selection_20261111/matrix")
FEATURES = Path("feature_sets/win_selected_20261112.txt")


def stage_build():
    store = FeatureStore()
    with store._access():
        frame = pd.read_parquet(store.featured_path, columns=STORE_COLUMNS + ["off_time", "track", "horse_name"])
        meta = store.meta()
    inputs = build(frame)
    inputs = inputs.merge(frame[["race_id", "horse_id", "race_date", "off_time", "track", "horse_name", "won"]],
                          on=["race_id", "horse_id"], validate="one_to_one")
    OUT.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(OUT / "inputs.parquet", index=False)
    (OUT / "build.json").write_text(json.dumps({"fe_signature": meta.get("fe_signature"),
        "data_signature": meta.get("data_signature"), "rows": len(inputs)}, indent=2))
    print(inputs[COLUMNS].describe().T.round(3).to_string())


def _row(x, p, label):
    x = x[np.isfinite(x[p])]
    won, exp = x.won.sum(), x[p].sum()
    sd = np.sqrt((x[p] * (1 - x[p])).sum())
    return {"group": label, "runners": len(x), "won": int(won), "expected": round(exp, 1),
            "ratio": round(won / exp, 3) if exp else np.nan, "z": round((won - exp) / sd, 1) if sd else np.nan}


def stage_screen():
    from src.betfair import attach_bsp, bsp_implied_prob
    inputs = pd.read_parquet(OUT / "inputs.parquet")
    history = pd.read_parquet("data/processed/forecast_history/history.parquet", columns=["race_id", "horse_id", "probability"])
    history["race_id"] = pd.to_numeric(history.race_id)
    history["horse_id"] = pd.to_numeric(history.horse_id)
    d = inputs.merge(history, on=["race_id", "horse_id"], how="inner")
    d = d[pd.to_datetime(d.race_date) < HOLDOUT_START].copy()
    d = attach_bsp(d)
    d["bsp_p"] = bsp_implied_prob(d.bsp, d.race_id.to_numpy())
    moved = d.tm_elo_change.notna()
    rows = []
    fresh = d[moved & (d.tm_runs_in_yard < 3)]
    edges = fresh.tm_elo_change.quantile([0, .2, .4, .6, .8, 1]).to_numpy()
    for base, label in (("probability", "model"), ("bsp_p", "BSP")):
        both = fresh[np.isfinite(fresh.bsp_p)]
        for k in range(5):
            lo, hi = edges[k], edges[k + 1]
            sel = both[(both.tm_elo_change >= lo) & ((both.tm_elo_change < hi) if k < 4 else (both.tm_elo_change <= hi))]
            rows.append({"base": label, **_row(sel, base, f"runs 1-3, Elo change {lo:+.0f} to {hi:+.0f}")})
        for r in range(3):
            rows.append({"base": label, **_row(both[both.tm_runs_in_yard == r], base, f"run {r + 1} for the new yard")})
        q = d[np.isfinite(d.bsp_p) & d.tm_arrivals_n.ge(30)]
        e = q.tm_arrival_edge.quantile([0, .2, .8, 1]).to_numpy()
        rows.append({"base": label, **_row(q[q.tm_arrival_edge <= e[1]], base, "arrival edge, bottom fifth of trainers")})
        rows.append({"base": label, **_row(q[q.tm_arrival_edge >= e[2]], base, "arrival edge, top fifth of trainers")})
        qa = q[(q.tm_elo_change.notna()) & (q.tm_runs_in_yard < 3)]
        rows.append({"base": label, **_row(qa[qa.tm_arrival_edge >= e[2]], base, "new arrivals at top-fifth trainers")})
        rows.append({"base": label, **_row(qa[qa.tm_arrival_edge <= e[1]], base, "new arrivals at bottom-fifth trainers")})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "screen.csv", index=False)
    print(f"development runners scored: {len(d):,} ({d.race_date.min()} to {d.race_date.max()}), "
          f"with BSP: {np.isfinite(d.bsp_p).sum():,}")
    print(table.to_string(index=False))


def stage_matrix():
    description = json.loads((SOURCE_MATRIX / "prepared.json").read_text())
    names = description["features"]
    features = [line.strip() for line in FEATURES.read_text().splitlines() if line.strip() and not line.startswith("#")]
    missing = set(features) - set(names)
    if missing:
        raise ValueError(f"Selection matrix lacks {sorted(missing)}")
    context = pd.read_parquet(SOURCE_MATRIX / "context.parquet")
    source = np.load(SOURCE_MATRIX / "matrix.npy", mmap_mode="r")
    inputs = pd.read_parquet(OUT / "inputs.parquet", columns=["race_id", "horse_id", *COLUMNS])
    keys = pd.MultiIndex.from_frame(context[["race_id", "horse_id"]].astype("int64"))
    positions = pd.MultiIndex.from_frame(inputs[["race_id", "horse_id"]].astype("int64")).get_indexer(keys)
    if (positions < 0).any():
        raise ValueError(f"{(positions < 0).sum()} matrix runners have no trainer-move inputs")
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
    receipt = {**description["receipt"], "features": out_names,
               "note": "research 098: 87 inputs copied from the 2026-11-11 selection matrix plus trainer-move inputs"}
    (target / "prepared.json").write_text(json.dumps({"receipt": receipt, "features": out_names,
        "context_signature": frame_signature(context, context.columns), "rows": len(context),
        "matrix_signature": matrix_signature(matrix, out_names)}, default=str))
    print(f"matrix {matrix.shape} written to {target}")


if __name__ == "__main__":
    {"build": stage_build, "screen": stage_screen, "matrix": stage_matrix}[sys.argv[1]]()
