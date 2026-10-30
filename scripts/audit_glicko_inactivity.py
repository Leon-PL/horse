"""Measure the verified Glicko time-accounting error without fitting a learner."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config
from src import ratings
from src.deployment_validation import file_sha256

OUTPUT = ROOT / "data/audit/established_horse_review_028"
STORE = ROOT / "data/processed/feature_store"
KEY = ["race_id", "horse_name"]


def run():
    legacy_path = OUTPUT / "ratings_before_fix.py"
    spec = importlib.util.spec_from_file_location("_legacy_glicko_028", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    # The audited active recipe has only horse Glicko-1 enabled. TrueSkill is
    # independent of Glicko; skip its unchanged sweep in BOTH audit arms.
    if any(getattr(config, name, False) for name in ("GLICKO_JOCKEY_TRAINER", "GLICKO_DIMENSIONAL", "GLICKO_MARGIN", "GLICKO2_ENABLED")):
        raise ValueError("This audit is scoped to the existing horse Glicko-1 recipe")
    config.TRUESKILL_ENABLED = False
    frame = pd.read_parquet(STORE / "processed.parquet", columns=KEY + ["horse_id", "race_date", "off_time", "finish_position"],
                             filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    started = time.monotonic()
    print(json.dumps({"stage": "legacy_sweep", "runners": len(frame)}), flush=True)
    before = legacy.compute_glicko_features(frame.copy())
    columns = [c for c in before if c.startswith("horse_glicko") or c == "has_horse_glicko"]
    stored = pd.read_parquet(STORE / "featured.parquet", columns=KEY + columns,
                             filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    before = before.merge(stored, on=KEY, validate="one_to_one", suffixes=("", "_stored"))
    # Reproduce the existing store before attributing any difference to the fix.
    for c in columns:
        np.testing.assert_allclose(before[c].astype(stored[c].dtype), before[c + "_stored"], rtol=0, atol=1e-10)
    print(json.dumps({"stage": "corrected_sweep", "legacy_store_reproduced": True}), flush=True)
    after = ratings.compute_glicko_features(frame.copy())
    after[KEY + columns].to_parquet(OUTPUT / "corrected_glicko_development.parquet", index=False)
    joined = before[KEY + ["race_date"] + columns].merge(after[KEY + columns], on=KEY, validate="one_to_one", suffixes=("_before", "_after"))
    context = pd.read_parquet(OUTPUT / "paired_context.parquet")
    paired = context.merge(joined.drop(columns="race_date"), on=KEY, validate="one_to_one")
    records = []
    for c in columns:
        delta = paired[c + "_after"] - paired[c + "_before"]
        established = paired.horse_prev_races.ge(5)
        records.append({"feature": c, "changed_runners_at_1e-6": int(delta.abs().gt(1e-6).sum()),
            "changed_established_runners_at_1e-6": int((delta.abs().gt(1e-6) & established).sum()),
            "mean_absolute_change": float(delta.abs().mean()), "maximum_absolute_change": float(delta.abs().max())})
    picked = pd.read_csv(OUTPUT / "selected_cases.csv")[KEY + ["case_group"]]
    selected = picked.merge(paired, on=KEY, validate="one_to_one")
    selected.to_csv(OUTPUT / "case_glicko_corrections.csv", index=False)
    report = {"scope": "Feature validity only; no retraining, no prediction improvement claim, no reserved-period scoring",
        "old_rating_source_sha256": file_sha256(legacy_path), "new_rating_source_sha256": file_sha256(ROOT / "src/ratings.py"),
        "source_sha256": file_sha256(Path(__file__)), "processed_rows_swept": len(frame),
        "paired_q1_runners": len(paired), "established_q1_runners": int(paired.horse_prev_races.ge(5).sum()),
        "changes": records, "legacy_store_exactly_reproduced_at_storage_precision": True,
        "elapsed_seconds": time.monotonic()-started,
        "interpretation": "Differences include downstream propagation through opponents and later rating updates; not all changed rows directly double-counted their own inactivity",
        "production_store_changed": False, "production_model_refitted": False}
    (OUTPUT / "glicko_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(selected.loc[selected.case_group.eq("missed_winners"), ["horse_name", "horse_glicko_rd_before", "horse_glicko_rd_after", "horse_glicko_rank_before", "horse_glicko_rank_after"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
