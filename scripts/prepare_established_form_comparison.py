"""Freeze one matched comparison from the fully rebuilt normal feature store."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.deployment_validation import file_sha256 as sha, RECIPE_SOURCES
from src.established_form_features import FEATURES, established_form_features
from src.featured_dataset import _FE_SOURCE_FILES, _model_load_columns, fe_signature
from src.feature_store import FeatureStore
from src.model import get_feature_columns

OUTPUT = ROOT / "data/audit/established_form_029/native_canonical"
STORE = ROOT / "data/processed/feature_store"
KEY = ["race_id", "horse_name"]


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise ValueError("Resume the frozen benchmark instead of overwriting its registration")
    metadata = json.loads((STORE / "meta.json").read_text())
    if metadata["fe_signature"] != fe_signature():
        raise ValueError("Complete the normal canonical feature rebuild first")
    schema = pq.read_schema(STORE / "featured.parquet")
    baseline = get_feature_columns(schema.empty_table().to_pandas())
    if len(baseline) != 151:
        raise ValueError("Keep the existing151-feature baseline for this comparison")
    # Current151 inputs retain Chase but omit the existing Hurdle indicator.
    # Expose it so Flat, bumper, hurdle and chase evidence can be distinguished.
    added = [*FEATURES, "race_type_Hurdle"]
    candidates = set(baseline) | set(added)
    ordered = [c for c in schema.names if c in candidates]
    if len(ordered) != 169:
        raise ValueError("The fixed family must add17 historical/context inputs and the existing Hurdle flag")
    # Share the normal training projection, including target/settlement metadata.
    # A win-only fit still prepares auxiliary targets; a handwritten short list
    # silently omitted finishing margins before the first component could fit.
    normal_columns = _model_load_columns(FeatureStore(str(STORE)))
    if normal_columns is None:
        raise ValueError("The registered comparison requires the normal feature allow-list")
    columns = list(dict.fromkeys([*ordered, *normal_columns, "horse_prev_races"]))
    required = {"race_id", "horse_name", "race_date", "won", "finish_position",
                "num_runners", "handicap", "odds", "lengths_behind", "beaten_lengths"}
    if not required.issubset(columns):
        raise ValueError(f"Native training metadata missing: {sorted(required-set(columns))}")
    frame = pd.read_parquet(STORE / "featured.parquet", columns=columns,
        filters=[("race_date", "<=", pd.Timestamp("2026-03-31"))])
    frame = frame.sort_values(["race_date", "race_id", "horse_name"], kind="stable").reset_index(drop=True)
    if frame.duplicated(KEY).any():
        raise ValueError("Do not silently drop duplicate runner identities")
    first_end = frame.race_date.min()+pd.Timedelta(days=365)
    raw = pd.read_parquet(STORE / "processed.parquet", filters=[("race_date", "<=", first_end)])
    values = established_form_features(raw)
    expected = pd.concat([raw[KEY], values], axis=1)
    checked = frame.loc[frame.race_date.le(first_end), KEY+list(FEATURES)].merge(expected, on=KEY, validate="one_to_one", suffixes=("_store", "_fresh"))
    if len(checked) != len(raw):
        raise ValueError("Native batch/state verification lost a runner")
    for c in FEATURES:
        np.testing.assert_allclose(checked[c+"_store"], checked[c+"_fresh"], rtol=0, atol=1e-12, equal_nan=True, err_msg=c)
    np.testing.assert_array_equal(frame.race_type_Hurdle.to_numpy(), frame.race_type.eq("Hurdle").to_numpy())
    boundary = raw.race_date.max()-pd.Timedelta(days=3)
    changed = raw.copy()
    changed.loc[changed.race_date.ge(boundary), ["finish_position", "finish_pos_label"]] = [0, "PU"]
    pd.testing.assert_frame_equal(values.loc[raw.race_date.le(boundary)], established_form_features(changed).loc[raw.race_date.le(boundary)])
    pd.testing.assert_frame_equal(values, established_form_features(raw.sample(frac=1, random_state=42)).sort_index())
    profile = []
    for c in added:
        x = frame[c]
        if np.isinf(x).any():
            raise ValueError(f"Infinite feature: {c}")
        profile.append({"feature": c, "known": int(x.notna().sum()), "unique": int(x.nunique()), "minimum": float(x.min()), "maximum": float(x.max())})
    pd.DataFrame(profile).to_csv(OUTPUT / "feature_coverage.csv", index=False)
    frame.to_parquet(OUTPUT / "inputs.parquet", index=False)
    sources = list(dict.fromkeys([*_FE_SOURCE_FILES, *RECIPE_SOURCES, "config.py", "feature_sets/win.txt",
        "scripts/prepare_established_form_comparison.py", "scripts/benchmark_established_form.py",
        "scripts/rebuild_established_form.py", "scripts/canonical_training_cache.py"]))
    manifest = {"scope": "One previously inspected Q1 2026 development comparison; no reserved-period or fresh-holdout claim",
        "fe_signature": fe_signature(), "store_meta": metadata, "baseline_features": baseline, "candidate_features": ordered,
        "training_projection_columns": columns,
        "history_cutoff": "2025-12-22", "evaluation_start": "2026-01-01", "evaluation_end": "2026-03-31",
        "max_trees": 4000, "threads": 2, "seed": 42, "model": "Unchanged native win learner and calibration; no parameter search",
        "teacher_sha256": sha(ROOT / "data/processed/betfair_bsp.parquet"), "inputs_sha256": sha(OUTPUT / "inputs.parquet"),
        "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm")},
        "reference_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT/"data/audit/canonical_repairs_026/historical_predictions.parquet",
            ROOT/"data/audit/canonical_repairs_026/market_matched_predictions.parquet",
            ROOT/"data/audit/established_horse_review_028/paired_context.parquet")},
        "source_sha256": {name: sha(ROOT/name) for name in sources},
        "native_state_checked_runners": len(raw), "real_causality_checks_passed": True,
        "mandatory_baseline_repairs": "Glicko inactivity, comparable marks and source-verified ratings apply to both arms regardless of experimental score",
        "decision_rule": "No automatic promotion. Consider the fixed family only when log loss and Brier both improve; report paired date-cluster uncertainty and established-field scores. No Q1 feature or parameter search.",
        "prior_registration_amendment": "Earlier input-only registrations were superseded by source validation before any candidate fit or comparison score. The first native registration then failed before any component fitted because target metadata was omitted; that registration and input are preserved in pre_target_metadata_fix. The corrected projection shares the ordinary native training metadata. No feature definitions, targets, learner settings or outcome-based decisions changed."}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT/name, name)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("baseline_features", "candidate_features", "source_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    OUTPUT = Path(parser.parse_args().output_dir).resolve()
    run()
