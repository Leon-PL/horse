"""Review every selected input after rebuilding, before screening a model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts import audit_selected_feature_semantics as semantic_audit
from scripts.audit_selected_feature_semantics import sha
from src.featured_dataset import fe_signature


def main():
    output = ROOT / "data/audit/feature_review_018"
    live = ROOT / "data/processed/feature_store"
    candidate = output / "candidate_store_exact"
    current_meta = json.loads((live / "meta.json").read_text())
    candidate_meta = json.loads((candidate / "meta.json").read_text())
    audit = json.loads((ROOT / "data/audit/model_improvement/018_feature_repairs/input_invariance.json").read_text())
    if not audit["passed"] or candidate_meta["fe_signature"] != audit["fe_signature"]:
        raise ValueError("Candidate generation differs from the audited source")
    if current_meta["fe_signature"] != fe_signature():
        raise ValueError("Installed feature source changed during research")
    if current_meta["featured_rows"] != candidate_meta["featured_rows"]:
        raise ValueError("Repair changed historical population")
    selected = [c.strip() for c in (ROOT / "feature_sets/win.txt").read_text().splitlines()
                if c.strip() and not c.startswith("#")]
    keys = ["race_id", "horse_name", "race_date"]
    columns = list(dict.fromkeys([*selected, *keys, "race_type", "won", "finish_position", "odds"]))
    original = pd.read_parquet(live / "featured.parquet", columns=columns)
    updated = pd.read_parquet(candidate / "featured.parquet", columns=columns)
    if not original[keys].equals(updated[keys]):
        raise ValueError("Repair lost or reordered a historical runner")
    for name in ["won", "finish_position", "odds"]:
        np.testing.assert_array_equal(original[name], updated[name])
    changes = {}
    for name in selected:
        a, b = original[name].to_numpy(np.float32), updated[name].to_numpy(np.float32)
        if np.isinf(b).any():
            raise ValueError(f"Infinite repaired input: {name}")
        differing = ~np.isclose(a, b, atol=0, rtol=0, equal_nan=True)
        if differing.any():
            finite = np.isfinite(a) & np.isfinite(b)
            maximum = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else None
            changes[name] = {"changed_rows": int(differing.sum()), "old_missing": int(np.isnan(a).sum()),
                "new_missing": int(np.isnan(b).sum()), "maximum_finite_difference": maximum}
    conserved = ["horse_elo_momentum", "horse_elo_rank", "horse_margin_elo_momentum", "horse_glicko_rd",
                 "horse_glicko_lcb", "horse_place_rate", "horse_avg_pos_3", "horse_prev_races_vs_field",
                 "days_since_last_race", "days_since_last_win", "sire_win_rate_shrunk", "jockey_place_rate",
                 "trainer_win_rate", "trainer_place_rate"]
    for name in conserved:
        np.testing.assert_allclose(original[name], updated[name], rtol=0, atol=1e-6, equal_nan=True, err_msg=name)
    # Independent real-data oracle: the installed code's single full pass is
    # mathematically correct; its truncated append state is the defect. Compare
    # the candidate's batched history against that full-pass calculation.
    from src.feature_engineer import (
        _add_result_history_flags, _event_sort_key, add_target_encoded_features,
        add_race_relative_signal_features, counts_state_scope,
    )
    raw_columns = ["race_id", "horse_id", "horse_name", "race_date", "off_time", "finish_position",
                   "finish_pos_label", "won", "num_runners", "jockey", "trainer", "track", "going"]
    oracle = pd.read_parquet(candidate / "processed.parquet", columns=raw_columns)
    oracle["_event_dt"] = _event_sort_key(oracle)
    oracle = _add_result_history_flags(oracle.sort_values(["_event_dt", "race_id"], kind="stable"))
    with counts_state_scope(None):
        oracle = add_race_relative_signal_features(add_target_encoded_features(oracle))
    oracle = oracle.set_index(["race_id", "horse_name"]).reindex(pd.MultiIndex.from_frame(updated[["race_id", "horse_name"]]))
    oracle_errors = {}
    for name in ["jockey_te_ewma_vs_best_in_race", "trainer_te_ewma_vs_field",
                 "trainer_te_ewma_z_in_race", "trainer_te_ewma_vs_best_in_race"]:
        actual, expected = updated[name].to_numpy(float), oracle[name].to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, equal_nan=True, err_msg=name)
        oracle_errors[name] = float(np.nanmax(np.abs(actual-expected)))
    del oracle
    updated[selected] = updated[selected].astype(np.float32)
    profiles = output / "repaired_profiles"
    profiles.mkdir(exist_ok=True)
    semantic_audit.ROOT = ROOT / "_research_snapshots/018_feature_repairs"
    result, aliases = semantic_audit.profile(updated, selected, profiles)
    if result.infinite_rows.gt(0).any() or result.known_rows.eq(0).any():
        raise ValueError("A selected feature became invalid or wholly unavailable")
    report = {"passed": True, "rows": len(updated), "selected_features": len(selected),
        "candidate_fe_signature": candidate_meta["fe_signature"], "installed_fe_signature": fe_signature(),
        "candidate_meta_sha256": sha(candidate / "meta.json"), "source_audit_sha256": sha(ROOT / "data/audit/model_improvement/018_feature_repairs/input_invariance.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "lineage_source_directory": str(semantic_audit.ROOT),
        "full_history_ewma_oracle_rows": len(updated), "full_history_ewma_oracle_max_error": oracle_errors,
        "ewma_oracle_source_sha256": sha(ROOT / "src/feature_engineer.py"),
        "changes": changes, "unchanged_control_features": conserved,
        "constant_selected": result.loc[result.constant_when_known, "feature"].tolist(),
        "scope": "Predictor/source validation only; no models, reserved outcomes or prices used for feature selection"}
    (output / "repaired_store_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2), flush=True)
    print(json.dumps({"changed_features": len(changes), "changed_names": list(changes)}), flush=True)


if __name__ == "__main__":
    main()
