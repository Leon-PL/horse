"""One fixed RTV-family screen on the corrected native feature history."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"data/audit/rtv_repair_features_044"
PROJECT = CASE/"project"
sys.path.insert(0, str(PROJECT))

import pandas as pd
import config
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore
from src.featured_dataset import _model_load_columns, fe_signature
from src.rtv_features import HISTORY_FEATURES
from src.source_pipeline import _publish
from scripts import quick_win_feature_screen as screen


def main():
    plan = json.loads((CASE/"feature_plan.json").read_text())
    built = json.loads((CASE/"rebuilt.json").read_text())
    if built["fe_signature"] != plan["fe_signature"] or fe_signature() != plan["fe_signature"]:
        raise ValueError("Use the frozen corrected generation")
    reference = CASE/"screen_reference_v2"
    reference.mkdir(exist_ok=True)
    parent = json.loads((ROOT/"data/runs/20261030_015917_823047/meta.json").read_text())
    baseline = parent["training_config"]["validation_reference"]["features"]
    store = FeatureStore()
    path = reference/"inputs.parquet"
    projection = reference/"rtv_projection.parquet"
    if not (reference/"manifest.json").exists():
        columns = _model_load_columns(store)
        columns = list(dict.fromkeys([*columns, "horse_prev_races"]))
        frame = pd.read_parquet(store.featured_path, columns=columns,
            filters=[("race_date", ">=", pd.Timestamp("2023-09-24")), ("race_date", "<=", pd.Timestamp("2025-12-31"))])
        frame.to_parquet(path, index=False)
        extras = pd.read_parquet(store.featured_path, columns=["race_id", "horse_name", "race_date", *HISTORY_FEATURES],
            filters=[("race_date", ">=", pd.Timestamp("2023-09-24")), ("race_date", "<=", pd.Timestamp("2025-12-31"))])
        extras.to_parquet(projection, index=False)
        _publish(reference/"manifest.json", {"baseline_features": baseline, "inputs_sha256": sha(path),
            "teacher_sha256": sha(PROJECT/"data/processed/betfair_bsp.parquet"), "projection_sha256": sha(projection),
            "source_repair_receipt_sha256": sha(CASE/"prepared_source.json"), "feature_plan_sha256": sha(CASE/"feature_plan.json")})
    screen.REFERENCE = reference
    screen.OUTPUT = CASE/"quick_screen_v3"
    screen.run(PROJECT/"feature_sets/win_with_rtv.txt", projection, allow_new_baseline=True)
    report = json.loads((screen.OUTPUT/"report.json").read_text())
    pairs = [report["comparison"], report["established_handicaps"]]
    passes = all(x["candidate"]["race_log_loss"] < x["baseline"]["race_log_loss"]
        and x["candidate"]["brier"] <= x["baseline"]["brier"] for x in pairs)
    interval = report["comparison"]["paired_delta_candidate_minus_baseline"]["date_cluster_race_log_loss"]["ci95"]
    _publish(CASE/"screen_decision.json", {"primary_point_rule_passed": passes,
        "overall_date_interval_excludes_zero_in_favour": interval[1] < 0,
        "decision": "requires_separate_native_validation" if passes and interval[1] < 0 else "not_selected_for_current_model",
        "new_fits": report["actual_new_fits"], "pre_race_features_evaluated": False,
        "scope": "Preset development screen; implementation remains available, no automatic model adoption or predictive gain claim"})


if __name__ == "__main__":
    main()
