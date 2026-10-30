"""One native win-only check after the corrected032 history finishes.

No baseline fit, place fit, promotion or reserved-period evaluation. Native
learner checkpoints can be reused by the later canonical qualification only
when their numeric arguments and complete learner semantics are identical.
"""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import argparse
import gc
import importlib.metadata
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.canonical_training_cache import CheckpointPredictor
from src.deployment_validation import RECIPE_SOURCES, file_sha256 as sha
from src.featured_dataset import fe_signature
from src.forecast_evaluation import valid_race_mask
from src.model import get_feature_columns
from src.training import TrainingConfig

CASE = ROOT / "data/audit/trip_suitability_032/installation"
RESEARCH = ROOT / "data/audit/prior_market_feasibility_033"
OUTPUT = RESEARCH / "native_check"
KEY = ["race_id", "horse_name", "race_date"]
SCRIPT_SHA_AT_START = sha(Path(__file__))


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(wait):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage":"waiting_for_corrected032_history"}), flush=True)
    deadline = time.monotonic()+3*3600
    while True:
        status = CASE / "rebuild_status.json"
        if status.exists():
            rebuild = json.loads(status.read_text())
            if rebuild["stage"] == "rebuilt":
                break
        if not wait or time.monotonic() >= deadline:
            raise ValueError("Complete the032 corrected history first")
        time.sleep(10)
    if rebuild["fe_signature"] != fe_signature():
        raise ValueError("Corrected feature sources changed after rebuilding")
    if sha(Path(__file__)) != SCRIPT_SHA_AT_START:
        raise ValueError("Restart the waiting check after changing its source")
    cfg = TrainingConfig(refresh_data=False, threads=2)
    columns = [s.strip() for s in (ROOT / "feature_sets/prior_market_033.txt").read_text().splitlines()
               if s.strip() and not s.lstrip().startswith("#")]
    if columns[:151] != rebuild["features"] or len(columns) != 154:
        raise ValueError("Only the fixed three-input proposal is allowed")
    runtime = {n:importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")}
    semantics = {"schema":"native_estimator_cache_v1", "seed":cfg.seed, "threads":2,
        "frameworks":cfg.frameworks, "win_calibration_method":cfg.win_calibration_method,
        "params":cfg.hyperparams, "config_sha256":sha(ROOT / "config.py"),
        "runtime":runtime, "sources":{p:sha(ROOT/p) for p in RECIPE_SOURCES}}
    scope = canonical_hash(semantics)
    cache = OUTPUT / "components"
    manifest = {"scope":"One already-inspected Q1 development check; no independent confirmation",
        "historical_cutoff":"2025-12-22", "test_dates":["2026-01-01", "2026-03-31"],
        "features":columns, "baseline":"Saved full native029 win/place qualification forecasts; no baseline refit",
        "base_fe_signature":fe_signature(), "learner_semantics":semantics, "component_cache_scope":scope,
        "inputs_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (
            CASE/"corrected_inputs.parquet", CASE/"baseline_2026Q1_predictions.parquet",
            RESEARCH/"projection.parquet", ROOT/"data/processed/betfair_bsp.parquet")},
        "source_sha256":{p:sha(ROOT/p) for p in ("src/prior_market_research.py", "scripts/check_prior_market_native.py")},
        "advance_rule":"Lower overall and established-handicap race log loss, with neither runner Brier worse. Report uncertainty; no retuning or automatic promotion."}
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Native check was already registered with different inputs/settings")
    else:
        write(manifest_path, manifest)
    base = pd.read_parquet(CASE / "corrected_inputs.parquet")
    projection = pd.read_parquet(RESEARCH / "projection.parquet")
    extra = columns[151:]
    frame = base.merge(projection[KEY+extra], on=KEY, how="left", validate="one_to_one", indicator=True)
    if not frame.pop("_merge").eq("both").all():
        raise ValueError("Projection misses corrected-history runners")
    del base, projection
    frame = frame.loc[valid_race_mask(frame)].sort_values(KEY[2:]+KEY[:2], kind="stable").reset_index(drop=True)
    test = frame.loc[frame.race_date.between(*manifest["test_dates"])].copy().reset_index(drop=True)
    baseline = pd.read_parquet(CASE / "baseline_2026Q1_predictions.parquet")
    label_keys = KEY+["won", "finish_position", "num_runners", "handicap"]
    pd.testing.assert_frame_equal(test[label_keys], baseline[label_keys], check_dtype=False)
    dataset_start = frame.race_date.min()
    train = frame.loc[frame.race_date.le(manifest["historical_cutoff"])].copy()
    del frame
    gc.collect()
    predictor = CheckpointPredictor(random_seed=cfg.seed, frameworks=cfg.frameworks,
        win_calibration_method=cfg.win_calibration_method, checkpoint_directory=cache, checkpoint_manifest=scope)
    predictor._autotune_njobs = 2
    started = time.monotonic()
    model_path = OUTPUT / "win_only_model.joblib"
    if model_path.exists():
        predictor.load(str(model_path))
        if predictor.eval_split_info.get("prior_market_check_manifest") != sha(manifest_path):
            raise ValueError("Saved native proposal differs")
    else:
        print(json.dumps({"stage":"native_candidate_win_only", "features":len(columns)}), flush=True)
        with patch.object(config, "FEATURE_ALLOWLIST", columns):
            if get_feature_columns(train) != columns:
                raise ValueError("Native feature selection/order differs from the proposed154 inputs")
            predictor.fit_as_of(train, manifest["historical_cutoff"], params=cfg.hyperparams,
                dataset_start=dataset_start, win_only=True)
        predictor.eval_split_info["prior_market_check_manifest"] = sha(manifest_path)
        predictor.save(str(model_path))
    if predictor.feature_cols != columns:
        raise ValueError("Native model changed the preset inputs")
    paired = test[label_keys].assign(baseline_probability=baseline.win_probability.to_numpy(),
        candidate_probability=align_predictions(predictor, test))
    if "horse_prev_races" in test:
        paired["horse_prev_races"] = test.horse_prev_races.to_numpy()
    else:
        context = pd.read_parquet(ROOT / "data/audit/established_form_029/native_canonical/inputs.parquet",
            columns=KEY+["horse_prev_races"], filters=[("race_date", ">=", pd.Timestamp("2026-01-01"))])
        paired = paired.merge(context, on=KEY, validate="one_to_one")
    established = paired.groupby("race_id",sort=False).horse_prev_races.transform("min").ge(5)
    handicap = paired.groupby("race_id",sort=False).handicap.transform("min").eq(1)
    overall = score_pair(paired)
    target = score_pair(paired.loc[established & handicap])
    advance = all(s["candidate"]["race_log_loss"] < s["baseline"]["race_log_loss"]
                  and s["candidate"]["brier"] <= s["baseline"]["brier"] for s in (overall,target))
    check_mask = test.race_id.isin(test.race_id.drop_duplicates().head(100))
    removed = align_predictions(predictor, test.loc[check_mask].assign(odds=np.nan, bsp=np.nan))
    price_error = float(np.max(np.abs(removed-paired.loc[check_mask.to_numpy(),"candidate_probability"].to_numpy())))
    if price_error > 1e-12:
        raise ValueError("Current prices affected the supposedly prior-only model")
    market = pd.read_parquet(ROOT / "data/audit/established_form_029/native_canonical/market_paired_predictions.parquet",
        columns=KEY+["market_probability"])
    matched = paired.merge(market,on=KEY,validate="one_to_one")
    market_scores = score_forecasts(matched, {c:matched[c+"_probability"].to_numpy() for c in ("baseline","candidate","market")})
    for p,digest in manifest["inputs_sha256"].items():
        if sha(ROOT/p) != digest:
            raise ValueError("Frozen native comparison inputs changed")
    if fe_signature() != manifest["base_fe_signature"]:
        raise ValueError("Feature sources changed during the native check")
    for p,digest in {**semantics["sources"], **manifest["source_sha256"], "config.py":semantics["config_sha256"]}.items():
        if sha(ROOT/p) != digest:
            raise ValueError("Native learner/proposal source changed during the check")
    paired.to_parquet(OUTPUT / "predictions.parquet",index=False)
    report = {"completed_at_utc":datetime.now(timezone.utc).isoformat(), "overall":overall,
        "established_handicaps":target, "market":market_scores, "advance_to_canonical":advance,
        "native_fit":predictor.eval_split_info, "component_cache_scope":scope,
        "elapsed_seconds":time.monotonic()-started, "production_changed":False,
        "current_price_independence_100_races_max_error":price_error,
        "limitations":["Previously inspected development; small point gains need forward confirmation.",
            "Candidate includes mandatory52-event metadata repairs alongside the new price-history inputs."]}
    write(OUTPUT / "report.json", report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-rebuild",action="store_true")
    main(parser.parse_args().wait_for_rebuild)
