"""020: audit and screen only the three currency-corrected prize predictors.

Reuse the exact matched 018 baseline. Reconstruct a narrow full-history prize
projection, verify it against the native original features before conversion,
then run one candidate fit. No whole feature-store rebuild or quarterly queue.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import config
from scripts.benchmark_forecasts import align_predictions, score_forecasts
from scripts.benchmark_independent_features import canonical_hash, score_pair
from scripts.screen_feature_batch import OUTPUT as PREVIOUS, REFERENCE, sha, runtime_versions, verify as verify_previous
from src.model import RacePredictor
from src.prize_currency_features import FEATURES, convert_prizes, native_prize_history

OUTPUT = ROOT / "data/audit/model_improvement/020_prize_currency"
AUDIT = ROOT / "data/audit/prize_currency_020"
FX = ROOT / "data/research/ecb_fx_020"
KEYS = ["race_id", "horse_name", "race_date"]
SOURCES = ("scripts/screen_prize_currency.py", "scripts/audit_prize_currency.py",
           "src/prize_currency_features.py", "tests/test_prize_currency_features.py")


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=str))


def prepare():
    old = json.loads((PREVIOUS / "manifest.json").read_text()); verify_previous(old)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        raise ValueError("Already registered; resume without --prepare")
    currency_audit = json.loads((AUDIT / "source_currency_audit.json").read_text())
    table = AUDIT / "source_currency_audit.csv"
    if (currency_audit["unresolved"] or currency_audit["error_races"] or currency_audit["amount_disagreements"]
            or sha(table) != currency_audit["audit_table_sha256"]
            or sha(ROOT / "scripts/audit_prize_currency.py") != currency_audit["source_code_sha256"]):
        raise ValueError("Finish explicit venue-currency verification first")
    source_receipts = pd.read_csv(table, dtype={"race_id": str})
    for row in source_receipts.itertuples():
        if sha(AUDIT / "source_races" / f"{row.race_id}.json") != row.source_sha256:
            raise ValueError("A source prize response changed")
    fx_receipt = json.loads((FX / "receipt.json").read_text())
    if sha(FX / "eurofxref-hist.xml") != fx_receipt["xml_sha256"] or sha(FX / "gbp_per_eur.parquet") != fx_receipt["projection_sha256"]:
        raise ValueError("Frozen historical exchange rates changed")
    store = ROOT / "data/processed/feature_store"
    frame = pd.read_parquet(store / "processed.parquet",
        columns=[*KEYS, "horse_id", "off_time", "track", "prize_money"])
    reference = pd.read_parquet(store / "featured.parquet", columns=[*KEYS, "horse_prev_races", *FEATURES])
    if not frame[KEYS].equals(reference[KEYS]):
        raise ValueError("Processed and feature histories do not align")
    frame["horse_prev_races"] = reference.horse_prev_races.to_numpy()
    original = native_prize_history(frame, frame.prize_money)
    oracle_errors = {}
    for name in FEATURES:
        np.testing.assert_allclose(original[name], reference[name], rtol=0, atol=1e-6, equal_nan=True, err_msg=name)
        oracle_errors[name] = float(np.nanmax(np.abs(original[name]-reference[name])))
    currencies = frame.track.map(currency_audit["currency_by_track"])
    corrected = native_prize_history(frame, convert_prizes(frame, currency_audit["currency_by_track"],
                                                           pd.read_parquet(FX / "gbp_per_eur.parquet")))
    never_euro = ~currencies.eq("EUR").groupby(frame.horse_id).transform("any")
    for name in FEATURES:
        np.testing.assert_allclose(original.loc[never_euro, name], corrected.loc[never_euro, name], rtol=0, atol=0, equal_nan=True)
    projection = pd.concat([frame[KEYS], corrected], axis=1)
    projection_path = AUDIT / "gbp_prize_features.parquet"
    projection.to_parquet(projection_path, index=False)
    review = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "source_currency_audit_sha256": sha(AUDIT / "source_currency_audit.json"),
        "exchange_rate_receipt_sha256": sha(FX / "receipt.json"), "projection_sha256": sha(projection_path),
        "native_original_replay_max_error": oracle_errors, "unaffected_all_gbp_history_rows": int(never_euro.sum()),
        "euro_rows": int(currencies.eq("EUR").sum()),
        "changed_rows": {name: int((~np.isclose(reference[name].to_numpy(np.float32), corrected[name].to_numpy(np.float32), rtol=0, atol=0, equal_nan=True)).sum()) for name in FEATURES},
        "scope": "Only monetary units change; preserve the native original zero/cold-start/denominator definitions"}
    write(AUDIT / "prize_feature_review.json", review)
    if sha(PREVIOUS / "baseline_inputs.parquet") != old["projection_sha256"]["baseline"]:
        raise ValueError("Matched baseline inputs changed")
    baseline = pd.read_parquet(PREVIOUS / "baseline_inputs.parquet")
    candidate = baseline.drop(columns=list(FEATURES)).merge(projection, on=KEYS, how="left", sort=False, validate="one_to_one")
    names = old["features"]["baseline"]
    candidate = candidate[baseline.columns].copy()
    candidate[names] = candidate[names].astype(np.float32)
    if not candidate[KEYS].equals(baseline[KEYS]) or candidate[list(FEATURES)].isna().any().any():
        raise ValueError("Currency projection lost a registered runner")
    untouched = [c for c in baseline if c not in FEATURES]
    pd.testing.assert_frame_equal(candidate[untouched], baseline[untouched])
    path = OUTPUT / "candidate_inputs.parquet"; candidate.to_parquet(path, index=False, compression="zstd")
    value = {"registered_at_utc": datetime.now(timezone.utc).isoformat(), "features": names,
        "changed_features": FEATURES, "params": old["params"], "dataset_start": old["dataset_start"],
        "test_start": old["test_start"], "test_end": old["test_end"], "training_cutoff": old["training_cutoff"],
        "source_sha256": {name: sha(ROOT / name) for name in SOURCES}, "runtime_versions": runtime_versions(),
        "projection_sha256": sha(path), "previous_manifest_sha256": sha(PREVIOUS / "manifest.json"),
        "previous_baseline_sha256": {name: sha(PREVIOUS / name) for name in ("baseline.joblib", "baseline_fit.json", "baseline_predictions.parquet")},
        "public_input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
            AUDIT / "source_currency_audit.json", table, AUDIT / "selected_races.json", AUDIT / "prize_feature_review.json",
            projection_path, FX / "receipt.json", FX / "gbp_per_eur.parquet", FX / "eurofxref-hist.xml", store / "meta.json")},
        "selection": "Shortlist only if race log loss and runner Brier both improve versus the cached matched baseline; uncertainty must be reported and any deployment requires wider validation",
        "scope": "One already inspected retrospective development window; one win-only candidate, native calibration,500trees/.025,two-year fitting rows,full feature history; no automatic promotion or more quarters"}
    write(OUTPUT / "manifest.json", value)
    with zipfile.ZipFile(OUTPUT / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    print(json.dumps({"stage": "registered", "review": review, "manifest_sha256": canonical_hash(value)}), flush=True)


def verify(manifest):
    old = json.loads((PREVIOUS / "manifest.json").read_text()); verify_previous(old)
    if runtime_versions() != manifest["runtime_versions"] or sha(PREVIOUS / "manifest.json") != manifest["previous_manifest_sha256"]:
        raise ValueError("Reference runtime or manifest changed")
    for name, expected in manifest["previous_baseline_sha256"].items():
        if sha(PREVIOUS / name) != expected:
            raise ValueError("Cached matched baseline changed")
    for name, expected in {**manifest["source_sha256"], **manifest["public_input_sha256"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Currency screen source changed: {name}")
    if sha(OUTPUT / "candidate_inputs.parquet") != manifest["projection_sha256"]:
        raise ValueError("Candidate input projection changed")


def run():
    manifest = json.loads((OUTPUT / "manifest.json").read_text()); verify(manifest)
    begin = time.perf_counter()
    frame = pd.read_parquet(OUTPUT / "candidate_inputs.parquet")
    config.FEATURE_ALLOWLIST = manifest["features"]
    config.WIN_SOFT_LABEL_DISTILL, config.WIN_SOFT_LABEL_BLEND = True, .5
    predictor = RacePredictor(random_seed=42, frameworks={"classifier": "lgbm", "place": "lgbm"})
    predictor._autotune_njobs = 2
    path = OUTPUT / "candidate.joblib"
    if path.exists():
        predictor.load(str(path))
        if predictor.eval_split_info.get("currency_screen_manifest_sha256") != canonical_hash(manifest):
            raise ValueError("Saved model belongs to another experiment")
    else:
        train = frame.loc[frame.race_date.le(manifest["training_cutoff"])].copy()
        print(json.dumps({"stage": "fitting", "training_rows": len(train), "features": len(manifest["features"])}), flush=True)
        predictor.fit_as_of(train, pd.Timestamp(manifest["training_cutoff"]), params=manifest["params"],
                            dataset_start=pd.Timestamp(manifest["dataset_start"]), win_only=True)
        predictor.eval_split_info["currency_screen_manifest_sha256"] = canonical_hash(manifest)
        verify(manifest)
        temporary = path.with_suffix(".incomplete.joblib")
        predictor.save(str(temporary)); temporary.replace(path)
    test = frame.loc[frame.race_date.ge(manifest["test_start"])].copy()
    baseline = pd.read_parquet(PREVIOUS / "baseline_predictions.parquet")
    if not baseline[KEYS + ["won"]].equals(test[KEYS + ["won"]].reset_index(drop=True)):
        raise ValueError("Candidate and cached baseline fields differ")
    result = baseline.drop(columns="probability").assign(baseline_probability=baseline.probability,
                                                          candidate_probability=align_predictions(predictor, test))
    replay = RacePredictor(); replay.load(str(path))
    np.testing.assert_array_equal(result.candidate_probability, align_predictions(replay, test))
    result.to_parquet(OUTPUT / "paired_predictions.parquet", index=False)
    comparison = score_pair(result)
    market_path = REFERENCE / "market_context_predictions.parquet"
    complete = json.loads((REFERENCE / "completion.json").read_text())
    if sha(market_path) != complete["evidence_sha256"][market_path.name]:
        raise ValueError("Reference market evidence changed")
    market = pd.read_parquet(market_path)
    market = market.loc[pd.to_datetime(market.race_date).between(manifest["test_start"], manifest["test_end"])]
    paired = market.drop(columns=["baseline_probability", "candidate_probability"]).merge(
        result[KEYS + ["won", "baseline_probability", "candidate_probability"]], on=KEYS + ["won"], how="left", sort=False, validate="one_to_one")
    if paired[["baseline_probability", "candidate_probability"]].isna().any().any():
        raise ValueError("Every runner on every market field must align")
    market_scores = score_forecasts(paired, {"market": paired.market_probability.to_numpy(),
        "baseline": paired.baseline_probability.to_numpy(), "candidate": paired.candidate_probability.to_numpy()})
    verify(manifest)
    report = {"manifest_sha256": canonical_hash(manifest), "comparison": comparison, "market": market_scores,
        "point_improves_both": bool(comparison["candidate"]["race_log_loss"] < comparison["baseline"]["race_log_loss"]
                                    and comparison["candidate"]["brier"] < comparison["baseline"]["brier"]),
        "fit": predictor.eval_split_info, "elapsed_seconds": time.perf_counter()-begin,
        "saved_model_replay_max_error": 0., "model_sha256": sha(path), "scope": manifest["scope"],
        "production_changed": False, "additional_quarters_queued": False}
    write(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    arguments = parser.parse_args()
    prepare() if arguments.prepare else run()
