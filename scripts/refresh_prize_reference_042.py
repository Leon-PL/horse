"""Repair the missing daily FX observations and only the post-fit history tail.

The official XML probe already proved every prior observation identical.
No feature/math source, raw result, trained weight or study protocol changes.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import certifi
from dotenv import get_key, set_key

from scripts.install_amended_results import read, write, replace_copy
from src import prize_currency_features as currency
from src.data_processor import process_data
from src.deployment_validation import file_sha256 as sha
from src.feature_store import FeatureStore, _stringify_objects
from src.featured_dataset import data_signature, fe_signature
from src.run_store import RUNS_DIR, get_production_run_id
from src.store_refresh import append_current_store

CASE = ROOT / "data/audit/result_amendments_042/consolidated"
OUTPUT = CASE / "fx_refresh"
REFERENCE = ROOT / "reference/ecb_gbp_per_eur.csv"
MODEL = "20261030_015917_823047"


def prepare():
    probe = read(OUTPUT / "probe.json")
    fit = read(CASE / "refit/fitted.json")
    if not probe["available"] or not probe["all_previous_rates_exact"] or get_production_run_id() != MODEL:
        raise ValueError("Qualify the verified source and installed model first")
    original = pd.read_csv(REFERENCE, float_precision="round_trip")
    fresh = pd.read_csv(OUTPUT / "ecb_gbp_per_eur.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(original, fresh.loc[fresh.observation_date.le(original.observation_date.max())].reset_index(drop=True), check_exact=True)
    main = FeatureStore()
    write(OUTPUT / "plan.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL, "model_sha256": sha(Path(RUNS_DIR) / MODEL / "model.joblib"),
        "fe_signature": fe_signature(), "original_store_meta": main.meta(),
        "old_reference_sha256": sha(REFERENCE), "new_reference_sha256": sha(OUTPUT / "ecb_gbp_per_eur.csv"),
        "raw_sha256": sha(ROOT / "data/raw/race_results.csv"), "training_prefix_signature": fit["signatures"]["data_signature"],
        "source_sha256": {p: sha(ROOT / p) for p in ("scripts/refresh_prize_reference_042.py", "src/prize_currency_features.py", "src/store_refresh.py")}})
    replace_copy(REFERENCE, OUTPUT / "previous_reference.csv")
    old = pd.read_parquet(main.processed_path)
    units, _ = currency._paths()
    with patch.object(currency, "_paths", lambda: (units, OUTPUT / "ecb_gbp_per_eur.csv")):
        processed = _stringify_objects(process_data(df=pd.read_csv(ROOT / "data/raw/race_results.csv"), save=False))
        if data_signature(processed, cutoff=fit["split"]["as_of"]) != fit["signatures"]["data_signature"]:
            raise ValueError("FX refresh changed the fitted historical prefix")
        keys = ["race_id", "horse_name"]
        a, b = (frame.set_index(keys).sort_index() for frame in (old, processed))
        pd.testing.assert_index_equal(a.index, b.index)
        changes = {}
        for name in a:
            if name not in {"prize_money", "prize_fx_date"}:
                pd.testing.assert_series_equal(a[name], b[name], check_exact=True, check_dtype=False)
            else:
                changed = ~(a[name].eq(b[name]) | (a[name].isna() & b[name].isna()))
                if changed.any() and not (b.loc[changed, "prize_currency"].eq("EUR")
                        & pd.to_datetime(b.loc[changed, "race_date"]).gt(fit["split"]["as_of"])).all():
                    raise ValueError("An unrelated runner's input changed")
                changes[name] = int(changed.sum())
        del old, a, b
        base = FeatureStore(str(CASE / "installation/ready_store"))
        ready = FeatureStore(str(OUTPUT / "ready_store"))
        with base._access():
            if base.meta()["data_signature"] != fit["signatures"]["data_signature"]:
                raise ValueError("Preserved fitted history is not the original corrected generation")
            for name in base._FILES:
                os.link(Path(base.root) / name, Path(ready.root) / name)
        if not append_current_store(processed, store=ready):
            raise ValueError("Corrected daily tail cannot be appended natively")
        expected_reference = currency.reference_signature()
        if ready.meta()["prize_reference_signature"] != expected_reference:
            raise ValueError("Prepared reference fingerprint differs")
    write(OUTPUT / "prepared.json", {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL, "changed_processed_cells": changes, "fitted_prefix_unchanged": True,
        "model_fits": 0, "rows": len(processed), "store_meta": ready.meta(),
        "expected_reference_signature": expected_reference,
        "store_files_sha256": {name: sha(Path(ready.root) / name) for name in ready._FILES}})
    print("Only the post-fit daily tail was rebuilt; all fitted historical inputs are unchanged.", flush=True)


def install():
    plan, prepared = read(OUTPUT / "plan.json"), read(OUTPUT / "prepared.json")
    paused = read(OUTPUT / "recorder_paused.json")
    main = FeatureStore()
    if not paused["paused"] or paused["model_id"] != MODEL or get_production_run_id() != MODEL:
        raise ValueError("Pause the current recorder before replacing its history/reference generation")
    if main.meta() != plan["original_store_meta"] or sha(REFERENCE) != plan["old_reference_sha256"]:
        raise ValueError("Current history/reference advanced after preparation")
    if sha(ROOT / "data/raw/race_results.csv") != plan["raw_sha256"]:
        raise ValueError("Current raw source changed")
    if any(sha(ROOT / p) != digest for p, digest in plan["source_sha256"].items()):
        raise ValueError("Prepared implementation changed")
    ready = FeatureStore(str(OUTPUT / "ready_store"))
    if any(sha(Path(ready.root) / name) != digest for name, digest in prepared["store_files_sha256"].items()):
        raise ValueError("Prepared history changed")
    if sha(OUTPUT / "ecb_gbp_per_eur.csv") != plan["new_reference_sha256"]:
        raise ValueError("Verified reference changed")
    # Project-local trust configuration uses the installed, verified CA bundle.
    # Preserve every other .env entry and never copy credentials into receipts.
    dotenv = ROOT / ".env"
    configured = get_key(str(dotenv), "SSL_CERT_FILE")
    bundle = Path(certifi.where()).as_posix()
    if configured and Path(configured).resolve() != Path(bundle).resolve():
        raise ValueError("Preserve the existing custom project CA configuration")
    set_key(str(dotenv), "SSL_CERT_FILE", bundle)
    backup = OUTPUT / "previous_store"
    backup.mkdir(exist_ok=False)
    with main._access():
        for name in main._FILES:
            os.link(Path(main.root) / name, backup / name)
        try:
            replace_copy(OUTPUT / "ecb_gbp_per_eur.csv", REFERENCE)
            with main._staged_update() as pending:
                for name in main._FILES:
                    os.link(Path(ready.root) / name, Path(pending.root) / name)
            if currency.reference_signature() != prepared["expected_reference_signature"]:
                raise ValueError("Installed reference fingerprint differs")
            if sha(Path(RUNS_DIR) / MODEL / "model.joblib") != plan["model_sha256"] or fe_signature() != plan["fe_signature"]:
                raise ValueError("The unchanged trained model/source differs")
        except Exception:
            replace_copy(OUTPUT / "previous_reference.csv", REFERENCE)
            with main._staged_update() as pending:
                for name in main._FILES:
                    os.link(backup / name, Path(pending.root) / name)
            raise
    write(ROOT / "data/reference/ecb_checked.json", {"date": datetime.now().date().isoformat(),
        "source_url": currency.ECB_URL, "checked_at_utc": read(OUTPUT / "probe.json")["checked_at_utc"]})
    write(OUTPUT / "installed.json", {"installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL, "model_bytes_unchanged": True, "fitted_inputs_unchanged": True,
        "new_model_fits": 0, "current_store_meta": main.meta(),
        "source_reference_sha256": sha(REFERENCE), "project_local_verified_ca_bundle": bundle,
        "changed_processed_cells": prepared["changed_processed_cells"]})
    print("Verified daily reference and matching history installed; model weights unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "install"])
    {"prepare": prepare, "install": install}[parser.parse_args().phase]()
